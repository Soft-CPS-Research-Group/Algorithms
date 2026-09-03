"""Runner integration for the standalone ``TIMARL`` execution unit."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch
import yaml

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.maddpg_agent import _select_torch_device
from algorithms.ti_marl.compiler.compiler import TypedInterfaceCompiler
from algorithms.ti_marl.contracts.compatibility import CompatibilitySignature
from algorithms.ti_marl.contracts.models import TypedTransition, canonical_value
from algorithms.ti_marl.learning.mappo import TIMAPPO
from algorithms.ti_marl.learning.behavior_cloning import (
    TypedBehaviorCloningWarmStart,
)
from algorithms.ti_marl.learning.ev_planning import CausalEVPlanner
from algorithms.ti_marl.learning.storage_planning import CausalStoragePlanner
from algorithms.ti_marl.learning.rollout import RolloutStep
from algorithms.ti_marl.policy.networks import (
    CentralSetCritic,
    LocalTypedCritic,
    TypedActor,
    TypedGroupCritic,
    parameter_count,
)
from algorithms.ti_marl.runtime.codec import CityLearnTypedActionCodec
from algorithms.ti_marl.runtime.commands import TypedCommandBuilder
from algorithms.ti_marl.runtime.feasibility import AnalyticLocalProjector
from algorithms.ti_marl.runtime.traces import BufferedTraceWriter


class TIMARL(BaseAgent):
    """Typed-interface control with PPO or MAPPO learning backbones."""

    supports_dynamic_topology = True
    handles_cross_topology_transitions = True
    requires_entity_observation_context = True
    requires_final_pipeline_stage = True
    _use_raw_observations = False

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = deepcopy(config)
        hyper = dict(config.get("algorithm", {}).get("hyperparameters", {}))
        feasibility_cfg = dict(hyper.get("feasibility", {}))
        simulator_cfg = dict(config.get("simulator") or {})
        community_market_cfg = dict(
            simulator_cfg.get("community_market") or {}
        )
        settlement_value_ratio = (
            float(
                community_market_cfg.get(
                    "local_price_ratio_to_grid_import",
                    community_market_cfg.get("intra_community_sell_ratio", 1.0),
                )
            )
            if bool(community_market_cfg.get("enabled", False))
            else 1.0
        )
        configured_v2g_value_ratio = feasibility_cfg.get(
            "ev_v2g_avoided_import_value_ratio"
        )
        effective_v2g_value_ratio = (
            settlement_value_ratio
            if configured_v2g_value_ratio is None
            else float(configured_v2g_value_ratio)
        )
        checkpoint_cfg = dict(config.get("checkpointing") or {})
        self.restore_optimizers = bool(
            checkpoint_cfg.get("restore_optimizers", True)
        )
        self.restore_rollout = bool(
            checkpoint_cfg.get("restore_replay_buffer", True)
        )
        backbone = dict(hyper.get("backbone", {}))
        critic_cfg = dict(hyper.get("critic", {}))
        self.backbone_name = str(backbone.get("name", "mappo")).lower()
        self.critic_kind = str(critic_cfg.get("kind", "set")).lower()
        expected_critic = {"ppo": "local", "mappo": "set"}.get(
            self.backbone_name
        )
        if expected_critic is None:
            raise ValueError("TIMARL v1 supports backbone.name in {'ppo', 'mappo'}")
        if self.critic_kind != expected_critic:
            raise ValueError(
                f"TIMARL backbone.name={self.backbone_name!r} requires "
                f"critic.kind={expected_critic!r}"
            )

        self.seed = int(config.get("training", {}).get("seed", 22))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        cuda_usable = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        if cuda_usable:
            torch.cuda.manual_seed_all(self.seed)
        require_cuda = bool(hyper.get("require_cuda", False))
        self.device = _select_torch_device(
            require_cuda=require_cuda,
            algorithm_name="TIMARL",
        )

        self.compiler = TypedInterfaceCompiler(
            contract_version=str(hyper.get("contract_version", "ti_marl_v1")),
            typed_interfaces_dir=str(hyper.get("typed_interfaces_dir", "")),
            interface_polling=bool(hyper.get("interface_polling", False)),
            simulator_bindings_path=hyper.get("simulator_bindings_path"),
        )
        if bool(hyper.get("require_declared_electrical_service", False)):
            self._require_declared_electrical_service()
        self.allow_checkpoint_compiler_migration = bool(
            hyper.get("allow_checkpoint_compiler_migration", False)
        )
        actor_cfg = dict(hyper.get("actor", {}))
        d_model = int(actor_cfg.get("d_model", 128))
        relation_layers = int(actor_cfg.get("relation_layers", 2))
        self.actor_group_context_kind = str(
            actor_cfg.get("group_context_kind", "local")
        )
        self.policy_credit_assignment = str(
            hyper.get("policy_credit_assignment", "joint_agent")
        )
        self.policy_anchor_reset_on_resume = bool(
            hyper.get("policy_anchor_reset_on_resume", False)
        )
        self.actor = TypedActor(
            self.compiler.type_registry,
            d_model=d_model,
            attention_heads=int(actor_cfg.get("attention_heads", 4)),
            relation_layers=relation_layers,
            group_context_kind=self.actor_group_context_kind,
            deterministic_mode_strategy=str(
                actor_cfg.get("deterministic_mode_strategy", "argmax")
            ),
            deterministic_mode_strategy_by_group_type=dict(
                actor_cfg.get("deterministic_mode_strategy_by_group_type", {})
            ),
            deterministic_expected_signed_gain_by_group_type=dict(
                actor_cfg.get(
                    "deterministic_expected_signed_gain_by_group_type", {}
                )
            ),
            deterministic_expected_signed_deadband_by_group_type=dict(
                actor_cfg.get(
                    "deterministic_expected_signed_deadband_by_group_type", {}
                )
            ),
            deterministic_non_idle_logit_margin_by_group_type=dict(
                actor_cfg.get(
                    "deterministic_non_idle_logit_margin_by_group_type", {}
                )
            ),
        ).to(self.device)
        critic_class = (
            CentralSetCritic if self.critic_kind == "set" else LocalTypedCritic
        )
        self.critic = critic_class(
            self.compiler.type_registry,
            d_model=d_model,
            relation_layers=relation_layers,
        ).to(self.device)
        self.group_critic = (
            TypedGroupCritic(
                self.compiler.type_registry,
                d_model=d_model,
                relation_layers=relation_layers,
                centralized=self.critic_kind == "set",
            ).to(self.device)
            if self.policy_credit_assignment == "typed_group"
            else None
        )
        self._parameter_count = self._current_parameter_count()
        ev_planning_cfg = dict(hyper.get("ev_planning", {}))
        ev_planning_auxiliary_coeff = float(
            ev_planning_cfg.get("auxiliary_coeff", 0.0)
        )
        configured_planner_v2g_value_ratio = ev_planning_cfg.get(
            "v2g_avoided_import_value_ratio"
        )
        planner_v2g_value_ratio = (
            effective_v2g_value_ratio
            if configured_planner_v2g_value_ratio is None
            else float(configured_planner_v2g_value_ratio)
        )
        ev_planner = (
            CausalEVPlanner(
                charge_fraction=float(
                    ev_planning_cfg.get("charge_fraction", 0.95)
                ),
                discharge_fraction=float(
                    ev_planning_cfg.get("discharge_fraction", 0.50)
                ),
                service_tolerance_ratio=float(
                    ev_planning_cfg.get("service_tolerance_ratio", 0.05)
                ),
                v2g_service_margin_ratio=float(
                    ev_planning_cfg.get("v2g_service_margin_ratio", 0.05)
                ),
                price_tie_tolerance=float(
                    ev_planning_cfg.get("price_tie_tolerance", 1.0e-6)
                ),
                urgency_duty_ratio=float(
                    ev_planning_cfg.get("urgency_duty_ratio", 0.85)
                ),
                minimum_price_spread=float(
                    ev_planning_cfg.get("minimum_price_spread", 0.0)
                ),
                minimum_v2g_price_spread=float(
                    ev_planning_cfg.get("minimum_v2g_price_spread", 0.01)
                ),
                minimum_v2g_departure_hours=float(
                    ev_planning_cfg.get("minimum_v2g_departure_hours", 1.0)
                ),
                v2g_avoided_import_value_ratio=planner_v2g_value_ratio,
                v2g_minimum_profit_margin_eur_per_kwh=float(
                    ev_planning_cfg.get(
                        "v2g_minimum_profit_margin_eur_per_kwh",
                        feasibility_cfg.get(
                            "ev_v2g_minimum_profit_margin_eur_per_kwh",
                            0.01,
                        ),
                    )
                ),
                v2g_degradation_cost_eur_per_kwh=float(
                    ev_planning_cfg.get(
                        "v2g_degradation_cost_eur_per_kwh",
                        feasibility_cfg.get(
                            "ev_v2g_degradation_cost_eur_per_kwh",
                            0.0,
                        ),
                    )
                ),
            )
            if ev_planning_auxiliary_coeff > 0.0
            else None
        )
        storage_planning_cfg = dict(hyper.get("storage_planning", {}))
        storage_planning_auxiliary_coeff = float(
            storage_planning_cfg.get("auxiliary_coeff", 0.0)
        )
        storage_planner = (
            CausalStoragePlanner(
                charge_fraction=float(
                    storage_planning_cfg.get("charge_fraction", 0.55)
                ),
                discharge_fraction=float(
                    storage_planning_cfg.get("discharge_fraction", 0.45)
                ),
                minimum_soc_ratio=float(
                    storage_planning_cfg.get("minimum_soc_ratio", 0.20)
                ),
                maximum_soc_ratio=float(
                    storage_planning_cfg.get("maximum_soc_ratio", 0.90)
                ),
                price_tie_tolerance=float(
                    storage_planning_cfg.get("price_tie_tolerance", 1.0e-6)
                ),
                minimum_price_spread=float(
                    storage_planning_cfg.get("minimum_price_spread", 0.01)
                ),
                pv_surplus_threshold_kw=float(
                    storage_planning_cfg.get("pv_surplus_threshold_kw", 0.25)
                ),
                import_threshold_kw=float(
                    storage_planning_cfg.get("import_threshold_kw", 0.25)
                ),
                price_regime_kind=str(
                    storage_planning_cfg.get(
                        "price_regime_kind",
                        "strict_extrema",
                    )
                ),
                forecast_mean_margin_fraction=float(
                    storage_planning_cfg.get(
                        "forecast_mean_margin_fraction",
                        0.20,
                    )
                ),
                forecast_edge_margin_fraction=float(
                    storage_planning_cfg.get(
                        "forecast_edge_margin_fraction",
                        0.10,
                    )
                ),
                forecast_spread_floor_ratio=float(
                    storage_planning_cfg.get(
                        "forecast_spread_floor_ratio",
                        0.05,
                    )
                ),
                scale_price_fraction_by_opportunity=bool(
                    storage_planning_cfg.get(
                        "scale_price_fraction_by_opportunity",
                        False,
                    )
                ),
                minimum_price_fraction_scale=float(
                    storage_planning_cfg.get(
                        "minimum_price_fraction_scale",
                        0.50,
                    )
                ),
            )
            if storage_planning_auxiliary_coeff > 0.0
            else None
        )
        self.learner = TIMAPPO(
            self.actor,
            self.critic,
            group_critic=self.group_critic,
            learning_rate=float(hyper.get("learning_rate", 3.0e-4)),
            gamma=float(hyper.get("gamma", 0.99)),
            gae_lambda=float(hyper.get("gae_lambda", 0.95)),
            discount_timebase_seconds=(
                None
                if hyper.get("discount_timebase_seconds") is None
                else float(hyper["discount_timebase_seconds"])
            ),
            clip_eps=float(hyper.get("clip_eps", 0.2)),
            ppo_epochs=int(hyper.get("ppo_epochs", 4)),
            entropy_coeff=float(hyper.get("entropy_coeff", 0.01)),
            entropy_coeff_by_group_type=dict(
                hyper.get("entropy_coeff_by_group_type", {})
            ),
            advantage_normalization=str(
                hyper.get("advantage_normalization", "global")
            ),
            policy_credit_assignment=self.policy_credit_assignment,
            ppo_policy_group_types=hyper.get("ppo_policy_group_types"),
            policy_anchor_coeff=float(hyper.get("policy_anchor_coeff", 0.0)),
            policy_anchor_coeff_by_group_type=dict(
                hyper.get("policy_anchor_coeff_by_group_type", {})
            ),
            exclude_intervened_actions_from_policy_loss=bool(
                hyper.get(
                    "exclude_intervened_actions_from_policy_loss",
                    False,
                )
            ),
            intervention_distillation_coeff=float(
                hyper.get("intervention_distillation_coeff", 0.0)
            ),
            ev_planner=ev_planner,
            ev_planning_auxiliary_coeff=ev_planning_auxiliary_coeff,
            ev_planning_balance_targets=bool(
                ev_planning_cfg.get("balance_targets", True)
            ),
            ev_planning_fraction_coeff=float(
                ev_planning_cfg.get("fraction_coeff", 0.25)
            ),
            ev_planning_replay_capacity_per_reason=int(
                ev_planning_cfg.get("replay_capacity_per_reason", 16)
            ),
            ev_planning_replay_samples_per_reason=int(
                ev_planning_cfg.get("replay_samples_per_reason", 8)
            ),
            storage_planner=storage_planner,
            storage_planning_auxiliary_coeff=(
                storage_planning_auxiliary_coeff
            ),
            storage_planning_balance_targets=bool(
                storage_planning_cfg.get("balance_targets", True)
            ),
            storage_planning_fraction_coeff=float(
                storage_planning_cfg.get("fraction_coeff", 0.25)
            ),
            storage_planning_replay_capacity_per_reason=int(
                storage_planning_cfg.get("replay_capacity_per_reason", 16)
            ),
            storage_planning_replay_samples_per_reason=int(
                storage_planning_cfg.get("replay_samples_per_reason", 8)
            ),
            value_coeff=float(hyper.get("value_coeff", 0.5)),
            max_grad_norm=float(hyper.get("max_grad_norm", 0.5)),
            target_kl=(None if hyper.get("target_kl", 0.03) is None else float(hyper.get("target_kl", 0.03))),
            rollout_steps=int(hyper.get("rollout_steps", 256)),
            normalize_value_targets=bool(hyper.get("normalize_value_targets", True)),
            value_target_scale_floor=float(
                hyper.get("value_target_scale_floor", 1.0)
            ),
            critic_loss=str(hyper.get("critic_loss", "huber")),
        )
        bc_cfg = hyper.get("behavior_cloning")
        self.behavior_cloning = None
        self._bc_teacher = None
        self._bc_teacher_spec: Mapping[str, Any] = {}
        self._training_progress_callback: Optional[
            Callable[[Mapping[str, Any]], None]
        ] = None
        if isinstance(bc_cfg, Mapping) and bool(bc_cfg.get("enabled", True)):
            teacher = dict(bc_cfg.get("teacher") or {})
            if str(teacher.get("policy", "RBCSmartPolicy")) != "RBCSmartPolicy":
                raise ValueError(
                    "TIMARL behavior_cloning.teacher.policy must be 'RBCSmartPolicy'"
                )
            self.behavior_cloning = TypedBehaviorCloningWarmStart(
                demonstration_episodes=int(bc_cfg.get("demonstration_episodes", 1)),
                max_samples=int(bc_cfg.get("max_samples", 4096)),
                pretraining_epochs=int(bc_cfg.get("pretraining_epochs", 4)),
                batch_size=int(bc_cfg.get("batch_size", 64)),
                learning_rate=float(bc_cfg.get("learning_rate", 3.0e-4)),
                balance_action_modes=bool(
                    bc_cfg.get("balance_action_modes", True)
                ),
                mode_balance_exponent=float(
                    bc_cfg.get("mode_balance_exponent", 0.5)
                ),
                max_mode_weight=float(bc_cfg.get("max_mode_weight", 4.0)),
                seed=self.seed,
                balanced_loss_kind=str(
                    bc_cfg.get("balanced_loss_kind", "weighted")
                ),
                calibration_epochs=int(bc_cfg.get("calibration_epochs", 0)),
                calibration_learning_rate=(
                    None
                    if bc_cfg.get("calibration_learning_rate") is None
                    else float(bc_cfg["calibration_learning_rate"])
                ),
            )
            self._bc_teacher_spec = teacher
        self.requires_raw_observation_context = self.behavior_cloning is not None
        self.projector = AnalyticLocalProjector(
            enforce_ev_service=bool(feasibility_cfg.get("enforce_ev_service", True)),
            ev_service_margin_ratio=float(
                feasibility_cfg.get("ev_service_margin_ratio", 0.05)
            ),
            ev_service_strategy=str(
                feasibility_cfg.get("ev_service_strategy", "average")
            ),
            ev_service_tolerance_ratio=float(
                feasibility_cfg.get("ev_service_tolerance_ratio", 0.05)
            ),
            ev_service_jit_buffer_seconds=float(
                feasibility_cfg.get("ev_service_jit_buffer_seconds", 0.0)
            ),
            ev_service_jit_minimum_average_fraction=float(
                feasibility_cfg.get(
                    "ev_service_jit_minimum_average_fraction",
                    0.0,
                )
            ),
            enforce_ev_discharge_reserve=bool(
                feasibility_cfg.get("enforce_ev_discharge_reserve", True)
            ),
            ev_v2g_reserve_margin_ratio=float(
                feasibility_cfg.get("ev_v2g_reserve_margin_ratio", 0.0)
            ),
            enforce_ev_economic_guard=bool(
                feasibility_cfg.get("enforce_ev_economic_guard", True)
            ),
            ev_v2g_avoided_import_value_ratio=float(
                effective_v2g_value_ratio
            ),
            ev_v2g_minimum_profit_margin_eur_per_kwh=float(
                feasibility_cfg.get(
                    "ev_v2g_minimum_profit_margin_eur_per_kwh",
                    0.01,
                )
            ),
            ev_v2g_degradation_cost_eur_per_kwh=float(
                feasibility_cfg.get(
                    "ev_v2g_degradation_cost_eur_per_kwh",
                    0.0,
                )
            ),
            ev_v2g_require_local_demand=bool(
                feasibility_cfg.get("ev_v2g_require_local_demand", True)
            ),
            headroom_reserve_kw=float(
                feasibility_cfg.get("headroom_reserve_kw", 0.0)
            ),
            deferrable_service_margin_seconds=float(
                feasibility_cfg.get("deferrable_service_margin_seconds", 0.0)
            ),
        )
        self.codec = CityLearnTypedActionCodec()
        self.command_builder = TypedCommandBuilder()

        trace_cfg = dict(hyper.get("trace", {}))
        job_dir = config.get("runtime", {}).get("job_dir")
        trace_dir = None if not job_dir else Path(job_dir) / "results" / "ti_marl_trace"
        self.trace_writer = BufferedTraceWriter(
            trace_dir,
            chunk_size=int(trace_cfg.get("chunk_size", 256)),
            snapshot_interval=int(trace_cfg.get("snapshot_interval", 256)),
            enabled=bool(trace_cfg.get("enabled", True)),
        )
        self.requires_deterministic_transition_observation = bool(
            self.trace_writer.enabled
        )

        self._current_snapshot = None
        self._next_snapshot = None
        self._transition_info: Mapping[str, Any] = {}
        self._pending: Optional[Dict[str, Any]] = None
        self._building_names: tuple[str, ...] = ()
        self._latest_training_metrics: Dict[str, float] = {}
        self._latest_diagnostics: Dict[str, float] = {}
        self._latest_raw_observations: Optional[List[np.ndarray]] = None
        self._episode_raw_modes: Counter[tuple[str, str]] = Counter()
        self._episode_final_modes: Counter[tuple[str, str]] = Counter()
        self._episode_raw_fraction_sums: Counter[tuple[str, str]] = Counter()
        self._episode_final_fraction_sums: Counter[tuple[str, str]] = Counter()
        self._episode_ev_control: Counter[str] = Counter()
        self._current_episode = 0
        self._current_episode_is_training = False

    def _require_declared_electrical_service(self) -> None:
        """Fail before training when controllable members lack grid contracts.

        The deployment-safe default for unknown headroom is zero.  That is the
        correct runtime fallback, but it can make a long training campaign look
        healthy while silently suppressing every flexible action.  Campaigns
        that expect active control can opt into this structural preflight.
        """

        missing: dict[str, tuple[str, ...]] = {}
        for agent_id in self.compiler.interface_registry.agent_ids:
            interface = self.compiler.interface_registry.for_agent(agent_id)
            modes = {
                action.mode
                for actuator in interface.actuators
                for action in actuator.actions
            }
            required = set()
            if any(
                mode.startswith("CHARGE_") or mode == "START"
                for mode in modes
            ):
                required.add("grid_import")
            absent = tuple(
                sorted(
                    key
                    for key in required
                    if not self._has_positive_constraint_limit(
                        interface.constraints.get(key)
                    )
                )
            )
            if absent:
                missing[agent_id] = absent
        if missing:
            rendered = ", ".join(
                f"{agent_id} ({'/'.join(keys)})"
                for agent_id, keys in sorted(missing.items())
            )
            raise ValueError(
                "TI-MARL requires explicit electrical-service constraints for "
                f"this campaign; missing: {rendered}"
            )

    @staticmethod
    def _has_positive_constraint_limit(raw: Any) -> bool:
        if not isinstance(raw, Mapping):
            return False
        value = raw.get("max")
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(float(value))
            and float(value) > 0.0
        )

    def _current_parameter_count(self) -> int:
        return (
            parameter_count(self.actor)
            + parameter_count(self.critic)
            + (
                0
                if self.group_critic is None
                else parameter_count(self.group_critic)
            )
        )

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = dict(metadata or {})
        if metadata.get("interface") != "entity":
            raise ValueError("TIMARL requires simulator.interface='entity'")
        topology_mode = str(metadata.get("topology_mode", "static"))
        if topology_mode not in {"static", "dynamic"}:
            raise ValueError(f"Unsupported TIMARL topology mode: {topology_mode}")
        entity_specs = metadata.get("entity_specs")
        if not isinstance(entity_specs, Mapping):
            raise ValueError("TIMARL attach_environment requires entity_specs")
        self.compiler.attach_entity_specs(
            entity_specs,
            seconds_per_time_step=float(metadata.get("seconds_per_time_step", 1.0)),
        )
        self.projector.set_seconds_per_time_step(
            float(metadata.get("seconds_per_time_step", 1.0))
        )
        self.learner.set_seconds_per_time_step(
            float(metadata.get("seconds_per_time_step", 1.0))
        )
        building_names = metadata.get("building_names") or entity_specs.get("tables", {}).get("building", {}).get("ids", [])
        self._building_names = tuple(str(item) for item in building_names)
        self.codec.attach(
            building_names=self._building_names,
            action_names=action_names,
            action_space=action_space,
        )
        if self.behavior_cloning is not None:
            from algorithms.agents.baseline_policies import RBCSmartPolicy

            teacher_config = deepcopy(self.config)
            teacher_config["algorithm"] = {
                "name": "RBCSmartPolicy",
                "hyperparameters": deepcopy(
                    dict(self._bc_teacher_spec.get("hyperparameters") or {})
                ),
            }
            self._bc_teacher = RBCSmartPolicy(teacher_config)
            self._bc_teacher.attach_environment(
                observation_names=observation_names,
                action_names=action_names,
                action_space=action_space,
                observation_space=observation_space,
                metadata=metadata,
            )
        if self._current_parameter_count() != self._parameter_count:
            raise AssertionError("TIMARL parameter count changed during topology attachment")

    def snapshot_topology_state(self) -> Mapping[str, Any]:
        return {
            "compiler": self.compiler.snapshot_state(),
            "building_names": self._building_names,
            "codec": deepcopy(self.codec.__dict__),
            "current_snapshot": self._current_snapshot,
            "next_snapshot": self._next_snapshot,
        }

    def restore_topology_state(self, snapshot: Mapping[str, Any]) -> None:
        self.compiler.restore_state(snapshot["compiler"])
        self._building_names = tuple(snapshot["building_names"])
        self.codec.__dict__.clear()
        self.codec.__dict__.update(deepcopy(snapshot["codec"]))
        self._current_snapshot = snapshot.get("current_snapshot")
        self._next_snapshot = snapshot.get("next_snapshot")

    def set_entity_observation_context(
        self,
        *,
        observation_payload: Mapping[str, Any],
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        del info
        meta = observation_payload.get("meta", {})
        if not (
            self._current_snapshot is not None
            and self._current_snapshot.time_step == int(meta.get("time_step", -1))
            and self._current_snapshot.topology_version
            == int(meta.get("topology_version", -1))
        ):
            started = time.perf_counter()
            self._current_snapshot = self.compiler.compile(observation_payload)
            self._latest_diagnostics["TI_MARL/health_derivation_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
        self._next_snapshot = None

    def set_observation_context(
        self,
        *,
        raw_observations: Optional[List[np.ndarray]] = None,
        encoded_observations: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Keep physical observations exclusively for the optional teacher."""

        del encoded_observations
        self._latest_raw_observations = (
            [np.asarray(item, dtype=np.float64) for item in raw_observations]
            if raw_observations is not None
            else None
        )

    def set_entity_transition_context(
        self,
        *,
        observation_payload: Mapping[str, Any],
        next_observation_payload: Mapping[str, Any],
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        del observation_payload
        if self._current_snapshot is None:
            raise RuntimeError("TIMARL transition context arrived before observation context")
        started = time.perf_counter()
        self._next_snapshot = self.compiler.compile(next_observation_payload)
        self._latest_diagnostics["TI_MARL/health_derivation_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        self._transition_info = dict(info or {})

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        del observations, context
        if self._current_snapshot is None:
            raise RuntimeError("TIMARL predict requires set_entity_observation_context")
        if self._in_demonstration_phase():
            if self._bc_teacher is None or self._latest_raw_observations is None:
                raise RuntimeError(
                    "TIMARL behavior cloning requires an attached SMART teacher "
                    "and raw physical observation context"
                )
            self._pending = None
            return self._bc_teacher.predict(
                self._latest_raw_observations,
                deterministic=True,
            )
        self.actor.eval()
        self.critic.eval()
        if self.group_critic is not None:
            self.group_critic.eval()
        with torch.no_grad():
            evaluation = self.actor(
                self._current_snapshot,
                deterministic=bool(deterministic),
            )
            values = self.critic(self._current_snapshot)
            group_values = (
                {}
                if self.group_critic is None
                else self.group_critic(self._current_snapshot)
            )
        raw_bundles = evaluation.bundles
        final_bundles = self.projector.project(self._current_snapshot, raw_bundles)
        self.projector.assert_feasible(self._current_snapshot, final_bundles)
        typed_commands = self.command_builder.build(
            self._current_snapshot,
            final_bundles,
        )
        commands = self.codec.encode_typed(self._current_snapshot, typed_commands)
        intervention_count = sum(len(bundle.interventions) for bundle in final_bundles)
        intervention_magnitude = sum(
            float(item.get("magnitude", 0.0))
            for bundle in final_bundles
            for item in bundle.interventions
        )
        total_groups = max(sum(len(bundle.decisions) for bundle in raw_bundles), 1)
        active_durations = [
            float(item.active_duration_seconds)
            for item in self._current_snapshot.fault_evidence
            if item.fault_mode is not None
        ]
        recovery_pending = [
            float(item.recovery_pending_seconds)
            for item in self._current_snapshot.health
            if item.recovery_pending_seconds > 0
        ]
        self._latest_diagnostics = {
            **self._latest_diagnostics,
            **self._action_mode_diagnostics(raw_bundles, final_bundles),
            **self._ev_actor_control_diagnostics(
                self._current_snapshot,
                raw_bundles,
                final_bundles,
            ),
            "TI_MARL/agents": float(len(self._current_snapshot.agent_ids)),
            "TI_MARL/registered_agents": float(
                len(self._current_snapshot.registered_agent_ids)
            ),
            "TI_MARL/groups": float(sum(len(self._current_snapshot.groups_for(agent)) for agent in self._current_snapshot.agent_ids)),
            "TI_MARL/ports": float(
                sum(len(group.ports) for group in self._current_snapshot.action_groups)
            ),
            "TI_MARL/invalid_port_rate": self._invalid_port_rate(self._current_snapshot),
            "TI_MARL/intervention_rate": float(intervention_count) / float(total_groups),
            "TI_MARL/raw_final_infeasibility": float(intervention_count) / float(total_groups),
            "TI_MARL/intervention_magnitude": float(intervention_magnitude),
            "TI_MARL/fallback_rate": float(
                sum(item.get("reason") == "invalid_port_fallback" for bundle in final_bundles for item in bundle.interventions)
            ) / float(total_groups),
            "TI_MARL/parameter_count": float(self._parameter_count),
            "TI_MARL/effective_gamma": float(self.learner.gamma),
            "TI_MARL/effective_gae_lambda": float(self.learner.gae_lambda),
            "TI_MARL/structure_recompilations": float(
                self.compiler.structure_recompilations
            ),
            "TI_MARL/binding_errors": float(
                sum(
                    bool(part.validity_reasons)
                    for part in self._current_snapshot.observation_parts
                )
            ),
            "TI_MARL/detection_latency_seconds": (
                float(np.mean(active_durations)) if active_durations else 0.0
            ),
            "TI_MARL/recovery_latency_seconds": (
                float(np.max(recovery_pending)) if recovery_pending else 0.0
            ),
        }
        self._pending = {
            "snapshot": self._current_snapshot,
            "raw_bundles": raw_bundles,
            "final_bundles": final_bundles,
            "commands": commands,
            "typed_commands": typed_commands,
            "old_log_probs": {
                key: float(value.detach().cpu()) for key, value in evaluation.log_prob_by_agent.items()
            },
            "values": {key: float(value.detach().cpu()) for key, value in values.items()},
            "group_values": {
                agent_id: {
                    group_id: float(value.detach().cpu())
                    for group_id, value in values_by_group.items()
                }
                for agent_id, values_by_group in group_values.items()
            },
        }
        return commands

    def _action_mode_diagnostics(
        self,
        raw_bundles,
        final_bundles,
    ) -> Mapping[str, float]:
        """Report cumulative learned versus post-feasibility action rates."""

        assert self._current_snapshot is not None
        group_types = {
            (group.owner_agent_id, group.group_id): group.group_type
            for group in self._current_snapshot.action_groups
        }
        for bundles, counter, fraction_sums in (
            (
                raw_bundles,
                self._episode_raw_modes,
                self._episode_raw_fraction_sums,
            ),
            (
                final_bundles,
                self._episode_final_modes,
                self._episode_final_fraction_sums,
            ),
        ):
            for bundle in bundles:
                for decision in bundle.decisions:
                    group_type = group_types.get(
                        (bundle.agent_id, decision.group_id),
                        "unknown",
                    )
                    counter[(group_type, decision.mode)] += 1
                    fraction_sums[(group_type, decision.mode)] += abs(
                        float(decision.fraction)
                    )

        metrics: dict[str, float] = {}
        group_types_seen = sorted(
            {
                group_type
                for group_type, _mode in (
                    set(self._episode_raw_modes) | set(self._episode_final_modes)
                )
            }
        )
        for label, counter, fraction_sums in (
            (
                "raw",
                self._episode_raw_modes,
                self._episode_raw_fraction_sums,
            ),
            (
                "final",
                self._episode_final_modes,
                self._episode_final_fraction_sums,
            ),
        ):
            for group_type in group_types_seen:
                total = sum(
                    count
                    for (candidate_type, _mode), count in counter.items()
                    if candidate_type == group_type
                )
                if total <= 0:
                    continue
                modes = sorted(
                    mode
                    for candidate_type, mode in counter
                    if candidate_type == group_type
                )
                for mode in modes:
                    metrics[
                        f"TI_MARL/{label}_mode_{group_type.lower()}_"
                        f"{mode.lower()}_rate"
                    ] = float(counter[(group_type, mode)]) / float(total)
                metrics[
                    f"TI_MARL/{label}_non_idle_rate_{group_type.lower()}"
                ] = float(
                    sum(
                        count
                        for (candidate_type, mode), count in counter.items()
                        if candidate_type == group_type and mode != "IDLE"
                    )
                ) / float(total)
                non_idle_total = sum(
                    count
                    for (candidate_type, mode), count in counter.items()
                    if candidate_type == group_type and mode != "IDLE"
                )
                metrics[
                    f"TI_MARL/{label}_mean_abs_fraction_{group_type.lower()}"
                ] = float(
                    sum(
                        value
                        for (candidate_type, _mode), value in fraction_sums.items()
                        if candidate_type == group_type
                    )
                ) / float(total)
                metrics[
                    f"TI_MARL/{label}_non_idle_mean_abs_fraction_"
                    f"{group_type.lower()}"
                ] = (
                    float(
                        sum(
                            value
                            for (candidate_type, mode), value in fraction_sums.items()
                            if candidate_type == group_type and mode != "IDLE"
                        )
                    )
                    / float(non_idle_total)
                    if non_idle_total > 0
                    else 0.0
                )
        return metrics

    def _ev_actor_control_diagnostics(
        self,
        snapshot,
        raw_bundles,
        final_bundles,
    ) -> Mapping[str, float]:
        """Distinguish learned EV control from safety-projector takeovers."""

        ev_groups = {
            (group.owner_agent_id, group.group_id): group
            for group in snapshot.action_groups
            if group.group_type == "ev_session"
        }
        raw = {
            (bundle.agent_id, decision.group_id): decision
            for bundle in raw_bundles
            for decision in bundle.decisions
            if (bundle.agent_id, decision.group_id) in ev_groups
        }
        final = {
            (bundle.agent_id, decision.group_id): decision
            for bundle in final_bundles
            for decision in bundle.decisions
            if (bundle.agent_id, decision.group_id) in ev_groups
        }
        self._episode_ev_control["groups"] += len(ev_groups)
        for key, final_decision in final.items():
            label = {
                "CHARGE_EV": "charge",
                "DISCHARGE_EV": "discharge",
            }.get(final_decision.mode)
            if label is None:
                continue
            self._episode_ev_control[f"final_{label}"] += 1
            raw_decision = raw.get(key)
            if (
                raw_decision is not None
                and raw_decision.mode == final_decision.mode
            ):
                self._episode_ev_control[f"actor_{label}"] += 1
            else:
                self._episode_ev_control[
                    f"projector_{label}_takeover"
                ] += 1

        planner = self.learner.ev_planner
        if planner is not None:
            for target in planner.targets(
                snapshot,
                seconds_per_time_step=self.learner.seconds_per_time_step,
            ):
                self._episode_ev_control["targets"] += 1
                label = {
                    "CHARGE_EV": "charge",
                    "DISCHARGE_EV": "discharge",
                    "IDLE": "idle",
                }[target.decision.mode]
                self._episode_ev_control[f"targets_{label}"] += 1
                raw_decision = raw.get((target.agent_id, target.group_id))
                if (
                    raw_decision is not None
                    and raw_decision.mode == target.decision.mode
                ):
                    self._episode_ev_control["agreements"] += 1
                    self._episode_ev_control[f"agreements_{label}"] += 1

        def ratio(numerator: str, denominator: str) -> float:
            total = self._episode_ev_control[denominator]
            return (
                float(self._episode_ev_control[numerator]) / float(total)
                if total > 0
                else 0.0
            )

        return {
            "TI_MARL/ev_planning_target_coverage_rate": ratio(
                "targets", "groups"
            ),
            "TI_MARL/ev_planning_mode_agreement_rate": ratio(
                "agreements", "targets"
            ),
            "TI_MARL/ev_planning_charge_recall": ratio(
                "agreements_charge", "targets_charge"
            ),
            "TI_MARL/ev_planning_discharge_recall": ratio(
                "agreements_discharge", "targets_discharge"
            ),
            "TI_MARL/ev_planning_idle_recall": ratio(
                "agreements_idle", "targets_idle"
            ),
            "TI_MARL/ev_actor_charge_ownership_rate": ratio(
                "actor_charge", "final_charge"
            ),
            "TI_MARL/ev_projector_charge_takeover_rate": ratio(
                "projector_charge_takeover", "final_charge"
            ),
            "TI_MARL/ev_actor_discharge_ownership_rate": ratio(
                "actor_discharge", "final_discharge"
            ),
            "TI_MARL/ev_projector_discharge_takeover_rate": ratio(
                "projector_discharge_takeover", "final_discharge"
            ),
        }

    def _reward_components_by_agent(
        self,
        snapshot,
    ) -> Mapping[str, Mapping[str, float]]:
        """Align unflattened reward evidence with stable typed agent IDs."""

        payload = self._transition_info.get("reward_components", {})
        rows = payload.get("per_agent", []) if isinstance(payload, Mapping) else []
        if not isinstance(rows, (list, tuple)):
            return {}
        aligned: dict[str, dict[str, float]] = {}
        for index, agent_id in enumerate(snapshot.agent_ids):
            if index >= len(rows) or not isinstance(rows[index], Mapping):
                continue
            numeric: dict[str, float] = {}
            for key, value in rows[index].items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    numeric[str(key)] = float(value)
            aligned[agent_id] = numeric
        return aligned

    def update(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        del observations, next_observations, update_target_step, global_learning_step, initial_exploration_done
        if self._next_snapshot is None:
            raise RuntimeError("TIMARL update requires a complete typed transition context")
        if self._in_demonstration_phase():
            assert self.behavior_cloning is not None
            assert self._current_snapshot is not None
            bundles = self.codec.decode_teacher_actions(
                self._current_snapshot,
                actions,
                group_modes=self.actor.group_modes,
            )
            self.behavior_cloning.record(self._current_snapshot, bundles)
            self._latest_training_metrics = {
                "TI_MARL/teacher_action_execution": 1.0,
                **{
                    f"TI_MARL/{key}": float(value)
                    for key, value in self.behavior_cloning.metrics().items()
                },
            }
            self._current_snapshot = self._next_snapshot
            self._next_snapshot = None
            self._pending = None
            return
        if self._pending is None:
            raise RuntimeError("TIMARL update requires a pending actor decision")
        current = self._pending["snapshot"]
        following = self._next_snapshot
        reward_by_agent = {
            agent_id: float(rewards[index]) if index < len(rewards) else 0.0
            for index, agent_id in enumerate(current.agent_ids)
        }
        with torch.no_grad():
            next_values = {
                key: float(value.detach().cpu()) for key, value in self.critic(following).items()
            }
            next_group_values = (
                {}
                if self.group_critic is None
                else {
                    agent_id: {
                        group_id: float(value.detach().cpu())
                        for group_id, value in values_by_group.items()
                    }
                    for agent_id, values_by_group in self.group_critic(
                        following
                    ).items()
                }
            )
        removed = set(current.agent_ids) - set(following.agent_ids)
        if terminated:
            removed.update(current.agent_ids)
        step = RolloutStep(
            snapshot=current,
            next_snapshot=following,
            bundles=tuple(self._pending["raw_bundles"]),
            old_log_probs=dict(self._pending["old_log_probs"]),
            values=dict(self._pending["values"]),
            next_values=next_values,
            rewards=reward_by_agent,
            terminated_agent_ids=tuple(sorted(removed)),
            truncated=bool(truncated),
            reward_components_by_agent=self._reward_components_by_agent(current),
            group_values=dict(self._pending.get("group_values", {})),
            next_group_values=next_group_values,
            final_bundles=tuple(self._pending["final_bundles"]),
        )
        self.learner.rollout.add(step)
        if update_step and self.learner.ready():
            self.actor.train()
            self.critic.train()
            if self.group_critic is not None:
                self.group_critic.train()
            metrics = self.learner.update(
                progress_callback=self._training_progress_callback,
            )
            self._latest_training_metrics = {
                f"TI_MARL/train_{key}": float(value) for key, value in metrics.items()
            }

        self._record_typed_transition(
            current=current,
            following=following,
            reward_by_agent=reward_by_agent,
            removed=removed,
        )

    def observe_transition(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Persist one deterministic transition without updating either critic."""

        del observations, actions, next_observations, truncated
        if self._next_snapshot is None:
            raise RuntimeError(
                "TIMARL observation requires a complete typed transition context"
            )
        if self._pending is None:
            raise RuntimeError("TIMARL observation requires a pending actor decision")
        current = self._pending["snapshot"]
        following = self._next_snapshot
        reward_by_agent = {
            agent_id: float(rewards[index]) if index < len(rewards) else 0.0
            for index, agent_id in enumerate(current.agent_ids)
        }
        removed = set(current.agent_ids) - set(following.agent_ids)
        if terminated:
            removed.update(current.agent_ids)
        self._record_typed_transition(
            current=current,
            following=following,
            reward_by_agent=reward_by_agent,
            removed=removed,
        )

    def _record_typed_transition(
        self,
        *,
        current,
        following,
        reward_by_agent: Mapping[str, float],
        removed: set[str],
    ) -> None:
        bootstrap = set(current.agent_ids) & set(following.agent_ids) - removed
        typed_transition = TypedTransition(
            snapshot_hash=current.snapshot_hash,
            next_snapshot_hash=following.snapshot_hash,
            agent_ids=current.agent_ids,
            next_agent_ids=following.agent_ids,
            raw_bundles=tuple(self._pending["raw_bundles"]),
            final_bundles=tuple(self._pending["final_bundles"]),
            commands=tuple(tuple(float(value) for value in command) for command in self._pending["commands"]),
            execution=dict(self._transition_info.get("entity_action_execution", {})),
            rewards=tuple(sorted(reward_by_agent.items())),
            reward_components=dict(self._transition_info.get("reward_components", {})),
            terminated_agent_ids=tuple(sorted(removed)),
            bootstrap_agent_ids=tuple(sorted(bootstrap)),
            health_events=tuple(canonical_value(item) for item in following.closure_log),
            topology_events=tuple(
                canonical_value(item) for item in self._transition_info.get("topology_events_applied", []) or []
            ),
            typed_commands=tuple(
                canonical_value(item) for item in self._pending["typed_commands"]
            ),
        )
        self.trace_writer.record(current, following, typed_transition)
        execution = self._transition_info.get("entity_action_execution", {})
        execution_entries = execution.get("entries", []) if isinstance(execution, Mapping) else []
        requested_applied_errors = [
            abs(float(item["requested_value"]) - float(item["applied_value"]))
            for item in execution_entries
            if item.get("requested_value") is not None and item.get("applied_value") is not None
        ]
        self._latest_diagnostics["TI_MARL/requested_applied_action_error"] = (
            float(np.mean(requested_applied_errors)) if requested_applied_errors else 0.0
        )
        self._latest_diagnostics["TI_MARL/trace_completeness"] = float(
            execution.get("version") == "entity_action_execution_v1"
            and typed_transition.snapshot_hash == current.snapshot_hash
            and typed_transition.next_snapshot_hash == following.snapshot_hash
        )
        self._current_snapshot = following
        self._next_snapshot = None
        self._pending = None

    def on_episode_end(self, *, episode: int, training: bool) -> None:
        if (
            self.behavior_cloning is not None
            and training
            and not self.behavior_cloning.pretraining_complete
            and int(episode) + 1 == self.behavior_cloning.demonstration_episodes
        ):
            metrics = self.behavior_cloning.pretrain(
                self.actor,
                max_grad_norm=self.learner.max_grad_norm,
                progress_callback=self._training_progress_callback,
            )
            self.learner.reset_policy_anchor()
            self._latest_training_metrics.update(
                {f"TI_MARL/{key}": float(value) for key, value in metrics.items()}
            )
        if training and len(self.learner.rollout):
            self.actor.train()
            self.critic.train()
            if self.group_critic is not None:
                self.group_critic.train()
            metrics = self.learner.update()
            self._latest_training_metrics = {
                f"TI_MARL/train_{key}": float(value) for key, value in metrics.items()
            }
        self.trace_writer.flush()

    def set_training_progress_callback(
        self,
        callback: Optional[Callable[[Mapping[str, Any]], None]],
    ) -> None:
        """Install best-effort progress telemetry for long boundary training.

        The callback is deliberately supplied by the runner rather than kept
        in checkpoints.  It lets behavior cloning expose epoch-level liveness
        without coupling the TI-MARL learner to the job orchestrator.
        """

        self._training_progress_callback = callback

    def on_episode_start(self, *, episode: int, training: bool) -> None:
        self._current_episode = int(episode)
        self._current_episode_is_training = bool(training)
        self.compiler.reset_runtime_state()
        self._current_snapshot = None
        self._next_snapshot = None
        self._pending = None
        self._episode_raw_modes.clear()
        self._episode_final_modes.clear()
        self._episode_raw_fraction_sums.clear()
        self._episode_final_fraction_sums.clear()
        self._episode_ev_control.clear()

    def consume_latest_training_metrics(self) -> Mapping[str, float]:
        metrics = dict(self._latest_training_metrics)
        self._latest_training_metrics.clear()
        return metrics

    def get_diagnostic_metrics(self) -> Mapping[str, float]:
        return dict(self._latest_diagnostics)

    def save_checkpoint(self, output_dir: str, step: int) -> str:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "latest_checkpoint.pth"
        temporary_path = root / f".{path.name}.tmp"
        payload = {
            "format": "ti_marl_checkpoint_v1",
            "step": int(step),
            "learning_architecture": {
                "backbone": self.backbone_name,
                "critic": self.critic_kind,
                "actor_group_context": self.actor_group_context_kind,
                "policy_credit_assignment": self.policy_credit_assignment,
            },
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "group_critic": (
                None
                if self.group_critic is None
                else self.group_critic.state_dict()
            ),
            "learner": self.learner.state_dict(),
            "compiler_state": self.compiler.checkpoint_state(),
            "compatibility_signature": asdict(self.compiler.compatibility_signature),
            "configuration": deepcopy(self.config),
            "versions": self._versions(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "normalizers": {},
            "behavior_cloning": (
                None
                if self.behavior_cloning is None
                else self.behavior_cloning.state_dict()
            ),
        }
        # Replace the latest checkpoint atomically.  Apart from protecting a
        # previously valid checkpoint from interrupted writes, this lets the
        # checkpoint manager preserve episode boundaries with hard links:
        # replacing ``latest`` then creates a new inode and cannot mutate an
        # older episode checkpoint.
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / "latest_checkpoint.pth"
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("format") != "ti_marl_checkpoint_v1":
            raise ValueError("Unsupported TIMARL checkpoint format")
        if (
            self.behavior_cloning is not None
            and payload.get("behavior_cloning") is None
        ):
            raise ValueError(
                "TI-MARL checkpoint has no behavior-cloning state while the "
                "current configuration enables behavior cloning. Disable "
                "behavior_cloning for continuation/inference, or resume from "
                "a checkpoint that preserves the teacher phase state."
            )
        architecture = dict(
            payload.get(
                "learning_architecture",
                {"backbone": "mappo", "critic": "set"},
            )
        )
        expected_architecture = {
            "backbone": self.backbone_name,
            "critic": self.critic_kind,
            "actor_group_context": self.actor_group_context_kind,
            "policy_credit_assignment": self.policy_credit_assignment,
        }
        architecture.setdefault("actor_group_context", "local")
        architecture.setdefault("policy_credit_assignment", "joint_agent")
        if architecture != expected_architecture:
            raise ValueError(
                "TIMARL checkpoint learning architecture does not match: "
                f"checkpoint={architecture}, configured={expected_architecture}"
            )
        raw_signature = dict(payload["compatibility_signature"])
        for key in ("supported_module_types", "supported_action_group_types"):
            raw_signature[key] = tuple(raw_signature[key])
        checkpoint_signature = CompatibilitySignature(**raw_signature)
        compatible = self.compiler.compatibility_signature.accepts(
            checkpoint_signature
        )
        compiler_migration = (
            self.allow_checkpoint_compiler_migration
            and self.compiler.compatibility_signature.accepts_explicit_compiler_migration(
                checkpoint_signature
            )
        )
        if not compatible and not compiler_migration:
            raise ValueError("TIMARL checkpoint compatibility signature is not accepted")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if self.group_critic is not None:
            group_critic_state = payload.get("group_critic")
            if group_critic_state is None:
                raise ValueError(
                    "TIMARL typed group checkpoint is missing its critic"
                )
            self.group_critic.load_state_dict(group_critic_state)
        self.learner.load_state_dict(
            payload["learner"],
            restore_optimizers=self.restore_optimizers,
            restore_rollout=self.restore_rollout,
        )
        if self.policy_anchor_reset_on_resume:
            self.learner.reset_policy_anchor()
        self.compiler.load_checkpoint_state(payload.get("compiler_state", {}))
        if self.behavior_cloning is not None:
            self.behavior_cloning.load_state_dict(payload["behavior_cloning"])
        rng = payload.get("rng", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda"):
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del context
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.trace_writer.close()
        model_path = root / "ti_marl_model.pth"
        deployment_path = root / "ti_marl_deployment_bundle.pth"
        interface_path = root / "typed_interfaces.resolved.yaml"
        with interface_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.compiler.resolved_typed_interface(),
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        torch.save(
            {
                "format": "ti_marl_torch",
                "deployable": False,
                "learning_architecture": {
                    "backbone": self.backbone_name,
                    "critic": self.critic_kind,
                    "policy_credit_assignment": self.policy_credit_assignment,
                },
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "group_critic": (
                    None
                    if self.group_critic is None
                    else self.group_critic.state_dict()
                ),
                "compatibility_signature": asdict(self.compiler.compatibility_signature),
                "versions": self._versions(),
            },
            model_path,
        )
        # Actor-only, technology-neutral handoff. The critic and optimizers are
        # intentionally absent; adapters provide runtime frames in deployment.
        torch.save(
            {
                "format": "ti_marl_deployment_bundle_v1",
                "actor": self.actor.state_dict(),
                "training_backbone": self.backbone_name,
                "actor_group_context": self.actor_group_context_kind,
                "policy_credit_assignment": self.policy_credit_assignment,
                "behavior_cloning_warm_start": self.behavior_cloning is not None,
                "typed_interfaces": self.compiler.resolved_typed_interface(),
                "compiler": {
                    "version": self._versions()["algorithms"],
                    "contract_version": self.compiler.contract_version,
                    "health_rules": deepcopy(self.compiler.health_rules),
                },
                "feasibility": dict(self.projector.configuration()),
                "normalisation": {"kind": "per_observation_declared"},
                "compatibility_signature": asdict(self.compiler.compatibility_signature),
                "versions": self._versions(),
            },
            deployment_path,
        )
        return {
            "format": "ti_marl_torch",
            "deployable": False,
            "model_path": model_path.name,
            "typed_interfaces_path": interface_path.name,
            "deployment_bundle_path": deployment_path.name,
            "auxiliary_files": [
                {
                    "path": interface_path.name,
                    "format": "typed_interface_registry_v1",
                    "editable": True,
                },
                {
                    "path": deployment_path.name,
                    "format": "ti_marl_deployment_bundle_v1",
                    "contains_critic": False,
                },
            ],
            "contract_version": self.compiler.contract_version,
            "learning_architecture": {
                "backbone": self.backbone_name,
                "critic": self.critic_kind,
                "policy_credit_assignment": self.policy_credit_assignment,
            },
            "compatibility_signature": asdict(self.compiler.compatibility_signature),
            "trace": dict(self.trace_writer.manifest()),
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": model_path.name,
                    "format": "ti_marl_torch",
                    "deployable": False,
                    "config": {"deployable": False},
                }
            ],
        }

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        del global_learning_step
        return True

    def _in_demonstration_phase(self) -> bool:
        return bool(
            self.behavior_cloning is not None
            and self.behavior_cloning.in_demonstration_phase(
                episode=self._current_episode,
                training=self._current_episode_is_training,
            )
        )

    @staticmethod
    def _invalid_port_rate(snapshot) -> float:
        ports = [port for group in snapshot.action_groups for port in group.ports if port.mode != "IDLE"]
        if not ports:
            return 0.0
        return float(sum(not port.valid for port in ports)) / float(len(ports))

    @staticmethod
    def _versions() -> Mapping[str, str]:
        try:
            import citylearn

            simulator_version = str(citylearn.__version__)
        except (ImportError, AttributeError):
            simulator_version = "unknown"
        return {
            "algorithms": os.environ.get("ALGORITHMS_VERSION", "working-tree"),
            "simulator": simulator_version,
        }
