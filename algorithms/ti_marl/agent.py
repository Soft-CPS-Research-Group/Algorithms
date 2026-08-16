"""Runner integration for the standalone ``TIMARL`` execution unit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from algorithms.agents.base_agent import BaseAgent
from algorithms.ti_marl.compiler.compiler import TypedInterfaceCompiler
from algorithms.ti_marl.contracts.compatibility import CompatibilitySignature
from algorithms.ti_marl.contracts.models import TypedTransition, canonical_value
from algorithms.ti_marl.learning.mappo import TIMAPPO
from algorithms.ti_marl.learning.rollout import RolloutStep
from algorithms.ti_marl.policy.networks import CentralSetCritic, TypedActor, parameter_count
from algorithms.ti_marl.runtime.codec import CityLearnTypedActionCodec
from algorithms.ti_marl.runtime.feasibility import AnalyticLocalProjector
from algorithms.ti_marl.runtime.traces import BufferedTraceWriter


class TIMARL(BaseAgent):
    """Typed Interface MARL with a TI-MAPPO first backbone."""

    supports_dynamic_topology = True
    handles_cross_topology_transitions = True
    requires_entity_observation_context = True
    requires_final_pipeline_stage = True
    _use_raw_observations = False

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = deepcopy(config)
        hyper = dict(config.get("algorithm", {}).get("hyperparameters", {}))
        backbone = dict(hyper.get("backbone", {}))
        if str(backbone.get("name", "mappo")).lower() != "mappo":
            raise ValueError("TIMARL v1 supports backbone.name='mappo' only")

        self.seed = int(config.get("training", {}).get("seed", 22))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        require_cuda = bool(hyper.get("require_cuda", False))
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("TIMARL require_cuda=true but CUDA is unavailable")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.compiler = TypedInterfaceCompiler(
            contract_version=str(hyper.get("contract_version", "ti_marl_v1")),
            agent_schema_path=str(hyper["agent_schema_path"]),
            type_registry_path=str(hyper["type_registry_path"]),
            health_rules_path=str(hyper["health_rules_path"]),
        )
        actor_cfg = dict(hyper.get("actor", {}))
        d_model = int(actor_cfg.get("d_model", 128))
        relation_layers = int(actor_cfg.get("relation_layers", 2))
        self.actor = TypedActor(
            self.compiler.type_registry,
            d_model=d_model,
            attention_heads=int(actor_cfg.get("attention_heads", 4)),
            relation_layers=relation_layers,
        ).to(self.device)
        self.critic = CentralSetCritic(
            self.compiler.type_registry,
            d_model=d_model,
            relation_layers=relation_layers,
        ).to(self.device)
        self._parameter_count = parameter_count(self.actor) + parameter_count(self.critic)
        self.learner = TIMAPPO(
            self.actor,
            self.critic,
            learning_rate=float(hyper.get("learning_rate", 3.0e-4)),
            gamma=float(hyper.get("gamma", 0.99)),
            gae_lambda=float(hyper.get("gae_lambda", 0.95)),
            clip_eps=float(hyper.get("clip_eps", 0.2)),
            ppo_epochs=int(hyper.get("ppo_epochs", 4)),
            entropy_coeff=float(hyper.get("entropy_coeff", 0.01)),
            value_coeff=float(hyper.get("value_coeff", 0.5)),
            max_grad_norm=float(hyper.get("max_grad_norm", 0.5)),
            target_kl=(None if hyper.get("target_kl", 0.03) is None else float(hyper.get("target_kl", 0.03))),
            rollout_steps=int(hyper.get("rollout_steps", 256)),
        )
        self.projector = AnalyticLocalProjector()
        self.codec = CityLearnTypedActionCodec()

        trace_cfg = dict(hyper.get("trace", {}))
        job_dir = config.get("runtime", {}).get("job_dir")
        trace_dir = None if not job_dir else Path(job_dir) / "results" / "ti_marl_trace"
        self.trace_writer = BufferedTraceWriter(
            trace_dir,
            chunk_size=int(trace_cfg.get("chunk_size", 256)),
            snapshot_interval=int(trace_cfg.get("snapshot_interval", 256)),
            enabled=bool(trace_cfg.get("enabled", True)),
        )

        self._current_snapshot = None
        self._next_snapshot = None
        self._transition_info: Mapping[str, Any] = {}
        self._pending: Optional[Dict[str, Any]] = None
        self._building_names: tuple[str, ...] = ()
        self._latest_training_metrics: Dict[str, float] = {}
        self._latest_diagnostics: Dict[str, float] = {}

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        del observation_names, observation_space
        metadata = dict(metadata or {})
        if metadata.get("interface") != "entity":
            raise ValueError("TIMARL requires simulator.interface='entity'")
        topology_mode = str(metadata.get("topology_mode", "static"))
        if topology_mode not in {"static", "dynamic"}:
            raise ValueError(f"Unsupported TIMARL topology mode: {topology_mode}")
        entity_specs = metadata.get("entity_specs")
        if not isinstance(entity_specs, Mapping):
            raise ValueError("TIMARL attach_environment requires entity_specs")
        self.compiler.attach_entity_specs(entity_specs)
        building_names = metadata.get("building_names") or entity_specs.get("tables", {}).get("building", {}).get("ids", [])
        self._building_names = tuple(str(item) for item in building_names)
        self.codec.attach(
            building_names=self._building_names,
            action_names=action_names,
            action_space=action_space,
        )
        if parameter_count(self.actor) + parameter_count(self.critic) != self._parameter_count:
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
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            evaluation = self.actor(
                self._current_snapshot,
                deterministic=bool(deterministic),
            )
            values = self.critic(self._current_snapshot)
        raw_bundles = evaluation.bundles
        final_bundles = self.projector.project(self._current_snapshot, raw_bundles)
        self.projector.assert_feasible(self._current_snapshot, final_bundles)
        commands = self.codec.encode(self._current_snapshot, final_bundles)
        intervention_count = sum(len(bundle.interventions) for bundle in final_bundles)
        intervention_magnitude = sum(
            float(item.get("magnitude", 0.0))
            for bundle in final_bundles
            for item in bundle.interventions
        )
        total_groups = max(sum(len(bundle.decisions) for bundle in raw_bundles), 1)
        active_durations = [
            float(item.active_duration_steps)
            for item in self._current_snapshot.fault_evidence
            if item.fault_mode is not None
        ]
        recovery_pending = [
            float(item.recovery_pending_steps)
            for item in self._current_snapshot.health
            if item.recovery_pending_steps > 0
        ]
        self._latest_diagnostics = {
            **self._latest_diagnostics,
            "TI_MARL/agents": float(len(self._current_snapshot.agent_ids)),
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
            "TI_MARL/binding_errors": 0.0,
            "TI_MARL/detection_latency_steps": (
                float(np.mean(active_durations)) if active_durations else 0.0
            ),
            "TI_MARL/recovery_latency_steps": (
                float(np.max(recovery_pending)) if recovery_pending else 0.0
            ),
        }
        self._pending = {
            "snapshot": self._current_snapshot,
            "raw_bundles": raw_bundles,
            "final_bundles": final_bundles,
            "commands": commands,
            "old_log_probs": {
                key: float(value.detach().cpu()) for key, value in evaluation.log_prob_by_agent.items()
            },
            "values": {key: float(value.detach().cpu()) for key, value in values.items()},
        }
        return commands

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
        del observations, actions, next_observations, update_target_step, global_learning_step, initial_exploration_done
        if self._pending is None or self._next_snapshot is None:
            raise RuntimeError("TIMARL update requires a complete typed transition context")
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
        removed = set(current.agent_ids) - set(following.agent_ids)
        if terminated:
            removed.update(current.agent_ids)
        bootstrap = set(current.agent_ids) & set(following.agent_ids) - removed
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
        )
        self.learner.rollout.add(step)
        if update_step and self.learner.ready():
            self.actor.train()
            self.critic.train()
            metrics = self.learner.update()
            self._latest_training_metrics = {
                f"TI_MARL/train_{key}": float(value) for key, value in metrics.items()
            }

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
        del episode
        if training and len(self.learner.rollout):
            self.actor.train()
            self.critic.train()
            metrics = self.learner.update()
            self._latest_training_metrics = {
                f"TI_MARL/train_{key}": float(value) for key, value in metrics.items()
            }
        self.trace_writer.flush()

    def on_episode_start(self, *, episode: int, training: bool) -> None:
        del episode, training
        self.compiler.health_deriver.reset()
        self._current_snapshot = None
        self._next_snapshot = None
        self._pending = None

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
        payload = {
            "format": "ti_marl_checkpoint_v1",
            "step": int(step),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
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
        }
        torch.save(payload, path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / "latest_checkpoint.pth"
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("format") != "ti_marl_checkpoint_v1":
            raise ValueError("Unsupported TIMARL checkpoint format")
        raw_signature = dict(payload["compatibility_signature"])
        for key in ("supported_module_types", "supported_action_group_types"):
            raw_signature[key] = tuple(raw_signature[key])
        checkpoint_signature = CompatibilitySignature(**raw_signature)
        if not self.compiler.compatibility_signature.accepts(checkpoint_signature):
            raise ValueError("TIMARL checkpoint compatibility signature is not accepted")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.learner.load_state_dict(payload["learner"])
        self.compiler.load_checkpoint_state(payload.get("compiler_state", {}))
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
        torch.save(
            {
                "format": "ti_marl_torch",
                "deployable": False,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "compatibility_signature": asdict(self.compiler.compatibility_signature),
                "versions": self._versions(),
            },
            model_path,
        )
        return {
            "format": "ti_marl_torch",
            "deployable": False,
            "model_path": model_path.name,
            "contract_version": self.compiler.contract_version,
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
