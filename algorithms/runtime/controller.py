from __future__ import annotations

import faulthandler
import inspect
import json
import re
import sys
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import psutil
import torch
from citylearn.agents.rlc import RLC
from citylearn.base import Environment
from citylearn.citylearn import CityLearnEnv
from loguru import logger

from algorithms.contracts import Transition, UpdateSchedule
from algorithms.execution_unit import ExecutionUnit
from algorithms.runtime.diagnostics import Diagnostics
from algorithms.runtime.interaction import decide, learn_transition
from algorithms.runtime.resources import ResourceLifecycle
from algorithms.runtime.simulator import SimulatorAdapter
from algorithms.runtime.tracking import mlflow
from utils.checkpoint_manager import CheckpointManager
from utils.entity_adapter import EntityContractAdapter
from utils.local_metrics import LocalMetricsLogger
from utils.preprocessing import (
    Encoder,
    NoNormalization,
    Normalize,
    NormalizeWithMissing,
    OnehotEncoding,
    PeriodicNormalization,
    RemoveFeature,
)
from utils.progress_tracker import ProgressTracker

ENCODER_RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "encoders" / "default.json"


ENCODER_TYPE_MAP: Dict[str, type[Encoder]] = {
    "NoNormalization": NoNormalization,
    "Normalize": Normalize,
    "NormalizeWithMissing": NormalizeWithMissing,
    "OnehotEncoding": OnehotEncoding,
    "PeriodicNormalization": PeriodicNormalization,
    "RemoveFeature": RemoveFeature,
}

WRAPPER_REWARD_PROFILES: Dict[str, Dict[str, Any]] = {
    "cost_limits_v1": {
        "version": "cost_limits_v1.0.0",
        "enabled_terms": {
            "energy_cost": True,
            "grid_violation": True,
            "ev_success": True,
            "community": True,
        },
        "weights": {
            "energy_cost": 1.0,
            "grid_violation": 1.0,
            "ev_success": 0.5,
            "community": 0.05,
        },
        "params": {
            "export_credit_ratio": 0.8,
            "community_export_bonus_ratio": 0.2,
            "ev_soc_tolerance": 0.1,
        },
    },
}


@lru_cache(maxsize=1)
def _load_encoder_rules() -> List[Dict[str, Any]]:
    if not ENCODER_RULES_PATH.exists():
        raise FileNotFoundError(f"Encoder rules file not found: {ENCODER_RULES_PATH}")
    with ENCODER_RULES_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rules = data.get("rules", [])
    if not rules:
        raise ValueError("Encoder rules configuration must define at least one rule")
    return rules


def _matches_rule(name: str, match_spec: Dict[str, Any]) -> bool:
    equals = match_spec.get("equals")
    if equals is not None and name in equals:
        return True

    contains = match_spec.get("contains")
    if contains is not None and any(token in name for token in contains):
        return True

    prefixes = match_spec.get("prefixes")
    if prefixes is not None and any(name.startswith(prefix) for prefix in prefixes):
        return True

    suffixes = match_spec.get("suffixes")
    if suffixes is not None and any(name.endswith(suffix) for suffix in suffixes):
        return True

    return bool(match_spec.get("default", False))


def _resolve_param(value: Any, space: Any, index: int) -> Any:
    if isinstance(value, str):
        if value == "space_high":
            return np.asarray(space.high)[index]
        if value == "space_low":
            return np.asarray(space.low)[index]
    if isinstance(value, list):
        return [_resolve_param(item, space, index) for item in value]
    return value


def _build_encoder(rule: Dict[str, Any], space: Any, index: int) -> Encoder:
    encoder_spec = rule.get("encoder", {})
    encoder_type = encoder_spec.get("type")
    if encoder_type is None:
        raise ValueError(f"Encoder rule missing type definition: {rule}")
    try:
        encoder_cls = ENCODER_TYPE_MAP[encoder_type]
    except KeyError as exc:
        raise ValueError(f"Unknown encoder type '{encoder_type}' in encoder rules") from exc

    raw_params = encoder_spec.get("params", {})
    params = {key: _resolve_param(value, space, index) for key, value in raw_params.items()}
    return encoder_cls(**params) if params else encoder_cls()


class SimulationController(ResourceLifecycle, Diagnostics, RLC):
    def __init__(
        self,
        env: CityLearnEnv,
        model: Optional[ExecutionUnit] = None,
        config=None,
        job_id=None,
        progress_path=None,
        **kwargs,
    ):
        """Wrapper for CityLearn RLC that delegates predict/update to a model.

        The ``model`` is any :class:`ExecutionUnit` — a single agent, a
        :class:`~algorithms.pipeline.Pipeline`, or an
        :class:`~algorithms.pipeline.Ensemble`. The wrapper interacts with
        the same surface regardless of the underlying architecture.
        """
        config = config or {}
        self.simulator = SimulatorAdapter(env)
        self.config = config
        simulator_cfg = config.get("simulator", {})
        interface_mode = (
            str(simulator_cfg.get("interface", getattr(env, "interface", "flat"))).strip().lower()
            or "flat"
        )
        self._entity_interface_mode = interface_mode == "entity"
        self._entity_topology_mode = (
            str(simulator_cfg.get("topology_mode", getattr(env, "topology_mode", "static")))
            .strip()
            .lower()
            or "static"
        )
        self._entity_dynamic_mode = (
            self._entity_interface_mode and self._entity_topology_mode == "dynamic"
        )
        self._repeat_episode_scenario = bool(simulator_cfg.get("repeat_episode_scenario", False))
        # Names of every algorithm that appears anywhere in the pipeline.
        # Used to gate behaviour (e.g. MADDPG cannot run with dynamic
        # entity topology) without depending on a single-algorithm model.
        self._algorithm_names: set[str] = {
            str(stage.get("algorithm", "")).strip()
            for stage in (config.get("pipeline") or [])
            if isinstance(stage, dict) and stage.get("algorithm")
        }
        self._entity_topology_version: Optional[int] = None
        self._topology_changed_during_step: bool = False
        self._pending_model_environment_attach: bool = False
        self._defer_model_environment_attach: bool = False
        self._entity_current_payload: Optional[Mapping[str, Any]] = None
        self._entity_current_info: Mapping[str, Any] = {}
        self._entity_next_payload: Optional[Mapping[str, Any]] = None
        self._entity_step_info: Mapping[str, Any] = {}
        self._entity_adapter: Optional[EntityContractAdapter] = None
        self._entity_profile_adapters: Dict[str, EntityContractAdapter] = {}
        self._profiled_observation_cache_key: Optional[tuple[int, ...]] = None
        self._profiled_observation_cache_value: Dict[str, List[np.ndarray]] = {}
        self._episode_execution_history: List[Dict[str, Any]] = []
        self.model = model

        entity_encoding_cfg = simulator_cfg.get("entity_encoding", {}) or {}
        default_entity_encoding_enabled = self._entity_interface_mode
        self._entity_encoding_enabled = bool(
            entity_encoding_cfg.get("enabled", default_entity_encoding_enabled)
        )
        self._entity_encoding_clip = bool(entity_encoding_cfg.get("clip", True))
        self._entity_encoding_policy = str(entity_encoding_cfg.get("normalization", "minmax_space"))
        if self._entity_encoding_policy != "minmax_space":
            logger.warning(
                "Unsupported entity encoding policy '{}'; falling back to 'minmax_space'.",
                self._entity_encoding_policy,
            )
            self._entity_encoding_policy = "minmax_space"
        self._entity_encoding_profile = (
            str(entity_encoding_cfg.get("profile", self._entity_encoding_policy)).strip().lower()
            or self._entity_encoding_policy
        )

        if self._entity_interface_mode:
            self._initialize_entity_agent_state(env=env)
        else:
            super().__init__(env, **kwargs)

        self.job_id = job_id
        self.initial_exploration_done = False
        self.update_step = False
        self.update_target_step = False
        self.global_step = 0
        self._encoded_observation_cache_key: Optional[tuple[int, ...]] = None
        self._encoded_observation_cache_value: Optional[List[np.ndarray]] = None
        self._entity_model_observations_direct = False
        self._action_bounds_cache: Optional[List[tuple[np.ndarray, np.ndarray]]] = None
        training_cfg = config.get("training", {})
        checkpoint_cfg = config.get("checkpointing", {})
        tracking_cfg = config.get("tracking", {})
        export_cfg = simulator_cfg.get("export", {}) or {}
        wrapper_reward_cfg = simulator_cfg.get("wrapper_reward", {})

        self.steps_between_training_updates = training_cfg.get("steps_between_training_updates", 1)
        self.default_episodes = int(simulator_cfg.get("episodes", 1) or 1)
        if self.default_episodes < 1:
            self.default_episodes = 1
        self.target_update_interval = training_cfg.get("target_update_interval", 0)
        self.log_dir = config.get("runtime", {}).get("log_dir")
        self.mlflow_enabled = tracking_cfg.get("mlflow_enabled", False)
        try:
            self.log_frequency = int(tracking_cfg.get("log_frequency", 1) or 1)
        except (TypeError, ValueError):
            self.log_frequency = 1
        if self.log_frequency < 1:
            self.log_frequency = 1
        try:
            self.mlflow_step_sample_interval = int(
                tracking_cfg.get("mlflow_step_sample_interval", 10) or 10
            )
        except (TypeError, ValueError):
            self.mlflow_step_sample_interval = 10
        if self.mlflow_step_sample_interval < 1:
            self.mlflow_step_sample_interval = 1
        self.step_metric_interval = max(self.log_frequency, self.mlflow_step_sample_interval)
        self.progress_updates_enabled = bool(tracking_cfg.get("progress_updates_enabled", True))
        try:
            self.progress_update_interval = int(
                tracking_cfg.get("progress_update_interval", 5) or 5
            )
        except (TypeError, ValueError):
            self.progress_update_interval = 5
        if self.progress_update_interval < 1:
            self.progress_update_interval = 1
        self.system_metrics_enabled = bool(tracking_cfg.get("system_metrics_enabled", False))
        try:
            self.system_metrics_interval = int(
                tracking_cfg.get("system_metrics_interval", 10) or 10
            )
        except (TypeError, ValueError):
            self.system_metrics_interval = 10
        if self.system_metrics_interval < 1:
            self.system_metrics_interval = 10
        self.action_diagnostics_enabled = bool(
            tracking_cfg.get("action_diagnostics_enabled", False)
        )
        self.action_diagnostics_detail = (
            str(tracking_cfg.get("action_diagnostics_detail", "summary") or "summary")
            .strip()
            .lower()
        )
        if self.action_diagnostics_detail not in {"summary", "per_action"}:
            logger.warning(
                "Unknown action_diagnostics_detail '{}'; falling back to 'summary'.",
                self.action_diagnostics_detail,
            )
            self.action_diagnostics_detail = "summary"
        self.action_saturation_tolerance = self._safe_float(
            tracking_cfg.get("action_saturation_tolerance"),
            default=0.01,
        )
        if self.action_saturation_tolerance < 0:
            self.action_saturation_tolerance = 0.01
        self.action_idle_tolerance = self._safe_float(
            tracking_cfg.get("action_idle_tolerance"),
            default=0.02,
        )
        if self.action_idle_tolerance < 0:
            self.action_idle_tolerance = 0.02
        self.reward_diagnostics_enabled = bool(tracking_cfg.get("reward_diagnostics_enabled", True))
        self.reward_diagnostics_detail = (
            str(tracking_cfg.get("reward_diagnostics_detail", "summary") or "summary")
            .strip()
            .lower()
        )
        if self.reward_diagnostics_detail not in {"summary", "per_agent"}:
            logger.warning(
                "Unknown reward_diagnostics_detail '{}'; falling back to 'summary'.",
                self.reward_diagnostics_detail,
            )
            self.reward_diagnostics_detail = "summary"
        self.runtime_profiling_enabled = bool(tracking_cfg.get("runtime_profiling_enabled", False))
        try:
            self.runtime_profiling_interval = int(
                tracking_cfg.get("runtime_profiling_interval", 512) or 512
            )
        except (TypeError, ValueError):
            self.runtime_profiling_interval = 512
        if self.runtime_profiling_interval < 1:
            self.runtime_profiling_interval = 512
        self.runtime_profiling_detail = (
            str(tracking_cfg.get("runtime_profiling_detail", "summary") or "summary")
            .strip()
            .lower()
        )
        if self.runtime_profiling_detail not in {"summary", "detailed"}:
            logger.warning(
                "Unknown runtime_profiling_detail '{}'; falling back to 'summary'.",
                self.runtime_profiling_detail,
            )
            self.runtime_profiling_detail = "summary"
        self.progress_phase_updates_enabled = bool(
            tracking_cfg.get("progress_phase_updates_enabled", False)
        )
        self.progress_phase_start_step = self._coerce_non_negative_int(
            tracking_cfg.get("progress_phase_start_step")
        )
        self.progress_phase_end_step = self._coerce_non_negative_int(
            tracking_cfg.get("progress_phase_end_step")
        )
        self.max_step_seconds = self._coerce_positive_float(tracking_cfg.get("max_step_seconds"))
        self.max_update_seconds = self._coerce_positive_float(
            tracking_cfg.get("max_update_seconds")
        )
        self.stall_watchdog_enabled = bool(tracking_cfg.get("stall_watchdog_enabled", False))
        self.stall_watchdog_timeout_seconds = self._coerce_positive_float(
            tracking_cfg.get("stall_watchdog_timeout_seconds")
        )
        if self.stall_watchdog_enabled and self.stall_watchdog_timeout_seconds is None:
            self.stall_watchdog_timeout_seconds = 900.0
        self.stall_watchdog_exit_on_timeout = bool(
            tracking_cfg.get("stall_watchdog_exit_on_timeout", True)
        )
        self.stall_watchdog_repeat = bool(tracking_cfg.get("stall_watchdog_repeat", False))
        self.stall_watchdog_traceback_file = self._optional_string(
            tracking_cfg.get("stall_watchdog_traceback_file")
        )
        self.stall_watchdog_context_interval_steps = self._coerce_positive_int(
            tracking_cfg.get("stall_watchdog_context_interval_steps", 1)
        )
        if self.stall_watchdog_context_interval_steps is None:
            self.stall_watchdog_context_interval_steps = 1
        self._stall_watchdog_traceback_path: Optional[Path] = None
        self._stall_watchdog_context_path: Optional[Path] = None
        self._stall_watchdog_file_handle = None
        self._stall_watchdog_armed_phase: Optional[str] = None
        self._stall_watchdog_last_context_global_step: Optional[int] = None
        self.resource_guard_enabled = bool(tracking_cfg.get("resource_guard_enabled", False))
        self.max_process_rss_mb = self._coerce_positive_float(
            tracking_cfg.get("max_process_rss_mb")
        )
        self.min_available_ram_mb = self._coerce_positive_float(
            tracking_cfg.get("min_available_ram_mb")
        )
        self._process = psutil.Process()
        self._deferrable_wait_steps: Dict[tuple[int, str], int] = {}
        self.progress_tracker = ProgressTracker(progress_path)
        self._configured_render_enabled = bool(getattr(self.env, "render_enabled", False))
        self._configured_export_kpis_on_episode_end = bool(
            export_cfg.get(
                "export_kpis_on_episode_end",
                getattr(self.env, "export_kpis_on_episode_end", False),
            )
        )
        self._export_final_episode_only = bool(export_cfg.get("final_episode_only", False))
        self._export_kpis_final_episode_only = (
            self._export_final_episode_only
            if export_cfg.get("kpis_final_episode_only") is None
            else bool(export_cfg["kpis_final_episode_only"])
        )
        self._export_timeseries_final_episode_only = (
            self._export_final_episode_only
            if export_cfg.get("timeseries_final_episode_only") is None
            else bool(export_cfg["timeseries_final_episode_only"])
        )
        self._export_profile = export_cfg.get("profile", "final")
        self._diagnostic_max_episodes = export_cfg.get("diagnostic_max_episodes", 3)
        self._diagnostic_max_mb = export_cfg.get("diagnostic_max_mb", 128)
        self.export_measurements = []
        self._export_include_business_as_usual = bool(
            export_cfg.get("include_business_as_usual", True)
        )
        self._export_business_as_usual_timeseries = bool(
            export_cfg.get("export_business_as_usual_timeseries", True)
        )
        self._export_kpi_round_decimals = export_cfg.get("kpi_round_decimals")
        self._manual_kpi_export = self._configured_export_kpis_on_episode_end
        self._manual_kpi_exported_episodes: set[int | None] = set()

        self.wrapper_reward_enabled = bool(wrapper_reward_cfg.get("enabled", False))
        self.wrapper_reward_profile = (
            str(wrapper_reward_cfg.get("profile", "cost_limits_v1")).strip() or "cost_limits_v1"
        )
        if self.wrapper_reward_profile not in WRAPPER_REWARD_PROFILES:
            logger.warning(
                "Unknown wrapper reward profile '{}'; falling back to 'cost_limits_v1'.",
                self.wrapper_reward_profile,
            )
            self.wrapper_reward_profile = "cost_limits_v1"
        self.wrapper_reward_profile_config = WRAPPER_REWARD_PROFILES[self.wrapper_reward_profile]
        self.wrapper_reward_version = str(
            self.wrapper_reward_profile_config.get("version", "unknown")
        )
        self.wrapper_reward_clip_enabled = bool(wrapper_reward_cfg.get("clip_enabled", True))
        self.wrapper_reward_clip_min = float(wrapper_reward_cfg.get("clip_min", -10.0))
        self.wrapper_reward_clip_max = float(wrapper_reward_cfg.get("clip_max", 10.0))
        if self.wrapper_reward_clip_max < self.wrapper_reward_clip_min:
            logger.warning(
                "Invalid wrapper reward clip range [{}, {}]; disabling clipping.",
                self.wrapper_reward_clip_min,
                self.wrapper_reward_clip_max,
            )
            self.wrapper_reward_clip_enabled = False
        self.wrapper_reward_squash = (
            str(wrapper_reward_cfg.get("squash", "none")).strip().lower() or "none"
        )
        if self.wrapper_reward_squash not in {"none", "tanh"}:
            logger.warning(
                "Unknown wrapper reward squash '{}'; falling back to 'none'.",
                self.wrapper_reward_squash,
            )
            self.wrapper_reward_squash = "none"

        checkpoint_base_dir = config.get("runtime", {}).get("job_dir") or self.log_dir
        self.checkpoint_manager = CheckpointManager(
            base_dir=checkpoint_base_dir,
            interval=checkpoint_cfg.get("checkpoint_interval"),
            save_final=bool(checkpoint_cfg.get("save_final", True)),
            min_free_disk_mb=checkpoint_cfg.get("min_free_disk_mb", 64),
            log_to_mlflow=tracking_cfg.get("mlflow_enabled", False),
            require_update_step=bool(checkpoint_cfg.get("require_update_step", True)),
            require_initial_exploration_done=bool(
                checkpoint_cfg.get("require_initial_exploration_done", True)
            ),
            checkpoint_on_episode_end=bool(checkpoint_cfg.get("checkpoint_on_episode_end", False)),
            keep_episode_checkpoints=bool(checkpoint_cfg.get("keep_episode_checkpoints", False)),
        )
        self.local_metrics_logger = None
        if not self.mlflow_enabled and tracking_cfg.get("metrics_enabled", True):
            self.local_metrics_logger = LocalMetricsLogger(self.log_dir)

        # Ensure encoders are initialised for observation metadata and encoding
        if not hasattr(self, "encoders") or not getattr(self, "encoders"):
            self.encoders = self.set_encoders()

    def _initialize_entity_agent_state(self, env: CityLearnEnv) -> None:
        self.env = env
        self.observation_names = []
        self.action_names = []
        self.observation_space = []
        self.action_space = []
        self.episode_time_steps = int(getattr(self.env.unwrapped, "time_steps", 0) or 0)
        self.building_metadata = (self.env.unwrapped.get_metadata() or {}).get("buildings", [])

        Environment.__init__(
            self,
            seconds_per_time_step=getattr(self.env.unwrapped, "seconds_per_time_step", None),
            random_seed=getattr(self.env.unwrapped, "random_seed", None),
            episode_tracker=getattr(self.env.unwrapped, "episode_tracker", None),
            time_step_ratio=getattr(self.env.unwrapped, "time_step_ratio", None),
        )

        # Keep RLC state available for methods shared with the flat path.
        self.hidden_dimension = None
        self.discount = None
        self.tau = None
        self.alpha = None
        self.lr = None
        self.batch_size = None
        self.replay_buffer_capacity = None
        self.standardize_start_time_step = None
        self.end_exploration_time_step = None
        self.action_scaling_coefficient = None
        self.reward_scaling = None
        self.update_per_time_step = None

        self._entity_adapter = EntityContractAdapter(
            self.env,
            normalization_enabled=self._entity_encoding_enabled
            and self._entity_encoding_policy == "minmax_space",
            clip=self._entity_encoding_clip,
            encoding_profile=self._entity_encoding_profile,
        )

        initial_observations, _ = self.simulator.reset()
        self._apply_entity_layout(initial_observations, force_attach=False)
        # Entity mode needs one real reset to discover the initial payload and
        # compile its layout.  That reset is bootstrap work, not a configured
        # training/evaluation episode.  CityLearn advances EpisodeTracker on
        # every reset, so leaving the index untouched silently shifts explicit
        # episode windows by one (and changes episode-derived EV randomness).
        # Restore the tracker before the learning loop to give entity mode the
        # same episode semantics as the flat wrapper, which does not consume a
        # reset while it is constructed.
        reset_episode_index = getattr(
            getattr(self.env.unwrapped, "episode_tracker", None),
            "reset_episode_index",
            None,
        )
        if callable(reset_episode_index):
            reset_episode_index()
        self.reset()

    def _apply_entity_layout(
        self,
        observation_payload: Mapping[str, Any],
        force_attach: bool,
        *,
        model_observations: bool = False,
        attach_model: bool = True,
    ) -> List[np.ndarray]:
        if not self._entity_interface_mode or self._entity_adapter is None:
            return []

        previous_version = self._entity_topology_version
        candidate_topology_change = force_attach or self._entity_adapter.needs_rebuild(
            observation_payload
        )
        previous_layout_state = (
            self._snapshot_entity_layout_state() if candidate_topology_change else None
        )
        model_topology_snapshot = None
        restore_model_topology = None
        try:
            if model_observations:
                agent_observations, observation_names, observation_spaces = (
                    self._entity_adapter.to_agent_encoded_observations(observation_payload)
                )
            else:
                agent_observations, observation_names, observation_spaces = (
                    self._entity_adapter.to_agent_observations(observation_payload)
                )
            self._entity_topology_version = self._entity_adapter.topology_version

            self.episode_time_steps = int(
                getattr(self.episode_tracker, "episode_time_steps", self.episode_time_steps)
            )

            topology_changed = (
                force_attach
                or previous_version is None
                or self._entity_topology_version != previous_version
            )
            self._topology_changed_during_step = (
                topology_changed and not force_attach and previous_version is not None
            )
            if topology_changed:
                self.observation_names = observation_names
                self.observation_space = observation_spaces
                self.action_space = list(getattr(self.env, "flat_action_space", []))
                self.action_names = [list(names) for names in getattr(self.env, "action_names", [])]
                if len(self.action_names) < len(self.action_space):
                    self.action_names.extend(
                        [[] for _ in range(len(self.action_space) - len(self.action_names))]
                    )
                elif len(self.action_names) > len(self.action_space):
                    self.action_names = self.action_names[: len(self.action_space)]
                self.encoders = self.set_encoders()
                self._action_bounds_cache = None

            if topology_changed and self._entity_dynamic_mode and previous_version is not None:
                from algorithms.registry import ALGORITHM_REGISTRY

                offenders = [
                    name
                    for name in self._algorithm_names
                    if (cls := ALGORITHM_REGISTRY.get(name)) is not None
                    and not bool(getattr(cls, "supports_dynamic_topology", False))
                ]
                if offenders:
                    if "MADDPG" in offenders:
                        raise ValueError(
                            "MADDPG supports entity interface only with topology_mode='static'. "
                            "Detected topology change during runtime."
                        )
                    raise ValueError(
                        f"{sorted(offenders)} do not support dynamic topology "
                        "(supports_dynamic_topology=False). Detected topology change during runtime."
                    )

            if topology_changed and attach_model and self.model is not None:
                if previous_version is not None and self._entity_dynamic_mode:
                    snapshot_model_topology = getattr(self.model, "snapshot_topology_state", None)
                    restore_model_topology = getattr(self.model, "restore_topology_state", None)
                    if callable(snapshot_model_topology) != callable(restore_model_topology):
                        raise RuntimeError(
                            "Model topology rollback requires both snapshot_topology_state() "
                            "and restore_topology_state()."
                        )
                    if callable(snapshot_model_topology):
                        model_topology_snapshot = snapshot_model_topology()
                self._attach_model_environment_metadata()
        except Exception:
            if model_topology_snapshot is not None and callable(restore_model_topology):
                restore_model_topology(model_topology_snapshot)
            if previous_layout_state is not None:
                self._restore_entity_layout_state(previous_layout_state)
            raise

        return [np.asarray(obs, dtype=np.float64) for obs in agent_observations]

    def _snapshot_entity_layout_state(self) -> Dict[str, Any]:
        """Capture wrapper and adapter metadata changed during layout rebuild."""
        adapter_state = None
        if self._entity_adapter is not None:
            adapter_state = {
                name: deepcopy(value)
                for name, value in self._entity_adapter.__dict__.items()
                if name != "env"
            }
        return {
            "wrapper": {
                "_entity_topology_version": self._entity_topology_version,
                "_topology_changed_during_step": self._topology_changed_during_step,
                "episode_time_steps": self.episode_time_steps,
                "observation_names": deepcopy(self.observation_names),
                "observation_space": deepcopy(self.observation_space),
                "action_space": deepcopy(self.action_space),
                "action_names": deepcopy(self.action_names),
                "encoders": deepcopy(getattr(self, "encoders", None)),
                "_action_bounds_cache": deepcopy(getattr(self, "_action_bounds_cache", None)),
            },
            "adapter": adapter_state,
        }

    def _restore_entity_layout_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore wrapper and adapter metadata after a failed rebuild."""
        for name, value in snapshot["wrapper"].items():
            setattr(self, name, deepcopy(value))
        adapter_state = snapshot.get("adapter")
        if self._entity_adapter is None or adapter_state is None:
            return
        for name in list(self._entity_adapter.__dict__):
            if name != "env":
                del self._entity_adapter.__dict__[name]
        self._entity_adapter.__dict__.update(deepcopy(adapter_state))

    def _model_requires_raw_observation_context(self) -> bool:
        if self.model is None:
            return False
        if bool(getattr(self.model, "use_raw_observations", False)):
            return True
        if bool(getattr(self.model, "requires_raw_observation_context", False)):
            return True
        if bool(getattr(self.model, "requires_entity_observation_context", False)):
            return True
        if getattr(self.model, "_warm_start_policy", None) is not None:
            return True
        if getattr(self.model, "warm_start_policy_name", None):
            return True
        return False

    def _can_use_direct_entity_model_observations(self) -> bool:
        return (
            self._entity_interface_mode
            and self._entity_adapter is not None
            and self._entity_encoding_profile != "minmax_space"
            and not self.wrapper_reward_enabled
            and not self.action_diagnostics_enabled
            and not self._model_requires_raw_observation_context()
        )

    def _attach_model_environment_metadata(self) -> None:
        if self.model is None:
            return

        building_names = self._resolve_building_names()
        raw_observation_names = [
            [str(name) for name in observation_group]
            for observation_group in getattr(self, "observation_names", [])
        ]
        encoded_observation_names = (
            self._entity_adapter.encoded_observation_names(raw_observation_names)
            if self._entity_interface_mode and self._entity_adapter is not None
            else raw_observation_names
        )
        profiled_encoded_observation_names = {
            profile: adapter.encoded_observation_names(raw_observation_names)
            for profile, adapter in self._entity_profile_adapters.items()
        }
        if self._entity_interface_mode and self._entity_adapter is not None:
            active_profile = str(self._entity_encoding_profile or "").strip().lower()
            if active_profile:
                # The active encoder is not duplicated in
                # ``_entity_profile_adapters``. Expose it under the same map so
                # a stage that also needs raw side-channel context still gets
                # its encoded model-input names during attach_environment().
                profiled_encoded_observation_names.setdefault(
                    active_profile,
                    encoded_observation_names,
                )
        metadata = {
            "community_market": {
                "enabled": bool(getattr(self.env.unwrapped, "community_market_enabled", False)),
                "local_price_ratio_to_grid_import": getattr(
                    self.env.unwrapped, "community_market_sell_ratio", 1.0
                ),
                "grid_export_price": getattr(
                    self.env.unwrapped, "community_market_grid_export_price", 0.0
                ),
            },
            "seconds_per_time_step": getattr(self.env, "seconds_per_time_step", None),
            "building_names": building_names,
            "interface": getattr(self.env, "interface", None),
            "topology_mode": self._entity_topology_mode
            if self._entity_interface_mode
            else getattr(self.env, "topology_mode", None),
            "entity_specs": getattr(self.env, "entity_specs", None)
            if self._entity_interface_mode
            else None,
            "raw_observation_names": raw_observation_names,
            "encoded_observation_names": encoded_observation_names,
            "profiled_encoded_observation_names": profiled_encoded_observation_names,
            "raw_observation_bounds": [
                {
                    "low": np.asarray(space.low, dtype=np.float64).reshape(-1).tolist(),
                    "high": np.asarray(space.high, dtype=np.float64).reshape(-1).tolist(),
                }
                for space in self.observation_space
            ],
            "building_battery_capacity_kwh": [
                float((item.get("electrical_storage") or {}).get("capacity", 0.0) or 0.0)
                for item in (self.building_metadata or [])
            ],
        }

        attach = getattr(self.model, "attach_environment", None)
        if callable(attach):
            from algorithms.runtime.randomness import call_hook

            call_hook(
                self.model,
                "attach_environment",
                stream="initialization",
                observation_names=self.observation_names,
                action_names=self.action_names,
                action_space=self.action_space,
                observation_space=self.observation_space,
                metadata=metadata,
            )
            if hasattr(self.model, "_export_references"):
                from algorithms.export.references import begin_capture

                begin_capture(self.model, self.config)

    def _resolve_building_names(self) -> Optional[List[str]]:
        env = getattr(self, "env", None)
        if env is None:
            return None
        raw_building_names = getattr(env, "building_names", None)
        if isinstance(raw_building_names, list):
            return [str(name) for name in raw_building_names]

        if self._entity_interface_mode:
            specs = getattr(self.env, "entity_specs", None)
            if isinstance(specs, Mapping):
                table_specs = (
                    specs.get("tables", {}) if isinstance(specs.get("tables", {}), Mapping) else {}
                )
                building_table = (
                    table_specs.get("building", {})
                    if isinstance(table_specs.get("building", {}), Mapping)
                    else {}
                )
                building_ids = building_table.get("ids")
                if isinstance(building_ids, list):
                    return [str(name) for name in building_ids]

        get_metadata = getattr(getattr(self.env, "unwrapped", self.env), "get_metadata", None)
        if callable(get_metadata):
            metadata = get_metadata() or {}
            buildings = metadata.get("buildings", []) if isinstance(metadata, Mapping) else []
            names = [
                str(building.get("name"))
                for building in buildings
                if isinstance(building, Mapping) and building.get("name")
            ]
            if names:
                return names

        return None

    @property
    def observation_dimension(self) -> List[int]:
        dimensions: List[int] = []
        for space in self.observation_space:
            if hasattr(space, "low"):
                dimensions.append(int(np.asarray(space.low).reshape(-1).shape[0]))
            else:
                dimensions.append(0)
        return dimensions

    @property
    def action_dimension(self) -> List[int]:
        dimensions: List[int] = []
        for space in self.action_space:
            if hasattr(space, "low"):
                dimensions.append(int(np.asarray(space.low).reshape(-1).shape[0]))
            else:
                dimensions.append(0)
        return dimensions

    def _configure_episode_exports(
        self, episode: int, episodes: int, *, deterministic=True
    ) -> bool:
        """Enable KPI export and timeseries rendering independently per episode."""

        is_final_episode = episode >= episodes - 1
        export_this_episode = self._configured_export_kpis_on_episode_end and (
            not self._export_kpis_final_episode_only or is_final_episode
        )
        render_this_episode = self._configured_render_enabled and (
            not self._export_timeseries_final_episode_only or is_final_episode
        )
        if getattr(self, "_export_profile", None) == "final":
            export_this_episode = export_this_episode and is_final_episode and deterministic
            render_this_episode = render_this_episode and is_final_episode and deterministic
        elif getattr(self, "_export_profile", None) == "diagnostic" and not (
            is_final_episode and deterministic
        ):
            allowed = episode < self._diagnostic_max_episodes
            export_this_episode = export_this_episode and allowed
            render_this_episode = render_this_episode and allowed
        self._export_episode_deterministic = deterministic

        if hasattr(self.env, "render_enabled"):
            self.env.render_enabled = render_this_episode
        if hasattr(self.env, "export_kpis_on_episode_end"):
            self.env.export_kpis_on_episode_end = (
                export_this_episode and not self._manual_kpi_export
            )

        return export_this_episode

    def _export_episode_kpis_if_needed(
        self,
        export_this_episode: bool,
        episode: int | None = None,
        *,
        is_final_episode: bool = False,
    ) -> None:
        if not export_this_episode or not self._manual_kpi_export:
            return
        if not hasattr(self.env, "export_final_kpis"):
            raise RuntimeError("Requested KPIs require simulator.export_final_kpis()")
        if episode in self._manual_kpi_exported_episodes:
            return

        export_business_as_usual_timeseries = self._export_business_as_usual_timeseries
        if self._export_timeseries_final_episode_only:
            export_business_as_usual_timeseries = (
                export_business_as_usual_timeseries and is_final_episode
            )

        candidate_kwargs: Dict[str, Any] = {
            "include_business_as_usual": self._export_include_business_as_usual,
            "export_business_as_usual_timeseries": export_business_as_usual_timeseries,
            "kpi_round_decimals": self._export_kpi_round_decimals,
        }
        if not self._export_kpis_final_episode_only and episode is not None:
            candidate_kwargs["filepath"] = f"exported_kpis_ep{episode}.csv"
        # A training diagnostic must never occupy the final-scorecard filename.
        if getattr(self, "_export_profile", None) == "diagnostic":
            if is_final_episode and getattr(self, "_export_episode_deterministic", False):
                candidate_kwargs.pop("filepath", None)
            else:
                candidate_kwargs["filepath"] = f"diagnostic_kpis_ep{episode}.csv"

        sig = inspect.signature(self.env.export_final_kpis)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_keyword:
            kwargs = candidate_kwargs
        else:
            kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
        export_started = time.perf_counter()
        self.env.export_final_kpis(**kwargs)
        if (
            getattr(self, "_export_profile", None) is None
            and not self._export_kpis_final_episode_only
            and is_final_episode
            and episode is not None
        ):
            final_kwargs = {k: v for k, v in kwargs.items() if k != "filepath"}
            self.env.export_final_kpis(**final_kwargs)

        self._manual_kpi_exported_episodes.add(episode)
        root = getattr(self.env, "render_output_root", None)
        size = (
            sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())
            if root
            else 0
        )
        if hasattr(self, "export_measurements"):
            self.export_measurements.append(
                {
                    "episode": episode,
                    "seconds": time.perf_counter() - export_started,
                    "total_bytes": size,
                    "final_evaluation": bool(
                        is_final_episode and getattr(self, "_export_episode_deterministic", False)
                    ),
                }
            )
        if (
            getattr(self, "_export_profile", None) == "diagnostic"
            and size > self._diagnostic_max_mb * 1024**2
        ):
            raise RuntimeError("Diagnostic export exceeded simulator.export.diagnostic_max_mb")

    def set_model(self, model: ExecutionUnit):
        """Set the model (any :class:`ExecutionUnit`) after initialization."""
        self.model = model
        self._configure_model_observation_profiles()
        self._attach_model_environment_metadata()

    def _configure_model_observation_profiles(self) -> None:
        """Build auxiliary entity encoders requested by hierarchical stages."""
        self._entity_profile_adapters = {}
        self._profiled_observation_cache_key = None
        self._profiled_observation_cache_value = {}
        if not self._entity_interface_mode or self.model is None:
            return
        profile_hook = getattr(self.model, "required_observation_encoding_profiles", None)
        if not callable(profile_hook):
            return
        for raw_profile in profile_hook() or []:
            profile = str(raw_profile or "").strip().lower()
            if not profile or profile == self._entity_encoding_profile:
                continue
            self._entity_profile_adapters[profile] = EntityContractAdapter(
                self.env,
                normalization_enabled=(
                    self._entity_encoding_enabled and self._entity_encoding_policy == "minmax_space"
                ),
                clip=self._entity_encoding_clip,
                encoding_profile=profile,
            )

    def _encode_profiled_observations(
        self,
        observations: List[Any],
    ) -> Dict[str, List[np.ndarray]]:
        """Encode one raw observation batch for every stage-specific profile."""
        cache_key = tuple(id(observation) for observation in observations)
        if (
            cache_key == self._profiled_observation_cache_key
            and self._profiled_observation_cache_value
        ):
            return self._profiled_observation_cache_value
        encoded: Dict[str, List[np.ndarray]] = {}
        for profile, adapter in self._entity_profile_adapters.items():
            encoded[profile] = [
                adapter.normalize_observation(
                    agent_index=index,
                    observation=observation,
                    observation_names=(
                        self.observation_names[index] if index < len(self.observation_names) else []
                    ),
                    observation_space=self.observation_space[index],
                ).astype(np.float64)
                for index, observation in enumerate(observations)
            ]
        self._profiled_observation_cache_key = cache_key
        self._profiled_observation_cache_value = encoded
        return encoded

    def _prepare_episode_reset(self) -> None:
        """Optionally replay one identical exogenous CityLearn scenario.

        CityLearn derives EV drift from ``random_seed + episode_index`` even
        when the time-series window itself is unchanged. Replaying the episode
        index makes repeated policy evaluations genuinely paired instead of
        comparing different EV realizations. This is opt-in because ordinary
        multi-episode training should retain CityLearn's default progression.
        """

        if not self._repeat_episode_scenario:
            self._deferrable_wait_steps.clear()
            return
        tracker = getattr(self.env.unwrapped, "episode_tracker", None)
        reset_index = getattr(tracker, "reset_episode_index", None)
        if not callable(reset_index):
            raise RuntimeError(
                "simulator.repeat_episode_scenario requires a CityLearn "
                "episode tracker with reset_episode_index()"
            )
        reset_index()
        self._deferrable_wait_steps.clear()

    def _record_episode_execution(
        self,
        *,
        episode: int,
        deterministic: bool,
        training: bool,
        export_enabled: bool,
    ) -> None:
        """Record the actual CityLearn window selected for an episode."""

        tracker = getattr(self.env.unwrapped, "episode_tracker", None)

        def _optional_int(name: str) -> Optional[int]:
            value = getattr(tracker, name, None)
            return None if value is None else int(value)

        record = {
            "episode": int(episode),
            "tracker_episode": _optional_int("episode"),
            "start_time_step": _optional_int("episode_start_time_step"),
            "end_time_step": _optional_int("episode_end_time_step"),
            "time_steps": int(self.episode_time_steps),
            "deterministic": bool(deterministic),
            "training": bool(training),
            "export_enabled": bool(export_enabled),
        }
        self._episode_execution_history.append(record)
        logger.info(
            "event=episode_window episode={episode} tracker_episode={tracker_episode} "
            "start={start_time_step} end={end_time_step} time_steps={time_steps} "
            "deterministic={deterministic} training={training} export_enabled={export_enabled}",
            **record,
        )

    def get_episode_execution_history(self) -> List[Dict[str, Any]]:
        """Return auditable runtime evidence for the last ``learn`` call."""

        return deepcopy(self._episode_execution_history)

    def learn(self, episodes=None, deterministic=None, deterministic_finish=False):
        """Run the common loop with this simulator/agent adapter."""
        from algorithms.runtime.loop import run_episodes

        return run_episodes(self, episodes, deterministic, deterministic_finish)

    def predict(self, observations, deterministic=None, *, episode_step=None):
        """
        Updates the predict action logic. It now uses a mix of algorithm and the next time step.
        """
        if self.model is None:
            raise ValueError("Model is not set. Use `set_model` to provide a model.")

        direct_entity_model_observations = bool(
            getattr(self, "_entity_model_observations_direct", False)
        )
        if direct_entity_model_observations:
            encoded_observations = [np.asarray(obs, dtype=np.float64) for obs in observations]
        else:
            encoded_observations = self._encode_observations_for_model(observations)

        self._current_model_episode_step = None if episode_step is None else int(episode_step)
        from algorithms.export.references import set_capture_step

        set_capture_step(self.model, self._current_model_episode_step)
        episode_context_hook = getattr(self.model, "set_episode_context", None)
        if callable(episode_context_hook):
            episode_context_hook(
                episode_step=self._current_model_episode_step,
                next_episode_step=None,
            )

        observation_context_hook = getattr(self.model, "set_observation_context", None)
        entity_observation_context_hook = getattr(
            self.model,
            "set_entity_observation_context",
            None,
        )
        profiled_observation_context_hook = getattr(
            self.model,
            "set_profiled_observation_context",
            None,
        )
        if callable(profiled_observation_context_hook):
            profiled_observation_context_hook(
                self._encode_profiled_observations(observations)
                if not direct_entity_model_observations
                else {}
            )
        if callable(observation_context_hook):
            observation_context_hook(
                raw_observations=None if direct_entity_model_observations else observations,
                encoded_observations=encoded_observations,
            )
        if callable(entity_observation_context_hook):
            if self._entity_current_payload is None:
                raise RuntimeError(
                    "Model requested entity observation context but no entity payload is active."
                )
            entity_observation_context_hook(
                observation_payload=self._entity_current_payload,
                info=self._entity_current_info,
            )

        actions = decide(self.model, encoded_observations, deterministic)
        if not self._entity_interface_mode:
            self.actions = actions
            self.next_time_step()
        else:
            Environment.next_time_step(self)
        return actions

    def _to_env_actions(self, actions: List[List[float]]) -> Any:
        if not self._entity_interface_mode:
            return actions

        if self._entity_adapter is None:
            raise RuntimeError("Entity adapter is not initialized.")

        return self._entity_adapter.to_entity_actions(actions, self.action_names)

    def _clip_actions(self, actions: List[List[float]]) -> List[List[float]]:
        """Reject malformed actions; clipping is an explicit, recorded policy."""
        if not isinstance(actions, list):
            raise ValueError("Model predicted actions must be provided as a list.")
        bounds = self._get_action_bounds_cache()
        if len(actions) != len(bounds):
            raise ValueError(
                f"Action agent count mismatch: predicted={len(actions)}, expected={len(bounds)}"
            )
        policy = self.config.get("simulator", {}).get("action_policy", "clip")
        result = []
        for index, (raw, (low, high)) in enumerate(zip(actions, bounds)):
            values = np.asarray(raw, dtype=np.float64)
            if values.ndim != 1 or values.shape != low.shape:
                raise ValueError(
                    f"Action dimension mismatch for agent {index}: predicted={values.shape}, expected={low.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite action for agent {index}")
            clipped = np.clip(values, low, high)
            if not np.array_equal(clipped, values):
                if policy == "reject":
                    raise ValueError(f"Action outside bounds for agent {index}")
                logger.warning(
                    "Action clipped: agent={} proposed={} submitted={}",
                    index,
                    values.tolist(),
                    clipped.tolist(),
                )
            result.append(clipped.tolist())
        return result

    def _get_action_bounds_cache(self) -> List[tuple[np.ndarray, np.ndarray]]:
        cached = getattr(self, "_action_bounds_cache", None)
        if cached is not None and len(cached) == len(self.action_space):
            return cached

        bounds: List[tuple[np.ndarray, np.ndarray]] = []
        for action_space in self.action_space:
            if hasattr(action_space, "low") and hasattr(action_space, "high"):
                low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
                high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
            else:
                low = np.asarray([], dtype=np.float64)
                high = np.asarray([], dtype=np.float64)
            bounds.append((low, high))
        self._action_bounds_cache = bounds
        return bounds

    def update(
        self,
        observations: List[List[float]],
        actions: List[List[float]],
        reward: List[float],
        next_observations: List[List[float]],
        terminated: bool,
        truncated: bool,
        *,
        learn: bool = True,
    ):
        """
        Prepare transition context, then learn or invoke a read-only observer.
        """

        if self.model is None:
            logger.error("Model is not set. Use `set_model` to provide a model.")
            raise ValueError("Model is not set. Use `set_model` to provide a model.")

        if learn:
            self._refresh_update_schedule()

        phase_start_time = time.perf_counter()
        direct_entity_model_observations = bool(
            getattr(self, "_entity_model_observations_direct", False)
        )
        if direct_entity_model_observations:
            encoded_observations = [np.asarray(obs, dtype=np.float64) for obs in observations]
            encoded_next_observations = [
                np.asarray(obs, dtype=np.float64) for obs in next_observations
            ]
            self._last_model_observation_encoding_seconds = 0.0
        else:
            encoded_observations = self._encode_observations_for_model(observations)
            encoded_next_observations = self._encode_observations_for_model(next_observations)
            self._last_model_observation_encoding_seconds = time.perf_counter() - phase_start_time

        transition_context_hook = getattr(self.model, "set_transition_context", None)
        entity_transition_context_hook = getattr(
            self.model,
            "set_entity_transition_context",
            None,
        )
        profiled_transition_context_hook = getattr(
            self.model,
            "set_profiled_transition_context",
            None,
        )
        episode_context_hook = getattr(self.model, "set_episode_context", None)
        current_episode_step = getattr(self, "_current_model_episode_step", None)
        if callable(episode_context_hook):
            episode_context_hook(
                episode_step=current_episode_step,
                next_episode_step=(
                    None
                    if terminated or truncated or current_episode_step is None
                    else int(current_episode_step) + 1
                ),
            )
        if callable(transition_context_hook):
            transition_context_hook(
                raw_observations=None if direct_entity_model_observations else observations,
                raw_next_observations=None
                if direct_entity_model_observations
                else next_observations,
                encoded_observations=encoded_observations,
                encoded_next_observations=encoded_next_observations,
            )
        if callable(entity_transition_context_hook):
            if self._entity_current_payload is None or self._entity_next_payload is None:
                raise RuntimeError(
                    "Model requested entity transition context but current/next payloads are incomplete."
                )
            entity_transition_context_hook(
                observation_payload=self._entity_current_payload,
                next_observation_payload=self._entity_next_payload,
                info=self._entity_step_info,
            )
        if callable(profiled_transition_context_hook):
            profiled_transition_context_hook(
                profiled_encoded_observations=(
                    self._encode_profiled_observations(observations)
                    if not direct_entity_model_observations
                    else {}
                ),
                profiled_encoded_next_observations=(
                    self._encode_profiled_observations(next_observations)
                    if not direct_entity_model_observations
                    else {}
                ),
            )

        if not learn:
            observer = getattr(self.model, "observe_transition", None)
            if not callable(observer):
                return None
            phase_start_time = time.perf_counter()
            observation_result = observer(
                observations=encoded_observations,
                actions=actions,
                rewards=reward,
                next_observations=encoded_next_observations,
                terminated=terminated,
                truncated=truncated,
            )
            self._last_model_update_seconds = time.perf_counter() - phase_start_time
            return observation_result

        # Pass updated parameters to model.update()
        phase_start_time = time.perf_counter()
        transition = Transition(
            observations=encoded_observations,
            next_observations=encoded_next_observations,
            rewards=reward,
            submitted_actions=actions,
            proposed_actions=getattr(self, "_last_proposed_actions", None),
            terminated=terminated,
            truncated=truncated,
            member_ids=tuple(self._resolve_building_names() or ()),
            duration_seconds=getattr(self, "_last_step_duration_seconds", None),
            info=self._entity_step_info if getattr(self, "_entity_interface_mode", False) else None,
        )
        schedule = UpdateSchedule(
            getattr(self, "learning_step", self.global_step),
            self.update_step,
            self.update_target_step,
            self.initial_exploration_done,
        )
        update_result = learn_transition(self.model, transition, schedule)
        self._last_model_update_seconds = time.perf_counter() - phase_start_time
        return update_result

    def _refresh_update_schedule(self) -> None:
        """Refresh per-step training and checkpoint scheduling state."""
        if not self.steps_between_training_updates or self.steps_between_training_updates <= 1:
            self.update_step = True
        else:
            self.update_step = (
                getattr(self, "learning_step", self.global_step)
                % self.steps_between_training_updates
                == 0
            )
        logger.debug(
            "Time step - Doing Update" if self.update_step else "Time step - Skipping Update"
        )

        # Exploration phase ownership belongs to the algorithm.
        self.initial_exploration_done = bool(
            self.model.is_initial_exploration_done(getattr(self, "learning_step", self.global_step))
        )
        logger.debug(
            "Initial exploration done: {} (global step={})",
            self.initial_exploration_done,
            getattr(self, "learning_step", self.global_step),
        )

        # Determine whether to update the target networks
        if not self.target_update_interval:
            self.update_target_step = False
        else:
            self.update_target_step = (
                getattr(self, "learning_step", self.global_step) % self.target_update_interval == 0
            )
        logger.debug(
            "Time step - Doing Target Update"
            if self.update_target_step
            else "Time step - Skipping Target Update"
        )

    def _should_log_step(self, step: int) -> bool:
        return step % self.step_metric_interval == 0

    def _encode_observations_for_model(self, observations: List[Any]) -> List[np.ndarray]:
        if getattr(self.model, "use_raw_observations", False):
            return [np.asarray(obs, dtype=np.float64) for obs in observations]

        cache_key = tuple(id(obs) for obs in observations)
        if (
            getattr(self, "_encoded_observation_cache_key", None) == cache_key
            and getattr(self, "_encoded_observation_cache_value", None) is not None
        ):
            return self._encoded_observation_cache_value

        encoded_observations = self.get_all_encoded_observations(observations)
        self._encoded_observation_cache_key = cache_key
        self._encoded_observation_cache_value = encoded_observations
        return encoded_observations

    def get_encoded_observations(self, index: int, observations: List[float]) -> np.ndarray:
        """Optimized encoding function using NumPy with proper type handling."""

        if self._entity_interface_mode:
            if index >= len(self.observation_space):
                return np.nan_to_num(
                    np.asarray(observations, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
                )
            if self._entity_adapter is None:
                return np.nan_to_num(
                    np.asarray(observations, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
                )

            return self._entity_adapter.normalize_observation(
                agent_index=index,
                observation=observations,
                observation_names=self.observation_names[index]
                if index < len(self.observation_names)
                else [],
                observation_space=self.observation_space[index],
            ).astype(np.float64)

        obs_array = np.array(observations, dtype=np.float64)  # Ensure numeric type

        # Apply encoding transformation correctly
        encoded = np.hstack(
            [
                encoder.transform(obs) if hasattr(encoder, "transform") else encoder * obs
                for encoder, obs in zip(self.encoders[index], obs_array)
            ]
        ).astype(np.float64)  # Convert everything to float

        return encoded[~np.isnan(encoded)]  # Remove NaN values safely

    def get_all_encoded_observations(self, observations: List[List[float]]) -> List[np.ndarray]:
        """Optimized version without joblib for better performance."""
        return [self.get_encoded_observations(idx, obs) for idx, obs in enumerate(observations)]

    def describe_environment(self) -> dict:
        """Return metadata required for inference encoders/decoders."""
        if not getattr(self, "encoders", None):
            self.encoders = self.set_encoders()

        def _encode_params(encoder):
            params = {}
            for attr in ("x_max", "x_min", "classes", "missing_value", "default"):
                if hasattr(encoder, attr):
                    value = getattr(encoder, attr)
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    elif isinstance(value, (list, tuple)):
                        value = list(value)
                    params[attr] = value
            return {
                "type": encoder.__class__.__name__,
                "params": params,
            }

        raw_observation_names = [
            [str(name) for name in observation_group]
            for observation_group in self.observation_names
        ]
        encoded_observation_names = (
            self._entity_adapter.encoded_observation_names(raw_observation_names)
            if self._entity_interface_mode and self._entity_adapter is not None
            else raw_observation_names
        )
        profiled_encoded_observation_names = {
            profile: adapter.encoded_observation_names(raw_observation_names)
            for profile, adapter in self._entity_profile_adapters.items()
        }
        if self._entity_interface_mode and self._entity_adapter is not None:
            active_profile = str(self._entity_encoding_profile or "").strip().lower()
            if active_profile:
                profiled_encoded_observation_names.setdefault(
                    active_profile,
                    encoded_observation_names,
                )
        serves_encoded_observations = (
            self._entity_interface_mode
            and self._entity_adapter is not None
            and self._entity_encoding_profile != "minmax_space"
        )
        serving_observation_names = (
            encoded_observation_names if serves_encoded_observations else raw_observation_names
        )

        if serves_encoded_observations:
            encoders_metadata = [
                [{"type": "NoNormalization", "params": {}} for _ in observation_group]
                for observation_group in serving_observation_names
            ]
        else:
            encoders_metadata = [
                [_encode_params(encoder) for encoder in encoder_list]
                for encoder_list in self.encoders
            ]

        action_bounds = []
        for space in self.action_space:
            if hasattr(space, "low") and hasattr(space, "high"):
                action_bounds.append(
                    {
                        "low": np.asarray(space.low).tolist(),
                        "high": np.asarray(space.high).tolist(),
                    }
                )
            else:
                action_bounds.append(None)

        action_names = getattr(self.env, "action_names", None)
        if action_names is None and hasattr(self, "action_names"):
            action_names = self.action_names

        action_names_by_agent = None
        flat_action_names = None
        if isinstance(action_names, list):
            if action_names and all(isinstance(item, (list, tuple)) for item in action_names):
                action_names_by_agent = {
                    str(index): [str(name) for name in names]
                    for index, names in enumerate(action_names)
                }
                flat_action_names = [str(name) for name in action_names[0]] if action_names else []
            else:
                flat_action_names = [str(name) for name in action_names]
                if len(self.observation_names) > 1:
                    action_names_by_agent = {
                        str(index): list(flat_action_names)
                        for index in range(len(self.observation_names))
                    }

        reward_fn = getattr(self.env, "reward_function", None)
        reward_config = None
        if reward_fn is not None:
            reward_config = {}
            for key, value in vars(reward_fn).items():
                if key.startswith("_"):
                    continue
                if isinstance(value, (int, float, str, bool)):
                    reward_config[key] = value
                elif isinstance(value, (list, tuple)):
                    reward_config[key] = list(value)

        building_names = self._resolve_building_names()

        return {
            "observation_names": serving_observation_names,
            "raw_observation_names": raw_observation_names,
            "encoded_observation_names": encoded_observation_names,
            "profiled_encoded_observation_names": profiled_encoded_observation_names,
            "encoders": encoders_metadata,
            "action_bounds": action_bounds,
            "action_names": flat_action_names,
            "action_names_by_agent": action_names_by_agent,
            "building_names": building_names,
            "interface": getattr(self.env, "interface", "flat"),
            "topology_mode": getattr(self.env, "topology_mode", "static"),
            "topology_version": getattr(self.env, "topology_version", None),
            "seconds_per_time_step": getattr(self.env, "seconds_per_time_step", None),
            "entity_encoding": {
                "enabled": bool(self._entity_encoding_enabled),
                "normalization": self._entity_encoding_policy,
                "profile": self._entity_encoding_profile,
                "clip": bool(self._entity_encoding_clip),
                "serving_observation_names": "encoded" if serves_encoded_observations else "raw",
            },
            "entity_specs": getattr(self.env, "entity_specs", None)
            if self._entity_interface_mode
            else None,
            "reward_function": {
                "name": reward_fn.__class__.__name__ if reward_fn else None,
                "params": reward_config,
            },
            "wrapper_reward": self.get_wrapper_reward_metadata(),
        }

    def get_wrapper_reward_metadata(self) -> dict:
        return {
            "enabled": bool(self.wrapper_reward_enabled),
            "profile": self.wrapper_reward_profile,
            "version": self.wrapper_reward_version,
            "clip_enabled": bool(self.wrapper_reward_clip_enabled),
            "clip_min": float(self.wrapper_reward_clip_min),
            "clip_max": float(self.wrapper_reward_clip_max),
            "squash": self.wrapper_reward_squash,
        }

    def set_encoders(self) -> List[List[Encoder]]:
        r"""Instantiate observation encoders from the shared JSON configuration."""

        if self._entity_interface_mode:
            # Entity mode normalization is handled directly from observation space bounds.
            return [
                [NoNormalization() for _ in observation_group]
                for observation_group in self.observation_names
            ]

        rules = _load_encoder_rules()
        encoders: List[List[Encoder]] = []
        missing: List[str] = []

        for observation_group, space in zip(self.observation_names, self.observation_space):
            group_encoders: List[Encoder] = []
            for index, name in enumerate(observation_group):
                rule = next((r for r in rules if _matches_rule(name, r.get("match", {}))), None)
                if rule is None:
                    missing.append(name)
                    continue

                if rule.get("warn_on_use"):
                    logger.warning("Encoder rule warning for observation '{}'", name)

                encoder = _build_encoder(rule, space, index)
                group_encoders.append(encoder)

            if len(group_encoders) != len(observation_group):
                raise ValueError(
                    "Failed to build encoders for all observations: "
                    f"expected {len(observation_group)}, built {len(group_encoders)}"
                )
            encoders.append(group_encoders)

        if missing:
            raise ValueError(
                "No encoder rule defined for observations: " + ", ".join(sorted(set(missing)))
            )

        logger.debug("Encoders initialised from external configuration")
        return encoders


# Compatibility name for existing imports and saved research scripts.
Wrapper_CityLearn = SimulationController
