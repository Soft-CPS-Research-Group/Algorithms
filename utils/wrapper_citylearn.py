import json
import faulthandler
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import mlflow
import numpy as np
import psutil
import torch
from citylearn.base import Environment
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from loguru import logger

from algorithms.agents.base_agent import BaseAgent
from utils.entity_adapter import EntityContractAdapter
from utils.checkpoint_manager import CheckpointManager
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


ENCODER_RULES_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "encoders" / "default.json"
)


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

class Wrapper_CityLearn(RLC):
    def __init__(
        self,
        env: CityLearnEnv,
        model: BaseAgent = None,
        config=None,
        job_id=None,
        progress_path=None,
        **kwargs,
    ):
        """
        Wrapper for CityLearn RLC that delegates custom behavior to a BaseAgent model.

        Parameters:
        - env: CityLearnEnv instance for the simulation environment.
        - model: BaseAgent instance implementing custom predict and update logic.
        - **kwargs: Additional arguments passed to the RLC constructor.
        """
        config = config or {}
        self.config = config
        simulator_cfg = config.get("simulator", {})
        interface_mode = str(simulator_cfg.get("interface", getattr(env, "interface", "flat"))).strip().lower() or "flat"
        self._entity_interface_mode = interface_mode == "entity"
        self._entity_topology_mode = str(
            simulator_cfg.get("topology_mode", getattr(env, "topology_mode", "static"))
        ).strip().lower() or "static"
        self._entity_dynamic_mode = self._entity_interface_mode and self._entity_topology_mode == "dynamic"
        self._algorithm_name = str((config.get("algorithm", {}) or {}).get("name", "")).strip()
        self._entity_topology_version: Optional[int] = None
        self._entity_adapter: Optional[EntityContractAdapter] = None
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
        self._entity_encoding_profile = str(
            entity_encoding_cfg.get("profile", self._entity_encoding_policy)
        ).strip().lower() or self._entity_encoding_policy

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
        self.mlflow_enabled = tracking_cfg.get("mlflow_enabled", True)
        try:
            self.log_frequency = int(tracking_cfg.get("log_frequency", 1) or 1)
        except (TypeError, ValueError):
            self.log_frequency = 1
        if self.log_frequency < 1:
            self.log_frequency = 1
        try:
            self.mlflow_step_sample_interval = int(tracking_cfg.get("mlflow_step_sample_interval", 10) or 10)
        except (TypeError, ValueError):
            self.mlflow_step_sample_interval = 10
        if self.mlflow_step_sample_interval < 1:
            self.mlflow_step_sample_interval = 1
        self.step_metric_interval = max(self.log_frequency, self.mlflow_step_sample_interval)
        self.progress_updates_enabled = bool(tracking_cfg.get("progress_updates_enabled", True))
        try:
            self.progress_update_interval = int(tracking_cfg.get("progress_update_interval", 5) or 5)
        except (TypeError, ValueError):
            self.progress_update_interval = 5
        if self.progress_update_interval < 1:
            self.progress_update_interval = 1
        self.system_metrics_enabled = bool(tracking_cfg.get("system_metrics_enabled", False))
        try:
            self.system_metrics_interval = int(tracking_cfg.get("system_metrics_interval", 10) or 10)
        except (TypeError, ValueError):
            self.system_metrics_interval = 10
        if self.system_metrics_interval < 1:
            self.system_metrics_interval = 10
        self.action_diagnostics_enabled = bool(tracking_cfg.get("action_diagnostics_enabled", False))
        self.action_diagnostics_detail = str(
            tracking_cfg.get("action_diagnostics_detail", "summary") or "summary"
        ).strip().lower()
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
        self.reward_diagnostics_detail = str(
            tracking_cfg.get("reward_diagnostics_detail", "summary") or "summary"
        ).strip().lower()
        if self.reward_diagnostics_detail not in {"summary", "per_agent"}:
            logger.warning(
                "Unknown reward_diagnostics_detail '{}'; falling back to 'summary'.",
                self.reward_diagnostics_detail,
            )
            self.reward_diagnostics_detail = "summary"
        self.runtime_profiling_enabled = bool(tracking_cfg.get("runtime_profiling_enabled", False))
        try:
            self.runtime_profiling_interval = int(tracking_cfg.get("runtime_profiling_interval", 512) or 512)
        except (TypeError, ValueError):
            self.runtime_profiling_interval = 512
        if self.runtime_profiling_interval < 1:
            self.runtime_profiling_interval = 512
        self.runtime_profiling_detail = str(
            tracking_cfg.get("runtime_profiling_detail", "summary") or "summary"
        ).strip().lower()
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
        self.max_step_seconds = self._coerce_positive_float(
            tracking_cfg.get("max_step_seconds")
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
        self._export_kpis_final_episode_only = bool(
            export_cfg.get("kpis_final_episode_only", self._export_final_episode_only)
        )
        self._export_timeseries_final_episode_only = bool(
            export_cfg.get("timeseries_final_episode_only", self._export_final_episode_only)
        )
        self._export_include_business_as_usual = bool(export_cfg.get("include_business_as_usual", True))
        self._export_business_as_usual_timeseries = bool(
            export_cfg.get("export_business_as_usual_timeseries", True)
        )
        self._export_kpi_round_decimals = export_cfg.get("kpi_round_decimals")
        self._manual_kpi_export = self._configured_export_kpis_on_episode_end and (
            not self._export_include_business_as_usual
            or not self._export_business_as_usual_timeseries
            or self._export_kpi_round_decimals is not None
            or not self._export_kpis_final_episode_only
        )
        self._manual_kpi_exported_episodes: set[int | None] = set()

        self.wrapper_reward_enabled = bool(wrapper_reward_cfg.get("enabled", False))
        self.wrapper_reward_profile = str(wrapper_reward_cfg.get("profile", "cost_limits_v1")).strip() or "cost_limits_v1"
        if self.wrapper_reward_profile not in WRAPPER_REWARD_PROFILES:
            logger.warning(
                "Unknown wrapper reward profile '{}'; falling back to 'cost_limits_v1'.",
                self.wrapper_reward_profile,
            )
            self.wrapper_reward_profile = "cost_limits_v1"
        self.wrapper_reward_profile_config = WRAPPER_REWARD_PROFILES[self.wrapper_reward_profile]
        self.wrapper_reward_version = str(self.wrapper_reward_profile_config.get("version", "unknown"))
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
        self.wrapper_reward_squash = str(wrapper_reward_cfg.get("squash", "none")).strip().lower() or "none"
        if self.wrapper_reward_squash not in {"none", "tanh"}:
            logger.warning(
                "Unknown wrapper reward squash '{}'; falling back to 'none'.",
                self.wrapper_reward_squash,
            )
            self.wrapper_reward_squash = "none"

        self.checkpoint_manager = CheckpointManager(
            base_dir=self.log_dir,
            interval=checkpoint_cfg.get("checkpoint_interval"),
            log_to_mlflow=tracking_cfg.get("mlflow_enabled", True),
            require_update_step=bool(checkpoint_cfg.get("require_update_step", True)),
            require_initial_exploration_done=bool(
                checkpoint_cfg.get("require_initial_exploration_done", True)
            ),
        )
        self.local_metrics_logger = None
        if not self.mlflow_enabled:
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
            normalization_enabled=self._entity_encoding_enabled and self._entity_encoding_policy == "minmax_space",
            clip=self._entity_encoding_clip,
            encoding_profile=self._entity_encoding_profile,
        )

        initial_observations, _ = self.env.reset()
        self._apply_entity_layout(initial_observations, force_attach=False)
        self.reset()

    def _apply_entity_layout(
        self,
        observation_payload: Mapping[str, Any],
        force_attach: bool,
        *,
        model_observations: bool = False,
    ) -> List[np.ndarray]:
        if not self._entity_interface_mode or self._entity_adapter is None:
            return []

        previous_version = self._entity_topology_version
        if model_observations:
            agent_observations, observation_names, observation_spaces = (
                self._entity_adapter.to_agent_encoded_observations(observation_payload)
            )
        else:
            agent_observations, observation_names, observation_spaces = (
                self._entity_adapter.to_agent_observations(observation_payload)
            )
        self._entity_topology_version = self._entity_adapter.topology_version

        self.episode_time_steps = int(getattr(self.episode_tracker, "episode_time_steps", self.episode_time_steps))

        topology_changed = (
            force_attach
            or previous_version is None
            or self._entity_topology_version != previous_version
        )
        if topology_changed:
            self.observation_names = observation_names
            self.observation_space = observation_spaces
            self.action_space = list(getattr(self.env, "flat_action_space", []))
            self.action_names = [list(names) for names in getattr(self.env, "action_names", [])]
            if len(self.action_names) < len(self.action_space):
                self.action_names.extend([[] for _ in range(len(self.action_space) - len(self.action_names))])
            elif len(self.action_names) > len(self.action_space):
                self.action_names = self.action_names[: len(self.action_space)]
            self.encoders = self.set_encoders()
            self._action_bounds_cache = None
        fixed_topology_algorithms = {"MADDPG", "MATD3", "MASAC", "IPPO", "MAPPO", "HAPPO"}
        if (
            topology_changed
            and self._entity_dynamic_mode
            and self._algorithm_name in fixed_topology_algorithms
            and previous_version is not None
        ):
            raise ValueError(
                f"{self._algorithm_name} supports entity interface only with topology_mode='static'. "
                "Detected topology change during runtime."
            )

        if topology_changed and self.model is not None:
            self._attach_model_environment_metadata()

        return [np.asarray(obs, dtype=np.float64) for obs in agent_observations]

    def _model_requires_raw_observation_context(self) -> bool:
        if self.model is None:
            return False
        if bool(getattr(self.model, "use_raw_observations", False)):
            return True
        if bool(getattr(self.model, "requires_raw_observation_context", False)):
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
        metadata = {
            "seconds_per_time_step": getattr(self.env, "seconds_per_time_step", None),
            "building_names": building_names,
            "interface": getattr(self.env, "interface", None),
            "topology_mode": getattr(self.env, "topology_mode", None),
            "entity_specs": getattr(self.env, "entity_specs", None) if self._entity_interface_mode else None,
        }

        try:
            self.model.attach_environment(
                observation_names=self.observation_names,
                action_names=self.action_names,
                action_space=self.action_space,
                observation_space=self.observation_space,
                metadata=metadata,
            )
        except AttributeError:
            pass

    def _resolve_building_names(self) -> Optional[List[str]]:
        raw_building_names = getattr(self.env, "building_names", None)
        if isinstance(raw_building_names, list):
            return [str(name) for name in raw_building_names]

        if self._entity_interface_mode:
            specs = getattr(self.env, "entity_specs", None)
            if isinstance(specs, Mapping):
                table_specs = specs.get("tables", {}) if isinstance(specs.get("tables", {}), Mapping) else {}
                building_table = table_specs.get("building", {}) if isinstance(table_specs.get("building", {}), Mapping) else {}
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

    @staticmethod
    def _coerce_positive_int(value) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _coerce_non_negative_int(value) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    @staticmethod
    def _coerce_positive_float(value) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(parsed) or np.isinf(parsed) or parsed <= 0.0:
            return None
        return parsed

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        parsed = str(value).strip()
        return parsed or None

    def _resolve_progress_totals(self, episodes: int) -> tuple[Optional[int], Optional[int]]:
        step_total = self._coerce_positive_int(self.episode_time_steps)
        if step_total is None:
            return None, None
        return step_total, episodes * step_total

    def _progress_phase_in_window(self) -> bool:
        if not self.progress_updates_enabled or not self.progress_phase_updates_enabled:
            return False
        if self.global_step % self.progress_update_interval != 0:
            return False
        if (
            self.progress_phase_start_step is not None
            and self.global_step < self.progress_phase_start_step
        ):
            return False
        if (
            self.progress_phase_end_step is not None
            and self.global_step > self.progress_phase_end_step
        ):
            return False
        return True

    def _runtime_resource_snapshot(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        try:
            rss_mb = self._process.memory_info().rss / (1024 ** 2)
            snapshot["process_rss_mb"] = round(float(rss_mb), 3)
        except Exception:
            pass

        try:
            virtual_memory = psutil.virtual_memory()
            snapshot["system_available_ram_mb"] = round(
                float(virtual_memory.available / (1024 ** 2)),
                3,
            )
            snapshot["system_ram_percent"] = round(float(virtual_memory.percent), 3)
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                snapshot["gpu_allocated_mb"] = round(
                    float(torch.cuda.memory_allocated() / (1024 ** 2)),
                    3,
                )
                snapshot["gpu_reserved_mb"] = round(
                    float(torch.cuda.memory_reserved() / (1024 ** 2)),
                    3,
                )
            except Exception:
                pass

        return snapshot

    def _write_phase_progress(
        self,
        *,
        phase: str,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
        rewards: Optional[List[float]] = None,
        status: str = "running",
        force: bool = False,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._update_stall_watchdog_for_phase(
            phase=phase,
            episode=episode,
            step=step,
            episode_total=episode_total,
            step_total=step_total,
            global_step_total=global_step_total,
            status=status,
            rewards=rewards,
        )

        if not force and not self._progress_phase_in_window():
            return

        payload_extra: Dict[str, Any] = {
            "phase": phase,
            "entity_topology_version": self._entity_topology_version,
            "entity_model_observations_direct": bool(
                getattr(self, "_entity_model_observations_direct", False)
            ),
        }
        payload_extra.update(self._runtime_resource_snapshot())
        if extra:
            payload_extra.update(dict(extra))

        self.progress_tracker.update(
            episode=episode,
            step=step,
            global_step=self.global_step,
            rewards=rewards,
            episode_total=episode_total,
            step_total=step_total,
            global_step_total=global_step_total,
            status=status,
            extra=payload_extra,
        )

    def _update_stall_watchdog_for_phase(
        self,
        *,
        phase: str,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
        status: str,
        rewards: Optional[List[float]],
    ) -> None:
        if not self.stall_watchdog_enabled:
            return

        if phase == "step_start" or phase.endswith("_start"):
            self._arm_stall_watchdog(
                phase=phase,
                episode=episode,
                step=step,
                episode_total=episode_total,
                step_total=step_total,
                global_step_total=global_step_total,
                status=status,
                rewards=rewards,
            )
            return

        if phase == "step_end" or phase.endswith("_end") or phase in {"model_update_skipped", "episode_end"}:
            self._cancel_stall_watchdog()

    def _stall_watchdog_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        if self._stall_watchdog_traceback_path is not None or self._stall_watchdog_context_path is not None:
            return self._stall_watchdog_traceback_path, self._stall_watchdog_context_path

        raw_path = self.stall_watchdog_traceback_file
        base_dir: Optional[Path] = None
        if self.log_dir:
            base_dir = Path(self.log_dir)
        elif self.progress_tracker.progress_path is not None:
            base_dir = self.progress_tracker.progress_path.parent
        else:
            runtime_job_dir = self.config.get("runtime", {}).get("job_dir") if isinstance(self.config, dict) else None
            if runtime_job_dir:
                base_dir = Path(runtime_job_dir) / "logs"

        if raw_path:
            traceback_path = Path(raw_path)
            if not traceback_path.is_absolute():
                root = base_dir or Path.cwd()
                traceback_path = root / traceback_path
        elif base_dir is not None:
            safe_job_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.job_id or "job")).strip("_") or "job"
            traceback_path = base_dir / f"{safe_job_id}_stall_watchdog.log"
        else:
            traceback_path = None

        context_path = None
        if traceback_path is not None:
            suffix = traceback_path.suffix
            context_path = traceback_path.with_suffix(f"{suffix}.context.json" if suffix else ".context.json")

        self._stall_watchdog_traceback_path = traceback_path
        self._stall_watchdog_context_path = context_path
        return traceback_path, context_path

    def _stall_watchdog_output_file(self):
        traceback_path, _ = self._stall_watchdog_paths()
        if traceback_path is None:
            return sys.stderr

        if self._stall_watchdog_file_handle is None or self._stall_watchdog_file_handle.closed:
            try:
                traceback_path.parent.mkdir(parents=True, exist_ok=True)
                self._stall_watchdog_file_handle = traceback_path.open("a", encoding="utf-8", buffering=1)
            except Exception as exc:
                logger.warning("Failed to open stall watchdog log {}: {}", traceback_path, exc)
                return sys.stderr

        return self._stall_watchdog_file_handle

    def _write_stall_watchdog_context(self, context: Mapping[str, Any], *, force: bool = False) -> None:
        _, context_path = self._stall_watchdog_paths()
        if context_path is None:
            return

        if not force:
            interval = max(int(self.stall_watchdog_context_interval_steps or 1), 1)
            global_step = self._coerce_non_negative_int(context.get("global_step")) or 0
            last_global_step = self._stall_watchdog_last_context_global_step
            if last_global_step is not None and (global_step - last_global_step) < interval:
                return
            self._stall_watchdog_last_context_global_step = global_step

        try:
            context_path.parent.mkdir(parents=True, exist_ok=True)
            context_path.write_text(json.dumps(dict(context), indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write stall watchdog context {}: {}", context_path, exc)

    def _arm_stall_watchdog(
        self,
        *,
        phase: str,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
        status: str,
        rewards: Optional[List[float]],
    ) -> None:
        timeout = self.stall_watchdog_timeout_seconds
        if timeout is None or timeout <= 0:
            return

        context: Dict[str, Any] = {
            "job_id": self.job_id,
            "phase": phase,
            "status": status,
            "episode": episode,
            "episode_current": max(0, episode) + 1,
            "step": step,
            "step_current": max(0, step) + 1,
            "global_step": self.global_step,
            "episode_total": episode_total,
            "step_total": step_total,
            "global_step_total": global_step_total,
            "timeout_seconds": float(timeout),
            "exit_on_timeout": self.stall_watchdog_exit_on_timeout,
            "repeat": self.stall_watchdog_repeat,
            "entity_topology_version": self._entity_topology_version,
            "entity_model_observations_direct": bool(
                getattr(self, "_entity_model_observations_direct", False)
            ),
            **self._runtime_resource_snapshot(),
        }
        if rewards is not None:
            context["rewards"] = list(rewards)
        self._write_stall_watchdog_context(context, force=phase != "step_start")

        try:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(
                float(timeout),
                repeat=self.stall_watchdog_repeat,
                file=self._stall_watchdog_output_file(),
                exit=self.stall_watchdog_exit_on_timeout,
            )
            self._stall_watchdog_armed_phase = phase
        except Exception as exc:
            logger.warning("Failed to arm stall watchdog for phase {}: {}", phase, exc)

    def _cancel_stall_watchdog(self) -> None:
        if not self.stall_watchdog_enabled:
            return

        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception as exc:
            logger.warning("Failed to cancel stall watchdog: {}", exc)
        self._stall_watchdog_armed_phase = None

    def _close_stall_watchdog_file(self) -> None:
        handle = self._stall_watchdog_file_handle
        if handle is None or handle is sys.stderr:
            return

        try:
            if not handle.closed:
                handle.close()
        except Exception as exc:
            logger.warning("Failed to close stall watchdog log handle: {}", exc)
        finally:
            self._stall_watchdog_file_handle = None

    def _enforce_resource_guards(
        self,
        *,
        phase: str,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
    ) -> None:
        if not self.resource_guard_enabled:
            return

        snapshot = self._runtime_resource_snapshot()
        failures: List[str] = []
        rss_mb = snapshot.get("process_rss_mb")
        available_mb = snapshot.get("system_available_ram_mb")
        if self.max_process_rss_mb is not None and rss_mb is not None and rss_mb > self.max_process_rss_mb:
            failures.append(
                f"process RSS {rss_mb:.1f} MB exceeded limit {self.max_process_rss_mb:.1f} MB"
            )
        if (
            self.min_available_ram_mb is not None
            and available_mb is not None
            and available_mb < self.min_available_ram_mb
        ):
            failures.append(
                f"available RAM {available_mb:.1f} MB below limit {self.min_available_ram_mb:.1f} MB"
            )
        if not failures:
            return

        message = "; ".join(failures)
        self._write_phase_progress(
            phase=phase,
            episode=episode,
            step=step,
            episode_total=episode_total,
            step_total=step_total,
            global_step_total=global_step_total,
            status="failed",
            force=True,
            extra={
                "error_type": "ResourceGuardError",
                "error_message": message,
                **snapshot,
            },
        )
        self._cancel_stall_watchdog()
        raise MemoryError(message)

    def _enforce_step_duration_guard(
        self,
        *,
        step_duration: float,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
        rewards: Optional[List[float]],
    ) -> None:
        if self.max_step_seconds is None or step_duration <= self.max_step_seconds:
            return

        message = (
            f"Step duration {step_duration:.3f}s exceeded configured limit "
            f"{self.max_step_seconds:.3f}s at global step {self.global_step}."
        )
        self._write_phase_progress(
            phase="step_duration_guard",
            episode=episode,
            step=step,
            episode_total=episode_total,
            step_total=step_total,
            global_step_total=global_step_total,
            rewards=rewards,
            status="failed",
            force=True,
            extra={
                "error_type": "StepDurationGuardError",
                "error_message": message,
                "step_duration_seconds": round(float(step_duration), 6),
                "max_step_seconds": round(float(self.max_step_seconds), 6),
            },
        )
        self._cancel_stall_watchdog()
        raise TimeoutError(message)

    def _configure_episode_exports(self, episode: int, episodes: int) -> bool:
        """Enable KPI export and timeseries rendering independently per episode."""

        is_final_episode = episode >= episodes - 1
        export_this_episode = self._configured_export_kpis_on_episode_end and (
            not self._export_kpis_final_episode_only or is_final_episode
        )
        render_this_episode = self._configured_render_enabled and (
            not self._export_timeseries_final_episode_only or is_final_episode
        )

        if hasattr(self.env, "render_enabled"):
            self.env.render_enabled = render_this_episode
        if hasattr(self.env, "export_kpis_on_episode_end"):
            self.env.export_kpis_on_episode_end = export_this_episode and not self._manual_kpi_export

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
            logger.warning("Simulator does not expose export_final_kpis(); skipping manual KPI export.")
            return
        if episode in self._manual_kpi_exported_episodes:
            return

        export_business_as_usual_timeseries = self._export_business_as_usual_timeseries
        if self._export_timeseries_final_episode_only:
            export_business_as_usual_timeseries = export_business_as_usual_timeseries and is_final_episode

        kwargs = {
            "include_business_as_usual": self._export_include_business_as_usual,
            "export_business_as_usual_timeseries": export_business_as_usual_timeseries,
            "kpi_round_decimals": self._export_kpi_round_decimals,
        }
        if not self._export_kpis_final_episode_only and episode is not None:
            kwargs["filepath"] = f"exported_kpis_ep{episode}.csv"

        self.env.export_final_kpis(**kwargs)
        if not self._export_kpis_final_episode_only and is_final_episode and episode is not None:
            final_kwargs = dict(kwargs)
            final_kwargs.pop("filepath", None)
            self.env.export_final_kpis(**final_kwargs)

        self._manual_kpi_exported_episodes.add(episode)

    def set_model(self, model: BaseAgent):
        """
        Set the model after initialization.
        """
        self.model = model
        self._attach_model_environment_metadata()


    def learn(self, episodes=None, deterministic=None, deterministic_finish=None):
        """
        Train agent with MLflow logging for rewards (per step and per agent), PyTorch GPU memory usage, and system usage.
        """
        if self.model is None:
            logger.error("Wrapper invoked without a model; aborting training.")
            raise ValueError("Model is not set. Use `set_model` to provide a model.")

        episodes = episodes or self.default_episodes
        deterministic_finish = deterministic_finish if deterministic_finish is not None else False
        deterministic = deterministic if deterministic is not None else False

        total_rewards_across_episodes = []  # To track overall reward trends

        for episode in range(episodes):
            start_episode_time = time.time()
            deterministic = deterministic or (deterministic_finish and episode >= episodes - 1)
            export_this_episode = self._configure_episode_exports(episode, episodes)
            self._entity_model_observations_direct = self._can_use_direct_entity_model_observations()
            self._write_phase_progress(
                phase="episode_reset_start",
                episode=episode,
                step=0,
                episode_total=episodes,
                step_total=None,
                global_step_total=None,
            )
            raw_observations, _ = self.env.reset()
            if self._entity_interface_mode:
                observations = self._apply_entity_layout(
                    raw_observations,
                    force_attach=True,
                    model_observations=self._entity_model_observations_direct,
                )
            else:
                observations = raw_observations
            self.episode_time_steps = self.episode_tracker.episode_time_steps
            episode_step_total, global_step_total = self._resolve_progress_totals(episodes)
            self._write_phase_progress(
                phase="episode_reset_end",
                episode=episode,
                step=0,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
            )
            terminated = False
            truncated = False
            time_step = 0
            rewards_list = []  # Stores rewards per step

            while not (terminated or truncated):
                step_start_time = time.time()
                self.global_step += 1
                should_profile_step = (
                    self.runtime_profiling_enabled
                    and self.global_step % self.runtime_profiling_interval == 0
                )
                step_profile_start_time = time.perf_counter() if should_profile_step else 0.0
                runtime_profile_metrics: Dict[str, float] = {}
                logger.debug(
                    "Global step {} (episode {}, timestep {})",
                    self.global_step,
                    episode,
                    time_step,
                )
                self._enforce_resource_guards(
                    phase="step_start",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                )
                self._write_phase_progress(
                    phase="step_start",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                )

                step_observations = [np.asarray(obs, dtype=np.float64) for obs in observations]
                self._write_phase_progress(
                    phase="predict_start",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                )
                phase_start_time = time.perf_counter() if should_profile_step else 0.0
                actions = self.predict(observations, deterministic=deterministic)
                if should_profile_step:
                    runtime_profile_metrics["Runtime/predict_seconds"] = (
                        time.perf_counter() - phase_start_time
                    )
                self._write_phase_progress(
                    phase="predict_end",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                )
                self._write_phase_progress(
                    phase="action_prepare_start",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                )
                phase_start_time = time.perf_counter() if should_profile_step else 0.0
                actions = self._clip_actions(actions)
                if not self._entity_interface_mode:
                    self.actions = actions
                logger.debug("Predicted actions: {}", actions)

                # Apply actions to CityLearn environment
                env_actions = self._to_env_actions(actions)
                if should_profile_step:
                    runtime_profile_metrics["Runtime/action_prepare_seconds"] = (
                        time.perf_counter() - phase_start_time
                    )
                self._write_phase_progress(
                    phase="action_prepare_end",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                )
                self._write_phase_progress(
                    phase="env_step_start",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                )
                phase_start_time = time.perf_counter() if should_profile_step else 0.0
                next_observations_raw, rewards, terminated, truncated, _ = self.env.step(env_actions)
                if should_profile_step:
                    runtime_profile_metrics["Runtime/env_step_seconds"] = (
                        time.perf_counter() - phase_start_time
                    )
                self._write_phase_progress(
                    phase="env_step_end",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=rewards,
                )
                self._write_phase_progress(
                    phase="entity_layout_start" if self._entity_interface_mode else "observation_prepare_start",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=rewards,
                )
                phase_start_time = time.perf_counter() if should_profile_step else 0.0
                if self._entity_interface_mode:
                    next_observations = self._apply_entity_layout(
                        next_observations_raw,
                        force_attach=False,
                        model_observations=self._entity_model_observations_direct,
                    )
                else:
                    next_observations = next_observations_raw
                if should_profile_step:
                    entity_layout_seconds = time.perf_counter() - phase_start_time
                    runtime_profile_metrics["Runtime/observation_encoding_seconds"] = entity_layout_seconds
                    runtime_profile_metrics["Runtime/entity_layout_seconds"] = entity_layout_seconds
                self._write_phase_progress(
                    phase="entity_layout_end" if self._entity_interface_mode else "observation_prepare_end",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=rewards,
                )
                self._write_phase_progress(
                    phase="reward_shaping_start",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=rewards,
                )
                phase_start_time = time.perf_counter() if should_profile_step else 0.0
                rewards = self._shape_rewards(rewards, next_observations)
                if should_profile_step:
                    runtime_profile_metrics["Runtime/reward_shaping_seconds"] = (
                        time.perf_counter() - phase_start_time
                    )
                self._write_phase_progress(
                    phase="reward_shaping_end",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=rewards,
                )
                rewards_list.append(rewards)

                # Update model if not in deterministic mode
                if not deterministic:
                    self._write_phase_progress(
                        phase="model_update_start",
                        episode=episode,
                        step=time_step,
                        episode_total=episodes,
                        step_total=episode_step_total,
                        global_step_total=global_step_total,
                        rewards=rewards,
                    )
                    phase_start_time = time.perf_counter() if should_profile_step else 0.0
                    self.update(
                        observations,
                        actions,
                        rewards,
                        next_observations,
                        terminated=terminated,
                        truncated=truncated,
                    )
                    if should_profile_step:
                        runtime_profile_metrics["Runtime/agent_update_seconds"] = (
                            time.perf_counter() - phase_start_time
                        )
                        runtime_profile_metrics["Runtime/model_observation_encoding_seconds"] = float(
                            getattr(self, "_last_model_observation_encoding_seconds", 0.0) or 0.0
                        )
                        runtime_profile_metrics["Runtime/model_update_seconds"] = float(
                            getattr(self, "_last_model_update_seconds", 0.0) or 0.0
                        )
                    logger.debug("Model update executed at global step {}", self.global_step)
                    self._write_phase_progress(
                        phase="model_update_end",
                        episode=episode,
                        step=time_step,
                        episode_total=episodes,
                        step_total=episode_step_total,
                        global_step_total=global_step_total,
                        rewards=rewards,
                    )

                    self._write_phase_progress(
                        phase="checkpoint_start",
                        episode=episode,
                        step=time_step,
                        episode_total=episodes,
                        step_total=episode_step_total,
                        global_step_total=global_step_total,
                        rewards=rewards,
                    )
                    phase_start_time = time.perf_counter() if should_profile_step else 0.0
                    self.checkpoint_manager.maybe_save(
                        agent=self.model,
                        step=self.global_step,
                        initial_exploration_done=self.initial_exploration_done,
                        update_step=self.update_step,
                    )
                    if should_profile_step:
                        runtime_profile_metrics["Runtime/checkpoint_seconds"] = (
                            time.perf_counter() - phase_start_time
                        )
                    self._write_phase_progress(
                        phase="checkpoint_end",
                        episode=episode,
                        step=time_step,
                        episode_total=episodes,
                        step_total=episode_step_total,
                        global_step_total=global_step_total,
                        rewards=rewards,
                    )
                elif should_profile_step:
                    runtime_profile_metrics["Runtime/agent_update_seconds"] = 0.0
                    runtime_profile_metrics["Runtime/model_observation_encoding_seconds"] = 0.0
                    runtime_profile_metrics["Runtime/model_update_seconds"] = 0.0
                    runtime_profile_metrics["Runtime/checkpoint_seconds"] = 0.0
                elif self._progress_phase_in_window():
                    self._write_phase_progress(
                        phase="model_update_skipped",
                        episode=episode,
                        step=time_step,
                        episode_total=episodes,
                        step_total=episode_step_total,
                        global_step_total=global_step_total,
                        rewards=rewards,
                    )

                observations = [o for o in next_observations]

                # Reduce system monitoring frequency
                cpu_usage = None
                ram_usage = None
                if self.system_metrics_enabled and (self.global_step % self.system_metrics_interval == 0):
                    cpu_usage = psutil.cpu_percent()
                    ram_usage = psutil.virtual_memory().percent

                # PyTorch-specific GPU memory tracking (kept only PyTorch measurement)
                if (
                    self.system_metrics_enabled
                    and (self.global_step % self.system_metrics_interval == 0)
                    and torch.cuda.is_available()
                ):
                    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # Convert to MB
                    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # Reserved for caching
                else:
                    gpu_mem_allocated = None
                    gpu_mem_reserved = None

                # Step duration calculation
                step_duration = time.time() - step_start_time
                self._enforce_resource_guards(
                    phase="step_end",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                )
                self._enforce_step_duration_guard(
                    step_duration=step_duration,
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=rewards,
                )
                self._write_phase_progress(
                    phase="step_end",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=rewards,
                    extra={"step_duration_seconds": round(float(step_duration), 6)},
                )
                if should_profile_step:
                    runtime_profile_metrics["Runtime/step_seconds"] = step_duration
                    runtime_profile_metrics["Runtime/step_perf_seconds"] = (
                        time.perf_counter() - step_profile_start_time
                    )
                should_log_step = self._should_log_step(self.global_step) or should_profile_step
                if should_log_step:
                    diagnostics_start_time = time.perf_counter()
                    metrics = {
                        f"Agent_{i}_Reward": reward for i, reward in enumerate(rewards)
                    }
                    if cpu_usage is not None:
                        metrics["CPU_Usage"] = cpu_usage
                        metrics["RAM_Usage"] = ram_usage
                    if gpu_mem_allocated is not None:
                        metrics["GPU_PyTorch_Allocated_MB"] = gpu_mem_allocated
                        metrics["GPU_PyTorch_Reserved_MB"] = gpu_mem_reserved
                    metrics["Step_Duration"] = step_duration
                    if should_profile_step:
                        metrics.update(runtime_profile_metrics)
                    metrics.update(self._collect_model_status_metrics())
                    metrics.update(self._build_action_diagnostic_metrics(actions, step_observations))
                    metrics.update(self._build_reward_component_metrics())
                    if not mlflow.active_run():
                        metrics.update(self._consume_model_training_metrics())
                    if should_profile_step:
                        metrics["Runtime/diagnostics_build_seconds"] = (
                            time.perf_counter() - diagnostics_start_time
                        )

                    if mlflow.active_run():
                        mlflow.log_metrics(metrics, step=self.global_step)
                    elif self.local_metrics_logger:
                        self.local_metrics_logger.log(metrics, self.global_step)

                    logger.info(
                        "Time step: {}/{}, Episode: {}/{}, Actions: {}, Rewards: {}, CPU: {}%, RAM: {}%, "
                        "GPU Allocated: {} MB, GPU Reserved: {} MB Step Duration: {}",
                        time_step + 1,
                        self.episode_time_steps,
                        episode + 1,
                        episodes,
                        actions,
                        rewards,
                        cpu_usage,
                        ram_usage,
                        gpu_mem_allocated,
                        gpu_mem_reserved,
                        step_duration,
                    )

                if self.progress_updates_enabled and (self.global_step % self.progress_update_interval == 0):
                    self.progress_tracker.update(
                        episode=episode,
                        step=time_step,
                        global_step=self.global_step,
                        rewards=rewards,
                        episode_total=episodes,
                        step_total=episode_step_total,
                        global_step_total=global_step_total,
                        status="running",
                        extra={
                            "phase": "step_end",
                            "step_duration_seconds": round(float(step_duration), 6),
                            **self._runtime_resource_snapshot(),
                        },
                    )

                time_step += 1

            last_rewards = rewards_list[-1] if rewards_list else None
            self._write_phase_progress(
                phase="episode_export_start",
                episode=episode,
                step=max(time_step - 1, 0),
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=last_rewards,
            )
            self._export_episode_kpis_if_needed(
                export_this_episode,
                episode=episode,
                is_final_episode=episode + 1 >= episodes,
            )
            self._write_phase_progress(
                phase="episode_export_end",
                episode=episode,
                step=max(time_step - 1, 0),
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=last_rewards,
            )

            if self.progress_updates_enabled and time_step > 0:
                self.progress_tracker.update(
                    episode=episode,
                    step=time_step - 1,
                    global_step=self.global_step,
                    rewards=rewards_list[-1],
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    status="completed" if episode + 1 >= episodes else "running",
                    extra={
                        "phase": "episode_end",
                        **self._runtime_resource_snapshot(),
                    },
                )

            # Compute rewards statistics for this episode
            reward_vectors = [np.asarray(step_rewards, dtype=np.float64).reshape(-1) for step_rewards in rewards_list]
            if len(reward_vectors) == 0:
                rewards_array = np.zeros((0, 0), dtype=np.float64)
            else:
                max_agents = max(vector.shape[0] for vector in reward_vectors)
                rewards_array = np.full((len(reward_vectors), max_agents), np.nan, dtype=np.float64)
                for row, vector in enumerate(reward_vectors):
                    rewards_array[row, : vector.shape[0]] = vector

            if rewards_array.size == 0:
                rewards_summary = {
                    "sum": np.array([], dtype=np.float64),
                    "mean": np.array([], dtype=np.float64),
                    "min": np.array([], dtype=np.float64),
                    "max": np.array([], dtype=np.float64),
                }
            else:
                valid_mask = ~np.isnan(rewards_array)
                valid_counts = valid_mask.sum(axis=0)
                sums = np.nansum(rewards_array, axis=0)
                means = np.divide(sums, np.maximum(valid_counts, 1), where=np.maximum(valid_counts, 1) > 0)
                mins = np.array(
                    [
                        np.nanmin(rewards_array[:, i]) if valid_counts[i] > 0 else 0.0
                        for i in range(rewards_array.shape[1])
                    ],
                    dtype=np.float64,
                )
                maxs = np.array(
                    [
                        np.nanmax(rewards_array[:, i]) if valid_counts[i] > 0 else 0.0
                        for i in range(rewards_array.shape[1])
                    ],
                    dtype=np.float64,
                )
                rewards_summary = {
                    "sum": sums,
                    "mean": means,
                    "min": mins,
                    "max": maxs,
                }

            # Store rewards for global tracking
            total_rewards_across_episodes.append(rewards_summary['sum'])

            # Log episode statistics
            episode_metrics = {}

            for i in range(len(rewards_summary['sum'])):
                episode_metrics[f"Agent_{i}_Episode_Reward_Sum"] = rewards_summary['sum'][i]
                episode_metrics[f"Agent_{i}_Episode_Reward_Mean"] = rewards_summary['mean'][i]
                episode_metrics[f"Agent_{i}_Episode_Reward_Min"] = rewards_summary['min'][i]
                episode_metrics[f"Agent_{i}_Episode_Reward_Max"] = rewards_summary['max'][i]

            episode_duration = time.time() - start_episode_time
            episode_metrics["Episode_Duration"] = episode_duration
            if mlflow.active_run():
                mlflow.log_metrics(episode_metrics, step=episode)
            elif self.local_metrics_logger:
                self.local_metrics_logger.log(episode_metrics, episode)

            logger.info(
                "Completed episode {}/{}, reward summary: {}, duration: {:.2f}s",
                episode + 1,
                episodes,
                rewards_summary,
                episode_duration,
            )

        # Aggregate rewards across episodes
        if len(total_rewards_across_episodes) == 0:
            total_rewards_matrix = np.zeros((0, 0), dtype=np.float64)
        else:
            max_agents = max(np.asarray(values).reshape(-1).shape[0] for values in total_rewards_across_episodes)
            total_rewards_matrix = np.full((len(total_rewards_across_episodes), max_agents), np.nan, dtype=np.float64)
            for row, values in enumerate(total_rewards_across_episodes):
                vector = np.asarray(values, dtype=np.float64).reshape(-1)
                total_rewards_matrix[row, : vector.shape[0]] = vector

        # Compute overall statistics across episodes
        if total_rewards_matrix.size == 0:
            overall_rewards_summary = {
                "sum": np.array([], dtype=np.float64),
                "mean": np.array([], dtype=np.float64),
                "min": np.array([], dtype=np.float64),
                "max": np.array([], dtype=np.float64),
            }
        else:
            valid_mask = ~np.isnan(total_rewards_matrix)
            valid_counts = valid_mask.sum(axis=0)
            sums = np.nansum(total_rewards_matrix, axis=0)
            means = np.divide(sums, np.maximum(valid_counts, 1), where=np.maximum(valid_counts, 1) > 0)
            mins = np.array(
                [
                    np.nanmin(total_rewards_matrix[:, i]) if valid_counts[i] > 0 else 0.0
                    for i in range(total_rewards_matrix.shape[1])
                ],
                dtype=np.float64,
            )
            maxs = np.array(
                [
                    np.nanmax(total_rewards_matrix[:, i]) if valid_counts[i] > 0 else 0.0
                    for i in range(total_rewards_matrix.shape[1])
                ],
                dtype=np.float64,
            )
            overall_rewards_summary = {
                "sum": sums,
                "mean": means,
                "min": mins,
                "max": maxs,
            }

        # Log overall statistics
        overall_metrics = {}

        for i in range(len(overall_rewards_summary['sum'])):
            overall_metrics[f"Agent_{i}_Overall_Reward_Sum"] = overall_rewards_summary['sum'][i]
            overall_metrics[f"Agent_{i}_Overall_Reward_Mean"] = overall_rewards_summary['mean'][i]
            overall_metrics[f"Agent_{i}_Overall_Reward_Min"] = overall_rewards_summary['min'][i]
            overall_metrics[f"Agent_{i}_Overall_Reward_Max"] = overall_rewards_summary['max'][i]

        if mlflow.active_run():
            mlflow.log_metrics(overall_metrics)
        elif self.local_metrics_logger:
            # Use -1 to denote aggregate metrics when logging locally.
            self.local_metrics_logger.log(overall_metrics, -1)
        self._cancel_stall_watchdog()
        self._close_stall_watchdog_file()

    def predict(self, observations, deterministic=None):
        """
        Updates the predict action logic. It now uses a mix of algorithm and the next time step.
        """
        if self.model is None:
            raise ValueError("Model is not set. Use `set_model` to provide a model.")

        direct_entity_model_observations = bool(
            getattr(self, "_entity_model_observations_direct", False)
        )
        if direct_entity_model_observations:
            encoded_observations = [
                np.asarray(obs, dtype=np.float64) for obs in observations
            ]
        else:
            encoded_observations = self._encode_observations_for_model(observations)

        observation_context_hook = getattr(self.model, "set_observation_context", None)
        if callable(observation_context_hook):
            observation_context_hook(
                raw_observations=None
                if direct_entity_model_observations
                else observations,
                encoded_observations=encoded_observations,
            )

        actions = self.model.predict(encoded_observations, deterministic)
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
        """Clip model actions to each agent action-space bounds."""
        if not isinstance(actions, list):
            raise ValueError("Model predicted actions must be provided as a list.")

        action_bounds = self._get_action_bounds_cache()
        clipped_actions: List[List[float]] = []
        for agent_idx, (low, high) in enumerate(action_bounds):
            raw = actions[agent_idx] if agent_idx < len(actions) else []
            action_array = np.asarray(raw, dtype=np.float64).reshape(-1)
            if low.shape[0] == 0 or high.shape[0] == 0:
                clipped_actions.append(action_array.tolist())
                continue

            expected_dim = low.shape[0]
            if action_array.shape[0] != expected_dim:
                logger.warning(
                    "Action dimension mismatch for agent {}: predicted={}, expected={}. "
                    "Padding/truncating before clipping.",
                    agent_idx,
                    action_array.shape[0],
                    expected_dim,
                )
                fixed = np.zeros(expected_dim, dtype=np.float64)
                copy_dim = min(expected_dim, action_array.shape[0])
                if copy_dim > 0:
                    fixed[:copy_dim] = action_array[:copy_dim]
                action_array = fixed

            action_array = np.clip(action_array, low, high)

            clipped_actions.append(action_array.tolist())

        return clipped_actions

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

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if np.isnan(parsed) or np.isinf(parsed):
            return default
        return parsed

    @staticmethod
    def _metric_safe_name(value: Any) -> str:
        text = str(value or "unknown")
        safe = [char if char.isalnum() or char in {"_", "-", "."} else "_" for char in text]
        return "".join(safe).strip("_") or "unknown"

    @staticmethod
    def _is_deferrable_action_name(action_name: str) -> bool:
        raw = str(action_name or "").lower()
        return "deferrable_appliance" in raw or raw.endswith("::start") or raw == "start"

    @staticmethod
    def _is_ev_action_name(action_name: str) -> bool:
        raw = str(action_name or "").lower()
        return "electric_vehicle_storage" in raw or "ev_storage" in raw

    @staticmethod
    def _charger_token_from_action_name(action_name: str) -> str | None:
        match = re.search(r"charger_\d+_\d+", str(action_name or "").lower())
        return match.group(0) if match else None

    @staticmethod
    def _is_storage_action_name(action_name: str) -> bool:
        raw = str(action_name or "").lower()
        return "electrical_storage" in raw or raw in {"battery", "storage"}

    def _action_bounds_for_agent(self, agent_index: int, action_count: int) -> tuple[np.ndarray, np.ndarray]:
        if agent_index < len(getattr(self, "action_space", [])):
            action_space = self.action_space[agent_index]
            if hasattr(action_space, "low") and hasattr(action_space, "high"):
                low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
                high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
                if low.shape[0] == action_count and high.shape[0] == action_count:
                    return low, high
        return (
            np.full(action_count, -1.0, dtype=np.float64),
            np.full(action_count, 1.0, dtype=np.float64),
        )

    def _ev_connected_for_action(
        self,
        *,
        agent_index: int,
        observation: np.ndarray,
        action_name: str,
    ) -> Optional[bool]:
        charger_token = self._charger_token_from_action_name(action_name)
        if not charger_token:
            return None
        names = (
            [str(name).lower() for name in self.observation_names[agent_index]]
            if agent_index < len(getattr(self, "observation_names", []))
            else []
        )
        if not names:
            return None
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        for obs_index, obs_name in enumerate(names[: values.shape[0]]):
            if charger_token not in obs_name:
                continue
            if obs_name.endswith("::connected_state") or obs_name.endswith("__connected_state"):
                return bool(values[obs_index] > 0.5)
        return None

    def _build_action_diagnostic_metrics(
        self,
        actions: List[List[float]],
        observations: List[np.ndarray],
    ) -> Dict[str, float]:
        if not getattr(self, "action_diagnostics_enabled", False):
            return {}

        rows: List[Dict[str, Any]] = []
        deferrable_start_delays: List[float] = []
        deferrable_pending_can_start_count = 0
        deferrable_start_command_count = 0
        deferrable_start_when_available_count = 0

        for agent_index, agent_actions in enumerate(actions):
            action_values = np.asarray(agent_actions, dtype=np.float64).reshape(-1)
            low, high = self._action_bounds_for_agent(agent_index, action_values.shape[0])
            span = np.maximum(high - low, 1.0e-9)
            names = (
                [str(name) for name in self.action_names[agent_index]]
                if agent_index < len(getattr(self, "action_names", []))
                else []
            )

            for action_index, value in enumerate(action_values):
                action_name = names[action_index] if action_index < len(names) else f"action_{action_index}"
                action_low = float(low[action_index])
                action_high = float(high[action_index])
                action_span = float(span[action_index])
                threshold = action_low + 0.5 * action_span
                category = "other"
                if self._is_deferrable_action_name(action_name):
                    category = "deferrable"
                    threshold = action_low + 0.5 * action_span
                    if getattr(self.model, "deferrable_trigger_threshold", None) is not None:
                        threshold = action_low + float(self.model.deferrable_trigger_threshold) * action_span
                    if value > threshold:
                        deferrable_start_command_count += 1
                    pending, can_start = self._deferrable_pending_can_start(
                        agent_index=agent_index,
                        observation=observations[agent_index] if agent_index < len(observations) else np.array([]),
                        action_name=action_name,
                    )
                    if pending and can_start:
                        deferrable_pending_can_start_count += 1
                        if value > threshold:
                            deferrable_start_when_available_count += 1
                            key = (agent_index, action_name)
                            deferrable_start_delays.append(float(self._deferrable_wait_steps.get(key, 0)))
                            self._deferrable_wait_steps.pop(key, None)
                        else:
                            key = (agent_index, action_name)
                            self._deferrable_wait_steps[key] = self._deferrable_wait_steps.get(key, 0) + 1
                    else:
                        self._deferrable_wait_steps.pop((agent_index, action_name), None)
                elif self._is_ev_action_name(action_name):
                    category = "ev"
                elif self._is_storage_action_name(action_name):
                    category = "storage"
                ev_connected = (
                    self._ev_connected_for_action(
                        agent_index=agent_index,
                        observation=observations[agent_index] if agent_index < len(observations) else np.array([]),
                        action_name=action_name,
                    )
                    if category == "ev"
                    else None
                )

                rows.append(
                    {
                        "agent_index": agent_index,
                        "action_index": action_index,
                        "action_name": action_name,
                        "category": category,
                        "ev_connected": ev_connected,
                        "value": float(value),
                        "low": action_low,
                        "high": action_high,
                        "span": action_span,
                        "threshold": threshold,
                    }
                )

        if not rows:
            return {}

        values = np.asarray([row["value"] for row in rows], dtype=np.float64)
        lows = np.asarray([row["low"] for row in rows], dtype=np.float64)
        highs = np.asarray([row["high"] for row in rows], dtype=np.float64)
        spans = np.maximum(highs - lows, 1.0e-9)
        saturation_tolerance = np.maximum(self.action_saturation_tolerance * spans, 1.0e-9)

        metrics: Dict[str, float] = {
            "Action/all_count": float(values.shape[0]),
            "Action/all_mean": float(np.mean(values)),
            "Action/all_std": float(np.std(values)),
            "Action/all_min": float(np.min(values)),
            "Action/all_max": float(np.max(values)),
            "Action/near_low_fraction": float(np.mean(values <= lows + saturation_tolerance)),
            "Action/near_high_fraction": float(np.mean(values >= highs - saturation_tolerance)),
            "Action/near_zero_fraction": float(np.mean(np.abs(values) <= self.action_idle_tolerance)),
        }

        for category in ("storage", "ev", "deferrable"):
            category_rows = [row for row in rows if row["category"] == category]
            if not category_rows:
                continue
            category_values = np.asarray([row["value"] for row in category_rows], dtype=np.float64)
            metrics[f"Action/{category}_count"] = float(category_values.shape[0])
            metrics[f"Action/{category}_mean"] = float(np.mean(category_values))
            metrics[f"Action/{category}_std"] = float(np.std(category_values))
            if category in {"storage", "ev"}:
                metrics[f"Action/{category}_positive_fraction"] = float(
                    np.mean(category_values > self.action_idle_tolerance)
                )
                metrics[f"Action/{category}_negative_fraction"] = float(
                    np.mean(category_values < -self.action_idle_tolerance)
                )
                metrics[f"Action/{category}_idle_fraction"] = float(
                    np.mean(np.abs(category_values) <= self.action_idle_tolerance)
                )
                if category == "ev":
                    connected_rows = [row for row in category_rows if row.get("ev_connected") is True]
                    disconnected_rows = [row for row in category_rows if row.get("ev_connected") is False]
                    for prefix, subset_rows in (
                        ("ev_connected", connected_rows),
                        ("ev_disconnected", disconnected_rows),
                    ):
                        if not subset_rows:
                            continue
                        subset_values = np.asarray([row["value"] for row in subset_rows], dtype=np.float64)
                        metrics[f"Action/{prefix}_count"] = float(subset_values.shape[0])
                        metrics[f"Action/{prefix}_mean"] = float(np.mean(subset_values))
                        metrics[f"Action/{prefix}_std"] = float(np.std(subset_values))
                        metrics[f"Action/{prefix}_positive_fraction"] = float(
                            np.mean(subset_values > self.action_idle_tolerance)
                        )
                        metrics[f"Action/{prefix}_negative_fraction"] = float(
                            np.mean(subset_values < -self.action_idle_tolerance)
                        )
                        metrics[f"Action/{prefix}_idle_fraction"] = float(
                            np.mean(np.abs(subset_values) <= self.action_idle_tolerance)
                        )
            elif category == "deferrable":
                thresholds = np.asarray([row["threshold"] for row in category_rows], dtype=np.float64)
                metrics["Action/deferrable_on_fraction"] = float(np.mean(category_values > thresholds))
                metrics["Action/deferrable_off_fraction"] = float(np.mean(category_values <= thresholds))

        metrics["Deferrable/pending_can_start_count"] = float(deferrable_pending_can_start_count)
        metrics["Deferrable/start_command_count"] = float(deferrable_start_command_count)
        metrics["Deferrable/start_when_available_count"] = float(deferrable_start_when_available_count)
        if deferrable_start_delays:
            delays = np.asarray(deferrable_start_delays, dtype=np.float64)
            metrics["Deferrable/start_delay_steps_mean"] = float(np.mean(delays))
            metrics["Deferrable/start_delay_steps_max"] = float(np.max(delays))

        if self.action_diagnostics_detail == "per_action":
            for row in rows:
                prefix = (
                    "Action/"
                    f"agent_{row['agent_index']}/"
                    f"{row['action_index']}_{self._metric_safe_name(row['action_name'])}"
                )
                span = max(float(row["span"]), 1.0e-9)
                tolerance = max(self.action_saturation_tolerance * span, 1.0e-9)
                metrics[f"{prefix}/value"] = float(row["value"])
                metrics[f"{prefix}/normalized"] = float((row["value"] - row["low"]) / span)
                metrics[f"{prefix}/near_low"] = float(row["value"] <= row["low"] + tolerance)
                metrics[f"{prefix}/near_high"] = float(row["value"] >= row["high"] - tolerance)

        return metrics

    def _deferrable_pending_can_start(
        self,
        *,
        agent_index: int,
        observation: np.ndarray,
        action_name: str,
    ) -> tuple[bool, bool]:
        asset_id = self._deferrable_asset_id(action_name)
        if not asset_id:
            return False, False

        lookup = self._build_observation_lookup(agent_index, observation)
        pending = self._find_deferrable_observation(lookup, asset_id, "pending")
        can_start = self._find_deferrable_observation(lookup, asset_id, "can_start")
        return pending > 0.5, can_start > 0.5

    @staticmethod
    def _deferrable_asset_id(action_name: str) -> Optional[str]:
        raw = str(action_name or "")
        if raw.startswith("deferrable_appliance_"):
            return raw[len("deferrable_appliance_") :]
        if "::" in raw:
            parts = raw.split("::")
            if len(parts) >= 2:
                asset = parts[-2].split("/")[-1]
                return asset or None
        return None

    @staticmethod
    def _find_deferrable_observation(
        observation_lookup: Dict[str, float],
        asset_id: str,
        signal: str,
    ) -> float:
        for name, value in observation_lookup.items():
            if "deferrable_appliance" not in name:
                continue
            if asset_id not in name:
                continue
            if name.endswith(f"::{signal}") or name.endswith(signal):
                return value
        return 0.0

    def _collect_model_status_metrics(self) -> Dict[str, float]:
        hook = getattr(self.model, "get_diagnostic_metrics", None)
        if callable(hook):
            return {str(key): self._safe_float(value) for key, value in hook().items()}
        return {}

    def _consume_model_training_metrics(self) -> Dict[str, float]:
        hook = getattr(self.model, "consume_latest_training_metrics", None)
        if callable(hook):
            return {str(key): self._safe_float(value) for key, value in hook().items()}
        return {}

    def _build_reward_component_metrics(self) -> Dict[str, float]:
        if not getattr(self, "reward_diagnostics_enabled", True):
            return {}

        reward_fn = getattr(self.env, "reward_function", None)
        if reward_fn is None and hasattr(self.env, "unwrapped"):
            reward_fn = getattr(self.env.unwrapped, "reward_function", None)
        if reward_fn is None:
            return {}

        components_hook = getattr(reward_fn, "get_last_components", None)
        if callable(components_hook):
            components = components_hook()
        else:
            components = {
                "per_agent": getattr(reward_fn, "last_components_by_agent", []),
                "community": getattr(reward_fn, "last_community_components", {}),
            }
        if not isinstance(components, Mapping):
            return {}

        metrics: Dict[str, float] = {}
        per_agent = components.get("per_agent", [])
        if isinstance(per_agent, list) and per_agent:
            numeric_keys = sorted(
                {
                    str(key)
                    for row in per_agent
                    if isinstance(row, Mapping)
                    for key, value in row.items()
                    if isinstance(value, (int, float))
                }
            )
            for key in numeric_keys:
                values = [
                    self._safe_float(row.get(key), default=0.0)
                    for row in per_agent
                    if isinstance(row, Mapping)
                ]
                if not values:
                    continue
                safe_key = self._metric_safe_name(key)
                array = np.asarray(values, dtype=np.float64)
                metrics[f"RewardComponent/{safe_key}_mean"] = float(np.mean(array))
                metrics[f"RewardComponent/{safe_key}_sum"] = float(np.sum(array))
                if self.reward_diagnostics_detail == "per_agent":
                    for agent_index, value in enumerate(array):
                        metrics[f"RewardComponent/agent_{agent_index}/{safe_key}"] = float(value)

        community = components.get("community", {})
        if isinstance(community, Mapping):
            for key, value in community.items():
                if not isinstance(value, (int, float)):
                    continue
                metrics[f"RewardComponent/community/{self._metric_safe_name(key)}"] = self._safe_float(value)

        return metrics

    def _build_observation_lookup(self, agent_index: int, observation: List[float]) -> Dict[str, float]:
        names: List[str] = []
        if agent_index < len(self.observation_names):
            names = [str(name) for name in self.observation_names[agent_index]]
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        return {name: self._safe_float(value) for name, value in zip(names, values)}

    @staticmethod
    def _extract_signal(observation_lookup: Dict[str, float], candidates: List[str], default: float = 0.0) -> float:
        for key in candidates:
            if key in observation_lookup:
                return observation_lookup[key]
        return default

    def _shape_rewards(self, rewards: List[float], observations: List[List[float]]) -> List[float]:
        if not self.wrapper_reward_enabled:
            return [self._safe_float(reward) for reward in rewards]

        profile = self.wrapper_reward_profile_config
        enabled_terms = profile.get("enabled_terms", {})
        weights = profile.get("weights", {})
        params = profile.get("params", {})

        observation_lookup = [
            self._build_observation_lookup(agent_index=i, observation=observation)
            for i, observation in enumerate(observations)
        ]

        community_net_consumption = sum(
            self._extract_signal(
                values,
                ["net_electricity_consumption", "net_electricity_consumption_without_storage"],
                default=0.0,
            )
            for values in observation_lookup
        )
        community_export_bonus_ratio = self._safe_float(params.get("community_export_bonus_ratio"), default=0.2)
        community_term = -abs(community_net_consumption) + (
            max(-community_net_consumption, 0.0) * community_export_bonus_ratio
        )

        shaped_rewards: List[float] = []
        export_credit_ratio = self._safe_float(params.get("export_credit_ratio"), default=0.8)
        ev_soc_tolerance = max(self._safe_float(params.get("ev_soc_tolerance"), default=0.1), 1e-6)

        for i, reward in enumerate(rewards):
            base_reward = self._safe_float(reward)
            obs_values = observation_lookup[i] if i < len(observation_lookup) else {}

            net_consumption = self._extract_signal(
                obs_values,
                ["net_electricity_consumption", "net_electricity_consumption_without_storage"],
                default=0.0,
            )
            electricity_price = max(
                self._extract_signal(
                    obs_values,
                    ["electricity_pricing", "electricity_price", "electricity_tariff"],
                    default=0.0,
                ),
                0.0,
            )
            import_cost = max(net_consumption, 0.0) * electricity_price
            export_credit = max(-net_consumption, 0.0) * electricity_price * export_credit_ratio
            energy_cost_term = -(import_cost - export_credit)

            grid_violation = self._extract_signal(
                obs_values,
                [
                    "electrical_service_violation",
                    "electrical_service_violation_kwh",
                    "service_violation",
                    "service_violation_kwh",
                ],
                default=0.0,
            )
            grid_violation_term = -max(grid_violation, 0.0)

            ev_success_signal = self._extract_signal(
                obs_values,
                [
                    "ev_departure_success",
                    "ev_departure_success_rate",
                    "ev_departure_status",
                    "departure_success",
                ],
                default=np.nan,
            )
            if not np.isnan(ev_success_signal):
                clipped_signal = float(np.clip(ev_success_signal, 0.0, 1.0))
                ev_term = 2.0 * clipped_signal - 1.0
            else:
                soc = self._extract_signal(obs_values, ["ev_soc", "soc"], default=np.nan)
                required_soc = self._extract_signal(
                    obs_values,
                    ["ev_required_soc_departure", "required_soc_departure", "required_soc"],
                    default=np.nan,
                )
                if np.isnan(soc) or np.isnan(required_soc):
                    ev_term = 0.0
                else:
                    distance = abs(soc - required_soc)
                    ev_term = float(np.clip(1.0 - (distance / ev_soc_tolerance), -1.0, 1.0))

            shaped_reward = base_reward
            if enabled_terms.get("energy_cost", True):
                shaped_reward += self._safe_float(weights.get("energy_cost"), default=1.0) * energy_cost_term
            if enabled_terms.get("grid_violation", True):
                shaped_reward += self._safe_float(weights.get("grid_violation"), default=1.0) * grid_violation_term
            if enabled_terms.get("ev_success", True):
                shaped_reward += self._safe_float(weights.get("ev_success"), default=1.0) * ev_term
            if enabled_terms.get("community", True):
                shaped_reward += self._safe_float(weights.get("community"), default=1.0) * community_term

            if self.wrapper_reward_clip_enabled:
                shaped_reward = float(
                    np.clip(shaped_reward, self.wrapper_reward_clip_min, self.wrapper_reward_clip_max)
                )

            if self.wrapper_reward_squash == "tanh":
                shaped_reward = float(np.tanh(shaped_reward))

            shaped_rewards.append(shaped_reward)

        return shaped_rewards

    def update(self, observations: List[List[float]], actions: List[List[float]], reward: List[float],
               next_observations: List[List[float]], terminated: bool, truncated: bool):
        """
        Delegates the update logic to the Algorithm, encoding observations before passing them.
        """

        if self.model is None:
            logger.error("Model is not set. Use `set_model` to provide a model.")
            raise ValueError("Model is not set. Use `set_model` to provide a model.")

        # Determine whether to update
        if not self.steps_between_training_updates or self.steps_between_training_updates <= 1:
            self.update_step = True
        else:
            self.update_step = self.global_step % self.steps_between_training_updates == 0
        logger.debug("Time step - Doing Update" if self.update_step else "Time step - Skipping Update")

        # Exploration phase ownership belongs to the algorithm.
        self.initial_exploration_done = bool(self.model.is_initial_exploration_done(self.global_step))
        logger.debug(
            "Initial exploration done: {} (global step={})",
            self.initial_exploration_done,
            self.global_step,
        )

        # Determine whether to update the target networks
        if not self.target_update_interval:
            self.update_target_step = False
        else:
            self.update_target_step = self.global_step % self.target_update_interval == 0
        logger.debug(
            "Time step - Doing Target Update" if self.update_target_step else "Time step - Skipping Target Update")

        phase_start_time = time.perf_counter()
        direct_entity_model_observations = bool(
            getattr(self, "_entity_model_observations_direct", False)
        )
        if direct_entity_model_observations:
            encoded_observations = [
                np.asarray(obs, dtype=np.float64) for obs in observations
            ]
            encoded_next_observations = [
                np.asarray(obs, dtype=np.float64) for obs in next_observations
            ]
            self._last_model_observation_encoding_seconds = 0.0
        else:
            encoded_observations = self._encode_observations_for_model(observations)
            encoded_next_observations = self._encode_observations_for_model(next_observations)
            self._last_model_observation_encoding_seconds = time.perf_counter() - phase_start_time

        transition_context_hook = getattr(self.model, "set_transition_context", None)
        if callable(transition_context_hook):
            transition_context_hook(
                raw_observations=None
                if direct_entity_model_observations
                else observations,
                raw_next_observations=None
                if direct_entity_model_observations
                else next_observations,
                encoded_observations=encoded_observations,
                encoded_next_observations=encoded_next_observations,
            )

        # Pass updated parameters to model.update()
        phase_start_time = time.perf_counter()
        update_result = self.model.update(observations = encoded_observations, actions= actions, rewards= reward,
                next_observations= encoded_next_observations, terminated = terminated,
                truncated = truncated,
                update_target_step=self.update_target_step, global_learning_step=self.global_step,
                update_step = self.update_step, initial_exploration_done= self.initial_exploration_done
        )
        self._last_model_update_seconds = time.perf_counter() - phase_start_time
        return update_result

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
                return np.nan_to_num(np.asarray(observations, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            if self._entity_adapter is None:
                return np.nan_to_num(np.asarray(observations, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

            return self._entity_adapter.normalize_observation(
                agent_index=index,
                observation=observations,
                observation_names=self.observation_names[index] if index < len(self.observation_names) else [],
                observation_space=self.observation_space[index],
            ).astype(np.float64)

        obs_array = np.array(observations, dtype=np.float64)  # Ensure numeric type

        # Apply encoding transformation correctly
        encoded = np.hstack([
            encoder.transform(obs) if hasattr(encoder, "transform") else encoder * obs
            for encoder, obs in zip(self.encoders[index], obs_array)
        ]).astype(np.float64)  # Convert everything to float

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
            "encoders": encoders_metadata,
            "action_bounds": action_bounds,
            "action_names": flat_action_names,
            "action_names_by_agent": action_names_by_agent,
            "building_names": building_names,
            "interface": getattr(self.env, "interface", "flat"),
            "topology_mode": getattr(self.env, "topology_mode", "static"),
            "seconds_per_time_step": getattr(self.env, "seconds_per_time_step", None),
            "entity_encoding": {
                "enabled": bool(self._entity_encoding_enabled),
                "normalization": self._entity_encoding_policy,
                "profile": self._entity_encoding_profile,
                "clip": bool(self._entity_encoding_clip),
                "serving_observation_names": "encoded" if serves_encoded_observations else "raw",
            },
            "entity_specs": getattr(self.env, "entity_specs", None) if self._entity_interface_mode else None,
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
