import json
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

        if self._entity_interface_mode:
            self._initialize_entity_agent_state(env=env)
        else:
            super().__init__(env, **kwargs)

        self.job_id = job_id
        self.initial_exploration_done = False
        self.update_step = False
        self.update_target_step = False
        self.global_step = 0
        training_cfg = config.get("training", {})
        checkpoint_cfg = config.get("checkpointing", {})
        tracking_cfg = config.get("tracking", {})
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
        self.progress_tracker = ProgressTracker(progress_path)

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
        )

        initial_observations, _ = self.env.reset()
        self._apply_entity_layout(initial_observations, force_attach=False)
        self.reset()

    def _apply_entity_layout(self, observation_payload: Mapping[str, Any], force_attach: bool) -> List[np.ndarray]:
        if not self._entity_interface_mode or self._entity_adapter is None:
            return []

        previous_version = self._entity_topology_version
        agent_observations, observation_names, observation_spaces = self._entity_adapter.to_agent_observations(observation_payload)
        self._entity_topology_version = self._entity_adapter.topology_version

        self.observation_names = observation_names
        self.observation_space = observation_spaces
        self.action_space = list(getattr(self.env, "flat_action_space", []))
        self.action_names = [list(names) for names in getattr(self.env, "action_names", [])]
        if len(self.action_names) < len(self.action_space):
            self.action_names.extend([[] for _ in range(len(self.action_space) - len(self.action_names))])
        elif len(self.action_names) > len(self.action_space):
            self.action_names = self.action_names[: len(self.action_space)]
        self.episode_time_steps = int(getattr(self.episode_tracker, "episode_time_steps", self.episode_time_steps))
        self.encoders = self.set_encoders()

        topology_changed = (
            force_attach
            or previous_version is None
            or self._entity_topology_version != previous_version
        )
        if topology_changed and self._entity_dynamic_mode and self._algorithm_name == "MADDPG" and previous_version is not None:
            raise ValueError(
                "MADDPG supports entity interface only with topology_mode='static'. "
                "Detected topology change during runtime."
            )

        if topology_changed and self.model is not None:
            self._attach_model_environment_metadata()

        return [np.asarray(obs, dtype=np.float64) for obs in agent_observations]

    def _attach_model_environment_metadata(self) -> None:
        if self.model is None:
            return

        metadata = {
            "seconds_per_time_step": getattr(self.env, "seconds_per_time_step", None),
            "building_names": getattr(self.env, "building_names", None),
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

    def _resolve_progress_totals(self, episodes: int) -> tuple[Optional[int], Optional[int]]:
        step_total = self._coerce_positive_int(self.episode_time_steps)
        if step_total is None:
            return None, None
        return step_total, episodes * step_total

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
            raw_observations, _ = self.env.reset()
            if self._entity_interface_mode:
                observations = self._apply_entity_layout(raw_observations, force_attach=True)
            else:
                observations = raw_observations
            self.episode_time_steps = self.episode_tracker.episode_time_steps
            episode_step_total, global_step_total = self._resolve_progress_totals(episodes)
            terminated = False
            truncated = False
            time_step = 0
            rewards_list = []  # Stores rewards per step

            while not (terminated or truncated):
                step_start_time = time.time()
                self.global_step += 1
                logger.debug(
                    "Global step {} (episode {}, timestep {})",
                    self.global_step,
                    episode,
                    time_step,
                )

                actions = self.predict(observations, deterministic=deterministic)
                actions = self._clip_actions(actions)
                if not self._entity_interface_mode:
                    self.actions = actions
                logger.debug("Predicted actions: {}", actions)

                # Apply actions to CityLearn environment
                env_actions = self._to_env_actions(actions)
                next_observations_raw, rewards, terminated, truncated, _ = self.env.step(env_actions)
                if self._entity_interface_mode:
                    next_observations = self._apply_entity_layout(next_observations_raw, force_attach=False)
                else:
                    next_observations = next_observations_raw
                rewards = self._shape_rewards(rewards, next_observations)
                rewards_list.append(rewards)

                # Update model if not in deterministic mode
                if not deterministic:
                    self.update(
                        observations,
                        actions,
                        rewards,
                        next_observations,
                        terminated=terminated,
                        truncated=truncated,
                    )
                    logger.debug("Model update executed at global step {}", self.global_step)

                    self.checkpoint_manager.maybe_save(
                        agent=self.model,
                        step=self.global_step,
                        initial_exploration_done=self.initial_exploration_done,
                        update_step=self.update_step,
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

                should_log_step = self._should_log_step(self.global_step)
                if should_log_step:
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
                    )

                time_step += 1

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

    def predict(self, observations, deterministic=None):
        """
        Updates the predict action logic. It now uses a mix of algorithm and the next time step.
        """
        if self.model is None:
            raise ValueError("Model is not set. Use `set_model` to provide a model.")

        if getattr(self.model, "use_raw_observations", False):
            encoded_observations = [np.asarray(obs, dtype=np.float64) for obs in observations]
        else:
            encoded_observations = self.get_all_encoded_observations(observations)

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

        clipped_actions: List[List[float]] = []
        for agent_idx, action_space in enumerate(self.action_space):
            raw = actions[agent_idx] if agent_idx < len(actions) else []
            action_array = np.asarray(raw, dtype=np.float64).reshape(-1)

            if hasattr(action_space, "low") and hasattr(action_space, "high"):
                low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
                high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
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

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if np.isnan(parsed) or np.isinf(parsed):
            return default
        return parsed

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
            self.update_step = self.time_step % self.steps_between_training_updates == 0
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
            self.update_target_step = self.time_step % self.target_update_interval == 0
        logger.debug(
            "Time step - Doing Target Update" if self.update_target_step else "Time step - Skipping Target Update")

        if getattr(self.model, "use_raw_observations", False):
            encoded_observations = [np.asarray(obs, dtype=np.float64) for obs in observations]
            encoded_next_observations = [np.asarray(obs, dtype=np.float64) for obs in next_observations]
        else:
            encoded_observations = self.get_all_encoded_observations(observations)
            encoded_next_observations = self.get_all_encoded_observations(next_observations)

        # Pass updated parameters to model.update()
        return self.model.update(observations = encoded_observations, actions= actions, rewards= reward,
                next_observations= encoded_next_observations, terminated = terminated,
                truncated = truncated,
                update_target_step=self.update_target_step, global_learning_step=self.global_step,
                update_step = self.update_step, initial_exploration_done= self.initial_exploration_done
        )

    def _should_log_step(self, step: int) -> bool:
        return step % self.step_metric_interval == 0

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

        raw_building_names = getattr(self.env, "building_names", None)
        building_names = None
        if isinstance(raw_building_names, list):
            building_names = [str(name) for name in raw_building_names]

        return {
            "observation_names": self.observation_names,
            "encoders": encoders_metadata,
            "action_bounds": action_bounds,
            "action_names": flat_action_names,
            "action_names_by_agent": action_names_by_agent,
            "building_names": building_names,
            "interface": getattr(self.env, "interface", "flat"),
            "topology_mode": getattr(self.env, "topology_mode", "static"),
            "entity_encoding": {
                "enabled": bool(self._entity_encoding_enabled),
                "normalization": self._entity_encoding_policy,
                "clip": bool(self._entity_encoding_clip),
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
