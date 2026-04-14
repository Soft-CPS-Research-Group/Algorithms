import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import psutil
import torch
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from loguru import logger

from algorithms.agents.base_agent import BaseAgent
from algorithms.utils.observation_enricher import ObservationEnricher
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
        # --- Initialize enrichment attributes BEFORE super().__init__() ---
        # These must exist before set_encoders() is called by parent.__init__
        algo_cfg = config.get("algorithm", {}) if config else {}
        self._tokenizer_config: Dict[str, Any] = algo_cfg.get("tokenizer", {})
        self._use_enrichment: bool = False  # Set in set_model() based on agent type
        self._enrichers: List[Optional[ObservationEnricher]] = []
        self._enriched_observation_names: List[List[str]] = []
        self._config = config
        
        super().__init__(env, **kwargs)
        self.model = model
        self.job_id = job_id
        self.initial_exploration_done = False
        self.update_step = False
        self.update_target_step = False
        self.global_step = 0
        training_cfg = config.get("training", {})
        simulator_cfg = config.get("simulator", {})
        checkpoint_cfg = config.get("checkpointing", {})
        tracking_cfg = config.get("tracking", {})

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

        # Detect if agent is Transformer-based (requires enrichment)
        # Check for AgentTransformerPPO or similar transformer agents
        agent_class_name = model.__class__.__name__
        self._use_enrichment = "Transformer" in agent_class_name

        if self._use_enrichment:
            logger.info("Enrichment enabled for Transformer-based agent: {}", agent_class_name)
            # Rebuild encoders with enrichment
            self.encoders = self.set_encoders()

        metadata = {
            "seconds_per_time_step": getattr(self.env, "seconds_per_time_step", None),
            "building_names": getattr(self.env, "building_names", None),
        }
        try:
            # Pass enriched names if enrichment is enabled
            obs_names = (
                self._enriched_observation_names
                if self._use_enrichment and self._enriched_observation_names
                else self.observation_names
            )
            self.model.attach_environment(
                observation_names=obs_names,
                action_names=self.action_names,
                action_space=self.action_space,
                observation_space=self.observation_space,
                metadata=metadata,
            )
        except AttributeError:
            # Older agents may not implement attach_environment.
            pass


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
            observations, _ = self.env.reset()
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
                logger.debug("Predicted actions: {}", actions)

                # Apply actions to CityLearn environment
                next_observations, rewards, terminated, truncated, _ = self.env.step(actions)
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
                if self.system_metrics_enabled and torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # Convert to MB
                    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # Reserved for caching
                else:
                    gpu_mem_allocated = None
                    gpu_mem_reserved = None

                # Step duration calculation
                step_duration = time.time() - step_start_time

                # Consolidated MLflow logging
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

                if self._should_log_step(self.global_step):
                    if mlflow.active_run():
                        mlflow.log_metrics(metrics, step=self.global_step)
                    elif self.local_metrics_logger:
                        self.local_metrics_logger.log(metrics, self.global_step)

                logger.info(
                    f'Time step: {time_step + 1}/{self.episode_time_steps},'
                    f' Episode: {episode + 1}/{episodes},'
                    f' Actions: {actions},'
                    f' Rewards: {rewards},'
                    f' CPU: {cpu_usage}%, RAM: {ram_usage}%,'
                    f' GPU Allocated: {gpu_mem_allocated} MB, GPU Reserved: {gpu_mem_reserved} MB'
                    f' Step Duration: {step_duration}'
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
            rewards_array = np.array(rewards_list, dtype='float')  # (time_steps, num_agents)
            rewards_summary = {
                'sum': rewards_array.sum(axis=0),  # Sum per agent
                'mean': rewards_array.mean(axis=0),
                'min': rewards_array.min(axis=0),
                'max': rewards_array.max(axis=0)
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
        total_rewards_across_episodes = np.array(total_rewards_across_episodes)  # Shape: (episodes, num_agents)

        # Compute overall statistics across episodes
        overall_rewards_summary = {
            'sum': total_rewards_across_episodes.sum(axis=0),
            'mean': total_rewards_across_episodes.mean(axis=0),
            'min': total_rewards_across_episodes.min(axis=0),
            'max': total_rewards_across_episodes.max(axis=0)
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
        self.actions = actions
        self.next_time_step()
        return actions

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

        # Apply enrichment if enabled for this building
        if self._use_enrichment and index < len(self._enrichers) and self._enrichers[index] is not None:
            enriched_obs = self._enrichers[index].enrich_values(observations)
            obs_array = np.array(enriched_obs, dtype=np.float64)
        else:
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
            "reward_function": {
                "name": reward_fn.__class__.__name__ if reward_fn else None,
                "params": reward_config,
            },
        }


    def set_encoders(self) -> List[List[Encoder]]:
        r"""Instantiate observation encoders from the shared JSON configuration.

        When enrichment is enabled (for Transformer-based agents), this method:
        1. Creates per-building ObservationEnrichers
        2. Enriches observation names with __tkn_*__ markers
        3. Builds encoders for enriched names (markers get NoNormalization)
        """

        rules = _load_encoder_rules()
        encoders: List[List[Encoder]] = []
        missing: List[str] = []

        # Reset enrichment state
        self._enrichers = []
        self._enriched_observation_names = []

        # Get action names by building (handle different formats)
        action_names_by_building = self._get_action_names_by_building()

        for building_idx, (observation_group, space) in enumerate(
            zip(self.observation_names, self.observation_space)
        ):
            # Get action names for this building
            action_names = (
                action_names_by_building[building_idx]
                if building_idx < len(action_names_by_building)
                else []
            )

            # --- Enrichment step (if enabled) ---
            if self._use_enrichment and self._tokenizer_config:
                enricher = ObservationEnricher(self._tokenizer_config)
                enrichment = enricher.enrich_names(observation_group, action_names)
                enriched_names = enrichment.enriched_names
                self._enrichers.append(enricher)
                self._enriched_observation_names.append(enriched_names)
            else:
                self._enrichers.append(None)
                enriched_names = list(observation_group)
                self._enriched_observation_names.append(enriched_names)

            # Build encoders from (possibly enriched) names
            group_encoders: List[Encoder] = []
            for index, name in enumerate(enriched_names):
                # Marker features use NoNormalization (pass-through)
                if name.startswith("__tkn_") and name.endswith("__"):
                    encoder = NoNormalization()
                    group_encoders.append(encoder)
                    continue

                rule = next((r for r in rules if _matches_rule(name, r.get("match", {}))), None)
                if rule is None:
                    missing.append(name)
                    continue

                if rule.get("warn_on_use"):
                    logger.warning("Encoder rule warning for observation '{}'", name)

                encoder = _build_encoder(rule, space, index)
                group_encoders.append(encoder)

            if len(group_encoders) != len(enriched_names):
                raise ValueError(
                    "Failed to build encoders for all observations: "
                    f"expected {len(enriched_names)}, built {len(group_encoders)}"
                )
            encoders.append(group_encoders)

        if missing:
            raise ValueError(
                "No encoder rule defined for observations: " + ", ".join(sorted(set(missing)))
            )

        logger.debug("Encoders initialised from external configuration")
        return encoders

    def _get_action_names_by_building(self) -> List[List[str]]:
        """Get action names grouped by building.

        Handles different CityLearn action_names formats:
        - List[List[str]]: Already per-building
        - List[str]: Same actions for all buildings
        """
        action_names = getattr(self, "action_names", None)
        if action_names is None:
            return [[] for _ in self.observation_names]

        if isinstance(action_names, list):
            if action_names and isinstance(action_names[0], list):
                # Already per-building
                return action_names
            else:
                # Same actions for all buildings
                return [list(action_names) for _ in self.observation_names]

        return [[] for _ in self.observation_names]
