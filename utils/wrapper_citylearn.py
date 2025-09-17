import time
from typing import List

import mlflow
import numpy as np
import psutil
import torch
from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from loguru import logger

from algorithms.agents.base_agent import BaseAgent
from utils.checkpoint_manager import CheckpointManager
from utils.local_metrics import LocalMetricsLogger
from utils.preprocessing import *
from utils.progress_tracker import ProgressTracker

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
        super().__init__(env, **kwargs)
        self.model = model
        self.job_id = job_id
        self.initial_exploration_done = False
        self.update_step = False
        self.update_target_step = False
        self.global_step = 0
        training_cfg = config.get("training", {})
        checkpoint_cfg = config.get("checkpointing", {})
        tracking_cfg = config.get("tracking", {})

        self.steps_between_training_updates = training_cfg.get("steps_between_training_updates", 1)
        self.end_initial_exploration_time_step = training_cfg.get("end_initial_exploration_time_step", 0)
        self.end_exploration_time_step = training_cfg.get("end_exploration_time_step", 0)
        self.target_update_interval = training_cfg.get("target_update_interval", 0)
        self.log_dir = config.get("runtime", {}).get("log_dir")
        self.mlflow_enabled = tracking_cfg.get("mlflow_enabled", True)
        self.progress_tracker = ProgressTracker(progress_path)
        self.checkpoint_manager = CheckpointManager(
            base_dir=self.log_dir,
            interval=checkpoint_cfg.get("checkpoint_interval"),
            log_to_mlflow=tracking_cfg.get("mlflow_enabled", True),
        )
        self.local_metrics_logger = None
        if not self.mlflow_enabled:
            self.local_metrics_logger = LocalMetricsLogger(self.log_dir)

        # Ensure encoders are initialised for observation metadata and encoding
        if not hasattr(self, "encoders") or not getattr(self, "encoders"):
            self.encoders = self.set_encoders()

    def set_model(self, model: BaseAgent):
        """
        Set the model after initialization.
        """
        self.model = model


    def learn(self, episodes=None, deterministic=None, deterministic_finish=None, logging_level=None):
        """
        Train agent with MLflow logging for rewards (per step and per agent), PyTorch GPU memory usage, and system usage.
        """
        if self.model is None:
            logger.error("Wrapper invoked without a model; aborting training.")
            raise ValueError("Model is not set. Use `set_model` to provide a model.")

        episodes = episodes or 1
        deterministic_finish = deterministic_finish if deterministic_finish is not None else False
        deterministic = deterministic if deterministic is not None else False

        total_rewards_across_episodes = []  # To track overall reward trends

        for episode in range(episodes):
            start_episode_time = time.time()
            deterministic = deterministic or (deterministic_finish and episode >= episodes - 1)
            observations, _ = self.env.reset()
            self.episode_time_steps = self.episode_tracker.episode_time_steps
            terminated = False
            time_step = 0
            rewards_list = []  # Stores rewards per step

            while not terminated:
                step_start_time = time.time()
                self.global_step = episode * self.episode_time_steps + time_step  # Global step
                logger.debug("Global step %s (episode %s, timestep %s)", self.global_step, episode, time_step)

                actions = self.predict(observations, deterministic=deterministic)
                logger.debug("Predicted actions: %s", actions)

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
                    logger.debug("Model update executed at global step %s", self.global_step)

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
                if self.global_step % 10 == 0:
                    cpu_usage = psutil.cpu_percent()
                    ram_usage = psutil.virtual_memory().percent

                # PyTorch-specific GPU memory tracking (kept only PyTorch measurement)
                if torch.cuda.is_available():
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

                if self.global_step % 5 == 0:
                    self.progress_tracker.update(
                        episode=episode,
                        step=time_step,
                        global_step=self.global_step,
                        rewards=rewards,
                    )

                time_step += 1

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
                "Completed episode %s/%s, reward summary: %s, duration: %.2fs",
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

        # Determine exploration phase
        self.initial_exploration_done = self.time_step >= self.end_initial_exploration_time_step
        logger.debug("Initial Exploration ended." if self.initial_exploration_done else "Doing Initial Exploration - Skipping Update")

        # Determine whether to update the target networks
        if not self.target_update_interval:
            self.update_target_step = False
        else:
            self.update_target_step = self.time_step % self.target_update_interval == 0
        logger.debug(
            "Time step - Doing Target Update" if self.update_target_step else "Time step - Skipping Target Update")

        encoded_observations = self.get_all_encoded_observations(observations)
        encoded_next_observations = self.get_all_encoded_observations(next_observations)

        # Pass updated parameters to model.update()
        return self.model.update(observations = encoded_observations, actions= actions, rewards= reward,
                next_observations= encoded_next_observations, terminated = terminated,
                truncated = truncated,
                update_target_step=self.update_target_step, global_learning_step=self.global_step,
                update_step = self.update_step, initial_exploration_done= self.initial_exploration_done
        )

    def get_encoded_observations(self, index: int, observations: List[float]) -> np.ndarray:
        """Optimized encoding function using NumPy with proper type handling."""

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

        return {
            "observation_names": self.observation_names,
            "encoders": encoders_metadata,
            "action_bounds": action_bounds,
            "action_names": action_names,
            "reward_function": {
                "name": reward_fn.__class__.__name__ if reward_fn else None,
                "params": reward_config,
            },
        }


    def set_encoders(self) -> List[List[Encoder]]:
        r"""Get observation value transformers/encoders for use in MARLISA agent internal regression model.

        The encoder classes are defined in the `preprocessing.py` module and include `PeriodicNormalization` for cyclic observations,
        `OnehotEncoding` for categorical obeservations, `RemoveFeature` for non-applicable observations given available storage systems and devices
        and `Normalize` for observations with known minimum and maximum boundaries.

        Returns
        -------
        encoders : List[Encoder]
            Encoder classes for observations ordered with respect to `active_observations`.
        """

        encoders = []

        for o, s in zip(self.observation_names, self.observation_space):
            e = []

            remove_features = [
                'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h',
                'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h',
                'outdoor_relative_humidity', 'outdoor_relative_humidity_predicted_6h',
                'outdoor_relative_humidity_predicted_12h', 'outdoor_relative_humidity_predicted_24h',
                'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_6h',
                'diffuse_solar_irradiance_predicted_12h', 'diffuse_solar_irradiance_predicted_24h'
            ]

            for i, n in enumerate(o):
                if n in ['month', 'hour']:
                    e.append(PeriodicNormalization(s.high[i]))

                elif any(item in n for item in ["connected_state", "incoming_state"]):
                    e.append(OnehotEncoding([0,1]))

                elif any(item in n for item in ["required_soc_departure", "estimated_soc_arrival", "electric_vehicle_soc"]):
                    e.append(NormalizeWithMissing(s.low[i], s.high[i]))

                elif any(item in n for item in ["departure_time", "arrival_time"]):
                    e.append(OnehotEncoding([-0.1] + list(range(0, 25)))) #-0.1 encodes missing values

                elif n in ['day_type']:
                    e.append(OnehotEncoding([1, 2, 3, 4, 5, 6, 7, 8]))

                elif n in ["daylight_savings_status"]:
                    e.append(OnehotEncoding([0, 1]))

                elif n in remove_features:
                    e.append(RemoveFeature())

                else:
                    e.append(NoNormalization())

            encoders.append(e)

        logger.debug("Encoders SET")

        return encoders
