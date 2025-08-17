from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv
from algorithms.agents.base_agent import BaseAgent
from loguru import logger
import mlflow
import numpy.typing as npt
from utils.preprocessing import *
import time
import psutil
import numpy as np
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
import torch

class Wrapper_CityLearn(RLC):
    def __init__(self, env: CityLearnEnv, model: BaseAgent = None, config=None, job_id=None, progress_path=None,
                 **kwargs):
        """Wrapper for CityLearn RLC that delegates custom behavior to a BaseAgent model."""

        super().__init__(env, **kwargs)
        self.model = model
        self.job_id = job_id
        self.progress_path = progress_path
        self.config = config or {}

        self.initial_exploration_done = False
        self.update_step = False
        self.update_target_step = False
        self.global_step = 0

        hp = config["algorithm"]["hyperparameters"]
        self.steps_between_training_updates = hp["steps_between_training_updates"]
        self.end_initial_exploration_time_step = hp["end_initial_exploration_time_step"]
        self.end_exploration_time_step = hp["end_exploration_time_step"]
        self.target_update_interval = hp["target_update_interval"]

        exp_cfg = config["experiment"]
        self.save_checkpoints = exp_cfg.get("save_checkpoints", False)
        self.checkpoint_interval = exp_cfg.get("checkpoint_interval_steps", 1000)
        self.save_final_model = exp_cfg.get("save_final_model", True)
        self.log_dir = exp_cfg["logging"]["log_dir"]
        self.progress_write_interval = exp_cfg.get("progress_write_interval", 5)

    def set_model(self, model: BaseAgent):
        """
        Set the model after initialization.
        """
        self.model = model

    def _write_progress(self, episode, step, rewards=None):
        """Write current progress to a JSON file."""
        try:
            import json, datetime
            progress = {
                "episode": episode,
                "step": step,
                "total_steps": self.episode_time_steps,
                "global_step": self.global_step,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            if rewards:
                progress["rewards"] = rewards
            with open(self.progress_path, "w") as f:
                json.dump(progress, f, indent=2)
            logger.debug(
                "Progress written: episode %s step %s/%s",
                episode,
                step,
                self.episode_time_steps,
            )
        except Exception as e:
            logger.warning(f"Could not write progress file: {e}")

    def learn(self, episodes=None, deterministic=None, deterministic_finish=None, logging_level=None):
        """
        Train agent with MLflow logging for rewards (per step and per agent), PyTorch GPU memory usage, and system usage.
        """
        if self.model is None:
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

                actions = self.predict(observations, deterministic=deterministic)

                # Apply actions to CityLearn environment
                next_observations, rewards, terminated, truncated, _ = self.env.step(actions)
                rewards_list.append(rewards)

                # Update model if not in deterministic mode
                if not deterministic:
                    self.update(observations, actions, rewards, next_observations, terminated=terminated,
                                truncated=truncated)

                observations = [o for o in next_observations]

                # Reduce system monitoring frequency
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

                mlflow.log_metrics(metrics, step=self.global_step)

                if self.save_checkpoints and self.global_step > 0 and self.global_step % self.checkpoint_interval == 0:
                    self.model.save_checkpoint(self.global_step)

                logger.info(
                    f'Time step: {time_step + 1}/{self.episode_time_steps},'
                    f' Episode: {episode + 1}/{episodes},'
                    f' Actions: {actions},'
                    f' Rewards: {rewards},'
                    f' CPU: {cpu_usage}%, RAM: {ram_usage}%,'
                    f' GPU Allocated: {gpu_mem_allocated} MB, GPU Reserved: {gpu_mem_reserved} MB'
                    f' Step Duration: {step_duration}'
                )

                if self.global_step % self.progress_write_interval == 0:
                    self._write_progress(
                        episode=episode, step=time_step, rewards=rewards
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
            mlflow.log_metrics(episode_metrics, step=episode)

            logger.info(
                f'Completed episode: {episode + 1}/{episodes}, Reward: {rewards_summary}, Duration: {episode_duration:.2f}s')

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

        mlflow.log_metrics(overall_metrics)

        # Save model at the end of training if requested
        if self.save_final_model:
            self.model.export_to_onnx(self.log_dir)

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
        self.update_step = self.time_step % self.steps_between_training_updates != 0
        logger.debug("Time step - Doing Update" if self.update_step else "Time step - Skipping Update")

        # Determine exploration phase
        self.initial_exploration_done = self.time_step >= self.end_initial_exploration_time_step
        logger.debug("Initial Exploration ended." if self.initial_exploration_done else "Doing Initial Exploration - Skipping Update")

        # Determine whether to update the target networks
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
        """Optimized encoding function avoiding joblib issues."""
        return [self.get_encoded_observations(idx, obs) for idx, obs in enumerate(observations)]

    def get_all_encoded_observations(self, observations: List[List[float]]) -> List[np.ndarray]:
        """Optimized version without joblib for better performance."""
        return [self.get_encoded_observations(idx, obs) for idx, obs in enumerate(observations)]

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