from loguru import logger
from torch.amp import GradScaler, autocast
from torch.nn.functional import mse_loss
from algorithms.utils.networks import Actor, Critic
from algorithms.utils.replay_buffer import *
from typing import List
import mlflow.pytorch
import time
import torch
import mlflow
from torch.nn.utils import clip_grad_norm_
import numpy as np
import onnx
import os
from algorithms.agents.base_agent import BaseAgent
import random


class MADDPG(BaseAgent):
    def __init__(self, config):
        super().__init__()
        """1. Initialize MADDPG with structured logging and streamlined config handling."""
        # 1.1. Store configuration
        self.config = config

        # 1.2. Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device selected: {self.device}")
        torch.backends.cudnn.benchmark = True

        # 1.3. Hyperparameters
        self.gamma = self.config["algorithm"]["exploration"]["params"]["gamma"]
        self.tau = self.config["algorithm"]["exploration"]["params"]["tau"]
        self.sigma = self.config["algorithm"]["exploration"]["params"]["sigma"]
        self.bias = self.config["algorithm"]["exploration"]["params"]["bias"]
        self.batch_size = self.config["algorithm"]["replay_buffer"]["params"]["batch_size"]
        self.lr_actor = float(self.config["algorithm"]["networks"]["actor_network"]["params"]["lr"])
        self.lr_critic = float(self.config["algorithm"]["networks"]["critic_network"]["params"]["lr"])
        self.seed = self.config["algorithm"].get("seed", 22)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.resume_training = self.config["experiment"]["resume_training"]
        self.checkpoint_run_id = self.config["experiment"]["checkpoint_run_id"]
        self.checkpoint_artifact = self.config["experiment"]["checkpoint_artifact"]
        self.use_best_checkpoint_artifact = self.config["experiment"]["use_best_checkpoint_artifact"]
        self.load_optimizer_state = self.config["experiment"].get("load_optimizer_state", True)

        # 1.4. Retrieve number of agents and their dimensions
        self.num_agents = self.config["algorithm"]["hyperparameters"]["num_agents"]
        self.observation_dimension = self.config["algorithm"]["hyperparameters"]["observation_dimensions"]
        self.action_dimension = self.config["algorithm"]["hyperparameters"]["action_dimensions"]

        # 1.5. Replay buffer
        self.replay_buffer = self._initialize_replay_buffer()

        # 1.6. Neural networks and optimizers
        self.actors, self.critics, self.actor_targets, self.critic_targets = self._initialize_networks()
        self.actor_optimizers, self.critic_optimizers = self._initialize_optimizers()
        self.scaler = GradScaler(self.device)

        # 1.7. Resume Training/Continous Training/Transfer Learning
        if self.resume_training:
            if self.use_best_checkpoint_artifact:
                self.checkpoint_run_id = self.get_best_checkpoint(self.config["experiment"]["name"])

            if self.checkpoint_run_id:
                self.load_checkpoint()

            # ✅ Freeze layers if needed for fine-tuning
            freeze_actor = self.config["experiment"].get("freeze_actors", False)
            freeze_critic = self.config["experiment"].get("freeze_critics", False)
            if self.config["experiment"].get("freeze_pretrained_layers", False):
                freeze_actor = True
            if freeze_actor or freeze_critic:
                self.freeze_layers(freeze_actor=freeze_actor, freeze_critic=freeze_critic)

        logger.info("MADDPG initialization complete.")


    def _initialize_replay_buffer(self):
        """Initialize the replay buffer dynamically based on configuration."""
        logger.debug("Initializing replay buffer.")
        try:
            replay_buffer_class = self.config["algorithm"]["replay_buffer"]["class"]
            replay_buffer_params = self.config["algorithm"]["replay_buffer"]["params"]
            replay_buffer_params["num_agents"] = self.num_agents

            # Convert string numbers to integers if necessary
            for key, value in replay_buffer_params.items():
                if isinstance(value, str) and value.isdigit():
                    replay_buffer_params[key] = int(value)

            ReplayBuffer = globals()[replay_buffer_class]
            return ReplayBuffer(**replay_buffer_params)

        except KeyError as e:
            logger.error(f"Missing key in configuration for replay buffer: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing replay buffer: {e}")
            raise

    def _initialize_networks(self):
        """1.7.1 Initialize actor and critic networks."""
        logger.debug("Initializing actor and critic networks.")
        actor_fc_units = self.config["algorithm"]["networks"]["actor_network"]["params"]["layers"]
        critic_fc_units = self.config["algorithm"]["networks"]["critic_network"]["params"]["layers"]

        actors, critics, actor_targets, critic_targets = [], [], [], []
        for i in range(self.num_agents):
            # Use agent-specific observation and action dimensions
            state_size = self.observation_dimension[i]
            action_size = self.action_dimension[i]

            # Initialize networks
            actors.append(Actor(state_size, action_size, self.seed, actor_fc_units).to(self.device))
            critics.append(Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, critic_fc_units).to(self.device))
            actor_targets.append(Actor(state_size, action_size, self.seed, actor_fc_units).to(self.device))
            critic_targets.append(Critic(sum(self.observation_dimension), sum(self.action_dimension), self.seed, critic_fc_units).to(self.device))

        # Target Network Initialization: the target networks are initially synchronized with the main networks
        # (i.e. copying weights at initialization).
        for actor, actor_target in zip(actors, actor_targets):
            actor_target.load_state_dict(actor.state_dict())
        for critic, critic_target in zip(critics, critic_targets):
            critic_target.load_state_dict(critic.state_dict())

        return actors, critics, actor_targets, critic_targets

    def _initialize_optimizers(self):
        """1.7.2 Initialize optimizers for actors and critics."""
        logger.debug("Initializing optimizers.")
        actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.lr_actor)
            for actor in self.actors
        ]
        critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.lr_critic)
            for critic in self.critics
        ]
        return actor_optimizers, critic_optimizers

    def update(self, observations, actions, rewards, next_observations, terminated, truncated,
               update_target_step, global_learning_step, update_step, initial_exploration_done):
        """
        Perform a MADDPG update step with optimized batch processing and full MLflow logging.

        Steps:
        1. Collect and store experiences in the replay buffer.
        2. Sample a batch from the replay buffer.
        3. Optimize all critics in a vectorized manner.
        4. Optimize each actor individually.
        5. Perform a soft update on target networks.
        6. Log metrics for training progress.
        """
        logger.debug("Starting update phase.")
        update_start_time = time.time()

        # Push experiences to replay buffer
        self.replay_buffer.push(observations, actions, rewards, next_observations, terminated)

        # Early exit conditions
        if len(self.replay_buffer) < self.batch_size:
            logger.warning("Not enough samples in the replay buffer. Skipping update.")
            return

        if not initial_exploration_done:
            logger.warning("Initial exploration phase not finished. Skipping update.")
            return

        if not update_step:
            logger.warning("Not updating due to time step constraint. Skipping update.")
            return

        # Sample batch from replay buffer
        # Sample batch from replay buffer (still in CPU)
        states, actions_all, rewards_all, next_states, dones_all = self.replay_buffer.sample()

        # ✅ Fix: Convert rewards_all to a single tensor before using in calculations
        rewards_all = torch.stack(rewards_all).to(self.device, dtype=torch.float32, non_blocking=True)

        # ✅ dones_all is already a tensor from sample()
        dones_all = dones_all.to(self.device, dtype=torch.float32, non_blocking=True)

        # ✅ Move batch to GPU efficiently using non_blocking=True
        states = [s.to(self.device, non_blocking=True) for s in states]
        actions_all = [a.to(self.device, non_blocking=True) for a in actions_all]
        next_states = [ns.to(self.device, non_blocking=True) for ns in next_states]

        # Compute global state representations
        global_state = torch.cat(states, dim=1)
        global_next_state = torch.cat(next_states, dim=1)
        global_actions = torch.cat(actions_all, dim=1)

        # Compute target next actions using target actor networks
        with torch.no_grad():
            global_next_actions = torch.cat(
                [self.actor_targets[i](next_states[i]) for i in range(self.num_agents)], dim=1
            )
            q_targets_next = torch.stack(
                [critic(global_next_state, global_next_actions) for critic in self.critic_targets])

            # Compute target Q-values with correct `dones_all`
            q_targets = rewards_all + self.gamma * q_targets_next * (1 - dones_all)

        # Optimize Critics
        q_expected = torch.stack([critic(global_state, global_actions) for critic in self.critics])
        with autocast(device_type="cuda"):
            critic_loss = mse_loss(q_expected, q_targets.expand_as(q_expected)).mean()

        self.critic_optimizers[0].zero_grad(set_to_none=True)
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizers[0])
        clip_grad_norm_([param for critic in self.critics for param in critic.parameters()], max_norm=1.0)
        self.scaler.step(self.critic_optimizers[0])
        self.scaler.update()

        logger.debug(f"Critics updated. Loss: {critic_loss.item():.4f}.")

        # **🔹 Log Individual Critic Losses to MLflow**
        for agent_num in range(self.num_agents):
            critic_loss_value = (q_expected[:, agent_num] - q_targets[:, agent_num].expand_as(
                q_expected[:, agent_num])).abs().mean()
            mlflow.log_metric(f"critic_loss_agent_{agent_num}", critic_loss_value.item(), step=global_learning_step)

        # **🔹 Optimize Each Actor Individually**
        total_actor_loss = 0.0

        for agent_num, (actor, critic, actor_optimizer) in enumerate(
                zip(self.actors, self.critics, self.actor_optimizers)
        ):
            logger.debug(f"Updating actor for agent {agent_num}.")

            obs = states[agent_num]

            # Compute predicted actions for the agent
            predicted_action = actor(obs)

            # Replace only the current agent's action in the global set
            global_predicted_actions = torch.cat([
                predicted_action if i == agent_num else actions_all[i].detach()
                for i in range(self.num_agents)
            ], dim=1)

            # Compute actor loss
            with autocast(device_type="cuda"):
                actor_loss = -critic(global_state, global_predicted_actions).mean()

            actor_optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(actor_optimizer)
            clip_grad_norm_(actor.parameters(), max_norm=1.0)  # Gradient clipping
            self.scaler.step(actor_optimizer)
            self.scaler.update()

            total_actor_loss += actor_loss.item()

            logger.debug(f"Actor {agent_num} updated. Loss: {actor_loss.item():.4f}.")

            # **🔹 Log Individual Actor Loss to MLflow**
            mlflow.log_metric(f"actor_loss_agent_{agent_num}", actor_loss.item(), step=global_learning_step)

            # **🔹 Soft Update Target Networks**
            if update_target_step:
                logger.debug(f"Updating target networks for agent {agent_num}.")
                self._soft_update(critic, self.critic_targets[agent_num], self.tau)
                self._soft_update(actor, self.actor_targets[agent_num], self.tau)

        # **🔹 Log Global Metrics**
        mlflow.log_metrics({
            "average_critic_loss": critic_loss.item(),
            "average_actor_loss": total_actor_loss / self.num_agents,
            "training_step_time": time.time() - update_start_time
        }, step=global_learning_step)

        logger.info(f"Update complete. Avg Critic Loss: {critic_loss.item():.4f}, "
                    f"Avg Actor Loss: {total_actor_loss / self.num_agents:.4f}.")

    def _soft_update(self, local_model, target_model, tau):
        """2.8.1 Soft update model parameters."""
        #for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        #    target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        with torch.no_grad():
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.lerp_(local_param.data, tau)

    def predict(self, observations, deterministic=False):
        """3. Predict actions based on observations.

        Parameters
        ----------
        observations : List[torch.Tensor]
            Observations for each agent.
        deterministic : bool, optional
            Whether to use deterministic predictions, by default False.

        Returns
        -------
        List[float]
            Predicted actions for each agent.
        """
        logger.debug("Predicting actions.")
        if deterministic:
            logger.debug("Using deterministic predictions.")
            return self._predict_deterministic(observations)
        else:
            logger.debug("Using exploration for predictions.")
            return self._predict_with_exploration(observations)

    def _predict_deterministic(self, observations):
        """3.1 Predict deterministic actions for given observations.

        Parameters
        ----------
        observations : List[npt.NDArray[np.float64]]
            Observations for each agent.

        Returns
        -------
        List[float]
            Deterministic actions for each agent.
        """
        logger.debug("Predicting deterministic actions.")
        actions = []
        with torch.no_grad():
            for actor, obs in zip(self.actors, observations):
                action = actor(torch.FloatTensor(obs).to(self.device)).cpu().numpy()
                actions.append(action)
        logger.debug(f"Deterministic actions predicted: {actions}")
        return actions

    def _predict_with_exploration(self, observations):
        """Predict actions with added exploration noise, including constraints and clipping.

        Parameters
        ----------
        observations : List[List[float]]
            Observations for each agent.

        Returns
        -------
        List[float]
            Actions with exploration noise for each agent.
        """
        logger.debug("Predicting actions with exploration noise.")

        # Predict deterministic actions first
        deterministic_actions = self._predict_deterministic(observations)

        # Apply noise using random generation, subtracting bias
        random_noises = []
        for action in deterministic_actions:
            noise = np.random.normal(scale=self.sigma) - self.bias
            random_noises.append(noise)

        # Add noise to deterministic actions
        actions = [noise + action for action, noise in zip(deterministic_actions, random_noises)]

        # Clip actions to stay within valid range (-1, 1)
        clipped_actions = [np.clip(action, -1, 1) for action in actions]

        # Convert to list format
        actions_return = [action.tolist() for action in clipped_actions]

        logger.debug(f"Actions with exploration noise and constraints applied: {actions_return}")
        return actions_return

    def freeze_layers(self, freeze_actor=True, freeze_critic=False):
        """Freeze parts of the network for transfer learning."""
        for i in range(self.num_agents):
            if freeze_actor:
                for param in self.actors[i].parameters():
                    param.requires_grad = False
            if freeze_critic:
                for param in self.critics[i].parameters():
                    param.requires_grad = False
        logger.info(f"Freezing actors: {freeze_actor}, Freezing critics: {freeze_critic}")

    def save_checkpoint(self, step: int) -> None:
        """Save a checkpoint to MLflow."""
        logger.info(f"Saving checkpoint at step {step}.")

        checkpoint = {}
        for i in range(self.num_agents):
            checkpoint[f"actor_state_dict_{i}"] = self.actors[i].state_dict()
            checkpoint[f"critic_state_dict_{i}"] = self.critics[i].state_dict()
            checkpoint[f"actor_optimizer_state_dict_{i}"] = self.actor_optimizers[i].state_dict()
            checkpoint[f"critic_optimizer_state_dict_{i}"] = self.critic_optimizers[i].state_dict()

        checkpoint["replay_buffer"] = self.replay_buffer.get_state()
        checkpoint_path = os.path.join("/tmp", self.checkpoint_artifact)
        torch.save(checkpoint, checkpoint_path)

        # Log to MLflow
        mlflow.log_artifact(checkpoint_path)
        logger.info(f"Checkpoint saved to MLflow: {self.checkpoint_artifact}")
    def load_checkpoint(self) -> None:
        """Load model parameters and optionally optimizer states from MLflow."""
        logger.info(f"Loading checkpoint from MLflow Run ID: {self.checkpoint_run_id}")

        try:
            checkpoint_path = os.path.join("/tmp", self.checkpoint_artifact)
            mlflow.artifacts.download_artifact(
                artifact_path=self.checkpoint_artifact,
                dst_path=checkpoint_path,
                run_id=self.checkpoint_run_id
            )

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            for i in range(self.num_agents):
                self.actors[i].load_state_dict(checkpoint[f"actor_state_dict_{i}"])
                self.critics[i].load_state_dict(checkpoint[f"critic_state_dict_{i}"])

                # ✅ Skip loading optimizer states if fine-tuning
                if self.load_optimizer_state:
                    self.actor_optimizers[i].load_state_dict(checkpoint[f"actor_optimizer_state_dict_{i}"])
                    self.critic_optimizers[i].load_state_dict(checkpoint[f"critic_optimizer_state_dict_{i}"])

            if "replay_buffer" in checkpoint and not self.config["experiment"].get("reset_replay_buffer", False):
                self.replay_buffer.set_state(checkpoint["replay_buffer"])

            logger.info("Checkpoint successfully loaded from MLflow.")

        except Exception as e:
            logger.error(f"Failed to load checkpoint from MLflow: {e}")
            raise RuntimeError("Error loading checkpoint from MLflow")

    @staticmethod
    def get_best_checkpoint(experiment_name):
        """Retrieve the best checkpoint (lowest validation loss) from MLflow."""
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found in MLflow.")

        runs = client.search_runs(experiment.experiment_id, order_by=["metrics.validation_loss ASC"], max_results=1)

        if runs:
            return runs[0].info.run_id
        else:
            raise ValueError(f"No runs found for experiment {experiment_name}.")

    def export_to_onnx(self, log_dir):
        """Export each agent's actor network to ONNX format inside the log directory."""
        export_dir = os.path.join(log_dir, "onnx_models")
        logger.info(f"Exporting MADDPG actors to ONNX. Saving to {export_dir}")

        os.makedirs(export_dir, exist_ok=True)  # Ensure directory exists

        for i, actor in enumerate(self.actors):
            export_path = os.path.join(export_dir, f"agent_{i}.onnx")
            dummy_input = torch.randn(1, self.observation_dimension[i]).to(self.device)  # Adjust per agent

            torch.onnx.export(
                actor,  # Export actor network for agent i
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=[f"observation_agent_{i}"],
                output_names=[f"action_agent_{i}"],
                dynamic_axes={f"observation_agent_{i}": {0: "batch_size"}, f"action_agent_{i}": {0: "batch_size"}}
            )

            mlflow.log_artifact(export_path)
            logger.info(f"ONNX model exported for Agent {i}: {export_path}")

