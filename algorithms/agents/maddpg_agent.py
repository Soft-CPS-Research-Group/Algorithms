import torch
from torch.nn.functional import mse_loss
from loguru import logger
from torch.amp import GradScaler, autocast
import random
from algorithms.utils.networks import Actor, Critic
from algorithms.utils.replay_buffer import *
from typing import List
import mlflow.pytorch
import os
import time
import torch
import mlflow
from contextlib import nullcontext
from torch.nn.utils import clip_grad_norm_
import numpy as np

class MADDPG:
    def __init__(self, config):
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

        logger.info("MADDPG initialization complete.")

    def _initialize_replay_buffer(self):
        """1.6.1 Initialize the replay buffer dynamically based on configuration."""
        logger.debug("Initializing replay buffer dynamically.")

        try:
            replay_buffer_class = self.config["algorithm"]["replay_buffer"]["class"]
            replay_buffer_params = self.config["algorithm"]["replay_buffer"]["params"]
            logger.debug(f"Replay buffer class: {replay_buffer_class}, params: {replay_buffer_params}")
            replay_buffer_params["device"] = self.device
            replay_buffer_params["num_agents"] = self.num_agents

            device_str = str(self.device)
            if "cuda" in device_str:
                # If there is extra info (like ":0"), you can split it off
                replay_buffer_params["device"] = "cuda"
            elif "cpu" in device_str:
                replay_buffer_params["device"] = "cpu"
            else:
                replay_buffer_params["device"] = device_str  # fallback

            replay_buffer_params["num_agents"] = self.num_agents

            # Clean up parameters: convert numeric strings to integers if necessary
            for key, value in replay_buffer_params.items():
                if isinstance(value, str):
                    # Remove commas and extra whitespace
                    cleaned_value = value.replace(',', '').strip()
                    # If the cleaned string is a digit, convert it to an integer
                    if cleaned_value.isdigit():
                        replay_buffer_params[key] = int(cleaned_value)
                    else:
                        # For non-numeric strings (like device names), just use the cleaned string
                        replay_buffer_params[key] = cleaned_value

            ReplayBuffer = globals()[replay_buffer_class]  # Dynamically fetch the class
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

    def save_models(self, run_id):
        """Save all actor and critic models to MLflow"""
        model_dir = f"./models/{run_id}"
        os.makedirs(model_dir, exist_ok=True)

        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            actor_path = os.path.join(model_dir, f"actor_{i}.pt")
            critic_path = os.path.join(model_dir, f"critic_{i}.pt")

            torch.save(actor.state_dict(), actor_path)
            torch.save(critic.state_dict(), critic_path)

            mlflow.log_artifact(actor_path, artifact_path="models")
            mlflow.log_artifact(critic_path, artifact_path="models")

        logger.info("Models saved to MLflow.")

    def save_checkpoint(self, run_id, checkpoint_step):
        """Save MADDPG checkpoint (actor & critic networks) and optimizer states"""
        checkpoint_dir = f"./checkpoints/{run_id}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_data = {
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
            "step": checkpoint_step
        }

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_step}.pt")
        torch.save(checkpoint_data, checkpoint_path)

        # Log to MLflow
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        logger.info(f"Checkpoint saved at step {checkpoint_step} - {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load MADDPG checkpoint (actor & critic networks) and optimizer states"""
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file {checkpoint_path} not found.")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        for actor, actor_state in zip(self.actors, checkpoint["actors"]):
            actor.load_state_dict(actor_state)

        for critic, critic_state in zip(self.critics, checkpoint["critics"]):
            critic.load_state_dict(critic_state)

        for opt, opt_state in zip(self.actor_optimizers, checkpoint["actor_optimizers"]):
            opt.load_state_dict(opt_state)

        for opt, opt_state in zip(self.critic_optimizers, checkpoint["critic_optimizers"]):
            opt.load_state_dict(opt_state)

        self.current_step = checkpoint["step"]

        logger.info(f"Checkpoint loaded from {checkpoint_path} at step {self.current_step}.")
        return True



