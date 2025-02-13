import torch
from torch.nn.functional import mse_loss
from loguru import logger
from torch.amp import GradScaler, autocast
import random
from algorithms.utils.networks import Actor, Critic
from algorithms.utils.replay_buffer import *
from typing import List
from algorithms.utils.noise import add_noise
import mlflow.pytorch
import os
from utils.mlflow_helper import log_to_mlflow

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


class MADDPG:
    def __init__(self, config):
        """1. Initialize MADDPG with structured logging and streamlined config handling."""
        # 1.1. Store configuration
        self.config = config

        # 1.2. Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device selected: {self.device}")

        # 1.3. Hyperparameters
        self.gamma = self.config["algorithm"]["exploration"]["params"]["gamma"]
        self.tau = self.config["algorithm"]["exploration"]["params"]["tau"]
        self.sigma = self.config["algorithm"]["exploration"]["params"]["sigma"]
        self.bias = self.config["algorithm"]["exploration"]["params"]["bias"]
        self.batch_size = self.config["algorithm"]["replay_buffer"]["params"]["batch_size"]
        self.lr_actor = float(self.config["algorithm"]["networks"]["actor_network"]["params"]["lr"])
        self.lr_critic = float(self.config["algorithm"]["networks"]["critic_network"]["params"]["lr"])

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

            print(replay_buffer_params)

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
        seed = self.config["algorithm"].get("seed", 22)
        actor_fc_units = self.config["algorithm"]["networks"]["actor_network"]["params"]["layers"]
        critic_fc_units = self.config["algorithm"]["networks"]["critic_network"]["params"]["layers"]

        actors, critics, actor_targets, critic_targets = [], [], [], []
        for i in range(self.num_agents):
            # Use agent-specific observation and action dimensions
            state_size = self.observation_dimension[i]
            action_size = self.action_dimension[i]

            # Initialize networks
            actors.append(Actor(state_size, action_size, seed, actor_fc_units).to(self.device))
            critics.append(Critic(sum(self.observation_dimension), sum(self.action_dimension), seed, critic_fc_units).to(self.device))
            actor_targets.append(Actor(state_size, action_size, seed, actor_fc_units).to(self.device))
            critic_targets.append(Critic(sum(self.observation_dimension), sum(self.action_dimension), seed, critic_fc_units).to(self.device))

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

    def update(self, observations, actions, rewards, next_observations, terminated, truncated, initial_exploration_done, update_step, update_target_step, time_step, *args, **kwargs):
        """2. Perform a training update."""
        logger.debug("Starting update phase.")

        # 2.1. Push experiences to replay buffer
        logger.debug("Pushing experiences to replay buffer.")
        self.replay_buffer.push(observations, actions, rewards, next_observations, terminated)

        # 2.2.1 Check if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            logger.warning("Not enough samples in the replay buffer. Skipping update.")
            return

        # 2.2.2 Check if exploration phase is finished
        if not initial_exploration_done:
            logger.warning("Initial Exploration phase is not yet finished. Skipping update.")
            return

        # 2.2.3 Check if it is an update time step
        if not update_step:
            logger.warning("Not Updating due to time step. Skipping update.")
            return

        # 2.3. Sample from replay buffer
        logger.debug("Sampling from replay buffer.")
        batch = self.replay_buffer.sample()
        # Unpack the batch from the replay buffer
        states, actions_all, rewards_all, next_states, dones_all = batch

        # For the critic we need global (concatenated) states and actions.
        global_state = torch.cat(states, dim=1)  # shape: (batch_size, sum(observation_dimensions))
        global_next_state = torch.cat(next_states, dim=1)
        global_actions = torch.cat(actions_all, dim=1)  # shape: (batch_size, sum(action_dimensions))
        # Compute global next actions by feeding each agent's next state into its corresponding actor_target.
        global_next_actions = torch.cat(
            [actor_target(next_states[i]) for i, actor_target in enumerate(self.actor_targets)], dim=1
        )

        for agent_num, (actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer) in enumerate(
            zip(self.actors, self.critics, self.actor_targets, self.critic_targets, self.actor_optimizers, self.critic_optimizers)
        ):
            logger.debug(f"Updating agent {agent_num}.")

            # 2.4. Prepare data for this agent
            # Extract individual data for the agent.
            obs = states[agent_num]  # Individual observation
            sampled_action = actions_all[agent_num]  # Sampled action
            reward = rewards_all[agent_num]
            terminated_agent = dones_all[agent_num]  # Numeric done flag (0.0 or 1.0)
            next_obs = next_states[agent_num]

            logger.debug(f"Agent {agent_num} data prepared.")

            # 2.5. Update critic
            # --------------------- Update Critic ---------------------
            logger.debug(f"Updating critic for agent {agent_num}.")
            with autocast(device_type=self.device.type):
                # Use global state and global actions for the critic.
                q_expected = critic(global_state, global_actions)
                q_targets_next = critic_target(global_next_state, global_next_actions)
                q_targets = reward + self.gamma * q_targets_next * (1 - terminated_agent)
                critic_loss = mse_loss(q_expected, q_targets.detach())

            self.scaler.scale(critic_loss).backward()
            self.scaler.step(critic_optimizer)
            critic_optimizer.zero_grad()
            self.scaler.update()
            logger.debug(f"Critic updated for agent {agent_num}. Loss: {critic_loss.item()}.")

            # 2.6. Update actor
            logger.debug(f"Updating actor for agent {agent_num}.")
            with autocast(device_type=self.device.type):
                # Predict action for the current agent using its own observation.
                predicted_action = actor(obs)
                # Build global predicted actions: replace the current agent's action with the predicted one,
                # while keeping the sampled actions for all other agents.
                global_predicted_actions = []
                for i in range(self.num_agents):
                    if i == agent_num:
                        global_predicted_actions.append(predicted_action)
                    else:
                        # Detach sampled actions for other agents so gradients do not flow through them.
                        global_predicted_actions.append(actions_all[i].detach())
                global_predicted_actions = torch.cat(global_predicted_actions, dim=1)
                # Actor loss computed using the critic on the global state and the new global predicted actions.
                actor_loss = -critic(global_state, global_predicted_actions).mean()

            self.scaler.scale(actor_loss).backward()
            self.scaler.step(actor_optimizer)
            actor_optimizer.zero_grad()
            self.scaler.update()
            logger.debug(f"Actor updated for agent {agent_num}. Loss: {actor_loss.item()}.")

            # 2.7. Log metrics using helper
            from utils.mlflow_helper import log_to_mlflow
            log_to_mlflow(f"critic_loss_agent_{agent_num}", critic_loss.item())
            log_to_mlflow(f"actor_loss_agent_{agent_num}", actor_loss.item())

            # 2.8. Update target networks
            if update_target_step:
                logger.debug(f"Updating target networks for agent {agent_num}.")
                self._soft_update(critic, critic_target, self.tau)
                self._soft_update(actor, actor_target, self.tau)

            log_to_mlflow(f"critic_loss_agent_{agent_num}", critic_loss.item(), step=time_step)
            log_to_mlflow(f"actor_loss_agent_{agent_num}", actor_loss.item(), step=time_step)

    def _soft_update(self, local_model, target_model, tau):
        """2.8.1 Soft update model parameters."""
        logger.debug("Performing soft update.")
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

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



