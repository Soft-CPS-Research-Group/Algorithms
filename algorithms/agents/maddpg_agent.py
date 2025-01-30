import torch
from torch.nn.functional import mse_loss
from loguru import logger
from torch.amp import GradScaler, autocast
import random
from algorithms.utils.networks import Actor, Critic
from algorithms.utils.replay_buffer import *
from typing import List
from algorithms.utils.noise import add_noise

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
        seed = self.config["hyperparameters"].get("seed", 22)
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

    def update(self, observations, actions, rewards, next_observations, dones, initital_exploration_done, update_step, update_target_step):
        """2. Perform a training update."""
        logger.debug("Starting update phase.")

        # 2.1. Push experiences to replay buffer
        logger.debug("Pushing experiences to replay buffer.")
        self.replay_buffer.push(observations, actions, rewards, next_observations, dones)

        # 2.2.1 Check if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            logger.warning("Not enough samples in the replay buffer. Skipping update.")
            return

        # 2.2.2 Check if exploration phase is finished
        if not initital_exploration_done:
            logger.warning("Initial Exploration phase is not yet finished. Skipping update.")
            return

        # 2.2.3 Check if it is an update time step
        if not update_step:
            logger.warning("Not Updating due to time step. Skipping update.")
            return

        # 2.3. Sample from replay buffer
        logger.debug("Sampling from replay buffer.")
        batch = self.replay_buffer.sample(self.batch_size)

        for agent_num, (actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer) in enumerate(
            zip(self.actors, self.critics, self.actor_targets, self.critic_targets, self.actor_optimizers, self.critic_optimizers)
        ):
            logger.debug(f"Updating agent {agent_num}.")

            # 2.4. Prepare data for this agent
            obs, actions, rewards, next_obs, dones = batch[agent_num]
            logger.debug(f"Agent {agent_num} data prepared.")

            # 2.5. Update critic
            logger.debug(f"Updating critic for agent {agent_num}.")
            with autocast(self.device):
                q_expected = critic(obs, actions)
                next_actions = [actor_target(next_obs) for actor_target in self.actor_targets]
                q_targets_next = critic_target(next_obs, next_actions)
                q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
                critic_loss = mse_loss(q_expected, q_targets.detach())

            self.scaler.scale(critic_loss).backward()
            self.scaler.step(critic_optimizer)
            critic_optimizer.zero_grad()
            self.scaler.update()
            logger.debug(f"Critic updated for agent {agent_num}. Loss: {critic_loss.item()}.")

            # 2.6. Update actor
            logger.debug(f"Updating actor for agent {agent_num}.")
            with autocast(self.device):
                predicted_actions = [actor(obs) for actor in self.actors]
                actor_loss = -critic(obs, predicted_actions).mean()

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
        observations : List[torch.Tensor]
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
            bias = self.config["hyperparameters"]["exploration"].get("bias", 0.3)  # Default bias to 0.3
            noise = np.random.normal(scale=self.config["hyperparameters"]["exploration"]["sigma"]) - bias
            random_noises.append(noise)

        # Add noise to deterministic actions
        actions = [noise + action for action, noise in zip(deterministic_actions, random_noises)]

        # Clip actions to stay within valid range (-1, 1)
        clipped_actions = [np.clip(action, -1, 1) for action in actions]

        # Convert to list format
        actions_return = [action.tolist() for action in clipped_actions]

        logger.debug(f"Actions with exploration noise and constraints applied: {actions_return}")
        return actions_return

