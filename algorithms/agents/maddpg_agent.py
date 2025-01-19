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
        self.gamma = self.config["hyperparameters"]["training"]["gamma"]
        self.batch_size = self.config["hyperparameters"]["training"]["batch_size"]
        self.tau = self.config["hyperparameters"]["training"]["tau"]
        self.lr_actor = float(self.config["hyperparameters"]["algorithm"]["optimizer"]["actor"]["params"]["lr"])
        self.lr_critic = float(self.config["hyperparameters"]["algorithm"]["optimizer"]["critic"]["params"]["lr"])
        self.target_update_interval = self.config["hyperparameters"]["training"]["target_update_interval"]

        # 1.4. Log parameters using the MLflow helper
        self._log_mlflow_params()

        # 1.5. Retrieve number of agents and their dimensions
        self.num_agents = self.config["hyperparameters"]["training"]["num_agents"]
        self.observation_dimension = self.config["simulator"]["observation_dimensions"]
        self.action_dimension = self.config["simulator"]["action_dimensions"]

        # 1.6. Replay buffer
        self.replay_buffer = self._initialize_replay_buffer()

        # 1.7. Neural networks and optimizers
        self.actors, self.critics, self.actor_targets, self.critic_targets = self._initialize_networks()
        self.actor_optimizers, self.critic_optimizers = self._initialize_optimizers()
        self.scaler = GradScaler(self.device)

        logger.info("MADDPG initialization complete.")

    def _log_mlflow_params(self):
        """1.4.1 Log hyperparameters to MLflow using a helper."""
        from utils.mlflow_helper import log_params_to_mlflow
        logger.debug("Logging hyperparameters to MLflow using helper.")
        log_params_to_mlflow({
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic
        })

    def _initialize_replay_buffer(self):
        """1.6.1 Initialize the replay buffer dynamically based on configuration."""
        logger.debug("Initializing replay buffer dynamically.")

        try:
            replay_buffer_class = self.config["replay_buffer"]["class"]
            replay_buffer_params = self.config["replay_buffer"]["params"]
            logger.debug(f"Replay buffer class: {replay_buffer_class}, params: {replay_buffer_params}")

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
        seed = self.config["hyperparameters"]["training"].get("seed", 0)  # Default to 0 if not specified

        actors, critics, actor_targets, critic_targets = [], [], [], []
        for i in range(self.num_agents):
            # Use agent-specific observation and action dimensions
            state_size = self.observation_dimension[i]
            action_size = self.action_dimension[i]

            # Initialize networks
            actors.append(Actor(state_size, action_size, seed).to(self.device))
            critics.append(Critic(sum(self.observation_dimension), sum(self.action_dimension), seed).to(self.device))
            actor_targets.append(Actor(state_size, action_size, seed).to(self.device))
            critic_targets.append(Critic(sum(self.observation_dimension), sum(self.action_dimension), seed).to(self.device))

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

    def update(self, observations, actions, rewards, next_observations, dones):
        """2. Perform a training update."""
        logger.debug("Starting update phase.")

        # 2.1. Push experiences to replay buffer
        logger.debug("Pushing experiences to replay buffer.")
        self.replay_buffer.push(observations, actions, rewards, next_observations, dones)

        # 2.2. Check if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            logger.warning("Not enough samples in the replay buffer. Skipping update.")
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
            if self.target_update_interval % self.config["hyperparameters"]["training"]["steps_between_training_updates"] == 0:
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
        """3.2 Predict actions with added exploration noise.

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

        # Apply noise using the external noise utility
        noisy_actions = [
            add_noise(
                action,
                sigma=self.config["hyperparameters"]["exploration"]["sigma"],
                bias=self.config["hyperparameters"]["exploration"].get("bias", 0.0),
            )
            for action in deterministic_actions
        ]

        #Hard Constraints to exploration
        #for i, b in enumerate(self.env.buildings):
        #    if b.chargers:
        #        for charger_index, charger in reversed(list(enumerate(b.chargers))):
        #            # If no EV is connected, set action to 0
        #            if not charger.connected_ev:
        #                actions_return[i][-charger_index - 1] = 0.0001

        logger.debug(f"Actions with exploration noise: {noisy_actions}")
        return noisy_actions

    """
    endoded observation
    set 
    ohysics constrains
    """
