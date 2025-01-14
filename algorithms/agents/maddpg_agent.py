from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from .base_agent import BaseAgent
import random
import yaml

class MADDPG(BaseAgent):
    def __init__(self, config_path: str, mlflow):
        """Initialize MADDPG agent using a YAML configuration."""
        # 1. Setup
        # 1.1. Load configurations from YAML
        self.config = self._load_config(config_path)

        # 1.2. Experiment settings
        self.experiment_name = self.config.get("experiment", {}).get("name", "default_experiment")
        self.log_dir = self.config.get("experiment", {}).get("log_dir", "logs")

        # Algorithm hyperparameters
        self.num_agents = self.config.get("num_agents", 1)
        self.gamma = self.config.get("gamma", 0.99)
        self.batch_size = self.config.get("batch_size", 128)
        self.tau = self.config.get("tau", 1e-3)
        self.sigma = self.config.get("sigma", 0.2)
        self.target_update_interval = self.config.get("target_update_interval", 2)
        self.steps_between_training_updates = self.config.get("steps_between_training_updates", 5)
        self.lr_actor = self.config.get("lr_actor", 1e-5)
        self.lr_critic = self.config.get("lr_critic", 1e-4)

        # Neural network dimensions
        self.actor_units = self.config.get("actor_units", [256, 128])
        self.critic_units = self.config.get("critic_units", [256, 128])

        # Seed and device setup
        self.seed = self.config.get("seed", random.randint(0, 100_000_000))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device selected: {self.device}")

        # Initialize components
        self.replay_buffer = self._initialize_replay_buffer()
        self.exploration_strategy = self._initialize_exploration()
        self.actors, self.critics, self.actors_target, self.critics_target = self._initialize_networks()
        self.actors_optimizer, self.critics_optimizer = self._initialize_optimizers()

        # GradScaler for mixed precision training
        self.scaler = GradScaler()
        self.exploration_done = False

    def _load_config(self, config_path: str):
        """Loads configuration from a YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _initialize_replay_buffer(self):
        """Dynamically initialize the replay buffer from the configuration."""
        buffer_config = self.config.get("replay_buffer", {})
        buffer_class_name = buffer_config.get("class", "ReplayBuffer1")
        buffer_params = buffer_config.get("params", {})

        # Dynamically load the replay buffer class
        buffer_class = getattr(my_replay_buffer_module, buffer_class_name)  # Replace with your replay buffer module
        return buffer_class(**buffer_params)

    def _initialize_exploration(self):
        """Dynamically initialize the exploration strategy."""
        exploration_config = self.config.get("exploration", {})
        strategy_name = exploration_config.get("strategy", "GaussianNoise")
        strategy_params = exploration_config.get("params", {})

        # Dynamically load the exploration strategy class
        strategy_class = getattr(my_exploration_module, strategy_name)  # Replace with your exploration module
        return strategy_class(**strategy_params)

    def _initialize_networks(self):
        """Initializes actor and critic networks and their target networks."""
        actors = [
            Actor(
                self.config["observation_dimension"][i],
                self.config["action_space"][i].shape[0],
                self.seed,
                self.actor_units,
            ).to(self.device)
            for i in range(self.num_agents)
        ]

        critics = [
            Critic(
                sum(self.config["observation_dimension"]),
                sum(self.config["action_dimension"]),
                self.seed,
                self.critic_units,
            ).to(self.device)
            for _ in range(self.num_agents)
        ]

        actors_target = [
            Actor(
                self.config["observation_dimension"][i],
                self.config["action_space"][i].shape[0],
                self.seed,
                self.actor_units,
            ).to(self.device)
            for i in range(self.num_agents)
        ]

        critics_target = [
            Critic(
                sum(self.config["observation_dimension"]),
                sum(self.config["action_dimension"]),
                self.seed,
                self.critic_units,
            ).to(self.device)
            for _ in range(self.num_agents)
        ]

        return actors, critics, actors_target, critics_target

    def _initialize_optimizers(self):
        """Initializes optimizers for actors and critics."""
        actors_optimizer = [
            torch.optim.Adam(actor.parameters(), lr=self.lr_actor)
            for actor in self.actors
        ]

        critics_optimizer = [
            torch.optim.Adam(critic.parameters(), lr=self.lr_critic)
            for critic in self.critics
        ]

        return actors_optimizer, critics_optimizer

    def update(self, observations, actions, rewards, next_observations, dones):
        """Perform a training update using the replay buffer."""
        self.replay_buffer.push(observations, actions, rewards, next_observations, dones)

        if len(self.replay_buffer) < self.batch_size:
            return

        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = self.replay_buffer.sample(
            self.batch_size
        )

        # Prepare tensors
        obs_full = torch.cat(obs_batch, dim=1).to(self.device)
        next_obs_full = torch.cat(next_obs_batch, dim=1).to(self.device)
        action_full = torch.cat(actions_batch, dim=1).to(self.device)

        for agent_num, (actor, critic, actor_target, critic_target, actor_optim, critic_optim) in enumerate(
            zip(self.actors, self.critics, self.actors_target, self.critics_target, self.actors_optimizer, self.critics_optimizer)
        ):
            with autocast():
                # Update critic
                Q_expected = critic(obs_full, action_full)
                next_actions = [
                    actor_target(next_obs) for next_obs in next_obs_batch
                ]
                next_actions_full = torch.cat(next_actions, dim=1)
                Q_targets_next = critic_target(next_obs_full, next_actions_full)
                Q_targets = rewards_batch[agent_num] + self.gamma * Q_targets_next * (1 - dones_batch[agent_num])
                critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

            self.scaler.scale(critic_loss).backward()
            self.scaler.step(critic_optim)
            critic_optim.zero_grad()
            self.scaler.update()

            with autocast():
                # Update actor
                predicted_actions = [actor(obs) for obs in obs_batch]
                predicted_actions_full = torch.cat(predicted_actions, dim=1)
                actor_loss = -critic(obs_full, predicted_actions_full).mean()

            self.scaler.scale(actor_loss).backward()
            self.scaler.step(actor_optim)
            actor_optim.zero_grad()
            self.scaler.update()

            # Update target networks
            if self.steps_between_training_updates % self.target_update_interval == 0:
                self.soft_update(critic, critic_target, self.tau)
                self.soft_update(actor, actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self):
        """Save the model's actor and critic networks."""
        save_path = f"models/{self.experiment_name}_maddpg.pth"
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actors_target': [actor_target.state_dict() for actor_target in self.actors_target],
            'critics_target': [critic_target.state_dict() for critic_target in self.critics_target],
            'optimizers': {
                'actors': [optim.state_dict() for optim in self.actors_optimizer],
                'critics': [optim.state_dict() for optim in self.critics_optimizer],
            },
        }, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Load the model's actor and critic networks."""
        checkpoint = torch.load(load_path)
        for actor, state_dict in zip(self.actors, checkpoint['actors']):
            actor.load_state_dict(state_dict)
        for critic, state_dict in zip(self.critics, checkpoint['critics']):
            critic.load_state_dict(state_dict)
        for actor_target, state_dict in zip(self.actors_target, checkpoint['actors_target']):
            actor_target.load_state_dict(state_dict)
        for critic_target, state_dict in zip(self.critics_target, checkpoint['critics_target']):
            critic_target.load_state_dict(state_dict)
        for optim, state_dict in zip(self.actors_optimizer, checkpoint['optimizers']['actors']):
            optim.load_state_dict(state_dict)
        for optim, state_dict in zip(self.critics_optimizer, checkpoint['optimizers']['critics']):
            optim.load_state_dict(state_dict)
        print(f"Model loaded from {load_path}")     