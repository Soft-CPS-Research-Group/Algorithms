import numpy as np
import random
from collections import namedtuple, deque
import copy

# conditional imports
try:
    import torch
    from torch.distributions import Normal
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise Exception("This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list): List of node counts in the hidden layers.
        """
        super(Actor, self).__init__()
        fc_units = fc_units or [256, 128]
        self.seed = torch.manual_seed(seed)

        # Input layer
        self.fc_layers = [nn.Linear(state_size, fc_units[0])]

        # Intermediate layers
        for i in range(1, len(fc_units)):
            self.fc_layers.append(nn.Linear(fc_units[i - 1], fc_units[i]))

        # Output layer
        self.fc_layers.append(nn.Linear(fc_units[-1], action_size))

        # ModuleList to register the layers with PyTorch
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fc_layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc_layers[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_layers[-1].bias, -3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        for fc in self.fc_layers[:-1]:
            x = F.relu(fc(x))
        return torch.tanh(self.fc_layers[-1](x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list): List of node counts in the hidden layers.
        """
        super(Critic, self).__init__()
        fc_units = fc_units or [256, 128]
        self.seed = torch.manual_seed(seed)
        self.fc_layers = nn.ModuleList()
        input_dim = state_size + action_size
        for hidden_dim in fc_units:
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.q_out = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fc_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.q_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_out.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=1)
        for fc in self.fc_layers:
            x = F.relu(fc(x))

        return self.q_out(x)


class LateFusionCritic(nn.Module):
    """Critic with separate state/action encoders before Q-value fusion.

    This keeps the centralized critic contract used by MADDPG-style agents, but
    avoids mixing the full global observation and full joint action vector in
    the first linear layer. State and action embeddings are learned separately
    and fused only in the joint Q head, which usually gives cleaner credit
    assignment for continuous-control actor-critic methods.
    """

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        fc_units=None,
        state_fc_units=None,
        action_fc_units=None,
        joint_fc_units=None,
    ):
        super(LateFusionCritic, self).__init__()
        fc_units = fc_units or [256, 128]
        state_fc_units, action_fc_units, joint_fc_units = self._resolve_layer_plan(
            fc_units,
            state_fc_units,
            action_fc_units,
            joint_fc_units,
        )
        self.seed = torch.manual_seed(seed)
        self.state_layers = self._build_layers(state_size, state_fc_units)
        self.action_layers = self._build_layers(action_size, action_fc_units)
        fusion_input_dim = (
            (state_fc_units[-1] if state_fc_units else state_size)
            + (action_fc_units[-1] if action_fc_units else action_size)
        )
        self.joint_layers = self._build_layers(fusion_input_dim, joint_fc_units)
        q_input_dim = joint_fc_units[-1] if joint_fc_units else fusion_input_dim
        self.q_out = nn.Linear(q_input_dim, 1)
        self.reset_parameters()

    @staticmethod
    def _resolve_layer_plan(fc_units, state_fc_units, action_fc_units, joint_fc_units):
        if state_fc_units is not None or action_fc_units is not None or joint_fc_units is not None:
            return (
                list(state_fc_units or []),
                list(action_fc_units or []),
                list(joint_fc_units or fc_units),
            )

        if len(fc_units) == 1:
            hidden_dim = int(fc_units[0])
            action_hidden_dim = max(16, hidden_dim // 2)
            return [hidden_dim], [action_hidden_dim], [hidden_dim]

        state_hidden_dim = int(fc_units[0])
        action_hidden_dim = max(16, state_hidden_dim // 2)
        return [state_hidden_dim], [action_hidden_dim], list(fc_units[1:])

    @staticmethod
    def _build_layers(input_dim, hidden_dims):
        layers = nn.ModuleList()
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        return layers

    def reset_parameters(self):
        for layer_group in (self.state_layers, self.action_layers, self.joint_layers):
            for layer in layer_group:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.q_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_out.bias, -3e-3, 3e-3)

    @staticmethod
    def _forward_layers(x, layers):
        for layer in layers:
            x = F.relu(layer(x))
        return x

    def forward(self, state, action):
        """Build a critic network that maps separately encoded (state, action) -> Q."""
        state_embedding = self._forward_layers(state, self.state_layers)
        action_embedding = self._forward_layers(action, self.action_layers)
        x = torch.cat((state_embedding, action_embedding), dim=1)
        x = self._forward_layers(x, self.joint_layers)
        return self.q_out(x)


def build_critic_network(state_size, action_size, seed, network_config):
    """Instantiate a critic from config while preserving the legacy default."""
    if isinstance(network_config, dict):
        class_name = network_config.get("class") or network_config.get("class_name") or "Critic"
        fc_units = network_config.get("layers")
        if class_name == "Critic":
            return Critic(state_size, action_size, seed, fc_units)
        if class_name == "LateFusionCritic":
            return LateFusionCritic(
                state_size,
                action_size,
                seed,
                fc_units,
                state_fc_units=network_config.get("state_layers"),
                action_fc_units=network_config.get("action_layers"),
                joint_fc_units=network_config.get("joint_layers"),
            )
        raise ValueError(f"Unsupported critic network class: {class_name}")

    return Critic(state_size, action_size, seed, network_config)


class ValueNetwork(nn.Module):
    """State-value model used by PPO-style agents."""

    def __init__(self, state_size, seed, fc_units=None):
        super(ValueNetwork, self).__init__()
        fc_units = fc_units or [256, 128]
        self.seed = torch.manual_seed(seed)
        self.fc_layers = nn.ModuleList()
        input_dim = state_size
        for hidden_dim in fc_units:
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.value_out = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fc_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.value_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.value_out.bias, -3e-3, 3e-3)

    def forward(self, state):
        x = state
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return self.value_out(x)


class GaussianActor(nn.Module):
    """Gaussian policy over normalized actions in [-1, 1]."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        fc_units=None,
        initial_log_std=-0.5,
        min_log_std=-5.0,
        max_log_std=1.0,
    ):
        super(GaussianActor, self).__init__()
        fc_units = fc_units or [256, 128]
        self.seed = torch.manual_seed(seed)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.fc_layers = nn.ModuleList()
        input_dim = state_size
        for hidden_dim in fc_units:
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.mean_out = nn.Linear(input_dim, action_size)
        self.log_std = nn.Parameter(torch.full((action_size,), float(initial_log_std)))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fc_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.mean_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_out.bias, -3e-3, 3e-3)

    def forward(self, state):
        x = state
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return torch.tanh(self.mean_out(x))

    def distribution(self, state):
        mean = self.forward(state)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def sample_normalized(self, state, epsilon=1.0e-6):
        """Sample a tanh-squashed normalized action and corrected log-prob."""
        distribution = self.distribution(state)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action)
        log_prob = log_prob - torch.log(torch.clamp(1.0 - action.pow(2), min=epsilon))
        return action, log_prob.sum(dim=-1, keepdim=True)
