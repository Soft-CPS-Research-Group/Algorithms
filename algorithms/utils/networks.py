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
