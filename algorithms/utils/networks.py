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
        self.feature_size = fc_units[-1]
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fc_layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc_layers[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_layers[-1].bias, -3e-3, 3e-3)

    def encode(self, state):
        x = state
        for fc in self.fc_layers[:-1]:
            x = F.relu(fc(x))
        return x

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.encode(state)
        return torch.tanh(self.fc_layers[-1](x))


class MultiHeadActor(nn.Module):
    """Actor with a shared state trunk and one output head per action dimension.

    The environment action layout is only attached after agent construction, so
    this intentionally avoids semantic EV/storage grouping. It still separates
    the final command heads, which reduces direct output competition between
    EV, storage and deferrable controls while preserving the existing actor
    contract.
    """

    def __init__(self, state_size, action_size, seed, fc_units=None, head_units=None):
        super(MultiHeadActor, self).__init__()
        fc_units = fc_units or [256, 128]
        head_units = list(head_units or [])
        self.seed = torch.manual_seed(seed)
        self.trunk_layers = nn.ModuleList()
        input_dim = state_size
        for hidden_dim in fc_units:
            self.trunk_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.feature_size = input_dim

        self.action_heads = nn.ModuleList()
        for _ in range(action_size):
            layers = nn.ModuleList()
            head_input_dim = input_dim
            for hidden_dim in head_units:
                layers.append(nn.Linear(head_input_dim, hidden_dim))
                head_input_dim = hidden_dim
            layers.append(nn.Linear(head_input_dim, 1))
            self.action_heads.append(layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.trunk_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        for head in self.action_heads:
            for layer in head[:-1]:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
            nn.init.uniform_(head[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(head[-1].bias, -3e-3, 3e-3)

    def set_output_bias(self, bias):
        bias_tensor = torch.as_tensor(bias, dtype=torch.float32)
        if bias_tensor.numel() != len(self.action_heads):
            raise ValueError("MultiHeadActor output bias size does not match action size.")
        with torch.no_grad():
            for index, head in enumerate(self.action_heads):
                output_layer = head[-1]
                output_layer.weight.zero_()
                output_layer.bias.fill_(float(bias_tensor[index].item()))

    def encode(self, state):
        x = state
        for layer in self.trunk_layers:
            x = F.relu(layer(x))
        return x

    def forward(self, state):
        squeeze_output = state.dim() == 1
        if squeeze_output:
            state = state.unsqueeze(0)
        x = self.encode(state)
        outputs = []
        for head in self.action_heads:
            y = x
            for layer in head[:-1]:
                y = F.relu(layer(y))
            outputs.append(head[-1](y))
        action = torch.tanh(torch.cat(outputs, dim=1))
        return action.squeeze(0) if squeeze_output else action


class SemanticMultiHeadActor(nn.Module):
    """Actor with shared trunk and output heads grouped by action category.

    ``action_groups`` maps a category name (for example ``"ev"`` or
    ``"storage"``) to the original action indices handled by that head. The
    forward pass scatters each head output back into the original action order,
    so the external actor contract stays unchanged.
    """

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        fc_units=None,
        head_units=None,
        action_groups=None,
    ):
        super(SemanticMultiHeadActor, self).__init__()
        fc_units = fc_units or [256, 128]
        head_units = list(head_units or [])
        self.seed = torch.manual_seed(seed)
        self.trunk_layers = nn.ModuleList()
        input_dim = state_size
        for hidden_dim in fc_units:
            self.trunk_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.feature_size = input_dim
        self.action_size = int(action_size)
        self.group_names, self.group_indices = self._resolve_groups(action_size, action_groups)

        self.group_heads = nn.ModuleList()
        for indices in self.group_indices:
            layers = nn.ModuleList()
            head_input_dim = input_dim
            for hidden_dim in head_units:
                layers.append(nn.Linear(head_input_dim, hidden_dim))
                head_input_dim = hidden_dim
            layers.append(nn.Linear(head_input_dim, len(indices)))
            self.group_heads.append(layers)
        self.reset_parameters()

    @staticmethod
    def _resolve_groups(action_size, action_groups):
        action_size = int(action_size)
        groups = []
        seen = set()
        if isinstance(action_groups, dict):
            for name in ("ev", "storage", "deferrable", "other"):
                raw_indices = action_groups.get(name, [])
                indices = []
                for value in raw_indices:
                    index = int(value)
                    if 0 <= index < action_size and index not in seen:
                        indices.append(index)
                        seen.add(index)
                if indices:
                    groups.append((name, indices))
        remaining = [index for index in range(action_size) if index not in seen]
        if remaining:
            groups.append(("other", remaining))
        if not groups:
            groups = [("other", list(range(action_size)))]
        return [name for name, _ in groups], [indices for _, indices in groups]

    def reset_parameters(self):
        for layer in self.trunk_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        for head in self.group_heads:
            for layer in head[:-1]:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
            nn.init.uniform_(head[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(head[-1].bias, -3e-3, 3e-3)

    def set_output_bias(self, bias):
        bias_tensor = torch.as_tensor(bias, dtype=torch.float32)
        if bias_tensor.numel() != self.action_size:
            raise ValueError("SemanticMultiHeadActor output bias size does not match action size.")
        with torch.no_grad():
            for head, indices in zip(self.group_heads, self.group_indices):
                output_layer = head[-1]
                output_layer.weight.zero_()
                output_layer.bias.copy_(
                    bias_tensor[torch.as_tensor(indices, dtype=torch.long, device=bias_tensor.device)].to(
                        dtype=output_layer.bias.dtype,
                        device=output_layer.bias.device,
                    )
                )

    def encode(self, state):
        x = state
        for layer in self.trunk_layers:
            x = F.relu(layer(x))
        return x

    def forward(self, state):
        squeeze_output = state.dim() == 1
        if squeeze_output:
            state = state.unsqueeze(0)
        x = self.encode(state)
        outputs = state.new_zeros((state.shape[0], self.action_size))
        for head, indices in zip(self.group_heads, self.group_indices):
            y = x
            for layer in head[:-1]:
                y = F.relu(layer(y))
            group_output = head[-1](y)
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=state.device)
            outputs.index_copy_(1, index_tensor, group_output)
        action = torch.tanh(outputs)
        return action.squeeze(0) if squeeze_output else action


def build_actor_network(state_size, action_size, seed, network_config):
    """Instantiate an actor from config while preserving the legacy default."""
    if isinstance(network_config, dict):
        class_name = network_config.get("class") or network_config.get("class_name") or "Actor"
        fc_units = network_config.get("layers")
        if class_name == "Actor":
            return Actor(state_size, action_size, seed, fc_units)
        if class_name == "MultiHeadActor":
            return MultiHeadActor(
                state_size,
                action_size,
                seed,
                fc_units,
                head_units=network_config.get("head_layers"),
            )
        if class_name == "SemanticMultiHeadActor":
            return SemanticMultiHeadActor(
                state_size,
                action_size,
                seed,
                fc_units,
                head_units=network_config.get("head_layers"),
                action_groups=network_config.get("action_groups"),
            )
        raise ValueError(f"Unsupported actor network class: {class_name}")

    return Actor(state_size, action_size, seed, network_config)


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
