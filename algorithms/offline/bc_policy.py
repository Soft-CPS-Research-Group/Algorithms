"""MLP policy used for Behavioral Cloning.

Architecture: ``obs_dim → 256 → 256 → action_dim`` with ReLU hidden
activations and Tanh output activation (so outputs are bounded to
``[-1, 1]`` — matching the CityLearn action space).

The same class is used by:

* :mod:`algorithms.offline.bc_trainer` — for training.
* :mod:`algorithms.agents.offline_bc_agent` — for inference inside CityLearn.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCPolicy(nn.Module):
    """Multi-layer perceptron behavioural-cloning policy."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_layers = list(hidden_layers)

        layer_sizes: List[int] = [obs_dim, *self.hidden_layers]
        self.hidden = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
        )
        self.output = nn.Linear(layer_sizes[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map a (batch of) standardized observations to a (batch of) actions in [-1, 1]."""
        x = obs
        for layer in self.hidden:
            x = F.relu(layer(x))
        return torch.tanh(self.output(x))

    def architecture_summary(self) -> dict:
        """Return a JSON-friendly summary of the network shape (used for metadata)."""
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "hidden_layers": list(self.hidden_layers),
            "hidden_activation": "relu",
            "output_activation": "tanh",
        }
