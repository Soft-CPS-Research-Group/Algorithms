"""BC policy network.

Self-contained re-implementation of v1's ``algorithms.offline.bc_policy.BCPolicy``,
imported by nothing in ``algorithms/offline/`` so v1 stays untouched.

Architecture matches v1 (validated baseline): ``obs_dim → 256 → 256 → action_dim``,
ReLU hidden activations, optional dropout between hidden layers, Tanh on the
output so actions are bounded to ``[-1, 1]`` (CityLearn's per-agent action
space; the env clips again on its side).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCPolicy(nn.Module):
    """Multi-layer perceptron behavioural-cloning policy ."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = (256, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if not hidden_layers:
            raise ValueError("hidden_layers must be non-empty")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_layers: List[int] = [int(h) for h in hidden_layers]
        self.dropout_p = float(dropout)

        sizes: List[int] = [self.obs_dim, *self.hidden_layers]
        self.hidden = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(sizes[:-1], sizes[1:])
        )
        self.dropout = (
            nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else nn.Identity()
        )
        self.output = nn.Linear(sizes[-1], self.action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map a (batch of) standardised observations to actions in [-1, 1]."""
        x = obs
        for layer in self.hidden:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return torch.tanh(self.output(x))

    def architecture_summary(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot of the net shape (used in metadata)."""
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_layers": list(self.hidden_layers),
            "hidden_activation": "relu",
            "output_activation": "tanh",
            "dropout": self.dropout_p,
        }
