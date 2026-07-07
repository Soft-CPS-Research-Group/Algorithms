"""Deterministic actor head for Transformer-MATD3.

Applies an MLP to each CA token embedding independently, producing one
scalar action per CA token. Output is tanh-squashed to [-1, 1].
Unlike PPO's stochastic ActorHead, this is purely deterministic —
exploration noise is added externally.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DeterministicActorHead(nn.Module):
    """MLP per CA embedding -> tanh-squashed scalar action."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ca_embeddings: torch.Tensor) -> torch.Tensor:
        """Return tanh-squashed actions [B, N_ca, 1]."""
        return torch.tanh(self.mlp(ca_embeddings))

    def forward_with_pre_tanh(
        self, ca_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (tanh_actions, pre_tanh_means) for target smoothing."""
        pre_tanh = self.mlp(ca_embeddings)
        return torch.tanh(pre_tanh), pre_tanh
