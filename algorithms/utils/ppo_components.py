"""PPO Components — Actor, Critic, RolloutBuffer, and loss functions.

These components are specific to the PPO algorithm. The Actor and Critic
share the Transformer backbone but have separate heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorHead(nn.Module):
    """Actor head that produces actions from CA embeddings.

    Applies an MLP to each CA embedding independently, producing action means.
    Uses a squashed Gaussian distribution (Normal + tanh) for sampling.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        log_std_init: float = -0.5,
    ) -> None:
        """Initialize the actor head.

        Args:
            d_model: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
            log_std_init: Initial value for log standard deviation.
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Learnable log standard deviation (shared across all CAs)
        self.log_std = nn.Parameter(torch.tensor(log_std_init))

    def forward(
        self,
        ca_embeddings: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce actions from CA embeddings.

        Args:
            ca_embeddings: [batch, N_ca, d_model] CA token embeddings.
            deterministic: If True, return mean action without sampling.

        Returns:
            Tuple of:
                - actions: [batch, N_ca, 1] sampled actions in [-1, 1].
                - log_probs: [batch, N_ca] log probability of actions.
                - means: [batch, N_ca, 1] action means (pre-tanh).
        """
        # Get action means
        means = self.mlp(ca_embeddings)  # [batch, N_ca, 1]
        
        # Get standard deviation
        std = torch.exp(self.log_std).expand_as(means)
        
        # Create normal distribution
        dist = Normal(means, std)
        
        if deterministic:
            # Use mean action
            pre_tanh_action = means
        else:
            # Sample from distribution
            pre_tanh_action = dist.rsample()
        
        # Apply tanh squashing
        actions = torch.tanh(pre_tanh_action)
        
        # Compute log probability with tanh correction
        log_probs = dist.log_prob(pre_tanh_action)
        # Correction for tanh squashing: log(1 - tanh(x)^2)
        log_probs = log_probs - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.squeeze(-1)  # [batch, N_ca]
        
        return actions, log_probs, means
