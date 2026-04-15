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


class CriticHead(nn.Module):
    """Critic head that produces state value from pooled embedding.

    Takes the mean-pooled representation of all tokens and outputs
    a scalar value estimate V(s).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
    ) -> None:
        """Initialize the critic head.

        Args:
            d_model: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Produce state value from pooled embedding.

        Args:
            pooled: [batch, d_model] mean-pooled token embeddings.

        Returns:
            values: [batch, 1] state value estimates.
        """
        return self.mlp(pooled)


@dataclass
class Batch:
    """A minibatch of transitions for PPO update."""
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """On-policy rollout buffer for PPO.

    Stores transitions from the current policy, computes GAE advantages,
    and provides minibatch iteration for PPO updates.
    """

    def __init__(self, gamma: float, gae_lambda: float) -> None:
        """Initialize the rollout buffer.

        Args:
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []
        
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            observation: Encoded observation tensor.
            action: Action tensor.
            log_prob: Log probability of the action.
            reward: Reward received.
            value: Value estimate from critic.
            done: Whether episode terminated.
        """
        self.observations.append(observation.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.values.append(value.detach())
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: torch.Tensor) -> None:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: Value estimate for the state after the last transition.
        """
        n = len(self.rewards)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)
        
        # Convert values to tensor
        values = torch.stack([v.squeeze() for v in self.values])
        
        # GAE computation (reverse order)
        gae = torch.tensor(0.0)
        next_value = last_value.squeeze()
        
        for t in reversed(range(n)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size: int) -> Iterator[Batch]:
        """Yield minibatches for PPO update.

        Args:
            batch_size: Size of each minibatch.

        Yields:
            Batch objects containing transition data.
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Must call compute_returns_and_advantages first")
        
        n = len(self.observations)
        indices = torch.randperm(n)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            yield Batch(
                observations=torch.stack([self.observations[i] for i in batch_indices]),
                actions=torch.stack([self.actions[i] for i in batch_indices]),
                log_probs=torch.stack([self.log_probs[i] for i in batch_indices]),
                advantages=self.advantages[batch_indices],
                returns=self.returns[batch_indices],
                values=torch.stack([self.values[i].squeeze() for i in batch_indices]),
            )

    def clear(self) -> None:
        """Clear all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.observations)


def compute_ppo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    value_coeff: float,
    entropy_coeff: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute PPO clipped surrogate loss.

    Args:
        log_probs_new: Log probabilities under current policy.
        log_probs_old: Log probabilities under old policy (detached).
        advantages: GAE advantages (normalized).
        values: Value estimates from critic.
        returns: Discounted returns.
        clip_eps: Clipping epsilon for probability ratio.
        value_coeff: Coefficient for value loss.
        entropy_coeff: Coefficient for entropy bonus.

    Returns:
        Tuple of:
            - total_loss: Combined loss for backprop.
            - metrics: Dict with policy_loss, value_loss, entropy.
    """
    # Probability ratio
    ratio = torch.exp(log_probs_new - log_probs_old)
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss (MSE)
    value_loss = F.mse_loss(values, returns)
    
    # Entropy bonus (approximate using log_probs)
    # For squashed Gaussian, entropy is complex; use simple approximation
    entropy = -log_probs_new.mean()
    
    # Combined loss
    total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
    
    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }
    
    return total_loss, metrics
