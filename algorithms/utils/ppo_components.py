"""PPO Components — Actor, Critic, RolloutBuffer, and loss functions.

These components are specific to the PPO algorithm. The Actor and Critic
share the Transformer backbone but have separate heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger
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
        logger.info(
            "Initialized ActorHead (d_model={}, hidden_dim={}, log_std_init={})",
            d_model,
            hidden_dim,
            log_std_init,
        )

    def forward(
        self,
        ca_embeddings: torch.Tensor,
        deterministic: bool = False,
        *,
        return_pre_tanh: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Produce actions from CA embeddings.

        Args:
            ca_embeddings: [batch, N_ca, d_model] CA token embeddings.
            deterministic: If True, return mean action without sampling.
            return_pre_tanh: If True, also return the unsquashed sample.

        Returns:
            Tuple of:
                 - actions: [batch, N_ca, 1] sampled actions in [-1, 1].
                 - log_probs: [batch, N_ca] log probability of actions.
                 - means: [batch, N_ca, 1] action means (pre-tanh).
                 - pre_tanh_actions: [batch, N_ca, 1] when requested.
        """
        # Get action means
        means = self.mlp(ca_embeddings)  # [batch, N_ca, 1]

        # Get standard deviation (clamped to prevent entropy collapse/divergence)
        log_std_clamped = torch.clamp(self.log_std, min=-2.0, max=0.5)
        std = torch.exp(log_std_clamped).expand_as(means)

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

        log_probs = self._squashed_log_prob(dist, pre_tanh_action)

        if return_pre_tanh:
            return actions, log_probs, means, pre_tanh_action

        return actions, log_probs, means

    def log_prob_from_pre_tanh(
        self,
        ca_embeddings: torch.Tensor,
        pre_tanh_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Score retained pre-tanh samples under the current actor policy."""
        means = self.mlp(ca_embeddings)
        log_std_clamped = torch.clamp(self.log_std, min=-2.0, max=0.5)
        std = torch.exp(log_std_clamped).expand_as(means)
        return self._squashed_log_prob(Normal(means, std), pre_tanh_actions)

    @staticmethod
    def _squashed_log_prob(dist: Normal, pre_tanh_actions: torch.Tensor) -> torch.Tensor:
        actions = torch.tanh(pre_tanh_actions)
        log_probs = dist.log_prob(pre_tanh_actions)
        log_probs = log_probs - torch.log(1 - actions.pow(2) + 1e-6)
        return log_probs.squeeze(-1)


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
        logger.info("Initialized CriticHead (d_model={}, hidden_dim={})", d_model, hidden_dim)

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
    pre_tanh_actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RunningValueNormalizer:
    """Maintain scalar running statistics for value normalization."""

    _EPSILON = 1e-8

    def __init__(self) -> None:
        self.mean = 0.0
        self.variance = 1.0
        self.count = 0

    def update(self, values: torch.Tensor) -> None:
        """Update running population statistics from a tensor of values."""
        if values.numel() == 0:
            return

        samples = values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        batch_count = samples.numel()
        batch_mean = samples.mean().item()
        batch_variance = samples.var(unbiased=False).item()

        if self.count == 0:
            self.mean = batch_mean
            self.variance = batch_variance
            self.count = batch_count
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        combined_m2 = (
            self.variance * self.count
            + batch_variance * batch_count
            + delta * delta * self.count * batch_count / total_count
        )
        self.mean += delta * batch_count / total_count
        self.variance = combined_m2 / total_count
        self.count = total_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize values with the current running statistics."""
        mean = values.new_tensor(self.mean)
        scale = values.new_tensor((max(self.variance, 0.0) + self._EPSILON) ** 0.5)
        return (values - mean) / scale

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Restore normalized values to the original scale."""
        mean = values.new_tensor(self.mean)
        scale = values.new_tensor((max(self.variance, 0.0) + self._EPSILON) ** 0.5)
        return values * scale + mean

    def state_dict(self) -> Dict[str, Union[float, int]]:
        """Return device-agnostic scalar statistics."""
        return {
            "mean": self.mean,
            "variance": self.variance,
            "count": self.count,
        }

    def load_state_dict(self, state: Dict[str, Union[float, int]]) -> None:
        """Restore scalar statistics from :meth:`state_dict`."""
        self.mean = float(state["mean"])
        self.variance = float(state["variance"])
        self.count = int(state["count"])


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
        self.pre_tanh_actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.terminated: List[bool] = []
        self.truncated: List[bool] = []

        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None
        logger.debug("Initialized RolloutBuffer (gamma={}, gae_lambda={})", gamma, gae_lambda)

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        terminated: bool,
        truncated: bool,
        pre_tanh_action: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            observation: Encoded observation tensor.
            action: Action tensor.
            pre_tanh_action: Unsquashed policy sample, when available.
            log_prob: Log probability of the action.
            reward: Reward received.
            value: Value estimate from critic.
            terminated: Whether the transition ended in a terminal state.
            truncated: Whether the transition ended due to a time or topology limit.
        """
        self.observations.append(observation.detach())
        self.actions.append(action.detach())
        self.pre_tanh_actions.append(
            (action if pre_tanh_action is None else pre_tanh_action).detach()
        )
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.values.append(value.detach())
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        logger.debug("Added transition to RolloutBuffer (size={})", len(self.observations))

    def compute_returns_and_advantages(self, last_value: torch.Tensor) -> None:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: Value estimate for the state after the last transition.
        """
        if any(self.truncated[:-1]):
            raise ValueError(
                "RolloutBuffer must be flushed at a truncation boundary; "
                "one last_value cannot bootstrap an interior truncation."
            )

        n = len(self.rewards)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)

        # Convert values to tensor
        values = torch.stack([v.squeeze() for v in self.values])

        # GAE computation (reverse order)
        gae = torch.tensor(0.0)
        next_value = last_value.squeeze()

        for t in reversed(range(n)):
            bootstrap_mask = 1.0 - float(self.terminated[t])
            is_truncated = self.truncated[t]
            bootstrap_value = last_value.squeeze() if is_truncated else next_value
            delta = self.rewards[t] + self.gamma * bootstrap_value * bootstrap_mask - values[t]
            continuation_mask = bootstrap_mask * (1.0 - float(is_truncated))
            gae = delta + self.gamma * self.gae_lambda * continuation_mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        # Normalize advantages. Guard against degenerate single-element
        # rollouts (where ``std()`` is undefined / NaN with the default
        # unbiased estimator) by skipping normalization in that case.
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        self.advantages = advantages
        self.returns = returns
        logger.debug("Computed rollout returns/advantages for {} transition(s)", n)

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
                pre_tanh_actions=torch.stack(
                    [self.pre_tanh_actions[i] for i in batch_indices]
                ),
                log_probs=torch.stack([self.log_probs[i] for i in batch_indices]),
                advantages=self.advantages[batch_indices],
                returns=self.returns[batch_indices],
                values=torch.stack([self.values[i].squeeze() for i in batch_indices]),
            )

    def clear(self) -> None:
        """Clear all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.pre_tanh_actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.advantages = None
        self.returns = None
        logger.debug("Cleared RolloutBuffer")

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
            - metrics: Dict with ``policy_loss``, ``value_loss``, ``entropy``,
               ``clip_fraction``, ``approx_kl``, ``ratio_error_max``, and
               ``explained_variance``.
    """
    # Float64 intermediates keep finite float16 and float32 inputs finite.
    # The clamp bounds every finite log-ratio before exponentiation.
    loss_dtype = torch.float64 if log_probs_new.dtype in (torch.float16, torch.float32) else log_probs_new.dtype
    log_probs_new_loss = log_probs_new.to(dtype=loss_dtype)
    log_probs_old_loss = log_probs_old.to(dtype=loss_dtype)
    advantages_loss = advantages.to(dtype=loss_dtype)
    values_loss = values.to(dtype=loss_dtype)
    returns_loss = returns.to(dtype=loss_dtype)

    log_ratio = torch.clamp(log_probs_new_loss - log_probs_old_loss, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    # Clipped surrogate objective
    def surrogate_with_safe_gradient(surrogate_ratio: torch.Tensor) -> torch.Tensor:
        surrogate = surrogate_ratio * advantages_loss
        max_source_gradient = torch.finfo(log_probs_new.dtype).max
        gradient_ratio = torch.minimum(
            surrogate_ratio,
            surrogate_ratio.new_tensor(max_source_gradient) / advantages_loss.abs().clamp_min(1.0),
        )
        gradient_surrogate = gradient_ratio * advantages_loss
        return surrogate.detach() + gradient_surrogate - gradient_surrogate.detach()

    surr1 = surrogate_with_safe_gradient(ratio)
    surr2 = surrogate_with_safe_gradient(torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps))
    policy_loss = -torch.min(surr1, surr2).mean()

    # Smooth L1 keeps gradients for large residuals without quadratic growth.
    value_loss = torch.nn.functional.smooth_l1_loss(values_loss, returns_loss)

    # Entropy bonus (approximate using log_probs)
    # For squashed Gaussian, entropy is complex; use simple approximation
    entropy = -log_probs_new_loss.mean()

    # Combined loss
    total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

    # Clip fraction: proportion of samples where ratio was clipped
    clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
    diagnostic_ratio = torch.nan_to_num(ratio.to(dtype=torch.float64))
    diagnostic_dtype = torch.float32 if log_probs_new.dtype == torch.float16 else log_probs_new.dtype
    diagnostic_log_ratio = log_ratio.to(dtype=diagnostic_dtype)
    diagnostic_ppo_ratio = ratio.to(dtype=diagnostic_dtype)
    approx_kl = ((diagnostic_ppo_ratio - 1.0) - diagnostic_log_ratio).mean()
    ratio_error_max = (diagnostic_ratio - 1.0).abs().max()
    diagnostic_returns = returns_loss.to(dtype=torch.float64)
    diagnostic_values = values_loss.to(dtype=torch.float64)
    returns_variance = diagnostic_returns.var(unbiased=False)
    explained_variance = 1.0 - (diagnostic_returns - diagnostic_values).var(unbiased=False) / torch.clamp(
        returns_variance,
        min=1e-8,
    )
    if not torch.isfinite(explained_variance):
        explained_variance = torch.zeros_like(explained_variance)

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl.item(),
        "ratio_error_max": ratio_error_max.item(),
        "explained_variance": explained_variance.item(),
    }

    logger.debug(
        "Computed PPO loss (policy_loss={:.6f}, value_loss={:.6f}, entropy={:.6f}, clip_frac={:.4f})",
        metrics["policy_loss"],
        metrics["value_loss"],
        metrics["entropy"],
        metrics["clip_fraction"],
    )

    return total_loss, metrics
