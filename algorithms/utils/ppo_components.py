"""PPO components — actor/critic heads, rollout buffer, and loss functions.

Implements:
* ``ActorHead`` — squashed Gaussian (Normal + tanh) applied to CA positions.
* ``CriticHead`` — value head applied to pooled representation.
* ``PPORolloutBuffer`` — on-policy storage with GAE computation.
* ``ppo_loss`` — clipped surrogate + value + entropy loss.
"""

from __future__ import annotations

import math
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Actor Head
# ---------------------------------------------------------------------------

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class ActorHead(nn.Module):
    """Actor head for PPO with squashed Gaussian distribution.

    Applied to CA token positions: produces one scalar action per CA in ``[-1, 1]``.

    Parameters
    ----------
    d_model : int
        Input dimension (from Transformer backbone).
    d_ff : int
        Hidden dimension of the 2-layer MLP.
    n_ca_types : int
        Number of distinct CA types. Each type gets a learnable log-std.
    """

    def __init__(self, d_model: int, d_ff: int, n_ca_types: int = 1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, 1)

        # Learnable log-std per CA type
        self.log_std = nn.Parameter(torch.zeros(max(n_ca_types, 1)))

    def forward(
        self,
        ca_embeddings: torch.Tensor,
        ca_type_indices: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute actions from CA embeddings.

        Parameters
        ----------
        ca_embeddings : Tensor[batch, N_ca, d_model]
        ca_type_indices : Tensor[N_ca] (long)
            Index into ``self.log_std`` for each CA position. If ``None``,
            uses index 0 for all CAs.
        deterministic : bool
            If True, use the mean action; otherwise sample.

        Returns
        -------
        actions : Tensor[batch, N_ca, 1]
            Actions in [-1, 1].
        log_probs : Tensor[batch, N_ca, 1]
            Log-probability of the chosen action (corrected for tanh squashing).
        entropy : Tensor[batch, N_ca, 1]
            Entropy estimate of the Gaussian (pre-squash).
        """

        x = self.norm(ca_embeddings)
        x = F.gelu(self.fc1(x))
        mu = self.fc2(x)  # [batch, N_ca, 1]

        # Per-CA-type std
        if ca_type_indices is not None:
            log_std = self.log_std[ca_type_indices]  # [N_ca]
            # Expand to [1, N_ca, 1] for broadcasting
            log_std = log_std.unsqueeze(0).unsqueeze(-1).expand_as(mu)
        else:
            log_std = self.log_std[0].expand_as(mu)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mu, std)

        if deterministic:
            u = mu
        else:
            u = dist.rsample()

        # Squash through tanh
        actions = torch.tanh(u)

        # Log-prob with tanh correction: log_pi = log_prob(u) - sum(log(1 - tanh(u)^2))
        log_probs = dist.log_prob(u) - torch.log(1 - actions.pow(2) + 1e-6)

        # Entropy of the pre-squash Gaussian
        entropy = dist.entropy()

        return actions, log_probs, entropy

    def evaluate_actions(
        self,
        ca_embeddings: torch.Tensor,
        old_actions: torch.Tensor,
        ca_type_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate log-probs and entropy for previously taken actions.

        Parameters
        ----------
        ca_embeddings : Tensor[batch, N_ca, d_model]
        old_actions : Tensor[batch, N_ca, 1]
            Previously taken actions (already squashed through tanh).
        ca_type_indices : Tensor[N_ca]

        Returns
        -------
        log_probs : Tensor[batch, N_ca, 1]
        entropy : Tensor[batch, N_ca, 1]
        """

        x = self.norm(ca_embeddings)
        x = F.gelu(self.fc1(x))
        mu = self.fc2(x)

        if ca_type_indices is not None:
            log_std = self.log_std[ca_type_indices].unsqueeze(0).unsqueeze(-1).expand_as(mu)
        else:
            log_std = self.log_std[0].expand_as(mu)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mu, std)

        # Invert tanh to get u from actions: u = atanh(actions)
        u = torch.atanh(torch.clamp(old_actions, -1 + 1e-6, 1 - 1e-6))

        log_probs = dist.log_prob(u) - torch.log(1 - old_actions.pow(2) + 1e-6)
        entropy = dist.entropy()

        return log_probs, entropy


# ---------------------------------------------------------------------------
# Critic Head
# ---------------------------------------------------------------------------


class CriticHead(nn.Module):
    """Value-function head for PPO.

    Applied to the pooled representation (mean over all token positions).

    Parameters
    ----------
    d_model : int
        Input dimension.
    d_ff : int
        Hidden dimension.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Parameters
        ----------
        pooled : Tensor[batch, d_model]

        Returns
        -------
        Tensor[batch, 1]
        """

        x = self.norm(pooled)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# PPO Rollout Buffer
# ---------------------------------------------------------------------------


class PPORolloutBuffer:
    """On-policy buffer storing complete trajectories for PPO updates.

    Stores transitions and computes GAE (Generalized Advantage Estimation)
    after each rollout.
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self) -> None:
        """Reset all stored data."""
        self.observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

        # Computed after rollout
        self._returns: Optional[torch.Tensor] = None
        self._advantages: Optional[torch.Tensor] = None

    def push(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self.observations.append(observation.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.values.append(value.detach())
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        last_done: bool,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_value : Tensor[1]
            V(s_{T+1}) — bootstrap value for the last state.
        last_done : bool
            Whether the last state was terminal.
        """

        n = len(self.rewards)
        advantages = torch.zeros(n)
        last_gae = 0.0
        last_val = last_value.detach().squeeze().item()

        for t in reversed(range(n)):
            mask = 0.0 if self.dones[t] else 1.0
            val_t = self.values[t].squeeze().item()

            delta = self.rewards[t] + self.gamma * last_val * mask - val_t
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae

            last_val = val_t

        returns = advantages + torch.tensor(
            [v.squeeze().item() for v in self.values]
        )

        self._advantages = advantages
        self._returns = returns

    def get_batches(
        self,
        batch_size: int,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield minibatches for PPO update epochs.

        Yields
        ------
        Dict with keys: ``observations``, ``actions``, ``old_log_probs``,
        ``returns``, ``advantages``.
        """

        if self._advantages is None:
            raise RuntimeError("Call compute_returns_and_advantages() first")

        n = len(self.rewards)
        obs = torch.stack(self.observations)
        acts = torch.stack(self.actions)
        old_lp = torch.stack(self.log_probs)
        rets = self._returns
        advs = self._advantages

        # Normalize advantages
        if advs.std() > 1e-8:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Random permutation
        indices = torch.randperm(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                "observations": obs[idx],
                "actions": acts[idx],
                "old_log_probs": old_lp[idx],
                "returns": rets[idx],
                "advantages": advs[idx],
            }


# ---------------------------------------------------------------------------
# PPO Loss
# ---------------------------------------------------------------------------


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_eps: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the combined PPO loss.

    Parameters
    ----------
    new_log_probs : Tensor
        Log-probs under the current policy.
    old_log_probs : Tensor
        Log-probs under the old (data-collection) policy.
    advantages : Tensor
        GAE advantages.
    values : Tensor
        Predicted V(s) from the critic.
    returns : Tensor
        Discounted returns (advantages + old values).
    entropy : Tensor
        Entropy estimate from the actor.
    clip_eps : float
        PPO clipping epsilon.
    value_coeff : float
        Coefficient for the value loss.
    entropy_coeff : float
        Coefficient for the entropy bonus.

    Returns
    -------
    total_loss : Tensor
        Combined scalar loss.
    metrics : dict
        Individual loss components for logging.
    """

    # Clipped surrogate objective
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value function loss (MSE)
    value_loss = F.mse_loss(values.squeeze(-1), returns)

    # Entropy bonus (negative to encourage exploration)
    entropy_loss = -entropy.mean()

    total = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": -entropy_loss.item(),
        "total_loss": total.item(),
    }

    return total, metrics
