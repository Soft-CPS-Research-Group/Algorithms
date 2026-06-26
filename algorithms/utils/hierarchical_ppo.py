"""Hierarchical PPO components for the Community Coordinator Agent.

This module provides a *set-based* actor-critic and matching rollout buffer
that let the Community Coordinator (CC) emit ONE signal per building while
remaining agnostic to the number of buildings ``N``.

Why a new module instead of reusing :mod:`algorithms.utils.ppo`?

  The existing ``PPOActorCritic`` is a single flat MLP ``(obs_dim -> action_dim)``.
  To emit ``N`` signals it would need ``action_dim = N`` and a flattened input of
  width ``c_dim + N * b_dim`` — both hardcode ``N``. A network trained on a
  17-building community could never run on a 5-building one.

  Here the actor applies the SAME small head to every building independently
  (weight sharing across the set). Parameter count is independent of ``N``:
  train on 17, run on any number. This is the generality the thesis requires.

  The PPO *algorithm* (clipped surrogate, GAE, KL early-stop) is unchanged and
  still lives in the agent; only the network forward pass and the buffer's
  storage shapes differ.

Glossary (same as ppo.py)
    community context : fixed-width vector shared by all buildings (price,
                        hour, community import/PV/net, carbon, headroom...).
    building features : fixed-width vector per building (battery soc, pv, net
                        load, EV state...). One row per building.
    actor head        : shared MLP mapping ``concat(context, building_i)`` to a
                        single scalar mean for building ``i``.
    centralized critic: mean-pools building encodings and maps
                        ``concat(context, pooled)`` to ONE community value.
                        This is the CTDE pattern: per-building actions, one
                        community-level value for credit assignment.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from algorithms.utils.ppo import _build_mlp


# ---------------------------------------------------------------------------
# Set-based actor-critic
# ---------------------------------------------------------------------------

class HierarchicalActorCritic(nn.Module):
    """Per-building actor with a centralized (community-level) critic.

    Actor (shared head, applied per building):
      - actor_head   : concat(context, building_i) -> mean_i   (one scalar)
      - actor_logstd : single learnable scalar (NOT state-dependent), shared
                       across buildings.

    Critic (centralized, permutation-invariant):
      - building_encoder : building_i -> enc_dim   (shared)
      - mean-pool over buildings -> pooled (fixed enc_dim regardless of N)
      - critic           : concat(context, pooled) -> V(community)  (scalar)

    Args:
        c_dim:       community-context dimensionality.
        b_dim:       per-building feature dimensionality.
        hidden_dims: hidden widths for actor_head and critic.
        enc_dim:     width of the per-building encoding pooled by the critic.

    Shapes (B = batch, N = number of buildings):
        community : (B, c_dim)
        buildings : (B, N, b_dim)
        action    : (B, N)        — one signal per building, in (-1, 1)
        log_prob  : (B,)          — joint over the N buildings
        entropy   : (B,)
        value     : (B, 1)        — single community value
    """

    def __init__(
        self,
        c_dim: int,
        b_dim: int,
        hidden_dims: list[int] | None = None,
        enc_dim: int = 64,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.c_dim = c_dim
        self.b_dim = b_dim
        self.enc_dim = enc_dim

        # Shared per-building encoder feeding the centralized critic pool.
        # Output std=sqrt(2) is the standard hidden-layer choice (it is a
        # hidden representation, not a policy/value head).
        self.building_encoder = _build_mlp(
            b_dim, hidden_dims, output_dim=enc_dim, output_std=np.sqrt(2)
        )

        # Shared actor head: applied identically to every building.
        # Output std=0.01 keeps the initial per-building mean near zero.
        self.actor_head = _build_mlp(
            c_dim + b_dim, hidden_dims, output_dim=1, output_std=0.01
        )

        # One log-std scalar, shared by all buildings. std ≈ 1 at init.
        self.actor_logstd = nn.Parameter(torch.zeros(1))

        # Centralized critic: community context + pooled building encoding.
        self.critic = _build_mlp(
            c_dim + enc_dim, hidden_dims, output_dim=1, output_std=1.0
        )

    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        community: torch.Tensor,
        buildings: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) per-building actions and the community value.

        During rollout (action=None): samples a fresh action per building.
        During PPO update (action given): evaluates log-prob of stored actions.

        Args:
            community: (B, c_dim)
            buildings: (B, N, b_dim)
            action:    (B, N) or None.

        Returns:
            action   : (B, N) in (-1, 1)
            log_prob : (B,)   joint over buildings
            entropy  : (B,)
            value    : (B, 1)
        """
        b, n, _ = buildings.shape

        # Broadcast community context to every building, concat, run shared head.
        c_exp = community.unsqueeze(1).expand(-1, n, -1)        # (B, N, c_dim)
        actor_in = torch.cat([c_exp, buildings], dim=-1)        # (B, N, c+b)

        # tanh keeps the mean in (-1, 1). We model the Gaussian directly on the
        # squashed mean (no tanh change-of-variables) — simpler and adequate for
        # a coarse community signal. Actions are clamped to (-1, 1) at sampling.
        means = torch.tanh(self.actor_head(actor_in).squeeze(-1))   # (B, N)

        std = torch.exp(self.actor_logstd).expand_as(means)         # (B, N)
        dist = Normal(means, std)

        if action is None:
            action = dist.sample().clamp(-1.0 + 1e-6, 1.0 - 1e-6)   # (B, N)

        log_prob = dist.log_prob(action).sum(dim=-1)                # (B,)
        entropy = dist.entropy().sum(dim=-1)                        # (B,)

        # Centralized value: encode each building, mean-pool (permutation- and
        # count-invariant), concat with community context.
        enc = self.building_encoder(buildings)                      # (B, N, enc)
        pooled = enc.mean(dim=1)                                    # (B, enc)
        value = self.critic(torch.cat([community, pooled], dim=-1))  # (B, 1)

        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class HierarchicalRolloutBuffer:
    """Fixed-size on-policy buffer for the hierarchical CC.

    Differs from :class:`algorithms.utils.ppo.RolloutBuffer` only in storage
    shapes: the observation is split into a community context and a per-building
    feature matrix, and the action is an ``N``-vector. Reward, value, logprob,
    advantage and return remain scalar-per-step, so the GAE recursion is
    byte-for-byte identical.

    Stored arrays:
        community  (num_steps, c_dim)
        buildings  (num_steps, N, b_dim)
        actions    (num_steps, N)
        logprobs   (num_steps,)
        rewards    (num_steps,)
        dones      (num_steps,)
        values     (num_steps,)
        advantages (num_steps,)   — filled by compute_gae
        returns    (num_steps,)   — filled by compute_gae
    """

    def __init__(self, num_steps: int, c_dim: int, b_dim: int, num_buildings: int) -> None:
        self.num_steps = num_steps
        self.c_dim = c_dim
        self.b_dim = b_dim
        self.num_buildings = num_buildings
        self.ptr = 0
        self.full = False

        self.community = np.zeros((num_steps, c_dim), dtype=np.float32)
        self.buildings = np.zeros((num_steps, num_buildings, b_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_buildings), dtype=np.float32)
        self.logprobs = np.zeros(num_steps, dtype=np.float32)
        self.rewards = np.zeros(num_steps, dtype=np.float32)
        self.dones = np.zeros(num_steps, dtype=np.float32)
        self.values = np.zeros(num_steps, dtype=np.float32)

        self.advantages = np.zeros(num_steps, dtype=np.float32)
        self.returns = np.zeros(num_steps, dtype=np.float32)

    # ------------------------------------------------------------------

    def add(
        self,
        community: np.ndarray,
        buildings: np.ndarray,
        action: np.ndarray,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Store one CC-level transition."""
        self.community[self.ptr] = community
        self.buildings[self.ptr] = buildings
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value

        self.ptr += 1
        if self.ptr >= self.num_steps:
            self.full = True

    # ------------------------------------------------------------------

    def compute_gae(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """GAE-λ. Identical recursion to the flat buffer (scalar reward/value).

        δ_t = r_t + γ·V(s_{t+1})·(1-done_t) − V(s_t)
        A_t = δ_t + γλ·(1-done_t)·A_{t+1}
        R_t = A_t + V(s_t)
        """
        last_gae = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------

    def get(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Return stored data as tensors. Advantages normalised (PPO trick)."""
        adv = self.advantages
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        return {
            "community":  torch.tensor(self.community, dtype=torch.float32, device=device),
            "buildings":  torch.tensor(self.buildings, dtype=torch.float32, device=device),
            "actions":    torch.tensor(self.actions,   dtype=torch.float32, device=device),
            "logprobs":   torch.tensor(self.logprobs,  dtype=torch.float32, device=device),
            "returns":    torch.tensor(self.returns,   dtype=torch.float32, device=device),
            "advantages": torch.tensor(adv_norm,       dtype=torch.float32, device=device),
        }

    def reset(self) -> None:
        """Clear the buffer for the next rollout."""
        self.ptr = 0
        self.full = False
        self.community[:] = 0
        self.buildings[:] = 0
        self.actions[:] = 0
        self.logprobs[:] = 0
        self.rewards[:] = 0
        self.dones[:] = 0
        self.values[:] = 0
        self.advantages[:] = 0
        self.returns[:] = 0
