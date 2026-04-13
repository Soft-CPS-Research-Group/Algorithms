"""PPO components for the Community Coordinator Agent.

Two classes live here:

  PPOActorCritic  — the neural network (actor + critic in one module)
  RolloutBuffer   — collects experience for one PPO update cycle

Design follows Huang et al. (2022) "The 37 Implementation Details of PPO"
but strips away Gym/vectorised-env dependencies so it fits BaseAgent.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Orthogonal weight init + constant bias — standard PPO trick.

    Orthogonal init keeps gradient magnitudes stable at the start of training.
    The output layer of the actor uses std=0.01 (near-zero) so the initial
    policy is nearly uniform, and the critic uses std=1.0.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, output_std: float) -> nn.Sequential:
    """Build a fully-connected MLP with Tanh activations.

    Args:
        input_dim:   Number of input features.
        hidden_dims: List of hidden layer widths, e.g. [64, 64].
        output_dim:  Number of output neurons.
        output_std:  Orthogonal init std for the output layer (controls how
                     'spread out' the initial outputs are).
    """
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(_layer_init(nn.Linear(in_dim, h)))
        layers.append(nn.Tanh())
        in_dim = h
    layers.append(_layer_init(nn.Linear(in_dim, output_dim), std=output_std))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class PPOActorCritic(nn.Module):
    """Shared-backbone actor-critic for a continuous action space.

    The actor outputs a Gaussian distribution over actions:
      - actor_mean  : state → action mean  (MLP)
      - actor_logstd: learnable parameter  (NOT state-dependent)

    The critic outputs a scalar state value:
      - critic: state → V(s)  (separate MLP, no action input)

    Args:
        obs_dim:      Dimensionality of the community state vector.
        action_dim:   Number of continuous actions (1 for O1 scalar).
        hidden_dims:  Hidden layer widths for both actor and critic MLPs.
    """

    def __init__(self, obs_dim: int, action_dim: int = 1, hidden_dims: list[int] | None = None) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        # Critic: state → scalar value estimate V(s)
        # Output std=1.0 is the standard PPO choice for the value head.
        self.critic = _build_mlp(obs_dim, hidden_dims, output_dim=1, output_std=1.0)

        # Actor: state → mean of the Gaussian policy
        # Output std=0.01 keeps initial actions close to zero (no strong bias).
        self.actor_mean = _build_mlp(obs_dim, hidden_dims, output_dim=action_dim, output_std=0.01)

        # Log-std is a single learnable parameter vector, NOT dependent on state.
        # Starting at 0 means std ≈ 1 — reasonable initial exploration.
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) an action and return everything PPO needs.

        During rollout collection (action=None):
          → samples a new action from the current policy.

        During the PPO update (action=stored action):
          → evaluates the log-probability of that stored action under the
            CURRENT (updated) policy — needed for the importance ratio.

        Args:
            x:      Community state tensor, shape (batch, obs_dim).
            action: If None, sample a fresh action. Otherwise evaluate this one.

        Returns:
            action    : sampled or passed-in action, shape (batch, action_dim)
            log_prob  : log π(action | state),       shape (batch,)
            entropy   : policy entropy H[π(·|state)], shape (batch,)
            value     : critic estimate V(state),     shape (batch,)
        """
        action_mean = self.actor_mean(x)

        # Expand actor_logstd to match the batch dimension of action_mean.
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # Gaussian distribution over actions.
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        # sum() over action_dim so log_prob is scalar per sample.
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        value = self.critic(x)

        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-size buffer that collects one PPO rollout (on-policy experience).

    PPO is on-policy: we collect `num_steps` transitions, do one (or a few)
    gradient update(s) using all of them, then discard everything.

    Stored arrays (all shape ``(num_steps,)`` or ``(num_steps, dim)``):
      obs       — community state at each step
      actions   — action taken (O1 scalar)
      logprobs  — log π(a|s) under the policy that collected the data
      rewards   — scalar reward received
      dones     — 1.0 if episode ended, else 0.0
      values    — V(s) estimated by the critic at collection time

    After the rollout is full, call ``compute_gae()`` to add:
      advantages — GAE-λ advantage estimates
      returns    — TD(λ) targets for the value function

    Then call ``get()`` to retrieve tensors ready for the PPO loss.
    """

    def __init__(self, num_steps: int, obs_dim: int, action_dim: int = 1) -> None:
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ptr = 0       # pointer to the next empty slot
        self.full = False  # True once we have collected num_steps transitions

        # Pre-allocate numpy arrays for speed.
        self.obs = np.zeros((num_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, action_dim), dtype=np.float32)
        self.logprobs = np.zeros(num_steps, dtype=np.float32)
        self.rewards = np.zeros(num_steps, dtype=np.float32)
        self.dones = np.zeros(num_steps, dtype=np.float32)
        self.values = np.zeros(num_steps, dtype=np.float32)

        # Filled by compute_gae().
        self.advantages = np.zeros(num_steps, dtype=np.float32)
        self.returns = np.zeros(num_steps, dtype=np.float32)

    # ------------------------------------------------------------------
    # Adding transitions
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: float,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Store one transition.  Call this once per environment step."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value

        self.ptr += 1
        if self.ptr >= self.num_steps:
            self.full = True

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalised Advantage Estimation (GAE-λ).

        Must be called after the rollout is full, before calling ``get()``.

        How GAE works
        -------------
        At each step t we compute the TD residual (a.k.a. "delta"):

            δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) − V(s_t)

        δ_t is the one-step advantage: how much better did we do vs. the
        critic's prediction

        GAE then builds a discounted sum of these residuals — going BACKWARD
        from the last step — with an extra λ decay:

            A_t = δ_t + γλ · (1 - done_t) · A_{t+1}

        When λ=0  → pure one-step TD (low variance, high bias)
        When λ=1  → full Monte Carlo (high variance, low bias)
        λ=0.95 is the canonical PPO default.

        The return (value target) is simply:

            R_t = A_t + V(s_t)

        Args:
            last_value: V(s_{T+1}) from the critic — needed to bootstrap the
                        advantage at the last step.
            last_done:  Whether the last step ended an episode.
            gamma:      Discount factor.
            gae_lambda: GAE smoothing parameter λ.
        """
        last_gae = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # TD residual
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]

            # GAE recursion (going backwards, so last_gae is A_{t+1})
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    # Retrieving data for the PPO update
    # ------------------------------------------------------------------

    def get(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Return all stored data as a flat dict of tensors.

        The advantages are normalised (zero mean, unit std) here — this is a
        standard PPO trick that stabilises training by keeping the gradient
        scale consistent regardless of reward magnitude.
        """
        adv = self.advantages
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        return {
            "obs":        torch.tensor(self.obs,      dtype=torch.float32, device=device),
            "actions":    torch.tensor(self.actions,  dtype=torch.float32, device=device),
            "logprobs":   torch.tensor(self.logprobs, dtype=torch.float32, device=device),
            "returns":    torch.tensor(self.returns,  dtype=torch.float32, device=device),
            "advantages": torch.tensor(adv_norm,      dtype=torch.float32, device=device),
        }

    def reset(self) -> None:
        """Clear the buffer so the next rollout starts fresh."""
        self.ptr = 0
        self.full = False
        self.obs[:] = 0
        self.actions[:] = 0
        self.logprobs[:] = 0
        self.rewards[:] = 0
        self.dones[:] = 0
        self.values[:] = 0
        self.advantages[:] = 0
        self.returns[:] = 0


# ---------------------------------------------------------------------------
# LR annealing
# ---------------------------------------------------------------------------

def anneal_lr(
    optimizer: torch.optim.Optimizer,
    initial_lr: float,
    current_step: int,
    total_steps: int,
) -> float:
    """Linearly decay the learning rate from ``initial_lr`` to 0.

    Call this once per rollout (before or after the PPO update).

    Args:
        optimizer:    The optimizer whose LR will be updated.
        initial_lr:   The LR at step 0 — same value passed to Adam at init.
        current_step: How many environment steps have been taken so far.
        total_steps:  The total number of steps the run will last.

    Returns:
        The new learning rate (useful for logging).

    Why anneal?
    -----------
    Early in training the policy changes a lot, so a larger LR is useful.
    Later, as it converges, a smaller LR avoids overshooting. Linear decay
    is the simplest schedule and works well in practice for PPO.
    """
    frac = 1.0 - current_step / total_steps
    new_lr = max(initial_lr * frac, 0.0)
    for group in optimizer.param_groups:
        group["lr"] = new_lr
    return new_lr




