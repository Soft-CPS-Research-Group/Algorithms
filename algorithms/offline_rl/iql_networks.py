"""IQL networks: MLP backbone, twin Q, value, Gaussian policy.

Self-contained — no imports from the trainer or dataset modules. Architecture
matches BC where applicable (hidden=[256, 256], dropout=0.1, ReLU hidden,
Adam-friendly init via PyTorch defaults).

Conventions
-----------
* All network forward methods take **standardised** observations (the
  ``ObservationStandardiser`` lives in the dataset module).
* Q networks take the raw dataset action (in ``[-1, 1]``) — no extra
  normalisation. Same convention as BC.
* The Gaussian policy outputs ``tanh(mean)`` deterministically and exposes a
  Gaussian ``log_prob`` over the pre-tanh "mean" head against a target action
  in ``[-1, 1]``. This is the standard IQL implementation shortcut: the AWR
  loss treats the policy as a diagonal Gaussian with mean=tanh(MLP(s)) and
  isotropic learned variance, scoring the dataset action directly. The
  tanh-Jacobian correction is omitted — it would shrink log-prob magnitudes
  near the boundary but does not change the gradient direction in the AWR
  setting (advantage-weighted MLE on bounded actions). This matches the
  reference IQL implementations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """ReLU MLP with optional dropout between hidden layers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (256, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim/out_dim must be positive: {in_dim}, {out_dim}")
        if not hidden:
            raise ValueError("hidden must be non-empty")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        sizes: List[int] = [int(in_dim), *(int(h) for h in hidden)]
        self.hidden_layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])
        )
        self.dropout = (
            nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        )
        self.output = nn.Linear(sizes[-1], int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.output(x)


# ---------------------------------------------------------------------------
# Q network
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Q(s, a) → scalar over concatenated (s, a)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (256, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.net = MLP(
            in_dim=self.obs_dim + self.action_dim,
            out_dim=1,
            hidden=hidden,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Value network
# ---------------------------------------------------------------------------


class ValueNetwork(nn.Module):
    """V(s) → scalar."""

    def __init__(
        self,
        obs_dim: int,
        hidden: Sequence[int] = (256, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.net = MLP(
            in_dim=self.obs_dim,
            out_dim=1,
            hidden=hidden,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Gaussian policy
# ---------------------------------------------------------------------------


_LOG_STD_INIT_DEFAULT: float = math.log(0.1)
_LOG_STD_MIN: float = -5.0   # σ ≈ 0.0067
_LOG_STD_MAX: float = 2.0    # σ ≈ 7.39


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy with tanh-squashed mean.

    * Mean head: MLP(s) → ``tanh(mean)`` ∈ [-1, 1]^A.
    * Log-σ: per-action learned ``nn.Parameter`` of shape ``(action_dim,)``,
      initialised at ``log(0.1)``. Not state-dependent (avoids stochasticity
      collapse during AWR; standard in IQL implementations).

    The ``log_prob(obs, action)`` method scores the dataset action under
    ``Normal(tanh(mean), σ)`` with no tanh-Jacobian correction (see module
    docstring).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (256, 256),
        dropout: float = 0.1,
        log_std_init: float = _LOG_STD_INIT_DEFAULT,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_layers: List[int] = [int(h) for h in hidden]
        self.dropout_p = float(dropout)
        self.mean_net = MLP(
            in_dim=self.obs_dim,
            out_dim=self.action_dim,
            hidden=hidden,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(log_std_init), dtype=torch.float32)
        )

    # --- forward primitives -------------------------------------------------

    def _mean(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean_net(obs))

    def _log_std_clamped(self) -> torch.Tensor:
        return self.log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (tanh_mean, log_std) — log_std is shared across the batch."""
        return self._mean(obs), self._log_std_clamped()

    def predict_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Inference path: ``tanh(mean)``. No sampling, no dropout effect."""
        return self._mean(obs)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Diag-Gaussian log-prob of ``action`` under N(tanh(mean), σ).

        Returns a scalar per row (shape ``(B,)``). Sums log-probs across
        action dims (independent diagonal Gaussian).
        """
        mean = self._mean(obs)
        log_std = self._log_std_clamped()
        # log N(a | mu, sigma) = -0.5 * ((a-mu)/sigma)^2 - log(sigma) - 0.5*log(2π)
        var = torch.exp(2.0 * log_std)
        diff = action - mean
        per_dim = -0.5 * (diff * diff) / var - log_std - 0.5 * math.log(2.0 * math.pi)
        return per_dim.sum(dim=-1)

    # --- metadata -----------------------------------------------------------

    def architecture_summary(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_layers": list(self.hidden_layers),
            "hidden_activation": "relu",
            "mean_activation": "tanh",
            "dropout": self.dropout_p,
            "log_std_init": float(self.log_std.detach().mean().item()),
        }
