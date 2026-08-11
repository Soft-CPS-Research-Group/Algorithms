"""
Community Coordinator — Phase 2 (Per-Building Price Signals).

Extends Phase 1 (CCLevel1Agent) from a single global price multiplier to a
vector of N per-building price multipliers, one per building.

    multipliers[i] ∈ [price_min, price_max]    (e.g. [0.5, 1.5])

The CC can now differentiate: e.g. "building 3 has a full battery and high
PV — give it a low multiplier to encourage discharge; building 11 is
importing heavily — give it a high multiplier to reduce consumption."

=============================================================================
DESIGN
=============================================================================
Observations:
    Uses the `cc_level2` entity-encoding profile.  Each building's encoded
    vector contains:
        • 16 legacy district features (time, price, carbon, community power),
          plus optional community electrical headroom
        • 6 per-building features: storage/PV/net power and EV service state

    The CC assembles a single context vector of shape
        (district_features + 6 * num_buildings,)
    taking district features from observations[0] (identical across all
    buildings) and per-building features from observations[i] for each i.

Action (continuous, N-dimensional vector):
    raw ~ Normal(mean(obs), std)               # PPO latent, shape (N,)
    multiplier[i] = price_min + (price_max - price_min) * (tanh(raw[i]) + 1) / 2

Policy network (CommunityMarketMakerNetV2):
    Shared encoder → two hidden layers →
        mean_head:   Linear(hidden, N)    # one mean per building
        critic_head: Linear(hidden, 1)    # community value estimate

Temporal abstraction:
    Same as Phase 1: CC decides every cc_action_interval env steps.

Reward:
    Same community-level reward (CCRewardLevel1 / CCRewardLevel2).
    The CC sums per-building rewards into one community scalar.

Training:
    PPO with a factorized diagonal Gaussian. Historical campaigns used one
    scalar community critic and broadcast the same advantage to all price
    factors. New campaigns can opt into ``member_decomposed`` credit: the
    centralized critic emits one value per building and each price factor is
    trained with its own settlement/service advantage. The shared encoder
    still sees the whole community, so coordination remains centralized.

Pipeline integration:
    predict() returns List[float] of length N.
    The Ensemble wrapper (pipeline.py) routes context[i] to building i, so
    SignalAwareRBC workers receive their individual multiplier unchanged.
    No changes to SignalAwareRBC are needed.
"""

from __future__ import annotations

import csv
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.optim import Adam

from algorithms.agents.base_agent import BaseAgent
from algorithms.constants import DEFAULT_ONNX_OPSET

# ── District features (identical to cc_level1) ───────────────────────────────
_CC_LEVEL2_DISTRICT_FEATURES = (
    "district__time_of_day_sin",
    "district__time_of_day_cos",
    "district__day_type_sin",
    "district__day_type_cos",
    "district__is_weekend",
    "district__month_sin",
    "district__month_cos",
    "district__electricity_pricing",
    "district__electricity_pricing_predicted_1",
    "district__electricity_pricing_predicted_2",
    "district__electricity_pricing_predicted_3",
    "district__carbon_intensity",
    "district__community_net_power_kw",
    "district__community_import_power_kw",
    "district__community_export_power_kw",
    "district__community_pv_power_kw",
)
_N_DISTRICT = len(_CC_LEVEL2_DISTRICT_FEATURES)  # 16
_CC_LEVEL2_HEADROOM_FEATURE = "district__community_building_headroom_kw"
_CC_LEVEL2_HISTORY_FEATURES = (
    "district__community_net_prev_1_kwh_step",
    "district__community_net_prev_3_mean_kwh_step",
)
_PRICE_FEATURE = "district__electricity_pricing"
_CAUSAL_ACTIVE_FEATURE = "causal__cheap_and_export_active"

# ── Per-building features (6 per building) ────────────────────────────────────
# Short patterns matched against encoded observation names at attach_environment time.
# Pattern rules:
#   "prefix::feature" → name must start with prefix:: AND end with ::feature
#   "feature"         → name must end with ::feature (or equal feature exactly)
# Buildings without chargers receive 0.0 for EV features (adapter zero-fills).
_CC_LEVEL2_BUILDING_FEATURES = (
    "storage::soc",                          # battery SoC [0, 1]
    "pv::generation_power_kw",               # local PV output [0, 1]
    "net_power_kw",                          # net consumption [-1, 1] (signed)
    "connected_state",                       # EV connected {0, 1}
    "connected_ev_soc_deficit",              # max(required-soc, 0) [0, 1]
    "connected_ev_departure_urgency_24h",    # 1 - hours_to_depart/24 [0, 1]
)
_N_BUILDING_FEATS = len(_CC_LEVEL2_BUILDING_FEATURES)  # 6


def _match_building_feature(encoded_name: str, pattern: str) -> bool:
    """True if encoded_name matches the short feature pattern.

    Handles two pattern forms:
      "prefix::feat"  → encoded name must start with "prefix::" and its last
                        "::" segment must equal "feat"
      "feat"          → encoded name must be "feat" exactly OR its last segment
                        must equal "feat" (catches charger::*::feat).
    """
    if "::" in encoded_name:
        tail = encoded_name.rsplit("::", 1)[1]
    else:
        tail = encoded_name
    if "::" in pattern:
        prefix, feat = pattern.split("::", 1)
        return tail == feat and encoded_name.startswith(f"{prefix}::")
    return tail == pattern or encoded_name == pattern


# ── Reward normaliser (Welford) ───────────────────────────────────────────────

class RunningMeanStd:
    """Online mean/variance estimator (Welford)."""

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, x: float) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (x - self._mean)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._n < 2:
            return 1.0
        return float(np.sqrt(max(self._M2 / self._n, 1e-12)))


# ── Policy network ────────────────────────────────────────────────────────────

class CommunityMarketMakerNetV2(nn.Module):
    """Shared encoder → N Gaussian means plus scalar or member values.

    All N means share the same encoder trunk, differing only in the final
    linear layer.  A single shared log_std parameter (per building) controls
    exploration independently of the input.
    """

    def __init__(
        self,
        c_dim: int,
        num_buildings: int,
        hidden_dims: List[int],
        initial_log_std: float = -2.5,
        value_dimension: int = 1,
        separate_value_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.num_buildings = num_buildings
        self.value_dimension = int(value_dimension)
        if self.value_dimension not in {1, self.num_buildings}:
            raise ValueError(
                "CCLevel2 value_dimension must be 1 or num_buildings"
            )
        layers: List[nn.Module] = []
        in_d = c_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        self.encoder     = nn.Sequential(*layers)
        self.separate_value_encoder = bool(separate_value_encoder)
        if self.separate_value_encoder:
            value_layers: List[nn.Module] = []
            value_in_d = c_dim
            for h in hidden_dims:
                value_layers += [nn.Linear(value_in_d, h), nn.Tanh()]
                value_in_d = h
            self.value_encoder: Optional[nn.Module] = nn.Sequential(*value_layers)
        else:
            self.value_encoder = None
        self.mean_head   = nn.Linear(in_d, num_buildings)
        self.critic_head = nn.Linear(in_d, self.value_dimension)
        # One learnable log_std per building, state-independent.
        self.log_std = nn.Parameter(
            torch.full((num_buildings,), float(initial_log_std))
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight,   gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def get_action_and_value(
        self,
        community: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for policy and value.

        Args:
            community: (batch, c_dim) encoded context.
            action:    (batch, N) pre-tanh raw actions, or None to sample.

        Returns:
            action:    (batch, N) raw samples (pre-tanh, stored in buffer)
            log_prob:  (batch, N) per-building corrected log-probabilities
            entropy:   (batch, N) per-building entropies
            value:     (batch,) or (batch, N) state value estimate
        """
        h     = self.encoder(community)                          # (batch, hidden)
        means = self.mean_head(h)                                # (batch, N)
        value_h = self.value_encoder(community) if self.value_encoder else h
        value = self.critic_head(value_h)
        if self.value_dimension == 1:
            value = value.squeeze(-1)                            # (batch,)
        # Keep enough room for genuinely conservative campaigns (e.g. -3.2)
        # while still preventing numerically degenerate or runaway variance.
        stds  = torch.exp(self.log_std.clamp(-6.0, 1.0)).unsqueeze(0).expand_as(means)
        dist  = torch.distributions.Normal(means, stds)
        if action is None:
            action = dist.sample()                               # (batch, N)
        # Keep the factor log-probabilities separate.  PPO clips each factor
        # independently with the same centralized community advantage.
        tanh_correction = torch.log(1.0 - torch.tanh(action) ** 2 + 1e-6)
        log_prob = dist.log_prob(action) - tanh_correction       # (batch, N)
        entropy  = dist.entropy()                                # (batch, N)
        return action, log_prob, entropy, value


class DeterministicVectorMultiplierPolicy(nn.Module):
    """Deployable CC-L2 policy including the price-range mapping."""

    def __init__(
        self,
        policy: CommunityMarketMakerNetV2,
        price_min: float,
        price_max: float,
        reference_multipliers: np.ndarray,
        policy_residual_scale: float,
        policy_parameterization: str = "absolute_blend",
        causal_initial_multipliers: Optional[np.ndarray] = None,
        causal_residual_scale: Optional[float] = None,
        causal_active_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = policy.encoder
        self.mean_head = policy.mean_head
        self.register_buffer("price_min", torch.tensor(float(price_min)))
        self.register_buffer("price_span", torch.tensor(float(price_max - price_min)))
        self.register_buffer(
            "reference_multipliers",
            torch.tensor(reference_multipliers, dtype=torch.float32),
        )
        self.register_buffer(
            "policy_residual_scale",
            torch.tensor(float(policy_residual_scale)),
        )
        self.policy_parameterization = str(policy_parameterization)
        self.causal_residual_scale = causal_residual_scale
        self.causal_active_index = causal_active_index
        self.register_buffer(
            "causal_initial_multipliers",
            torch.tensor(
                (
                    causal_initial_multipliers
                    if causal_initial_multipliers is not None
                    else reference_multipliers
                ),
                dtype=torch.float32,
            ),
        )

    def forward(self, community: torch.Tensor) -> torch.Tensor:
        raw = self.mean_head(self.encoder(community))
        if self.policy_parameterization == "causal_active_only":
            # Restrict learned Level-2 prices to the causal region established
            # by the deployable Level-1 controller. Outside a currently cheap
            # and exporting interval the exact neutral vector is emitted.
            if self.causal_active_index is not None:
                active = (
                    community[:, int(self.causal_active_index)] > 0.5
                ).unsqueeze(-1)
            else:
                price = community[:, 7]
                forecasts = community[:, 8:11]
                forecast_mean = forecasts.mean(dim=1)
                forecast_min = forecasts.min(dim=1).values
                forecast_max = forecasts.max(dim=1).values
                spread = torch.maximum(
                    forecast_max - forecast_min,
                    torch.maximum(
                        forecast_mean.abs() * 0.05,
                        torch.full_like(forecast_mean, 1.0e-9),
                    ),
                )
                cheap = torch.logical_or(
                    price <= forecast_mean - 0.20 * spread,
                    price <= forecast_min + 0.10 * spread,
                )
                exporting = community[:, 14] > 1.0e-9
                active = torch.logical_and(cheap, exporting).unsqueeze(-1)
            if self.causal_residual_scale is None:
                active_multiplier = self.price_min + self.price_span * (
                    torch.tanh(raw) + 1.0
                ) / 2.0
            else:
                unit = torch.tanh(raw)
                price_max = self.price_min + self.price_span
                upward_distance = price_max - self.causal_initial_multipliers
                downward_distance = self.causal_initial_multipliers - self.price_min
                distance = torch.where(unit >= 0.0, upward_distance, downward_distance)
                active_multiplier = self.causal_initial_multipliers + (
                    float(self.causal_residual_scale) * distance * unit
                )
            return torch.where(
                active,
                active_multiplier,
                torch.ones_like(active_multiplier),
            )
        if self.policy_parameterization == "centered_residual":
            unit = torch.tanh(raw)
            upward_distance = self.price_min + self.price_span - self.reference_multipliers
            downward_distance = self.reference_multipliers - self.price_min
            distance = torch.where(unit >= 0.0, upward_distance, downward_distance)
            output = self.reference_multipliers + (
                self.policy_residual_scale * distance * unit
            )
            return torch.minimum(
                torch.maximum(output, self.price_min),
                self.price_min + self.price_span,
            )
        full = self.price_min + self.price_span * (torch.tanh(raw) + 1.0) / 2.0
        return self.reference_multipliers + self.policy_residual_scale * (
            full - self.reference_multipliers
        )


# ── Rollout buffer (N-dim actions) ───────────────────────────────────────────

class RolloutBufferV2:
    """Fixed-size rollout buffer for continuous N-dimensional PPO."""

    def __init__(
        self,
        num_steps: int,
        c_dim: int,
        num_buildings: int,
        *,
        member_credit: bool = False,
    ) -> None:
        self.num_steps    = num_steps
        self.num_buildings = num_buildings
        self.member_credit = bool(member_credit)
        value_shape = (num_steps, num_buildings) if self.member_credit else (num_steps,)
        self._ptr  = 0
        self.full  = False
        self.communities = np.zeros((num_steps, c_dim),           dtype=np.float32)
        self.actions     = np.zeros((num_steps, num_buildings),   dtype=np.float32)
        self.logprobs    = np.zeros((num_steps, num_buildings),  dtype=np.float32)
        self.actor_masks = np.ones((num_steps, num_buildings),   dtype=np.float32)
        self.rewards     = np.zeros(value_shape,                  dtype=np.float32)
        self.dones       = np.zeros(num_steps,                    dtype=np.float32)
        self.values      = np.zeros(value_shape,                  dtype=np.float32)
        self.returns     = np.zeros(value_shape,                  dtype=np.float32)
        self.advantages  = np.zeros(value_shape,                  dtype=np.float32)

    def add(
        self,
        community,
        action,
        logprob,
        reward,
        done,
        value,
        actor_mask=None,
    ) -> None:
        self.communities[self._ptr] = community
        self.actions[self._ptr]     = action          # (N,)
        self.logprobs[self._ptr]    = logprob         # (N,)
        self.actor_masks[self._ptr] = (
            1.0 if actor_mask is None else actor_mask
        )
        self.rewards[self._ptr]     = reward
        self.dones[self._ptr]       = float(done)
        self.values[self._ptr]      = value
        self._ptr += 1
        if self._ptr >= self.num_steps:
            self.full = True

    def compute_gae(self, last_value, last_done, gamma, gae_lambda) -> None:
        gae = np.zeros(self.num_buildings, dtype=np.float32) if self.member_credit else 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_nt, next_value = 1.0 - float(last_done), last_value
            else:
                next_nt, next_value = 1.0 - self.dones[t], self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nt - self.values[t]
            gae = delta + gamma * gae_lambda * next_nt * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get(self) -> dict:
        return {
            "community":  torch.tensor(self.communities, dtype=torch.float32),
            "actions":    torch.tensor(self.actions,     dtype=torch.float32),
            "logprobs":   torch.tensor(self.logprobs,    dtype=torch.float32),
            "actor_masks": torch.tensor(self.actor_masks, dtype=torch.float32),
            "returns":    torch.tensor(self.returns,     dtype=torch.float32),
            "advantages": torch.tensor(self.advantages,  dtype=torch.float32),
        }

    def reset(self) -> None:
        self._ptr = 0
        self.full = False


# ── Agent ─────────────────────────────────────────────────────────────────────

class CCLevel2Agent(BaseAgent):
    """Phase-2 Community Coordinator: emits one price multiplier per building."""

    _use_raw_observations: bool = False
    observation_encoding_profile: str = "cc_level2"

    # ──────────────────────────── Construction ──────────────────────────────

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.use_raw_observations = False

        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})

        # PPO hyperparameters
        self._gamma           = float(hyper.get("gamma",         0.99))
        self._gae_lambda      = float(hyper.get("gae_lambda",    0.95))
        self._num_epochs      = int(hyper.get("num_epochs",      10))
        self._mini_batch_size = int(hyper.get("mini_batch_size", 64))
        self._clip_coef       = float(hyper.get("clip_coef",     0.2))
        self._vf_coef         = float(hyper.get("vf_coef",       1.0))
        self._ent_coef        = float(hyper.get("ent_coef",      0.05))
        self._max_grad_norm   = float(hyper.get("max_grad_norm", 0.5))
        self._target_kl       = hyper.get("target_kl",           0.1)

        self._credit_assignment = str(
            hyper.get("credit_assignment", "global")
        ).strip().lower()
        if self._credit_assignment not in {"global", "member_decomposed"}:
            raise ValueError(
                "CCLevel2 credit_assignment must be 'global' or "
                "'member_decomposed'"
            )
        self._member_credit = self._credit_assignment == "member_decomposed"
        self._reward_normalization = str(
            hyper.get("reward_normalization", "running_zscore")
        ).strip().lower()
        if self._reward_normalization not in {"running_zscore", "none"}:
            raise ValueError(
                "CCLevel2 reward_normalization must be 'running_zscore' or 'none'"
            )
        self._team_reward_mix = float(hyper.get("team_reward_mix", 0.0))
        if not 0.0 <= self._team_reward_mix <= 1.0:
            raise ValueError("CCLevel2 team_reward_mix must be within [0, 1]")

        # Price-multiplier bounds
        self._price_min = float(hyper.get("price_min", 0.5))
        self._price_max = float(hyper.get("price_max", 1.5))
        if self._price_max <= self._price_min:
            raise ValueError("CCLevel2 price_max must be greater than price_min")

        # Auxiliary reward weights
        self._w_factor     = float(hyper.get("w_factor",     0.3))
        self._w_smoothness = float(hyper.get("w_smoothness", 0.02))
        self._prev_multipliers: Optional[np.ndarray] = None  # (N,), init at first step

        # Community size — must be set before rollout buffer is created.
        self._num_buildings = int(hyper.get("num_buildings", 17))
        reference = hyper.get("reference_multipliers")
        if reference is None:
            reference = [1.0] * self._num_buildings
        if len(reference) != self._num_buildings:
            raise ValueError(
                "CCLevel2 reference_multipliers length must equal num_buildings"
            )
        self._reference_multipliers = np.asarray(reference, dtype=np.float32)
        if not np.all(np.isfinite(self._reference_multipliers)):
            raise ValueError("CCLevel2 reference_multipliers must be finite")
        if np.any(self._reference_multipliers < self._price_min) or np.any(
            self._reference_multipliers > self._price_max
        ):
            raise ValueError(
                "CCLevel2 reference_multipliers must lie within the configured price range"
            )
        self._policy_residual_scale = float(hyper.get("policy_residual_scale", 1.0))
        if not 0.0 <= self._policy_residual_scale <= 1.0:
            raise ValueError("CCLevel2 policy_residual_scale must be within [0, 1]")
        self._policy_parameterization = str(
            hyper.get("policy_parameterization", "absolute_blend")
        ).strip().lower()
        if self._policy_parameterization not in {
            "absolute_blend",
            "centered_residual",
            "causal_active_only",
        }:
            raise ValueError(
                "CCLevel2 policy_parameterization must be 'absolute_blend' or "
                "'centered_residual' or 'causal_active_only'"
            )
        self._causal_use_physical_context = bool(
            hyper.get("causal_use_physical_context", False)
        )
        self._causal_initial_multiplier = float(
            hyper.get("causal_initial_multiplier", 0.90)
        )
        causal_initial = hyper.get("causal_initial_multipliers")
        if causal_initial is None:
            causal_initial = [
                self._causal_initial_multiplier
            ] * self._num_buildings
        if len(causal_initial) != self._num_buildings:
            raise ValueError(
                "CCLevel2 causal_initial_multipliers length must equal "
                "num_buildings"
            )
        self._causal_initial_multipliers = np.asarray(
            causal_initial, dtype=np.float32
        )
        raw_causal_residual_scale = hyper.get("causal_residual_scale")
        self._causal_residual_scale: Optional[float] = (
            None
            if raw_causal_residual_scale is None
            else float(raw_causal_residual_scale)
        )
        if self._causal_residual_scale is not None and not (
            0.0 <= self._causal_residual_scale <= 1.0
        ):
            raise ValueError("CCLevel2 causal_residual_scale must be within [0, 1]")
        if not np.all(np.isfinite(self._causal_initial_multipliers)):
            raise ValueError("CCLevel2 causal_initial_multipliers must be finite")
        if self._policy_parameterization == "causal_active_only":
            if self._price_max > 1.0 + 1.0e-9:
                raise ValueError(
                    "CCLevel2 causal_active_only requires price_max <= 1.0"
                )
            if not self._price_min <= self._causal_initial_multiplier <= self._price_max:
                raise ValueError(
                    "CCLevel2 causal_initial_multiplier must lie within the price range"
                )
            if np.any(self._causal_initial_multipliers < self._price_min) or np.any(
                self._causal_initial_multipliers > self._price_max
            ):
                raise ValueError(
                    "CCLevel2 causal_initial_multipliers must lie within the price range"
                )

        # Keep the historical 118-wide contract by default. New campaigns may
        # opt into physical headroom and causal community-history features.
        self._include_community_headroom = bool(
            hyper.get("include_community_headroom", False)
        )
        self._include_community_history = bool(
            hyper.get("include_community_history", False)
        )
        self._district_feature_names = _CC_LEVEL2_DISTRICT_FEATURES + (
            (_CC_LEVEL2_HEADROOM_FEATURE,)
            if self._include_community_headroom
            else ()
        ) + (
            _CC_LEVEL2_HISTORY_FEATURES
            if self._include_community_history
            else ()
        )
        self._n_district = len(self._district_feature_names)
        self._causal_context_feature_names = (
            (_CAUSAL_ACTIVE_FEATURE,)
            if (
                self._policy_parameterization == "causal_active_only"
                and self._causal_use_physical_context
            )
            else ()
        )
        self._n_causal_context = len(self._causal_context_feature_names)
        self._building_context_start = self._n_district + self._n_causal_context
        self._n_building_feats = _N_BUILDING_FEATS
        default_c_dim = (
            self._building_context_start
            + _N_BUILDING_FEATS * self._num_buildings
        )
        self._c_dim = int(hyper.get("c_dim", default_c_dim))
        if self._c_dim != default_c_dim:
            raise ValueError(
                "CCLevel2 c_dim does not match the selected observation contract: "
                f"expected {default_c_dim}, got {self._c_dim}"
            )

        # Per-building feature positions in encoded obs — populated at attach_environment.
        self._district_positions: List[int] = []
        self._building_feat_positions: List[List[int]] = []

        self._hidden_dims = list(hyper.get("hidden_dims", [256, 256]))
        self._lr = float(hyper.get("lr", 1e-4))
        self._initial_log_std = float(hyper.get("initial_log_std", -2.5))
        self._separate_value_encoder = bool(
            hyper.get("separate_value_encoder", False)
        )
        self.policy = CommunityMarketMakerNetV2(
            self._c_dim,
            self._num_buildings,
            self._hidden_dims,
            initial_log_std=self._initial_log_std,
            value_dimension=self._num_buildings if self._member_credit else 1,
            separate_value_encoder=self._separate_value_encoder,
        )
        self._initialize_policy_at_reference()
        self.ppo_optim = Adam(self.policy.parameters(), lr=self._lr)

        self._reward_rms = RunningMeanStd()
        self._member_reward_rms = [
            RunningMeanStd() for _ in range(self._num_buildings)
        ]

        self.rollout_buffer = RolloutBufferV2(
            int(hyper.get("num_steps", 96)),
            self._c_dim,
            self._num_buildings,
            member_credit=self._member_credit,
        )
        self._ppo_update_count = 0

        # Temporal abstraction
        self._cc_action_interval = int(hyper.get("cc_action_interval", 4))
        self._step_in_interval = 0
        self._decision_interval_complete = False
        self._episode_step_context: Optional[int] = None

        # Cached decision (arrays instead of scalars)
        self._cached_multipliers: np.ndarray = self._reference_multipliers.copy()
        initial_action_multipliers = (
            self._causal_initial_multipliers
            if self._policy_parameterization == "causal_active_only"
            else self._reference_multipliers
        )
        self._cached_action: np.ndarray = self._multipliers_to_raw(
            initial_action_multipliers
        )
        self._cached_community:   Optional[np.ndarray] = None
        self._cached_logprob:     np.ndarray = np.zeros(
            self._num_buildings, dtype=np.float32
        )
        self._cached_value: float | np.ndarray = (
            np.zeros(self._num_buildings, dtype=np.float32)
            if self._member_credit
            else 0.0
        )
        self._cached_policy_sample: bool = False
        self._cached_actor_mask = np.ones(
            self._num_buildings, dtype=np.float32
        )
        self._accumulated_reward: float | np.ndarray = (
            np.zeros(self._num_buildings, dtype=np.float64)
            if self._member_credit
            else 0.0
        )

        # BC warm-start
        self._bc_enabled       = bool(hyper.get("bc_pretrain_enabled", False))
        if self._policy_parameterization == "causal_active_only" and self._bc_enabled:
            raise ValueError(
                "CCLevel2 causal_active_only starts from its causal incumbent "
                "and does not support BC pretraining"
            )
        self._bc_use_physical_teacher_context = bool(
            hyper.get("bc_use_physical_teacher_context", False)
        )
        self.requires_raw_observation_context = (
            (self._bc_enabled and self._bc_use_physical_teacher_context)
            or (
                self._policy_parameterization == "causal_active_only"
                and self._causal_use_physical_context
            )
        )
        self._bc_collect_steps = int(hyper.get("bc_collect_steps", 336))
        self._bc_train_steps   = int(hyper.get("bc_train_steps",   2000))
        self._bc_lr            = float(hyper.get("bc_lr",           1e-3))
        self._bc_train_chunk_steps = int(
            hyper.get("bc_train_chunk_steps", 256)
        )
        self._bc_max_torch_threads = int(
            hyper.get("bc_max_torch_threads", 1)
        )
        self._bc_progress_interval = int(
            hyper.get(
                "bc_progress_interval",
                max(self._bc_train_steps // 8, 1),
            )
        )
        if self._bc_train_chunk_steps <= 0:
            raise ValueError("CCLevel2 bc_train_chunk_steps must be positive")
        if self._bc_max_torch_threads <= 0:
            raise ValueError("CCLevel2 bc_max_torch_threads must be positive")
        if self._bc_progress_interval <= 0:
            raise ValueError("CCLevel2 bc_progress_interval must be positive")
        self._bc_pretrain_done: bool = not self._bc_enabled
        self._bc_contexts:  List[np.ndarray] = []
        self._bc_teacher_contexts: List[np.ndarray] = []
        # targets: shape (N_steps, num_buildings)
        self._bc_targets:   List[np.ndarray] = []
        # Community-level reference values (auto-calibrated from episode-0 data)
        self._bc_dt_hours         = float(hyper.get("bc_dt_hours", 0.25))
        self._bc_target_import    = hyper.get("bc_target_import",    None)
        self._bc_reference_peak   = hyper.get("bc_reference_peak",   None)
        self._bc_reference_export = hyper.get("bc_reference_export", None)
        self._bc_import_samples: List[float] = []
        self._bc_export_samples: List[float] = []
        self._latest_raw_observations: Optional[List[np.ndarray]] = None
        self._latest_raw_next_observations: Optional[List[np.ndarray]] = None
        self._raw_obs_indices: List[Dict[str, int]] = []
        # Weights mirroring CCRewardLevel1
        self._bc_w_cost   = float(hyper.get("bc_w_cost",   1.0))
        self._bc_w_peak   = float(hyper.get("bc_w_peak",   0.3))
        self._bc_w_export = float(hyper.get("bc_w_export", 0.1))
        self._bc_mult_scale = float(hyper.get("bc_mult_scale", 1.0))
        self._bc_teacher_mode = str(
            hyper.get("bc_teacher_mode", "continuous_score")
        ).strip().lower()
        if self._bc_teacher_mode not in {
            "continuous_score",
            "cheap_and_export",
        }:
            raise ValueError(
                "CCLevel2 bc_teacher_mode must be 'continuous_score' or "
                "'cheap_and_export'"
            )
        self._bc_discount_multiplier = float(
            hyper.get("bc_discount_multiplier", 0.90)
        )
        if not self._price_min <= self._bc_discount_multiplier <= 1.0:
            raise ValueError(
                "CCLevel2 bc_discount_multiplier must lie between price_min and 1"
            )
        self._bc_export_activation_kw = max(
            float(hyper.get("bc_export_activation_kw", 1.0e-9)),
            0.0,
        )
        # Per-building modulation weights for BC teacher (mirror CCRewardLevel2)
        self._bc_w_soc    = float(hyper.get("bc_w_soc",  0.2))
        self._bc_w_net    = float(hyper.get("bc_w_net",  0.1))   # legacy; TODO: remove after EV redesign
        self._bc_w_ev     = float(hyper.get("bc_w_ev",   0.5))   # mirrors CCRewardLevel2 w_ev
        # Must match CCRewardLevel2.urgency_horizon so BC and reward use the same urgency scale.
        # The encoded obs feature connected_ev_departure_urgency_24h uses a fixed 24h horizon;
        # we invert it to recover actual hours and re-apply this horizon.
        self._bc_urgency_horizon = float(hyper.get("bc_urgency_horizon", 4.0))

        # Incremental BC state.  Running all BC optimizer updates inside one
        # predict() call made remote jobs look alive while blocking environment
        # progress for tens of minutes.  Keep the prepared tensors between
        # decisions and execute only a bounded chunk at a time.
        self._bc_train_inputs: Optional[torch.Tensor] = None
        self._bc_train_targets: Optional[torch.Tensor] = None
        self._bc_train_optimizer: Optional[Adam] = None
        self._bc_train_step = 0
        self._bc_train_losses: List[float] = []
        self._bc_train_started_at: Optional[float] = None

        # Obs layout (set in attach_environment)
        self._obs_index: Dict[str, int] = {}  # feature name → index in obs

        # Diagnostics
        self._episode_count = 0
        self._global_cc_step = 0
        self._decision_trace: List[dict] = []
        self._completed_decision_traces: List[dict] = []

    def _multipliers_to_raw(self, multipliers: np.ndarray) -> np.ndarray:
        values = np.asarray(multipliers, dtype=np.float64)
        if self._policy_parameterization == "causal_active_only":
            if self._causal_residual_scale is None:
                unit = (
                    (values - self._price_min)
                    / (self._price_max - self._price_min)
                    * 2.0
                    - 1.0
                )
            else:
                reference = self._causal_initial_multipliers.astype(np.float64)
                delta = values - reference
                distance = np.where(
                    delta >= 0.0,
                    self._price_max - reference,
                    reference - self._price_min,
                )
                denominator = self._causal_residual_scale * distance
                unit = np.divide(
                    delta,
                    denominator,
                    out=np.zeros_like(delta, dtype=np.float64),
                    where=denominator > 1.0e-12,
                )
            return np.arctanh(np.clip(unit, -0.999, 0.999)).astype(np.float32)
        reference = self._reference_multipliers.astype(np.float64)
        if self._policy_parameterization == "centered_residual":
            delta = values - reference
            distance = np.where(
                delta >= 0.0,
                self._price_max - reference,
                reference - self._price_min,
            )
            denominator = self._policy_residual_scale * distance
            unit = np.divide(
                delta,
                denominator,
                out=np.zeros_like(delta, dtype=np.float64),
                where=denominator > 1.0e-12,
            )
        else:
            if self._policy_residual_scale <= 1.0e-12:
                full = reference
            else:
                full = reference + (
                    (values - reference) / self._policy_residual_scale
                )
            unit = (
                (full - self._price_min)
                / (self._price_max - self._price_min)
                * 2.0
                - 1.0
            )
        return np.arctanh(np.clip(unit, -0.999, 0.999)).astype(np.float32)

    def _initialize_policy_at_reference(self) -> None:
        """Start deterministic inference at a measured safe/reference signal."""
        if self._policy_parameterization == "centered_residual":
            raw_reference = np.zeros_like(
                self._reference_multipliers, dtype=np.float32
            )
        elif self._policy_parameterization == "causal_active_only":
            raw_reference = self._multipliers_to_raw(
                self._causal_initial_multipliers
            )
        else:
            raw_reference = self._multipliers_to_raw(
                self._reference_multipliers
            )
        with torch.no_grad():
            self.policy.mean_head.weight.zero_()
            self.policy.mean_head.bias.copy_(torch.from_numpy(raw_reference))

    def _causal_intervention_active(self, context: np.ndarray) -> bool:
        if self._causal_use_physical_context:
            return bool(float(context[self._n_district]) > 0.5)
        index = self._district_feature_names.index
        price = float(context[index("district__electricity_pricing")])
        forecasts = np.asarray(
            [
                context[
                    index(f"district__electricity_pricing_predicted_{horizon}")
                ]
                for horizon in (1, 2, 3)
            ],
            dtype=np.float64,
        )
        forecast_mean = float(forecasts.mean())
        forecast_min = float(forecasts.min())
        forecast_max = float(forecasts.max())
        spread = max(
            forecast_max - forecast_min,
            abs(forecast_mean) * 0.05,
            1.0e-9,
        )
        cheap = (
            price <= forecast_mean - 0.20 * spread
            or price <= forecast_min + 0.10 * spread
        )
        exporting = (
            float(
                context[index("district__community_export_power_kw")]
            )
            > 1.0e-9
        )
        return bool(cheap and exporting)

    def _physical_causal_intervention_active(
        self,
        raw_observations: Optional[List[np.ndarray]] = None,
    ) -> bool:
        physical_observations = (
            self._latest_raw_observations
            if raw_observations is None
            else raw_observations
        )
        if not physical_observations or not self._raw_obs_indices:
            raise RuntimeError(
                "CCLevel2 physical causal gate did not receive raw observation context"
            )
        raw = np.asarray(physical_observations[0], dtype=np.float64)
        raw_index = self._raw_obs_indices[0]

        def value(name: str) -> float:
            position = raw_index.get(name)
            if position is None or position >= len(raw):
                raise RuntimeError(
                    f"CCLevel2 physical causal feature is unavailable: {name}"
                )
            parsed = float(raw[position])
            if not np.isfinite(parsed):
                raise RuntimeError(
                    f"CCLevel2 physical causal feature is non-finite: {name}"
                )
            return parsed

        price = value("district__electricity_pricing")
        forecasts = np.asarray(
            [
                value(f"district__electricity_pricing_predicted_{horizon}")
                for horizon in (1, 2, 3)
            ],
            dtype=np.float64,
        )
        forecast_mean = float(forecasts.mean())
        forecast_min = float(forecasts.min())
        forecast_max = float(forecasts.max())
        spread = max(
            forecast_max - forecast_min,
            abs(forecast_mean) * 0.05,
            1.0e-9,
        )
        cheap = (
            price <= forecast_mean - 0.20 * spread
            or price <= forecast_min + 0.10 * spread
        )
        exporting = value("district__community_export_power_kw") > 1.0e-9
        return bool(cheap and exporting)

    def _raw_to_multipliers(
        self,
        raw: np.ndarray,
        *,
        context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._policy_parameterization == "causal_active_only":
            if context is None:
                raise ValueError(
                    "CCLevel2 causal_active_only mapping requires policy context"
                )
            if not self._causal_intervention_active(context):
                return np.ones(self._num_buildings, dtype=np.float32)
            if self._causal_residual_scale is not None:
                unit = np.tanh(raw)
                reference = self._causal_initial_multipliers
                distance = np.where(
                    unit >= 0.0,
                    self._price_max - reference,
                    reference - self._price_min,
                )
                return np.clip(
                    reference
                    + self._causal_residual_scale * distance * unit,
                    self._price_min,
                    self._price_max,
                )
            return self._price_min + (
                (self._price_max - self._price_min)
                * (np.tanh(raw) + 1.0)
                / 2.0
            )
        if self._policy_parameterization == "centered_residual":
            unit = np.tanh(raw)
            reference = self._reference_multipliers
            distance = np.where(
                unit >= 0.0,
                self._price_max - reference,
                reference - self._price_min,
            )
            output = reference + self._policy_residual_scale * distance * unit
            return np.clip(output, self._price_min, self._price_max)
        full = self._price_min + (
            (self._price_max - self._price_min) * (np.tanh(raw) + 1.0) / 2.0
        )
        return self._reference_multipliers + self._policy_residual_scale * (
            full - self._reference_multipliers
        )

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Keep a flat name→index map from building 0 for any legacy BC lookups.
        self._obs_index = {n: i for i, n in enumerate(observation_names[0])}

        # --- District feature positions (same names across all buildings) ------
        obs0_idx = self._obs_index
        self._district_positions = [
            obs0_idx.get(name, -1) for name in self._district_feature_names
        ]
        missing_district = [
            name
            for name, position in zip(
                self._district_feature_names,
                self._district_positions,
            )
            if position < 0
        ]
        if missing_district:
            raise ValueError(
                "CCLevel2 required district features are missing from the selected "
                f"observation profile: {missing_district}"
            )

        raw_observation_names = (
            (metadata or {}).get("raw_observation_names")
            if isinstance(metadata, dict)
            else None
        )
        self._raw_obs_indices = []
        if isinstance(raw_observation_names, list):
            self._raw_obs_indices = [
                {str(name): index for index, name in enumerate(names)}
                for names in raw_observation_names
            ]
        if self.requires_raw_observation_context:
            required_physical = {
                "district__electricity_pricing",
                "district__electricity_pricing_predicted_1",
                "district__electricity_pricing_predicted_2",
                "district__electricity_pricing_predicted_3",
                "district__community_export_power_kw",
            }
            if self._bc_enabled and self._bc_use_physical_teacher_context:
                required_physical.add("district__community_import_power_kw")
            available = set(self._raw_obs_indices[0]) if self._raw_obs_indices else set()
            missing_physical = sorted(required_physical - available)
            if missing_physical:
                raise ValueError(
                    "CCLevel2 physical causal/BC context requires raw district features: "
                    f"{missing_physical}"
                )

        # --- Per-building feature positions (pattern-matched per building) -----
        # Each building's encoded obs has different qualified IDs, e.g.
        #   storage::Building1/electrical_storage::soc
        # We match by short pattern using _match_building_feature.
        self._building_feat_positions = []
        for names_i in observation_names:
            positions: List[int] = []
            for pattern in _CC_LEVEL2_BUILDING_FEATURES:
                pos = -1
                for j, n in enumerate(names_i):
                    if _match_building_feature(n, pattern):
                        pos = j
                        break
                positions.append(pos)
            self._building_feat_positions.append(positions)

        # Validate: warn if expected per-building features are missing.
        missing = {
            pat for pat in _CC_LEVEL2_BUILDING_FEATURES
            if all(pos == -1 for pos in
                   [self._building_feat_positions[i][k]
                    for i, (k, p) in enumerate(
                        [(list(_CC_LEVEL2_BUILDING_FEATURES).index(pat), pat)]
                        * len(self._building_feat_positions)
                    )])
        }
        if missing:
            logger.warning(
                "CCLevel2: per-building features not found in obs: {}. "
                "These will be zero-filled.", missing,
            )

        # Update num_buildings from the actual environment if not overridden.
        if len(observation_names) != self._num_buildings:
            logger.warning(
                "CCLevel2: config num_buildings={} but env has {} buildings; "
                "updating to match environment.",
                self._num_buildings, len(observation_names),
            )
            self._num_buildings = len(observation_names)
            # Rebuild policy and buffer with corrected size.
            self._rebuild_for_num_buildings()

    def set_observation_context(
        self,
        *,
        raw_observations: Optional[List[np.ndarray]] = None,
        encoded_observations: Optional[List[np.ndarray]] = None,
    ) -> None:
        _ = encoded_observations
        self._latest_raw_observations = (
            [
                np.asarray(observation, dtype=np.float64)
                for observation in raw_observations
            ]
            if raw_observations is not None
            else None
        )

    def set_transition_context(
        self,
        *,
        raw_observations: Optional[List[np.ndarray]] = None,
        raw_next_observations: Optional[List[np.ndarray]] = None,
        encoded_observations: Optional[List[np.ndarray]] = None,
        encoded_next_observations: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Keep physical current/next context aligned for PPO bootstrapping."""
        _ = encoded_observations, encoded_next_observations
        if raw_observations is not None:
            self._latest_raw_observations = [
                np.asarray(observation, dtype=np.float64)
                for observation in raw_observations
            ]
        self._latest_raw_next_observations = (
            [
                np.asarray(observation, dtype=np.float64)
                for observation in raw_next_observations
            ]
            if raw_next_observations is not None
            else None
        )

    def _rebuild_for_num_buildings(self) -> None:
        """Reconstruct policy and buffer when num_buildings changes at env attach."""
        self._c_dim = (
            self._building_context_start
            + _N_BUILDING_FEATS * self._num_buildings
        )
        self.policy = CommunityMarketMakerNetV2(
            self._c_dim,
            self._num_buildings,
            self._hidden_dims,
            initial_log_std=self._initial_log_std,
            value_dimension=self._num_buildings if self._member_credit else 1,
            separate_value_encoder=self._separate_value_encoder,
        )
        if len(self._reference_multipliers) != self._num_buildings:
            raise ValueError(
                "CCLevel2 environment building count does not match reference_multipliers"
            )
        if len(self._causal_initial_multipliers) != self._num_buildings:
            raise ValueError(
                "CCLevel2 environment building count does not match "
                "causal_initial_multipliers"
            )
        self._initialize_policy_at_reference()
        self.ppo_optim = Adam(self.policy.parameters(), lr=self._lr)
        self.rollout_buffer = RolloutBufferV2(
            self.rollout_buffer.num_steps,
            self._c_dim,
            self._num_buildings,
            member_credit=self._member_credit,
        )
        self._cached_multipliers = self._reference_multipliers.copy()
        rebuild_initial = (
            self._causal_initial_multipliers
            if self._policy_parameterization == "causal_active_only"
            else self._reference_multipliers
        )
        self._cached_action = self._multipliers_to_raw(rebuild_initial)
        self._cached_logprob     = np.zeros(
            self._num_buildings, dtype=np.float32
        )
        self._cached_actor_mask = np.ones(
            self._num_buildings, dtype=np.float32
        )
        self._member_reward_rms = [
            RunningMeanStd() for _ in range(self._num_buildings)
        ]
        self._cached_policy_sample = False

    # ───────────────────────── Per-step interaction ──────────────────────────

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[float]:
        """Return list of N price multipliers (one per building)."""
        if self._step_in_interval == 0:
            if not self._bc_pretrain_done:
                ctx = self._build_context(observations)
                teacher_ctx = self._build_teacher_context(ctx)
                # Compute teacher targets per building.
                teacher_targets = self._bc_teacher_multipliers_per_building(
                    teacher_ctx,
                    observations,
                )
                # Use per-building teacher as cached output.
                self._cached_action = self._multipliers_to_raw(teacher_targets)
                self._cached_multipliers = self._raw_to_multipliers(
                    self._cached_action
                ).astype(np.float32)
                self._cached_community   = ctx
                self._cached_logprob     = np.zeros(
                    self._num_buildings, dtype=np.float32
                )
                self._cached_actor_mask = np.zeros(
                    self._num_buildings, dtype=np.float32
                )
                self._cached_value = (
                    np.zeros(self._num_buildings, dtype=np.float32)
                    if self._member_credit
                    else 0.0
                )
                # A BC-teacher decision has no policy log-probability.  Even
                # when this same decision completes BC collection, it must not
                # enter the PPO rollout as an on-policy transition.
                self._cached_policy_sample = False
                if self._bc_train_inputs is None:
                    self._bc_contexts.append(ctx.copy())
                    self._bc_teacher_contexts.append(teacher_ctx.copy())
                    self._bc_targets.append(teacher_targets.copy())
                    # Accumulate community import/export for BC calibration.
                    _idx = self._district_feature_names.index
                    dt = self._bc_dt_hours
                    self._bc_import_samples.append(
                        float(
                            teacher_ctx[
                                _idx("district__community_import_power_kw")
                            ]
                        )
                        * dt
                    )
                    self._bc_export_samples.append(
                        float(
                            teacher_ctx[
                                _idx("district__community_export_power_kw")
                            ]
                        )
                        * dt
                    )
                    if len(self._bc_contexts) >= self._bc_collect_steps:
                        self._prepare_bc_pretraining()
                if self._bc_train_inputs is not None:
                    self._run_bc_pretraining_chunk()
            else:
                self._sample_new_decision(observations, deterministic)
        self._step_in_interval = (
            self._step_in_interval + 1
        ) % self._cc_action_interval
        self._decision_interval_complete = self._step_in_interval == 0
        return self._cached_multipliers.tolist()

    def set_episode_context(
        self,
        *,
        episode_step: Optional[int] = None,
        next_episode_step: Optional[int] = None,
    ) -> None:
        _ = next_episode_step
        normalized_step = None if episode_step is None else int(episode_step)
        if normalized_step == 0 and self._episode_step_context != 0:
            self._step_in_interval = 0
            self._decision_interval_complete = False
            self._reset_accumulated_reward()
            self._prev_multipliers = None
        self._episode_step_context = normalized_step

    def update(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        done = terminated or truncated
        if self._member_credit:
            reward_vector = np.asarray(rewards, dtype=np.float64).reshape(-1)
            if reward_vector.size != self._num_buildings:
                raise ValueError(
                    "CCLevel2 member_decomposed credit requires one reward per building"
                )
            self._accumulated_reward += reward_vector
        else:
            self._accumulated_reward += float(sum(rewards))

        if not (self._decision_interval_complete or done):
            return

        assert self._cached_community is not None, "predict() must run before update()"

        if not self._bc_pretrain_done or not self._cached_policy_sample:
            self._decision_interval_complete = False
            self._reset_accumulated_reward()
            if done:
                self._step_in_interval = 0
                self._prev_multipliers = None
                self._flush_decision_trace()
            return

        if self._prev_multipliers is None:
            self._prev_multipliers = self._reference_multipliers.copy()

        factor_penalties = (
            self._cached_multipliers - self._reference_multipliers
        ) ** 2
        smoothness_penalties = (
            self._cached_multipliers - self._prev_multipliers
        ) ** 2
        if self._member_credit:
            aux = (
                -self._w_factor * factor_penalties
                -self._w_smoothness * smoothness_penalties
            )
        else:
            aux = float(
                -self._w_factor * np.mean(factor_penalties)
                -self._w_smoothness * np.mean(smoothness_penalties)
            )
        self._prev_multipliers = self._cached_multipliers.copy()

        raw = self._accumulated_reward + aux
        if self._member_credit:
            raw = np.asarray(raw, dtype=np.float64)
            team_share = float(raw.sum()) / self._num_buildings
            raw = (
                (1.0 - self._team_reward_mix) * raw
                + self._team_reward_mix * team_share
            )
        if self._reward_normalization == "none":
            scaled = (
                np.asarray(raw, dtype=np.float32)
                if self._member_credit
                else float(raw)
            )
        elif self._member_credit:
            scaled = np.empty(self._num_buildings, dtype=np.float32)
            for index, value in enumerate(raw):
                normalizer = self._member_reward_rms[index]
                normalizer.update(float(value))
                scaled[index] = (
                    float(value) - normalizer.mean
                ) / max(normalizer.std, 1e-8)
        else:
            raw = float(raw)
            self._reward_rms.update(raw)
            scaled = float(
                (raw - self._reward_rms.mean)
                / max(self._reward_rms.std, 1e-8)
            )

        self.rollout_buffer.add(
            community=self._cached_community,
            action=self._cached_action,
            logprob=self._cached_logprob,
            reward=scaled,
            done=done,
            value=self._cached_value,
            actor_mask=self._cached_actor_mask,
        )

        self._decision_interval_complete = False
        self._reset_accumulated_reward()
        if done:
            self._step_in_interval = 0
            self._prev_multipliers = None
            self._flush_decision_trace()

        if self.rollout_buffer.full:
            self._learn_from_rollout(next_observations, done)

    def _reset_accumulated_reward(self) -> None:
        self._accumulated_reward = (
            np.zeros(self._num_buildings, dtype=np.float64)
            if self._member_credit
            else 0.0
        )

    # ──────────────────────── BC warm-start ──────────────────────────────────

    def _community_signal(self, ctx: np.ndarray) -> float:
        """Community-level raw signal (mirrors CCLevel1 BC teacher)."""
        _idx = self._district_feature_names.index
        price    = float(ctx[_idx("district__electricity_pricing")])
        price_p1 = float(ctx[_idx("district__electricity_pricing_predicted_1")])
        price_p2 = float(ctx[_idx("district__electricity_pricing_predicted_2")])
        price_p3 = float(ctx[_idx("district__electricity_pricing_predicted_3")])
        imp_kw   = float(ctx[_idx("district__community_import_power_kw")])
        exp_kw   = float(ctx[_idx("district__community_export_power_kw")])
        dt       = self._bc_dt_hours
        imp_kwh  = imp_kw * dt
        exp_kwh  = exp_kw * dt

        ref_price   = (price_p1 + price_p2 + price_p3) / 3.0
        cost_signal = (price - ref_price) / max(ref_price, 1e-8)

        peak_excess  = max(0.0, imp_kwh - self._bc_target_import)
        peak_signal  = peak_excess ** 2 / self._bc_reference_peak

        export_signal = -(exp_kwh / self._bc_reference_export)

        return (self._bc_w_cost * cost_signal
                + self._bc_w_peak  * peak_signal
                + self._bc_w_export * export_signal)

    def _cheap_and_export_teacher_active(self, ctx: np.ndarray) -> bool:
        """Causal version of the successful V5 cheap-and-export probe."""

        _idx = self._district_feature_names.index
        price = float(ctx[_idx("district__electricity_pricing")])
        forecasts = [
            float(ctx[_idx(f"district__electricity_pricing_predicted_{index}")])
            for index in (1, 2, 3)
        ]
        forecast_mean = float(np.mean(forecasts))
        forecast_min = float(np.min(forecasts))
        forecast_max = float(np.max(forecasts))
        spread = max(
            forecast_max - forecast_min,
            abs(forecast_mean) * 0.05,
            1.0e-9,
        )
        cheap = (
            price <= forecast_mean - 0.20 * spread
            or price <= forecast_min + 0.10 * spread
        )
        export_kw = float(ctx[_idx("district__community_export_power_kw")])
        return bool(cheap and export_kw > self._bc_export_activation_kw)

    def _cheap_and_export_teacher_multiplier(
        self,
        ctx: np.ndarray,
        *,
        soc: float,
    ) -> float:
        if not self._cheap_and_export_teacher_active(ctx):
            return 1.0
        # Preserve the known-good 0.90 global intervention while making it
        # slightly more attractive for emptier local batteries. This teaches
        # useful Level-2 diversity without using future outcome traces.
        multiplier = self._bc_discount_multiplier + (
            (float(soc) - 0.5) * self._bc_w_soc
        )
        return float(np.clip(multiplier, self._price_min, 1.0))

    def _bc_teacher_multipliers_per_building(
        self,
        ctx: np.ndarray,
        observations: List[np.ndarray],
    ) -> np.ndarray:
        """Compute per-building BC teacher multipliers.

        Mirrors CCRewardLevel2 term structure:
            base       = community signal (cost + peak + export)
            ev_mod[i]  = -w_ev * urgency[i] * gap[i]
                         (high urgency + large deficit → lower mult to allow charging)

        Building block positions (resolved at attach_environment()):
            [0] storage::soc                       → legacy soc_mod (kept for stability)
            [2] net_power_kw                       → legacy net_mod (kept for stability)
            [3] connected_state (EV)
            [4] connected_ev_soc_deficit           = max(req - soc, 0) ∈ [0, 1]
            [5] connected_ev_departure_urgency_24h = 1 - hours/24 ∈ [0, 1]
        """
        if self._bc_target_import is None or self._bc_reference_peak is None:
            return np.ones(self._num_buildings, dtype=np.float32)

        base = self._community_signal(ctx)
        mults = np.empty(self._num_buildings, dtype=np.float32)
        for i in range(self._num_buildings):
            obs_i = observations[i] if i < len(observations) else None
            if obs_i is not None and i < len(self._building_feat_positions):
                positions = self._building_feat_positions[i]
                soc          = float(obs_i[positions[0]]) if positions[0] >= 0 else 0.5
                net          = float(obs_i[positions[2]]) if positions[2] >= 0 else 0.0
                ev_conn      = float(obs_i[positions[3]]) if positions[3] >= 0 else 0.0
                soc_def      = float(obs_i[positions[4]]) if positions[4] >= 0 else 0.0
                urgency_24h  = float(obs_i[positions[5]]) if positions[5] >= 0 else 0.0
            else:
                soc, net, ev_conn, soc_def, urgency_24h = 0.5, 0.0, 0.0, 0.0, 0.0

            # The encoded feature connected_ev_departure_urgency_24h = 1 - hours/24.
            # Invert to recover actual hours, then re-apply the same horizon as
            # CCRewardLevel2 (bc_urgency_horizon, default 4 h) so the teacher
            # and the reward function use identical urgency values.
            actual_hours = (1.0 - urgency_24h) * 24.0
            urgency = max(1.0 - actual_hours / self._bc_urgency_horizon, 0.0)

            if self._bc_teacher_mode == "cheap_and_export":
                mults[i] = self._cheap_and_export_teacher_multiplier(
                    ctx,
                    soc=soc,
                )
            else:
                # SignalAwareRBC interprets multiplier > 1 as expensive
                # (discharge/conserve) and < 1 as cheap (charge/consume). A
                # fuller battery must therefore raise, not lower, the price.
                soc_mod = (soc - 0.5) * self._bc_w_soc
                net_mod = net * self._bc_w_net
                ev_mod = -self._bc_w_ev * urgency * soc_def * ev_conn
                raw = float(np.clip(
                    (base + soc_mod + net_mod + ev_mod) * self._bc_mult_scale,
                    -0.8,
                    0.8,
                ))
                mults[i] = float(np.clip(
                    1.0 + raw,
                    self._price_min,
                    self._price_max,
                ))
        return mults

    def _prepare_bc_pretraining(self) -> None:
        """Prepare supervised BC tensors without running a long optimizer loop."""
        X = np.stack(self._bc_contexts)          # (N_steps, c_dim)
        teacher_X = np.stack(self._bc_teacher_contexts)
        T = np.stack(self._bc_targets)            # (N_steps, num_buildings)

        # Auto-calibrate reference values from community import/export distribution.
        imp_arr = np.array(self._bc_import_samples, dtype=np.float64)
        exp_arr = np.array(self._bc_export_samples, dtype=np.float64)

        if self._bc_target_import is None:
            self._bc_target_import = float(np.percentile(imp_arr, 75))
        if self._bc_reference_peak is None:
            excess_sq = np.maximum(0.0, imp_arr - self._bc_target_import) ** 2
            self._bc_reference_peak = max(float(np.percentile(excess_sq, 90)), 1e-6)
        if self._bc_reference_export is None:
            self._bc_reference_export = max(float(np.percentile(exp_arr, 90)), 1e-6)

        logger.info(
            "CC-L2 BC | collected {} contexts | "
            "target_import={:.3f} ref_peak={:.4f} ref_export={:.3f}",
            len(X),
            self._bc_target_import, self._bc_reference_peak, self._bc_reference_export,
        )

        # Re-compute targets now that reference values are calibrated.
        # Per-building block: [soc, pv, net, ev_conn, soc_deficit, urgency_24h]
        for j in range(len(X)):
            base = self._community_signal(teacher_X[j])
            d_start = self._building_context_start
            for i in range(self._num_buildings):
                feat_start = d_start + i * _N_BUILDING_FEATS
                soc         = float(X[j][feat_start + 0])  # storage::soc [0,1]
                net         = float(X[j][feat_start + 2])  # net_power_kw [-1,1]
                ev_conn     = float(X[j][feat_start + 3])  # connected_state {0,1}
                soc_def     = float(X[j][feat_start + 4])  # soc_deficit [0,1]
                urgency_24h = float(X[j][feat_start + 5])  # departure_urgency_24h [0,1]
                # Recover actual hours from the 24h-horizon encoded feature,
                # then re-apply bc_urgency_horizon (same as CCRewardLevel2).
                actual_hours = (1.0 - urgency_24h) * 24.0
                urgency = max(1.0 - actual_hours / self._bc_urgency_horizon, 0.0)
                if self._bc_teacher_mode == "cheap_and_export":
                    T[j, i] = self._cheap_and_export_teacher_multiplier(
                        teacher_X[j],
                        soc=soc,
                    )
                else:
                    soc_mod = (soc - 0.5) * self._bc_w_soc
                    net_mod = net * self._bc_w_net
                    ev_mod = -self._bc_w_ev * urgency * soc_def * ev_conn
                    raw = float(np.clip(
                        (base + soc_mod + net_mod + ev_mod) * self._bc_mult_scale,
                        -0.8,
                        0.8,
                    ))
                    T[j, i] = float(np.clip(
                        1.0 + raw,
                        self._price_min,
                        self._price_max,
                    ))

        # Convert reachable multiplier targets to the configured pre-tanh
        # representation.  This is especially important for centered residual
        # policies whose reference may lie exactly on a price bound.
        T_raw = self._multipliers_to_raw(T)   # (N_steps, num_buildings)

        self._bc_train_inputs = torch.tensor(X, dtype=torch.float32)
        self._bc_train_targets = torch.tensor(T_raw, dtype=torch.float32)

        bc_params = (list(self.policy.encoder.parameters())
                     + list(self.policy.mean_head.parameters()))
        self._bc_train_optimizer = Adam(bc_params, lr=self._bc_lr)
        self._bc_train_step = 0
        self._bc_train_losses = []
        self._bc_train_started_at = time.perf_counter()

        # The tensors above now own the collected data.  Free the Python lists
        # before episode two starts instead of retaining duplicate annual data.
        self._bc_contexts.clear()
        self._bc_teacher_contexts.clear()
        self._bc_targets.clear()
        self._bc_import_samples.clear()
        self._bc_export_samples.clear()

    def _run_bc_pretraining_chunk(self) -> None:
        """Run a bounded BC chunk so simulator progress remains observable."""
        X_t = self._bc_train_inputs
        T_t = self._bc_train_targets
        bc_opt = self._bc_train_optimizer
        if X_t is None or T_t is None or bc_opt is None:
            return

        N = len(X_t)
        bc_params = [
            parameter
            for group in bc_opt.param_groups
            for parameter in group["params"]
        ]
        start_step = self._bc_train_step
        end_step = min(
            start_step + self._bc_train_chunk_steps,
            self._bc_train_steps,
        )
        chunk_started_at = time.perf_counter()
        original_threads = torch.get_num_threads()
        bounded_threads = min(original_threads, self._bc_max_torch_threads)

        try:
            if bounded_threads != original_threads:
                torch.set_num_threads(bounded_threads)
            for _ in range(start_step, end_step):
                idx_mb = np.random.randint(0, N, size=min(64, N))
                h      = self.policy.encoder(X_t[idx_mb])
                pred   = self.policy.mean_head(h)             # (batch, num_buildings)
                loss   = (pred - T_t[idx_mb]).pow(2).mean()
                bc_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    bc_params,
                    max_norm=1.0,
                )
                bc_opt.step()
                self._bc_train_losses.append(float(loss.item()))
        finally:
            if bounded_threads != original_threads:
                torch.set_num_threads(original_threads)

        self._bc_train_step = end_step
        crossed_progress_boundary = (
            start_step // self._bc_progress_interval
            != end_step // self._bc_progress_interval
        )
        if end_step == self._bc_train_steps or crossed_progress_boundary:
            logger.info(
                "CC-L2 BC pretraining progress | steps={}/{} | "
                "chunk_seconds={:.3f} | mean_loss={:.6f} | torch_threads={}",
                end_step,
                self._bc_train_steps,
                time.perf_counter() - chunk_started_at,
                float(np.mean(self._bc_train_losses)),
                bounded_threads,
            )

        if end_step < self._bc_train_steps:
            return

        mean_loss = float(np.mean(self._bc_train_losses))
        total_seconds = (
            time.perf_counter() - self._bc_train_started_at
            if self._bc_train_started_at is not None
            else 0.0
        )
        logger.info(
            "CC-L2 BC pretraining done | steps={} | loss={:.6f} | seconds={:.3f}",
            self._bc_train_steps,
            mean_loss,
            total_seconds,
        )
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "CC2/bc_pretrain_loss":          mean_loss,
                    "CC2/bc_pretrain_collect_n":     float(N),
                    "CC2/bc_pretrain_target_import": self._bc_target_import,
                    "CC2/bc_pretrain_ref_peak":      self._bc_reference_peak,
                    "CC2/bc_pretrain_ref_export":    self._bc_reference_export,
                    "CC2/bc_pretrain_seconds":       total_seconds,
                },
                step=0,
            )

        self._bc_pretrain_done = True
        self._bc_train_inputs = None
        self._bc_train_targets = None
        self._bc_train_optimizer = None
        self._bc_train_losses = []
        self._bc_train_started_at = None

    # ───────────────────────── Internal: decision ────────────────────────────

    def _build_context(
        self,
        observations: List[np.ndarray],
        *,
        raw_observations: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Build the compact policy context from encoded observations.

        Layout:
            [0:D]       district features (from obs[0])
            [D:D+6*N]   per-building features (obs[i] for i in range(N))

        Within each building block of 6:
            [0] storage::soc                       [0, 1]
            [1] pv::generation_power_kw            [0, 1]
            [2] net_power_kw                       [-1, 1]  (signed)
            [3] connected_state (EV)               {0, 1}
            [4] connected_ev_soc_deficit           [0, 1]
            [5] connected_ev_departure_urgency_24h [0, 1]

        Feature positions were resolved at attach_environment() via pattern
        matching against encoded observation names.  Missing features → 0.0.
        """
        obs0 = observations[0]

        # District features (positions resolved at attach time)
        district = np.array(
            [float(obs0[p]) if p >= 0 else 0.0 for p in self._district_positions],
            dtype=np.float32,
        )
        causal_context = np.asarray(
            [
                float(
                    self._physical_causal_intervention_active(raw_observations)
                )
            ],
            dtype=np.float32,
        ) if self._causal_use_physical_context else np.empty(0, dtype=np.float32)

        # Per-building features
        building_parts: List[np.ndarray] = []
        n_feat = _N_BUILDING_FEATS
        for i in range(self._num_buildings):
            obs_i = observations[i] if i < len(observations) else obs0
            if i < len(self._building_feat_positions):
                positions = self._building_feat_positions[i]
                bfeat = np.array(
                    [float(obs_i[p]) if p >= 0 else 0.0 for p in positions],
                    dtype=np.float32,
                )
            else:
                bfeat = np.zeros(n_feat, dtype=np.float32)
            building_parts.append(bfeat)

        return np.concatenate([district, causal_context] + building_parts)

    def _build_teacher_context(self, encoded_context: np.ndarray) -> np.ndarray:
        """Overlay physical district values for the BC teacher only.

        The policy deliberately consumes normalized features. The heuristic
        teacher, however, compares import/export energy against physical kWh
        references and therefore must not use the min-max encoded values.
        Per-building SoC/net/EV features remain encoded in their documented
        ranges.
        """
        teacher = np.asarray(encoded_context, dtype=np.float32).copy()
        if not self._bc_use_physical_teacher_context:
            return teacher
        if not self._latest_raw_observations or not self._raw_obs_indices:
            raise RuntimeError(
                "CCLevel2 physical BC teacher did not receive raw observation context"
            )

        raw = self._latest_raw_observations[0]
        raw_index = self._raw_obs_indices[0]
        aliases = {
            "district__time_of_day_sin": "district__seconds_of_day_sin",
            "district__time_of_day_cos": "district__seconds_of_day_cos",
        }
        for index, name in enumerate(self._district_feature_names):
            physical_name = aliases.get(name, name)
            position = raw_index.get(physical_name)
            if position is None or position >= len(raw):
                continue
            value = float(raw[position])
            if np.isfinite(value):
                teacher[index] = value
        return teacher

    def _sample_new_decision(self, observations, deterministic: bool | None) -> None:
        ctx   = self._build_context(observations)
        ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                h    = self.policy.encoder(ctx_t)
                raw  = self.policy.mean_head(h)           # (1, N)
                _, logprob, _, value = self.policy.get_action_and_value(ctx_t, raw)
            else:
                raw, logprob, _, value = self.policy.get_action_and_value(ctx_t)

        raw_np   = raw.squeeze(0).numpy()                 # (N,)
        mults = self._raw_to_multipliers(raw_np, context=ctx)

        self._cached_action      = raw_np.astype(np.float32)
        self._cached_multipliers = mults.astype(np.float32)
        self._cached_community   = ctx
        self._cached_logprob     = logprob.squeeze(0).numpy().astype(np.float32)
        value_np = value.squeeze(0).numpy()
        self._cached_value = (
            value_np.astype(np.float32)
            if self._member_credit
            else float(np.asarray(value_np).item())
        )
        self._cached_policy_sample = True
        active = (
            self._causal_intervention_active(ctx)
            if self._policy_parameterization == "causal_active_only"
            else True
        )
        self._cached_actor_mask = np.full(
            self._num_buildings,
            float(active),
            dtype=np.float32,
        )

        self._log_decision()

    # ───────────────────────── Internal: learning ────────────────────────────

    def _learn_from_rollout(self, next_observations, done: bool) -> None:
        ctx = torch.tensor(
            self._build_context(
                next_observations,
                raw_observations=self._latest_raw_next_observations,
            ),
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            _, _, _, last_value = self.policy.get_action_and_value(ctx)
        last_value_np = last_value.squeeze(0).numpy()
        self.rollout_buffer.compute_gae(
            (
                last_value_np.astype(np.float32)
                if self._member_credit
                else float(np.asarray(last_value_np).item())
            ),
            done,
            self._gamma,
            self._gae_lambda,
        )
        self._run_ppo_update()
        self.rollout_buffer.reset()

    def _run_ppo_update(self) -> None:
        data = self.rollout_buffer.get()
        community    = data["community"]   # (T, c_dim)
        actions      = data["actions"]     # (T, N)
        old_logprobs = data["logprobs"]    # (T, N)
        actor_masks  = data["actor_masks"]
        returns      = data["returns"]
        advantages   = data["advantages"]
        old_values   = torch.tensor(self.rollout_buffer.values, dtype=torch.float32)

        if self._member_credit:
            advantages = self._normalize_masked_actor_advantages(
                advantages,
                actor_masks,
            )
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_steps = self.rollout_buffer.num_steps
        kl_stop   = False
        approx_kl = 0.0
        pg_loss = v_loss = ent_loss = torch.tensor(0.0)

        for _ in range(self._num_epochs):
            if kl_stop:
                break
            permutation = np.random.permutation(num_steps)
            for start in range(0, num_steps, self._mini_batch_size):
                mb = permutation[start : start + self._mini_batch_size]

                _, new_logprobs, entropy, new_values = self.policy.get_action_and_value(
                    community[mb], actions[mb]
                )
                if not self._member_credit:
                    new_values = new_values.squeeze()

                log_ratio  = new_logprobs - old_logprobs[mb]
                ratio      = torch.exp(log_ratio)
                mask = actor_masks[mb]
                active_count = mask.sum()
                if active_count.item() > 0.0:
                    approx_kl = float(
                        ((((ratio - 1) - log_ratio) * mask).sum() / active_count).item()
                    )
                else:
                    approx_kl = 0.0
                if self._target_kl is not None and approx_kl > 1.5 * self._target_kl:
                    kl_stop = True
                    break

                mb_adv = (
                    advantages[mb]
                    if self._member_credit
                    else advantages[mb].unsqueeze(-1)
                )
                pg_elements = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef),
                )
                if active_count.item() > 0.0:
                    pg_loss = (pg_elements * mask).sum() / active_count
                    ent_loss = (entropy * mask).sum() / active_count
                else:
                    pg_loss = new_logprobs.sum() * 0.0
                    ent_loss = entropy.sum() * 0.0

                v_unclipped = (new_values - returns[mb]) ** 2
                v_clipped   = old_values[mb] + (new_values - old_values[mb]).clamp(
                    -self._clip_coef, self._clip_coef
                )
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - returns[mb]) ** 2).mean()

                loss     = pg_loss + self._vf_coef * v_loss - self._ent_coef * ent_loss

                self.ppo_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._max_grad_norm)
                self.ppo_optim.step()

        self._ppo_update_count += 1
        log_std_mean = float(self.policy.log_std.mean().item())
        logger.info(
            "CC-L2 PPO | pg={:.4f} v={:.4f} ent={:.4f} kl={:.4f} log_std={:.3f} kl_stop={}",
            pg_loss.item(), v_loss.item(), ent_loss.item(), approx_kl, log_std_mean, kl_stop,
        )
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "CC2/PPO_pg_loss":    pg_loss.item(),
                    "CC2/PPO_v_loss":     v_loss.item(),
                    "CC2/PPO_entropy":    ent_loss.item(),
                    "CC2/PPO_approx_kl":  approx_kl,
                    "CC2/PPO_kl_stop":    float(kl_stop),
                    "CC2/PPO_log_std":    log_std_mean,
                    "CC2/reward_mean": float(
                        np.mean([rms.mean for rms in self._member_reward_rms])
                        if self._member_credit
                        else self._reward_rms.mean
                    ),
                    "CC2/reward_std": float(
                        np.mean([rms.std for rms in self._member_reward_rms])
                        if self._member_credit
                        else self._reward_rms.std
                    ),
                },
                step=self._ppo_update_count,
            )

    @staticmethod
    def _normalize_masked_actor_advantages(
        advantages: torch.Tensor,
        actor_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize each price factor over transitions it could influence.

        Causal CC policies are exactly neutral for most transitions. Those
        samples remain useful to the critic, but including them in the actor's
        advantage mean/variance shifts and dilutes the much smaller active
        subset. The returned inactive entries are zero because the policy loss
        masks them in every case.
        """

        if advantages.shape != actor_masks.shape:
            raise ValueError(
                "CCLevel2 actor advantages and masks must have matching shapes"
            )
        active_counts = actor_masks.sum(dim=0, keepdim=True)
        safe_counts = active_counts.clamp_min(1.0)
        means = (advantages * actor_masks).sum(
            dim=0,
            keepdim=True,
        ) / safe_counts
        variances = (
            ((advantages - means) ** 2 * actor_masks).sum(
                dim=0,
                keepdim=True,
            )
            / safe_counts
        )
        normalized = (advantages - means) / torch.sqrt(variances + 1.0e-8)
        return torch.where(
            actor_masks > 0.0,
            normalized,
            torch.zeros_like(normalized),
        )

    # ──────────────────────── Internal: logging ──────────────────────────────

    def _log_decision(self) -> None:
        ctx   = self._cached_community
        mults = self._cached_multipliers
        _idx  = self._district_feature_names.index

        # Per-building EV state from context blocks [district | b0 | b1 | … | bN]
        # Block layout: [soc, pv, net, ev_conn, soc_def, urgency_24h]
        d = self._building_context_start
        k = _N_BUILDING_FEATS
        ev_harm_vals: List[float] = []
        n_ev_connected = 0
        for i in range(self._num_buildings):
            base = d + i * k
            if base + 5 < len(ctx):
                ev_conn     = float(ctx[base + 3])
                soc_def     = float(ctx[base + 4])
                urgency_24h = float(ctx[base + 5])
                # Recover actual hours and reapply urgency horizon (same as BC teacher)
                actual_hours = (1.0 - urgency_24h) * 24.0
                urgency = max(1.0 - actual_hours / self._bc_urgency_horizon, 0.0)
                harm = urgency * soc_def * ev_conn
                ev_harm_vals.append(harm)
                if ev_conn > 0.5:
                    n_ev_connected += 1

        ev_harm_mean = float(np.mean(ev_harm_vals)) if ev_harm_vals else 0.0
        ev_harm_max  = float(np.max(ev_harm_vals))  if ev_harm_vals else 0.0

        record: dict = {
            "timestep":        self._global_cc_step,
            "causal_active": int(
                self._causal_intervention_active(ctx)
                if self._policy_parameterization == "causal_active_only"
                else True
            ),
            "mult_mean":       float(mults.mean()),
            "mult_std":        float(mults.std()),
            "mult_min":        float(mults.min()),
            "mult_max":        float(mults.max()),
            "value_est":       float(np.asarray(self._cached_value).mean()),
            "import_norm":     float(ctx[_idx("district__community_import_power_kw")]),
            "pv_norm":         float(ctx[_idx("district__community_pv_power_kw")]),
            "carbon_norm":     float(ctx[_idx("district__carbon_intensity")]),
            "ev_harm_mean":    ev_harm_mean,
            "ev_harm_max":     ev_harm_max,
            "n_ev_connected":  float(n_ev_connected),
        }
        # Per-building multipliers (b0..b16) — allows post-hoc per-building analysis
        for i, m in enumerate(mults):
            record[f"mult_b{i}"] = float(m)

        self._decision_trace.append(record)
        self._global_cc_step += 1

    def _flush_decision_trace(self) -> None:
        if not self._decision_trace:
            return
        self._episode_count += 1
        ep     = self._episode_count
        fields = list(self._decision_trace[0].keys())
        self._completed_decision_traces.extend(
            {"episode": ep, **row} for row in self._decision_trace
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=f"cc2_ep{ep}_", delete=False
        ) as f:
            tmp_path = f.name
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self._decision_trace)

        # ── Multiplier statistics ─────────────────────────────────────────────
        mult_means = np.array([r["mult_mean"] for r in self._decision_trace], dtype=np.float64)
        mult_stds  = np.array([r["mult_std"]  for r in self._decision_trace], dtype=np.float64)
        # Per-building mean multiplier over the episode (b0..bN-1)
        per_b_means = {}
        for i in range(self._num_buildings):
            key = f"mult_b{i}"
            if key in self._decision_trace[0]:
                per_b_means[i] = float(np.mean([r[key] for r in self._decision_trace]))

        mult_spread      = float(mult_means.std())          # how much CC differentiates over time
        mult_mean_ep     = float(mult_means.mean())
        intra_spread_ep  = float(mult_stds.mean())          # avg per-step spread across buildings

        # ── EV KPIs ───────────────────────────────────────────────────────────
        ev_harm_means = np.array([r["ev_harm_mean"] for r in self._decision_trace], dtype=np.float64)
        ev_harm_maxes = np.array([r["ev_harm_max"]  for r in self._decision_trace], dtype=np.float64)
        n_ev_arr      = np.array([r["n_ev_connected"] for r in self._decision_trace], dtype=np.float64)

        ev_harm_ep        = float(ev_harm_means.mean())     # avg urgency*deficit: lower is better
        ev_harm_peak_ep   = float(ev_harm_maxes.max())      # worst single-step worst-building harm
        ev_harm_high_frac = float(np.mean(ev_harm_maxes > 0.1))  # fraction of steps with any urgent unmet EV
        n_ev_mean         = float(n_ev_arr.mean())          # avg EVs connected per step

        # ── Community proxy KPIs (normalised values — meaningful for trend tracking) ──
        # NOTE: these are encoded [0,1] values, not kW. Use for relative trends only.
        imports = np.array([r["import_norm"] for r in self._decision_trace], dtype=np.float64)
        pvs     = np.array([r["pv_norm"]     for r in self._decision_trace], dtype=np.float64)
        carbons = np.array([r["carbon_norm"] for r in self._decision_trace], dtype=np.float64)
        denom   = float(imports.sum() + pvs.sum())
        self_sufficiency_proxy = 1.0 - float(imports.sum()) / denom if denom > 1e-6 else float("nan")
        carbon_import_proxy    = float(np.mean(imports * carbons))

        logger.info(
            "CC-L2 ep{} | {} steps | mult_mean={:.3f} intra_spread={:.3f} "
            "ev_harm={:.4f} ev_harm_peak={:.4f} ev_high_frac={:.3f} n_ev={:.1f}",
            ep, len(self._decision_trace),
            mult_mean_ep, intra_spread_ep,
            ev_harm_ep, ev_harm_peak_ep, ev_harm_high_frac, n_ev_mean,
        )
        if mlflow.active_run():
            mlflow.log_artifact(tmp_path, artifact_path="decision_traces")
            metrics: dict = {
                # Multiplier quality
                "CC2_ep/mult_mean":         mult_mean_ep,
                "CC2_ep/mult_spread_time":  mult_spread,       # variation over episode
                "CC2_ep/mult_spread_intra": intra_spread_ep,   # avg spread across buildings per step
                # EV service quality — the Phase 2 differentiator
                "CC2_ep/ev_harm_mean":      ev_harm_ep,
                "CC2_ep/ev_harm_peak":      ev_harm_peak_ep,
                "CC2_ep/ev_high_risk_frac": ev_harm_high_frac,
                "CC2_ep/n_ev_connected":    n_ev_mean,
                # Community proxies (normalised — track relative trends)
                "CC2_ep/self_suff_proxy":   self_sufficiency_proxy if not np.isnan(self_sufficiency_proxy) else 0.0,
                "CC2_ep/carbon_import_proxy": carbon_import_proxy,
            }
            # Per-building mean multiplier — reveals systematic bias toward specific buildings
            for i, mean_m in per_b_means.items():
                metrics[f"CC2_ep/mult_b{i}_mean"] = mean_m
            mlflow.log_metrics(metrics, step=ep)

        self._decision_trace = []

    # ───────────────────────── Lifecycle / artifacts ─────────────────────────

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return True

    def export_artifacts(self, output_dir, context=None):
        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        export_path = onnx_dir / "cc2_market_maker.onnx"
        deterministic_policy = DeterministicVectorMultiplierPolicy(
            self.policy,
            self._price_min,
            self._price_max,
            self._reference_multipliers,
            self._policy_residual_scale,
            self._policy_parameterization,
            self._causal_initial_multipliers,
            self._causal_residual_scale,
            (
                self._n_district
                if self._causal_use_physical_context
                else None
            ),
        ).eval()
        torch.onnx.export(
            deterministic_policy,
            torch.randn(1, self._c_dim),
            str(export_path),
            export_params=True,
            opset_version=DEFAULT_ONNX_OPSET,
            do_constant_folding=True,
            input_names=["community_context"],
            output_names=["price_multipliers"],
            dynamic_axes={
                "community_context": {0: "batch"},
                "price_multipliers": {0: "batch"},
            },
        )
        self._flush_decision_trace()
        artifacts = [
            {
                "agent_index": 0,
                "path": str(export_path.relative_to(export_root)),
                "format": "onnx",
            }
        ]
        diagnostic_artifacts: List[Dict[str, Any]] = []
        if self._completed_decision_traces:
            trace_path = export_root / "decision_trace.csv"
            fields = list(self._completed_decision_traces[0].keys())
            with trace_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self._completed_decision_traces)
            diagnostic_artifacts.append(
                {
                    "path": str(trace_path.relative_to(export_root)),
                    "format": "csv",
                }
            )
        return {
            "format": "onnx",
            "output_contract": "deterministic_per_building_price_multiplier_vector",
            "num_buildings": self._num_buildings,
            "price_min": self._price_min,
            "price_max": self._price_max,
            "reference_multipliers": self._reference_multipliers.tolist(),
            "policy_residual_scale": self._policy_residual_scale,
            "policy_parameterization": self._policy_parameterization,
            "causal_initial_multiplier": self._causal_initial_multiplier,
            "causal_initial_multipliers": (
                self._causal_initial_multipliers.tolist()
            ),
            "causal_residual_scale": self._causal_residual_scale,
            "reward_normalization": self._reward_normalization,
            "credit_assignment": self._credit_assignment,
            "team_reward_mix": self._team_reward_mix,
            "community_context_dimension": self._c_dim,
            "district_features": list(self._district_feature_names),
            "causal_context_features": list(self._causal_context_feature_names),
            "include_community_headroom": self._include_community_headroom,
            "include_community_history": self._include_community_history,
            "bc_use_physical_teacher_context": (
                self._bc_use_physical_teacher_context
            ),
            "causal_use_physical_context": self._causal_use_physical_context,
            "separate_value_encoder": self._separate_value_encoder,
            "artifacts": artifacts,
            "diagnostic_artifacts": diagnostic_artifacts,
        }

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        path = Path(output_dir) / f"cc2_step_{step}.pt"
        torch.save(
            {
                "step":               step,
                "policy":             self.policy.state_dict(),
                "optimizer":          self.ppo_optim.state_dict(),
                "num_buildings":      self._num_buildings,
                "c_dim":              self._c_dim,
                "district_features":  list(self._district_feature_names),
                "reward_rms_n":       self._reward_rms._n,
                "reward_rms_mean":    self._reward_rms._mean,
                "reward_rms_M2":      self._reward_rms._M2,
                "member_reward_rms": [
                    {
                        "n": normalizer._n,
                        "mean": normalizer._mean,
                        "M2": normalizer._M2,
                    }
                    for normalizer in self._member_reward_rms
                ],
                "ppo_update_count":   self._ppo_update_count,
                "global_cc_step":     self._global_cc_step,
                "reference_multipliers": self._reference_multipliers.tolist(),
                "credit_assignment": self._credit_assignment,
                "team_reward_mix": self._team_reward_mix,
                "policy_parameterization": self._policy_parameterization,
                "causal_initial_multipliers": (
                    self._causal_initial_multipliers.tolist()
                ),
                "causal_residual_scale": self._causal_residual_scale,
                "causal_use_physical_context": self._causal_use_physical_context,
                "separate_value_encoder": self._separate_value_encoder,
                "bc_pretrain_done":   self._bc_pretrain_done,
                "bc_target_import":   self._bc_target_import,
                "bc_reference_peak":  self._bc_reference_peak,
                "bc_reference_export": self._bc_reference_export,
            },
            path,
        )
        logger.info("CC-L2 checkpoint saved → {}", path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        root = Path(checkpoint_path)
        if root.is_dir():
            cands = sorted(root.glob("cc2_step_*.pt"), key=lambda p: p.stat().st_mtime)
            if not cands:
                raise FileNotFoundError(f"No CC-L2 checkpoint in {root}")
            path = cands[-1]
        else:
            path = root
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        checkpoint_c_dim = ckpt.get("c_dim")
        if checkpoint_c_dim is not None and int(checkpoint_c_dim) != self._c_dim:
            raise ValueError(
                "CC-L2 checkpoint observation dimension does not match the current "
                f"config: checkpoint={checkpoint_c_dim}, config={self._c_dim}"
            )
        checkpoint_reference = ckpt.get("reference_multipliers")
        if checkpoint_reference is not None and not np.allclose(
            np.asarray(checkpoint_reference, dtype=np.float32),
            self._reference_multipliers,
        ):
            raise ValueError(
                "CC-L2 checkpoint reference_multipliers do not match the current config"
            )
        checkpoint_credit = ckpt.get("credit_assignment", "global")
        if checkpoint_credit != self._credit_assignment:
            raise ValueError(
                "CC-L2 checkpoint credit_assignment does not match the current config"
            )
        checkpoint_parameterization = ckpt.get("policy_parameterization")
        if (
            checkpoint_parameterization is not None
            and checkpoint_parameterization != self._policy_parameterization
        ):
            raise ValueError(
                "CC-L2 checkpoint policy_parameterization does not match the current config"
            )
        checkpoint_causal_initial = ckpt.get("causal_initial_multipliers")
        if checkpoint_causal_initial is not None and not np.allclose(
            np.asarray(checkpoint_causal_initial, dtype=np.float32),
            self._causal_initial_multipliers,
        ):
            raise ValueError(
                "CC-L2 checkpoint causal_initial_multipliers do not match "
                "the current config"
            )
        checkpoint_causal_residual = ckpt.get("causal_residual_scale")
        if checkpoint_causal_residual != self._causal_residual_scale:
            raise ValueError(
                "CC-L2 checkpoint causal_residual_scale does not match the current config"
            )
        checkpoint_separate_value = bool(
            ckpt.get("separate_value_encoder", False)
        )
        if checkpoint_separate_value != self._separate_value_encoder:
            raise ValueError(
                "CC-L2 checkpoint separate_value_encoder does not match the current config"
            )
        self.policy.load_state_dict(ckpt["policy"])
        self.ppo_optim.load_state_dict(ckpt["optimizer"])
        self._reward_rms._n    = ckpt.get("reward_rms_n",    0)
        self._reward_rms._mean = ckpt.get("reward_rms_mean", 0.0)
        self._reward_rms._M2   = ckpt.get("reward_rms_M2",  0.0)
        member_states = ckpt.get("member_reward_rms")
        if isinstance(member_states, list):
            if len(member_states) != self._num_buildings:
                raise ValueError(
                    "CC-L2 checkpoint member reward normalizer count does not match"
                )
            for normalizer, state in zip(
                self._member_reward_rms,
                member_states,
            ):
                normalizer._n = int(state.get("n", 0))
                normalizer._mean = float(state.get("mean", 0.0))
                normalizer._M2 = float(state.get("M2", 0.0))
        self._ppo_update_count  = int(ckpt.get("ppo_update_count", 0))
        self._global_cc_step    = int(ckpt.get("global_cc_step", 0))
        if "bc_pretrain_done" in ckpt:
            self._bc_pretrain_done = bool(ckpt["bc_pretrain_done"])
        for key in ("bc_target_import", "bc_reference_peak", "bc_reference_export"):
            if key in ckpt and ckpt[key] is not None:
                setattr(self, f"_{key}", float(ckpt[key]))
        logger.info("CC-L2 checkpoint loaded ← {}", path)
