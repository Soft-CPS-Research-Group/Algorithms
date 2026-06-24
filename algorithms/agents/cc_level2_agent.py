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
        • 17 district features (same as cc_level1: time, price, carbon,
          community power, community building headroom)
        • 6 per-building features: storage::soc, pv::generation_power_kw,
          net_power_kw, ev connected_state, ev soc_deficit, ev urgency

    The CC assembles a single context vector of shape
        (17 + 6 * num_buildings,)
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
    PPO with factorized diagonal Gaussian.  Log-prob of the joint action is
    the sum of per-building log-probs (independent actions, shared state).
    GAE, reward scaling, and all Phase 1 hyperparameters are preserved.

Pipeline integration:
    predict() returns List[float] of length N.
    The Ensemble wrapper (pipeline.py) routes context[i] to building i, so
    SignalAwareRBC workers receive their individual multiplier unchanged.
    No changes to SignalAwareRBC are needed.
"""

from __future__ import annotations

import csv
import tempfile
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
    "district__community_building_headroom_kw",  # [16] grid headroom — helps avoid violations
)
_N_DISTRICT = len(_CC_LEVEL2_DISTRICT_FEATURES)  # 17
_PRICE_FEATURE = "district__electricity_pricing"

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
    """Permutation-equivariant per-building actor + community critic.

    Actor:
        district_encoder maps the shared district block → a district embedding
        (computed once). A SINGLE shared building_head then maps
            [district_embedding, building_i_features] → building i's pre-tanh mean
        with the SAME weights for every building. This guarantees that two
        buildings in the same state receive the same multiplier, and forces all
        per-building differentiation to come from each building's own features —
        there are no independent per-building output rows or biases, so the
        policy cannot encode arbitrary frozen per-building tiers.

    Critic:
        a separate monolithic MLP over the full context → one community value.
        Equivariance is unnecessary for a single scalar value, so the critic
        stays expressive over the whole context.

    A single shared log_std (buildings are symmetric) controls exploration.
    """

    def __init__(
        self,
        c_dim: int,
        num_buildings: int,
        hidden_dims: List[int],
        *,
        n_district: int,
        n_building_feats: int,
        building_hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.num_buildings = num_buildings
        self.n_district = n_district
        self.n_building_feats = n_building_feats

        # ── District encoder (shared, applied once) ──────────────────────────
        layers: List[nn.Module] = []
        in_d = n_district
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        self.district_encoder = nn.Sequential(*layers)
        self._district_emb_dim = in_d

        # ── Shared per-building head: [district_emb, building_feats] → 1 ──────
        bdims = list(building_hidden_dims) if building_hidden_dims else list(hidden_dims)
        layers = []
        in_b = self._district_emb_dim + n_building_feats
        for h in bdims:
            layers += [nn.Linear(in_b, h), nn.Tanh()]
            in_b = h
        layers += [nn.Linear(in_b, 1)]
        self.building_head = nn.Sequential(*layers)

        # ── Critic: monolithic MLP over the full context → scalar value ──────
        layers = []
        in_c = c_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_c, h), nn.Tanh()]
            in_c = h
        layers += [nn.Linear(in_c, 1)]
        self.critic = nn.Sequential(*layers)

        # Single shared log_std (buildings are symmetric, state-independent).
        self.log_std = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Final building-head layer near-zero so the policy starts near neutral
        # (mean ≈ 0 → tanh ≈ 0 → multiplier ≈ 1.0).
        last_building = [m for m in self.building_head if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_building.weight, gain=0.01)
        last_critic = [m for m in self.critic if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_critic.weight, gain=1.0)

    def action_means(self, community: torch.Tensor) -> torch.Tensor:
        """Return (batch, N) pre-tanh per-building means via the shared head."""
        district  = community[:, : self.n_district]
        buildings = community[:, self.n_district :].reshape(
            -1, self.num_buildings, self.n_building_feats
        )
        d = self.district_encoder(district)                       # (batch, emb)
        d_rep = d.unsqueeze(1).expand(-1, self.num_buildings, -1)  # (batch, N, emb)
        x = torch.cat([d_rep, buildings], dim=-1)                 # (batch, N, emb+feats)
        return self.building_head(x).squeeze(-1)                  # (batch, N)

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
            log_prob:  (batch,)   joint log-prob summed over N buildings
            entropy:   (batch,)   joint entropy summed over N buildings
            value:     (batch,)   state value estimate
        """
        means = self.action_means(community)                     # (batch, N)
        value = self.critic(community).squeeze(-1)               # (batch,)
        # Clamp log_std to [-3, 1] to prevent runaway variance.
        stds  = torch.exp(self.log_std.clamp(-3.0, 1.0)).expand_as(means)
        dist  = torch.distributions.Normal(means, stds)
        if action is None:
            action = dist.sample()                               # (batch, N)
        # Tanh correction; sum over building dimension for joint log-prob.
        tanh_correction = torch.log(1.0 - torch.tanh(action) ** 2 + 1e-6)
        log_prob = (dist.log_prob(action) - tanh_correction).sum(dim=-1)  # (batch,)
        entropy  = dist.entropy().sum(dim=-1)                    # (batch,)
        return action, log_prob, entropy, value


# ── Rollout buffer (N-dim actions) ───────────────────────────────────────────

class RolloutBufferV2:
    """Fixed-size rollout buffer for continuous N-dimensional PPO."""

    def __init__(self, num_steps: int, c_dim: int, num_buildings: int) -> None:
        self.num_steps    = num_steps
        self.num_buildings = num_buildings
        self._ptr  = 0
        self.full  = False
        self.communities = np.zeros((num_steps, c_dim),           dtype=np.float32)
        self.actions     = np.zeros((num_steps, num_buildings),   dtype=np.float32)
        self.logprobs    = np.zeros(num_steps,                    dtype=np.float32)
        self.rewards     = np.zeros(num_steps,                    dtype=np.float32)
        self.dones       = np.zeros(num_steps,                    dtype=np.float32)
        self.values      = np.zeros(num_steps,                    dtype=np.float32)
        self.returns     = np.zeros(num_steps,                    dtype=np.float32)
        self.advantages  = np.zeros(num_steps,                    dtype=np.float32)

    def add(self, community, action, logprob, reward, done, value) -> None:
        self.communities[self._ptr] = community
        self.actions[self._ptr]     = action          # (N,)
        self.logprobs[self._ptr]    = logprob         # scalar (summed over N)
        self.rewards[self._ptr]     = reward
        self.dones[self._ptr]       = float(done)
        self.values[self._ptr]      = value
        self._ptr += 1
        if self._ptr >= self.num_steps:
            self.full = True

    def compute_gae(self, last_value, last_done, gamma, gae_lambda) -> None:
        gae = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_nt, next_value = 1.0 - float(last_done), last_value
            else:
                next_nt, next_value = 1.0 - self.dones[t + 1], self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nt - self.values[t]
            gae = delta + gamma * gae_lambda * next_nt * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get(self) -> dict:
        return {
            "community":  torch.tensor(self.communities, dtype=torch.float32),
            "actions":    torch.tensor(self.actions,     dtype=torch.float32),
            "logprobs":   torch.tensor(self.logprobs,    dtype=torch.float32),
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

        # Price-multiplier bounds
        self._price_min = float(hyper.get("price_min", 0.5))
        self._price_max = float(hyper.get("price_max", 1.5))

        # Auxiliary reward weights
        self._w_factor     = float(hyper.get("w_factor",     0.3))
        self._w_smoothness = float(hyper.get("w_smoothness", 0.02))
        self._prev_multipliers: Optional[np.ndarray] = None  # (N,), init at first step

        # Community size — must be set before rollout buffer is created.
        self._num_buildings = int(hyper.get("num_buildings", 17))

        # Context dimension: 17 district + 6 per-building × N = 119 for N=17
        self._n_district = _N_DISTRICT
        self._n_building_feats = _N_BUILDING_FEATS
        default_c_dim = _N_DISTRICT + _N_BUILDING_FEATS * self._num_buildings
        self._c_dim = int(hyper.get("c_dim", default_c_dim))

        # Per-building feature positions in encoded obs — populated at attach_environment.
        self._district_positions: List[int] = []
        self._building_feat_positions: List[List[int]] = []

        self._hidden_dims = list(hyper.get("hidden_dims", [256, 256]))
        # Optional separate sizing for the shared per-building head; defaults to hidden_dims.
        self._building_hidden_dims = list(hyper.get("building_hidden_dims", self._hidden_dims))
        self._lr = float(hyper.get("lr", 1e-4))
        self.policy = CommunityMarketMakerNetV2(
            self._c_dim, self._num_buildings, self._hidden_dims,
            n_district=self._n_district, n_building_feats=self._n_building_feats,
            building_hidden_dims=self._building_hidden_dims,
        )
        self.ppo_optim = Adam(self.policy.parameters(), lr=self._lr)

        self._reward_rms = RunningMeanStd()

        self.rollout_buffer = RolloutBufferV2(
            int(hyper.get("num_steps", 96)), self._c_dim, self._num_buildings
        )
        self._ppo_update_count = 0

        # Temporal abstraction
        self._cc_action_interval = int(hyper.get("cc_action_interval", 4))
        self._step_in_interval = 0

        # Cached decision (arrays instead of scalars)
        self._cached_multipliers: np.ndarray = np.ones(self._num_buildings, dtype=np.float32)
        self._cached_action:      np.ndarray = np.zeros(self._num_buildings, dtype=np.float32)
        self._cached_community:   Optional[np.ndarray] = None
        self._cached_logprob:     float = 0.0
        self._cached_value:       float = 0.0
        self._accumulated_reward: float = 0.0

        # BC warm-start
        self._bc_enabled       = bool(hyper.get("bc_pretrain_enabled", False))
        self._bc_collect_steps = int(hyper.get("bc_collect_steps", 336))
        self._bc_train_steps   = int(hyper.get("bc_train_steps",   2000))
        self._bc_lr            = float(hyper.get("bc_lr",           1e-3))
        self._bc_pretrain_done: bool = not self._bc_enabled
        self._bc_contexts:  List[np.ndarray] = []
        # targets: shape (N_steps, num_buildings)
        self._bc_targets:   List[np.ndarray] = []
        # Community-level reference values (auto-calibrated from episode-0 data)
        self._bc_dt_hours         = float(hyper.get("bc_dt_hours", 0.25))
        self._bc_target_import    = hyper.get("bc_target_import",    None)
        self._bc_reference_peak   = hyper.get("bc_reference_peak",   None)
        self._bc_reference_export = hyper.get("bc_reference_export", None)
        self._bc_reference_price  = hyper.get("bc_reference_price",  None)  # €/kWh p50, auto if None
        self._bc_reference_ramping = float(hyper.get("bc_reference_ramping", 1.878))  # p90 step-to-step Δimport
        self._bc_reference_headroom = float(hyper.get("bc_reference_headroom", 2.0))  # kW low-headroom threshold
        self._bc_import_samples: List[float] = []
        self._bc_export_samples: List[float] = []
        self._bc_price_samples:  List[float] = []
        # Ramp/violation tracking during BC collection (mirror CCLevel1Agent).
        self._bc_prev_import_kwh: float = 0.0
        self._bc_ramp_samples: List[float] = []
        self._bc_violation_samples: List[float] = []
        # Community weights mirroring CCRewardLevel1 (all 5 terms).
        self._bc_w_cost      = float(hyper.get("bc_w_cost",      1.0))
        self._bc_w_peak      = float(hyper.get("bc_w_peak",      0.6))
        self._bc_w_ramp      = float(hyper.get("bc_w_ramp",      0.4))
        self._bc_w_export    = float(hyper.get("bc_w_export",    0.05))
        self._bc_w_violation = float(hyper.get("bc_w_violation", 2.0))
        self._bc_w_headroom  = float(hyper.get("bc_w_headroom",  1.0))
        self._bc_mult_scale  = float(hyper.get("bc_mult_scale",  1.0))
        # Per-building EV modulation weight (mirrors CCRewardLevel2 w_ev).
        self._bc_w_ev     = float(hyper.get("bc_w_ev",   0.5))   # mirrors CCRewardLevel2 w_ev
        # Must match CCRewardLevel2.urgency_horizon so BC and reward use the same urgency scale.
        # The encoded obs feature connected_ev_departure_urgency_24h uses a fixed 24h horizon;
        # we invert it to recover actual hours and re-apply this horizon.
        self._bc_urgency_horizon = float(hyper.get("bc_urgency_horizon", 4.0))

        # Obs layout (set in attach_environment)
        self._obs_index: Dict[str, int] = {}  # feature name → index in obs

        # Diagnostics
        self._episode_count = 0
        self._global_cc_step = 0
        self._decision_trace: List[dict] = []

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
        self._district_positions = [obs0_idx.get(name, -1) for name in _CC_LEVEL2_DISTRICT_FEATURES]

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

    def _rebuild_for_num_buildings(self) -> None:
        """Reconstruct policy and buffer when num_buildings changes at env attach."""
        self._c_dim = _N_DISTRICT + _N_BUILDING_FEATS * self._num_buildings  # 17 + 6×N
        self.policy = CommunityMarketMakerNetV2(
            self._c_dim,
            self._num_buildings,
            self._hidden_dims,
            n_district=self._n_district,
            n_building_feats=self._n_building_feats,
            building_hidden_dims=self._building_hidden_dims,
        )
        self.ppo_optim = Adam(self.policy.parameters(), lr=self._lr)
        self.rollout_buffer = RolloutBufferV2(
            self.rollout_buffer.num_steps, self._c_dim, self._num_buildings
        )
        self._cached_multipliers = np.ones(self._num_buildings, dtype=np.float32)
        self._cached_action      = np.zeros(self._num_buildings, dtype=np.float32)

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
                self._bc_contexts.append(ctx.copy())
                # Accumulate community import/export/price for BC calibration.
                _idx = _CC_LEVEL2_DISTRICT_FEATURES.index
                dt = self._bc_dt_hours
                imp_kwh = float(ctx[_idx("district__community_import_power_kw")]) * dt
                self._bc_import_samples.append(imp_kwh)
                self._bc_export_samples.append(
                    float(ctx[_idx("district__community_export_power_kw")]) * dt
                )
                self._bc_price_samples.append(
                    float(ctx[_idx("district__electricity_pricing")])
                )
                # Ramp: step-to-step change in community import (kWh).
                ramp_kwh = abs(imp_kwh - self._bc_prev_import_kwh)
                self._bc_ramp_samples.append(ramp_kwh)
                self._bc_prev_import_kwh = imp_kwh
                # Violation: sum charging_constraint_violation_kwh across buildings.
                viol_idx = self._obs_index.get("charging_constraint_violation_kwh")
                total_viol = (
                    sum(float(obs[viol_idx]) for obs in observations)
                    if viol_idx is not None else 0.0
                )
                self._bc_violation_samples.append(total_viol)
                # Compute teacher targets per building (5-term community signal + EV mod).
                teacher_targets = self._bc_teacher_multipliers_per_building(
                    ctx, observations, ramp_kwh=ramp_kwh, violation_kwh=total_viol
                )
                self._bc_targets.append(teacher_targets.copy())
                # Use per-building teacher as cached output.
                self._cached_multipliers = teacher_targets.copy()
                self._cached_community   = ctx
                self._cached_action      = teacher_targets - 1.0
                self._cached_logprob     = 0.0
                self._cached_value       = 0.0
                if len(self._bc_contexts) >= self._bc_collect_steps:
                    self._run_bc_pretraining()
                    self._bc_pretrain_done = True
            else:
                self._sample_new_decision(observations, deterministic)
        return self._cached_multipliers.tolist()

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
        self._accumulated_reward += float(sum(rewards))
        self._step_in_interval   += 1

        if not ((self._step_in_interval >= self._cc_action_interval) or done):
            return

        assert self._cached_community is not None, "predict() must run before update()"

        if not self._bc_pretrain_done:
            self._step_in_interval   = 0
            self._accumulated_reward = 0.0
            if done:
                self._prev_multipliers = None
                self._flush_decision_trace()
            return

        if self._prev_multipliers is None:
            self._prev_multipliers = np.ones(self._num_buildings, dtype=np.float32)

        # Factor and smoothness penalties averaged over buildings.
        factor_penalty     = float(np.mean((self._cached_multipliers - 1.0) ** 2))
        smoothness_penalty = float(np.mean((self._cached_multipliers - self._prev_multipliers) ** 2))
        aux = (
            - self._w_factor     * factor_penalty
            - self._w_smoothness * smoothness_penalty
        )
        self._prev_multipliers = self._cached_multipliers.copy()

        raw = self._accumulated_reward + aux
        self._reward_rms.update(raw)
        scaled = float((raw - self._reward_rms.mean) / max(self._reward_rms.std, 1e-8))

        self.rollout_buffer.add(
            community=self._cached_community,
            action=self._cached_action,
            logprob=self._cached_logprob,
            reward=scaled,
            done=done,
            value=self._cached_value,
        )

        self._step_in_interval   = 0
        self._accumulated_reward = 0.0
        if done:
            self._prev_multipliers = None
            self._flush_decision_trace()

        if self.rollout_buffer.full:
            self._learn_from_rollout(next_observations, done)

    # ──────────────────────── BC warm-start ──────────────────────────────────

    def _community_signal(
        self, ctx: np.ndarray, *, ramp_kwh: float = 0.0, violation_kwh: float = 0.0
    ) -> float:
        """Community-level raw signal — mirrors CCLevel1Agent's 5-term BC teacher.

          cost_signal      = (price - ref_price) / ref_price   (fixed p50 reference)
          peak_signal      = max(0, imp_kWh - target)² / ref_peak
          ramp_signal      = ramp_kwh / ref_ramping
          export_signal    = -(exp_kWh / ref_export)
          violation_signal = violation_kwh
          headroom_signal  = max(0, (ref_headroom - headroom_kw) / ref_headroom)
        """
        _idx = _CC_LEVEL2_DISTRICT_FEATURES.index
        price    = float(ctx[_idx("district__electricity_pricing")])
        price_p1 = float(ctx[_idx("district__electricity_pricing_predicted_1")])
        price_p2 = float(ctx[_idx("district__electricity_pricing_predicted_2")])
        price_p3 = float(ctx[_idx("district__electricity_pricing_predicted_3")])
        imp_kw   = float(ctx[_idx("district__community_import_power_kw")])
        exp_kw   = float(ctx[_idx("district__community_export_power_kw")])
        headroom_kw = float(ctx[_idx("district__community_building_headroom_kw")])
        dt       = self._bc_dt_hours
        imp_kwh  = imp_kw * dt
        exp_kwh  = exp_kw * dt

        # Price vs fixed p50 reference (avoids the rolling-average whitening bug).
        # Until calibration runs (end of collection) fall back to the forecast mean.
        ref_price   = (
            self._bc_reference_price if self._bc_reference_price is not None
            else (price_p1 + price_p2 + price_p3) / 3.0
        )
        cost_signal = (price - ref_price) / max(ref_price, 1e-8)

        # Peak/export skipped until reference values are calibrated (kept None
        # during collection so we never bake in hardcoded community thresholds).
        if self._bc_target_import is not None and self._bc_reference_peak is not None:
            peak_excess = max(0.0, imp_kwh - self._bc_target_import)
            peak_signal = peak_excess ** 2 / self._bc_reference_peak
        else:
            peak_signal = 0.0

        if self._bc_reference_export is not None:
            export_signal = -(exp_kwh / self._bc_reference_export)
        else:
            export_signal = 0.0

        ramp_signal      = ramp_kwh / max(self._bc_reference_ramping, 1e-8)
        violation_signal = violation_kwh
        ref_headroom     = max(self._bc_reference_headroom, 1e-8)
        headroom_signal  = max(0.0, (ref_headroom - headroom_kw) / ref_headroom)

        return (self._bc_w_cost      * cost_signal
                + self._bc_w_peak      * peak_signal
                + self._bc_w_ramp      * ramp_signal
                + self._bc_w_export    * export_signal
                + self._bc_w_violation * violation_signal
                + self._bc_w_headroom  * headroom_signal)

    def _bc_teacher_multipliers_per_building(
        self,
        ctx: np.ndarray,
        observations: List[np.ndarray],
        *,
        ramp_kwh: float = 0.0,
        violation_kwh: float = 0.0,
    ) -> np.ndarray:
        """Compute per-building BC teacher multipliers.

        Mirrors CCRewardLevel2 term structure:
            base       = community signal (cost + peak + ramp + export
                         + violation + headroom — same 5 terms as CCRewardLevel1)
            ev_mod[i]  = -w_ev * urgency[i] * gap[i]
                         (high urgency + large deficit → lower mult to allow charging)

        Building block positions (resolved at attach_environment()):
            [3] connected_state (EV)
            [4] connected_ev_soc_deficit           = max(req - soc, 0) ∈ [0, 1]
            [5] connected_ev_departure_urgency_24h = 1 - hours/24 ∈ [0, 1]
        """
        # During collection (before calibration) the community signal still
        # yields the cost/ramp/violation/headroom terms; peak/export stay 0.
        base = self._community_signal(ctx, ramp_kwh=ramp_kwh, violation_kwh=violation_kwh)
        mults = np.empty(self._num_buildings, dtype=np.float32)
        for i in range(self._num_buildings):
            obs_i = observations[i] if i < len(observations) else None
            if obs_i is not None and i < len(self._building_feat_positions):
                positions = self._building_feat_positions[i]
                ev_conn      = float(obs_i[positions[3]]) if positions[3] >= 0 else 0.0
                soc_def      = float(obs_i[positions[4]]) if positions[4] >= 0 else 0.0
                urgency_24h  = float(obs_i[positions[5]]) if positions[5] >= 0 else 0.0
            else:
                ev_conn, soc_def, urgency_24h = 0.0, 0.0, 0.0

            # The encoded feature connected_ev_departure_urgency_24h = 1 - hours/24.
            # Invert to recover actual hours, then re-apply the same horizon as
            # CCRewardLevel2 (bc_urgency_horizon, default 4 h) so the teacher
            # and the reward function use identical urgency values.
            actual_hours = (1.0 - urgency_24h) * 24.0
            urgency = max(1.0 - actual_hours / self._bc_urgency_horizon, 0.0)

            # EV modulation: mirrors -w_ev * urgency * gap in CCRewardLevel2.
            # Negative sign: high harm → lower multiplier → cheaper price → EV charges.
            ev_mod = -self._bc_w_ev * urgency * soc_def * ev_conn

            raw = float(np.clip((base + ev_mod) * self._bc_mult_scale, -0.8, 0.8))
            mults[i] = float(np.clip(1.0 + raw, self._price_min, self._price_max))
        return mults

    def _run_bc_pretraining(self) -> None:
        """Supervised pretraining of encoder + mean_head against per-building teacher targets."""
        X = np.stack(self._bc_contexts)          # (N_steps, c_dim)
        T = np.stack(self._bc_targets)            # (N_steps, num_buildings)

        # Auto-calibrate reference values from community import/export/price distribution.
        imp_arr   = np.array(self._bc_import_samples, dtype=np.float64)
        exp_arr   = np.array(self._bc_export_samples, dtype=np.float64)
        price_arr = np.array(self._bc_price_samples,  dtype=np.float64)

        if self._bc_target_import is None:
            self._bc_target_import = float(np.percentile(imp_arr, 75))
        if self._bc_reference_peak is None:
            excess_sq = np.maximum(0.0, imp_arr - self._bc_target_import) ** 2
            self._bc_reference_peak = max(float(np.percentile(excess_sq, 90)), 1e-6)
        if self._bc_reference_export is None:
            self._bc_reference_export = max(float(np.percentile(exp_arr, 90)), 1e-6)
        if self._bc_reference_price is None:
            # p50 so ~half the year price is above (expensive) and half below (cheap).
            self._bc_reference_price = max(float(np.percentile(price_arr, 50)), 1e-6)

        logger.info(
            "CC-L2 BC | collected {} contexts | "
            "target_import={:.3f} ref_peak={:.4f} ref_export={:.3f} ref_price={:.4f} "
            "ref_ramping={:.3f} ref_headroom={:.3f} | "
            "w_cost={:.2f} w_peak={:.2f} w_ramp={:.2f} w_export={:.2f} w_violation={:.2f} "
            "w_headroom={:.2f} w_ev={:.2f}",
            len(X),
            self._bc_target_import, self._bc_reference_peak, self._bc_reference_export,
            self._bc_reference_price, self._bc_reference_ramping, self._bc_reference_headroom,
            self._bc_w_cost, self._bc_w_peak, self._bc_w_ramp, self._bc_w_export,
            self._bc_w_violation, self._bc_w_headroom, self._bc_w_ev,
        )

        # Re-compute targets now that reference values are calibrated.
        # Per-building block: [soc, pv, net, ev_conn, soc_deficit, urgency_24h]
        for j in range(len(X)):
            ramp_kwh = self._bc_ramp_samples[j]      if j < len(self._bc_ramp_samples)      else 0.0
            viol_kwh = self._bc_violation_samples[j] if j < len(self._bc_violation_samples) else 0.0
            base = self._community_signal(X[j], ramp_kwh=ramp_kwh, violation_kwh=viol_kwh)
            d_start = _N_DISTRICT
            for i in range(self._num_buildings):
                feat_start = d_start + i * _N_BUILDING_FEATS
                ev_conn     = float(X[j][feat_start + 3])  # connected_state {0,1}
                soc_def     = float(X[j][feat_start + 4])  # soc_deficit [0,1]
                urgency_24h = float(X[j][feat_start + 5])  # departure_urgency_24h [0,1]
                # Recover actual hours from the 24h-horizon encoded feature,
                # then re-apply bc_urgency_horizon (same as CCRewardLevel2).
                actual_hours = (1.0 - urgency_24h) * 24.0
                urgency = max(1.0 - actual_hours / self._bc_urgency_horizon, 0.0)
                ev_mod  = -self._bc_w_ev * urgency * soc_def * ev_conn
                raw = float(np.clip((base + ev_mod) * self._bc_mult_scale, -0.8, 0.8))
                T[j, i] = float(np.clip(1.0 + raw, self._price_min, self._price_max))

        # Convert multiplier targets to pre-tanh space (atanh).
        def to_raw(mult: np.ndarray) -> np.ndarray:
            t = (mult - self._price_min) / (self._price_max - self._price_min) * 2.0 - 1.0
            return np.arctanh(np.clip(t, -0.999, 0.999))

        T_raw = to_raw(T)   # (N_steps, num_buildings)

        X_t   = torch.tensor(X,     dtype=torch.float32)
        T_t   = torch.tensor(T_raw, dtype=torch.float32)

        # Train the actor (district encoder + shared building head) against the
        # per-building teacher targets.
        bc_params = (list(self.policy.district_encoder.parameters())
                     + list(self.policy.building_head.parameters()))
        bc_opt    = Adam(bc_params, lr=self._bc_lr)
        N         = len(X_t)
        losses: List[float] = []

        for step in range(self._bc_train_steps):
            idx_mb = np.random.randint(0, N, size=min(64, N))
            pred   = self.policy.action_means(X_t[idx_mb])  # (batch, num_buildings)
            loss   = (pred - T_t[idx_mb]).pow(2).mean()
            bc_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(bc_params, max_norm=1.0)
            bc_opt.step()
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses))
        logger.info(
            "CC-L2 BC pretraining done | steps={} | loss={:.6f}",
            self._bc_train_steps, mean_loss,
        )
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "CC2/bc_pretrain_loss":              mean_loss,
                    "CC2/bc_pretrain_collect_n":         float(N),
                    "CC2/bc_pretrain_target_import":     self._bc_target_import,
                    "CC2/bc_pretrain_ref_peak":          self._bc_reference_peak,
                    "CC2/bc_pretrain_ref_export":        self._bc_reference_export,
                    "CC2/bc_pretrain_ref_price":         self._bc_reference_price,
                    "CC2/bc_pretrain_ref_ramping":       self._bc_reference_ramping,
                    "CC2/bc_pretrain_ref_headroom":      self._bc_reference_headroom,
                    "CC2/bc_pretrain_w_cost":            self._bc_w_cost,
                    "CC2/bc_pretrain_w_peak":            self._bc_w_peak,
                    "CC2/bc_pretrain_w_ramp":            self._bc_w_ramp,
                    "CC2/bc_pretrain_w_export":          self._bc_w_export,
                    "CC2/bc_pretrain_w_violation":       self._bc_w_violation,
                    "CC2/bc_pretrain_w_headroom":        self._bc_w_headroom,
                    "CC2/bc_pretrain_w_ev":              self._bc_w_ev,
                },
                step=0,
            )

        self._bc_contexts.clear()
        self._bc_targets.clear()
        self._bc_import_samples.clear()
        self._bc_export_samples.clear()
        self._bc_price_samples.clear()
        self._bc_ramp_samples.clear()
        self._bc_violation_samples.clear()
        self._bc_prev_import_kwh = 0.0

    # ───────────────────────── Internal: decision ────────────────────────────

    def _build_context(self, observations: List[np.ndarray]) -> np.ndarray:
        """Build (17 + 6*N,) context vector from all buildings' encoded observations.

        Layout:
            [0:17]          district features (from obs[0])
            [17 : 17+6*N]   per-building features (obs[i] for i in range(N))

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

        return np.concatenate([district] + building_parts)

    def _sample_new_decision(self, observations, deterministic: bool | None) -> None:
        ctx   = self._build_context(observations)
        ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                raw  = self.policy.action_means(ctx_t)    # (1, N)
                _, logprob, _, value = self.policy.get_action_and_value(ctx_t, raw)
            else:
                raw, logprob, _, value = self.policy.get_action_and_value(ctx_t)

        raw_np   = raw.squeeze(0).numpy()                 # (N,)
        squashed = np.tanh(raw_np)
        mults    = self._price_min + (self._price_max - self._price_min) * (squashed + 1.0) / 2.0

        self._cached_action      = raw_np.astype(np.float32)
        self._cached_multipliers = mults.astype(np.float32)
        self._cached_community   = ctx
        self._cached_logprob     = float(logprob.item())
        self._cached_value       = float(value.item())

        self._log_decision()

    # ───────────────────────── Internal: learning ────────────────────────────

    def _learn_from_rollout(self, next_observations, done: bool) -> None:
        ctx = torch.tensor(
            self._build_context(next_observations), dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            _, _, _, last_value = self.policy.get_action_and_value(ctx)
        self.rollout_buffer.compute_gae(
            float(last_value.item()), done, self._gamma, self._gae_lambda
        )
        self._run_ppo_update()
        self.rollout_buffer.reset()

    def _run_ppo_update(self) -> None:
        data = self.rollout_buffer.get()
        community    = data["community"]   # (T, c_dim)
        actions      = data["actions"]     # (T, N)
        old_logprobs = data["logprobs"]    # (T,)
        returns      = data["returns"]     # (T,)
        advantages   = data["advantages"]  # (T,)
        old_values   = torch.tensor(self.rollout_buffer.values, dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_steps = self.rollout_buffer.num_steps
        kl_stop   = False
        approx_kl = 0.0
        pg_loss = v_loss = ent_loss = torch.tensor(0.0)

        for _ in range(self._num_epochs):
            if kl_stop:
                break
            for start in range(0, num_steps, self._mini_batch_size):
                mb = np.random.permutation(num_steps)[start : start + self._mini_batch_size]

                _, new_logprobs, entropy, new_values = self.policy.get_action_and_value(
                    community[mb], actions[mb]
                )
                new_values = new_values.squeeze()

                log_ratio  = new_logprobs - old_logprobs[mb]
                ratio      = torch.exp(log_ratio)
                approx_kl  = ((ratio - 1) - log_ratio).mean().item()
                if self._target_kl is not None and approx_kl > 1.5 * self._target_kl:
                    kl_stop = True
                    break

                mb_adv  = advantages[mb]
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef),
                ).mean()

                v_unclipped = (new_values - returns[mb]) ** 2
                v_clipped   = old_values[mb] + (new_values - old_values[mb]).clamp(
                    -self._clip_coef, self._clip_coef
                )
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - returns[mb]) ** 2).mean()

                ent_loss = entropy.mean()
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
                    "CC2/reward_mean":    float(self._reward_rms.mean),
                    "CC2/reward_std":     float(self._reward_rms.std),
                },
                step=self._ppo_update_count,
            )

    # ──────────────────────── Internal: logging ──────────────────────────────

    def _log_decision(self) -> None:
        ctx   = self._cached_community
        mults = self._cached_multipliers
        _idx  = _CC_LEVEL2_DISTRICT_FEATURES.index

        # Per-building EV state from context blocks [district | b0 | b1 | … | bN]
        # Block layout: [soc, pv, net, ev_conn, soc_def, urgency_24h]
        d = _N_DISTRICT
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
            "mult_mean":       float(mults.mean()),
            "mult_std":        float(mults.std()),
            "mult_min":        float(mults.min()),
            "mult_max":        float(mults.max()),
            "value_est":       self._cached_value,
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

        # Export the actor: full community context → per-building pre-tanh means.
        # (The old export only saved the encoder, which no longer exists and was
        # insufficient for inference anyway.)
        class _ActorWrapper(nn.Module):
            def __init__(self, policy: nn.Module) -> None:
                super().__init__()
                self.policy = policy

            def forward(self, community: torch.Tensor) -> torch.Tensor:
                return self.policy.action_means(community)

        torch.onnx.export(
            _ActorWrapper(self.policy),
            torch.randn(1, self._c_dim),
            str(export_path),
            export_params=True,
            opset_version=DEFAULT_ONNX_OPSET,
            do_constant_folding=True,
            input_names=["community_context"],
            output_names=["per_building_means"],
            dynamic_axes={"community_context": {0: "batch"}, "per_building_means": {0: "batch"}},
        )
        return {
            "format": "onnx",
            "artifacts": [{"agent_index": 0, "path": str(export_path.relative_to(export_root)), "format": "onnx"}],
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
                "reward_rms_n":       self._reward_rms._n,
                "reward_rms_mean":    self._reward_rms._mean,
                "reward_rms_M2":      self._reward_rms._M2,
                "ppo_update_count":   self._ppo_update_count,
                "global_cc_step":     self._global_cc_step,
                "bc_pretrain_done":   self._bc_pretrain_done,
                "bc_target_import":   self._bc_target_import,
                "bc_reference_peak":  self._bc_reference_peak,
                "bc_reference_export": self._bc_reference_export,
                "bc_reference_price": self._bc_reference_price,
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
        self.policy.load_state_dict(ckpt["policy"])
        self.ppo_optim.load_state_dict(ckpt["optimizer"])
        self._reward_rms._n    = ckpt.get("reward_rms_n",    0)
        self._reward_rms._mean = ckpt.get("reward_rms_mean", 0.0)
        self._reward_rms._M2   = ckpt.get("reward_rms_M2",  0.0)
        self._ppo_update_count  = int(ckpt.get("ppo_update_count", 0))
        self._global_cc_step    = int(ckpt.get("global_cc_step", 0))
        if "bc_pretrain_done" in ckpt:
            self._bc_pretrain_done = bool(ckpt["bc_pretrain_done"])
        for key in ("bc_target_import", "bc_reference_peak", "bc_reference_export",
                    "bc_reference_price"):
            if key in ckpt and ckpt[key] is not None:
                setattr(self, f"_{key}", float(ckpt[key]))
        logger.info("CC-L2 checkpoint loaded ← {}", path)
