"""
Community Coordinator — Phase 1 (Global Community Market Maker).

The high-level agent for the HRL pipeline. It observes only community-level
aggregates and emits ONE global economic signal: a community price multiplier.

    multiplier ∈ [price_min, price_max]   (e.g. [0.5, 1.5])

        > 1.0  →  energy is "more expensive"  →  low-level reduces consumption
        < 1.0  →  energy is "cheaper"         →  low-level increases consumption
        = 1.0  →  neutral

The CC does NOT command assets directly. It sets a price the low-level
controllers (e.g. RBCCommunityPolicy) react to as if it were the real price.
This is the "market maker" formulation: the manager steers behaviour through
incentives, not orders.

=============================================================================
DESIGN
=============================================================================
Observations:
    Pre-encoded community vector from the `cc_level1` entity-encoding profile.
    The encoding layer (EntityContractAdapter) already selects, derives, and
    minmax-normalises the features — this agent does NO feature engineering
    and NO normalisation. It consumes the vector as-is.

Action (continuous, single scalar):
    raw ~ Normal(mean(obs), std)               # PPO latent
    multiplier = clip(1.0 + raw, price_min, price_max)

Temporal abstraction:
    CC decides every `cc_action_interval` env steps (default 4 = hourly at
    15-min resolution). The multiplier persists across the interval; the
    low-level acts every step.

Reward:
    Community-level only (CCRewardLevel1: cost + peak + export). The CC sums
    the per-building rewards into one community scalar. The reward reaches the
    CC through the env: multiplier → RBC behaviour → community net → cost.

Training:
    Continuous PPO (Gaussian policy), GAE, reward scaling by running return std.
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

# Ordered community features delivered by the `cc_level1` encoding profile.
# Selecting by name guarantees a stable vector layout regardless of adapter
# ordering. Values are already minmax-normalised by the encoding layer.
_CC_LEVEL1_FEATURES = (
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
_PRICE_FEATURE = "district__electricity_pricing"


class RunningMeanStd:
    """Online mean/variance estimator (Welford). Used for reward scaling only."""

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


class CommunityMarketMakerNet(nn.Module):
    """MLP: community context → Gaussian(price latent) + scalar value."""

    def __init__(self, c_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_d = c_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        self.encoder     = nn.Sequential(*layers)
        self.mean_head   = nn.Linear(in_d, 1)
        self.critic_head = nn.Linear(in_d, 1)
        self.log_std     = nn.Parameter(torch.zeros(1))  # state-independent std

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight,   gain=0.01)  # start near neutral
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def get_action_and_value(
        self,
        community: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h     = self.encoder(community)
        mean  = self.mean_head(h).squeeze(-1)
        value = self.critic_head(h).squeeze(-1)
        # Clamp log_std to [-3, 1] to prevent runaway variance.
        std  = torch.exp(self.log_std.clamp(-3.0, 1.0)).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()  # pre-tanh raw value; stored in rollout buffer
        # Tanh correction: log π(a|s) = log N(raw|μ,σ) − log(1 − tanh²(raw))
        log_prob = dist.log_prob(action) - torch.log(1.0 - torch.tanh(action) ** 2 + 1e-6)
        return action, log_prob, dist.entropy(), value


class RolloutBuffer:
    """Fixed-size rollout buffer for continuous single-action PPO."""

    def __init__(self, num_steps: int, c_dim: int) -> None:
        self.num_steps = num_steps
        self._ptr = 0
        self.full = False
        self.communities = np.zeros((num_steps, c_dim), dtype=np.float32)
        self.actions     = np.zeros(num_steps, dtype=np.float32)
        self.logprobs    = np.zeros(num_steps, dtype=np.float32)
        self.rewards     = np.zeros(num_steps, dtype=np.float32)
        self.dones       = np.zeros(num_steps, dtype=np.float32)
        self.values      = np.zeros(num_steps, dtype=np.float32)
        self.returns     = np.zeros(num_steps, dtype=np.float32)
        self.advantages  = np.zeros(num_steps, dtype=np.float32)

    def add(self, community, action, logprob, reward, done, value) -> None:
        self.communities[self._ptr] = community
        self.actions[self._ptr]     = action
        self.logprobs[self._ptr]    = logprob
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


class CCLevel1Agent(BaseAgent):
    """Phase-1 Community Coordinator: emits a global price multiplier."""

    # Consume the pre-encoded cc_level1 vector (16 features incl. cyclic time).
    # Raw observations would omit the encoder-derived cyclic features
    # (time_of_day_sin/cos, day_type_sin/cos, month_sin/cos, is_weekend),
    # leaving them silently zero — so we MUST use encoded observations.
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

        # Price-multiplier bounds (market-maker action range)
        self._price_min = float(hyper.get("price_min", 0.5))
        self._price_max = float(hyper.get("price_max", 1.5))

        # Auxiliary reward weights (supervisor Phase 1 design)
        self._w_factor     = float(hyper.get("w_factor",     0.3))
        self._w_smoothness = float(hyper.get("w_smoothness", 0.02))
        self._prev_multiplier: float = 1.0

        # Network
        self._c_dim = int(hyper.get("c_dim", len(_CC_LEVEL1_FEATURES)))
        hidden_dims = hyper.get("hidden_dims", [128, 128])
        self.policy = CommunityMarketMakerNet(self._c_dim, hidden_dims)
        # Network
        self._c_dim = int(hyper.get("c_dim", len(_CC_LEVEL1_FEATURES)))
        hidden_dims = hyper.get("hidden_dims", [128, 128])
        self.policy = CommunityMarketMakerNet(self._c_dim, hidden_dims)
        self.ppo_optim = Adam(self.policy.parameters(), lr=float(hyper.get("lr", 1e-4)))

        # Reward normalisation: track mean + std of RAW step rewards (not returns).
        # Centering around zero ensures the value function sees both positive and
        # negative targets instead of a constant negative signal, preventing it
        # from collapsing to a trivial solution (v_loss → 0).
        self._reward_rms = RunningMeanStd()

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(int(hyper.get("num_steps", 96)), self._c_dim)
        self._ppo_update_count = 0

        # Temporal abstraction
        self._cc_action_interval = int(hyper.get("cc_action_interval", 4))
        self._step_in_interval = 0

        # Cached decision
        self._cached_multiplier: float = 1.0
        self._cached_action:     float = 0.0          # PPO latent (raw Gaussian sample)
        self._cached_community:  Optional[np.ndarray] = None
        self._cached_price:      float = 0.0
        self._cached_logprob:    float = 0.0
        self._cached_value:      float = 0.0
        self._accumulated_reward: float = 0.0

        # Behaviour-cloning warm-start (Option A: pretrain before any RL).
        # Mirrors the MADDPG actor_offline_bc_pretrain_* pattern.
        self._bc_enabled      = bool(hyper.get("bc_pretrain_enabled", False))
        self._bc_collect_steps = int(hyper.get("bc_collect_steps", 336))   # CC decisions to collect
        self._bc_train_steps   = int(hyper.get("bc_train_steps",   2000))
        self._bc_lr            = float(hyper.get("bc_lr",           1e-3))
        self._bc_pretrain_done: bool = not self._bc_enabled  # skip if disabled
        self._bc_contexts: List[np.ndarray] = []
        # Sample buffers for auto-calibration of reference values.
        # Populated during BC collection; cleared after pretraining.
        self._bc_import_samples: List[float] = []
        self._bc_export_samples: List[float] = []
        self._bc_price_samples:  List[float] = []
        # Fixed thresholds (kept for backward-compat / logging; no longer drive teacher).
        self._bc_price_p20 = float(hyper.get("bc_price_p20", 0.10))
        self._bc_price_p80 = float(hyper.get("bc_price_p80", 0.21))
        # Multi-signal teacher parameters — mirror CCRewardLevel1 term weights.
        self._bc_w_cost          = float(hyper.get("bc_w_cost",          1.0))
        self._bc_w_peak          = float(hyper.get("bc_w_peak",          0.6))
        self._bc_w_export        = float(hyper.get("bc_w_export",        0.05))
        self._bc_w_ramp          = float(hyper.get("bc_w_ramp",          0.4))
        self._bc_w_violation     = float(hyper.get("bc_w_violation",     2.0))
        # Community oversight: proactively raise the multiplier as the shared
        # electrical-service headroom shrinks, so workers conserve BEFORE the
        # community trips its three-phase limit. This is the CC analogue of
        # RBCCommunityPolicy's community-stress backoff — the per-building Smart
        # RBC stays selfish, the CC watches the community as a whole.
        self._bc_w_headroom      = float(hyper.get("bc_w_headroom",      1.0))
        # kW margin below which the CC starts pushing prices up (mirrors the
        # RBC's low_headroom_threshold_kw). Stress ramps 0→1 as headroom falls
        # from this threshold to 0, and exceeds 1 once headroom goes negative.
        self._bc_reference_headroom = float(hyper.get("bc_reference_headroom", 2.0))
        # Reference values for the BC teacher (peak/export thresholds).
        # None = auto-calibrate from episode-0 data (p75/p90 of observed distribution).
        # Set explicitly in config only to force fixed community-specific values.
        self._bc_dt_hours            = float(hyper.get("bc_dt_hours", 0.25))
        self._bc_target_import       = hyper.get("bc_target_import",    None)  # kWh p75, auto if None
        self._bc_reference_peak      = hyper.get("bc_reference_peak",   None)  # kWh² p90, auto if None
        self._bc_reference_export    = hyper.get("bc_reference_export", None)  # kWh p90, auto if None
        self._bc_reference_ramping   = float(hyper.get("bc_reference_ramping", 1.878))  # p90 step-to-step Δimport
        self._bc_reference_price     = hyper.get("bc_reference_price",  None)  # €/kWh p50, auto if None
        # Scaling factor applied to the combined raw signal before clipping to ±0.8.
        self._bc_mult_scale      = float(hyper.get("bc_mult_scale",      1.0))
        # State for ramp/violation tracking during BC collection.
        self._bc_prev_import_kwh: float = 0.0
        self._bc_ramp_samples: List[float] = []
        self._bc_violation_samples: List[float] = []

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
        # Only the first building's vector is needed — district features are
        # identical across buildings in entity mode.
        self._obs_index = {n: i for i, n in enumerate(observation_names[0])}

    # ───────────────────────── Per-step interaction ──────────────────────────

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> float:
        """Return the global price multiplier (Pipeline context for low-level)."""
        if self._step_in_interval == 0:
            if not self._bc_pretrain_done:
                # BC collection phase: store context, act with teacher.
                ctx = self._build_context(observations)
                self._bc_contexts.append(ctx.copy())
                # Accumulate import/export samples for auto-calibration of reference values.
                _idx = _CC_LEVEL1_FEATURES.index
                imp_kwh = float(ctx[_idx("district__community_import_power_kw")]) * self._bc_dt_hours
                self._bc_import_samples.append(imp_kwh)
                self._bc_export_samples.append(
                    float(ctx[_idx("district__community_export_power_kw")]) * self._bc_dt_hours
                )
                self._bc_price_samples.append(
                    float(ctx[_idx("district__electricity_pricing")])
                )
                # Ramp: step-to-step change in community import.
                ramp_kwh = abs(imp_kwh - self._bc_prev_import_kwh)
                self._bc_ramp_samples.append(ramp_kwh)
                self._bc_prev_import_kwh = imp_kwh
                # Violation: sum charging_constraint_violation_kwh across buildings.
                viol_idx = self._obs_index.get("charging_constraint_violation_kwh")
                total_viol = sum(float(obs[viol_idx]) for obs in observations) if viol_idx is not None else 0.0
                self._bc_violation_samples.append(total_viol)
                # Compute teacher multiplier using full context (5-term logic).
                price_feat = float(ctx[_CC_LEVEL1_FEATURES.index(_PRICE_FEATURE)])  # kept for diagnostics
                self._cached_multiplier = self._bc_teacher_multiplier(ctx, ramp_kwh=ramp_kwh, violation_kwh=total_viol)
                self._cached_community  = ctx
                self._cached_price      = price_feat
                # Dummy PPO fields — not used in rollout buffer during BC.
                self._cached_action   = self._cached_multiplier - 1.0
                self._cached_logprob  = 0.0
                self._cached_value    = 0.0
                self._log_decision()
                # Trigger pretraining once enough contexts are collected.
                if len(self._bc_contexts) >= self._bc_collect_steps:
                    self._run_bc_pretraining()
                    self._bc_pretrain_done = True
            else:
                self._sample_new_decision(observations, deterministic)
        return self._cached_multiplier

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
        """Accumulate community reward over the interval, then push a transition."""
        done = terminated or truncated
        self._accumulated_reward += float(sum(rewards))
        self._step_in_interval += 1

        if not ((self._step_in_interval >= self._cc_action_interval) or done):
            return

        assert self._cached_community is not None, "predict() must run before update()"

        # BC collection phase: flush interval state but do NOT add to rollout buffer.
        if not self._bc_pretrain_done:
            self._step_in_interval = 0
            self._accumulated_reward = 0.0
            if done:
                self._prev_multiplier = 1.0
                self._flush_decision_trace()
            return

        # Auxiliary penalties on the CC action (supervisor Phase 1 design).
        # Applied once per decision interval, not per env step.
        factor_penalty     = (self._cached_multiplier - 1.0) ** 2
        smoothness_penalty = (self._cached_multiplier - self._prev_multiplier) ** 2
        aux = (
            - self._w_factor     * factor_penalty
            - self._w_smoothness * smoothness_penalty
        )
        self._prev_multiplier = self._cached_multiplier

        raw = self._accumulated_reward + aux
        # Centre and scale by running mean/std of raw rewards so the value
        # function sees targets distributed around zero (some positive, some
        # negative) rather than a uniformly negative constant signal.
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

        self._step_in_interval = 0
        self._accumulated_reward = 0.0
        if done:
            self._prev_multiplier = 1.0
            self._flush_decision_trace()

        if self.rollout_buffer.full:
            self._learn_from_rollout(next_observations, done)

    # ──────────────────────── BC warm-start ──────────────────────────────────

    def _bc_teacher_multiplier(self, ctx: np.ndarray, *, ramp_kwh: float = 0.0, violation_kwh: float = 0.0) -> float:
        """Multi-signal teacher mirroring CCRewardLevel1 (cost + peak + ramp + export + violation).

        Features arrive as raw kW (NOT normalised to [0,1]).
        Convert to kWh per step via bc_dt_hours, then apply the same
        formulas as CCRewardLevel1 so BC and RL targets are consistent.

          cost_signal      = (price - ref_price) / ref_price   # relative ∈ [-1, +1]
          peak_signal      = max(0, imp_kWh - target)² / ref_peak
          ramp_signal      = ramp_kwh / ref_ramping            # ≈ [0, 3]
          export_signal    = -(exp_kWh / ref_export)           # ≈ [-1, 0]
          violation_signal = violation_kwh / 1.0               # kWh of violations
          headroom_signal  = max(0, (ref_headroom - headroom_kw) / ref_headroom)  # ≥ 0

          raw  = w_cost*cost + w_peak*peak + w_ramp*ramp + w_export*export
                 + w_violation*violation + w_headroom*headroom
          mult = clip(1.0 + raw * scale, price_min, price_max)

        The headroom_signal is the CC's community-oversight reflex: as the shared
        electrical-service margin shrinks, the multiplier rises so workers
        conserve BEFORE a violation occurs (proactive), while violation_signal
        reacts once a limit is already breached.
        """
        _idx = _CC_LEVEL1_FEATURES.index
        price    = float(ctx[_idx("district__electricity_pricing")])
        price_p1 = float(ctx[_idx("district__electricity_pricing_predicted_1")])
        price_p2 = float(ctx[_idx("district__electricity_pricing_predicted_2")])
        price_p3 = float(ctx[_idx("district__electricity_pricing_predicted_3")])
        imp_kw   = float(ctx[_idx("district__community_import_power_kw")])
        exp_kw   = float(ctx[_idx("district__community_export_power_kw")])
        headroom_kw = float(ctx[_idx("district__community_building_headroom_kw")])

        # Convert power (kW) → energy (kWh) for this timestep
        dt      = self._bc_dt_hours
        imp_kwh = imp_kw * dt
        exp_kwh = exp_kw * dt

        # Price vs fixed p50 reference — same percentile approach as every other signal.
        # Using a rolling 3-step average whitens the signal to near-zero (prices vary slowly).
        # A fixed annual median gives real variance: above median = expensive, below = cheap.
        ref_price   = self._bc_reference_price if self._bc_reference_price is not None else (price_p1 + price_p2 + price_p3) / 3.0
        cost_signal = (price - ref_price) / max(ref_price, 1e-8)

        # Peak import — identical formula to CCRewardLevel1.
        # During the collection phase the reference values are still None
        # (auto-calibration runs at the end of collection). Until then the
        # peak/export terms are skipped and the teacher uses the cost signal
        # only — this avoids hardcoded community thresholds while still
        # collecting a representative trajectory.
        if self._bc_target_import is not None and self._bc_reference_peak is not None:
            peak_excess  = max(0.0, imp_kwh - self._bc_target_import)
            peak_signal  = peak_excess ** 2 / self._bc_reference_peak
        else:
            peak_signal = 0.0

        # Export — identical formula to CCRewardLevel1
        if self._bc_reference_export is not None:
            export_signal = -(exp_kwh / self._bc_reference_export)
        else:
            export_signal = 0.0

        # Ramping — positive when import just jumped, signals "price expensive"
        ramp_signal = ramp_kwh / max(self._bc_reference_ramping, 1e-8)

        # Violation — strong positive signal when grid limits breached
        violation_signal = violation_kwh  # reference is 1.0 kWh (same as CCRewardLevel1)

        # Community headroom — proactive oversight. Stress ramps 0→1 as the
        # shared electrical-service margin falls from ref_headroom to 0, and
        # exceeds 1 once headroom is negative (already over the limit). Positive
        # stress pushes the multiplier UP so workers conserve before a violation.
        ref_headroom = max(self._bc_reference_headroom, 1e-8)
        headroom_signal = max(0.0, (ref_headroom - headroom_kw) / ref_headroom)

        raw = (self._bc_w_cost      * cost_signal
               + self._bc_w_peak      * peak_signal
               + self._bc_w_ramp      * ramp_signal
               + self._bc_w_export    * export_signal
               + self._bc_w_violation * violation_signal
               + self._bc_w_headroom  * headroom_signal)

        mult = 1.0 + float(np.clip(raw * self._bc_mult_scale, -0.8, 0.8))
        logger.debug(
            "BC teacher | price={:.3f} ref={:.3f} imp_kWh={:.3f} exp_kWh={:.3f} headroom_kW={:.3f} "
            "cost={:.3f} peak={:.3f} ramp={:.3f} exp_sig={:.3f} viol={:.3f} headroom_sig={:.3f} raw={:.3f} mult={:.3f}",
            price, ref_price, imp_kwh, exp_kwh, headroom_kw,
            cost_signal, peak_signal, ramp_signal, export_signal, violation_signal, headroom_signal, raw,
            float(np.clip(mult, self._price_min, self._price_max)),
        )
        return float(np.clip(mult, self._price_min, self._price_max))

    def _run_bc_pretraining(self) -> None:
        """Supervised pretraining of encoder + mean_head against teacher targets.

        Mirrors MADDPG._maybe_run_actor_offline_bc_pretraining: same Adam +
        grad-clip-1.0 pattern, same MLflow metric keys (prefixed CC/bc_*).
        """
        X = np.stack(self._bc_contexts)                        # [N, feat_dim]
        price_idx = _CC_LEVEL1_FEATURES.index(_PRICE_FEATURE)
        prices = X[:, price_idx]                               # kept for corr diagnostic

        # Auto-calibrate reference values from the collected distribution.
        # This makes the teacher community-agnostic: thresholds are derived from
        # what was actually observed rather than hardcoded community statistics.
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
            "BC | collected {} contexts | "
            "target_import={:.3f} ref_peak={:.4f} ref_export={:.3f} ref_ramping={:.3f} ref_price={:.4f} ref_headroom={:.3f} | "
            "w_cost={:.2f} w_peak={:.2f} w_ramp={:.2f} w_export={:.2f} w_violation={:.2f} w_headroom={:.2f}",
            len(X),
            self._bc_target_import, self._bc_reference_peak, self._bc_reference_export,
            self._bc_reference_ramping, self._bc_reference_price, self._bc_reference_headroom,
            self._bc_w_cost, self._bc_w_peak, self._bc_w_ramp, self._bc_w_export, self._bc_w_violation,
            self._bc_w_headroom,
        )

        # Teacher targets: invert the tanh squash so BC trains in pre-tanh space.
        #   multiplier = price_min + (price_max - price_min) * (tanh(raw) + 1) / 2
        #   → raw = atanh(clamp((mult - price_min) / (price_max - price_min) * 2 - 1))
        # Clamp to ±0.999 to avoid atanh(±1) = ±inf at the hard boundaries.
        def _teacher_raw(ctx_row: np.ndarray, i: int) -> float:
            ramp_kwh    = self._bc_ramp_samples[i]      if i < len(self._bc_ramp_samples)      else 0.0
            viol_kwh    = self._bc_violation_samples[i] if i < len(self._bc_violation_samples) else 0.0
            mult = self._bc_teacher_multiplier(ctx_row, ramp_kwh=ramp_kwh, violation_kwh=viol_kwh)
            t = (mult - self._price_min) / (self._price_max - self._price_min) * 2.0 - 1.0
            return float(np.arctanh(np.clip(t, -0.999, 0.999)))

        targets = np.array([_teacher_raw(X[i], i) for i in range(len(X))], dtype=np.float32)

        X_t = torch.tensor(X,       dtype=torch.float32)
        T_t = torch.tensor(targets, dtype=torch.float32)

        bc_params = list(self.policy.encoder.parameters()) + list(self.policy.mean_head.parameters())
        bc_opt    = Adam(bc_params, lr=self._bc_lr)
        N         = len(X_t)
        losses: List[float] = []

        for step in range(self._bc_train_steps):
            idx    = np.random.randint(0, N, size=min(64, N))
            h      = self.policy.encoder(X_t[idx])
            pred   = self.policy.mean_head(h).squeeze(-1)
            loss   = (pred - T_t[idx]).pow(2).mean()
            bc_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(bc_params, max_norm=1.0)
            bc_opt.step()
            losses.append(float(loss.item()))

        # Validation metrics.
        with torch.no_grad():
            h    = self.policy.encoder(X_t)
            pred = self.policy.mean_head(h).squeeze(-1).numpy()
        pred_mults = self._price_min + (self._price_max - self._price_min) * (np.tanh(pred) + 1.0) / 2.0
        corr = (
            float(np.corrcoef(prices, pred_mults)[0, 1])
            if prices.std() > 1e-6 and pred_mults.std() > 1e-6
            else float("nan")
        )
        mean_loss = float(np.mean(losses))

        logger.info(
            "BC pretraining done | steps={} | loss={:.6f} | corr(price,mult)={:+.3f}",
            self._bc_train_steps, mean_loss, corr,
        )
        if mlflow.active_run():
            metrics: dict = {
                "CC/bc_pretrain_loss":             mean_loss,
                "CC/bc_pretrain_collect_n":        float(N),
                # Calibrated reference values — these reflect the actual community,
                # not hardcoded constants, so they vary across deployments.
                "CC/bc_pretrain_target_import":    self._bc_target_import,
                "CC/bc_pretrain_reference_peak":   self._bc_reference_peak,
                "CC/bc_pretrain_reference_export": self._bc_reference_export,
                "CC/bc_pretrain_w_cost":           self._bc_w_cost,
                "CC/bc_pretrain_w_peak":           self._bc_w_peak,
                "CC/bc_pretrain_w_ramp":           self._bc_w_ramp,
                "CC/bc_pretrain_w_export":         self._bc_w_export,
                "CC/bc_pretrain_w_violation":      self._bc_w_violation,
                "CC/bc_pretrain_w_headroom":       self._bc_w_headroom,
                "CC/bc_pretrain_reference_ramping": self._bc_reference_ramping,
                "CC/bc_pretrain_reference_price":   self._bc_reference_price,
                "CC/bc_pretrain_reference_headroom": self._bc_reference_headroom,
            }
            if not np.isnan(corr):
                metrics["CC/bc_pretrain_corr_price_mult"] = corr
            mlflow.log_metrics(metrics, step=0)

        # Free collected data.
        self._bc_contexts.clear()
        self._bc_import_samples.clear()
        self._bc_export_samples.clear()
        self._bc_ramp_samples.clear()
        self._bc_violation_samples.clear()
        self._bc_price_samples.clear()
        self._bc_prev_import_kwh = 0.0

    # ───────────────────────── Internal: decision ────────────────────────────

    def _build_context(self, observations: List[np.ndarray]) -> np.ndarray:
        """Select the cc_level1 feature vector (already normalised by encoding)."""
        obs0 = observations[0]
        idx = self._obs_index
        return np.array(
            [float(obs0[idx[name]]) if name in idx else 0.0 for name in _CC_LEVEL1_FEATURES],
            dtype=np.float32,
        )

    def _sample_new_decision(self, observations, deterministic: bool | None) -> None:
        ctx = self._build_context(observations)
        ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                h = self.policy.encoder(ctx_t)
                raw = self.policy.mean_head(h).squeeze(-1)
                # Delegate to get_action_and_value so the tanh correction is included.
                _, logprob, _, value = self.policy.get_action_and_value(ctx_t, raw)
            else:
                raw, logprob, _, value = self.policy.get_action_and_value(ctx_t)

        raw_f = float(raw.item())
        # Map pre-tanh raw → [price_min, price_max] without a hard clip.
        # tanh(raw) ∈ (-1, 1), so bounds are approached asymptotically and
        # the gradient is always non-zero — preventing the one-way drift.
        squashed = float(np.tanh(raw_f))
        self._cached_action     = raw_f   # store pre-tanh for PPO re-evaluation
        self._cached_multiplier = float(
            self._price_min + (self._price_max - self._price_min) * (squashed + 1.0) / 2.0
        )
        self._cached_community  = ctx
        self._cached_price      = float(ctx[_CC_LEVEL1_FEATURES.index(_PRICE_FEATURE)])
        self._cached_logprob    = float(logprob.item())
        self._cached_value      = float(value.item())

        self._log_decision()

    # ───────────────────────── Internal: learning ────────────────────────────

    def _learn_from_rollout(self, next_observations, done: bool) -> None:
        ctx = torch.tensor(self._build_context(next_observations), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, _, _, last_value = self.policy.get_action_and_value(ctx)
        self.rollout_buffer.compute_gae(
            float(last_value.item()), done, self._gamma, self._gae_lambda
        )
        self._run_ppo_update()
        self.rollout_buffer.reset()

    def _run_ppo_update(self) -> None:
        data = self.rollout_buffer.get()
        community, actions = data["community"], data["actions"]
        old_logprobs, returns, advantages = data["logprobs"], data["returns"], data["advantages"]
        old_values = torch.tensor(self.rollout_buffer.values, dtype=torch.float32)

        # Advantage normalisation
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_steps = self.rollout_buffer.num_steps
        kl_stop = False
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

                log_ratio = new_logprobs - old_logprobs[mb]
                ratio = torch.exp(log_ratio)
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                if self._target_kl is not None and approx_kl > 1.5 * self._target_kl:
                    kl_stop = True
                    break

                mb_adv = advantages[mb]
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef),
                ).mean()

                v_unclipped = (new_values - returns[mb]) ** 2
                v_clipped = old_values[mb] + (new_values - old_values[mb]).clamp(
                    -self._clip_coef, self._clip_coef
                )
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - returns[mb]) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss + self._vf_coef * v_loss - self._ent_coef * ent_loss

                self.ppo_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._max_grad_norm)
                self.ppo_optim.step()

        self._ppo_update_count += 1
        logger.info(
            "CC PPO | pg={:.4f} v={:.4f} ent={:.4f} kl_stop={}",
            pg_loss.item(), v_loss.item(), ent_loss.item(), kl_stop,
        )
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "PPO_pg_loss":      pg_loss.item(),
                    "PPO_v_loss":       v_loss.item(),
                    "PPO_entropy":      ent_loss.item(),
                    "PPO_kl_stop":      float(kl_stop),
                    # Reward distribution — should be centred near 0 with std ≈ 1
                    "PPO_reward_mean":  float(self._reward_rms.mean),
                    "PPO_reward_std":   float(self._reward_rms.std),
                },
                step=self._ppo_update_count,
            )

    # ──────────────────────── Internal: logging ──────────────────────────────

    def _log_decision(self) -> None:
        ctx = self._cached_community
        _idx = _CC_LEVEL1_FEATURES.index
        self._decision_trace.append(
            {
                "timestep":    self._global_cc_step,
                "cc_step":     self._global_cc_step,
                "price":       self._cached_price,
                "multiplier":  self._cached_multiplier,
                "value_est":   self._cached_value,
                # Community power (normalised by encoding layer — used for relative KPIs only)
                "import_norm": float(ctx[_idx("district__community_import_power_kw")]),
                "pv_norm":     float(ctx[_idx("district__community_pv_power_kw")]),
                "carbon_norm": float(ctx[_idx("district__carbon_intensity")]),
            }
        )
        self._global_cc_step += 1

    def _flush_decision_trace(self) -> None:
        if not self._decision_trace:
            return
        self._episode_count += 1
        ep = self._episode_count
        fields = list(self._decision_trace[0].keys())

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=f"cc_ep{ep}_", delete=False
        ) as f:
            tmp_path = f.name
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self._decision_trace)

        imports = np.array([r["import_norm"] for r in self._decision_trace], dtype=np.float64)
        pvs     = np.array([r["pv_norm"]     for r in self._decision_trace], dtype=np.float64)
        carbons = np.array([r["carbon_norm"] for r in self._decision_trace], dtype=np.float64)

        # Load factor: ratio within the same (normalised) variable — scale cancels out.
        load_factor = float(imports.mean() / imports.max()) if imports.max() > 1e-6 else float("nan")
        # Carbon-weighted import: relative index (both normalised; directionally valid).
        carbon_import = float(np.mean(imports * carbons))
        # Self-sufficiency: import/(import+pv) — valid only when both share comparable scale.
        denom = float(imports.sum() + pvs.sum())
        self_sufficiency = 1.0 - float(imports.sum()) / denom if denom > 1e-6 else float("nan")
        peak_import = float(imports.max())

        logger.info(
            "CC ep{} | {} decisions | self_suff={:.3f} load_factor={:.3f} "
            "carbon_import={:.4f} peak_import={:.3f}",
            ep, len(imports), self_sufficiency, load_factor, carbon_import, peak_import,
        )
        if mlflow.active_run():
            mlflow.log_artifact(tmp_path, artifact_path="decision_traces")
            metrics = {
                "CC_self_sufficiency": self_sufficiency if not np.isnan(self_sufficiency) else 0.0,
                "CC_load_factor":      load_factor if not np.isnan(load_factor) else 0.0,
                "CC_carbon_import":    carbon_import,
                "CC_peak_import":      peak_import,
            }
            mlflow.log_metrics(metrics, step=ep)

        self._decision_trace = []

    # ───────────────────────── Lifecycle / artifacts ─────────────────────────

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return True

    def export_artifacts(self, output_dir, context=None):
        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        export_path = onnx_dir / "cc_market_maker.onnx"
        torch.onnx.export(
            self.policy.encoder,
            torch.randn(1, self._c_dim),
            str(export_path),
            export_params=True,
            opset_version=DEFAULT_ONNX_OPSET,
            do_constant_folding=True,
            input_names=["community_context"],
            output_names=["hidden"],
            dynamic_axes={"community_context": {0: "batch"}, "hidden": {0: "batch"}},
        )
        return {
            "format": "onnx",
            "artifacts": [{"agent_index": 0, "path": str(export_path.relative_to(export_root)), "format": "onnx"}],
        }

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        path = Path(output_dir) / f"cc_step_{step}.pt"
        torch.save(
            {
                "step":               step,
                "policy":             self.policy.state_dict(),
                "optimizer":          self.ppo_optim.state_dict(),
                "reward_rms_n":       self._reward_rms._n,
                "reward_rms_mean":    self._reward_rms._mean,
                "reward_rms_M2":      self._reward_rms._M2,
                "ppo_update_count":   self._ppo_update_count,
                "global_cc_step":     self._global_cc_step,
                "bc_pretrain_done":   self._bc_pretrain_done,
            },
            path,
        )
        logger.info("CC checkpoint saved → {}", path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        root = Path(checkpoint_path)
        if root.is_dir():
            cands = sorted(root.glob("cc_step_*.pt"), key=lambda p: p.stat().st_mtime)
            if not cands:
                raise FileNotFoundError(f"No CC checkpoint in {root}")
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
        # On resume: if BC was completed before checkpoint, don't re-run it.
        if "bc_pretrain_done" in ckpt:
            self._bc_pretrain_done = bool(ckpt["bc_pretrain_done"])
        logger.info("CC checkpoint loaded ← {}", path)
