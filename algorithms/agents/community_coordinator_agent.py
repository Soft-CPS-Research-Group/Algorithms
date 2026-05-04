"""
Community Coordinator (CC) Agent.

The CC is the *manager* in a hierarchical multi-agent setup:

    ┌────────────────────────────────────────────────────────────┐
    │  CommunityCoordinatorAgent (this file)                     │
    │  ─ neural policy that outputs ONE scalar action `o1` ∈     │
    │    (-1, 1) every CC decision                               │
    │  ─ learned with PPO                                        │
    └────────────────────────────────────────────────────────────┘
                              │  o1 (community-wide signal)
                              ▼
    ┌────────────────────────────────────────────────────────────┐
    │  Rule-Based Controller (RBC) — _rbc_act()                  │
    │  ─ fixed (NOT learned) per-building translator             │
    │  ─ takes o1 + per-building observation → action vector     │
    │    for that building (battery + EV chargers + appliances)  │
    └────────────────────────────────────────────────────────────┘
                              │  per-building actions
                              ▼
                          CityLearn env

Reinforcement-learning glossary used below
    state          : numerical summary of the environment the CC sees.
    action (o1)    : scalar in (-1, 1). Positive = charge batteries.
                     Negative = discharge.
    reward         : scalar feedback per env step (e.g. -cost).
    rollout buffer : a fixed-size queue of (state, action, reward …)
                     tuples. PPO needs many tuples before each update
                     because it estimates gradients from a batch.
    actor-critic   : two networks sharing inputs.
                       - actor:  state → action distribution
                       - critic: state → expected return (value)
    GAE            : Generalized Advantage Estimation. Smooths the
                     reward signal into "advantages" used by the actor.
    PPO            : Proximal Policy Optimization. Trust-region style
                     update that clips how much the new policy can
                     differ from the old one.
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
from algorithms.utils.ppo import PPOActorCritic, RolloutBuffer


class RunningMeanStd:
    """
    Online estimator of mean and variance, vector or scalar.

    Implements Welford's algorithm: numerically stable, O(1) per update,
    no need to keep the full history. Used in two places:

      1. State normalization. Neural networks train poorly when input
         features have very different scales (e.g. price ~0.5 vs net
         power ~30). We track running mean/std per feature and feed the
         network `(x - mean) / std` instead of raw `x`.
      2. Reward scaling. PPO's combined loss mixes pg_loss, v_loss, and
         entropy. If returns are huge, v_loss dominates and the actor
         barely trains. We track the std of running discounted returns
         and divide rewards by it before storing in the rollout buffer.

    `shape` is the feature dimension (e.g. 9 for the CC state).
    `shape=()` (the default) makes it a scalar tracker.
    """

    def __init__(self, shape: tuple = ()) -> None:
        self._n: int          = 0
        self._mean: np.ndarray = np.zeros(shape, dtype=np.float64)
        self._M2:  np.ndarray = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray | float) -> None:
        x = np.asarray(x, dtype=np.float64)
        self._n += 1
        delta  = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def std(self) -> np.ndarray:
        if self._n < 2:
            return np.ones_like(self._mean)
        return np.sqrt(np.maximum(self._M2 / self._n, 1e-12))


class CommunityCoordinatorAgent(BaseAgent):
    """PPO-trained manager that outputs a single community-level signal."""

    # ─────────────────────────── Construction ───────────────────────────

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        # The wrapper reads this flag to decide which observations to pass.
        # `True` = give us raw per-building observations from CityLearn.
        self.use_raw_observations = True

        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})

        # ── PPO hyperparameters ────────────────────────────────────────
        # `gamma` (discount) and `gae_lambda` shape how rewards turn into
        # returns/advantages. `clip_coef` bounds the PPO policy update.
        # `vf_coef` / `ent_coef` weigh the critic loss / entropy bonus
        # inside the combined PPO loss. `target_kl` early-stops PPO epochs
        # if the new policy drifts too far from the old one.
        self._gamma           = hyper.get("gamma",           0.99)
        self._gae_lambda      = hyper.get("gae_lambda",      0.95)
        self._num_epochs      = hyper.get("num_epochs",      10)
        self._mini_batch_size = hyper.get("mini_batch_size", 64)
        self._clip_coef       = hyper.get("clip_coef",       0.2)
        self._vf_coef         = hyper.get("vf_coef",         0.5)
        self._ent_coef        = hyper.get("ent_coef",        0.01)
        self._max_grad_norm   = hyper.get("max_grad_norm",   0.5)
        self._target_kl       = hyper.get("target_kl",       0.02)

        # ── Network + optimiser ────────────────────────────────────────
        # `obs_dim` MUST equal the length of the list returned by
        # `_build_community_state`. The actor-critic outputs one scalar
        # action; that is the `1` below.
        obs_dim = hyper.get("obs_dim", 9)
        self._obs_dim       = obs_dim
        self.actor_critic   = PPOActorCritic(obs_dim, 1)
        self.ppo_optim      = Adam(self.actor_critic.parameters(), lr=hyper.get("lr"))

        # ── Input / reward normalization ───────────────────────────────
        # Both stats are updated as data streams in (online), then used to
        # normalize the state fed to the network (`obs_rms`) and the reward
        # written to the buffer (`ret_rms`). See RunningMeanStd docstring.
        self._obs_rms       = RunningMeanStd(shape=(obs_dim,))
        self._ret_rms       = RunningMeanStd()
        self._return_running: float = 0.0   # discounted-return accumulator

        # PPO collects `num_steps` transitions before each gradient update.
        # Larger = more stable but slower wall-clock.
        self.rollout_buffer = RolloutBuffer(hyper.get("num_steps"), obs_dim, 1)
        self._ppo_update_count = 0

        # ── Temporal abstraction (HRL idea) ─────────────────────────────
        # If `cc_action_interval` (K) > 1, the CC re-decides only every K
        # env steps and its action is held constant in between. The reward
        # for that one CC decision is the SUM of the K env-step rewards.
        # K=1 disables abstraction (CC acts every env step).
        self._cc_action_interval: int   = hyper.get("cc_action_interval", 1)
        self._step_in_interval:   int   = 0
        self._cached_o1:          float = 0.0
        self._cached_state:       Optional[np.ndarray] = None
        self._cached_logprob:     float = 0.0
        self._cached_value:       float = 0.0
        self._accumulated_reward: float = 0.0

        # ── Decision trace (diagnostic only) ────────────────────────────
        # One row per CC decision: state + action + value estimate.
        # Flushed to CSV + MLflow at episode end so we can plot whether
        # the CC actually learned to charge cheap / discharge expensive.
        self._episode_count:  int        = 0
        self._global_cc_step: int        = 0
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
        """Cache name → index lookups so we can read observations by name."""
        self._obs_index    = [{n: i for i, n in enumerate(ns)} for ns in observation_names]
        self._action_index = [{n: i for i, n in enumerate(ns)} for ns in action_names]
        self._action_dims  = [len(ns) for ns in action_names]

    # ─────────────────────── Per-step interaction ───────────────────────

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        """
        Called once per env step. Returns a per-building action vector.

        At the START of a CC interval (every K env steps) we run the
        actor-critic to sample a fresh `o1`. For the remaining K-1 steps
        we re-use the cached `o1`. Either way, the RBC translates `o1` +
        the per-building observation into the action vector CityLearn
        expects.
        """
        if self._step_in_interval == 0:
            self._sample_new_decision(observations)

        return [self._rbc_act(observations[i], i, self._cached_o1)
                for i in range(len(observations))]

    def update(
        self,
        observations:      List[np.ndarray],
        actions:           List[np.ndarray],
        rewards:           List[float],
        next_observations: List[np.ndarray],
        terminated:        bool,
        truncated:         bool,
        *,
        update_target_step:       bool,
        global_learning_step:     int,
        update_step:              bool,
        initial_exploration_done: bool,
    ) -> None:
        """
        Called once per env step. Accumulates reward over the CC interval
        and, at the interval boundary, writes one transition into the
        rollout buffer. When the buffer fills, runs a PPO update.
        """
        done = terminated or truncated

        # Accumulate reward across env steps that fall under the same CC decision.
        self._accumulated_reward += float(sum(rewards))
        self._step_in_interval   += 1

        interval_complete = (self._step_in_interval >= self._cc_action_interval) or done
        if not interval_complete:
            return

        # ── Push one CC-level transition to the rollout buffer ──────────
        assert self._cached_state is not None, "predict() must run before update()"

        # Reward scaling: maintain a running discounted return, track its
        # std, and divide the raw reward by that std before storing. This
        # keeps PPO's value targets at O(1) regardless of reward units.
        # Sign is preserved (we do NOT subtract the mean).
        raw_reward = self._accumulated_reward
        self._return_running = self._gamma * self._return_running + raw_reward
        self._ret_rms.update(self._return_running)
        scaled_reward = float(raw_reward / max(float(self._ret_rms.std), 1e-8))

        self.rollout_buffer.add(
            obs     = self._cached_state,
            action  = self._cached_o1,
            logprob = self._cached_logprob,
            reward  = scaled_reward,
            done    = done,
            value   = self._cached_value,
        )

        # Reset interval accumulators.
        self._step_in_interval   = 0
        self._accumulated_reward = 0.0
        if done:
            # Discounted return resets at episode boundary; std accumulates across episodes.
            self._return_running = 0.0

        # Episode end → write the decision trace for this episode to disk.
        if done:
            self._flush_decision_trace()

        # Buffer full → compute advantages and run PPO.
        if self.rollout_buffer.full:
            self._learn_from_rollout(next_observations, done)

    # ─────────────────────── Lifecycle / artifacts ──────────────────────

    def export_artifacts(self, output_dir, context=None):
        """Export the actor (just the policy mean network) to ONNX."""
        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        export_path = onnx_dir / "community_coordinator.onnx"

        obs_dim = self.actor_critic.actor_mean[0].in_features
        dummy_input = torch.randn(1, obs_dim)

        torch.onnx.export(
            self.actor_critic.actor_mean,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=DEFAULT_ONNX_OPSET,
            do_constant_folding=True,
            input_names=["community_state"],
            output_names=["o1"],
            dynamic_axes={
                "community_state": {0: "batch_size"},
                "o1":              {0: "batch_size"},
            },
        )

        return {
            "format": "onnx",
            "artifacts": [
                {
                    "path":         str(export_path.relative_to(export_root)),
                    "format":       "onnx",
                    "agent_index":  i,
                }
                for i in range(len(self._action_dims))
            ],
        }

    # Required by BaseAgent but not implemented yet.
    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        raise NotImplementedError("Agent does not implement checkpointing.")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        raise NotImplementedError("Agent does not implement checkpoint loading.")

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        # No warm-up exploration phase: PPO learns from the very first rollout.
        return True

    # ───────────────────── Internal: state & action ─────────────────────

    def _sample_new_decision(self, observations: List[np.ndarray]) -> None:
        """Run the actor-critic once and cache the result for K env steps."""
        raw_state = np.array(self._build_community_state(observations), dtype=np.float32)

        # Update running input stats with the *raw* state, then feed the
        # *normalized* state to the network. Storing the normalized state
        # in the buffer keeps train and infer distributions matched.
        norm_state = self._normalize_state(raw_state, update_stats=True)

        state_tensor = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0)

        # No grad: this is inference, not training.
        with torch.no_grad():
            action, log_prob, _, value = self.actor_critic.get_action_and_value(state_tensor)

        self._cached_o1      = float(action.squeeze().item())
        self._cached_state   = norm_state          # normalized: this is what PPO sees
        self._cached_logprob = float(log_prob.item())
        self._cached_value   = float(value.item())

        # Decision trace logs the RAW state so plots are interpretable
        # (price stays in $/kWh, net stays in kW, etc.).
        self._log_decision(observations, raw_state)

    def _normalize_state(self, raw_state: np.ndarray, *, update_stats: bool) -> np.ndarray:
        """
        Standardize a 9-dim state with the current running mean/std.

        `update_stats=True` is for the live decision path (we see this
        state once and want to fold it into the running estimate).
        `update_stats=False` is for re-evaluating the post-rollout
        bootstrap state inside `_learn_from_rollout` (we don't want to
        bias the stats by counting that one extra observation).
        """
        if update_stats:
            self._obs_rms.update(raw_state)
        return ((raw_state - self._obs_rms.mean) / self._obs_rms.std).astype(np.float32)

    def _build_community_state(self, observations: List[np.ndarray]) -> List[float]:
        """
        Build the 9-dim state vector the actor-critic sees.

        District-level fields are precomputed by CityLearn (entity mode);
        we still average SoC manually because CityLearn doesn't expose
        a district aggregate for it.

        IMPORTANT: this list's length must equal `obs_dim` in the config.
        """
        obs0 = observations[0]
        idx0 = self._obs_index[0]

        community_net     = float(obs0[idx0["district__community_net_power_kw"]])
        community_pv      = float(obs0[idx0["district__community_pv_power_kw"]])
        community_import  = float(obs0[idx0["district__community_import_power_kw"]])
        current_price     = float(obs0[idx0["district__electricity_pricing"]])
        predicted_price_1 = float(obs0[idx0["district__electricity_pricing_predicted_1"]])
        predicted_price_2 = float(obs0[idx0["district__electricity_pricing_predicted_2"]])
        predicted_price_3 = float(obs0[idx0["district__electricity_pricing_predicted_3"]])
        hour_of_day       = float(obs0[idx0["district__hour"]])

        total_soc, n = 0.0, 0
        for i, building_obs in enumerate(observations):
            idx = self._obs_index[i]
            if "electrical_storage_soc_ratio" in idx:
                total_soc += float(building_obs[idx["electrical_storage_soc_ratio"]])
                n += 1
        avg_soc = total_soc / max(n, 1)

        return [
            community_net, community_pv, community_import, avg_soc,
            current_price, predicted_price_1, predicted_price_2, predicted_price_3,
            hour_of_day,
        ]

    # Hours-to-departure threshold below which the RBC forces full EV
    # charging regardless of o1. Ensures EVs always leave with enough charge.
    _EV_MUST_CHARGE_HOURS: int = 3

    # SoC gap threshold (as % points: required_soc - current_soc). If the
    # EV is this far below its required departure SoC, force charging even
    # if departure is not imminent.
    _EV_MUST_CHARGE_SOC_GAP: float = 50.0

    def _rbc_act(
        self,
        building_obs: np.ndarray,
        building_idx: int,
        o1:           float,
    ) -> List[float]:
        """
        Rule-based local controller. Translates the CC's `o1` into an
        action vector for one building.

        Battery rule:
          Follow `o1` directly, clamped at SoC boundaries [0.05, 0.95].

        EV rule (smart charging):
          The CC is a community-level manager — it doesn't know departure
          schedules of individual EVs. The RBC enforces those constraints.

          Two-zone logic:
            MUST-CHARGE zone: departure ≤ 3h OR SoC gap > 50 pp
              → charge at 1.0 regardless of o1 (EV WILL leave charged).
            FLEXIBLE zone: departure > 3h AND SoC gap ≤ 50 pp
              → o1 > 0 (CC says energy is cheap) → charge at 1.0
              → o1 ≤ 0 (CC says energy is expensive) → pause (0.0)

          EVs cannot be discharged (no V2G), so the negative side of o1
          only pauses charging, never reverses it.

        Everything else (e.g. washing machines): no-op (0.0).
        """
        idx = self._obs_index[building_idx]
        building_actions = [0.0] * self._action_dims[building_idx]

        for action_name, slot in self._action_index[building_idx].items():

            if action_name == "electrical_storage":
                soc = (float(building_obs[idx["electrical_storage_soc_ratio"]])
                       if "electrical_storage_soc_ratio" in idx else 0.5)
                if   o1 > 0 and soc >= 0.95: building_actions[slot] = 0.0
                elif o1 < 0 and soc <= 0.05: building_actions[slot] = 0.0
                else:                        building_actions[slot] = o1

            elif action_name.startswith("electric_vehicle_storage_charger_"):
                building_actions[slot] = self._ev_action(
                    building_obs, idx, action_name, o1
                )

        return building_actions

    def _ev_action(
        self,
        building_obs: np.ndarray,
        idx:          dict,
        action_name:  str,
        o1:           float,
    ) -> float:
        """
        Decide EV charger action for one charger slot.
        Returns 1.0 (charge), 0.0 (pause), or 0.0 (not connected).
        """
        charger_id   = action_name[len("electric_vehicle_storage_charger_"):]
        building_num = charger_id.split("_")[0]
        prefix       = f"charger::Building_{building_num}/charger_{charger_id}"

        connected = float(building_obs[idx[f"{prefix}::connected_state"]]) \
                    if f"{prefix}::connected_state" in idx else 0.0

        if connected <= 0:
            return 0.0  # no EV plugged in

        # Read departure countdown and SoC info (all in the same obs vector).
        hours_to_departure = float(building_obs[idx[f"{prefix}::connected_ev_departure_time_step"]]) \
                             if f"{prefix}::connected_ev_departure_time_step" in idx else 24.0
        ev_soc             = float(building_obs[idx[f"{prefix}::connected_ev_soc"]]) \
                             if f"{prefix}::connected_ev_soc" in idx else 0.0
        required_soc       = float(building_obs[idx[f"{prefix}::connected_ev_required_soc_departure"]]) \
                             if f"{prefix}::connected_ev_required_soc_departure" in idx else 80.0

        soc_gap = max(0.0, required_soc - ev_soc)

        # Must-charge: RBC overrides CC to protect departure readiness.
        must_charge = (
            hours_to_departure <= self._EV_MUST_CHARGE_HOURS
            or soc_gap >= self._EV_MUST_CHARGE_SOC_GAP
        )
        if must_charge:
            return 1.0

        # Flexible window: follow CC signal (positive = charge, ≤0 = pause).
        return 1.0 if o1 > 0 else 0.0

    # ──────────────────────── Internal: learning ────────────────────────

    def _learn_from_rollout(self, next_observations: List[np.ndarray], done: bool) -> None:
        """
        Buffer is full → finalise advantages with GAE and run PPO.

        GAE needs an estimate of the value AFTER the last collected step,
        which is the critic's prediction at `next_observations`.
        """
        raw_next_state  = np.array(self._build_community_state(next_observations), dtype=np.float32)
        norm_next_state = self._normalize_state(raw_next_state, update_stats=False)
        next_state_tensor = torch.tensor(norm_next_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            last_value = float(self.actor_critic.critic(next_state_tensor).item())

        self.rollout_buffer.compute_gae(
            last_value  = last_value,
            last_done   = done,
            gae_lambda  = self._gae_lambda,
        )
        self._run_ppo_update()
        self.rollout_buffer.reset()

    def _run_ppo_update(self) -> None:
        """
        One PPO learning phase over the contents of the rollout buffer.

        The combined loss being minimised is:

            loss = pg_loss  +  vf_coef * v_loss  -  ent_coef * entropy

        - pg_loss (policy / actor): clipped surrogate. Uses the ratio
              r = exp(log π_new(a|s) - log π_old(a|s))
          and minimises the worse of `-A · r` and `-A · clip(r, 1±ε)`.
          The clip prevents large policy jumps from a single update.
        - v_loss (value / critic): MSE between predicted value and the
          GAE target return, also clipped for stability.
        - entropy: encourages exploration. Subtracted (negative sign)
          so higher entropy lowers loss.

        We make `num_epochs` passes over the same data, in mini-batches.
        After each mini-batch we estimate KL divergence between old and
        new policy; if it crosses `1.5 * target_kl` we early-stop to
        avoid destructive updates.
        """
        data             = self.rollout_buffer.get()
        batch_obs        = data["obs"]
        batch_actions    = data["actions"]
        batch_logprobs   = data["logprobs"]
        batch_returns    = data["returns"]
        batch_advantages = data["advantages"]

        # Pre-update predicted values, used for clipped value loss below.
        old_values = torch.tensor(self.rollout_buffer.values, dtype=torch.float32)

        num_steps   = self.rollout_buffer.num_steps
        kl_exceeded = False
        pg_loss = v_loss = entropy_loss = torch.tensor(0.0)  # for logger

        for _ in range(self._num_epochs):
            if kl_exceeded:
                break

            indices = np.random.permutation(num_steps)

            for start in range(0, num_steps, self._mini_batch_size):
                mb = indices[start : start + self._mini_batch_size]

                _, new_logprobs, entropy, new_values = self.actor_critic.get_action_and_value(
                    batch_obs[mb], action=batch_actions[mb]
                )
                new_values = new_values.squeeze()

                # Importance-sampling ratio (new vs old policy probability).
                log_ratio = new_logprobs - batch_logprobs[mb]
                ratio     = torch.exp(log_ratio)

                # KL early-stopping check (Schulman et al.'s low-variance estimator).
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                if self._target_kl is not None and approx_kl > 1.5 * self._target_kl:
                    kl_exceeded = True
                    break

                # Clipped surrogate policy loss.
                mb_adv  = batch_advantages[mb]
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef),
                ).mean()

                # Clipped value loss (PPO-G).
                v_unclipped = (new_values - batch_returns[mb]) ** 2
                v_clipped   = old_values[mb] + (new_values - old_values[mb]).clamp(
                    -self._clip_coef, self._clip_coef
                )
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - batch_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss + self._vf_coef * v_loss - self._ent_coef * entropy_loss

                self.ppo_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self._max_grad_norm)
                self.ppo_optim.step()

        self._ppo_update_count += 1

        logger.info(
            "PPO update | pg={:.4f}  v={:.4f}  ent={:.4f}  kl_stop={}",
            pg_loss.item(), v_loss.item(), entropy_loss.item(), kl_exceeded,
        )

        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "PPO_pg_loss": pg_loss.item(),
                    "PPO_v_loss":  v_loss.item(),
                    "PPO_entropy": entropy_loss.item(),
                    "PPO_kl_stop": float(kl_exceeded),
                },
                step=self._ppo_update_count,
            )

    # ──────────────────────── Internal: logging ─────────────────────────

    def _log_decision(self, observations: List[np.ndarray], state: np.ndarray) -> None:
        """Append one row to the in-memory decision trace for this episode."""
        idx0 = self._obs_index[0]
        obs0 = observations[0]
        self._decision_trace.append({
            "cc_step":      self._global_cc_step,
            "hour":         float(obs0[idx0["district__hour"]]),
            "month":        float(obs0[idx0["district__month"]]),
            "price":        float(obs0[idx0["district__electricity_pricing"]]),
            "pred_price_1": float(obs0[idx0["district__electricity_pricing_predicted_1"]]),
            "pred_price_2": float(obs0[idx0["district__electricity_pricing_predicted_2"]]),
            "pred_price_3": float(obs0[idx0["district__electricity_pricing_predicted_3"]]),
            "community_net":float(obs0[idx0["district__community_net_power_kw"]]),
            "avg_soc":      float(state[3]),     # index 3 in _build_community_state
            "o1":           self._cached_o1,
            "value_est":    self._cached_value,
        })
        self._global_cc_step += 1

    def _flush_decision_trace(self) -> None:
        """Write the current episode's decision trace to a CSV + MLflow artifact."""
        if not self._decision_trace:
            return

        self._episode_count += 1
        ep = self._episode_count
        fields = list(self._decision_trace[0].keys())

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=f"cc_decisions_ep{ep}_", delete=False,
        ) as f:
            tmp_path = f.name
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._decision_trace)

        # Per-episode arbitrage signal: correlation between current price and
        # the CC's action. We want this NEGATIVE — high price → discharge (o1<0),
        # low price → charge (o1>0). Logged to MLflow as a chart we can watch.
        prices  = np.array([row["price"] for row in self._decision_trace], dtype=np.float64)
        actions = np.array([row["o1"]    for row in self._decision_trace], dtype=np.float64)
        if len(prices) > 1 and prices.std() > 0 and actions.std() > 0:
            corr_price_o1 = float(np.corrcoef(prices, actions)[0, 1])
        else:
            corr_price_o1 = 0.0

        logger.info(
            "CC ep{} → {} decisions → corr(price,o1)={:+.3f} mean_o1={:+.3f} → {}",
            ep, len(self._decision_trace), corr_price_o1, float(actions.mean()), tmp_path,
        )

        if mlflow.active_run():
            mlflow.log_artifact(tmp_path, artifact_path="decision_traces")
            mlflow.log_metrics(
                {
                    "CC_corr_price_o1": corr_price_o1,
                    "CC_mean_o1":       float(actions.mean()),
                    "CC_std_o1":        float(actions.std()),
                },
                step=ep,
            )

        self._decision_trace = []
