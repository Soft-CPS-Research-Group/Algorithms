"""
Community Coordinator (CC) Agent — Phase 1.

The CC is the *manager* in a hierarchical multi-agent setup.
It outputs one continuous action per building: o1 ∈ (-1, 1).

    o1 > 0  →  charge battery
    o1 < 0  →  discharge battery
    o1 = 0  →  idle

EV charger slots are zeroed (idle) — EV management is delegated to
Building Agents in Phase 2.

The CC is trained with PPO. Its reward is computed *internally* from the
quality of its own decisions, not from the environment reward signal:

    urgency_i = price_delta + net_weight × net_power_i
    reward_i  = -o1_i × urgency_i
    reward    = sum_i( reward_i )

    price_delta = price_now - mean(price_now, pred_1, pred_2, pred_3)
    net_power_i = building i's net grid draw (kW); positive = importing

Two complementary signals:
  1. price_delta  — temporal: charge cheap, discharge dear (global)
  2. net_power_i  — spatial:  prioritise buildings that are importing
                              heavily (per-building urgency)

A building heavily importing at peak price = maximum urgency to
discharge.  Both signals must align for full reward — this prevents
the CC from ignoring per-building context and just following price.
Evaluated at decision time — independent of BA actions or env outcomes.
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
from algorithms.utils.hierarchical_ppo import (
    HierarchicalActorCritic,
    HierarchicalRolloutBuffer,
)


class RunningMeanStd:
    """
    Online estimator of mean and variance (Welford's algorithm).

    Used for:
      1. State normalisation — feeds (x - mean) / std to the network.
      2. Reward scaling    — divides by running return std to keep
                             PPO value targets at O(1).
    """

    def __init__(self, shape: tuple = ()) -> None:
        self._n: int           = 0
        self._mean: np.ndarray = np.zeros(shape, dtype=np.float64)
        self._M2:  np.ndarray  = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray | float) -> None:
        x = np.asarray(x, dtype=np.float64)
        self._n += 1
        delta        = x - self._mean
        self._mean  += delta / self._n
        self._M2    += delta * (x - self._mean)

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def std(self) -> np.ndarray:
        if self._n < 2:
            return np.ones_like(self._mean)
        return np.sqrt(np.maximum(self._M2 / self._n, 1e-12))


class CommunityCoordinatorAgent(BaseAgent):
    """PPO-trained manager that outputs one battery action per building."""

    _use_raw_observations: bool = True

    # ─────────────────────────── Construction ───────────────────────────

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.use_raw_observations = True

        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})

        # ── PPO hyperparameters ────────────────────────────────────────
        self._gamma           = hyper.get("gamma",           0.99)
        self._gae_lambda      = hyper.get("gae_lambda",      0.95)
        self._num_epochs      = hyper.get("num_epochs",      10)
        self._mini_batch_size = hyper.get("mini_batch_size", 64)
        self._clip_coef       = hyper.get("clip_coef",       0.2)
        self._vf_coef         = hyper.get("vf_coef",         0.5)
        self._ent_coef        = hyper.get("ent_coef",        0.01)
        self._max_grad_norm   = hyper.get("max_grad_norm",   0.5)
        self._target_kl       = hyper.get("target_kl",       0.02)

        # ── Network + optimiser (set-based, N-agnostic) ─────────────────
        c_dim         = hyper.get("c_dim",          12)
        b_dim         = hyper.get("b_dim",           7)
        num_buildings = hyper.get("num_buildings",  17)
        hidden_dims   = hyper.get("hidden_dims",     [128, 128])
        self._c_dim         = c_dim
        self._b_dim         = b_dim
        self._num_buildings = num_buildings
        # Weight on per-building net-power term in decision reward.
        # Balances ~0.05 price_delta against ~5 kW net_power → default 0.01.
        self._net_weight: float = float(hyper.get("net_weight", 0.01))

        self.actor_critic = HierarchicalActorCritic(c_dim, b_dim, hidden_dims)
        self.ppo_optim    = Adam(self.actor_critic.parameters(), lr=hyper.get("lr", 1e-4))

        # ── Normalisation ──────────────────────────────────────────────
        self._community_rms   = RunningMeanStd(shape=(c_dim,))
        self._building_rms    = RunningMeanStd(shape=(b_dim,))
        self._ret_rms         = RunningMeanStd()
        self._return_running: float = 0.0

        # ── Rollout buffer ─────────────────────────────────────────────
        self.rollout_buffer  = HierarchicalRolloutBuffer(
            hyper.get("num_steps"), c_dim, b_dim, num_buildings
        )
        self._ppo_update_count = 0

        # ── Temporal abstraction ───────────────────────────────────────
        self._cc_action_interval: int = hyper.get("cc_action_interval", 1)
        self._step_in_interval:   int = 0

        # Cached state from the most recent CC decision.
        self._cached_o1:            np.ndarray            = np.zeros(num_buildings, dtype=np.float32)
        self._cached_raw_community: Optional[np.ndarray]  = None  # (c_dim,) raw values
        self._cached_raw_buildings: Optional[np.ndarray]  = None  # (N, b_dim) raw values — for reward
        self._cached_community:     Optional[np.ndarray]  = None  # (c_dim,) normalised
        self._cached_buildings:     Optional[np.ndarray]  = None  # (N, b_dim) normalised
        self._cached_logprob:       float                 = 0.0
        self._cached_value:         float                 = 0.0
        self._accumulated_reward:   float                 = 0.0

        # ── Output mode ────────────────────────────────────────────────
        # "actions" (default, Phase 1): CC sends actions directly to env.
        # "signal"  (Phase 2): CC passes o1 array downstream to BAs.
        self._output_mode: str = hyper.get("output_mode", "actions")

        # ── Decision trace (diagnostics) ──────────────────────────────
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
        """Cache name→index lookups for observations and actions."""
        self._obs_index    = [{n: i for i, n in enumerate(ns)} for ns in observation_names]
        self._action_index = [{n: i for i, n in enumerate(ns)} for ns in action_names]
        self._action_dims  = [len(ns) for ns in action_names]

    # ─────────────────────── Per-step interaction ───────────────────────

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        """
        Returns a per-building action vector each env step.

        Battery slot = o1 (clamped at SoC limits).
        EV charger slots = 0.0 (idle — EV management is BA's job).
        All other slots = 0.0.
        """
        if self._step_in_interval == 0:
            self._sample_new_decision(observations)

        if self._output_mode == "signal":
            return self._cached_o1

        actions = []
        for i, (building_obs, action_dim) in enumerate(
            zip(observations, self._action_dims)
        ):
            idx         = self._obs_index[i]
            act         = [0.0] * action_dim
            o1          = float(self._cached_o1[i]) if i < len(self._cached_o1) else 0.0

            for action_name, slot in self._action_index[i].items():
                if action_name == "electrical_storage":
                    soc = float(building_obs[idx["electrical_storage_soc_ratio"]]) \
                          if "electrical_storage_soc_ratio" in idx else 0.5
                    if o1 > 0 and soc >= 0.95:
                        act[slot] = 0.0   # full — can't charge more
                    elif o1 < 0 and soc <= 0.05:
                        act[slot] = 0.0   # empty — can't discharge more
                    else:
                        act[slot] = o1
                # EV chargers and everything else: idle (0.0)

            actions.append(act)

        return actions

    def update(
        self,
        observations:             List[np.ndarray],
        actions:                  List[np.ndarray],
        rewards:                  List[float],      # env reward — IGNORED (CC uses internal)
        next_observations:        List[np.ndarray],
        terminated:               bool,
        truncated:                bool,
        *,
        update_target_step:       bool,
        global_learning_step:     int,
        update_step:              bool,
        initial_exploration_done: bool,
    ) -> None:
        """
        Accumulates CC's internally-computed reward over the action interval,
        then pushes one transition to the rollout buffer.
        PPO fires when the buffer fills.
        """
        done = terminated or truncated

        # ── Internal reward: decision quality at decision time ──────────
        # reward_i = -o1_i × (price_now - ref_price)
        # Summed over buildings → one scalar per env step.
        cc_reward = self._compute_decision_reward()
        self._accumulated_reward += cc_reward
        self._step_in_interval   += 1

        interval_complete = (self._step_in_interval >= self._cc_action_interval) or done
        if not interval_complete:
            return

        assert self._cached_community is not None, "predict() must run before update()"

        # Reward scaling: divide by running return std (keeps PPO targets O(1)).
        raw_reward = self._accumulated_reward
        self._return_running  = self._gamma * self._return_running + raw_reward
        self._ret_rms.update(self._return_running)
        scaled_reward = float(raw_reward / max(float(self._ret_rms.std), 1e-8))

        self.rollout_buffer.add(
            community = self._cached_community,
            buildings = self._cached_buildings,
            action    = self._cached_o1,
            logprob   = self._cached_logprob,
            reward    = scaled_reward,
            done      = done,
            value     = self._cached_value,
        )

        self._step_in_interval   = 0
        self._accumulated_reward = 0.0
        if done:
            self._return_running = 0.0
            self._flush_decision_trace()

        if self.rollout_buffer.full:
            self._learn_from_rollout(next_observations, done)

    # ─────────────────────── Lifecycle / artifacts ──────────────────────

    def export_artifacts(self, output_dir, context=None):
        """Export the shared actor head to ONNX (N-agnostic)."""
        export_root = Path(output_dir)
        onnx_dir    = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        export_path = onnx_dir / "community_coordinator.onnx"

        in_dim      = self._c_dim + self._b_dim
        dummy_input = torch.randn(1, in_dim)

        torch.onnx.export(
            self.actor_critic.actor_head,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=DEFAULT_ONNX_OPSET,
            do_constant_folding=True,
            input_names=["community_building_state"],
            output_names=["o1"],
            dynamic_axes={
                "community_building_state": {0: "n_buildings"},
                "o1":                       {0: "n_buildings"},
            },
        )

        return {
            "format": "onnx",
            "artifacts": [
                {"path": str(export_path.relative_to(export_root)), "format": "onnx",
                 "agent_index": i}
                for i in range(len(self._action_dims))
            ],
        }

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        path = Path(output_dir) / f"cc_step_{step}.pt"
        torch.save(
            {
                "step":               step,
                "actor_critic":       self.actor_critic.state_dict(),
                "optimizer":          self.ppo_optim.state_dict(),
                "community_rms_n":    self._community_rms._n,
                "community_rms_mean": self._community_rms._mean.copy(),
                "community_rms_M2":   self._community_rms._M2.copy(),
                "building_rms_n":     self._building_rms._n,
                "building_rms_mean":  self._building_rms._mean.copy(),
                "building_rms_M2":    self._building_rms._M2.copy(),
                "ret_rms_n":          self._ret_rms._n,
                "ret_rms_mean":       self._ret_rms._mean.copy(),
                "ret_rms_M2":         self._ret_rms._M2.copy(),
                "return_running":     self._return_running,
                "ppo_update_count":   self._ppo_update_count,
                "global_cc_step":     self._global_cc_step,
            },
            path,
        )
        logger.info("CC checkpoint saved → {}", path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        root = Path(checkpoint_path)
        if root.is_dir():
            candidates = sorted(root.glob("cc_step_*.pt"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(f"No CC checkpoint found in {root}")
            path = candidates[-1]
        else:
            path = root

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.ppo_optim.load_state_dict(ckpt["optimizer"])
        self._community_rms._n    = ckpt["community_rms_n"]
        self._community_rms._mean = ckpt["community_rms_mean"]
        self._community_rms._M2   = ckpt["community_rms_M2"]
        self._building_rms._n     = ckpt["building_rms_n"]
        self._building_rms._mean  = ckpt["building_rms_mean"]
        self._building_rms._M2    = ckpt["building_rms_M2"]
        self._ret_rms._n          = ckpt["ret_rms_n"]
        self._ret_rms._mean       = ckpt["ret_rms_mean"]
        self._ret_rms._M2         = ckpt["ret_rms_M2"]
        self._return_running      = float(ckpt["return_running"])
        self._ppo_update_count    = int(ckpt.get("ppo_update_count", 0))
        self._global_cc_step      = int(ckpt.get("global_cc_step", 0))
        logger.info("CC checkpoint loaded ← {} (step {})", path, ckpt.get("step"))

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return True

    # ───────────────────── Internal: state & action ─────────────────────

    def _sample_new_decision(self, observations: List[np.ndarray]) -> None:
        """Run actor-critic once and cache result for the next K env steps."""
        raw_community = self._build_community_context(observations)
        raw_buildings = self._build_building_features(observations)

        norm_community, norm_buildings = self._normalize_state(
            raw_community, raw_buildings, update_stats=True
        )

        community_t = torch.tensor(norm_community, dtype=torch.float32).unsqueeze(0)
        buildings_t = torch.tensor(norm_buildings, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = self.actor_critic.get_action_and_value(
                community_t, buildings_t
            )

        self._cached_o1            = action.squeeze(0).numpy().astype(np.float32)
        self._cached_raw_community = raw_community     # raw: for internal reward
        self._cached_raw_buildings = raw_buildings     # raw: for per-building reward
        self._cached_community     = norm_community    # normalised: for PPO buffer
        self._cached_buildings     = norm_buildings
        self._cached_logprob       = float(log_prob.item())
        self._cached_value         = float(value.item())

        self._log_decision(observations, raw_community)

    def _compute_decision_reward(self) -> float:
        """
        reward_i = -o1_i × (price_delta + net_weight × net_power_i)
        reward   = sum_i( reward_i )

        Two complementary signals:
          price_delta  = price_now − mean(price_now, pred_1..3)
                         Temporal: charge cheap, discharge dear.
          net_power_i  = building i net grid draw at decision time (kW).
                         Spatial: prioritise buildings that are importing.

        Together: discharge at peak price in a heavily-importing building
        = maximum reward. Charge at low price in an exporting building
        (absorbing surplus PV) = also rewarded.

        Returns 0 if no decision has been cached yet.

        Building feature indices (must match _build_building_features):
          0: soc_ratio  1: pv_power_kw  2: net_power_kw  3: ev_charging_kw
          4: active_chargers  5: ev_soc  6: ev_is_flexible
        """
        if self._cached_raw_community is None or self._cached_raw_buildings is None:
            return 0.0

        # ── Temporal signal (community-level) ────────────────────────────
        c = self._cached_raw_community
        # c indices: 0=price, 1=pred_1, 2=pred_2, 3=pred_3
        price       = float(c[0])
        ref_price   = float((c[0] + c[1] + c[2] + c[3]) / 4.0)
        price_delta = price - ref_price

        # ── Spatial signal (per-building) ─────────────────────────────────
        # net_power_kw at feature index 2; positive = importing from grid.
        N = len(self._cached_o1)
        net_power = self._cached_raw_buildings[:N, 2]   # shape (N,)

        # ── Combined urgency + decision quality ───────────────────────────
        urgency      = price_delta + self._net_weight * net_power   # shape (N,)
        per_building = -self._cached_o1 * urgency                   # shape (N,)
        return float(np.sum(per_building))

    def _normalize_state(
        self,
        raw_community: np.ndarray,
        raw_buildings: np.ndarray,
        *,
        update_stats: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if update_stats:
            self._community_rms.update(raw_community)
            for row in raw_buildings:
                self._building_rms.update(row)

        norm_community = (
            (raw_community - self._community_rms.mean) / self._community_rms.std
        ).astype(np.float32)
        norm_buildings = (
            (raw_buildings - self._building_rms.mean) / self._building_rms.std
        ).astype(np.float32)
        return norm_community, norm_buildings

    def _build_community_context(self, observations: List[np.ndarray]) -> np.ndarray:
        """
        Community-context vector (length c_dim) from district aggregates.
        All values read from the first building (district fields are identical
        across buildings in entity mode).

        Index layout (must stay in sync with _compute_decision_reward):
          0: price_now   1: pred_1   2: pred_2   3: pred_3
          4: hour        5: carbon   6: net_kw   7: import_kw
          8: pv_kw       9: headroom 10: evs     11: chargers
        """
        obs0 = observations[0]
        idx0 = self._obs_index[0]

        def g(name: str) -> float:
            return float(obs0[idx0[name]]) if name in idx0 else 0.0

        return np.array([
            g("district__electricity_pricing"),
            g("district__electricity_pricing_predicted_1"),
            g("district__electricity_pricing_predicted_2"),
            g("district__electricity_pricing_predicted_3"),
            g("district__hour"),
            g("district__carbon_intensity"),
            g("district__community_net_power_kw"),
            g("district__community_import_power_kw"),
            g("district__community_pv_power_kw"),
            g("district__community_building_headroom_kw"),
            g("district__active_evs_count"),
            g("district__active_chargers_count"),
        ], dtype=np.float32)

    def _build_building_features(self, observations: List[np.ndarray]) -> np.ndarray:
        """
        Per-building feature matrix, shape (num_buildings, b_dim).
        Missing fields default to 0 (e.g. building without EV).
        """
        features = np.zeros((self._num_buildings, self._b_dim), dtype=np.float32)

        for i, building_obs in enumerate(observations):
            if i >= self._num_buildings:
                break
            idx = self._obs_index[i]

            def g(name: str) -> float:
                return float(building_obs[idx[name]]) if name in idx else 0.0

            features[i] = [
                g("electrical_storage_soc_ratio"),
                g("pv_power_kw"),
                g("net_power_kw"),
                g("ev_charging_power_kw"),
                g("active_chargers_count"),
                g("electric_vehicle_soc"),
                g("electric_vehicle_is_flexible"),
            ]

        return features

    # ──────────────────────── Internal: learning ────────────────────────

    def _learn_from_rollout(self, next_observations: List[np.ndarray], done: bool) -> None:
        raw_next_community = self._build_community_context(next_observations)
        raw_next_buildings = self._build_building_features(next_observations)
        norm_next_community, norm_next_buildings = self._normalize_state(
            raw_next_community, raw_next_buildings, update_stats=False
        )
        community_t = torch.tensor(norm_next_community, dtype=torch.float32).unsqueeze(0)
        buildings_t = torch.tensor(norm_next_buildings, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, _, _, last_value_t = self.actor_critic.get_action_and_value(
                community_t, buildings_t
            )

        self.rollout_buffer.compute_gae(
            last_value = float(last_value_t.item()),
            last_done  = done,
            gamma      = self._gamma,
            gae_lambda = self._gae_lambda,
        )
        self._run_ppo_update()
        self.rollout_buffer.reset()

    def _run_ppo_update(self) -> None:
        data             = self.rollout_buffer.get()
        batch_community  = data["community"]
        batch_buildings  = data["buildings"]
        batch_actions    = data["actions"]
        batch_logprobs   = data["logprobs"]
        batch_returns    = data["returns"]
        batch_advantages = data["advantages"]

        old_values  = torch.tensor(self.rollout_buffer.values, dtype=torch.float32)
        num_steps   = self.rollout_buffer.num_steps
        kl_exceeded = False
        pg_loss = v_loss = entropy_loss = torch.tensor(0.0)

        for _ in range(self._num_epochs):
            if kl_exceeded:
                break
            indices = np.random.permutation(num_steps)

            for start in range(0, num_steps, self._mini_batch_size):
                mb = indices[start : start + self._mini_batch_size]

                _, new_logprobs, entropy, new_values = self.actor_critic.get_action_and_value(
                    batch_community[mb], batch_buildings[mb], action=batch_actions[mb]
                )
                new_values = new_values.squeeze()

                log_ratio  = new_logprobs - batch_logprobs[mb]
                ratio      = torch.exp(log_ratio)
                approx_kl  = ((ratio - 1) - log_ratio).mean().item()

                if self._target_kl is not None and approx_kl > 1.5 * self._target_kl:
                    kl_exceeded = True
                    break

                mb_adv  = batch_advantages[mb]
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef),
                ).mean()

                v_unclipped = (new_values - batch_returns[mb]) ** 2
                v_clipped   = old_values[mb] + (new_values - old_values[mb]).clamp(
                    -self._clip_coef, self._clip_coef
                )
                v_loss = 0.5 * torch.max(
                    v_unclipped, (v_clipped - batch_returns[mb]) ** 2
                ).mean()

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

    def _log_decision(self, observations: List[np.ndarray], community: np.ndarray) -> None:
        """Append one row to the in-memory decision trace."""
        idx0 = self._obs_index[0]
        obs0 = observations[0]

        def _g(name: str) -> float:
            return float(obs0[idx0[name]]) if name in idx0 else 0.0

        o1 = np.asarray(self._cached_o1, dtype=np.float64)

        row: dict = {
            "cc_step":              self._global_cc_step,
            "month":                _g("district__month"),
            "hour":                 _g("district__hour"),
            "price":                _g("district__electricity_pricing"),
            "pred_price_1":         _g("district__electricity_pricing_predicted_1"),
            "pred_price_2":         _g("district__electricity_pricing_predicted_2"),
            "pred_price_3":         _g("district__electricity_pricing_predicted_3"),
            "carbon_intensity":     _g("district__carbon_intensity"),
            "community_net_kw":     _g("district__community_net_power_kw"),
            "community_import_kw":  _g("district__community_import_power_kw"),
            "community_pv_kw":      _g("district__community_pv_power_kw"),
            "community_headroom_kw":_g("district__community_building_headroom_kw"),
            "active_evs":           _g("district__active_evs_count"),
            "active_chargers":      _g("district__active_chargers_count"),
            "o1_mean":              float(o1.mean()),
            "o1_std":               float(o1.std()),
            "o1_min":               float(o1.min()),
            "o1_max":               float(o1.max()),
            "value_est":            self._cached_value,
            "decision_reward":      self._compute_decision_reward(),
        }

        for i, sig in enumerate(o1):
            row[f"o1_b{i}"] = float(sig)

        self._decision_trace.append(row)
        self._global_cc_step += 1

    def _flush_decision_trace(self) -> None:
        """Write episode decision trace to CSV + log MLflow metrics."""
        if not self._decision_trace:
            return

        self._episode_count += 1
        ep     = self._episode_count
        fields = list(self._decision_trace[0].keys())

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=f"cc_decisions_ep{ep}_", delete=False,
        ) as f:
            tmp_path = f.name
            writer   = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._decision_trace)

        prices  = np.array([r["price"]   for r in self._decision_trace], dtype=np.float64)
        actions = np.array([r["o1_mean"] for r in self._decision_trace], dtype=np.float64)
        rewards = np.array([r["decision_reward"] for r in self._decision_trace], dtype=np.float64)

        if len(prices) > 1 and prices.std() > 1e-6 and actions.std() > 1e-6:
            corr_price_o1 = float(np.corrcoef(prices, actions)[0, 1])
        else:
            corr_price_o1 = float("nan")

        cross_building_std = float(np.mean([r["o1_std"] for r in self._decision_trace]))

        logger.info(
            "CC ep{} → {} steps | corr(price,o1)={:+.3f} | mean_reward={:+.4f} | "
            "cross_std={:.3f} → {}",
            ep, len(self._decision_trace), corr_price_o1,
            float(rewards.mean()), cross_building_std, tmp_path,
        )

        if mlflow.active_run():
            mlflow.log_artifact(tmp_path, artifact_path="decision_traces")
            mlflow.log_metrics(
                {
                    "CC_corr_price_o1":        corr_price_o1 if not np.isnan(corr_price_o1) else 0.0,
                    "CC_mean_o1":              float(actions.mean()),
                    "CC_mean_o1_over_time_std":float(actions.std()),
                    "CC_cross_building_std":   cross_building_std,
                    "CC_mean_decision_reward": float(rewards.mean()),
                },
                step=ep,
            )

        self._decision_trace = []
