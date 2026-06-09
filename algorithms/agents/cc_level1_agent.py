"""
Community Coordinator — Level 1 (Global Community Signal).

Validates the HRL pipeline with the simplest possible high-level agent:

    observe community aggregates → emit one global discrete signal
    low-level rule maps signal to per-building battery actions

=============================================================================
ARCHITECTURE
=============================================================================

Observations (community-only — no per-building features):
    price, price forecasts, time (cyclic), day-of-week, month, community
    net/import/export/pv power, carbon intensity, moving-average import,
    rolling peak-import reference.  No SoC, no EV, no flexibility info.

Action (one global signal):
    0 → reduce  (-1):  community should cut consumption/import
    1 → neutral ( 0):  do nothing
    2 → increase(+1):  community should absorb surplus / increase load

Low-level rule:
    reduce   → charge all batteries at -1.0 (discharge, give back to grid)
    neutral  → all batteries idle
    increase → charge all batteries at +1.0 (store cheap energy)

Reward:
    Comes from CCRewardLevel1 (community cost + peak penalty + export penalty).
    CC sums per-building rewards as its scalar training signal.
    NO internal reward: this level is trained purely on community outcomes.

KPIs logged per episode:
    - distribution of reduce/neutral/increase decisions (%)
    - decision alignment:
        * "price-reduce":  % of reduce decisions when price > ref_price
        * "price-increase": % of increase decisions when price < ref_price
        * "import-reduce":  % of reduce decisions when community is net-importing
    - mean episode reward

=============================================================================
TRAINING
=============================================================================
PPO with Categorical distribution over 3 actions.
Community context normalised online (Welford RunningMeanStd).
Reward scaled by running return std.
"""

from __future__ import annotations

import csv
import tempfile
from collections import deque
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

# ───────────────────────────── Helpers ──────────────────────────────────────

class RunningMeanStd:
    """Online mean/variance estimator (Welford)."""

    def __init__(self, shape: tuple = ()) -> None:
        self._n:    int           = 0
        self._mean: np.ndarray    = np.zeros(shape, dtype=np.float64)
        self._M2:   np.ndarray    = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray | float) -> None:
        x = np.asarray(x, dtype=np.float64)
        self._n += 1
        delta       = x - self._mean
        self._mean += delta / self._n
        self._M2   += delta * (x - self._mean)

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def std(self) -> np.ndarray:
        if self._n < 2:
            return np.ones_like(self._mean)
        return np.sqrt(np.maximum(self._M2 / self._n, 1e-12))


# ─────────────────────── Policy Network ─────────────────────────────────────

class CommunityLevel1Net(nn.Module):
    """MLP: community context → Categorical(3) + scalar value.

    Actor head outputs logits for {reduce(0), neutral(1), increase(2)}.
    Critic head outputs scalar state-value estimate.
    """

    def __init__(self, c_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_d = c_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        self.encoder     = nn.Sequential(*layers)
        self.actor_head  = nn.Linear(in_d, 3)   # logits: reduce / neutral / increase
        self.critic_head = nn.Linear(in_d, 1)   # scalar value

        # Orthogonal initialisation (standard PPO practice)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def get_action_and_value(
        self,
        community: torch.Tensor,
        action:    Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h      = self.encoder(community)
        logits = self.actor_head(h)
        value  = self.critic_head(h).squeeze(-1)
        dist   = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ───────────────────────── Rollout Buffer ────────────────────────────────────

class DiscreteRolloutBuffer:
    """Fixed-size rollout buffer for discrete-action PPO."""

    def __init__(self, num_steps: int, c_dim: int) -> None:
        self.num_steps = num_steps
        self._ptr  = 0
        self.full  = False

        self.communities = np.zeros((num_steps, c_dim), dtype=np.float32)
        self.actions     = np.zeros(num_steps,          dtype=np.int64)
        self.logprobs    = np.zeros(num_steps,          dtype=np.float32)
        self.rewards     = np.zeros(num_steps,          dtype=np.float32)
        self.dones       = np.zeros(num_steps,          dtype=np.float32)
        self.values      = np.zeros(num_steps,          dtype=np.float32)
        self.returns     = np.zeros(num_steps,          dtype=np.float32)
        self.advantages  = np.zeros(num_steps,          dtype=np.float32)

    def add(
        self,
        community: np.ndarray,
        action:    int,
        logprob:   float,
        reward:    float,
        done:      bool,
        value:     float,
    ) -> None:
        self.communities[self._ptr] = community
        self.actions[self._ptr]     = action
        self.logprobs[self._ptr]    = logprob
        self.rewards[self._ptr]     = reward
        self.dones[self._ptr]       = float(done)
        self.values[self._ptr]      = value
        self._ptr += 1
        if self._ptr >= self.num_steps:
            self.full = True

    def compute_gae(
        self,
        last_value: float,
        last_done:  bool,
        gamma:      float,
        gae_lambda: float,
    ) -> None:
        gae = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_nt    = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_nt    = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta              = self.rewards[t] + gamma * next_value * next_nt - self.values[t]
            gae                = delta + gamma * gae_lambda * next_nt * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get(self) -> dict:
        return {
            "community":  torch.tensor(self.communities, dtype=torch.float32),
            "actions":    torch.tensor(self.actions,     dtype=torch.int64),
            "logprobs":   torch.tensor(self.logprobs,    dtype=torch.float32),
            "returns":    torch.tensor(self.returns,     dtype=torch.float32),
            "advantages": torch.tensor(self.advantages,  dtype=torch.float32),
        }

    def reset(self) -> None:
        self._ptr = 0
        self.full = False


# ─────────────────────── Main Agent ─────────────────────────────────────────

# Map action index → human label / battery multiplier
_ACTION_LABEL = {0: "reduce", 1: "neutral", 2: "increase"}
_ACTION_TO_BATTERY = {0: -1.0, 1: 0.0, 2: 1.0}


class CCLevel1Agent(BaseAgent):
    """
    Level-1 Community Coordinator.

    Sees only community aggregates. Emits one global discrete signal
    {reduce, neutral, increase}.  Low-level rule maps signal to battery
    actions for every building.
    """

    _use_raw_observations: bool = True

    # ──────────────────────────── Construction ──────────────────────────────

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.use_raw_observations = True

        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})

        # PPO hyperparameters
        self._gamma           = float(hyper.get("gamma",           0.99))
        self._gae_lambda      = float(hyper.get("gae_lambda",      0.95))
        self._num_epochs      = int(hyper.get("num_epochs",        10))
        self._mini_batch_size = int(hyper.get("mini_batch_size",   64))
        self._clip_coef       = float(hyper.get("clip_coef",       0.2))
        self._vf_coef         = float(hyper.get("vf_coef",         0.5))
        self._ent_coef        = float(hyper.get("ent_coef",        0.01))
        self._max_grad_norm   = float(hyper.get("max_grad_norm",   0.5))
        self._target_kl       = hyper.get("target_kl",             0.1)

        # Network dimensions
        c_dim       = int(hyper.get("c_dim",       18))
        hidden_dims = hyper.get("hidden_dims",      [128, 128])
        self._c_dim = c_dim

        # Network + optimiser
        self.policy    = CommunityLevel1Net(c_dim, hidden_dims)
        self.ppo_optim = Adam(self.policy.parameters(), lr=float(hyper.get("lr", 1e-4)))

        # Observation normalisation + reward scaling
        self._community_rms  = RunningMeanStd(shape=(c_dim,))
        self._ret_rms        = RunningMeanStd()
        self._return_running = 0.0

        # Rollout buffer
        num_steps = int(hyper.get("num_steps", 96))
        self.rollout_buffer = DiscreteRolloutBuffer(num_steps, c_dim)
        self._ppo_update_count = 0

        # Temporal abstraction: CC decides every K env steps
        self._cc_action_interval = int(hyper.get("cc_action_interval", 1))
        self._step_in_interval   = 0

        # Moving-average window for import/export (obs features 16,17)
        ma_window = int(hyper.get("ma_window", 96))   # default 24h at 15-min
        self._import_ma_buf  = deque(maxlen=ma_window)
        self._export_ma_buf  = deque(maxlen=ma_window)
        self._peak_import_buf = deque(maxlen=ma_window)

        # Cached decision
        self._cached_action:    int                   = 1    # neutral
        self._cached_community: Optional[np.ndarray]  = None  # normalised
        self._cached_raw_comm:  Optional[np.ndarray]  = None  # raw — for trace
        self._cached_logprob:   float                 = 0.0
        self._cached_value:     float                 = 0.0
        self._accumulated_reward: float               = 0.0

        # Diagnostics
        self._episode_count  = 0
        self._global_cc_step = 0
        self._decision_trace: List[dict] = []

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names:      List[List[str]],
        action_space:      List[Any],
        observation_space: List[Any],
        metadata:          Optional[Dict[str, Any]] = None,
    ) -> None:
        self._obs_index    = [{n: i for i, n in enumerate(ns)} for ns in observation_names]
        self._action_index = [{n: i for i, n in enumerate(ns)} for ns in action_names]
        self._action_dims  = [len(ns) for ns in action_names]

    # ───────────────────────── Per-step interaction ──────────────────────────

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        """
        Sample (or reuse) global signal and translate to per-building battery actions.

        Global action indices:
            0 (reduce)   → battery = -1.0  (discharge: give back energy)
            1 (neutral)  → battery =  0.0  (idle)
            2 (increase) → battery = +1.0  (charge: absorb cheap energy)

        SoC boundary clamping applied per building.
        """
        if self._step_in_interval == 0:
            self._sample_new_decision(observations)

        battery_cmd = _ACTION_TO_BATTERY[self._cached_action]

        actions = []
        for i, (building_obs, action_dim) in enumerate(
            zip(observations, self._action_dims)
        ):
            idx = self._obs_index[i]
            act = [0.0] * action_dim

            for action_name, slot in self._action_index[i].items():
                if action_name == "electrical_storage":
                    soc = (
                        float(building_obs[idx["electrical_storage_soc_ratio"]])
                        if "electrical_storage_soc_ratio" in idx
                        else 0.5
                    )
                    if battery_cmd > 0 and soc >= 0.95:
                        act[slot] = 0.0    # full — can't charge
                    elif battery_cmd < 0 and soc <= 0.05:
                        act[slot] = 0.0    # empty — can't discharge
                    else:
                        act[slot] = battery_cmd
                # EV chargers and everything else: idle (0.0)

            actions.append(act)

        return actions

    def update(
        self,
        observations:             List[np.ndarray],
        actions:                  List[np.ndarray],
        rewards:                  List[float],          # from CCRewardLevel1
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
        Accumulate community reward over the action interval then push to buffer.
        PPO fires when the buffer fills.
        """
        done = terminated or truncated

        # CC reward = sum of per-building env rewards (all from CCRewardLevel1)
        self._accumulated_reward += float(sum(rewards))
        self._step_in_interval   += 1

        interval_complete = (self._step_in_interval >= self._cc_action_interval) or done
        if not interval_complete:
            return

        assert self._cached_community is not None, "predict() must run before update()"

        # Reward scaling
        raw_reward           = self._accumulated_reward
        self._return_running = self._gamma * self._return_running + raw_reward
        self._ret_rms.update(self._return_running)
        scaled_reward = float(raw_reward / max(float(self._ret_rms.std), 1e-8))

        self.rollout_buffer.add(
            community = self._cached_community,
            action    = self._cached_action,
            logprob   = self._cached_logprob,
            reward    = scaled_reward,
            done      = done,
            value     = self._cached_value,
        )

        self._step_in_interval    = 0
        self._accumulated_reward  = 0.0
        if done:
            self._return_running = 0.0
            self._flush_decision_trace()

        if self.rollout_buffer.full:
            self._learn_from_rollout(next_observations, done)

    # ───────────────────────── Lifecycle / artifacts ─────────────────────────

    def export_artifacts(self, output_dir, context=None):
        export_root = Path(output_dir)
        onnx_dir    = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        export_path = onnx_dir / "cc_level1.onnx"

        dummy = torch.randn(1, self._c_dim)
        torch.onnx.export(
            self.policy.encoder,
            dummy,
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
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": str(export_path.relative_to(export_root)),
                    "format": "onnx",
                }
            ],
        }

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        path = Path(output_dir) / f"cc_l1_step_{step}.pt"
        torch.save(
            {
                "step":             step,
                "policy":           self.policy.state_dict(),
                "optimizer":        self.ppo_optim.state_dict(),
                "comm_rms_n":       self._community_rms._n,
                "comm_rms_mean":    self._community_rms._mean.copy(),
                "comm_rms_M2":      self._community_rms._M2.copy(),
                "ret_rms_n":        self._ret_rms._n,
                "ret_rms_mean":     self._ret_rms._mean.copy(),
                "ret_rms_M2":       self._ret_rms._M2.copy(),
                "return_running":   self._return_running,
                "ppo_update_count": self._ppo_update_count,
                "global_cc_step":   self._global_cc_step,
            },
            path,
        )
        logger.info("CC-L1 checkpoint saved → {}", path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        root = Path(checkpoint_path)
        if root.is_dir():
            candidates = sorted(root.glob("cc_l1_step_*.pt"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(f"No CC-L1 checkpoint in {root}")
            path = candidates[-1]
        else:
            path = root
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.ppo_optim.load_state_dict(ckpt["optimizer"])
        self._community_rms._n    = ckpt["comm_rms_n"]
        self._community_rms._mean = ckpt["comm_rms_mean"]
        self._community_rms._M2   = ckpt["comm_rms_M2"]
        self._ret_rms._n          = ckpt["ret_rms_n"]
        self._ret_rms._mean       = ckpt["ret_rms_mean"]
        self._ret_rms._M2         = ckpt["ret_rms_M2"]
        self._return_running      = float(ckpt["return_running"])
        self._ppo_update_count    = int(ckpt.get("ppo_update_count", 0))
        self._global_cc_step      = int(ckpt.get("global_cc_step", 0))
        logger.info("CC-L1 checkpoint loaded ← {}", path)

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return True

    # ───────────────────────── Internal: state ───────────────────────────────

    def _sample_new_decision(self, observations: List[np.ndarray]) -> None:
        raw_comm = self._build_community_context(observations)

        self._community_rms.update(raw_comm)
        norm_comm = (
            (raw_comm - self._community_rms.mean) / self._community_rms.std
        ).astype(np.float32)

        comm_t = torch.tensor(norm_comm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = self.policy.get_action_and_value(comm_t)

        self._cached_action    = int(action.item())
        self._cached_community = norm_comm
        self._cached_raw_comm  = raw_comm
        self._cached_logprob   = float(log_prob.item())
        self._cached_value     = float(value.item())

        self._log_decision(raw_comm)

    def _build_community_context(self, observations: List[np.ndarray]) -> np.ndarray:
        """
        Community-context vector (length c_dim).
        All values from the first building's district__ fields.

        Index layout (must match c_dim=18 in config):
          0:  electricity_pricing          (current price)
          1:  electricity_pricing_pred_1
          2:  electricity_pricing_pred_2
          3:  electricity_pricing_pred_3
          4:  time_of_day_sin              (cyclic: position in 24h)
          5:  time_of_day_cos
          6:  day_type_sin                 (cyclic: day of week)
          7:  day_type_cos
          8:  is_weekend
          9:  month_sin                    (cyclic: seasonal)
          10: month_cos
          11: community_net_power_kw       (+ = net import, - = net export)
          12: community_import_power_kw
          13: community_export_power_kw
          14: community_pv_power_kw
          15: carbon_intensity
          16: import_moving_avg_kw         (internal: rolling 24h mean)
          17: peak_import_ref_kw           (internal: rolling 24h max)
        """
        obs0 = observations[0]
        idx0 = self._obs_index[0]

        def g(name: str) -> float:
            return float(obs0[idx0[name]]) if name in idx0 else 0.0

        # Update internal rolling buffers from current step
        import_kw = g("district__community_import_power_kw")
        export_kw = g("district__community_export_power_kw")
        self._import_ma_buf.append(import_kw)
        self._export_ma_buf.append(export_kw)
        self._peak_import_buf.append(import_kw)

        import_ma  = float(np.mean(self._import_ma_buf)) if self._import_ma_buf else 0.0
        peak_ref   = float(np.max(self._peak_import_buf)) if self._peak_import_buf else 0.0

        return np.array([
            g("district__electricity_pricing"),
            g("district__electricity_pricing_predicted_1"),
            g("district__electricity_pricing_predicted_2"),
            g("district__electricity_pricing_predicted_3"),
            g("district__time_of_day_sin"),
            g("district__time_of_day_cos"),
            g("district__day_type_sin"),
            g("district__day_type_cos"),
            g("district__is_weekend"),
            g("district__month_sin"),
            g("district__month_cos"),
            g("district__community_net_power_kw"),
            import_kw,
            export_kw,
            g("district__community_pv_power_kw"),
            g("district__carbon_intensity"),
            import_ma,
            peak_ref,
        ], dtype=np.float32)

    # ───────────────────────── Internal: learning ────────────────────────────

    def _learn_from_rollout(self, next_observations: List[np.ndarray], done: bool) -> None:
        raw_next = self._build_community_context(next_observations)
        self._community_rms.update(raw_next)
        norm_next = (
            (raw_next - self._community_rms.mean) / self._community_rms.std
        ).astype(np.float32)

        comm_t = torch.tensor(norm_next, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, _, _, last_value_t = self.policy.get_action_and_value(comm_t)

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
        batch_actions    = data["actions"]
        batch_logprobs   = data["logprobs"]
        batch_returns    = data["returns"]
        batch_advantages = data["advantages"]

        old_values   = torch.tensor(self.rollout_buffer.values, dtype=torch.float32)
        num_steps    = self.rollout_buffer.num_steps
        kl_exceeded  = False
        pg_loss = v_loss = entropy_loss = torch.tensor(0.0)

        for _ in range(self._num_epochs):
            if kl_exceeded:
                break
            indices = np.random.permutation(num_steps)
            for start in range(0, num_steps, self._mini_batch_size):
                mb = indices[start : start + self._mini_batch_size]

                _, new_logprobs, entropy, new_values = self.policy.get_action_and_value(
                    batch_community[mb], batch_actions[mb]
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
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._max_grad_norm)
                self.ppo_optim.step()

        self._ppo_update_count += 1

        logger.info(
            "CC-L1 PPO | pg={:.4f}  v={:.4f}  ent={:.4f}  kl_stop={}",
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

    # ──────────────────────── Internal: logging ──────────────────────────────

    def _log_decision(self, raw_comm: np.ndarray) -> None:
        """Append one row to the episode decision trace."""
        # c[0]=price, c[1..3]=pred, c[11]=net, c[12]=import, c[13]=export, c[14]=pv
        price      = float(raw_comm[0])
        ref_price  = float((raw_comm[0] + raw_comm[1] + raw_comm[2] + raw_comm[3]) / 4.0)
        net_kw     = float(raw_comm[11])
        import_kw  = float(raw_comm[12])
        export_kw  = float(raw_comm[13])
        pv_kw      = float(raw_comm[14])

        action     = self._cached_action
        label      = _ACTION_LABEL[action]

        row: dict = {
            "cc_step":          self._global_cc_step,
            "price":            price,
            "ref_price":        ref_price,
            "net_kw":           net_kw,
            "import_kw":        import_kw,
            "export_kw":        export_kw,
            "pv_kw":            pv_kw,
            "import_ma_kw":     float(raw_comm[16]),
            "peak_import_kw":   float(raw_comm[17]),
            "action_idx":       action,
            "action_label":     label,
            "value_est":        self._cached_value,
            # Alignment flags (simple: is decision justified by context?)
            "price_justified":  int(
                (action == 0 and price > ref_price) or   # reduce when expensive
                (action == 2 and price < ref_price)       # increase when cheap
            ),
            "import_justified": int(
                (action == 0 and net_kw > 0) or   # reduce when net-importing
                (action == 2 and net_kw < 0)       # increase when net-exporting (absorb)
            ),
        }
        self._decision_trace.append(row)
        self._global_cc_step += 1

    def _flush_decision_trace(self) -> None:
        """Write episode decision trace to CSV and log KPIs to MLflow."""
        if not self._decision_trace:
            return

        self._episode_count += 1
        ep     = self._episode_count
        fields = list(self._decision_trace[0].keys())

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=f"cc_l1_ep{ep}_", delete=False,
        ) as f:
            tmp_path = f.name
            writer   = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._decision_trace)

        n       = len(self._decision_trace)
        actions = np.array([r["action_idx"] for r in self._decision_trace])
        rewards = np.array([r.get("reward", 0.0) for r in self._decision_trace])

        pct_reduce   = float(np.mean(actions == 0)) * 100
        pct_neutral  = float(np.mean(actions == 1)) * 100
        pct_increase = float(np.mean(actions == 2)) * 100

        # Alignment metrics
        price_just  = float(np.mean([r["price_justified"]  for r in self._decision_trace])) * 100
        import_just = float(np.mean([r["import_justified"] for r in self._decision_trace])) * 100

        # Directional breakdown: reduce when price>ref (correct), reduce when price<ref (wrong)
        reduce_rows   = [r for r in self._decision_trace if r["action_idx"] == 0]
        increase_rows = [r for r in self._decision_trace if r["action_idx"] == 2]
        pct_reduce_at_peak = (
            float(np.mean([r["price"] > r["ref_price"] for r in reduce_rows])) * 100
            if reduce_rows else float("nan")
        )
        pct_increase_at_cheap = (
            float(np.mean([r["price"] < r["ref_price"] for r in increase_rows])) * 100
            if increase_rows else float("nan")
        )

        logger.info(
            "CC-L1 ep{} | {} steps | "
            "reduce={:.1f}%  neutral={:.1f}%  increase={:.1f}% | "
            "price_align={:.1f}%  import_align={:.1f}% | "
            "reduce@peak={:.1f}%  increase@cheap={:.1f}%",
            ep, n,
            pct_reduce, pct_neutral, pct_increase,
            price_just, import_just,
            pct_reduce_at_peak, pct_increase_at_cheap,
        )

        if mlflow.active_run():
            mlflow.log_artifact(tmp_path, artifact_path="decision_traces")
            metrics: dict = {
                "L1_pct_reduce":              pct_reduce,
                "L1_pct_neutral":             pct_neutral,
                "L1_pct_increase":            pct_increase,
                "L1_price_align_pct":         price_just,
                "L1_import_align_pct":        import_just,
            }
            if not np.isnan(pct_reduce_at_peak):
                metrics["L1_reduce_at_peak_pct"]    = pct_reduce_at_peak
            if not np.isnan(pct_increase_at_cheap):
                metrics["L1_increase_at_cheap_pct"] = pct_increase_at_cheap
            mlflow.log_metrics(metrics, step=ep)

        self._decision_trace = []
