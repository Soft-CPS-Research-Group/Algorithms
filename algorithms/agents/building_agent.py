"""
Building Agent (BA) — per-building PPO worker in the HIRO architecture.

    ┌────────────────────────────────────────────────────────────┐
    │  CommunityCoordinatorAgent (stage 0)                       │
    │  outputs o1  ∈ (-1, 1) — community-wide scalar signal      │
    └────────────────────────────────────────────────────────────┘
                              │  o1  (context)
                              ▼
    ┌────────────────────────────────────────────────────────────┐
    │  BuildingAgent (this file, one per building)               │
    │  receives:  own obs  +  o1  concatenated as input          │
    │  outputs:   per-building action vector                     │
    │  learns:    PPO from start (no warm-up exploration)        │
    └────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.optim import Adam

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.community_coordinator_agent import RunningMeanStd
from algorithms.constants import DEFAULT_ONNX_OPSET
from algorithms.utils.ppo import PPOActorCritic, RolloutBuffer


class _BuildingActorExport(nn.Module):
    """Deterministic deployment wrapper for a BuildingAgent PPO actor."""

    def __init__(self, actor_mean: nn.Module) -> None:
        super().__init__()
        self.actor_mean = actor_mean

    def forward(self, observation_with_context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor_mean(observation_with_context))


class BuildingAgent(BaseAgent):
    """PPO-trained per-building worker that receives the CC's o1 signal."""

    _use_raw_observations: bool = True

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        # The wrapper passes raw CityLearn observations (not encoded).
        # Set AFTER super().__init__() which sets the instance attr to False.
        self.use_raw_observations = True

        hyper: Dict[str, Any] = (config.get("algorithm", {}).get("hyperparameters") or {})
        self._hyper = hyper  # stored for late init in attach_environment

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
        self._num_steps       = hyper.get("num_steps",       2048)
        self._hidden_dims: List[int] = list(hyper.get("hidden_dims", [64, 64]))

        # ── Reward-weighting hyperparams (stored for reference) ────────
        self._building_cost_weight     = hyper.get("building_cost_weight",     1.0)
        self._community_import_weight  = hyper.get("community_import_weight",  0.3)
        self._constraint_penalty_weight = hyper.get("constraint_penalty_weight", 0.5)

        # ── Dims (0 = lazy / auto from env) ───────────────────────────
        self._obs_dim    = int(hyper.get("obs_dim",    0))
        self._action_dim = int(hyper.get("action_dim", 0))

        # ── Running statistics ─────────────────────────────────────────
        # Initialised to scalar placeholders; resized in _init_network once
        # dims are known.
        self._obs_rms:     RunningMeanStd = RunningMeanStd()
        self._ret_rms:     RunningMeanStd = RunningMeanStd()
        self._return_running: float = 0.0

        # ── Network / buffer (lazy) ────────────────────────────────────
        self._network_initialized: bool = False
        self.actor_critic:   Optional[PPOActorCritic] = None
        self.ppo_optim:      Optional[Adam]           = None
        self.rollout_buffer: Optional[RolloutBuffer]  = None
        self._ppo_update_count: int = 0

        # ── Cached step data ───────────────────────────────────────────
        self._last_o1:       float                = 0.0
        self._cached_obs:    Optional[np.ndarray] = None
        self._cached_logprob: float               = 0.0
        self._cached_value:  float                = 0.0

        # ── Environment index maps (set in attach_environment) ─────────
        self._obs_index:   Optional[Dict[str, int]] = None
        self._action_index: Optional[Dict[str, int]] = None
        self._action_dim_env: int = 0

        # If dims are already known eagerly, init now.
        if self._obs_dim > 0 and self._action_dim > 0:
            self._init_network(hyper)

    # ------------------------------------------------------------------
    # Network initialisation helper
    # ------------------------------------------------------------------

    def _init_network(self, hyper: Dict[str, Any]) -> None:
        """Initialise actor-critic, optimiser, and rollout buffer."""
        # The network input is the building obs concatenated with o1 (1 scalar).
        net_input_dim = self._obs_dim + 1

        self._obs_rms = RunningMeanStd(shape=(net_input_dim,))

        self.actor_critic = PPOActorCritic(
            obs_dim=net_input_dim,
            action_dim=self._action_dim,
            hidden_dims=self._hidden_dims,
        )
        self.ppo_optim = Adam(
            self.actor_critic.parameters(),
            lr=hyper.get("lr", 3e-4),
        )
        self.rollout_buffer = RolloutBuffer(
            num_steps=self._num_steps,
            obs_dim=net_input_dim,
            action_dim=self._action_dim,
        )
        self._network_initialized = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set dims from env and (lazily) initialise network."""
        # Each BA receives a single-building slice: [[name, ...]]
        if observation_names:
            self._obs_index = {n: i for i, n in enumerate(observation_names[0])}
            if self._obs_dim == 0:
                self._obs_dim = len(observation_names[0])

        if action_names:
            self._action_index = {n: i for i, n in enumerate(action_names[0])}
            self._action_dim_env = len(action_names[0])
            if self._action_dim == 0:
                self._action_dim = self._action_dim_env

        if not self._network_initialized and self._obs_dim > 0 and self._action_dim > 0:
            self._init_network(self._hyper)

    # ------------------------------------------------------------------
    # Core interaction loop
    # ------------------------------------------------------------------

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: Optional[bool] = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        """
        Produce an action vector for this building.

        ``context`` is the o1 scalar from the CC.  It is concatenated to
        the building's observation before being fed to the network.
        """
        o1 = float(context) if context is not None else 0.0
        self._last_o1 = o1

        obs_raw = np.asarray(observations[0], dtype=np.float32)

        if not self._network_initialized:
            # Network not yet ready (dims still unknown) — return zeros.
            action_dim = self._action_dim or (len(self._action_index) if self._action_index else 1)
            return [[0.0] * action_dim]

        # Concatenate o1 to obs to form the network input.
        net_input_raw = np.concatenate([obs_raw, [o1]], dtype=np.float32)

        # Normalize.
        norm_input = self._normalize_obs(net_input_raw, update_stats=True)
        self._cached_obs = norm_input

        state_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = self.actor_critic.get_action_and_value(state_tensor)

        self._cached_logprob = float(log_prob.item())
        self._cached_value   = float(value.item())

        # action shape: (1, action_dim) — squeeze to list
        action_np = action.squeeze(0).cpu().numpy().tolist()
        if not isinstance(action_np, list):
            action_np = [action_np]

        return [action_np]

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
        """Store transition; trigger PPO when rollout buffer is full."""
        if not self._network_initialized or self._cached_obs is None:
            return

        done = terminated or truncated
        reward = float(rewards[0]) if rewards else 0.0

        # Reward normalisation using running discounted return.
        self._return_running = self._gamma * self._return_running + reward
        self._ret_rms.update(self._return_running)
        scaled_reward = float(reward / max(float(self._ret_rms.std), 1e-8))

        # Action stored in buffer should be the tanh-squashed value that was
        # sampled in predict().
        stored_action = np.asarray(actions[0], dtype=np.float32) if actions else np.zeros(self._action_dim, dtype=np.float32)

        self.rollout_buffer.add(
            obs     = self._cached_obs,
            action  = stored_action,
            logprob = self._cached_logprob,
            reward  = scaled_reward,
            done    = done,
            value   = self._cached_value,
        )

        if done:
            self._return_running = 0.0

        if self.rollout_buffer.full:
            self._learn_from_rollout(next_observations, done)

    # ------------------------------------------------------------------
    # Internal: learning
    # ------------------------------------------------------------------

    def _learn_from_rollout(
        self,
        next_observations: List[np.ndarray],
        done: bool,
    ) -> None:
        """Compute GAE and run a PPO update."""
        next_obs_raw = np.asarray(next_observations[0], dtype=np.float32)
        next_input_raw = np.concatenate([next_obs_raw, [self._last_o1]], dtype=np.float32)
        norm_next = self._normalize_obs(next_input_raw, update_stats=False)

        next_tensor = torch.tensor(norm_next, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            last_value = float(self.actor_critic.critic(next_tensor).item())

        self.rollout_buffer.compute_gae(
            last_value=last_value,
            last_done=done,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
        )
        self._run_ppo_update()
        self.rollout_buffer.reset()

    def _run_ppo_update(self) -> None:
        """PPO update loop — mirrors CC's _run_ppo_update exactly."""
        data             = self.rollout_buffer.get()
        batch_obs        = data["obs"]
        batch_actions    = data["actions"]
        batch_logprobs   = data["logprobs"]
        batch_returns    = data["returns"]
        batch_advantages = data["advantages"]

        old_values = torch.tensor(self.rollout_buffer.values, dtype=torch.float32)

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
                    batch_obs[mb], action=batch_actions[mb]
                )
                new_values = new_values.squeeze()

                log_ratio = new_logprobs - batch_logprobs[mb]
                ratio     = torch.exp(log_ratio)

                approx_kl = ((ratio - 1) - log_ratio).mean().item()
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
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - batch_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss + self._vf_coef * v_loss - self._ent_coef * entropy_loss

                self.ppo_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self._max_grad_norm)
                self.ppo_optim.step()

        self._ppo_update_count += 1
        logger.info(
            "BA PPO update #{} | pg={:.4f}  v={:.4f}  ent={:.4f}  kl_stop={}",
            self._ppo_update_count,
            pg_loss.item(), v_loss.item(), entropy_loss.item(), kl_exceeded,
        )

    # ------------------------------------------------------------------
    # Internal: obs normalization
    # ------------------------------------------------------------------

    def _normalize_obs(self, raw: np.ndarray, *, update_stats: bool) -> np.ndarray:
        if update_stats:
            self._obs_rms.update(raw)
        return ((raw - self._obs_rms.mean) / self._obs_rms.std).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence / lifecycle
    # ------------------------------------------------------------------

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return True

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Any] = None,
    ) -> dict:
        """Export deterministic actor policy for this building."""
        if not self._network_initialized or self.actor_critic is None:
            return {"format": "onnx", "artifacts": []}

        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        export_path = onnx_dir / "building_agent.onnx"

        export_model = _BuildingActorExport(self.actor_critic.actor_mean)
        export_model.eval()
        dummy_input = torch.randn(1, self._obs_dim + 1)
        torch.onnx.export(
            export_model,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=DEFAULT_ONNX_OPSET,
            do_constant_folding=True,
            input_names=["observation_with_context"],
            output_names=["building_action"],
            dynamic_axes={
                "observation_with_context": {0: "batch_size"},
                "building_action": {0: "batch_size"},
            },
        )

        return {
            "format": "onnx",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": str(export_path.relative_to(export_root)),
                    "format": "onnx",
                    "observation_dimension": self._obs_dim + 1,
                    "action_dimension": self._action_dim,
                }
            ],
        }

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        raise NotImplementedError("BuildingAgent does not implement checkpointing.")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        raise NotImplementedError("BuildingAgent does not implement checkpoint loading.")
