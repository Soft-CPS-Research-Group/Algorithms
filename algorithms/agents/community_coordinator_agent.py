


from algorithms.agents.base_agent import BaseAgent
from typing import Any, Dict, List, Optional, Sequence
import torch
from torch import nn
from loguru import logger
from pathlib import Path
from algorithms.constants import DEFAULT_ONNX_OPSET
from algorithms.utils.ppo import PPOActorCritic, RolloutBuffer
from torch.optim import Adam

import numpy as np


class CommunityCoordinatorAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self._config = config
        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})
        self.community = hyper.get("community")
        self.use_raw_observations = True

        self._num_epochs      = hyper.get("num_epochs",      10)
        self._mini_batch_size = hyper.get("mini_batch_size", 64)
        self._clip_coef       = hyper.get("clip_coef",       0.2)
        self._ent_coef        = hyper.get("ent_coef",        0.01)
        self._vf_coef         = hyper.get("vf_coef",         0.5)
        self._max_grad_norm   = hyper.get("max_grad_norm",   0.5)
        self._target_kl       = hyper.get("target_kl",       0.02)
        self._gae_lambda      = hyper.get("gae_lambda",      0.95)

        self.actor_critic    = PPOActorCritic(3, 1)
        self.ppo_optim       = Adam(self.actor_critic.parameters(), lr=hyper.get("lr"))
        self.rollout_buffer  = RolloutBuffer(hyper.get("num_steps"), 3, 1)


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
        # --- Build community state ---
        state = np.array(self._build_community_state(observations), dtype=np.float32)

        # --- Community reward: sum of all per-building rewards ---
        community_reward = float(sum(rewards)) / len(rewards)

        done = terminated or truncated

        # --- Re-evaluate the network on the state + action that was taken ---
        # O1 is whatever was broadcast to the buildings — read it back from actions[0][0]
        o1 = float(actions[0][0])
        state_tensor  = torch.tensor(state,   dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([[o1]],  dtype=torch.float32)

        with torch.no_grad():
            _, log_prob, _, value = self.actor_critic.get_action_and_value(state_tensor, action=action_tensor)

        # --- Store the transition ---
        self.rollout_buffer.add(
            obs=state,
            action=o1,
            logprob=float(log_prob.item()),
            reward=community_reward,
            done=done,
            value=float(value.item()),
        )

        # --- If the buffer is full, learn ---
        if self.rollout_buffer.full:
            next_state = np.array(self._build_community_state(next_observations), dtype=np.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                last_value = float(self.actor_critic.critic(next_state_tensor).item())

            self.rollout_buffer.compute_gae(last_value=last_value, last_done=done, gae_lambda=self._gae_lambda)
            self._ppo_update()
            self.rollout_buffer.reset()


    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        state = self._build_community_state(observations)

        with torch.no_grad():
            action, log_prob, entropy, value = self.actor_critic.get_action_and_value(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            o1 = action.squeeze().item()

        return [[o1] * self._action_dims[i] for i in range(len(observations))]


    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        """Persist training state and return the checkpoint path."""
        raise NotImplementedError("Agent does not implement checkpointing.")

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._obs_index = [
            {name: idx for idx, name in enumerate(names)}
            for names in observation_names
        ]
        self._action_dims = [len(names) for names in action_names]

    def export_artifacts(self, output_dir, context=None):

        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        export_path = onnx_dir / "community_coordinator.onnx"
        dummy_input = torch.randn(1, 3)

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
                "o1": {0: "batch_size"},
            },
        )

        relative_path = export_path.relative_to(export_root)

        return {
            "format": "onnx",
            "artifacts": [
                {
                    "path": str(relative_path),
                    "format": "onnx",
                    "agent_index": i,
                }
                for i in range(len(self._action_dims))
            ]
        }

    def load_checkpoint(self, checkpoint_path: str) -> None:
        raise NotImplementedError("Agent does not implement checkpoint loading.")

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        pass


    def _ppo_update(self) -> None:
        data             = self.rollout_buffer.get()
        batch_obs        = data["obs"]
        batch_actions    = data["actions"]
        batch_logprobs   = data["logprobs"]
        batch_returns    = data["returns"]
        batch_advantages = data["advantages"]

        # Old values from the buffer — needed for value clipping
        old_values = torch.tensor(self.rollout_buffer.values, dtype=torch.float32)

        num_steps       = self.rollout_buffer.num_steps
        mini_batch_size = self._mini_batch_size
        kl_exceeded     = False

        for _ in range(self._num_epochs):
            if kl_exceeded:
                break

            indices = np.random.permutation(num_steps)

            for start in range(0, num_steps, mini_batch_size):
                mb = indices[start : start + mini_batch_size]

                _, new_logprobs, entropy, new_values = self.actor_critic.get_action_and_value(
                    batch_obs[mb], action=batch_actions[mb]
                )
                new_values = new_values.squeeze()

                # --- KL early stopping ---
                log_ratio = new_logprobs - batch_logprobs[mb]
                ratio     = torch.exp(log_ratio)
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

                if self._target_kl is not None and approx_kl > 1.5 * self._target_kl:
                    kl_exceeded = True
                    break

                # --- Policy loss (clipped) ---
                mb_adv  = batch_advantages[mb]
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef),
                ).mean()

                # --- Value loss (clipped) ---
                v_unclipped = (new_values - batch_returns[mb]) ** 2
                v_clipped   = old_values[mb] + (new_values - old_values[mb]).clamp(-self._clip_coef, self._clip_coef)
                v_loss      = 0.5 * torch.max(v_unclipped, (v_clipped - batch_returns[mb]) ** 2).mean()

                # --- Entropy bonus ---
                entropy_loss = entropy.mean()

                loss = pg_loss + self._vf_coef * v_loss - self._ent_coef * entropy_loss

                self.ppo_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self._max_grad_norm)
                self.ppo_optim.step()

        logger.info("PPO update | pg={:.4f}  v={:.4f}  ent={:.4f}  kl_stop={}",
                    pg_loss.item(), v_loss.item(), entropy_loss.item(), kl_exceeded)


    def _build_community_state(
            self, 
            observations: List[np.ndarray]
    ) -> List[Any]:
        total_net_electricity_consumption = 0
        total_solar = 0
        total_load = 0

        for idx, building_obs in enumerate(observations):
            obs_indexes = self._obs_index[idx]

            net_electricity_consumption_idx = obs_indexes["net_electricity_consumption"]
            solar_idx = obs_indexes["solar_generation"]
            load_idx = obs_indexes["non_shiftable_load"]
            # aggregate price

            total_net_electricity_consumption += building_obs[net_electricity_consumption_idx]
            total_solar += building_obs[solar_idx]
            total_load += building_obs[load_idx]

        return [total_net_electricity_consumption, total_solar, total_load]