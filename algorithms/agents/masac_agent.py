from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import numpy as np
import torch
from loguru import logger
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_

from algorithms.agents.maddpg_agent import MADDPG
from algorithms.utils.networks import Critic, GaussianActor


class MASAC(MADDPG):
    """Multi-Agent SAC with centralized twin critics and decentralized actors."""

    def _initialize_networks(self):
        logger.debug("Initializing MASAC stochastic actor and twin critic networks.")
        actor_cfg = self.config["algorithm"]["networks"]["actor"]
        critic_cfg = self.config["algorithm"]["networks"]["critic"]
        actor_fc_units = actor_cfg["layers"]
        critic_fc_units = critic_cfg["layers"]
        exploration_cfg = self.config["algorithm"]["exploration"]["params"]

        self.initial_log_std = float(exploration_cfg.get("initial_log_std", -0.5))
        self.min_log_std = float(exploration_cfg.get("min_log_std", -5.0))
        self.max_log_std = float(exploration_cfg.get("max_log_std", 1.0))
        self.entropy_alpha = float(max(exploration_cfg.get("entropy_alpha", 0.2), 1.0e-8))
        self.automatic_entropy_tuning = bool(exploration_cfg.get("automatic_entropy_tuning", True))

        actors, critics, actor_targets, critic_targets = [], [], [], []
        self.critics_2, self.critic_targets_2 = [], []
        self.target_entropy = []
        global_state_size = sum(self.observation_dimension)
        global_action_size = sum(self.action_dimension)
        for agent_idx in range(self.num_agents):
            state_size = self.observation_dimension[agent_idx]
            action_size = self.action_dimension[agent_idx]
            actors.append(
                GaussianActor(
                    state_size,
                    action_size,
                    self.seed + agent_idx,
                    actor_fc_units,
                    initial_log_std=self.initial_log_std,
                    min_log_std=self.min_log_std,
                    max_log_std=self.max_log_std,
                ).to(self.device)
            )
            actor_targets.append(
                GaussianActor(
                    state_size,
                    action_size,
                    self.seed + agent_idx,
                    actor_fc_units,
                    initial_log_std=self.initial_log_std,
                    min_log_std=self.min_log_std,
                    max_log_std=self.max_log_std,
                ).to(self.device)
            )
            critics.append(Critic(global_state_size, global_action_size, self.seed, critic_fc_units).to(self.device))
            critic_targets.append(Critic(global_state_size, global_action_size, self.seed, critic_fc_units).to(self.device))
            self.critics_2.append(Critic(global_state_size, global_action_size, self.seed + 7919, critic_fc_units).to(self.device))
            self.critic_targets_2.append(
                Critic(global_state_size, global_action_size, self.seed + 7919, critic_fc_units).to(self.device)
            )
            configured_target = exploration_cfg.get("target_entropy")
            self.target_entropy.append(
                float(configured_target) if configured_target is not None else -float(action_size)
            )

        for actor, actor_target in zip(actors, actor_targets):
            actor_target.load_state_dict(actor.state_dict())
        for critic, critic_target in zip(critics, critic_targets):
            critic_target.load_state_dict(critic.state_dict())
        for critic, critic_target in zip(self.critics_2, self.critic_targets_2):
            critic_target.load_state_dict(critic.state_dict())

        return actors, critics, actor_targets, critic_targets

    def _initialize_optimizers(self):
        self.log_alpha = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.log(torch.as_tensor(self.entropy_alpha, dtype=torch.float32, device=self.device))
                )
                for _ in range(self.num_agents)
            ]
        )
        actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.lr_actor) for actor in self.actors]
        critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=self.lr_critic) for critic in self.critics]
        self.critic_optimizers_2 = [
            torch.optim.Adam(critic.parameters(), lr=self.lr_critic) for critic in self.critics_2
        ]
        alpha_lr = float(self.config["algorithm"]["exploration"]["params"].get("alpha_lr", self.lr_actor))
        self.alpha_optimizers = [
            torch.optim.Adam([self.log_alpha[agent_idx]], lr=alpha_lr)
            for agent_idx in range(self.num_agents)
        ]
        return actor_optimizers, critic_optimizers

    def predict(self, observations, deterministic: bool = False) -> List[List[float]]:
        self.exploration_step += 1
        if not deterministic and self.exploration_step <= self.random_exploration_steps:
            initial_strategy = getattr(self, "initial_exploration_strategy", "uniform_full_range")
            if initial_strategy == "policy":
                return self._predict_warm_start_policy()
            if initial_strategy == "noop_centered":
                return self._predict_noop_centered()
            return self._predict_random()

        actions: List[List[float]] = []
        with torch.inference_mode():
            for agent_idx, (actor, obs) in enumerate(zip(self.actors, observations)):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
                if deterministic:
                    normalized_action = actor(obs_tensor)
                else:
                    normalized_action, _ = actor.sample_normalized(obs_tensor)
                scaled_action = self._scale_action_tensor(agent_idx, normalized_action)
                actions.append(scaled_action.squeeze(0).cpu().numpy().tolist())
        return actions

    def update(
        self,
        observations: List[torch.Tensor],
        actions: List[torch.Tensor],
        rewards: List[float],
        next_observations: List[torch.Tensor],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        update_start_time = time.time()

        done = bool(terminated or truncated)
        self._update_reward_normalizer(rewards)
        behavior_actions = self._transition_behavior_actions(actions)
        priority_boost = self._transition_observation_event_priority_boost()
        self._push_replay_transition(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            done=done,
            behavior_actions=behavior_actions,
            priority_boost=priority_boost,
        )

        if len(self.replay_buffer) < self.batch_size:
            return
        if not self._should_train_on_step(initial_exploration_done, global_learning_step):
            return
        if not update_step:
            return

        if hasattr(self.replay_buffer, "sample_with_behavior_actions"):
            states, actions_all, rewards_all, next_states, dones_all, behavior_actions_all = (
                self.replay_buffer.sample_with_behavior_actions()
            )
        else:
            states, actions_all, rewards_all, next_states, dones_all = self.replay_buffer.sample()
            behavior_actions_all = actions_all

        raw_rewards_all = torch.stack(rewards_all).to(self.device, dtype=torch.float32, non_blocking=True)
        rewards_all = self._normalize_reward_tensor(raw_rewards_all)
        dones_all = dones_all.to(self.device, dtype=torch.float32, non_blocking=True)
        states = [state.to(self.device, non_blocking=True) for state in states]
        actions_all = [action.to(self.device, non_blocking=True) for action in actions_all]
        behavior_actions_all = [action.to(self.device, non_blocking=True) for action in behavior_actions_all]
        next_states = [state.to(self.device, non_blocking=True) for state in next_states]

        global_state = torch.cat(states, dim=1)
        global_next_state = torch.cat(next_states, dim=1)
        global_actions = torch.cat(actions_all, dim=1)

        with torch.no_grad():
            next_policy_actions = []
            next_log_probs = []
            for agent_idx in range(self.num_agents):
                normalized_next_action, next_log_prob = self.actor_targets[agent_idx].sample_normalized(
                    next_states[agent_idx]
                )
                next_policy_actions.append(self._scale_action_tensor(agent_idx, normalized_next_action))
                next_log_probs.append(next_log_prob)
            global_next_actions = torch.cat(next_policy_actions, dim=1)

            q_targets = []
            for agent_idx in range(self.num_agents):
                q1_next = self.critic_targets[agent_idx](global_next_state, global_next_actions)
                q2_next = self.critic_targets_2[agent_idx](global_next_state, global_next_actions)
                min_q_next = torch.minimum(q1_next, q2_next)
                alpha = self._alpha(agent_idx).detach()
                soft_value = min_q_next - alpha * next_log_probs[agent_idx]
                target = rewards_all[agent_idx] + self.gamma * soft_value * (1 - dones_all[agent_idx])
                if self.critic_target_clip_abs > 0.0:
                    target = torch.clamp(target, -self.critic_target_clip_abs, self.critic_target_clip_abs)
                q_targets.append(target)

        critic_loss_values: List[float] = []
        critic_2_loss_values: List[float] = []
        critic_td_abs_values: List[float] = []
        critic_gap_values: List[float] = []
        critic_grad_norm_values: List[float] = []
        q1_expected_tensors: List[torch.Tensor] = []
        q2_expected_tensors: List[torch.Tensor] = []

        for agent_idx in range(self.num_agents):
            critic_1 = self.critics[agent_idx]
            critic_2 = self.critics_2[agent_idx]
            optimizer_1 = self.critic_optimizers[agent_idx]
            optimizer_2 = self.critic_optimizers_2[agent_idx]
            target = q_targets[agent_idx]

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                q1_expected = critic_1(global_state, global_actions)
                q2_expected = critic_2(global_state, global_actions)
                critic_1_loss = self._critic_loss(q1_expected, target)
                critic_2_loss = self._critic_loss(q2_expected, target)
                critic_loss = critic_1_loss + critic_2_loss

            optimizer_1.zero_grad(set_to_none=True)
            optimizer_2.zero_grad(set_to_none=True)
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(optimizer_1)
            self.scaler.unscale_(optimizer_2)
            grad_norm = clip_grad_norm_(
                [*critic_1.parameters(), *critic_2.parameters()],
                max_norm=1.0,
            )
            self.scaler.step(optimizer_1)
            self.scaler.step(optimizer_2)
            self.scaler.update()

            critic_loss_values.append(float(critic_1_loss.detach().item()))
            critic_2_loss_values.append(float(critic_2_loss.detach().item()))
            critic_td_abs_values.append(float((q1_expected.detach() - target.detach()).abs().mean().item()))
            critic_gap_values.append(float((q1_expected.detach() - q2_expected.detach()).abs().mean().item()))
            critic_grad_norm_values.append(float(grad_norm))
            q1_expected_tensors.append(q1_expected.detach())
            q2_expected_tensors.append(q2_expected.detach())

        actor_update_due = global_learning_step % self.actor_update_interval == 0
        actor_loss_values: List[float] = []
        actor_log_prob_values: List[float] = []
        alpha_loss_values: List[float] = []
        alpha_values: List[float] = []
        actor_grad_norm_values: List[float] = []
        actor_behavior_cloning_values: List[float] = []
        actor_regularization_values: List[float] = []
        actor_behavior_cloning_effective_weight = self._actor_behavior_cloning_effective_weight(
            global_learning_step
        )
        if actor_update_due:
            with torch.no_grad():
                detached_policy_actions = []
                for agent_idx, actor in enumerate(self.actors):
                    normalized_action, _ = actor.sample_normalized(states[agent_idx])
                    detached_policy_actions.append(self._scale_action_tensor(agent_idx, normalized_action).detach())

            for agent_idx, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
                normalized_action, log_prob = actor.sample_normalized(states[agent_idx])
                scaled_action = self._scale_action_tensor(agent_idx, normalized_action)
                joint_policy_actions = list(detached_policy_actions)
                joint_policy_actions[agent_idx] = scaled_action
                global_policy_actions = torch.cat(joint_policy_actions, dim=1)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    q1_policy = self.critics[agent_idx](global_state, global_policy_actions)
                    q2_policy = self.critics_2[agent_idx](global_state, global_policy_actions)
                    min_q_policy = torch.minimum(q1_policy, q2_policy)
                    alpha = self._alpha(agent_idx).detach()
                    actor_policy_loss = (alpha * log_prob - min_q_policy).mean()
                    (
                        _action_l2,
                        _action_saturation,
                        _storage_action_l2,
                        _ev_v2g_action_l2,
                        actor_regularization,
                    ) = self._actor_action_regularization_terms(agent_idx, scaled_action)
                    behavior_cloning_loss = self._actor_behavior_cloning_loss(
                        agent_idx,
                        scaled_action,
                        behavior_actions_all[agent_idx],
                    )
                    actor_loss = (
                        actor_policy_loss
                        + actor_regularization
                        + actor_behavior_cloning_effective_weight * behavior_cloning_loss
                    )

                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(optimizer)
                actor_grad_norm = clip_grad_norm_(actor.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                actor_loss_values.append(float(actor_loss.detach().item()))
                actor_log_prob_values.append(float(log_prob.detach().mean().item()))
                actor_grad_norm_values.append(float(actor_grad_norm))
                actor_behavior_cloning_values.append(float(behavior_cloning_loss.detach().item()))
                actor_regularization_values.append(float(actor_regularization.detach().item()))

                if self.automatic_entropy_tuning:
                    alpha_loss = -(
                        self.log_alpha[agent_idx]
                        * (log_prob.detach() + float(self.target_entropy[agent_idx]))
                    ).mean()
                    self.alpha_optimizers[agent_idx].zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.alpha_optimizers[agent_idx].step()
                    alpha_loss_values.append(float(alpha_loss.detach().item()))
                else:
                    alpha_loss_values.append(0.0)
                alpha_values.append(float(self._alpha(agent_idx).detach().item()))

        if update_target_step:
            for agent_idx in range(self.num_agents):
                self._soft_update(self.critics[agent_idx], self.critic_targets[agent_idx], self.tau)
                self._soft_update(self.critics_2[agent_idx], self.critic_targets_2[agent_idx], self.tau)
                self._soft_update(self.actors[agent_idx], self.actor_targets[agent_idx], self.tau)

        if self._should_log_training_step(global_learning_step) and self.training_diagnostics_enabled:
            q1_flat = torch.cat([tensor.reshape(-1) for tensor in q1_expected_tensors])
            q2_flat = torch.cat([tensor.reshape(-1) for tensor in q2_expected_tensors])
            q_target_flat = torch.cat([target.reshape(-1) for target in q_targets])
            metrics: Dict[str, float] = {
                "MASAC/critic_1_loss_mean": float(np.mean(critic_loss_values)),
                "MASAC/critic_2_loss_mean": float(np.mean(critic_2_loss_values)),
                "MASAC/critic_td_abs_mean": float(np.mean(critic_td_abs_values)),
                "MASAC/critic_gap_abs_mean": float(np.mean(critic_gap_values)),
                "MASAC/critic_grad_norm_mean": float(np.mean(critic_grad_norm_values)),
                "MASAC/actor_update_performed": float(actor_update_due),
                "MASAC/actor_loss_mean": float(np.mean(actor_loss_values) if actor_loss_values else 0.0),
                "MASAC/actor_log_prob_mean": float(np.mean(actor_log_prob_values) if actor_log_prob_values else 0.0),
                "MASAC/actor_grad_norm_mean": float(np.mean(actor_grad_norm_values) if actor_grad_norm_values else 0.0),
                "MASAC/actor_behavior_cloning_loss_mean": float(
                    np.mean(actor_behavior_cloning_values) if actor_behavior_cloning_values else 0.0
                ),
                "MASAC/actor_behavior_cloning_effective_weight": float(actor_behavior_cloning_effective_weight),
                "MASAC/actor_regularization_mean": float(
                    np.mean(actor_regularization_values) if actor_regularization_values else 0.0
                ),
                "MASAC/alpha_loss_mean": float(np.mean(alpha_loss_values) if alpha_loss_values else 0.0),
                "MASAC/alpha_mean": float(np.mean(alpha_values) if alpha_values else self.entropy_alpha),
                "MASAC/q1_expected_mean": float(q1_flat.mean().item()),
                "MASAC/q2_expected_mean": float(q2_flat.mean().item()),
                "MASAC/q_target_mean": float(q_target_flat.mean().item()),
                "MASAC/reward_raw_mean": float(raw_rewards_all.mean().item()),
                "MASAC/reward_train_mean": float(rewards_all.mean().item()),
                "MASAC/replay_buffer_size": float(len(self.replay_buffer)),
                "MASAC/automatic_entropy_tuning": float(self.automatic_entropy_tuning),
                "MASAC/training_step_time": time.time() - update_start_time,
            }
            if self.training_diagnostics_detail == "per_agent":
                for agent_idx in range(self.num_agents):
                    metrics[f"MASAC/critic_1_loss_agent_{agent_idx}"] = critic_loss_values[agent_idx]
                    metrics[f"MASAC/critic_2_loss_agent_{agent_idx}"] = critic_2_loss_values[agent_idx]
                    metrics[f"MASAC/critic_gap_abs_agent_{agent_idx}"] = critic_gap_values[agent_idx]
                    metrics[f"MASAC/alpha_agent_{agent_idx}"] = float(self._alpha(agent_idx).detach().item())
            self._record_training_metrics(metrics, global_learning_step)

    def _alpha(self, agent_idx: int) -> torch.Tensor:
        if self.automatic_entropy_tuning:
            return self.log_alpha[agent_idx].exp()
        return torch.as_tensor(self.entropy_alpha, dtype=torch.float32, device=self.device)

    def get_diagnostic_metrics(self) -> Dict[str, float]:
        metrics = super().get_diagnostic_metrics()
        metrics.update(
            {
                "MASAC/enabled": 1.0,
                "MASAC/automatic_entropy_tuning": float(self.automatic_entropy_tuning),
                "MASAC/entropy_alpha_mean": float(
                    np.mean([float(self._alpha(agent_idx).detach().item()) for agent_idx in range(self.num_agents)])
                ),
                "MASAC/actor_update_interval": float(getattr(self, "actor_update_interval", 1)),
            }
        )
        return metrics

    def save_checkpoint(self, output_dir: str, step: int) -> str:
        checkpoint: Dict[str, Any] = {}
        for agent_idx in range(self.num_agents):
            checkpoint[f"actor_state_dict_{agent_idx}"] = self.actors[agent_idx].state_dict()
            checkpoint[f"actor_target_state_dict_{agent_idx}"] = self.actor_targets[agent_idx].state_dict()
            checkpoint[f"critic_state_dict_{agent_idx}"] = self.critics[agent_idx].state_dict()
            checkpoint[f"critic_2_state_dict_{agent_idx}"] = self.critics_2[agent_idx].state_dict()
            checkpoint[f"critic_target_state_dict_{agent_idx}"] = self.critic_targets[agent_idx].state_dict()
            checkpoint[f"critic_target_2_state_dict_{agent_idx}"] = self.critic_targets_2[agent_idx].state_dict()
            checkpoint[f"actor_optimizer_state_dict_{agent_idx}"] = self.actor_optimizers[agent_idx].state_dict()
            checkpoint[f"critic_optimizer_state_dict_{agent_idx}"] = self.critic_optimizers[agent_idx].state_dict()
            checkpoint[f"critic_optimizer_2_state_dict_{agent_idx}"] = self.critic_optimizers_2[agent_idx].state_dict()
            checkpoint[f"log_alpha_{agent_idx}"] = self.log_alpha[agent_idx].detach().cpu()
            checkpoint[f"alpha_optimizer_state_dict_{agent_idx}"] = self.alpha_optimizers[agent_idx].state_dict()

        if hasattr(self.replay_buffer, "get_state"):
            checkpoint["replay_buffer"] = self.replay_buffer.get_state()
        checkpoint["exploration_state"] = {
            "sigma": float(getattr(self, "sigma", 0.0)),
            "exploration_step": int(getattr(self, "exploration_step", 0)),
        }
        checkpoint["reward_normalization_state"] = {
            "enabled": bool(getattr(self, "reward_normalization_enabled", False)),
            "count": int(getattr(self, "reward_norm_count", 0)),
            "mean": float(getattr(self, "reward_norm_mean", 0.0)),
            "m2": float(getattr(self, "reward_norm_m2", 0.0)),
        }
        checkpoint["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        latest_path = output_dir_path / (self.checkpoint_artifact or "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)
        return str(latest_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        for agent_idx in range(self.num_agents):
            self.actors[agent_idx].load_state_dict(checkpoint[f"actor_state_dict_{agent_idx}"])
            self.actor_targets[agent_idx].load_state_dict(
                checkpoint.get(f"actor_target_state_dict_{agent_idx}", checkpoint[f"actor_state_dict_{agent_idx}"])
            )
            self.critics[agent_idx].load_state_dict(checkpoint[f"critic_state_dict_{agent_idx}"])
            self.critics_2[agent_idx].load_state_dict(checkpoint[f"critic_2_state_dict_{agent_idx}"])
            self.critic_targets[agent_idx].load_state_dict(
                checkpoint.get(f"critic_target_state_dict_{agent_idx}", checkpoint[f"critic_state_dict_{agent_idx}"])
            )
            self.critic_targets_2[agent_idx].load_state_dict(
                checkpoint.get(
                    f"critic_target_2_state_dict_{agent_idx}",
                    checkpoint[f"critic_2_state_dict_{agent_idx}"],
                )
            )
            if f"log_alpha_{agent_idx}" in checkpoint:
                with torch.no_grad():
                    self.log_alpha[agent_idx].copy_(
                        checkpoint[f"log_alpha_{agent_idx}"].to(self.device)
                    )
            if not self.fine_tune:
                self.actor_optimizers[agent_idx].load_state_dict(
                    checkpoint[f"actor_optimizer_state_dict_{agent_idx}"]
                )
                self.critic_optimizers[agent_idx].load_state_dict(
                    checkpoint[f"critic_optimizer_state_dict_{agent_idx}"]
                )
                self.critic_optimizers_2[agent_idx].load_state_dict(
                    checkpoint[f"critic_optimizer_2_state_dict_{agent_idx}"]
                )
                self.alpha_optimizers[agent_idx].load_state_dict(
                    checkpoint[f"alpha_optimizer_state_dict_{agent_idx}"]
                )

        if "replay_buffer" in checkpoint and not self.reset_replay_buffer:
            self.replay_buffer.set_state(checkpoint["replay_buffer"])

        exploration_state = checkpoint.get("exploration_state")
        if isinstance(exploration_state, dict):
            self.sigma = float(exploration_state.get("sigma", self.sigma))
            self.exploration_step = int(exploration_state.get("exploration_step", self.exploration_step))

        reward_norm_state = checkpoint.get("reward_normalization_state")
        if isinstance(reward_norm_state, dict):
            self.reward_norm_count = int(reward_norm_state.get("count", self.reward_norm_count))
            self.reward_norm_mean = float(reward_norm_state.get("mean", self.reward_norm_mean))
            self.reward_norm_m2 = float(reward_norm_state.get("m2", self.reward_norm_m2))

        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            if rng_state.get("python") is not None:
                random.setstate(rng_state["python"])
            if rng_state.get("numpy") is not None:
                np.random.set_state(rng_state["numpy"])
            if rng_state.get("torch") is not None:
                torch.set_rng_state(rng_state["torch"])
            if rng_state.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

        if self.freeze_pretrained_layers:
            self.freeze_layers(freeze_actor=True, freeze_critic=False)

        logger.info("MASAC checkpoint loaded from {}", checkpoint_file)
