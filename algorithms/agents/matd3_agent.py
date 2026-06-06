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
from algorithms.utils.networks import Actor, build_critic_network


class MATD3(MADDPG):
    """Multi-Agent TD3 variant.

    This keeps the same runtime/export contract as :class:`MADDPG`, but uses
    twin centralized critics per agent, clipped target policy smoothing, and
    delayed actor/target updates. It is the closest controlled comparator for
    checking whether MADDPG is limited by critic overestimation/instability.
    """

    def _initialize_networks(self):
        logger.debug("Initializing MATD3 actor and twin critic networks.")
        actor_fc_units = self.config["algorithm"]["networks"]["actor"]["layers"]
        critic_cfg = self.config["algorithm"]["networks"]["critic"]

        actors, critics, actor_targets, critic_targets = [], [], [], []
        self.critics_2, self.critic_targets_2 = [], []
        global_state_size = sum(self.observation_dimension)
        global_action_size = self._critic_global_action_feature_size()
        for agent_idx in range(self.num_agents):
            state_size = self.observation_dimension[agent_idx]
            action_size = self.action_dimension[agent_idx]

            actors.append(Actor(state_size, action_size, self.seed, actor_fc_units).to(self.device))
            actor_targets.append(Actor(state_size, action_size, self.seed, actor_fc_units).to(self.device))

            critics.append(build_critic_network(global_state_size, global_action_size, self.seed, critic_cfg).to(self.device))
            critic_targets.append(
                build_critic_network(global_state_size, global_action_size, self.seed, critic_cfg).to(self.device)
            )
            self.critics_2.append(
                build_critic_network(global_state_size, global_action_size, self.seed + 7919, critic_cfg).to(self.device)
            )
            self.critic_targets_2.append(
                build_critic_network(global_state_size, global_action_size, self.seed + 7919, critic_cfg).to(self.device)
            )

        for actor, actor_target in zip(actors, actor_targets):
            actor_target.load_state_dict(actor.state_dict())
        for critic, critic_target in zip(critics, critic_targets):
            critic_target.load_state_dict(critic.state_dict())
        for critic, critic_target in zip(self.critics_2, self.critic_targets_2):
            critic_target.load_state_dict(critic.state_dict())

        # TD3 should use smoothing/delayed actors by default, while still
        # allowing configs to override the concrete values.
        self.target_policy_smoothing = bool(
            self.config["algorithm"]["exploration"]["params"].get("target_policy_smoothing", True)
        )
        self.actor_update_interval = max(
            1,
            int(self.config["algorithm"]["exploration"]["params"].get("actor_update_interval", 2) or 2),
        )
        return actors, critics, actor_targets, critic_targets

    def _initialize_optimizers(self):
        actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.lr_actor) for actor in self.actors]
        critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=self.lr_critic) for critic in self.critics]
        self.critic_optimizers_2 = [
            torch.optim.Adam(critic.parameters(), lr=self.lr_critic) for critic in self.critics_2
        ]
        return actor_optimizers, critic_optimizers

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
        next_behavior_actions = self._transition_next_behavior_actions(behavior_actions)
        priority_boost = self._transition_observation_event_priority_boost()
        self._store_replay_transition(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            done=done,
            behavior_actions=behavior_actions,
            next_behavior_actions=next_behavior_actions,
            priority_boost=priority_boost,
        )
        self._maybe_run_actor_offline_bc_pretraining(global_learning_step)

        if len(self.replay_buffer) < self.batch_size:
            return
        if not self._should_train_on_step(initial_exploration_done, global_learning_step):
            return
        if not update_step:
            return

        if hasattr(self.replay_buffer, "sample_with_policy_context_actions"):
            states, actions_all, rewards_all, next_states, dones_all, behavior_actions_all, next_behavior_actions_all = (
                self.replay_buffer.sample_with_policy_context_actions()
            )
        elif hasattr(self.replay_buffer, "sample_with_behavior_actions"):
            states, actions_all, rewards_all, next_states, dones_all, behavior_actions_all = (
                self.replay_buffer.sample_with_behavior_actions()
            )
            next_behavior_actions_all = behavior_actions_all
        else:
            states, actions_all, rewards_all, next_states, dones_all = self.replay_buffer.sample()
            behavior_actions_all = actions_all
            next_behavior_actions_all = actions_all

        raw_rewards_all = torch.stack(rewards_all).to(self.device, dtype=torch.float32, non_blocking=True)
        rewards_all = self._normalize_reward_tensor(raw_rewards_all)
        dones_all = dones_all.to(self.device, dtype=torch.float32, non_blocking=True)
        states = [state.to(self.device, non_blocking=True) for state in states]
        actions_all = [action.to(self.device, non_blocking=True) for action in actions_all]
        behavior_actions_all = [action.to(self.device, non_blocking=True) for action in behavior_actions_all]
        next_behavior_actions_all = [action.to(self.device, non_blocking=True) for action in next_behavior_actions_all]
        next_states = [state.to(self.device, non_blocking=True) for state in next_states]

        global_state = torch.cat(states, dim=1)
        global_next_state = torch.cat(next_states, dim=1)
        global_actions = self._critic_action_features(
            actions_all,
            behavior_actions_all,
        )

        with torch.no_grad():
            next_policy_actions = []
            for agent_idx in range(self.num_agents):
                next_action = self._policy_action_from_actor_output(
                    agent_idx,
                    self.actor_targets[agent_idx](next_states[agent_idx]),
                    base_action=next_behavior_actions_all[agent_idx],
                    global_learning_step=global_learning_step,
                )
                next_action = self._add_target_policy_smoothing(agent_idx, next_action)
                next_policy_actions.append(next_action)
            global_next_actions = self._critic_action_features(
                next_policy_actions,
                next_behavior_actions_all,
            )

            q_targets = []
            for agent_idx in range(self.num_agents):
                q1_next = self.critic_targets[agent_idx](global_next_state, global_next_actions)
                q2_next = self.critic_targets_2[agent_idx](global_next_state, global_next_actions)
                q_next = torch.minimum(q1_next, q2_next)
                bootstrap_gamma = float(getattr(self, "n_step_gamma", getattr(self, "gamma", 0.99))) ** int(
                    getattr(self, "n_step_returns", 1) or 1
                )
                target = rewards_all[agent_idx] + bootstrap_gamma * q_next * (1 - dones_all[agent_idx])
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

        should_log_step_metrics = self._should_log_training_step(global_learning_step)

        with torch.no_grad():
            detached_policy_actions = [
                self._policy_action_from_actor_output(
                    agent_idx,
                    actor(state),
                    base_action=behavior_actions_all[agent_idx],
                    global_learning_step=global_learning_step,
                ).detach()
                for agent_idx, (actor, state) in enumerate(zip(self.actors, states))
            ]

        actor_update_due = global_learning_step % self.actor_update_interval == 0
        total_actor_loss = 0.0
        actor_loss_values: List[float] = []
        actor_policy_loss_values: List[float] = []
        actor_policy_loss_weighted_values: List[float] = []
        actor_policy_loss_scale_values: List[float] = []
        actor_policy_q_abs_mean_values: List[float] = []
        actor_grad_norm_values: List[float] = []
        actor_bc_values: List[float] = []
        actor_reg_values: List[float] = []
        actor_residual_delta_l2_values: List[float] = []
        actor_behavior_cloning_effective_weight = self._actor_behavior_cloning_effective_weight(
            global_learning_step
        )
        actor_policy_loss_effective_weight = self._actor_policy_loss_effective_weight(
            global_learning_step
        )

        if actor_update_due:
            for agent_idx, (actor, critic, optimizer) in enumerate(
                zip(self.actors, self.critics, self.actor_optimizers)
            ):
                predicted_action = self._policy_action_from_actor_output(
                    agent_idx,
                    actor(states[agent_idx]),
                    base_action=behavior_actions_all[agent_idx],
                    global_learning_step=global_learning_step,
                )
                joint_policy_actions = list(detached_policy_actions)
                joint_policy_actions[agent_idx] = predicted_action
                global_predicted_actions = self._critic_action_features(
                    joint_policy_actions,
                    behavior_actions_all,
                )

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    (
                        actor_policy_loss,
                        actor_policy_loss_for_optimization,
                        actor_policy_loss_scale,
                        actor_policy_q_abs_mean,
                    ) = self._actor_policy_loss_from_critic(
                        critic,
                        global_state,
                        global_predicted_actions,
                    )
                    weighted_actor_policy_loss = (
                        actor_policy_loss_effective_weight * actor_policy_loss_for_optimization
                    )
                    (
                        _action_l2,
                        _action_saturation,
                        _storage_action_l2,
                        _ev_v2g_action_l2,
                        _ev_v2g_action_mass,
                        actor_regularization,
                    ) = self._actor_action_regularization_terms(
                        agent_idx,
                        predicted_action,
                        base_action=behavior_actions_all[agent_idx],
                    )
                    behavior_cloning_loss = self._actor_behavior_cloning_loss(
                        agent_idx,
                        predicted_action,
                        behavior_actions_all[agent_idx],
                    )
                    residual_delta_l2 = self._residual_delta_l2(
                        agent_idx,
                        predicted_action,
                        behavior_actions_all[agent_idx],
                    )
                    behavior_cloning_regularization = (
                        actor_behavior_cloning_effective_weight * behavior_cloning_loss
                    )
                    actor_loss = (
                        weighted_actor_policy_loss
                        + actor_regularization
                        + behavior_cloning_regularization
                    )

                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(optimizer)
                actor_grad_norm = clip_grad_norm_(actor.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                with torch.no_grad():
                    detached_policy_actions[agent_idx] = self._policy_action_from_actor_output(
                        agent_idx,
                        actor(states[agent_idx]),
                        base_action=behavior_actions_all[agent_idx],
                        global_learning_step=global_learning_step,
                    ).detach()

                total_actor_loss += float(actor_loss.detach().item())
                actor_loss_values.append(float(actor_loss.detach().item()))
                actor_policy_loss_values.append(float(actor_policy_loss.detach().item()))
                actor_policy_loss_weighted_values.append(float(weighted_actor_policy_loss.detach().item()))
                actor_policy_loss_scale_values.append(float(actor_policy_loss_scale.detach().item()))
                actor_policy_q_abs_mean_values.append(float(actor_policy_q_abs_mean.detach().item()))
                actor_reg_values.append(float(actor_regularization.detach().item()))
                actor_bc_values.append(float(behavior_cloning_loss.detach().item()))
                actor_residual_delta_l2_values.append(float(residual_delta_l2.detach().item()))
                actor_grad_norm_values.append(float(actor_grad_norm))

                if update_target_step:
                    self._soft_update(actor, self.actor_targets[agent_idx], self.tau)
                    self._soft_update(self.critics[agent_idx], self.critic_targets[agent_idx], self.tau)
                    self._soft_update(self.critics_2[agent_idx], self.critic_targets_2[agent_idx], self.tau)
        else:
            actor_loss_values = [0.0 for _ in range(self.num_agents)]
            actor_policy_loss_values = [0.0 for _ in range(self.num_agents)]
            actor_policy_loss_weighted_values = [0.0 for _ in range(self.num_agents)]
            actor_policy_loss_scale_values = [1.0 for _ in range(self.num_agents)]
            actor_policy_q_abs_mean_values = [0.0 for _ in range(self.num_agents)]
            actor_reg_values = [0.0 for _ in range(self.num_agents)]
            actor_bc_values = [0.0 for _ in range(self.num_agents)]
            actor_residual_delta_l2_values = [0.0 for _ in range(self.num_agents)]
            actor_grad_norm_values = [0.0 for _ in range(self.num_agents)]

        if should_log_step_metrics and self.training_diagnostics_enabled:
            q1_flat = torch.cat([tensor.reshape(-1) for tensor in q1_expected_tensors])
            q2_flat = torch.cat([tensor.reshape(-1) for tensor in q2_expected_tensors])
            q_target_flat = torch.cat([target.reshape(-1) for target in q_targets])
            policy_deviation_metrics = self._tensor_action_deviation_metrics(
                detached_policy_actions,
                behavior_actions_all,
                prefix="MATD3/policy_vs_teacher",
            )
            replay_deviation_metrics = self._tensor_action_deviation_metrics(
                actions_all,
                behavior_actions_all,
                prefix="MATD3/replay_vs_teacher",
            )
            metrics: Dict[str, float] = {
                "MATD3/critic_1_loss_mean": float(np.mean(critic_loss_values)),
                "MATD3/critic_2_loss_mean": float(np.mean(critic_2_loss_values)),
                "MATD3/critic_loss_mean": float(np.mean(critic_loss_values) + np.mean(critic_2_loss_values)),
                "MATD3/critic_td_abs_mean": float(np.mean(critic_td_abs_values)),
                "MATD3/critic_gap_abs_mean": float(np.mean(critic_gap_values)),
                "MATD3/critic_grad_norm_mean": float(np.mean(critic_grad_norm_values)),
                "MATD3/actor_update_performed": float(actor_update_due),
                "MATD3/actor_loss_mean": float(np.mean(actor_loss_values)),
                "MATD3/actor_policy_loss_mean": float(np.mean(actor_policy_loss_values)),
                "MATD3/actor_policy_loss_weighted_mean": float(np.mean(actor_policy_loss_weighted_values)),
                "MATD3/actor_policy_loss_effective_weight": float(actor_policy_loss_effective_weight),
                "MATD3/actor_policy_loss_normalization_enabled": float(
                    getattr(self, "actor_policy_loss_normalization", False)
                ),
                "MATD3/actor_policy_loss_scale_mean": float(np.mean(actor_policy_loss_scale_values)),
                "MATD3/actor_policy_q_abs_mean": float(np.mean(actor_policy_q_abs_mean_values)),
                "MATD3/actor_regularization_loss_mean": float(np.mean(actor_reg_values)),
                "MATD3/actor_residual_delta_l2_mean": float(np.mean(actor_residual_delta_l2_values)),
                "MATD3/actor_behavior_cloning_loss_mean": float(np.mean(actor_bc_values)),
                "MATD3/actor_behavior_cloning_effective_weight": float(
                    actor_behavior_cloning_effective_weight
                ),
                "MATD3/critic_action_input_mode_final_base_delta": float(
                    getattr(self, "critic_action_input_mode", "final")
                    in {"final_base_delta", "final_base_delta_normalized"}
                ),
                "MATD3/critic_action_input_mode_delta_normalized": float(
                    getattr(self, "critic_action_input_mode", "final") == "final_base_delta_normalized"
                ),
                "MATD3/residual_policy_enabled": float(
                    getattr(self, "residual_policy_enabled", False)
                ),
                "MATD3/residual_action_scale_effective": float(
                    getattr(self, "_last_residual_action_scale", 0.0)
                ),
                "MATD3/actor_grad_norm_mean": float(np.mean(actor_grad_norm_values)),
                "MATD3/q1_expected_mean": float(q1_flat.mean().item()),
                "MATD3/q2_expected_mean": float(q2_flat.mean().item()),
                "MATD3/q_min_expected_mean": float(torch.minimum(q1_flat, q2_flat).mean().item()),
                "MATD3/q_target_mean": float(q_target_flat.mean().item()),
                "MATD3/reward_raw_mean": float(raw_rewards_all.mean().item()),
                "MATD3/reward_train_mean": float(rewards_all.mean().item()),
                "MATD3/replay_buffer_size": float(len(self.replay_buffer)),
                "MATD3/replay_push_count": float(getattr(self, "_replay_push_count", 0)),
                "MATD3/n_step_returns": float(getattr(self, "n_step_returns", 1)),
                "MATD3/n_step_queue_size": float(getattr(self, "_last_n_step_queue_size", 0)),
                "MATD3/target_policy_smoothing": float(getattr(self, "target_policy_smoothing", False)),
                "MATD3/target_policy_noise": float(getattr(self, "target_policy_noise", 0.0)),
                "MATD3/target_policy_noise_clip": float(getattr(self, "target_policy_noise_clip", 0.0)),
                "MATD3/actor_update_interval": float(getattr(self, "actor_update_interval", 1)),
                "MATD3/training_step_time": time.time() - update_start_time,
            }
            if self.training_diagnostics_detail == "per_agent":
                for agent_idx in range(self.num_agents):
                    metrics[f"MATD3/critic_1_loss_agent_{agent_idx}"] = critic_loss_values[agent_idx]
                    metrics[f"MATD3/critic_2_loss_agent_{agent_idx}"] = critic_2_loss_values[agent_idx]
                    metrics[f"MATD3/critic_gap_abs_agent_{agent_idx}"] = critic_gap_values[agent_idx]
                    metrics[f"MATD3/actor_loss_agent_{agent_idx}"] = actor_loss_values[agent_idx]
            metrics.update(policy_deviation_metrics)
            metrics.update(replay_deviation_metrics)
            self._record_training_metrics(metrics, global_learning_step)

    def get_diagnostic_metrics(self) -> Dict[str, float]:
        metrics = super().get_diagnostic_metrics()
        metrics.update(
            {
                "MATD3/enabled": 1.0,
                "MATD3/target_policy_smoothing": float(getattr(self, "target_policy_smoothing", False)),
                "MATD3/actor_update_interval": float(getattr(self, "actor_update_interval", 1)),
            }
        )
        return metrics

    def save_checkpoint(self, output_dir: str, step: int) -> str:
        checkpoint: Dict[str, Any] = {}
        for agent_idx in range(self.num_agents):
            checkpoint[f"actor_state_dict_{agent_idx}"] = self.actors[agent_idx].state_dict()
            checkpoint[f"critic_state_dict_{agent_idx}"] = self.critics[agent_idx].state_dict()
            checkpoint[f"critic_2_state_dict_{agent_idx}"] = self.critics_2[agent_idx].state_dict()
            checkpoint[f"actor_target_state_dict_{agent_idx}"] = self.actor_targets[agent_idx].state_dict()
            checkpoint[f"critic_target_state_dict_{agent_idx}"] = self.critic_targets[agent_idx].state_dict()
            checkpoint[f"critic_target_2_state_dict_{agent_idx}"] = self.critic_targets_2[agent_idx].state_dict()
            checkpoint[f"actor_optimizer_state_dict_{agent_idx}"] = self.actor_optimizers[agent_idx].state_dict()
            checkpoint[f"critic_optimizer_state_dict_{agent_idx}"] = self.critic_optimizers[agent_idx].state_dict()
            checkpoint[f"critic_optimizer_2_state_dict_{agent_idx}"] = self.critic_optimizers_2[agent_idx].state_dict()

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
            self.critics[agent_idx].load_state_dict(checkpoint[f"critic_state_dict_{agent_idx}"])
            self.critics_2[agent_idx].load_state_dict(checkpoint[f"critic_2_state_dict_{agent_idx}"])
            self.actor_targets[agent_idx].load_state_dict(
                checkpoint.get(f"actor_target_state_dict_{agent_idx}", checkpoint[f"actor_state_dict_{agent_idx}"])
            )
            self.critic_targets[agent_idx].load_state_dict(
                checkpoint.get(f"critic_target_state_dict_{agent_idx}", checkpoint[f"critic_state_dict_{agent_idx}"])
            )
            self.critic_targets_2[agent_idx].load_state_dict(
                checkpoint.get(
                    f"critic_target_2_state_dict_{agent_idx}",
                    checkpoint[f"critic_2_state_dict_{agent_idx}"],
                )
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

        logger.info("MATD3 checkpoint loaded from {}", checkpoint_file)
