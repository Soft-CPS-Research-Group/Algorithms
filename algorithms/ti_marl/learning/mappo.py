"""Hybrid-action multi-agent PPO over typed interface snapshots."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from algorithms.ti_marl.learning.rollout import TypedRolloutBuffer
from algorithms.ti_marl.policy.networks import TypedActor


class TIMAPPO:
    def __init__(
        self,
        actor: TypedActor,
        critic: nn.Module,
        *,
        group_critic: nn.Module | None = None,
        learning_rate: float = 3.0e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        entropy_coeff: float = 0.01,
        entropy_coeff_by_group_type: Mapping[str, float] | None = None,
        advantage_normalization: str = "global",
        policy_credit_assignment: str = "joint_agent",
        policy_anchor_coeff: float = 0.0,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = 0.03,
        rollout_steps: int = 256,
        normalize_value_targets: bool = True,
        value_target_scale_floor: float = 1.0,
        critic_loss: str = "huber",
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.group_critic = group_critic
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.ppo_epochs = int(ppo_epochs)
        self.entropy_coeff = float(entropy_coeff)
        self.entropy_coeff_by_group_type = {
            str(key): float(value)
            for key, value in dict(entropy_coeff_by_group_type or {}).items()
        }
        unknown_group_types = sorted(
            set(self.entropy_coeff_by_group_type) - set(actor.group_modes)
        ) if hasattr(actor, "group_modes") else []
        if unknown_group_types:
            raise ValueError(
                "Unknown TI-MAPPO entropy action-group type(s): "
                f"{unknown_group_types}"
            )
        if any(value < 0.0 for value in self.entropy_coeff_by_group_type.values()):
            raise ValueError("TI-MAPPO entropy coefficients must be non-negative")
        self.advantage_normalization = str(advantage_normalization)
        if self.advantage_normalization not in {"global", "per_agent"}:
            raise ValueError(
                "TI-MAPPO advantage_normalization must be 'global' or 'per_agent'"
            )
        self.policy_credit_assignment = str(policy_credit_assignment)
        if self.policy_credit_assignment not in {"joint_agent", "typed_group"}:
            raise ValueError(
                "TI-MAPPO policy_credit_assignment must be 'joint_agent' or "
                "'typed_group'"
            )
        if (
            self.policy_credit_assignment == "typed_group"
            and self.group_critic is None
        ):
            raise ValueError(
                "TI-MAPPO typed_group credit requires a typed group critic"
            )
        self.policy_anchor_coeff = float(policy_anchor_coeff)
        if self.policy_anchor_coeff < 0.0:
            raise ValueError("TI-MAPPO policy_anchor_coeff must be non-negative")
        self.policy_anchor_actor: TypedActor | None = None
        if self.policy_anchor_coeff > 0.0:
            self.reset_policy_anchor()
        self.value_coeff = float(value_coeff)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = None if target_kl is None else float(target_kl)
        self.rollout_steps = int(rollout_steps)
        self.normalize_value_targets = bool(normalize_value_targets)
        self.value_target_scale_floor = float(value_target_scale_floor)
        if critic_loss not in {"mse", "huber"}:
            raise ValueError("TIMAPPO critic_loss must be one of {'mse', 'huber'}")
        self.critic_loss = str(critic_loss)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(learning_rate))
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(learning_rate))
        self.group_critic_optimizer = (
            None
            if self.group_critic is None
            else torch.optim.Adam(
                self.group_critic.parameters(), lr=float(learning_rate)
            )
        )
        self.rollout = TypedRolloutBuffer()
        self.update_count = 0

    def ready(self) -> bool:
        return len(self.rollout) >= self.rollout_steps

    def reset_policy_anchor(self) -> None:
        """Freeze the current actor as the conservative PPO reference policy."""

        if self.policy_anchor_coeff <= 0.0:
            self.policy_anchor_actor = None
            return
        anchor = deepcopy(self.actor).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        self.policy_anchor_actor = anchor

    def update(self) -> Mapping[str, float]:
        update_started = time.perf_counter()
        samples = self.rollout.advantages(gamma=self.gamma, gae_lambda=self.gae_lambda)
        if not samples:
            return {}
        advantages = self._normalize_advantages(samples)
        normalized = {
            (sample.step_index, sample.agent_id): float(value)
            for sample, value in zip(samples, advantages)
        }
        returns = {
            (sample.step_index, sample.agent_id): float(sample.return_value)
            for sample in samples
        }
        group_samples = ()
        normalized_groups: dict[tuple[int, str, str], float] = {}
        policy_advantages = advantages
        if self.policy_credit_assignment == "typed_group":
            group_samples = self.rollout.typed_group_advantages(
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            group_advantages = self._normalize_group_advantages(group_samples)
            normalized_groups = {
                (sample.step_index, sample.agent_id, sample.group_id): float(value)
                for sample, value in zip(group_samples, group_advantages)
            }
            group_returns = {
                (sample.step_index, sample.agent_id, sample.group_id): float(
                    sample.return_value
                )
                for sample in group_samples
            }
            policy_advantages = group_advantages
        else:
            group_returns = {}

        metrics = defaultdict(float)
        epochs_completed = 0
        decisions_by_step = [
            {
                bundle.agent_id: {
                    decision.group_id: decision
                    for decision in bundle.decisions
                }
                for bundle in step.bundles
            }
            for step in self.rollout.steps
        ]
        anchor_evaluation = None
        if self.policy_anchor_coeff > 0.0:
            if self.policy_anchor_actor is None:
                raise RuntimeError("TI-MAPPO policy anchor is not initialized")
            with torch.no_grad():
                anchor_evaluation = self.policy_anchor_actor.evaluate_actions_many(
                    tuple(
                        (step.snapshot, decisions)
                        for step, decisions in zip(
                            self.rollout.steps, decisions_by_step
                        )
                    )
                )
        for _epoch in range(self.ppo_epochs):
            actor_losses: list[Tensor] = []
            policy_anchor_losses: list[Tensor] = []
            critic_predictions: list[Tensor] = []
            critic_targets: list[Tensor] = []
            group_critic_predictions: list[Tensor] = []
            group_critic_targets: list[Tensor] = []
            entropies: list[Tensor] = []
            entropy_bonuses: list[Tensor] = []
            entropies_by_group_type: dict[str, list[Tensor]] = defaultdict(list)
            log_ratios: list[Tensor] = []
            ratios: list[Tensor] = []
            evaluation = self.actor.evaluate_actions_many(
                tuple(
                    (step.snapshot, decisions)
                    for step, decisions in zip(
                        self.rollout.steps, decisions_by_step
                    )
                )
            )
            values_by_step = self.critic.forward_many(
                tuple(step.snapshot for step in self.rollout.steps)
            )
            group_values_by_step = (
                None
                if self.group_critic is None
                else self.group_critic.forward_many(
                    tuple(step.snapshot for step in self.rollout.steps)
                )
            )
            for step_index, step in enumerate(self.rollout.steps):
                log_prob_by_agent = evaluation.log_prob_by_step[step_index]
                entropy_by_agent = evaluation.entropy_by_step[step_index]
                values = values_by_step[step_index]
                for agent_id in step.snapshot.agent_ids:
                    key = (step_index, agent_id)
                    if key not in normalized:
                        continue
                    group_types = {
                        group.group_id: group.group_type
                        for group in step.snapshot.groups_for(agent_id)
                    }
                    if self.policy_credit_assignment == "joint_agent":
                        old_log_prob = torch.tensor(
                            float(step.old_log_probs[agent_id]),
                            dtype=torch.float32,
                            device=log_prob_by_agent[agent_id].device,
                        )
                        new_log_prob = log_prob_by_agent[agent_id]
                        log_ratio = torch.clamp(
                            new_log_prob - old_log_prob, -20.0, 20.0
                        )
                        ratio = torch.exp(log_ratio)
                        advantage = torch.tensor(
                            normalized[key], device=ratio.device
                        )
                        unclipped = ratio * advantage
                        clipped = torch.clamp(
                            ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                        ) * advantage
                        actor_losses.append(-torch.minimum(unclipped, clipped))
                        if anchor_evaluation is not None:
                            anchor_log_prob = anchor_evaluation.log_prob_by_step[
                                step_index
                            ][agent_id]
                            policy_anchor_losses.append(
                                (new_log_prob - anchor_log_prob).pow(2)
                            )
                        entropies.append(entropy_by_agent[agent_id])
                        agent_entropy_bonus = torch.zeros((), device=ratio.device)
                        for group_id, group_entropy in (
                            evaluation.entropy_by_group_step[step_index]
                            .get(agent_id, {})
                            .items()
                        ):
                            group_type = group_types[group_id]
                            coefficient = self.entropy_coeff_by_group_type.get(
                                group_type, self.entropy_coeff
                            )
                            agent_entropy_bonus = (
                                agent_entropy_bonus + coefficient * group_entropy
                            )
                            entropies_by_group_type[group_type].append(
                                group_entropy
                            )
                        entropy_bonuses.append(agent_entropy_bonus)
                        log_ratios.append(log_ratio)
                        ratios.append(ratio)
                    else:
                        stored_decisions = decisions_by_step[step_index][agent_id]
                        current_log_probs = (
                            evaluation.log_prob_by_group_step[step_index]
                            .get(agent_id, {})
                        )
                        current_entropies = (
                            evaluation.entropy_by_group_step[step_index]
                            .get(agent_id, {})
                        )
                        for group_id, group_type in group_types.items():
                            group_key = (step_index, agent_id, group_id)
                            if group_key not in normalized_groups:
                                continue
                            new_log_prob = current_log_probs[group_id]
                            old_log_prob = torch.tensor(
                                float(stored_decisions[group_id].raw_log_prob),
                                dtype=torch.float32,
                                device=new_log_prob.device,
                            )
                            log_ratio = torch.clamp(
                                new_log_prob - old_log_prob, -20.0, 20.0
                            )
                            ratio = torch.exp(log_ratio)
                            advantage = torch.tensor(
                                normalized_groups[group_key], device=ratio.device
                            )
                            unclipped = ratio * advantage
                            clipped = torch.clamp(
                                ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                            ) * advantage
                            actor_losses.append(
                                -torch.minimum(unclipped, clipped)
                            )
                            if anchor_evaluation is not None:
                                anchor_log_prob = (
                                    anchor_evaluation.log_prob_by_group_step[
                                        step_index
                                    ][agent_id][group_id]
                                )
                                policy_anchor_losses.append(
                                    (new_log_prob - anchor_log_prob).pow(2)
                                )
                            group_entropy = current_entropies[group_id]
                            entropies.append(group_entropy)
                            coefficient = self.entropy_coeff_by_group_type.get(
                                group_type, self.entropy_coeff
                            )
                            entropy_bonuses.append(
                                coefficient * group_entropy
                            )
                            entropies_by_group_type[group_type].append(
                                group_entropy
                            )
                            log_ratios.append(log_ratio)
                            ratios.append(ratio)
                            assert group_values_by_step is not None
                            group_critic_predictions.append(
                                group_values_by_step[step_index][agent_id][
                                    group_id
                                ]
                            )
                            group_critic_targets.append(
                                torch.tensor(
                                    group_returns[group_key],
                                    device=ratio.device,
                                )
                            )
                    target = torch.tensor(returns[key], device=values[agent_id].device)
                    critic_predictions.append(values[agent_id])
                    critic_targets.append(target)

            if not actor_losses:
                break
            actor_loss = torch.stack(actor_losses).mean()
            policy_anchor_loss = (
                torch.stack(policy_anchor_losses).mean()
                if policy_anchor_losses
                else actor_loss.new_zeros(())
            )
            entropy = torch.stack(entropies).mean()
            entropy_bonus = torch.stack(entropy_bonuses).mean()
            critic_loss, target_mean, target_scale, raw_mse = self._value_loss(
                torch.stack(critic_predictions),
                torch.stack(critic_targets),
            )
            group_critic_loss = None
            group_target_mean = None
            group_target_scale = None
            group_raw_mse = None
            if group_critic_predictions:
                (
                    group_critic_loss,
                    group_target_mean,
                    group_target_scale,
                    group_raw_mse,
                ) = self._value_loss(
                    torch.stack(group_critic_predictions),
                    torch.stack(group_critic_targets),
                )
            ratio_tensor = torch.stack(ratios)
            log_ratio_tensor = torch.stack(log_ratios)
            # Schulman's non-negative sample estimator: (r - 1) - log(r).
            approximate_kl = ((ratio_tensor - 1.0) - log_ratio_tensor).mean()
            clip_fraction = (
                torch.abs(ratio_tensor - 1.0) > self.clip_eps
            ).float().mean()
            ratio_error_max = torch.abs(ratio_tensor - 1.0).max()
            prediction_tensor = torch.stack(critic_predictions)
            target_tensor = torch.stack(critic_targets)
            target_variance = torch.var(target_tensor, unbiased=False)
            explained_variance = torch.where(
                target_variance > 1.0e-8,
                1.0
                - torch.var(
                    target_tensor - prediction_tensor.detach(), unbiased=False
                )
                / target_variance,
                torch.zeros_like(target_variance),
            )

            actor_objective = (
                actor_loss
                - entropy_bonus
                + self.policy_anchor_coeff * policy_anchor_loss
            )
            critic_objective = self.value_coeff * critic_loss
            group_critic_objective = (
                None
                if group_critic_loss is None
                else self.value_coeff * group_critic_loss
            )
            for name, value in {
                "actor_objective": actor_objective,
                "critic_objective": critic_objective,
                "approximate_kl": approximate_kl,
                **(
                    {}
                    if group_critic_objective is None
                    else {"group_critic_objective": group_critic_objective}
                ),
            }.items():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Non-finite TI-PPO {name}")

            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            if self.group_critic_optimizer is not None:
                self.group_critic_optimizer.zero_grad(set_to_none=True)
            actor_objective.backward()
            critic_objective.backward()
            if group_critic_objective is not None:
                group_critic_objective.backward()
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            group_critic_grad = (
                None
                if self.group_critic is None
                else nn.utils.clip_grad_norm_(
                    self.group_critic.parameters(), self.max_grad_norm
                )
            )
            if not bool(torch.isfinite(torch.as_tensor(actor_grad))):
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite TI-PPO actor gradient")
            if not bool(torch.isfinite(torch.as_tensor(critic_grad))):
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite TI-PPO critic gradient")
            if group_critic_grad is not None and not bool(
                torch.isfinite(torch.as_tensor(group_critic_grad))
            ):
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                assert self.group_critic_optimizer is not None
                self.group_critic_optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "Non-finite TI-PPO group critic gradient"
                )
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            if self.group_critic_optimizer is not None:
                self.group_critic_optimizer.step()

            metrics["actor_loss"] += float(actor_loss.detach().cpu())
            metrics["policy_anchor_loss"] += float(
                policy_anchor_loss.detach().cpu()
            )
            metrics["critic_loss"] += float(critic_loss.detach().cpu())
            metrics["entropy"] += float(entropy.detach().cpu())
            metrics["entropy_bonus"] += float(entropy_bonus.detach().cpu())
            for group_type, values in entropies_by_group_type.items():
                metrics[f"entropy_{group_type}"] += float(
                    torch.stack(values).mean().detach().cpu()
                )
            metrics["approx_kl"] += float(approximate_kl.detach().cpu())
            metrics["actor_grad_norm"] += float(torch.as_tensor(actor_grad).detach().cpu())
            metrics["critic_grad_norm"] += float(torch.as_tensor(critic_grad).detach().cpu())
            metrics["critic_target_mean"] += float(target_mean.detach().cpu())
            metrics["critic_target_scale"] += float(target_scale.detach().cpu())
            metrics["critic_raw_mse"] += float(raw_mse.detach().cpu())
            if group_critic_loss is not None:
                assert group_target_mean is not None
                assert group_target_scale is not None
                assert group_raw_mse is not None
                metrics["group_critic_loss"] += float(
                    group_critic_loss.detach().cpu()
                )
                metrics["group_critic_grad_norm"] += float(
                    torch.as_tensor(group_critic_grad).detach().cpu()
                )
                metrics["group_critic_target_mean"] += float(
                    group_target_mean.detach().cpu()
                )
                metrics["group_critic_target_scale"] += float(
                    group_target_scale.detach().cpu()
                )
                metrics["group_critic_raw_mse"] += float(
                    group_raw_mse.detach().cpu()
                )
            metrics["clip_fraction"] += float(clip_fraction.detach().cpu())
            metrics["ratio_error_max"] += float(ratio_error_max.detach().cpu())
            metrics["explained_variance"] += float(explained_variance.detach().cpu())
            metrics["advantage_mean"] += float(np.mean(policy_advantages))
            metrics["advantage_std"] += float(np.std(policy_advantages))
            metrics["return_mean"] += float(target_tensor.detach().mean().cpu())
            metrics["return_std"] += float(
                target_tensor.detach().std(unbiased=False).cpu()
            )
            epochs_completed += 1
            if self.target_kl is not None and float(approximate_kl.detach().cpu()) > self.target_kl:
                break

        self.update_count += 1
        self.rollout.clear()
        divisor = max(epochs_completed, 1)
        result = {key: value / divisor for key, value in metrics.items()}
        result.update(
            {
                "epochs": float(epochs_completed),
                "updates": float(self.update_count),
                "samples": float(len(samples)),
                "actor_samples": float(
                    len(group_samples) if group_samples else len(samples)
                ),
                "policy_anchor_coeff": float(self.policy_anchor_coeff),
                "update_seconds": float(time.perf_counter() - update_started),
            }
        )
        result["evaluated_samples_per_second"] = (
            float(len(samples) * epochs_completed)
            / max(result["update_seconds"], 1.0e-9)
        )
        return result

    def _normalize_advantages(self, samples) -> np.ndarray:
        advantages = np.asarray(
            [sample.advantage for sample in samples], dtype=np.float32
        )
        if self.advantage_normalization == "global":
            return (advantages - advantages.mean()) / max(
                float(advantages.std()), 1.0e-6
            )
        normalized = np.zeros_like(advantages)
        indices_by_agent: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            indices_by_agent[sample.agent_id].append(index)
        for indices in indices_by_agent.values():
            values = advantages[indices]
            normalized[indices] = (values - values.mean()) / max(
                float(values.std()), 1.0e-6
            )
        return normalized

    def _normalize_group_advantages(self, samples) -> np.ndarray:
        advantages = np.asarray(
            [sample.advantage for sample in samples], dtype=np.float32
        )
        if not len(advantages):
            return advantages
        if self.advantage_normalization == "global":
            return (advantages - advantages.mean()) / max(
                float(advantages.std()), 1.0e-6
            )
        normalized = np.zeros_like(advantages)
        indices_by_agent_and_type: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            indices_by_agent_and_type[(sample.agent_id, sample.group_type)].append(
                index
            )
        for indices in indices_by_agent_and_type.values():
            values = advantages[indices]
            normalized[indices] = (values - values.mean()) / max(
                float(values.std()), 1.0e-6
            )
        return normalized

    def _value_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        target_mean = targets.detach().mean()
        target_scale = targets.detach().std(unbiased=False)
        if not self.normalize_value_targets:
            target_mean = torch.zeros_like(target_mean)
            target_scale = torch.ones_like(target_scale)
        else:
            target_scale = torch.clamp(
                target_scale,
                min=self.value_target_scale_floor,
            )
        normalized_predictions = (predictions - target_mean) / target_scale
        normalized_targets = (targets - target_mean) / target_scale
        if self.critic_loss == "huber":
            loss = F.smooth_l1_loss(
                normalized_predictions,
                normalized_targets,
            )
        else:
            loss = F.mse_loss(normalized_predictions, normalized_targets)
        raw_mse = F.mse_loss(predictions.detach(), targets.detach())
        return loss, target_mean, target_scale, raw_mse

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "group_critic_optimizer": (
                None
                if self.group_critic_optimizer is None
                else self.group_critic_optimizer.state_dict()
            ),
            "rollout": self.rollout.state_dict(),
            "update_count": self.update_count,
            "policy_anchor_actor": (
                None
                if self.policy_anchor_actor is None
                else self.policy_anchor_actor.state_dict()
            ),
        }

    def load_state_dict(
        self,
        payload: Mapping[str, Any],
        *,
        restore_optimizers: bool = True,
        restore_rollout: bool = True,
    ) -> None:
        if restore_optimizers:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if self.group_critic_optimizer is not None and restore_optimizers:
            group_optimizer = payload.get("group_critic_optimizer")
            if group_optimizer is None:
                raise ValueError(
                    "TI-MARL typed group checkpoint is missing its optimizer"
                )
            self.group_critic_optimizer.load_state_dict(group_optimizer)
        if restore_rollout:
            self.rollout.load_state_dict(
                payload.get(
                    "rollout",
                    {"format": "ti_marl_rollout_v1", "steps": []},
                )
            )
        else:
            self.rollout.clear()
        self.update_count = int(payload.get("update_count", 0))
        if self.policy_anchor_coeff > 0.0:
            anchor_state = payload.get("policy_anchor_actor")
            self.reset_policy_anchor()
            if anchor_state is not None:
                assert self.policy_anchor_actor is not None
                self.policy_anchor_actor.load_state_dict(anchor_state)
        else:
            self.policy_anchor_actor = None
