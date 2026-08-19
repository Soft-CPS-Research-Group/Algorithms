"""Hybrid-action multi-agent PPO over typed interface snapshots."""

from __future__ import annotations

from collections import defaultdict
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
        learning_rate: float = 3.0e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        entropy_coeff: float = 0.01,
        entropy_coeff_by_group_type: Mapping[str, float] | None = None,
        advantage_normalization: str = "global",
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
        self.rollout = TypedRolloutBuffer()
        self.update_count = 0

    def ready(self) -> bool:
        return len(self.rollout) >= self.rollout_steps

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

        metrics = defaultdict(float)
        epochs_completed = 0
        for _epoch in range(self.ppo_epochs):
            actor_losses: list[Tensor] = []
            critic_predictions: list[Tensor] = []
            critic_targets: list[Tensor] = []
            entropies: list[Tensor] = []
            entropy_bonuses: list[Tensor] = []
            entropies_by_group_type: dict[str, list[Tensor]] = defaultdict(list)
            log_ratios: list[Tensor] = []
            ratios: list[Tensor] = []
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
            for step_index, step in enumerate(self.rollout.steps):
                log_prob_by_agent = evaluation.log_prob_by_step[step_index]
                entropy_by_agent = evaluation.entropy_by_step[step_index]
                values = values_by_step[step_index]
                for agent_id in step.snapshot.agent_ids:
                    key = (step_index, agent_id)
                    if key not in normalized:
                        continue
                    old_log_prob = torch.tensor(
                        float(step.old_log_probs[agent_id]),
                        dtype=torch.float32,
                        device=log_prob_by_agent[agent_id].device,
                    )
                    new_log_prob = log_prob_by_agent[agent_id]
                    log_ratio = torch.clamp(new_log_prob - old_log_prob, -20.0, 20.0)
                    ratio = torch.exp(log_ratio)
                    advantage = torch.tensor(normalized[key], device=ratio.device)
                    unclipped = ratio * advantage
                    clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
                    actor_losses.append(-torch.minimum(unclipped, clipped))
                    entropies.append(entropy_by_agent[agent_id])
                    group_types = {
                        group.group_id: group.group_type
                        for group in step.snapshot.groups_for(agent_id)
                    }
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
                        entropies_by_group_type[group_type].append(group_entropy)
                    entropy_bonuses.append(agent_entropy_bonus)
                    target = torch.tensor(returns[key], device=values[agent_id].device)
                    critic_predictions.append(values[agent_id])
                    critic_targets.append(target)
                    log_ratios.append(log_ratio)
                    ratios.append(ratio)

            if not actor_losses:
                break
            actor_loss = torch.stack(actor_losses).mean()
            entropy = torch.stack(entropies).mean()
            entropy_bonus = torch.stack(entropy_bonuses).mean()
            critic_loss, target_mean, target_scale, raw_mse = self._value_loss(
                torch.stack(critic_predictions),
                torch.stack(critic_targets),
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

            actor_objective = actor_loss - entropy_bonus
            critic_objective = self.value_coeff * critic_loss
            for name, value in {
                "actor_objective": actor_objective,
                "critic_objective": critic_objective,
                "approximate_kl": approximate_kl,
            }.items():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Non-finite TI-PPO {name}")

            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            actor_objective.backward()
            critic_objective.backward()
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            if not bool(torch.isfinite(torch.as_tensor(actor_grad))):
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite TI-PPO actor gradient")
            if not bool(torch.isfinite(torch.as_tensor(critic_grad))):
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite TI-PPO critic gradient")
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            metrics["actor_loss"] += float(actor_loss.detach().cpu())
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
            metrics["clip_fraction"] += float(clip_fraction.detach().cpu())
            metrics["ratio_error_max"] += float(ratio_error_max.detach().cpu())
            metrics["explained_variance"] += float(explained_variance.detach().cpu())
            metrics["advantage_mean"] += float(np.mean(advantages))
            metrics["advantage_std"] += float(np.std(advantages))
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
            "rollout": self.rollout.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.rollout.load_state_dict(payload.get("rollout", {"format": "ti_marl_rollout_v1", "steps": []}))
        self.update_count = int(payload.get("update_count", 0))
