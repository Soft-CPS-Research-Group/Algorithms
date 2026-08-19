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
        advantages = np.asarray([sample.advantage for sample in samples], dtype=np.float32)
        advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1.0e-6)
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
                    target = torch.tensor(returns[key], device=values[agent_id].device)
                    critic_predictions.append(values[agent_id])
                    critic_targets.append(target)
                    log_ratios.append(log_ratio)
                    ratios.append(ratio)

            if not actor_losses:
                break
            actor_loss = torch.stack(actor_losses).mean()
            entropy = torch.stack(entropies).mean()
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

            actor_objective = actor_loss - self.entropy_coeff * entropy
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
