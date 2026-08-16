"""Hybrid-action multi-agent PPO over typed interface snapshots."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from algorithms.ti_marl.learning.rollout import TypedRolloutBuffer
from algorithms.ti_marl.policy.networks import CentralSetCritic, TypedActor


class TIMAPPO:
    def __init__(
        self,
        actor: TypedActor,
        critic: CentralSetCritic,
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
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(learning_rate))
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(learning_rate))
        self.rollout = TypedRolloutBuffer()
        self.update_count = 0

    def ready(self) -> bool:
        return len(self.rollout) >= self.rollout_steps

    def update(self) -> Mapping[str, float]:
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
            critic_losses: list[Tensor] = []
            entropies: list[Tensor] = []
            kls: list[Tensor] = []
            for step_index, step in enumerate(self.rollout.steps):
                decisions = {
                    bundle.agent_id: {decision.group_id: decision for decision in bundle.decisions}
                    for bundle in step.bundles
                }
                evaluation = self.actor(step.snapshot, decisions=decisions)
                values = self.critic(step.snapshot)
                for agent_id in step.snapshot.agent_ids:
                    key = (step_index, agent_id)
                    if key not in normalized:
                        continue
                    old_log_prob = torch.tensor(
                        float(step.old_log_probs[agent_id]),
                        dtype=torch.float32,
                        device=evaluation.log_prob_by_agent[agent_id].device,
                    )
                    new_log_prob = evaluation.log_prob_by_agent[agent_id]
                    ratio = torch.exp(new_log_prob - old_log_prob)
                    advantage = torch.tensor(normalized[key], device=ratio.device)
                    unclipped = ratio * advantage
                    clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
                    actor_losses.append(-torch.minimum(unclipped, clipped))
                    entropies.append(evaluation.entropy_by_agent[agent_id])
                    target = torch.tensor(returns[key], device=values[agent_id].device)
                    critic_losses.append((values[agent_id] - target).pow(2))
                    kls.append(old_log_prob - new_log_prob)

            if not actor_losses:
                break
            actor_loss = torch.stack(actor_losses).mean()
            entropy = torch.stack(entropies).mean()
            critic_loss = torch.stack(critic_losses).mean()
            approximate_kl = torch.stack(kls).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            (actor_loss - self.entropy_coeff * entropy).backward()
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=True)
            (self.value_coeff * critic_loss).backward()
            critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            metrics["actor_loss"] += float(actor_loss.detach().cpu())
            metrics["critic_loss"] += float(critic_loss.detach().cpu())
            metrics["entropy"] += float(entropy.detach().cpu())
            metrics["approx_kl"] += float(approximate_kl.detach().cpu())
            metrics["actor_grad_norm"] += float(torch.as_tensor(actor_grad).detach().cpu())
            metrics["critic_grad_norm"] += float(torch.as_tensor(critic_grad).detach().cpu())
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
            }
        )
        return result

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
