"""PPO on recorded choices with critic-first fitting and physical-time returns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from algorithms.ti_marl.learning.rollout import TypedRolloutBuffer


def credit_for_stages(variant, values, target):
    """All credits are fixed while the actor is updated (ch. 6 and §7.2)."""
    values = values.detach()
    if variant == "v2":
        return values[1:] - values[:-1]
    if variant == "v1c":
        return (values[-1] - values[0]).expand(len(values) - 1)
    return (target - values[0]).expand(len(values) - 1)


@dataclass(frozen=True)
class PrefixStep:
    snapshot: object
    traces: Mapping
    community_reward: float
    local_adjustments: Mapping[str, float]
    next_values: Mapping[str, float]
    terminated: bool
    truncated: bool
    seconds: float


def return_targets(steps, gamma_per_hour, timebase_seconds):
    """Bootstrap at rollout boundaries; do not cut the community return on site departure."""
    targets = [{} for _ in steps]
    community_tail = 0.0
    local_tail = {}
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        if index == len(steps) - 1 or step.terminated or step.truncated:
            bootstrap = {} if step.terminated else step.next_values
            community_tail = sum(bootstrap.values()) / max(1, len(bootstrap))
            local_tail = {site: value - community_tail for site, value in bootstrap.items()}
        discount = gamma_per_hour ** (step.seconds / timebase_seconds)
        community_tail = step.community_reward + discount * community_tail
        local_tail = {
            site: step.local_adjustments.get(site, 0.0) + discount * local_tail.get(site, 0.0)
            for site in step.snapshot.agent_ids
        }
        targets[index] = {site: community_tail + value for site, value in local_tail.items()}
    return targets


class PrefixPPO:
    def __init__(self, actor, critic, hyper):
        self.actor = actor
        self.critic = critic
        self.variant = actor.variant
        self.gamma_reference = float(hyper.get("gamma", 0.99))
        self.timebase = float(hyper.get("discount_timebase_seconds") or 3600.0)
        self.ppo_epochs = int(hyper.get("ppo_epochs", 4))
        self.clip_eps = float(hyper.get("clip_eps", 0.2))
        self.entropy_coeff = float(hyper.get("entropy_coeff", 0.01))
        self.max_grad_norm = float(hyper.get("max_grad_norm", 0.5))
        self.scale_floor = float(hyper.get("value_target_scale_floor", 1.0))
        self.rollout_steps = int(hyper.get("rollout_steps", 256))
        lr = float(hyper.get("learning_rate", 3e-4))
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
        self.rollout = TypedRolloutBuffer()
        self.set_seconds_per_time_step(900.0)

    def set_seconds_per_time_step(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("The TI-MARL decision duration must be positive and finite")
        self.actor.step_seconds = float(seconds)
        self.gamma = self.gamma_reference ** (seconds / self.timebase)
        # This branch uses bootstrapped Monte Carlo returns.
        self.gae_lambda = 1.0

    def ready(self):
        return len(self.rollout) >= self.rollout_steps

    def update(self, *, progress_callback=None):
        steps = self.rollout.steps
        if not steps:
            return {}
        targets = return_targets(steps, self.gamma_reference, self.timebase)
        reference = next(self.critic.parameters())
        amounts = [value for target in targets for value in target.values()]
        scale = (
            max(self.scale_floor, float(reference.new_tensor(amounts).std(unbiased=False)))
            if amounts
            else self.scale_floor
        )
        site_count = sum(not trace.supervised for step in steps for trace in step.traces.values())
        if not site_count:
            self.rollout.clear()
            return {"sampled_choices": 0.0}
        self.actor.encoder.begin_replay_preparation_cache()
        self.critic.encoder.begin_replay_preparation_cache()
        try:
            for epoch in range(self.ppo_epochs):
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss = 0.0
                for step, target in zip(steps, targets):
                    values = self.critic.prefix_values(step.snapshot, step.traces)
                    terms = [
                        (v - target[site]).square().mean()
                        for site, v in values.items()
                        if not step.traces[site].supervised
                    ]
                    if terms:
                        loss = torch.stack(terms).sum() / site_count
                        loss.backward()
                        value_loss += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

            with torch.no_grad():
                fixed = []
                for step, target in zip(steps, targets):
                    values = self.critic.prefix_values(step.snapshot, step.traces)
                    fixed.append(
                        {
                            site: credit_for_stages(self.variant, v, target[site]) / scale
                            for site, v in values.items()
                        }
                    )
            for epoch in range(self.ppo_epochs):
                self.actor_optimizer.zero_grad(set_to_none=True)
                policy_loss, sampled = 0.0, 0
                for step, credits in zip(steps, fixed):
                    scores = self.actor.replay(step.snapshot, step.traces)
                    terms = []
                    for site, choices in scores.items():
                        for k, log_prob, entropy in choices:
                            old = step.traces[site].choices[k].old_log_prob
                            ratio = (log_prob - old).exp()
                            clipped = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
                            advantage = credits[site][k]
                            terms.append(
                                -torch.minimum(ratio * advantage, clipped * advantage)
                                - self.entropy_coeff * entropy
                            )
                    if terms:
                        loss = torch.stack(terms).sum() / len(steps)
                        loss.backward()
                        policy_loss += float(loss.detach())
                        sampled += len(terms)
                if sampled:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                if progress_callback:
                    progress_callback({"phase": "prefix_ppo", "epoch": epoch + 1})
            metrics = {
                "actor_loss": policy_loss,
                "critic_loss": value_loss,
                "sampled_choices": float(sampled),
                "rollout_steps": float(len(steps)),
                "return_scale": scale,
            }
            self.rollout.clear()
            return metrics
        finally:
            self.actor.encoder.end_replay_preparation_cache()
            self.critic.encoder.end_replay_preparation_cache()

    def state_dict(self):
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "rollout": self.rollout.state_dict(),
        }

    def load_state_dict(self, state, *, restore_optimizers, restore_rollout):
        if restore_optimizers:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if restore_rollout:
            self.rollout.load_state_dict(state["rollout"])
