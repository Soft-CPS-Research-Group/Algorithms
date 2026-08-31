"""Hybrid-action multi-agent PPO over typed interface snapshots."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import wraps
import random
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional as F

from algorithms.ti_marl.contracts.models import ActionDecision, InterfaceSnapshot
from algorithms.ti_marl.learning.ev_planning import (
    CausalEVPlanner,
    EVPlanningTarget,
)
from algorithms.ti_marl.learning.storage_planning import (
    CausalStoragePlanner,
    StoragePlanningTarget,
)
from algorithms.ti_marl.learning.rollout import TypedRolloutBuffer
from algorithms.ti_marl.policy.networks import TypedActor


@dataclass(frozen=True)
class _EVPlanningReplayItem:
    """One immutable, deployment-causal auxiliary replay item."""

    snapshot: InterfaceSnapshot
    decisions: Mapping[str, Mapping[str, ActionDecision]]
    targets: tuple[EVPlanningTarget, ...]
    reason: str


@dataclass(frozen=True)
class _StoragePlanningReplayItem:
    """One immutable, deployment-causal storage auxiliary replay item."""

    snapshot: InterfaceSnapshot
    decisions: Mapping[str, Mapping[str, ActionDecision]]
    targets: tuple[StoragePlanningTarget, ...]
    reason: str


def _with_replay_preparation_cache(
    method: Callable[..., Mapping[str, float]],
) -> Callable[..., Mapping[str, float]]:
    """Reuse typed structural inputs throughout one PPO optimization.

    The same rollout snapshots are evaluated by the actor and critics for
    every PPO epoch.  Their values stay immutable during an update, while the
    neural parameters do not, so only the NumPy/index/device preparation is
    cached.  Cleanup is unconditional, including failed updates.
    """

    @wraps(method)
    def wrapped(self: "TIMAPPO", *args: Any, **kwargs: Any) -> Mapping[str, float]:
        encoders = []
        seen: set[int] = set()
        for model in (self.actor, self.critic, self.group_critic):
            encoder = getattr(model, "encoder", None)
            if encoder is None or id(encoder) in seen:
                continue
            seen.add(id(encoder))
            encoder.begin_replay_preparation_cache()
            encoders.append(encoder)
        try:
            return method(self, *args, **kwargs)
        finally:
            for encoder in reversed(encoders):
                encoder.end_replay_preparation_cache()

    return wrapped


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
        discount_timebase_seconds: float | None = None,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        entropy_coeff: float = 0.01,
        entropy_coeff_by_group_type: Mapping[str, float] | None = None,
        advantage_normalization: str = "global",
        policy_credit_assignment: str = "joint_agent",
        policy_anchor_coeff: float = 0.0,
        exclude_intervened_actions_from_policy_loss: bool = False,
        intervention_distillation_coeff: float = 0.0,
        ev_planner: CausalEVPlanner | None = None,
        ev_planning_auxiliary_coeff: float = 0.0,
        ev_planning_balance_targets: bool = True,
        ev_planning_fraction_coeff: float = 0.25,
        ev_planning_replay_capacity_per_reason: int = 16,
        ev_planning_replay_samples_per_reason: int = 8,
        storage_planner: CausalStoragePlanner | None = None,
        storage_planning_auxiliary_coeff: float = 0.0,
        storage_planning_balance_targets: bool = True,
        storage_planning_fraction_coeff: float = 0.25,
        storage_planning_replay_capacity_per_reason: int = 16,
        storage_planning_replay_samples_per_reason: int = 8,
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
        self.gamma_reference = float(gamma)
        self.gae_lambda_reference = float(gae_lambda)
        self.discount_timebase_seconds = (
            None
            if discount_timebase_seconds is None
            else float(discount_timebase_seconds)
        )
        if (
            self.discount_timebase_seconds is not None
            and self.discount_timebase_seconds <= 0.0
        ):
            raise ValueError("TI-MAPPO discount_timebase_seconds must be positive")
        self.seconds_per_time_step = 1.0
        self.gamma = self.gamma_reference
        self.gae_lambda = self.gae_lambda_reference
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
        self.exclude_intervened_actions_from_policy_loss = bool(
            exclude_intervened_actions_from_policy_loss
        )
        if (
            self.exclude_intervened_actions_from_policy_loss
            and self.policy_credit_assignment != "typed_group"
        ):
            raise ValueError(
                "TI-MAPPO intervention-aware masking requires typed_group credit"
            )
        self.intervention_distillation_coeff = float(
            intervention_distillation_coeff
        )
        if self.intervention_distillation_coeff < 0.0:
            raise ValueError(
                "TI-MAPPO intervention_distillation_coeff must be non-negative"
            )
        if (
            self.intervention_distillation_coeff > 0.0
            and not self.exclude_intervened_actions_from_policy_loss
        ):
            raise ValueError(
                "TI-MAPPO intervention distillation requires intervention-aware "
                "policy masking"
            )
        self.ev_planner = ev_planner
        self.ev_planning_auxiliary_coeff = float(ev_planning_auxiliary_coeff)
        if self.ev_planning_auxiliary_coeff < 0.0:
            raise ValueError(
                "TI-MAPPO ev_planning_auxiliary_coeff must be non-negative"
            )
        if self.ev_planning_auxiliary_coeff > 0.0 and self.ev_planner is None:
            raise ValueError(
                "TI-MAPPO EV planning auxiliary loss requires a causal EV planner"
            )
        self.ev_planning_balance_targets = bool(ev_planning_balance_targets)
        self.ev_planning_fraction_coeff = float(ev_planning_fraction_coeff)
        if self.ev_planning_fraction_coeff < 0.0:
            raise ValueError(
                "TI-MAPPO EV planning fraction coefficient must be non-negative"
            )
        self.ev_planning_replay_capacity_per_reason = int(
            ev_planning_replay_capacity_per_reason
        )
        self.ev_planning_replay_samples_per_reason = int(
            ev_planning_replay_samples_per_reason
        )
        if self.ev_planning_replay_capacity_per_reason < 0:
            raise ValueError(
                "TI-MAPPO EV planning replay capacity must be non-negative"
            )
        if self.ev_planning_replay_samples_per_reason < 0:
            raise ValueError(
                "TI-MAPPO EV planning replay sample count must be non-negative"
            )
        self._ev_planning_replay: dict[str, list[_EVPlanningReplayItem]] = (
            defaultdict(list)
        )
        self._ev_planning_replay_seen: Counter[str] = Counter()
        self.storage_planner = storage_planner
        self.storage_planning_auxiliary_coeff = float(
            storage_planning_auxiliary_coeff
        )
        if self.storage_planning_auxiliary_coeff < 0.0:
            raise ValueError(
                "TI-MAPPO storage_planning_auxiliary_coeff must be non-negative"
            )
        if (
            self.storage_planning_auxiliary_coeff > 0.0
            and self.storage_planner is None
        ):
            raise ValueError(
                "TI-MAPPO storage planning auxiliary loss requires a causal "
                "storage planner"
            )
        self.storage_planning_balance_targets = bool(
            storage_planning_balance_targets
        )
        self.storage_planning_fraction_coeff = float(
            storage_planning_fraction_coeff
        )
        if self.storage_planning_fraction_coeff < 0.0:
            raise ValueError(
                "TI-MAPPO storage planning fraction coefficient must be "
                "non-negative"
            )
        self.storage_planning_replay_capacity_per_reason = int(
            storage_planning_replay_capacity_per_reason
        )
        self.storage_planning_replay_samples_per_reason = int(
            storage_planning_replay_samples_per_reason
        )
        if self.storage_planning_replay_capacity_per_reason < 0:
            raise ValueError(
                "TI-MAPPO storage planning replay capacity must be non-negative"
            )
        if self.storage_planning_replay_samples_per_reason < 0:
            raise ValueError(
                "TI-MAPPO storage planning replay sample count must be non-negative"
            )
        self._storage_planning_replay: dict[
            str, list[_StoragePlanningReplayItem]
        ] = defaultdict(list)
        self._storage_planning_replay_seen: Counter[str] = Counter()
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

    def set_seconds_per_time_step(self, seconds: float) -> None:
        """Resolve PPO discount factors against physical rather than step time.

        When ``discount_timebase_seconds`` is configured, ``gamma`` and
        ``gae_lambda`` describe that reference period (normally one hour).  The
        effective per-transition values are exponentiated by the actual runtime
        step duration.  With no timebase configured, historical step-based
        behaviour is preserved exactly.
        """

        seconds = float(seconds)
        if not np.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("TI-MAPPO seconds_per_time_step must be positive")
        self.seconds_per_time_step = seconds
        if self.discount_timebase_seconds is None:
            self.gamma = self.gamma_reference
            self.gae_lambda = self.gae_lambda_reference
            return
        exponent = seconds / self.discount_timebase_seconds
        self.gamma = self.gamma_reference**exponent
        self.gae_lambda = self.gae_lambda_reference**exponent

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

    @_with_replay_preparation_cache
    def update(
        self,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Mapping[str, float]:
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
        final_decisions_by_step = [
            self._overlay_final_decisions(step, decisions)
            for step, decisions in zip(self.rollout.steps, decisions_by_step)
        ]
        intervened_groups_by_step = [
            self._intervened_group_keys(step)
            for step in self.rollout.steps
        ]
        intervened_policy_samples = (
            sum(len(items) for items in intervened_groups_by_step)
            if self.exclude_intervened_actions_from_policy_loss
            else 0
        )
        distillable_groups_by_step = [
            intervened
            & {
                (bundle.agent_id, decision.group_id)
                for bundle in step.final_bundles
                for decision in bundle.decisions
            }
            for step, intervened in zip(
                self.rollout.steps, intervened_groups_by_step
            )
        ]
        intervention_distillation_samples = (
            sum(len(items) for items in distillable_groups_by_step)
            if self.intervention_distillation_coeff > 0.0
            else 0
        )
        ev_targets_by_step = [
            (
                self.ev_planner.targets(
                    step.snapshot,
                    seconds_per_time_step=self.seconds_per_time_step,
                )
                if self.ev_planner is not None
                and self.ev_planning_auxiliary_coeff > 0.0
                else ()
            )
            for step in self.rollout.steps
        ]
        current_ev_planning_items = self._ev_planning_items(
            snapshots=tuple(step.snapshot for step in self.rollout.steps),
            decisions_by_step=decisions_by_step,
            targets_by_step=ev_targets_by_step,
        )
        replay_ev_planning_items = self._sample_ev_planning_replay()
        ev_planning_items = current_ev_planning_items + replay_ev_planning_items
        ev_planning_current_samples = sum(
            len(item.targets) for item in current_ev_planning_items
        )
        ev_planning_replay_samples = sum(
            len(item.targets) for item in replay_ev_planning_items
        )
        ev_planning_samples = sum(
            len(item.targets) for item in ev_planning_items
        )
        ev_planning_charge_samples = sum(
            target.decision.mode == "CHARGE_EV"
            for item in ev_planning_items
            for target in item.targets
        )
        ev_planning_discharge_samples = sum(
            target.decision.mode == "DISCHARGE_EV"
            for item in ev_planning_items
            for target in item.targets
        )
        ev_planning_reason_counts = Counter(
            target.reason
            for item in ev_planning_items
            for target in item.targets
        )
        storage_targets_by_step = [
            (
                self.storage_planner.targets(
                    step.snapshot,
                    seconds_per_time_step=self.seconds_per_time_step,
                )
                if self.storage_planner is not None
                and self.storage_planning_auxiliary_coeff > 0.0
                else ()
            )
            for step in self.rollout.steps
        ]
        current_storage_planning_items = self._storage_planning_items(
            snapshots=tuple(step.snapshot for step in self.rollout.steps),
            decisions_by_step=decisions_by_step,
            targets_by_step=storage_targets_by_step,
        )
        replay_storage_planning_items = self._sample_storage_planning_replay()
        storage_planning_items = (
            current_storage_planning_items + replay_storage_planning_items
        )
        storage_planning_current_samples = sum(
            len(item.targets) for item in current_storage_planning_items
        )
        storage_planning_replay_samples = sum(
            len(item.targets) for item in replay_storage_planning_items
        )
        storage_planning_samples = sum(
            len(item.targets) for item in storage_planning_items
        )
        storage_planning_charge_samples = sum(
            target.decision.mode == "CHARGE_STATIONARY"
            for item in storage_planning_items
            for target in item.targets
        )
        storage_planning_discharge_samples = sum(
            target.decision.mode == "DISCHARGE_STATIONARY"
            for item in storage_planning_items
            for target in item.targets
        )
        storage_planning_reason_counts = Counter(
            target.reason
            for item in storage_planning_items
            for target in item.targets
        )
        actor_samples_by_group_type = Counter(
            sample.group_type for sample in group_samples
        )
        intervened_policy_samples_by_group_type: Counter[str] = Counter()
        if self.exclude_intervened_actions_from_policy_loss:
            for sample in group_samples:
                if (sample.agent_id, sample.group_id) in (
                    intervened_groups_by_step[sample.step_index]
                ):
                    intervened_policy_samples_by_group_type[sample.group_type] += 1
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
        for epoch in range(self.ppo_epochs):
            epoch_started = time.perf_counter()
            actor_losses: list[Tensor] = []
            policy_anchor_losses: list[Tensor] = []
            intervention_distillation_losses: list[Tensor] = []
            ev_planning_losses: list[Tensor] = []
            ev_planning_losses_by_mode_and_reason: dict[
                str, dict[str, list[Tensor]]
            ] = defaultdict(lambda: defaultdict(list))
            ev_planning_mode_correct: list[Tensor] = []
            ev_planning_mode_correct_by_mode: dict[str, list[Tensor]] = defaultdict(
                list
            )
            ev_planning_fraction_losses: list[Tensor] = []
            storage_planning_losses: list[Tensor] = []
            storage_planning_losses_by_mode_and_reason: dict[
                str, dict[str, list[Tensor]]
            ] = defaultdict(lambda: defaultdict(list))
            storage_planning_mode_correct: list[Tensor] = []
            storage_planning_mode_correct_by_mode: dict[
                str, list[Tensor]
            ] = defaultdict(list)
            storage_planning_fraction_losses: list[Tensor] = []
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
            intervention_evaluation = None
            if (
                self.intervention_distillation_coeff > 0.0
                and intervention_distillation_samples > 0
            ):
                intervention_evaluation = self.actor.evaluate_actions_many(
                    tuple(
                        (step.snapshot, decisions)
                        for step, decisions in zip(
                            self.rollout.steps, final_decisions_by_step
                        )
                    )
                )
            ev_planning_evaluation = None
            if ev_planning_samples > 0:
                ev_planning_evaluation = self.actor.evaluate_actions_many(
                    tuple(
                        (item.snapshot, item.decisions)
                        for item in ev_planning_items
                    )
                )
            storage_planning_evaluation = None
            if storage_planning_samples > 0:
                storage_planning_evaluation = self.actor.evaluate_actions_many(
                    tuple(
                        (item.snapshot, item.decisions)
                        for item in storage_planning_items
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
                            assert group_values_by_step is not None
                            group_critic_predictions.append(
                                group_values_by_step[step_index][agent_id][
                                    group_id
                                ]
                            )
                            group_critic_targets.append(
                                torch.tensor(
                                    group_returns[group_key],
                                    device=values[agent_id].device,
                                )
                            )
                            if (
                                self.exclude_intervened_actions_from_policy_loss
                                and (agent_id, group_id)
                                in intervened_groups_by_step[step_index]
                            ):
                                if (
                                    intervention_evaluation is not None
                                    and (agent_id, group_id)
                                    in distillable_groups_by_step[step_index]
                                ):
                                    intervention_distillation_losses.append(
                                        -intervention_evaluation.log_prob_by_group_step[
                                            step_index
                                        ][agent_id][group_id]
                                    )
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
                    target = torch.tensor(returns[key], device=values[agent_id].device)
                    critic_predictions.append(values[agent_id])
                    critic_targets.append(target)

            if ev_planning_evaluation is not None:
                for replay_index, item in enumerate(ev_planning_items):
                    for target in item.targets:
                        target_mode_log_prob = (
                            ev_planning_evaluation.mode_log_prob_by_group_step[
                                replay_index
                            ][target.agent_id][target.group_id]
                        )
                        target_loss = -target_mode_log_prob
                        ev_planning_losses.append(target_loss)
                        ev_planning_losses_by_mode_and_reason[
                            target.decision.mode
                        ][target.reason].append(target_loss)
                        predicted_mode = (
                            ev_planning_evaluation.predicted_mode_by_group_step[
                                replay_index
                            ][target.agent_id][target.group_id]
                        )
                        mode_correct = (
                            predicted_mode == int(target.decision.mode_index)
                        ).float()
                        ev_planning_mode_correct.append(mode_correct)
                        ev_planning_mode_correct_by_mode[
                            target.decision.mode
                        ].append(mode_correct)
                        if target.decision.mode in {
                            "CHARGE_EV",
                            "DISCHARGE_EV",
                        }:
                            predicted_fraction = (
                                ev_planning_evaluation
                                .predicted_fraction_by_group_step[replay_index][
                                    target.agent_id
                                ][target.group_id]
                            )
                            ev_planning_fraction_losses.append(
                                F.smooth_l1_loss(
                                    predicted_fraction,
                                    predicted_fraction.new_tensor(
                                        float(target.decision.fraction)
                                    ),
                                )
                            )

            if storage_planning_evaluation is not None:
                for replay_index, item in enumerate(storage_planning_items):
                    for target in item.targets:
                        target_mode_log_prob = (
                            storage_planning_evaluation
                            .mode_log_prob_by_group_step[replay_index][
                                target.agent_id
                            ][target.group_id]
                        )
                        target_loss = -target_mode_log_prob
                        storage_planning_losses.append(target_loss)
                        storage_planning_losses_by_mode_and_reason[
                            target.decision.mode
                        ][target.reason].append(target_loss)
                        predicted_mode = (
                            storage_planning_evaluation
                            .predicted_mode_by_group_step[replay_index][
                                target.agent_id
                            ][target.group_id]
                        )
                        mode_correct = (
                            predicted_mode == int(target.decision.mode_index)
                        ).float()
                        storage_planning_mode_correct.append(mode_correct)
                        storage_planning_mode_correct_by_mode[
                            target.decision.mode
                        ].append(mode_correct)
                        if target.decision.mode in {
                            "CHARGE_STATIONARY",
                            "DISCHARGE_STATIONARY",
                        }:
                            predicted_fraction = (
                                storage_planning_evaluation
                                .predicted_fraction_by_group_step[replay_index][
                                    target.agent_id
                                ][target.group_id]
                            )
                            storage_planning_fraction_losses.append(
                                F.smooth_l1_loss(
                                    predicted_fraction,
                                    predicted_fraction.new_tensor(
                                        float(target.decision.fraction)
                                    ),
                                )
                            )

            actor_loss = (
                torch.stack(actor_losses).mean()
                if actor_losses
                else next(self.actor.parameters()).sum() * 0.0
            )
            policy_anchor_loss = (
                torch.stack(policy_anchor_losses).mean()
                if policy_anchor_losses
                else actor_loss.new_zeros(())
            )
            intervention_distillation_loss = (
                torch.stack(intervention_distillation_losses).mean()
                if intervention_distillation_losses
                else actor_loss.new_zeros(())
            )
            ev_planning_unbalanced_mode_loss = (
                torch.stack(ev_planning_losses).mean()
                if ev_planning_losses
                else actor_loss.new_zeros(())
            )
            ev_planning_mode_loss = self._ev_planning_loss(
                ev_planning_losses_by_mode_and_reason,
                fallback=ev_planning_unbalanced_mode_loss,
            )
            ev_planning_fraction_loss = self._mean_or_zero(
                ev_planning_fraction_losses,
                reference=actor_loss,
            )
            ev_planning_loss = (
                ev_planning_mode_loss
                + self.ev_planning_fraction_coeff * ev_planning_fraction_loss
            )
            ev_planning_mode_accuracy = (
                torch.stack(ev_planning_mode_correct).mean()
                if ev_planning_mode_correct
                else actor_loss.new_zeros(())
            )
            ev_planning_charge_recall = self._mean_or_zero(
                ev_planning_mode_correct_by_mode.get("CHARGE_EV", ()),
                reference=actor_loss,
            )
            ev_planning_discharge_recall = self._mean_or_zero(
                ev_planning_mode_correct_by_mode.get("DISCHARGE_EV", ()),
                reference=actor_loss,
            )
            ev_planning_idle_recall = self._mean_or_zero(
                ev_planning_mode_correct_by_mode.get("IDLE", ()),
                reference=actor_loss,
            )
            storage_planning_unbalanced_mode_loss = (
                torch.stack(storage_planning_losses).mean()
                if storage_planning_losses
                else actor_loss.new_zeros(())
            )
            storage_planning_mode_loss = self._storage_planning_loss(
                storage_planning_losses_by_mode_and_reason,
                fallback=storage_planning_unbalanced_mode_loss,
            )
            storage_planning_fraction_loss = self._mean_or_zero(
                storage_planning_fraction_losses,
                reference=actor_loss,
            )
            storage_planning_loss = (
                storage_planning_mode_loss
                + self.storage_planning_fraction_coeff
                * storage_planning_fraction_loss
            )
            storage_planning_mode_accuracy = (
                torch.stack(storage_planning_mode_correct).mean()
                if storage_planning_mode_correct
                else actor_loss.new_zeros(())
            )
            storage_planning_charge_recall = self._mean_or_zero(
                storage_planning_mode_correct_by_mode.get(
                    "CHARGE_STATIONARY", ()
                ),
                reference=actor_loss,
            )
            storage_planning_discharge_recall = self._mean_or_zero(
                storage_planning_mode_correct_by_mode.get(
                    "DISCHARGE_STATIONARY", ()
                ),
                reference=actor_loss,
            )
            storage_planning_idle_recall = self._mean_or_zero(
                storage_planning_mode_correct_by_mode.get("IDLE", ()),
                reference=actor_loss,
            )
            entropy = (
                torch.stack(entropies).mean()
                if entropies
                else actor_loss.new_zeros(())
            )
            entropy_bonus = (
                torch.stack(entropy_bonuses).mean()
                if entropy_bonuses
                else actor_loss.new_zeros(())
            )
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
            ratio_tensor = (
                torch.stack(ratios)
                if ratios
                else actor_loss.new_ones((1,))
            )
            log_ratio_tensor = (
                torch.stack(log_ratios)
                if log_ratios
                else actor_loss.new_zeros((1,))
            )
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
                + self.intervention_distillation_coeff
                * intervention_distillation_loss
                + self.ev_planning_auxiliary_coeff * ev_planning_loss
                + self.storage_planning_auxiliary_coeff
                * storage_planning_loss
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
            metrics["intervention_distillation_loss"] += float(
                intervention_distillation_loss.detach().cpu()
            )
            metrics["ev_planning_loss"] += float(
                ev_planning_loss.detach().cpu()
            )
            metrics["ev_planning_mode_loss"] += float(
                ev_planning_mode_loss.detach().cpu()
            )
            metrics["ev_planning_unbalanced_mode_loss"] += float(
                ev_planning_unbalanced_mode_loss.detach().cpu()
            )
            metrics["ev_planning_fraction_loss"] += float(
                ev_planning_fraction_loss.detach().cpu()
            )
            metrics["ev_planning_mode_accuracy"] += float(
                ev_planning_mode_accuracy.detach().cpu()
            )
            metrics["ev_planning_charge_recall"] += float(
                ev_planning_charge_recall.detach().cpu()
            )
            metrics["ev_planning_discharge_recall"] += float(
                ev_planning_discharge_recall.detach().cpu()
            )
            metrics["ev_planning_idle_recall"] += float(
                ev_planning_idle_recall.detach().cpu()
            )
            metrics["storage_planning_loss"] += float(
                storage_planning_loss.detach().cpu()
            )
            metrics["storage_planning_mode_loss"] += float(
                storage_planning_mode_loss.detach().cpu()
            )
            metrics["storage_planning_unbalanced_mode_loss"] += float(
                storage_planning_unbalanced_mode_loss.detach().cpu()
            )
            metrics["storage_planning_fraction_loss"] += float(
                storage_planning_fraction_loss.detach().cpu()
            )
            metrics["storage_planning_mode_accuracy"] += float(
                storage_planning_mode_accuracy.detach().cpu()
            )
            metrics["storage_planning_charge_recall"] += float(
                storage_planning_charge_recall.detach().cpu()
            )
            metrics["storage_planning_discharge_recall"] += float(
                storage_planning_discharge_recall.detach().cpu()
            )
            metrics["storage_planning_idle_recall"] += float(
                storage_planning_idle_recall.detach().cpu()
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
            epoch_duration = time.perf_counter() - epoch_started
            epoch_payload = {
                "phase": "ti_mappo_epoch",
                "update_current": self.update_count + 1,
                "epoch_current": epoch + 1,
                "epoch_total": self.ppo_epochs,
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "approx_kl": float(approximate_kl.detach().cpu()),
                "epoch_duration_seconds": epoch_duration,
            }
            logger.info(
                "event=ti_mappo_epoch update={} epoch={}/{} actor_loss={:.8f} "
                "critic_loss={:.8f} approx_kl={:.8f} duration_seconds={:.3f}",
                self.update_count + 1,
                epoch + 1,
                self.ppo_epochs,
                epoch_payload["actor_loss"],
                epoch_payload["critic_loss"],
                epoch_payload["approx_kl"],
                epoch_duration,
            )
            if progress_callback is not None:
                try:
                    progress_callback(epoch_payload)
                except Exception as exc:  # progress is best-effort telemetry
                    logger.warning("Unable to report TI-MAPPO progress: {}", exc)
            if self.target_kl is not None and float(approximate_kl.detach().cpu()) > self.target_kl:
                break

        self._remember_ev_planning_items(current_ev_planning_items)
        self._remember_storage_planning_items(current_storage_planning_items)
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
                "intervention_distillation_coeff": float(
                    self.intervention_distillation_coeff
                ),
                "intervention_distillation_samples": float(
                    intervention_distillation_samples
                ),
                "ev_planning_auxiliary_coeff": float(
                    self.ev_planning_auxiliary_coeff
                ),
                "ev_planning_balance_targets": float(
                    self.ev_planning_balance_targets
                ),
                "ev_planning_fraction_coeff": float(
                    self.ev_planning_fraction_coeff
                ),
                "ev_planning_samples": float(ev_planning_samples),
                "ev_planning_current_samples": float(
                    ev_planning_current_samples
                ),
                "ev_planning_replay_samples": float(
                    ev_planning_replay_samples
                ),
                "ev_planning_charge_samples": float(
                    ev_planning_charge_samples
                ),
                "ev_planning_discharge_samples": float(
                    ev_planning_discharge_samples
                ),
                "ev_planning_idle_samples": float(
                    ev_planning_samples
                    - ev_planning_charge_samples
                    - ev_planning_discharge_samples
                ),
                "storage_planning_auxiliary_coeff": float(
                    self.storage_planning_auxiliary_coeff
                ),
                "storage_planning_balance_targets": float(
                    self.storage_planning_balance_targets
                ),
                "storage_planning_fraction_coeff": float(
                    self.storage_planning_fraction_coeff
                ),
                "storage_planning_samples": float(storage_planning_samples),
                "storage_planning_current_samples": float(
                    storage_planning_current_samples
                ),
                "storage_planning_replay_samples": float(
                    storage_planning_replay_samples
                ),
                "storage_planning_charge_samples": float(
                    storage_planning_charge_samples
                ),
                "storage_planning_discharge_samples": float(
                    storage_planning_discharge_samples
                ),
                "storage_planning_idle_samples": float(
                    storage_planning_samples
                    - storage_planning_charge_samples
                    - storage_planning_discharge_samples
                ),
                "effective_gamma": float(self.gamma),
                "effective_gae_lambda": float(self.gae_lambda),
                "seconds_per_time_step": float(self.seconds_per_time_step),
                "intervened_policy_samples": float(intervened_policy_samples),
                "eligible_actor_samples": float(
                    max(
                        (len(group_samples) if group_samples else len(samples))
                        - intervened_policy_samples,
                        0,
                    )
                ),
                "update_seconds": float(time.perf_counter() - update_started),
            }
        )
        for group_type, sample_count in sorted(
            actor_samples_by_group_type.items()
        ):
            intervened_count = intervened_policy_samples_by_group_type.get(
                group_type, 0
            )
            result[
                f"intervened_policy_samples_{group_type}"
            ] = float(intervened_count)
            result[
                f"eligible_actor_samples_{group_type}"
            ] = float(max(sample_count - intervened_count, 0))
        for reason, count in sorted(ev_planning_reason_counts.items()):
            result[f"ev_planning_samples_{reason}"] = float(count)
        result["ev_planning_replay_size"] = float(
            sum(len(items) for items in self._ev_planning_replay.values())
        )
        for reason, items in sorted(self._ev_planning_replay.items()):
            result[f"ev_planning_replay_size_{reason}"] = float(len(items))
            result[f"ev_planning_replay_seen_{reason}"] = float(
                self._ev_planning_replay_seen[reason]
            )
        for reason, count in sorted(storage_planning_reason_counts.items()):
            result[f"storage_planning_samples_{reason}"] = float(count)
        result["storage_planning_replay_size"] = float(
            sum(len(items) for items in self._storage_planning_replay.values())
        )
        for reason, items in sorted(self._storage_planning_replay.items()):
            result[f"storage_planning_replay_size_{reason}"] = float(len(items))
            result[f"storage_planning_replay_seen_{reason}"] = float(
                self._storage_planning_replay_seen[reason]
            )
        result["evaluated_samples_per_second"] = (
            float(len(samples) * epochs_completed)
            / max(result["update_seconds"], 1.0e-9)
        )
        return result

    @staticmethod
    def _ev_planning_items(
        *,
        snapshots,
        decisions_by_step,
        targets_by_step,
    ) -> tuple[_EVPlanningReplayItem, ...]:
        """Group current targets by causal reason for stratified replay."""

        items: list[_EVPlanningReplayItem] = []
        for snapshot, raw_decisions, targets in zip(
            snapshots,
            decisions_by_step,
            targets_by_step,
        ):
            targets_by_reason: dict[str, list[EVPlanningTarget]] = defaultdict(
                list
            )
            for target in targets:
                targets_by_reason[target.reason].append(target)
            for reason, reason_targets in sorted(targets_by_reason.items()):
                target_agent_ids = tuple(
                    agent_id
                    for agent_id in snapshot.agent_ids
                    if any(
                        target.agent_id == agent_id
                        for target in reason_targets
                    )
                )
                replay_snapshot = TIMAPPO._compact_planning_snapshot(
                    snapshot,
                    agent_ids=target_agent_ids,
                )
                overlaid = {
                    agent_id: dict(raw_decisions.get(agent_id, {}))
                    for agent_id in target_agent_ids
                }
                for target in reason_targets:
                    overlaid.setdefault(target.agent_id, {})[
                        target.group_id
                    ] = target.decision
                items.append(
                    _EVPlanningReplayItem(
                        snapshot=replay_snapshot,
                        decisions=overlaid,
                        targets=tuple(reason_targets),
                        reason=reason,
                    )
                )
        return tuple(items)

    @staticmethod
    def _storage_planning_items(
        *,
        snapshots,
        decisions_by_step,
        targets_by_step,
    ) -> tuple[_StoragePlanningReplayItem, ...]:
        """Group current storage targets by causal reason for replay."""

        items: list[_StoragePlanningReplayItem] = []
        for snapshot, raw_decisions, targets in zip(
            snapshots,
            decisions_by_step,
            targets_by_step,
        ):
            targets_by_reason: dict[
                str, list[StoragePlanningTarget]
            ] = defaultdict(list)
            for target in targets:
                targets_by_reason[target.reason].append(target)
            for reason, reason_targets in sorted(targets_by_reason.items()):
                target_agent_ids = tuple(
                    agent_id
                    for agent_id in snapshot.agent_ids
                    if any(
                        target.agent_id == agent_id
                        for target in reason_targets
                    )
                )
                replay_snapshot = TIMAPPO._compact_planning_snapshot(
                    snapshot,
                    agent_ids=target_agent_ids,
                )
                overlaid = {
                    agent_id: dict(raw_decisions.get(agent_id, {}))
                    for agent_id in target_agent_ids
                }
                for target in reason_targets:
                    overlaid.setdefault(target.agent_id, {})[
                        target.group_id
                    ] = target.decision
                items.append(
                    _StoragePlanningReplayItem(
                        snapshot=replay_snapshot,
                        decisions=overlaid,
                        targets=tuple(reason_targets),
                        reason=reason,
                    )
                )
        return tuple(items)

    @staticmethod
    def _compact_planning_snapshot(
        snapshot: InterfaceSnapshot,
        *,
        agent_ids: tuple[str, ...],
    ) -> InterfaceSnapshot:
        """Keep only local actor inputs for agents carrying causal targets."""

        selected = frozenset(agent_ids)

        return replace(
            snapshot,
            agent_ids=agent_ids,
            modules=(),
            entities=(),
            fault_evidence=(),
            health=(),
            observation_parts=tuple(
                part
                for part in snapshot.observation_parts
                if part.policy_input and part.owner_agent_id in selected
            ),
            action_groups=tuple(
                group
                for group in snapshot.action_groups
                if group.owner_agent_id in selected
            ),
            dependencies=(),
            constraints=(),
            shared_resources=(),
            closure_log=(),
            execution_feedback=(),
            topology_events=(),
        )

    def _sample_ev_planning_replay(self) -> tuple[_EVPlanningReplayItem, ...]:
        count = self.ev_planning_replay_samples_per_reason
        if count <= 0:
            return ()
        samples: list[_EVPlanningReplayItem] = []
        for reason, reservoir in sorted(self._ev_planning_replay.items()):
            if not reservoir:
                continue
            samples.extend(
                random.sample(reservoir, min(count, len(reservoir)))
            )
        return tuple(samples)

    def _sample_storage_planning_replay(
        self,
    ) -> tuple[_StoragePlanningReplayItem, ...]:
        count = self.storage_planning_replay_samples_per_reason
        if count <= 0:
            return ()
        samples: list[_StoragePlanningReplayItem] = []
        for _reason, reservoir in sorted(self._storage_planning_replay.items()):
            if reservoir:
                samples.extend(
                    random.sample(reservoir, min(count, len(reservoir)))
                )
        return tuple(samples)

    def _remember_ev_planning_items(
        self,
        items: tuple[_EVPlanningReplayItem, ...],
    ) -> None:
        """Maintain a bounded reservoir independently for each target reason."""

        capacity = self.ev_planning_replay_capacity_per_reason
        if capacity <= 0:
            return
        for item in items:
            reason = item.reason
            self._ev_planning_replay_seen[reason] += 1
            seen = self._ev_planning_replay_seen[reason]
            reservoir = self._ev_planning_replay[reason]
            if len(reservoir) < capacity:
                reservoir.append(item)
                continue
            replacement_index = random.randrange(seen)
            if replacement_index < capacity:
                reservoir[replacement_index] = item

    def _remember_storage_planning_items(
        self,
        items: tuple[_StoragePlanningReplayItem, ...],
    ) -> None:
        """Maintain a bounded storage reservoir for every causal reason."""

        capacity = self.storage_planning_replay_capacity_per_reason
        if capacity <= 0:
            return
        for item in items:
            reason = item.reason
            self._storage_planning_replay_seen[reason] += 1
            seen = self._storage_planning_replay_seen[reason]
            reservoir = self._storage_planning_replay[reason]
            if len(reservoir) < capacity:
                reservoir.append(item)
                continue
            replacement_index = random.randrange(seen)
            if replacement_index < capacity:
                reservoir[replacement_index] = item

    def _ev_planning_loss(
        self,
        losses_by_mode_and_reason: Mapping[str, Mapping[str, list[Tensor]]],
        *,
        fallback: Tensor,
    ) -> Tensor:
        """Return a target-balanced EV loss without duplicating samples.

        Connected EV data naturally contains many more safe ``IDLE`` instants
        than useful charging opportunities.  An ordinary sample mean therefore
        rewards an actor that predicts IDLE almost everywhere.  The balanced
        objective gives equal mass to the available action modes and, within a
        mode, equal mass to the available causal reasons (for example urgent
        service versus a cheap tariff opportunity).  This affects training
        only; the planner never replaces the actor at inference time.
        """

        if not self.ev_planning_balance_targets:
            return fallback
        mode_losses: list[Tensor] = []
        for reason_groups in losses_by_mode_and_reason.values():
            reason_losses = [
                torch.stack(losses).mean()
                for losses in reason_groups.values()
                if losses
            ]
            if reason_losses:
                mode_losses.append(torch.stack(reason_losses).mean())
        return torch.stack(mode_losses).mean() if mode_losses else fallback

    def _storage_planning_loss(
        self,
        losses_by_mode_and_reason: Mapping[
            str, Mapping[str, list[Tensor]]
        ],
        *,
        fallback: Tensor,
    ) -> Tensor:
        """Balance rare storage charge/discharge modes against abundant IDLE."""

        if not self.storage_planning_balance_targets:
            return fallback
        mode_losses: list[Tensor] = []
        for reason_groups in losses_by_mode_and_reason.values():
            reason_losses = [
                torch.stack(losses).mean()
                for losses in reason_groups.values()
                if losses
            ]
            if reason_losses:
                mode_losses.append(torch.stack(reason_losses).mean())
        return torch.stack(mode_losses).mean() if mode_losses else fallback

    @staticmethod
    def _mean_or_zero(values, *, reference: Tensor) -> Tensor:
        return (
            torch.stack(tuple(values)).mean()
            if values
            else reference.new_zeros(())
        )

    @staticmethod
    def _overlay_final_decisions(step, raw_decisions):
        """Build a complete action map while preserving absent-group masking."""

        overlaid = {
            agent_id: dict(decisions)
            for agent_id, decisions in raw_decisions.items()
        }
        for bundle in tuple(getattr(step, "final_bundles", ()) or ()):
            agent = overlaid.setdefault(bundle.agent_id, {})
            for decision in bundle.decisions:
                agent[decision.group_id] = decision
        return overlaid

    @staticmethod
    def _intervened_group_keys(step) -> set[tuple[str, str]]:
        """Return groups whose executed local decision differs from the sample.

        PPO ratios are valid for sampled actions.  A safety-projected decision
        is still useful to the critics, but attributing its return to the raw
        categorical/continuous sample creates a false policy-gradient signal.
        """

        final_bundles = tuple(getattr(step, "final_bundles", ()) or ())
        if not final_bundles:
            return set()
        raw = {
            (bundle.agent_id, decision.group_id): decision
            for bundle in step.bundles
            for decision in bundle.decisions
        }
        final = {
            (bundle.agent_id, decision.group_id): decision
            for bundle in final_bundles
            for decision in bundle.decisions
        }
        return {
            key
            for key, raw_decision in raw.items()
            if key not in final
            or raw_decision.mode != final[key].mode
            or abs(float(raw_decision.fraction) - float(final[key].fraction))
            > 1.0e-9
        }

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
            "discount": {
                "gamma_reference": self.gamma_reference,
                "gae_lambda_reference": self.gae_lambda_reference,
                "discount_timebase_seconds": self.discount_timebase_seconds,
                "seconds_per_time_step": self.seconds_per_time_step,
                "effective_gamma": self.gamma,
                "effective_gae_lambda": self.gae_lambda,
            },
            "ev_planning": {
                "auxiliary_coeff": self.ev_planning_auxiliary_coeff,
                "balance_targets": self.ev_planning_balance_targets,
                "fraction_coeff": self.ev_planning_fraction_coeff,
                "replay_capacity_per_reason": (
                    self.ev_planning_replay_capacity_per_reason
                ),
                "replay_samples_per_reason": (
                    self.ev_planning_replay_samples_per_reason
                ),
                "replay": {
                    reason: tuple(items)
                    for reason, items in self._ev_planning_replay.items()
                },
                "replay_seen": dict(self._ev_planning_replay_seen),
                "planner": (
                    None
                    if self.ev_planner is None
                    else dict(self.ev_planner.configuration())
                ),
            },
            "storage_planning": {
                "auxiliary_coeff": self.storage_planning_auxiliary_coeff,
                "balance_targets": self.storage_planning_balance_targets,
                "fraction_coeff": self.storage_planning_fraction_coeff,
                "replay_capacity_per_reason": (
                    self.storage_planning_replay_capacity_per_reason
                ),
                "replay_samples_per_reason": (
                    self.storage_planning_replay_samples_per_reason
                ),
                "replay": {
                    reason: tuple(items)
                    for reason, items in self._storage_planning_replay.items()
                },
                "replay_seen": dict(self._storage_planning_replay_seen),
                "planner": (
                    None
                    if self.storage_planner is None
                    else dict(self.storage_planner.configuration())
                ),
            },
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
        ev_planning_state = dict(payload.get("ev_planning", {}))
        restored_replay = (
            dict(ev_planning_state.get("replay", {}))
            if restore_rollout
            else {}
        )
        capacity = self.ev_planning_replay_capacity_per_reason
        self._ev_planning_replay = defaultdict(
            list,
            {
                str(reason): list(items)[-capacity:]
                if capacity > 0
                else []
                for reason, items in restored_replay.items()
            },
        )
        self._ev_planning_replay_seen = Counter(
            {
                str(reason): int(count)
                for reason, count in (
                    dict(ev_planning_state.get("replay_seen", {})).items()
                    if restore_rollout
                    else ()
                )
            }
        )
        storage_planning_state = dict(payload.get("storage_planning", {}))
        restored_storage_replay = (
            dict(storage_planning_state.get("replay", {}))
            if restore_rollout
            else {}
        )
        storage_capacity = self.storage_planning_replay_capacity_per_reason
        self._storage_planning_replay = defaultdict(
            list,
            {
                str(reason): list(items)[-storage_capacity:]
                if storage_capacity > 0
                else []
                for reason, items in restored_storage_replay.items()
            },
        )
        self._storage_planning_replay_seen = Counter(
            {
                str(reason): int(count)
                for reason, count in (
                    dict(storage_planning_state.get("replay_seen", {})).items()
                    if restore_rollout
                    else ()
                )
            }
        )
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
