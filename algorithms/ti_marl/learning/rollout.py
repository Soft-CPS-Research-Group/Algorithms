"""Variable-population on-policy rollout with stable-ID GAE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

from algorithms.ti_marl.contracts.models import InterfaceSnapshot, LocalActionBundle


@dataclass(frozen=True)
class RolloutStep:
    snapshot: InterfaceSnapshot
    next_snapshot: InterfaceSnapshot
    bundles: Tuple[LocalActionBundle, ...]
    old_log_probs: Mapping[str, float]
    values: Mapping[str, float]
    next_values: Mapping[str, float]
    rewards: Mapping[str, float]
    terminated_agent_ids: Tuple[str, ...]
    truncated: bool
    reward_components_by_agent: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    group_values: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    next_group_values: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class AdvantageSample:
    step_index: int
    agent_id: str
    advantage: float
    return_value: float


@dataclass(frozen=True)
class GroupAdvantageSample:
    step_index: int
    agent_id: str
    group_id: str
    group_type: str
    advantage: float
    return_value: float


class TypedRolloutBuffer:
    def __init__(self) -> None:
        self.steps: list[RolloutStep] = []

    def __len__(self) -> int:
        return len(self.steps)

    def add(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def clear(self) -> None:
        self.steps.clear()

    def advantages(self, *, gamma: float, gae_lambda: float) -> Tuple[AdvantageSample, ...]:
        running: Dict[str, float] = {}
        samples: list[AdvantageSample] = []
        for step_index in reversed(range(len(self.steps))):
            step = self.steps[step_index]
            current_ids = set(step.snapshot.agent_ids)
            next_ids = set(step.next_snapshot.agent_ids)
            terminated = set(step.terminated_agent_ids)
            for agent_id in sorted(current_ids):
                value = float(step.values.get(agent_id, 0.0))
                survives = agent_id in next_ids and agent_id not in terminated
                next_value = float(step.next_values.get(agent_id, 0.0)) if survives else 0.0
                reward = float(step.rewards.get(agent_id, 0.0))
                delta = reward + float(gamma) * next_value * float(survives) - value
                continuation = running.get(agent_id, 0.0) if survives else 0.0
                advantage = delta + float(gamma) * float(gae_lambda) * continuation
                running[agent_id] = advantage
                samples.append(
                    AdvantageSample(
                        step_index=step_index,
                        agent_id=agent_id,
                        advantage=advantage,
                        return_value=advantage + value,
                    )
                )
            # A newly joined agent has no predecessor and therefore cannot
            # leak a GAE term into this earlier transition.
            for agent_id in next_ids - current_ids:
                running.pop(agent_id, None)
        return tuple(sorted(samples, key=lambda item: (item.step_index, item.agent_id)))

    def typed_group_advantages(
        self,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[GroupAdvantageSample, ...]:
        """Route constraint evidence only to causally related action groups.

        Every group still receives the complete economic/shared part of the
        member reward.  Penalties belonging to other typed assets are removed,
        so an EV service event cannot directly teach the stationary battery or
        a deferrable head to change its action.  The existing state-only agent
        value remains a valid (although deliberately shared) baseline.
        """

        running: Dict[tuple[str, str], float] = {}
        samples: list[GroupAdvantageSample] = []
        for step_index in reversed(range(len(self.steps))):
            step = self.steps[step_index]
            terminated_agents = set(step.terminated_agent_ids)
            next_groups = {
                (agent_id, group.group_id)
                for agent_id in step.next_snapshot.agent_ids
                for group in step.next_snapshot.groups_for(agent_id)
            }
            current_groups = {
                (agent_id, group.group_id)
                for agent_id in step.snapshot.agent_ids
                for group in step.snapshot.groups_for(agent_id)
            }
            for agent_id in step.snapshot.agent_ids:
                components = step.reward_components_by_agent.get(agent_id, {})
                total_reward = float(step.rewards.get(agent_id, 0.0))
                for group in step.snapshot.groups_for(agent_id):
                    key = (agent_id, group.group_id)
                    survives = key in next_groups and agent_id not in terminated_agents
                    value = float(
                        step.group_values.get(agent_id, {}).get(
                            group.group_id,
                            step.values.get(agent_id, 0.0),
                        )
                    )
                    next_value = float(
                        step.next_group_values.get(agent_id, {}).get(
                            group.group_id,
                            step.next_values.get(agent_id, 0.0),
                        )
                    )
                    group_reward = self._typed_group_reward(
                        total_reward,
                        components,
                        group.group_type,
                    )
                    delta = (
                        group_reward
                        + float(gamma) * next_value * float(survives)
                        - value
                    )
                    continuation = running.get(key, 0.0) if survives else 0.0
                    advantage = (
                        delta
                        + float(gamma) * float(gae_lambda) * continuation
                    )
                    running[key] = advantage
                    samples.append(
                        GroupAdvantageSample(
                            step_index=step_index,
                            agent_id=agent_id,
                            group_id=group.group_id,
                            group_type=group.group_type,
                            advantage=advantage,
                            return_value=advantage + value,
                        )
                    )
            # A newly appearing module/session has no predecessor.  Likewise,
            # a removed group cannot leak its continuation into earlier data.
            for key in next_groups - current_groups:
                running.pop(key, None)
        return tuple(
            sorted(
                samples,
                key=lambda item: (
                    item.step_index,
                    item.agent_id,
                    item.group_id,
                ),
            )
        )

    @staticmethod
    def _typed_group_reward(
        total_reward: float,
        components: Mapping[str, float],
        group_type: str,
    ) -> float:
        """Remove only penalties proven unrelated by the typed contract."""

        if not components:
            return float(total_reward)
        penalties = {
            "stationary_storage": "battery_safety_penalty",
            "ev_session": "ev_service_penalty",
            "deferrable": "deferrable_service_penalty",
        }
        own_penalty = penalties.get(str(group_type))
        if own_penalty is None:
            return float(total_reward)
        unrelated = sum(
            max(float(components.get(component, 0.0)), 0.0)
            for component in penalties.values()
            if component != own_penalty
        )
        return float(total_reward) + unrelated

    def state_dict(self) -> Mapping[str, object]:
        return {"format": "ti_marl_rollout_v1", "steps": list(self.steps)}

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("format") != "ti_marl_rollout_v1":
            raise ValueError("Unsupported TI-MARL rollout checkpoint format")
        self.steps = list(payload.get("steps", []))
