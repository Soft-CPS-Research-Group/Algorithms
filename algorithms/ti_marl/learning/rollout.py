"""Variable-population on-policy rollout with stable-ID GAE."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class AdvantageSample:
    step_index: int
    agent_id: str
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

    def state_dict(self) -> Mapping[str, object]:
        return {"format": "ti_marl_rollout_v1", "steps": list(self.steps)}

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("format") != "ti_marl_rollout_v1":
            raise ValueError("Unsupported TI-MARL rollout checkpoint format")
        self.steps = list(payload.get("steps", []))
