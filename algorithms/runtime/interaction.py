"""Typed runtime calls with one compatibility adapter for legacy/duck-typed agents."""

from __future__ import annotations

from typing import Any

from algorithms.contracts import StageContext, Transition, UpdateSchedule


def decide(
    unit: Any, observations: Any, deterministic: bool | None, context: StageContext | None = None
) -> Any:
    hook = getattr(unit, "decide", None)
    if callable(hook):
        return hook(observations, deterministic, context=context)
    if context is None:
        return unit.predict(observations, deterministic)
    return unit.predict(observations, deterministic, context=context.value)


def learn_transition(unit: Any, transition: Transition, schedule: UpdateSchedule) -> Any:
    hook = getattr(unit, "learn_transition", None)
    if callable(hook):
        return hook(transition, schedule)
    return unit.update(
        observations=transition.observations,
        actions=transition.submitted_actions,
        rewards=transition.rewards,
        next_observations=transition.next_observations,
        terminated=transition.terminated,
        truncated=transition.truncated,
        update_target_step=schedule.update_targets,
        global_learning_step=schedule.global_step,
        update_step=schedule.update,
        initial_exploration_done=schedule.exploration_done,
    )
