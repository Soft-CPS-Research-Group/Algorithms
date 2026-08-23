"""MATD3-specific replay-based behavior-cloning helpers.

The agent owns update orchestration and optimizer steps. This module owns the
BC-A schedule, action weighting, target reachability, and loss calculations.
The helpers accept the agent as a narrow protocol-like object to avoid a
dependency cycle with the concrete learner.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch


def effective_weight(agent: Any, global_learning_step: int) -> float:
    if not agent.bc_a_enabled or agent.bc_a_weight <= 0.0:
        return 0.0
    if agent.bc_a_decay_steps <= 0:
        return agent.bc_a_weight
    if global_learning_step <= agent.bc_a_decay_start_step:
        return agent.bc_a_weight
    progress = min(
        max(
            (global_learning_step - agent.bc_a_decay_start_step)
            / agent.bc_a_decay_steps,
            0.0,
        ),
        1.0,
    )
    return agent.bc_a_weight + (
        agent.bc_a_min_weight - agent.bc_a_weight
    ) * progress


def action_weights(agent: Any, index: int, like: torch.Tensor) -> torch.Tensor:
    values = []
    for action_name in agent._per_building[index].action_names:
        multiplier = 1.0
        if agent._is_ev_action_name(action_name):
            multiplier *= agent.bc_a_ev_multiplier
        if agent._is_storage_action_name(action_name):
            multiplier *= agent.bc_a_storage_multiplier
        if agent._is_deferrable_action_name(action_name):
            multiplier *= agent.bc_a_deferrable_multiplier
        values.append(multiplier)
    return torch.as_tensor(values, dtype=like.dtype, device=like.device).view(1, -1)


def reachable_target(
    agent: Any,
    index: int,
    cloning_action: torch.Tensor,
    *,
    base_action: Optional[torch.Tensor],
) -> torch.Tensor:
    if (
        not agent.bc_a_clip_target_to_residual_authority
        or not agent.residual_policy_enabled
        or base_action is None
    ):
        return cloning_action
    base = base_action.detach().to(cloning_action)
    if base.shape != cloning_action.shape:
        raise ValueError("BC-A base and cloning action shapes must match")
    state = agent._per_building[index]
    authority = agent._residual_action_effective_scale() * (
        agent._residual_action_scale_mask(index, cloning_action)
    )
    maximum_delta = 0.5 * (state.action_high - state.action_low) * authority
    return torch.maximum(
        torch.minimum(cloning_action, base + maximum_delta),
        base - maximum_delta,
    )


def actor_loss(
    agent: Any,
    index: int,
    predicted_action: torch.Tensor,
    cloning_action: torch.Tensor,
    *,
    base_action: Optional[torch.Tensor],
) -> torch.Tensor:
    target = reachable_target(
        agent,
        index,
        cloning_action.detach(),
        base_action=base_action,
    )
    predicted = agent._normalize_action(index, predicted_action)
    normalized_target = agent._normalize_action(index, target)
    weights = action_weights(agent, index, predicted)
    return (
        (predicted - normalized_target).square() * weights
    ).sum() / weights.expand_as(predicted).sum().clamp_min(1.0)


def actor_type_losses(
    agent: Any,
    index: int,
    predicted_action: torch.Tensor,
    cloning_action: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    error = (
        agent._normalize_action(index, predicted_action)
        - agent._normalize_action(index, cloning_action.detach())
    ).square()
    result: Dict[str, torch.Tensor] = {}
    predicates = {
        "ev": agent._is_ev_action_name,
        "storage": agent._is_storage_action_name,
        "deferrable": agent._is_deferrable_action_name,
    }
    known = torch.zeros(error.shape[-1], dtype=torch.bool, device=error.device)
    for label, predicate in predicates.items():
        mask = torch.as_tensor(
            [predicate(name) for name in agent._per_building[index].action_names],
            dtype=torch.bool,
            device=error.device,
        )
        known |= mask
        result[label] = (
            error[..., mask].mean() if mask.any() else error.new_tensor(0.0)
        )
    result["other"] = (
        error[..., ~known].mean() if (~known).any() else error.new_tensor(0.0)
    )
    return result


def extra_updates_are_due(
    agent: Any,
    *,
    effective_weight_value: float,
    cloning_actions: Optional[Sequence[torch.Tensor]],
    global_learning_step: int,
    update_count: Optional[int],
) -> bool:
    count = agent.bc_a_extra_updates if update_count is None else update_count
    return not (
        effective_weight_value <= 0.0
        or cloning_actions is None
        or count <= 0
        or (
            update_count is None
            and global_learning_step < agent.bc_a_extra_update_start_step
        )
        or (
            update_count is None
            and agent.bc_a_extra_update_end_step > 0
            and global_learning_step > agent.bc_a_extra_update_end_step
        )
    )
