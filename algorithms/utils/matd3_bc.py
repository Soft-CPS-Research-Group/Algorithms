"""Replay-native behavior cloning for AgentTransformerMATD3."""
from __future__ import annotations

from typing import List

import torch


def compute_bc_effective_weight(
    *,
    global_learning_step: int,
    initial_weight: float,
    min_weight: float,
    decay_start_step: int,
    decay_steps: int,
) -> float:
    """Compute current BC effective weight using linear decay."""
    if initial_weight <= 0.0:
        return 0.0
    if global_learning_step < decay_start_step:
        return float(initial_weight)
    if decay_steps <= 0:
        return float(initial_weight)
    progress = min(
        max(float(global_learning_step - decay_start_step) / float(decay_steps), 0.0),
        1.0,
    )
    return float(initial_weight + (min_weight - initial_weight) * progress)


def compute_ca_type_weights(
    *,
    ca_type_names: List[str],
    ev_multiplier: float,
    storage_multiplier: float,
) -> torch.Tensor:
    """Build per-CA-token weights based on type names."""
    weights: List[float] = []
    for type_name in ca_type_names:
        name_lower = type_name.lower()
        if name_lower == "storage":
            weights.append(storage_multiplier)
        elif name_lower == "charger":
            weights.append(ev_multiplier)
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_bc_loss(
    *,
    actor_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    ca_type_weights: torch.Tensor,
    effective_weight: float,
) -> torch.Tensor:
    """Compute weighted MSE BC loss from replay-sampled teacher actions."""
    if effective_weight <= 0.0:
        return actor_actions.new_tensor(0.0)
    squared_error = (actor_actions - teacher_actions).pow(2)
    weights_expanded = ca_type_weights.view(1, -1).expand_as(squared_error)
    weighted_error = squared_error * weights_expanded
    denominator = weights_expanded.sum().clamp_min(1.0)
    return (weighted_error.sum() / denominator) * effective_weight


__all__ = [
    "compute_bc_loss",
    "compute_bc_effective_weight",
    "compute_ca_type_weights",
]
