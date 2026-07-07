"""Residual policy composition and direct action scaling for AgentTransformerMATD3."""
from __future__ import annotations

from typing import List

import torch


def compose_residual_actions(
    *,
    teacher_actions: torch.Tensor,
    actor_outputs: torch.Tensor,
    action_span: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    residual_action_scale: float,
    scale_mask: torch.Tensor,
) -> torch.Tensor:
    """Compose final actions using the residual formula."""
    residual = 0.5 * action_span * residual_action_scale * scale_mask * actor_outputs
    composed = teacher_actions + residual
    return torch.clamp(composed, min=action_low, max=action_high)


def scale_direct_actions(
    *,
    actor_outputs: torch.Tensor,
    action_span: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    """Scale actor outputs directly when no teacher is present."""
    del action_high
    return action_low + 0.5 * (actor_outputs + 1.0) * action_span


def build_ca_type_scale_mask(
    *,
    ca_type_names: List[str],
    storage_multiplier: float,
    ev_multiplier: float,
) -> torch.Tensor:
    """Build per-CA-token scale mask from type names."""
    multipliers: List[float] = []
    for type_name in ca_type_names:
        name_lower = type_name.lower()
        if name_lower == "storage":
            multipliers.append(storage_multiplier)
        elif name_lower == "charger":
            multipliers.append(ev_multiplier)
        else:
            multipliers.append(1.0)
    return torch.tensor(multipliers, dtype=torch.float32)


__all__ = [
    "compose_residual_actions",
    "scale_direct_actions",
    "build_ca_type_scale_mask",
]
