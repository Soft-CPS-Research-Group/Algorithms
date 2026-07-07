"""Exploration noise, sigma decay, gating, and phaseout for AgentTransformerMATD3."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ExplorationConfig:
    """Configuration container for exploration parameters."""
    sigma_initial: float = 0.3
    sigma_final: float = 0.05
    sigma_decay_steps: int = 10000
    noise_clip: float = 0.5
    end_initial_exploration_time_step: int = 0
    train_during_initial_exploration: bool = False
    initial_exploration_training_start_step: int = 0
    random_exploration_steps: int = 0
    phaseout_steps: int = 0
    phaseout_mode: str = "blend"


def is_initial_exploration_done(
    *,
    global_learning_step: int,
    end_initial_exploration_time_step: int,
) -> bool:
    """True when global_learning_step >= end_initial_exploration_time_step."""
    return global_learning_step >= end_initial_exploration_time_step


def should_train_on_step(
    *,
    initial_exploration_done: bool,
    train_during_initial_exploration: bool,
    global_learning_step: int,
    initial_exploration_training_start_step: int,
) -> bool:
    """Determine whether training updates should happen this step."""
    if initial_exploration_done:
        return True
    if not train_during_initial_exploration:
        return False
    return global_learning_step >= initial_exploration_training_start_step


def compute_sigma(
    *,
    exploration_step: int,
    sigma_initial: float,
    sigma_final: float,
    sigma_decay_steps: int,
) -> float:
    """Compute current exploration sigma using linear decay."""
    if sigma_decay_steps <= 0:
        return sigma_initial
    progress = min(max(float(exploration_step) / float(sigma_decay_steps), 0.0), 1.0)
    return float(sigma_initial + (sigma_final - sigma_initial) * progress)


def add_exploration_noise(
    *,
    actions: torch.Tensor,
    action_span: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    sigma: float,
    noise_clip: float,
) -> torch.Tensor:
    """Add clipped Gaussian exploration noise to actions."""
    if sigma <= 0.0:
        return actions
    noise = torch.randn_like(actions) * (sigma * action_span)
    if noise_clip > 0.0:
        clip_bound = noise_clip * action_span
        noise = torch.clamp(noise, min=-clip_bound, max=clip_bound)
    return torch.clamp(actions + noise, min=action_low, max=action_high)


def compute_phaseout_probability(
    *,
    exploration_step: int,
    phaseout_steps: int,
) -> float:
    """Compute teacher phaseout probability, linearly decaying from 1 to 0."""
    if phaseout_steps <= 0:
        return 0.0
    return max(0.0, 1.0 - float(exploration_step) / float(phaseout_steps))


def apply_exploration_phaseout(
    *,
    actor_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    phaseout_probability: float,
    mode: str,
    deterministic: bool,
) -> torch.Tensor:
    """Apply exploration phaseout blending or replacement."""
    if deterministic or phaseout_probability <= 0.0:
        return actor_actions
    if mode == "blend":
        return phaseout_probability * teacher_actions + (1.0 - phaseout_probability) * actor_actions
    if phaseout_probability >= 1.0:
        return teacher_actions
    mask = torch.rand(actor_actions.shape[0], 1, device=actor_actions.device) < phaseout_probability
    return torch.where(mask.expand_as(actor_actions), teacher_actions, actor_actions)


__all__ = [
    "ExplorationConfig",
    "add_exploration_noise",
    "compute_sigma",
    "is_initial_exploration_done",
    "should_train_on_step",
    "compute_phaseout_probability",
    "apply_exploration_phaseout",
]
