"""Critic update helper for AgentTransformerMATD3."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from algorithms.utils.matd3_critic import TwinTransformerCritics
from algorithms.utils.matd3_global_packer import PackedGlobalSequence


@dataclass
class CriticUpdateResult:
    """Diagnostic output from a single critic update step."""
    critic_1_loss: float
    critic_2_loss: float
    total_loss: float
    mean_q1: float
    mean_q2: float
    mean_target_q: float


@torch.no_grad()
def compute_target_q(
    target_critics: TwinTransformerCritics,
    packed_next_state: PackedGlobalSequence,
    rewards: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute TD3 target: r + gamma * (1 - done) * min(Q1, Q2)."""
    min_q_next = target_critics.min_q(
        packed_next_state.global_tokens,
        packed_next_state.type_ids,
        packed_next_state.building_ids,
        packed_next_state.padding_mask,
        packed_next_state.controlled_building_indices,
    )
    return rewards + gamma * (1.0 - done) * min_q_next


def critic_update_step(
    online_critics: TwinTransformerCritics,
    optimizer: torch.optim.Optimizer,
    packed_current_state: PackedGlobalSequence,
    target_q: torch.Tensor,
) -> CriticUpdateResult:
    """Perform one gradient step on both online critics using MSE loss."""
    q1, q2 = online_critics(
        packed_current_state.global_tokens,
        packed_current_state.type_ids,
        packed_current_state.building_ids,
        packed_current_state.padding_mask,
        packed_current_state.controlled_building_indices,
    )
    loss_1 = F.mse_loss(q1, target_q)
    loss_2 = F.mse_loss(q2, target_q)
    total_loss = loss_1 + loss_2

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return CriticUpdateResult(
        critic_1_loss=loss_1.item(),
        critic_2_loss=loss_2.item(),
        total_loss=total_loss.item(),
        mean_q1=q1.mean().item(),
        mean_q2=q2.mean().item(),
        mean_target_q=target_q.mean().item(),
    )
