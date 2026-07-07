"""Unit tests for the critic update loop."""
from __future__ import annotations

import torch

from algorithms.utils.matd3_critic import TwinTransformerCritics
from algorithms.utils.matd3_global_packer import PackedGlobalSequence
from algorithms.utils.matd3_critic_update import (
    compute_target_q,
    critic_update_step,
    CriticUpdateResult,
)


def _make_critics(d_model=16) -> TwinTransformerCritics:
    return TwinTransformerCritics(
        d_model=d_model, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0.0,
        num_token_types=8, max_buildings=4,
    )


def _make_packed_sequence(batch_size=4, n_tokens=8, d_model=16) -> PackedGlobalSequence:
    return PackedGlobalSequence(
        global_tokens=torch.randn(batch_size, n_tokens, d_model),
        type_ids=torch.zeros(batch_size, n_tokens, dtype=torch.long),
        building_ids=torch.zeros(batch_size, n_tokens, dtype=torch.long),
        padding_mask=torch.zeros(batch_size, n_tokens, dtype=torch.bool),
        controlled_building_indices=[0],
    )


class TestComputeTargetQ:
    def test_target_q_shape(self):
        target_critics = _make_critics()
        target_critics.eval()
        packed_next = _make_packed_sequence(batch_size=4, n_tokens=8)
        rewards = torch.randn(4, 1)
        done = torch.zeros(4, 1)

        target_q = compute_target_q(
            target_critics=target_critics,
            packed_next_state=packed_next,
            rewards=rewards,
            done=done,
            gamma=0.99,
        )
        assert target_q.shape == (4, 1)

    def test_target_q_terminal_state(self):
        """When done=1, target Q should equal reward (no bootstrap)."""
        target_critics = _make_critics()
        target_critics.eval()
        packed_next = _make_packed_sequence(batch_size=2, n_tokens=6)
        rewards = torch.tensor([[1.0], [2.0]])
        done = torch.ones(2, 1)

        target_q = compute_target_q(
            target_critics=target_critics,
            packed_next_state=packed_next,
            rewards=rewards,
            done=done,
            gamma=0.99,
        )
        assert torch.allclose(target_q, rewards)

    def test_target_q_uses_min_of_twins(self):
        """Target Q uses min(Q1, Q2) for overestimation reduction."""
        target_critics = _make_critics()
        target_critics.eval()
        packed_next = _make_packed_sequence(batch_size=3, n_tokens=6)
        rewards = torch.zeros(3, 1)
        done = torch.zeros(3, 1)

        target_q = compute_target_q(
            target_critics=target_critics,
            packed_next_state=packed_next,
            rewards=rewards,
            done=done,
            gamma=0.99,
        )
        with torch.no_grad():
            q1 = target_critics.critic_1(
                packed_next.global_tokens, packed_next.type_ids,
                packed_next.building_ids, packed_next.padding_mask,
                packed_next.controlled_building_indices,
            )
            q2 = target_critics.critic_2(
                packed_next.global_tokens, packed_next.type_ids,
                packed_next.building_ids, packed_next.padding_mask,
                packed_next.controlled_building_indices,
            )
        expected_target = rewards + 0.99 * (1.0 - done) * torch.min(q1, q2)
        assert torch.allclose(target_q, expected_target)

    def test_target_q_no_gradient(self):
        """Target Q computation should not require grad."""
        target_critics = _make_critics()
        packed_next = _make_packed_sequence(batch_size=2, n_tokens=6)
        rewards = torch.randn(2, 1)
        done = torch.zeros(2, 1)

        target_q = compute_target_q(
            target_critics=target_critics,
            packed_next_state=packed_next,
            rewards=rewards,
            done=done,
            gamma=0.99,
        )
        assert not target_q.requires_grad


class TestCriticUpdateStep:
    def test_returns_result_object(self):
        online_critics = _make_critics()
        optimizer = torch.optim.Adam(online_critics.parameters(), lr=1e-3)
        packed_current = _make_packed_sequence(batch_size=4, n_tokens=8)
        target_q = torch.randn(4, 1)

        result = critic_update_step(
            online_critics=online_critics,
            optimizer=optimizer,
            packed_current_state=packed_current,
            target_q=target_q,
        )
        assert isinstance(result, CriticUpdateResult)
        assert result.critic_1_loss >= 0
        assert result.critic_2_loss >= 0

    def test_loss_decreases_over_steps(self):
        """Critic loss should decrease over multiple update steps."""
        torch.manual_seed(42)
        online_critics = _make_critics()
        optimizer = torch.optim.Adam(online_critics.parameters(), lr=1e-2)
        packed_current = _make_packed_sequence(batch_size=8, n_tokens=6)
        target_q = torch.zeros(8, 1)

        losses = []
        for _ in range(20):
            result = critic_update_step(
                online_critics=online_critics,
                optimizer=optimizer,
                packed_current_state=packed_current,
                target_q=target_q,
            )
            losses.append(result.critic_1_loss + result.critic_2_loss)

        assert losses[-1] < losses[0]

    def test_both_critics_updated(self):
        """Both critic parameters should change after update."""
        online_critics = _make_critics()
        optimizer = torch.optim.Adam(online_critics.parameters(), lr=1e-3)
        packed_current = _make_packed_sequence(batch_size=4, n_tokens=6)
        target_q = torch.randn(4, 1)

        params_1_before = [p.clone() for p in online_critics.critic_1.parameters()]
        params_2_before = [p.clone() for p in online_critics.critic_2.parameters()]

        critic_update_step(
            online_critics=online_critics,
            optimizer=optimizer,
            packed_current_state=packed_current,
            target_q=target_q,
        )

        params_1_changed = any(
            not torch.allclose(p_before, p_after)
            for p_before, p_after in zip(params_1_before, online_critics.critic_1.parameters())
        )
        params_2_changed = any(
            not torch.allclose(p_before, p_after)
            for p_before, p_after in zip(params_2_before, online_critics.critic_2.parameters())
        )
        assert params_1_changed, "Critic 1 params unchanged after update"
        assert params_2_changed, "Critic 2 params unchanged after update"

    def test_mse_loss_used(self):
        """Verify that MSE loss is used (loss = 0 when Q matches target)."""
        torch.manual_seed(0)
        online_critics = _make_critics()
        optimizer = torch.optim.Adam(online_critics.parameters(), lr=1e-3)
        packed_current = _make_packed_sequence(batch_size=4, n_tokens=6)

        with torch.no_grad():
            q1, _q2 = online_critics(
                packed_current.global_tokens, packed_current.type_ids,
                packed_current.building_ids, packed_current.padding_mask,
                packed_current.controlled_building_indices,
            )
        result = critic_update_step(
            online_critics=online_critics,
            optimizer=optimizer,
            packed_current_state=packed_current,
            target_q=q1.detach(),
        )
        assert result.critic_1_loss < 1e-6
