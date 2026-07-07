"""Unit tests for twin independent Transformer critic stacks."""
from __future__ import annotations

import torch

from algorithms.utils.matd3_critic import (
    TransformerCriticStack,
    TwinTransformerCritics,
)


class TestTransformerCriticStack:
    """Single critic stack behavior."""

    def test_output_shape_single_building(self):
        """Q output per controlled building."""
        critic = TransformerCriticStack(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        global_tokens = torch.randn(2, 5, 16)
        type_ids = torch.zeros(2, 5, dtype=torch.long)
        building_ids = torch.zeros(2, 5, dtype=torch.long)
        padding_mask = torch.zeros(2, 5, dtype=torch.bool)

        q_values = critic(
            global_tokens, type_ids, building_ids,
            padding_mask, [0],
        )
        assert q_values.shape == (2, 1)

    def test_output_shape_multi_building(self):
        """Q output for multiple controlled buildings."""
        critic = TransformerCriticStack(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        global_tokens = torch.randn(4, 10, 16)
        type_ids = torch.zeros(4, 10, dtype=torch.long)
        building_ids = torch.cat([
            torch.zeros(4, 5, dtype=torch.long),
            torch.ones(4, 5, dtype=torch.long),
        ], dim=1)
        padding_mask = torch.zeros(4, 10, dtype=torch.bool)

        q_values = critic(
            global_tokens, type_ids, building_ids,
            padding_mask, [0, 1],
        )
        assert q_values.shape == (4, 2)

    def test_padding_mask_respected(self):
        """Padded tokens should not affect output of non-padded tokens."""
        torch.manual_seed(42)
        critic = TransformerCriticStack(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        critic.eval()

        global_tokens = torch.randn(1, 6, 16)
        type_ids = torch.zeros(1, 6, dtype=torch.long)
        building_ids = torch.zeros(1, 6, dtype=torch.long)

        mask_pad = torch.tensor([[False, False, False, False, True, True]])
        q_no_junk = critic(global_tokens, type_ids, building_ids, mask_pad, [0])
        tokens_with_junk = global_tokens.clone()
        tokens_with_junk[:, 4:, :] = torch.randn(1, 2, 16) * 100.0
        q_with_pad = critic(tokens_with_junk, type_ids, building_ids, mask_pad, [0])

        assert torch.allclose(q_no_junk, q_with_pad, atol=1e-5)

    def test_type_embeddings_affect_output(self):
        """Different type_ids should produce different outputs."""
        torch.manual_seed(0)
        critic = TransformerCriticStack(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        critic.eval()

        global_tokens = torch.randn(1, 4, 16)
        building_ids = torch.zeros(1, 4, dtype=torch.long)
        padding_mask = torch.zeros(1, 4, dtype=torch.bool)
        type_ids_a = torch.tensor([[0, 1, 2, 3]])
        type_ids_b = torch.tensor([[4, 5, 6, 7]])

        q_a = critic(global_tokens, type_ids_a, building_ids, padding_mask, [0])
        q_b = critic(global_tokens, type_ids_b, building_ids, padding_mask, [0])

        assert not torch.allclose(q_a, q_b)


class TestTwinTransformerCritics:
    """Twin critics independence and min-Q."""

    def test_critics_are_independent(self):
        """No shared parameters between critic 1 and critic 2."""
        twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        params_1 = set(id(p) for p in twins.critic_1.parameters())
        params_2 = set(id(p) for p in twins.critic_2.parameters())
        assert params_1.isdisjoint(params_2), "Twin critics share parameters!"

    def test_critics_produce_different_outputs(self):
        """After init, twin critics produce different Q values."""
        torch.manual_seed(99)
        twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        twins.eval()

        global_tokens = torch.randn(2, 6, 16)
        type_ids = torch.zeros(2, 6, dtype=torch.long)
        building_ids = torch.zeros(2, 6, dtype=torch.long)
        padding_mask = torch.zeros(2, 6, dtype=torch.bool)

        q1, q2 = twins(global_tokens, type_ids, building_ids, padding_mask, [0])
        assert q1.shape == q2.shape == (2, 1)
        assert not torch.allclose(q1, q2)

    def test_min_q_helper(self):
        """min_q returns element-wise minimum."""
        twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        twins.eval()

        global_tokens = torch.randn(3, 4, 16)
        type_ids = torch.zeros(3, 4, dtype=torch.long)
        building_ids = torch.zeros(3, 4, dtype=torch.long)
        padding_mask = torch.zeros(3, 4, dtype=torch.bool)

        min_q = twins.min_q(global_tokens, type_ids, building_ids, padding_mask, [0])
        q1, q2 = twins(global_tokens, type_ids, building_ids, padding_mask, [0])
        assert torch.allclose(min_q, torch.min(q1, q2))

    def test_soft_update(self):
        """Soft update with tau=1.0 makes target equal to online."""
        twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        target_twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        target_twins.soft_update_from(twins, tau=1.0)

        for p_online, p_target in zip(twins.parameters(), target_twins.parameters()):
            assert torch.allclose(p_online, p_target)

    def test_soft_update_partial(self):
        """Soft update with tau=0.0 leaves target unchanged."""
        torch.manual_seed(7)
        twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        target_twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        before = [p.clone() for p in target_twins.parameters()]
        target_twins.soft_update_from(twins, tau=0.0)
        for p_before, p_after in zip(before, target_twins.parameters()):
            assert torch.allclose(p_before, p_after)
