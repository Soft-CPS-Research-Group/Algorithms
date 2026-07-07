"""Tests for MATD3 residual policy composition."""
from __future__ import annotations

import pytest
import torch

from algorithms.utils.matd3_residual import (
    compose_residual_actions,
    scale_direct_actions,
    build_ca_type_scale_mask,
)


class TestResidualComposition:
    def test_basic_formula(self):
        """action = clip(teacher + 0.5 * span * scale * mask * actor, low, high)"""
        teacher = torch.tensor([[0.5, -0.3]])
        actor = torch.tensor([[0.4, -0.6]])
        span = torch.tensor([2.0, 2.0])
        low = torch.tensor([-1.0, -1.0])
        high = torch.tensor([1.0, 1.0])

        result = compose_residual_actions(
            teacher_actions=teacher,
            actor_outputs=actor,
            action_span=span,
            action_low=low,
            action_high=high,
            residual_action_scale=0.5,
            scale_mask=torch.tensor([1.0, 1.0]),
        )
        assert torch.allclose(result, torch.tensor([[0.7, -0.6]]), atol=1e-6)

    def test_clipping_at_bounds(self):
        """Result must be clipped to [low, high]."""
        result = compose_residual_actions(
            teacher_actions=torch.tensor([[0.9]]),
            actor_outputs=torch.tensor([[1.0]]),
            action_span=torch.tensor([2.0]),
            action_low=torch.tensor([-1.0]),
            action_high=torch.tensor([1.0]),
            residual_action_scale=1.0,
            scale_mask=torch.tensor([1.0]),
        )
        assert result.item() == pytest.approx(1.0)

    def test_per_ca_type_mask(self):
        """Different mask values for different CA types."""
        result = compose_residual_actions(
            teacher_actions=torch.tensor([[0.0, 0.0, 0.0]]),
            actor_outputs=torch.tensor([[1.0, 1.0, 1.0]]),
            action_span=torch.tensor([2.0, 2.0, 2.0]),
            action_low=torch.tensor([-1.0, -1.0, -1.0]),
            action_high=torch.tensor([1.0, 1.0, 1.0]),
            residual_action_scale=1.0,
            scale_mask=torch.tensor([0.5, 1.0, 0.25]),
        )
        assert torch.allclose(result, torch.tensor([[0.5, 1.0, 0.25]]), atol=1e-6)

    def test_batch_dimension(self):
        """Works with batch > 1."""
        result = compose_residual_actions(
            teacher_actions=torch.zeros(4, 3),
            actor_outputs=torch.ones(4, 3) * 0.5,
            action_span=torch.full((3,), 2.0),
            action_low=torch.full((3,), -1.0),
            action_high=torch.full((3,), 1.0),
            residual_action_scale=0.4,
            scale_mask=torch.ones(3),
        )
        assert result.shape == (4, 3)
        assert torch.allclose(result, torch.full((4, 3), 0.2), atol=1e-6)

    def test_zero_scale_returns_teacher(self):
        """With scale=0, output equals teacher (clipped)."""
        teacher = torch.tensor([[0.3, -0.7]])
        result = compose_residual_actions(
            teacher_actions=teacher,
            actor_outputs=torch.tensor([[0.9, -0.9]]),
            action_span=torch.tensor([2.0, 2.0]),
            action_low=torch.tensor([-1.0, -1.0]),
            action_high=torch.tensor([1.0, 1.0]),
            residual_action_scale=0.0,
            scale_mask=torch.ones(2),
        )
        assert torch.allclose(result, teacher, atol=1e-6)


class TestDirectScaling:
    def test_direct_scaling_formula(self):
        """action = low + 0.5 * (actor + 1) * span"""
        result = scale_direct_actions(
            actor_outputs=torch.tensor([[0.0, 1.0, -1.0]]),
            action_span=torch.tensor([2.0, 2.0, 2.0]),
            action_low=torch.tensor([-1.0, -1.0, -1.0]),
            action_high=torch.tensor([1.0, 1.0, 1.0]),
        )
        assert torch.allclose(result, torch.tensor([[0.0, 1.0, -1.0]]), atol=1e-6)

    def test_asymmetric_bounds(self):
        """Correctly handles non-symmetric bounds."""
        result = scale_direct_actions(
            actor_outputs=torch.tensor([[0.0]]),
            action_span=torch.tensor([4.0]),
            action_low=torch.tensor([-1.0]),
            action_high=torch.tensor([3.0]),
        )
        assert result.item() == pytest.approx(1.0)


class TestBuildCATypeScaleMask:
    def test_default_multipliers(self):
        """All types get 1.0 with default multipliers."""
        mask = build_ca_type_scale_mask(
            ca_type_names=["storage", "charger", "pv"],
            storage_multiplier=1.0,
            ev_multiplier=1.0,
        )
        assert mask.shape == (3,)
        assert torch.allclose(mask, torch.ones(3))

    def test_custom_multipliers(self):
        """Per-type multipliers applied correctly."""
        mask = build_ca_type_scale_mask(
            ca_type_names=["storage", "charger", "charger", "pv"],
            storage_multiplier=0.5,
            ev_multiplier=2.0,
        )
        assert torch.allclose(mask, torch.tensor([0.5, 2.0, 2.0, 1.0]))

    def test_empty_list(self):
        """Empty CA list returns empty tensor."""
        mask = build_ca_type_scale_mask(
            ca_type_names=[],
            storage_multiplier=0.5,
            ev_multiplier=2.0,
        )
        assert mask.shape == (0,)
