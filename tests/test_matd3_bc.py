"""Tests for MATD3 replay-native behavior cloning loss."""
from __future__ import annotations

import pytest
import torch

from algorithms.utils.matd3_bc import (
    compute_bc_loss,
    compute_bc_effective_weight,
    compute_ca_type_weights,
)


class TestBCEffectiveWeight:
    def test_before_decay_start(self):
        w = compute_bc_effective_weight(
            global_learning_step=50,
            initial_weight=1.0,
            min_weight=0.0,
            decay_start_step=100,
            decay_steps=200,
        )
        assert w == pytest.approx(1.0)

    def test_at_decay_start(self):
        w = compute_bc_effective_weight(
            global_learning_step=100,
            initial_weight=1.0,
            min_weight=0.0,
            decay_start_step=100,
            decay_steps=200,
        )
        assert w == pytest.approx(1.0)

    def test_midway_through_decay(self):
        w = compute_bc_effective_weight(
            global_learning_step=200,
            initial_weight=1.0,
            min_weight=0.0,
            decay_start_step=100,
            decay_steps=200,
        )
        assert w == pytest.approx(0.5)

    def test_after_full_decay(self):
        w = compute_bc_effective_weight(
            global_learning_step=300,
            initial_weight=1.0,
            min_weight=0.0,
            decay_start_step=100,
            decay_steps=200,
        )
        assert w == pytest.approx(0.0)

    def test_respects_min_weight(self):
        w = compute_bc_effective_weight(
            global_learning_step=9999,
            initial_weight=1.0,
            min_weight=0.2,
            decay_start_step=0,
            decay_steps=100,
        )
        assert w == pytest.approx(0.2)

    def test_zero_initial_weight(self):
        w = compute_bc_effective_weight(
            global_learning_step=50,
            initial_weight=0.0,
            min_weight=0.0,
            decay_start_step=0,
            decay_steps=100,
        )
        assert w == pytest.approx(0.0)

    def test_zero_decay_steps(self):
        """No decay steps means weight stays at initial."""
        w = compute_bc_effective_weight(
            global_learning_step=9999,
            initial_weight=0.5,
            min_weight=0.0,
            decay_start_step=0,
            decay_steps=0,
        )
        assert w == pytest.approx(0.5)


class TestCATypeWeights:
    def test_default_all_ones(self):
        weights = compute_ca_type_weights(
            ca_type_names=["storage", "charger", "pv"],
            ev_multiplier=1.0,
            storage_multiplier=1.0,
        )
        assert torch.allclose(weights, torch.ones(3))

    def test_custom_multipliers(self):
        weights = compute_ca_type_weights(
            ca_type_names=["storage", "charger", "charger", "pv", "storage"],
            ev_multiplier=3.0,
            storage_multiplier=0.5,
        )
        assert torch.allclose(weights, torch.tensor([0.5, 3.0, 3.0, 1.0, 0.5]))

    def test_empty_returns_empty(self):
        weights = compute_ca_type_weights(
            ca_type_names=[],
            ev_multiplier=2.0,
            storage_multiplier=0.5,
        )
        assert weights.shape == (0,)


class TestBCLoss:
    def test_basic_mse_loss(self):
        """BC loss = weighted MSE between actor and teacher."""
        loss = compute_bc_loss(
            actor_actions=torch.tensor([[0.5, 0.3], [0.2, -0.1]]),
            teacher_actions=torch.tensor([[0.4, 0.3], [0.0, -0.1]]),
            ca_type_weights=torch.tensor([1.0, 1.0]),
            effective_weight=1.0,
        )
        assert loss.item() == pytest.approx(0.0125, abs=1e-6)

    def test_ca_type_weighting(self):
        """Per-type weights affect loss magnitude."""
        loss = compute_bc_loss(
            actor_actions=torch.tensor([[1.0, 1.0]]),
            teacher_actions=torch.tensor([[0.0, 0.0]]),
            ca_type_weights=torch.tensor([0.5, 2.0]),
            effective_weight=1.0,
        )
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_effective_weight_multiplied(self):
        """Loss is multiplied by effective_weight."""
        actor_actions = torch.tensor([[0.5]])
        teacher_actions = torch.tensor([[0.0]])
        weights = torch.tensor([1.0])
        loss_full = compute_bc_loss(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            ca_type_weights=weights,
            effective_weight=1.0,
        )
        loss_half = compute_bc_loss(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            ca_type_weights=weights,
            effective_weight=0.5,
        )
        assert loss_half.item() == pytest.approx(loss_full.item() * 0.5, abs=1e-7)

    def test_zero_weight_returns_zero(self):
        """Zero effective weight means zero loss."""
        loss = compute_bc_loss(
            actor_actions=torch.tensor([[0.9, -0.8]]),
            teacher_actions=torch.tensor([[0.0, 0.0]]),
            ca_type_weights=torch.tensor([1.0, 1.0]),
            effective_weight=0.0,
        )
        assert loss.item() == 0.0

    def test_gradient_flows_to_actor(self):
        """BC loss supports gradient computation for actor parameters."""
        actor_actions = torch.tensor([[0.5, 0.3]], requires_grad=True)
        loss = compute_bc_loss(
            actor_actions=actor_actions,
            teacher_actions=torch.tensor([[0.0, 0.0]]),
            ca_type_weights=torch.tensor([1.0, 1.0]),
            effective_weight=1.0,
        )
        loss.backward()
        assert actor_actions.grad is not None
        assert actor_actions.grad.shape == (1, 2)

    def test_batch_independence(self):
        """Each sample in batch contributes independently."""
        weights = torch.tensor([1.0, 1.0])
        loss_single = compute_bc_loss(
            actor_actions=torch.tensor([[1.0, 0.0]]),
            teacher_actions=torch.tensor([[0.0, 0.0]]),
            ca_type_weights=weights,
            effective_weight=1.0,
        )
        loss_double = compute_bc_loss(
            actor_actions=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            teacher_actions=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            ca_type_weights=weights,
            effective_weight=1.0,
        )
        assert loss_single.item() == pytest.approx(loss_double.item(), abs=1e-6)
