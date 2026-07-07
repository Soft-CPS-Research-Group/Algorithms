"""Tests for MATD3 exploration noise, gating, and phaseout."""
from __future__ import annotations

import pytest
import torch

from algorithms.utils.matd3_exploration import (
    add_exploration_noise,
    compute_sigma,
    is_initial_exploration_done,
    should_train_on_step,
    compute_phaseout_probability,
    apply_exploration_phaseout,
)


class TestInitialExplorationGating:
    def test_not_done_before_threshold(self):
        assert is_initial_exploration_done(
            global_learning_step=99,
            end_initial_exploration_time_step=100,
        ) is False

    def test_done_at_threshold(self):
        assert is_initial_exploration_done(
            global_learning_step=100,
            end_initial_exploration_time_step=100,
        ) is True

    def test_done_after_threshold(self):
        assert is_initial_exploration_done(
            global_learning_step=200,
            end_initial_exploration_time_step=100,
        ) is True


class TestShouldTrainOnStep:
    def test_trains_when_exploration_done(self):
        assert should_train_on_step(
            initial_exploration_done=True,
            train_during_initial_exploration=False,
            global_learning_step=0,
            initial_exploration_training_start_step=0,
        ) is True

    def test_skips_during_exploration_when_disabled(self):
        assert should_train_on_step(
            initial_exploration_done=False,
            train_during_initial_exploration=False,
            global_learning_step=50,
            initial_exploration_training_start_step=0,
        ) is False

    def test_trains_during_exploration_when_enabled_and_past_start(self):
        assert should_train_on_step(
            initial_exploration_done=False,
            train_during_initial_exploration=True,
            global_learning_step=50,
            initial_exploration_training_start_step=30,
        ) is True

    def test_skips_during_exploration_before_start_step(self):
        assert should_train_on_step(
            initial_exploration_done=False,
            train_during_initial_exploration=True,
            global_learning_step=20,
            initial_exploration_training_start_step=30,
        ) is False


class TestSigmaDecay:
    def test_initial_sigma(self):
        sigma = compute_sigma(
            exploration_step=0,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=1000,
        )
        assert sigma == pytest.approx(0.3)

    def test_final_sigma(self):
        sigma = compute_sigma(
            exploration_step=1000,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=1000,
        )
        assert sigma == pytest.approx(0.05)

    def test_midway_sigma(self):
        sigma = compute_sigma(
            exploration_step=500,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=1000,
        )
        assert sigma == pytest.approx(0.175)

    def test_past_decay_stays_at_final(self):
        sigma = compute_sigma(
            exploration_step=5000,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=1000,
        )
        assert sigma == pytest.approx(0.05)

    def test_zero_decay_steps_returns_initial(self):
        sigma = compute_sigma(
            exploration_step=100,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=0,
        )
        assert sigma == pytest.approx(0.3)


class TestExplorationNoise:
    def test_output_shape_preserved(self):
        torch.manual_seed(42)
        actions = torch.tensor([[0.5, -0.3, 0.0]])
        noisy = add_exploration_noise(
            actions=actions,
            action_span=torch.tensor([2.0, 2.0, 2.0]),
            action_low=torch.tensor([-1.0, -1.0, -1.0]),
            action_high=torch.tensor([1.0, 1.0, 1.0]),
            sigma=0.2,
            noise_clip=0.5,
        )
        assert noisy.shape == actions.shape

    def test_output_within_bounds(self):
        torch.manual_seed(0)
        actions = torch.rand(100, 4) * 2 - 1
        noisy = add_exploration_noise(
            actions=actions,
            action_span=torch.full((4,), 2.0),
            action_low=torch.full((4,), -1.0),
            action_high=torch.full((4,), 1.0),
            sigma=0.5,
            noise_clip=1.0,
        )
        assert noisy.min() >= -1.0
        assert noisy.max() <= 1.0

    def test_zero_sigma_no_change(self):
        actions = torch.tensor([[0.5, -0.3]])
        noisy = add_exploration_noise(
            actions=actions,
            action_span=torch.tensor([2.0, 2.0]),
            action_low=torch.tensor([-1.0, -1.0]),
            action_high=torch.tensor([1.0, 1.0]),
            sigma=0.0,
            noise_clip=0.5,
        )
        assert torch.allclose(noisy, actions)

    def test_noise_clip_bounds_noise(self):
        """Noise magnitude bounded by noise_clip * span."""
        torch.manual_seed(1)
        actions = torch.zeros(1000, 2)
        span = torch.tensor([2.0, 2.0])
        noise_clip = 0.25
        noisy = add_exploration_noise(
            actions=actions,
            action_span=span,
            action_low=torch.full((2,), -10.0),
            action_high=torch.full((2,), 10.0),
            sigma=5.0,
            noise_clip=noise_clip,
        )
        assert (noisy - actions).abs().max() <= noise_clip * span.max() + 1e-6


class TestPhaseoutProbability:
    def test_full_probability_at_start(self):
        p = compute_phaseout_probability(exploration_step=0, phaseout_steps=100)
        assert p == pytest.approx(1.0)

    def test_zero_probability_at_end(self):
        p = compute_phaseout_probability(exploration_step=100, phaseout_steps=100)
        assert p == pytest.approx(0.0)

    def test_linear_decay_midpoint(self):
        p = compute_phaseout_probability(exploration_step=50, phaseout_steps=100)
        assert p == pytest.approx(0.5)

    def test_zero_phaseout_steps_returns_zero(self):
        p = compute_phaseout_probability(exploration_step=0, phaseout_steps=0)
        assert p == pytest.approx(0.0)


class TestExplorationPhaseout:
    def test_deterministic_skips_phaseout(self):
        actor_actions = torch.tensor([[0.5, -0.3]])
        teacher_actions = torch.tensor([[0.0, 0.0]])
        result = apply_exploration_phaseout(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            phaseout_probability=0.9,
            mode="blend",
            deterministic=True,
        )
        assert torch.allclose(result, actor_actions)

    def test_blend_mode_interpolates(self):
        actor_actions = torch.tensor([[1.0, 0.0]])
        teacher_actions = torch.tensor([[0.0, 1.0]])
        result = apply_exploration_phaseout(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            phaseout_probability=0.5,
            mode="blend",
            deterministic=False,
        )
        assert torch.allclose(result, torch.tensor([[0.5, 0.5]]))

    def test_blend_full_teacher(self):
        actor_actions = torch.tensor([[1.0]])
        teacher_actions = torch.tensor([[-1.0]])
        result = apply_exploration_phaseout(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            phaseout_probability=1.0,
            mode="blend",
            deterministic=False,
        )
        assert torch.allclose(result, teacher_actions)

    def test_blend_zero_probability_returns_actor(self):
        actor_actions = torch.tensor([[0.7, -0.2]])
        teacher_actions = torch.tensor([[0.0, 0.0]])
        result = apply_exploration_phaseout(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            phaseout_probability=0.0,
            mode="blend",
            deterministic=False,
        )
        assert torch.allclose(result, actor_actions)
