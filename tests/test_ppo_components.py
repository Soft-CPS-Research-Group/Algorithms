"""Tests for PPO components."""

import math

import pytest
import torch
import torch.nn as nn

from algorithms.utils.ppo_components import (
    ActorHead,
    CriticHead,
    RolloutBuffer,
    RunningValueNormalizer,
    compute_ppo_loss,
)


class TestActorHead:
    """Tests for ActorHead class."""

    def test_actor_creation(self) -> None:
        """ActorHead should create with correct architecture."""
        actor = ActorHead(d_model=64, hidden_dim=128)

        assert actor is not None
        assert isinstance(actor, nn.Module)

    def test_actor_output_shape(self) -> None:
        """Actor should output actions and log_probs with correct shapes."""
        d_model = 64
        actor = ActorHead(d_model=d_model, hidden_dim=128)

        batch_size = 2
        n_ca = 3
        ca_embeddings = torch.randn(batch_size, n_ca, d_model)

        actions, log_probs, means = actor(ca_embeddings, deterministic=False)

        assert actions.shape == (batch_size, n_ca, 1)
        assert log_probs.shape == (batch_size, n_ca)
        assert means.shape == (batch_size, n_ca, 1)

    def test_actor_output_range(self) -> None:
        """Actions should be in [-1, 1] range after tanh."""
        d_model = 64
        actor = ActorHead(d_model=d_model, hidden_dim=128)

        ca_embeddings = torch.randn(2, 3, d_model)
        actions, _, _ = actor(ca_embeddings, deterministic=False)

        assert (actions >= -1.0).all()
        assert (actions <= 1.0).all()

    def test_actor_deterministic_mode(self) -> None:
        """Deterministic mode should return mean actions."""
        d_model = 64
        actor = ActorHead(d_model=d_model, hidden_dim=128)

        ca_embeddings = torch.randn(1, 2, d_model)

        # Multiple calls in deterministic mode should return same result
        actions1, _, means1 = actor(ca_embeddings, deterministic=True)
        actions2, _, means2 = actor(ca_embeddings, deterministic=True)

        assert torch.allclose(actions1, actions2)
        assert torch.allclose(actions1, torch.tanh(means1))

    def test_actor_stochastic_mode_yields_valid_actions(self) -> None:
        """Stochastic sampling stays within the tanh-squashed range."""
        d_model = 64
        actor = ActorHead(d_model=d_model, hidden_dim=128)

        ca_embeddings = torch.randn(1, 2, d_model)

        torch.manual_seed(42)
        actions1, _, _ = actor(ca_embeddings, deterministic=False)
        torch.manual_seed(123)
        actions2, _, _ = actor(ca_embeddings, deterministic=False)

        assert (actions1 >= -1.0).all() and (actions1 <= 1.0).all()
        assert (actions2 >= -1.0).all() and (actions2 <= 1.0).all()
        # Different seeds should give different samples with overwhelming probability.
        assert not torch.allclose(actions1, actions2)


class TestCriticHead:
    """Tests for CriticHead class."""

    def test_critic_creation(self) -> None:
        """CriticHead should create with correct architecture."""
        critic = CriticHead(d_model=64, hidden_dim=128)

        assert critic is not None
        assert isinstance(critic, nn.Module)

    def test_critic_output_shape(self) -> None:
        """Critic should output scalar value per batch."""
        d_model = 64
        critic = CriticHead(d_model=d_model, hidden_dim=128)

        batch_size = 2
        pooled = torch.randn(batch_size, d_model)

        values = critic(pooled)

        assert values.shape == (batch_size, 1)

    def test_critic_gradient_flow(self) -> None:
        """Gradients should flow through critic."""
        d_model = 64
        critic = CriticHead(d_model=d_model, hidden_dim=128)

        pooled = torch.randn(2, d_model, requires_grad=True)
        values = critic(pooled)
        loss = values.sum()
        loss.backward()

        assert pooled.grad is not None


class TestRolloutBuffer:
    """Tests for RolloutBuffer class."""

    def test_buffer_creation(self) -> None:
        """RolloutBuffer should create with specified hyperparameters."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)

        assert buffer.gamma == 0.99
        assert buffer.gae_lambda == 0.95

    def test_buffer_add_transition(self) -> None:
        """Buffer should store transitions."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)

        buffer.add(
            observation=torch.randn(10),
            action=torch.randn(2),
            log_prob=torch.tensor(-0.5),
            reward=1.0,
            value=torch.tensor(0.5),
            terminated=False,
            truncated=False,
        )

        assert len(buffer.observations) == 1
        assert len(buffer.rewards) == 1
        assert buffer.terminated == [False]
        assert buffer.truncated == [False]
        assert not hasattr(buffer, "dones")

    def test_buffer_compute_gae(self) -> None:
        """Buffer should compute GAE advantages."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)

        # Add a few transitions
        for i in range(5):
            buffer.add(
                observation=torch.randn(10),
                action=torch.randn(2),
                log_prob=torch.tensor(-0.5),
                reward=1.0,
                value=torch.tensor(0.5),
                terminated=(i == 4),
                truncated=False,
            )

        buffer.compute_returns_and_advantages(last_value=torch.tensor(0.0))

        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert len(buffer.advantages) == 5

    def test_buffer_get_batches(self) -> None:
        """Buffer should yield minibatches."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)

        for i in range(10):
            buffer.add(
                observation=torch.randn(10),
                action=torch.randn(2),
                log_prob=torch.tensor(-0.5),
                reward=1.0,
                value=torch.tensor(0.5),
                terminated=False,
                truncated=False,
            )

        buffer.compute_returns_and_advantages(last_value=torch.tensor(0.0))

        batches = list(buffer.get_batches(batch_size=4))
        assert len(batches) >= 2  # At least 2 batches of size 4 from 10 samples

    def test_buffer_clear(self) -> None:
        """Buffer should clear all data."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)

        buffer.add(
            observation=torch.randn(10),
            action=torch.randn(2),
            log_prob=torch.tensor(-0.5),
            reward=1.0,
            value=torch.tensor(0.5),
            terminated=False,
            truncated=False,
        )

        buffer.clear()

        assert len(buffer.observations) == 0
        assert len(buffer.rewards) == 0
        assert buffer.terminated == []
        assert buffer.truncated == []

    @pytest.mark.parametrize(
        ("terminated", "truncated", "expected_return"),
        [(True, False, 1.0), (False, True, 2.8)],
        ids=["terminated", "truncated"],
    )
    def test_buffer_uses_termination_but_not_truncation_for_bootstrap(
        self,
        terminated: bool,
        truncated: bool,
        expected_return: float,
    ) -> None:
        """Termination prevents bootstrap, while truncation retains it."""
        buffer = RolloutBuffer(gamma=0.9, gae_lambda=1.0)
        buffer.add(
            observation=torch.tensor([0.0]),
            action=torch.tensor([0.0]),
            log_prob=torch.tensor(0.0),
            reward=1.0,
            value=torch.tensor(0.0),
            terminated=terminated,
            truncated=truncated,
        )

        buffer.compute_returns_and_advantages(last_value=torch.tensor(2.0))

        assert buffer.returns is not None
        assert buffer.returns.item() == pytest.approx(expected_return)

    def test_buffer_rejects_interior_truncation_without_boundary_flush(self) -> None:
        """A single final bootstrap value cannot serve an interior truncation."""
        buffer = RolloutBuffer(gamma=0.9, gae_lambda=1.0)
        for truncated in (True, False):
            buffer.add(
                observation=torch.tensor([0.0]),
                action=torch.tensor([0.0]),
                log_prob=torch.tensor(0.0),
                reward=1.0,
                value=torch.tensor(0.0),
                terminated=False,
                truncated=truncated,
            )

        with pytest.raises(ValueError, match="flush.*truncation boundary"):
            buffer.compute_returns_and_advantages(last_value=torch.tensor(2.0))


class TestPPOLoss:
    """Tests for PPO loss computation."""

    def test_ppo_loss_shape(self) -> None:
        """PPO loss should return scalar tensor and metrics dict."""
        batch_size = 4

        log_probs_new = torch.randn(batch_size)
        log_probs_old = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        values = torch.randn(batch_size)
        returns = torch.randn(batch_size)

        loss, metrics = compute_ppo_loss(
            log_probs_new=log_probs_new,
            log_probs_old=log_probs_old,
            advantages=advantages,
            values=values,
            returns=returns,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
        )

        assert loss.ndim == 0  # Scalar
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "approx_kl" in metrics
        assert "ratio_error_max" in metrics
        assert "explained_variance" in metrics

    def test_ppo_loss_clipping(self) -> None:
        """PPO loss should clip probability ratios."""
        batch_size = 4

        # Create scenario where ratio would be clipped
        log_probs_new = torch.zeros(batch_size)
        log_probs_old = torch.ones(batch_size) * -1.0  # ratio = exp(1) ≈ 2.7
        advantages = torch.ones(batch_size)
        values = torch.zeros(batch_size)
        returns = torch.ones(batch_size)

        loss, metrics = compute_ppo_loss(
            log_probs_new=log_probs_new,
            log_probs_old=log_probs_old,
            advantages=advantages,
            values=values,
            returns=returns,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
        )

        # Loss should be finite (clipping prevents explosion)
        assert torch.isfinite(loss)

    def test_ppo_loss_gradient_flow(self) -> None:
        """Gradients should flow through PPO loss."""
        log_probs_new = torch.randn(4, requires_grad=True)
        log_probs_old = torch.randn(4)
        advantages = torch.randn(4)
        values = torch.randn(4, requires_grad=True)
        returns = torch.randn(4)

        loss, _ = compute_ppo_loss(
            log_probs_new=log_probs_new,
            log_probs_old=log_probs_old,
            advantages=advantages,
            values=values,
            returns=returns,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
        )

        loss.backward()

        assert log_probs_new.grad is not None
        assert values.grad is not None

    def test_ppo_loss_smooth_l1_value_gradient_for_large_residual(self) -> None:
        """A large critic residual must retain a finite, nonzero gradient."""
        values = torch.tensor([20.0], requires_grad=True)
        loss, _ = compute_ppo_loss(
            log_probs_new=torch.zeros(1, requires_grad=True),
            log_probs_old=torch.zeros(1),
            advantages=torch.zeros(1),
            values=values,
            returns=torch.zeros(1),
            clip_eps=0.2,
            value_coeff=1.0,
            entropy_coeff=0.0,
        )

        loss.backward()

        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert values.grad.abs().max().item() > 0.0

    def test_ppo_loss_reports_diagnostics(self) -> None:
        """PPO diagnostics should report ratio and critic-fit information."""
        loss, metrics = compute_ppo_loss(
            log_probs_new=torch.tensor([0.0, math.log(2.0)]),
            log_probs_old=torch.zeros(2),
            advantages=torch.ones(2),
            values=torch.tensor([1.0, 2.0]),
            returns=torch.tensor([1.0, 3.0]),
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.0,
        )

        assert torch.isfinite(loss)
        assert metrics["approx_kl"] == pytest.approx((1.0 - math.log(2.0)) / 2.0)
        assert metrics["ratio_error_max"] == pytest.approx(1.0)
        assert metrics["explained_variance"] == pytest.approx(0.75)

    def test_ppo_loss_approx_kl_uses_safe_ppo_ratio_expression(self) -> None:
        """The diagnostic must use the same safe ratio and log-ratio as PPO."""
        log_probs_new = torch.linspace(-10.0, 10.0, steps=10_001, dtype=torch.float32)
        log_probs_old = torch.zeros_like(log_probs_new)

        _, metrics = compute_ppo_loss(
            log_probs_new=log_probs_new,
            log_probs_old=log_probs_old,
            advantages=torch.ones_like(log_probs_new),
            values=torch.zeros_like(log_probs_new),
            returns=torch.zeros_like(log_probs_new),
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.0,
        )

        log_ratio = log_probs_new - log_probs_old
        ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
        expected = ((ratio - 1.0) - log_ratio).mean().item()

        assert metrics["approx_kl"] == expected

    @pytest.mark.parametrize(
        ("returns", "values"),
        [
            (torch.tensor([2.0]), torch.tensor([1.0])),
            (torch.tensor([2.0, 2.0]), torch.tensor([1.0, 3.0])),
        ],
        ids=["one-return", "constant-returns"],
    )
    def test_ppo_loss_diagnostics_are_finite_for_degenerate_returns(
        self,
        returns: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Diagnostics must remain finite for degenerate return samples."""
        _, metrics = compute_ppo_loss(
            log_probs_new=torch.zeros_like(returns),
            log_probs_old=torch.zeros_like(returns),
            advantages=torch.ones_like(returns),
            values=values,
            returns=returns,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.0,
        )

        for metric_name in ("approx_kl", "ratio_error_max", "explained_variance"):
            assert math.isfinite(metrics[metric_name])

    def test_ppo_loss_explained_variance_is_finite_for_large_float16_residuals(self) -> None:
        """Finite float16 inputs must not overflow explained-variance diagnostics."""
        returns = torch.tensor([1.0, 1.0], dtype=torch.float16)
        values = torch.tensor([300.0, -300.0], dtype=torch.float16)

        _, metrics = compute_ppo_loss(
            log_probs_new=torch.zeros_like(returns),
            log_probs_old=torch.zeros_like(returns),
            advantages=torch.ones_like(returns),
            values=values,
            returns=returns,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.0,
        )

        assert math.isfinite(metrics["explained_variance"])

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_ppo_loss_is_finite_for_large_log_ratio(self, dtype: torch.dtype) -> None:
        """Finite log probabilities must not overflow PPO loss or diagnostics."""
        log_probs_new = torch.tensor([100.0], dtype=dtype, requires_grad=True)
        values = torch.tensor([0.0], dtype=dtype, requires_grad=True)

        loss, metrics = compute_ppo_loss(
            log_probs_new=log_probs_new,
            log_probs_old=torch.tensor([0.0], dtype=dtype),
            advantages=torch.tensor([1.0], dtype=dtype),
            values=values,
            returns=torch.tensor([0.0], dtype=dtype),
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert math.isfinite(metrics["approx_kl"])
        assert math.isfinite(metrics["ratio_error_max"])
        assert log_probs_new.grad is not None
        assert torch.isfinite(log_probs_new.grad).all()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()

    def test_ppo_loss_is_finite_for_near_limit_float32_inputs(self) -> None:
        """Finite near-limit float32 tensors must retain finite losses and gradients."""
        limit = torch.finfo(torch.float32).max / 2.0
        log_probs_new = torch.tensor([0.0, 20.0], dtype=torch.float32, requires_grad=True)
        values = torch.tensor([limit, -limit], dtype=torch.float32, requires_grad=True)

        loss, metrics = compute_ppo_loss(
            log_probs_new=log_probs_new,
            log_probs_old=torch.zeros_like(log_probs_new),
            advantages=torch.tensor([limit, -limit], dtype=torch.float32),
            values=values,
            returns=-values.detach(),
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert all(math.isfinite(metric) for metric in metrics.values())
        assert log_probs_new.grad is not None
        assert torch.isfinite(log_probs_new.grad).all()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


class TestRunningValueNormalizer:
    """Tests for RunningValueNormalizer."""

    def test_normalizer_round_trip_preserves_dtype_and_device(self) -> None:
        """Normalization and denormalization should restore input values."""
        normalizer = RunningValueNormalizer()
        values = torch.tensor([1.0, 3.0], dtype=torch.float64)
        normalizer.update(values)

        normalized = normalizer.normalize(values)
        restored = normalizer.denormalize(normalized)

        assert normalized.dtype == values.dtype
        assert normalized.device == values.device
        assert restored.dtype == values.dtype
        assert restored.device == values.device
        assert torch.allclose(restored, values)
        assert normalizer.mean == pytest.approx(2.0)
        assert normalizer.variance == pytest.approx(1.0)
        assert normalizer.count == 2

    def test_normalizer_state_round_trip_is_device_agnostic(self) -> None:
        """A saved scalar state should restore equivalent normalization."""
        normalizer = RunningValueNormalizer()
        normalizer.update(torch.tensor([1.0, 2.0, 5.0]))
        state = normalizer.state_dict()

        restored = RunningValueNormalizer()
        restored.load_state_dict(state)
        values = torch.tensor([2.0, 8.0], dtype=torch.float64)

        assert all(not isinstance(value, torch.Tensor) for value in state.values())
        assert restored.state_dict() == state
        assert torch.allclose(restored.normalize(values), normalizer.normalize(values))
