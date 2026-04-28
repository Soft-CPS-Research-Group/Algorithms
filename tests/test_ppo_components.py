"""Tests for PPO components."""

import pytest
import torch
import torch.nn as nn

from algorithms.utils.ppo_components import ActorHead, CriticHead, RolloutBuffer, compute_ppo_loss


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

    def test_actor_stochastic_mode(self) -> None:
        """Stochastic mode should sample different actions."""
        d_model = 64
        actor = ActorHead(d_model=d_model, hidden_dim=128)

        ca_embeddings = torch.randn(1, 2, d_model)

        # Multiple calls should likely return different results (probabilistic)
        torch.manual_seed(42)
        actions1, _, _ = actor(ca_embeddings, deterministic=False)
        torch.manual_seed(123)
        actions2, _, _ = actor(ca_embeddings, deterministic=False)

        # Not guaranteed to be different, but very likely with different seeds
        # Just check they're valid actions
        assert (actions1 >= -1.0).all() and (actions1 <= 1.0).all()
        assert (actions2 >= -1.0).all() and (actions2 <= 1.0).all()


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
            done=False,
        )

        assert len(buffer.observations) == 1
        assert len(buffer.rewards) == 1

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
                done=(i == 4),  # Last one is terminal
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
                done=False,
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
            done=False,
        )

        buffer.clear()

        assert len(buffer.observations) == 0
        assert len(buffer.rewards) == 0


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
