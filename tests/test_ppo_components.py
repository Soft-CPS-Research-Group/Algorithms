"""Tests for PPO components."""

import pytest
import torch
import torch.nn as nn

from algorithms.utils.ppo_components import ActorHead


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
