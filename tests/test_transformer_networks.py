"""Tests for Transformer-based neural network architectures.

Covers:
- TransformerConfig validation
- TransformerActor output shape and variable cardinality
- TransformerCritic forward pass and aggregation
- Gradient flow through all tokens
- Permutation equivariance for CA tokens
"""

from __future__ import annotations

import pytest
import torch

from algorithms.utils.transformer_networks import (
    CLSTokenStrategy,
    CriticAggregationStrategy,
    MeanPoolStrategy,
    TransformerActor,
    TransformerConfig,
    TransformerCritic,
)


# --- Fixtures ---


@pytest.fixture
def default_config() -> TransformerConfig:
    """Default configuration for tests."""
    return TransformerConfig(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        max_tokens=128,
        action_dim=3,
    )


@pytest.fixture
def actor(default_config: TransformerConfig) -> TransformerActor:
    """Create a TransformerActor for testing."""
    return TransformerActor(default_config)


@pytest.fixture
def critic(default_config: TransformerConfig) -> TransformerCritic:
    """Create a TransformerCritic for testing."""
    return TransformerCritic(default_config)


def make_random_inputs(
    cfg: TransformerConfig,
    batch_size: int,
    n_ca: int,
    n_sro: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random token inputs for testing."""
    ca = torch.randn(batch_size, n_ca, cfg.d_model, device=device)
    sro = torch.randn(batch_size, n_sro, cfg.d_model, device=device)
    nfc = torch.randn(batch_size, 1, cfg.d_model, device=device)
    return ca, sro, nfc


# --- TransformerConfig Tests ---


class TestTransformerConfig:
    """Tests for TransformerConfig validation."""

    def test_valid_config(self) -> None:
        """Valid configuration should not raise."""
        cfg = TransformerConfig(d_model=64, nhead=4)
        assert cfg.d_model == 64
        assert cfg.nhead == 4

    def test_d_model_not_divisible_by_nhead(self) -> None:
        """d_model must be divisible by nhead."""
        with pytest.raises(ValueError, match="divisible by nhead"):
            TransformerConfig(d_model=65, nhead=4)

    def test_d_model_too_small(self) -> None:
        """d_model must be at least 8."""
        with pytest.raises(ValueError, match="d_model must be >= 8"):
            TransformerConfig(d_model=4, nhead=2)

    def test_nhead_too_small(self) -> None:
        """nhead must be at least 1."""
        with pytest.raises(ValueError, match="nhead must be >= 1"):
            TransformerConfig(d_model=64, nhead=0)

    def test_num_layers_too_small(self) -> None:
        """num_layers must be at least 1."""
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            TransformerConfig(num_layers=0)

    def test_dropout_out_of_range(self) -> None:
        """dropout must be in [0, 1]."""
        with pytest.raises(ValueError, match="dropout must be in"):
            TransformerConfig(dropout=1.5)
        with pytest.raises(ValueError, match="dropout must be in"):
            TransformerConfig(dropout=-0.1)

    def test_max_tokens_too_small(self) -> None:
        """max_tokens must be at least 4."""
        with pytest.raises(ValueError, match="max_tokens must be >= 4"):
            TransformerConfig(max_tokens=2)

    def test_action_dim_too_small(self) -> None:
        """action_dim must be at least 1."""
        with pytest.raises(ValueError, match="action_dim must be >= 1"):
            TransformerConfig(action_dim=0)


# --- TransformerActor Tests ---


class TestTransformerActor:
    """Tests for TransformerActor."""

    def test_output_shape_matches_ca_count(
        self, actor: TransformerActor, default_config: TransformerConfig
    ) -> None:
        """Output shape must match [B, N_ca, action_dim]."""
        B, N_ca, N_sro = 2, 5, 3
        ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
        
        actions = actor(ca, sro, nfc)
        
        assert actions.shape == (B, N_ca, default_config.action_dim)

    def test_variable_ca_count_same_model(
        self, actor: TransformerActor, default_config: TransformerConfig
    ) -> None:
        """Same model should handle different N_ca values."""
        B = 1
        N_sro = 4

        for N_ca in [2, 5, 10, 20]:
            ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
            actions = actor(ca, sro, nfc)
            assert actions.shape == (B, N_ca, default_config.action_dim)

    def test_sro_count_independence(
        self, actor: TransformerActor, default_config: TransformerConfig
    ) -> None:
        """Varying N_sro should not change output count."""
        B = 1
        N_ca = 4

        for N_sro in [0, 5, 15, 30]:
            ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
            actions = actor(ca, sro, nfc)
            # Output count must equal N_ca regardless of N_sro
            assert actions.shape[1] == N_ca

    def test_actions_bounded(
        self, actor: TransformerActor, default_config: TransformerConfig
    ) -> None:
        """Actions must be bounded in [-1, 1] due to tanh."""
        ca, sro, nfc = make_random_inputs(default_config, 4, 8, 5)
        actions = actor(ca, sro, nfc)
        
        assert torch.all(actions >= -1.0)
        assert torch.all(actions <= 1.0)

    def test_max_tokens_exceeded_raises(
        self, default_config: TransformerConfig
    ) -> None:
        """Exceeding max_tokens should raise."""
        cfg = TransformerConfig(max_tokens=10, action_dim=3)
        actor = TransformerActor(cfg)
        
        # 8 CA + 5 SRO + 1 NFC = 14 > 10
        ca, sro, nfc = make_random_inputs(cfg, 1, 8, 5)
        with pytest.raises(ValueError, match="exceeds max_tokens"):
            actor(ca, sro, nfc)

    def test_permutation_equivariance(
        self, actor: TransformerActor, default_config: TransformerConfig
    ) -> None:
        """Permuting CA inputs should permute outputs correspondingly.
        
        This validates strict 1-to-1 mapping: output[i] corresponds to ca_tokens[i].
        """
        torch.manual_seed(42)
        B, N_ca, N_sro = 1, 6, 4
        ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
        
        # Get baseline outputs
        actor.eval()
        with torch.no_grad():
            base = actor(ca, sro, nfc)
        
        # Permute CA tokens
        perm = torch.tensor([2, 5, 1, 0, 4, 3])
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm))
        
        ca_perm = ca[:, perm, :]
        with torch.no_grad():
            out_perm = actor(ca_perm, sro, nfc)
        
        # Reorder permuted outputs back
        out_perm_reordered = out_perm[:, inv_perm, :]
        
        # Should match baseline (within numerical tolerance)
        diff = (base - out_perm_reordered).abs().max().item()
        assert diff < 1e-5, f"Permutation equivariance violated, max diff: {diff}"

    def test_gradient_flow(
        self, actor: TransformerActor, default_config: TransformerConfig
    ) -> None:
        """Gradients should flow through all input tokens."""
        B, N_ca, N_sro = 2, 4, 3
        ca = torch.randn(B, N_ca, default_config.d_model, requires_grad=True)
        sro = torch.randn(B, N_sro, default_config.d_model, requires_grad=True)
        nfc = torch.randn(B, 1, default_config.d_model, requires_grad=True)
        
        actions = actor(ca, sro, nfc)
        loss = actions.sum()
        loss.backward()
        
        # All inputs should have gradients
        assert ca.grad is not None and ca.grad.abs().sum() > 0
        assert sro.grad is not None and sro.grad.abs().sum() > 0
        assert nfc.grad is not None and nfc.grad.abs().sum() > 0

    def test_with_masks(
        self, actor: TransformerActor, default_config: TransformerConfig
    ) -> None:
        """Actor should handle explicit masks."""
        B, N_ca, N_sro = 2, 5, 3
        ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
        
        ca_mask = torch.ones(B, N_ca, dtype=torch.bool)
        sro_mask = torch.ones(B, N_sro, dtype=torch.bool)
        
        # Mask out some tokens
        ca_mask[0, 3:] = False
        sro_mask[1, 1:] = False
        
        actions = actor(ca, sro, nfc, ca_mask=ca_mask, sro_mask=sro_mask)
        assert actions.shape == (B, N_ca, default_config.action_dim)


# --- TransformerCritic Tests ---


class TestTransformerCritic:
    """Tests for TransformerCritic."""

    def test_forward_produces_scalar(
        self, critic: TransformerCritic, default_config: TransformerConfig
    ) -> None:
        """Critic should produce [B, 1] Q-value."""
        B, N_ca, N_sro = 2, 5, 3
        ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
        actions = torch.randn(B, N_ca, default_config.action_dim)
        
        q = critic(ca, sro, nfc, actions)
        
        assert q.shape == (B, 1)

    def test_accepts_variable_cardinality(
        self, critic: TransformerCritic, default_config: TransformerConfig
    ) -> None:
        """Critic should handle different N_ca and N_sro."""
        B = 1
        
        for N_ca, N_sro in [(2, 5), (8, 0), (10, 20)]:
            ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
            actions = torch.randn(B, N_ca, default_config.action_dim)
            
            q = critic(ca, sro, nfc, actions)
            assert q.shape == (B, 1)

    def test_gradient_flow_through_all_inputs(
        self, critic: TransformerCritic, default_config: TransformerConfig
    ) -> None:
        """Gradients should flow through all inputs including actions."""
        B, N_ca, N_sro = 2, 4, 3
        ca = torch.randn(B, N_ca, default_config.d_model, requires_grad=True)
        sro = torch.randn(B, N_sro, default_config.d_model, requires_grad=True)
        nfc = torch.randn(B, 1, default_config.d_model, requires_grad=True)
        actions = torch.randn(B, N_ca, default_config.action_dim, requires_grad=True)
        
        q = critic(ca, sro, nfc, actions)
        loss = q.sum()
        loss.backward()
        
        # All inputs should have gradients
        assert ca.grad is not None and ca.grad.abs().sum() > 0
        assert sro.grad is not None and sro.grad.abs().sum() > 0
        assert nfc.grad is not None and nfc.grad.abs().sum() > 0
        assert actions.grad is not None and actions.grad.abs().sum() > 0

    def test_with_masks(
        self, critic: TransformerCritic, default_config: TransformerConfig
    ) -> None:
        """Critic should handle explicit masks."""
        B, N_ca, N_sro = 2, 5, 3
        ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
        actions = torch.randn(B, N_ca, default_config.action_dim)
        
        ca_mask = torch.ones(B, N_ca, dtype=torch.bool)
        sro_mask = torch.ones(B, N_sro, dtype=torch.bool)
        
        ca_mask[0, 3:] = False
        sro_mask[1, 1:] = False
        
        q = critic(ca, sro, nfc, actions, ca_mask=ca_mask, sro_mask=sro_mask)
        assert q.shape == (B, 1)


# --- Aggregation Strategy Tests ---


class TestAggregationStrategies:
    """Tests for CriticAggregationStrategy implementations."""

    def test_mean_pool_basic(self) -> None:
        """MeanPoolStrategy should compute correct mean."""
        strategy = MeanPoolStrategy()
        
        # [B=1, T=3, D=2]
        encoded = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = torch.tensor([[True, True, True]])
        
        pooled = strategy.aggregate(encoded, mask)
        
        expected = torch.tensor([[3.0, 4.0]])  # mean over T
        assert torch.allclose(pooled, expected)

    def test_mean_pool_with_mask(self) -> None:
        """MeanPoolStrategy should ignore masked positions."""
        strategy = MeanPoolStrategy()
        
        # [B=1, T=3, D=2]
        encoded = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])  # Mask out last
        
        pooled = strategy.aggregate(encoded, mask)
        
        expected = torch.tensor([[2.0, 3.0]])  # mean of first two
        assert torch.allclose(pooled, expected)

    def test_mean_pool_all_masked(self) -> None:
        """MeanPoolStrategy should handle all-masked gracefully (clamp to 1)."""
        strategy = MeanPoolStrategy()
        
        encoded = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.tensor([[False, False]])  # All masked
        
        pooled = strategy.aggregate(encoded, mask)
        
        # Should return zeros (masked values) / 1 = zeros
        assert pooled.shape == (1, 2)

    def test_cls_token_strategy(self) -> None:
        """CLSTokenStrategy should return last token."""
        strategy = CLSTokenStrategy()
        
        encoded = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = torch.tensor([[True, True, True]])
        
        pooled = strategy.aggregate(encoded, mask)
        
        expected = torch.tensor([[5.0, 6.0]])  # last token
        assert torch.allclose(pooled, expected)

    def test_critic_with_cls_strategy(self) -> None:
        """Critic should work with CLSTokenStrategy."""
        cfg = TransformerConfig(d_model=64, action_dim=3)
        critic = TransformerCritic(cfg, aggregation_strategy=CLSTokenStrategy())
        
        B, N_ca, N_sro = 2, 4, 3
        ca, sro, nfc = make_random_inputs(cfg, B, N_ca, N_sro)
        actions = torch.randn(B, N_ca, cfg.action_dim)
        
        q = critic(ca, sro, nfc, actions)
        assert q.shape == (B, 1)


# --- Integration Tests ---


class TestActorCriticIntegration:
    """Integration tests for Actor and Critic working together."""

    def test_actor_output_feeds_to_critic(
        self, default_config: TransformerConfig
    ) -> None:
        """Actor output should be compatible with Critic input."""
        actor = TransformerActor(default_config)
        critic = TransformerCritic(default_config)
        
        B, N_ca, N_sro = 2, 5, 3
        ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
        
        # Actor produces actions
        actions = actor(ca, sro, nfc)
        
        # Critic evaluates state-action pair
        q = critic(ca, sro, nfc, actions)
        
        assert actions.shape == (B, N_ca, default_config.action_dim)
        assert q.shape == (B, 1)

    def test_policy_gradient_end_to_end(
        self, default_config: TransformerConfig
    ) -> None:
        """Verify policy gradient can flow from critic to actor."""
        actor = TransformerActor(default_config)
        critic = TransformerCritic(default_config)
        
        B, N_ca, N_sro = 2, 4, 3
        ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
        
        # Forward pass
        actions = actor(ca, sro, nfc)
        q = critic(ca.detach(), sro.detach(), nfc.detach(), actions)
        
        # Actor loss (policy gradient)
        actor_loss = -q.mean()
        actor_loss.backward()
        
        # Actor parameters should have gradients
        actor_params = list(actor.parameters())
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in actor_params)

    def test_td_learning_end_to_end(
        self, default_config: TransformerConfig
    ) -> None:
        """Verify TD learning gradient flows to critic."""
        critic = TransformerCritic(default_config)
        critic_target = TransformerCritic(default_config)
        critic_target.load_state_dict(critic.state_dict())
        
        B, N_ca, N_sro = 2, 4, 3
        gamma = 0.99
        
        # Current state-action
        ca, sro, nfc = make_random_inputs(default_config, B, N_ca, N_sro)
        actions = torch.randn(B, N_ca, default_config.action_dim)
        rewards = torch.randn(B, 1)
        
        # Next state-action
        next_ca, next_sro, next_nfc = make_random_inputs(default_config, B, N_ca, N_sro)
        next_actions = torch.randn(B, N_ca, default_config.action_dim)
        
        # TD target
        with torch.no_grad():
            q_next = critic_target(next_ca, next_sro, next_nfc, next_actions)
            q_target = rewards + gamma * q_next
        
        # TD prediction
        q_pred = critic(ca, sro, nfc, actions)
        
        # TD loss
        critic_loss = torch.nn.functional.mse_loss(q_pred, q_target)
        critic_loss.backward()
        
        # Critic parameters should have gradients
        critic_params = list(critic.parameters())
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in critic_params)
