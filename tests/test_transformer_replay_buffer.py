"""Tests for Transformer Replay Buffer.

Covers:
- Basic push and sample operations
- Variable cardinality experiences
- Mask correctness
- State serialization roundtrip
- Capacity overflow (circular buffer)
"""

from __future__ import annotations

import pytest
import torch

from algorithms.utils.transformer_replay_buffer import (
    PaddedStorageStrategy,
    TransformerExperience,
    TransformerReplayBuffer,
    TransformerReplayBufferConfig,
)


# --- Fixtures ---


@pytest.fixture
def buffer_config() -> TransformerReplayBufferConfig:
    """Default buffer configuration for tests."""
    return TransformerReplayBufferConfig(
        capacity=100,
        batch_size=8,
        max_ca=16,
        max_sro=8,
        d_model=32,
        action_dim=2,
    )


@pytest.fixture
def buffer(buffer_config: TransformerReplayBufferConfig) -> TransformerReplayBuffer:
    """Create a buffer for testing."""
    return TransformerReplayBuffer(buffer_config)


def make_experience_tensors(
    n_ca: int = 4,
    n_sro: int = 2,
    d_model: int = 32,
    action_dim: int = 2,
) -> dict:
    """Generate random tensors for a single experience."""
    return {
        "ca_tokens": torch.randn(1, n_ca, d_model),
        "sro_tokens": torch.randn(1, n_sro, d_model),
        "nfc_token": torch.randn(1, 1, d_model),
        "ca_mask": torch.ones(1, n_ca, dtype=torch.bool),
        "sro_mask": torch.ones(1, n_sro, dtype=torch.bool),
        "actions": torch.randn(1, n_ca, action_dim),
        "rewards": torch.randn(1, 1),
        "next_ca_tokens": torch.randn(1, n_ca, d_model),
        "next_sro_tokens": torch.randn(1, n_sro, d_model),
        "next_nfc_token": torch.randn(1, 1, d_model),
        "next_ca_mask": torch.ones(1, n_ca, dtype=torch.bool),
        "next_sro_mask": torch.ones(1, n_sro, dtype=torch.bool),
        "done": torch.tensor([[0.0]]),
    }


# --- Configuration Tests ---


class TestTransformerReplayBufferConfig:
    """Tests for TransformerReplayBufferConfig."""

    def test_default_values(self) -> None:
        """Default config should have sensible values."""
        cfg = TransformerReplayBufferConfig()
        assert cfg.capacity == 100000
        assert cfg.batch_size == 256
        assert cfg.max_ca == 64
        assert cfg.max_sro == 32
        assert cfg.d_model == 64
        assert cfg.action_dim == 1

    def test_custom_values(self) -> None:
        """Custom values should be respected."""
        cfg = TransformerReplayBufferConfig(
            capacity=1000,
            batch_size=32,
            max_ca=8,
            max_sro=4,
            d_model=128,
            action_dim=3,
        )
        assert cfg.capacity == 1000
        assert cfg.batch_size == 32


# --- Basic Operations Tests ---


class TestTransformerReplayBuffer:
    """Tests for TransformerReplayBuffer basic operations."""

    def test_push_basic(
        self, buffer: TransformerReplayBuffer, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """Push should add experience to buffer."""
        assert len(buffer) == 0
        
        exp = make_experience_tensors(
            d_model=buffer_config.d_model,
            action_dim=buffer_config.action_dim,
        )
        buffer.push(**exp)
        
        assert len(buffer) == 1

    def test_push_multiple(
        self, buffer: TransformerReplayBuffer, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """Multiple pushes should accumulate."""
        for _ in range(10):
            exp = make_experience_tensors(
                d_model=buffer_config.d_model,
                action_dim=buffer_config.action_dim,
            )
            buffer.push(**exp)
        
        assert len(buffer) == 10

    def test_sample_batch_shapes(
        self, buffer: TransformerReplayBuffer, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """Sample should return correct shapes."""
        # Fill buffer with enough samples
        for _ in range(20):
            exp = make_experience_tensors(
                d_model=buffer_config.d_model,
                action_dim=buffer_config.action_dim,
            )
            buffer.push(**exp)
        
        batch = buffer.sample(batch_size=8)
        
        # Check all expected keys present
        expected_keys = [
            "ca_tokens", "sro_tokens", "nfc_token",
            "ca_mask", "sro_mask", "actions", "rewards",
            "next_ca_tokens", "next_sro_tokens", "next_nfc_token",
            "next_ca_mask", "next_sro_mask", "dones",
            "n_ca", "n_sro",
        ]
        for key in expected_keys:
            assert key in batch, f"Missing key: {key}"
        
        # Check shapes
        B = 8
        assert batch["ca_tokens"].shape == (B, buffer_config.max_ca, buffer_config.d_model)
        assert batch["sro_tokens"].shape == (B, buffer_config.max_sro, buffer_config.d_model)
        assert batch["nfc_token"].shape == (B, 1, buffer_config.d_model)
        assert batch["ca_mask"].shape == (B, buffer_config.max_ca)
        assert batch["sro_mask"].shape == (B, buffer_config.max_sro)
        assert batch["actions"].shape == (B, buffer_config.max_ca, buffer_config.action_dim)
        assert batch["rewards"].shape == (B, 1)
        assert batch["dones"].shape == (B, 1)
        assert batch["n_ca"].shape == (B,)
        assert batch["n_sro"].shape == (B,)

    def test_sample_raises_on_empty(self, buffer: TransformerReplayBuffer) -> None:
        """Sample should raise when buffer is empty."""
        with pytest.raises(ValueError, match="Not enough samples"):
            buffer.sample()

    def test_sample_raises_on_insufficient(
        self, buffer: TransformerReplayBuffer, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """Sample should raise when buffer has fewer samples than batch_size."""
        # Add fewer samples than batch_size
        for _ in range(5):
            exp = make_experience_tensors(
                d_model=buffer_config.d_model,
                action_dim=buffer_config.action_dim,
            )
            buffer.push(**exp)
        
        with pytest.raises(ValueError, match="Not enough samples"):
            buffer.sample(batch_size=10)


# --- Variable Cardinality Tests ---


class TestVariableCardinality:
    """Tests for handling variable N_ca and N_sro."""

    def test_variable_cardinality_experiences(
        self, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """Buffer should handle experiences with different cardinalities."""
        buffer = TransformerReplayBuffer(buffer_config)
        
        # Push experiences with varying cardinalities
        cardinalities = [(2, 1), (4, 3), (8, 4), (3, 2), (6, 2)]
        for n_ca, n_sro in cardinalities:
            exp = make_experience_tensors(
                n_ca=n_ca,
                n_sro=n_sro,
                d_model=buffer_config.d_model,
                action_dim=buffer_config.action_dim,
            )
            buffer.push(**exp)
        
        assert len(buffer) == 5
        
        # Sample should work
        batch = buffer.sample(batch_size=3)
        assert batch["ca_tokens"].shape[0] == 3

    def test_masks_correctly_indicate_padding(
        self, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """Masks should correctly identify valid vs padded positions."""
        buffer = TransformerReplayBuffer(buffer_config)
        
        # Add experience with specific cardinality
        n_ca, n_sro = 3, 2
        exp = make_experience_tensors(
            n_ca=n_ca,
            n_sro=n_sro,
            d_model=buffer_config.d_model,
            action_dim=buffer_config.action_dim,
        )
        
        # Add multiple copies with same cardinality
        for _ in range(10):
            buffer.push(**exp)
        
        batch = buffer.sample(batch_size=8)
        
        # Check masks
        ca_mask = batch["ca_mask"]
        sro_mask = batch["sro_mask"]
        
        for i in range(8):
            # First n_ca positions should be True
            assert ca_mask[i, :n_ca].all()
            # Remaining should be False
            if n_ca < buffer_config.max_ca:
                assert not ca_mask[i, n_ca:].any()
            
            # Same for SRO
            assert sro_mask[i, :n_sro].all()
            if n_sro < buffer_config.max_sro:
                assert not sro_mask[i, n_sro:].any()

    def test_n_ca_n_sro_counts(
        self, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """n_ca and n_sro should reflect actual cardinalities."""
        buffer = TransformerReplayBuffer(buffer_config)
        
        # Add experiences with known cardinalities
        for n_ca in [2, 4, 6]:
            n_sro = n_ca // 2
            exp = make_experience_tensors(
                n_ca=n_ca,
                n_sro=n_sro,
                d_model=buffer_config.d_model,
                action_dim=buffer_config.action_dim,
            )
            for _ in range(5):
                buffer.push(**exp)
        
        batch = buffer.sample(batch_size=10)
        
        n_ca_vals = batch["n_ca"]
        n_sro_vals = batch["n_sro"]
        
        # All n_ca should be in [2, 4, 6]
        for n in n_ca_vals.tolist():
            assert n in [2, 4, 6]
        
        # n_sro should be n_ca // 2
        for n_ca, n_sro in zip(n_ca_vals.tolist(), n_sro_vals.tolist()):
            assert n_sro == n_ca // 2


# --- State Serialization Tests ---


class TestStateSerialization:
    """Tests for checkpoint save/restore."""

    def test_state_serialization_roundtrip(
        self, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """Buffer state should survive serialization roundtrip."""
        buffer = TransformerReplayBuffer(buffer_config)
        
        # Add experiences
        for _ in range(20):
            exp = make_experience_tensors(
                d_model=buffer_config.d_model,
                action_dim=buffer_config.action_dim,
            )
            buffer.push(**exp)
        
        original_len = len(buffer)
        
        # Get state
        state = buffer.get_state()
        
        # Create new buffer and restore
        new_buffer = TransformerReplayBuffer(buffer_config)
        new_buffer.set_state(state)
        
        assert len(new_buffer) == original_len

    def test_set_state_none_clears(
        self, buffer: TransformerReplayBuffer, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """set_state(None) should clear buffer."""
        # Add some experiences
        for _ in range(10):
            exp = make_experience_tensors(
                d_model=buffer_config.d_model,
                action_dim=buffer_config.action_dim,
            )
            buffer.push(**exp)
        
        assert len(buffer) > 0
        
        buffer.set_state(None)
        
        assert len(buffer) == 0

    def test_clear(
        self, buffer: TransformerReplayBuffer, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """clear() should empty the buffer."""
        for _ in range(10):
            exp = make_experience_tensors(
                d_model=buffer_config.d_model,
                action_dim=buffer_config.action_dim,
            )
            buffer.push(**exp)
        
        assert len(buffer) > 0
        
        buffer.clear()
        
        assert len(buffer) == 0


# --- Capacity Tests ---


class TestCapacity:
    """Tests for buffer capacity handling."""

    def test_capacity_overflow_circular(self) -> None:
        """Buffer should act as circular buffer when capacity exceeded."""
        cfg = TransformerReplayBufferConfig(
            capacity=10,
            batch_size=4,
            max_ca=4,
            max_sro=2,
            d_model=16,
            action_dim=1,
        )
        buffer = TransformerReplayBuffer(cfg)
        
        # Fill to capacity
        for _ in range(10):
            exp = make_experience_tensors(
                d_model=cfg.d_model,
                action_dim=cfg.action_dim,
            )
            buffer.push(**exp)
        
        assert len(buffer) == 10
        
        # Add more - should maintain capacity
        for _ in range(5):
            exp = make_experience_tensors(
                d_model=cfg.d_model,
                action_dim=cfg.action_dim,
            )
            buffer.push(**exp)
        
        assert len(buffer) == 10  # Still capped at capacity


# --- PaddedStorageStrategy Tests ---


class TestPaddedStorageStrategy:
    """Tests for PaddedStorageStrategy."""

    def test_collate_batch_basic(self) -> None:
        """Strategy should correctly collate experiences."""
        strategy = PaddedStorageStrategy()
        cfg = TransformerReplayBufferConfig(
            max_ca=8,
            max_sro=4,
            d_model=32,
            action_dim=2,
        )
        device = torch.device("cpu")
        
        # Create experiences with different cardinalities
        experiences = []
        for n_ca, n_sro in [(2, 1), (4, 2), (3, 1)]:
            exp = TransformerExperience(
                ca_tokens=torch.randn(n_ca, cfg.d_model),
                sro_tokens=torch.randn(n_sro, cfg.d_model),
                nfc_token=torch.randn(1, cfg.d_model),
                ca_mask=torch.ones(n_ca, dtype=torch.bool),
                sro_mask=torch.ones(n_sro, dtype=torch.bool),
                actions=torch.randn(n_ca, cfg.action_dim),
                rewards=torch.randn(1),
                next_ca_tokens=torch.randn(n_ca, cfg.d_model),
                next_sro_tokens=torch.randn(n_sro, cfg.d_model),
                next_nfc_token=torch.randn(1, cfg.d_model),
                next_ca_mask=torch.ones(n_ca, dtype=torch.bool),
                next_sro_mask=torch.ones(n_sro, dtype=torch.bool),
                done=torch.tensor([0.0]),
                n_ca=n_ca,
                n_sro=n_sro,
            )
            experiences.append(exp)
        
        batch = strategy.collate_batch(experiences, cfg, device)
        
        # Check shapes
        assert batch["ca_tokens"].shape == (3, cfg.max_ca, cfg.d_model)
        assert batch["sro_tokens"].shape == (3, cfg.max_sro, cfg.d_model)
        
        # Check n_ca tracking
        assert batch["n_ca"].tolist() == [2, 4, 3]
        assert batch["n_sro"].tolist() == [1, 2, 1]


# --- Integration Tests ---


class TestReplayBufferIntegration:
    """Integration tests with Transformer networks."""

    def test_sampled_batch_usable_by_networks(
        self, buffer_config: TransformerReplayBufferConfig
    ) -> None:
        """Sampled batch should be directly usable by Transformer networks."""
        from algorithms.utils.transformer_networks import (
            TransformerActor,
            TransformerConfig,
            TransformerCritic,
        )
        
        buffer = TransformerReplayBuffer(buffer_config)
        
        # Fill buffer
        for _ in range(20):
            exp = make_experience_tensors(
                n_ca=4,
                n_sro=2,
                d_model=buffer_config.d_model,
                action_dim=buffer_config.action_dim,
            )
            buffer.push(**exp)
        
        batch = buffer.sample(batch_size=8)
        
        # Create networks
        net_cfg = TransformerConfig(
            d_model=buffer_config.d_model,
            action_dim=buffer_config.action_dim,
            max_tokens=buffer_config.max_ca + buffer_config.max_sro + 1,
        )
        actor = TransformerActor(net_cfg)
        critic = TransformerCritic(net_cfg)
        
        # Forward passes should work
        actions = actor(
            batch["ca_tokens"],
            batch["sro_tokens"],
            batch["nfc_token"],
            batch["ca_mask"],
            batch["sro_mask"],
        )
        assert actions.shape == (8, buffer_config.max_ca, buffer_config.action_dim)
        
        q = critic(
            batch["ca_tokens"],
            batch["sro_tokens"],
            batch["nfc_token"],
            batch["actions"],
            batch["ca_mask"],
            batch["sro_mask"],
        )
        assert q.shape == (8, 1)
