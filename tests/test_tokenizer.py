"""Tests for the Observation Tokenizer.

Covers:
- TokenizerConfig defaults
- FeatureClassifier pattern matching
- ObservationTokenizer shape correctness
- Variable agent counts
- Mask validity
- Edge cases (zero SRO, missing NFC)
"""

from __future__ import annotations

import pytest
import numpy as np
import torch

from algorithms.utils.tokenizer import (
    FeatureClassifier,
    FeatureEmbedding,
    ObservationTokenizer,
    TokenizerConfig,
)


# --- Fixtures ---


@pytest.fixture
def basic_config() -> TokenizerConfig:
    """Basic tokenizer configuration."""
    return TokenizerConfig(d_model=64)


@pytest.fixture
def sample_observation_names() -> list[list[str]]:
    """Sample observation names for 2 agents."""
    return [
        [
            "month",
            "hour",
            "electric_vehicle_soc",
            "connected_state",
            "departure_time",
            "arrival_time",
            "outdoor_dry_bulb_temperature",
            "non_shiftable_load",
            "carbon_intensity",
        ],
        [
            "month",
            "hour",
            "electric_vehicle_soc",
            "connected_state",
            "departure_time",
            "arrival_time",
            "outdoor_dry_bulb_temperature",
            "non_shiftable_load",
            "carbon_intensity",
        ],
    ]


@pytest.fixture
def tokenizer(
    basic_config: TokenizerConfig,
    sample_observation_names: list[list[str]],
) -> ObservationTokenizer:
    """Create tokenizer with sample configuration."""
    return ObservationTokenizer(basic_config, sample_observation_names)


def make_dummy_observations(
    num_agents: int,
    obs_dim: int,
) -> list[np.ndarray]:
    """Generate dummy encoded observations."""
    return [np.random.randn(obs_dim).astype(np.float32) for _ in range(num_agents)]


# --- TokenizerConfig Tests ---


class TestTokenizerConfig:
    """Tests for TokenizerConfig."""

    def test_default_values(self) -> None:
        """Default config should have sensible values."""
        cfg = TokenizerConfig()
        assert cfg.d_model == 64
        assert len(cfg.ca_feature_patterns) > 0
        assert len(cfg.sro_feature_patterns) > 0
        assert len(cfg.nfc_feature_patterns) > 0

    def test_custom_d_model(self) -> None:
        """Custom d_model should be respected."""
        cfg = TokenizerConfig(d_model=128)
        assert cfg.d_model == 128

    def test_custom_patterns(self) -> None:
        """Custom patterns should override defaults."""
        cfg = TokenizerConfig(
            ca_feature_patterns=["custom_ca"],
            sro_feature_patterns=["custom_sro"],
            nfc_feature_patterns=["custom_nfc"],
        )
        assert cfg.ca_feature_patterns == ["custom_ca"]
        assert cfg.sro_feature_patterns == ["custom_sro"]
        assert cfg.nfc_feature_patterns == ["custom_nfc"]


# --- FeatureClassifier Tests ---


class TestFeatureClassifier:
    """Tests for FeatureClassifier."""

    def test_classify_ca_features(self, basic_config: TokenizerConfig) -> None:
        """CA features should be classified correctly."""
        classifier = FeatureClassifier(basic_config)
        
        ca_names = [
            "electric_vehicle_soc",
            "connected_state",
            "departure_time",
            "arrival_time",
        ]
        for name in ca_names:
            assert classifier.classify(name) == FeatureClassifier.TYPE_CA, f"Failed for {name}"

    def test_classify_sro_features(self, basic_config: TokenizerConfig) -> None:
        """SRO features should be classified correctly."""
        classifier = FeatureClassifier(basic_config)
        
        sro_names = [
            "outdoor_dry_bulb_temperature",
            "carbon_intensity",
            "electricity_pricing",
            "solar_generation",
        ]
        for name in sro_names:
            assert classifier.classify(name) == FeatureClassifier.TYPE_SRO, f"Failed for {name}"

    def test_classify_nfc_features(self, basic_config: TokenizerConfig) -> None:
        """NFC features should be classified correctly."""
        classifier = FeatureClassifier(basic_config)
        
        nfc_names = [
            "non_shiftable_load",
            "non_flexible_load",
        ]
        for name in nfc_names:
            assert classifier.classify(name) == FeatureClassifier.TYPE_NFC, f"Failed for {name}"

    def test_classify_unknown_defaults_to_sro(self, basic_config: TokenizerConfig) -> None:
        """Unknown features should default to SRO."""
        classifier = FeatureClassifier(basic_config)
        
        # These don't match any pattern
        unknown_names = ["month", "hour", "day_type"]
        for name in unknown_names:
            assert classifier.classify(name) == FeatureClassifier.TYPE_SRO, f"Failed for {name}"

    def test_classify_all(self, basic_config: TokenizerConfig) -> None:
        """classify_all should group features by type."""
        classifier = FeatureClassifier(basic_config)
        
        feature_names = [
            "electric_vehicle_soc",
            "outdoor_dry_bulb_temperature",
            "non_shiftable_load",
            "connected_state",
        ]
        
        result = classifier.classify_all(feature_names)
        
        # Check CA features
        ca_names = [name for _, name in result[FeatureClassifier.TYPE_CA]]
        assert "electric_vehicle_soc" in ca_names
        assert "connected_state" in ca_names
        
        # Check SRO features
        sro_names = [name for _, name in result[FeatureClassifier.TYPE_SRO]]
        assert "outdoor_dry_bulb_temperature" in sro_names
        
        # Check NFC features
        nfc_names = [name for _, name in result[FeatureClassifier.TYPE_NFC]]
        assert "non_shiftable_load" in nfc_names

    def test_case_insensitive(self, basic_config: TokenizerConfig) -> None:
        """Pattern matching should be case insensitive."""
        classifier = FeatureClassifier(basic_config)
        
        assert classifier.classify("ELECTRIC_VEHICLE_SOC") == FeatureClassifier.TYPE_CA
        assert classifier.classify("Electric_Vehicle_Soc") == FeatureClassifier.TYPE_CA


# --- FeatureEmbedding Tests ---


class TestFeatureEmbedding:
    """Tests for FeatureEmbedding."""

    def test_output_dimension(self) -> None:
        """Output should have d_model dimension."""
        embed = FeatureEmbedding(input_dim=10, d_model=64)
        x = torch.randn(2, 5, 10)  # [B, N, input_dim]
        
        out = embed(x)
        
        assert out.shape == (2, 5, 64)

    def test_single_vector(self) -> None:
        """Should handle single vectors."""
        embed = FeatureEmbedding(input_dim=10, d_model=64)
        x = torch.randn(10)  # [input_dim]
        
        out = embed(x)
        
        assert out.shape == (64,)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through embedding."""
        embed = FeatureEmbedding(input_dim=10, d_model=64)
        x = torch.randn(2, 10, requires_grad=True)
        
        out = embed(x)
        loss = out.sum()
        loss.backward()
        
        # Check that embedding parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in embed.parameters()
        )
        assert has_grad, "Embedding parameters should have gradients"


# --- ObservationTokenizer Tests ---


class TestObservationTokenizer:
    """Tests for ObservationTokenizer."""

    def test_tokenize_produces_correct_shapes(
        self,
        tokenizer: ObservationTokenizer,
        sample_observation_names: list[list[str]],
    ) -> None:
        """tokenize should produce correct output shapes."""
        num_agents = len(sample_observation_names)
        obs_dim = len(sample_observation_names[0])
        observations = make_dummy_observations(num_agents, obs_dim)
        
        ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask = tokenizer.tokenize(
            observations
        )
        
        d_model = tokenizer.d_model
        
        # CA: one token per agent
        assert ca_tokens.shape == (1, num_agents, d_model)
        # SRO: aggregated into 1 token
        assert sro_tokens.shape == (1, 1, d_model)
        # NFC: exactly 1 token
        assert nfc_token.shape == (1, 1, d_model)
        # Masks
        assert ca_mask.shape == (1, num_agents)
        assert sro_mask.shape == (1, 1)

    def test_variable_agent_count(self, basic_config: TokenizerConfig) -> None:
        """Tokenizer should handle different agent counts."""
        for num_agents in [1, 3, 5, 10]:
            obs_names = [
                ["electric_vehicle_soc", "connected_state", "non_shiftable_load"]
                for _ in range(num_agents)
            ]
            tokenizer = ObservationTokenizer(basic_config, obs_names)
            observations = make_dummy_observations(num_agents, 3)
            
            ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask = tokenizer.tokenize(
                observations
            )
            
            assert ca_tokens.shape[1] == num_agents

    def test_embedding_dimensions_match_d_model(
        self, sample_observation_names: list[list[str]]
    ) -> None:
        """All embeddings should have d_model dimension."""
        d_model = 128
        config = TokenizerConfig(d_model=d_model)
        tokenizer = ObservationTokenizer(config, sample_observation_names)
        
        num_agents = len(sample_observation_names)
        obs_dim = len(sample_observation_names[0])
        observations = make_dummy_observations(num_agents, obs_dim)
        
        ca_tokens, sro_tokens, nfc_token, _, _ = tokenizer.tokenize(observations)
        
        assert ca_tokens.shape[-1] == d_model
        assert sro_tokens.shape[-1] == d_model
        assert nfc_token.shape[-1] == d_model

    def test_masks_indicate_valid_tokens(
        self,
        tokenizer: ObservationTokenizer,
        sample_observation_names: list[list[str]],
    ) -> None:
        """All tokens should be valid (True) in non-batched case."""
        observations = make_dummy_observations(
            len(sample_observation_names),
            len(sample_observation_names[0]),
        )
        
        _, _, _, ca_mask, sro_mask = tokenizer.tokenize(observations)
        
        assert ca_mask.all()
        assert sro_mask.all()

    def test_get_feature_indices(
        self,
        tokenizer: ObservationTokenizer,
    ) -> None:
        """get_feature_indices should return index mappings."""
        indices = tokenizer.get_feature_indices()
        
        assert "ca_indices" in indices
        assert "sro_indices" in indices
        assert "nfc_indices" in indices
        assert "ca_dims" in indices
        assert "sro_dim" in indices
        assert "nfc_dim" in indices

    def test_edge_case_zero_sro(self, basic_config: TokenizerConfig) -> None:
        """Tokenizer should handle observations with no SRO features."""
        # Only CA and NFC features
        obs_names = [
            ["electric_vehicle_soc", "connected_state", "non_shiftable_load"],
            ["electric_vehicle_soc", "connected_state", "non_shiftable_load"],
        ]
        tokenizer = ObservationTokenizer(basic_config, obs_names)
        observations = make_dummy_observations(2, 3)
        
        ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask = tokenizer.tokenize(
            observations
        )
        
        # Should still produce valid outputs (empty SRO gets padded)
        assert ca_tokens.shape == (1, 2, basic_config.d_model)
        assert sro_tokens.shape == (1, 1, basic_config.d_model)
        assert nfc_token.shape == (1, 1, basic_config.d_model)

    def test_edge_case_no_nfc(self, basic_config: TokenizerConfig) -> None:
        """Tokenizer should handle observations with no NFC features."""
        # Only CA and SRO features
        obs_names = [
            ["electric_vehicle_soc", "outdoor_dry_bulb_temperature"],
            ["electric_vehicle_soc", "outdoor_dry_bulb_temperature"],
        ]
        tokenizer = ObservationTokenizer(basic_config, obs_names)
        observations = make_dummy_observations(2, 2)
        
        ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask = tokenizer.tokenize(
            observations
        )
        
        # Should still produce valid outputs
        assert ca_tokens.shape == (1, 2, basic_config.d_model)
        assert nfc_token.shape == (1, 1, basic_config.d_model)

    def test_tokenize_batch(
        self,
        tokenizer: ObservationTokenizer,
        sample_observation_names: list[list[str]],
    ) -> None:
        """tokenize_batch should handle multiple observations."""
        num_agents = len(sample_observation_names)
        obs_dim = len(sample_observation_names[0])
        batch_size = 4
        
        observations_batch = [
            make_dummy_observations(num_agents, obs_dim)
            for _ in range(batch_size)
        ]
        
        ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask = tokenizer.tokenize_batch(
            observations_batch
        )
        
        d_model = tokenizer.d_model
        assert ca_tokens.shape == (batch_size, num_agents, d_model)
        assert sro_tokens.shape == (batch_size, 1, d_model)
        assert nfc_token.shape == (batch_size, 1, d_model)
        assert ca_mask.shape == (batch_size, num_agents)
        assert sro_mask.shape == (batch_size, 1)

    def test_gradient_flow(
        self,
        tokenizer: ObservationTokenizer,
        sample_observation_names: list[list[str]],
    ) -> None:
        """Gradients should flow through tokenization."""
        num_agents = len(sample_observation_names)
        obs_dim = len(sample_observation_names[0])
        
        # Use numpy arrays (tokenizer expects numpy input)
        observations = make_dummy_observations(num_agents, obs_dim)
        
        ca_tokens, sro_tokens, nfc_token, _, _ = tokenizer.tokenize(observations)
        
        # Tokens should require grad (from embedding params)
        loss = ca_tokens.sum() + sro_tokens.sum() + nfc_token.sum()
        loss.backward()
        
        # Check embedding parameters have gradients
        for embed in tokenizer.ca_embeddings:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in embed.parameters()
            )
            assert has_grad, "CA embedding parameters should have gradients"

    def test_device_placement(
        self,
        tokenizer: ObservationTokenizer,
        sample_observation_names: list[list[str]],
    ) -> None:
        """Tensors should be placed on specified device."""
        num_agents = len(sample_observation_names)
        obs_dim = len(sample_observation_names[0])
        observations = make_dummy_observations(num_agents, obs_dim)
        
        device = torch.device("cpu")
        ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask = tokenizer.tokenize(
            observations, device=device
        )
        
        assert ca_tokens.device == device
        assert sro_tokens.device == device
        assert nfc_token.device == device
        assert ca_mask.device == device
        assert sro_mask.device == device


# --- Integration Tests ---


class TestTokenizerIntegration:
    """Integration tests with Transformer networks."""

    def test_tokenizer_output_compatible_with_transformer(
        self,
        tokenizer: ObservationTokenizer,
        sample_observation_names: list[list[str]],
    ) -> None:
        """Tokenizer output should be compatible with TransformerActor."""
        from algorithms.utils.transformer_networks import (
            TransformerActor,
            TransformerConfig,
        )
        
        num_agents = len(sample_observation_names)
        obs_dim = len(sample_observation_names[0])
        observations = make_dummy_observations(num_agents, obs_dim)
        
        # Tokenize
        ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask = tokenizer.tokenize(
            observations
        )
        
        # Create actor with matching config
        actor_config = TransformerConfig(
            d_model=tokenizer.d_model,
            action_dim=2,
        )
        actor = TransformerActor(actor_config)
        
        # Forward pass should work
        actions = actor(ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask)
        
        # Output should be [B, N_ca, action_dim]
        assert actions.shape == (1, num_agents, actor_config.action_dim)
