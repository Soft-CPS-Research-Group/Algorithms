"""Tests for tokenizer config schema validation."""

import json
import pytest
from pathlib import Path

from utils.config_schema import (
    TokenizerConfig,
    CATypeConfig,
    SROTypeConfig,
    NFCConfig,
    MarkerValuesConfig,
)


class TestTokenizerConfigSchema:
    """Tests for TokenizerConfig Pydantic model."""

    def test_load_default_config(self) -> None:
        """Default tokenizer config should validate successfully."""
        config_path = Path("configs/tokenizers/default.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        
        config = TokenizerConfig(**raw_config)
        
        assert config.marker_values.ca_base == 1000
        assert config.marker_values.sro_base == 2000
        assert config.marker_values.nfc == 3001

    def test_ca_types_parsed(self) -> None:
        """CA types should be parsed with correct structure."""
        config_path = Path("configs/tokenizers/default.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        
        config = TokenizerConfig(**raw_config)
        
        assert "battery" in config.ca_types
        assert "ev_charger" in config.ca_types
        assert "washing_machine" in config.ca_types
        
        battery = config.ca_types["battery"]
        assert battery.action_name == "electrical_storage"
        assert battery.input_dim == 1
        assert "electrical_storage_soc" in battery.features

    def test_sro_types_parsed(self) -> None:
        """SRO types should be parsed with correct structure."""
        config_path = Path("configs/tokenizers/default.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        
        config = TokenizerConfig(**raw_config)
        
        assert "temporal" in config.sro_types
        assert "pricing" in config.sro_types
        assert "carbon" in config.sro_types
        
        temporal = config.sro_types["temporal"]
        assert temporal.input_dim == 12
        assert "month" in temporal.features

    def test_nfc_config_parsed(self) -> None:
        """NFC config should be parsed correctly."""
        config_path = Path("configs/tokenizers/default.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        
        config = TokenizerConfig(**raw_config)
        
        assert "non_shiftable_load" in config.nfc.demand_features
        assert "solar_generation" in config.nfc.generation_features
        assert config.nfc.input_dim == 3

    def test_invalid_ca_type_missing_input_dim(self) -> None:
        """CA type without input_dim should fail validation."""
        invalid_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    # Missing input_dim
                }
            },
            "sro_types": {},
            "nfc": {
                "demand_features": [],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 0,
            },
        }
        
        with pytest.raises(ValueError):
            TokenizerConfig(**invalid_config)

    def test_marker_values_must_be_positive(self) -> None:
        """Marker base values must be positive integers."""
        invalid_config = {
            "marker_values": {"ca_base": -1, "sro_base": 2000, "nfc": 3001},
            "ca_types": {},
            "sro_types": {},
            "nfc": {
                "demand_features": [],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 0,
            },
        }
        
        with pytest.raises(ValueError):
            TokenizerConfig(**invalid_config)


class TestTransformerPPOConfigSchema:
    """Tests for TransformerPPO algorithm config schema."""

    def test_transformer_config_valid(self) -> None:
        """Valid Transformer config should parse successfully."""
        from utils.config_schema import TransformerConfig
        
        config = TransformerConfig(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        )
        
        assert config.d_model == 64
        assert config.nhead == 4
        assert config.num_layers == 2

    def test_transformer_config_invalid_nhead(self) -> None:
        """nhead must divide d_model evenly."""
        from utils.config_schema import TransformerConfig
        
        with pytest.raises(ValueError):
            TransformerConfig(
                d_model=64,
                nhead=5,  # 64 % 5 != 0
                num_layers=2,
            )

    def test_transformer_ppo_hyperparameters_valid(self) -> None:
        """Valid PPO hyperparameters should parse successfully."""
        from utils.config_schema import TransformerPPOHyperparameters
        
        config = TransformerPPOHyperparameters(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            ppo_epochs=4,
            minibatch_size=64,
            entropy_coeff=0.01,
            value_coeff=0.5,
            max_grad_norm=0.5,
        )
        
        assert config.gamma == 0.99
        assert config.clip_eps == 0.2

    def test_transformer_ppo_algorithm_config_valid(self) -> None:
        """Full TransformerPPO algorithm config should parse."""
        from utils.config_schema import TransformerPPOAlgorithmConfig
        
        config = TransformerPPOAlgorithmConfig(
            name="AgentTransformerPPO",
            tokenizer_config_path="configs/tokenizers/default.json",
            transformer={
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.1,
            },
            hyperparameters={
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "ppo_epochs": 4,
                "minibatch_size": 64,
                "entropy_coeff": 0.01,
                "value_coeff": 0.5,
                "max_grad_norm": 0.5,
            },
        )
        
        assert config.name == "AgentTransformerPPO"
        assert config.transformer.d_model == 64
