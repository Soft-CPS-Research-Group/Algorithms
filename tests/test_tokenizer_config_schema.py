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
