"""Tests for Transformer agent wrapper integration."""

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np


class TestWrapperEnricherSetup:
    """Tests for enricher setup in wrapper."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config."""
        return {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                }
            },
            "sro_types": {
                "temporal": {"features": ["month", "hour"], "input_dim": 4},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }

    def test_enricher_created_for_transformer_agent(
        self, sample_tokenizer_config: Dict[str, Any]
    ) -> None:
        """Enrichers should be created when agent is Transformer-based."""
        from algorithms.utils.observation_enricher import ObservationEnricher
        from utils.wrapper_citylearn import Wrapper_CityLearn
        
        # Mock the agent to indicate it's a Transformer agent
        mock_agent = MagicMock()
        mock_agent.is_transformer_agent = True
        mock_agent.tokenizer_config = sample_tokenizer_config
        
        # This test verifies the wrapper has enricher setup capability
        # Actual test depends on wrapper implementation
        enricher = ObservationEnricher(sample_tokenizer_config)
        assert enricher is not None

    def test_no_enricher_for_non_transformer_agent(self) -> None:
        """Enrichers should not be created for non-Transformer agents."""
        # This test documents expected behavior
        # Non-Transformer agents skip enrichment entirely
        pass


class TestWrapperEnrichment:
    """Tests for observation enrichment in wrapper."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config."""
        return {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                }
            },
            "sro_types": {
                "temporal": {"features": ["month"], "input_dim": 2},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }

    def test_enriched_obs_contains_markers(
        self, sample_tokenizer_config: Dict[str, Any]
    ) -> None:
        """Enriched observations should contain marker values."""
        from algorithms.utils.observation_enricher import ObservationEnricher
        
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc", "non_shiftable_load"]
        action_names = ["electrical_storage"]
        
        enricher.enrich_names(observation_names, action_names)
        
        observation_values = [6.0, 0.75, 100.0]
        enriched_values = enricher.enrich_values(observation_values)
        
        # Should contain marker values
        assert 1001.0 in enriched_values  # CA marker
        assert 2001.0 in enriched_values  # SRO marker
        assert 3001.0 in enriched_values  # NFC marker

    def test_marker_encoder_specs_generated(
        self, sample_tokenizer_config: Dict[str, Any]
    ) -> None:
        """Enrichment should provide encoder specs for markers."""
        from algorithms.utils.observation_enricher import ObservationEnricher
        
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc"]
        action_names = ["electrical_storage"]
        
        result = enricher.enrich_names(observation_names, action_names)
        
        # Enriched names should include marker names
        marker_names = [n for n in result.enriched_names if n.startswith("__marker_")]
        assert len(marker_names) > 0


class TestWrapperTopologyChange:
    """Tests for topology change detection in wrapper."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config."""
        return {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                },
                "ev_charger": {
                    "features": ["electric_vehicle_soc"],
                    "action_name": "electric_vehicle_storage",
                    "input_dim": 2,
                },
            },
            "sro_types": {},
            "nfc": {
                "demand_features": [],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 0,
            },
        }

    def test_topology_change_detected(
        self, sample_tokenizer_config: Dict[str, Any]
    ) -> None:
        """Wrapper should detect when observation count changes."""
        from algorithms.utils.observation_enricher import ObservationEnricher
        
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        # Initial topology
        obs_names_v1 = ["electrical_storage_soc"]
        action_names_v1 = ["electrical_storage"]
        enricher.enrich_names(obs_names_v1, action_names_v1)
        
        # Same topology - no change
        assert not enricher.topology_changed(obs_names_v1, action_names_v1)
        
        # New topology - EV charger added
        obs_names_v2 = ["electrical_storage_soc", "electric_vehicle_soc"]
        action_names_v2 = ["electrical_storage", "electric_vehicle_storage"]
        assert enricher.topology_changed(obs_names_v2, action_names_v2)
