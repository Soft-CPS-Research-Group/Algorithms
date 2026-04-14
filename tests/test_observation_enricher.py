"""Tests for ObservationEnricher."""

import pytest

from algorithms.utils.observation_enricher import (
    EnrichmentResult,
    ObservationEnricher,
)


class TestEnrichmentResult:
    """Tests for EnrichmentResult dataclass."""

    def test_enrichment_result_creation(self) -> None:
        """EnrichmentResult should store enriched names and positions."""
        result = EnrichmentResult(
            enriched_names=["__marker__", "feature1", "feature2"],
            marker_positions={"__marker__": [0]},
            marker_to_type={"__marker__": ("ca", "battery", None)},
        )
        
        assert result.enriched_names == ["__marker__", "feature1", "feature2"]
        assert result.marker_positions == {"__marker__": [0]}
        assert result.marker_to_type["__marker__"] == ("ca", "battery", None)


class TestDeviceIdExtraction:
    """Tests for extracting device IDs from action names."""

    def test_extract_single_instance_battery(self) -> None:
        """Single-instance CA has no device ID (None)."""
        from algorithms.utils.observation_enricher import _extract_device_ids
        
        action_names = ["electrical_storage"]
        ca_config = {
            "battery": {
                "features": ["electrical_storage_soc"],
                "action_name": "electrical_storage",
                "input_dim": 1,
            }
        }
        
        result = _extract_device_ids(action_names, ca_config)
        
        assert "battery" in result
        assert result["battery"] == [None]

    def test_extract_multi_instance_ev_chargers(self) -> None:
        """Multi-instance CAs have device IDs extracted from suffix."""
        from algorithms.utils.observation_enricher import _extract_device_ids
        
        action_names = [
            "electrical_storage",
            "electric_vehicle_storage_charger_1_1",
            "electric_vehicle_storage_charger_1_2",
        ]
        ca_config = {
            "battery": {
                "features": ["electrical_storage_soc"],
                "action_name": "electrical_storage",
                "input_dim": 1,
            },
            "ev_charger": {
                "features": ["electric_vehicle_soc"],
                "action_name": "electric_vehicle_storage",
                "input_dim": 61,
            },
        }
        
        result = _extract_device_ids(action_names, ca_config)
        
        assert "battery" in result
        assert result["battery"] == [None]
        assert "ev_charger" in result
        assert result["ev_charger"] == ["charger_1_1", "charger_1_2"]

    def test_extract_no_matching_actions(self) -> None:
        """CA type with no matching actions returns empty list."""
        from algorithms.utils.observation_enricher import _extract_device_ids
        
        action_names = ["electrical_storage"]
        ca_config = {
            "ev_charger": {
                "features": ["electric_vehicle_soc"],
                "action_name": "electric_vehicle_storage",
                "input_dim": 61,
            },
        }
        
        result = _extract_device_ids(action_names, ca_config)
        
        assert "ev_charger" not in result or result.get("ev_charger") == []
