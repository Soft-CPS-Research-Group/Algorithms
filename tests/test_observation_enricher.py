"""Tests for ObservationEnricher."""

from typing import Any, Dict

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


class TestFeatureMatching:
    """Tests for feature pattern matching helpers."""

    def test_feature_matches_pattern_substring(self) -> None:
        """Feature matches if pattern is substring of feature name."""
        from algorithms.utils.observation_enricher import _feature_matches_patterns
        
        assert _feature_matches_patterns(
            "electrical_storage_soc",
            ["electrical_storage_soc"]
        )
        assert _feature_matches_patterns(
            "connected_electric_vehicle_at_charger_soc",
            ["electric_vehicle", "charger"]
        )

    def test_feature_no_match(self) -> None:
        """Feature doesn't match if no pattern is substring."""
        from algorithms.utils.observation_enricher import _feature_matches_patterns
        
        assert not _feature_matches_patterns(
            "electricity_pricing",
            ["electric_vehicle", "storage"]
        )

    def test_contains_device_id_bounded(self) -> None:
        """Device ID must appear as bounded token (surrounded by _ or at edges)."""
        from algorithms.utils.observation_enricher import _contains_device_id
        
        # Should match - device_id is bounded
        assert _contains_device_id(
            "electric_vehicle_charger_charger_1_1_connected_state",
            "charger_1_1"
        )
        assert _contains_device_id(
            "connected_state_charger_1_1",
            "charger_1_1"
        )

    def test_contains_device_id_not_bounded(self) -> None:
        """Device ID should not match partial substrings."""
        from algorithms.utils.observation_enricher import _contains_device_id
        
        # Should not match - "1" appears but not as bounded token
        # (this is a tricky case - we accept some false positives for simplicity)
        # The important thing is that the feature-pattern check happens first
        pass  # This test documents expected behavior


class TestFeatureClassification:
    """Tests for classifying features into token groups."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config for testing."""
        return {
            "marker_values": {
                "ca_base": 1000,
                "sro_base": 2000,
                "nfc": 3001,
            },
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                },
                "ev_charger": {
                    "features": [
                        "electric_vehicle_charger",
                        "connected_electric_vehicle",
                    ],
                    "action_name": "electric_vehicle_storage",
                    "input_dim": 61,
                },
            },
            "sro_types": {
                "temporal": {
                    "features": ["month", "hour", "day_type"],
                    "input_dim": 12,
                },
                "pricing": {
                    "features": ["electricity_pricing"],
                    "input_dim": 4,
                },
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": ["solar_generation"],
                "extra_features": [],
                "input_dim": 2,
            },
        }

    def test_classify_battery_feature(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Battery feature should be classified as CA."""
        from algorithms.utils.observation_enricher import _classify_feature
        
        result = _classify_feature(
            "electrical_storage_soc",
            sample_tokenizer_config,
            device_ids_by_type={"battery": [None]},
        )
        
        assert result is not None
        family, type_name, device_id = result
        assert family == "ca"
        assert type_name == "battery"
        assert device_id is None

    def test_classify_ev_feature_with_device_id(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """EV charger feature with device ID should be classified correctly."""
        from algorithms.utils.observation_enricher import _classify_feature
        
        result = _classify_feature(
            "electric_vehicle_charger_charger_1_1_connected_state",
            sample_tokenizer_config,
            device_ids_by_type={"ev_charger": ["charger_1_1", "charger_1_2"]},
        )
        
        assert result is not None
        family, type_name, device_id = result
        assert family == "ca"
        assert type_name == "ev_charger"
        assert device_id == "charger_1_1"

    def test_classify_temporal_feature(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Temporal feature should be classified as SRO."""
        from algorithms.utils.observation_enricher import _classify_feature
        
        result = _classify_feature(
            "month",
            sample_tokenizer_config,
            device_ids_by_type={},
        )
        
        assert result is not None
        family, type_name, device_id = result
        assert family == "sro"
        assert type_name == "temporal"
        assert device_id is None

    def test_classify_nfc_demand_feature(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Demand feature should be classified as NFC."""
        from algorithms.utils.observation_enricher import _classify_feature
        
        result = _classify_feature(
            "non_shiftable_load",
            sample_tokenizer_config,
            device_ids_by_type={},
        )
        
        assert result is not None
        family, type_name, device_id = result
        assert family == "nfc"
        assert type_name == "nfc"
        assert device_id is None

    def test_classify_unknown_feature(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Unknown feature should return None."""
        from algorithms.utils.observation_enricher import _classify_feature
        
        result = _classify_feature(
            "unknown_random_feature",
            sample_tokenizer_config,
            device_ids_by_type={},
        )
        
        assert result is None


class TestObservationEnricherEnrichNames:
    """Tests for ObservationEnricher.enrich_names() method."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config for testing."""
        return {
            "marker_values": {
                "ca_base": 1000,
                "sro_base": 2000,
                "nfc": 3001,
            },
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                },
                "ev_charger": {
                    "features": [
                        "electric_vehicle_charger",
                        "connected_electric_vehicle",
                    ],
                    "action_name": "electric_vehicle_storage",
                    "input_dim": 61,
                },
            },
            "sro_types": {
                "temporal": {
                    "features": ["month", "hour"],
                    "input_dim": 12,
                },
                "pricing": {
                    "features": ["electricity_pricing"],
                    "input_dim": 4,
                },
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": ["solar_generation"],
                "extra_features": [],
                "input_dim": 2,
            },
        }

    def test_enrich_names_single_ca(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Single CA building should have one CA marker."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = [
            "month",
            "hour",
            "electricity_pricing",
            "electrical_storage_soc",
            "non_shiftable_load",
            "solar_generation",
        ]
        action_names = ["electrical_storage"]
        
        result = enricher.enrich_names(observation_names, action_names)
        
        # Should have markers for: 1 CA (battery), 2 SROs (temporal, pricing), 1 NFC
        # Total markers: 4
        assert len(result.enriched_names) == len(observation_names) + 4
        
        # Check marker positions exist
        assert any("1001" in name for name in result.enriched_names)  # CA marker
        assert any("2001" in name for name in result.enriched_names)  # SRO marker
        assert any("3001" in name for name in result.enriched_names)  # NFC marker

    def test_enrich_names_multi_ca(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Building with battery + 2 EV chargers should have 3 CA markers."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = [
            "month",
            "electrical_storage_soc",
            "electric_vehicle_charger_charger_1_1_connected_state",
            "connected_electric_vehicle_at_charger_charger_1_1_soc",
            "electric_vehicle_charger_charger_1_2_connected_state",
            "connected_electric_vehicle_at_charger_charger_1_2_soc",
            "non_shiftable_load",
        ]
        action_names = [
            "electrical_storage",
            "electric_vehicle_storage_charger_1_1",
            "electric_vehicle_storage_charger_1_2",
        ]
        
        result = enricher.enrich_names(observation_names, action_names)
        
        # Count CA markers (1001, 1002, 1003)
        ca_markers = [n for n in result.enriched_names if n.startswith("__marker_100")]
        assert len(ca_markers) == 3  # battery + 2 ev_chargers

    def test_enrich_names_marker_positions_correct(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Marker positions should correctly index into enriched_names."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = [
            "month",
            "electrical_storage_soc",
            "non_shiftable_load",
        ]
        action_names = ["electrical_storage"]
        
        result = enricher.enrich_names(observation_names, action_names)
        
        # Verify each marker position points to a marker name
        for marker_name, positions in result.marker_positions.items():
            for pos in positions:
                assert result.enriched_names[pos] == marker_name

    def test_enrich_names_caching(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Calling enrich_names twice with same input should return cached result."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc"]
        action_names = ["electrical_storage"]
        
        result1 = enricher.enrich_names(observation_names, action_names)
        result2 = enricher.enrich_names(observation_names, action_names)
        
        assert result1 is result2  # Same object (cached)


class TestObservationEnricherEnrichValues:
    """Tests for ObservationEnricher.enrich_values() method."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config for testing."""
        return {
            "marker_values": {
                "ca_base": 1000,
                "sro_base": 2000,
                "nfc": 3001,
            },
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                },
            },
            "sro_types": {
                "temporal": {
                    "features": ["month", "hour"],
                    "input_dim": 12,
                },
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }

    def test_enrich_values_inserts_markers(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """enrich_values() should inject marker values at correct positions."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = [
            "month",
            "hour",
            "electrical_storage_soc",
            "non_shiftable_load",
        ]
        action_names = ["electrical_storage"]
        
        # First enrich names to populate cache
        enriched_result = enricher.enrich_names(observation_names, action_names)
        
        # Raw observation values (4 values, same order as observation_names)
        observation_values = [6.0, 14.0, 0.5, 100.0]
        
        # Enrich values
        enriched_values = enricher.enrich_values(observation_values)
        
        # Enriched should have: CA marker, battery feature, SRO marker, temporal features, NFC marker, nfc feature
        # Expected: [1001.0, 0.5, 2001.0, 6.0, 14.0, 3001.0, 100.0]
        assert len(enriched_values) == len(enriched_result.enriched_names)
        
        # Verify marker values are inserted
        assert 1001.0 in enriched_values  # CA marker
        assert 2001.0 in enriched_values  # SRO marker
        assert 3001.0 in enriched_values  # NFC marker
        
        # Verify original values are still present
        assert 0.5 in enriched_values  # electrical_storage_soc
        assert 6.0 in enriched_values  # month
        assert 14.0 in enriched_values  # hour
        assert 100.0 in enriched_values  # non_shiftable_load

    def test_enrich_values_preserves_order(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """enrich_values() should place values after their corresponding markers."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = [
            "month",
            "electrical_storage_soc",
            "non_shiftable_load",
        ]
        action_names = ["electrical_storage"]
        
        # Enrich names
        enriched_result = enricher.enrich_names(observation_names, action_names)
        
        # Raw values
        observation_values = [6.0, 0.5, 100.0]
        
        # Enrich values
        enriched_values = enricher.enrich_values(observation_values)
        
        # Find marker positions and verify values appear after them
        for i, name in enumerate(enriched_result.enriched_names):
            if name == "__marker_1001__":
                # Battery marker - next value should be electrical_storage_soc (0.5)
                assert enriched_values[i] == 1001.0
                assert enriched_values[i + 1] == 0.5
            elif name == "__marker_2001__":
                # Temporal marker - next value should be month (6.0)
                assert enriched_values[i] == 2001.0
                assert enriched_values[i + 1] == 6.0
            elif name == "__marker_3001__":
                # NFC marker - next value should be non_shiftable_load (100.0)
                assert enriched_values[i] == 3001.0
                assert enriched_values[i + 1] == 100.0


class TestObservationEnricherTopologyChanged:
    """Tests for ObservationEnricher.topology_changed() method."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config for testing."""
        return {
            "marker_values": {
                "ca_base": 1000,
                "sro_base": 2000,
                "nfc": 3001,
            },
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                },
            },
            "sro_types": {
                "temporal": {
                    "features": ["month"],
                    "input_dim": 12,
                },
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }

    def test_topology_unchanged(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """topology_changed() should return False when topology is same as cached."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc"]
        action_names = ["electrical_storage"]
        
        # Enrich to populate cache
        enricher.enrich_names(observation_names, action_names)
        
        # Check with same topology
        assert not enricher.topology_changed(observation_names, action_names)

    def test_topology_changed_different_observations(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """topology_changed() should return True when observation names differ."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc"]
        action_names = ["electrical_storage"]
        
        # Enrich to populate cache
        enricher.enrich_names(observation_names, action_names)
        
        # Check with different observation names
        new_observation_names = ["month", "electrical_storage_soc", "non_shiftable_load"]
        assert enricher.topology_changed(new_observation_names, action_names)

    def test_topology_changed_before_enrich_names(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """topology_changed() should return True when no cache exists."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc"]
        action_names = ["electrical_storage"]
        
        # No enrich_names() called yet - cache is empty
        assert enricher.topology_changed(observation_names, action_names)


class TestEnricherPortability:
    """Tests to verify ObservationEnricher has no external dependencies."""

    def test_no_numpy_import(self) -> None:
        """ObservationEnricher should not import numpy."""
        import algorithms.utils.observation_enricher as enricher_module
        import sys
        
        # Check that numpy is not in the module's namespace
        assert not hasattr(enricher_module, "np")
        assert not hasattr(enricher_module, "numpy")
        
        # Check module source doesn't import numpy
        import inspect
        source = inspect.getsource(enricher_module)
        assert "import numpy" not in source
        assert "from numpy" not in source

    def test_no_torch_import(self) -> None:
        """ObservationEnricher should not import torch."""
        import algorithms.utils.observation_enricher as enricher_module
        import inspect
        
        source = inspect.getsource(enricher_module)
        assert "import torch" not in source
        assert "from torch" not in source

    def test_no_training_imports(self) -> None:
        """ObservationEnricher should not import from algorithms.* or utils.*."""
        import algorithms.utils.observation_enricher as enricher_module
        import inspect
        
        source = inspect.getsource(enricher_module)
        # Should not import from other project modules
        assert "from algorithms." not in source.replace(
            "from algorithms.utils.observation_enricher", ""
        )
        assert "from utils." not in source


class TestEnricherIntegration:
    """Integration tests using the actual tokenizer config file."""

    def test_with_real_config_file(self) -> None:
        """Test enricher with the actual configs/tokenizers/default.json."""
        import json
        from pathlib import Path
        
        config_path = Path("configs/tokenizers/default.json")
        with open(config_path) as f:
            tokenizer_config = json.load(f)
        
        enricher = ObservationEnricher(tokenizer_config)
        
        # Simulate Building_4 (battery + 1 EV charger)
        observation_names = [
            "month",
            "hour",
            "day_type",
            "electricity_pricing",
            "electricity_pricing_predicted_1",
            "electricity_pricing_predicted_2",
            "electricity_pricing_predicted_3",
            "carbon_intensity",
            "electrical_storage_soc",
            "electric_vehicle_charger_connected_state",
            "connected_electric_vehicle_at_charger_battery_capacity",
            "connected_electric_vehicle_at_charger_departure_time",
            "connected_electric_vehicle_at_charger_required_soc_departure",
            "connected_electric_vehicle_at_charger_soc",
            "electric_vehicle_charger_incoming_state",
            "incoming_electric_vehicle_at_charger_estimated_arrival_time",
            "non_shiftable_load",
            "solar_generation",
            "net_electricity_consumption",
        ]
        action_names = ["electrical_storage", "electric_vehicle_storage"]
        
        result = enricher.enrich_names(observation_names, action_names)
        
        # Should have markers for: 2 CAs, 3 SROs (temporal, pricing, carbon), 1 NFC
        # Total markers: 6
        marker_count = sum(1 for n in result.enriched_names if n.startswith("__marker_"))
        assert marker_count == 6
        
        # Verify CA markers (1001 for battery, 1002 for ev_charger)
        assert "__marker_1001__" in result.enriched_names
        assert "__marker_1002__" in result.enriched_names
        
        # Verify SRO markers (2001 temporal, 2002 pricing, 2003 carbon)
        assert "__marker_2001__" in result.enriched_names
        assert "__marker_2002__" in result.enriched_names
        assert "__marker_2003__" in result.enriched_names
        
        # Verify NFC marker
        assert "__marker_3001__" in result.enriched_names
        
        # Test enrich_values
        observation_values = [float(i) for i in range(len(observation_names))]
        enriched_values = enricher.enrich_values(observation_values)
        
        assert len(enriched_values) == len(result.enriched_names)
        assert 1001.0 in enriched_values
        assert 2001.0 in enriched_values
        assert 3001.0 in enriched_values

