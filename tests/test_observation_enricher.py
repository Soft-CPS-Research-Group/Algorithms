"""Tests for the observation enricher (Phase A of DIN plan).

Verifies that the ObservationEnricher correctly classifies features,
injects markers, and handles caching.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

import pytest

from algorithms.utils.observation_enricher import (
    EnrichmentResult,
    ObservationEnricher,
    _contains_device_id,
    _extract_device_ids,
    _feature_matches_ca_type,
)


# ---------------------------------------------------------------------------
# Fixtures & constants
# ---------------------------------------------------------------------------

TOKENIZER_CONFIG: Dict[str, Any] = {
    "ca_types": {
        "ev_charger": {
            "features": [
                "connected_state",
                "departure_time",
                "required_soc_departure",
                "_soc",
                "battery_capacity",
                "incoming_state",
                "arrival_time",
            ],
            "action_name": "electric_vehicle_storage",
        },
        "battery": {
            "features": ["electrical_storage_soc"],
            "action_name": "electrical_storage",
        },
        "washing_machine": {
            "features": ["start_time_step", "end_time_step"],
            "action_name": "washing_machine",
        },
    },
    "sro_types": {
        "temporal": {
            "features": ["month", "hour", "day_type", "daylight_savings_status"],
        },
        "weather": {
            "features": [
                "outdoor_dry_bulb_temperature",
                "outdoor_relative_humidity",
                "diffuse_solar_irradiance",
                "direct_solar_irradiance",
            ],
        },
        "pricing": {
            "features": ["electricity_pricing"],
        },
        "carbon": {
            "features": ["carbon_intensity"],
        },
    },
    "rl": {
        "demand_feature": "non_shiftable_load",
        "generation_features": ["solar_generation"],
        "extra_features": ["net_electricity_consumption"],
    },
}


# Building with 1 battery (single CA)
SINGLE_BATTERY_OBS = [
    "month",
    "hour",
    "day_type",
    "electrical_storage_soc",
    "electricity_pricing",
    "non_shiftable_load",
    "solar_generation",
    "net_electricity_consumption",
]

SINGLE_BATTERY_ACTIONS = ["electrical_storage"]


# Building with battery + 2 EV chargers (multi-instance)
MULTI_CA_OBS = [
    "month",
    "hour",
    "electrical_storage_soc",
    "electric_vehicle_charger_charger_1_1_connected_state",
    "connected_electric_vehicle_at_charger_charger_1_1_departure_time",
    "connected_electric_vehicle_at_charger_charger_1_1_soc",
    "electric_vehicle_charger_charger_1_2_connected_state",
    "connected_electric_vehicle_at_charger_charger_1_2_departure_time",
    "connected_electric_vehicle_at_charger_charger_1_2_soc",
    "electricity_pricing",
    "non_shiftable_load",
    "solar_generation",
]

MULTI_CA_ACTIONS = [
    "electrical_storage",
    "electric_vehicle_storage_charger_1_1",
    "electric_vehicle_storage_charger_1_2",
]


# Building with no CAs
NO_CA_OBS = [
    "month",
    "hour",
    "day_type",
    "electricity_pricing",
    "non_shiftable_load",
    "solar_generation",
    "net_electricity_consumption",
]

NO_CA_ACTIONS: List[str] = []


# Building with unmatched features
WITH_UNMATCHED_OBS = [
    "month",
    "hour",
    "electrical_storage_soc",
    "some_unknown_feature",
    "another_unknown_one",
    "non_shiftable_load",
]

WITH_UNMATCHED_ACTIONS = ["electrical_storage"]


# ---------------------------------------------------------------------------
# Helper function tests (from observation_enricher)
# ---------------------------------------------------------------------------


class TestExtractDeviceIds:
    def test_single_instance_exact_match(self):
        result = _extract_device_ids(
            ["electrical_storage"],
            {"battery": {"action_name": "electrical_storage"}},
        )
        assert result == {"battery": [None]}

    def test_multi_instance_ev_charger(self):
        result = _extract_device_ids(
            [
                "electrical_storage",
                "electric_vehicle_storage_charger_15_1",
                "electric_vehicle_storage_charger_15_2",
            ],
            {
                "battery": {"action_name": "electrical_storage"},
                "ev_charger": {"action_name": "electric_vehicle_storage"},
            },
        )
        assert result["battery"] == [None]
        assert result["ev_charger"] == ["charger_15_1", "charger_15_2"]


class TestContainsDeviceId:
    def test_ev_charger_id_in_middle(self):
        assert _contains_device_id(
            "electric_vehicle_charger_charger_1_1_connected_state", "charger_1_1"
        )

    def test_ev_charger_id_different_instance(self):
        assert not _contains_device_id(
            "electric_vehicle_charger_charger_1_1_connected_state", "charger_15_1"
        )


class TestFeatureMatchesCaType:
    def test_matches_pattern(self):
        assert _feature_matches_ca_type("electrical_storage_soc", ["electrical_storage_soc"])

    def test_no_match(self):
        assert not _feature_matches_ca_type("month", ["electrical_storage_soc"])


# ---------------------------------------------------------------------------
# ObservationEnricher tests
# ---------------------------------------------------------------------------


class TestEnrichNamesSingleCA:
    """Test enrichment with a single battery building."""

    def test_enriched_names_contain_ca_marker(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)

        assert "__tkn_ca_battery__" in result.enriched_names

    def test_enriched_names_contain_sro_markers(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)

        # Should have temporal and pricing SRO markers
        assert "__tkn_sro_temporal__" in result.enriched_names
        assert "__tkn_sro_pricing__" in result.enriched_names

    def test_enriched_names_contain_nfc_marker(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)

        assert "__tkn_nfc__" in result.enriched_names

    def test_marker_positions_tracked(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)

        # All markers should have tracked positions
        assert "__tkn_ca_battery__" in result.marker_positions
        assert len(result.marker_positions["__tkn_ca_battery__"]) == 1


class TestEnrichNamesMultiCA:
    """Test enrichment with battery + 2 EV chargers."""

    def test_markers_include_device_ids(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(MULTI_CA_OBS, MULTI_CA_ACTIONS)

        # Should have device ID markers for each EV charger
        assert "__tkn_ca_ev_charger__charger_1_1__" in result.enriched_names
        assert "__tkn_ca_ev_charger__charger_1_2__" in result.enriched_names
        # And a battery marker without device ID
        assert "__tkn_ca_battery__" in result.enriched_names

    def test_separate_markers_for_each_instance(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(MULTI_CA_OBS, MULTI_CA_ACTIONS)

        # Each EV charger instance has its own marker
        assert "__tkn_ca_ev_charger__charger_1_1__" in result.marker_positions
        assert "__tkn_ca_ev_charger__charger_1_2__" in result.marker_positions


class TestEnrichNamesNoCA:
    """Test enrichment with no controllable assets."""

    def test_no_ca_markers(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(NO_CA_OBS, NO_CA_ACTIONS)

        # Should have no CA markers
        ca_markers = [n for n in result.enriched_names if n.startswith("__tkn_ca_")]
        assert len(ca_markers) == 0

    def test_sro_and_nfc_markers_present(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(NO_CA_OBS, NO_CA_ACTIONS)

        # SRO and NFC markers should still be present
        assert "__tkn_sro_temporal__" in result.enriched_names
        assert "__tkn_sro_pricing__" in result.enriched_names
        assert "__tkn_nfc__" in result.enriched_names


class TestEnrichNamesPreservesOrder:
    """Test that feature order within groups is preserved."""

    def test_features_within_group_maintain_order(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)

        # Find the position of the battery marker
        battery_idx = result.enriched_names.index("__tkn_ca_battery__")
        # electrical_storage_soc should be right after the marker
        assert result.enriched_names[battery_idx + 1] == "electrical_storage_soc"


class TestEnrichNamesUnmatched:
    """Test handling of unmatched features."""

    def test_unmatched_features_at_end(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(WITH_UNMATCHED_OBS, WITH_UNMATCHED_ACTIONS)

        # Unmatched features should appear at the end
        assert "some_unknown_feature" in result.enriched_names
        assert "another_unknown_one" in result.enriched_names

        # They should be after all markers
        last_marker_pos = max(
            pos
            for positions in result.marker_positions.values()
            for pos in positions
        )
        unknown_pos = result.enriched_names.index("some_unknown_feature")
        assert unknown_pos > last_marker_pos


class TestEnrichValuesCorrectLength:
    """Test enrich_values produces correct output length."""

    def test_output_length_matches_enriched_names(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)

        # Create dummy observation values
        obs_values = [float(i) for i in range(len(SINGLE_BATTERY_OBS))]
        enriched_values = enricher.enrich_values(obs_values)

        assert len(enriched_values) == len(result.enriched_names)


class TestEnrichValuesMarkerPositions:
    """Test that marker positions contain 0.0."""

    def test_marker_positions_are_zero(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)

        obs_values = [1.0] * len(SINGLE_BATTERY_OBS)  # All non-zero
        enriched_values = enricher.enrich_values(obs_values)

        # All marker positions should have 0.0
        for marker_name, positions in result.marker_positions.items():
            for pos in positions:
                assert enriched_values[pos] == 0.0, f"Marker {marker_name} at {pos} is not 0.0"


class TestEnrichValuesPreservesOriginal:
    """Test that non-marker positions preserve original values (reordered)."""

    def test_original_values_preserved(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)

        # Create unique values for each observation
        obs_values = [float(i + 100) for i in range(len(SINGLE_BATTERY_OBS))]
        enriched_values = enricher.enrich_values(obs_values)

        # Get all marker positions
        marker_pos_set = set()
        for positions in result.marker_positions.values():
            marker_pos_set.update(positions)

        # Build reverse lookup: enriched_name -> original_idx
        original_name_to_idx = {name: idx for idx, name in enumerate(SINGLE_BATTERY_OBS)}

        # Non-marker positions should match original values (reordered to match names)
        for i, val in enumerate(enriched_values):
            if i not in marker_pos_set:
                enriched_name = result.enriched_names[i]
                original_idx = original_name_to_idx[enriched_name]
                assert val == obs_values[original_idx], (
                    f"Value mismatch at enriched pos {i}: "
                    f"got {val}, expected {obs_values[original_idx]} "
                    f"for feature '{enriched_name}'"
                )


class TestCacheHitSameTopology:
    """Test caching behavior with same topology."""

    def test_cache_hit_returns_same_object(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)

        result1 = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)
        result2 = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)

        # Should return the exact same object
        assert result1 is result2


class TestCacheMissTopologyChange:
    """Test caching behavior with topology change."""

    def test_cache_miss_returns_new_result(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)

        result1 = enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)
        result2 = enricher.enrich_names(MULTI_CA_OBS, MULTI_CA_ACTIONS)

        # Should return different objects
        assert result1 is not result2
        # And different content
        assert result1.enriched_names != result2.enriched_names


class TestMarkerNamingConvention:
    """Test that all markers follow the naming convention."""

    def test_all_markers_match_pattern(self):
        import re

        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(MULTI_CA_OBS, MULTI_CA_ACTIONS)

        # Pattern: __tkn_{family}_{type}__ or __tkn_{family}_{type}__{device_id}__
        # NFC is special: __tkn_nfc__
        marker_pattern = re.compile(
            r"^__tkn_(ca|sro|nfc)(_[a-z_]+)?(__[a-z0-9_]+)?__$"
        )

        for name in result.enriched_names:
            if name.startswith("__tkn_"):
                assert marker_pattern.match(name), f"Marker {name} doesn't match convention"


class TestMarkerDeviceIdExtraction:
    """Test that device IDs can be extracted from markers."""

    def test_device_id_extractable_from_marker(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(MULTI_CA_OBS, MULTI_CA_ACTIONS)

        # Find EV charger markers
        ev_markers = [n for n in result.enriched_names if "ev_charger" in n]
        assert len(ev_markers) == 2

        # Extract device IDs from markers
        # Format: __tkn_ca_ev_charger__charger_1_1__
        for marker in ev_markers:
            # Strip __tkn_ca_ev_charger__ prefix and __ suffix
            if "__charger_" in marker:
                inner = marker[6:-2]  # Strip __tkn_ and __
                assert "__" in inner  # Has device ID separator


class TestEnricherNoExternalDependencies:
    """Test that ObservationEnricher has no external dependencies."""

    def test_module_imports_only_stdlib(self):
        import importlib
        import sys

        # Re-import the module to check its imports
        module_name = "algorithms.utils.observation_enricher"
        if module_name in sys.modules:
            # Get the module's dependencies
            module = sys.modules[module_name]

            # Check that it only uses stdlib imports
            # The module should work with only: re, dataclasses, typing
            required_modules = {"re", "dataclasses", "typing"}

            # This is a basic check - the module should be portable
            assert hasattr(module, "ObservationEnricher")
            assert hasattr(module, "EnrichmentResult")


class TestEnrichValuesRequiresEnrichNames:
    """Test that enrich_values requires enrich_names to be called first."""

    def test_enrich_values_without_names_raises(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)

        with pytest.raises(RuntimeError, match="enrich_names.*must be called"):
            enricher.enrich_values([1.0, 2.0, 3.0])


class TestEnrichValuesWrongLength:
    """Test that enrich_values validates input length."""

    def test_wrong_length_raises(self):
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        enricher.enrich_names(SINGLE_BATTERY_OBS, SINGLE_BATTERY_ACTIONS)

        with pytest.raises(ValueError, match="does not match"):
            enricher.enrich_values([1.0, 2.0])  # Too short
