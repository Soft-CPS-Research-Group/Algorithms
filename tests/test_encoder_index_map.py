"""Tests for the encoder index map utility (Phase 0).

Verifies that build_encoder_index_map correctly maps raw observation names
to their post-encoding slice indices using the same matching logic as the
wrapper's set_encoders pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithms.utils.encoder_index_map import (
    EncoderSlice,
    build_encoder_index_map,
    _compute_encoded_dims,
    _matches_rule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ENCODER_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "encoders" / "default.json"
)


@pytest.fixture(scope="module")
def encoder_config() -> dict:
    """Load the real encoder configuration from disk."""
    with ENCODER_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Observation name sets for realistic test scenarios
# ---------------------------------------------------------------------------


# Typical building from citylearn_challenge_2022 with 1 EV charger + battery + PV
BUILDING_1_OBS = [
    "month",
    "hour",
    "day_type",
    "daylight_savings_status",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_6h",
    "outdoor_dry_bulb_temperature_predicted_12h",
    "outdoor_dry_bulb_temperature_predicted_24h",
    "outdoor_relative_humidity",
    "outdoor_relative_humidity_predicted_6h",
    "outdoor_relative_humidity_predicted_12h",
    "outdoor_relative_humidity_predicted_24h",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_6h",
    "diffuse_solar_irradiance_predicted_12h",
    "diffuse_solar_irradiance_predicted_24h",
    "electricity_pricing",
    "electricity_pricing_predicted_6h",
    "electricity_pricing_predicted_12h",
    "electricity_pricing_predicted_24h",
    "carbon_intensity",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "electric_vehicle_soc",
    "electric_vehicle_charger_connected_state",
    "electric_vehicle_departure_time",
    "electric_vehicle_required_soc_departure",
    "electric_vehicle_battery_capacity",
    "electric_vehicle_incoming_state",
    "electric_vehicle_arrival_time",
]


# Building_15: 2 EV chargers (multi-instance)
BUILDING_15_PARTIAL_OBS = [
    "month",
    "hour",
    "day_type",
    "electricity_pricing",
    "electric_vehicle_soc_charger_15_1",
    "electric_vehicle_charger_connected_state_charger_15_1",
    "electric_vehicle_departure_time_charger_15_1",
    "electric_vehicle_soc_charger_15_2",
    "electric_vehicle_charger_connected_state_charger_15_2",
    "electric_vehicle_departure_time_charger_15_2",
]


# ---------------------------------------------------------------------------
# _matches_rule tests
# ---------------------------------------------------------------------------


class TestMatchesRule:
    def test_equals_match(self):
        assert _matches_rule("month", {"equals": ["month", "hour"]})

    def test_equals_no_match(self):
        assert not _matches_rule("year", {"equals": ["month", "hour"]})

    def test_contains_match(self):
        assert _matches_rule(
            "electric_vehicle_charger_connected_state_charger_15_1",
            {"contains": ["connected_state"]},
        )

    def test_contains_no_match(self):
        assert not _matches_rule("month", {"contains": ["connected_state"]})

    def test_prefixes_match(self):
        assert _matches_rule("electric_vehicle_soc", {"prefixes": ["electric_vehicle"]})

    def test_suffixes_match(self):
        assert _matches_rule("electricity_pricing_predicted_6h", {"suffixes": ["_6h"]})

    def test_default_match(self):
        assert _matches_rule("anything", {"default": True})

    def test_default_false(self):
        assert not _matches_rule("anything", {"default": False})

    def test_empty_spec_no_match(self):
        assert not _matches_rule("anything", {})


# ---------------------------------------------------------------------------
# _compute_encoded_dims tests
# ---------------------------------------------------------------------------


class TestComputeEncodedDims:
    def test_periodic_normalization(self):
        assert _compute_encoded_dims({"type": "PeriodicNormalization"}) == 2

    def test_onehot_encoding(self):
        spec = {"type": "OnehotEncoding", "params": {"classes": [0, 1]}}
        assert _compute_encoded_dims(spec) == 2

    def test_onehot_encoding_8_classes(self):
        spec = {"type": "OnehotEncoding", "params": {"classes": [1, 2, 3, 4, 5, 6, 7, 8]}}
        assert _compute_encoded_dims(spec) == 8

    def test_onehot_encoding_27_classes(self):
        classes = [-1, -0.1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        spec = {"type": "OnehotEncoding", "params": {"classes": classes}}
        assert _compute_encoded_dims(spec) == 27

    def test_remove_feature(self):
        assert _compute_encoded_dims({"type": "RemoveFeature"}) == 0

    def test_no_normalization(self):
        assert _compute_encoded_dims({"type": "NoNormalization"}) == 1

    def test_normalize(self):
        assert _compute_encoded_dims({"type": "Normalize"}) == 1

    def test_normalize_with_missing(self):
        assert _compute_encoded_dims({"type": "NormalizeWithMissing"}) == 1


# ---------------------------------------------------------------------------
# build_encoder_index_map tests
# ---------------------------------------------------------------------------


class TestBuildEncoderIndexMap:
    """Test the main public API with the real encoder configuration."""

    def test_empty_rules_raises(self):
        with pytest.raises(ValueError, match="at least one rule"):
            build_encoder_index_map(["month"], {"rules": []})

    def test_no_matching_rule_raises(self):
        config = {"rules": [{"match": {"equals": ["hour"]}, "encoder": {"type": "NoNormalization"}}]}
        with pytest.raises(ValueError, match="No encoder rule matches"):
            build_encoder_index_map(["unknown_feature"], config)

    def test_single_periodic_feature(self, encoder_config):
        index_map = build_encoder_index_map(["month"], encoder_config)
        assert index_map["month"] == EncoderSlice(start_idx=0, end_idx=2, n_dims=2)

    def test_remove_feature_produces_zero_dims(self, encoder_config):
        index_map = build_encoder_index_map(
            ["outdoor_dry_bulb_temperature"], encoder_config,
        )
        assert index_map["outdoor_dry_bulb_temperature"].n_dims == 0
        assert index_map["outdoor_dry_bulb_temperature"].start_idx == 0
        assert index_map["outdoor_dry_bulb_temperature"].end_idx == 0

    def test_weather_features_all_zero_dims(self, encoder_config):
        """All weather features must produce 0 post-encoding dims."""
        weather_names = [
            "outdoor_dry_bulb_temperature",
            "outdoor_dry_bulb_temperature_predicted_6h",
            "outdoor_dry_bulb_temperature_predicted_12h",
            "outdoor_dry_bulb_temperature_predicted_24h",
            "outdoor_relative_humidity",
            "outdoor_relative_humidity_predicted_6h",
            "outdoor_relative_humidity_predicted_12h",
            "outdoor_relative_humidity_predicted_24h",
            "diffuse_solar_irradiance",
            "diffuse_solar_irradiance_predicted_6h",
            "diffuse_solar_irradiance_predicted_12h",
            "diffuse_solar_irradiance_predicted_24h",
        ]
        index_map = build_encoder_index_map(weather_names, encoder_config)
        for name in weather_names:
            assert index_map[name].n_dims == 0, f"Expected 0 dims for {name}"
        # Total encoded dims should be 0
        total_dims = sum(s.n_dims for s in index_map.values())
        assert total_dims == 0

    def test_periodic_features_produce_2_dims(self, encoder_config):
        """month and hour must each produce 2 post-encoding dims."""
        index_map = build_encoder_index_map(["month", "hour"], encoder_config)
        assert index_map["month"].n_dims == 2
        assert index_map["hour"].n_dims == 2
        # month: [0, 2), hour: [2, 4)
        assert index_map["month"] == EncoderSlice(0, 2, 2)
        assert index_map["hour"] == EncoderSlice(2, 4, 2)

    def test_day_type_onehot_8_classes(self, encoder_config):
        index_map = build_encoder_index_map(["day_type"], encoder_config)
        assert index_map["day_type"].n_dims == 8

    def test_departure_time_onehot_27_classes(self, encoder_config):
        """departure_time uses OnehotEncoding with 27 classes."""
        index_map = build_encoder_index_map(
            ["electric_vehicle_departure_time"], encoder_config,
        )
        assert index_map["electric_vehicle_departure_time"].n_dims == 27

    def test_arrival_time_onehot_27_classes(self, encoder_config):
        index_map = build_encoder_index_map(
            ["electric_vehicle_arrival_time"], encoder_config,
        )
        assert index_map["electric_vehicle_arrival_time"].n_dims == 27

    def test_connected_state_onehot_2_classes(self, encoder_config):
        index_map = build_encoder_index_map(
            ["electric_vehicle_charger_connected_state"], encoder_config,
        )
        assert index_map["electric_vehicle_charger_connected_state"].n_dims == 2

    def test_no_normalization_features(self, encoder_config):
        """Features that hit the default rule produce 1 dim each."""
        index_map = build_encoder_index_map(
            ["electricity_pricing", "non_shiftable_load", "solar_generation"],
            encoder_config,
        )
        for name in ["electricity_pricing", "non_shiftable_load", "solar_generation"]:
            assert index_map[name].n_dims == 1

    def test_normalize_with_missing_features(self, encoder_config):
        """EV SOC and required_soc_departure use NormalizeWithMissing → 1 dim."""
        index_map = build_encoder_index_map(
            ["electric_vehicle_soc", "electric_vehicle_required_soc_departure"],
            encoder_config,
        )
        assert index_map["electric_vehicle_soc"].n_dims == 1
        assert index_map["electric_vehicle_required_soc_departure"].n_dims == 1

    def test_contiguous_slices(self, encoder_config):
        """Slices must be contiguous: each start_idx == previous end_idx."""
        index_map = build_encoder_index_map(BUILDING_1_OBS, encoder_config)
        slices = list(index_map.values())
        for i in range(1, len(slices)):
            assert slices[i].start_idx == slices[i - 1].end_idx, (
                f"Gap between {list(index_map.keys())[i-1]} and "
                f"{list(index_map.keys())[i]}: {slices[i-1]} → {slices[i]}"
            )

    def test_total_dims_building_1(self, encoder_config):
        """Verify total post-encoding dims for a realistic building.

        Building_1 features breakdown:
        - month (2) + hour (2) + day_type (8) + daylight_savings_status (2) = 14
        - 12 weather features × 0 = 0
        - electricity_pricing ×4 (4) + carbon_intensity (1) = 5
        - non_shiftable_load (1) + solar_generation (1) = 2
        - electrical_storage_soc (1) = 1
        - electric_vehicle_soc (1) + connected_state (2) + departure_time (27)
          + required_soc_departure (1) + battery_capacity (1) + incoming_state (2)
          + arrival_time (27) = 61
        Total: 14 + 0 + 5 + 2 + 1 + 61 = 83
        """
        index_map = build_encoder_index_map(BUILDING_1_OBS, encoder_config)
        total = sum(s.n_dims for s in index_map.values())
        assert total == 83

    def test_suffixed_features_match_contains_rules(self, encoder_config):
        """Multi-instance features with device suffixes must still match.

        e.g. 'electric_vehicle_soc_charger_15_1' contains 'electric_vehicle_soc'
        and should match the NormalizeWithMissing rule.
        """
        index_map = build_encoder_index_map(BUILDING_15_PARTIAL_OBS, encoder_config)
        # Suffixed SOC features should get 1 dim (NormalizeWithMissing)
        assert index_map["electric_vehicle_soc_charger_15_1"].n_dims == 1
        assert index_map["electric_vehicle_soc_charger_15_2"].n_dims == 1
        # Suffixed connected_state should get 2 dims (OnehotEncoding [0, 1])
        assert index_map["electric_vehicle_charger_connected_state_charger_15_1"].n_dims == 2
        assert index_map["electric_vehicle_charger_connected_state_charger_15_2"].n_dims == 2
        # Suffixed departure_time should get 27 dims
        assert index_map["electric_vehicle_departure_time_charger_15_1"].n_dims == 27
        assert index_map["electric_vehicle_departure_time_charger_15_2"].n_dims == 27

    def test_ordering_preserved(self, encoder_config):
        """The OrderedDict preserves insertion order (= observation_names order)."""
        names = ["hour", "month", "day_type"]
        index_map = build_encoder_index_map(names, encoder_config)
        assert list(index_map.keys()) == names

    def test_building_15_partial_total_dims(self, encoder_config):
        """Verify partial Building_15 observations.

        month (2) + hour (2) + day_type (8) + electricity_pricing (1)
        + soc_1 (1) + connected_1 (2) + departure_1 (27)
        + soc_2 (1) + connected_2 (2) + departure_2 (27)
        = 73
        """
        index_map = build_encoder_index_map(BUILDING_15_PARTIAL_OBS, encoder_config)
        total = sum(s.n_dims for s in index_map.values())
        assert total == 73

    def test_daylight_savings_status(self, encoder_config):
        """daylight_savings_status uses OnehotEncoding [0, 1] → 2 dims."""
        index_map = build_encoder_index_map(["daylight_savings_status"], encoder_config)
        assert index_map["daylight_savings_status"].n_dims == 2

    def test_electrical_storage_soc(self, encoder_config):
        """electrical_storage_soc → NormalizeWithMissing → 1 dim."""
        # electrical_storage_soc does NOT contain the substring tokens for
        # NormalizeWithMissing (required_soc_departure, estimated_soc_arrival, electric_vehicle_soc).
        # It falls through to the default rule → NoNormalization → 1 dim.
        index_map = build_encoder_index_map(["electrical_storage_soc"], encoder_config)
        assert index_map["electrical_storage_soc"].n_dims == 1
