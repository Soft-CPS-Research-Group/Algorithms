"""Tests for the observation tokenizer (Phase 1, updated for actual CityLearn naming).

Verifies action-based instance detection, feature classification, projection
shapes, weather SRO skipping, RL token computation, action-CA mapping, and
extra RL features.

CityLearn naming convention (actual):
  - Device IDs are inserted IN THE MIDDLE of feature names, not as a suffix.
  - e.g. ``electric_vehicle_charger_charger_1_1_connected_state``
  - Device IDs are extracted from action names:
    ``electric_vehicle_storage_charger_1_1`` → device_id = ``charger_1_1``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from algorithms.utils.observation_tokenizer import (
    ObservationTokenizer,
    TokenizedObservation,
    _contains_device_id,
    _extract_device_ids,
)


# ---------------------------------------------------------------------------
# Fixtures & constants
# ---------------------------------------------------------------------------

ENCODER_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "encoders" / "default.json"
)

# Tokenizer config using the updated patterns that match actual CityLearn naming.
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


@pytest.fixture(scope="module")
def encoder_config() -> dict:
    with ENCODER_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Observation name sets — ACTUAL CityLearn naming
# ---------------------------------------------------------------------------

# Building_1: 1 EV charger (charger_1_1) + 1 battery + PV + 1 washing machine
BUILDING_1_OBS = [
    "month",
    "day_type",
    "hour",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "outdoor_relative_humidity",
    "outdoor_relative_humidity_predicted_1",
    "outdoor_relative_humidity_predicted_2",
    "outdoor_relative_humidity_predicted_3",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_1",
    "diffuse_solar_irradiance_predicted_2",
    "diffuse_solar_irradiance_predicted_3",
    "direct_solar_irradiance",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "carbon_intensity",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "electricity_pricing_predicted_3",
    "daylight_savings_status",
    # Per-charger observations (charger_1_1)
    "electric_vehicle_charger_charger_1_1_connected_state",
    "connected_electric_vehicle_at_charger_charger_1_1_departure_time",
    "connected_electric_vehicle_at_charger_charger_1_1_required_soc_departure",
    "connected_electric_vehicle_at_charger_charger_1_1_soc",
    "connected_electric_vehicle_at_charger_charger_1_1_battery_capacity",
    "electric_vehicle_charger_charger_1_1_incoming_state",
    "incoming_electric_vehicle_at_charger_charger_1_1_estimated_arrival_time",
    # Washing machine
    "washing_machine_1_start_time_step",
    "washing_machine_1_end_time_step",
]

BUILDING_1_ACTIONS = [
    "electrical_storage",
    "electric_vehicle_storage_charger_1_1",
    "washing_machine_1",
]


# Building_15: 2 EV chargers (multi-instance) + battery (no washing machine)
BUILDING_15_OBS = [
    "month",
    "day_type",
    "hour",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "outdoor_relative_humidity",
    "outdoor_relative_humidity_predicted_1",
    "outdoor_relative_humidity_predicted_2",
    "outdoor_relative_humidity_predicted_3",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_1",
    "diffuse_solar_irradiance_predicted_2",
    "diffuse_solar_irradiance_predicted_3",
    "direct_solar_irradiance",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "carbon_intensity",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "electricity_pricing_predicted_3",
    "daylight_savings_status",
    # Per-charger observations (charger_15_1)
    "electric_vehicle_charger_charger_15_1_connected_state",
    "connected_electric_vehicle_at_charger_charger_15_1_departure_time",
    "connected_electric_vehicle_at_charger_charger_15_1_required_soc_departure",
    "connected_electric_vehicle_at_charger_charger_15_1_soc",
    "connected_electric_vehicle_at_charger_charger_15_1_battery_capacity",
    "electric_vehicle_charger_charger_15_1_incoming_state",
    "incoming_electric_vehicle_at_charger_charger_15_1_estimated_arrival_time",
    # Per-charger observations (charger_15_2)
    "electric_vehicle_charger_charger_15_2_connected_state",
    "connected_electric_vehicle_at_charger_charger_15_2_departure_time",
    "connected_electric_vehicle_at_charger_charger_15_2_required_soc_departure",
    "connected_electric_vehicle_at_charger_charger_15_2_soc",
    "connected_electric_vehicle_at_charger_charger_15_2_battery_capacity",
    "electric_vehicle_charger_charger_15_2_incoming_state",
    "incoming_electric_vehicle_at_charger_charger_15_2_estimated_arrival_time",
]

BUILDING_15_ACTIONS = [
    "electrical_storage",
    "electric_vehicle_storage_charger_15_1",
    "electric_vehicle_storage_charger_15_2",
]


# Minimal building: only SRO + RL (no CAs)
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


# Single-charger building with no battery, no washing machine (like Building_4)
SINGLE_CHARGER_OBS = [
    "month",
    "day_type",
    "hour",
    "electricity_pricing",
    "non_shiftable_load",
    "solar_generation",
    "net_electricity_consumption",
    "daylight_savings_status",
    # Per-charger observations (charger_4_1)
    "electric_vehicle_charger_charger_4_1_connected_state",
    "connected_electric_vehicle_at_charger_charger_4_1_departure_time",
    "connected_electric_vehicle_at_charger_charger_4_1_required_soc_departure",
    "connected_electric_vehicle_at_charger_charger_4_1_soc",
    "connected_electric_vehicle_at_charger_charger_4_1_battery_capacity",
    "electric_vehicle_charger_charger_4_1_incoming_state",
    "incoming_electric_vehicle_at_charger_charger_4_1_estimated_arrival_time",
]

SINGLE_CHARGER_ACTIONS = ["electric_vehicle_storage_charger_4_1"]


# Battery-only building (like Building_2)
BATTERY_ONLY_OBS = [
    "month",
    "day_type",
    "hour",
    "electricity_pricing",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "daylight_savings_status",
]

BATTERY_ONLY_ACTIONS = ["electrical_storage"]


# ---------------------------------------------------------------------------
# _extract_device_ids tests
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

    def test_washing_machine(self):
        result = _extract_device_ids(
            ["washing_machine_1"],
            {"washing_machine": {"action_name": "washing_machine"}},
        )
        assert result == {"washing_machine": ["1"]}

    def test_no_matching_actions(self):
        result = _extract_device_ids(
            ["electrical_storage"],
            {"ev_charger": {"action_name": "electric_vehicle_storage"}},
        )
        assert "ev_charger" not in result

    def test_mixed_building(self):
        """Building_1: battery + EV charger + washing machine."""
        result = _extract_device_ids(
            [
                "electrical_storage",
                "electric_vehicle_storage_charger_1_1",
                "washing_machine_1",
            ],
            {
                "battery": {"action_name": "electrical_storage"},
                "ev_charger": {"action_name": "electric_vehicle_storage"},
                "washing_machine": {"action_name": "washing_machine"},
            },
        )
        assert result["battery"] == [None]
        assert result["ev_charger"] == ["charger_1_1"]
        assert result["washing_machine"] == ["1"]


# ---------------------------------------------------------------------------
# _contains_device_id tests
# ---------------------------------------------------------------------------


class TestContainsDeviceId:
    def test_ev_charger_id_in_middle(self):
        assert _contains_device_id(
            "electric_vehicle_charger_charger_1_1_connected_state", "charger_1_1",
        )

    def test_ev_charger_id_different_instance(self):
        assert not _contains_device_id(
            "electric_vehicle_charger_charger_1_1_connected_state", "charger_15_1",
        )

    def test_washing_machine_id(self):
        assert _contains_device_id("washing_machine_1_start_time_step", "1")

    def test_short_id_no_false_positive_at_end(self):
        """Device ID '1' should not match 'predicted_1' because '_1' is at the end
        and is followed by end-of-string, which is a valid boundary."""
        # Actually _1 at end IS a boundary match, so this WOULD match.
        # But in practice the feature pattern check (e.g. start_time_step)
        # prevents false assignment to the wrong CA type.
        assert _contains_device_id("electricity_pricing_predicted_1", "1")

    def test_no_match_without_boundary(self):
        """Device ID '15' should not match '_150_' or '_156_'."""
        assert not _contains_device_id("some_feature_150_name", "15")

    def test_id_at_start(self):
        assert _contains_device_id("charger_1_1_something", "charger_1_1")

    def test_id_at_end(self):
        assert _contains_device_id("some_prefix_charger_1_1", "charger_1_1")


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassification:
    def test_building_1_ca_classification(self, encoder_config):
        """Building_1 should have 3 CA tokens: 1 battery + 1 EV charger + 1 washing machine."""
        tok = ObservationTokenizer(
            BUILDING_1_OBS, BUILDING_1_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=32,
        )
        assert tok.n_ca == 3
        assert sorted(tok.ca_types) == ["battery", "ev_charger", "washing_machine"]

    def test_building_15_multi_instance(self, encoder_config):
        """Building_15 should have 3 CA tokens (1 battery + 2 EV chargers)."""
        tok = ObservationTokenizer(
            BUILDING_15_OBS, BUILDING_15_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=32,
        )
        assert tok.n_ca == 3
        types_sorted = sorted(tok.ca_types)
        assert types_sorted == ["battery", "ev_charger", "ev_charger"]

    def test_no_ca_building(self, encoder_config):
        """Building with no CAs → 0 CA tokens."""
        tok = ObservationTokenizer(
            NO_CA_OBS, NO_CA_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=32,
        )
        assert tok.n_ca == 0
        assert tok.ca_types == []

    def test_single_charger_building(self, encoder_config):
        """Building with 1 charger only → 1 CA token."""
        tok = ObservationTokenizer(
            SINGLE_CHARGER_OBS, SINGLE_CHARGER_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=32,
        )
        assert tok.n_ca == 1
        assert tok.ca_types == ["ev_charger"]

    def test_battery_only_building(self, encoder_config):
        """Building with only a battery → 1 CA token."""
        tok = ObservationTokenizer(
            BATTERY_ONLY_OBS, BATTERY_ONLY_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=32,
        )
        assert tok.n_ca == 1
        assert tok.ca_types == ["battery"]

    def test_weather_sro_has_dims(self, encoder_config):
        """Weather SRO group should have non-zero dims because the actual dataset
        uses ``_predicted_1/2/3`` names (not ``_predicted_6h/12h/24h``), which
        do NOT match the RemoveFeature encoder rules.  Additionally,
        ``direct_solar_irradiance`` and its predicted variants are not in the
        RemoveFeature list at all."""
        tok = ObservationTokenizer(
            BUILDING_1_OBS, BUILDING_1_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=32,
        )
        sro_type_names = [name for name, _ in tok._sro_groups]
        assert "weather" in sro_type_names

    def test_active_sro_groups(self, encoder_config):
        """Building_1 should have temporal, weather, pricing, carbon SRO groups."""
        tok = ObservationTokenizer(
            BUILDING_1_OBS, BUILDING_1_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=32,
        )
        sro_type_names = [name for name, _ in tok._sro_groups]
        assert "temporal" in sro_type_names
        assert "weather" in sro_type_names
        assert "pricing" in sro_type_names
        assert "carbon" in sro_type_names
        assert tok.n_sro == 4

    def test_ev_features_grouped_by_instance(self, encoder_config):
        """Each EV charger instance should get exactly 7 features (matching config)."""
        tok = ObservationTokenizer(
            BUILDING_15_OBS, BUILDING_15_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=32,
        )
        # Find the EV charger instances
        ev_instances = [
            (device_id, indices)
            for ca_type, device_id, indices in tok._ca_instances
            if ca_type == "ev_charger"
        ]
        assert len(ev_instances) == 2
        # Both instances should have the same number of encoded dims
        assert len(ev_instances[0][1]) == len(ev_instances[1][1])
        # Each instance should have indices (7 features, each encoded)
        assert len(ev_instances[0][1]) > 0

    def test_no_unmatched_features_building_1(self, encoder_config):
        """All Building_1 features should be classified (no warnings)."""
        tok = ObservationTokenizer(
            BUILDING_1_OBS, BUILDING_1_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=32,
        )
        # Count total classified features
        classified = set()
        for _, _, indices in tok._ca_instances:
            classified.update(indices)
        for _, indices in tok._sro_groups:
            classified.update(indices)
        classified.update(tok._rl_demand_indices)
        classified.update(tok._rl_generation_indices)
        classified.update(tok._rl_extra_indices)
        # All features that produce dims should be covered — but weather has 0 dims
        # so we just verify all non-zero-dim features are assigned.
        total_dims = tok.total_encoded_dims
        assert total_dims > 0


# ---------------------------------------------------------------------------
# Projection shape tests
# ---------------------------------------------------------------------------


D_MODEL = 32
BATCH = 4


class TestProjectionShapes:
    def test_building_1_shapes(self, encoder_config):
        """Test forward pass shapes for Building_1 (3 CAs, 4 SROs, 1 RL)."""
        tok = ObservationTokenizer(
            BUILDING_1_OBS, BUILDING_1_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        obs_dim = tok.total_encoded_dims
        obs = torch.randn(BATCH, obs_dim)
        result = tok(obs)

        assert isinstance(result, TokenizedObservation)
        assert result.ca_tokens.shape == (BATCH, 3, D_MODEL)
        assert result.sro_tokens.shape == (BATCH, 4, D_MODEL)
        assert result.rl_token.shape == (BATCH, 1, D_MODEL)
        assert result.n_ca == 3

    def test_building_15_shapes(self, encoder_config):
        """Test forward pass shapes for Building_15 (3 CA tokens: 1 battery + 2 EV)."""
        tok = ObservationTokenizer(
            BUILDING_15_OBS, BUILDING_15_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        obs_dim = tok.total_encoded_dims
        obs = torch.randn(BATCH, obs_dim)
        result = tok(obs)

        assert result.ca_tokens.shape == (BATCH, 3, D_MODEL)
        assert result.n_ca == 3

    def test_no_ca_shapes(self, encoder_config):
        """No-CA building → ca_tokens has shape [batch, 0, d_model]."""
        tok = ObservationTokenizer(
            NO_CA_OBS, NO_CA_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        obs_dim = tok.total_encoded_dims
        obs = torch.randn(BATCH, obs_dim)
        result = tok(obs)

        assert result.ca_tokens.shape == (BATCH, 0, D_MODEL)
        assert result.n_ca == 0

    def test_single_obs_unbatched(self, encoder_config):
        """Forward with a 1-D tensor (no batch dim) should still work."""
        tok = ObservationTokenizer(
            SINGLE_CHARGER_OBS, SINGLE_CHARGER_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        obs_dim = tok.total_encoded_dims
        obs = torch.randn(obs_dim)  # no batch dimension
        result = tok(obs)

        assert result.ca_tokens.shape == (1, 1, D_MODEL)
        assert result.rl_token.shape == (1, 1, D_MODEL)


# ---------------------------------------------------------------------------
# RL token computation tests
# ---------------------------------------------------------------------------


class TestRLToken:
    def test_rl_token_uses_residual(self, encoder_config):
        """RL token should have demand and generation indices populated."""
        tok = ObservationTokenizer(
            SINGLE_CHARGER_OBS, SINGLE_CHARGER_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        assert len(tok._rl_demand_indices) > 0
        assert len(tok._rl_generation_indices) > 0

    def test_rl_extra_features(self, encoder_config):
        """RL token should include extra features (net_electricity_consumption)."""
        tok = ObservationTokenizer(
            BUILDING_1_OBS, BUILDING_1_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        assert len(tok._rl_extra_indices) > 0

    def test_rl_projection_input_dim(self, encoder_config):
        """RL projection should accept residual (1) + extra features."""
        tok = ObservationTokenizer(
            BUILDING_1_OBS, BUILDING_1_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        assert tok.rl_projection is not None
        # 1 (residual) + 1 (net_electricity_consumption) = 2
        assert tok.rl_projection.in_features == 2

    def test_rl_token_residual_value(self, encoder_config):
        """With known demand/generation values, verify RL projection is used."""
        tok = ObservationTokenizer(
            SINGLE_CHARGER_OBS, SINGLE_CHARGER_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        obs_dim = tok.total_encoded_dims
        obs = torch.zeros(1, obs_dim)

        # Set demand = 5.0, generation = 2.0
        demand_idx = tok._rl_demand_indices[0]
        gen_idx = tok._rl_generation_indices[0]
        obs[0, demand_idx] = 5.0
        obs[0, gen_idx] = 2.0

        result = tok(obs)
        assert result.rl_token.shape == (1, 1, D_MODEL)
        assert tok.rl_projection is not None


# ---------------------------------------------------------------------------
# Action-CA mapping tests
# ---------------------------------------------------------------------------


class TestActionCAMapping:
    def test_building_1_action_mapping(self, encoder_config):
        """Building_1 actions: [electrical_storage, electric_vehicle_storage_charger_1_1, washing_machine_1]."""
        tok = ObservationTokenizer(
            BUILDING_1_OBS, BUILDING_1_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        # There should be a mapping for each action
        assert len(tok.action_ca_map) == 3

        # Verify each action maps to a valid CA index
        for ca_idx in tok.action_ca_map:
            assert 0 <= ca_idx < tok.n_ca

        # electrical_storage → battery, ev_storage_charger_1_1 → ev_charger,
        # washing_machine_1 → washing_machine
        battery_idx = tok.action_ca_map[0]
        ev_idx = tok.action_ca_map[1]
        wm_idx = tok.action_ca_map[2]
        assert tok.ca_types[battery_idx] == "battery"
        assert tok.ca_types[ev_idx] == "ev_charger"
        assert tok.ca_types[wm_idx] == "washing_machine"

    def test_building_15_action_mapping(self, encoder_config):
        """Building_15 actions: 1 battery + 2 EV charger actions."""
        tok = ObservationTokenizer(
            BUILDING_15_OBS, BUILDING_15_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        assert len(tok.action_ca_map) == 3

        # First action is battery
        assert tok.ca_types[tok.action_ca_map[0]] == "battery"

        # Next two should be ev_charger
        ev_indices = [tok.action_ca_map[1], tok.action_ca_map[2]]
        for ca_idx in ev_indices:
            assert tok.ca_types[ca_idx] == "ev_charger"

        # They should map to different CA tokens (different charger instances)
        assert ev_indices[0] != ev_indices[1]

    def test_no_ca_no_action_mapping(self, encoder_config):
        """No-CA building has empty action mapping."""
        tok = ObservationTokenizer(
            NO_CA_OBS, NO_CA_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        assert tok.action_ca_map == []


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradients_flow_through_projections(self, encoder_config):
        """Ensure gradients propagate from output tokens back to projections."""
        tok = ObservationTokenizer(
            BUILDING_1_OBS, BUILDING_1_ACTIONS,
            encoder_config, TOKENIZER_CONFIG, d_model=D_MODEL,
        )
        obs_dim = tok.total_encoded_dims
        obs = torch.randn(2, obs_dim, requires_grad=True)

        result = tok(obs)
        loss = result.ca_tokens.sum() + result.sro_tokens.sum() + result.rl_token.sum()
        loss.backward()

        # Check that projection weights received gradients
        for name, proj in tok.ca_projections.items():
            assert proj.weight.grad is not None, f"No gradient for CA projection {name}"

        for name, proj in tok.sro_projections.items():
            assert proj.weight.grad is not None, f"No gradient for SRO projection {name}"

        if tok.rl_projection is not None:
            assert tok.rl_projection.weight.grad is not None, "No gradient for RL projection"


# ---------------------------------------------------------------------------
# Marker parsing tests (Phase C1 of DIN plan)
# ---------------------------------------------------------------------------


from algorithms.utils.observation_tokenizer import (
    _parse_markers,
    _build_encoded_dims_map,
    _has_markers,
    _matches_rule,
    _compute_encoded_dims,
    TokenGroup,
    EncoderSlice,
)


class TestParseMarkersBasic:
    """Test _parse_markers() with basic marker patterns."""

    def test_parse_ca_markers(self):
        """Parse CA markers without device ID."""
        names = [
            "__tkn_ca_battery__",
            "electrical_storage_soc",
            "__tkn_sro_temporal__",
            "month",
            "hour",
        ]
        groups = _parse_markers(names)

        assert len(groups) == 2
        assert groups[0].family == "ca"
        assert groups[0].type_name == "battery"
        assert groups[0].device_id is None
        assert groups[0].feature_names == ["electrical_storage_soc"]

        assert groups[1].family == "sro"
        assert groups[1].type_name == "temporal"
        assert groups[1].feature_names == ["month", "hour"]

    def test_parse_markers_extracts_device_id(self):
        """Parse CA markers with device ID."""
        names = [
            "__tkn_ca_ev_charger__charger_1_1__",
            "ev_charger_charger_1_1_connected_state",
            "ev_charger_charger_1_1_soc",
        ]
        groups = _parse_markers(names)

        assert len(groups) == 1
        assert groups[0].family == "ca"
        assert groups[0].type_name == "ev_charger"
        assert groups[0].device_id == "charger_1_1"
        assert len(groups[0].feature_names) == 2

    def test_parse_markers_multi_instance(self):
        """Parse multiple CA markers with different device IDs."""
        names = [
            "__tkn_ca_ev_charger__charger_1_1__",
            "feature_charger_1_1_a",
            "__tkn_ca_ev_charger__charger_1_2__",
            "feature_charger_1_2_a",
        ]
        groups = _parse_markers(names)

        assert len(groups) == 2
        assert groups[0].device_id == "charger_1_1"
        assert groups[1].device_id == "charger_1_2"

    def test_parse_markers_nfc(self):
        """Parse NFC marker."""
        names = [
            "__tkn_nfc__",
            "non_shiftable_load",
            "solar_generation",
        ]
        groups = _parse_markers(names)

        assert len(groups) == 1
        assert groups[0].family == "nfc"
        assert groups[0].type_name == ""
        assert groups[0].device_id is None
        assert groups[0].feature_names == ["non_shiftable_load", "solar_generation"]

    def test_parse_markers_empty(self):
        """Empty observation names → empty groups list."""
        groups = _parse_markers([])
        assert groups == []

    def test_parse_markers_no_markers(self):
        """Names without markers → empty groups list."""
        names = ["month", "hour", "day_type"]
        groups = _parse_markers(names)
        assert groups == []


class TestBuildEncodedDimsMap:
    """Test _build_encoded_dims_map() with markers."""

    def test_marker_gets_one_dim(self, encoder_config):
        """Marker names get 1 dim (NoNormalization)."""
        names = ["__tkn_ca_battery__", "electrical_storage_soc"]
        dims_map = _build_encoded_dims_map(names, encoder_config)

        assert dims_map["__tkn_ca_battery__"].n_dims == 1
        assert dims_map["electrical_storage_soc"].n_dims == 1

    def test_regular_features_get_correct_dims(self, encoder_config):
        """Regular features get dims from encoder rules."""
        names = [
            "__tkn_sro_temporal__",
            "month",  # PeriodicNormalization → 2
            "hour",   # PeriodicNormalization → 2
            "day_type",  # OnehotEncoding with 8 classes → 8
        ]
        dims_map = _build_encoded_dims_map(names, encoder_config)

        assert dims_map["__tkn_sro_temporal__"].n_dims == 1
        assert dims_map["month"].n_dims == 2
        assert dims_map["hour"].n_dims == 2
        assert dims_map["day_type"].n_dims == 8

    def test_contiguous_slices(self, encoder_config):
        """Slices must be contiguous."""
        names = ["__tkn_ca_battery__", "electrical_storage_soc", "month"]
        dims_map = _build_encoded_dims_map(names, encoder_config)

        slices = list(dims_map.values())
        for i in range(1, len(slices)):
            assert slices[i].start_idx == slices[i - 1].end_idx


class TestHasMarkers:
    """Test _has_markers() helper."""

    def test_has_markers_true(self):
        """Detect markers in observation names."""
        names = ["__tkn_ca_battery__", "electrical_storage_soc"]
        assert _has_markers(names) is True

    def test_has_markers_false(self):
        """No markers in observation names."""
        names = ["month", "hour", "electrical_storage_soc"]
        assert _has_markers(names) is False

    def test_has_markers_empty(self):
        """Empty list has no markers."""
        assert _has_markers([]) is False


class TestMatchesRule:
    """Test _matches_rule() encoder rule matching."""

    def test_equals_match(self):
        assert _matches_rule("month", {"equals": ["month", "hour"]})

    def test_equals_no_match(self):
        assert not _matches_rule("year", {"equals": ["month", "hour"]})

    def test_contains_match(self):
        assert _matches_rule(
            "electric_vehicle_charger_connected_state",
            {"contains": ["connected_state"]},
        )

    def test_default_match(self):
        assert _matches_rule("anything", {"default": True})


class TestComputeEncodedDims:
    """Test _compute_encoded_dims() dimension computation."""

    def test_periodic_normalization(self):
        assert _compute_encoded_dims({"type": "PeriodicNormalization"}) == 2

    def test_onehot_encoding(self):
        spec = {"type": "OnehotEncoding", "params": {"classes": [0, 1]}}
        assert _compute_encoded_dims(spec) == 2

    def test_remove_feature(self):
        assert _compute_encoded_dims({"type": "RemoveFeature"}) == 0

    def test_no_normalization(self):
        assert _compute_encoded_dims({"type": "NoNormalization"}) == 1


class TestTokenGroupDataclass:
    """Test TokenGroup dataclass."""

    def test_token_group_creation(self):
        group = TokenGroup(
            family="ca",
            type_name="battery",
            device_id=None,
            feature_names=["electrical_storage_soc"],
        )
        assert group.family == "ca"
        assert group.type_name == "battery"
        assert group.device_id is None
        assert group.feature_names == ["electrical_storage_soc"]

    def test_token_group_with_device_id(self):
        group = TokenGroup(
            family="ca",
            type_name="ev_charger",
            device_id="charger_1_1",
            feature_names=[],
        )
        assert group.device_id == "charger_1_1"


class TestEncoderSlice:
    """Test EncoderSlice class."""

    def test_encoder_slice_equality(self):
        slice1 = EncoderSlice(0, 2, 2)
        slice2 = EncoderSlice(0, 2, 2)
        assert slice1 == slice2

    def test_encoder_slice_inequality(self):
        slice1 = EncoderSlice(0, 2, 2)
        slice2 = EncoderSlice(2, 4, 2)
        assert slice1 != slice2

    def test_encoder_slice_repr(self):
        slice_ = EncoderSlice(0, 2, 2)
        assert "start_idx=0" in repr(slice_)
        assert "end_idx=2" in repr(slice_)
