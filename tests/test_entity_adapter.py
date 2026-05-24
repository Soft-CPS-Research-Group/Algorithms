from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from utils.entity_adapter import EntityContractAdapter


class _DummyEntityEnv:
    def __init__(self):
        self.interface = "entity"
        self.topology_mode = "dynamic"

        self._entity_specs = {
            "tables": {
                "district": {"ids": ["district_0"], "features": ["hour", "minutes"]},
                "building": {"ids": ["B1", "B2"], "features": ["load_power_kw", "pv_power_kw"]},
                "charger": {
                    "ids": ["B1/C1", "B2/C2"],
                    "features": [
                        "connected_state",
                        "connected_ev_soc",
                        "connected_ev_required_soc_departure",
                        "connected_ev_departure_time_step",
                    ],
                },
                "storage": {"ids": ["B1/electrical_storage"], "features": ["soc"]},
                "pv": {"ids": ["B1/pv"], "features": ["generation_power_kw"]},
                "ev": {"ids": ["EV1", "EV2"], "features": ["soc"]},
                "deferrable_appliance": {
                    "ids": ["B1/deferrable_appliance_1", "B2/deferrable_appliance_1"],
                    "features": [
                        "pending",
                        "running",
                        "can_start",
                        "urgency_ratio",
                        "slack_ratio",
                        "priority",
                    ],
                },
            },
            "actions": {
                "building": {"ids": ["B1", "B2"], "features": ["electrical_storage"]},
                "charger": {"ids": ["B1/C1", "B2/C2"], "features": ["electric_vehicle_storage"]},
                "deferrable_appliance": {
                    "ids": ["B1/deferrable_appliance_1", "B2/deferrable_appliance_1"],
                    "features": ["start"],
                },
            },
        }

        self._observation_space = spaces.Dict(
            {
                "tables": spaces.Dict(
                    {
                        "district": spaces.Box(
                            low=np.array([[0.0, 0.0]], dtype=np.float32),
                            high=np.array([[23.0, 59.0]], dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "building": spaces.Box(
                            low=np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
                            high=np.array([[100.0, 0.0], [100.0, 0.0]], dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "charger": spaces.Box(
                            low=np.zeros((2, 4), dtype=np.float32),
                            high=np.array([[1.0, 100.0, 100.0, 24.0], [1.0, 100.0, 100.0, 24.0]], dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "storage": spaces.Box(
                            low=np.array([[0.0]], dtype=np.float32),
                            high=np.array([[1.0]], dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "pv": spaces.Box(
                            low=np.array([[0.0]], dtype=np.float32),
                            high=np.array([[20.0]], dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "ev": spaces.Box(
                            low=np.array([[0.0], [0.0]], dtype=np.float32),
                            high=np.array([[100.0], [100.0]], dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "deferrable_appliance": spaces.Box(
                            low=np.zeros((2, 6), dtype=np.float32),
                            high=np.ones((2, 6), dtype=np.float32),
                            dtype=np.float32,
                        ),
                    }
                )
            }
        )

        self._action_names = [
            ["electrical_storage", "electric_vehicle_storage_C1"],
            ["electrical_storage", "electric_vehicle_storage_C2"],
        ]

    @property
    def entity_specs(self):
        return self._entity_specs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_names(self):
        return self._action_names


def _sample_observation_payload() -> dict:
    return {
        "tables": {
            "district": np.array([[12.0, 30.0]], dtype=np.float32),
            "building": np.array([[50.0, 0.0], [80.0, 0.0]], dtype=np.float32),
            "charger": np.array(
                [
                    [1.0, 45.0, 80.0, 18.0],
                    [0.0, 20.0, 70.0, 19.0],
                ],
                dtype=np.float32,
            ),
            "storage": np.array([[0.4]], dtype=np.float32),
            "pv": np.array([[8.0]], dtype=np.float32),
            "ev": np.array([[45.0], [22.0]], dtype=np.float32),
            "deferrable_appliance": np.array(
                [
                    [1.0, 0.0, 1.0, 0.8, 0.1, 0.9],
                    [1.0, 0.0, 0.0, 0.2, 0.9, 0.2],
                ],
                dtype=np.float32,
            ),
        },
        "edges": {
            "building_to_charger": np.array([[0, 0], [1, 1]], dtype=np.int32),
            "building_to_storage": np.array([[0, 0]], dtype=np.int32),
            "building_to_pv": np.array([[0, 0]], dtype=np.int32),
            "building_to_deferrable_appliance": np.array([[0, 0], [1, 1]], dtype=np.int32),
            "charger_to_ev_connected": np.array([[0, 0], [1, 1]], dtype=np.int32),
            "charger_to_ev_connected_mask": np.array([1.0, 0.0], dtype=np.float32),
            "charger_to_ev_incoming": np.array([[0, 1], [1, 0]], dtype=np.int32),
            "charger_to_ev_incoming_mask": np.array([0.0, 1.0], dtype=np.float32),
        },
        "meta": {"topology_version": 0},
    }


def test_entity_adapter_stable_order_and_aliases():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(env, normalization_enabled=True, clip=True)

    observations, observation_names, observation_spaces = adapter.to_agent_observations(_sample_observation_payload())

    assert len(observations) == 2
    assert len(observation_names) == 2
    assert observation_names[0][0] == "district__hour"
    assert "electric_vehicle_charger_state" in observation_names[0]
    assert "electric_vehicle_soc" in observation_names[0]
    assert any(name.startswith("charger::B1/C1::") for name in observation_names[0])
    assert "deferrable_appliance::B1/deferrable_appliance_1::can_start" in observation_names[0]
    assert "active_deferrable_appliances_count" in observation_names[0]
    assert observation_spaces[0].shape[0] == observations[0].shape[0]


def test_entity_adapter_cached_source_plan_matches_collected_layout():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(env, normalization_enabled=True, clip=True)
    payload = _sample_observation_payload()

    collected, observation_names, observation_spaces = adapter.to_agent_observations(payload)
    cached, cached_names, cached_spaces = adapter.to_agent_observations(payload)

    assert cached_names == observation_names
    assert [space.shape for space in cached_spaces] == [space.shape for space in observation_spaces]
    assert len(cached) == len(collected)
    for cached_obs, collected_obs in zip(cached, collected):
        np.testing.assert_allclose(cached_obs, collected_obs, atol=1e-9)


def test_entity_adapter_direct_encoded_observations_match_normalize_path():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(
        env,
        normalization_enabled=True,
        clip=True,
        encoding_profile="maddpg_v3_operational",
    )
    payload = _sample_observation_payload()

    observations, observation_names, observation_spaces = adapter.to_agent_observations(payload)
    expected = [
        adapter.normalize_observation(
            agent_index=idx,
            observation=obs,
            observation_names=observation_names[idx],
            observation_space=observation_spaces[idx],
        )
        for idx, obs in enumerate(observations)
    ]
    direct, direct_names, direct_spaces = adapter.to_agent_encoded_observations(payload)

    assert direct_names == observation_names
    assert [space.shape for space in direct_spaces] == [space.shape for space in observation_spaces]
    assert len(direct) == len(expected)
    for direct_obs, expected_obs in zip(direct, expected):
        np.testing.assert_allclose(direct_obs, expected_obs, atol=1e-9)


def test_entity_adapter_minmax_normalization_with_invalid_bounds_passthrough():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(env, normalization_enabled=True, clip=True)

    observations, observation_names, observation_spaces = adapter.to_agent_observations(_sample_observation_payload())
    encoded = adapter.normalize_observation(
        agent_index=0,
        observation=observations[0],
        observation_names=observation_names[0],
        observation_space=observation_spaces[0],
    )

    load_idx = observation_names[0].index("load_power_kw")
    assert encoded[load_idx] == pytest.approx(0.5, abs=1e-6)

    # pv_power_kw has low==high in dummy observation space -> passthrough expected.
    pv_idx = observation_names[0].index("pv_power_kw")
    assert encoded[pv_idx] == pytest.approx(observations[0][pv_idx], abs=1e-6)


def test_entity_adapter_maddpg_v1_profile_encodes_time_and_ev_features():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(
        env,
        normalization_enabled=True,
        clip=True,
        encoding_profile="maddpg_v1",
    )

    observations, observation_names, observation_spaces = adapter.to_agent_observations(_sample_observation_payload())
    encoded = adapter.normalize_observation(
        agent_index=0,
        observation=observations[0],
        observation_names=observation_names[0],
        observation_space=observation_spaces[0],
    )
    encoded_names = adapter.encoded_observation_names(observation_names)[0]

    assert len(encoded) == len(encoded_names)
    assert "district__time_of_day_sin" in encoded_names
    assert "district__time_of_day_cos" in encoded_names
    assert "district__hour" not in encoded_names
    assert "electric_vehicle_soc" not in encoded_names

    deficit_name = "charger::B1/C1::connected_ev_soc_deficit"
    assert deficit_name in encoded_names
    assert encoded[encoded_names.index(deficit_name)] == pytest.approx(0.35, abs=1e-6)

    hours_name = "charger::B1/C1::connected_ev_hours_until_departure"
    assert hours_name in encoded_names
    assert encoded[encoded_names.index(hours_name)] == pytest.approx(18.0 / 24.0, abs=1e-6)


def test_entity_adapter_maddpg_v1_profile_encodes_simulator_043_features():
    env = _DummyEntityEnv()
    env.seconds_per_time_step = 15.0
    adapter = EntityContractAdapter(
        env,
        normalization_enabled=True,
        clip=True,
        encoding_profile="maddpg_v1",
    )

    observation_names = [
        "charger::B1/C1::incoming_ev_estimated_soc_arrival",
        "charger::B1/C1::incoming_ev_required_soc_departure",
        "charger::B1/C1::incoming_ev_departure_time_step",
        "deferrable_appliance::B1/deferrable_appliance_1::must_run",
        "deferrable_appliance::B1/deferrable_appliance_1::remaining_duration_steps",
        "deferrable_appliance::B1/deferrable_appliance_1::cycle_peak_step_offset_ratio",
    ]
    observation = np.array([0.25, 0.80, 960.0, 1.0, 480.0, -0.5], dtype=np.float32)
    observation_space = spaces.Box(
        low=np.array([-0.1, -0.1, -1.0, 0.0, 0.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 5760.0, 1.0, 5760.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    encoded = adapter.normalize_observation(
        agent_index=0,
        observation=observation,
        observation_names=observation_names,
        observation_space=observation_space,
    )
    encoded_names = adapter.encoded_observation_names([observation_names])[0]

    incoming_required = "charger::B1/C1::incoming_ev_required_soc_departure"
    incoming_deficit = "charger::B1/C1::incoming_ev_soc_deficit"
    incoming_hours = "charger::B1/C1::incoming_ev_hours_until_departure_from_time_step"
    incoming_available = "charger::B1/C1::incoming_ev_departure_available"
    incoming_urgency = "charger::B1/C1::incoming_ev_departure_urgency_24h"
    must_run = "deferrable_appliance::B1/deferrable_appliance_1::must_run"
    remaining_duration = "deferrable_appliance::B1/deferrable_appliance_1::remaining_duration_steps_day_ratio"
    peak_offset = "deferrable_appliance::B1/deferrable_appliance_1::cycle_peak_step_offset_ratio"

    assert incoming_required in encoded_names
    assert encoded[encoded_names.index(incoming_required)] == pytest.approx(0.80, abs=1e-6)
    assert encoded[encoded_names.index(incoming_deficit)] == pytest.approx(0.55, abs=1e-6)
    assert encoded[encoded_names.index(incoming_hours)] == pytest.approx(4.0 / 24.0, abs=1e-6)
    assert encoded[encoded_names.index(incoming_available)] == pytest.approx(1.0, abs=1e-6)
    assert encoded[encoded_names.index(incoming_urgency)] == pytest.approx(20.0 / 24.0, abs=1e-6)
    assert encoded[encoded_names.index(must_run)] == pytest.approx(1.0, abs=1e-6)
    assert encoded[encoded_names.index(remaining_duration)] == pytest.approx(480.0 / 5760.0, abs=1e-6)
    assert encoded[encoded_names.index(peak_offset)] == pytest.approx(-0.5, abs=1e-6)


def test_entity_adapter_maddpg_v2_compact_profile_drops_redundant_features():
    env = _DummyEntityEnv()
    env.seconds_per_time_step = 15.0
    adapter = EntityContractAdapter(
        env,
        normalization_enabled=True,
        clip=True,
        encoding_profile="maddpg_v2_compact",
    )

    observation_names = [
        "district__hour",
        "district__minutes",
        "district__community_net_power_kw",
        "district__community_net_energy_kwh_step",
        "district__active_evs_count",
        "load_power_kw",
        "load_energy_kwh_step",
        "active_chargers_count",
        "charger::B1/C1::connected_ev_soc",
        "charger::B1/C1::connected_ev_required_soc_departure",
        "charger::B1/C1::connected_ev_departure_time_step",
        "charger::B1/C1::applied_power_kw",
        "charger::B1/C1::applied_energy_kwh_step",
        "charger::B1/C1::incoming_ev_estimated_soc_arrival",
        "charger::B1/C1::incoming_ev_required_soc_departure",
        "charger::B1/C1::incoming_ev_departure_time_step",
        "deferrable_appliance::B1/deferrable_appliance_1::must_run",
        "deferrable_appliance::B1/deferrable_appliance_1::remaining_duration_steps",
        "deferrable_appliance::B1/deferrable_appliance_1::earliest_start_time_step",
    ]
    observation = np.array(
        [
            12.0,
            30.0,
            20.0,
            0.1,
            8.0,
            4.0,
            0.01,
            1.0,
            0.45,
            0.80,
            960.0,
            3.5,
            0.014,
            0.25,
            0.70,
            1920.0,
            1.0,
            480.0,
            600.0,
        ],
        dtype=np.float32,
    )
    observation_space = spaces.Box(
        low=np.array(
            [
                0.0,
                0.0,
                -100.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.1,
                -0.1,
                -1.0,
                -22.0,
                -1.0,
                -0.1,
                -0.1,
                -1.0,
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                23.0,
                59.0,
                100.0,
                1.0,
                20.0,
                20.0,
                1.0,
                10.0,
                1.0,
                1.0,
                5760.0,
                22.0,
                1.0,
                1.0,
                1.0,
                5760.0,
                1.0,
                5760.0,
                5760.0,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )

    encoded = adapter.normalize_observation(
        agent_index=0,
        observation=observation,
        observation_names=observation_names,
        observation_space=observation_space,
    )
    encoded_names = adapter.encoded_observation_names([observation_names])[0]

    assert len(encoded) == len(encoded_names)
    assert "district__time_of_day_sin" in encoded_names
    assert "district__community_net_power_kw" in encoded_names
    assert "load_power_kw" in encoded_names
    assert "charger::B1/C1::applied_power_kw" in encoded_names
    assert "charger::B1/C1::connected_ev_soc_deficit" in encoded_names
    assert "charger::B1/C1::connected_ev_departure_urgency_24h" in encoded_names
    assert "charger::B1/C1::incoming_ev_soc_deficit" in encoded_names
    assert "charger::B1/C1::incoming_ev_departure_urgency_24h" in encoded_names
    assert "deferrable_appliance::B1/deferrable_appliance_1::must_run" in encoded_names
    assert "deferrable_appliance::B1/deferrable_appliance_1::remaining_duration_steps_day_ratio" in encoded_names

    assert "district__community_net_energy_kwh_step" not in encoded_names
    assert "district__active_evs_count" not in encoded_names
    assert "load_energy_kwh_step" not in encoded_names
    assert "active_chargers_count" not in encoded_names
    assert "charger::B1/C1::applied_energy_kwh_step" not in encoded_names
    assert "charger::B1/C1::connected_ev_hours_until_departure" not in encoded_names
    assert "charger::B1/C1::incoming_ev_hours_until_departure_from_time_step" not in encoded_names
    assert "deferrable_appliance::B1/deferrable_appliance_1::earliest_start_time_of_day_sin" not in encoded_names

    connected_deficit = "charger::B1/C1::connected_ev_soc_deficit"
    incoming_deficit = "charger::B1/C1::incoming_ev_soc_deficit"
    assert encoded[encoded_names.index(connected_deficit)] == pytest.approx(0.35, abs=1e-6)
    assert encoded[encoded_names.index(incoming_deficit)] == pytest.approx(0.45, abs=1e-6)


def test_entity_adapter_maddpg_v3_operational_keeps_simulator_100_features():
    env = _DummyEntityEnv()
    env.seconds_per_time_step = 15.0
    adapter = EntityContractAdapter(
        env,
        normalization_enabled=True,
        clip=True,
        encoding_profile="maddpg_v3_operational",
    )

    observation_names = [
        "district__hour",
        "district__minutes",
        "district__forecast_price_mean_1h",
        "district__forecast_community_import_power_mean_1h_kw",
        "district__community_flexible_charge_capacity_kw",
        "forecast_load_power_mean_1h_kw",
        "storage::B1/electrical_storage::can_charge",
        "storage::B1/electrical_storage::available_charge_action_normalized",
        "storage::B1/electrical_storage::last_projection_error_kw",
        "charger::B1/C1::connected_ev_soc",
        "charger::B1/C1::connected_ev_required_soc_departure",
        "charger::B1/C1::hours_until_departure",
        "charger::B1/C1::can_charge",
        "charger::B1/C1::available_charge_action_normalized",
        "charger::B1/C1::departure_feasibility_ratio",
        "charger::B1/C1::departure_energy_margin_kwh",
        "charger::B1/C1::min_required_action_normalized",
        "charger::B1/C1::last_requested_action_normalized",
        "charger::B1/C1::clip_reason_headroom",
        "deferrable_appliance::B1/deferrable_appliance_1::must_start_now",
        "deferrable_appliance::B1/deferrable_appliance_1::remaining_duration_hours",
        "deferrable_appliance::B1/deferrable_appliance_1::start_blocked",
    ]
    observation = np.array(
        [
            12.0,
            30.0,
            0.12,
            8.0,
            15.0,
            4.0,
            1.0,
            0.7,
            2.0,
            0.45,
            0.80,
            3.0,
            1.0,
            0.6,
            0.75,
            -2.0,
            0.35,
            0.4,
            1.0,
            1.0,
            0.5,
            1.0,
        ],
        dtype=np.float32,
    )
    observation_space = spaces.Box(
        low=np.array(
            [
                0.0,
                0.0,
                0.0,
                -100.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -22.0,
                -0.1,
                -0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                -100.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                23.0,
                59.0,
                1.0,
                100.0,
                100.0,
                20.0,
                1.0,
                1.0,
                22.0,
                1.0,
                1.0,
                24.0,
                1.0,
                1.0,
                1.0,
                100.0,
                1.0,
                1.0,
                1.0,
                1.0,
                24.0,
                1.0,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )

    encoded = adapter.normalize_observation(
        agent_index=0,
        observation=observation,
        observation_names=observation_names,
        observation_space=observation_space,
    )
    encoded_names = adapter.encoded_observation_names([observation_names])[0]

    assert len(encoded) == len(encoded_names)
    assert "district__forecast_price_mean_1h" in encoded_names
    assert "district__forecast_community_import_power_mean_1h_kw" in encoded_names
    assert "district__community_flexible_charge_capacity_kw" in encoded_names
    assert "forecast_load_power_mean_1h_kw" in encoded_names
    assert "storage::B1/electrical_storage::available_charge_action_normalized" in encoded_names
    assert "storage::B1/electrical_storage::last_projection_error_kw" in encoded_names
    assert "charger::B1/C1::hours_until_departure_24h" in encoded_names
    assert "charger::B1/C1::available_charge_action_normalized" in encoded_names
    assert "charger::B1/C1::departure_feasibility_ratio" in encoded_names
    assert "charger::B1/C1::departure_energy_margin_kwh" in encoded_names
    assert "charger::B1/C1::min_required_action_normalized" in encoded_names
    assert "charger::B1/C1::last_requested_action_normalized" in encoded_names
    assert "charger::B1/C1::clip_reason_headroom" in encoded_names
    assert "deferrable_appliance::B1/deferrable_appliance_1::must_start_now" in encoded_names
    assert "deferrable_appliance::B1/deferrable_appliance_1::remaining_duration_hours" in encoded_names
    assert "deferrable_appliance::B1/deferrable_appliance_1::start_blocked" in encoded_names

    required = "charger::B1/C1::min_required_action_normalized"
    hours = "charger::B1/C1::hours_until_departure_24h"
    assert encoded[encoded_names.index(required)] == pytest.approx(0.35, abs=1e-6)
    assert encoded[encoded_names.index(hours)] == pytest.approx(3.0 / 24.0, abs=1e-6)


def test_entity_adapter_maddpg_v3_realtime_drops_simulator_perfect_forecasts():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(
        env,
        normalization_enabled=True,
        clip=True,
        encoding_profile="maddpg_v3_realtime",
    )

    observation_names = [
        "district__electricity_pricing",
        "district__forecast_price_mean_1h",
        "forecast_load_power_mean_1h_kw",
        "charger::B1/C1::min_required_action_normalized",
    ]
    observation = np.array([0.2, 0.1, 4.0, 0.35], dtype=np.float32)
    observation_space = spaces.Box(
        low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 20.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    encoded = adapter.normalize_observation(
        agent_index=0,
        observation=observation,
        observation_names=observation_names,
        observation_space=observation_space,
    )
    encoded_names = adapter.encoded_observation_names([observation_names])[0]

    assert len(encoded) == len(encoded_names)
    assert "district__electricity_pricing" in encoded_names
    assert "charger::B1/C1::min_required_action_normalized" in encoded_names
    assert "district__forecast_price_mean_1h" not in encoded_names
    assert "forecast_load_power_mean_1h_kw" not in encoded_names


def test_entity_adapter_observation_dimension_is_stable_when_ev_links_toggle():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(env, normalization_enabled=True, clip=True)

    payload_a = _sample_observation_payload()
    obs_a, names_a, spaces_a = adapter.to_agent_observations(payload_a)

    payload_b = _sample_observation_payload()
    payload_b["edges"]["charger_to_ev_connected_mask"] = np.array([0.0, 0.0], dtype=np.float32)
    payload_b["edges"]["charger_to_ev_incoming_mask"] = np.array([0.0, 0.0], dtype=np.float32)
    payload_b["tables"]["ev"] = np.array([[0.0], [0.0]], dtype=np.float32)
    obs_b, names_b, spaces_b = adapter.to_agent_observations(payload_b)

    assert len(obs_a) == len(obs_b)
    assert names_a[0] == names_b[0]
    assert spaces_a[0].shape == spaces_b[0].shape
    assert obs_a[0].shape[0] == obs_b[0].shape[0]


def test_entity_adapter_decodes_agent_actions_to_entity_tables():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(env, normalization_enabled=True, clip=True)

    action_payload = adapter.to_entity_actions(
        actions=[
            [0.25, 0.75],
            [0.5, 0.10],
        ],
        action_names=env.action_names,
    )

    building_table = action_payload["tables"]["building"]
    charger_table = action_payload["tables"]["charger"]

    assert building_table.shape == (2, 1)
    assert charger_table.shape == (2, 1)
    assert building_table[0, 0] == pytest.approx(0.25, abs=1e-6)
    assert building_table[1, 0] == pytest.approx(0.5, abs=1e-6)
    assert charger_table[0, 0] == pytest.approx(0.75, abs=1e-6)
    assert charger_table[1, 0] == pytest.approx(0.10, abs=1e-6)


def test_entity_adapter_decodes_namespaced_charger_actions():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(env, normalization_enabled=True, clip=True)

    action_payload = adapter.to_entity_actions(
        actions=[
            [0.15, 0.65],
            [0.35, 0.45],
        ],
        action_names=[
            ["electrical_storage", "charger::B1/C1::electric_vehicle_storage"],
            ["electrical_storage", "B2/C2::electric_vehicle_storage"],
        ],
    )

    building_table = action_payload["tables"]["building"]
    charger_table = action_payload["tables"]["charger"]

    assert building_table[0, 0] == pytest.approx(0.15, abs=1e-6)
    assert building_table[1, 0] == pytest.approx(0.35, abs=1e-6)
    assert charger_table[0, 0] == pytest.approx(0.65, abs=1e-6)
    assert charger_table[1, 0] == pytest.approx(0.45, abs=1e-6)


def test_entity_adapter_decodes_direct_charger_feature_when_unique_per_building():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(env, normalization_enabled=True, clip=True)

    action_payload = adapter.to_entity_actions(
        actions=[
            [0.20, 0.90],
            [0.40, 0.30],
        ],
        action_names=[
            ["electrical_storage", "electric_vehicle_storage"],
            ["electrical_storage", "electric_vehicle_storage"],
        ],
    )

    building_table = action_payload["tables"]["building"]
    charger_table = action_payload["tables"]["charger"]

    assert building_table[0, 0] == pytest.approx(0.20, abs=1e-6)
    assert building_table[1, 0] == pytest.approx(0.40, abs=1e-6)
    assert charger_table[0, 0] == pytest.approx(0.90, abs=1e-6)
    assert charger_table[1, 0] == pytest.approx(0.30, abs=1e-6)


def test_entity_adapter_decodes_deferrable_actions():
    env = _DummyEntityEnv()
    adapter = EntityContractAdapter(env, normalization_enabled=True, clip=True)

    action_payload = adapter.to_entity_actions(
        actions=[
            [1.0],
            [0.25],
        ],
        action_names=[
            ["deferrable_appliance_deferrable_appliance_1"],
            ["deferrable_appliance::B2/deferrable_appliance_1::start"],
        ],
    )

    deferrable_table = action_payload["tables"]["deferrable_appliance"]

    assert deferrable_table.shape == (2, 1)
    assert deferrable_table[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert deferrable_table[1, 0] == pytest.approx(0.25, abs=1e-6)
