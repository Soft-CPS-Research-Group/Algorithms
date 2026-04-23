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
            },
            "actions": {
                "building": {"ids": ["B1", "B2"], "features": ["electrical_storage"]},
                "charger": {"ids": ["B1/C1", "B2/C2"], "features": ["electric_vehicle_storage"]},
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
        },
        "edges": {
            "building_to_charger": np.array([[0, 0], [1, 1]], dtype=np.int32),
            "building_to_storage": np.array([[0, 0]], dtype=np.int32),
            "building_to_pv": np.array([[0, 0]], dtype=np.int32),
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
    assert observation_spaces[0].shape[0] == observations[0].shape[0]


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
