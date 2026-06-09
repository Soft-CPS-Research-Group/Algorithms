import json

import numpy as np
import pytest

from algorithms.agents.rbc_agent import RuleBasedPolicy


class DummySpace:
    def __init__(self, low, high):
        self.low = low
        self.high = high


def make_agent():
    config = {
        "simulator": {"dataset_path": "datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"},
        "algorithm": {
            "name": "RuleBasedPolicy",
            "hyperparameters": {
                "pv_charge_threshold": 2.0,
                "flexibility_hours": 3.0,
                "emergency_hours": 1.0,
                "pv_preferred_charge_rate": 0.5,
                "flex_trickle_charge": 0.1,
                "min_charge_rate": 0.05,
                "emergency_charge_rate": 1.0,
                "energy_epsilon": 1e-3,
                "default_capacity_kwh": 60.0,
            },
        },
    }
    agent = RuleBasedPolicy(config)
    observation_names = [
        [
            "hour",
            "electric_vehicle_charger_state",
            "electric_vehicle_soc",
            "electric_vehicle_required_soc_departure",
            "electric_vehicle_departure_time",
            "solar_generation",
        ]
    ]
    action_names = [["electric_vehicle_storage"]]
    action_space = [DummySpace([0.0], [1.0])]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=action_names,
        action_space=action_space,
        observation_space=[None],
        metadata={"building_names": ["Building_5"], "seconds_per_time_step": 3600},
    )
    return agent


def test_rbc_delays_when_flexible_and_no_pv():
    agent = make_agent()
    observations = [
        np.array([8.0, 1.0, 40.0, 80.0, 18.0, 0.5], dtype=float)
    ]
    actions = agent.predict(observations)
    assert actions[0][0] == pytest.approx(0.1, rel=1e-6)


def test_rbc_charges_with_pv():
    agent = make_agent()
    observations = [
        np.array([10.0, 1.0, 40.0, 80.0, 18.0, 5.0], dtype=float)
    ]
    actions = agent.predict(observations)
    assert actions[0][0] >= 0.5


def test_rbc_emergency_window_forces_full_charge():
    agent = make_agent()
    observations = [
        np.array([17.5, 1.0, 40.0, 80.0, 18.0, 0.0], dtype=float)
    ]
    actions = agent.predict(observations)
    assert actions[0][0] == pytest.approx(1.0, rel=1e-6)


def test_rbc_reads_namespaced_citylearn_ev_observations():
    agent = RuleBasedPolicy(
        {
            "simulator": {"dataset_path": "datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"},
            "algorithm": {
                "name": "RuleBasedPolicy",
                "hyperparameters": {
                    "pv_charge_threshold": 2.0,
                    "flexibility_hours": 3.0,
                    "emergency_hours": 1.0,
                    "pv_preferred_charge_rate": 0.5,
                    "flex_trickle_charge": 0.0,
                    "min_charge_rate": 0.05,
                    "emergency_charge_rate": 1.0,
                },
            },
        }
    )
    observation_names = [
        [
            "hour",
            "solar_generation",
            "electric_vehicle_charger_charger_1_1_connected_state",
            "connected_electric_vehicle_at_charger_charger_1_1_departure_time",
            "connected_electric_vehicle_at_charger_charger_1_1_required_soc_departure",
            "connected_electric_vehicle_at_charger_charger_1_1_soc",
            "connected_electric_vehicle_at_charger_charger_1_1_battery_capacity",
        ]
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[["electric_vehicle_storage_charger_1_1"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    actions = agent.predict([np.array([8.0, 4.0, 1.0, 2.0, 0.9, 0.5, 60.0], dtype=float)])

    assert actions[0][0] > 0.0


def test_rbc_reads_entity_charger_specific_ev_observations():
    agent = RuleBasedPolicy(
        {
            "simulator": {"dataset_path": "datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json"},
            "algorithm": {
                "name": "RuleBasedPolicy",
                "hyperparameters": {
                    "pv_charge_threshold": 2.0,
                    "flexibility_hours": 3.0,
                    "emergency_hours": 1.0,
                    "flex_trickle_charge": 0.0,
                    "emergency_charge_rate": 1.0,
                },
            },
        }
    )
    observation_names = [
        [
            "solar_generation",
            "charger::Building_15/charger_15_1::connected_state",
            "charger::Building_15/charger_15_1::connected_ev_soc",
            "charger::Building_15/charger_15_1::connected_ev_required_soc_departure",
            "charger::Building_15/charger_15_1::connected_ev_battery_capacity_kwh",
            "charger::Building_15/charger_15_1::hours_until_departure",
            "charger::Building_15/charger_15_2::connected_state",
            "charger::Building_15/charger_15_2::connected_ev_soc",
            "charger::Building_15/charger_15_2::connected_ev_required_soc_departure",
            "charger::Building_15/charger_15_2::connected_ev_battery_capacity_kwh",
            "charger::Building_15/charger_15_2::hours_until_departure",
            "electric_vehicle_charger_state",
            "electric_vehicle_soc",
            "electric_vehicle_required_soc_departure",
            "electric_vehicle_departure_time",
        ]
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[
            [
                "electrical_storage",
                "electric_vehicle_storage_charger_15_1",
                "electric_vehicle_storage_charger_15_2",
            ]
        ],
        action_space=[DummySpace([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_15"], "seconds_per_time_step": 900},
    )

    observations = [
        np.array(
            [
                0.0,
                0.0,
                20.0,
                90.0,
                60.0,
                1.0,
                1.0,
                20.0,
                80.0,
                60.0,
                2.0,
                1.0,
                20.0,
                90.0,
                1.0,
            ],
            dtype=float,
        )
    ]

    actions = agent.predict(observations)

    assert actions[0][0] == pytest.approx(0.0, abs=1e-6)
    assert actions[0][1] == pytest.approx(0.0, abs=1e-6)
    assert actions[0][2] > 0.0


def test_rbc_clips_ev_charge_to_phase_headroom():
    agent = RuleBasedPolicy(
        {
            "simulator": {"dataset_path": "datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json"},
            "algorithm": {
                "name": "RuleBasedPolicy",
                "hyperparameters": {
                    "pv_charge_threshold": 2.0,
                    "flexibility_hours": 3.0,
                    "emergency_hours": 1.0,
                    "flex_trickle_charge": 0.0,
                    "emergency_charge_rate": 1.0,
                },
            },
        }
    )
    observation_names = [
        [
            "solar_generation",
            "charging_building_headroom_kw",
            "charging_phase_L1_headroom_kw",
            "charging_phase_L2_headroom_kw",
            "charging_phase_L3_headroom_kw",
            "charger::Building_15/charger_15_1::connected_state",
            "charger::Building_15/charger_15_1::connected_ev_soc",
            "charger::Building_15/charger_15_1::connected_ev_required_soc_departure",
            "charger::Building_15/charger_15_1::connected_ev_battery_capacity_kwh",
            "charger::Building_15/charger_15_1::hours_until_departure",
        ]
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[["electric_vehicle_storage_charger_15_1"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_15"], "seconds_per_time_step": 3600},
    )

    observations = [np.array([0.0, 10.0, 1.85, 10.0, 10.0, 1.0, 0.0, 1.0, 60.0, 0.5], dtype=float)]
    actions = agent.predict(observations)

    assert actions[0][0] == pytest.approx(1.85 / 7.4, abs=1e-6)


def test_rbc_ev_headroom_clip_allows_existing_charger_power():
    agent = RuleBasedPolicy(
        {
            "simulator": {"dataset_path": "datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json"},
            "algorithm": {
                "name": "RuleBasedPolicy",
                "hyperparameters": {
                    "pv_charge_threshold": 2.0,
                    "flexibility_hours": 3.0,
                    "emergency_hours": 1.0,
                    "flex_trickle_charge": 0.0,
                    "emergency_charge_rate": 1.0,
                },
            },
        }
    )
    observation_names = [
        [
            "solar_generation",
            "charging_building_headroom_kw",
            "charging_phase_L1_headroom_kw",
            "charging_phase_L2_headroom_kw",
            "charging_phase_L3_headroom_kw",
            "charger::Building_15/charger_15_1::connected_state",
            "charger::Building_15/charger_15_1::connected_ev_soc",
            "charger::Building_15/charger_15_1::connected_ev_required_soc_departure",
            "charger::Building_15/charger_15_1::connected_ev_battery_capacity_kwh",
            "charger::Building_15/charger_15_1::hours_until_departure",
            "charger::Building_15/charger_15_1::applied_power_kw",
        ]
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[["electric_vehicle_storage_charger_15_1"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_15"], "seconds_per_time_step": 3600},
    )

    observations = [np.array([0.0, 0.0, 0.0, 10.0, 10.0, 1.0, 0.0, 1.0, 60.0, 0.5, 7.0], dtype=float)]
    actions = agent.predict(observations)

    assert actions[0][0] == pytest.approx(7.0 / 7.4, abs=1e-6)


def test_rbc_converts_entity_departure_steps_to_hours():
    agent = RuleBasedPolicy(
        {
            "simulator": {"dataset_path": "datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json"},
            "algorithm": {
                "name": "RuleBasedPolicy",
                "hyperparameters": {
                    "pv_charge_threshold": 2.0,
                    "flexibility_hours": 3.0,
                    "emergency_hours": 1.0,
                    "flex_trickle_charge": 0.0,
                    "emergency_charge_rate": 1.0,
                },
            },
        }
    )
    observation_names = [
        [
            "solar_generation",
            "charger::Building_1/charger_1_1::connected_state",
            "charger::Building_1/charger_1_1::connected_ev_soc",
            "charger::Building_1/charger_1_1::connected_ev_required_soc_departure",
            "charger::Building_1/charger_1_1::connected_ev_battery_capacity_kwh",
            "charger::Building_1/charger_1_1::connected_ev_departure_time_step",
        ]
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[["electric_vehicle_storage_charger_1_1"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 900},
    )

    observations = [np.array([0.0, 1.0, 20.0, 80.0, 60.0, 8.0], dtype=float)]
    actions = agent.predict(observations)

    assert actions[0][0] == pytest.approx(1.0, abs=1e-6)


def test_rbc_can_disable_ev_and_deferrable_controls():
    agent = RuleBasedPolicy(
        {
            "simulator": {},
            "algorithm": {
                "name": "RuleBasedPolicy",
                "hyperparameters": {
                    "control_evs": False,
                    "control_deferrables": False,
                    "deferrable_urgency_threshold": 0.75,
                },
            },
        }
    )
    observation_names = [
        [
            "electric_vehicle_charger_state",
            "electric_vehicle_soc",
            "electric_vehicle_required_soc_departure",
            "electric_vehicle_departure_time",
            "deferrable_appliance_deferrable_appliance_1_pending",
            "deferrable_appliance_deferrable_appliance_1_running",
            "deferrable_appliance_deferrable_appliance_1_can_start",
            "deferrable_appliance_deferrable_appliance_1_urgency_ratio",
        ]
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[["electric_vehicle_storage", "deferrable_appliance_deferrable_appliance_1"]],
        action_space=[DummySpace([0.0, 0.0], [1.0, 1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    actions = agent.predict([np.array([1.0, 20.0, 80.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=float)])

    assert actions[0] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_rbc_starts_flat_deferrable_appliance_when_urgent():
    agent = RuleBasedPolicy(
        {
            "simulator": {},
            "algorithm": {
                "name": "RuleBasedPolicy",
                "hyperparameters": {
                    "deferrable_urgency_threshold": 0.75,
                    "deferrable_slack_threshold": 0.25,
                    "deferrable_priority_threshold": 0.5,
                },
            },
        }
    )
    observation_names = [
        [
            "deferrable_appliance_deferrable_appliance_1_pending",
            "deferrable_appliance_deferrable_appliance_1_running",
            "deferrable_appliance_deferrable_appliance_1_can_start",
            "deferrable_appliance_deferrable_appliance_1_urgency_ratio",
            "deferrable_appliance_deferrable_appliance_1_slack_ratio",
            "deferrable_appliance_deferrable_appliance_1_priority",
        ]
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[["deferrable_appliance_deferrable_appliance_1"]],
        action_space=[DummySpace([0.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 900},
    )

    actions = agent.predict([np.array([1.0, 0.0, 1.0, 0.8, 0.5, 0.1], dtype=float)])
    assert actions[0][0] == pytest.approx(1.0, abs=1e-6)


def test_rbc_matches_entity_deferrable_appliance_by_action_suffix():
    agent = RuleBasedPolicy({"simulator": {}, "algorithm": {"name": "RuleBasedPolicy"}})
    observation_names = [
        [
            "deferrable_appliance::B1/deferrable_appliance_1::pending",
            "deferrable_appliance::B1/deferrable_appliance_1::running",
            "deferrable_appliance::B1/deferrable_appliance_1::can_start",
            "deferrable_appliance::B1/deferrable_appliance_1::urgency_ratio",
            "deferrable_appliance::B1/deferrable_appliance_1::slack_ratio",
            "deferrable_appliance::B1/deferrable_appliance_1::priority",
            "deferrable_appliance::B1/deferrable_appliance_2::pending",
            "deferrable_appliance::B1/deferrable_appliance_2::running",
            "deferrable_appliance::B1/deferrable_appliance_2::can_start",
            "deferrable_appliance::B1/deferrable_appliance_2::urgency_ratio",
            "deferrable_appliance::B1/deferrable_appliance_2::slack_ratio",
            "deferrable_appliance::B1/deferrable_appliance_2::priority",
        ]
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[["deferrable_appliance_deferrable_appliance_1", "deferrable_appliance_deferrable_appliance_2"]],
        action_space=[DummySpace([0.0, 0.0], [1.0, 1.0])],
        observation_space=[None],
        metadata={"building_names": ["B1"], "seconds_per_time_step": 3600},
    )

    obs = np.array(
        [
            1.0,
            0.0,
            1.0,
            0.8,
            0.2,
            0.9,
            1.0,
            0.0,
            1.0,
            0.1,
            0.9,
            0.1,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])
    assert actions[0] == pytest.approx([1.0, 0.0], abs=1e-6)


def test_rbc_export_artifacts_uses_rule_based_contract(tmp_path):
    agent = make_agent()
    agent._action_labels = [["electric_vehicle_storage"], ["electric_vehicle_storage_2"]]
    metadata = agent.export_artifacts(
        output_dir=str(tmp_path),
        context={
            "config": {
                "bundle": {
                    "require_observations_envelope": True,
                    "artifact_config": {
                        "input_site_key": "site_default",
                        "use_preprocessor": True,
                    },
                    "per_agent_artifact_config": {
                        "1": {
                            "input_site_key": "site_b",
                            "use_preprocessor": False,
                            "require_observations_envelope": False,
                        }
                    },
                }
            }
        },
    )

    assert metadata["format"] == "rule_based"
    assert len(metadata["artifacts"]) == 2

    first = metadata["artifacts"][0]
    second = metadata["artifacts"][1]

    assert first["path"] == "policy_agent_0.json"
    assert first["format"] == "rule_based"
    assert first["config"]["require_observations_envelope"] is True
    assert first["config"]["input_site_key"] == "site_default"
    assert first["config"]["use_preprocessor"] is True

    assert second["path"] == "policy_agent_1.json"
    assert second["format"] == "rule_based"
    assert second["config"]["require_observations_envelope"] is True
    assert second["config"]["input_site_key"] == "site_b"
    assert second["config"]["use_preprocessor"] is False

    for artifact in metadata["artifacts"]:
        exported_path = tmp_path / artifact["path"]
        assert exported_path.exists()
        payload = json.loads(exported_path.read_text(encoding="utf-8"))
        assert isinstance(payload.get("default_actions"), dict)
        assert isinstance(payload.get("rules"), list)


def test_rbc_export_artifacts_respects_agent_index_offset(tmp_path):
    agent = make_agent()
    agent._action_labels = [["electric_vehicle_storage"]]

    metadata = agent.export_artifacts(
        output_dir=str(tmp_path),
        context={"agent_index_offset": 4, "config": {"bundle": {}}},
    )

    artifact = metadata["artifacts"][0]
    assert artifact["agent_index"] == 4
    assert artifact["path"] == "policy_agent_4.json"

    exported_path = tmp_path / artifact["path"]
    assert exported_path.exists()
    payload = json.loads(exported_path.read_text(encoding="utf-8"))
    assert payload["agent_index"] == 4
