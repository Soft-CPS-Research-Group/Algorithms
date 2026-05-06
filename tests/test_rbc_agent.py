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
            "deferrable_appliance_washing_machine_1_pending",
            "deferrable_appliance_washing_machine_1_running",
            "deferrable_appliance_washing_machine_1_can_start",
            "deferrable_appliance_washing_machine_1_urgency_ratio",
            "deferrable_appliance_washing_machine_1_slack_ratio",
            "deferrable_appliance_washing_machine_1_priority",
        ]
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[["deferrable_appliance_washing_machine_1"]],
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
            "deferrable_appliance::B1/washer::pending",
            "deferrable_appliance::B1/washer::running",
            "deferrable_appliance::B1/washer::can_start",
            "deferrable_appliance::B1/washer::urgency_ratio",
            "deferrable_appliance::B1/washer::slack_ratio",
            "deferrable_appliance::B1/washer::priority",
            "deferrable_appliance::B1/dryer::pending",
            "deferrable_appliance::B1/dryer::running",
            "deferrable_appliance::B1/dryer::can_start",
            "deferrable_appliance::B1/dryer::urgency_ratio",
            "deferrable_appliance::B1/dryer::slack_ratio",
            "deferrable_appliance::B1/dryer::priority",
        ]
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[["deferrable_appliance_washer", "deferrable_appliance_dryer"]],
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
