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
