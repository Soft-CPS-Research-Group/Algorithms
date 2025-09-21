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
