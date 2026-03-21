from __future__ import annotations

import pytest

from reward_function.cost_hard_constraint_reward import CostHardConstraintReward
from reward_function.cost_minimization_reward import CostMinimizationReward


def test_cost_minimization_reward_matches_import_export_cost_math():
    reward = CostMinimizationReward(env_metadata={"central_agent": False}, export_credit_ratio=0.8)

    observations = [
        {"net_electricity_consumption": 2.0, "electricity_pricing": 0.5},
        {"net_electricity_consumption": -1.0, "electricity_pricing": 0.5},
    ]

    rewards = reward.calculate(observations)

    assert rewards[0] == pytest.approx(-1.0)
    assert rewards[1] == pytest.approx(0.4)


def test_cost_hard_constraint_reward_penalizes_ev_and_grid_constraints():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        export_credit_ratio=0.8,
        grid_violation_penalty=60.0,
        power_outage_penalty=120.0,
        ev_departure_window_hours=1.0,
        ev_departure_deficit_penalty=120.0,
        battery_soc_min=0.1,
        battery_soc_max=0.95,
        battery_soc_violation_penalty=30.0,
    )

    observations = [
        {
            "net_electricity_consumption": 2.0,
            "electricity_pricing": 0.5,
            "electrical_service_violation_kwh": 1.0,
            "power_outage": 1.0,
            "electrical_storage_soc": 0.0,
            "electric_vehicles_chargers_dict": {
                "charger_a": {
                    "connected": True,
                    "battery_soc": 0.7,
                    "required_soc": 0.9,
                    "hours_until_departure": 0.5,
                }
            },
        }
    ]

    rewards = reward.calculate(observations)

    # cost term: -1.0
    # penalties: service(60) + outage(120) + battery_soc_violation(0.1*30=3) + ev_deficit(0.2*120=24)
    assert rewards[0] == pytest.approx(-208.0)


def test_cost_hard_constraint_reward_supports_central_agent_output():
    reward = CostHardConstraintReward(env_metadata={"central_agent": True})

    observations = [
        {"net_electricity_consumption": 1.0, "electricity_pricing": 0.2},
        {"net_electricity_consumption": 2.0, "electricity_pricing": 0.2},
    ]

    rewards = reward.calculate(observations)

    assert len(rewards) == 1
    assert rewards[0] == pytest.approx(-0.6)
