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
            "charging_constraint_violation_kwh": 1.0,
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
    # penalties: service(60) + outage(120) + battery_soc_violation(0.1*30=3)
    # + EV service shortfall((0.9 - 0.05 - 0.7)*120=18)
    assert rewards[0] == pytest.approx(-202.0)

    components = reward.get_last_components()
    assert components["per_agent"][0]["local_cost_reward"] == pytest.approx(-1.0)
    assert components["per_agent"][0]["service_violation_penalty"] == pytest.approx(60.0)
    assert components["per_agent"][0]["power_outage_penalty"] == pytest.approx(120.0)
    assert components["per_agent"][0]["battery_safety_penalty"] == pytest.approx(3.0)
    assert components["per_agent"][0]["ev_service_penalty"] == pytest.approx(18.0)
    assert components["per_agent"][0]["ev_soc_deficit_sum"] == pytest.approx(0.2)
    assert components["per_agent"][0]["ev_soc_shortfall_beyond_tolerance_sum"] == pytest.approx(0.15)
    assert components["per_agent"][0]["ev_soc_strict_gap_within_tolerance_sum"] == pytest.approx(0.05)
    assert components["per_agent"][0]["ev_departure_window_penalty"] == pytest.approx(18.0)
    assert components["per_agent"][0]["reward_total"] == pytest.approx(-202.0)


def test_cost_hard_constraint_reward_penalizes_battery_throughput():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        battery_throughput_penalty=2.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electrical_storage_electricity_consumption": -1.5,
            }
        ]
    )

    assert rewards[0] == pytest.approx(-3.0)


def test_cost_hard_constraint_reward_ev_schedule_penalty_is_zero_when_departure_is_feasible():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_connected_deficit_penalty=0.0,
        ev_schedule_deficit_penalty=120.0,
        ev_departure_window_hours=1.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.2,
                        "required_soc": 0.8,
                        "hours_until_departure": 10.0,
                        "battery_capacity": 60.0,
                        "max_charging_power_kw": 7.4,
                    }
                },
            }
        ]
    )

    assert rewards[0] == pytest.approx(0.0)
    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_soc_deficit_sum"] == pytest.approx(0.6)
    assert components["ev_schedule_soc_deficit_sum"] == pytest.approx(0.0)
    assert components["ev_service_penalty"] == pytest.approx(0.0)


def test_cost_hard_constraint_reward_ev_schedule_penalty_grows_when_departure_becomes_infeasible():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_connected_deficit_penalty=0.0,
        ev_schedule_deficit_penalty=120.0,
        ev_departure_window_hours=1.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.2,
                        "required_soc": 0.8,
                        "hours_until_departure": 3.0,
                        "battery_capacity": 60.0,
                        "max_charging_power_kw": 6.0,
                    }
                },
            }
        ]
    )

    # Max remaining SOC gain is 6kW*3h/60kWh = 0.3, so minimum acceptable SOC now is 0.45.
    assert rewards[0] == pytest.approx(-30.0)
    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_schedule_min_soc_required_sum"] == pytest.approx(0.45)
    assert components["ev_schedule_soc_deficit_sum"] == pytest.approx(0.25)
    assert components["ev_schedule_deficit_penalty"] == pytest.approx(30.0)


def test_cost_hard_constraint_reward_uses_observed_storage_soc_limits():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        battery_soc_min=0.1,
        battery_soc_max=0.95,
        battery_soc_violation_penalty=30.0,
        use_observed_storage_soc_limits=True,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "storage::Building_1/electrical_storage::soc": 0.0,
                "storage::Building_1/electrical_storage::soc_min_ratio": 0.0,
                "storage::Building_1/electrical_storage::soc_max_ratio": 1.0,
            }
        ]
    )

    assert rewards[0] == pytest.approx(0.0)
    components = reward.get_last_components()["per_agent"][0]
    assert components["battery_soc_min_limit"] == pytest.approx(0.0)
    assert components["battery_soc_max_limit"] == pytest.approx(1.0)
    assert components["battery_safety_penalty"] == pytest.approx(0.0)


def test_cost_hard_constraint_reward_penalizes_observed_storage_soc_max_limit():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        battery_soc_min=0.0,
        battery_soc_max=1.0,
        battery_soc_violation_penalty=30.0,
        use_observed_storage_soc_limits=True,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electrical_storage_soc": 0.9,
                "electrical_storage_soc_min_ratio": 0.0,
                "electrical_storage_soc_max_ratio": 0.8,
            }
        ]
    )

    assert rewards[0] == pytest.approx(-3.0)
    components = reward.get_last_components()["per_agent"][0]
    assert components["battery_soc_above_limit"] == pytest.approx(0.1)
    assert components["battery_soc_violation_penalty_amount"] == pytest.approx(3.0)


def test_cost_hard_constraint_reward_can_scale_dense_state_penalties_by_timestep():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False, "seconds_per_time_step": 15},
        power_outage_penalty=120.0,
        ev_departure_window_hours=1.0,
        ev_departure_deficit_penalty=120.0,
        battery_soc_min=0.1,
        battery_soc_max=0.95,
        battery_soc_violation_penalty=30.0,
        scale_state_penalties_by_time_step=True,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
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
    )

    scale = 15.0 / 3600.0
    # outage(120) + battery SOC(0.1*30=3) + EV service shortfall(0.15*120=18), all dense/state-scaled
    assert rewards[0] == pytest.approx(-(120.0 + 3.0 + 18.0) * scale)
    components = reward.get_last_components()
    assert components["per_agent"][0]["state_penalty_scale"] == pytest.approx(scale)


def test_cost_hard_constraint_reward_penalizes_deferrable_service_from_dict():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        deferrable_deadline_missed_penalty=100.0,
        deferrable_urgency_penalty=10.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "deferrable_appliances_dict": {
                    "deferrable_appliance_1": {
                        "pending": 1.0,
                        "running": 0.0,
                        "can_start": 1.0,
                        "deadline_missed": 1.0,
                        "urgency_ratio": 0.8,
                        "priority": 2.0,
                        "remaining_energy_kwh": 5.0,
                        "cycle_energy_kwh": 10.0,
                    }
                },
            }
        ]
    )

    # deadline missed: 1*100*2=200; urgency: 0.8*(5/10)*10*2=8
    assert rewards[0] == pytest.approx(-208.0)
    components = reward.get_last_components()["per_agent"][0]
    assert components["deferrable_deadline_missed_penalty_amount"] == pytest.approx(200.0)
    assert components["deferrable_urgency_penalty_amount"] == pytest.approx(8.0)
    assert components["deferrable_pending_count"] == pytest.approx(1.0)


def test_cost_hard_constraint_reward_reads_flat_deferrable_observations():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        deferrable_urgency_penalty=10.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "deferrable_appliance_deferrable_appliance_1_pending": 1.0,
                "deferrable_appliance_deferrable_appliance_1_running": 0.0,
                "deferrable_appliance_deferrable_appliance_1_can_start": 1.0,
                "deferrable_appliance_deferrable_appliance_1_urgency_ratio": 0.5,
                "deferrable_appliance_deferrable_appliance_1_priority": 1.0,
                "deferrable_appliance_deferrable_appliance_1_remaining_energy_kwh": 2.0,
                "deferrable_appliance_deferrable_appliance_1_cycle_energy_kwh": 4.0,
            }
        ]
    )

    assert rewards[0] == pytest.approx(-2.5)


def test_cost_hard_constraint_reward_reads_flat_namespaced_ev_observations():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_departure_window_hours=1.0,
        ev_departure_deficit_penalty=120.0,
        ev_connected_deficit_penalty=30.0,
    )

    observations = [
        {
            "net_electricity_consumption": 0.0,
            "electricity_pricing": 0.0,
            "electric_vehicle_charger_charger_1_1_connected_state": 1.0,
            "connected_electric_vehicle_at_charger_charger_1_1_departure_time": 0.5,
            "connected_electric_vehicle_at_charger_charger_1_1_required_soc_departure": 0.9,
            "connected_electric_vehicle_at_charger_charger_1_1_soc": 0.7,
        }
    ]

    rewards = reward.calculate(observations)

    # dense penalty: 0.2 * 30 / 1h = 6; departure-window service shortfall: 0.15 * 120 = 18
    assert rewards[0] == pytest.approx(-24.0)


def test_cost_hard_constraint_reward_treats_within_service_tolerance_as_acceptable():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_departure_service_tolerance=0.05,
        ev_departure_window_hours=1.0,
        ev_departure_deficit_penalty=120.0,
        ev_departure_missed_penalty=250.0,
        ev_connected_deficit_penalty=30.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.76,
                        "required_soc": 0.80,
                        "hours_until_departure": 0.0,
                    }
                },
            }
        ]
    )

    # Strict deficit still gives a small dense learning signal, but the hard
    # departure-window/missed penalties are zero because the user tolerance is met.
    assert rewards[0] == pytest.approx(-1.2)
    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_soc_deficit_sum"] == pytest.approx(0.04)
    assert components["ev_soc_shortfall_beyond_tolerance_sum"] == pytest.approx(0.0)
    assert components["ev_departure_window_penalty"] == pytest.approx(0.0)
    assert components["ev_departure_missed_penalty_amount"] == pytest.approx(0.0)


def test_cost_hard_constraint_reward_supports_central_agent_output():
    reward = CostHardConstraintReward(env_metadata={"central_agent": True})

    observations = [
        {"net_electricity_consumption": 1.0, "electricity_pricing": 0.2},
        {"net_electricity_consumption": 2.0, "electricity_pricing": 0.2},
    ]

    rewards = reward.calculate(observations)

    assert len(rewards) == 1
    assert rewards[0] == pytest.approx(-0.6)


def test_cost_hard_constraint_reward_logs_community_components():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        community_import_penalty=0.5,
    )

    rewards = reward.calculate(
        [
            {"net_electricity_consumption": 2.0, "electricity_pricing": 0.0},
            {"net_electricity_consumption": -1.0, "electricity_pricing": 0.0},
        ]
    )

    assert rewards == pytest.approx([-1.0, -1.0])
    components = reward.get_last_components()
    assert components["community"]["community_import_energy"] == pytest.approx(2.0)
    assert components["community"]["community_export_energy"] == pytest.approx(0.0)
    assert components["community"]["community_import_penalty"] == pytest.approx(1.0)
    assert components["per_agent"][0]["community_import_penalty"] == pytest.approx(1.0)


def test_cost_hard_constraint_reward_can_share_community_peak_penalty_between_agents():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        community_import_penalty=0.5,
        community_peak_import_penalty=0.25,
        community_penalty_divide_by_agents=True,
    )

    rewards = reward.calculate(
        [
            {"net_electricity_consumption": 2.0, "electricity_pricing": 0.0},
            {"net_electricity_consumption": 1.0, "electricity_pricing": 0.0},
        ]
    )

    # linear: 3*0.5=1.5; peak: 3^2*0.25=2.25; shared per agent: 3.75/2=1.875
    assert rewards == pytest.approx([-1.875, -1.875])
    components = reward.get_last_components()
    assert components["community"]["community_import_linear_penalty"] == pytest.approx(1.5)
    assert components["community"]["community_peak_import_penalty"] == pytest.approx(2.25)
    assert components["community"]["community_shared_penalty"] == pytest.approx(3.75)
    assert components["community"]["community_shared_penalty_per_agent"] == pytest.approx(1.875)
    assert components["per_agent"][0]["community_import_penalty"] == pytest.approx(1.875)
