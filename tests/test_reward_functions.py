from __future__ import annotations

import pytest

from reward_function.cost_hard_constraint_reward import (
    CostHardConstraintReward,
    CostServiceCostBalancedRewardV3,
    CostServiceCommunityBandRewardV4,
    CostServiceCommunityBatteryValueRewardV43,
    CostServiceCommunityFeasibleServiceRewardV45,
    CostServiceCommunityFeasiblePrecisionRewardV46,
    CostServiceCommunityFeasiblePrecisionRewardV47,
    CostServiceCommunityDeadlineValueRewardV50,
    CostServiceCommunityPeakDeadlineRewardV52,
    CostServiceCommunityPrecisionValueRewardV51,
    CostServiceCommunityServiceBandRewardV42,
    CostServiceCommunitySmoothServiceRewardV44,
    CostServiceCommunityStorageBandRewardV41,
    CostServiceCommunityStorageValueRewardV49,
    CostServiceGuardRewardV2,
)
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


def test_cost_reward_prefers_storing_pv_export_to_avoid_later_import():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        local_cost_weight=1.0,
        export_credit_ratio=0.0,
        community_settlement_cost_weight=0.0,
    )

    export_then_import = reward.calculate(
        [{"net_electricity_consumption": -1.0, "electricity_pricing": 0.10}]
    )[0] + reward.calculate(
        [{"net_electricity_consumption": 1.0, "electricity_pricing": 0.50}]
    )[0]
    store_then_avoid_import = reward.calculate(
        [{"net_electricity_consumption": 0.0, "electricity_pricing": 0.10}]
    )[0] + reward.calculate(
        [{"net_electricity_consumption": 0.0, "electricity_pricing": 0.50}]
    )[0]

    assert store_then_avoid_import > export_then_import
    assert export_then_import == pytest.approx(-0.50)
    assert store_then_avoid_import == pytest.approx(0.0)


def test_community_settlement_prefers_local_self_consumption_to_grid_import():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        local_cost_weight=0.0,
        community_settlement_cost_weight=1.0,
        community_local_price_ratio=0.8,
        community_grid_export_price=0.0,
    )

    export_then_grid_import = sum(
        reward.calculate(
            [
                {"net_electricity_consumption": -1.0, "electricity_pricing": 0.10},
                {"net_electricity_consumption": 0.0, "electricity_pricing": 0.10},
            ]
        )
    ) + sum(
        reward.calculate(
            [
                {"net_electricity_consumption": 0.0, "electricity_pricing": 0.50},
                {"net_electricity_consumption": 1.0, "electricity_pricing": 0.50},
            ]
        )
    )
    local_self_consumption = sum(
        reward.calculate(
            [
                {"net_electricity_consumption": 0.0, "electricity_pricing": 0.10},
                {"net_electricity_consumption": 0.0, "electricity_pricing": 0.10},
            ]
        )
    ) + sum(
        reward.calculate(
            [
                {"net_electricity_consumption": 0.0, "electricity_pricing": 0.50},
                {"net_electricity_consumption": 0.0, "electricity_pricing": 0.50},
            ]
        )
    )

    assert local_self_consumption > export_then_grid_import
    assert export_then_grid_import == pytest.approx(-0.50)
    assert local_self_consumption == pytest.approx(0.0)


def test_named_cost_service_reward_profiles_set_default_weights():
    guard = CostServiceGuardRewardV2(env_metadata={"central_agent": False})
    balanced = CostServiceCostBalancedRewardV3(env_metadata={"central_agent": False})
    community_band = CostServiceCommunityBandRewardV4(env_metadata={"central_agent": False})
    storage_band = CostServiceCommunityStorageBandRewardV41(env_metadata={"central_agent": False})
    service_band = CostServiceCommunityServiceBandRewardV42(env_metadata={"central_agent": False})
    battery_value = CostServiceCommunityBatteryValueRewardV43(env_metadata={"central_agent": False})
    smooth_service = CostServiceCommunitySmoothServiceRewardV44(env_metadata={"central_agent": False})
    feasible_service = CostServiceCommunityFeasibleServiceRewardV45(env_metadata={"central_agent": False})
    feasible_precision = CostServiceCommunityFeasiblePrecisionRewardV46(env_metadata={"central_agent": False})
    feasible_precision_guard = CostServiceCommunityFeasiblePrecisionRewardV47(env_metadata={"central_agent": False})
    storage_value = CostServiceCommunityStorageValueRewardV49(env_metadata={"central_agent": False})
    deadline_value = CostServiceCommunityDeadlineValueRewardV50(env_metadata={"central_agent": False})
    precision_value = CostServiceCommunityPrecisionValueRewardV51(env_metadata={"central_agent": False})
    peak_deadline = CostServiceCommunityPeakDeadlineRewardV52(env_metadata={"central_agent": False})

    assert guard.ev_departure_window_hours == pytest.approx(4.0)
    assert guard.ev_v2g_service_penalty == pytest.approx(200.0)
    assert balanced.ev_departure_window_hours == pytest.approx(3.0)
    assert balanced.ev_v2g_service_penalty == pytest.approx(140.0)
    assert balanced.battery_throughput_penalty == pytest.approx(0.002)
    assert community_band.local_cost_weight == pytest.approx(0.0)
    assert community_band.community_settlement_cost_weight == pytest.approx(1.0)
    assert community_band.ev_over_service_penalty == pytest.approx(40.0)
    assert community_band.battery_throughput_penalty == pytest.approx(0.05)
    assert storage_band.ev_over_service_tolerance == pytest.approx(0.02)
    assert storage_band.ev_over_service_penalty == pytest.approx(120.0)
    assert storage_band.battery_throughput_penalty == pytest.approx(1.0)
    assert service_band.community_settlement_cost_weight == pytest.approx(0.8)
    assert service_band.ev_connected_deficit_penalty == pytest.approx(180.0)
    assert service_band.ev_schedule_deficit_penalty == pytest.approx(900.0)
    assert service_band.ev_over_service_penalty == pytest.approx(350.0)
    assert battery_value.battery_soc_min == pytest.approx(0.0)
    assert battery_value.battery_soc_max == pytest.approx(1.0)
    assert battery_value.battery_throughput_penalty == pytest.approx(0.02)
    assert battery_value.scale_state_penalties_by_time_step is True
    assert battery_value.state_penalty_reference_seconds == pytest.approx(3600.0)
    assert smooth_service.ev_connected_deficit_exponent == pytest.approx(2.0)
    assert smooth_service.ev_schedule_deficit_exponent == pytest.approx(2.0)
    assert smooth_service.ev_departure_deficit_penalty == pytest.approx(1500.0)
    assert smooth_service.ev_v2g_service_penalty == pytest.approx(600.0)
    assert feasible_service.ev_use_effective_charging_power_for_schedule is True
    assert feasible_service.ev_departure_window_penalty_mode == "schedule_deficit"
    assert feasible_service.ev_schedule_deficit_penalty == pytest.approx(720.0)
    assert feasible_precision.ev_over_service_tolerance == pytest.approx(0.02)
    assert feasible_precision.ev_over_service_penalty == pytest.approx(420.0)
    assert feasible_precision.ev_schedule_deficit_cap_soc == pytest.approx(0.08)
    assert feasible_precision.ev_departure_window_shortfall_cap_soc == pytest.approx(0.08)
    assert feasible_precision_guard.ev_over_service_tolerance == pytest.approx(0.01)
    assert feasible_precision_guard.ev_over_service_penalty == pytest.approx(760.0)
    assert feasible_precision_guard.ev_schedule_deficit_cap_soc == pytest.approx(0.08)
    assert storage_value.community_settlement_cost_weight == pytest.approx(1.15)
    assert storage_value.community_peak_import_penalty == pytest.approx(0.0010)
    assert storage_value.battery_throughput_penalty == pytest.approx(0.003)
    assert storage_value.ev_v2g_service_penalty == pytest.approx(700.0)
    assert deadline_value.ev_departure_window_hours == pytest.approx(6.0)
    assert deadline_value.ev_schedule_deficit_penalty == pytest.approx(820.0)
    assert deadline_value.ev_schedule_deficit_exponent == pytest.approx(1.5)
    assert deadline_value.ev_schedule_deficit_cap_soc == pytest.approx(0.10)
    assert deadline_value.ev_departure_missed_penalty == pytest.approx(3000.0)
    assert precision_value.community_settlement_cost_weight == pytest.approx(1.08)
    assert precision_value.ev_schedule_deficit_penalty == pytest.approx(760.0)
    assert precision_value.ev_schedule_deficit_cap_soc == pytest.approx(0.08)
    assert precision_value.ev_over_service_tolerance == pytest.approx(0.03)
    assert precision_value.ev_over_service_penalty == pytest.approx(1200.0)
    assert precision_value.battery_throughput_penalty == pytest.approx(0.004)
    assert peak_deadline.community_settlement_cost_weight == pytest.approx(1.12)
    assert peak_deadline.community_peak_import_penalty == pytest.approx(0.0014)
    assert peak_deadline.community_export_penalty == pytest.approx(0.00035)
    assert peak_deadline.ev_schedule_deficit_penalty == pytest.approx(820.0)
    assert peak_deadline.ev_over_service_tolerance == pytest.approx(0.035)
    assert peak_deadline.ev_over_service_penalty == pytest.approx(720.0)
    assert peak_deadline.battery_throughput_penalty == pytest.approx(0.0035)


def test_cost_hard_constraint_reward_declares_minimal_observation_payload():
    required = set(CostHardConstraintReward.required_observation_names)

    assert "net_electricity_consumption" in required
    assert "electricity_pricing" in required
    assert "electric_vehicles_chargers_dict" in required
    assert "deferrable_appliances_dict" in required
    assert "charging_building_headroom_kw" in required


def test_battery_value_reward_does_not_penalize_empty_idle_storage_by_default():
    reward = CostServiceCommunityBatteryValueRewardV43(env_metadata={"central_agent": False})

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electrical_storage_soc": 0.0,
                "electrical_storage_electricity_consumption": 0.0,
            }
        ]
    )

    assert rewards[0] == pytest.approx(0.0)
    components = reward.get_last_components()["per_agent"][0]
    assert components["battery_soc_min_limit"] == pytest.approx(0.0)
    assert components["battery_soc_below_limit"] == pytest.approx(0.0)
    assert components["battery_safety_penalty"] == pytest.approx(0.0)


def test_battery_value_reward_respects_observed_storage_soc_limits():
    reward = CostServiceCommunityBatteryValueRewardV43(env_metadata={"central_agent": False})

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electrical_storage_soc": 0.1,
                "electrical_storage_soc_min_ratio": 0.2,
                "electrical_storage_soc_max_ratio": 0.9,
                "electrical_storage_electricity_consumption": 0.0,
            }
        ]
    )

    components = reward.get_last_components()["per_agent"][0]
    assert components["battery_soc_min_limit"] == pytest.approx(0.2)
    assert components["battery_soc_max_limit"] == pytest.approx(0.9)
    assert components["battery_soc_below_limit"] == pytest.approx(0.1)
    assert components["battery_safety_penalty"] > 0.0
    assert rewards[0] < 0.0


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


def test_cost_hard_constraint_reward_can_shape_dense_ev_penalties_quadratically():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_connected_deficit_penalty=60.0,
        ev_connected_deficit_exponent=2.0,
        ev_schedule_deficit_penalty=120.0,
        ev_schedule_deficit_exponent=2.0,
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

    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_dense_deficit_penalty"] == pytest.approx(7.2)
    assert components["ev_schedule_deficit_penalty"] == pytest.approx(7.5)
    assert rewards[0] == pytest.approx(-14.7)


def test_cost_hard_constraint_reward_can_schedule_against_effective_phase_headroom():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_connected_deficit_penalty=0.0,
        ev_schedule_deficit_penalty=100.0,
        ev_use_effective_charging_power_for_schedule=True,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "charging_building_headroom_kw": 12.0,
                "charging_phase_L2_headroom_kw": 5.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.2,
                        "required_soc": 0.8,
                        "hours_until_departure": 3.0,
                        "battery_capacity": 60.0,
                        "max_charging_power_kw": 10.0,
                        "phase_connection": "L2",
                    }
                },
            }
        ]
    )

    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_nominal_max_charging_power_kw_sum"] == pytest.approx(10.0)
    assert components["ev_effective_max_charging_power_kw_sum"] == pytest.approx(5.0)
    assert components["ev_schedule_min_soc_required_sum"] == pytest.approx(0.5)
    assert components["ev_schedule_soc_deficit_sum"] == pytest.approx(0.3)
    assert rewards[0] == pytest.approx(-30.0)


def test_cost_hard_constraint_reward_can_use_metadata_charger_phase_map_for_effective_headroom():
    reward = CostHardConstraintReward(
        env_metadata={
            "central_agent": False,
            "buildings": [
                {
                    "name": "Building_15",
                    "charger_phase_map": {"charger_15_2": "L2"},
                }
            ],
        },
        ev_connected_deficit_penalty=0.0,
        ev_schedule_deficit_penalty=100.0,
        ev_use_effective_charging_power_for_schedule=True,
    )

    reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "charging_building_headroom_kw": 12.0,
                "charging_phase_L2_headroom_kw": 5.0,
                "electric_vehicles_chargers_dict": {
                    "charger_15_2": {
                        "connected": True,
                        "battery_soc": 0.2,
                        "required_soc": 0.8,
                        "hours_until_departure": 3.0,
                        "battery_capacity": 60.0,
                        "max_charging_power_kw": 10.0,
                    }
                },
            }
        ]
    )

    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_effective_max_charging_power_kw_sum"] == pytest.approx(5.0)
    assert components["ev_schedule_soc_deficit_sum"] == pytest.approx(0.3)


def test_cost_hard_constraint_reward_refreshes_late_metadata_charger_phase_map():
    metadata = {"central_agent": False, "buildings": [{"name": "Building_15"}]}
    reward = CostHardConstraintReward(
        env_metadata=metadata,
        ev_connected_deficit_penalty=0.0,
        ev_schedule_deficit_penalty=100.0,
        ev_use_effective_charging_power_for_schedule=True,
    )
    metadata["buildings"][0]["charger_phase_map"] = {"charger_15_2": "L2"}

    reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "charging_building_headroom_kw": 12.0,
                "charging_phase_L2_headroom_kw": 5.0,
                "electric_vehicles_chargers_dict": {
                    "charger_15_2": {
                        "connected": True,
                        "battery_soc": 0.2,
                        "required_soc": 0.8,
                        "hours_until_departure": 3.0,
                        "battery_capacity": 60.0,
                        "max_charging_power_kw": 10.0,
                    }
                },
            }
        ]
    )

    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_effective_max_charging_power_kw_sum"] == pytest.approx(5.0)


def test_cost_hard_constraint_reward_uses_static_phase_limit_when_headroom_is_post_action():
    reward = CostHardConstraintReward(
        env_metadata={
            "central_agent": False,
            "buildings": [
                {
                    "name": "Building_15",
                    "charger_phase_map": {"charger_15_2": "L2"},
                    "electrical_service": {
                        "limits": {
                            "total": {"import_kw": 12.0},
                            "per_phase": {"L2": {"import_kw": 5.0}},
                        }
                    },
                }
            ],
        },
        ev_connected_deficit_penalty=0.0,
        ev_schedule_deficit_penalty=100.0,
        ev_use_effective_charging_power_for_schedule=True,
    )

    reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "charging_building_headroom_kw": 0.0,
                "charging_phase_L2_headroom_kw": 0.0,
                "electric_vehicles_chargers_dict": {
                    "charger_15_2": {
                        "connected": True,
                        "battery_soc": 0.2,
                        "required_soc": 0.8,
                        "hours_until_departure": 3.0,
                        "battery_capacity": 60.0,
                        "max_charging_power_kw": 11.0,
                    }
                },
            }
        ]
    )

    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_effective_max_charging_power_kw_sum"] == pytest.approx(5.0)
    assert components["ev_schedule_min_soc_required_sum"] == pytest.approx(0.5)
    assert components["ev_schedule_soc_deficit_sum"] == pytest.approx(0.3)


def test_cost_hard_constraint_reward_can_use_schedule_deficit_in_departure_window():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_connected_deficit_penalty=0.0,
        ev_schedule_deficit_penalty=0.0,
        ev_departure_window_hours=4.0,
        ev_departure_deficit_penalty=100.0,
        ev_use_effective_charging_power_for_schedule=True,
        ev_departure_window_penalty_mode="schedule_deficit",
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "charging_building_headroom_kw": 12.0,
                "charging_phase_L2_headroom_kw": 5.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.2,
                        "required_soc": 0.8,
                        "hours_until_departure": 3.0,
                        "battery_capacity": 60.0,
                        "max_charging_power_kw": 10.0,
                        "phase_connection": "L2",
                    }
                },
            }
        ]
    )

    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_soc_shortfall_beyond_tolerance_sum"] == pytest.approx(0.55)
    assert components["ev_schedule_soc_deficit_sum"] == pytest.approx(0.3)
    assert components["ev_departure_window_penalty"] == pytest.approx(30.0)
    assert rewards[0] == pytest.approx(-30.0)


def test_cost_hard_constraint_reward_can_cap_unrecoverable_ev_training_penalty():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_connected_deficit_penalty=0.0,
        ev_schedule_deficit_penalty=100.0,
        ev_schedule_deficit_exponent=1.0,
        ev_schedule_deficit_cap_soc=0.05,
        ev_departure_window_hours=4.0,
        ev_departure_deficit_penalty=200.0,
        ev_departure_missed_penalty=0.0,
        ev_departure_window_shortfall_cap_soc=0.05,
        ev_use_effective_charging_power_for_schedule=True,
        ev_departure_window_penalty_mode="schedule_deficit",
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "charging_building_headroom_kw": 0.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.2,
                        "required_soc": 0.9,
                        "hours_until_departure": 0.0,
                        "battery_capacity": 60.0,
                        "max_charging_power_kw": 10.0,
                        "phase_connection": "L2",
                    }
                },
            }
        ]
    )

    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_schedule_soc_deficit_sum"] == pytest.approx(0.65)
    assert components["ev_schedule_penalty_deficit_sum"] == pytest.approx(0.05)
    assert components["ev_departure_window_penalty_shortfall_sum"] == pytest.approx(0.05)
    assert components["ev_schedule_deficit_penalty"] == pytest.approx(5.0)
    assert components["ev_departure_window_penalty"] == pytest.approx(10.0)
    assert rewards[0] == pytest.approx(-15.0)


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


def test_cost_hard_constraint_reward_penalizes_v2g_when_ev_is_below_service_target():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_departure_service_tolerance=0.05,
        ev_departure_window_hours=4.0,
        ev_departure_deficit_penalty=0.0,
        ev_v2g_service_penalty=200.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.50,
                        "required_soc": 0.80,
                        "hours_until_departure": 2.0,
                        "last_charged_kwh": -0.10,
                    }
                },
            }
        ]
    )

    # service target is 0.75. Physical schedule risk also applies because only
    # 2h remain at the default 7.4 kW charger limit:
    # schedule min SOC = 0.75 - 7.4*2/75 ~= 0.552667;
    # service risk = 0.25 + 0.052667; urgency = 1.5 inside a 4h window;
    # 0.10 kWh * 200 * (1 + 0.302667) * 1.5 = 39.08.
    assert rewards[0] == pytest.approx(-39.08)
    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_v2g_discharge_kwh_sum"] == pytest.approx(0.10)
    assert components["ev_v2g_service_risk_sum"] == pytest.approx(0.3026666667)
    assert components["ev_v2g_service_abuse_penalty"] == pytest.approx(39.08)
    assert components["ev_service_penalty"] == pytest.approx(39.08)


def test_cost_hard_constraint_reward_can_penalize_any_ev_v2g_discharge():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_departure_service_tolerance=0.05,
        ev_departure_window_hours=4.0,
        ev_v2g_service_penalty=200.0,
        ev_v2g_discharge_penalty=1.5,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.90,
                        "required_soc": 0.80,
                        "hours_until_departure": 4.0,
                        "last_charged_kwh": -2.0,
                    }
                },
            }
        ]
    )

    assert rewards[0] == pytest.approx(-3.0)
    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_v2g_discharge_kwh_sum"] == pytest.approx(2.0)
    assert components["ev_v2g_service_risk_sum"] == pytest.approx(0.0)
    assert components["ev_v2g_service_abuse_penalty"] == pytest.approx(0.0)
    assert components["ev_v2g_discharge_penalty_amount"] == pytest.approx(3.0)
    assert components["ev_service_penalty"] == pytest.approx(3.0)


def test_cost_hard_constraint_reward_penalizes_ev_over_service_above_target_band():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_over_service_tolerance=0.05,
        ev_over_service_penalty=10.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.90,
                        "required_soc": 0.80,
                        "hours_until_departure": 2.0,
                    }
                },
            }
        ]
    )

    assert rewards[0] == pytest.approx(-0.5)
    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_soc_surplus_sum"] == pytest.approx(0.10)
    assert components["ev_soc_over_service_sum"] == pytest.approx(0.05)
    assert components["ev_over_service_penalty_amount"] == pytest.approx(0.5)
    assert components["ev_service_penalty"] == pytest.approx(0.5)


def test_cost_hard_constraint_reward_allows_v2g_when_ev_service_target_is_met():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_departure_service_tolerance=0.05,
        ev_v2g_service_penalty=200.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.78,
                        "required_soc": 0.80,
                        "hours_until_departure": 1.0,
                        "last_charged_kwh": -0.10,
                    }
                },
            }
        ]
    )

    assert rewards[0] == pytest.approx(0.0)
    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_v2g_discharge_kwh_sum"] == pytest.approx(0.10)
    assert components["ev_v2g_service_abuse_penalty"] == pytest.approx(0.0)


def test_cost_hard_constraint_reward_ignores_unknown_ev_departure_sentinel_from_entity_dict():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_departure_service_tolerance=0.05,
        ev_departure_missed_penalty=250.0,
        ev_connected_deficit_penalty=30.0,
        ev_schedule_deficit_penalty=120.0,
        ev_v2g_service_penalty=200.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electric_vehicles_chargers_dict": {
                    "charger_a": {
                        "connected": True,
                        "battery_soc": 0.70,
                        "required_soc": 0.82,
                        "hours_until_departure": 0.0,
                        "departure_time": -1.0,
                        "connected_ev_departure_time_step": -1.0,
                        "last_charged_kwh": -0.10,
                    }
                },
            }
        ]
    )

    assert rewards[0] == pytest.approx(0.0)
    components = reward.get_last_components()["per_agent"][0]
    assert components["ev_soc_deficit_sum"] == pytest.approx(0.12)
    assert components["ev_service_penalty"] == pytest.approx(0.0)
    assert components["ev_v2g_service_abuse_penalty"] == pytest.approx(0.0)
    assert components["ev_departure_missed_count"] == pytest.approx(0.0)
    assert components["ev_departure_missed_penalty_amount"] == pytest.approx(0.0)


def test_cost_hard_constraint_reward_ignores_unknown_ev_departure_sentinel_from_flat_names():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        ev_departure_service_tolerance=0.05,
        ev_departure_missed_penalty=250.0,
        ev_connected_deficit_penalty=30.0,
    )

    rewards = reward.calculate(
        [
            {
                "net_electricity_consumption": 0.0,
                "electricity_pricing": 0.0,
                "electric_vehicle_charger_charger_10_1_connected_state": 1.0,
                "connected_electric_vehicle_at_charger_charger_10_1_departure_time": 0.0,
                "connected_electric_vehicle_at_charger_charger_10_1_departure_time_step": -1.0,
                "connected_electric_vehicle_at_charger_charger_10_1_required_soc_departure": 0.82,
                "connected_electric_vehicle_at_charger_charger_10_1_soc": 0.70,
            }
        ]
    )

    assert rewards[0] == pytest.approx(0.0)


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


def test_cost_hard_constraint_reward_can_train_on_community_settlement_cost():
    reward = CostHardConstraintReward(
        env_metadata={"central_agent": False},
        local_cost_weight=0.0,
        community_settlement_cost_weight=1.0,
        community_local_price_ratio=0.8,
        community_grid_export_price=0.0,
    )

    rewards = reward.calculate(
        [
            {"net_electricity_consumption": 2.0, "electricity_pricing": 0.5},
            {"net_electricity_consumption": -1.0, "electricity_pricing": 0.5},
        ]
    )

    assert rewards == pytest.approx([-0.9, 0.4])
    components = reward.get_last_components()
    assert components["per_agent"][0]["community_settlement_cost"] == pytest.approx(0.9)
    assert components["per_agent"][0]["community_local_import_energy"] == pytest.approx(1.0)
    assert components["per_agent"][1]["community_settlement_cost"] == pytest.approx(-0.4)
    assert components["per_agent"][1]["community_local_export_energy"] == pytest.approx(1.0)
    assert components["community"]["community_settlement_cost_total"] == pytest.approx(0.5)
    assert components["community"]["community_local_traded_energy"] == pytest.approx(1.0)


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
