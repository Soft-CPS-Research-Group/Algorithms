import numpy as np
import pytest

from algorithms.agents.baseline_policies import (
    NormalNoBatteryPolicy,
    NormalPolicy,
    RBCBasicPolicy,
    RBCCommunityPolicy,
    RBCSmartPolicy,
    RandomPolicy,
)


class DummySpace:
    def __init__(self, low, high):
        self.low = low
        self.high = high


def _attach(agent, observation_names, action_names):
    agent.attach_environment(
        observation_names=[observation_names],
        action_names=[action_names],
        action_space=[DummySpace([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )


def _base_observation_names():
    return [
        "district__electricity_pricing",
        "district__electricity_pricing_predicted_1",
        "district__electricity_pricing_predicted_2",
        "district__electricity_pricing_predicted_3",
        "solar_generation",
        "load_power_kw",
        "import_power_kw",
        "charging_building_headroom_kw",
        "electrical_storage_soc",
        "charger::Building_1/charger_1_1::connected_state",
        "charger::Building_1/charger_1_1::connected_ev_soc",
        "charger::Building_1/charger_1_1::connected_ev_required_soc_departure",
        "charger::Building_1/charger_1_1::connected_ev_battery_capacity_kwh",
        "charger::Building_1/charger_1_1::hours_until_departure",
        "deferrable_appliance::Building_1/deferrable_appliance_1::pending",
        "deferrable_appliance::Building_1/deferrable_appliance_1::running",
        "deferrable_appliance::Building_1/deferrable_appliance_1::can_start",
        "deferrable_appliance::Building_1/deferrable_appliance_1::deadline_missed",
        "deferrable_appliance::Building_1/deferrable_appliance_1::urgency_ratio",
        "deferrable_appliance::Building_1/deferrable_appliance_1::slack_ratio",
    ]


def _action_names():
    return [
        "electrical_storage",
        "electric_vehicle_storage_charger_1_1",
        "deferrable_appliance_deferrable_appliance_1",
    ]


def test_random_policy_samples_every_action_within_bounds():
    agent = RandomPolicy({"training": {"seed": 123}, "algorithm": {"name": "RandomPolicy", "hyperparameters": {}}})
    _attach(agent, ["hour"], _action_names())

    actions = agent.predict([np.array([12.0], dtype=float)])

    assert len(actions) == 1
    assert len(actions[0]) == 3
    assert -1.0 <= actions[0][0] <= 1.0
    assert -1.0 <= actions[0][1] <= 1.0
    assert 0.0 <= actions[0][2] <= 1.0


def test_random_policy_respects_phase_headroom_when_available():
    agent = RandomPolicy(
        {
            "simulator": {"dataset_path": "datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json"},
            "training": {"seed": 123},
            "algorithm": {"name": "RandomPolicy", "hyperparameters": {}},
        }
    )
    agent.attach_environment(
        observation_names=[["charging_building_headroom_kw", "charging_phase_L1_headroom_kw"]],
        action_names=[["electric_vehicle_storage_charger_15_1"]],
        action_space=[DummySpace([0.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_15"], "seconds_per_time_step": 3600},
    )

    actions = agent.predict([np.array([10.0, 0.74], dtype=float)])

    assert 0.0 <= actions[0][0] <= 0.74 / 7.4


def test_random_policy_respects_observed_storage_soc_max():
    class _Rng:
        def uniform(self, low, high):
            return 0.8

    agent = RandomPolicy({"training": {"seed": 123}, "algorithm": {"name": "RandomPolicy", "hyperparameters": {}}})
    agent._rng = _Rng()
    agent.attach_environment(
        observation_names=[["electrical_storage_soc", "electrical_storage_soc_max_ratio"]],
        action_names=[["electrical_storage"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    actions = agent.predict([np.array([0.9, 0.9], dtype=float)])

    assert actions[0][0] == pytest.approx(0.0)


def test_normal_policy_controls_storage_ev_and_deferrable():
    agent = NormalPolicy({"algorithm": {"name": "NormalPolicy", "hyperparameters": {}}})
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.2,
            0.3,
            0.5,
            0.5,
            5.0,
            1.0,
            0.0,
            10.0,
            0.4,
            1.0,
            0.2,
            0.8,
            60.0,
            3.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.1,
            0.9,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] > 0.0
    assert actions[0][1] > 0.0
    assert actions[0][2] == pytest.approx(1.0)


def test_normal_policy_prefers_pv_power_kw_over_legacy_solar_generation():
    agent = NormalPolicy({"algorithm": {"name": "NormalPolicy", "hyperparameters": {}}})
    agent.attach_environment(
        observation_names=[["pv_power_kw", "solar_generation", "load_power_kw", "electrical_storage_soc"]],
        action_names=[["electrical_storage"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 15},
    )

    actions = agent.predict([np.array([5.0, 0.01, 1.0, 0.5], dtype=float)])

    assert actions[0][0] > 0.0


def test_normal_policy_charges_to_full_target_not_only_required_soc():
    agent = NormalPolicy({"algorithm": {"name": "NormalPolicy", "hyperparameters": {}}})
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.2,
            0.3,
            0.4,
            0.5,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.8,
            0.8,
            60.0,
            3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] > 0.0


def test_normal_no_battery_policy_leaves_storage_idle():
    agent = NormalNoBatteryPolicy({"algorithm": {"name": "NormalNoBatteryPolicy", "hyperparameters": {}}})
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.2,
            0.3,
            0.4,
            0.5,
            5.0,
            1.0,
            0.0,
            10.0,
            0.5,
            1.0,
            0.2,
            0.8,
            60.0,
            3.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.1,
            0.9,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] == pytest.approx(0.0)
    assert actions[0][1] > 0.0
    assert actions[0][2] == pytest.approx(1.0)


def test_normal_policy_never_uses_ev_v2g():
    agent = NormalPolicy({"algorithm": {"name": "NormalPolicy", "hyperparameters": {}}})
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.90,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            8.0,
            10.0,
            0.5,
            1.0,
            0.95,
            0.5,
            60.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] >= 0.0


def test_normal_policy_clips_storage_to_three_phase_headroom():
    agent = NormalPolicy(
        {
            "simulator": {"dataset_path": "datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json"},
            "algorithm": {"name": "NormalPolicy", "hyperparameters": {}},
        }
    )
    observation_names = [
        "solar_generation",
        "load_power_kw",
        "electrical_storage_soc",
        "charging_building_headroom_kw",
        "charging_phase_L1_headroom_kw",
        "charging_phase_L2_headroom_kw",
        "charging_phase_L3_headroom_kw",
    ]
    agent.attach_environment(
        observation_names=[observation_names],
        action_names=[["electrical_storage"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_15"], "seconds_per_time_step": 3600},
    )

    actions = agent.predict([np.array([5.0, 1.0, 0.5, 10.0, 0.50, 10.0, 10.0], dtype=float)])

    assert actions[0][0] == pytest.approx((0.50 * 3.0) / 5.0, abs=1e-6)


def test_rbc_smart_policy_can_charge_storage_from_local_pv_surplus_without_import_headroom():
    agent = RBCSmartPolicy(
        {
            "simulator": {"dataset_path": "datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json"},
            "algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {"pv_charge_rate": 0.5}},
        }
    )
    observation_names = [
        "pv_power_kw",
        "load_power_kw",
        "electrical_storage_soc",
        "charging_building_headroom_kw",
        "charging_phase_L1_headroom_kw",
        "charging_phase_L2_headroom_kw",
        "charging_phase_L3_headroom_kw",
    ]
    agent.attach_environment(
        observation_names=[observation_names],
        action_names=[["electrical_storage"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_15"], "seconds_per_time_step": 3600},
    )

    actions = agent.predict([np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)])

    assert actions[0][0] == pytest.approx(1.0 / 5.0, abs=1e-6)


def test_rbc_smart_policy_price_charge_is_clipped_by_physical_headroom():
    agent = RBCSmartPolicy(
        {
            "simulator": {"dataset_path": "datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json"},
            "algorithm": {
                "name": "RBCSmartPolicy",
                "hyperparameters": {"price_charge_rate": 0.5, "pv_charge_rate": 0.0},
            },
        }
    )
    observation_names = [
        "district__electricity_pricing",
        "district__electricity_pricing_predicted_1",
        "district__electricity_pricing_predicted_2",
        "district__electricity_pricing_predicted_3",
        "pv_power_kw",
        "load_power_kw",
        "electrical_storage_soc",
        "charging_building_headroom_kw",
        "charging_phase_L1_headroom_kw",
        "charging_phase_L2_headroom_kw",
        "charging_phase_L3_headroom_kw",
    ]
    agent.attach_environment(
        observation_names=[observation_names],
        action_names=[["electrical_storage"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_15"], "seconds_per_time_step": 3600},
    )

    obs = np.array([0.10, 0.40, 0.50, 0.60, 2.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
    actions = agent.predict([obs])

    assert actions[0][0] == pytest.approx(1.0 / 5.0, abs=1e-6)


def test_rbc_smart_policy_does_not_charge_storage_at_observed_soc_max():
    agent = RBCSmartPolicy(
        {
            "simulator": {"dataset_path": "datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json"},
            "algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {"pv_charge_rate": 0.5}},
        }
    )
    observation_names = [
        "pv_power_kw",
        "load_power_kw",
        "electrical_storage_soc",
        "electrical_storage_soc_min_ratio",
        "electrical_storage_soc_max_ratio",
        "charging_building_headroom_kw",
    ]
    agent.attach_environment(
        observation_names=[observation_names],
        action_names=[["electrical_storage"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    actions = agent.predict([np.array([4.0, 1.0, 0.85, 0.10, 0.85, 10.0], dtype=float)])

    assert actions[0][0] == pytest.approx(0.0)


def test_rbc_smart_policy_does_not_discharge_storage_at_observed_soc_min():
    agent = RBCSmartPolicy(
        {
            "algorithm": {
                "name": "RBCSmartPolicy",
                "hyperparameters": {"storage_min_soc": 0.0, "storage_price_discharge_soc_floor": 0.0},
            }
        }
    )
    observation_names = [
        "district__electricity_pricing",
        "district__electricity_pricing_predicted_1",
        "district__electricity_pricing_predicted_2",
        "district__electricity_pricing_predicted_3",
        "pv_power_kw",
        "load_power_kw",
        "import_power_kw",
        "charging_building_headroom_kw",
        "electrical_storage_soc",
        "electrical_storage_soc_min_ratio",
        "electrical_storage_soc_max_ratio",
    ]
    agent.attach_environment(
        observation_names=[observation_names],
        action_names=[["electrical_storage"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    actions = agent.predict([np.array([0.60, 0.10, 0.20, 0.30, 0.0, 2.0, 9.0, 1.0, 0.20, 0.20, 0.95], dtype=float)])

    assert actions[0][0] == pytest.approx(0.0)


def test_rbc_basic_policy_uses_price_for_storage_ev_and_deferrable():
    agent = RBCBasicPolicy(
        {
            "algorithm": {
                "name": "RBCBasicPolicy",
                "hyperparameters": {"flex_trickle_charge": 0.0},
            }
        }
    )
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.10,
            0.40,
            0.50,
            0.60,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.2,
            0.8,
            60.0,
            6.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.1,
            0.9,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] > 0.0
    assert actions[0][1] > 0.0
    assert actions[0][2] == pytest.approx(1.0)


def test_rbc_basic_policy_never_uses_ev_v2g():
    agent = RBCBasicPolicy({"algorithm": {"name": "RBCBasicPolicy", "hyperparameters": {"flex_trickle_charge": 0.0}}})
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.90,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            8.0,
            10.0,
            0.5,
            1.0,
            0.95,
            0.5,
            60.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] >= 0.0


def test_rbc_basic_policy_ignores_long_price_forecasts():
    agent = RBCBasicPolicy({"algorithm": {"name": "RBCBasicPolicy", "hyperparameters": {}}})
    observation_names = [
        "district__electricity_pricing",
        "district__electricity_pricing_predicted_1",
        "district__electricity_pricing_predicted_2",
        "district__electricity_pricing_predicted_3",
        "district__electricity_pricing_predicted_4",
        "electrical_storage_soc",
    ]
    agent.attach_environment(
        observation_names=[observation_names],
        action_names=[["electrical_storage"]],
        action_space=[DummySpace([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    actions = agent.predict([np.array([10.0, 9.0, 10.0, 11.0, 1.0, 0.5], dtype=float)])

    assert actions[0][0] == pytest.approx(0.0)


def test_rbc_basic_policy_keeps_minimum_ev_service_rate_when_not_cheap():
    agent = RBCBasicPolicy(
        {
            "algorithm": {
                "name": "RBCBasicPolicy",
                "hyperparameters": {
                    "flex_trickle_charge": 0.0,
                    "ev_service_margin_rate": 0.0,
                    "ev_deadline_buffer_hours": 0.0,
                    "ev_service_lookahead_hours": 0.0,
                },
            }
        }
    )
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
            [
                0.20,
                0.10,
                0.20,
                0.30,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.2,
            0.8,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] == pytest.approx((0.8 - 0.2) * 60.0 / (10.0 * 7.4), abs=1e-6)


def test_rbc_basic_policy_can_use_service_target_above_required_soc():
    agent = RBCBasicPolicy(
        {
            "algorithm": {
                "name": "RBCBasicPolicy",
                "hyperparameters": {
                    "ev_service_target_soc": 1.0,
                    "ev_service_margin_rate": 0.0,
                    "ev_deadline_buffer_hours": 0.0,
                    "ev_service_floor_rate": 0.0,
                    "ev_service_lookahead_hours": 0.0,
                },
            }
        }
    )
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.20,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.8,
            0.8,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] == pytest.approx((1.0 - 0.8) * 60.0 / (10.0 * 7.4), abs=1e-6)


def test_rbc_smart_policy_ev_required_rate_uses_connected_battery_capacity():
    agent = RBCSmartPolicy({"algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {}}})
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.20,
            np.nan,
            np.nan,
            np.nan,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.2,
            0.8,
            20.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )
    low_capacity_action = agent.predict([obs])[0][1]

    obs[12] = 80.0
    high_capacity_action = agent.predict([obs])[0][1]

    assert high_capacity_action > low_capacity_action


def test_rbc_smart_policy_default_service_rate_is_not_bau_max_charge():
    agent = RBCSmartPolicy({"algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {}}})
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.20,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.7,
            0.8,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert 0.0 < actions[0][1] < 0.5


def test_rbc_smart_policy_keeps_required_ev_rate_under_grid_stress():
    agent = RBCSmartPolicy(
        {
            "algorithm": {
                "name": "RBCSmartPolicy",
                "hyperparameters": {
                    "flex_trickle_charge": 0.0,
                    "ev_service_margin_rate": 0.0,
                    "ev_deadline_buffer_hours": 0.0,
                    "ev_service_lookahead_hours": 0.0,
                },
            }
        }
    )
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.60,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            8.0,
            1.0,
            0.5,
            1.0,
            0.2,
            0.8,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] == pytest.approx((0.8 - 0.2) * 60.0 / (10.0 * 7.4), abs=1e-6)


def test_rbc_smart_policy_uses_simulator_min_required_ev_action():
    agent = RBCSmartPolicy(
        {
            "algorithm": {
                "name": "RBCSmartPolicy",
                "hyperparameters": {
                    "flex_trickle_charge": 0.0,
                    "ev_service_floor_rate": 0.0,
                    "ev_service_margin_rate": 0.0,
                    "ev_service_lookahead_hours": 0.0,
                    "ev_deadline_buffer_hours": 0.0,
                },
            }
        }
    )
    names = _base_observation_names() + [
        "charger::Building_1/charger_1_1::min_required_action_normalized",
        "charger::Building_1/charger_1_1::available_charge_action_normalized",
    ]
    _attach(agent, names, _action_names())
    obs = np.array(
        [
            0.60,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            8.0,
            10.0,
            0.5,
            1.0,
            0.78,
            0.80,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.40,
            1.00,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] == pytest.approx(0.40)


def test_rbc_smart_policy_respects_simulator_ev_charge_capacity():
    agent = RBCSmartPolicy({"algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {}}})
    names = _base_observation_names() + [
        "charger::Building_1/charger_1_1::can_charge",
        "charger::Building_1/charger_1_1::available_charge_action_normalized",
    ]
    _attach(agent, names, _action_names())
    obs = np.array(
        [
            0.10,
            0.40,
            0.50,
            0.60,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.2,
            0.8,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.20,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] == pytest.approx(0.20)

    obs[-2] = 0.0
    actions = agent.predict([obs])
    assert actions[0][1] == pytest.approx(0.0)


def test_rbc_smart_policy_respects_simulator_ev_discharge_capacity():
    agent = RBCSmartPolicy(
        {
            "algorithm": {
                "name": "RBCSmartPolicy",
                "hyperparameters": {"allow_v2g": True, "ev_v2g_discharge_rate": 0.50},
            }
        }
    )
    names = _base_observation_names() + [
        "charger::Building_1/charger_1_1::can_discharge",
        "charger::Building_1/charger_1_1::available_discharge_action_normalized",
    ]
    _attach(agent, names, _action_names())
    obs = np.array(
        [
            0.60,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            8.0,
            1.0,
            0.5,
            1.0,
            0.9,
            0.5,
            60.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.10,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] == pytest.approx(-0.10)

    obs[-2] = 0.0
    actions = agent.predict([obs])
    assert actions[0][1] == pytest.approx(0.0)


def test_rbc_smart_policy_charges_storage_on_cheap_price_when_headroom_exists():
    agent = RBCSmartPolicy({"algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {}}})
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.10,
            0.40,
            0.50,
            0.60,
            0.0,
            2.0,
            8.0,
            10.0,
            0.4,
            0.0,
            0.0,
            0.0,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] > 0.0


def test_rbc_smart_policy_respects_simulator_storage_action_capacity():
    agent = RBCSmartPolicy({"algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {}}})
    names = _base_observation_names() + [
        "storage::Building_1/electrical_storage::can_charge",
        "storage::Building_1/electrical_storage::available_charge_action_normalized",
        "storage::Building_1/electrical_storage::can_discharge",
        "storage::Building_1/electrical_storage::available_discharge_action_normalized",
    ]
    _attach(agent, names, _action_names())
    obs = np.array(
        [
            0.10,
            0.40,
            0.50,
            0.60,
            0.0,
            2.0,
            8.0,
            10.0,
            0.4,
            0.0,
            0.0,
            0.0,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.15,
            0.0,
            0.40,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] == pytest.approx(0.15)

    obs[20] = 0.0
    actions = agent.predict([obs])
    assert actions[0][0] == pytest.approx(0.0)


def test_rbc_smart_policy_uses_local_forecast_surplus_for_storage_ev_and_deferrable():
    agent = RBCSmartPolicy({"algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {}}})
    names = _base_observation_names() + [
        "forecast_pv_surplus_mean_bucket_01_15m_kw",
    ]
    _attach(agent, names, _action_names())
    obs = np.array(
        [
            0.20,
            np.nan,
            np.nan,
            np.nan,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.7,
            0.8,
            60.0,
            10.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.1,
            0.9,
            4.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] == pytest.approx(0.75)
    assert actions[0][1] == pytest.approx(0.85)
    assert actions[0][2] == pytest.approx(1.0)


def test_rbc_smart_policy_uses_local_forecast_stress_for_storage_and_v2g():
    agent = RBCSmartPolicy(
        {
            "algorithm": {
                "name": "RBCSmartPolicy",
                "hyperparameters": {"allow_v2g": True, "ev_v2g_discharge_rate": 0.25},
            }
        }
    )
    names = _base_observation_names() + [
        "forecast_import_peak_next_15m_kw",
    ]
    _attach(agent, names, _action_names())
    obs = np.array(
        [
            0.20,
            np.nan,
            np.nan,
            np.nan,
            0.0,
            2.0,
            2.0,
            10.0,
            0.6,
            1.0,
            0.9,
            0.5,
            60.0,
            5.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.1,
            0.9,
            12.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] < 0.0
    assert actions[0][1] == pytest.approx(-0.25)
    assert actions[0][2] == pytest.approx(0.0)


def test_rbc_smart_policy_boosts_ev_service_when_departure_feasibility_is_low():
    agent = RBCSmartPolicy(
        {
            "algorithm": {
                "name": "RBCSmartPolicy",
                "hyperparameters": {
                    "flex_trickle_charge": 0.0,
                    "ev_service_floor_rate": 0.0,
                    "ev_service_margin_rate": 0.0,
                    "ev_service_lookahead_hours": 0.0,
                    "ev_deadline_buffer_hours": 0.0,
                    "emergency_charge_rate": 0.75,
                },
            }
        }
    )
    names = _base_observation_names() + [
        "charger::Building_1/charger_1_1::departure_feasibility_ratio",
    ]
    _attach(agent, names, _action_names())
    obs = np.array(
        [
            0.20,
            np.nan,
            np.nan,
            np.nan,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.79,
            0.80,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.60,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] == pytest.approx(0.75)


def test_rbc_smart_policy_price_charge_ceiling_is_not_capped_by_storage_target():
    agent = RBCSmartPolicy(
        {
            "algorithm": {
                "name": "RBCSmartPolicy",
                "hyperparameters": {
                    "price_charge_rate": 0.30,
                    "pv_charge_rate": 0.0,
                    "storage_target_soc": 0.40,
                    "storage_price_charge_soc_ceiling": 0.80,
                },
            }
        }
    )
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.10,
            0.40,
            0.50,
            0.60,
            0.0,
            2.0,
            8.0,
            10.0,
            0.60,
            0.0,
            0.0,
            0.0,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] == pytest.approx(0.30)


def test_rbc_smart_policy_charges_storage_when_headroom_is_low_but_positive():
    agent = RBCSmartPolicy({"algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {}}})
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.10,
            0.40,
            0.50,
            0.60,
            0.0,
            2.0,
            8.0,
            1.0,
            0.4,
            0.0,
            0.0,
            0.0,
            60.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] > 0.0


def test_rbc_smart_policy_can_use_conservative_v2g_when_enabled():
    agent = RBCSmartPolicy(
        {
            "algorithm": {
                "name": "RBCSmartPolicy",
                "hyperparameters": {"allow_v2g": True, "ev_v2g_discharge_rate": 0.25},
            }
        }
    )
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.60,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            8.0,
            1.0,
            0.5,
            1.0,
            0.9,
            0.5,
            60.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] < 0.0
    assert actions[0][1] < 0.0
    assert actions[0][2] == pytest.approx(0.0)


def test_rbc_smart_policy_deferrable_uses_simulator_must_start_now():
    agent = RBCSmartPolicy({"algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {}}})
    names = _base_observation_names() + [
        "deferrable_appliance::Building_1/deferrable_appliance_1::must_start_now",
    ]
    _attach(agent, names, _action_names())
    obs = np.array(
        [
            0.60,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.8,
            0.8,
            60.0,
            10.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][2] == pytest.approx(1.0)


def test_rbc_smart_policy_blocks_v2g_near_departure_even_when_enabled():
    agent = RBCSmartPolicy(
        {
            "algorithm": {
                "name": "RBCSmartPolicy",
                "hyperparameters": {"allow_v2g": True, "ev_v2g_discharge_rate": 0.25},
            }
        }
    )
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.60,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            8.0,
            1.0,
            0.5,
            1.0,
            0.9,
            0.5,
            60.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][1] >= 0.0


def test_rbc_community_policy_uses_community_surplus_for_storage_ev_and_deferrable():
    agent = RBCCommunityPolicy(
        {
            "algorithm": {
                "name": "RBCCommunityPolicy",
                "hyperparameters": {"community_storage_charge_rate": 0.55},
            }
        }
    )
    observation_names = _base_observation_names() + [
        "district__community_export_power_kw",
        "district__community_import_power_kw",
        "district__community_pv_power_kw",
        "district__community_building_headroom_kw",
    ]
    _attach(agent, observation_names, _action_names())
    obs = np.array(
        [
            0.20,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.7,
            0.8,
            60.0,
            10.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.1,
            0.9,
            8.0,
            0.0,
            8.0,
            10.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] == pytest.approx(0.55)
    assert actions[0][1] == pytest.approx(0.85)
    assert actions[0][2] == pytest.approx(1.0)


def test_rbc_community_policy_uses_forecast_community_surplus_for_storage_ev_and_deferrable():
    agent = RBCCommunityPolicy(
        {
            "algorithm": {
                "name": "RBCCommunityPolicy",
                "hyperparameters": {"community_storage_charge_rate": 0.55},
            }
        }
    )
    observation_names = _base_observation_names() + [
        "district__forecast_community_export_mean_bucket_01_15m_kw",
        "district__forecast_community_import_mean_bucket_01_15m_kw",
    ]
    _attach(agent, observation_names, _action_names())
    obs = np.array(
        [
            0.20,
            np.nan,
            np.nan,
            np.nan,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.7,
            0.8,
            60.0,
            10.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.1,
            0.9,
            8.0,
            0.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] == pytest.approx(0.55)
    assert actions[0][1] == pytest.approx(0.85)
    assert actions[0][2] == pytest.approx(1.0)


def test_rbc_community_policy_uses_safe_v2g_under_community_stress():
    agent = RBCCommunityPolicy(
        {
            "algorithm": {
                "name": "RBCCommunityPolicy",
                "hyperparameters": {"allow_v2g": True, "community_v2g_discharge_rate": 0.35},
            }
        }
    )
    observation_names = _base_observation_names() + [
        "district__community_export_power_kw",
        "district__community_import_power_kw",
        "district__community_building_headroom_kw",
    ]
    _attach(agent, observation_names, _action_names())
    obs = np.array(
        [
            0.20,
            0.10,
            0.20,
            0.30,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            1.0,
            0.9,
            0.5,
            60.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            20.0,
            1.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] < 0.0
    assert actions[0][1] == pytest.approx(-0.35)


def test_rbc_community_policy_uses_forecast_community_stress_for_storage_and_v2g():
    agent = RBCCommunityPolicy(
        {
            "algorithm": {
                "name": "RBCCommunityPolicy",
                "hyperparameters": {"allow_v2g": True, "community_v2g_discharge_rate": 0.35},
            }
        }
    )
    observation_names = _base_observation_names() + [
        "district__forecast_community_import_peak_next_15m_kw",
    ]
    _attach(agent, observation_names, _action_names())
    obs = np.array(
        [
            0.20,
            np.nan,
            np.nan,
            np.nan,
            0.0,
            2.0,
            2.0,
            10.0,
            0.6,
            1.0,
            0.9,
            0.5,
            60.0,
            5.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.1,
            0.9,
            20.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][0] < 0.0
    assert actions[0][1] == pytest.approx(-0.35)
    assert actions[0][2] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "policy_cls",
    [NormalPolicy, RBCBasicPolicy, RBCSmartPolicy, RBCCommunityPolicy],
)
def test_baseline_deferrable_actions_are_binary_start_commands(policy_cls):
    agent = policy_cls({"algorithm": {"name": policy_cls.__name__, "hyperparameters": {}}})
    _attach(agent, _base_observation_names(), _action_names())
    obs = np.array(
        [
            0.10,
            0.40,
            0.50,
            0.60,
            0.0,
            2.0,
            2.0,
            10.0,
            0.5,
            0.0,
            0.0,
            0.0,
            60.0,
            10.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        ],
        dtype=float,
    )

    actions = agent.predict([obs])

    assert actions[0][2] in {0.0, 1.0}
