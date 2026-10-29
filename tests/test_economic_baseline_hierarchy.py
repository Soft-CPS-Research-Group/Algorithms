"""Operational baseline capabilities without assuming universal KPI dominance."""

import numpy as np
import pytest

from algorithms.agents.baseline_policies import (
    RBCBasicPolicy,
    RBCCommunityPolicy,
    RBCSmartLocalPolicy,
)
from algorithms.agents.rbc_agent import StorageInfo


class Space:
    low = np.array([-1.0, 0.0, 0.0])
    high = np.array([1.0, 1.0, 1.0])


NAMES = [
    "electricity_pricing",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "electricity_pricing_predicted_3",
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
    "district__community_export_power_kw",
    "district__community_import_power_kw",
    "district__community_pv_power_kw",
    "district__community_building_headroom_kw",
]
ACTIONS = ["electrical_storage", "electric_vehicle_storage_charger_1_1"]


def make(cls, *, market=False, hyper=None, members=1):
    policy = cls({"algorithm": {"hyperparameters": hyper or {}}})
    policy.attach_environment(
        observation_names=[NAMES],
        action_names=[ACTIONS],
        action_space=[Space()],
        observation_space=[None],
        metadata={
            "seconds_per_time_step": 3600,
            "building_names": ["Building_" + str(i + 1) for i in range(members)],
            "community_market": {"enabled": market, "local_price_ratio_to_grid_import": 0.8},
        },
    )
    policy._get_storage_info = lambda _: StorageInfo(nominal_power=5.0, capacity=20.0)
    return policy


def obs(*, price=0.1, pv=0.0, load=2.0, soc=0.5, ev=True, shared_pv=0.0, shared_import=0.0):
    return np.array(
        [
            price,
            0.2,
            0.3,
            0.4,
            pv,
            load,
            max(0.0, load - pv),
            20.0,
            soc,
            float(ev),
            0.7,
            0.8,
            60.0,
            10.0,
            shared_pv,
            shared_import,
            shared_pv,
            100.0,
        ]
    )


@pytest.mark.parametrize("cls", [RBCSmartLocalPolicy, RBCCommunityPolicy])
def test_smart_preserves_basic_cheap_ev_opportunity_before_urgency(cls):
    x = obs()
    assert make(cls).predict([x])[0][1] == pytest.approx(make(RBCBasicPolicy).predict([x])[0][1])
    assert make(cls).predict([x])[0][1] > 0.5


@pytest.mark.parametrize("cls", [RBCSmartLocalPolicy, RBCCommunityPolicy])
def test_smart_does_not_discharge_against_cheap_imports(cls):
    x = obs(ev=False)
    action = make(cls).predict([x])[0][0]
    assert action > 0.0
    assert action * 5.0 + 2.0 <= 7.0


@pytest.mark.parametrize("cls", [RBCSmartLocalPolicy, RBCCommunityPolicy])
def test_smart_captures_local_solar_that_basic_ignores(cls):
    x = obs(price=0.6, pv=5.0, ev=False)
    assert make(RBCBasicPolicy).predict([x])[0][0] == 0.0
    assert make(cls).predict([x])[0][0] == pytest.approx(0.6)


def test_smart_keeps_basic_price_discharge_and_clips_to_actual_local_load():
    x = obs(price=0.6, ev=False)
    assert make(RBCSmartLocalPolicy).predict([x])[0][0] == pytest.approx(-0.4)


def test_community_without_market_is_identical_to_local_despite_remote_demand():
    x = obs(price=0.6, ev=False, shared_import=100.0)
    assert make(RBCCommunityPolicy).predict([x]) == make(RBCSmartLocalPolicy).predict([x])
    assert make(RBCCommunityPolicy, hyper={"community_market_enabled": True}).predict([x]) == make(
        RBCSmartLocalPolicy
    ).predict([x])


def test_actual_market_enables_surplus_use_and_does_not_count_it_twice():
    x = obs(price=0.25, load=0.0, shared_pv=8.0)
    local = make(RBCSmartLocalPolicy, market=True).predict([x])[0]
    shared = make(RBCCommunityPolicy, market=True).predict([x])[0]
    assert shared[1] > local[1]
    assert shared[0] * 5.0 + shared[1] * 7.4 <= 8.0 + 1e-9


def test_community_aggregate_peak_default_scales_with_member_count():
    policy = make(RBCCommunityPolicy, market=True, members=17)
    assert policy.community_import_threshold_kw == 119.0
    explicit = make(
        RBCCommunityPolicy, market=True, members=17, hyper={"community_import_threshold_kw": 80.0}
    )
    assert explicit.community_import_threshold_kw == 80.0


def test_community_supports_expensive_peak_only_with_settlement_enabled():
    x = obs(price=0.6, load=0.0, ev=False, shared_import=10.0)
    assert make(RBCCommunityPolicy, market=False).predict([x])[0][0] == 0.0
    assert make(RBCCommunityPolicy, market=True).predict([x])[0][0] == pytest.approx(-0.6)
    assert make(RBCSmartLocalPolicy, market=True).predict([x])[0][0] == 0.0


def test_explicit_zero_community_peak_is_not_replaced_by_default():
    policy = make(RBCCommunityPolicy, market=True, hyper={"community_import_threshold_kw": 0.0})
    assert policy.community_import_threshold_kw == 0.0


def test_specialised_community_teacher_keeps_its_configured_market_assumption():
    from algorithms.agents.baseline_policies import SignalAwareRBCCommunity

    # Its separate policy audit is deferred: binding actual market assumptions
    # is scoped to the operational Community, not silently to trained teachers.
    policy = SignalAwareRBCCommunity(
        {
            "algorithm": {"hyperparameters": {"community_local_price_ratio": 1.0}},
            "simulator": {
                "community_market": {"enabled": True, "local_price_ratio_to_grid_import": 0.5}
            },
        }
    )
    policy.attach_environment(
        observation_names=[NAMES],
        action_names=[ACTIONS],
        action_space=[Space()],
        observation_space=[None],
        metadata={
            "community_market": {"enabled": True, "local_price_ratio_to_grid_import": 0.5},
        },
    )
    assert policy.community_local_price_ratio == 1.0
    assert not policy._community_energy_is_discounted()
