"""Behavior, rather than reward ranking, defines each operational baseline."""

from types import SimpleNamespace

import numpy as np
import pytest

from algorithms.agents.baseline_policies import (
    NormalNoBatteryPolicy,
    NormalPolicy,
    RBCBasicPolicy,
    RBCCommunityPolicy,
    RBCSmartLocalPolicy,
    RBCSmartPolicy,
)
from algorithms.agents.rbc_agent import ChargerInfo, StorageInfo
from algorithms.configuration.agents import RuleBasedAlgorithmConfig

POLICIES = (
    NormalPolicy,
    NormalNoBatteryPolicy,
    RBCBasicPolicy,
    RBCSmartLocalPolicy,
    RBCCommunityPolicy,
)


@pytest.mark.parametrize("cls", POLICIES)
def test_baseline_does_not_silently_disable_unsupported_active_hvac(cls):
    with pytest.raises(ValueError, match="does not define control.*cooling_device"):
        policy(cls, {}, ("cooling_device",))


def policy(cls, values, actions=("electrical_storage",), hyper=None, seconds=3600):
    agent = cls({"algorithm": {"name": cls.__name__, "hyperparameters": hyper or {}}})
    agent.attach_environment(
        observation_names=[list(values)],
        action_names=[list(actions)],
        action_space=[SimpleNamespace(low=np.full(len(actions), -1.0), high=np.ones(len(actions)))],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": seconds},
    )
    agent._dataset_info = {"storage_info": {"Building_1": StorageInfo(10.0, 100.0)}}
    return agent, np.array(list(values.values()), dtype=float)


def ev(charger="charger_1", *, soc=0.2, required=0.8, hours=10.0, **extra):
    values = dict(
        connected_state=1.0,
        connected_ev_soc=soc,
        connected_ev_required_soc_departure=required,
        connected_ev_battery_capacity_kwh=60.0,
        hours_until_departure=hours,
        **extra,
    )
    return {f"charger::Building_1/{charger}::{key}": value for key, value in values.items()}


def appliance(name="washer", **extra):
    values = dict(
        pending=1.0,
        running=0.0,
        can_start=1.0,
        deadline_missed=0.0,
        must_run=1.0,
        must_start_now=0.0,
        slack_steps=10.0,
        urgency_ratio=0.0,
        slack_ratio=1.0,
        start_power_kw=2.0,
        **extra,
    )
    return {
        f"deferrable_appliance::Building_1/{name}::{key}": value for key, value in values.items()
    }


@pytest.mark.parametrize("cls", [NormalPolicy, NormalNoBatteryPolicy])
@pytest.mark.parametrize("hour", [2.0, 12.0, 18.0])
def test_bau_starts_requested_washer_and_ev_independent_of_price_or_hour(cls, hour):
    values = {"hour": hour, "electricity_pricing": 1.0, **ev(), **appliance()}
    agent, obs = policy(
        cls, values, ("electric_vehicle_storage_charger_1", "deferrable_appliance_washer")
    )
    assert agent.predict([obs])[0] == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("cls", [NormalPolicy, NormalNoBatteryPolicy])
def test_bau_does_not_reuse_departure_target_tolerance_near_full(cls):
    values = ev(soc=0.9995, required=0.8, hours=20.0)
    agent, obs = policy(cls, values, ("electric_vehicle_storage_charger_1",))
    assert agent.predict([obs])[0] == pytest.approx([1.0])


@pytest.mark.parametrize("cls", [RBCBasicPolicy, RBCSmartLocalPolicy, RBCCommunityPolicy])
def test_rbc_continues_service_with_a_three_percentage_point_soc_gap(cls):
    values = ev(
        soc=0.77, required=0.8, hours=0.5, min_charging_power_kw=1.38, max_charging_power_kw=7.36
    )
    agent, obs = policy(cls, values, ("electric_vehicle_storage_charger_1",), seconds=900)
    assert agent.ev_service_soc_tolerance == pytest.approx(0.001)
    assert agent.predict([obs])[0][0] >= 1.38 / 7.36


def test_rbc_final_service_command_crosses_charger_deadband_without_exceeding_budget():
    agent, _ = policy(RBCBasicPolicy, {})
    ctx = {
        "connected": True,
        "soc": 0.797,
        "required_soc": 0.8,
        "capacity": 50.0,
        "max_power": 7.36,
        "min_power": 1.38,
    }
    assert agent._safe_ev_charge_rate(ctx, 1.0, (-1.0, 1.0)) == pytest.approx(1.38 / 7.36)
    # An insufficient PV or physical-power budget must not be rounded up.
    assert agent._safe_ev_charge_rate(ctx, 0.1, (-1.0, 1.0)) == 0.0


@pytest.mark.parametrize("cls", POLICIES)
@pytest.mark.parametrize("flag", ["pending", "can_start"])
def test_appliance_never_starts_before_a_request_and_an_allowed_window(cls, flag):
    values = appliance()
    values[f"deferrable_appliance::Building_1/washer::{flag}"] = 0.0
    agent, obs = policy(cls, values, ("deferrable_appliance_washer",))
    assert agent.predict([obs])[0] == [0.0]


@pytest.mark.parametrize("cls", [RBCBasicPolicy, RBCSmartLocalPolicy, RBCCommunityPolicy])
def test_required_washing_cycle_can_wait_until_latest_start(cls):
    values = {"electricity_pricing": 0.2, **appliance()}
    agent, obs = policy(cls, values, ("deferrable_appliance_washer",))
    assert agent.predict([obs])[0] == [0.0]
    obs[list(values).index("deferrable_appliance::Building_1/washer::must_start_now")] = 1.0
    assert agent.predict([obs])[0] == [1.0]


@pytest.mark.parametrize("pv,bess,net,expected", [(10.0, 1.6, 0.0, 0.16), (0.0, -4.0, 4.4, -0.84)])
def test_bau_does_not_double_count_previous_settled_ev_and_bess(pv, bess, net, expected):
    values = {
        "electrical_storage_soc": 0.5,
        "pv_power_kw": pv,
        "load_power_kw": 8.4,
        "net_power_kw": net,
        "import_power_kw": max(0.0, net),
        "export_power_kw": max(0.0, -net),
        "bess_power_kw": bess,
        "ev_charging_power_kw": 7.4,
        **ev(),
    }
    agent, obs = policy(
        NormalPolicy, values, ("electrical_storage", "electric_vehicle_storage_charger_1")
    )
    # The next EV command is 7.4 kW. Passive household demand is only 1 kW.
    assert agent.predict([obs])[0] == pytest.approx([expected, 1.0])


def test_bau_battery_uses_only_pv_left_after_ev_and_washer():
    values = {
        "electrical_storage_soc": 0.5,
        "pv_power_kw": 10.0,
        "load_power_kw": 0.0,
        **ev(),
        **appliance(),
    }
    actions = (
        "electrical_storage",
        "electric_vehicle_storage_charger_1",
        "deferrable_appliance_washer",
    )
    agent, obs = policy(NormalPolicy, values, actions)
    result = agent.predict([obs])[0]
    assert result == pytest.approx([0.06, 1.0, 1.0])  # 10 - 7.4 - 2 = 0.6 kW


def test_bau_battery_supplies_new_washer_demand_without_waiting_a_step():
    values = {
        "electrical_storage_soc": 0.5,
        "pv_power_kw": 0.0,
        "load_power_kw": 0.0,
        **appliance(),
    }
    agent, obs = policy(NormalPolicy, values, ("electrical_storage", "deferrable_appliance_washer"))
    assert agent.predict([obs])[0] == pytest.approx([-0.2, 1.0])


@pytest.mark.parametrize("cls", [RBCSmartPolicy, RBCSmartLocalPolicy, RBCCommunityPolicy])
def test_two_nonurgent_evs_cannot_reuse_the_same_pv(cls):
    values = {
        "electrical_storage_soc": 0.5,
        "pv_power_kw": 10.0,
        "load_power_kw": 0.0,
        **ev("one", hours=30.0),
        **ev("two", hours=30.0),
    }
    agent, obs = policy(
        cls, values, ("electric_vehicle_storage_one", "electric_vehicle_storage_two")
    )
    result = agent.predict([obs])[0]
    assert sum(result) * 7.4 <= 10.0 + 1e-8
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(2.6 / 7.4)


@pytest.mark.parametrize("cls", [RBCBasicPolicy, RBCSmartLocalPolicy, RBCCommunityPolicy])
def test_constant_tariff_has_no_cheap_or_expensive_incentive(cls):
    values = {
        "electrical_storage_soc": 0.5,
        "electricity_pricing": 0.2,
        **{f"electricity_pricing_predicted_{i}": 0.2 for i in (1, 2, 3)},
    }
    agent, obs = policy(cls, values)
    context = agent._get_price_context(obs, agent._obs_index[0])
    assert not context["cheap"] and not context["expensive"]
    assert agent.predict([obs])[0] == [0.0]


@pytest.mark.parametrize("hour,expected", [(2.0, 0.6), (12.0, 0.0), (18.0, -0.45), (23.0, 0.6)])
def test_basic_hourly_fallback_is_explicit_when_price_is_missing(hour, expected):
    agent, obs = policy(
        RBCBasicPolicy, {"hour": hour, "electrical_storage_soc": 0.5, "import_power_kw": 5.0}
    )
    assert agent.predict([obs])[0] == pytest.approx([expected])


@pytest.mark.parametrize("cls", POLICIES + (RBCSmartPolicy,))
def test_direct_and_validated_defaults_produce_same_actions(cls):
    values = {
        "electrical_storage_soc": 0.5,
        "pv_power_kw": 0.0,
        "load_power_kw": 0.0,
        "electricity_pricing": 0.1,
        "electricity_pricing_predicted_1": 0.3,
    }
    hyper = RuleBasedAlgorithmConfig(algorithm=cls.__name__).model_dump()["hyperparameters"]
    hyper = {key: value for key, value in hyper.items() if value is not None}
    direct, obs = policy(cls, values)
    configured, _ = policy(cls, values, hyper=hyper)
    assert direct.predict([obs]) == configured.predict([obs])


@pytest.mark.parametrize("seconds", [15, 900, 3600])
def test_flat_step_energy_and_entity_power_mean_the_same_physical_surplus(seconds):
    hours = seconds / 3600
    common = {"electrical_storage_soc": 0.5}
    flat, flat_obs = policy(
        NormalPolicy,
        {**common, "solar_generation": 5 * hours, "non_shiftable_load": 1 * hours},
        seconds=seconds,
    )
    entity, entity_obs = policy(
        NormalPolicy, {**common, "pv_power_kw": 5.0, "load_power_kw": 1.0}, seconds=seconds
    )
    assert flat.predict([flat_obs])[0] == pytest.approx([0.4])
    assert flat.predict([flat_obs]) == entity.predict([entity_obs])


def test_charger_metadata_matches_identity_not_action_order():
    agent = NormalPolicy({})
    a = ChargerInfo("one", 7.4, 0.0, 7.4, 0.0, 1.0, 60.0)
    b = ChargerInfo("two", 11.0, 0.0, 11.0, 0.0, 1.0, 80.0)
    agent._dataset_info = {"charger_info": {"Building_1": [a, b]}}
    agent._agent_buildings = {0: "Building_1"}
    agent._build_ev_action_mapping(
        [["electric_vehicle_storage_two", "electric_vehicle_storage_one"]]
    )
    assert agent._get_charger_info(0, 0) is b
    assert agent._get_charger_info(0, 1) is a


def test_dynamic_charger_uses_observed_power_in_pv_budget():
    values = {
        "pv_power_kw": 10.0,
        "load_power_kw": 0.0,
        **ev("new", hours=30.0, max_charging_power_kw=22.0),
    }
    agent, obs = policy(RBCSmartLocalPolicy, values, ("electric_vehicle_storage_new",))
    assert agent.predict([obs])[0] == pytest.approx([10.0 / 22.0])


def test_dynamic_charger_obeys_building_headroom_without_static_phase_metadata():
    values = {
        "pv_power_kw": 10.0,
        "load_power_kw": 0.0,
        "charging_building_headroom_kw": 5.0,
        **ev("new", hours=30.0, max_charging_power_kw=22.0),
    }
    agent, obs = policy(RBCSmartLocalPolicy, values, ("electric_vehicle_storage_new",))
    assert agent.predict([obs])[0] == pytest.approx([(5.0 - 0.05) / 22.0])


def test_kpi_baseline_reference_distinguishes_native_bau_from_normal_policy():
    from algorithms.runtime.kpis import baseline_reference

    assert baseline_reference(False) is None
    reference = baseline_reference(True)
    assert reference["implementation"] == "citylearn.agents.baseline.BusinessAsUsualAgent"
    assert reference["equivalent_to_normal_policy"] is False


@pytest.mark.parametrize("donor_has_local_pv", [False, True])
def test_community_surplus_is_shared_once_across_buildings(donor_has_local_pv):
    values = {
        "pv_power_kw": 0.0,
        "load_power_kw": 0.0,
        "district__community_export_power_kw": 10.0,
        "district__community_import_power_kw": 0.0,
        "district__community_building_headroom_kw": 100.0,
        **ev("one", hours=30.0),
    }
    agent = RBCCommunityPolicy({})
    second = [name.replace("Building_1", "Building_2") for name in values]
    space = SimpleNamespace(low=np.array([-1.0]), high=np.array([1.0]))
    agent.attach_environment(
        observation_names=[list(values), second],
        action_names=[["electric_vehicle_storage_one"]] * 2,
        action_space=[space, space],
        observation_space=[None, None],
        metadata={"building_names": ["Building_1", "Building_2"], "seconds_per_time_step": 3600},
    )
    obs = np.array(list(values.values()))
    donor_obs = obs.copy()
    if donor_has_local_pv:
        donor_obs[list(values).index("pv_power_kw")] = 10.0
    actions = agent.predict([obs, donor_obs])
    assert actions[0][0] == pytest.approx(1.0)
    assert actions[1][0] == pytest.approx(2.6 / 7.4)
    assert sum(a[0] * 7.4 for a in actions) == pytest.approx(10.0)


def test_retired_rulebased_name_is_rejected_with_explicit_alternatives():
    from algorithms.registry import build_execution_unit

    with pytest.raises(ValueError, match="retired.*NormalPolicy.*RBCBasicPolicy"):
        build_execution_unit({"pipeline": [{"algorithm": "RuleBasedPolicy"}]})


def test_basic_does_not_discharge_into_unpaid_export_at_high_price():
    values = {
        "electrical_storage_soc": 0.7,
        "electricity_pricing": 0.5,
        "electricity_pricing_predicted_1": 0.1,
        "import_power_kw": 0.0,
    }
    agent, obs = policy(RBCBasicPolicy, values)
    assert agent.predict([obs])[0] == [0.0]


def test_missing_price_and_clock_does_not_invent_a_night_tariff():
    agent, obs = policy(RBCBasicPolicy, {"electrical_storage_soc": 0.5})
    assert agent.predict([obs])[0] == [0.0]
