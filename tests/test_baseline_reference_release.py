"""Public baseline identity, information limits and old/new executable bundles."""

import json

import pytest

from algorithms.agents.baseline_policies import (
    NormalTargetPolicy,
    RBCBasicPolicy,
    RBCCommunityPolicy,
    RBCSmartLocalPolicy,
    RBCSmartPolicy,
)
from algorithms.configuration.agents import RuleBasedAlgorithmConfig
from algorithms.configuration.baseline_defaults import NORMAL_DEFAULTS
from algorithms.export.policy_json import load_policy, policy_state
from algorithms.registry import ALGORITHM_REGISTRY, agent_spec
from tests.test_baseline_behavior_contract import appliance, ev, policy


def signals(price=0.1, forecast=0.3):
    return {
        "electricity_pricing": price,
        **{f"electricity_pricing_predicted_{i}": forecast for i in (1, 2, 3)},
        "district__forecast_price_next_24h": forecast,
        "electrical_storage_soc": 0.5,
        "import_power_kw": 2.0,
    }


def test_basic_cannot_change_decisions_by_reading_price_forecasts():
    actions = (
        "electrical_storage",
        "electric_vehicle_storage_charger_1",
        "deferrable_appliance_washer",
    )
    results = []
    for forecast in (0.01, 0.10, 0.40):
        agent, observation = policy(
            RBCBasicPolicy, {**signals(forecast=forecast), **ev(), **appliance()}, actions
        )
        results.append(agent.predict([observation]))
    assert results[0] == results[1] == results[2]
    assert results[0][0][0] < 0.0 and results[0][0][2] == 0.0


@pytest.mark.parametrize("price,expected", [(0.05, 0.6), (0.10, 0.0), (0.15, -0.2)])
def test_legacy_basic_price_only_storage_works_without_any_forecasts(price, expected):
    values = {"electricity_pricing": price, "electrical_storage_soc": 0.5, "import_power_kw": 2.0}
    agent, observation = policy(
        RBCBasicPolicy, values, hyper={"basic_storage_policy": "price_only"}
    )
    assert agent.predict([observation])[0][0] == pytest.approx(expected)


@pytest.mark.parametrize("hour", [2, 12, 18, 23])
def test_basic_missing_price_is_neutral_but_urgent_service_still_runs(hour):
    values = {
        "hour": hour,
        "electrical_storage_soc": 0.5,
        "import_power_kw": 2.0,
        **ev(hours=0.25),
        **appliance(),
        "deferrable_appliance::Building_1/washer::must_start_now": 1.0,
    }
    agent, observation = policy(
        RBCBasicPolicy,
        values,
        ("electric_vehicle_storage_charger_1", "deferrable_appliance_washer"),
    )
    action = agent.predict([observation])[0]
    context = agent._get_price_context(observation, agent._obs_index[0])
    assert not context["cheap"] and not context["expensive"]
    assert action == [1.0, 1.0]


@pytest.mark.parametrize("cls", [RBCSmartLocalPolicy, RBCCommunityPolicy])
def test_smart_and_community_retain_forecast_price_decisions(cls):
    decisions = []
    for forecast in (0.05, 0.3):
        agent, observation = policy(cls, signals(forecast=forecast))
        decisions.append(agent.predict([observation])[0][0])
    assert decisions[0] < 0 < decisions[1]


@pytest.mark.parametrize("name", ["Smart", "RBCSmartLocalPolicy"])
def test_smart_public_name_is_the_exact_audited_local_implementation(name):
    assert ALGORITHM_REGISTRY[name] is RBCSmartLocalPolicy
    config = RuleBasedAlgorithmConfig(algorithm=name)
    assert config.hyperparameters.price_charge_rate == 0.6
    assert agent_spec(name).capabilities == agent_spec("RBCSmartLocalPolicy").capabilities


@pytest.mark.parametrize("name", ["RBCBasicPolicy", "Smart", "RBCCommunityPolicy"])
def test_current_baseline_json_roundtrip_preserves_actions_and_identity(name, tmp_path):
    cls = ALGORITHM_REGISTRY[name]
    instance, observation = policy(cls, signals(price=0.05))
    instance.agent_name = name
    expected = instance.predict([observation])
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {"format": "algorithms_policy_v1", "algorithm": name, "state": policy_state(instance)}
        )
    )
    loaded = load_policy(path)
    assert loaded.agent_name == name
    assert loaded.predict([observation]) == expected
    if name == "RBCBasicPolicy":
        assert loaded.basic_price_policy == "fixed_thresholds"
        assert loaded.basic_cheap_price_eur_kwh == 0.0765
    elif name == "RBCCommunityPolicy":
        assert loaded.community_import_threshold_kw == 15.0


def test_old_basic_bundle_preserves_forecast_based_actions(tmp_path):
    instance, observation = policy(
        RBCBasicPolicy, signals(), hyper={"basic_price_policy": "forecast_relative"}
    )
    expected = instance.predict([observation])
    state = policy_state(instance)
    for key in ("basic_price_policy", "basic_cheap_price_eur_kwh", "basic_expensive_price_eur_kwh"):
        state.pop(key)
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {"format": "algorithms_policy_v1", "algorithm": "RBCBasicPolicy", "state": state}
        )
    )
    assert load_policy(path).predict([observation]) == expected
    fixed, _ = policy(RBCBasicPolicy, signals())
    assert fixed.predict([observation]) != expected


@pytest.mark.parametrize("price", [0.01, 0.09, 0.20])
@pytest.mark.parametrize("pv,load", [(0.0, 3.0), (5.0, 1.0), (0.0, 0.0)])
def test_basic_self_consumption_matches_bau_battery_without_reading_price(price, pv, load):
    values = {
        "electricity_pricing": price,
        "pv_power_kw": pv,
        "load_power_kw": load,
        "electrical_storage_soc": 0.5,
    }
    bau, obs = policy(NormalTargetPolicy, values)
    basic, _ = policy(
        RBCBasicPolicy,
        values,
        hyper={**NORMAL_DEFAULTS, "basic_storage_policy": "self_consumption"},
    )
    assert basic.predict([obs]) == bau.predict([obs])


@pytest.mark.parametrize("price", [0.01, 0.09, 0.20])
def test_basic_storage_choice_preserves_ev_and_appliance_timing(price):
    values = {**signals(price=price), **ev(hours=30), **appliance()}
    actions = ("electric_vehicle_storage_charger_1", "deferrable_appliance_washer")
    old, obs = policy(RBCBasicPolicy, values, actions, hyper={"basic_storage_policy": "price_only"})
    new, _ = policy(
        RBCBasicPolicy,
        values,
        actions,
        hyper={**NORMAL_DEFAULTS, "basic_storage_policy": "self_consumption"},
    )
    assert old.predict([obs]) == new.predict([obs])


def test_basic_storage_bundle_keeps_new_and_historical_rules(tmp_path):
    values = {**signals(price=0.05), "pv_power_kw": 0.0, "load_power_kw": 2.0}
    for mode in ("price_only", "self_consumption"):
        agent, obs = policy(
            RBCBasicPolicy, values, hyper={**NORMAL_DEFAULTS, "basic_storage_policy": mode}
        )
        expected = agent.predict([obs])
        state = policy_state(agent)
        if mode == "price_only":
            state.pop("basic_storage_policy")
        path = tmp_path / f"{mode}.json"
        path.write_text(
            json.dumps(
                {"format": "algorithms_policy_v1", "algorithm": "RBCBasicPolicy", "state": state}
            )
        )
        assert load_policy(path).predict([obs]) == expected


def test_basic_selected_defaults_match_direct_and_validated_construction():
    current = RBCBasicPolicy({})
    schema = RuleBasedAlgorithmConfig(algorithm="RBCBasicPolicy").hyperparameters
    assert current.basic_storage_policy == schema.basic_storage_policy == "self_consumption"
    for key, value in NORMAL_DEFAULTS.items():
        assert getattr(current, key) == getattr(schema, key) == value
    legacy = RBCBasicPolicy(
        {"algorithm": {"hyperparameters": {"basic_storage_policy": "price_only"}}}
    )
    legacy_schema = RuleBasedAlgorithmConfig(
        algorithm="RBCBasicPolicy", hyperparameters={"basic_storage_policy": "price_only"}
    ).hyperparameters
    assert legacy.storage_charge_rate == legacy_schema.storage_charge_rate == 0.35
    assert legacy.pv_surplus_threshold_kw == legacy_schema.pv_surplus_threshold_kw == 0.25


@pytest.mark.parametrize("cls", [RBCSmartPolicy, RBCSmartLocalPolicy, RBCCommunityPolicy])
def test_basic_default_change_preserves_smart_and_community_storage_defaults(cls):
    agent = cls({})
    schema = RuleBasedAlgorithmConfig(algorithm=cls.__name__).hyperparameters
    assert agent.storage_charge_rate == schema.storage_charge_rate == 0.35
    assert agent.storage_discharge_rate == schema.storage_discharge_rate == 0.35
    assert agent.pv_surplus_threshold_kw == schema.pv_surplus_threshold_kw == 0.25


@pytest.mark.parametrize(
    "cheap,expensive", [(0.2, 0.1), (0.1, 0.1), (float("nan"), 0.2), (0.1, float("inf"))]
)
def test_basic_rejects_ambiguous_or_nonfinite_thresholds(cheap, expensive):
    hyper = {"basic_cheap_price_eur_kwh": cheap, "basic_expensive_price_eur_kwh": expensive}
    with pytest.raises(ValueError):
        RBCBasicPolicy({"algorithm": {"hyperparameters": hyper}})
    with pytest.raises(ValueError):
        RuleBasedAlgorithmConfig(algorithm="RBCBasicPolicy", hyperparameters=hyper)
