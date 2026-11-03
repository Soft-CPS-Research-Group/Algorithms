"""Public baseline identity, information limits and old/new executable bundles."""

import json

import pytest

from algorithms.agents.baseline_policies import (
    RBCBasicPolicy,
    RBCCommunityPolicy,
    RBCSmartLocalPolicy,
)
from algorithms.configuration.agents import RuleBasedAlgorithmConfig
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
    assert results[0][0][0] == 0.0 and results[0][0][2] == 0.0


@pytest.mark.parametrize("price,expected", [(0.05, 0.6), (0.10, 0.0), (0.15, -0.2)])
def test_basic_fixed_tariff_works_without_any_forecasts(price, expected):
    values = {"electricity_pricing": price, "electrical_storage_soc": 0.5, "import_power_kw": 2.0}
    agent, observation = policy(RBCBasicPolicy, values)
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


@pytest.mark.parametrize(
    "cheap,expensive", [(0.2, 0.1), (0.1, 0.1), (float("nan"), 0.2), (0.1, float("inf"))]
)
def test_basic_rejects_ambiguous_or_nonfinite_thresholds(cheap, expensive):
    hyper = {"basic_cheap_price_eur_kwh": cheap, "basic_expensive_price_eur_kwh": expensive}
    with pytest.raises(ValueError):
        RBCBasicPolicy({"algorithm": {"hyperparameters": hyper}})
    with pytest.raises(ValueError):
        RuleBasedAlgorithmConfig(algorithm="RBCBasicPolicy", hyperparameters=hyper)
