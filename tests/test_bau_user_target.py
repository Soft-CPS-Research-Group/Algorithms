"""Immediate charging respects a user SOC ceiling, including the final interval."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from algorithms.agents.baseline_policies import NormalPolicy, NormalTargetPolicy
from algorithms.configuration.agents import RuleBasedAlgorithmConfig
from algorithms.export.policy_json import load_policy, policy_state
from algorithms.registry import ALGORITHM_REGISTRY
from tests.test_baseline_behavior_contract import ev, policy


@pytest.mark.parametrize("seconds", [60, 900, 3600])
@pytest.mark.parametrize("soc", [0.2, 0.79, 0.7999, 0.8, 0.9])
@pytest.mark.parametrize("hours", [0.01, 10.0])
def test_charge_energy_cannot_cross_user_target_even_at_departure(seconds, soc, hours):
    values = ev(soc=soc, required=0.8, hours=hours, max_charging_power_kw=7.0)
    agent, obs = policy(
        NormalTargetPolicy, values, ("electric_vehicle_storage_charger_1",), seconds=seconds
    )
    action = agent.predict([obs])[0][0]
    assert 0 <= action <= 1
    # Even at 100% physical efficiency, the command cannot take SOC above X.
    assert action * 7 * seconds / 3600 <= max(0.0, 0.8 - soc) * 60 + 1e-8
    if soc == 0.2:
        assert action == 1.0
    if soc >= 0.8:
        assert action == 0.0


def test_user_limit_never_rounds_up_to_charger_minimum():
    values = ev(soc=0.799, min_charging_power_kw=1.38, max_charging_power_kw=7.36)
    agent, obs = policy(
        NormalTargetPolicy, values, ("electric_vehicle_storage_charger_1",), seconds=900
    )
    assert agent.predict([obs]) == [[0.0]]


@pytest.mark.parametrize(
    "feature",
    [
        "connected_ev_soc",
        "connected_ev_required_soc_departure",
        "connected_ev_battery_capacity_kwh",
    ],
)
def test_user_limit_does_not_charge_from_guessed_state(feature):
    values = ev()
    del values[f"charger::Building_1/charger_1::{feature}"]
    agent, obs = policy(NormalTargetPolicy, values, ("electric_vehicle_storage_charger_1",))
    assert agent.predict([obs]) == [[0.0]]


def test_target_bau_export_retains_identity_and_strict_last_charge(tmp_path):
    agent, obs = policy(
        NormalTargetPolicy, ev(soc=0.79), ("electric_vehicle_storage_charger_1",), seconds=900
    )
    expected = agent.predict([obs])
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "format": "algorithms_policy_v1",
                "algorithm": "NormalTargetPolicy",
                "state": policy_state(agent),
            }
        )
    )
    assert load_policy(path).predict([obs]) == expected
    assert 0 < expected[0][0] < 1


def test_user_target_is_a_separate_validated_policy():
    assert ALGORITHM_REGISTRY["NormalTargetPolicy"] is NormalTargetPolicy
    config = RuleBasedAlgorithmConfig(algorithm="NormalTargetPolicy")
    assert config.hyperparameters.ev_normal_target_soc == 0.0
    assert config.hyperparameters.storage_charge_rate == 1.0
    for factory in [
        lambda h: RuleBasedAlgorithmConfig(algorithm="NormalTargetPolicy", hyperparameters=h),
        lambda h: NormalTargetPolicy({"algorithm": {"hyperparameters": h}}),
    ]:
        with pytest.raises(ValueError, match="user target"):
            factory({"ev_normal_target_soc": 1.0})
    normal, obs = policy(NormalPolicy, ev(soc=0.81), ("electric_vehicle_storage_charger_1",))
    target, _ = policy(NormalTargetPolicy, ev(soc=0.81), ("electric_vehicle_storage_charger_1",))
    assert normal.predict([obs]) == [[1.0]]
    assert target.predict([obs]) == [[0.0]]


def test_core30_default_bau_and_full_charge_variant_are_unambiguous():
    root = Path(__file__).resolve().parents[1] / "configs/templates/baselines/core30"
    main = yaml.safe_load((root / "bau.yaml").read_text())
    alias = yaml.safe_load((root / "bau_target.yaml").read_text())
    full = yaml.safe_load((root / "bau_full_charge.yaml").read_text())
    assert main == alias
    assert main["pipeline"][0]["algorithm"] == "NormalTargetPolicy"
    assert full["pipeline"][0]["algorithm"] == "NormalPolicy"
    for name in ("bau", "bau_full_charge", "basic", "smart", "community"):
        recipe = yaml.safe_load((root / f"{name}.yaml").read_text())
        assert recipe["simulator"]["export"]["include_business_as_usual"] is False


def test_lossy_physical_charger_approaches_target_without_overshoot():
    from citylearn.base import EpisodeTracker
    from citylearn.energy_model import Battery

    # The real battery model includes losses and float32 SOC. The controller
    # sees the current state and does not inject a hidden physical SOC cap.
    tracker = EpisodeTracker(0, 95)
    tracker.next_episode(96, False, False, 1)
    battery = Battery(
        capacity=60.0,
        nominal_power=7.0,
        initial_soc=0.2,
        seconds_per_time_step=900,
        episode_tracker=tracker,
        random_seed=1,
    )
    battery.reset()
    current = 0.2
    for _ in range(60):
        values = ev(soc=current, min_charging_power_kw=1.38, max_charging_power_kw=7.0)
        agent, obs = policy(
            NormalTargetPolicy, values, ("electric_vehicle_storage_charger_1",), seconds=900
        )
        action = agent.predict([obs])[0][0]
        battery.charge(action * 7 * 0.25 * 0.9)
        current = float(battery.soc[battery.time_step])
        assert current <= 0.8 + np.finfo(np.float32).eps
        battery.next_time_step()
    assert current >= 0.8 - 1.38 * 0.25 / 60 - 0.001
