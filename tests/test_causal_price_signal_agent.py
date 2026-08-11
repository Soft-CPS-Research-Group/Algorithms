from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from algorithms.agents.causal_price_signal_agent import CausalPriceSignalAgent
from algorithms.registry import ALGORITHM_REGISTRY, build_execution_unit
from scripts.generate_cc_ppo_causal_online_v5p3 import (
    CHARGE_RATES,
    EXPERIMENT_NAME,
    causal_online_recipe,
    generate,
)
from utils.config_schema import CausalPriceSignalHyperparameters, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments" / EXPERIMENT_NAME
NAMES = [
    "district__electricity_pricing",
    "district__electricity_pricing_predicted_1",
    "district__electricity_pricing_predicted_2",
    "district__electricity_pricing_predicted_3",
    "district__community_export_power_kw",
]


def _agent(**hyperparameters) -> CausalPriceSignalAgent:
    agent = CausalPriceSignalAgent(
        {"algorithm": {"hyperparameters": hyperparameters}}
    )
    agent.attach_environment(
        observation_names=[NAMES],
        action_names=[[]],
        action_space=[],
        observation_space=[],
    )
    return agent


def _observation(price: float, export_kw: float) -> list[np.ndarray]:
    return [np.asarray([price, 2.0, 2.0, 2.0, export_kw], dtype=np.float64)]


def test_causal_signal_requires_both_cheap_price_and_current_export():
    agent = _agent(cc_action_interval=1)
    assert agent.predict(_observation(1.0, 3.0), deterministic=True) == pytest.approx(0.95)
    assert agent.predict(_observation(3.0, 3.0), deterministic=True) == pytest.approx(1.0)
    assert agent.predict(_observation(1.0, 0.0), deterministic=True) == pytest.approx(1.0)


def test_causal_signal_can_emit_one_discount_per_building():
    discounts = [0.89, 0.90, 0.91]
    agent = CausalPriceSignalAgent(
        {
            "algorithm": {
                "hyperparameters": {
                    "cc_action_interval": 1,
                    "discount_multipliers": discounts,
                }
            }
        }
    )
    agent.attach_environment(
        observation_names=[NAMES, NAMES, NAMES],
        action_names=[[], [], []],
        action_space=[],
        observation_space=[],
    )

    assert agent.predict(_observation(1.0, 3.0)) == pytest.approx(discounts)
    assert agent.predict(_observation(3.0, 3.0)) == pytest.approx([1.0] * 3)


def test_causal_vector_can_allocate_with_a_per_building_surcharge():
    multipliers = [0.85, 1.05, 0.95]
    agent = CausalPriceSignalAgent(
        {
            "algorithm": {
                "hyperparameters": {"discount_multipliers": multipliers}
            }
        }
    )
    agent.attach_environment(
        observation_names=[NAMES, NAMES, NAMES],
        action_names=[[], [], []],
        action_space=[],
        observation_space=[],
    )

    assert agent.predict(_observation(1.0, 3.0)) == pytest.approx(multipliers)


def test_causal_vector_signal_validates_environment_building_count():
    agent = CausalPriceSignalAgent(
        {
            "algorithm": {
                "hyperparameters": {"discount_multipliers": [0.9, 0.9]}
            }
        }
    )
    with pytest.raises(ValueError, match="environment building count"):
        agent.attach_environment(
            observation_names=[NAMES],
            action_names=[[]],
            action_space=[],
            observation_space=[],
        )


def test_causal_signal_holds_decision_for_the_complete_interval():
    agent = _agent(cc_action_interval=4)
    agent.set_episode_context(episode_step=0)
    assert agent.predict(_observation(1.0, 3.0)) == pytest.approx(0.95)
    for step in (1, 2, 3):
        agent.set_episode_context(episode_step=step)
        assert agent.predict(_observation(3.0, 0.0)) == pytest.approx(0.95)
    agent.set_episode_context(episode_step=4)
    assert agent.predict(_observation(3.0, 0.0)) == pytest.approx(1.0)


def test_causal_signal_episode_reset_forces_a_fresh_pre_action_decision():
    agent = _agent(cc_action_interval=4)
    agent.set_episode_context(episode_step=0)
    assert agent.predict(_observation(1.0, 3.0)) == pytest.approx(0.95)
    agent.set_episode_context(episode_step=1)
    assert agent.predict(_observation(3.0, 0.0)) == pytest.approx(0.95)
    agent.set_episode_context(episode_step=0)
    assert agent.predict(_observation(3.0, 0.0)) == pytest.approx(1.0)


def test_causal_signal_manifest_exports_auditable_decision_trace(tmp_path: Path):
    agent = _agent(cc_action_interval=1)
    agent.set_episode_context(episode_step=0)
    agent.predict(_observation(1.0, 3.0))
    manifest = agent.export_artifacts(str(tmp_path))
    trace = tmp_path / "decision_trace.csv"
    rows = list(csv.DictReader(trace.open(encoding="utf-8")))
    assert manifest["rule"] == "current_native_cheap_and_current_community_export"
    assert manifest["output_contract"] == "causal_global_price_multiplier"
    assert rows[0]["episode_step"] == "0"
    assert rows[0]["cheap"] == "1"
    assert rows[0]["exporting"] == "1"


def test_causal_vector_manifest_records_every_member(tmp_path: Path):
    discounts = [0.89, 0.90]
    agent = CausalPriceSignalAgent(
        {
            "algorithm": {
                "hyperparameters": {
                    "cc_action_interval": 1,
                    "discount_multipliers": discounts,
                }
            }
        }
    )
    agent.attach_environment(
        observation_names=[NAMES, NAMES],
        action_names=[[], []],
        action_space=[],
        observation_space=[],
    )
    agent.set_episode_context(episode_step=0)
    agent.predict(_observation(1.0, 3.0))

    manifest = agent.export_artifacts(str(tmp_path))
    rows = list(csv.DictReader((tmp_path / "decision_trace.csv").open()))

    assert manifest["output_contract"] == (
        "causal_per_building_price_multiplier_vector"
    )
    assert manifest["discount_multipliers"] == discounts
    assert float(rows[0]["multiplier_b0"]) == pytest.approx(0.89)
    assert float(rows[0]["multiplier_b1"]) == pytest.approx(0.90)


def test_causal_signal_fails_fast_when_causal_features_are_missing():
    agent = CausalPriceSignalAgent({"algorithm": {"hyperparameters": {}}})
    with pytest.raises(ValueError, match="missing required observation"):
        agent.attach_environment(
            observation_names=[["district__electricity_pricing"]],
            action_names=[[]],
            action_space=[],
            observation_space=[],
        )


def test_causal_signal_schema_rejects_non_discount():
    with pytest.raises(ValueError, match="must be below"):
        CausalPriceSignalHyperparameters(
            neutral_multiplier=1.0,
            discount_multiplier=1.0,
        )
    CausalPriceSignalHyperparameters(
        neutral_multiplier=1.0,
        discount_multipliers=[0.9, 1.05],
    )
    with pytest.raises(ValueError, match="vector multiplier range"):
        CausalPriceSignalHyperparameters(
            neutral_multiplier=1.0,
            discount_multipliers=[0.9, 1.31],
        )


@pytest.mark.parametrize("charge_rate", CHARGE_RATES)
def test_v5p3_recipe_is_causal_annual_and_keeps_frozen_local_ppo(charge_rate: float):
    config = causal_online_recipe(charge_rate)
    validate_config(config)
    manager, ppo = config["pipeline"]
    assert manager["algorithm"] == "CausalPriceSignal"
    assert manager["frozen"] is True
    assert "schedule" not in manager["hyperparameters"]
    assert manager["hyperparameters"]["cc_action_interval"] == 4
    assert config["simulator"]["episode_time_steps"] == 35040
    assert config["simulator"]["community_market"]["enabled"] is True
    assert config["tracking"]["tags"]["trace_derived"] == "False"
    assert config["tracking"]["tags"]["uses_future_realized_data"] == "False"
    assert ppo["frozen"] is True
    residual = ppo["exploration"]["params"][
        "residual_base_policy_hyperparameters"
    ]
    assert residual["signal_price_charge_rate"] == pytest.approx(charge_rate)


def test_generated_v5p3_templates_match_committed_templates(tmp_path: Path):
    for path in generate(tmp_path):
        committed = CONFIG_ROOT / path.name
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == yaml.safe_load(
            committed.read_text(encoding="utf-8")
        )


def test_v5p3_smoke_is_short_and_not_promotion_eligible(tmp_path: Path):
    configs = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in generate(tmp_path, smoke=True)
    ]
    for config in configs:
        validate_config(config)
        assert config["simulator"]["episode_time_steps"] == 385
        assert config["tracking"]["tags"]["evidence"] == "functional_smoke"
        assert config["tracking"]["tags"]["promotion_eligible"] == "False"


def test_causal_signal_is_registered_as_raw_observation_pipeline_manager():
    assert ALGORITHM_REGISTRY["CausalPriceSignal"]._use_raw_observations is True
    unit = build_execution_unit(
        {
            "pipeline": [
                {"algorithm": "CausalPriceSignal", "count": 1, "frozen": True},
                {"algorithm": "SignalAwareRBCSmartLocal", "count": 1, "frozen": True},
            ]
        }
    )
    assert unit.stages[0].__class__.__name__ == "CausalPriceSignalAgent"
