"""Contracts for 15-minute Transformer-PPO demonstration templates."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
from algorithms.registry import build_execution_unit
from utils.config_schema import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_CASES = (
    (
        "smoke",
        REPO_ROOT
        / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_smoke.yaml",
        1,
        64,
    ),
    (
        "week",
        REPO_ROOT
        / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_week.yaml",
        1,
        672,
    ),
    (
        "month",
        REPO_ROOT
        / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_month.yaml",
        1,
        2880,
    ),
    (
        "year",
        REPO_ROOT
        / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_year.yaml",
        1,
        34816,
    ),
)
SMOKE_TEMPLATE_PATH = TEMPLATE_CASES[0][1]
DATASET_PATH = (
    REPO_ROOT
    / "datasets/citylearn_three_phase_dynamic_assets_only_demo_15min_parquet/schema.json"
)


def _load_template(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _assert_no_legacy_bc_fields(value: object) -> None:
    legacy_fields = {
        "phaseout_steps",
        "phaseout_mode",
        "noise_scale",
        "warm_start",
    }
    if isinstance(value, dict):
        assert not legacy_fields.intersection(value)
        for child in value.values():
            _assert_no_legacy_bc_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_legacy_bc_fields(child)


@pytest.mark.parametrize(
    ("duration", "template_path", "demonstration_episodes", "sample_limit"),
    TEMPLATE_CASES,
    ids=[case[0] for case in TEMPLATE_CASES],
)
def test_15min_bc_templates_use_valid_demonstration_contracts(
    duration: str,
    template_path: Path,
    demonstration_episodes: int,
    sample_limit: int,
) -> None:
    config = _load_template(template_path)
    stage = config["pipeline"][0]
    transformer = stage["transformer"]
    hyperparameters = stage["hyperparameters"]
    behavior_cloning = stage["behavior_cloning"]

    for name in (config["metadata"]["experiment_name"], config["metadata"]["run_name"]):
        assert "15min" in name.lower()
        assert duration in name.lower()
        assert "blend" not in name.lower()
    assert transformer["dropout"] == pytest.approx(0.0)
    assert config["training"]["steps_between_training_updates"] == 256
    assert config["training"]["steps_between_training_updates"] >= hyperparameters["minibatch_size"]
    assert "actor_log_std_init" in hyperparameters
    assert behavior_cloning["enabled"] is True
    assert behavior_cloning["demonstration_episodes"] == demonstration_episodes
    assert behavior_cloning["max_samples_per_building"] == sample_limit
    assert behavior_cloning["pretraining_epochs"] >= 1
    assert behavior_cloning["batch_size"] >= 1
    assert behavior_cloning["weight"] == pytest.approx(0.42)
    assert behavior_cloning["min_weight"] == pytest.approx(0.0)
    assert behavior_cloning["decay_start_step"] == 0
    assert behavior_cloning["decay_steps"] == sample_limit
    assert behavior_cloning["ev_multiplier"] == pytest.approx(24.0)
    assert behavior_cloning["storage_multiplier"] == pytest.approx(0.18)
    assert behavior_cloning["teacher"] == {
        "policy": "RBCSmartPolicy",
        "deterministic": True,
        "hyperparameters": {},
    }
    _assert_no_legacy_bc_fields(behavior_cloning)
    assert config["simulator"]["episodes"] == 3
    assert config["simulator"]["deterministic_finish"] is True
    assert (
        config["simulator"]["episodes"]
        - behavior_cloning["demonstration_episodes"]
        - int(config["simulator"]["deterministic_finish"])
    ) == 1

    validate_config(config)


def test_smoke_template_window_crosses_the_shifted_dynamic_topology_event() -> None:
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    simulator = _load_template(SMOKE_TEMPLATE_PATH)["simulator"]

    assert dataset["seconds_per_time_step"] == 900
    assert dataset["topology_mode"] == "dynamic"
    event = next(event for event in dataset["topology_events"] if event["time_step"] == 5200)
    assert simulator["simulation_start_time_step"] == 5184
    assert simulator["simulation_end_time_step"] == 5248
    assert event["time_step"] + simulator["topology_event_time_offset"] == 16


def test_smoke_template_builds_a_demonstration_enabled_tppo_agent() -> None:
    agent = build_execution_unit(_load_template(SMOKE_TEMPLATE_PATH))

    assert isinstance(agent, AgentTransformerPPO)
    assert agent._bc is not None
    assert agent._bc.policy == "RBCSmartPolicy"
