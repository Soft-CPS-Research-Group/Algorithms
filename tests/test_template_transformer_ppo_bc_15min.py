"""Contracts for 15-minute Transformer-PPO demonstration templates."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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

    validate_config(config)
