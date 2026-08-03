"""Contracts for the local Transformer-PPO demonstration template."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from utils.config_schema import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    REPO_ROOT / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic.yaml"
)
DOC_PATH = REPO_ROOT / "docs/transformer_ppo_spec.md"


def _load_template() -> dict:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
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


def test_local_bc_template_uses_demonstrations_without_action_blending() -> None:
    config = _load_template()
    stage = config["pipeline"][0]
    transformer = stage["transformer"]
    hyperparameters = stage["hyperparameters"]
    behavior_cloning = stage["behavior_cloning"]

    assert transformer["dropout"] == pytest.approx(0.0)
    assert config["training"]["steps_between_training_updates"] == 256
    assert config["training"]["steps_between_training_updates"] >= hyperparameters["minibatch_size"]
    assert "actor_log_std_init" in hyperparameters
    assert behavior_cloning["enabled"] is True
    assert behavior_cloning["demonstration_episodes"] >= 1
    assert behavior_cloning["max_samples_per_building"] == 3400
    assert behavior_cloning["pretraining_epochs"] >= 1
    assert behavior_cloning["batch_size"] >= 1
    assert behavior_cloning["weight"] == pytest.approx(0.42)
    assert behavior_cloning["min_weight"] == pytest.approx(0.24)
    assert behavior_cloning["decay_start_step"] == 512
    assert behavior_cloning["decay_steps"] == 3584
    assert behavior_cloning["ev_multiplier"] == pytest.approx(24.0)
    assert behavior_cloning["storage_multiplier"] == pytest.approx(0.18)
    assert behavior_cloning["teacher"] == {
        "policy": "RBCSmartPolicy",
        "deterministic": True,
        "hyperparameters": {},
    }
    _assert_no_legacy_bc_fields(behavior_cloning)

    validate_config(config)


def test_docs_define_the_remaining_tppo_correctness_decisions() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").lower()

    assert "pending decisions" in text
    assert "huber" in text
    assert "value normalization" in text
    assert "separate demonstration episodes" in text
    assert "actor-only ppo" in text
    assert "final deterministic evaluation" in text
    assert "required diagnostics" in text
    assert "rbcsmartpolicy" in text
    assert "action blending" not in text
