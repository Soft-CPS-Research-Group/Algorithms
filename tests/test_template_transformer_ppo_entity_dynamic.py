"""Contracts for the local Transformer-PPO entity-dynamic template."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from utils.config_schema import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml"


def _load_template() -> dict:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_template_has_a_valid_non_bc_tppo_training_contract() -> None:
    config = _load_template()
    stage = config["pipeline"][0]
    transformer = stage["transformer"]
    hyperparameters = stage["hyperparameters"]

    assert stage["algorithm"] == "AgentTransformerPPO"
    assert transformer["dropout"] == pytest.approx(0.0)
    assert config["training"]["steps_between_training_updates"] == 256
    assert config["training"]["steps_between_training_updates"] >= hyperparameters["minibatch_size"]
    assert "actor_log_std_init" in hyperparameters
    assert "behavior_cloning" not in stage

    validate_config(config)
