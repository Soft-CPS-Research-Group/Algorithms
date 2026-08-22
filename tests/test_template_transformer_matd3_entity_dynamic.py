"""Contracts for the shipped Transformer MATD3 dynamic templates."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from algorithms.registry import ALGORITHM_REGISTRY
from algorithms.transformer_matd3.agent import AgentTransformerMATD3
from utils.config_schema import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "configs/templates/dynamic"
TEMPLATES = {
    "default": TEMPLATE_DIR / "transformer_matd3_entity_dynamic.yaml",
    "residual": TEMPLATE_DIR / "transformer_matd3_entity_dynamic_residual.yaml",
    "bc": TEMPLATE_DIR / "transformer_matd3_entity_dynamic_bc.yaml",
}


def _load(name: str) -> dict:
    return yaml.safe_load(TEMPLATES[name].read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_transformer_matd3_template_validates_and_resolves(name: str) -> None:
    config = _load(name)

    validated = validate_config(config)

    assert validated.pipeline[0].algorithm == "AgentTransformerMATD3"
    assert ALGORITHM_REGISTRY["AgentTransformerMATD3"] is AgentTransformerMATD3
    assert config["simulator"]["interface"] == "entity"
    assert config["simulator"]["topology_mode"] == "dynamic"
    assert config["simulator"]["entity_encoding"]["profile"] == "minmax_space"
    assert config["tracking"]["mlflow_enabled"] is False
    assert config["simulator"]["export"]["final_episode_only"] is True


def test_default_template_disables_optional_action_paths() -> None:
    stage = _load("default")["pipeline"][0]
    hyperparameters = stage["hyperparameters"]
    behavior_cloning = stage["behavior_cloning"]

    assert hyperparameters["residual_policy_enabled"] is False
    assert hyperparameters["local_action_safety_enabled"] is False
    assert hyperparameters["local_price_conditioning_enabled"] is False
    assert behavior_cloning["replay_based"]["enabled"] is False
    assert behavior_cloning["demonstration_based"]["enabled"] is False


def test_residual_template_declares_runtime_export_dependency() -> None:
    hyperparameters = _load("residual")["pipeline"][0]["hyperparameters"]

    assert hyperparameters["residual_policy_enabled"] is True
    assert hyperparameters["warm_start_policy_name"] == "RBCSmartPolicy"
    assert hyperparameters["residual_policy_runtime_only_export"] is True


def test_bc_template_enables_independent_bc_paths() -> None:
    stage = _load("bc")["pipeline"][0]
    behavior_cloning = stage["behavior_cloning"]
    hyperparameters = stage["hyperparameters"]

    replay = behavior_cloning["replay_based"]
    demonstration = behavior_cloning["demonstration_based"]
    assert replay["enabled"] is True
    assert replay["teacher"] == "replay_action"
    assert demonstration["enabled"] is True
    assert demonstration["demonstration_episodes"] >= 1
    assert demonstration["teacher"]["policy"] == "RBCSmartPolicy"
    assert hyperparameters["local_action_safety_enabled"] is True
    assert hyperparameters["local_action_safety_ev_minimum_mode"] == "deadline_feasible"
    assert hyperparameters["local_action_safety_protect_ev_service_target"] is True
    assert hyperparameters["local_action_safety_runtime_only_export"] is True


def test_templates_reference_the_validated_tokenizer() -> None:
    from utils.entity_tokenizer_schema import (
        _load_default_sample,
        load_entity_tokenizer_config,
        validate_against_payload,
    )

    tokenizer_path = REPO_ROOT / _load("default")["pipeline"][0][
        "tokenizer_config_path"
    ]
    tokenizer = load_entity_tokenizer_config(tokenizer_path)
    action_names = [
        ["electrical_storage", "electric_vehicle_storage"] for _ in range(3)
    ]

    validate_against_payload(tokenizer, _load_default_sample(), action_names)


def test_no_transformer_compatibility_shim_or_cross_algorithm_import_remains() -> None:
    shim_names = {
        "behavior_cloning.py",
        "entity_observation_tokenizer.py",
        "entity_token_layout.py",
        "transformer_backbone.py",
    }
    ppo_dir = REPO_ROOT / "algorithms/transformer_ppo"
    assert shim_names.isdisjoint(path.name for path in ppo_dir.glob("*.py"))

    matd3_dir = REPO_ROOT / "algorithms/transformer_matd3"
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in matd3_dir.glob("*.py")
    )
    assert "algorithms.transformer_ppo" not in sources
