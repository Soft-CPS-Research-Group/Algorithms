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
    "cost4": TEMPLATE_DIR / "transformer_matd3_entity_dynamic_cost4_faithful.yaml",
    "cost4_realistic_pilot": TEMPLATE_DIR / "transformer_matd3_entity_dynamic_cost4_realistic_pilot.yaml",
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


def test_templates_declare_actor_policy_loss_weight() -> None:
    for name in sorted(TEMPLATES):
        hyperparameters = _load(name)["pipeline"][0]["hyperparameters"]
        assert hyperparameters["actor_policy_loss_weight"] >= 0.0


def test_cost4_template_matches_named_recipe_translation() -> None:
    stage = _load("cost4")["pipeline"][0]
    hyperparameters = stage["hyperparameters"]
    replay = stage["behavior_cloning"]["replay_based"]

    assert _load("cost4")["simulator"]["dataset_name"].endswith("15min_parquet")
    assert hyperparameters["actor_policy_loss_weight"] == pytest.approx(0.085)
    assert hyperparameters["warm_start_policy_name"] == "RBCCommunityPolicy"
    assert hyperparameters["residual_action_final_scale"] == pytest.approx(0.30)
    assert hyperparameters["residual_storage_action_scale_multiplier"] == pytest.approx(0.75)
    assert hyperparameters["residual_ev_action_scale_multiplier"] == pytest.approx(0.25)
    assert replay["weight"] == pytest.approx(0.24)
    assert replay["ev_multiplier"] == pytest.approx(18.0)


def test_cost4_realistic_pilot_keeps_recipe_and_limits_episode_budget() -> None:
    config = _load("cost4_realistic_pilot")
    stage = config["pipeline"][0]
    hyperparameters = stage["hyperparameters"]
    replay = stage["behavior_cloning"]["replay_based"]

    assert config["simulator"]["episodes"] == 2
    assert config["simulator"]["episode_time_steps"] == 3401
    assert config["simulator"]["dataset_name"].endswith("15min_parquet")
    assert config["simulator"]["reward_function"] == (
        "CostServiceCommunityDenseEVResidualRewardV54"
    )
    assert config["simulator"]["reward_function_kwargs"] == {
        "community_settlement_cost_weight": pytest.approx(1.55),
        "battery_throughput_penalty": pytest.approx(0.0015),
    }
    assert hyperparameters["actor_policy_loss_weight"] == pytest.approx(0.085)
    assert hyperparameters["warm_start_policy_name"] == "RBCCommunityPolicy"
    assert hyperparameters["residual_action_final_scale"] == pytest.approx(0.30)
    assert hyperparameters["residual_storage_action_scale_multiplier"] == pytest.approx(0.75)
    assert hyperparameters["residual_ev_action_scale_multiplier"] == pytest.approx(0.25)
    assert replay["ev_multiplier"] == pytest.approx(18.0)


@pytest.mark.parametrize(
    "path, algorithm",
    [
        (
            TEMPLATE_DIR / "rbc_community_entity_dynamic_15min_cost4.yaml",
            "RBCCommunityPolicy",
        ),
        (
            TEMPLATE_DIR / "rbc_smart_entity_dynamic_15min_cost4.yaml",
            "RBCSmartPolicy",
        ),
    ],
)
def test_matching_dynamic_cost4_baselines_validate(path: Path, algorithm: str) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validated = validate_config(config)

    assert validated.pipeline[0].algorithm == algorithm
    assert config["simulator"]["dataset_name"].endswith("15min_parquet")
    assert config["simulator"]["topology_mode"] == "dynamic"


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
