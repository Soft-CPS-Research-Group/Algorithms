"""Schema/registry sanity tests for the transformer-PPO entity-dynamic template."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml"


def _load_template() -> dict:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_template_passes_schema_validation() -> None:
    """The shipped template MUST validate against the v2 config schema."""
    from utils.config_schema import validate_config

    cfg = _load_template()
    validate_config(cfg)  # should not raise


def test_transformer_ppo_stage_accepts_behavior_cloning_config() -> None:
    from utils.config_schema import validate_config

    cfg = _load_template()
    cfg["pipeline"][0]["behavior_cloning"] = {
        "enabled": True,
        "weight": 0.42,
        "min_weight": 0.24,
        "decay_start_step": 512,
        "decay_steps": 3584,
        "ev_multiplier": 24.0,
        "storage_multiplier": 0.18,
        "warm_start": {
            "policy": "RBCCommunityPolicy",
            "deterministic": True,
            "noise_scale": 0.0,
            "phaseout_steps": 6144,
            "phaseout_mode": "blend",
            "hyperparameters": {},
        },
    }

    stage = validate_config(cfg).pipeline[0]

    assert stage.behavior_cloning is not None
    assert stage.behavior_cloning.enabled is True
    assert stage.behavior_cloning.weight == pytest.approx(0.42)
    assert stage.behavior_cloning.min_weight == pytest.approx(0.24)
    assert stage.behavior_cloning.decay_start_step == 512
    assert stage.behavior_cloning.decay_steps == 3584
    assert stage.behavior_cloning.ev_multiplier == pytest.approx(24.0)
    assert stage.behavior_cloning.storage_multiplier == pytest.approx(0.18)
    assert stage.behavior_cloning.warm_start is not None
    assert stage.behavior_cloning.warm_start.policy == "RBCCommunityPolicy"
    assert stage.behavior_cloning.warm_start.deterministic is True
    assert stage.behavior_cloning.warm_start.noise_scale == pytest.approx(0.0)
    assert stage.behavior_cloning.warm_start.phaseout_steps == 6144
    assert stage.behavior_cloning.warm_start.phaseout_mode == "blend"
    assert stage.behavior_cloning.warm_start.hyperparameters == {}


def test_transformer_ppo_stage_rejects_invalid_behavior_cloning_phaseout_mode() -> None:
    from utils.config_schema import validate_config

    cfg = _load_template()
    cfg["pipeline"][0]["behavior_cloning"] = {
        "warm_start": {
            "policy": "RBCCommunityPolicy",
            "phaseout_mode": "invalid",
        },
    }

    with pytest.raises(ValidationError):
        validate_config(cfg)


def test_transformer_ppo_stage_rejects_enabled_behavior_cloning_without_warm_start() -> None:
    from utils.config_schema import validate_config

    cfg = _load_template()
    cfg["pipeline"][0]["behavior_cloning"] = {
        "enabled": True,
        "weight": 0.42,
    }

    with pytest.raises(ValidationError, match="warm_start"):
        validate_config(cfg)


def test_transformer_ppo_stage_without_behavior_cloning_defaults_to_none() -> None:
    from utils.config_schema import validate_config

    cfg = _load_template()

    stage = validate_config(cfg).pipeline[0]

    assert stage.behavior_cloning is None


def test_template_resolves_to_registered_agent() -> None:
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    from algorithms.registry import ALGORITHM_REGISTRY

    cfg = _load_template()
    assert cfg["pipeline"][0]["algorithm"] == "AgentTransformerPPO"
    assert ALGORITHM_REGISTRY["AgentTransformerPPO"] is AgentTransformerPPO


def test_template_tokenizer_path_validates_against_bundled_sample() -> None:
    """Tokenizer JSON pointed to by the template MUST pass the 5-rule
    validation against the bundled sample payload + per-building action_field
    declarations declared by the simulator."""
    from utils.entity_tokenizer_schema import (
        _load_default_sample,
        load_entity_tokenizer_config,
        validate_against_payload,
    )

    cfg = _load_template()
    tok_path = REPO_ROOT / cfg["pipeline"][0]["tokenizer_config_path"]
    assert tok_path.exists(), tok_path

    tok = load_entity_tokenizer_config(tok_path)
    sample = _load_default_sample()
    # The bundled sample has 3 buildings; the tokenizer rule 5 needs every
    # CA action_field to appear in each building's action_names. Provide the
    # canonical assets-only action set per building.
    action_names_per_building = [
        ["electrical_storage", "electric_vehicle_storage"],
        ["electrical_storage", "electric_vehicle_storage"],
        ["electrical_storage", "electric_vehicle_storage"],
    ]
    validate_against_payload(tok, sample, action_names_per_building)
