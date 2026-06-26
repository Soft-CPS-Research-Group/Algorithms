"""WP06 — Schema/registry sanity tests for the transformer-PPO entity-dynamic template."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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


def test_template_resolves_to_registered_agent() -> None:
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    from algorithms.registry import ALGORITHM_REGISTRY

    cfg = _load_template()
    assert cfg["pipeline"][0]["algorithm"] == "AgentTransformerPPO"
    assert ALGORITHM_REGISTRY["AgentTransformerPPO"] is AgentTransformerPPO


def test_template_tokenizer_path_validates_against_bundled_sample() -> None:
    """Tokenizer JSON pointed to by the template MUST pass §13.4 5-rule
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
