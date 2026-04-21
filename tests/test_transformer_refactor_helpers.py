"""Tests for extracted Transformer helper classes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithms.utils.observation_enricher import ObservationEnricher


class _WrapperLike:
    def __init__(self, tokenizer_config: dict) -> None:
        self._is_transformer_agent = True
        self._enrichers = [ObservationEnricher(tokenizer_config)]
        self.observation_names = [["month", "electrical_storage_soc", "non_shiftable_load"]]
        self.action_names = [["electrical_storage"]]


def _tokenizer_config() -> dict:
    return {
        "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
        "ca_types": {
            "battery": {
                "features": ["electrical_storage_soc"],
                "action_name": "electrical_storage",
                "input_dim": 1,
            }
        },
        "sro_types": {
            "temporal": {"features": ["month"], "input_dim": 2},
        },
        "nfc": {
            "demand_features": ["non_shiftable_load"],
            "generation_features": [],
            "extra_features": [],
            "input_dim": 1,
        },
    }


def test_transformer_observation_coordinator_enriches_values() -> None:
    """Coordinator helper should inject marker values via enricher cache."""
    from utils.wrapper_transformer.transformer_observation_coordinator import (
        TransformerObservationCoordinator,
    )

    tokenizer_config = _tokenizer_config()
    wrapper = _WrapperLike(tokenizer_config)
    wrapper._enrichers[0].enrich_names(wrapper.observation_names[0], wrapper.action_names[0])

    enriched = TransformerObservationCoordinator.enrich_observation_values(
        wrapper,
        0,
        [6.0, 0.55, 130.0],
    )

    assert 1001.0 in enriched
    assert 2001.0 in enriched
    assert 3001.0 in enriched


@pytest.fixture
def transformer_config(tmp_path: Path) -> dict:
    tokenizer_config = _tokenizer_config()
    tokenizer_path = tmp_path / "tokenizer.json"
    with tokenizer_path.open("w", encoding="utf-8") as handle:
        json.dump(tokenizer_config, handle)

    return {
        "algorithm": {
            "name": "AgentTransformerPPO",
            "tokenizer_config_path": str(tokenizer_path),
            "transformer": {
                "d_model": 32,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.1,
            },
            "hyperparameters": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "ppo_epochs": 1,
                "minibatch_size": 4,
                "entropy_coeff": 0.01,
                "value_coeff": 0.5,
                "max_grad_norm": 0.5,
                "hidden_dim": 32,
                "rollout_length": 16,
            },
        }
    }


def test_export_helper_builds_dummy_observation_from_config(
    transformer_config: dict,
) -> None:
    """Export helper should include CA/SRO/NFC markers in dummy observation."""
    from algorithms.agents.transformer_ppo.export_helper import TransformerPPOExportHelper
    from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO

    agent = AgentTransformerPPO(transformer_config)

    dummy_obs = TransformerPPOExportHelper.build_dummy_observation(agent, agent_index=0)

    assert tuple(dummy_obs.shape[:1]) == (1,)
    values = dummy_obs.squeeze(0).tolist()
    assert 1001.0 in values
    assert 2001.0 in values
    assert 3001.0 in values
