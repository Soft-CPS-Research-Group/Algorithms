"""Tests for entity tokenizer JSON schema and the 5 hard-fail validation rules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


def test_valid_json_loads():
    from utils.entity_tokenizer_schema import load_entity_tokenizer_config

    cfg = load_entity_tokenizer_config(
        "configs/tokenizers/entity_default.json"
    )
    assert cfg.type_embeddings == {"SRO": 0, "NFC": 1, "CA": 2}
    assert cfg.nfc.type_name == "building_nfc"
    assert cfg.nfc.entity_table == "building"
    assert cfg.nfc.expression.op == "subtract"
    assert "storage" in cfg.ca_types
    assert "charger" in cfg.ca_types
    assert "district_time" in cfg.sro_types
    assert "pv" in cfg.sro_types


def test_missing_ca_type_raises():
    from utils.entity_tokenizer_schema import EntityTokenizerConfig

    raw = json.loads(
        Path("configs/tokenizers/entity_default.json").read_text()
    )
    raw["ca_types"].pop("charger")
    with pytest.raises(ValueError, match="charger"):
        EntityTokenizerConfig.model_validate(raw)


def test_unknown_field_raises():
    from utils.entity_tokenizer_schema import EntityTokenizerConfig

    raw = json.loads(
        Path("configs/tokenizers/entity_default.json").read_text()
    )
    raw["mystery_extra_field"] = 42
    with pytest.raises(ValueError, match="mystery_extra_field|extra"):
        EntityTokenizerConfig.model_validate(raw)


def test_default_payload_sample_loads_from_repo():
    from utils.entity_tokenizer_schema import _load_default_sample

    sample = _load_default_sample()
    # Expected coverage: district has 46 features, building has 38.
    assert len(sample.feature_names_per_table["district"]) == 46
    assert len(sample.feature_names_per_table["building"]) == 38
    assert "district__hour" in sample.feature_names_per_table["district"]
    assert (
        "non_shiftable_load" in sample.feature_names_per_table["building"]
    )


# ---------------------------------------------------------------------------
# Rule 1 — coverage
# ---------------------------------------------------------------------------


def test_rule1_default_config_passes_coverage():
    from utils.entity_tokenizer_schema import (
        _load_default_sample,
        load_entity_tokenizer_config,
        validate_against_payload,
    )

    cfg = load_entity_tokenizer_config(
        "configs/tokenizers/entity_default.json"
    )
    sample = _load_default_sample()
    action_names = [["electrical_storage", "electric_vehicle_storage"]]
    validate_against_payload(cfg, sample, action_names)  # must not raise


def test_rule1_unmatched_feature_fails():
    from utils.entity_tokenizer_schema import (
        EntityPayloadSample,
        load_entity_tokenizer_config,
        validate_against_payload,
    )

    cfg = load_entity_tokenizer_config(
        "configs/tokenizers/entity_default.json"
    )
    sample = EntityPayloadSample(
        feature_names_per_table={
            "district": ["district__hour", "district__some_new_thing"],
            "building": [
                "non_shiftable_load",
                "solar_generation",
                "electrical_storage_soc",
                "electrical_storage_soc_ratio",
            ],
            "storage": [],
            "charger": [],
            "pv": [],
            "ev": [],
        },
    )
    with pytest.raises(ValueError, match="district__some_new_thing"):
        validate_against_payload(
            cfg,
            sample,
            [["electrical_storage", "electric_vehicle_storage"]],
        )


# ---------------------------------------------------------------------------
# Rule 2 — uniqueness
# ---------------------------------------------------------------------------


def test_rule2_ambiguous_pattern_fails():
    from utils.entity_tokenizer_schema import (
        EntityTokenizerConfig,
        _load_default_sample,
        validate_against_payload,
    )

    cfg_dict = json.loads(
        Path("configs/tokenizers/entity_default.json").read_text()
    )
    cfg_dict["sro_types"]["district_time_dup"] = {
        "entity_table": "district",
        "cardinality": "singleton",
        "feature_patterns": ["^district__hour$"],
        "input_dim_fallback": 1,
    }
    cfg = EntityTokenizerConfig.model_validate(cfg_dict)
    with pytest.raises(
        ValueError,
        match=r"district__hour.*district_time.*district_time_dup|district_time_dup.*district_time",
    ):
        validate_against_payload(
            cfg,
            _load_default_sample(),
            [["electrical_storage", "electric_vehicle_storage"]],
        )


def test_excluded_feature_cannot_match_an_sro_type():
    """If a feature matches both excluded and an SRO pattern, rule 2 catches it."""
    from utils.entity_tokenizer_schema import (
        EntityTokenizerConfig,
        _load_default_sample,
        validate_against_payload,
    )

    cfg_dict = json.loads(
        Path("configs/tokenizers/entity_default.json").read_text()
    )
    cfg_dict["sro_types"]["district_meta"]["feature_patterns"].append(
        "^district__topology_version$"
    )
    cfg = EntityTokenizerConfig.model_validate(cfg_dict)
    with pytest.raises(ValueError, match="district__topology_version"):
        validate_against_payload(
            cfg,
            _load_default_sample(),
            [["electrical_storage", "electric_vehicle_storage"]],
        )


# ---------------------------------------------------------------------------
# Rule 3 — NFC sources
# ---------------------------------------------------------------------------


def test_rule3_missing_nfc_source_fails():
    from utils.entity_tokenizer_schema import (
        EntityTokenizerConfig,
        _load_default_sample,
        validate_against_payload,
    )

    cfg_dict = json.loads(
        Path("configs/tokenizers/entity_default.json").read_text()
    )
    cfg_dict["nfc"]["expression"]["left"]["feature"] = "does_not_exist"
    cfg = EntityTokenizerConfig.model_validate(cfg_dict)
    with pytest.raises(ValueError, match="does_not_exist"):
        validate_against_payload(
            cfg,
            _load_default_sample(),
            [["electrical_storage", "electric_vehicle_storage"]],
        )


# ---------------------------------------------------------------------------
# Rule 4 — regex compilation
# ---------------------------------------------------------------------------


def test_rule4_bad_regex_fails():
    from utils.entity_tokenizer_schema import (
        EntityTokenizerConfig,
        _load_default_sample,
        validate_against_payload,
    )

    cfg_dict = json.loads(
        Path("configs/tokenizers/entity_default.json").read_text()
    )
    cfg_dict["sro_types"]["district_time"]["feature_patterns"] = ["^[unclosed"]
    cfg = EntityTokenizerConfig.model_validate(cfg_dict)
    with pytest.raises(
        ValueError, match=r"sro_types\.district_time\.feature_patterns"
    ):
        validate_against_payload(
            cfg,
            _load_default_sample(),
            [["electrical_storage", "electric_vehicle_storage"]],
        )


# ---------------------------------------------------------------------------
# Rule 5 — action-field coverage
# ---------------------------------------------------------------------------


def test_rule5_missing_action_field_fails():
    from utils.entity_tokenizer_schema import (
        _load_default_sample,
        load_entity_tokenizer_config,
        validate_against_payload,
    )

    cfg = load_entity_tokenizer_config(
        "configs/tokenizers/entity_default.json"
    )
    sample = _load_default_sample()
    # Building 0 missing the charger action_field.
    bad_action_names = [["electrical_storage"]]
    with pytest.raises(ValueError, match="electric_vehicle_storage"):
        validate_against_payload(cfg, sample, bad_action_names)


def test_rule5_default_passes():
    from utils.entity_tokenizer_schema import (
        _load_default_sample,
        load_entity_tokenizer_config,
        validate_against_payload,
    )

    cfg = load_entity_tokenizer_config(
        "configs/tokenizers/entity_default.json"
    )
    sample = _load_default_sample()
    action_names = [["electrical_storage", "electric_vehicle_storage"]]
    validate_against_payload(cfg, sample, action_names)


# ---------------------------------------------------------------------------
# Excluded patterns matcher
# ---------------------------------------------------------------------------


def test_excluded_pattern_matches_topology_version():
    from utils.entity_tokenizer_schema import (
        _excluded_matchers,
        load_entity_tokenizer_config,
    )

    cfg = load_entity_tokenizer_config(
        "configs/tokenizers/entity_default.json"
    )
    matchers = _excluded_matchers(cfg)
    assert any(p.fullmatch("district__topology_version") for p in matchers)
    assert any(
        p.fullmatch("electric_vehicle_charger_state") for p in matchers
    )
    assert not any(p.fullmatch("district__hour") for p in matchers)


def test_excluded_feature_pattern_removes_topology_version():
    """The exclusion regex removes the ``topology_version`` feature."""
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
    from tests._entity_sample_obs_names import (
        load_sample_observation_names_for_first_building,
    )
    from utils.entity_tokenizer_schema import load_entity_tokenizer_config

    cfg = load_entity_tokenizer_config(
        "configs/tokenizers/entity_default.json"
    )
    builder = EntityTokenLayoutBuilder(cfg)
    layout = builder.build(
        "Building_1",
        load_sample_observation_names_for_first_building(),
        ["electrical_storage", "electric_vehicle_storage"],
    )
    assert "district__topology_version" in layout.excluded_feature_names


# ---------------------------------------------------------------------------
# validate_config integration (Tasks 12-13)
# ---------------------------------------------------------------------------


def _make_minimal_transformer_ppo_cfg(
    *,
    tokenizer_path: str = "configs/tokenizers/entity_default.json",
) -> Dict[str, Any]:
    """Construct a minimal ProjectConfig dict naming AgentTransformerPPO."""
    return {
        "metadata": {
            "experiment_name": "x",
            "run_name": "x",
            "community_name": "x",
        },
        "runtime": {},
        "tracking": {
            "mlflow_enabled": False,
            "log_level": "INFO",
            "log_frequency": 1,
            "mlflow_step_sample_interval": 10,
            "mlflow_artifacts_profile": "minimal",
            "progress_updates_enabled": True,
            "progress_update_interval": 5,
            "system_metrics_enabled": False,
            "system_metrics_interval": 10,
        },
        "checkpointing": {
            "resume_training": False,
            "checkpoint_artifact": "x.pt",
            "use_best_checkpoint_artifact": False,
            "reset_replay_buffer": False,
            "freeze_pretrained_layers": False,
            "fine_tune": False,
        },
        "bundle": {
            "require_observations_envelope": False,
            "artifact_config": {},
            "per_agent_artifact_config": {},
        },
        "simulator": {
            "dataset_name": "citylearn_three_phase_dynamic_assets_only_demo",
            "dataset_path": (
                "./datasets/citylearn_three_phase_dynamic_assets_only_demo"
                "/schema.json"
            ),
            "central_agent": False,
            "interface": "entity",
            "topology_mode": "dynamic",
            "entity_encoding": {
                "enabled": True,
                "normalization": "minmax_space",
                "clip": True,
            },
            "reward_function": "RewardFunction",
            "reward_function_kwargs": {},
            "episodes": 1,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": 100,
            "episode_time_steps": 101,
            "export": {
                "mode": "end",
                "export_kpis_on_episode_end": True,
            },
            "wrapper_reward": {
                "enabled": False,
                "profile": "cost_limits_v1",
                "clip_enabled": True,
                "clip_min": -10.0,
                "clip_max": 10.0,
                "squash": "none",
            },
        },
        "training": {
            "seed": 0,
            "steps_between_training_updates": 1,
            "target_update_interval": 0,
        },
        "topology": {},
        "pipeline": [
            {
                "algorithm": "AgentTransformerPPO",
                "count": 1,
                "tokenizer_config_path": tokenizer_path,
                "transformer": {
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                    "dim_feedforward": 128,
                    "dropout": 0.1,
                },
                "hyperparameters": {
                    "learning_rate": 3.0e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_eps": 0.2,
                    "ppo_epochs": 4,
                    "minibatch_size": 64,
                    "entropy_coeff": 0.01,
                    "value_coeff": 0.5,
                    "max_grad_norm": 0.5,
                },
            }
        ],
        "execution": None,
    }


def test_validate_config_accepts_transformer_ppo_algorithm():
    from utils.config_schema import validate_config

    validate_config(_make_minimal_transformer_ppo_cfg())


def test_validate_config_loads_tokenizer_json(tmp_path):
    """validate_config opens the JSON path and runs the 5 rules."""
    from utils.config_schema import validate_config

    bad_tokenizer = tmp_path / "bad.json"
    bad_tokenizer.write_text(
        json.dumps(
            {
                "type_embeddings": {"SRO": 0, "NFC": 1, "CA": 2},
                "excluded_features": {"patterns": []},
                "nfc": {
                    "type_name": "x",
                    "entity_table": "building",
                    "expression": {
                        "op": "subtract",
                        "left": {"feature": "ghost"},
                        "right": {"feature": "solar_generation"},
                    },
                },
                "ca_types": {
                    "storage": {
                        "entity_table": "storage",
                        "action_field": "electrical_storage",
                        "input_dim_fallback": 1,
                    },
                    "charger": {
                        "entity_table": "charger",
                        "action_field": "electric_vehicle_storage",
                        "input_dim_fallback": 1,
                    },
                },
                "sro_types": {},
                "validation": {
                    "unmatched_features": "fail",
                    "ambiguous_pattern_match": "fail",
                    "input_dim_mismatch": "fail",
                },
            }
        )
    )
    cfg = _make_minimal_transformer_ppo_cfg(tokenizer_path=str(bad_tokenizer))
    with pytest.raises(ValueError, match="ghost"):
        validate_config(cfg)
