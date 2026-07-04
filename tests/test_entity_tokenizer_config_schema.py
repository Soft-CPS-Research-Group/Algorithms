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


