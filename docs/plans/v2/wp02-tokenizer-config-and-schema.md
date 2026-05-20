# WP02 — Tokenizer Config & Schema Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For all v2 WPs:** every task that writes production code MUST follow `superpowers:test-driven-development` (red → verify-red → green → verify-green → refactor → commit). The WP MUST end with `superpowers:requesting-code-review`.

**Goal:** Author the canonical tokenizer JSON (`configs/tokenizers/entity_default.json`) per `docs/specv2.md` §13.1 and add the Pydantic schema + 5 hard-fail validation rules (§13.4, §13.5) to `utils/config_schema.py` so that `validate_config(...)` rejects malformed tokenizer configs before any agent is instantiated.

**Architecture:** Pure-Python work — **no torch import allowed in this WP**. The validation pipeline is: YAML config → existing `validate_config(...)` → if `algorithm.name == "AgentTransformerPPO"`, open the path at `algorithm.tokenizer_config_path`, parse JSON, build `EntityTokenizerConfig`, then run rules 1–5 against a real entity-payload sample (the dataset-derived one if a simulator exists, otherwise the bundled `datasets/tmp_entity_obs_full_step2200_named.json`). Validation runs once, in one place. The agent (WP05) re-runs the same rules at runtime against `entity_specs`.

**Tech Stack:** Python 3.11, Pydantic v2 (already in repo), pytest, `re` stdlib.

**Branch:** `gj/wp02-tokenizer-config`
**Base branch:** `gj/wp01-port-plan-c`

---

## Scope

**Files created:**

- `configs/tokenizers/entity_default.json` — full content from spec §13.1 (verbatim).
- `utils/entity_tokenizer_schema.py` — Pydantic models (`EntityTokenizerConfig`, `NfcConfig`, `NfcExpressionConfig`, `NfcOperandConfig`, `CaTypeConfig`, `SroSingletonTypeConfig`, `SroPerAssetTypeConfig`, `ExcludedFeaturesConfig`) + the JSON loader + the 5 validation rules.
- `tests/test_entity_tokenizer_config_schema.py` — covers all of §16.6.

**Files modified:**

- `utils/config_schema.py` — add `TransformerConfig`, `TransformerPPOHyperparameters`, `TransformerPPOAlgorithmConfig`; add `TransformerPPOAlgorithmConfig` to the `ProjectConfig.algorithm` discriminated union; in `validate_config(...)`, after structural validation, dispatch on `algorithm.name == "AgentTransformerPPO"` to invoke the tokenizer JSON loader + 5 rules.

**Out of scope:**

- Any torch usage.
- The `EntityTokenLayoutBuilder` itself (that is WP03 — but the **classification regex catalog** lives in this WP's JSON because the JSON is the single source of truth).
- The agent class.
- The wrapper hook.

---

## File Structure

```
configs/
  tokenizers/
    entity_default.json                    # NEW (verbatim from spec §13.1)
utils/
  config_schema.py                          # MODIFIED (add Transformer* models + dispatch)
  entity_tokenizer_schema.py                # NEW (Pydantic + loader + rules 1-5)
tests/
  test_entity_tokenizer_config_schema.py    # NEW (covers §16.6)
```

`utils/entity_tokenizer_schema.py` is a separate module (not inside `config_schema.py`) because it has its own surface area (≈ 6 model classes + 5 rule functions + 1 loader) and `config_schema.py` is already large. `config_schema.py` only adds the Transformer-PPO algorithm config models and a single dispatch call.

**Design split inside `entity_tokenizer_schema.py`:**

- Pydantic models — top of file.
- `load_entity_tokenizer_config(path: str) -> EntityTokenizerConfig` — single JSON loader entry point.
- `EntityPayloadSample` dataclass — wraps a feature-name listing per entity table (so rule 1/2/3 can be tested in isolation without a live wrapper).
- `validate_against_payload(cfg, sample, action_names_per_building) -> None` — runs rules 1–5; raises `ValueError` on first failure with a structured message.
- `_load_default_sample() -> EntityPayloadSample` — fallback that reads `datasets/tmp_entity_obs_full_step2200_named.json`.

---

## Tasks

### Task 1: Branch + skeleton

- [ ] **Step 1: Create branch from WP01**

```bash
git checkout gj/wp01-port-plan-c
git checkout -b gj/wp02-tokenizer-config
```

- [ ] **Step 2: Verify ports are present**

```bash
test -f algorithms/utils/transformer_backbone.py && test -f algorithms/utils/ppo_components.py && echo "OK"
```

- [ ] **Step 3: Confirm target paths absent**

```bash
test ! -e configs/tokenizers/entity_default.json && test ! -e utils/entity_tokenizer_schema.py && echo "OK"
```

---

### Task 2: Author `entity_default.json` (verbatim from spec)

This task is config-only — no code yet — so we deviate from RED/GREEN: we author the JSON, then write a single load+parse smoke test that the next task will turn green.

**Files:**
- Create: `configs/tokenizers/entity_default.json`

- [ ] **Step 1: Copy the JSON literally from `docs/specv2.md` §13.1 lines 970–1175**

The file content is reproduced here for self-containment (DO NOT diverge from the spec — if you find yourself wanting to change a regex, STOP and update the spec first):

```json
{
  "type_embeddings": { "SRO": 0, "NFC": 1, "CA": 2 },

  "excluded_features": {
    "patterns": [
      "^district__topology_version$",
      "^electric_vehicle_charger_state$",
      "^electric_vehicle_soc$",
      "^electric_vehicle_required_soc_departure$",
      "^electric_vehicle_departure_time$",
      "^electric_vehicle_is_flexible$"
    ]
  },

  "nfc": {
    "type_name": "building_nfc",
    "entity_table": "building",
    "expression": {
      "op": "subtract",
      "left":  { "feature": "non_shiftable_load" },
      "right": { "feature": "solar_generation" }
    }
  },

  "ca_types": {
    "storage": {
      "entity_table": "storage",
      "action_field": "electrical_storage",
      "input_dim_fallback": 9
    },
    "charger": {
      "entity_table": "charger",
      "action_field": "electric_vehicle_storage",
      "input_dim_fallback": 16
    }
  },

  "sro_types": {
    "district_time": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__month$",
        "^district__day_type$",
        "^district__hour$"
      ],
      "input_dim_fallback": 3
    },
    "district_weather_current": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__outdoor_dry_bulb_temperature$",
        "^district__outdoor_relative_humidity$",
        "^district__diffuse_solar_irradiance$",
        "^district__direct_solar_irradiance$"
      ],
      "input_dim_fallback": 4
    },
    "district_weather_forecast": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__outdoor_dry_bulb_temperature_predicted_\\d+$",
        "^district__outdoor_relative_humidity_predicted_\\d+$",
        "^district__diffuse_solar_irradiance_predicted_\\d+$",
        "^district__direct_solar_irradiance_predicted_\\d+$"
      ],
      "input_dim_fallback": 12
    },
    "district_carbon": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [ "^district__carbon_intensity$" ],
      "input_dim_fallback": 1
    },
    "district_pricing_current": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [ "^district__electricity_pricing$" ],
      "input_dim_fallback": 1
    },
    "district_pricing_forecast": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [ "^district__electricity_pricing_predicted_\\d+$" ],
      "input_dim_fallback": 3
    },
    "district_community_energy": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__community_(net|import|export|pv|bess|ev)_(power_kw|energy_kwh_step)$"
      ],
      "input_dim_fallback": 12
    },
    "district_community_headroom": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__community_(building|phase)(_export)?_headroom_kw$"
      ],
      "input_dim_fallback": 4
    },
    "district_community_history": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__community_net_prev_\\d+_(kwh_step|mean_kwh_step)$"
      ],
      "input_dim_fallback": 2
    },
    "district_meta": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [ "^district__active_(buildings|chargers|evs)_count$" ],
      "input_dim_fallback": 3
    },

    "building_storage_state": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [
        "^electrical_storage_soc$",
        "^electrical_storage_soc_ratio$"
      ],
      "input_dim_fallback": 2
    },
    "building_charging_phase_onehot": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [ "^charging_phase_one_hot_.+$" ],
      "input_dim_fallback": 6
    },
    "building_charging_headroom": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [
        "^charging_(building|phase_L\\d+)(_export)?_headroom_kw$"
      ],
      "input_dim_fallback": 8
    },
    "building_charging_violation": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [ "^charging_constraint_violation_kwh$" ],
      "input_dim_fallback": 1
    },
    "building_energy_current": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [
        "^net_electricity_consumption$",
        "^(net|import|export|load|pv|bess|ev_charging)_(power_kw|energy_kwh_step)$"
      ],
      "input_dim_fallback": 15
    },
    "building_energy_history": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [
        "^(net|import|export)_energy_prev_\\d+_(kwh_step|mean_kwh_step)$"
      ],
      "input_dim_fallback": 4
    },
    "building_meta": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [
        "^active_chargers_count$",
        "^active_storages_count$",
        "^active_pvs_count$"
      ],
      "input_dim_fallback": 3
    },

    "pv": {
      "entity_table": "pv",
      "cardinality": "per_asset",
      "adapter_prefix": "pv::",
      "input_dim_fallback": 3
    },
    "ev_connected": {
      "entity_table": "ev",
      "cardinality": "per_asset",
      "adapter_prefix": "charger::",
      "adapter_label": "connected_ev",
      "input_dim_fallback": 8
    },
    "ev_incoming": {
      "entity_table": "ev",
      "cardinality": "per_asset",
      "adapter_prefix": "charger::",
      "adapter_label": "incoming_ev",
      "input_dim_fallback": 8
    }
  },

  "validation": {
    "unmatched_features": "fail",
    "ambiguous_pattern_match": "fail",
    "input_dim_mismatch": "fail"
  }
}
```

- [ ] **Step 2: Validate JSON parses**

```bash
python -c "import json; json.load(open('configs/tokenizers/entity_default.json')); print('OK')"
```
Expected: `OK`. If this fails the JSON has a syntax error — fix it before proceeding.

- [ ] **Step 3: Commit**

```bash
git add configs/tokenizers/entity_default.json
git commit -m "feat(wp02): add entity_default.json tokenizer config (spec §13.1)"
```

---

### Task 3: Pydantic models — failing test first

**Files:**
- Create (test): `tests/test_entity_tokenizer_config_schema.py`
- Create (impl): `utils/entity_tokenizer_schema.py`

- [ ] **Step 1: Write the first failing test**

```python
# tests/test_entity_tokenizer_config_schema.py
"""Tests for entity tokenizer JSON schema and validation rules.

Covers spec §13.4 (5 hard-fail rules) and §16.6.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_valid_json_loads():
    from utils.entity_tokenizer_schema import load_entity_tokenizer_config

    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    assert cfg.type_embeddings == {"SRO": 0, "NFC": 1, "CA": 2}
    assert cfg.nfc.type_name == "building_nfc"
    assert cfg.nfc.entity_table == "building"
    assert cfg.nfc.expression.op == "subtract"
    assert "storage" in cfg.ca_types
    assert "charger" in cfg.ca_types
    assert "district_time" in cfg.sro_types
    assert "pv" in cfg.sro_types
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/test_entity_tokenizer_config_schema.py::test_valid_json_loads -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.entity_tokenizer_schema'`.

- [ ] **Step 3: Implement minimal Pydantic models + loader**

Create `utils/entity_tokenizer_schema.py`. The file is non-trivial; implement only what the failing test requires:

```python
"""Pydantic models, loader, and validation rules for the entity tokenizer config.

See docs/specv2.md §13.1, §13.4, §13.5.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    """Forbid unknown fields anywhere — catches typos in user config."""
    model_config = ConfigDict(extra="forbid", frozen=True)


class NfcOperandConfig(_Strict):
    feature: str


class NfcExpressionConfig(_Strict):
    op: Literal["subtract"]
    left: NfcOperandConfig
    right: NfcOperandConfig


class NfcConfig(_Strict):
    type_name: str
    entity_table: str
    expression: NfcExpressionConfig


class CaTypeConfig(_Strict):
    entity_table: str
    action_field: str
    input_dim_fallback: int = Field(ge=1)


class SroSingletonTypeConfig(_Strict):
    entity_table: str
    cardinality: Literal["singleton"]
    feature_patterns: List[str]
    input_dim_fallback: int = Field(ge=1)


class SroPerAssetTypeConfig(_Strict):
    entity_table: str
    cardinality: Literal["per_asset"]
    adapter_prefix: str
    adapter_label: Optional[str] = None
    input_dim_fallback: int = Field(ge=1)


SroTypeConfig = Union[SroSingletonTypeConfig, SroPerAssetTypeConfig]


class ExcludedFeaturesConfig(_Strict):
    patterns: List[str]


class _ValidationFlags(_Strict):
    unmatched_features: Literal["fail"]
    ambiguous_pattern_match: Literal["fail"]
    input_dim_mismatch: Literal["fail"]


class EntityTokenizerConfig(_Strict):
    type_embeddings: Dict[str, int]
    excluded_features: ExcludedFeaturesConfig
    nfc: NfcConfig
    ca_types: Dict[str, CaTypeConfig]
    sro_types: Dict[str, SroTypeConfig]
    validation: _ValidationFlags

    @field_validator("type_embeddings")
    @classmethod
    def _check_type_embeddings(cls, v: Dict[str, int]) -> Dict[str, int]:
        expected = {"SRO": 0, "NFC": 1, "CA": 2}
        if v != expected:
            raise ValueError(
                f"type_embeddings must equal {expected!r}, got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _require_storage_and_charger(self) -> "EntityTokenizerConfig":
        missing = {"storage", "charger"} - set(self.ca_types)
        if missing:
            raise ValueError(f"ca_types missing required entries: {sorted(missing)}")
        return self


def load_entity_tokenizer_config(path: str | Path) -> EntityTokenizerConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EntityTokenizerConfig.model_validate(raw)
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_entity_tokenizer_config_schema.py::test_valid_json_loads -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/entity_tokenizer_schema.py tests/test_entity_tokenizer_config_schema.py
git commit -m "feat(wp02): add Pydantic models + loader for entity tokenizer config"
```

---

### Task 4: Pydantic strictness tests (rules-by-Pydantic, not rules 1-5)

These tests exercise the Pydantic-level checks (`extra="forbid"`, required fields, type embeddings constant). One sub-step per test, using the standard RED/GREEN cycle.

**Files:**
- Modify: `tests/test_entity_tokenizer_config_schema.py`
- Possibly modify: `utils/entity_tokenizer_schema.py` (only if a test reveals a missing constraint)

For each of the following tests (covers §16.6 first three rows): write the test → run (expect FAIL or red → if it already passes because of `_Strict`, document that it passes) → if fail, fix → commit.

- [ ] **Step 1: `test_missing_ca_type_raises`**

```python
def test_missing_ca_type_raises():
    from utils.entity_tokenizer_schema import EntityTokenizerConfig

    raw = json.loads(Path("configs/tokenizers/entity_default.json").read_text())
    raw["ca_types"].pop("charger")
    with pytest.raises(ValueError, match="charger"):
        EntityTokenizerConfig.model_validate(raw)
```
Run: `pytest tests/test_entity_tokenizer_config_schema.py::test_missing_ca_type_raises -v`
Expected: PASS (because of `_require_storage_and_charger` model_validator). If it doesn't, add the missing constraint.

- [ ] **Step 2: `test_unknown_field_raises`**

```python
def test_unknown_field_raises():
    from utils.entity_tokenizer_schema import EntityTokenizerConfig

    raw = json.loads(Path("configs/tokenizers/entity_default.json").read_text())
    raw["mystery_extra_field"] = 42
    with pytest.raises(ValueError, match="mystery_extra_field|extra"):
        EntityTokenizerConfig.model_validate(raw)
```
Run: `pytest tests/test_entity_tokenizer_config_schema.py::test_unknown_field_raises -v`
Expected: PASS (because of `extra="forbid"`).

- [ ] **Step 3: Commit Pydantic-level strictness tests**

```bash
git add tests/test_entity_tokenizer_config_schema.py
git commit -m "test(wp02): cover Pydantic strictness for tokenizer schema"
```

---

### Task 5: `EntityPayloadSample` dataclass + default loader

**Files:**
- Modify: `utils/entity_tokenizer_schema.py`
- Modify: `tests/test_entity_tokenizer_config_schema.py`

- [ ] **Step 1: Write failing test**

```python
def test_default_payload_sample_loads_from_repo():
    from utils.entity_tokenizer_schema import _load_default_sample

    sample = _load_default_sample()
    # Per spec §13.1 coverage: district has 46 features, building has 38
    assert len(sample.feature_names_per_table["district"]) == 46
    assert len(sample.feature_names_per_table["building"]) == 38
    assert "district__hour" in sample.feature_names_per_table["district"]
    assert "non_shiftable_load" in sample.feature_names_per_table["building"]
```
Run: `pytest tests/test_entity_tokenizer_config_schema.py::test_default_payload_sample_loads_from_repo -v`
Expected: FAIL (`_load_default_sample` not defined).

- [ ] **Step 2: Implement**

Append to `utils/entity_tokenizer_schema.py`:

```python
@dataclass(frozen=True)
class EntityPayloadSample:
    """Lightweight view of an entity payload used for tokenizer validation.

    Only feature names (not values) are needed for rules 1-5; this lets the
    validator run before any wrapper/simulator instantiation.
    """
    feature_names_per_table: Dict[str, List[str]]
    # Per-asset adapter labels we have observed in the sample, e.g.
    # {"pv::Building_1/pv", "charger::Building_1/charger_1", ...}.
    # Used only by rule 1 to know which per-asset SRO segments are populated.
    adapter_observed_prefixes: Dict[str, List[str]] = field(default_factory=dict)


_DEFAULT_SAMPLE_PATH = Path("datasets/tmp_entity_obs_full_step2200_named.json")


def _load_default_sample() -> EntityPayloadSample:
    payload = json.loads(_DEFAULT_SAMPLE_PATH.read_text(encoding="utf-8"))
    tables = payload["tables"]
    feature_names_per_table: Dict[str, List[str]] = {}
    for table_name, table in tables.items():
        # Each table is { "<entity_id>": { "<feature>": value, ... }, ... }
        # Take the union of feature names across all rows; in practice they
        # are identical per table.
        names: List[str] = []
        seen: set[str] = set()
        for row in table.values():
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    names.append(k)
        feature_names_per_table[table_name] = names
    return EntityPayloadSample(
        feature_names_per_table=feature_names_per_table,
    )
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_entity_tokenizer_config_schema.py::test_default_payload_sample_loads_from_repo -v
```
Expected: PASS. If the count assertions fail (e.g. district != 46), STOP — the sample payload may have changed. Verify against §13.1 coverage table; if the spec is wrong, escalate; if the sample is wrong, fix the sample.

- [ ] **Step 4: Commit**

```bash
git add utils/entity_tokenizer_schema.py tests/test_entity_tokenizer_config_schema.py
git commit -m "feat(wp02): add EntityPayloadSample loader (default = bundled sample)"
```

---

### Task 6: Rule 1 — Coverage

**Files:**
- Modify: `utils/entity_tokenizer_schema.py`
- Modify: `tests/test_entity_tokenizer_config_schema.py`

- [ ] **Step 1: Failing test — happy path**

```python
def test_rule1_default_config_passes_coverage():
    from utils.entity_tokenizer_schema import (
        load_entity_tokenizer_config,
        _load_default_sample,
        validate_against_payload,
    )
    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    sample = _load_default_sample()
    # action_names per building: storage + charger (use placeholder, rule 5 tested separately)
    action_names = [["electrical_storage", "electric_vehicle_storage"]]
    validate_against_payload(cfg, sample, action_names)  # must not raise
```
Run: `pytest tests/test_entity_tokenizer_config_schema.py::test_rule1_default_config_passes_coverage -v`
Expected: FAIL (`validate_against_payload` not defined).

- [ ] **Step 2: Failing test — unmatched feature**

```python
def test_rule1_unmatched_feature_fails():
    import copy
    from utils.entity_tokenizer_schema import (
        load_entity_tokenizer_config,
        EntityPayloadSample,
        validate_against_payload,
    )
    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    sample = EntityPayloadSample(
        feature_names_per_table={
            "district": ["district__hour", "district__some_new_thing"],
            "building": ["non_shiftable_load", "solar_generation",
                         "electrical_storage_soc", "electrical_storage_soc_ratio"],
            "storage": [], "charger": [], "pv": [], "ev": [],
        },
    )
    with pytest.raises(ValueError, match="district__some_new_thing"):
        validate_against_payload(
            cfg, sample,
            [["electrical_storage", "electric_vehicle_storage"]],
        )
```
Run: confirm it FAILs (function not defined).

- [ ] **Step 3: Implement rule 1**

Append to `utils/entity_tokenizer_schema.py`:

```python
def _compile_pattern(pattern: str, json_path: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(
            f"Invalid regex at {json_path}: {pattern!r} -> {exc}"
        ) from exc


def _excluded_matchers(cfg: EntityTokenizerConfig) -> List[re.Pattern[str]]:
    return [
        _compile_pattern(p, f"excluded_features.patterns[{i}]")
        for i, p in enumerate(cfg.excluded_features.patterns)
    ]


def _sro_matchers(cfg: EntityTokenizerConfig) -> Dict[str, List[re.Pattern[str]]]:
    out: Dict[str, List[re.Pattern[str]]] = {}
    for type_name, sro in cfg.sro_types.items():
        if isinstance(sro, SroSingletonTypeConfig):
            out[type_name] = [
                _compile_pattern(p, f"sro_types.{type_name}.feature_patterns[{i}]")
                for i, p in enumerate(sro.feature_patterns)
            ]
    return out


def _classify_feature(
    feature: str,
    table: str,
    cfg: EntityTokenizerConfig,
    sro_matchers: Mapping[str, List[re.Pattern[str]]],
    excluded_matchers: List[re.Pattern[str]],
) -> List[str]:
    """Return the labels that match a feature: 'excluded', 'nfc', or sro type names.
    Empty list = unmatched. More than one label = ambiguous."""
    labels: List[str] = []
    if any(p.fullmatch(feature) for p in excluded_matchers):
        labels.append("excluded")
    if (
        table == cfg.nfc.entity_table
        and feature in (cfg.nfc.expression.left.feature, cfg.nfc.expression.right.feature)
    ):
        labels.append("nfc")
    for sro_name, sro in cfg.sro_types.items():
        if not isinstance(sro, SroSingletonTypeConfig):
            continue
        if sro.entity_table != table:
            continue
        if any(p.fullmatch(feature) for p in sro_matchers.get(sro_name, [])):
            labels.append(sro_name)
    return labels


def _validate_rule_1_coverage(
    cfg: EntityTokenizerConfig,
    sample: EntityPayloadSample,
    sro_matchers: Mapping[str, List[re.Pattern[str]]],
    excluded_matchers: List[re.Pattern[str]],
) -> None:
    # Only validate tables that are referenced by an SRO singleton type or NFC.
    referenced_tables = {cfg.nfc.entity_table}
    for sro in cfg.sro_types.values():
        if isinstance(sro, SroSingletonTypeConfig):
            referenced_tables.add(sro.entity_table)
    unmatched: List[tuple[str, str]] = []  # (table, feature)
    for table in referenced_tables:
        for feature in sample.feature_names_per_table.get(table, []):
            labels = _classify_feature(
                feature, table, cfg, sro_matchers, excluded_matchers
            )
            if not labels:
                unmatched.append((table, feature))
    if unmatched:
        bullets = "\n".join(
            f"  - table={t!r}, feature={f!r}" for t, f in unmatched
        )
        raise ValueError(
            "Tokenizer rule 1 (coverage) failed — the following features are "
            "not matched by any SRO type, NFC source, or excluded pattern:\n"
            f"{bullets}\n"
            "Fix: add to an existing SRO type's feature_patterns, define a new "
            "SRO type, or add to excluded_features.patterns."
        )


def validate_against_payload(
    cfg: EntityTokenizerConfig,
    sample: EntityPayloadSample,
    action_names_per_building: List[List[str]],
) -> None:
    """Run all 5 hard-fail rules. Raises ValueError on first failure."""
    sro_matchers = _sro_matchers(cfg)
    excluded_matchers = _excluded_matchers(cfg)
    _validate_rule_1_coverage(cfg, sample, sro_matchers, excluded_matchers)
    # Rules 2-5 added in subsequent tasks.
```

- [ ] **Step 4: Run both rule-1 tests**

```bash
pytest tests/test_entity_tokenizer_config_schema.py -k rule1 -v
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/entity_tokenizer_schema.py tests/test_entity_tokenizer_config_schema.py
git commit -m "feat(wp02): implement tokenizer validation rule 1 (feature coverage)"
```

---

### Task 7: Rule 2 — Uniqueness (no feature in two SRO types)

**Files:**
- Modify: `utils/entity_tokenizer_schema.py`
- Modify: `tests/test_entity_tokenizer_config_schema.py`

- [ ] **Step 1: Failing test**

```python
def test_rule2_ambiguous_pattern_fails():
    from utils.entity_tokenizer_schema import (
        load_entity_tokenizer_config,
        _load_default_sample,
        validate_against_payload,
    )
    cfg_dict = json.loads(Path("configs/tokenizers/entity_default.json").read_text())
    # Inject overlap: another SRO type that also matches district__hour
    cfg_dict["sro_types"]["district_time_dup"] = {
        "entity_table": "district",
        "cardinality": "singleton",
        "feature_patterns": ["^district__hour$"],
        "input_dim_fallback": 1,
    }
    from utils.entity_tokenizer_schema import EntityTokenizerConfig
    cfg = EntityTokenizerConfig.model_validate(cfg_dict)
    with pytest.raises(ValueError, match=r"district__hour.*district_time.*district_time_dup|district_time_dup.*district_time"):
        validate_against_payload(
            cfg, _load_default_sample(),
            [["electrical_storage", "electric_vehicle_storage"]],
        )


def test_excluded_feature_cannot_match_an_sro_type():
    """If a feature matches both an excluded pattern and an SRO pattern, rule 2 catches it."""
    from utils.entity_tokenizer_schema import (
        EntityTokenizerConfig, _load_default_sample, validate_against_payload,
    )
    cfg_dict = json.loads(Path("configs/tokenizers/entity_default.json").read_text())
    # district__topology_version is excluded by default; also force an SRO match.
    cfg_dict["sro_types"]["district_meta"]["feature_patterns"].append(
        "^district__topology_version$"
    )
    cfg = EntityTokenizerConfig.model_validate(cfg_dict)
    with pytest.raises(ValueError, match="district__topology_version"):
        validate_against_payload(
            cfg, _load_default_sample(),
            [["electrical_storage", "electric_vehicle_storage"]],
        )
```
Run: confirm both FAIL (currently rule 2 not implemented → either test passes silently or rule 1 catches a different issue). FAIL = expected. If they pass, STOP.

- [ ] **Step 2: Implement rule 2**

Add inside `utils/entity_tokenizer_schema.py`:

```python
def _validate_rule_2_uniqueness(
    cfg: EntityTokenizerConfig,
    sample: EntityPayloadSample,
    sro_matchers: Mapping[str, List[re.Pattern[str]]],
    excluded_matchers: List[re.Pattern[str]],
) -> None:
    referenced_tables = {cfg.nfc.entity_table}
    for sro in cfg.sro_types.values():
        if isinstance(sro, SroSingletonTypeConfig):
            referenced_tables.add(sro.entity_table)
    conflicts: List[tuple[str, str, List[str]]] = []  # (table, feature, labels)
    for table in referenced_tables:
        for feature in sample.feature_names_per_table.get(table, []):
            labels = _classify_feature(
                feature, table, cfg, sro_matchers, excluded_matchers
            )
            if len(labels) > 1:
                conflicts.append((table, feature, labels))
    if conflicts:
        bullets = "\n".join(
            f"  - table={t!r}, feature={f!r}, matches={labels}"
            for t, f, labels in conflicts
        )
        raise ValueError(
            "Tokenizer rule 2 (uniqueness) failed — the following features "
            "match more than one SRO type / NFC / excluded pattern:\n"
            f"{bullets}\n"
            "Fix: tighten the offending feature_patterns so each feature "
            "belongs to exactly one bucket."
        )
```

Wire it into `validate_against_payload`:

```python
def validate_against_payload(
    cfg: EntityTokenizerConfig,
    sample: EntityPayloadSample,
    action_names_per_building: List[List[str]],
) -> None:
    sro_matchers = _sro_matchers(cfg)
    excluded_matchers = _excluded_matchers(cfg)
    _validate_rule_1_coverage(cfg, sample, sro_matchers, excluded_matchers)
    _validate_rule_2_uniqueness(cfg, sample, sro_matchers, excluded_matchers)
```

**Note on rule order:** rule 2 is checked AFTER rule 1 — but a single feature can independently trigger both. The tests above use scenarios where rule 1 passes, so we will catch rule 2.

Actually wait — in `test_excluded_feature_cannot_match_an_sro_type`, `district__topology_version` is excluded AND in district_meta. Rule 1 considers it matched (because labels=`["excluded","district_meta"]` is non-empty), so rule 1 passes — rule 2 then fires. Good.

- [ ] **Step 3: Run both rule-2 tests**

```bash
pytest tests/test_entity_tokenizer_config_schema.py -k "rule2 or excluded_feature_cannot" -v
```
Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add utils/entity_tokenizer_schema.py tests/test_entity_tokenizer_config_schema.py
git commit -m "feat(wp02): implement tokenizer validation rule 2 (pattern uniqueness)"
```

---

### Task 8: Rule 3 — NFC sources exist

**Files:**
- Modify: `utils/entity_tokenizer_schema.py`
- Modify: `tests/test_entity_tokenizer_config_schema.py`

- [ ] **Step 1: Failing test**

```python
def test_rule3_missing_nfc_source_fails():
    from utils.entity_tokenizer_schema import (
        EntityTokenizerConfig, _load_default_sample, validate_against_payload,
    )
    cfg_dict = json.loads(Path("configs/tokenizers/entity_default.json").read_text())
    cfg_dict["nfc"]["expression"]["left"]["feature"] = "does_not_exist"
    cfg = EntityTokenizerConfig.model_validate(cfg_dict)
    with pytest.raises(ValueError, match="does_not_exist"):
        validate_against_payload(
            cfg, _load_default_sample(),
            [["electrical_storage", "electric_vehicle_storage"]],
        )
```
Run: confirm FAIL.

- [ ] **Step 2: Implement rule 3 — must run BEFORE rule 1, because rule 1 expects NFC sources to be present in the payload**

```python
def _validate_rule_3_nfc_sources_exist(
    cfg: EntityTokenizerConfig, sample: EntityPayloadSample
) -> None:
    table = cfg.nfc.entity_table
    available = set(sample.feature_names_per_table.get(table, []))
    missing = [
        cfg.nfc.expression.left.feature,
        cfg.nfc.expression.right.feature,
    ]
    missing = [m for m in missing if m not in available]
    if missing:
        raise ValueError(
            f"Tokenizer rule 3 (nfc sources exist) failed — features "
            f"{missing!r} are not present in entity_table {table!r}."
        )
```

Reorder `validate_against_payload`:

```python
def validate_against_payload(cfg, sample, action_names_per_building):
    sro_matchers = _sro_matchers(cfg)
    excluded_matchers = _excluded_matchers(cfg)
    _validate_rule_3_nfc_sources_exist(cfg, sample)
    _validate_rule_1_coverage(cfg, sample, sro_matchers, excluded_matchers)
    _validate_rule_2_uniqueness(cfg, sample, sro_matchers, excluded_matchers)
```

- [ ] **Step 3: Run**

```bash
pytest tests/test_entity_tokenizer_config_schema.py -k rule3 -v
```
Expected: PASS. Also re-run all prior tests to confirm no regression: `pytest tests/test_entity_tokenizer_config_schema.py -v`.

- [ ] **Step 4: Commit**

```bash
git add utils/entity_tokenizer_schema.py tests/test_entity_tokenizer_config_schema.py
git commit -m "feat(wp02): implement tokenizer validation rule 3 (NFC sources exist)"
```

---

### Task 9: Rule 4 — Pattern compilation

**Files:**
- Modify: `utils/entity_tokenizer_schema.py`
- Modify: `tests/test_entity_tokenizer_config_schema.py`

- [ ] **Step 1: Failing test**

```python
def test_rule4_bad_regex_fails():
    from utils.entity_tokenizer_schema import (
        EntityTokenizerConfig, _load_default_sample, validate_against_payload,
    )
    cfg_dict = json.loads(Path("configs/tokenizers/entity_default.json").read_text())
    cfg_dict["sro_types"]["district_time"]["feature_patterns"] = ["^[unclosed"]
    cfg = EntityTokenizerConfig.model_validate(cfg_dict)
    with pytest.raises(ValueError, match=r"sro_types\.district_time\.feature_patterns"):
        validate_against_payload(
            cfg, _load_default_sample(),
            [["electrical_storage", "electric_vehicle_storage"]],
        )
```
Run: confirm FAIL.

- [ ] **Step 2: Implementation**

Rule 4 already happens implicitly inside `_compile_pattern`, called by `_sro_matchers` and `_excluded_matchers`. The current `validate_against_payload` calls those at the top, so the failure is raised before any other rule. Run the test:

```bash
pytest tests/test_entity_tokenizer_config_schema.py::test_rule4_bad_regex_fails -v
```
Expected: PASS — `_compile_pattern` already produces a `ValueError` whose message matches the regex.

If the regex match string is wrong, adjust the test (e.g. include `re.error` text). Do NOT loosen the matcher unnecessarily.

- [ ] **Step 3: Commit**

```bash
git add tests/test_entity_tokenizer_config_schema.py
git commit -m "test(wp02): cover tokenizer validation rule 4 (regex compilation)"
```

---

### Task 10: Rule 5 — Action-field coverage

**Files:**
- Modify: `utils/entity_tokenizer_schema.py`
- Modify: `tests/test_entity_tokenizer_config_schema.py`

- [ ] **Step 1: Failing test**

```python
def test_rule5_missing_action_field_fails():
    from utils.entity_tokenizer_schema import (
        load_entity_tokenizer_config, _load_default_sample, validate_against_payload,
    )
    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    sample = _load_default_sample()
    # action_names without "electric_vehicle_storage" (the charger action_field).
    bad_action_names = [["electrical_storage"]]  # Building 0 missing charger action
    with pytest.raises(ValueError, match="electric_vehicle_storage"):
        validate_against_payload(cfg, sample, bad_action_names)


def test_rule5_default_passes():
    from utils.entity_tokenizer_schema import (
        load_entity_tokenizer_config, _load_default_sample, validate_against_payload,
    )
    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    sample = _load_default_sample()
    action_names = [["electrical_storage", "electric_vehicle_storage"]]
    validate_against_payload(cfg, sample, action_names)  # must not raise
```
Run: confirm `test_rule5_missing_action_field_fails` FAILs (rule not implemented).

- [ ] **Step 2: Implement**

Add to `utils/entity_tokenizer_schema.py`:

```python
def _validate_rule_5_action_field_coverage(
    cfg: EntityTokenizerConfig,
    action_names_per_building: List[List[str]],
) -> None:
    required = {ca.action_field for ca in cfg.ca_types.values()}
    missing_per_building: List[tuple[int, List[str]]] = []
    for b, names in enumerate(action_names_per_building):
        missing = [r for r in required if r not in names]
        if missing:
            missing_per_building.append((b, missing))
    if missing_per_building:
        bullets = "\n".join(
            f"  - building_idx={b}, missing_action_fields={m}"
            for b, m in missing_per_building
        )
        raise ValueError(
            "Tokenizer rule 5 (action-field coverage) failed — the following "
            "buildings are missing one or more action_field declared in "
            f"ca_types:\n{bullets}\n"
            "Fix: ensure every CA type's action_field appears in the building's "
            "action_names (this is normally produced by the simulator schema)."
        )
```

Wire into `validate_against_payload`:

```python
def validate_against_payload(cfg, sample, action_names_per_building):
    sro_matchers = _sro_matchers(cfg)
    excluded_matchers = _excluded_matchers(cfg)
    _validate_rule_3_nfc_sources_exist(cfg, sample)
    _validate_rule_1_coverage(cfg, sample, sro_matchers, excluded_matchers)
    _validate_rule_2_uniqueness(cfg, sample, sro_matchers, excluded_matchers)
    _validate_rule_5_action_field_coverage(cfg, action_names_per_building)
```

- [ ] **Step 3: Run rule-5 tests**

```bash
pytest tests/test_entity_tokenizer_config_schema.py -k rule5 -v
```
Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add utils/entity_tokenizer_schema.py tests/test_entity_tokenizer_config_schema.py
git commit -m "feat(wp02): implement tokenizer validation rule 5 (action-field coverage)"
```

---

### Task 11: Auxiliary `excluded_features` test

The §16.6 row `test_excluded_feature_pattern_removes_topology_version` lives in WP03 (the layout builder produces `excluded_feature_names`). For now, write a smaller WP02-scoped version: confirm the excluded pattern matches `district__topology_version` in classification.

- [ ] **Step 1: Failing test**

```python
def test_excluded_pattern_matches_topology_version():
    from utils.entity_tokenizer_schema import (
        load_entity_tokenizer_config, _excluded_matchers,
    )
    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    matchers = _excluded_matchers(cfg)
    assert any(p.fullmatch("district__topology_version") for p in matchers)
    assert any(p.fullmatch("electric_vehicle_charger_state") for p in matchers)
    assert not any(p.fullmatch("district__hour") for p in matchers)
```
Run: should already PASS since matchers exist. If not, fix.

- [ ] **Step 2: Commit if new**

```bash
git add tests/test_entity_tokenizer_config_schema.py
git commit -m "test(wp02): cover excluded-pattern matching for topology_version + legacy aliases"
```

---

### Task 12: Add Pydantic models for `TransformerPPOAlgorithmConfig` to `config_schema.py`

**Files:**
- Modify: `utils/config_schema.py`
- Modify: `tests/test_entity_tokenizer_config_schema.py`

- [ ] **Step 1: Read the existing algorithm-config union to identify where to insert**

```bash
grep -n "discriminator\|RuleBasedPolicyAlgorithmConfig\|MADDPGAlgorithmConfig\|class.*AlgorithmConfig" utils/config_schema.py
```

- [ ] **Step 2: Failing test — `validate_config` must accept the new template**

Even though we don't have the template yet (WP06), we can construct an in-memory config dict that mimics §13.2 and validate it:

```python
def test_validate_config_accepts_transformer_ppo_algorithm():
    """A minimal config naming AgentTransformerPPO should pass schema validation."""
    from utils.config_schema import validate_config

    cfg = {
        "metadata": {"experiment_name": "x", "run_name": "x", "community_name": "x"},
        "runtime": {},
        "tracking": {"mlflow_enabled": False, "log_level": "INFO", "log_frequency": 1,
                     "mlflow_step_sample_interval": 10,
                     "mlflow_artifacts_profile": "minimal",
                     "progress_updates_enabled": True, "progress_update_interval": 5,
                     "system_metrics_enabled": False, "system_metrics_interval": 10},
        "checkpointing": {"resume_training": False,
                          "checkpoint_artifact": "x.pt",
                          "use_best_checkpoint_artifact": False,
                          "reset_replay_buffer": False,
                          "freeze_pretrained_layers": False, "fine_tune": False},
        "bundle": {"require_observations_envelope": False,
                   "artifact_config": {}, "per_agent_artifact_config": {}},
        "simulator": {
            "dataset_name": "citylearn_three_phase_dynamic_assets_only_demo",
            "dataset_path": "./datasets/citylearn_three_phase_dynamic_assets_only_demo/schema.json",
            "central_agent": False,
            "interface": "entity",
            "topology_mode": "dynamic",
            "entity_encoding": {"enabled": True, "normalization": "minmax_space", "clip": True},
            "reward_function": "RewardFunction",
            "reward_function_kwargs": {},
            "episodes": 1,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": 100,
            "episode_time_steps": 101,
            "export": {"mode": "end", "export_kpis_on_episode_end": True},
            "wrapper_reward": {"enabled": False, "profile": "cost_limits_v1",
                               "clip_enabled": True, "clip_min": -10.0, "clip_max": 10.0,
                               "squash": "none"},
        },
        "training": {"seed": 0, "steps_between_training_updates": 1, "target_update_interval": 0},
        "topology": {},
        "algorithm": {
            "name": "AgentTransformerPPO",
            "tokenizer_config_path": "configs/tokenizers/entity_default.json",
            "transformer": {"d_model": 64, "nhead": 4, "num_layers": 2,
                            "dim_feedforward": 128, "dropout": 0.1},
            "hyperparameters": {"learning_rate": 3.0e-4, "gamma": 0.99,
                                "gae_lambda": 0.95, "clip_eps": 0.2, "ppo_epochs": 4,
                                "minibatch_size": 64, "entropy_coeff": 0.01,
                                "value_coeff": 0.5, "max_grad_norm": 0.5},
        },
        "execution": None,
    }
    validate_config(cfg)  # must not raise
```
Run: FAIL — `algorithm.name` not in the discriminated union.

- [ ] **Step 3: Implement — add models + extend union**

In `utils/config_schema.py`, near the existing `*AlgorithmConfig` classes, add:

```python
from typing import Literal as _Literal


class TransformerConfig(_Strict):  # use the project's existing strict base, or define
    d_model: int = Field(ge=1)
    nhead: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    dim_feedforward: int = Field(ge=1)
    dropout: float = Field(ge=0.0, le=1.0)


class TransformerPPOHyperparameters(_Strict):
    learning_rate: float = Field(gt=0.0)
    gamma: float = Field(ge=0.0, le=1.0)
    gae_lambda: float = Field(ge=0.0, le=1.0)
    clip_eps: float = Field(gt=0.0)
    ppo_epochs: int = Field(ge=1)
    minibatch_size: int = Field(ge=1)
    entropy_coeff: float = Field(ge=0.0)
    value_coeff: float = Field(ge=0.0)
    max_grad_norm: float = Field(gt=0.0)


class TransformerPPOAlgorithmConfig(_Strict):
    name: _Literal["AgentTransformerPPO"]
    tokenizer_config_path: str
    transformer: TransformerConfig
    hyperparameters: TransformerPPOHyperparameters
    networks: Optional[Any] = None
    replay_buffer: Optional[Any] = None
    exploration: Optional[Any] = None
```

Replace `_Strict` with whatever strict base the existing classes use (likely `BaseModel` with `model_config = ConfigDict(extra="forbid")`). Inspect the file and reuse the same idiom.

Then extend the discriminated union:

```python
AlgorithmConfig = Annotated[
    Union[
        RuleBasedPolicyAlgorithmConfig,
        MADDPGAlgorithmConfig,
        TransformerPPOAlgorithmConfig,   # NEW
    ],
    Field(discriminator="name"),
]
```

- [ ] **Step 4: Run**

```bash
pytest tests/test_entity_tokenizer_config_schema.py::test_validate_config_accepts_transformer_ppo_algorithm -v
```
Expected: PASS. Also run the existing test suite: `pytest -x -q`. No prior test should regress.

- [ ] **Step 5: Commit**

```bash
git add utils/config_schema.py tests/test_entity_tokenizer_config_schema.py
git commit -m "feat(wp02): add TransformerPPOAlgorithmConfig to config schema union"
```

---

### Task 13: Wire tokenizer JSON validation into `validate_config`

**Files:**
- Modify: `utils/config_schema.py`
- Modify: `tests/test_entity_tokenizer_config_schema.py`

- [ ] **Step 1: Failing test**

```python
def test_validate_config_loads_tokenizer_json(tmp_path):
    """validate_config opens the JSON path and runs the 5 rules."""
    import yaml
    from utils.config_schema import validate_config

    bad_tokenizer = tmp_path / "bad.json"
    bad_tokenizer.write_text(json.dumps({
        "type_embeddings": {"SRO": 0, "NFC": 1, "CA": 2},
        "excluded_features": {"patterns": []},
        "nfc": {"type_name": "x", "entity_table": "building",
                "expression": {"op": "subtract",
                               "left": {"feature": "ghost"},
                               "right": {"feature": "solar_generation"}}},
        "ca_types": {
            "storage": {"entity_table": "storage", "action_field": "electrical_storage", "input_dim_fallback": 1},
            "charger": {"entity_table": "charger", "action_field": "electric_vehicle_storage", "input_dim_fallback": 1},
        },
        "sro_types": {},
        "validation": {"unmatched_features": "fail",
                       "ambiguous_pattern_match": "fail",
                       "input_dim_mismatch": "fail"},
    }))
    cfg = _make_minimal_transformer_ppo_cfg(tokenizer_path=str(bad_tokenizer))
    with pytest.raises(ValueError, match="ghost"):
        validate_config(cfg)
```

`_make_minimal_transformer_ppo_cfg` is a tiny helper inside the test module — extract from the previous test to avoid duplication.

Run: FAIL (no JSON loading yet).

- [ ] **Step 2: Implement dispatch**

In `utils/config_schema.py`, find the `validate_config(...)` function. After Pydantic validation succeeds, add:

```python
# Tokenizer JSON validation for AgentTransformerPPO (spec §13.4)
if validated.algorithm.name == "AgentTransformerPPO":
    from utils.entity_tokenizer_schema import (
        load_entity_tokenizer_config,
        _load_default_sample,
        validate_against_payload,
    )
    tokenizer_cfg = load_entity_tokenizer_config(
        validated.algorithm.tokenizer_config_path
    )
    sample = _load_default_sample()
    # At config load time we don't have action_names from a live wrapper; supply
    # the canonical ca_types.action_field set so rule 5 is a no-op here. The
    # agent will re-run all 5 rules at attach_environment time against the live
    # entity_specs.
    canonical_action_names = [
        [ca.action_field for ca in tokenizer_cfg.ca_types.values()]
    ]
    validate_against_payload(tokenizer_cfg, sample, canonical_action_names)
```

- [ ] **Step 3: Run all tests**

```bash
pytest tests/test_entity_tokenizer_config_schema.py -v
```
Expected: every test PASS, including the new `test_validate_config_loads_tokenizer_json`.

Also run full sweep: `pytest -x -q`. No prior test should regress.

- [ ] **Step 4: Commit**

```bash
git add utils/config_schema.py tests/test_entity_tokenizer_config_schema.py
git commit -m "feat(wp02): wire entity tokenizer JSON validation into validate_config"
```

---

## Self-Review Checklist (run before requesting code review)

Mark each box only after observing the evidence.

- [ ] **Spec coverage:** Re-read §13.1, §13.4, §13.5, §16.6 of `docs/specv2.md`. Confirm:
  - JSON file content equals §13.1 verbatim: `diff <(cat configs/tokenizers/entity_default.json) <(sed -n '970,1175p' docs/specv2.md | sed -n '/```json/,/```/p' | sed '1d;$d')` (allow whitespace differences).
  - All 5 rules implemented and tested with both happy path and failure path.
  - `EntityTokenizerConfig` exposes every field listed in §13.5.
  - `TransformerPPOAlgorithmConfig` added to the union.
  - `_Strict` (`extra="forbid"`) on every model.
- [ ] **All §16.6 test rows are covered or deferred to WP03 with a comment in code:**
  - `test_valid_json_loads` ✅
  - `test_missing_ca_type_raises` ✅
  - `test_unknown_field_raises` ✅
  - `test_validate_config_loads_tokenizer_json` ✅
  - `test_rule1_unmatched_feature_fails` ✅
  - `test_rule2_ambiguous_pattern_fails` ✅
  - `test_rule3_missing_nfc_source_fails` ✅
  - `test_rule4_bad_regex_fails` ✅
  - `test_rule5_missing_action_field_fails` ✅
  - `test_excluded_feature_pattern_removes_topology_version` — DEFERRED to WP03 (needs layout builder); WP02 has the smaller `test_excluded_pattern_matches_topology_version`.
  - `test_excluded_feature_cannot_match_an_sro_type` ✅
- [ ] **No torch import:** `grep -RIn "import torch\|from torch" utils/entity_tokenizer_schema.py utils/config_schema.py tests/test_entity_tokenizer_config_schema.py`. Expected: no matches.
- [ ] **Default config validates against itself:**
  ```bash
  python -c "from utils.entity_tokenizer_schema import load_entity_tokenizer_config, _load_default_sample, validate_against_payload; cfg = load_entity_tokenizer_config('configs/tokenizers/entity_default.json'); validate_against_payload(cfg, _load_default_sample(), [['electrical_storage', 'electric_vehicle_storage']]); print('OK')"
  ```
  Expected: `OK`.
- [ ] **Coverage accounting matches spec §13.1 table:**
  ```bash
  python -c "
  from utils.entity_tokenizer_schema import load_entity_tokenizer_config, _load_default_sample, _sro_matchers, _excluded_matchers, _classify_feature
  cfg = load_entity_tokenizer_config('configs/tokenizers/entity_default.json')
  s = _load_default_sample()
  sm = _sro_matchers(cfg); em = _excluded_matchers(cfg)
  from collections import Counter
  for table in ('district','building'):
      counts = Counter()
      for f in s.feature_names_per_table[table]:
          for label in _classify_feature(f, table, cfg, sm, em):
              counts[label] += 1
      print(table, dict(counts))
  "
  ```
  Compare output against §13.1 coverage table. Expected: district counts add up to 46 (with `excluded` = 1), building counts add up to 38 (with `nfc` = 2 because both NFC sources are matched).
- [ ] **Full repo test suite passes:** `pytest -x -q` → exit 0.
- [ ] **No new lint warnings introduced:** if ruff is configured, `ruff check utils/ tests/test_entity_tokenizer_config_schema.py` is clean.

---

## Code Review

After the self-review checklist passes, invoke `superpowers:requesting-code-review` to dispatch a fresh subagent that reviews the diff against this plan and §13.1/§13.4/§13.5/§16.6 of `docs/specv2.md`. Resolve any blocking findings before opening the PR.

---

## PR Description

```markdown
## Summary
Introduces the canonical entity tokenizer config (`configs/tokenizers/entity_default.json`) as the single source of truth for v2 token taxonomy, and adds the Pydantic schema + 5 hard-fail validation rules (coverage, uniqueness, NFC sources, regex compilation, action-field coverage) per spec §13.4. `validate_config(...)` now refuses to load any `AgentTransformerPPO` config whose tokenizer JSON is malformed, before the agent or wrapper are instantiated.

## Key Changes
- Add `configs/tokenizers/entity_default.json` (verbatim from spec §13.1, 19 SRO types + NFC + 2 CA types + 6 excluded patterns).
- Add `utils/entity_tokenizer_schema.py`: Pydantic models (`EntityTokenizerConfig`, `NfcConfig`, `CaTypeConfig`, `SroSingletonTypeConfig`, `SroPerAssetTypeConfig`, `ExcludedFeaturesConfig`), `EntityPayloadSample` dataclass, `load_entity_tokenizer_config(...)`, `validate_against_payload(...)` implementing rules 1-5.
- Extend `utils/config_schema.py`: add `TransformerConfig`, `TransformerPPOHyperparameters`, `TransformerPPOAlgorithmConfig` to the algorithm discriminated union; in `validate_config(...)` dispatch on `algorithm.name == "AgentTransformerPPO"` to load and validate the tokenizer JSON against the bundled sample payload.
- Add `tests/test_entity_tokenizer_config_schema.py` covering all of spec §16.6 (except the layout-builder-dependent row, deferred to WP03).
- No torch dependency.
```
