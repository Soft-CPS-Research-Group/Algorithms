"""Pydantic models, loader, and validation rules for the entity tokenizer config.

See ``docs/specv2.md`` §13.1 (catalog), §13.4 (5 hard-fail rules), §13.5 (model
surface area).

This module is pure-Python (no torch). It is consumed at config-validation time
by :func:`utils.config_schema.validate_config` and re-used at runtime by the
v2 agent (WP05) when the wrapper notifies a topology change.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Union

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
    """Top-level entity tokenizer configuration (see spec §13.5)."""

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
            raise ValueError(
                f"ca_types missing required entries: {sorted(missing)}"
            )
        return self


def load_entity_tokenizer_config(path: str | Path) -> EntityTokenizerConfig:
    """Read the JSON file at ``path`` and validate it as an EntityTokenizerConfig."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EntityTokenizerConfig.model_validate(raw)


@dataclass(frozen=True)
class EntityPayloadSample:
    """Lightweight view of an entity payload used for tokenizer validation.

    Only feature names (not values) are needed for rules 1–5; this lets the
    validator run before any wrapper/simulator instantiation.
    """

    feature_names_per_table: Dict[str, List[str]]
    # Per-asset adapter labels we have observed in the sample, e.g.
    # {"pv": ["Building_1/pv", ...], "charger": ["Building_1/charger_1", ...]}.
    # Currently only used as carry-through metadata.
    adapter_observed_prefixes: Dict[str, List[str]] = field(default_factory=dict)


_DEFAULT_SAMPLE_PATH = Path("datasets/tmp_entity_obs_full_step2200_named.json")


def _load_default_sample() -> EntityPayloadSample:
    """Load the bundled entity payload sample as an EntityPayloadSample.

    The sample lives at ``datasets/tmp_entity_obs_full_step2200_named.json``
    and is the single canonical fixture for tokenizer validation.

    The entity payload stores **bare** feature column names per table (e.g.
    ``hour`` in the district table). The tokenizer config matches against
    **adapter-emitted** names (see ``utils/entity_adapter.py:215``), where
    district columns are prefixed ``district__<feature>`` and building columns
    are emitted unprefixed. This loader applies that prefixing convention so
    the validation rules see the same names the agent sees at runtime.
    """
    payload = json.loads(_DEFAULT_SAMPLE_PATH.read_text(encoding="utf-8"))
    tables = payload["tables"]
    feature_names_per_table: Dict[str, List[str]] = {}
    # Match utils/entity_adapter.py: only the district block is prefixed when
    # adapted into per-building observation names. All other tables keep bare
    # column names; per-asset prefixing (storage::, pv::, charger::) is *not*
    # applied to the per-table feature lists used by SRO singleton matchers
    # (those operate on the asset's column names directly).
    table_prefix = {
        "district": "district__",
    }
    for table_name, table in tables.items():
        if not isinstance(table, dict):
            continue
        features = table.get("features")
        if not isinstance(features, list):
            continue
        prefix = table_prefix.get(table_name, "")
        feature_names_per_table[table_name] = [
            f"{prefix}{name}" for name in features
        ]
    return EntityPayloadSample(
        feature_names_per_table=feature_names_per_table,
    )


# ---------------------------------------------------------------------------
# Five hard-fail validation rules (spec §13.4)
# ---------------------------------------------------------------------------


def _compile_pattern(pattern: str, json_path: str) -> re.Pattern[str]:
    """Compile a regex; on failure raise ValueError with the originating path.

    This is rule 4 (regex compilation): malformed patterns surface here before
    any other rule is evaluated.
    """
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


def _sro_matchers(
    cfg: EntityTokenizerConfig,
) -> Dict[str, List[re.Pattern[str]]]:
    out: Dict[str, List[re.Pattern[str]]] = {}
    for type_name, sro in cfg.sro_types.items():
        if isinstance(sro, SroSingletonTypeConfig):
            out[type_name] = [
                _compile_pattern(
                    p, f"sro_types.{type_name}.feature_patterns[{i}]"
                )
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
    """Return the labels a feature matches: ``"excluded"``, ``"nfc"``, or SRO type names.

    Empty list = unmatched. More than one label = ambiguous.
    """
    labels: List[str] = []
    if any(p.fullmatch(feature) for p in excluded_matchers):
        labels.append("excluded")
    if (
        table == cfg.nfc.entity_table
        and feature in (
            cfg.nfc.expression.left.feature,
            cfg.nfc.expression.right.feature,
        )
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


def _validate_rule_3_nfc_sources_exist(
    cfg: EntityTokenizerConfig, sample: EntityPayloadSample
) -> None:
    table = cfg.nfc.entity_table
    available = set(sample.feature_names_per_table.get(table, []))
    declared = [
        cfg.nfc.expression.left.feature,
        cfg.nfc.expression.right.feature,
    ]
    missing = [m for m in declared if m not in available]
    if missing:
        raise ValueError(
            f"Tokenizer rule 3 (nfc sources exist) failed — features "
            f"{missing!r} are not present in entity_table {table!r}."
        )


def _validate_rule_1_coverage(
    cfg: EntityTokenizerConfig,
    sample: EntityPayloadSample,
    sro_matchers: Mapping[str, List[re.Pattern[str]]],
    excluded_matchers: List[re.Pattern[str]],
) -> None:
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
            "Fix: add to an existing SRO type's feature_patterns, define a "
            "new SRO type, or add to excluded_features.patterns."
        )


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
    conflicts: List[tuple[str, str, List[str]]] = []
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


def _validate_rule_5_action_field_coverage(
    cfg: EntityTokenizerConfig,
    action_names_per_building: List[List[str]],
) -> None:
    """Rule 5 (action-field coverage).

    Every ``ca_types[*].action_field`` declared in the tokenizer config must
    appear in at least one building's ``action_names``. Per-building gaps are
    expected (e.g. assets-only datasets where some buildings have no
    charger), so we enforce coverage at the *dataset* level rather than the
    per-building level. The per-building post-condition is enforced by
    `EntityTokenLayoutBuilder` via the count match
    (``len(ca_segments) == len(action_names)``); a building that lacks a
    given CA simply has no segment for that type.

    Diverges from the literal ``docs/specv2.md`` §13.4 r5 wording ("for
    every building") because real datasets (e.g. the bundled
    ``citylearn_three_phase_dynamic_assets_only_demo``) intentionally have
    asset-free buildings, and the simulator (softcpsrecsimulator >= 0.5.0)
    correctly omits the corresponding actions.
    """
    required = {ca.action_field for ca in cfg.ca_types.values()}
    declared = {n for names in action_names_per_building for n in names}
    missing = sorted(r for r in required if r not in declared)
    if missing:
        raise ValueError(
            "Tokenizer rule 5 (action-field coverage) failed — the following "
            "CA action_fields declared in ca_types do not appear in any "
            f"building's action_names:\n"
            + "\n".join(f"  - {m!r}" for m in missing)
            + "\nFix: either remove the unused ca_type from the tokenizer "
            "config, or ensure the simulator schema declares the "
            "corresponding asset on at least one building."
        )


def validate_against_payload(
    cfg: EntityTokenizerConfig,
    sample: EntityPayloadSample,
    action_names_per_building: List[List[str]],
    *,
    include_rule_5: bool = True,
) -> None:
    """Run hard-fail validation rules. Raises ``ValueError`` on first failure.

    Order: rule 4 (regex compile) is implicit during matcher construction;
    then rule 3 (NFC sources exist); then rule 1 (coverage); then rule 2
    (uniqueness); then optionally rule 5 (action-field coverage).

    ``include_rule_5`` controls whether the dataset-level action-field
    coverage check runs. Callers SHOULD enable it at startup (to catch
    misconfigured tokenizers vs the simulator schema) but MAY disable it
    when re-validating after a runtime topology mutation, where the set of
    active assets can legitimately become a strict subset of the configured
    CA types (e.g. a topology event removes the last EV charger).
    """
    sro_matchers = _sro_matchers(cfg)
    excluded_matchers = _excluded_matchers(cfg)
    _validate_rule_3_nfc_sources_exist(cfg, sample)
    _validate_rule_1_coverage(cfg, sample, sro_matchers, excluded_matchers)
    _validate_rule_2_uniqueness(cfg, sample, sro_matchers, excluded_matchers)
    if include_rule_5:
        _validate_rule_5_action_field_coverage(cfg, action_names_per_building)
