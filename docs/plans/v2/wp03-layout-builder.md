# WP03 — `EntityTokenLayoutBuilder` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For all v2 WPs:** every production-code task MUST follow `superpowers:test-driven-development`. The WP MUST end with `superpowers:requesting-code-review`.

**Goal:** Implement `EntityTokenLayoutBuilder` (`algorithms/utils/entity_token_layout.py`) per `docs/specv2.md` §7. The builder consumes per-building `observation_names` + `action_names` and produces a deterministic `BuildingTokenLayout` whose CA segment order is defined exclusively by `action_names[building]`. Pure Python (stdlib only).

**Architecture:** A pure-function library that re-uses the regex catalog and validated config from WP02. The builder owns CA ordering as a single source of truth (spec §7.3 final paragraph). Layouts are cached per `(observation_names, action_names)` tuple so repeated `build()` calls during a stable topology cost nothing. `topology_changed(...)` is the cheap predicate the wrapper polls (alongside `meta.topology_version`) to decide whether to rebuild.

**Tech Stack:** Python 3.11 stdlib only — `typing`, `re`, `dataclasses`. **No torch, no numpy, no algorithms.* / utils.* imports** (per spec §7.4).

**Branch:** `gj/wp03-layout-builder`
**Base branch:** `gj/wp02-tokenizer-config`

---

## Scope

**Files created:**

- `algorithms/utils/entity_token_layout.py` — public dataclasses (`TokenSegment`, `NfcExpression`, `BuildingTokenLayout`) + `EntityTokenLayoutBuilder` class.
- `tests/test_entity_token_layout.py` — covers all of spec §16.1.

**Files modified:**

- `tests/test_entity_tokenizer_config_schema.py` — add the deferred row from WP02 (`test_excluded_feature_pattern_removes_topology_version`) using the now-available builder.

**Out of scope:**

- Anything torch (the tokenizer itself is WP04).
- Adapter changes (`utils/entity_adapter.py`) — the builder consumes adapter-output names as-is.
- Wrapper changes.
- Agent.

---

## File Structure

```
algorithms/
  utils/
    entity_token_layout.py    # NEW (~250 lines)
tests/
  test_entity_token_layout.py             # NEW (~22 tests, covers §16.1)
  test_entity_tokenizer_config_schema.py  # MODIFIED (1 test added)
```

**Module layout (`entity_token_layout.py`):**

1. Public dataclasses (top): `TokenSegment`, `NfcExpression`, `BuildingTokenLayout`.
2. Private helpers: `_compile_patterns`, `_strip_district_prefix`, `_extract_per_asset_id`, `_classify_observation_name`.
3. Public class `EntityTokenLayoutBuilder` with: `__init__`, `build`, `topology_changed`, `_compute_layout`, `_classify`, `_order_segments`, `_assemble_ca_in_action_order`.

The class fits in ~250 lines because the regex catalog + 5 rules already live in `utils/entity_tokenizer_schema.py` (WP02). The builder *consumes* an `EntityTokenizerConfig` instance — it never re-parses JSON.

---

## Key Design Decisions (encoded in the dataclasses)

- `TokenSegment.feature_indices` is a `Tuple[int, ...]` so the tokenizer can do `obs[..., list(seg.feature_indices)]` (works even for non-contiguous indices, as in the interleaved district forecast feature group).
- `NfcExpression.left_index_in_segment` and `right_index_in_segment` are **offsets into `feature_indices`**, not absolute observation indices. This decouples the NFC compute from the layout's positional details.
- `BuildingTokenLayout.ca_action_names` is the single contract that the agent's startup assertion (§10.1) compares against `action_names[building]`. The builder enforces position-equality at construction time — if the assertion ever fires, the builder is broken, not the agent.
- Caching key = `(building_id, tuple(observation_names), tuple(action_names))`. Tuples are hashable; this avoids cache invalidation bugs from list mutation.
- The builder receives a parsed `EntityTokenizerConfig` (from WP02), not raw JSON or a `Mapping[str, Any]`. The spec lists `Mapping[str, Any]` in the interface but we tighten the type since WP02 already produced the Pydantic model. This keeps the builder honest about what it expects.

---

## Tasks

### Task 1: Branch + skeleton

- [ ] **Step 1: Create branch from WP02**

```bash
git checkout gj/wp02-tokenizer-config
git checkout -b gj/wp03-layout-builder
```

- [ ] **Step 2: Verify WP02 deliverables present**

```bash
test -f utils/entity_tokenizer_schema.py && test -f configs/tokenizers/entity_default.json && echo "OK"
```

- [ ] **Step 3: Confirm target paths absent**

```bash
test ! -e algorithms/utils/entity_token_layout.py && test ! -e tests/test_entity_token_layout.py && echo "OK"
```

---

### Task 2: Public dataclasses (TDD — write a constructor smoke test first)

**Files:**
- Create: `algorithms/utils/entity_token_layout.py`
- Create: `tests/test_entity_token_layout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_entity_token_layout.py
"""Tests for EntityTokenLayoutBuilder. Covers spec §16.1."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_dataclasses_constructible():
    from algorithms.utils.entity_token_layout import (
        TokenSegment, NfcExpression, BuildingTokenLayout,
    )
    seg = TokenSegment(
        family="sro", type_name="district_time", instance_id="Building_1",
        feature_indices=(0, 1, 2),
        feature_names=("district__month", "district__day_type", "district__hour"),
    )
    nfc = NfcExpression(op="subtract", left_index_in_segment=0, right_index_in_segment=1)
    layout = BuildingTokenLayout(
        building_id="Building_1",
        segments=(seg,),
        n_sro=1, n_ca=0,
        ca_action_names=(),
        excluded_feature_names=(),
    )
    assert layout.n_sro == 1
    assert seg.derived is None
    assert nfc.op == "subtract"
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_entity_token_layout.py::test_dataclasses_constructible -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement minimal dataclasses**

```python
# algorithms/utils/entity_token_layout.py
"""EntityTokenLayoutBuilder — pure-Python token segmentation.

See docs/specv2.md §7. No torch / numpy / algorithms.* / utils.* imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NfcExpression:
    op: str
    left_index_in_segment: int
    right_index_in_segment: int


@dataclass(frozen=True)
class TokenSegment:
    family: str
    type_name: str
    instance_id: Optional[str]
    feature_indices: Tuple[int, ...]
    feature_names: Tuple[str, ...]
    derived: Optional[NfcExpression] = None


@dataclass(frozen=True)
class BuildingTokenLayout:
    building_id: str
    segments: Tuple[TokenSegment, ...]
    n_sro: int
    n_ca: int
    ca_action_names: Tuple[str, ...]
    excluded_feature_names: Tuple[str, ...]
```

- [ ] **Step 4: Run test to confirm PASS**

```bash
pytest tests/test_entity_token_layout.py::test_dataclasses_constructible -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/entity_token_layout.py tests/test_entity_token_layout.py
git commit -m "feat(wp03): add token-layout dataclasses (TokenSegment, NfcExpression, BuildingTokenLayout)"
```

---

### Task 3: Builder skeleton + happy-path single-building build

**Files:**
- Modify: `algorithms/utils/entity_token_layout.py`
- Modify: `tests/test_entity_token_layout.py`

This task implements end-to-end `build()` for ONE realistic building (the first building in the bundled sample payload). Every subsequent §16.1 test then refines specific behaviors.

- [ ] **Step 1: Write the failing test (use the real sample payload)**

```python
def _load_sample_observation_names_for_first_building():
    """Compute observation_names for Building_1 by passing the sample payload
    through the adapter — mirrors what the wrapper would do."""
    from utils.entity_adapter import EntityContractAdapter

    payload = json.loads(
        Path("datasets/tmp_entity_obs_full_step2200_named.json").read_text()
    )
    # Adapter expects the full envelope shape produced by the simulator;
    # the bundled sample is already in that shape.
    adapter = EntityContractAdapter()
    obs_per_building, observation_names_per_building = (
        adapter.to_agent_observations(payload)
    )
    return observation_names_per_building[0]


def test_uses_real_sample_payload():
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
    from utils.entity_tokenizer_schema import load_entity_tokenizer_config

    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    builder = EntityTokenLayoutBuilder(cfg)
    obs_names = _load_sample_observation_names_for_first_building()
    action_names = ["electrical_storage", "electric_vehicle_storage"]
    layout = builder.build("Building_1", obs_names, action_names)

    # Per spec §13.1 coverage table: every district + building feature is
    # accounted for. The layout must classify all of them.
    classified_indices = set()
    for seg in layout.segments:
        classified_indices.update(seg.feature_indices)
    excluded_count = len(layout.excluded_feature_names)
    assert len(classified_indices) + excluded_count == len(obs_names), (
        f"unclassified features: total={len(obs_names)}, "
        f"classified={len(classified_indices)}, excluded={excluded_count}"
    )
    assert layout.ca_action_names == ("electrical_storage", "electric_vehicle_storage")
```

- [ ] **Step 2: Run to confirm FAIL** — `EntityTokenLayoutBuilder` not defined.

- [ ] **Step 3: Implement the builder end-to-end**

Below is the complete builder implementation. Append to `algorithms/utils/entity_token_layout.py`:

```python
# Per-asset adapter prefixes use "::" as separator (see utils/entity_adapter.py).
_PER_ASSET_ID_RE = re.compile(r"^(?P<prefix>[a-z_]+)::(?P<id>[^:]+(?:::[^:]+)?)::(?P<feature>.+)$")
# Two id forms appear in adapter output:
#   pv::Building_1/pv::generation_power_kw                   → id = "Building_1/pv"
#   charger::Building_1/charger_1::connected_ev::soc          → id = "Building_1/charger_1", label = "connected_ev"
#   charger::Building_1/charger_1::power_kw                   → id = "Building_1/charger_1", label = None
# The single regex above is too loose for the labelled charger case; split into two:
_PER_ASSET_LABELLED_RE = re.compile(
    r"^(?P<prefix>charger)::(?P<id>[^:]+(?:/[^:]+)?)::(?P<label>connected_ev|incoming_ev)::(?P<feature>.+)$"
)
_PER_ASSET_UNLABELLED_RE = re.compile(
    r"^(?P<prefix>pv|storage|charger)::(?P<id>[^:]+(?:/[^:]+)?)::(?P<feature>.+)$"
)
_DISTRICT_PREFIX = "district__"


def _detect_table(name: str, cfg) -> str:
    """Return the entity table this observation name originates from.

    The adapter prefix tells us; the tokenizer config's per-asset SROs map
    each prefix → entity_table.
    """
    if name.startswith(_DISTRICT_PREFIX):
        return "district"
    # Try per-asset patterns
    if _PER_ASSET_LABELLED_RE.fullmatch(name):
        return "ev"
    m = _PER_ASSET_UNLABELLED_RE.fullmatch(name)
    if m:
        prefix = m.group("prefix")
        # Map adapter prefix → entity_table via SRO/CA configs.
        # storage and charger are CA tables.
        return {"pv": "pv", "storage": "storage", "charger": "charger"}[prefix]
    # Otherwise: building-scoped feature (no prefix).
    return "building"


class EntityTokenLayoutBuilder:
    """Builds per-building TokenLayout. Caches by (building_id, obs, actions)."""

    def __init__(self, tokenizer_config) -> None:
        self._cfg = tokenizer_config
        self._cache: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], BuildingTokenLayout] = {}
        # Pre-compile pattern matchers for speed.
        from utils.entity_tokenizer_schema import (
            _sro_matchers, _excluded_matchers,
        )
        self._sro_matchers: Mapping[str, List[re.Pattern[str]]] = _sro_matchers(
            tokenizer_config
        )
        self._excluded_matchers: List[re.Pattern[str]] = _excluded_matchers(
            tokenizer_config
        )
        # SRO declaration order = order of insertion into the JSON dict.
        self._sro_declaration_order: Dict[str, int] = {
            name: i for i, name in enumerate(tokenizer_config.sro_types.keys())
        }
        # action_field per CA type (storage → electrical_storage, etc.)
        self._ca_action_field_by_type: Dict[str, str] = {
            type_name: ca.action_field
            for type_name, ca in tokenizer_config.ca_types.items()
        }
        # CA prefix (the adapter prefix, e.g. "storage") = the CA type_name.
        # WP02 schema requires "storage" and "charger" entries — use that mapping.
        self._ca_prefix_to_type: Dict[str, str] = {
            "storage": "storage",
            "charger": "charger",
        }

    def build(
        self,
        building_id: str,
        observation_names: Sequence[str],
        action_names: Sequence[str],
    ) -> BuildingTokenLayout:
        key = (building_id, tuple(observation_names), tuple(action_names))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        layout = self._compute_layout(building_id, list(observation_names), list(action_names))
        self._cache[key] = layout
        return layout

    def topology_changed(
        self,
        building_id: str,
        observation_names: Sequence[str],
        action_names: Sequence[str],
    ) -> bool:
        key = (building_id, tuple(observation_names), tuple(action_names))
        # Topology is "unchanged" iff some cache entry exists for this building
        # with EXACTLY this names tuple. If none, treat as changed.
        return key not in self._cache

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _compute_layout(
        self,
        building_id: str,
        observation_names: List[str],
        action_names: List[str],
    ) -> BuildingTokenLayout:
        # Step 1: drop excluded features
        excluded_names: List[str] = []
        keep_indices: List[int] = []
        for i, name in enumerate(observation_names):
            if any(p.fullmatch(name) for p in self._excluded_matchers):
                excluded_names.append(name)
            else:
                keep_indices.append(i)

        # Step 2: detect NFC sources
        nfc_left = self._cfg.nfc.expression.left.feature
        nfc_right = self._cfg.nfc.expression.right.feature
        idx_left: Optional[int] = None
        idx_right: Optional[int] = None
        for i in keep_indices:
            if observation_names[i] == nfc_left and idx_left is None:
                idx_left = i
            elif observation_names[i] == nfc_right and idx_right is None:
                idx_right = i
        if idx_left is None or idx_right is None:
            raise ValueError(
                f"NFC sources missing for building {building_id!r}: "
                f"left={nfc_left!r} (found={idx_left is not None}), "
                f"right={nfc_right!r} (found={idx_right is not None})"
            )
        nfc_segment = TokenSegment(
            family="nfc",
            type_name=self._cfg.nfc.type_name,
            instance_id=building_id,
            feature_indices=(idx_left, idx_right),
            feature_names=(nfc_left, nfc_right),
            derived=NfcExpression(op="subtract", left_index_in_segment=0, right_index_in_segment=1),
        )
        # NFC sources consumed → exclude from SRO classification.
        sro_candidates = [i for i in keep_indices if i != idx_left and i != idx_right]

        # Steps 3-4: classify each remaining feature into SRO or CA.
        # Buckets: sro_buckets[(type_name, instance_id)] = list of (idx, name)
        # ca_buckets[(type_name, instance_id)]            = list of (idx, name)
        sro_buckets: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        ca_buckets: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        unmatched: List[str] = []

        for i in sro_candidates:
            name = observation_names[i]
            classification = self._classify_one(name, building_id)
            if classification is None:
                unmatched.append(name)
            elif classification[0] == "sro":
                _, type_name, instance_id = classification
                sro_buckets.setdefault((type_name, instance_id), []).append((i, name))
            else:  # "ca"
                _, type_name, instance_id = classification
                ca_buckets.setdefault((type_name, instance_id), []).append((i, name))

        # Step 5: reject leftovers
        if unmatched:
            bullets = "\n".join(f"  - {n!r}" for n in unmatched)
            raise ValueError(
                f"Tokenizer rule 1 (coverage) failed at runtime for building "
                f"{building_id!r} — the following observation names did not "
                f"match any SRO type, NFC source, or excluded pattern:\n"
                f"{bullets}\n"
                "Add to a feature_pattern, NFC source, or excluded_features.patterns."
            )

        # Order: SROs first by (declaration_order, instance_id), then NFC, then CAs.
        sro_segments: List[TokenSegment] = []
        for (type_name, instance_id), feats in sorted(
            sro_buckets.items(),
            key=lambda kv: (self._sro_declaration_order[kv[0][0]], kv[0][1]),
        ):
            indices = tuple(idx for idx, _ in feats)
            names = tuple(n for _, n in feats)
            sro_segments.append(TokenSegment(
                family="sro", type_name=type_name, instance_id=instance_id,
                feature_indices=indices, feature_names=names,
            ))

        ca_segments_unordered: List[TokenSegment] = []
        for (type_name, instance_id), feats in ca_buckets.items():
            indices = tuple(idx for idx, _ in feats)
            names = tuple(n for _, n in feats)
            ca_segments_unordered.append(TokenSegment(
                family="ca", type_name=type_name, instance_id=instance_id,
                feature_indices=indices, feature_names=names,
            ))

        # CA ordering — single owner: sort to match action_names exactly.
        ca_segments = self._order_cas_by_action_names(
            ca_segments_unordered, list(action_names), building_id,
        )

        segments = tuple(sro_segments) + (nfc_segment,) + tuple(ca_segments)
        ca_action_names = tuple(
            self._ca_action_field_by_type[seg.type_name] for seg in ca_segments
        )
        # Post-condition (defensive — _order_cas_by_action_names guarantees this).
        if ca_action_names != tuple(action_names):
            raise ValueError(
                f"BuildingTokenLayout.ca_action_names {ca_action_names!r} does "
                f"not match action_names[{building_id}] {tuple(action_names)!r}"
            )
        return BuildingTokenLayout(
            building_id=building_id,
            segments=segments,
            n_sro=len(sro_segments),
            n_ca=len(ca_segments),
            ca_action_names=ca_action_names,
            excluded_feature_names=tuple(excluded_names),
        )

    def _classify_one(
        self, name: str, building_id: str,
    ) -> Optional[Tuple[str, str, str]]:
        """Return ('sro'|'ca', type_name, instance_id) or None."""
        # CA detection (storage::… or charger::… without per-asset label)
        m_lab = _PER_ASSET_LABELLED_RE.fullmatch(name)
        if m_lab is None:
            m_un = _PER_ASSET_UNLABELLED_RE.fullmatch(name)
            if m_un is not None:
                prefix = m_un.group("prefix")
                instance_id = m_un.group("id")
                if prefix in self._ca_prefix_to_type:
                    return ("ca", self._ca_prefix_to_type[prefix], instance_id)
                # else: it's a per-asset SRO (pv).
                if prefix == "pv":
                    return ("sro", "pv", instance_id)
        else:
            label = m_lab.group("label")
            instance_id = m_lab.group("id")
            sro_type = "ev_connected" if label == "connected_ev" else "ev_incoming"
            return ("sro", sro_type, instance_id)

        # Singleton SRO match (district or building scoped).
        for sro_name, patterns in self._sro_matchers.items():
            sro = self._cfg.sro_types[sro_name]
            # Per-asset SROs are handled above; here we deal with singletons only.
            if getattr(sro, "cardinality", None) != "singleton":
                continue
            if any(p.fullmatch(name) for p in patterns):
                return ("sro", sro_name, building_id)
        return None

    def _order_cas_by_action_names(
        self,
        ca_segments: List[TokenSegment],
        action_names: List[str],
        building_id: str,
    ) -> List[TokenSegment]:
        """Sort CA segments so that ca_action_names[i] == action_names[i]
        element-wise. There must be exactly as many CA segments as action_names,
        and a 1:1 match by action_field. Raises ValueError otherwise."""
        if len(ca_segments) != len(action_names):
            raise ValueError(
                f"CA count mismatch for building {building_id!r}: "
                f"{len(ca_segments)} CA segments vs {len(action_names)} action names "
                f"(actions={action_names!r})"
            )
        # Group available segments by action_field. Each action_field may have
        # multiple instances (e.g. two chargers → two `electric_vehicle_storage`).
        by_action_field: Dict[str, List[TokenSegment]] = {}
        for seg in ca_segments:
            af = self._ca_action_field_by_type[seg.type_name]
            by_action_field.setdefault(af, []).append(seg)
        # For determinism within a field, sort by instance_id.
        for v in by_action_field.values():
            v.sort(key=lambda s: s.instance_id or "")
        ordered: List[TokenSegment] = []
        consumed: Dict[str, int] = {k: 0 for k in by_action_field}
        for action_field in action_names:
            pool = by_action_field.get(action_field, [])
            i = consumed.get(action_field, 0)
            if i >= len(pool):
                raise ValueError(
                    f"Cannot satisfy action_names ordering for building "
                    f"{building_id!r}: ran out of CA segments for "
                    f"action_field {action_field!r} (have {len(pool)})"
                )
            ordered.append(pool[i])
            consumed[action_field] = i + 1
        return ordered
```

- [ ] **Step 4: Run the failing test**

```bash
pytest tests/test_entity_token_layout.py::test_uses_real_sample_payload -v
```
Expected: PASS. If it fails:
  - Coverage shortfall → some feature is not classified. Print the offending names. Most likely cause: an SRO regex in `entity_default.json` didn't compile to what the sample produces. STOP — this is a real spec/regex bug; do not silently special-case it.
  - CA action mismatch → action_names from the adapter are not what we expect. Inspect the adapter output.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/entity_token_layout.py tests/test_entity_token_layout.py
git commit -m "feat(wp03): implement EntityTokenLayoutBuilder.build (happy path on real sample)"
```

---

### Task 4: Classification tests (§16.1 rows 1–9)

For each of the 9 single-name classification tests below, follow the standard cycle: write test → run (FAIL or PASS — if PASS without code change, it means existing behavior already covers it; that's fine, just commit). The tests are small and largely independent.

For brevity, all 9 tests are listed below in one block. The execution should add them one at a time, run after each, and commit in groups of 3.

```python
# All of these use a single shared fixture:
@pytest.fixture
def builder_and_obs():
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
    from utils.entity_tokenizer_schema import load_entity_tokenizer_config

    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    builder = EntityTokenLayoutBuilder(cfg)
    obs_names = _load_sample_observation_names_for_first_building()
    action_names = ["electrical_storage", "electric_vehicle_storage"]
    layout = builder.build("Building_1", obs_names, action_names)
    return builder, obs_names, layout


def _find_segment(layout, family, type_name, instance_substr=None):
    for s in layout.segments:
        if s.family == family and s.type_name == type_name:
            if instance_substr is None or instance_substr in (s.instance_id or ""):
                return s
    return None


def test_classifies_district_time_to_sro_singleton(builder_and_obs):
    _, obs, layout = builder_and_obs
    seg = _find_segment(layout, "sro", "district_time")
    assert seg is not None
    idx_hour = obs.index("district__hour")
    assert idx_hour in seg.feature_indices


def test_classifies_district_pricing_current_separately_from_forecast(builder_and_obs):
    _, obs, layout = builder_and_obs
    cur = _find_segment(layout, "sro", "district_pricing_current")
    fwd = _find_segment(layout, "sro", "district_pricing_forecast")
    assert cur is not None and fwd is not None
    assert obs.index("district__electricity_pricing") in cur.feature_indices
    # at least one forecast feature
    forecast_names = [n for n in obs if "electricity_pricing_predicted" in n]
    assert any(obs.index(n) in fwd.feature_indices for n in forecast_names)


def test_classifies_district_carbon_separately_from_pricing(builder_and_obs):
    _, obs, layout = builder_and_obs
    carbon = _find_segment(layout, "sro", "district_carbon")
    assert carbon is not None
    assert obs.index("district__carbon_intensity") in carbon.feature_indices
    # ensure carbon_intensity is NOT in any pricing segment
    cur = _find_segment(layout, "sro", "district_pricing_current")
    fwd = _find_segment(layout, "sro", "district_pricing_forecast")
    assert obs.index("district__carbon_intensity") not in cur.feature_indices
    assert obs.index("district__carbon_intensity") not in fwd.feature_indices


def test_classifies_building_storage_state_to_sro(builder_and_obs):
    _, obs, layout = builder_and_obs
    seg = _find_segment(layout, "sro", "building_storage_state")
    assert seg is not None
    assert obs.index("electrical_storage_soc") in seg.feature_indices
    assert obs.index("electrical_storage_soc_ratio") in seg.feature_indices


def test_classifies_per_asset_pv_to_sro(builder_and_obs):
    _, obs, layout = builder_and_obs
    pv_segs = [s for s in layout.segments if s.family == "sro" and s.type_name == "pv"]
    assert len(pv_segs) >= 1
    # All pv:: features in obs must appear in some pv segment
    pv_obs = [n for n in obs if n.startswith("pv::")]
    classified = set()
    for s in pv_segs:
        classified.update(s.feature_indices)
    for n in pv_obs:
        assert obs.index(n) in classified


def test_classifies_per_asset_ev_connected_to_sro(builder_and_obs):
    _, obs, layout = builder_and_obs
    segs = [s for s in layout.segments if s.family == "sro" and s.type_name == "ev_connected"]
    assert len(segs) >= 1
    for s in segs:
        for n in s.feature_names:
            assert "::connected_ev::" in n


def test_classifies_per_asset_ev_incoming_to_sro(builder_and_obs):
    _, obs, layout = builder_and_obs
    segs = [s for s in layout.segments if s.family == "sro" and s.type_name == "ev_incoming"]
    assert len(segs) >= 1
    for s in segs:
        for n in s.feature_names:
            assert "::incoming_ev::" in n


def test_classifies_storage_prefix_to_ca(builder_and_obs):
    _, obs, layout = builder_and_obs
    segs = [s for s in layout.segments if s.family == "ca" and s.type_name == "storage"]
    assert len(segs) >= 1
    for s in segs:
        for n in s.feature_names:
            assert n.startswith("storage::")


def test_classifies_charger_prefix_to_ca(builder_and_obs):
    _, obs, layout = builder_and_obs
    segs = [s for s in layout.segments if s.family == "ca" and s.type_name == "charger"]
    assert len(segs) >= 1
    for s in segs:
        for n in s.feature_names:
            # CA charger features have NO "::connected_ev::" / "::incoming_ev::" segment
            assert "::connected_ev::" not in n
            assert "::incoming_ev::" not in n
            assert n.startswith("charger::")
```

- [ ] **Step 1: Add tests in batches of 3, run, commit**

```bash
# After adding tests 1-3:
pytest tests/test_entity_token_layout.py -k "district_time or pricing_current or carbon_separately" -v
git add tests/test_entity_token_layout.py
git commit -m "test(wp03): cover district SRO classification (time/pricing/carbon)"

# After adding tests 4-6:
pytest tests/test_entity_token_layout.py -k "building_storage or per_asset_pv or per_asset_ev_connected" -v
git add tests/test_entity_token_layout.py
git commit -m "test(wp03): cover building SRO and per-asset SRO classification"

# After adding tests 7-9:
pytest tests/test_entity_token_layout.py -k "ev_incoming or storage_prefix or charger_prefix" -v
git add tests/test_entity_token_layout.py
git commit -m "test(wp03): cover ev_incoming SRO and CA classification"
```

If any test fails, fix the builder (or the regex in `entity_default.json` if the spec is wrong) — never weaken a test to pass.

---

### Task 5: NFC tests (§16.1 rows 10–11)

```python
def test_nfc_segment_has_two_source_indices_and_subtract_op(builder_and_obs):
    _, obs, layout = builder_and_obs
    nfc = next((s for s in layout.segments if s.family == "nfc"), None)
    assert nfc is not None
    assert nfc.derived is not None
    assert nfc.derived.op == "subtract"
    assert nfc.feature_indices == (obs.index("non_shiftable_load"), obs.index("solar_generation"))
    assert nfc.derived.left_index_in_segment == 0
    assert nfc.derived.right_index_in_segment == 1


def test_nfc_source_features_not_in_any_sro_group(builder_and_obs):
    _, obs, layout = builder_and_obs
    for s in layout.segments:
        if s.family == "sro":
            assert "non_shiftable_load" not in s.feature_names
            assert "solar_generation" not in s.feature_names
```

- [ ] **Step 1: Run, expect PASS** (the builder already implements this).

```bash
pytest tests/test_entity_token_layout.py -k nfc -v
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_entity_token_layout.py
git commit -m "test(wp03): cover NFC segment shape and source feature exclusivity"
```

---

### Task 6: Excluded features test (§16.1 row 12 and the deferred WP02 row)

```python
def test_excluded_features_dropped_before_classification(builder_and_obs):
    _, obs, layout = builder_and_obs
    # district__topology_version is in the sample payload AND in
    # excluded_features.patterns. It must appear in excluded_feature_names
    # and NOT in any segment.
    assert "district__topology_version" in layout.excluded_feature_names
    for s in layout.segments:
        assert "district__topology_version" not in s.feature_names
```

Plus the deferred WP02 row, written into `tests/test_entity_tokenizer_config_schema.py`:

```python
def test_excluded_feature_pattern_removes_topology_version():
    """Builder-level corollary of the WP02 exclusion regex."""
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
    from utils.entity_tokenizer_schema import load_entity_tokenizer_config
    # Reuse helper from WP03 tests
    from tests.test_entity_token_layout import _load_sample_observation_names_for_first_building

    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    builder = EntityTokenLayoutBuilder(cfg)
    layout = builder.build(
        "Building_1",
        _load_sample_observation_names_for_first_building(),
        ["electrical_storage", "electric_vehicle_storage"],
    )
    assert "district__topology_version" in layout.excluded_feature_names
```

(If pytest doesn't allow cross-test-module imports cleanly, copy `_load_sample_observation_names_for_first_building` into a small `tests/conftest.py` fixture instead. Use whichever is cleanest.)

- [ ] **Step 1: Run**

```bash
pytest tests/test_entity_token_layout.py::test_excluded_features_dropped_before_classification tests/test_entity_tokenizer_config_schema.py::test_excluded_feature_pattern_removes_topology_version -v
```
Expected: both PASS.

- [ ] **Step 2: Commit**

```bash
git add tests/test_entity_token_layout.py tests/test_entity_tokenizer_config_schema.py
git commit -m "test(wp03): cover excluded-feature handling (incl. deferred WP02 row)"
```

---

### Task 7: Failure-mode tests (§16.1 rows 13–14)

```python
def test_unmatched_feature_raises():
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
    from utils.entity_tokenizer_schema import load_entity_tokenizer_config

    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    builder = EntityTokenLayoutBuilder(cfg)
    obs = _load_sample_observation_names_for_first_building() + [
        "district__some_new_feature"
    ]
    with pytest.raises(ValueError, match="district__some_new_feature"):
        builder.build("Building_1", obs, ["electrical_storage", "electric_vehicle_storage"])


def test_ambiguous_pattern_raises(monkeypatch):
    """Two SRO singletons matching the same district feature."""
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
    from utils.entity_tokenizer_schema import EntityTokenizerConfig

    raw = json.loads(Path("configs/tokenizers/entity_default.json").read_text())
    raw["sro_types"]["district_time_dup"] = {
        "entity_table": "district",
        "cardinality": "singleton",
        "feature_patterns": ["^district__hour$"],
        "input_dim_fallback": 1,
    }
    cfg = EntityTokenizerConfig.model_validate(raw)
    builder = EntityTokenLayoutBuilder(cfg)
    obs = _load_sample_observation_names_for_first_building()
    # The current builder picks the FIRST matching SRO type and stops; that
    # silently masks ambiguity. Spec §13.4 rule 2 catches this at config-load
    # time, but the builder itself should also catch it for symmetry.
    with pytest.raises(ValueError, match=r"district__hour.*(district_time|district_time_dup)"):
        builder.build("Building_1", obs, ["electrical_storage", "electric_vehicle_storage"])
```

- [ ] **Step 1: Run; `test_unmatched_feature_raises` should already PASS, `test_ambiguous_pattern_raises` will FAIL** (current `_classify_one` returns on first match).

- [ ] **Step 2: Make `_classify_one` ambiguity-aware**

In `_classify_one`, replace the singleton SRO loop with one that collects ALL matches:

```python
        # Singleton SRO match (district or building scoped).
        matches: List[str] = []
        for sro_name, patterns in self._sro_matchers.items():
            sro = self._cfg.sro_types[sro_name]
            if getattr(sro, "cardinality", None) != "singleton":
                continue
            if any(p.fullmatch(name) for p in patterns):
                matches.append(sro_name)
        if len(matches) > 1:
            raise ValueError(
                f"Tokenizer rule 2 (uniqueness) violated at runtime: "
                f"observation name {name!r} matches multiple SRO types {matches}. "
                f"Tighten feature_patterns in the tokenizer config."
            )
        if matches:
            return ("sro", matches[0], building_id)
        return None
```

- [ ] **Step 3: Run both**

```bash
pytest tests/test_entity_token_layout.py -k "unmatched_feature or ambiguous_pattern" -v
```
Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add algorithms/utils/entity_token_layout.py tests/test_entity_token_layout.py
git commit -m "feat(wp03): hard-fail on ambiguous SRO matches at build time"
```

---

### Task 8: Ordering tests (§16.1 rows 15–17)

```python
def test_sro_segment_order_follows_config_declaration(builder_and_obs):
    _, _, layout = builder_and_obs
    sros = [s for s in layout.segments if s.family == "sro"]
    type_names_seen = [s.type_name for s in sros]
    # The first three district types declared in the JSON are: district_time,
    # district_weather_current, district_weather_forecast (in that order).
    # Verify any of them present appear in declaration order.
    declared_order = [
        "district_time", "district_weather_current", "district_weather_forecast",
        "district_carbon", "district_pricing_current", "district_pricing_forecast",
        "district_community_energy", "district_community_headroom",
        "district_community_history", "district_meta",
        "building_storage_state", "building_charging_phase_onehot",
        "building_charging_headroom", "building_charging_violation",
        "building_energy_current", "building_energy_history", "building_meta",
        "pv", "ev_connected", "ev_incoming",
    ]
    seen_in_layout = [t for t in type_names_seen if t in declared_order]
    expected_order = [t for t in declared_order if t in type_names_seen]
    # NOTE: same type may appear multiple times for per-asset (one per instance).
    # Map to first-occurrence order for comparison.
    def first_occurrence(seq):
        out, seen = [], set()
        for x in seq:
            if x not in seen:
                out.append(x); seen.add(x)
        return out
    assert first_occurrence(seen_in_layout) == first_occurrence(expected_order)


def test_per_asset_sro_segments_sorted_by_instance_id(builder_and_obs):
    _, _, layout = builder_and_obs
    pv_segs = [s for s in layout.segments if s.family == "sro" and s.type_name == "pv"]
    if len(pv_segs) > 1:
        ids = [s.instance_id for s in pv_segs]
        assert ids == sorted(ids), f"pv instance ids not sorted: {ids}"
    # same for ev_connected / ev_incoming
    for tname in ("ev_connected", "ev_incoming"):
        segs = [s for s in layout.segments if s.family == "sro" and s.type_name == tname]
        if len(segs) > 1:
            ids = [s.instance_id for s in segs]
            assert ids == sorted(ids), f"{tname} instance ids not sorted: {ids}"


def test_segment_overall_order(builder_and_obs):
    _, _, layout = builder_and_obs
    families = [s.family for s in layout.segments]
    # Family blocks must be: zero or more sro, then exactly one nfc, then zero or more ca.
    n_sro = layout.n_sro
    assert families[:n_sro] == ["sro"] * n_sro
    assert families[n_sro] == "nfc"
    assert families[n_sro + 1:] == ["ca"] * layout.n_ca
```

- [ ] **Step 1: Run**

```bash
pytest tests/test_entity_token_layout.py -k "segment_order or per_asset_sro_segments_sorted or segment_overall_order" -v
```
Expected: PASS.

- [ ] **Step 2: Commit**

```bash
git add tests/test_entity_token_layout.py
git commit -m "test(wp03): cover SRO declaration order, per-asset id sort, and segment block order"
```

---

### Task 9: Topology / cache tests (§16.1 rows 18–20)

```python
def test_topology_changed_when_names_differ(builder_and_obs):
    builder, obs, _ = builder_and_obs
    new_obs = list(obs) + ["charger::Building_1/charger_99::power_kw"]
    assert builder.topology_changed(
        "Building_1", new_obs, ["electrical_storage", "electric_vehicle_storage"]
    ) is True


def test_topology_unchanged_for_identical_names(builder_and_obs):
    builder, obs, _ = builder_and_obs
    assert builder.topology_changed(
        "Building_1", obs, ["electrical_storage", "electric_vehicle_storage"]
    ) is False


def test_layout_is_cached(builder_and_obs):
    builder, obs, layout = builder_and_obs
    layout2 = builder.build(
        "Building_1", obs, ["electrical_storage", "electric_vehicle_storage"]
    )
    assert layout2 is layout, "expected cached object identity"
```

- [ ] **Step 1: Run**

```bash
pytest tests/test_entity_token_layout.py -k "topology_changed or topology_unchanged or layout_is_cached" -v
```
Expected: PASS.

- [ ] **Step 2: Commit**

```bash
git add tests/test_entity_token_layout.py
git commit -m "test(wp03): cover layout caching and topology_changed predicate"
```

---

### Task 10: Portability + coverage assertion tests (§16.1 rows 21–22)

```python
def test_no_external_imports():
    """algorithms/utils/entity_token_layout.py imports only stdlib + typing + re."""
    import ast
    src = Path("algorithms/utils/entity_token_layout.py").read_text()
    tree = ast.parse(src)
    forbidden_prefixes = ("torch", "numpy", "algorithms.", "utils.", "pydantic")
    # Note: utils.entity_tokenizer_schema is imported INSIDE __init__ for the
    # matchers — that violates the strict §7.4 rule. The spec acknowledges
    # the builder consumes a parsed config; for portability the matchers must
    # be derivable from the config alone WITHOUT the schema module. See
    # follow-up note in self-review.
    # For now: require no top-level forbidden imports.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                assert not any(n.name.startswith(p) for p in forbidden_prefixes), (
                    f"forbidden top-level import: {n.name}"
                )
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not any(mod.startswith(p) for p in forbidden_prefixes), (
                f"forbidden top-level from-import: {mod}"
            )


def test_coverage_accounting_matches_spec(builder_and_obs):
    _, _, layout = builder_and_obs
    # Per spec §13.1 coverage table for one building:
    counts = {}
    for s in layout.segments:
        if s.family == "sro":
            counts.setdefault(s.type_name, 0)
            counts[s.type_name] += len(s.feature_indices)
    expected_singletons = {
        "district_time": 3, "district_weather_current": 4,
        "district_weather_forecast": 12, "district_carbon": 1,
        "district_pricing_current": 1, "district_pricing_forecast": 3,
        "district_community_energy": 12, "district_community_headroom": 4,
        "district_community_history": 2, "district_meta": 3,
        "building_storage_state": 2, "building_charging_phase_onehot": 6,
        "building_charging_headroom": 8, "building_charging_violation": 1,
        "building_energy_current": 15, "building_energy_history": 4,
        "building_meta": 3,
    }
    for k, expected in expected_singletons.items():
        assert counts.get(k) == expected, (
            f"{k}: expected {expected} features, got {counts.get(k)}"
        )
```

- [ ] **Step 1: Run `test_no_external_imports`** — likely FAIL because the builder imports from `utils.entity_tokenizer_schema` inside `__init__`.

  **Fix:** move the matcher-building helpers (`_sro_matchers`, `_excluded_matchers`) into `entity_token_layout.py` itself as private functions that operate on the parsed config (the config object exposes `.sro_types`, `.excluded_features.patterns`, etc.). The schema module remains the validation owner; the builder just consumes the parsed config without re-importing schema.

  Implement: lift those two functions verbatim from `utils/entity_tokenizer_schema.py` into `entity_token_layout.py` as private module-level functions, then drop the import in `EntityTokenLayoutBuilder.__init__`.

  After the lift, the schema module STILL needs them (for its own validation). Keep them in both places (DRY violation justified by §7.4 portability — the schema module isn't a portability target, the layout builder is).

- [ ] **Step 2: Run both new tests**

```bash
pytest tests/test_entity_token_layout.py -k "no_external_imports or coverage_accounting" -v
```
Expected: both PASS.

- [ ] **Step 3: Commit**

```bash
git add algorithms/utils/entity_token_layout.py tests/test_entity_token_layout.py
git commit -m "feat(wp03): make EntityTokenLayoutBuilder portable (no schema/utils imports) + cover spec coverage table"
```

---

### Task 11: Full test sweep

- [ ] **Step 1: Run everything**

```bash
pytest -x -q
```
Expected: exit 0; the §16.1 row count (22) matches the test count in `tests/test_entity_token_layout.py` (or close — `test_dataclasses_constructible` is an extra smoke test we added, so the count may be 23).

- [ ] **Step 2: Commit any cleanup**

---

## Self-Review Checklist (run before requesting code review)

- [ ] **Spec §16.1 coverage:** Every row of the §16.1 table has a corresponding test in `tests/test_entity_token_layout.py`. Verify by `grep -c "^def test_" tests/test_entity_token_layout.py` — expect ≥ 22.
- [ ] **Builder is pure stdlib:**
  ```bash
  python -c "
  import ast
  tree = ast.parse(open('algorithms/utils/entity_token_layout.py').read())
  forbidden = ('torch','numpy','pydantic','algorithms.','utils.')
  bad = []
  for node in ast.walk(tree):
      if isinstance(node, ast.Import):
          bad += [n.name for n in node.names if any(n.name.startswith(p) for p in forbidden)]
      if isinstance(node, ast.ImportFrom):
          if any((node.module or '').startswith(p) for p in forbidden):
              bad.append(node.module)
  print('forbidden imports:', bad)
  assert not bad
  "
  ```
  Expected: `forbidden imports: []`.
- [ ] **CA ordering invariant on real sample:** for every building in the sample payload, `layout.ca_action_names == tuple(action_names[building])`. Check with a one-off script:
  ```bash
  python -c "
  import json
  from utils.entity_adapter import EntityContractAdapter
  from utils.entity_tokenizer_schema import load_entity_tokenizer_config
  from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
  payload = json.load(open('datasets/tmp_entity_obs_full_step2200_named.json'))
  adapter = EntityContractAdapter()
  obs, names = adapter.to_agent_observations(payload)
  # Build action_names from the schema or the adapter's action_names emission;
  # this script depends on adapter API. If unsure: print and inspect manually.
  print('Skipping — verify manually for now (action_names plumbing in WP05).')
  "
  ```
  If the builder can't be exercised end-to-end yet because action_names plumbing isn't ready, document that and rely on the unit test `test_uses_real_sample_payload` plus the §16.1 coverage rows.
- [ ] **All failure modes tested:** `test_unmatched_feature_raises`, `test_ambiguous_pattern_raises`, the implicit NFC-source-missing `ValueError` in `_compute_layout` is exercised somewhere — if not, add a test:
  ```python
  def test_missing_nfc_source_raises():
      from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
      from utils.entity_tokenizer_schema import load_entity_tokenizer_config
      cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
      builder = EntityTokenLayoutBuilder(cfg)
      obs = ["non_shiftable_load"]  # solar_generation missing
      with pytest.raises(ValueError, match="solar_generation"):
          builder.build("X", obs, [])
  ```
- [ ] **CA count mismatch tested:** add if missing
  ```python
  def test_ca_count_mismatch_raises(builder_and_obs):
      builder, obs, _ = builder_and_obs
      # Provide one fewer action_name than there are CA segments
      with pytest.raises(ValueError, match="CA count mismatch|action_names"):
          builder.build("Building_1", obs, ["electrical_storage"])  # missing charger
  ```
- [ ] **All tests pass, no regression:** `pytest -x -q` exits 0.
- [ ] **Type names from spec §7.2 match exactly:** `TokenSegment`, `NfcExpression`, `BuildingTokenLayout`, `EntityTokenLayoutBuilder`. `grep -E "^(class|@dataclass)" algorithms/utils/entity_token_layout.py` — confirm exact spelling.

---

## Code Review

After the self-review checklist passes, invoke `superpowers:requesting-code-review` to dispatch a fresh subagent. The reviewer should check the diff against this plan and §7 + §16.1 of `docs/specv2.md`. Resolve blocking findings before opening the PR.

---

## PR Description

```markdown
## Summary
Implements `EntityTokenLayoutBuilder` (`algorithms/utils/entity_token_layout.py`) per spec §7. The builder consumes per-building `observation_names` + `action_names` and produces a deterministic `BuildingTokenLayout` whose CA segment order is element-wise equal to `action_names[building]` by construction. Pure stdlib — no torch, numpy, or repo-internal imports — so it is portable to the inference repo.

## Key Changes
- Add `algorithms/utils/entity_token_layout.py`: `TokenSegment`, `NfcExpression`, `BuildingTokenLayout` dataclasses + `EntityTokenLayoutBuilder` (build, topology_changed, layout cache, classification, ordering).
- Owns CA ordering as a single source of truth: builder permutes CA segments to satisfy `ca_action_names == tuple(action_names[building])`; raises `ValueError` if no permutation works.
- Hard-fails on unmatched features (runtime mirror of spec §13.4 rule 1) and on ambiguous SRO matches (runtime mirror of rule 2).
- Add `tests/test_entity_token_layout.py` covering all of spec §16.1 (22 rows) plus a couple of defensive cases (missing NFC source, CA count mismatch).
- Add the deferred WP02 row `test_excluded_feature_pattern_removes_topology_version` (now feasible because the builder exists).
```
