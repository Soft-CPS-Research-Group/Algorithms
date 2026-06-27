"""EntityTokenLayoutBuilder — pure-Python token segmentation.

No torch / numpy / pydantic / algorithms.* / utils.* imports — this module
is portable to the inference repo.

The builder is **passive**: it consumes per-building observation/action
names that the wrapper has already produced and returns a deterministic
``BuildingTokenLayout``. CA segments are permuted to match ``action_names``
position-by-position; this is the single source of truth that the agent's
startup assertion checks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NfcExpression:
    """Compiled NFC computation: ``op(features[left], features[right])``.

    Indices are offsets into the owning ``TokenSegment.feature_indices``,
    NOT absolute observation indices.
    """

    op: str
    left_index_in_segment: int
    right_index_in_segment: int


@dataclass(frozen=True)
class TokenSegment:
    """One token's worth of slice metadata into the per-building observation.

    ``feature_indices`` is a tuple (so it is hashable) and supports
    non-contiguous selection — important for interleaved district forecast
    feature groups.
    """

    family: str  # "sro" | "nfc" | "ca"
    type_name: str
    instance_id: Optional[str]
    feature_indices: Tuple[int, ...]
    feature_names: Tuple[str, ...]
    derived: Optional[NfcExpression] = None


@dataclass(frozen=True)
class BuildingTokenLayout:
    """Deterministic layout for a single building.

    ``ca_action_names`` MUST equal ``tuple(action_names[building])`` element
    by element — enforced at construction time so the agent's startup
    assertion is a redundant check, not the first failure point.
    """

    building_id: str
    segments: Tuple[TokenSegment, ...]
    n_sro: int
    n_ca: int
    ca_action_names: Tuple[str, ...]
    excluded_feature_names: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# Per-asset adapter prefixes use "::" as separator (see utils/entity_adapter.py).
# Two id forms appear in adapter output:
#   pv::Building_1/pv::generation_power_kw                  → unlabelled
#   charger::Building_1/C1::power_kw                        → unlabelled
#   storage::Building_1/storage::soc                        → unlabelled
#   charger::Building_1/C1::connected_ev::soc               → labelled
#   charger::Building_1/C1::incoming_ev::soc                → labelled
_PER_ASSET_LABELLED_RE = re.compile(
    r"^(?P<prefix>charger)::(?P<id>[^:]+(?:/[^:]+)*)::"
    r"(?P<label>connected_ev|incoming_ev)::(?P<feature>.+)$"
)
_PER_ASSET_UNLABELLED_RE = re.compile(
    r"^(?P<prefix>pv|storage|charger|deferrable_appliance)::(?P<id>[^:]+(?:/[^:]+)*)::"
    r"(?P<feature>(?!connected_ev::|incoming_ev::).+)$"
)
_DISTRICT_PREFIX = "district__"


def _compile_pattern(pattern: str, json_path: str) -> "re.Pattern[str]":
    """Compile a regex with a contextual error path."""
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(
            f"Invalid regex at {json_path}: {pattern!r} -> {exc}"
        ) from exc


def _build_excluded_matchers(tokenizer_config: Any) -> List["re.Pattern[str]"]:
    """Compile excluded_features.patterns. Lifted from
    ``utils/entity_tokenizer_schema.py`` so this module stays portable
    (no schema-module import allowed)."""
    return [
        _compile_pattern(p, f"excluded_features.patterns[{i}]")
        for i, p in enumerate(tokenizer_config.excluded_features.patterns)
    ]


def _build_sro_matchers(
    tokenizer_config: Any,
) -> Dict[str, List["re.Pattern[str]"]]:
    """Compile feature_patterns for every singleton SRO type (same
    portability rationale as above)."""
    out: Dict[str, List["re.Pattern[str]"]] = {}
    for type_name, sro in tokenizer_config.sro_types.items():
        # Skip per-asset entries; they have no feature_patterns.
        if getattr(sro, "cardinality", None) != "singleton":
            continue
        out[type_name] = [
            _compile_pattern(
                p, f"sro_types.{type_name}.feature_patterns[{i}]"
            )
            for i, p in enumerate(sro.feature_patterns)
        ]
    return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class EntityTokenLayoutBuilder:
    """Builds per-building TokenLayout. Caches by
    ``(building_id, observation_names, action_names)``."""

    def __init__(self, tokenizer_config: Any) -> None:
        self._cfg = tokenizer_config
        self._cache: Dict[
            Tuple[str, Tuple[str, ...], Tuple[str, ...]], BuildingTokenLayout
        ] = {}
        self._sro_matchers = _build_sro_matchers(tokenizer_config)
        self._excluded_matchers = _build_excluded_matchers(tokenizer_config)
        # SRO declaration order = order of insertion into the JSON dict
        # (preserved by Python 3.7+ dict semantics).
        self._sro_declaration_order: Dict[str, int] = {
            name: i for i, name in enumerate(tokenizer_config.sro_types.keys())
        }
        # action_field per CA type (storage → electrical_storage, etc.)
        self._ca_action_field_by_type: Dict[str, str] = {
            type_name: ca.action_field
            for type_name, ca in tokenizer_config.ca_types.items()
        }
        # Adapter prefix (e.g. "storage", "charger") → CA type_name.
        self._ca_prefix_to_type: Dict[str, str] = {
            "storage": "storage",
            "charger": "charger",
        }

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

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
        layout = self._compute_layout(
            building_id, list(observation_names), list(action_names)
        )
        self._cache[key] = layout
        return layout

    def topology_changed(
        self,
        building_id: str,
        observation_names: Sequence[str],
        action_names: Sequence[str],
    ) -> bool:
        """Cheap predicate: ``True`` iff no cached layout exists for this
        exact ``(building, observation_names, action_names)`` combination."""
        key = (building_id, tuple(observation_names), tuple(action_names))
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
        # --- Step 1: drop excluded features --------------------------------
        excluded_names: List[str] = []
        keep_indices: List[int] = []
        for i, name in enumerate(observation_names):
            if any(p.fullmatch(name) for p in self._excluded_matchers):
                excluded_names.append(name)
            else:
                keep_indices.append(i)

        # --- Step 2: detect NFC sources ------------------------------------
        nfc_left = self._cfg.nfc.expression.left.feature
        nfc_right = self._cfg.nfc.expression.right.feature
        idx_left: Optional[int] = None
        idx_right: Optional[int] = None
        for i in keep_indices:
            name = observation_names[i]
            if name == nfc_left and idx_left is None:
                idx_left = i
            elif name == nfc_right and idx_right is None:
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
            derived=NfcExpression(
                op=self._cfg.nfc.expression.op,
                left_index_in_segment=0,
                right_index_in_segment=1,
            ),
        )
        # NFC sources consumed → exclude from SRO classification.
        sro_candidates = [
            i for i in keep_indices if i != idx_left and i != idx_right
        ]

        # --- Steps 3-4: classify each remaining feature into SRO or CA -----
        sro_buckets: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        ca_buckets: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        unmatched: List[str] = []

        for i in sro_candidates:
            name = observation_names[i]
            classification = self._classify_one(name, building_id)
            if classification is None:
                unmatched.append(name)
            else:
                family, type_name, instance_id = classification
                bucket = sro_buckets if family == "sro" else ca_buckets
                bucket.setdefault((type_name, instance_id), []).append(
                    (i, name)
                )

        # --- Step 5: hard-fail on leftovers ---
        if unmatched:
            bullets = "\n".join(f"  - {n!r}" for n in unmatched)
            raise ValueError(
                f"Tokenizer rule 1 (coverage) failed at runtime for building "
                f"{building_id!r} — the following observation names did not "
                f"match any SRO type, NFC source, or excluded pattern:\n"
                f"{bullets}\n"
                "Add to a feature_pattern, NFC source, or "
                "excluded_features.patterns."
            )

        # --- Order: SROs by (declaration_order, instance_id), then NFC,
        # then CAs in action_names order. -----------------------------------
        sro_segments: List[TokenSegment] = []
        for (type_name, instance_id), feats in sorted(
            sro_buckets.items(),
            key=lambda kv: (
                self._sro_declaration_order[kv[0][0]],
                kv[0][1] or "",
            ),
        ):
            indices = tuple(idx for idx, _ in feats)
            names = tuple(n for _, n in feats)
            sro_segments.append(
                TokenSegment(
                    family="sro",
                    type_name=type_name,
                    instance_id=instance_id,
                    feature_indices=indices,
                    feature_names=names,
                )
            )

        ca_segments_unordered: List[TokenSegment] = []
        for (type_name, instance_id), feats in ca_buckets.items():
            indices = tuple(idx for idx, _ in feats)
            names = tuple(n for _, n in feats)
            ca_segments_unordered.append(
                TokenSegment(
                    family="ca",
                    type_name=type_name,
                    instance_id=instance_id,
                    feature_indices=indices,
                    feature_names=names,
                )
            )

        ca_segments = self._order_cas_by_action_names(
            ca_segments_unordered, list(action_names), building_id
        )

        segments = (
            tuple(sro_segments) + (nfc_segment,) + tuple(ca_segments)
        )
        ca_action_names = tuple(
            self._ca_action_field_by_type[seg.type_name]
            for seg in ca_segments
        )
        # Defensive post-condition (the orderer guarantees this).
        # Each entry of ``ca_action_names`` (the segment's action_field) must
        # either equal the corresponding ``action_name`` exactly, or be a
        # ``"<action_field>_<instance_id>"`` prefix of it (CityLearn appends
        # a charger-id suffix when multiple CAs of the same type exist).
        for af, an in zip(ca_action_names, action_names):
            if an == af:
                continue
            if an.startswith(af + "_"):
                continue
            raise ValueError(
                f"BuildingTokenLayout.ca_action_names {ca_action_names!r} "
                f"does not match action_names[{building_id}] "
                f"{tuple(action_names)!r}"
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
        self, name: str, building_id: str
    ) -> Optional[Tuple[str, str, str]]:
        """Return ``(family, type_name, instance_id)`` or ``None``.

        Hard-fails on ambiguous singleton SRO matches.
        """
        # Per-asset (labelled) → ev_connected / ev_incoming SRO.
        m_lab = _PER_ASSET_LABELLED_RE.fullmatch(name)
        if m_lab is not None:
            label = m_lab.group("label")
            instance_id = m_lab.group("id")
            sro_type = (
                "ev_connected" if label == "connected_ev" else "ev_incoming"
            )
            return ("sro", sro_type, instance_id)

        # Per-asset (unlabelled) → CA (storage/charger) or per-asset SRO (pv).
        m_un = _PER_ASSET_UNLABELLED_RE.fullmatch(name)
        if m_un is not None:
            prefix = m_un.group("prefix")
            instance_id = m_un.group("id")
            if prefix in self._ca_prefix_to_type:
                return ("ca", self._ca_prefix_to_type[prefix], instance_id)
            if prefix == "pv":
                return ("sro", "pv", instance_id)
            if prefix == "deferrable_appliance":
                return ("sro", "deferrable_appliance", instance_id)

        # Singleton SRO match (district or building scoped). Collect ALL
        # matches to detect ambiguity.
        matches: List[str] = []
        for sro_name, patterns in self._sro_matchers.items():
            if any(p.fullmatch(name) for p in patterns):
                matches.append(sro_name)
        if len(matches) > 1:
            raise ValueError(
                f"Tokenizer rule 2 (uniqueness) violated at runtime: "
                f"observation name {name!r} matches multiple SRO types "
                f"{matches}. Tighten feature_patterns in the tokenizer config."
            )
        if matches:
            return ("sro", matches[0], building_id)
        return None

    def _order_cas_by_action_names(
        self,
        ca_segments: List[TokenSegment],
        action_names: List[str],
        building_id: str,
    ) -> List[TokenSegment]:
        """Sort CA segments so ``ca_action_names[i] == action_names[i]``.

        Matching strategy (in order):

        1. Exact match: ``action_name == action_field``. This holds when the
           simulator emits unsuffixed names (e.g., ``electrical_storage`` for
           the single building-level battery).
        2. Prefix match: ``action_name == f"{action_field}_{instance_id}"``.
           CityLearn appends a charger ID suffix when multiple CAs of the
           same type exist (e.g.,
           ``electric_vehicle_storage_charger_1_1``); we match by checking
           ``action_name.startswith(action_field + "_")`` and verifying the
           suffix equals the segment's ``instance_id``. Falls back to first
           prefix-matching segment if exact instance match fails.

        Within an action_field, segments are picked in sorted ``instance_id``
        order for determinism (only relevant for the prefix fallback).
        """
        if len(ca_segments) != len(action_names):
            raise ValueError(
                f"CA count mismatch for building {building_id!r}: "
                f"{len(ca_segments)} CA segments vs {len(action_names)} "
                f"action names (actions={action_names!r})"
            )
        by_action_field: Dict[str, List[TokenSegment]] = {}
        for seg in ca_segments:
            af = self._ca_action_field_by_type[seg.type_name]
            by_action_field.setdefault(af, []).append(seg)
        for v in by_action_field.values():
            v.sort(key=lambda s: s.instance_id or "")

        # Build (action_field, action_name) pairs by matching each action
        # name to its action_field via exact-or-prefix.
        all_action_fields = set(by_action_field.keys())
        ordered: List[TokenSegment] = []
        # Track which segments we've already consumed (by id()) so each one
        # is used exactly once.
        consumed_ids: set = set()
        for action_name in action_names:
            chosen_field: Optional[str] = None
            if action_name in all_action_fields:
                chosen_field = action_name
            else:
                # Prefix match: action_name = f"{action_field}_{instance_id}"
                candidates = [
                    af for af in all_action_fields
                    if action_name.startswith(af + "_")
                ]
                if len(candidates) == 1:
                    chosen_field = candidates[0]
                elif len(candidates) > 1:
                    # Ambiguity: pick the longest matching prefix.
                    chosen_field = max(candidates, key=len)
            if chosen_field is None:
                raise ValueError(
                    f"Cannot map action_name {action_name!r} to any "
                    f"action_field in {sorted(all_action_fields)} for "
                    f"building {building_id!r}"
                )

            # Find the next unconsumed segment in this action_field's pool.
            # Prefer instance_id == suffix-of-action_name when present.
            pool = by_action_field[chosen_field]
            suffix = action_name[len(chosen_field) + 1 :] if action_name != chosen_field else None
            picked: Optional[TokenSegment] = None
            if suffix:
                for seg in pool:
                    if id(seg) in consumed_ids:
                        continue
                    if seg.instance_id == suffix:
                        picked = seg
                        break
            if picked is None:
                # Fall back to first unconsumed segment in deterministic order.
                for seg in pool:
                    if id(seg) not in consumed_ids:
                        picked = seg
                        break
            if picked is None:
                raise ValueError(
                    f"Cannot satisfy action_names ordering for building "
                    f"{building_id!r}: ran out of CA segments for "
                    f"action_field {chosen_field!r} (have {len(pool)})"
                )
            ordered.append(picked)
            consumed_ids.add(id(picked))
        return ordered
