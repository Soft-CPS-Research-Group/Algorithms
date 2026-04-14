"""Observation Enricher — injects token-type markers into observation names/values.

This module classifies raw observation features into token groups (CA, SRO, NFC)
and injects marker names/values that allow the tokenizer to identify token boundaries
without heuristic-based classification.

Portable: no dependencies on training-only code. Can be used in both the training
wrapper and the production inference preprocessor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentResult:
    """Result of observation name enrichment.

    Attributes:
        enriched_names: List of observation names with __tkn_*__ markers inserted.
        marker_encoder_specs: List of (marker_name, encoder_spec) pairs for each
            injected marker. encoder_spec is {"type": "NoNormalization"}.
        marker_positions: Dict mapping marker_name → list of positions in enriched_names.
    """

    enriched_names: List[str]
    marker_encoder_specs: List[Tuple[str, dict]]
    marker_positions: Dict[str, List[int]]


# ---------------------------------------------------------------------------
# Classification helpers (moved from observation_tokenizer.py)
# ---------------------------------------------------------------------------


def _extract_device_ids(
    action_names: List[str],
    ca_config: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Optional[str]]]:
    """Extract device IDs from action names, grouped by CA type.

    For each CA type, strips the configured ``action_name`` prefix from each
    matching action name.  The remainder (if any) is the device ID.

    Returns
    -------
    Dict mapping ``ca_type_name`` → list of device IDs (``None`` for single-instance).

    Examples
    --------
    >>> _extract_device_ids(
    ...     ["electrical_storage", "electric_vehicle_storage_charger_1_1"],
    ...     {"battery": {"action_name": "electrical_storage"},
    ...      "ev_charger": {"action_name": "electric_vehicle_storage"}},
    ... )
    {'battery': [None], 'ev_charger': ['charger_1_1']}
    """
    result: Dict[str, List[Optional[str]]] = {}

    for ca_type_name, ca_spec in ca_config.items():
        action_prefix = ca_spec.get("action_name", "")
        if not action_prefix:
            continue

        device_ids: List[Optional[str]] = []
        for act_name in action_names:
            if act_name == action_prefix:
                # Exact match → single-instance, no device ID
                device_ids.append(None)
            elif act_name.startswith(action_prefix + "_"):
                # Has a device ID suffix
                device_id = act_name[len(action_prefix) + 1 :]
                device_ids.append(device_id)

        if device_ids:
            result[ca_type_name] = device_ids

    return result


def _contains_device_id(raw_name: str, device_id: str) -> bool:
    """Check whether *device_id* appears as a bounded token in *raw_name*.

    A bounded match means the device_id is surrounded by ``_`` (or at the
    start/end of the string).  This prevents false positives when the
    device_id is short (e.g. ``"1"``).

    Examples
    --------
    >>> _contains_device_id("washing_machine_1_start_time_step", "1")
    True
    >>> _contains_device_id("electricity_pricing_predicted_1", "1")
    True

    In practice this is acceptable because the CA feature-pattern check
    happens **first** (e.g. ``start_time_step`` must be in the name), so
    ``electricity_pricing_predicted_1`` is never tested against a
    washing-machine CA type.
    """
    # Direct regex: device_id as a whole "word" with _ as separator
    pattern = r"(?:^|_)" + re.escape(device_id) + r"(?:_|$)"
    return re.search(pattern, raw_name) is not None


def _feature_matches_ca_type(
    raw_name: str,
    feature_patterns: List[str],
) -> bool:
    """Check if *raw_name* contains any of the configured feature substrings."""
    return any(pattern in raw_name for pattern in feature_patterns)


# ---------------------------------------------------------------------------
# ObservationEnricher
# ---------------------------------------------------------------------------


class ObservationEnricher:
    """Classifies observation features and injects token-type markers.

    Portable: no dependencies on training-only code. Can be used in
    both the training wrapper and the production inference preprocessor.
    """

    def __init__(self, tokenizer_config: Dict[str, Any]) -> None:
        """Initialize the enricher with tokenizer configuration.

        Args:
            tokenizer_config: The tokenizer section from config YAML.
                Must contain 'ca_types', 'sro_types', 'rl' keys.
        """
        self._ca_config: Dict[str, Dict[str, Any]] = tokenizer_config.get("ca_types", {})
        self._sro_config: Dict[str, Dict[str, Any]] = tokenizer_config.get("sro_types", {})
        self._rl_config: Dict[str, Any] = tokenizer_config.get("rl", {})

        # Cache for topology change detection
        self._cache_key: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None
        self._cached_result: Optional[EnrichmentResult] = None
        self._insertion_positions: List[int] = []  # Positions where markers are inserted
        self._original_to_enriched: List[int] = []  # Maps original index -> enriched index

    def enrich_names(
        self,
        observation_names: List[str],
        action_names: List[str],
    ) -> EnrichmentResult:
        """Classify features and inject marker names.

        Args:
            observation_names: Raw observation names for one building.
            action_names: Action names for one building (used to detect CA instances).

        Returns:
            EnrichmentResult with enriched_names, marker_encoder_specs, and marker_positions.
        """
        key = (tuple(observation_names), tuple(action_names))
        if key == self._cache_key and self._cached_result is not None:
            return self._cached_result

        # --- Step 1: Classify features into token groups ---
        assigned: set[str] = set()

        # CA classification (action-based instance detection)
        device_ids_by_type = _extract_device_ids(action_names, self._ca_config)

        # {ca_type_name: {device_id_or_None: [raw_name, ...]}}
        ca_groups: Dict[str, Dict[Optional[str], List[str]]] = {}

        for ca_type_name, ca_spec in self._ca_config.items():
            ca_feature_patterns: List[str] = ca_spec.get("features", [])
            device_ids = device_ids_by_type.get(ca_type_name, [])

            if not device_ids:
                continue

            instance_map: Dict[Optional[str], List[str]] = {}
            for dev_id in device_ids:
                instance_map[dev_id] = []

            for raw_name in observation_names:
                if raw_name in assigned:
                    continue

                if not _feature_matches_ca_type(raw_name, ca_feature_patterns):
                    continue

                if len(device_ids) == 1 and device_ids[0] is None:
                    instance_map[None].append(raw_name)
                    assigned.add(raw_name)
                else:
                    for dev_id in device_ids:
                        if dev_id is not None and _contains_device_id(raw_name, dev_id):
                            instance_map[dev_id].append(raw_name)
                            assigned.add(raw_name)
                            break

            # Only keep instances with features
            instance_map = {k: v for k, v in instance_map.items() if v}
            if instance_map:
                ca_groups[ca_type_name] = instance_map

        # SRO classification
        sro_groups: Dict[str, List[str]] = {}

        for sro_type_name, sro_spec in self._sro_config.items():
            sro_features: List[str] = sro_spec.get("features", [])
            matched_names: List[str] = []

            for raw_name in observation_names:
                if raw_name in assigned:
                    continue
                for sro_feat in sro_features:
                    if sro_feat in raw_name:
                        matched_names.append(raw_name)
                        assigned.add(raw_name)
                        break

            if matched_names:
                sro_groups[sro_type_name] = matched_names

        # RL (NFC) classification
        rl_demand_feature = self._rl_config.get("demand_feature")
        rl_generation_features: List[str] = self._rl_config.get("generation_features", [])
        rl_extra_features: List[str] = self._rl_config.get("extra_features", [])
        rl_names: List[str] = []

        for raw_name in observation_names:
            if raw_name in assigned:
                continue
            if rl_demand_feature and rl_demand_feature in raw_name:
                rl_names.append(raw_name)
                assigned.add(raw_name)
            elif any(gen in raw_name for gen in rl_generation_features):
                rl_names.append(raw_name)
                assigned.add(raw_name)
            elif any(extra in raw_name for extra in rl_extra_features):
                rl_names.append(raw_name)
                assigned.add(raw_name)

        # Collect unmatched features
        unmatched = [n for n in observation_names if n not in assigned]

        # --- Step 2: Build enriched names list with markers ---
        # Also build mapping from original index to enriched index
        enriched_names: List[str] = []
        marker_encoder_specs: List[Tuple[str, dict]] = []
        marker_positions: Dict[str, List[int]] = {}
        insertion_positions: List[int] = []
        
        # Build reverse lookup: raw_name -> original index
        raw_name_to_original_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(observation_names)
        }
        # Track original_idx -> enriched_idx (excluding markers)
        original_to_enriched: List[int] = [-1] * len(observation_names)

        def add_marker(marker_name: str) -> None:
            """Helper to add a marker and track its position."""
            pos = len(enriched_names)
            enriched_names.append(marker_name)
            insertion_positions.append(pos)
            marker_encoder_specs.append((marker_name, {"type": "NoNormalization"}))
            if marker_name not in marker_positions:
                marker_positions[marker_name] = []
            marker_positions[marker_name].append(pos)
        
        def add_features(features: List[str]) -> None:
            """Helper to add features and track their original->enriched mapping."""
            for feat in features:
                enriched_idx = len(enriched_names)
                original_idx = raw_name_to_original_idx[feat]
                original_to_enriched[original_idx] = enriched_idx
                enriched_names.append(feat)

        # Add CA groups (sorted by type name for determinism)
        for ca_type_name in sorted(ca_groups.keys()):
            instance_map = ca_groups[ca_type_name]
            # Sort device IDs for determinism
            for device_id in sorted(instance_map.keys(), key=lambda x: x or ""):
                features = instance_map[device_id]
                # Build marker name with device ID if multi-instance
                if device_id is not None:
                    marker_name = f"__tkn_ca_{ca_type_name}__{device_id}__"
                else:
                    marker_name = f"__tkn_ca_{ca_type_name}__"
                add_marker(marker_name)
                add_features(features)

        # Add SRO groups (sorted by type name for determinism)
        for sro_type_name in sorted(sro_groups.keys()):
            features = sro_groups[sro_type_name]
            marker_name = f"__tkn_sro_{sro_type_name}__"
            add_marker(marker_name)
            add_features(features)

        # Add NFC (RL) group if present
        if rl_names:
            marker_name = "__tkn_nfc__"
            add_marker(marker_name)
            add_features(rl_names)

        # Add unmatched features at the end (no marker)
        add_features(unmatched)

        result = EnrichmentResult(
            enriched_names=enriched_names,
            marker_encoder_specs=marker_encoder_specs,
            marker_positions=marker_positions,
        )

        # Cache results
        self._cache_key = key
        self._cached_result = result
        self._insertion_positions = insertion_positions
        self._original_to_enriched = original_to_enriched

        return result

    def enrich_values(
        self,
        observation_values: List[float],
    ) -> List[float]:
        """Insert marker values and reorder to match enriched names.

        Must be called AFTER enrich_names() for the same building.
        Uses cached positions and mapping from the last enrich_names() call.

        Marker values are 0.0 (NoNormalization passes them through as-is;
        the tokenizer ignores the actual value — it uses the name for classification).

        Args:
            observation_values: Raw observation values (same length as
                the original observation_names passed to enrich_names).

        Returns:
            Enriched values list (same length as enriched_names), reordered
            to match the enriched names order.
        """
        if self._cached_result is None:
            raise RuntimeError("enrich_names() must be called before enrich_values()")

        # Calculate expected original length
        n_markers = len(self._insertion_positions)
        expected_original_len = len(self._cached_result.enriched_names) - n_markers

        if len(observation_values) != expected_original_len:
            raise ValueError(
                f"observation_values length ({len(observation_values)}) does not match "
                f"expected original length ({expected_original_len})"
            )

        # Build enriched values: markers get 0.0, features get reordered values
        enriched_len = len(self._cached_result.enriched_names)
        enriched_values: List[float] = [0.0] * enriched_len

        # Place each original value at its enriched position
        for original_idx, enriched_idx in enumerate(self._original_to_enriched):
            if enriched_idx >= 0:  # -1 means not placed (shouldn't happen)
                enriched_values[enriched_idx] = observation_values[original_idx]

        return enriched_values
