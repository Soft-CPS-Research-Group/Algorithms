"""Observation Enricher — injects token-type markers into observation names/values.

This module classifies raw observation features into token groups (CA, SRO, NFC)
and injects marker values that allow the tokenizer to identify token boundaries
without heuristic-based classification.

Portable: no dependencies on training-only code. Can be used in both the training
wrapper and the production inference preprocessor.

No PyTorch/NumPy dependencies — pure Python with stdlib + typing only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EnrichmentResult:
    """Result of observation name enrichment.

    Attributes:
        enriched_names: List of observation names with marker names inserted.
        marker_positions: Dict mapping marker_name -> list of positions in enriched_names.
        marker_to_type: Dict mapping marker_name -> (family, type_name, device_id).
            family is "ca", "sro", or "nfc".
            type_name is e.g., "battery", "temporal".
            device_id is e.g., "charger_1_1" for multi-instance CAs, None otherwise.
    """

    enriched_names: List[str]
    marker_positions: Dict[str, List[int]]
    marker_to_type: Dict[str, Tuple[str, str, Optional[str]]]


def _extract_device_ids(
    action_names: List[str],
    ca_config: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Optional[str]]]:
    """Extract device IDs from action names, grouped by CA type.

    For each CA type, strips the configured ``action_name`` prefix from each
    matching action name. The remainder (if any) is the device ID.

    Args:
        action_names: List of action names for one building.
        ca_config: CA types configuration with action_name patterns.

    Returns:
        Dict mapping ca_type_name -> list of device IDs (None for single-instance).

    Examples:
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
                # Exact match -> single-instance, no device ID
                device_ids.append(None)
            elif act_name.startswith(action_prefix + "_"):
                # Has a device ID suffix
                device_id = act_name[len(action_prefix) + 1:]
                device_ids.append(device_id)

        if device_ids:
            result[ca_type_name] = device_ids

    return result


def _feature_matches_patterns(feature_name: str, patterns: List[str]) -> bool:
    """Check if feature_name contains any of the patterns as substring.

    Args:
        feature_name: Raw observation feature name.
        patterns: List of pattern substrings to match against.

    Returns:
        True if any pattern is a substring of feature_name.
    """
    return any(pattern in feature_name for pattern in patterns)


def _contains_device_id(feature_name: str, device_id: str) -> bool:
    """Check whether device_id appears as a bounded token in feature_name.

    A bounded match means the device_id is surrounded by ``_`` (or at the
    start/end of the string). This prevents false positives when the
    device_id is short.

    Args:
        feature_name: Raw observation feature name.
        device_id: Device ID to search for (e.g., "charger_1_1").

    Returns:
        True if device_id appears as bounded token in feature_name.
    """
    pattern = r"(?:^|_)" + re.escape(device_id) + r"(?:_|$)"
    return re.search(pattern, feature_name) is not None


def _classify_feature(
    feature_name: str,
    tokenizer_config: Dict[str, Any],
    device_ids_by_type: Dict[str, List[Optional[str]]],
) -> Optional[Tuple[str, str, Optional[str]]]:
    """Classify a feature into a token group.

    Checks in order: CA types, SRO types, NFC.

    Args:
        feature_name: Raw observation feature name.
        tokenizer_config: Full tokenizer config dict.
        device_ids_by_type: Device IDs per CA type (from _extract_device_ids).

    Returns:
        Tuple of (family, type_name, device_id) or None if unmatched.
        family is "ca", "sro", or "nfc".
    """
    ca_config = tokenizer_config.get("ca_types", {})
    sro_config = tokenizer_config.get("sro_types", {})
    nfc_config = tokenizer_config.get("nfc", {})

    # 1. Try CA types
    for ca_type_name, ca_spec in ca_config.items():
        patterns = ca_spec.get("features", [])
        if _feature_matches_patterns(feature_name, patterns):
            # Determine which device instance this belongs to
            device_ids = device_ids_by_type.get(ca_type_name, [None])
            
            # If only one instance (or no device IDs), use first/None
            if len(device_ids) <= 1:
                return ("ca", ca_type_name, device_ids[0] if device_ids else None)
            
            # Multiple instances - find which device ID is in the feature name
            for device_id in device_ids:
                if device_id is not None and _contains_device_id(feature_name, device_id):
                    return ("ca", ca_type_name, device_id)
            
            # Feature matches CA type but no specific device ID found
            # This shouldn't happen with well-formed data, but return first as fallback
            return ("ca", ca_type_name, device_ids[0])

    # 2. Try SRO types
    for sro_type_name, sro_spec in sro_config.items():
        patterns = sro_spec.get("features", [])
        if _feature_matches_patterns(feature_name, patterns):
            return ("sro", sro_type_name, None)

    # 3. Try NFC
    demand_patterns = nfc_config.get("demand_features", [])
    generation_patterns = nfc_config.get("generation_features", [])
    extra_patterns = nfc_config.get("extra_features", [])
    all_nfc_patterns = demand_patterns + generation_patterns + extra_patterns
    
    if _feature_matches_patterns(feature_name, all_nfc_patterns):
        return ("nfc", "nfc", None)

    # 4. Unmatched
    return None


class ObservationEnricher:
    """Classifies observation features and injects token-type markers.

    Portable: no dependencies on training-only code. Can be used in
    both the training wrapper and the production inference preprocessor.
    """

    def __init__(self, tokenizer_config: Dict[str, Any]) -> None:
        """Initialize the enricher with tokenizer configuration.

        Args:
            tokenizer_config: The tokenizer config dict.
                Must contain 'marker_values', 'ca_types', 'sro_types', 'nfc' keys.
        """
        self._config = tokenizer_config
        self._marker_values = tokenizer_config.get("marker_values", {})
        self._ca_config = tokenizer_config.get("ca_types", {})
        self._sro_config = tokenizer_config.get("sro_types", {})
        self._nfc_config = tokenizer_config.get("nfc", {})

        # Cache for topology change detection
        self._cache_key: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None
        self._cached_result: Optional[EnrichmentResult] = None
        self._insertion_positions: List[int] = []
        self._marker_values_list: List[float] = []

    def enrich_names(
        self,
        observation_names: List[str],
        action_names: List[str],
    ) -> EnrichmentResult:
        """Classify features and produce enriched observation names.

        Called once per topology (cached until topology changes).

        Args:
            observation_names: Raw observation names for one building.
            action_names: Action names for one building.

        Returns:
            EnrichmentResult with enriched_names, marker_positions, marker_to_type.
        """
        # Check cache
        cache_key = (tuple(observation_names), tuple(action_names))
        if cache_key == self._cache_key and self._cached_result is not None:
            return self._cached_result

        # Extract device IDs from action names
        device_ids_by_type = _extract_device_ids(action_names, self._ca_config)

        # Classify all features
        classified: List[Tuple[str, Optional[Tuple[str, str, Optional[str]]]]] = []
        for feature_name in observation_names:
            classification = _classify_feature(
                feature_name, self._config, device_ids_by_type
            )
            classified.append((feature_name, classification))

        # Group features by (family, type_name, device_id)
        groups: Dict[Tuple[str, str, Optional[str]], List[str]] = {}
        unclassified: List[str] = []

        for feature_name, classification in classified:
            if classification is None:
                unclassified.append(feature_name)
            else:
                key = classification
                if key not in groups:
                    groups[key] = []
                groups[key].append(feature_name)

        # Build enriched names with markers
        enriched_names: List[str] = []
        marker_positions: Dict[str, List[int]] = {}
        marker_to_type: Dict[str, Tuple[str, str, Optional[str]]] = {}
        insertion_positions: List[int] = []
        marker_values_list: List[float] = []

        ca_base = self._marker_values.get("ca_base", 1000)
        sro_base = self._marker_values.get("sro_base", 2000)
        nfc_marker_value = self._marker_values.get("nfc", 3001)

        ca_counter = 1
        sro_counter = 1

        # Determine order: CAs first (sorted by type then device_id), then SROs, then NFC
        # This ensures marker order = action order

        # Sort CA groups to match action order
        ca_groups = [(k, v) for k, v in groups.items() if k[0] == "ca"]
        # Sort by the order they appear in action_names
        def ca_sort_key(item: Tuple[Tuple[str, str, Optional[str]], List[str]]) -> int:
            key, _ = item
            _, type_name, device_id = key
            action_prefix = self._ca_config.get(type_name, {}).get("action_name", "")
            if device_id:
                action_name = f"{action_prefix}_{device_id}"
            else:
                action_name = action_prefix
            try:
                return action_names.index(action_name)
            except ValueError:
                return 999

        ca_groups.sort(key=ca_sort_key)

        # Add CA groups
        for (family, type_name, device_id), features in ca_groups:
            marker_value = ca_base + ca_counter
            marker_name = f"__marker_{marker_value}__"
            
            insertion_positions.append(len(enriched_names))
            marker_values_list.append(float(marker_value))
            
            enriched_names.append(marker_name)
            marker_positions[marker_name] = [len(enriched_names) - 1]
            marker_to_type[marker_name] = (family, type_name, device_id)
            
            enriched_names.extend(features)
            ca_counter += 1

        # Add SRO groups (in config order)
        for sro_type_name in self._sro_config.keys():
            key = ("sro", sro_type_name, None)
            if key in groups:
                features = groups[key]
                marker_value = sro_base + sro_counter
                marker_name = f"__marker_{marker_value}__"
                
                insertion_positions.append(len(enriched_names))
                marker_values_list.append(float(marker_value))
                
                enriched_names.append(marker_name)
                marker_positions[marker_name] = [len(enriched_names) - 1]
                marker_to_type[marker_name] = key
                
                enriched_names.extend(features)
                sro_counter += 1

        # Add NFC group
        nfc_key = ("nfc", "nfc", None)
        if nfc_key in groups:
            features = groups[nfc_key]
            marker_name = f"__marker_{nfc_marker_value}__"
            
            insertion_positions.append(len(enriched_names))
            marker_values_list.append(float(nfc_marker_value))
            
            enriched_names.append(marker_name)
            marker_positions[marker_name] = [len(enriched_names) - 1]
            marker_to_type[marker_name] = nfc_key
            
            enriched_names.extend(features)

        # Add unclassified features at the end (no marker)
        enriched_names.extend(unclassified)

        # Cache result
        result = EnrichmentResult(
            enriched_names=enriched_names,
            marker_positions=marker_positions,
            marker_to_type=marker_to_type,
        )
        self._cache_key = cache_key
        self._cached_result = result
        self._insertion_positions = insertion_positions
        self._marker_values_list = marker_values_list

        return result
