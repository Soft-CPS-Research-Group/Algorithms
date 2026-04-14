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
    """Placeholder stub for future implementation.
    
    This class will be implemented in subsequent tasks.
    """
    pass
