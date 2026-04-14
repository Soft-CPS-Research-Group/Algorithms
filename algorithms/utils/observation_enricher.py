"""Observation Enricher — injects token-type markers into observation names/values.

This module classifies raw observation features into token groups (CA, SRO, NFC)
and injects marker values that allow the tokenizer to identify token boundaries
without heuristic-based classification.

Portable: no dependencies on training-only code. Can be used in both the training
wrapper and the production inference preprocessor.

No PyTorch/NumPy dependencies — pure Python with stdlib + typing only.
"""

from __future__ import annotations

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


class ObservationEnricher:
    """Placeholder stub for future implementation.
    
    This class will be implemented in subsequent tasks.
    """
    pass
