"""Observation Enricher — injects token-type markers into observation names/values.

This module classifies raw observation features into token groups (CA, SRO, NFC)
and injects marker values that allow the tokenizer to identify token boundaries.

Portable: no dependencies on training-only code. Can be used in both the training
wrapper and the production inference preprocessor.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


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
            # Validate consistency: all None or all present, never mixed
            has_none = None in device_ids
            has_non_none = any(d is not None for d in device_ids)
            
            if has_none and has_non_none:
                single_instance_names = [
                    act for act in action_names 
                    if act == action_prefix
                ]
                multi_instance_names = [
                    act for act in action_names 
                    if act.startswith(action_prefix + "_")
                ]
                raise ValueError(
                    f"Inconsistent device ID naming for CA type '{ca_type_name}': "
                    f"Found both single-instance {single_instance_names} and "
                    f"multi-instance {multi_instance_names} action names. "
                    f"All instances must use consistent naming: either all without "
                    f"suffix OR all with '_<device_id>' suffix."
                )
            
            result[ca_type_name] = device_ids

    return result


def _feature_matches_patterns(feature_name: str, patterns: List[str]) -> bool:
    """Check if feature_name contains any of the patterns as substring.

    Args:
        feature_name: Raw observation feature name.
        patterns: List of pattern substrings to match against.

    Returns:
        True if any pattern is a substring of feature_name, or if all pattern
        tokens appear in order in feature_name tokens (allowing inserted
        device-id tokens such as ``charger_1_1``).
    """
    feature_tokens = [token for token in feature_name.split("_") if token]

    for pattern in patterns:
        if pattern in feature_name:
            return True

        pattern_tokens = [token for token in pattern.split("_") if token]
        if not pattern_tokens:
            continue

        # Ordered-subsequence match to tolerate injected identifiers in names.
        token_idx = 0
        for feature_token in feature_tokens:
            if feature_token == pattern_tokens[token_idx]:
                token_idx += 1
                if token_idx == len(pattern_tokens):
                    return True

    return False


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

        logger.debug(
            "ObservationEnricher initialized with %d CA type(s), %d SRO type(s).",
            len(self._ca_config),
            len(self._sro_config),
        )

    def _classify_and_group_features(
        self,
        observation_names: List[str],
        action_names: List[str],
    ) -> Tuple[Dict[Tuple[str, str, Optional[str]], List[str]], List[str]]:
        """Classify features and group them by token type.
        
        Returns:
            Tuple of (groups dict, unclassified features list).
            Groups are keyed by (family, type_name, device_id).
        """
        device_ids_by_type = _extract_device_ids(action_names, self._ca_config)
        
        groups: Dict[Tuple[str, str, Optional[str]], List[str]] = {}
        unclassified: List[str] = []

        for feature_name in observation_names:
            classification = _classify_feature(
                feature_name, self._config, device_ids_by_type
            )
            if classification is None:
                unclassified.append(feature_name)
            else:
                if classification not in groups:
                    groups[classification] = []
                groups[classification].append(feature_name)

        return groups, unclassified

    def _sort_ca_groups_by_action_order(
        self,
        groups: Dict[Tuple[str, str, Optional[str]], List[str]],
        action_names: List[str],
    ) -> List[Tuple[Tuple[str, str, Optional[str]], List[str]]]:
        """Sort CA groups to match action name ordering."""
        ca_groups = [(k, v) for k, v in groups.items() if k[0] == "ca"]
        
        def sort_key(item: Tuple[Tuple[str, str, Optional[str]], List[str]]) -> int:
            key, _ = item
            _, type_name, device_id = key
            action_prefix = self._ca_config.get(type_name, {}).get("action_name", "")
            action_name = f"{action_prefix}_{device_id}" if device_id else action_prefix
            try:
                return action_names.index(action_name)
            except ValueError:
                return 999

        ca_groups.sort(key=sort_key)
        return ca_groups

    def _build_enriched_output(
        self,
        groups: Dict[Tuple[str, str, Optional[str]], List[str]],
        unclassified: List[str],
        ca_groups: List[Tuple[Tuple[str, str, Optional[str]], List[str]]],
    ) -> Tuple[List[str], Dict[str, List[int]], Dict[str, Tuple[str, str, Optional[str]]]]:
        """Build enriched names with markers inserted.
        
        Returns:
            Tuple of (enriched_names, marker_positions, marker_to_type).
        """
        enriched_names: List[str] = []
        marker_positions: Dict[str, List[int]] = {}
        marker_to_type: Dict[str, Tuple[str, str, Optional[str]]] = {}

        ca_base = self._marker_values.get("ca_base", 1000)
        sro_base = self._marker_values.get("sro_base", 2000)
        nfc_marker_value = self._marker_values.get("nfc", 3001)

        # Add unclassified features before any marker
        if unclassified:
            enriched_names.extend(unclassified)
            logger.warning(
                "ObservationEnricher found %d unclassified feature(s): %s",
                len(unclassified),
                ", ".join(unclassified),
            )

        # Add CA groups
        for idx, ((family, type_name, device_id), features) in enumerate(ca_groups, start=1):
            marker_value = ca_base + idx
            marker_name = f"__marker_{marker_value}__"
            
            enriched_names.append(marker_name)
            marker_positions[marker_name] = [len(enriched_names) - 1]
            marker_to_type[marker_name] = (family, type_name, device_id)
            enriched_names.extend(features)

        # Add SRO groups (in config order)
        sro_counter = 1
        for sro_type_name in self._sro_config.keys():
            key = ("sro", sro_type_name, None)
            if key in groups:
                features = groups[key]
                marker_value = sro_base + sro_counter
                marker_name = f"__marker_{marker_value}__"
                
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
            
            enriched_names.append(marker_name)
            marker_positions[marker_name] = [len(enriched_names) - 1]
            marker_to_type[marker_name] = nfc_key
            enriched_names.extend(features)

        return enriched_names, marker_positions, marker_to_type

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
            logger.debug("ObservationEnricher cache hit for current topology.")
            return self._cached_result

        logger.debug(
            "Enriching observations for topology (observations=%d, actions=%d).",
            len(observation_names),
            len(action_names),
        )

        # Classify and group features
        groups, unclassified = self._classify_and_group_features(
            observation_names, action_names
        )
        
        # Sort CA groups by action order
        ca_groups = self._sort_ca_groups_by_action_order(groups, action_names)
        
        # Build enriched output
        enriched_names, marker_positions, marker_to_type = self._build_enriched_output(
            groups, unclassified, ca_groups
        )

        logger.debug(
            "Observation enrichment complete (raw=%d, enriched=%d, markers=%d).",
            len(observation_names),
            len(enriched_names),
            len(marker_to_type),
        )

        # Cache result
        result = EnrichmentResult(
            enriched_names=enriched_names,
            marker_positions=marker_positions,
            marker_to_type=marker_to_type,
        )
        self._cache_key = cache_key
        self._cached_result = result

        return result

    def enrich_values(self, observation_values: List[float]) -> List[float]:
        """Inject marker values at cached positions.

        Must be called after enrich_names() to populate the cache.

        Args:
            observation_values: Raw observation values (same length as observation_names
                passed to enrich_names()).

        Returns:
            Enriched values list with marker values inserted.

        Raises:
            RuntimeError: If enrich_names() has not been called yet.
        """
        if self._cached_result is None or self._cache_key is None:
            raise RuntimeError("enrich_names() must be called before enrich_values()")

        # Build mapping from feature name to value
        observation_names = list(self._cache_key[0])  # Original observation names
        value_map = dict(zip(observation_names, observation_values))

        enriched_values: List[float] = []

        for name in self._cached_result.enriched_names:
            if name.startswith("__marker_") and name.endswith("__"):
                # Extract marker value from name: "__marker_1001__" -> 1001.0
                marker_str = name[9:-2]  # Strip "__marker_" prefix and "__" suffix
                marker_value = float(marker_str)
                enriched_values.append(marker_value)
            else:
                # Regular feature - look up value by name
                enriched_values.append(value_map[name])

        logger.debug(
            "Enriched observation values (raw=%d, enriched=%d).",
            len(observation_values),
            len(enriched_values),
        )

        return enriched_values

    def topology_changed(
        self,
        observation_names: List[str],
        action_names: List[str],
    ) -> bool:
        """Check if topology differs from cached state.

        Args:
            observation_names: Current observation names.
            action_names: Current action names.

        Returns:
            True if no cache exists or if topology has changed.
            False if topology matches cached state.
        """
        if self._cache_key is None:
            logger.debug("Topology check reports changed because cache is empty.")
            return True

        current_key = (tuple(observation_names), tuple(action_names))
        changed = current_key != self._cache_key
        if changed:
            logger.info(
                "Topology change detected (observations=%d, actions=%d).",
                len(observation_names),
                len(action_names),
            )
        return changed
