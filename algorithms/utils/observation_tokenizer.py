"""Observation Tokenizer — converts flat post-encoded vectors into typed token embeddings.

Given a building's observation names (raw or enriched with __tkn_*__ markers),
action names, encoder config, and tokenizer config, the tokenizer:

1. If enriched names: Parses __tkn_*__ markers to identify token groups.
   If raw names: Uses heuristic classification (legacy mode).
2. Uses the encoder config to determine post-encoding slice indices.
3. Creates per-type Linear projections.
4. At forward time, slices the encoded vector and projects each group to ``d_model``.

Marker-based mode (DIN plan)
----------------------------
When observation names contain ``__tkn_*__`` markers (injected by ObservationEnricher),
the tokenizer uses marker positions to identify token boundaries. This eliminates
heuristic-based classification and ensures consistency with the wrapper's encoding.

Marker format:
- ``__tkn_ca_{type}__`` — single-instance CA (e.g., ``__tkn_ca_battery__``)
- ``__tkn_ca_{type}__{device_id}__`` — multi-instance CA (e.g., ``__tkn_ca_ev_charger__charger_1_1__``)
- ``__tkn_sro_{type}__`` — SRO group (e.g., ``__tkn_sro_temporal__``)
- ``__tkn_nfc__`` — NFC/RL token

Legacy mode (CityLearn naming convention)
-----------------------------------------
CityLearn inserts device IDs **in the middle** of feature names, not as a suffix.
For example, a charger named ``charger_1_1`` produces observation names like:

- ``electric_vehicle_charger_charger_1_1_connected_state``
- ``connected_electric_vehicle_at_charger_charger_1_1_soc``

The device ID is extracted from **action names** (e.g., ``electric_vehicle_storage_charger_1_1``)
by stripping the configured ``action_name`` prefix (``electric_vehicle_storage``).
Features are then assigned to instances by checking whether the device ID appears
anywhere in the observation name.
"""

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger




# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class TokenizedObservation:
    """Output of the observation tokenizer.

    Attributes:
        ca_tokens:  [batch, N_ca, d_model] — one token per controllable-asset instance.
        sro_tokens: [batch, N_sro, d_model] — one token per SRO group (with >0 dims).
        rl_token:   [batch, 1, d_model] — residual-load token.
        ca_types:   Type name per CA token position (for action ordering).
        n_ca:       Number of CA tokens.
    """

    ca_tokens: torch.Tensor
    sro_tokens: torch.Tensor
    rl_token: torch.Tensor
    ca_types: List[str]
    n_ca: int


@dataclass
class TokenGroup:
    """Parsed token group from marker-based observation names.

    Attributes:
        family: Token family ("ca", "sro", or "nfc").
        type_name: Type within family (e.g., "battery", "temporal"). Empty for nfc.
        device_id: Device ID for multi-instance CAs (e.g., "charger_1_1"). None otherwise.
        feature_names: List of feature names belonging to this group.
    """

    family: str
    type_name: str
    device_id: Optional[str]
    feature_names: List[str] = field(default_factory=list)


class EncoderSlice:
    """Post-encoding position of a single raw observation feature.

    Attributes:
        start_idx: Inclusive start index in the post-encoded flat vector.
        end_idx: Exclusive end index in the post-encoded flat vector.
        n_dims: Number of post-encoding dimensions (end_idx - start_idx).
    """

    __slots__ = ("start_idx", "end_idx", "n_dims")

    def __init__(self, start_idx: int, end_idx: int, n_dims: int) -> None:
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.n_dims = n_dims

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EncoderSlice):
            return NotImplemented
        return (
            self.start_idx == other.start_idx
            and self.end_idx == other.end_idx
            and self.n_dims == other.n_dims
        )

    def __repr__(self) -> str:
        return f"EncoderSlice(start_idx={self.start_idx}, end_idx={self.end_idx}, n_dims={self.n_dims})"


# ---------------------------------------------------------------------------
# Internal helpers — classification (legacy mode)
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
                device_id = act_name[len(action_prefix) + 1:]
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
# Internal helpers — encoder dimension computation
# ---------------------------------------------------------------------------


def _matches_rule(name: str, match_spec: Dict[str, Any]) -> bool:
    """Check whether *name* satisfies the match specification of an encoder rule.

    Supports the same match types as ``wrapper_citylearn._matches_rule``:
    ``equals``, ``contains``, ``prefixes``, ``suffixes``, and ``default``.
    """
    equals = match_spec.get("equals")
    if equals is not None and name in equals:
        return True

    contains = match_spec.get("contains")
    if contains is not None and any(token in name for token in contains):
        return True

    prefixes = match_spec.get("prefixes")
    if prefixes is not None and any(name.startswith(p) for p in prefixes):
        return True

    suffixes = match_spec.get("suffixes")
    if suffixes is not None and any(name.endswith(s) for s in suffixes):
        return True

    return bool(match_spec.get("default", False))


def _compute_encoded_dims(encoder_spec: Dict[str, Any]) -> int:
    """Return the number of post-encoding dimensions for an encoder spec.

    Rules
    -----
    * ``PeriodicNormalization`` → **2** (sin, cos)
    * ``OnehotEncoding``        → **len(classes)**
    * ``RemoveFeature``         → **0**
    * Everything else (``NoNormalization``, ``Normalize``,
      ``NormalizeWithMissing``) → **1**
    """
    encoder_type = encoder_spec.get("type")

    if encoder_type == "RemoveFeature":
        return 0

    if encoder_type == "PeriodicNormalization":
        return 2

    if encoder_type == "OnehotEncoding":
        classes = encoder_spec.get("params", {}).get("classes", [])
        return len(classes)

    # NoNormalization, Normalize, NormalizeWithMissing → 1 dim
    return 1


# ---------------------------------------------------------------------------
# Internal helpers — marker parsing (DIN plan)
# ---------------------------------------------------------------------------


def _parse_markers(observation_names: List[str]) -> List[TokenGroup]:
    """Parse __tkn_*__ markers from enriched observation names.

    Returns ordered list of TokenGroup, one per marker encountered.
    Features before the first marker (if any) are unassigned.

    Marker format:
        - __tkn_{family}_{type}__ → family="ca"/"sro", type_name="{type}", device_id=None
        - __tkn_{family}_{type}__{device_id}__ → family="ca", type_name="{type}", device_id="{device_id}"
        - __tkn_nfc__ → family="nfc", type_name="", device_id=None

    Examples:
        - __tkn_ca_battery__ → family="ca", type_name="battery", device_id=None
        - __tkn_ca_ev_charger__charger_1_1__ → family="ca", type_name="ev_charger", device_id="charger_1_1"
        - __tkn_sro_temporal__ → family="sro", type_name="temporal", device_id=None
        - __tkn_nfc__ → family="nfc", type_name="", device_id=None
    """
    groups: List[TokenGroup] = []
    current_group: Optional[TokenGroup] = None

    for name in observation_names:
        if name.startswith("__tkn_") and name.endswith("__"):
            # Parse marker: __tkn_{family}_{type}__ or __tkn_{family}_{type}__{device_id}__
            inner = name[6:-2]  # strip __tkn_ and __

            # Check for device_id (indicated by double underscore in the middle)
            if "__" in inner:
                # Has device_id: e.g., "ca_ev_charger__charger_1_1"
                type_part, device_id = inner.rsplit("__", 1)
                parts = type_part.split("_", 1)
                family = parts[0]
                type_name = parts[1] if len(parts) > 1 else ""
            else:
                # No device_id: e.g., "ca_battery" or "sro_temporal" or "nfc"
                parts = inner.split("_", 1)
                family = parts[0]
                type_name = parts[1] if len(parts) > 1 else ""
                device_id = None

            current_group = TokenGroup(
                family=family,
                type_name=type_name,
                device_id=device_id,
                feature_names=[],
            )
            groups.append(current_group)
        elif current_group is not None:
            current_group.feature_names.append(name)
        else:
            # Feature before any marker — unassigned (skip)
            pass

    return groups


def _build_encoded_dims_map(
    observation_names: List[str],
    encoder_config: Dict[str, Any],
) -> OrderedDict[str, EncoderSlice]:
    """Map each feature name to (start_idx, end_idx, n_dims) in the encoded vector.

    For __tkn_*__ markers, uses NoNormalization → 1 dim.
    For regular features, matches against encoder rules.

    Parameters
    ----------
    observation_names:
        Feature names (may include __tkn_*__ markers).
    encoder_config:
        The full encoder configuration dict (must contain a ``"rules"`` key).

    Returns
    -------
    OrderedDict[str, EncoderSlice]
        Maps each name to its post-encoding slice.

    Raises
    ------
    ValueError
        If encoder config has no rules or if no rule matches a feature name.
    """
    rules = encoder_config.get("rules", [])
    has_non_marker = any(
        not (name.startswith("__tkn_") and name.endswith("__"))
        for name in observation_names
    )
    if has_non_marker and not rules:
        raise ValueError("Encoder config must contain at least one rule")
    
    result: OrderedDict[str, EncoderSlice] = OrderedDict()
    current_idx = 0

    for name in observation_names:
        if name.startswith("__tkn_") and name.endswith("__"):
            n_dims = 1  # NoNormalization → 1 dim
        else:
            rule = next(
                (r for r in rules if _matches_rule(name, r.get("match", {}))),
                None,
            )
            if rule is None:
                raise ValueError(f"No encoder rule matches: {name!r}")
            n_dims = _compute_encoded_dims(rule.get("encoder", {}))

        result[name] = EncoderSlice(current_idx, current_idx + n_dims, n_dims)
        current_idx += n_dims

    return result


def _has_markers(observation_names: List[str]) -> bool:
    """Check if observation names contain __tkn_*__ markers."""
    return any(
        name.startswith("__tkn_") and name.endswith("__")
        for name in observation_names
    )


# ---------------------------------------------------------------------------
# ObservationTokenizer
# ---------------------------------------------------------------------------


class ObservationTokenizer(nn.Module):
    """Converts a flat post-encoded observation vector into typed token embeddings.

    Supports both marker-based (enriched names) and heuristic-based (raw names) modes.
    When observation names contain __tkn_*__ markers, uses marker parsing.
    Otherwise, falls back to heuristic classification.
    """

    def __init__(
        self,
        observation_names: List[str],
        action_names: List[str],
        encoder_config: Dict[str, Any],
        tokenizer_config: Dict[str, Any],
        d_model: int,
        global_ca_type_dims: Optional[Dict[str, int]] = None,
        global_sro_type_dims: Optional[Dict[str, int]] = None,
        max_rl_input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.observation_names = observation_names
        self.action_names = action_names

        ca_config: Dict[str, Dict[str, Any]] = tokenizer_config.get("ca_types", {})
        sro_config: Dict[str, Dict[str, Any]] = tokenizer_config.get("sro_types", {})
        rl_config: Dict[str, Any] = tokenizer_config.get("rl", {})

        # --- Choose classification mode based on marker presence ---
        # Both modes use _build_encoded_dims_map for index computation
        self._index_map = _build_encoded_dims_map(observation_names, encoder_config)
        
        if _has_markers(observation_names):
            # Marker-based mode: parse markers for token group assignment
            self._init_from_markers(rl_config)
        else:
            # Heuristic-based mode: infer token groups from feature patterns
            self._init_from_heuristics(ca_config, sro_config, rl_config)

        # --- Build projections ---
        self._build_projections(
            ca_config, global_ca_type_dims, global_sro_type_dims, max_rl_input_dim,
        )

        # --- Build action-CA mapping ---
        self._action_ca_map: List[int] = self._build_action_ca_map(action_names, ca_config)

        # --- Register index buffers ---
        self._register_buffers()

    def _init_from_markers(self, rl_config: Dict[str, Any]) -> None:
        """Initialize token groups from marker-based observation names."""
        token_groups = _parse_markers(self.observation_names)

        # CA instances
        self._ca_instances: List[Tuple[str, Optional[str], List[int]]] = []
        self._ca_type_names: List[str] = []

        # SRO groups
        self._sro_groups: List[Tuple[str, List[int]]] = []
        self._sro_group_dims: Dict[str, int] = {}

        # RL indices
        self._rl_demand_indices: List[int] = []
        self._rl_generation_indices: List[int] = []
        self._rl_extra_indices: List[int] = []

        # CA type dims for projection sizing
        self._ca_type_dims: Dict[str, int] = {}

        rl_demand_feature = rl_config.get("demand_feature")
        rl_generation_features: List[str] = rl_config.get("generation_features", [])
        rl_extra_features: List[str] = rl_config.get("extra_features", [])

        for group in token_groups:
            if group.family == "ca" and group.type_name:
                # Compute indices for this CA instance
                indices: List[int] = []
                for n in group.feature_names:
                    s = self._index_map.get(n)
                    if s and s.n_dims > 0:
                        indices.extend(range(s.start_idx, s.end_idx))

                if indices:
                    self._ca_instances.append((group.type_name, group.device_id, indices))
                    self._ca_type_names.append(group.type_name)

                    # Track dims per type (first instance defines the dims)
                    if group.type_name not in self._ca_type_dims:
                        self._ca_type_dims[group.type_name] = len(indices)

            elif group.family == "sro" and group.type_name:
                indices = []
                for n in group.feature_names:
                    s = self._index_map.get(n)
                    if s and s.n_dims > 0:
                        indices.extend(range(s.start_idx, s.end_idx))

                if indices:
                    self._sro_groups.append((group.type_name, indices))
                    self._sro_group_dims[group.type_name] = len(indices)

            elif group.family == "nfc":
                # Parse RL features into demand/generation/extra
                for n in group.feature_names:
                    s = self._index_map.get(n)
                    if not s or s.n_dims == 0:
                        continue

                    idxs = list(range(s.start_idx, s.end_idx))

                    if rl_demand_feature and rl_demand_feature in n:
                        self._rl_demand_indices.extend(idxs)
                    elif any(gen in n for gen in rl_generation_features):
                        self._rl_generation_indices.extend(idxs)
                    elif any(extra in n for extra in rl_extra_features):
                        self._rl_extra_indices.extend(idxs)
                    else:
                        # Default to extra if not matched
                        self._rl_extra_indices.extend(idxs)

    def _init_from_heuristics(
        self,
        ca_config: Dict[str, Dict[str, Any]],
        sro_config: Dict[str, Dict[str, Any]],
        rl_config: Dict[str, Any],
    ) -> None:
        """Initialize token groups using heuristic classification (legacy mode)."""
        assigned: set[str] = set()

        # CA classification
        device_ids_by_type = _extract_device_ids(self.action_names, ca_config)
        ca_groups: Dict[str, Dict[Optional[str], List[str]]] = {}

        for ca_type_name, ca_spec in ca_config.items():
            ca_feature_patterns: List[str] = ca_spec.get("features", [])
            device_ids = device_ids_by_type.get(ca_type_name, [])

            if not device_ids:
                continue

            instance_map: Dict[Optional[str], List[str]] = defaultdict(list)

            for raw_name in self.observation_names:
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

            if instance_map:
                ca_groups[ca_type_name] = dict(instance_map)

        # SRO classification
        sro_groups: Dict[str, List[str]] = {}

        for sro_type_name, sro_spec in sro_config.items():
            sro_features: List[str] = sro_spec.get("features", [])
            matched_names: List[str] = []

            for raw_name in self.observation_names:
                if raw_name in assigned:
                    continue
                for sro_feat in sro_features:
                    if sro_feat in raw_name:
                        matched_names.append(raw_name)
                        assigned.add(raw_name)
                        break

            if matched_names:
                sro_groups[sro_type_name] = matched_names

        # RL classification
        rl_demand_feature = rl_config.get("demand_feature")
        rl_generation_features: List[str] = rl_config.get("generation_features", [])
        rl_extra_features: List[str] = rl_config.get("extra_features", [])
        rl_names: List[str] = []

        for raw_name in self.observation_names:
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

        # Warn about unmatched
        unmatched = [n for n in self.observation_names if n not in assigned]
        for name in unmatched:
            logger.warning("Tokenizer: unmatched observation feature: {}", name)

        # --- Build internal structures ---
        self._ca_instances = []
        self._ca_type_names = []

        for ca_type_name, instance_map in ca_groups.items():
            for device_id in sorted(instance_map.keys(), key=lambda s: s or ""):
                names_for_instance = instance_map[device_id]
                indices: List[int] = []
                for n in names_for_instance:
                    s = self._index_map[n]
                    if s.n_dims > 0:
                        indices.extend(range(s.start_idx, s.end_idx))
                if indices:
                    self._ca_instances.append((ca_type_name, device_id, indices))
                    self._ca_type_names.append(ca_type_name)

        self._ca_type_dims: Dict[str, int] = {}
        for ca_type_name, instance_map in ca_groups.items():
            first_device_id = next(iter(instance_map))
            first_names = instance_map[first_device_id]
            dims = sum(self._index_map[n].n_dims for n in first_names)
            if dims > 0:
                self._ca_type_dims[ca_type_name] = dims

        self._sro_groups = []
        self._sro_group_dims: Dict[str, int] = {}

        for sro_type_name, names in sro_groups.items():
            indices: List[int] = []
            for n in names:
                s = self._index_map[n]
                if s.n_dims > 0:
                    indices.extend(range(s.start_idx, s.end_idx))
            if indices:
                self._sro_groups.append((sro_type_name, indices))
                self._sro_group_dims[sro_type_name] = len(indices)

        self._rl_demand_indices = []
        self._rl_generation_indices = []
        self._rl_extra_indices = []

        for n in rl_names:
            s = self._index_map[n]
            if s.n_dims == 0:
                continue
            idxs = list(range(s.start_idx, s.end_idx))
            if rl_demand_feature and rl_demand_feature in n:
                self._rl_demand_indices.extend(idxs)
            elif any(gen in n for gen in rl_generation_features):
                self._rl_generation_indices.extend(idxs)
            else:
                self._rl_extra_indices.extend(idxs)

    def _build_projections(
        self,
        ca_config: Dict[str, Dict[str, Any]],
        global_ca_type_dims: Optional[Dict[str, int]],
        global_sro_type_dims: Optional[Dict[str, int]],
        max_rl_input_dim: Optional[int],
    ) -> None:
        """Build Linear projections for CA, SRO, and RL tokens."""
        # CA projections
        final_ca_type_dims = (
            global_ca_type_dims if global_ca_type_dims is not None else self._ca_type_dims
        )

        self.ca_projections = nn.ModuleDict()
        for ca_type_name, dims in final_ca_type_dims.items():
            self.ca_projections[ca_type_name] = nn.Linear(dims, self.d_model)

        # SRO projections
        final_sro_type_dims = {}
        if global_sro_type_dims is not None:
            final_sro_type_dims.update(global_sro_type_dims)
        for sro_type_name, dims in self._sro_group_dims.items():
            if sro_type_name not in final_sro_type_dims:
                final_sro_type_dims[sro_type_name] = dims

        self.sro_projections = nn.ModuleDict()
        for sro_type_name, dims in final_sro_type_dims.items():
            self.sro_projections[sro_type_name] = nn.Linear(dims, self.d_model)

        # RL projection
        rl_input_dim = 0
        if bool(self._rl_demand_indices) or bool(self._rl_generation_indices):
            rl_input_dim += 1  # residual
        rl_input_dim += len(self._rl_extra_indices)

        final_rl_input_dim = (
            max_rl_input_dim if max_rl_input_dim is not None else rl_input_dim
        )
        self.rl_projection: Optional[nn.Linear] = None
        if final_rl_input_dim > 0:
            self.rl_projection = nn.Linear(final_rl_input_dim, self.d_model)

        self._rl_has_residual = bool(self._rl_demand_indices) or bool(self._rl_generation_indices)
        self._local_rl_input_dim = rl_input_dim

    def _register_buffers(self) -> None:
        """Register index tensors as buffers for device handling."""
        for i, (ca_type, device_id, indices) in enumerate(self._ca_instances):
            self.register_buffer(f"_ca_idx_{i}", torch.tensor(indices, dtype=torch.long))
        for i, (sro_type, indices) in enumerate(self._sro_groups):
            self.register_buffer(f"_sro_idx_{i}", torch.tensor(indices, dtype=torch.long))
        if self._rl_demand_indices:
            self.register_buffer("_rl_demand_idx", torch.tensor(self._rl_demand_indices, dtype=torch.long))
        if self._rl_generation_indices:
            self.register_buffer("_rl_gen_idx", torch.tensor(self._rl_generation_indices, dtype=torch.long))
        if self._rl_extra_indices:
            self.register_buffer("_rl_extra_idx", torch.tensor(self._rl_extra_indices, dtype=torch.long))

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    @property
    def n_ca(self) -> int:
        """Number of CA tokens this tokenizer produces."""
        return len(self._ca_instances)

    @property
    def n_sro(self) -> int:
        """Number of SRO tokens this tokenizer produces."""
        return len(self._sro_groups)

    @property
    def ca_types(self) -> List[str]:
        """Type name per CA token position."""
        return list(self._ca_type_names)

    @property
    def action_ca_map(self) -> List[int]:
        """action_index → ca_token_index mapping."""
        return list(self._action_ca_map)

    @property
    def total_encoded_dims(self) -> int:
        """Total post-encoding observation dimension expected by forward()."""
        if not self._index_map:
            return 0
        last = list(self._index_map.values())[-1]
        return last.end_idx

    # --------------------------------------------------------------------- #
    # Action ↔ CA mapping
    # --------------------------------------------------------------------- #

    def _build_action_ca_map(
        self,
        action_names: List[str],
        ca_config: Dict[str, Dict[str, Any]],
    ) -> List[int]:
        """Map each action index to the corresponding CA token index.

        Uses the same device-ID extraction logic as the classifier: strip the
        ``action_name`` prefix from each action name to get the device ID,
        then find the CA instance with that device ID.
        """

        action_to_ca: List[int] = []

        for act_name in action_names:
            matched = False
            for ca_idx, (ca_type, device_id, _indices) in enumerate(self._ca_instances):
                ca_action_name = ca_config[ca_type].get("action_name", "")
                if not ca_action_name:
                    continue

                if device_id is None:
                    # Single-instance: exact match on action name
                    if act_name == ca_action_name:
                        action_to_ca.append(ca_idx)
                        matched = True
                        break
                else:
                    # Multi-instance: action_name + "_" + device_id
                    expected_action = f"{ca_action_name}_{device_id}"
                    if act_name == expected_action:
                        action_to_ca.append(ca_idx)
                        matched = True
                        break

            if not matched:
                logger.warning(
                    "Tokenizer: action '{}' does not map to any CA token", act_name,
                )

        return action_to_ca

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #

    def forward(self, encoded_obs: torch.Tensor) -> TokenizedObservation:
        """Convert a flat post-encoded vector into typed token embeddings.

        Parameters
        ----------
        encoded_obs : Tensor[batch, obs_dim]
            Flat post-encoded observation from the wrapper.

        Returns
        -------
        TokenizedObservation
        """

        batch = encoded_obs.shape[0] if encoded_obs.dim() > 1 else 1
        if encoded_obs.dim() == 1:
            encoded_obs = encoded_obs.unsqueeze(0)

        # -- CA tokens --------------------------------------------------------
        ca_token_list: List[torch.Tensor] = []
        for i, (ca_type, device_id, _indices) in enumerate(self._ca_instances):
            idx_buf: torch.Tensor = getattr(self, f"_ca_idx_{i}")
            features = encoded_obs[:, idx_buf]  # [batch, n_features]
            projection = self.ca_projections[ca_type]
            token = projection(features)  # [batch, d_model]
            ca_token_list.append(token)

        if ca_token_list:
            ca_tokens = torch.stack(ca_token_list, dim=1)  # [batch, N_ca, d_model]
        else:
            ca_tokens = torch.zeros(batch, 0, self.d_model, device=encoded_obs.device)

        # -- SRO tokens -------------------------------------------------------
        sro_token_list: List[torch.Tensor] = []
        for i, (sro_type, _indices) in enumerate(self._sro_groups):
            idx_buf = getattr(self, f"_sro_idx_{i}")
            features = encoded_obs[:, idx_buf]  # [batch, n_features]
            projection = self.sro_projections[sro_type]
            token = projection(features)  # [batch, d_model]
            sro_token_list.append(token)

        if sro_token_list:
            sro_tokens = torch.stack(sro_token_list, dim=1)  # [batch, N_sro, d_model]
        else:
            sro_tokens = torch.zeros(batch, 0, self.d_model, device=encoded_obs.device)

        # -- RL token ---------------------------------------------------------
        if self.rl_projection is not None:
            parts: List[torch.Tensor] = []

            # Residual = demand - generation (if either is present)
            if self._rl_has_residual:
                demand = torch.zeros(batch, 1, device=encoded_obs.device)
                generation = torch.zeros(batch, 1, device=encoded_obs.device)

                if self._rl_demand_indices:
                    demand_idx = getattr(self, "_rl_demand_idx")
                    demand = encoded_obs[:, demand_idx].sum(dim=-1, keepdim=True)
                if self._rl_generation_indices:
                    gen_idx = getattr(self, "_rl_gen_idx")
                    generation = encoded_obs[:, gen_idx].sum(dim=-1, keepdim=True)

                parts.append(demand - generation)  # [batch, 1]

            # Extra features (e.g. net_electricity_consumption)
            if self._rl_extra_indices:
                extra_idx = getattr(self, "_rl_extra_idx")
                parts.append(encoded_obs[:, extra_idx])  # [batch, n_extra]

            rl_input = torch.cat(parts, dim=-1)  # [batch, local_rl_input_dim]
            
            # Pad to max_rl_input_dim if necessary
            if rl_input.shape[-1] < self.rl_projection.in_features:
                padding_size = self.rl_projection.in_features - rl_input.shape[-1]
                padding = torch.zeros(batch, padding_size, device=encoded_obs.device)
                rl_input = torch.cat([rl_input, padding], dim=-1)
            
            rl_token = self.rl_projection(rl_input).unsqueeze(1)  # [batch, 1, d_model]
        else:
            rl_token = torch.zeros(batch, 1, self.d_model, device=encoded_obs.device)

        return TokenizedObservation(
            ca_tokens=ca_tokens,
            sro_tokens=sro_tokens,
            rl_token=rl_token,
            ca_types=self._ca_type_names,
            n_ca=len(self._ca_instances),
        )
