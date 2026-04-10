"""Observation Tokenizer — converts flat post-encoded vectors into typed token embeddings.

Given a building's raw observation names, action names, encoder config, and tokenizer
config, the tokenizer:

1. Detects CA instances from action names (device ID extraction).
2. Classifies each raw feature into a token group (CA instance, SRO group, or RL).
3. Uses the encoder index map to determine post-encoding slice indices.
4. Creates per-type Linear projections.
5. At forward time, slices the encoded vector and projects each group to ``d_model``.

CityLearn naming convention
---------------------------
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

from algorithms.utils.encoder_index_map import (
    EncoderSlice,
    build_encoder_index_map,
)


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


# ---------------------------------------------------------------------------
# Internal helpers
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
# ObservationTokenizer
# ---------------------------------------------------------------------------


class ObservationTokenizer(nn.Module):
    """Converts a flat post-encoded observation vector into typed token embeddings."""

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

        # --- Step 1: Build encoder index map ---------------------------------
        self._index_map: OrderedDict[str, EncoderSlice] = build_encoder_index_map(
            observation_names, encoder_config,
        )

        # --- Step 2: Classify features into token groups ---------------------
        ca_config: Dict[str, Dict[str, Any]] = tokenizer_config.get("ca_types", {})
        sro_config: Dict[str, Dict[str, Any]] = tokenizer_config.get("sro_types", {})
        rl_config: Dict[str, Any] = tokenizer_config.get("rl", {})

        # Track which names have been assigned
        assigned: set[str] = set()

        # -- CA classification (action-based instance detection) ---------------
        # Step 2a: Extract device IDs from action names
        device_ids_by_type: Dict[str, List[Optional[str]]] = _extract_device_ids(
            action_names, ca_config,
        )

        # Step 2b: Classify features into CA instances
        # {ca_type_name: {device_id_or_None: [raw_name, ...]}}
        ca_groups: Dict[str, Dict[Optional[str], List[str]]] = {}

        for ca_type_name, ca_spec in ca_config.items():
            ca_feature_patterns: List[str] = ca_spec.get("features", [])
            device_ids = device_ids_by_type.get(ca_type_name, [])

            if not device_ids:
                # No actions for this CA type in this building → skip
                continue

            instance_map: Dict[Optional[str], List[str]] = defaultdict(list)

            for raw_name in observation_names:
                if raw_name in assigned:
                    continue

                # Check if this feature matches any of the CA type's patterns
                if not _feature_matches_ca_type(raw_name, ca_feature_patterns):
                    continue

                # Determine which instance this feature belongs to
                if len(device_ids) == 1 and device_ids[0] is None:
                    # Single-instance: no device ID → assign directly
                    instance_map[None].append(raw_name)
                    assigned.add(raw_name)
                else:
                    # Multi-instance: find which device ID is in the name
                    for dev_id in device_ids:
                        if dev_id is not None and _contains_device_id(raw_name, dev_id):
                            instance_map[dev_id].append(raw_name)
                            assigned.add(raw_name)
                            break
                    # If no device ID matched, the feature might be shared or
                    # mismatched — it stays unassigned

            if instance_map:
                ca_groups[ca_type_name] = dict(instance_map)

        # -- SRO classification -----------------------------------------------
        # {sro_type_name: [raw_name, ...]}
        sro_groups: Dict[str, List[str]] = {}

        for sro_type_name, sro_spec in sro_config.items():
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

        # -- RL classification ------------------------------------------------
        rl_demand_feature: Optional[str] = rl_config.get("demand_feature")
        rl_generation_features: List[str] = rl_config.get("generation_features", [])
        rl_extra_features: List[str] = rl_config.get("extra_features", [])
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

        # -- Warn about unmatched features ------------------------------------
        unmatched = [n for n in observation_names if n not in assigned]
        for name in unmatched:
            logger.warning("Tokenizer: unmatched observation feature: {}", name)

        # --- Step 3: Resolve post-encoding slices per group ------------------

        # CA instances — flatten to ordered list [(ca_type, device_id, indices), ...]
        self._ca_instances: List[Tuple[str, Optional[str], List[int]]] = []
        self._ca_type_names: List[str] = []

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

        # Per-CA-type input dims (for shared projection)
        ca_type_dims: Dict[str, int] = {}
        for ca_type_name, instance_map in ca_groups.items():
            # Use the first instance to determine dims (all instances of the
            # same type have the same feature set → same dims).
            first_device_id = next(iter(instance_map))
            first_names = instance_map[first_device_id]
            dims = sum(self._index_map[n].n_dims for n in first_names)
            if dims > 0:
                ca_type_dims[ca_type_name] = dims

        # SRO groups — skip groups with 0 total dims
        self._sro_groups: List[Tuple[str, List[int]]] = []
        sro_group_dims: Dict[str, int] = {}

        for sro_type_name, names in sro_groups.items():
            indices: List[int] = []
            for n in names:
                s = self._index_map[n]
                if s.n_dims > 0:
                    indices.extend(range(s.start_idx, s.end_idx))
            if indices:
                self._sro_groups.append((sro_type_name, indices))
                sro_group_dims[sro_type_name] = len(indices)

        # RL group
        self._rl_demand_indices: List[int] = []
        self._rl_generation_indices: List[int] = []
        self._rl_extra_indices: List[int] = []

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

        rl_has_features = (
            bool(self._rl_demand_indices)
            or bool(self._rl_generation_indices)
            or bool(self._rl_extra_indices)
        )
        # RL token input dim: residual (1 scalar from demand - generation)
        # plus any extra features (e.g. net_electricity_consumption).
        rl_input_dim = 0
        if bool(self._rl_demand_indices) or bool(self._rl_generation_indices):
            rl_input_dim += 1  # residual = demand - generation
        rl_input_dim += len(self._rl_extra_indices)

        # --- Step 4: Create per-type projections -----------------------------
        # CA projections: use global dims if provided, otherwise fall back to local
        final_ca_type_dims = global_ca_type_dims if global_ca_type_dims is not None else ca_type_dims
        
        self.ca_projections = nn.ModuleDict()
        for ca_type_name, dims in final_ca_type_dims.items():
            self.ca_projections[ca_type_name] = nn.Linear(dims, d_model)

        # SRO projections: merge global + local dims
        # Global dims are for consistent SRO types; local dims fill in the rest
        final_sro_type_dims = {}
        if global_sro_type_dims is not None:
            final_sro_type_dims.update(global_sro_type_dims)
        # Add any local SRO types that aren't in global (e.g., inconsistent types)
        for sro_type_name, dims in sro_group_dims.items():
            if sro_type_name not in final_sro_type_dims:
                final_sro_type_dims[sro_type_name] = dims
        
        self.sro_projections = nn.ModuleDict()
        for sro_type_name, dims in final_sro_type_dims.items():
            self.sro_projections[sro_type_name] = nn.Linear(dims, d_model)

        # RL projection uses max_rl_input_dim if provided
        final_rl_input_dim = max_rl_input_dim if max_rl_input_dim is not None else rl_input_dim
        self.rl_projection: Optional[nn.Linear] = None
        if final_rl_input_dim > 0:
            self.rl_projection = nn.Linear(final_rl_input_dim, d_model)
        self._rl_has_residual = bool(self._rl_demand_indices) or bool(self._rl_generation_indices)
        self._local_rl_input_dim = rl_input_dim  # Store local dim for padding in forward()

        # --- Step 5: Build action-CA mapping ---------------------------------
        self._action_ca_map: List[int] = self._build_action_ca_map(
            action_names, ca_config,
        )

        # Pre-register index tensors as buffers for device handling
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
