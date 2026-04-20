"""Observation Tokenizer — scans markers and projects features to d_model.

Given an encoded observation tensor with marker values, the tokenizer:
1. Scans for marker values (1001-1999 for CAs, 2001-2999 for SROs, 3001 for NFC)
2. Splits the tensor into token groups based on marker positions
3. Projects each group to d_model via per-type Linear layers
4. Returns TokenizedObservation with ca_tokens, sro_tokens, nfc_token

This tokenizer is generic and reusable by any Transformer-based agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class TokenizedObservation:
    """Output of the observation tokenizer.

    Attributes:
        ca_tokens: [batch, N_ca, d_model] — one token per controllable asset instance.
        sro_tokens: [batch, N_sro, d_model] — one token per SRO group.
        nfc_token: [batch, 1, d_model] — non-flexible context token.
        ca_types: Type name per CA token position (e.g., ["battery", "ev_charger"]).
        n_ca: Number of CA tokens.
    """

    ca_tokens: torch.Tensor
    sro_tokens: torch.Tensor
    nfc_token: torch.Tensor
    ca_types: List[str]
    n_ca: int


MarkerTypeInfo = Tuple[str, str, Optional[str]]


def _lookup_marker_type(
    marker_registry: Dict[float, MarkerTypeInfo],
    marker_value: float,
) -> Optional[MarkerTypeInfo]:
    """Lookup marker type metadata with light tolerance for numeric keys."""
    if marker_value in marker_registry:
        return marker_registry[marker_value]

    rounded_marker = float(int(round(marker_value)))
    if rounded_marker in marker_registry:
        return marker_registry[rounded_marker]

    return None


def _find_marker_positions(
    encoded: torch.Tensor,
    ca_base: int,
    sro_base: int,
    nfc_marker: int,
) -> Tuple[List[List[int]], List[List[int]], List[Optional[int]]]:
    """Find positions of marker values in encoded tensor.

    Args:
        encoded: [batch, obs_dim] encoded observation tensor.
        ca_base: Base value for CA markers (e.g., 1000 -> markers are 1001, 1002, ...).
        sro_base: Base value for SRO markers (e.g., 2000 -> markers are 2001, 2002, ...).
        nfc_marker: Exact marker value for NFC (e.g., 3001).

    Returns:
        Tuple of:
            - ca_positions: List of lists, one per batch, containing CA marker positions.
            - sro_positions: List of lists, one per batch, containing SRO marker positions.
            - nfc_position: List, one per batch, containing NFC marker position (or None).
    """
    batch_size = encoded.shape[0]
    ca_positions: List[List[int]] = []
    sro_positions: List[List[int]] = []
    nfc_positions: List[Optional[int]] = []

    for b in range(batch_size):
        row = encoded[b]
        ca_pos: List[int] = []
        sro_pos: List[int] = []
        nfc_pos: Optional[int] = None

        for i, val in enumerate(row.tolist()):
            # Check if value is a CA marker (ca_base < val < ca_base + 1000)
            if ca_base < val < ca_base + 1000:
                ca_pos.append(i)
            # Check if value is an SRO marker (sro_base < val < sro_base + 1000)
            elif sro_base < val < sro_base + 1000:
                sro_pos.append(i)
            # Check if value is NFC marker
            elif abs(val - nfc_marker) < 0.01:  # Float comparison tolerance
                nfc_pos = i

        ca_positions.append(ca_pos)
        sro_positions.append(sro_pos)
        nfc_positions.append(nfc_pos)

    return ca_positions, sro_positions, nfc_positions


def _extract_groups(
    encoded: torch.Tensor,
    ca_positions: List[List[int]],
    sro_positions: List[List[int]],
    nfc_positions: List[Optional[int]],
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], List[torch.Tensor]]:
    """Extract feature groups from encoded tensor based on marker positions.

    Features between a marker and the next marker (or end of tensor) belong to that group.

    Args:
        encoded: [batch, obs_dim] encoded observation tensor.
        ca_positions: CA marker positions per batch.
        sro_positions: SRO marker positions per batch.
        nfc_positions: NFC marker position per batch (or None).

    Returns:
        Tuple of:
            - ca_groups: List of lists of tensors, [batch][ca_idx] -> features tensor
            - sro_groups: List of lists of tensors, [batch][sro_idx] -> features tensor
            - nfc_group: List of tensors, [batch] -> features tensor (empty if no NFC)
    """
    batch_size = encoded.shape[0]
    obs_dim = encoded.shape[1]

    ca_groups: List[List[torch.Tensor]] = []
    sro_groups: List[List[torch.Tensor]] = []
    nfc_groups: List[torch.Tensor] = []

    for b in range(batch_size):
        row = encoded[b]

        # Collect all marker positions to determine group boundaries
        all_markers: List[Tuple[int, str, int]] = []  # (position, type, index)

        for idx, pos in enumerate(ca_positions[b]):
            all_markers.append((pos, "ca", idx))
        for idx, pos in enumerate(sro_positions[b]):
            all_markers.append((pos, "sro", idx))
        if nfc_positions[b] is not None:
            all_markers.append((nfc_positions[b], "nfc", 0))

        # Sort by position
        all_markers.sort(key=lambda x: x[0])

        # Extract groups
        batch_ca_groups: List[torch.Tensor] = []
        batch_sro_groups: List[torch.Tensor] = []
        batch_nfc_group: torch.Tensor = torch.tensor([])

        for i, (pos, marker_type, _) in enumerate(all_markers):
            # Find end position (next marker or end of tensor)
            if i + 1 < len(all_markers):
                end_pos = all_markers[i + 1][0]
            else:
                end_pos = obs_dim

            # Extract features (skip the marker itself)
            features = row[pos + 1:end_pos]

            if marker_type == "ca":
                batch_ca_groups.append(features)
            elif marker_type == "sro":
                batch_sro_groups.append(features)
            elif marker_type == "nfc":
                batch_nfc_group = features

        ca_groups.append(batch_ca_groups)
        sro_groups.append(batch_sro_groups)
        nfc_groups.append(batch_nfc_group)

    return ca_groups, sro_groups, nfc_groups


class ObservationTokenizer(nn.Module):
    """Tokenizes encoded observations with marker values into typed token embeddings.

    Scans for marker values, extracts feature groups, and projects each group
    to d_model via per-type Linear layers.
    """

    def __init__(
        self,
        tokenizer_config: Dict[str, Any],
        d_model: int,
    ) -> None:
        """Initialize the tokenizer with config and embedding dimension.

        Args:
            tokenizer_config: Loaded from configs/tokenizers/default.json.
                Contains ca_types, sro_types, nfc with input_dim per type.
            d_model: Embedding dimension for all tokens.
        """
        super().__init__()
        self.d_model = d_model
        self._config = tokenizer_config

        # Extract marker values
        marker_values = tokenizer_config.get("marker_values", {})
        self.ca_base = marker_values.get("ca_base", 1000)
        self.sro_base = marker_values.get("sro_base", 2000)
        self.nfc_marker = marker_values.get("nfc", 3001)

        # Extract type configs
        ca_config = tokenizer_config.get("ca_types", {})
        sro_config = tokenizer_config.get("sro_types", {})
        nfc_config = tokenizer_config.get("nfc", {})

        # Create CA projections (one per type)
        self.ca_projections = nn.ModuleDict()
        self._ca_type_order = list(ca_config.keys())  # Preserve order
        for ca_type, spec in ca_config.items():
            input_dim = spec.get("input_dim", 1)
            self.ca_projections[ca_type] = nn.Linear(input_dim, d_model)

        # Create SRO projections (one per type)
        self.sro_projections = nn.ModuleDict()
        self._sro_type_order = list(sro_config.keys())  # Preserve order
        for sro_type, spec in sro_config.items():
            input_dim = spec.get("input_dim", 1)
            if input_dim > 0:
                self.sro_projections[sro_type] = nn.Linear(input_dim, d_model)

        # Create NFC projection
        nfc_input_dim = nfc_config.get("input_dim", 1)
        self.nfc_projection = nn.Linear(nfc_input_dim, d_model) if nfc_input_dim > 0 else None

        # Build input_dim -> type mapping for CA type inference
        self._ca_dim_to_type: Dict[int, str] = {}
        for ca_type, spec in ca_config.items():
            input_dim = spec.get("input_dim", 1)
            self._ca_dim_to_type[input_dim] = ca_type

        # Build input_dim -> type mapping for SRO type inference
        self._sro_dim_to_type: Dict[int, str] = {}
        for sro_type, spec in sro_config.items():
            input_dim = spec.get("input_dim", 1)
            self._sro_dim_to_type[input_dim] = sro_type

        # Warning deduplication caches for noisy per-step mismatches.
        self._warned_ca_dim_mismatch: set[Tuple[str, int, int]] = set()
        self._warned_ca_fallback: set[int] = set()
        self._warned_sro_dim_mismatch: set[Tuple[str, int, int]] = set()
        self._warned_sro_skip: set[int] = set()
        self._warned_nfc_truncation: set[Tuple[int, int]] = set()

        logger.info(
            "Initialized ObservationTokenizer (d_model={}, CA types={}, SRO types={}, nfc_input_dim={})",
            d_model,
            len(self._ca_type_order),
            len(self._sro_type_order),
            nfc_input_dim,
        )

    def forward(
        self,
        encoded_obs: torch.Tensor,
        marker_registry: Optional[Dict[float, MarkerTypeInfo]] = None,
    ) -> TokenizedObservation:
        """Tokenize encoded observations.

        Args:
            encoded_obs: [batch, obs_dim] flat encoded observation with markers.
            marker_registry: Optional mapping marker value -> (family, type_name, device_id).
                When provided, projection type selection uses explicit marker metadata
                rather than relying only on group feature count.

        Returns:
            TokenizedObservation with ca_tokens, sro_tokens, nfc_token, metadata.
        """
        batch_size = encoded_obs.shape[0]
        device = encoded_obs.device

        # Find marker positions
        ca_positions, sro_positions, nfc_positions = _find_marker_positions(
            encoded_obs, self.ca_base, self.sro_base, self.nfc_marker
        )

        if not any(ca_positions):
            logger.warning("ObservationTokenizer did not find CA markers in current batch.")

        # Extract groups
        ca_groups, sro_groups, nfc_groups = _extract_groups(
            encoded_obs, ca_positions, sro_positions, nfc_positions
        )

        resolved_registry = marker_registry or {}
        ca_marker_values = [
            [float(encoded_obs[b, pos].item()) for pos in ca_positions[b]]
            for b in range(batch_size)
        ]
        sro_marker_values = [
            [float(encoded_obs[b, pos].item()) for pos in sro_positions[b]]
            for b in range(batch_size)
        ]

        # Project CA groups
        all_ca_tokens: List[torch.Tensor] = []
        all_ca_types: List[str] = []

        for b in range(batch_size):
            batch_ca_tokens: List[torch.Tensor] = []
            for ca_idx, features in enumerate(ca_groups[b]):
                feature_count = int(features.shape[0])
                marker_value: Optional[float] = None
                if ca_idx < len(ca_marker_values[b]):
                    marker_value = ca_marker_values[b][ca_idx]

                ca_type: Optional[str] = None
                if marker_value is not None and resolved_registry:
                    marker_info = _lookup_marker_type(resolved_registry, marker_value)
                    if marker_info is not None:
                        family, type_name, _ = marker_info
                        if family == "ca" and type_name in self.ca_projections:
                            ca_type = type_name

                if ca_type is None:
                    ca_type = self._ca_dim_to_type.get(feature_count)

                if ca_type is not None and ca_type in self.ca_projections:
                    projection = self.ca_projections[ca_type]
                    expected_dim = int(projection.in_features)
                    if feature_count < expected_dim:
                        mismatch_key = (ca_type, feature_count, expected_dim)
                        if mismatch_key not in self._warned_ca_dim_mismatch:
                            logger.warning(
                                "CA token '{}' feature size {} is smaller than configured input_dim {}. "
                                "Padding with zeros.",
                                ca_type,
                                feature_count,
                                expected_dim,
                            )
                            self._warned_ca_dim_mismatch.add(mismatch_key)
                        features = torch.cat([
                            features,
                            torch.zeros(expected_dim - feature_count, device=device)
                        ])
                    elif feature_count > expected_dim:
                        mismatch_key = (ca_type, feature_count, expected_dim)
                        if mismatch_key not in self._warned_ca_dim_mismatch:
                            logger.warning(
                                "CA token '{}' feature size {} exceeds configured input_dim {}. "
                                "Truncating trailing values.",
                                ca_type,
                                feature_count,
                                expected_dim,
                            )
                            self._warned_ca_dim_mismatch.add(mismatch_key)
                        features = features[:expected_dim]

                    token = projection(features.unsqueeze(0))  # [1, d_model]
                    batch_ca_tokens.append(token)
                    if b == 0:  # Only collect types once
                        all_ca_types.append(ca_type)
                else:
                    # Unknown type - use first available projection as fallback
                    if self._ca_type_order:
                        fallback_type = self._ca_type_order[0]
                        if feature_count not in self._warned_ca_fallback:
                            logger.warning(
                                "Unknown CA token feature size {}. Falling back to '{}' projection.",
                                feature_count,
                                fallback_type,
                            )
                            self._warned_ca_fallback.add(feature_count)
                        # Pad or truncate features to match expected dim
                        expected_dim = self.ca_projections[fallback_type].in_features
                        if feature_count < expected_dim:
                            features = torch.cat([
                                features,
                                torch.zeros(expected_dim - feature_count, device=device)
                            ])
                        elif feature_count > expected_dim:
                            features = features[:expected_dim]
                        token = self.ca_projections[fallback_type](features.unsqueeze(0))
                        batch_ca_tokens.append(token)
                        if b == 0:
                            all_ca_types.append(fallback_type)

            if batch_ca_tokens:
                all_ca_tokens.append(torch.cat(batch_ca_tokens, dim=0))
            else:
                all_ca_tokens.append(torch.zeros(0, self.d_model, device=device))

        # Stack CA tokens across batch
        n_ca = len(ca_groups[0]) if ca_groups else 0
        if n_ca > 0:
            ca_tokens = torch.stack([t for t in all_ca_tokens], dim=0)  # [batch, n_ca, d_model]
        else:
            ca_tokens = torch.zeros(batch_size, 0, self.d_model, device=device)

        # Project SRO groups
        all_sro_tokens: List[torch.Tensor] = []
        for b in range(batch_size):
            batch_sro_tokens: List[torch.Tensor] = []
            for sro_idx, features in enumerate(sro_groups[b]):
                feature_count = int(features.shape[0])
                marker_value: Optional[float] = None
                if sro_idx < len(sro_marker_values[b]):
                    marker_value = sro_marker_values[b][sro_idx]

                sro_type: Optional[str] = None
                if marker_value is not None and resolved_registry:
                    marker_info = _lookup_marker_type(resolved_registry, marker_value)
                    if marker_info is not None:
                        family, type_name, _ = marker_info
                        if family == "sro" and type_name in self.sro_projections:
                            sro_type = type_name

                if sro_type is None:
                    sro_type = self._sro_dim_to_type.get(feature_count)

                if sro_type is not None and sro_type in self.sro_projections:
                    projection = self.sro_projections[sro_type]
                    expected_dim = int(projection.in_features)
                    if feature_count < expected_dim:
                        mismatch_key = (sro_type, feature_count, expected_dim)
                        if mismatch_key not in self._warned_sro_dim_mismatch:
                            logger.warning(
                                "SRO token '{}' feature size {} is smaller than configured input_dim {}. "
                                "Padding with zeros.",
                                sro_type,
                                feature_count,
                                expected_dim,
                            )
                            self._warned_sro_dim_mismatch.add(mismatch_key)
                        features = torch.cat([
                            features,
                            torch.zeros(expected_dim - feature_count, device=device)
                        ])
                    elif feature_count > expected_dim:
                        mismatch_key = (sro_type, feature_count, expected_dim)
                        if mismatch_key not in self._warned_sro_dim_mismatch:
                            logger.warning(
                                "SRO token '{}' feature size {} exceeds configured input_dim {}. "
                                "Truncating trailing values.",
                                sro_type,
                                feature_count,
                                expected_dim,
                            )
                            self._warned_sro_dim_mismatch.add(mismatch_key)
                        features = features[:expected_dim]

                    token = projection(features.unsqueeze(0))
                    batch_sro_tokens.append(token)
                else:
                    if feature_count not in self._warned_sro_skip:
                        logger.warning(
                            "Skipping SRO token with unsupported feature size {}.",
                            feature_count,
                        )
                        self._warned_sro_skip.add(feature_count)

            if batch_sro_tokens:
                all_sro_tokens.append(torch.cat(batch_sro_tokens, dim=0))
            else:
                all_sro_tokens.append(torch.zeros(0, self.d_model, device=device))

        n_sro = len(sro_groups[0]) if sro_groups else 0
        if n_sro > 0:
            sro_tokens = torch.stack(all_sro_tokens, dim=0)
        else:
            sro_tokens = torch.zeros(batch_size, 0, self.d_model, device=device)

        # Project NFC group
        if self.nfc_projection is not None and any(g.numel() > 0 for g in nfc_groups):
            nfc_tokens_list = []
            expected_nfc_dim = self.nfc_projection.in_features
            for b in range(batch_size):
                if nfc_groups[b].numel() > 0:
                    nfc_features = nfc_groups[b]
                    feature_count = nfc_features.shape[0]
                    if feature_count < expected_nfc_dim:
                        nfc_features = torch.cat([
                            nfc_features,
                            torch.zeros(expected_nfc_dim - feature_count, device=device),
                        ])
                    elif feature_count > expected_nfc_dim:
                        truncation_key = (feature_count, expected_nfc_dim)
                        if truncation_key not in self._warned_nfc_truncation:
                            logger.warning(
                                "NFC token feature size {} exceeds configured input_dim {}. "
                                "Truncating trailing values.",
                                feature_count,
                                expected_nfc_dim,
                            )
                            self._warned_nfc_truncation.add(truncation_key)
                        nfc_features = nfc_features[:expected_nfc_dim]

                    nfc_token = self.nfc_projection(nfc_features.unsqueeze(0))
                else:
                    nfc_token = torch.zeros(1, self.d_model, device=device)
                nfc_tokens_list.append(nfc_token)
            nfc_token = torch.stack(nfc_tokens_list, dim=0)  # [batch, 1, d_model]
        else:
            nfc_token = torch.zeros(batch_size, 1, self.d_model, device=device)

        return TokenizedObservation(
            ca_tokens=ca_tokens,
            sro_tokens=sro_tokens,
            nfc_token=nfc_token,
            ca_types=all_ca_types,
            n_ca=n_ca,
        )
