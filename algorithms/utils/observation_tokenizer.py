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
