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
