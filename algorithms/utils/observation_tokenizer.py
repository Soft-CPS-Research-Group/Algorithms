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
