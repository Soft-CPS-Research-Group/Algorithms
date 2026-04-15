"""Transformer Backbone — self-attention over token sequence.

Processes concatenated [CA tokens, SRO tokens, NFC token] through
self-attention with type embeddings. Produces contextual embeddings
where each token has attended to all others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TransformerOutput:
    """Output of the Transformer backbone.

    Attributes:
        all_embeddings: [batch, N_total, d_model] — all token embeddings.
        ca_embeddings: [batch, N_ca, d_model] — CA token embeddings (sliced from all).
        pooled: [batch, d_model] — mean-pooled over all tokens.
        n_ca: Number of CA tokens.
    """

    all_embeddings: torch.Tensor
    ca_embeddings: torch.Tensor
    pooled: torch.Tensor
    n_ca: int
