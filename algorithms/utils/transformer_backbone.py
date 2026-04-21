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
from loguru import logger


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


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone with type embeddings.

    Processes concatenated [CA tokens, SRO tokens, NFC token] through
    self-attention. Each token attends to all others, producing contextual
    embeddings.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the Transformer backbone.

        Args:
            d_model: Embedding dimension for all tokens.
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            dim_feedforward: Feedforward network dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model

        # Type embeddings: 0=CA, 1=SRO, 2=NFC
        self.type_embedding = nn.Embedding(3, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        logger.info(
            "Initialized TransformerBackbone (d_model={}, nhead={}, layers={}, ff_dim={}, dropout={})",
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
        )

    def forward(
        self,
        ca_tokens: torch.Tensor,
        sro_tokens: torch.Tensor,
        nfc_token: torch.Tensor,
    ) -> TransformerOutput:
        """Process tokens through self-attention.

        Args:
            ca_tokens: [batch, N_ca, d_model] CA token embeddings.
            sro_tokens: [batch, N_sro, d_model] SRO token embeddings.
            nfc_token: [batch, 1, d_model] NFC token embedding.

        Returns:
            TransformerOutput with all embeddings, CA embeddings, pooled, and n_ca.
        """
        batch_size = ca_tokens.shape[0]
        n_ca = ca_tokens.shape[1]
        n_sro = sro_tokens.shape[1]
        device = ca_tokens.device

        # Concatenate all tokens: [batch, N_total, d_model]
        all_tokens = torch.cat([ca_tokens, sro_tokens, nfc_token], dim=1)
        n_total = all_tokens.shape[1]

        # Build type IDs
        type_ids = torch.cat([
            torch.zeros(batch_size, n_ca, dtype=torch.long, device=device),  # CA = 0
            torch.ones(batch_size, n_sro, dtype=torch.long, device=device),  # SRO = 1
            torch.full((batch_size, 1), 2, dtype=torch.long, device=device),  # NFC = 2
        ], dim=1)

        # Add type embeddings
        all_tokens = all_tokens + self.type_embedding(type_ids)

        # Self-attention
        encoded = self.encoder(all_tokens)

        # Extract CA embeddings (first n_ca positions)
        ca_embeddings = encoded[:, :n_ca, :]

        # Mean pooling over all tokens
        pooled = encoded.mean(dim=1)

        return TransformerOutput(
            all_embeddings=encoded,
            ca_embeddings=ca_embeddings,
            pooled=pooled,
            n_ca=n_ca,
        )
