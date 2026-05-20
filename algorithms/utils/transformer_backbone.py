"""Transformer Backbone — self-attention over the per-building token sequence.

Spec ``docs/specv2.md`` §9: token order in the concatenated sequence is
``[SRO tokens, NFC token, CA tokens]``. Type-embedding semantics are fixed:
``SRO = 0``, ``NFC = 1``, ``CA = 2``. ``forward(sros, nfc, cas)`` returns
``(ca_embeddings, pooled)`` — the CA slice is sliced at the post-NFC offset
and ``pooled`` is the mean across all tokens (used by the critic).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from loguru import logger


# Type-id constants — the integer values index ``self.type_embedding``.
_TYPE_SRO = 0
_TYPE_NFC = 1
_TYPE_CA = 2


class TransformerBackbone(nn.Module):
    """Self-attention encoder over ``[SRO, NFC, CA]`` token sequence.

    Input shapes are per-building: ``[batch, n_*, d_model]``. Token counts
    can vary between calls (variable topology); only ``d_model`` is fixed.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # 3-entry table: SRO=0, NFC=1, CA=2 (spec §9).
        self.type_embedding = nn.Embedding(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        logger.info(
            "Initialized TransformerBackbone "
            "(d_model={}, nhead={}, layers={}, ff_dim={}, dropout={})",
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
        )

    def forward(
        self,
        sros: torch.Tensor,  # [B, N_sro, d_model]
        nfc: torch.Tensor,  # [B, 1, d_model]
        cas: torch.Tensor,  # [B, N_ca, d_model]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the full sequence and return ``(ca_embeddings, pooled)``.

        Args:
          sros: ``[B, N_sro, d_model]`` SRO tokens.
          nfc: ``[B, 1, d_model]`` single NFC token.
          cas: ``[B, N_ca, d_model]`` CA tokens (one per actuator).

        Returns:
          (``ca_embeddings`` of shape ``[B, N_ca, d_model]``,
          ``pooled`` of shape ``[B, d_model]``).
        """
        n_sro = sros.shape[1]
        n_ca = cas.shape[1]
        device = sros.device

        # Build the per-position type-id sequence
        # ``[SRO]*n_sro + [NFC] + [CA]*n_ca``.
        type_ids = torch.cat(
            [
                torch.full(
                    (n_sro,), _TYPE_SRO, dtype=torch.long, device=device
                ),
                torch.full(
                    (1,), _TYPE_NFC, dtype=torch.long, device=device
                ),
                torch.full(
                    (n_ca,), _TYPE_CA, dtype=torch.long, device=device
                ),
            ]
        )
        # Concat in the spec order [sros, nfc, cas].
        seq = torch.cat([sros, nfc, cas], dim=1)
        # Add type embeddings (broadcast across batch).
        seq = seq + self.type_embedding(type_ids).unsqueeze(0)

        encoded = self.encoder(seq)

        ca_embeddings = encoded[:, n_sro + 1 : n_sro + 1 + n_ca, :]
        pooled = encoded.mean(dim=1)
        return ca_embeddings, pooled
