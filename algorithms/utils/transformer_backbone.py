"""Transformer backbone — shared encoder that processes all token types.

Takes the tokenized observation (CA + SRO + RL tokens) and produces
contextual embeddings through self-attention.  The architecture uses
learnable type embeddings (CA=0, SRO=1, RL=2) — no positional embeddings
(pure set semantics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from algorithms.utils.observation_tokenizer import TokenizedObservation


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class TransformerOutput:
    """Output of the Transformer backbone.

    Attributes:
        all_embeddings: [batch, N_total, d_model] — contextual embeddings at
            every token position.
        ca_embeddings: [batch, N_ca, d_model] — slice corresponding to CA positions.
        pooled: [batch, d_model] — mean-pooled over all token positions.
        n_ca: Number of CA tokens.
    """

    all_embeddings: torch.Tensor
    ca_embeddings: torch.Tensor
    pooled: torch.Tensor
    n_ca: int


# ---------------------------------------------------------------------------
# TransformerBackbone
# ---------------------------------------------------------------------------


class TransformerBackbone(nn.Module):
    """Shared Transformer encoder that processes all token types.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of encoder layers.
    dim_feedforward : int
        Width of the FFN in each encoder layer.
    dropout : float
        Dropout rate.
    """

    # Type IDs for the three token types
    TYPE_CA = 0
    TYPE_SRO = 1
    TYPE_RL = 2

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Learnable type embeddings (3 types: CA, SRO, RL)
        self.type_embedding = nn.Embedding(3, d_model)

        # Standard Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, tokenized: TokenizedObservation) -> TransformerOutput:
        """Process tokenized observations through self-attention.

        Parameters
        ----------
        tokenized : TokenizedObservation
            Output from ``ObservationTokenizer.forward()``.

        Returns
        -------
        TransformerOutput
        """

        device = self._get_device()
        n_ca = tokenized.n_ca
        batch = tokenized.rl_token.shape[0]

        # --- Assemble tokens in deterministic order: CA, SRO, RL -----------
        parts: list[torch.Tensor] = []
        type_ids: list[torch.Tensor] = []

        if n_ca > 0:
            parts.append(tokenized.ca_tokens)
            type_ids.append(
                torch.full((batch, n_ca), self.TYPE_CA, dtype=torch.long, device=device)
            )

        n_sro = tokenized.sro_tokens.shape[1]
        if n_sro > 0:
            parts.append(tokenized.sro_tokens)
            type_ids.append(
                torch.full((batch, n_sro), self.TYPE_SRO, dtype=torch.long, device=device)
            )

        # RL token always present (even if zero-initialized)
        parts.append(tokenized.rl_token)
        type_ids.append(
            torch.full((batch, 1), self.TYPE_RL, dtype=torch.long, device=device)
        )

        # Concatenate: [batch, N_total, d_model]
        tokens = torch.cat(parts, dim=1)
        type_id_tensor = torch.cat(type_ids, dim=1)  # [batch, N_total]

        # Add type embeddings
        tokens = tokens + self.type_embedding(type_id_tensor)

        # --- Transformer encoder -------------------------------------------
        embeddings = self.encoder(tokens)  # [batch, N_total, d_model]

        # --- Output slicing ------------------------------------------------
        ca_embeddings = embeddings[:, :n_ca, :]  # [batch, N_ca, d_model]
        pooled = embeddings.mean(dim=1)  # [batch, d_model]

        return TransformerOutput(
            all_embeddings=embeddings,
            ca_embeddings=ca_embeddings,
            pooled=pooled,
            n_ca=n_ca,
        )

    def _get_device(self) -> torch.device:
        """Infer the device from model parameters."""
        return next(self.parameters()).device
