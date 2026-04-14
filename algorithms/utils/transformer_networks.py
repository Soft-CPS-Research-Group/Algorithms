"""Transformer-based neural network architectures for variable-cardinality MADDPG.

This module provides Transformer encoder-based Actor and Critic networks that support
variable numbers of Controllable Assets (CAs) and Shared Read-Only observations (SROs)
at runtime without retraining.

Key design principles:
- CA tokens produce exactly one output each (1-to-1 mapping)
- SRO tokens provide context but don't produce outputs
- NFC token represents global context (non-flexible consumption)
- Token type embeddings distinguish CA/SRO/NFC tokens
- No positional embeddings (permutation invariant within token types)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for Transformer network hyperparameters.
    
    Attributes:
        d_model: Embedding dimension (must be divisible by nhead).
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Hidden dimension of the feedforward network.
        dropout: Dropout rate for regularization.
        max_tokens: Maximum total tokens (N_ca + N_sro + 1).
        action_dim: Dimension of action output per CA.
    """
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    max_tokens: int = 128
    action_dim: int = 1

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.d_model < 8:
            raise ValueError(f"d_model must be >= 8, got {self.d_model}")
        if self.nhead < 1:
            raise ValueError(f"nhead must be >= 1, got {self.nhead}")
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.dim_feedforward < 8:
            raise ValueError(f"dim_feedforward must be >= 8, got {self.dim_feedforward}")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.max_tokens < 4:
            raise ValueError(f"max_tokens must be >= 4, got {self.max_tokens}")
        if self.action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {self.action_dim}")


class TransformerActor(nn.Module):
    """Transformer encoder-based policy network producing CA-aligned actions.
    
    Architecture:
    - Type embeddings for CA/SRO/NFC token distinction
    - Transformer encoder for cross-token attention
    - MLP head applied only to CA tokens for action output
    
    The forward pass ensures strict 1-to-1 mapping: output[i] corresponds to ca_tokens[i].
    
    Args:
        cfg: TransformerConfig with network hyperparameters.
    """

    # Token type IDs
    TYPE_CA = 0
    TYPE_SRO = 1
    TYPE_NFC = 2

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Type embeddings: 0=CA, 1=SRO, 2=NFC
        self.type_emb = nn.Embedding(3, cfg.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Action head (applied only to CA outputs)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.action_dim),
        )

    def forward(
        self,
        ca_tokens: torch.Tensor,
        sro_tokens: torch.Tensor,
        nfc_token: torch.Tensor,
        ca_mask: Optional[torch.Tensor] = None,
        sro_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass producing CA-aligned actions.
        
        Args:
            ca_tokens: [B, N_ca, d_model] embeddings for controllable assets.
            sro_tokens: [B, N_sro, d_model] embeddings for shared read-only observations.
            nfc_token: [B, 1, d_model] embedding for non-flexible consumption.
            ca_mask: [B, N_ca] boolean mask (True=valid, False=padding). Optional.
            sro_mask: [B, N_sro] boolean mask (True=valid, False=padding). Optional.
            
        Returns:
            actions: [B, N_ca, action_dim] bounded actions in [-1, 1].
        """
        B, N_ca, D = ca_tokens.shape
        _, N_sro, _ = sro_tokens.shape
        
        assert D == self.cfg.d_model, f"Expected d_model={self.cfg.d_model}, got {D}"
        assert nfc_token.shape == (B, 1, D), f"Expected nfc_token shape ({B}, 1, {D}), got {nfc_token.shape}"

        total = N_ca + N_sro + 1
        if total > self.cfg.max_tokens:
            raise ValueError(
                f"Total tokens ({total}) exceeds max_tokens ({self.cfg.max_tokens})"
            )

        # Concatenate tokens: [CA..., SRO..., NFC]
        x = torch.cat([ca_tokens, sro_tokens, nfc_token], dim=1)  # [B, T, D]

        # Build type IDs
        device = x.device
        type_ids = torch.cat([
            torch.full((B, N_ca), self.TYPE_CA, dtype=torch.long, device=device),
            torch.full((B, N_sro), self.TYPE_SRO, dtype=torch.long, device=device),
            torch.full((B, 1), self.TYPE_NFC, dtype=torch.long, device=device),
        ], dim=1)  # [B, T]

        # Add type embeddings
        x = x + self.type_emb(type_ids)

        # Build key padding mask (True=ignore in PyTorch convention)
        if ca_mask is None:
            ca_mask = torch.ones(B, N_ca, dtype=torch.bool, device=device)
        if sro_mask is None:
            sro_mask = torch.ones(B, N_sro, dtype=torch.bool, device=device)
        nfc_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        keep_mask = torch.cat([ca_mask, sro_mask, nfc_mask], dim=1)  # [B, T], True=valid
        key_padding_mask = ~keep_mask  # True=pad/ignore

        # Encode
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, T, D]

        # Strict 1-to-1 CA mapping: slice only CA outputs
        h_ca = h[:, :N_ca, :]  # [B, N_ca, D]

        # Apply action head
        actions = self.head(h_ca)  # [B, N_ca, action_dim]

        # Bound to [-1, 1]
        return torch.tanh(actions)


class CriticAggregationStrategy(ABC):
    """Abstract base class for critic aggregation strategies.
    
    Strategies define how to pool encoded token representations
    into a single vector for Q-value estimation.
    """

    @abstractmethod
    def aggregate(
        self, encoded: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate encoded tokens into a single representation.
        
        Args:
            encoded: [B, T, D] encoded token representations.
            mask: [B, T] boolean mask (True=valid, False=padding).
            
        Returns:
            pooled: [B, D] aggregated representation.
        """


class MeanPoolStrategy(CriticAggregationStrategy):
    """Mean pooling over valid tokens.
    
    Computes the mean of encoded representations, ignoring padded positions.
    """

    def aggregate(
        self, encoded: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pool over valid tokens.
        
        Args:
            encoded: [B, T, D] encoded representations.
            mask: [B, T] boolean mask (True=valid).
            
        Returns:
            pooled: [B, D] mean-pooled representation.
        """
        # Expand mask for broadcasting: [B, T, 1]
        mask_expanded = mask.unsqueeze(-1).float()
        
        # Masked sum and count
        masked_encoded = encoded * mask_expanded
        summed = masked_encoded.sum(dim=1)  # [B, D]
        counts = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1], avoid div by 0
        
        return summed / counts


class CLSTokenStrategy(CriticAggregationStrategy):
    """Use the NFC token output as the aggregate representation.
    
    The NFC token (last position) acts like a [CLS] token in BERT,
    aggregating information from all tokens through attention.
    """

    def aggregate(
        self, encoded: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Return the last token (NFC) representation.
        
        Args:
            encoded: [B, T, D] encoded representations.
            mask: [B, T] boolean mask (unused, NFC always valid).
            
        Returns:
            pooled: [B, D] NFC token representation.
        """
        return encoded[:, -1, :]  # NFC is always last


class TransformerCritic(nn.Module):
    """Transformer encoder-based Q-value estimator.
    
    Architecture:
    - Action embedding projection to combine with CA tokens
    - Type embeddings for CA/SRO/NFC distinction
    - Transformer encoder for cross-token attention
    - Aggregation strategy to pool encoded tokens
    - MLP head for scalar Q-value output
    
    Args:
        cfg: TransformerConfig with network hyperparameters.
        aggregation_strategy: Strategy for pooling encoded tokens. Defaults to MeanPoolStrategy.
    """

    # Token type IDs
    TYPE_CA = 0
    TYPE_SRO = 1
    TYPE_NFC = 2

    def __init__(
        self,
        cfg: TransformerConfig,
        aggregation_strategy: Optional[CriticAggregationStrategy] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.aggregation_strategy = aggregation_strategy or MeanPoolStrategy()

        # Type embeddings: 0=CA, 1=SRO, 2=NFC
        self.type_emb = nn.Embedding(3, cfg.d_model)

        # Action embedding: project actions to d_model for fusion with CA tokens
        self.action_embed = nn.Linear(cfg.action_dim, cfg.d_model)

        # Transformer encoder (separate from actor)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Q-value head
        self.q_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self,
        ca_tokens: torch.Tensor,
        sro_tokens: torch.Tensor,
        nfc_token: torch.Tensor,
        actions: torch.Tensor,
        ca_mask: Optional[torch.Tensor] = None,
        sro_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass computing Q-value.
        
        Args:
            ca_tokens: [B, N_ca, d_model] embeddings for controllable assets.
            sro_tokens: [B, N_sro, d_model] embeddings for shared read-only observations.
            nfc_token: [B, 1, d_model] embedding for non-flexible consumption.
            actions: [B, N_ca, action_dim] actions for each CA.
            ca_mask: [B, N_ca] boolean mask (True=valid). Optional.
            sro_mask: [B, N_sro] boolean mask (True=valid). Optional.
            
        Returns:
            q_value: [B, 1] estimated Q-value.
        """
        B, N_ca, D = ca_tokens.shape
        _, N_sro, _ = sro_tokens.shape
        
        assert D == self.cfg.d_model, f"Expected d_model={self.cfg.d_model}, got {D}"
        assert nfc_token.shape == (B, 1, D)
        assert actions.shape == (B, N_ca, self.cfg.action_dim)

        total = N_ca + N_sro + 1
        if total > self.cfg.max_tokens:
            raise ValueError(
                f"Total tokens ({total}) exceeds max_tokens ({self.cfg.max_tokens})"
            )

        # Embed actions and add to CA tokens
        action_embed = self.action_embed(actions)  # [B, N_ca, d_model]
        ca_with_actions = ca_tokens + action_embed  # [B, N_ca, d_model]

        # Concatenate tokens: [CA+actions, SRO..., NFC]
        x = torch.cat([ca_with_actions, sro_tokens, nfc_token], dim=1)  # [B, T, D]

        # Build type IDs
        device = x.device
        type_ids = torch.cat([
            torch.full((B, N_ca), self.TYPE_CA, dtype=torch.long, device=device),
            torch.full((B, N_sro), self.TYPE_SRO, dtype=torch.long, device=device),
            torch.full((B, 1), self.TYPE_NFC, dtype=torch.long, device=device),
        ], dim=1)  # [B, T]

        # Add type embeddings
        x = x + self.type_emb(type_ids)

        # Build key padding mask
        if ca_mask is None:
            ca_mask = torch.ones(B, N_ca, dtype=torch.bool, device=device)
        if sro_mask is None:
            sro_mask = torch.ones(B, N_sro, dtype=torch.bool, device=device)
        nfc_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        keep_mask = torch.cat([ca_mask, sro_mask, nfc_mask], dim=1)
        key_padding_mask = ~keep_mask

        # Encode
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, T, D]

        # Aggregate
        pooled = self.aggregation_strategy.aggregate(h, keep_mask)  # [B, D]

        # Q-value
        q = self.q_head(pooled)  # [B, 1]

        return q
