"""Twin independent Transformer critic stacks for AgentTransformerMATD3.

Each critic stack has its own TransformerEncoder, type embeddings, building
embeddings, and Q head. The two critics are fully independent to preserve
TD3's overestimation reduction property.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class TransformerCriticStack(nn.Module):
    """Single Transformer critic: global sequence -> per-building Q values."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_token_types: int,
        max_buildings: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_token_types = num_token_types
        self.max_buildings = max_buildings
        self.type_embedding = nn.Embedding(num_token_types, d_model)
        self.building_embedding = nn.Embedding(max_buildings, d_model)
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
        self.q_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        building_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        controlled_building_indices: List[int],
    ) -> torch.Tensor:
        """Return Q-values [B, n_controlled] for each controlled building."""
        seq = global_tokens + self.type_embedding(type_ids) + self.building_embedding(building_ids)
        seq = seq.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        encoded = self.encoder(seq, src_key_padding_mask=padding_mask)
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        q_values = []
        for b_idx in controlled_building_indices:
            building_mask = (building_ids == b_idx) & (~padding_mask)
            building_mask_expanded = building_mask.unsqueeze(-1).float()
            summed = (encoded * building_mask_expanded).sum(dim=1)
            count = building_mask_expanded.sum(dim=1).clamp(min=1.0)
            pooled = summed / count
            q_values.append(self.q_head(pooled))
        return torch.cat(q_values, dim=-1)


class TwinTransformerCritics(nn.Module):
    """Container for two fully independent TransformerCriticStack instances."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_token_types: int,
        max_buildings: int,
    ) -> None:
        super().__init__()
        self.critic_1 = TransformerCriticStack(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            num_token_types=num_token_types, max_buildings=max_buildings,
        )
        self.critic_2 = TransformerCriticStack(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            num_token_types=num_token_types, max_buildings=max_buildings,
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        building_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        controlled_building_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (q1, q2), each of shape [B, n_controlled]."""
        q1 = self.critic_1(
            global_tokens, type_ids, building_ids,
            padding_mask, controlled_building_indices,
        )
        q2 = self.critic_2(
            global_tokens, type_ids, building_ids,
            padding_mask, controlled_building_indices,
        )
        return q1, q2

    def min_q(
        self,
        global_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        building_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        controlled_building_indices: List[int],
    ) -> torch.Tensor:
        """Return element-wise min(q1, q2)."""
        q1, q2 = self.forward(
            global_tokens, type_ids, building_ids,
            padding_mask, controlled_building_indices,
        )
        return torch.min(q1, q2)

    @torch.no_grad()
    def soft_update_from(self, source: "TwinTransformerCritics", tau: float) -> None:
        """Polyak-average: target = tau * source + (1 - tau) * target."""
        for p_target, p_source in zip(self.parameters(), source.parameters()):
            p_target.data.mul_(1.0 - tau).add_(p_source.data, alpha=tau)
