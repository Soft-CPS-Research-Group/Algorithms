from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from algorithms.transformer_shared.entity_observation_tokenizer import (
    EntityObservationTokenizer,
)
from algorithms.transformer_shared.entity_token_layout import BuildingTokenLayout
from algorithms.transformer_shared.transformer_backbone import TransformerBackbone


class DeterministicActorHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self._d_model = int(d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self._d_model),
            nn.Linear(self._d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ca_embeddings: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        del deterministic
        if ca_embeddings.dim() != 3 or ca_embeddings.shape[-1] != self._d_model:
            raise ValueError(
                "ca_embeddings must have shape [batch, n_ca, "
                f"{self._d_model}], got {tuple(ca_embeddings.shape)}"
            )
        return self.mlp(ca_embeddings)


class ActionInjectionMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self._d_model = int(d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self._d_model + 1),
            nn.Linear(self._d_model + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self._d_model),
        )

    def forward(
        self,
        ca_embeddings: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        expected_action_shape = ca_embeddings.shape[:2]
        if ca_embeddings.dim() != 3 or ca_embeddings.shape[-1] != self._d_model:
            raise ValueError(
                "ca_embeddings must have shape [batch, n_ca, "
                f"{self._d_model}], got {tuple(ca_embeddings.shape)}"
            )
        if actions.dim() != 2 or actions.shape != expected_action_shape:
            raise ValueError(
                "action shape must match CA embeddings [batch, n_ca], got "
                f"{tuple(actions.shape)} for embeddings "
                f"{tuple(ca_embeddings.shape)}"
            )
        return self.mlp(torch.cat((ca_embeddings, actions.unsqueeze(-1)), dim=-1))


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        hidden_dim: int,
        dropout: float,
        *,
        tokenizer_config: Any,
        type_input_dims: Mapping[str, int],
    ) -> None:
        super().__init__()
        self.tokenizer = EntityObservationTokenizer(
            tokenizer_config=tokenizer_config,
            d_model=d_model,
            type_input_dims=type_input_dims,
        )
        self.backbone = TransformerBackbone(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.action_injection = ActionInjectionMLP(d_model, hidden_dim)
        self.building_embedding = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.deep_set_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.q_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        per_building_obs: Sequence[torch.Tensor],
        per_building_layouts: Sequence[BuildingTokenLayout],
        per_building_actions: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        community_size = len(per_building_obs)
        if community_size == 0:
            raise ValueError("centralized critic requires at least one building")
        if len(per_building_layouts) != community_size:
            raise ValueError("layout count must match observation count")
        if len(per_building_actions) != community_size:
            raise ValueError("action count must match observation count")

        batch_size = per_building_obs[0].shape[0]
        building_embeddings = []
        for index, (observation, layout, actions) in enumerate(
            zip(per_building_obs, per_building_layouts, per_building_actions)
        ):
            if observation.dim() != 2 or observation.shape[0] != batch_size:
                raise ValueError(
                    "all building observations must have the same batch size; "
                    f"building {index} has shape {tuple(observation.shape)}"
                )
            if actions.dim() != 2 or actions.shape[0] != batch_size:
                raise ValueError(
                    "all building actions must have the same batch size; "
                    f"building {index} has shape {tuple(actions.shape)}"
                )
            if actions.shape[1] != layout.n_ca:
                raise ValueError(
                    f"building {index} action width is {actions.shape[1]}, "
                    f"expected {layout.n_ca}"
                )

            tokens = self.tokenizer(observation, layout)
            ca_embeddings, pooled = self.backbone(
                tokens.sro_tokens,
                tokens.nfc_token,
                tokens.ca_tokens,
            )
            action_conditioned = self.action_injection(ca_embeddings, actions)
            if layout.n_ca == 0:
                action_summary = torch.zeros_like(pooled)
            else:
                action_summary = action_conditioned.mean(dim=1)
            building_embeddings.append(
                self.building_embedding(
                    torch.cat((pooled, action_summary), dim=-1)
                )
            )

        projected = [
            self.deep_set_projection(embedding)
            for embedding in building_embeddings
        ]
        community_embedding = torch.stack(projected, dim=1).mean(dim=1)
        return self.q_head(community_embedding)
