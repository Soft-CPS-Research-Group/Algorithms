"""Global critic token packer for AgentTransformerMATD3.

Concatenates all buildings' observation tokens and action tokens into one
global sequence for centralized twin critics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


TYPE_OBS_SRO = 0
TYPE_OBS_NFC = 1
TYPE_OBS_CA = 2
TYPE_ACTION = 3


@dataclass
class BuildingLayout:
    """Lightweight layout summary for the packer."""
    building_index: int
    n_sro: int
    n_nfc: int
    n_ca: int
    is_controlled: bool

    @property
    def n_obs_tokens(self) -> int:
        return self.n_sro + self.n_nfc + self.n_ca

    @property
    def n_action_tokens(self) -> int:
        return self.n_ca

    @property
    def n_total_tokens(self) -> int:
        return self.n_obs_tokens + self.n_action_tokens


@dataclass
class PackedGlobalSequence:
    """Output of the global token packer, ready for twin critics."""
    global_tokens: torch.Tensor
    type_ids: torch.Tensor
    building_ids: torch.Tensor
    padding_mask: torch.Tensor
    controlled_building_indices: List[int]


class GlobalTokenPacker(nn.Module):
    """Pack per-building obs and action tokens into a global sequence."""

    def __init__(
        self,
        d_model: int,
        num_token_types: int,
        max_buildings: int,
        action_input_mode: str = "final",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.action_input_mode = action_input_mode
        self.num_token_types = num_token_types
        self.max_buildings = max_buildings

        if action_input_mode == "final":
            action_input_dim = 1
        elif action_input_mode in ("final_base_delta", "final_base_delta_normalized"):
            action_input_dim = 3
        else:
            raise ValueError(f"Unknown action_input_mode: {action_input_mode!r}")
        self.action_projection = nn.Linear(action_input_dim, d_model)

    def pack(
        self,
        obs_tokens_per_building: List[torch.Tensor],
        action_values_per_building: List[torch.Tensor],
        layouts: List[BuildingLayout],
        *,
        base_actions: Optional[List[torch.Tensor]] = None,
        action_span: float = 2.0,
    ) -> PackedGlobalSequence:
        """Pack all buildings into a global sequence."""
        assert len(obs_tokens_per_building) == len(layouts)
        assert len(action_values_per_building) == len(layouts)
        batch_size = obs_tokens_per_building[0].shape[0]
        device = obs_tokens_per_building[0].device

        all_tokens: List[torch.Tensor] = []
        all_type_ids: List[torch.Tensor] = []
        all_building_ids: List[torch.Tensor] = []
        controlled_building_indices: List[int] = []

        for i, layout in enumerate(layouts):
            obs_toks = obs_tokens_per_building[i]
            obs_type_ids = torch.cat([
                torch.full((layout.n_sro,), TYPE_OBS_SRO, dtype=torch.long, device=device),
                torch.full((layout.n_nfc,), TYPE_OBS_NFC, dtype=torch.long, device=device),
                torch.full((layout.n_ca,), TYPE_OBS_CA, dtype=torch.long, device=device),
            ])

            actions = action_values_per_building[i]
            n_ca = layout.n_ca
            if n_ca > 0:
                action_toks = self._encode_action_tokens(
                    actions, base_actions[i] if base_actions else None,
                    action_span, device,
                )
                action_type_ids = torch.full((n_ca,), TYPE_ACTION, dtype=torch.long, device=device)
            else:
                action_toks = torch.zeros(batch_size, 0, self.d_model, device=device)
                action_type_ids = torch.zeros(0, dtype=torch.long, device=device)

            building_tokens = torch.cat([obs_toks, action_toks], dim=1)
            building_type_ids = torch.cat([obs_type_ids, action_type_ids])
            building_id_vec = torch.full(
                (building_tokens.shape[1],), layout.building_index, dtype=torch.long, device=device,
            )
            all_tokens.append(building_tokens)
            all_type_ids.append(building_type_ids)
            all_building_ids.append(building_id_vec)
            if layout.is_controlled:
                controlled_building_indices.append(layout.building_index)

        global_tokens = torch.cat(all_tokens, dim=1)
        type_ids = torch.cat(all_type_ids).unsqueeze(0).expand(batch_size, -1)
        building_ids = torch.cat(all_building_ids).unsqueeze(0).expand(batch_size, -1)
        padding_mask = torch.zeros(
            batch_size, global_tokens.shape[1], dtype=torch.bool, device=device,
        )

        return PackedGlobalSequence(
            global_tokens=global_tokens,
            type_ids=type_ids,
            building_ids=building_ids,
            padding_mask=padding_mask,
            controlled_building_indices=controlled_building_indices,
        )

    def _encode_action_tokens(
        self,
        actions: torch.Tensor,
        base_actions: Optional[torch.Tensor],
        action_span: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Project action scalars to d_model embeddings."""
        del device
        if self.action_input_mode == "final":
            action_input = actions.unsqueeze(-1)
        elif self.action_input_mode == "final_base_delta":
            if base_actions is None:
                raise ValueError("base_actions required for action_input_mode='final_base_delta'")
            delta = actions - base_actions
            action_input = torch.stack([actions, base_actions, delta], dim=-1)
        elif self.action_input_mode == "final_base_delta_normalized":
            if base_actions is None:
                raise ValueError(
                    "base_actions required for action_input_mode='final_base_delta_normalized'"
                )
            delta = (actions - base_actions) / max(action_span, 1e-8)
            action_input = torch.stack([actions, base_actions, delta], dim=-1)
        else:
            raise ValueError(f"Unknown action_input_mode: {self.action_input_mode!r}")
        return self.action_projection(action_input)
