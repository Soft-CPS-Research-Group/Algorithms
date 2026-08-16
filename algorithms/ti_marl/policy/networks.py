"""Type-shared relational actor and centralized set critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Beta, Categorical
from torch.nn import functional as F

from algorithms.ti_marl.contracts.enums import HealthState
from algorithms.ti_marl.contracts.models import (
    ActionDecision,
    ActionGroupInstance,
    InterfaceSnapshot,
    LocalActionBundle,
    ObservationPart,
)


HEALTH_INDEX = {state: index for index, state in enumerate(HealthState)}


@dataclass
class ActorEvaluation:
    bundles: Tuple[LocalActionBundle, ...]
    log_prob_by_agent: Mapping[str, Tensor]
    entropy_by_agent: Mapping[str, Tensor]
    latent_by_agent: Mapping[str, Tensor]


class RelationalMessageLayer(nn.Module):
    """Cardinality-independent local relational message passing."""

    def __init__(self, d_model: int):
        super().__init__()
        self.self_projection = nn.Linear(d_model, d_model)
        self.neighbour_projection = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("RelationalMessageLayer expects [tokens, d_model]")
        if tokens.shape[0] == 0:
            return tokens
        aggregate = tokens.mean(dim=0, keepdim=True).expand_as(tokens)
        updated = self.norm(tokens + self.self_projection(tokens) + self.neighbour_projection(aggregate))
        return self.norm(updated + self.feed_forward(updated))


class TypedSnapshotEncoder(nn.Module):
    """Encode typed observation parts into one latent per building."""

    def __init__(self, type_registry: Mapping[str, object], d_model: int, relation_layers: int):
        super().__init__()
        entity_types = dict(type_registry.get("entity_types", {}))
        if not entity_types:
            raise ValueError("TI-MARL type registry must define entity_types")
        self.feature_width = int(type_registry.get("feature_width", 16))
        self.d_model = int(d_model)
        input_width = self.feature_width + len(HealthState) + 2
        self.type_encoders = nn.ModuleDict(
            {
                entity_type: nn.Sequential(
                    nn.Linear(input_width, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for entity_type in sorted(entity_types)
            }
        )
        self.relation_index = {
            entity_type: index for index, entity_type in enumerate(sorted(entity_types))
        }
        self.relation_embedding = nn.Embedding(len(self.relation_index), d_model)
        self.semantic_types = sorted(
            {str(config.get("semantic_type", "local_energy")) for config in entity_types.values()}
        )
        self.semantic_index = {name: index for index, name in enumerate(self.semantic_types)}
        self.semantic_embedding = nn.Embedding(len(self.semantic_types), d_model)
        self.layers = nn.ModuleList(
            [RelationalMessageLayer(d_model) for _ in range(int(relation_layers))]
        )
        self.pool_query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())

    def forward(self, snapshot: InterfaceSnapshot, agent_id: str, device: torch.device) -> Tensor:
        entity_type_by_id = {entity.entity_id: entity.entity_type for entity in snapshot.entities}
        parts = tuple(part for part in snapshot.parts_for(agent_id) if part.valid)
        tokens = []
        for part in parts:
            entity_type = entity_type_by_id.get(part.source_entity_id)
            if entity_type not in self.type_encoders:
                continue
            features = self._feature_tensor(part, device)
            encoded = self.type_encoders[entity_type](features)
            # Every per-agent token is connected through the local building
            # star; the entity type identifies the declared relation role
            # (district/building/storage/charger/EV/deferrable/PV).
            encoded = encoded + self.relation_embedding(
                torch.tensor(self.relation_index[entity_type], dtype=torch.long, device=device)
            )
            semantic_index = self.semantic_index.get(part.semantic_type, 0)
            encoded = encoded + self.semantic_embedding(
                torch.tensor(semantic_index, dtype=torch.long, device=device)
            )
            tokens.append(encoded)
        if not tokens:
            return torch.zeros(self.d_model, dtype=torch.float32, device=device)
        token_tensor = torch.stack(tokens, dim=0)
        for layer in self.layers:
            token_tensor = layer(token_tensor)
        scores = (token_tensor * self.pool_query).sum(dim=-1) / np.sqrt(float(self.d_model))
        weights = torch.softmax(scores, dim=0)
        return self.pool_projection((weights.unsqueeze(-1) * token_tensor).sum(dim=0))

    def _feature_tensor(self, part: ObservationPart, device: torch.device) -> Tensor:
        values = torch.zeros(self.feature_width, dtype=torch.float32, device=device)
        count = min(len(part.values), self.feature_width)
        if count:
            raw = torch.tensor(part.values[:count], dtype=torch.float32, device=device)
            values[:count] = torch.sign(raw) * torch.log1p(torch.abs(raw))
        health = torch.zeros(len(HealthState), dtype=torch.float32, device=device)
        health[HEALTH_INDEX[part.health]] = 1.0
        flags = torch.tensor([float(part.valid), float(part.estimated)], device=device)
        return torch.cat((values, health, flags), dim=0)


class TypedActor(nn.Module):
    """Shared grouped actor with categorical ports and Beta fractions."""

    def __init__(
        self,
        type_registry: Mapping[str, object],
        *,
        d_model: int = 128,
        attention_heads: int = 4,
        relation_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.encoder = TypedSnapshotEncoder(type_registry, d_model, relation_layers)
        action_types = dict(type_registry.get("action_group_types", {}))
        if not action_types:
            raise ValueError("TI-MARL type registry must define action_group_types")
        self.group_modes = {
            name: tuple(str(mode) for mode in config.get("modes", []))
            for name, config in sorted(action_types.items())
        }
        self.group_index = {name: index for index, name in enumerate(self.group_modes)}
        self.group_embedding = nn.Embedding(len(self.group_modes), d_model)
        self.group_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=int(attention_heads),
            batch_first=True,
        )
        self.group_norm = nn.LayerNorm(d_model)
        self.mode_heads = nn.ModuleDict(
            {name: nn.Linear(d_model, len(modes)) for name, modes in self.group_modes.items()}
        )
        self.beta_heads = nn.ModuleDict(
            {name: nn.Linear(d_model, 2) for name in self.group_modes}
        )

    def forward(
        self,
        snapshot: InterfaceSnapshot,
        *,
        deterministic: bool = False,
        decisions: Mapping[str, Mapping[str, ActionDecision]] | None = None,
    ) -> ActorEvaluation:
        device = next(self.parameters()).device
        bundles = []
        log_probs: Dict[str, Tensor] = {}
        entropies: Dict[str, Tensor] = {}
        latents: Dict[str, Tensor] = {}
        for agent_id in snapshot.agent_ids:
            local_latent = self.encoder(snapshot, agent_id, device)
            latents[agent_id] = local_latent
            groups = snapshot.groups_for(agent_id)
            if groups:
                contexts = torch.stack(
                    [
                        local_latent
                        + self.group_embedding(
                            torch.tensor(self.group_index[group.group_type], device=device)
                        )
                        for group in groups
                    ],
                    dim=0,
                ).unsqueeze(0)
                attended, _ = self.group_attention(contexts, contexts, contexts, need_weights=False)
                contexts = self.group_norm(contexts + attended).squeeze(0)
            else:
                contexts = torch.empty((0, self.d_model), device=device)

            agent_decisions = []
            agent_log_prob = torch.zeros((), device=device)
            agent_entropy = torch.zeros((), device=device)
            expected = (decisions or {}).get(agent_id, {})
            for index, group in enumerate(groups):
                decision, log_prob, entropy = self._group_decision(
                    group,
                    contexts[index],
                    deterministic=deterministic,
                    expected=expected.get(group.group_id),
                )
                agent_decisions.append(decision)
                agent_log_prob = agent_log_prob + log_prob
                agent_entropy = agent_entropy + entropy
            bundles.append(LocalActionBundle(agent_id=agent_id, decisions=tuple(agent_decisions)))
            log_probs[agent_id] = agent_log_prob
            entropies[agent_id] = agent_entropy
        return ActorEvaluation(tuple(bundles), log_probs, entropies, latents)

    def _group_decision(
        self,
        group: ActionGroupInstance,
        context: Tensor,
        *,
        deterministic: bool,
        expected: ActionDecision | None,
    ) -> Tuple[ActionDecision, Tensor, Tensor]:
        modes = self.group_modes[group.group_type]
        logits = self.mode_heads[group.group_type](context)
        valid_by_mode = {port.mode: port.valid and group.enabled for port in group.ports}
        mask = torch.tensor(
            [mode == "IDLE" or valid_by_mode.get(mode, False) for mode in modes],
            dtype=torch.bool,
            device=context.device,
        )
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        categorical = Categorical(logits=masked_logits)

        beta_params = F.softplus(self.beta_heads[group.group_type](context)) + 1.0
        beta_distribution = Beta(beta_params[0], beta_params[1])
        if expected is not None:
            mode_index = int(expected.mode_index)
            if mode_index < 0 or mode_index >= len(modes) or not bool(mask[mode_index]):
                raise ValueError(f"Stored invalid TI-MARL mode for {group.group_id}: {mode_index}")
            fraction = torch.tensor(
                float(np.clip(expected.fraction, 1.0e-6, 1.0 - 1.0e-6)),
                dtype=torch.float32,
                device=context.device,
            )
        elif deterministic:
            mode_index = int(torch.argmax(masked_logits).item())
            fraction = beta_params[0] / beta_params.sum()
        else:
            mode_index = int(categorical.sample().item())
            fraction = beta_distribution.rsample()

        mode = modes[mode_index]
        categorical_log_prob = categorical.log_prob(
            torch.tensor(mode_index, dtype=torch.long, device=context.device)
        )
        categorical_entropy = categorical.entropy()
        parameterized = "CHARGE" in mode or "DISCHARGE" in mode
        if mode == "IDLE" or not parameterized:
            fraction_value = 0.0 if mode == "IDLE" else 1.0
            log_prob = categorical_log_prob
            entropy = categorical_entropy
        else:
            fraction = torch.clamp(fraction, 1.0e-6, 1.0 - 1.0e-6)
            fraction_value = float(fraction.detach().cpu())
            log_prob = categorical_log_prob + beta_distribution.log_prob(fraction)
            entropy = categorical_entropy + beta_distribution.entropy()
        return (
            ActionDecision(
                group_id=group.group_id,
                mode=mode,
                fraction=fraction_value,
                mode_index=mode_index,
                raw_log_prob=float(log_prob.detach().cpu()),
            ),
            log_prob,
            entropy,
        )


class CentralSetCritic(nn.Module):
    """Permutation-equivariant critic returning one value per stable agent ID."""

    def __init__(
        self,
        type_registry: Mapping[str, object],
        *,
        d_model: int = 128,
        relation_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = TypedSnapshotEncoder(type_registry, d_model, relation_layers)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, snapshot: InterfaceSnapshot) -> Mapping[str, Tensor]:
        device = next(self.parameters()).device
        local = {
            agent_id: self.encoder(snapshot, agent_id, device)
            for agent_id in snapshot.agent_ids
        }
        if not local:
            return {}
        community = torch.stack([local[key] for key in sorted(local)], dim=0).mean(dim=0)
        return {
            agent_id: self.value_head(torch.cat((latent, community), dim=-1)).squeeze(-1)
            for agent_id, latent in local.items()
        }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
