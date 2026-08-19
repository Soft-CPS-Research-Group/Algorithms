"""Type-shared relational actor and local/centralized critics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Mapping, Sequence, Tuple

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


@dataclass
class ActorReplayEvaluation:
    log_prob_by_step: Tuple[Mapping[str, Tensor], ...]
    entropy_by_step: Tuple[Mapping[str, Tensor], ...]


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
        groups = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
        return self.forward_grouped(tokens, groups, 1)

    def forward_grouped(
        self,
        tokens: Tensor,
        group_indices: Tensor,
        group_count: int,
    ) -> Tensor:
        """Apply the same set layer to several packed variable-size sets."""
        if tokens.ndim != 2:
            raise ValueError("RelationalMessageLayer expects [tokens, d_model]")
        if tokens.shape[0] == 0:
            return tokens
        aggregate = _group_mean(tokens, group_indices, group_count)[group_indices]
        updated = self.norm(tokens + self.self_projection(tokens) + self.neighbour_projection(aggregate))
        return self.norm(updated + self.feed_forward(updated))


def _group_mean(tokens: Tensor, group_indices: Tensor, group_count: int) -> Tensor:
    sums = torch.zeros(
        (int(group_count), tokens.shape[-1]),
        dtype=tokens.dtype,
        device=tokens.device,
    ).index_add(0, group_indices, tokens)
    counts = torch.zeros(
        int(group_count), dtype=tokens.dtype, device=tokens.device
    ).index_add(
        0,
        group_indices,
        torch.ones(group_indices.shape[0], dtype=tokens.dtype, device=tokens.device),
    )
    return sums / counts.clamp_min(1.0).unsqueeze(-1)


class TypedSnapshotEncoder(nn.Module):
    """Hierarchical observation → channel → sensor → local latent encoder."""

    def __init__(self, type_registry: Mapping[str, object], d_model: int, relation_layers: int):
        super().__init__()
        semantic_types = tuple(
            sorted(str(item) for item in type_registry.get("semantic_types", []))
        )
        if not semantic_types:
            raise ValueError("TI-MARL type registry must define semantic_types")
        self.d_model = int(d_model)
        input_width = 4 + len(HealthState) + 2 + 1 + 1 + 3
        self.observation_encoder = nn.Sequential(
            nn.Linear(input_width, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.semantic_types = list(semantic_types)
        self.semantic_index = {name: index for index, name in enumerate(self.semantic_types)}
        self.semantic_embedding = nn.Embedding(len(self.semantic_types), d_model)
        self.channel_encoder = RelationalMessageLayer(d_model)
        self.sensor_encoder = RelationalMessageLayer(d_model)
        self.layers = nn.ModuleList(
            [RelationalMessageLayer(d_model) for _ in range(int(relation_layers))]
        )
        self.role_embedding = nn.Embedding(3, d_model)
        self.agent_type_embedding = nn.Embedding(5, d_model)
        self.pool_query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())

    def forward(self, snapshot: InterfaceSnapshot, agent_id: str, device: torch.device) -> Tensor:
        return self.forward_many(((snapshot, agent_id),), device)[0]

    def forward_many(
        self,
        requests: Sequence[tuple[InterfaceSnapshot, str]],
        device: torch.device,
    ) -> Tensor:
        """Encode packed snapshot-agent pairs without changing set semantics."""
        request_count = len(requests)
        if request_count == 0:
            return torch.empty((0, self.d_model), device=device)

        # Missing/invalid samples remain explicit tokens: health and validity
        # are policy inputs, while the value itself is zero-filled by the TIC.
        parts_by_request = [
            tuple(part for part in snapshot.parts_for(agent_id) if part.policy_input)
            for snapshot, agent_id in requests
        ]
        all_parts = [part for parts in parts_by_request for part in parts]
        if not all_parts:
            return torch.zeros(
                (request_count, self.d_model), dtype=torch.float32, device=device
            )

        feature_batch = torch.as_tensor(
            np.stack([self._feature_array(part) for part in all_parts]),
            dtype=torch.float32,
            device=device,
        )
        semantic_indices = torch.tensor(
            [self.semantic_index.get(part.semantic_type, 0) for part in all_parts],
            dtype=torch.long,
            device=device,
        )
        encoded_batch = self.observation_encoder(feature_batch) + self.semantic_embedding(
            semantic_indices
        )

        channel_lookup: Dict[tuple[int, str, str], int] = {}
        channel_request_indices: list[int] = []
        channel_sensor_ids: list[str] = []
        observation_channel_indices: list[int] = []
        for request_index, parts in enumerate(parts_by_request):
            for part in parts:
                key = (request_index, part.sensor_id, part.channel_id)
                channel_index = channel_lookup.get(key)
                if channel_index is None:
                    channel_index = len(channel_lookup)
                    channel_lookup[key] = channel_index
                    channel_request_indices.append(request_index)
                    channel_sensor_ids.append(part.sensor_id)
                observation_channel_indices.append(channel_index)
        observation_channels = torch.tensor(
            observation_channel_indices, dtype=torch.long, device=device
        )
        encoded_observations = self.channel_encoder.forward_grouped(
            encoded_batch,
            observation_channels,
            len(channel_lookup),
        )
        channel_latents = _group_mean(
            encoded_observations,
            observation_channels,
            len(channel_lookup),
        )

        sensor_lookup: Dict[tuple[int, str], int] = {}
        sensor_request_indices: list[int] = []
        channel_sensor_indices: list[int] = []
        for request_index, sensor_id in zip(
            channel_request_indices, channel_sensor_ids
        ):
            key = (request_index, sensor_id)
            sensor_index = sensor_lookup.get(key)
            if sensor_index is None:
                sensor_index = len(sensor_lookup)
                sensor_lookup[key] = sensor_index
                sensor_request_indices.append(request_index)
            channel_sensor_indices.append(sensor_index)
        channel_sensors = torch.tensor(
            channel_sensor_indices, dtype=torch.long, device=device
        )
        encoded_channels = self.sensor_encoder.forward_grouped(
            channel_latents,
            channel_sensors,
            len(sensor_lookup),
        )
        sensor_latents = _group_mean(
            encoded_channels,
            channel_sensors,
            len(sensor_lookup),
        )
        sensor_requests = torch.tensor(
            sensor_request_indices, dtype=torch.long, device=device
        )
        for layer in self.layers:
            sensor_latents = layer.forward_grouped(
                sensor_latents,
                sensor_requests,
                request_count,
            )

        scores = (sensor_latents * self.pool_query).sum(dim=-1) / np.sqrt(
            float(self.d_model)
        )
        maxima = torch.full(
            (request_count,),
            -torch.inf,
            dtype=scores.dtype,
            device=device,
        ).scatter_reduce(
            0,
            sensor_requests,
            scores,
            reduce="amax",
            include_self=True,
        )
        exponentials = torch.exp(scores - maxima[sensor_requests])
        denominators = torch.zeros(
            request_count, dtype=scores.dtype, device=device
        ).index_add(0, sensor_requests, exponentials)
        weights = exponentials / denominators[sensor_requests].clamp_min(1.0e-12)
        pooled = torch.zeros(
            (request_count, self.d_model),
            dtype=sensor_latents.dtype,
            device=device,
        ).index_add(0, sensor_requests, weights.unsqueeze(-1) * sensor_latents)

        roles: list[int] = []
        agent_types: list[int] = []
        for snapshot, agent_id in requests:
            metadata = {
                row[0]: (row[1], row[2]) for row in snapshot.agent_metadata
            }.get(agent_id, ("consumer", "other"))
            roles.append(
                {"consumer": 0, "producer": 1, "prosumer": 2}.get(metadata[0], 0)
            )
            agent_types.append(
                {
                    "residential": 0,
                    "office": 1,
                    "commercial": 2,
                    "industrial": 3,
                    "other": 4,
                }.get(metadata[1], 4)
            )
        pooled = (
            pooled
            + self.role_embedding(torch.tensor(roles, dtype=torch.long, device=device))
            + self.agent_type_embedding(
                torch.tensor(agent_types, dtype=torch.long, device=device)
            )
        )
        encoded = self.pool_projection(pooled)
        active = torch.zeros(
            request_count, dtype=torch.bool, device=device
        )
        active[sensor_requests] = True
        return torch.where(active.unsqueeze(-1), encoded, torch.zeros_like(encoded))

    @staticmethod
    def _feature_array(part: ObservationPart) -> np.ndarray:
        raw = np.asarray(part.values or (0.0,), dtype=np.float32)
        transformed = np.sign(raw) * np.log1p(np.abs(raw))
        values = np.asarray(
            (
                transformed[0],
                transformed.mean(),
                transformed.min(),
                transformed.max(),
            ),
            dtype=np.float32,
        )
        health = np.zeros(len(HealthState), dtype=np.float32)
        health[HEALTH_INDEX[part.health]] = 1.0
        flags = np.asarray(
            (float(part.valid), float(part.estimated)), dtype=np.float32
        )
        age = np.asarray(
            (np.log1p(max(part.age_seconds, 0.0)),), dtype=np.float32
        )
        unit_hash = int(hashlib.sha256(part.unit.encode("utf-8")).hexdigest()[:8], 16)
        unit = np.asarray(
            ((unit_hash % 1000) / 999.0,), dtype=np.float32
        )
        criticality = np.zeros(3, dtype=np.float32)
        criticality[{"advisory": 0, "operational": 1, "safety": 2}.get(part.criticality, 1)] = 1.0
        return np.concatenate((values, health, flags, age, unit, criticality))


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
        materialize_bundles: bool = True,
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
                group_indices = torch.tensor(
                    [self.group_index[group.group_type] for group in groups],
                    dtype=torch.long,
                    device=device,
                )
                contexts = (
                    local_latent.unsqueeze(0) + self.group_embedding(group_indices)
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
                    materialize_decision=materialize_bundles,
                )
                if decision is not None:
                    agent_decisions.append(decision)
                agent_log_prob = agent_log_prob + log_prob
                agent_entropy = agent_entropy + entropy
            if materialize_bundles:
                bundles.append(
                    LocalActionBundle(
                        agent_id=agent_id,
                        decisions=tuple(agent_decisions),
                    )
                )
            log_probs[agent_id] = agent_log_prob
            entropies[agent_id] = agent_entropy
        return ActorEvaluation(tuple(bundles), log_probs, entropies, latents)

    def evaluate_actions_many(
        self,
        items: Sequence[
            tuple[
                InterfaceSnapshot,
                Mapping[str, Mapping[str, ActionDecision]],
            ]
        ],
    ) -> ActorReplayEvaluation:
        """Evaluate packed rollout actions without materializing runtime bundles."""
        device = next(self.parameters()).device
        requests = [
            (snapshot, agent_id)
            for snapshot, _decisions in items
            for agent_id in snapshot.agent_ids
        ]
        request_keys = [
            (step_index, agent_id)
            for step_index, (snapshot, _decisions) in enumerate(items)
            for agent_id in snapshot.agent_ids
        ]
        if not requests:
            empty = tuple({} for _item in items)
            return ActorReplayEvaluation(empty, empty)

        latents = self.encoder.forward_many(requests, device)
        group_entries: list[
            tuple[int, ActionGroupInstance, ActionDecision]
        ] = []
        group_counts = [0] * len(requests)
        for request_index, ((snapshot, agent_id), (step_index, _key_agent)) in enumerate(
            zip(requests, request_keys)
        ):
            expected_by_group = items[step_index][1].get(agent_id, {})
            for group in snapshot.groups_for(agent_id):
                expected = expected_by_group.get(group.group_id)
                if expected is None:
                    raise ValueError(
                        f"Missing stored TI-MARL action for {group.group_id}"
                    )
                group_entries.append((request_index, group, expected))
                group_counts[request_index] += 1

        request_log_probs = torch.zeros(len(requests), device=device)
        request_entropies = torch.zeros(len(requests), device=device)
        if group_entries:
            max_groups = max(group_counts)
            positions = []
            seen = [0] * len(requests)
            for request_index, _group, _expected in group_entries:
                positions.append(seen[request_index])
                seen[request_index] += 1
            request_indices = torch.tensor(
                [entry[0] for entry in group_entries],
                dtype=torch.long,
                device=device,
            )
            position_indices = torch.tensor(
                positions, dtype=torch.long, device=device
            )
            type_indices = torch.tensor(
                [self.group_index[entry[1].group_type] for entry in group_entries],
                dtype=torch.long,
                device=device,
            )
            flat_contexts = (
                latents[request_indices] + self.group_embedding(type_indices)
            )
            padded = torch.zeros(
                (len(requests), max_groups, self.d_model),
                dtype=flat_contexts.dtype,
                device=device,
            )
            padded[request_indices, position_indices] = flat_contexts
            padding_mask = torch.ones(
                (len(requests), max_groups), dtype=torch.bool, device=device
            )
            padding_mask[request_indices, position_indices] = False
            active_request_values = [
                index for index, count in enumerate(group_counts) if count
            ]
            active_request_indices = torch.tensor(
                active_request_values,
                dtype=torch.long,
                device=device,
            )
            attended, _ = self.group_attention(
                padded[active_request_indices],
                padded[active_request_indices],
                padded[active_request_indices],
                key_padding_mask=padding_mask[active_request_indices],
                need_weights=False,
            )
            attended = self.group_norm(
                padded[active_request_indices] + attended
            )
            active_row_by_request = {
                int(request_index): row_index
                for row_index, request_index in enumerate(
                    active_request_values
                )
            }
            active_rows = torch.tensor(
                [active_row_by_request[entry[0]] for entry in group_entries],
                dtype=torch.long,
                device=device,
            )
            flat_contexts = attended[active_rows, position_indices]

            for group_type, modes in self.group_modes.items():
                selected = [
                    index
                    for index, entry in enumerate(group_entries)
                    if entry[1].group_type == group_type
                ]
                if not selected:
                    continue
                selected_tensor = torch.tensor(
                    selected, dtype=torch.long, device=device
                )
                contexts = flat_contexts[selected_tensor]
                entries = [group_entries[index] for index in selected]
                logits = self.mode_heads[group_type](contexts)
                mask_values = [
                    [
                        mode == "IDLE"
                        or {
                            port.mode: port.valid and group.enabled
                            for port in group.ports
                        }.get(mode, False)
                        for mode in modes
                    ]
                    for _request_index, group, _expected in entries
                ]
                mode_indices = [entry[2].mode_index for entry in entries]
                for row, mode_index in enumerate(mode_indices):
                    if (
                        mode_index < 0
                        or mode_index >= len(modes)
                        or not mask_values[row][mode_index]
                    ):
                        raise ValueError(
                            "Stored invalid TI-MARL mode for "
                            f"{entries[row][1].group_id}: {mode_index}"
                        )
                mask = torch.tensor(
                    mask_values, dtype=torch.bool, device=device
                )
                categorical = Categorical(
                    logits=logits.masked_fill(
                        ~mask, torch.finfo(logits.dtype).min
                    )
                )
                selected_modes = torch.tensor(
                    mode_indices, dtype=torch.long, device=device
                )
                log_prob = categorical.log_prob(selected_modes)
                entropy = categorical.entropy()
                beta_params = F.softplus(self.beta_heads[group_type](contexts)) + 1.0
                beta_distribution = Beta(beta_params[:, 0], beta_params[:, 1])
                fractions = torch.tensor(
                    [
                        float(np.clip(entry[2].fraction, 1.0e-6, 1.0 - 1.0e-6))
                        for entry in entries
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                parameterized = torch.tensor(
                    [
                        modes[mode_index].startswith(("CHARGE_", "DISCHARGE_"))
                        for mode_index in mode_indices
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                log_prob = log_prob + torch.where(
                    parameterized,
                    beta_distribution.log_prob(fractions),
                    torch.zeros_like(log_prob),
                )
                entropy = entropy + torch.where(
                    parameterized,
                    beta_distribution.entropy(),
                    torch.zeros_like(entropy),
                )
                target_requests = request_indices[selected_tensor]
                request_log_probs = request_log_probs.index_add(
                    0, target_requests, log_prob
                )
                request_entropies = request_entropies.index_add(
                    0, target_requests, entropy
                )

        log_prob_by_step: list[Dict[str, Tensor]] = [dict() for _item in items]
        entropy_by_step: list[Dict[str, Tensor]] = [dict() for _item in items]
        for request_index, (step_index, agent_id) in enumerate(request_keys):
            log_prob_by_step[step_index][agent_id] = request_log_probs[request_index]
            entropy_by_step[step_index][agent_id] = request_entropies[request_index]
        return ActorReplayEvaluation(
            tuple(log_prob_by_step), tuple(entropy_by_step)
        )

    def _group_decision(
        self,
        group: ActionGroupInstance,
        context: Tensor,
        *,
        deterministic: bool,
        expected: ActionDecision | None,
        materialize_decision: bool,
    ) -> Tuple[ActionDecision | None, Tensor, Tensor]:
        modes = self.group_modes[group.group_type]
        logits = self.mode_heads[group.group_type](context)
        valid_by_mode = {port.mode: port.valid and group.enabled for port in group.ports}
        mask_values = [
            mode == "IDLE" or valid_by_mode.get(mode, False) for mode in modes
        ]
        mask = torch.tensor(
            mask_values,
            dtype=torch.bool,
            device=context.device,
        )
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        categorical = Categorical(logits=masked_logits)

        beta_params = F.softplus(self.beta_heads[group.group_type](context)) + 1.0
        beta_distribution = Beta(beta_params[0], beta_params[1])
        if expected is not None:
            mode_index = int(expected.mode_index)
            if (
                mode_index < 0
                or mode_index >= len(modes)
                or not mask_values[mode_index]
            ):
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
        parameterized = mode.startswith(("CHARGE_", "DISCHARGE_"))
        if mode == "IDLE" or not parameterized:
            log_prob = categorical_log_prob
            entropy = categorical_entropy
        else:
            fraction = torch.clamp(fraction, 1.0e-6, 1.0 - 1.0e-6)
            log_prob = categorical_log_prob + beta_distribution.log_prob(fraction)
            entropy = categorical_entropy + beta_distribution.entropy()
        decision = None
        if materialize_decision:
            fraction_value = (
                0.0
                if mode == "IDLE"
                else 1.0
                if not parameterized
                else float(fraction.detach().cpu())
            )
            decision = ActionDecision(
                group_id=group.group_id,
                mode=mode,
                fraction=fraction_value,
                mode_index=mode_index,
                raw_log_prob=float(log_prob.detach().cpu()),
            )
        return decision, log_prob, entropy


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
        return self.forward_many((snapshot,))[0]

    def forward_many(
        self,
        snapshots: Sequence[InterfaceSnapshot],
    ) -> Tuple[Mapping[str, Tensor], ...]:
        device = next(self.parameters()).device
        requests = [
            (snapshot, agent_id)
            for snapshot in snapshots
            for agent_id in snapshot.agent_ids
        ]
        if not requests:
            return tuple({} for _snapshot in snapshots)
        local = self.encoder.forward_many(requests, device)
        snapshot_indices = torch.tensor(
            [
                snapshot_index
                for snapshot_index, snapshot in enumerate(snapshots)
                for _agent_id in snapshot.agent_ids
            ],
            dtype=torch.long,
            device=device,
        )
        community = _group_mean(local, snapshot_indices, len(snapshots))
        values = self.value_head(
            torch.cat((local, community[snapshot_indices]), dim=-1)
        ).squeeze(-1)
        result: list[Dict[str, Tensor]] = [dict() for _snapshot in snapshots]
        offset = 0
        for snapshot_index, snapshot in enumerate(snapshots):
            for agent_id in snapshot.agent_ids:
                result[snapshot_index][agent_id] = values[offset]
                offset += 1
        return tuple(result)


class LocalTypedCritic(nn.Module):
    """Parameter-shared critic whose value depends only on the local interface."""

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
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, snapshot: InterfaceSnapshot) -> Mapping[str, Tensor]:
        return self.forward_many((snapshot,))[0]

    def forward_many(
        self,
        snapshots: Sequence[InterfaceSnapshot],
    ) -> Tuple[Mapping[str, Tensor], ...]:
        device = next(self.parameters()).device
        requests = [
            (snapshot, agent_id)
            for snapshot in snapshots
            for agent_id in snapshot.agent_ids
        ]
        if not requests:
            return tuple({} for _snapshot in snapshots)
        values = self.value_head(
            self.encoder.forward_many(requests, device)
        ).squeeze(-1)
        result: list[Dict[str, Tensor]] = [dict() for _snapshot in snapshots]
        offset = 0
        for snapshot_index, snapshot in enumerate(snapshots):
            for agent_id in snapshot.agent_ids:
                result[snapshot_index][agent_id] = values[offset]
                offset += 1
        return tuple(result)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
