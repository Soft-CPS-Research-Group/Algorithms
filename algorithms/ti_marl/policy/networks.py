"""Type-shared relational actor and local/centralized critics."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import re
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
    log_prob_by_group_step: Tuple[Mapping[str, Mapping[str, Tensor]], ...]
    entropy_by_group_step: Tuple[Mapping[str, Mapping[str, Tensor]], ...]
    mode_log_prob_by_group_step: Tuple[Mapping[str, Mapping[str, Tensor]], ...]
    predicted_mode_by_group_step: Tuple[Mapping[str, Mapping[str, Tensor]], ...]
    predicted_fraction_by_group_step: Tuple[
        Mapping[str, Mapping[str, Tensor]], ...
    ]


@dataclass
class _PreparedSnapshotRequests:
    """Immutable tensor inputs shared by repeated actor replay evaluations.

    TI-MARL BC revisits the same typed snapshots for many epochs.  Preparing
    their NumPy features and structural indices on every visit used to make
    warm-start training CPU-bound even when the actor lived on a GPU.
    """

    parts_by_request: tuple[tuple[ObservationPart, ...], ...]
    feature_batch: np.ndarray
    semantic_indices: np.ndarray
    unit_indices: np.ndarray
    use_indices: np.ndarray
    scope_indices: np.ndarray
    observation_channels: np.ndarray
    channel_types: np.ndarray
    channel_sensors: np.ndarray
    sensor_types: np.ndarray
    sensor_requests: np.ndarray
    roles: np.ndarray
    agent_types: np.ndarray
    channel_count: int
    sensor_count: int
    request_count: int
    tensors: dict[tuple[str, str], Tensor] = field(default_factory=dict)

    def tensor(
        self,
        name: str,
        values: np.ndarray,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        key = (name, str(device))
        cached = self.tensors.get(key)
        if cached is None:
            cached = torch.as_tensor(values, dtype=dtype, device=device)
            self.tensors[key] = cached
        return cached


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
        identity_width = 8
        temporal_width = 3
        input_width = (
            4
            + len(HealthState)
            + 2
            + 1
            + 3
            + identity_width
            + temporal_width
        )
        self.observation_encoder = nn.Sequential(
            nn.Linear(input_width, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.semantic_types = list(semantic_types)
        self.semantic_index = {name: index for index, name in enumerate(self.semantic_types)}
        self.semantic_embedding = nn.Embedding(len(self.semantic_types), d_model)
        self.sensor_types, self.sensor_index, self.sensor_type_embedding = (
            self._typed_embedding(type_registry, "sensor_types", d_model)
        )
        self.channel_types, self.channel_index, self.channel_type_embedding = (
            self._typed_embedding(type_registry, "channel_types", d_model)
        )
        self.unit_types, self.unit_index, self.unit_embedding = self._typed_embedding(
            type_registry, "unit_types", d_model
        )
        self.observation_uses, self.use_index, self.use_embedding = (
            self._typed_embedding(type_registry, "observation_uses", d_model)
        )
        self.scopes, self.scope_index, self.scope_embedding = self._typed_embedding(
            type_registry, "scopes", d_model
        )
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
        self._replay_preparation_cache_enabled = False
        self._replay_preparation_cache: dict[
            tuple[tuple[int, str], ...], _PreparedSnapshotRequests
        ] = {}

    def begin_replay_preparation_cache(self) -> None:
        """Cache constant typed inputs while a fixed replay dataset is reused."""

        self._replay_preparation_cache.clear()
        self._replay_preparation_cache_enabled = True

    def end_replay_preparation_cache(self) -> None:
        """Release cached CPU/GPU inputs after replay/BC training."""

        self._replay_preparation_cache_enabled = False
        self._replay_preparation_cache.clear()

    def forward(self, snapshot: InterfaceSnapshot, agent_id: str, device: torch.device) -> Tensor:
        return self.forward_many(((snapshot, agent_id),), device)[0]

    def forward_many(
        self,
        requests: Sequence[tuple[InterfaceSnapshot, str]],
        device: torch.device,
    ) -> Tensor:
        """Encode packed snapshot-agent pairs without changing set semantics."""
        latents, _parts, _tokens = self.forward_many_with_parts(requests, device)
        return latents

    def forward_many_with_parts(
        self,
        requests: Sequence[tuple[InterfaceSnapshot, str]],
        device: torch.device,
    ) -> tuple[Tensor, tuple[tuple[ObservationPart, ...], ...], Tensor]:
        """Encode agents and retain their already encoded observation tokens."""
        request_count = len(requests)
        if request_count == 0:
            return (
                torch.empty((0, self.d_model), device=device),
                (),
                torch.empty((0, self.d_model), device=device),
            )

        prepared = self._prepared_requests(requests)
        if prepared.feature_batch.shape[0] == 0:
            return (
                torch.zeros(
                    (request_count, self.d_model),
                    dtype=torch.float32,
                    device=device,
                ),
                prepared.parts_by_request,
                torch.empty((0, self.d_model), dtype=torch.float32, device=device),
            )

        encoded_batch = self._encode_prepared_observations(prepared, device)
        observation_channels = prepared.tensor(
            "observation_channels",
            prepared.observation_channels,
            dtype=torch.long,
            device=device,
        )
        encoded_observations = self.channel_encoder.forward_grouped(
            encoded_batch,
            observation_channels,
            prepared.channel_count,
        )
        channel_latents = _group_mean(
            encoded_observations,
            observation_channels,
            prepared.channel_count,
        )
        channel_type_indices = prepared.tensor(
            "channel_types",
            prepared.channel_types,
            dtype=torch.long,
            device=device,
        )
        channel_latents = channel_latents + self.channel_type_embedding(
            channel_type_indices
        )

        channel_sensors = prepared.tensor(
            "channel_sensors",
            prepared.channel_sensors,
            dtype=torch.long,
            device=device,
        )
        encoded_channels = self.sensor_encoder.forward_grouped(
            channel_latents,
            channel_sensors,
            prepared.sensor_count,
        )
        sensor_latents = _group_mean(
            encoded_channels,
            channel_sensors,
            prepared.sensor_count,
        )
        sensor_type_indices = prepared.tensor(
            "sensor_types",
            prepared.sensor_types,
            dtype=torch.long,
            device=device,
        )
        sensor_latents = sensor_latents + self.sensor_type_embedding(
            sensor_type_indices
        )
        sensor_requests = prepared.tensor(
            "sensor_requests",
            prepared.sensor_requests,
            dtype=torch.long,
            device=device,
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

        pooled = (
            pooled
            + self.role_embedding(
                prepared.tensor(
                    "roles", prepared.roles, dtype=torch.long, device=device
                )
            )
            + self.agent_type_embedding(
                prepared.tensor(
                    "agent_types",
                    prepared.agent_types,
                    dtype=torch.long,
                    device=device,
                )
            )
        )
        encoded = self.pool_projection(pooled)
        active = torch.zeros(
            request_count, dtype=torch.bool, device=device
        )
        active[sensor_requests] = True
        return (
            torch.where(active.unsqueeze(-1), encoded, torch.zeros_like(encoded)),
            prepared.parts_by_request,
            encoded_batch,
        )

    def _prepared_requests(
        self,
        requests: Sequence[tuple[InterfaceSnapshot, str]],
    ) -> _PreparedSnapshotRequests:
        cache_key = tuple((id(snapshot), agent_id) for snapshot, agent_id in requests)
        if self._replay_preparation_cache_enabled:
            cached = self._replay_preparation_cache.get(cache_key)
            if cached is not None:
                return cached

        parts_by_request = tuple(
            tuple(part for part in snapshot.parts_for(agent_id) if part.policy_input)
            for snapshot, agent_id in requests
        )
        all_parts = tuple(part for parts in parts_by_request for part in parts)
        if all_parts:
            feature_batch = np.stack(
                [self._feature_array(part) for part in all_parts]
            ).astype(np.float32, copy=False)
        else:
            feature_batch = np.empty(
                (0, self.observation_encoder[0].in_features), dtype=np.float32
            )

        semantic_indices = self._index_array(
            [part.semantic_type for part in all_parts],
            self.semantic_index,
            "semantic type",
        )
        unit_indices = self._index_array(
            [part.unit for part in all_parts], self.unit_index, "unit"
        )
        use_indices = self._index_array(
            [part.use for part in all_parts], self.use_index, "observation use"
        )
        scope_indices = self._index_array(
            [part.scope for part in all_parts], self.scope_index, "scope"
        )

        channel_lookup: Dict[tuple[int, str, str], int] = {}
        channel_request_indices: list[int] = []
        channel_sensor_ids: list[str] = []
        channel_sensor_types: list[str] = []
        channel_types: list[str] = []
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
                    channel_sensor_types.append(part.sensor_type)
                    channel_types.append(part.channel_id)
                elif (
                    channel_sensor_types[channel_index] != part.sensor_type
                    or channel_types[channel_index] != part.channel_id
                ):
                    raise ValueError(
                        "TI-MARL channel contains inconsistent sensor/channel types"
                    )
                observation_channel_indices.append(channel_index)

        sensor_lookup: Dict[tuple[int, str], int] = {}
        sensor_request_indices: list[int] = []
        sensor_types: list[str] = []
        channel_sensor_indices: list[int] = []
        for request_index, sensor_id, sensor_type in zip(
            channel_request_indices, channel_sensor_ids, channel_sensor_types
        ):
            key = (request_index, sensor_id)
            sensor_index = sensor_lookup.get(key)
            if sensor_index is None:
                sensor_index = len(sensor_lookup)
                sensor_lookup[key] = sensor_index
                sensor_request_indices.append(request_index)
                sensor_types.append(sensor_type)
            elif sensor_types[sensor_index] != sensor_type:
                raise ValueError("TI-MARL sensor contains inconsistent sensor types")
            channel_sensor_indices.append(sensor_index)

        roles: list[int] = []
        agent_types: list[int] = []
        for snapshot, agent_id in requests:
            role, agent_type = snapshot.metadata_for(agent_id)
            roles.append(
                {"consumer": 0, "producer": 1, "prosumer": 2}.get(role, 0)
            )
            agent_types.append(
                {
                    "residential": 0,
                    "office": 1,
                    "commercial": 2,
                    "industrial": 3,
                    "other": 4,
                }.get(agent_type, 4)
            )

        prepared = _PreparedSnapshotRequests(
            parts_by_request=parts_by_request,
            feature_batch=feature_batch,
            semantic_indices=semantic_indices,
            unit_indices=unit_indices,
            use_indices=use_indices,
            scope_indices=scope_indices,
            observation_channels=np.asarray(
                observation_channel_indices, dtype=np.int64
            ),
            channel_types=self._index_array(
                channel_types, self.channel_index, "channel type"
            ),
            channel_sensors=np.asarray(channel_sensor_indices, dtype=np.int64),
            sensor_types=self._index_array(
                sensor_types, self.sensor_index, "sensor type"
            ),
            sensor_requests=np.asarray(sensor_request_indices, dtype=np.int64),
            roles=np.asarray(roles, dtype=np.int64),
            agent_types=np.asarray(agent_types, dtype=np.int64),
            channel_count=len(channel_lookup),
            sensor_count=len(sensor_lookup),
            request_count=len(requests),
        )
        if self._replay_preparation_cache_enabled:
            self._replay_preparation_cache[cache_key] = prepared
        return prepared

    def _encode_prepared_observations(
        self,
        prepared: _PreparedSnapshotRequests,
        device: torch.device,
    ) -> Tensor:
        feature_batch = prepared.tensor(
            "feature_batch",
            prepared.feature_batch,
            dtype=torch.float32,
            device=device,
        )
        return (
            self.observation_encoder(feature_batch)
            + self.semantic_embedding(
                prepared.tensor(
                    "semantic_indices",
                    prepared.semantic_indices,
                    dtype=torch.long,
                    device=device,
                )
            )
            + self.unit_embedding(
                prepared.tensor(
                    "unit_indices",
                    prepared.unit_indices,
                    dtype=torch.long,
                    device=device,
                )
            )
            + self.use_embedding(
                prepared.tensor(
                    "use_indices",
                    prepared.use_indices,
                    dtype=torch.long,
                    device=device,
                )
            )
            + self.scope_embedding(
                prepared.tensor(
                    "scope_indices",
                    prepared.scope_indices,
                    dtype=torch.long,
                    device=device,
                )
            )
        )

    @staticmethod
    def _index_array(
        values: Sequence[str],
        lookup: Mapping[str, int],
        label: str,
    ) -> np.ndarray:
        unknown = sorted({str(value) for value in values if str(value) not in lookup})
        if unknown:
            raise ValueError(f"Unknown TI-MARL {label}(s): {unknown}")
        return np.asarray([lookup[str(value)] for value in values], dtype=np.int64)

    def encode_observation_parts(
        self,
        parts: Sequence[ObservationPart],
        device: torch.device,
    ) -> Tensor:
        """Encode typed samples without pooling away their individual identity."""

        if not parts:
            return torch.empty((0, self.d_model), dtype=torch.float32, device=device)
        feature_batch = torch.as_tensor(
            np.stack([self._feature_array(part) for part in parts]),
            dtype=torch.float32,
            device=device,
        )
        semantic_indices = self._indices(
            parts,
            self.semantic_index,
            lambda part: part.semantic_type,
            "semantic type",
            device,
        )
        unit_indices = self._indices(
            parts,
            self.unit_index,
            lambda part: part.unit,
            "unit",
            device,
        )
        use_indices = self._indices(
            parts,
            self.use_index,
            lambda part: part.use,
            "observation use",
            device,
        )
        scope_indices = self._indices(
            parts,
            self.scope_index,
            lambda part: part.scope,
            "scope",
            device,
        )
        return (
            self.observation_encoder(feature_batch)
            + self.semantic_embedding(semantic_indices)
            + self.unit_embedding(unit_indices)
            + self.use_embedding(use_indices)
            + self.scope_embedding(scope_indices)
        )

    @staticmethod
    def _typed_embedding(
        type_registry: Mapping[str, object],
        key: str,
        d_model: int,
    ) -> tuple[list[str], Dict[str, int], nn.Embedding]:
        values = sorted(str(item) for item in type_registry.get(key, []))
        if not values:
            raise ValueError(f"TI-MARL type registry must define {key}")
        return values, {name: index for index, name in enumerate(values)}, nn.Embedding(
            len(values), d_model
        )

    @staticmethod
    def _indices(
        parts: Sequence[ObservationPart],
        lookup: Mapping[str, int],
        value_getter,
        label: str,
        device: torch.device,
    ) -> Tensor:
        return TypedSnapshotEncoder._indices_from_values(
            [str(value_getter(part)) for part in parts],
            lookup,
            label,
            device,
        )

    @staticmethod
    def _indices_from_values(
        values: Sequence[str],
        lookup: Mapping[str, int],
        label: str,
        device: torch.device,
    ) -> Tensor:
        unknown = sorted({str(value) for value in values if str(value) not in lookup})
        if unknown:
            raise ValueError(f"Unknown TI-MARL {label}(s): {unknown}")
        return torch.tensor(
            [lookup[str(value)] for value in values],
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def _feature_array(part: ObservationPart) -> np.ndarray:
        raw = np.asarray(part.values or (0.0,), dtype=np.float32)
        if part.normalisation == "signed_log1p":
            transformed = np.sign(raw) * np.log1p(np.abs(raw))
        elif part.normalisation in {"identity", "none"}:
            transformed = raw
        else:
            raise ValueError(
                f"Unknown TI-MARL normalisation: {part.normalisation!r}"
            )
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
        criticality = np.zeros(3, dtype=np.float32)
        criticality[{"advisory": 0, "operational": 1, "safety": 2}.get(part.criticality, 1)] = 1.0
        identity = TypedSnapshotEncoder._identity_features(part)
        temporal = TypedSnapshotEncoder._temporal_features(part.observation_id)
        return np.concatenate(
            (values, health, flags, age, criticality, identity, temporal)
        )

    @staticmethod
    def _identity_features(part: ObservationPart) -> np.ndarray:
        # Instance IDs are intentionally absent: a second charger shares the
        # same model parameters.  The exact typed observation still receives a
        # stable fingerprint, preventing equal-valued load/PV/price signals
        # from becoming indistinguishable inside a semantic family.
        return np.asarray(
            TypedSnapshotEncoder._identity_feature_tuple(
                part.semantic_type,
                part.channel_id,
                part.observation_id,
            ),
            dtype=np.float32,
        )

    @staticmethod
    @lru_cache(maxsize=4096)
    def _identity_feature_tuple(
        semantic_type: str,
        channel_id: str,
        observation_id: str,
    ) -> tuple[float, ...]:
        identity = "\x1f".join((semantic_type, channel_id, observation_id))
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        values = (
            np.frombuffer(digest[:8], dtype=np.uint8).astype(np.float32) / 127.5
            - 1.0
        )
        return tuple(float(value) for value in values)

    @staticmethod
    def _temporal_features(observation_id: str) -> np.ndarray:
        return np.asarray(
            TypedSnapshotEncoder._temporal_feature_tuple(observation_id),
            dtype=np.float32,
        )

    @staticmethod
    @lru_cache(maxsize=4096)
    def _temporal_feature_tuple(observation_id: str) -> tuple[float, float, float]:
        key = str(observation_id).lower()
        future = re.search(r"(?:next|predicted)_(\d+)(m|h)?", key)
        past = re.search(r"prev_(\d+)", key)
        amount_minutes = 0.0
        if future is not None:
            amount_minutes = float(future.group(1))
            if future.group(2) == "h":
                amount_minutes *= 60.0
        elif past is not None:
            amount_minutes = float(past.group(1))
        scaled_amount = np.log1p(amount_minutes) / np.log1p(1440.0)
        return (
            float(future is not None),
            float(past is not None),
            float(scaled_amount),
        )


class TypedActor(nn.Module):
    """Shared grouped actor with categorical ports and Beta fractions."""

    def __init__(
        self,
        type_registry: Mapping[str, object],
        *,
        d_model: int = 128,
        attention_heads: int = 4,
        relation_layers: int = 2,
        group_context_kind: str = "local",
        deterministic_mode_strategy: str = "argmax",
        deterministic_mode_strategy_by_group_type: Mapping[str, str] | None = None,
        deterministic_expected_signed_gain_by_group_type: Mapping[
            str, float
        ] | None = None,
        deterministic_expected_signed_deadband_by_group_type: Mapping[
            str, float
        ] | None = None,
        deterministic_non_idle_logit_margin_by_group_type: Mapping[
            str, float
        ] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.group_context_kind = str(group_context_kind)
        if self.group_context_kind not in {"local", "action_conditioned"}:
            raise ValueError(
                "TI-MARL actor group_context_kind must be 'local' or "
                "'action_conditioned'"
            )
        self.deterministic_mode_strategy = str(deterministic_mode_strategy)
        if self.deterministic_mode_strategy not in {
            "argmax",
            "expected_signed",
        }:
            raise ValueError(
                "TI-MARL actor deterministic_mode_strategy must be 'argmax' "
                "or 'expected_signed'"
            )
        self.encoder = TypedSnapshotEncoder(type_registry, d_model, relation_layers)
        action_types = dict(type_registry.get("action_group_types", {}))
        if not action_types:
            raise ValueError("TI-MARL type registry must define action_group_types")
        self.deterministic_mode_strategy_by_group_type = {
            str(group_type): str(strategy)
            for group_type, strategy in dict(
                deterministic_mode_strategy_by_group_type or {}
            ).items()
        }
        unknown_strategy_group_types = sorted(
            set(self.deterministic_mode_strategy_by_group_type) - set(action_types)
        )
        if unknown_strategy_group_types:
            raise ValueError(
                "TI-MARL deterministic mode strategy overrides reference unknown "
                f"action group types: {unknown_strategy_group_types}"
            )
        invalid_strategies = sorted(
            {
                strategy
                for strategy in self.deterministic_mode_strategy_by_group_type.values()
                if strategy not in {"argmax", "expected_signed"}
            }
        )
        if invalid_strategies:
            raise ValueError(
                "TI-MARL deterministic mode strategy overrides must be 'argmax' "
                f"or 'expected_signed': {invalid_strategies}"
            )
        self.deterministic_expected_signed_gain_by_group_type = {
            str(group_type): float(gain)
            for group_type, gain in dict(
                deterministic_expected_signed_gain_by_group_type or {}
            ).items()
        }
        unknown_gain_group_types = sorted(
            set(self.deterministic_expected_signed_gain_by_group_type)
            - set(action_types)
        )
        if unknown_gain_group_types:
            raise ValueError(
                "TI-MARL deterministic expected-signed gains reference unknown "
                f"action group types: {unknown_gain_group_types}"
            )
        if any(
            gain < 0.0
            for gain in self.deterministic_expected_signed_gain_by_group_type.values()
        ):
            raise ValueError(
                "TI-MARL deterministic expected-signed gains must be non-negative"
            )
        self.deterministic_expected_signed_deadband_by_group_type = {
            str(group_type): float(deadband)
            for group_type, deadband in dict(
                deterministic_expected_signed_deadband_by_group_type or {}
            ).items()
        }
        unknown_deadband_group_types = sorted(
            set(self.deterministic_expected_signed_deadband_by_group_type)
            - set(action_types)
        )
        if unknown_deadband_group_types:
            raise ValueError(
                "TI-MARL deterministic expected-signed deadbands reference unknown "
                f"action group types: {unknown_deadband_group_types}"
            )
        if any(
            deadband < 0.0 or deadband > 1.0
            for deadband in self.deterministic_expected_signed_deadband_by_group_type.values()
        ):
            raise ValueError(
                "TI-MARL deterministic expected-signed deadbands must be between "
                "zero and one"
            )
        self.deterministic_non_idle_logit_margin_by_group_type = {
            str(group_type): float(margin)
            for group_type, margin in dict(
                deterministic_non_idle_logit_margin_by_group_type or {}
            ).items()
        }
        unknown_margin_group_types = sorted(
            set(self.deterministic_non_idle_logit_margin_by_group_type)
            - set(action_types)
        )
        if unknown_margin_group_types:
            raise ValueError(
                "TI-MARL deterministic non-idle margins reference unknown action "
                f"group types: {unknown_margin_group_types}"
            )
        if any(
            margin < 0.0
            for margin in self.deterministic_non_idle_logit_margin_by_group_type.values()
        ):
            raise ValueError(
                "TI-MARL deterministic non-idle logit margins must be non-negative"
            )
        self.group_modes = {
            name: tuple(str(mode) for mode in config.get("modes", []))
            for name, config in sorted(action_types.items())
        }
        self.group_index = {name: index for index, name in enumerate(self.group_modes)}
        self.group_embedding = nn.Embedding(len(self.group_modes), d_model)
        if self.group_context_kind == "action_conditioned":
            self.group_observation_relation_embedding = nn.Embedding(4, d_model)
            self.group_observation_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=int(attention_heads),
                batch_first=True,
            )
            self.group_observation_norm = nn.LayerNorm(d_model)
        else:
            self.group_observation_relation_embedding = None
            self.group_observation_attention = None
            self.group_observation_norm = None
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
        agent_ids = tuple(snapshot.agent_ids)
        requests = tuple((snapshot, agent_id) for agent_id in agent_ids)
        local_latents, preencoded_parts = self._encode_actor_requests(
            requests, device
        )
        groups_by_request = tuple(snapshot.groups_for(agent_id) for agent_id in agent_ids)
        flat_contexts = self._group_contexts_many(
            requests,
            groups_by_request,
            local_latents,
            device,
            preencoded_parts=preencoded_parts,
        )
        context_offset = 0
        for request_index, (agent_id, local_latent) in enumerate(
            zip(agent_ids, local_latents)
        ):
            latents[agent_id] = local_latent
            groups = groups_by_request[request_index]
            contexts = flat_contexts[context_offset : context_offset + len(groups)]
            context_offset += len(groups)

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
            return ActorReplayEvaluation(
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
            )

        latents, preencoded_parts = self._encode_actor_requests(requests, device)
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
        group_log_prob_by_step: list[Dict[str, Dict[str, Tensor]]] = [
            {} for _item in items
        ]
        group_entropy_by_step: list[Dict[str, Dict[str, Tensor]]] = [
            {} for _item in items
        ]
        group_mode_log_prob_by_step: list[Dict[str, Dict[str, Tensor]]] = [
            {} for _item in items
        ]
        group_predicted_mode_by_step: list[Dict[str, Dict[str, Tensor]]] = [
            {} for _item in items
        ]
        group_predicted_fraction_by_step: list[
            Dict[str, Dict[str, Tensor]]
        ] = [{} for _item in items]
        if group_entries:
            grouped_entries: list[list[ActionGroupInstance]] = [
                [] for _request in requests
            ]
            for request_index, group, _expected in group_entries:
                grouped_entries[request_index].append(group)
            groups_by_request = tuple(tuple(groups) for groups in grouped_entries)
            flat_contexts = self._group_contexts_many(
                requests,
                groups_by_request,
                latents,
                device,
                preencoded_parts=preencoded_parts,
            )
            request_indices = torch.tensor(
                [entry[0] for entry in group_entries],
                dtype=torch.long,
                device=device,
            )

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
                masked_logits = logits.masked_fill(
                    ~mask, torch.finfo(logits.dtype).min
                )
                categorical = Categorical(logits=masked_logits)
                selected_modes = torch.tensor(
                    mode_indices, dtype=torch.long, device=device
                )
                mode_log_prob = categorical.log_prob(selected_modes)
                entropy = categorical.entropy()
                predicted_modes = torch.argmax(masked_logits, dim=1)
                beta_params = F.softplus(self.beta_heads[group_type](contexts)) + 1.0
                beta_distribution = Beta(beta_params[:, 0], beta_params[:, 1])
                predicted_fractions = beta_params[:, 0] / beta_params.sum(dim=1)
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
                log_prob = mode_log_prob + torch.where(
                    parameterized,
                    beta_distribution.log_prob(fractions),
                    torch.zeros_like(mode_log_prob),
                )
                entropy = entropy + torch.where(
                    parameterized,
                    beta_distribution.entropy(),
                    torch.zeros_like(entropy),
                )
                for row, (_request_index, group, _expected) in enumerate(entries):
                    step_index, agent_id = request_keys[_request_index]
                    group_log_prob_by_step[step_index].setdefault(agent_id, {})[
                        group.group_id
                    ] = log_prob[row]
                    group_entropy_by_step[step_index].setdefault(agent_id, {})[
                        group.group_id
                    ] = entropy[row]
                    group_mode_log_prob_by_step[step_index].setdefault(
                        agent_id, {}
                    )[group.group_id] = mode_log_prob[row]
                    group_predicted_mode_by_step[step_index].setdefault(
                        agent_id, {}
                    )[group.group_id] = predicted_modes[row]
                    group_predicted_fraction_by_step[step_index].setdefault(
                        agent_id, {}
                    )[group.group_id] = predicted_fractions[row]
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
            tuple(log_prob_by_step),
            tuple(entropy_by_step),
            tuple(group_log_prob_by_step),
            tuple(group_entropy_by_step),
            tuple(group_mode_log_prob_by_step),
            tuple(group_predicted_mode_by_step),
            tuple(group_predicted_fraction_by_step),
        )

    def _group_contexts_many(
        self,
        requests: Sequence[tuple[InterfaceSnapshot, str]],
        groups_by_request: Sequence[Sequence[ActionGroupInstance]],
        local_latents: Tensor,
        device: torch.device,
        *,
        preencoded_parts: tuple[
            tuple[tuple[ObservationPart, ...], ...], Tensor
        ] | None = None,
    ) -> Tensor:
        """Build action-conditioned contexts without using concrete asset IDs."""

        flat_groups = [group for groups in groups_by_request for group in groups]
        if not flat_groups:
            return torch.empty((0, self.d_model), device=device)
        flat_request_indices = [
            request_index
            for request_index, groups in enumerate(groups_by_request)
            for _group in groups
        ]
        request_indices = torch.tensor(
            flat_request_indices,
            dtype=torch.long,
            device=device,
        )
        type_indices = torch.tensor(
            [self.group_index[group.group_type] for group in flat_groups],
            dtype=torch.long,
            device=device,
        )
        base_contexts = local_latents[request_indices] + self.group_embedding(type_indices)

        if self.group_context_kind == "action_conditioned":
            if preencoded_parts is None:
                parts_by_request = [
                    tuple(
                        part
                        for part in snapshot.parts_for(agent_id)
                        if part.policy_input
                    )
                    for snapshot, agent_id in requests
                ]
                encoded_parts = self.encoder.encode_observation_parts(
                    [part for parts in parts_by_request for part in parts],
                    device,
                )
            else:
                parts_by_request = list(preencoded_parts[0])
                encoded_parts = preencoded_parts[1]
            relation_by_group: list[tuple[int, ...]] = []
            for request_index, groups in enumerate(groups_by_request):
                parts = parts_by_request[request_index]
                for group in groups:
                    relation_by_group.append(
                        tuple(self._group_part_relation(group, part) for part in parts)
                    )
            max_parts = max((len(parts) for parts in parts_by_request), default=0)
        else:
            parts_by_request = []
            relation_by_group = []
            max_parts = 0
        if max_parts:
            assert self.group_observation_relation_embedding is not None
            assert self.group_observation_attention is not None
            assert self.group_observation_norm is not None
            request_offsets = []
            offset = 0
            for parts in parts_by_request:
                request_offsets.append(offset)
                offset += len(parts)
            padded = torch.zeros(
                (len(flat_groups), max_parts, self.d_model),
                dtype=encoded_parts.dtype,
                device=device,
            )
            padding_mask = torch.ones(
                (len(flat_groups), max_parts), dtype=torch.bool, device=device
            )
            for group_index, (request_index, relations) in enumerate(
                zip(flat_request_indices, relation_by_group)
            ):
                parts = parts_by_request[request_index]
                count = len(parts)
                if not count:
                    continue
                relation_indices = torch.tensor(
                    relations, dtype=torch.long, device=device
                )
                padded[group_index, :count] = (
                    encoded_parts[
                        request_offsets[request_index] :
                        request_offsets[request_index] + count
                    ]
                    + self.group_observation_relation_embedding(relation_indices)
                )
                padding_mask[group_index, :count] = False
            active = ~padding_mask.all(dim=1)
            attended = torch.zeros_like(base_contexts)
            if bool(active.any()):
                cross, _ = self.group_observation_attention(
                    base_contexts[active].unsqueeze(1),
                    padded[active],
                    padded[active],
                    key_padding_mask=padding_mask[active],
                    need_weights=False,
                )
                attended[active] = cross.squeeze(1)
            base_contexts = self.group_observation_norm(base_contexts + attended)

        max_groups = max((len(groups) for groups in groups_by_request), default=0)
        positions = []
        for groups in groups_by_request:
            positions.extend(range(len(groups)))
        position_indices = torch.tensor(positions, dtype=torch.long, device=device)
        padded_groups = torch.zeros(
            (len(requests), max_groups, self.d_model),
            dtype=base_contexts.dtype,
            device=device,
        )
        padded_groups[request_indices, position_indices] = base_contexts
        group_padding = torch.ones(
            (len(requests), max_groups), dtype=torch.bool, device=device
        )
        group_padding[request_indices, position_indices] = False
        active_requests = ~group_padding.all(dim=1)
        output = torch.zeros_like(padded_groups)
        self_attended, _ = self.group_attention(
            padded_groups[active_requests],
            padded_groups[active_requests],
            padded_groups[active_requests],
            key_padding_mask=group_padding[active_requests],
            need_weights=False,
        )
        output[active_requests] = self.group_norm(
            padded_groups[active_requests] + self_attended
        )
        return output[request_indices, position_indices]

    def _encode_actor_requests(
        self,
        requests: Sequence[tuple[InterfaceSnapshot, str]],
        device: torch.device,
    ) -> tuple[
        Tensor,
        tuple[tuple[tuple[ObservationPart, ...], ...], Tensor] | None,
    ]:
        if self.group_context_kind == "action_conditioned":
            latents, parts, encoded_parts = self.encoder.forward_many_with_parts(
                requests, device
            )
            return latents, (parts, encoded_parts)
        return self.encoder.forward_many(requests, device), None

    @staticmethod
    def _group_part_relation(
        group: ActionGroupInstance,
        part: ObservationPart,
    ) -> int:
        if part.sensor_id == group.module_id:
            return 0
        group_suffix = group.module_id.rsplit("_", 1)[-1]
        sensor_suffix = part.sensor_id.rsplit("_", 1)[-1]
        if (
            group.group_type == "ev_session"
            and part.sensor_type == "ev_session"
            and group_suffix == sensor_suffix
        ):
            return 0
        if part.scope == "community":
            return 3
        if part.scope == "local":
            return 2
        return 1

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
            fraction = beta_params[0] / beta_params.sum()
            mode_index, fraction = self._deterministic_mode_and_fraction(
                group.group_type,
                modes,
                categorical.probs,
                fraction,
                mask_values,
                masked_logits,
            )
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

    def _deterministic_mode_and_fraction(
        self,
        group_type: str,
        modes: Sequence[str],
        probabilities: Tensor,
        beta_mean: Tensor,
        mask_values: Sequence[bool],
        masked_logits: Tensor,
    ) -> tuple[int, Tensor]:
        strategy = self.deterministic_mode_strategy_by_group_type.get(
            group_type, self.deterministic_mode_strategy
        )
        if strategy == "argmax":
            mode_index = int(torch.argmax(masked_logits).item())
            margin = self.deterministic_non_idle_logit_margin_by_group_type.get(
                group_type, 0.0
            )
            if margin > 0.0 and modes[mode_index] != "IDLE":
                idle_index = modes.index("IDLE")
                non_idle_advantage = masked_logits[mode_index] - masked_logits[idle_index]
                if float(non_idle_advantage.detach().cpu()) < margin:
                    return idle_index, beta_mean.new_zeros(())
            return mode_index, beta_mean

        charge_indices = [
            index for index, mode in enumerate(modes) if mode.startswith("CHARGE_")
        ]
        discharge_indices = [
            index
            for index, mode in enumerate(modes)
            if mode.startswith("DISCHARGE_")
        ]
        if not charge_indices and not discharge_indices:
            return int(torch.argmax(masked_logits).item()), beta_mean

        signed_probability = probabilities.new_zeros(())
        for index in charge_indices:
            signed_probability = signed_probability + probabilities[index]
        for index in discharge_indices:
            signed_probability = signed_probability - probabilities[index]
        gain = self.deterministic_expected_signed_gain_by_group_type.get(
            group_type, 1.0
        )
        signed_fraction = signed_probability * beta_mean * gain
        deadband = self.deterministic_expected_signed_deadband_by_group_type.get(
            group_type, 0.0
        )
        if abs(float(signed_fraction.detach().cpu())) <= max(deadband, 1.0e-8):
            return modes.index("IDLE"), beta_mean.new_zeros(())
        candidates = charge_indices if signed_fraction > 0.0 else discharge_indices
        valid_candidates = [index for index in candidates if mask_values[index]]
        if not valid_candidates:
            return modes.index("IDLE"), beta_mean.new_zeros(())
        mode_index = max(
            valid_candidates,
            key=lambda index: float(probabilities[index].detach().cpu()),
        )
        return mode_index, torch.clamp(torch.abs(signed_fraction), 0.0, 1.0)


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


class TypedGroupCritic(nn.Module):
    """Typed value baseline per action group with invariant parameters."""

    def __init__(
        self,
        type_registry: Mapping[str, object],
        *,
        d_model: int = 128,
        relation_layers: int = 2,
        centralized: bool = True,
    ) -> None:
        super().__init__()
        self.centralized = bool(centralized)
        self.encoder = TypedSnapshotEncoder(
            type_registry, d_model, relation_layers
        )
        group_types = tuple(
            sorted(
                str(item)
                for item in dict(
                    type_registry.get("action_group_types", {})
                )
            )
        )
        if not group_types:
            raise ValueError(
                "TI-MARL type registry must define action_group_types"
            )
        self.group_index = {
            name: index for index, name in enumerate(group_types)
        }
        self.group_embedding = nn.Embedding(len(group_types), d_model)
        input_width = d_model * (3 if self.centralized else 2)
        self.value_head = nn.Sequential(
            nn.Linear(input_width, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        snapshot: InterfaceSnapshot,
    ) -> Mapping[str, Mapping[str, Tensor]]:
        return self.forward_many((snapshot,))[0]

    def forward_many(
        self,
        snapshots: Sequence[InterfaceSnapshot],
    ) -> Tuple[Mapping[str, Mapping[str, Tensor]], ...]:
        device = next(self.parameters()).device
        requests = [
            (snapshot, agent_id)
            for snapshot in snapshots
            for agent_id in snapshot.agent_ids
        ]
        result: list[Dict[str, Dict[str, Tensor]]] = [
            {} for _snapshot in snapshots
        ]
        if not requests:
            return tuple(result)
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

        request_entries: list[tuple[int, int, str, ActionGroupInstance]] = []
        request_index = 0
        for snapshot_index, snapshot in enumerate(snapshots):
            for agent_id in snapshot.agent_ids:
                result[snapshot_index][agent_id] = {}
                request_entries.extend(
                    (request_index, snapshot_index, agent_id, group)
                    for group in snapshot.groups_for(agent_id)
                )
                request_index += 1
        if not request_entries:
            return tuple(result)
        request_indices = torch.tensor(
            [item[0] for item in request_entries],
            dtype=torch.long,
            device=device,
        )
        group_indices = torch.tensor(
            [self.group_index[item[3].group_type] for item in request_entries],
            dtype=torch.long,
            device=device,
        )
        features = [
            local[request_indices],
            self.group_embedding(group_indices),
        ]
        if self.centralized:
            features.insert(
                1,
                community[snapshot_indices[request_indices]],
            )
        values = self.value_head(torch.cat(features, dim=-1)).squeeze(-1)
        for index, (_request, snapshot_index, agent_id, group) in enumerate(
            request_entries
        ):
            result[snapshot_index][agent_id][group.group_id] = values[index]
        return tuple(result)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
