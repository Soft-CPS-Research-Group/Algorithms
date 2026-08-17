"""Technology-neutral runtime contracts shared by training and deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from algorithms.ti_marl.contracts.enums import (
    AvailabilityState,
    ConnectionState,
    EventDomain,
    QualityState,
)


RUNTIME_CONTRACT_VERSION = "typed_runtime_v1"


@dataclass(frozen=True)
class TypedObservationSample:
    agent_id: str
    sensor_id: str
    channel_id: str
    observation_id: str
    value: Tuple[float, ...]
    shape: Tuple[int, ...]
    unit: str
    timestamp_seconds: float
    age_seconds: float = 0.0
    availability: AvailabilityState = AvailabilityState.AVAILABLE
    quality: QualityState = QualityState.NOMINAL
    fault_mode: Optional[str] = None
    provenance: str = "unknown_adapter"
    source_entity_type: Optional[str] = None
    source_entity_id: Optional[str] = None
    source_feature: Optional[str] = None
    estimated: bool = False

    @property
    def sample_id(self) -> str:
        return (
            f"{self.agent_id}:{self.sensor_id}:{self.channel_id}:"
            f"{self.observation_id}"
        )


@dataclass(frozen=True)
class TypedHealthEvidence:
    evidence_id: str
    agent_id: Optional[str]
    sensor_id: Optional[str]
    channel_id: Optional[str]
    observation_id: Optional[str]
    event_domain: EventDomain
    fault_mode: Optional[str]
    target_type: str
    target_id: str
    target_feature: str
    availability: AvailabilityState = AvailabilityState.UNKNOWN
    connection: ConnectionState = ConnectionState.UNKNOWN
    quality: QualityState = QualityState.UNKNOWN
    started_at_seconds: Optional[float] = None
    active_duration_seconds: float = 0.0
    last_update_seconds: Optional[float] = None
    last_fresh_seconds: Optional[float] = None
    age_seconds: Optional[float] = None
    event_ids: Tuple[str, ...] = ()
    provenance: str = "unknown_adapter"


@dataclass(frozen=True)
class TypedRuntimeEntity:
    entity_id: str
    entity_type: str
    owner_agent_id: Optional[str]
    active: bool
    values: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TypedExecutionFeedback:
    agent_id: str
    actuator_id: str
    target_entity_id: str
    action_name: str
    requested_value: Optional[float]
    post_channel_value: Optional[float]
    limited_value: Optional[float]
    applied_value: Optional[float]
    applied_power_kw: Optional[float]
    limitation_reasons: Tuple[str, ...] = ()
    timestamp_seconds: float = 0.0
    provenance: str = "unknown_adapter"


@dataclass(frozen=True)
class TypedRuntimeFrame:
    version: str
    frame_id: str
    timestamp_seconds: float
    sequence: int
    topology_version: int
    registered_agent_ids: Tuple[str, ...]
    active_agent_ids: Tuple[str, ...]
    samples: Tuple[TypedObservationSample, ...]
    health_evidence: Tuple[TypedHealthEvidence, ...]
    entities: Tuple[TypedRuntimeEntity, ...]
    execution_feedback: Tuple[TypedExecutionFeedback, ...] = ()
    topology_events: Tuple[Mapping[str, Any], ...] = ()
    provenance: str = "unknown_adapter"


@dataclass(frozen=True)
class TypedActionCommand:
    agent_id: str
    actuator_id: str
    action_id: str
    mode: str
    value: float
    unit: str
    target_entity_id: Optional[str]
    timestamp_seconds: float
    command_id: str
    constraints_applied: Tuple[str, ...] = ()
    fallback_reason: Optional[str] = None
