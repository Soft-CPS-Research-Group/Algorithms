"""Canonical immutable object model for ``ti_marl_v1``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Optional, Sequence, Tuple

from algorithms.ti_marl.contracts.enums import (
    AvailabilityState,
    ConnectionState,
    EventDomain,
    HealthState,
    QualityState,
)


def canonical_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""

    if is_dataclass(value):
        return canonical_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [canonical_value(item) for item in value]
    if isinstance(value, set):
        return sorted(canonical_value(item) for item in value)
    if hasattr(value, "tolist"):
        try:
            return canonical_value(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return canonical_value(value.item())
        except (TypeError, ValueError):
            pass
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgentSchema:
    version: str
    agent_entity_type: str
    module_types: Tuple[str, ...]
    action_group_types: Tuple[str, ...]
    observation_semantic_types: Tuple[str, ...]


@dataclass(frozen=True)
class ModuleInstance:
    module_id: str
    module_type: str
    owner_agent_id: str
    entity_id: str
    available: AvailabilityState = AvailabilityState.AVAILABLE
    connected: ConnectionState = ConnectionState.NOT_APPLICABLE


@dataclass(frozen=True)
class EntityInstance:
    entity_id: str
    entity_type: str
    owner_agent_id: Optional[str]
    row_index: int
    feature_names: Tuple[str, ...]
    values: Tuple[float, ...]


@dataclass(frozen=True)
class FaultEvidence:
    evidence_id: str
    event_domain: EventDomain
    fault_mode: Optional[str]
    target_type: str
    target_id: str
    target_feature: str
    availability: AvailabilityState = AvailabilityState.UNKNOWN
    connection: ConnectionState = ConnectionState.UNKNOWN
    quality: QualityState = QualityState.UNKNOWN
    start_time_step: Optional[int] = None
    active_duration_steps: int = 0
    last_update_time_step: Optional[int] = None
    last_fresh_time_step: Optional[int] = None
    age_steps: Optional[int] = None
    event_ids: Tuple[str, ...] = ()
    start_time_seconds: Optional[float] = None
    active_duration_seconds: float = 0.0
    last_update_seconds: Optional[float] = None
    last_fresh_seconds: Optional[float] = None
    age_seconds: Optional[float] = None
    agent_id: Optional[str] = None
    sensor_id: Optional[str] = None
    channel_name: Optional[str] = None
    observation_id: Optional[str] = None


@dataclass(frozen=True)
class ChannelStatus:
    """Facts-only status for one sensor, actuator or communication channel."""

    channel_id: str
    event_domain: EventDomain
    source_id: Optional[str]
    target_id: str
    target_feature: str
    availability: AvailabilityState
    quality: QualityState
    fault_mode: Optional[str] = None
    last_update_time_step: Optional[int] = None
    last_fresh_time_step: Optional[int] = None
    age_steps: Optional[int] = None
    event_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class HealthAssessment:
    subject_id: str
    semantic_type: str
    criticality: str
    state: HealthState
    rule_id: str
    evidence_ids: Tuple[str, ...] = ()
    since_time_step: Optional[int] = None
    recovery_pending_steps: int = 0
    explanation: str = ""
    since_seconds: Optional[float] = None
    recovery_pending_seconds: float = 0.0


@dataclass(frozen=True)
class ObservationPart:
    part_id: str
    owner_agent_id: str
    source_entity_id: str
    semantic_type: str
    feature_names: Tuple[str, ...]
    values: Tuple[float, ...]
    health: HealthState
    shape: Tuple[int, ...] = (1,)
    valid: bool = True
    validity_reasons: Tuple[str, ...] = ()
    estimated: bool = False
    sensor_id: str = "unknown_sensor"
    channel_id: str = "state"
    observation_id: str = "value"
    unit: str = "scalar"
    scope: str = "local"
    use: str = "policy_input"
    policy_input: bool = True
    criticality: str = "operational"
    age_seconds: float = 0.0
    normalisation: str = "signed_log1p"


@dataclass(frozen=True)
class ActionPortInstance:
    port_id: str
    mode: str
    target_entity_id: str
    action_name: str
    lower_bound: float
    upper_bound: float
    valid: bool = True
    invalid_reasons: Tuple[str, ...] = ()
    contracted_by: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionGroupInstance:
    group_id: str
    group_type: str
    owner_agent_id: str
    module_id: str
    ports: Tuple[ActionPortInstance, ...]
    enabled: bool = True
    degraded_mode: Optional[str] = None
    fallback_mode: str = "IDLE"
    max_charge_power_kw: float = 0.0
    max_discharge_power_kw: float = 0.0
    activation_power_kw: float = 0.0
    forced_mode: Optional[str] = None
    forced_fraction: Optional[float] = None
    fallback_reason: Optional[str] = None
    adapter_target_entity_id: Optional[str] = None


@dataclass(frozen=True)
class Dependency:
    dependency_id: str
    source_kind: str
    source_type: str
    target_group_type: Optional[str]
    target_semantic_type: Optional[str]
    consequence: str
    condition_states: Tuple[HealthState, ...]
    parameter: Optional[float] = None


@dataclass(frozen=True)
class HealthRule:
    rule_id: str
    semantic_type: str
    criticality: str
    degraded_after_steps: int
    stale_after_steps: int
    missing_after_steps: int
    recovery_hysteresis_steps: int
    cache_allowed: bool = True
    degraded_after_seconds: float = 0.0
    stale_after_seconds: float = 0.0
    missing_after_seconds: float = 0.0
    recovery_hysteresis_seconds: float = 0.0


@dataclass(frozen=True)
class LocalConstraint:
    constraint_id: str
    owner_agent_id: str
    constraint_type: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    member_group_ids: Tuple[str, ...] = ()
    member_group_coefficients: Tuple[Tuple[str, float], ...] = ()
    active: bool = True


@dataclass(frozen=True)
class SharedResource:
    resource_id: str
    resource_type: str
    member_agent_ids: Tuple[str, ...]
    observable_only: bool = True


@dataclass(frozen=True)
class InterfaceSnapshot:
    contract_version: str
    compiler_version: str
    topology_version: int
    time_step: int
    agent_ids: Tuple[str, ...]
    modules: Tuple[ModuleInstance, ...]
    entities: Tuple[EntityInstance, ...]
    fault_evidence: Tuple[FaultEvidence, ...]
    health: Tuple[HealthAssessment, ...]
    observation_parts: Tuple[ObservationPart, ...]
    action_groups: Tuple[ActionGroupInstance, ...]
    dependencies: Tuple[Dependency, ...]
    constraints: Tuple[LocalConstraint, ...]
    shared_resources: Tuple[SharedResource, ...]
    closure_log: Tuple[Mapping[str, Any], ...] = ()
    timestamp_seconds: float = 0.0
    registered_agent_ids: Tuple[str, ...] = ()
    agent_metadata: Tuple[Tuple[str, str, str], ...] = ()
    registry_hash: str = ""
    execution_feedback: Tuple[Mapping[str, Any], ...] = ()
    topology_events: Tuple[Mapping[str, Any], ...] = ()

    @property
    def snapshot_hash(self) -> str:
        return content_hash(self)

    def groups_for(self, agent_id: str) -> Tuple[ActionGroupInstance, ...]:
        return tuple(group for group in self.action_groups if group.owner_agent_id == agent_id)

    def parts_for(self, agent_id: str) -> Tuple[ObservationPart, ...]:
        return tuple(part for part in self.observation_parts if part.owner_agent_id == agent_id)


@dataclass(frozen=True)
class ActionDecision:
    group_id: str
    mode: str
    fraction: float
    mode_index: int
    raw_log_prob: float = 0.0


@dataclass(frozen=True)
class LocalActionBundle:
    agent_id: str
    decisions: Tuple[ActionDecision, ...]
    interventions: Tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class TypedTransition:
    snapshot_hash: str
    next_snapshot_hash: str
    agent_ids: Tuple[str, ...]
    next_agent_ids: Tuple[str, ...]
    raw_bundles: Tuple[LocalActionBundle, ...]
    final_bundles: Tuple[LocalActionBundle, ...]
    commands: Tuple[Tuple[float, ...], ...]
    execution: Mapping[str, Any]
    rewards: Tuple[Tuple[str, float], ...]
    reward_components: Mapping[str, Any]
    terminated_agent_ids: Tuple[str, ...]
    bootstrap_agent_ids: Tuple[str, ...]
    health_events: Tuple[Mapping[str, Any], ...] = ()
    topology_events: Tuple[Mapping[str, Any], ...] = ()
    typed_commands: Tuple[Mapping[str, Any], ...] = ()


def tuple_of_strings(values: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(str(value) for value in values)
