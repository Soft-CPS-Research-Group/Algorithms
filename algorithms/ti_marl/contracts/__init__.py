"""Immutable TI-MARL contracts and compatibility helpers."""

from algorithms.ti_marl.contracts.enums import (
    AvailabilityState,
    ConnectionState,
    EventDomain,
    HealthState,
    QualityState,
)
from algorithms.ti_marl.contracts.interface_definition import (
    TYPED_INTERFACE_VERSION,
    TypedInterfaceDefinition,
)
from algorithms.ti_marl.contracts.models import (
    ActionDecision,
    ActionGroupInstance,
    ActionPortInstance,
    AgentSchema,
    ChannelStatus,
    Dependency,
    EntityInstance,
    FaultEvidence,
    HealthAssessment,
    HealthRule,
    InterfaceSnapshot,
    LocalActionBundle,
    LocalConstraint,
    ModuleInstance,
    ObservationPart,
    SharedResource,
    TypedTransition,
)

__all__ = [
    "ActionDecision",
    "ActionGroupInstance",
    "ActionPortInstance",
    "AgentSchema",
    "AvailabilityState",
    "ChannelStatus",
    "ConnectionState",
    "Dependency",
    "EntityInstance",
    "EventDomain",
    "FaultEvidence",
    "HealthAssessment",
    "HealthRule",
    "HealthState",
    "InterfaceSnapshot",
    "LocalActionBundle",
    "LocalConstraint",
    "ModuleInstance",
    "ObservationPart",
    "QualityState",
    "SharedResource",
    "TypedTransition",
    "TYPED_INTERFACE_VERSION",
    "TypedInterfaceDefinition",
]
