# TI-MARL object model

Runtime semantic objects are immutable and instance-bound. Configuration
models may use Pydantic; per-step objects use frozen, slotted dataclasses.

## Capability and runtime objects

- `AgentSchema`: potential compatible roles, module/entity types, observations,
  groups, ports, constraints, timing and fallback.
- `ModuleInstance`: installed functional module and its availability evidence.
- `EntityInstance`: physical/logical target such as an EV session.
- `TypedRelation`: validated ownership, containment, connection, observation,
  control, dependency and resource relation.

## Runtime evidence and health

- `ConnectionState`: connected, disconnected, unknown or not applicable.
- `AvailabilityState`: available, unavailable or unknown.
- `ChannelStatus`: sensor, actuator or communication-channel evidence.
- `FaultEvidence`: immutable cause, domain, target, timing and quality facts.
- `HealthAssessment`: TIC-derived health plus rule/evidence references.

`FaultEvidence.fault_mode` is never a `HealthState`. Health assessments may
change over time while the underlying cause remains unchanged.

## Decision objects

- `ObservationPart`: semantic feature bound to entity and source, including
  unit, scope, timestamp, provenance and derived health.
- `ActionGroupInstance`: exactly-one port family bound to one module/entity.
- `ActionPortInstance`: port type, continuous parameters, validity, bounds,
  dependencies and resource effects.
- `LocalConstraint`: typed local hard/soft constraint.
- `SharedResource`: community resource visible to rewards/critic/trace.
- `InterfaceSnapshot`: complete immutable compiler result and compatibility
  signature for one agent and step.
- `LocalActionBundle`: selected port/parameter per active group.
- `TypedTransition`: current/next snapshots, raw/final/executed bundles,
  rewards, termination and runtime events.

## Identity invariants

- Stable IDs never derive solely from row positions.
- Session replacement creates a new entity identity.
- Every port has one owner, one target and one executable simulator route.
- Every observation identifies entity, source and scope.
- Unknown action semantics are never executed automatically.

