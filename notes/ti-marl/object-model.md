# TI-MARL object model

Runtime semantic objects are immutable and instance-bound. Configuration
models may use Pydantic; per-step objects use frozen dataclasses.

## Capability and runtime objects

- `TypedAgentInterface`: one versioned, editable, technology-neutral contract
  for a registered agent, its sensor/channel/observation hierarchy, actuators,
  constraints and explicit dependencies.
- `InterfaceRegistry`: immutable generation of agent interfaces with atomic
  directory reload and composition-independent compatibility validation.
- `CapabilityProfile`: policy-packaged reusable observation/action semantics,
  units, shapes and fail-safe defaults expanded into a resolved interface.
- `AgentSchema`: versioned compatible agent entity, module, observation and
  action-group types.
- `ModuleInstance`: installed functional module and its availability evidence.
- `EntityInstance`: physical/logical target such as an EV session.
- Simulator edge tables are validated during binding; the compiled
  `Dependency` objects encode versioned health consequences.

## Runtime evidence and health

- `ConnectionState`: connected, disconnected, unknown or not applicable.
- `AvailabilityState`: available, unavailable or unknown.
- `ChannelStatus`: sensor, actuator or communication-channel evidence.
- `FaultEvidence`: immutable cause, domain, target, timing and quality facts.
- `HealthAssessment`: TIC-derived health plus rule/evidence references.
- `TypedObservationSample`: one timestamped, unit-bearing observation value.
- `TypedHealthEvidence`: adapter-neutral cause, quality, availability and
  freshness evidence.
- `TypedRuntimeFrame`: one decision-boundary population frame containing the
  latest samples, runtime entities, membership/topology events and evidence.

`FaultEvidence.fault_mode` is never a `HealthState`. Health assessments may
change over time while the underlying cause remains unchanged.

## Decision objects

- `ObservationPart`: stable typed scalar/vector observation bound to agent,
  sensor, channel, entity/session, semantic type and independently derived
  health.
- `ChannelInstance`: related observation parts with shared source/channel
  provenance and optional channel-level health evidence.
- `SensorInstance`: local or community observation source containing channels.
- `ActionGroupInstance`: exactly-one port family bound to one module/entity.
- `ActionPortInstance`: port type, continuous parameters, validity, bounds,
  dependencies and resource effects.
- `LocalConstraint`: typed local hard/soft constraint.
- `SharedResource`: community resource visible to rewards/critic/trace.
- `InterfaceSnapshot`: complete immutable compiler result for one population
  and step. The compiler owns the composition-independent compatibility
  signature used by checkpoints.
- `LocalActionBundle`: selected port/parameter per active group.
- `TypedTransition`: current/next snapshots, raw/final/executed bundles,
  rewards, termination and runtime events.
- `TypedActionCommand`: technology-neutral selected action and physical unit.
- `TypedExecutionFeedback`: requested, limited and applied execution evidence.

## Identity invariants

- Stable IDs never derive solely from row positions.
- Session replacement creates a new entity identity.
- Every port has one owner, one target and one executable simulator route.
- Every observation identifies entity, source and scope.
- Community observations use the same sensor/channel model as local data.
- A registered but inactive member/asset remains distinct from an unknown one.
- Unknown action semantics are never executed automatically.
