# TI-MARL implementation RFC v1

Status: approved for implementation on 2026-08-16.

## Purpose

Typed-Interface Multi-Agent Reinforcement Learning (TI-MARL) is implemented
as a first-class Algorithms execution unit. A building is one logical agent;
chargers, EV sessions, batteries and deferrable appliances are typed module
and entity instances inside that agent. Runtime instance changes alter the
compiled interface, never the neural parameter dimensions.

The first learning backbone is TI-MAPPO. The Typed Interface Compiler (TIC),
typed policy, action-bundle codec, feasibility layer and trace contracts are
backbone-independent so a TI-MATD3 adapter can be added later.

Archived PILS/JFAL designs, flexibility offers, normal central allocation and
Community Coordinator agents are outside this method. Community coordination
is learned through authorised aggregate observations, coupled rewards, joint
transitions and a training-only variable-cardinality critic.

## Simulator boundary

CityLearn `entity_v1` remains the source of truth for active entities,
relations, observation/action features, ordering, units and bounds. Simulator
1.7.0 adds two backward-compatible subcontracts:

- `runtime_status_v1`: raw events, connection, availability, channel quality,
  timestamps and freshness evidence;
- `entity_action_execution_v1`: requested, post-channel, limited and applied
  commands, with stable entity ownership where observable.

The Simulator never emits a TI-MARL `HealthState`. In particular,
`fault_mode=stuck` remains a cause and is not translated to `STALE`. The TIC
derives health from the cause, duration, freshness, channel semantics,
criticality and versioned rules.

The runtime contract distinguishes:

1. asset connection;
2. asset availability;
3. sensor-channel state;
4. actuator-channel state;
5. community/cloud communication state; and
6. value-quality perturbations.

The Simulator exposes facts and real execution. Algorithms owns dependency
closure, port validity, bound contraction, degraded operating modes and
fallback.

## Typed execution model

For every active building and step, the TIC deterministically produces an
immutable `InterfaceSnapshot` containing entity-bound observations, modules,
entities, relations, health assessments, grouped ports, dynamic bounds, local
constraints, authorised community observations, operational mode and a
compatibility signature.

The initial action groups are:

- stationary storage: `IDLE`, `CHARGE_STATIONARY(fraction)`,
  `DISCHARGE_STATIONARY(fraction)`;
- charger/EV session: `IDLE`, `CHARGE_EV(fraction)`,
  `DISCHARGE_EV(fraction)`;
- deferrable appliance: `IDLE`, `START`.

Continuous fractions are in `[0, 1]` and are converted to the signed,
normalised CityLearn action using the current entity action space and
`available_*_action_normalized` evidence. PV and meters are observation-only
in the first slice.

The shared local actor uses type-conditioned encoders, relation-aware
interaction, one local latent, contextual action-group embeddings, a
categorical port selector and Beta-distributed continuous fractions. All
building controllers share actor parameters. A `CentralSetCritic` pools a
variable set of agent/action summaries during training and emits one value per
stable agent ID. It is absent from decentralised execution.

Local feasibility converts the raw bundle into a final locally feasible
bundle using deterministic typed bounds and the existing analytic projection
semantics. It records every adjustment and does no community optimisation.

## Algorithms integration

The registry name is `TIMARL`. Configuration requires entity interface,
decentralised CityLearn agents and one versioned `typed_interface_v1` YAML.
That public file contains fixed semantics/dependencies, the ordered editable
observation and action view, and health rules. It may be authored by hand or
enriched with a generated Simulator observation/action catalog; every form is
validated against live `entity_specs` before execution. The compiler may split
it internally, but users do not coordinate separate schema/type/health files.
One `TIMARL` execution unit internally manages the changing population of
logical building agents.

The wrapper gains optional structured entity observation/transition hooks.
Existing agents keep the current vector path. TI-MARL stores current and next
snapshots across topology changes: joining agents have no predecessor,
departing agents terminate individually and surviving agents bootstrap.

Checkpoints contain policy/critic/optimiser state, partial rollout, RNG state,
normalisers, compiler/schema/rule hashes and simulator/repository versions.
The first bundle format is `ti_marl_torch` with `deployable=false`; dynamic
inference export is a later milestone.

Typed traces are buffered and snapshot-deduplicated. Every transition remains
reconstructable without per-object/per-step filesystem writes.

## Delivery order

1. Freeze this RFC, the decision log and evidence plan.
2. Implement, test, document and release Simulator 1.7.0.
3. Pin Algorithms to that published release.
4. Implement typed contracts, compiler and health closure.
5. Implement typed actor, grouped decoder, codec and local feasibility.
6. Implement TI-MAPPO, the variable set critic and population-aware rollout.
7. Complete the multi-agent dynamic/health vertical slice.
8. Validate CI, Docker, Union and Deucalion artefacts.
9. Run development and confirmatory experiment campaigns.

Campaign configs, checkpoints and numerical results remain local until
explicitly approved for publication. Design documents, generic templates and
tests are versioned.
