# TI-MARL implementation RFC v1

Status: approved for implementation on 2026-08-16; deployment-neutral
per-agent interface revision approved on 2026-08-17.

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

## Public agent interface and adapter boundary

The public contract is `typed_agent_interface_v1`: one human-editable YAML
document per registered building agent.  It describes stable agent identity,
role, building type, installed or pre-registered sensor/asset instances,
channels, typed observations, actuators, actions, local constraints and
explicit health dependencies.  It contains no Simulator, CityLearn, MQTT,
Modbus or endpoint-specific bindings; logical names may coincide with a
technology when that adapter requires no translation.

An observation is addressed by the stable path
`agent/sensor/channel/observation`.  Every available field is typed and
classified as policy input, safety dependency, runtime bound, trace-only or
explicitly excluded with a reason.  Community aggregates use exactly the same
model as local telemetry: `community` is a sensor with `scope=community`.
Only fields declared `policy_input=true` enter the local actor.

Capability profiles keep repeated definitions compact.  The resolved
interface expands every inherited semantic type, unit, shape, default and
health consequence and is the auditable contract used by compatibility,
learning, safety and traces.  Profiles are packaged with the policy; new
instances of known compatible types do not resize a network.  Unknown action
semantics are never executed automatically.

Technology-specific adapters produce a neutral `TypedRuntimeFrame` and
consume neutral `TypedActionCommand` objects.  The Simulator adapter is one
such adapter; real deployments may provide MQTT, Modbus or service adapters
without changing the public interface or the TIC.

## Simulator boundary

CityLearn `entity_v1` remains the source of runtime facts for simulated runs.
The Simulator adapter binds those facts to the deployment-neutral registered
interfaces. Simulator 1.7.0 adds two backward-compatible subcontracts:

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

The Simulator exposes facts and real execution. The adapter owns technological
bindings. Algorithms owns semantic validation, health derivation, dependency
closure, port validity, bound contraction, degraded operating modes and
fallback.

## Typed execution model

For every active building and decision boundary, the TIC deterministically
updates an immutable `InterfaceSnapshot`. Static structure is compiled only at
registration, atomic reload, topology or session boundaries; current values,
freshness, health and bounds are updated incrementally from each
`TypedRuntimeFrame`. Thresholds use physical duration rather than timestep
counts.

The TIC is not a latent model and has no learned parameters. Its output is the
typed, causal snapshot consumed by the learned policy. The neural boundary is
therefore explicit: the hierarchical encoder learns representations from
snapshot observations, health and metadata; the grouped actor learns action
mode and continuous-fraction distributions; and the centralized set critic is
used only to learn values during training. Safety closure and final local
feasibility remain deterministic on both training and deployment paths.

The observation encoder is hierarchical: typed observations are aggregated
into channel representations, then sensor/asset representations, before local
asset/group interaction and pooling into one local latent. An EV session is a
runtime entity distinct from its persistent charger and never inherits cached
state from a previous session.

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
semantics. Total-service and per-phase constraints are represented separately,
including action-group phase incidence and replacement of the previously
applied controllable power. Causal EV-service reservations precede
discretionary charging, and binary deferrable starts are admitted only when
their complete first-step demand fits. It records every adjustment and does no
community optimisation.

## Health and fail-safe policy

Health is assessed per observation/channel. Each action dependency explicitly
defines consequences for non-nominal states; there is no universal
`DEGRADED -> lower power` translation. Supported consequences include port
invalidation, bound contraction, degraded-mode selection, declared fallback,
group suspension and agent safe mode. Constraint and channel safety precedes
service preservation, which precedes optimisation.

The initial fail-safe profile includes: agent isolation when the main/grid
meter is unavailable; no charging without trustworthy site/phase headroom; no
export without trustworthy export headroom; no charger action without a
confirmed connection and actuator capability; no V2G under uncertain EV SoC
or schedule; service-priority maximum safe charging under explicitly declared
uncertain-SoC rules; no stationary storage action without trustworthy SoC and
bounds; local continuation without community/price/forecast telemetry; and
deterministic deferrable behaviour only when start/running evidence is safe.
Every fallback remains subject to physical, BMS, grid, phase and local
feasibility bounds.

## Algorithms integration

The registry name is `TIMARL`. Configuration uses `typed_interfaces_dir`, with
one `typed_agent_interface_v1` file per registered logical building. The
prototype global `typed_interface_v1` is rejected rather than maintained as a
second runtime representation. One `TIMARL` execution unit may manage the
changing population during central simulation/training; the same actor, TIC,
profiles, normalisation and feasibility components run per edge agent during
deployment. The training-only critic is not deployed.

The interface registry provides all-or-nothing reload. Invalid changes retain
the previous generation. Polling is optional and disabled by default, so YAML
parsing and filesystem access are absent from the steady-state inference path.

The wrapper gains optional structured entity observation/transition hooks.
Existing agents keep the current vector path. TI-MARL stores current and next
snapshots across topology changes: joining agents have no predecessor,
departing agents terminate individually and surviving agents bootstrap.

Checkpoints contain policy/critic/optimiser state, partial rollout, RNG state,
normalisers, capability/profile/compiler hashes and resolved interface hashes.
Compatibility signatures depend on semantic types and contracts, not concrete
instance IDs. Deployment bundles contain the actor, TIC, profiles,
normalisation and feasibility components but exclude the central critic.

Typed traces are buffered and snapshot-deduplicated. Every transition remains
reconstructable without per-object/per-step filesystem writes.

## Delivery order

1. Freeze this RFC, the decision log and evidence plan.
2. Implement, test, document and release Simulator 1.7.0.
3. Pin Algorithms to that published release.
4. Replace the prototype with per-agent contracts, registry and neutral runtime
   frames/commands.
5. Implement the Simulator adapter, hierarchical compiler/actor and fail-safe
   closure.
6. Generate and validate interfaces for the canonical static 15-minute and
   dynamic-topology datasets.
7. Complete static annual, dynamic, health, deployment-parity and scale gates.
8. Validate CI, Docker, Union and Deucalion artefacts.
9. Run development and confirmatory experiment campaigns.

Campaign configs, checkpoints and numerical results remain local until
explicitly approved for publication. Design documents, generic templates and
tests are versioned.

Development-only electrical-service scenarios follow the same provenance
boundary. An external versioned overlay may populate missing site/phase limits
on an in-memory schema copy, but it cannot replace conflicting dataset facts
and is never written back into the dataset. Canonical results must state
whether limits came from the dataset or from such an explicit local overlay.
