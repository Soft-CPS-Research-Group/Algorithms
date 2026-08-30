# TI-MARL v1

`TIMARL` is a first-class Algorithms agent that compiles deployment-neutral
typed runtime frames into a health-aware local control interface.
Its first learning family is PPO with a shared decentralised actor. Two
explicit training variants isolate the value of centralised training:

- `backbone.name: ppo` with `critic.kind: local` (TI-PPO);
- `backbone.name: mappo` with `critic.kind: set` (TI-MAPPO).

Both deploy exactly the same local actor contract. The critic is training-only.

## Runtime boundary

The Simulator owns runtime facts and actual execution:

- `entity_v1` observations;
- additive `runtime_status_v1` evidence;
- additive `entity_action_execution_v1` feedback;
- applied topology events.

The typed interface compiler (TIC) owns control meaning:

- discovery and owner/session binding;
- health derivation and recovery hysteresis;
- dependency closure and valid ports;
- dynamic bound contraction and local degraded modes;
- local feasibility projection and fallback.

The TIC is deliberately **not** a neural network and contains no learned
weights.  It deterministically converts the registered typed interfaces and
runtime facts into an immutable `InterfaceSnapshot`.  The learned boundary
starts only after that snapshot:

```text
typed_agent_interface_v1 + TypedRuntimeFrame
                    ↓
       deterministic TIC compilation
  (health, validity, bounds and constraints)
                    ↓
       neural hierarchical encoder
 (observation → channel → sensor → local latent)
                    ↓
        neural grouped action actor
       (categorical mode + Beta fraction)
                    ↓
     deterministic local feasibility
                    ↓
          adapter action command
```

The actor and encoder are trained jointly and share parameters across
compatible agents and asset instances. The critic is either a shared local
critic or a centralised variable-set critic. It is a separate neural component
used only during training; neither it nor privileged training context belongs
to the deployment bundle.

The learned encoder does not infer signal identity from numeric values alone.
Each observation carries an explicit instance-free semantic family (for
example local load, local PV, price, EV SoC or grid constraint), sensor type,
channel type, unit, scope, use, health and an exact deterministic observation
fingerprint.  Sensor and asset instance IDs are deliberately absent. Thus an
equal-valued load and PV sample remain distinguishable, while adding
`charger_2` reuses the parameters of the known charger type instead of growing
the model. Unknown semantic families fail before action selection and require
an explicit contract/model migration.

`fault_mode` is retained as evidence. It is never treated as a Simulator
health label and there is deliberately no universal `fault_mode → HealthState`
mapping.

## Package map

```text
contracts/   immutable object model, enums and compatibility signatures
compiler/    discovery, binding, health derivation, closure and snapshots
policy/      type-shared relational actor plus local and set critics
learning/    stable-ID rollout/GAE and hybrid-action PPO optimisation
runtime/     local projection, CityLearn codec and buffered typed traces
agent.py     BaseAgent, checkpoint and artifact integration
```

The public algorithm name is `TIMARL`. It requires one standalone pipeline
stage with `simulator.interface: entity`, `central_agent: false` and a static
or dynamic topology. `TIMARL.supports_dynamic_topology` and
`TIMARL.handles_cross_topology_transitions` are both true.

The public registry is a directory containing one
`typed_agent_interface_v1` YAML per formally registered member. Each file is
technology-neutral and declares agent role/type, nested sensor → channel →
observation contracts, actuator ports with exact dependencies, constraints
and fallback. Community data is an ordinary `scope: community` sensor.

There is no runtime support for the retired global document or split contract.
Runs use only:

```yaml
hyperparameters:
  contract_version: ti_marl_v1
  typed_interfaces_dir: /local/path/generated_interfaces
  interface_polling: false
  simulator_bindings_path: /local/path/generated_interfaces/technology_bindings/simulator.yaml
  backbone: {name: mappo}  # or ppo
  actor:
    group_context_kind: action_conditioned  # or local
    deterministic_mode_strategy: expected_signed
    deterministic_expected_signed_gain_by_group_type:
      stationary_storage: 2.0  # optional deterministic calibration; default 1.0
  critic: {kind: set}      # local when backbone.name is ppo
  policy_credit_assignment: typed_group  # or joint_agent
```

Every observation is classified as policy input, safety dependency, runtime
bound, trace-only or excluded with a reason. Profiles fill compact defaults;
the resolved registry expands every default for review and checkpoints.
Trace-only samples remain in snapshots/traces but are filtered before the
actor. Runtime unit, shape and finite-value mismatches invalidate the sample
and trigger declared local safety instead of being coerced silently.

To pin the observations/actions exposed by a saved or live Simulator contract
into a reviewable copy of the same file:

```bash
python scripts/generate_typed_interfaces.py \
  --config /path/to/entity_run.yaml \
  --output /local/path/generated_interfaces
```

Generation writes one file per member, `observation_coverage.csv` and
`interface_manifest.json`; every YAML is reloaded and validated. The optional
`technology_bindings/simulator.yaml` is generated separately and is never part
of an agent's public interface. Simulator
bindings are confined to `SimulatorAdapter`; logical observation names may
deliberately coincide when no translation is necessary. MQTT, Modbus or API gateways map to
the same `TypedRuntimeFrame`, `TypedActionCommand` and execution feedback
contracts.

The generator obtains physical charger/storage bounds and electrical-service
constraints from the selected dataset when they are declared. It never fills
unknown site or phase limits with invented defaults. The resolved registry
expands the compact profile-based YAML for audit.

For a deliberately synthetic development scenario, `simulator.building_ids`
may select a dataset subset and `simulator.electrical_service_overrides_path`
may supply an external `electrical_service_overrides_v1` contract.  The runner
applies that contract only to an in-memory schema copy, refuses to replace a
different electrical-service fact already present in the dataset, and never
modifies `schema.json`.  Such overlays and the interfaces generated from them
are experimental inputs and remain under ignored local run paths.

Experiment configurations are intentionally not stored here. Campaign YAML,
checkpoints, traces and scorecards remain under ignored local experiment/run
paths.

## Action semantics

The action groups are stationary storage, charger/EV session and
deferrable start. The actor selects one valid categorical port and, where
applicable, a Beta-distributed fraction in `[0, 1]`. The fraction is relative
to the currently compiled port; its dynamic bound is applied exactly once by
the CityLearn codec.

Before action encoding, the analytic projector enforces compiled validity,
causal EV service, deferrable must-start and joint local import/export
headroom.  Total and per-phase constraints are kept separate and projected
using each action group's declared phase incidence.  A deferrable `START` is
treated as a binary first-step demand rather than a fractionally scalable
action.  Optional headroom reserve absorbs explicitly configured uncertainty;
it is zero by default. A deployment-neutral
`deferrable_service_margin_seconds` can force a mandatory cycle to start before
its last admissible instant, leaving time to survive a later headroom conflict;
it is also zero by default and every early start is recorded as a safety
intervention. Mandatory EV charging and indivisible deferrable starts share
the same headroom reservation queue ordered by their physical time-to-service
deadline. A binary start is admitted only in full; it cannot consume a partial
reservation that the actuator cannot execute. The projector performs no
community optimisation. Raw and final bundles plus interventions remain in
the typed trace.

Deterministic evaluation also forwards read-only transitions to TI-MARL, so
tracing never requires a policy or critic update. Every transition is buffered;
complete value-bearing snapshots are stored only at the configured interval or
on health/topology events. This keeps deterministic auditability available
without turning `snapshot_interval` into an accidental per-step export.

Both PPO variants support normalized value targets and a Huber critic loss so large
service penalties remain visible without numerically dominating every critic
update. PPO ratio clipping, approximate KL, clip fraction, explained variance
and finite-gradient guards are reported explicitly. These options stabilize
learning; they do not change reward or feasibility semantics.

Shared policies may use `advantage_normalization: per_agent` so a building with
many EV service terms does not set the scale of every other building's policy
gradient. Exploration can likewise be assigned by typed action family through
`entropy_coeff_by_group_type`: for example, stationary storage can retain more
entropy while EV and deferrable decisions remain conservative. The default is
the original global normalization and one shared entropy coefficient.

`policy_credit_assignment: typed_group` uses the interface dependency graph
for actor credit. PPO clipping is applied to each stored action-group
log-probability instead of one joint agent ratio, and a penalty is routed only
to the typed family that can cause it: EV service evidence does not directly
update storage or deferrable choices, for example. Every group still receives
the complete economic and shared-grid reward. A parameter-shared typed group
critic supplies the matching baseline while the centralized set critic still
learns the authoritative total member return. Both critics accept variable
populations and asset counts. Typed routing changes the actor estimator and is
reported as a separate experimental ablation.
`joint_agent` remains the backward-compatible default.

With typed-group credit, the optional
`exclude_intervened_actions_from_policy_loss: true` makes PPO aware of the
local safety shield. If feasibility changes a sampled mode or fraction, the
resulting transition still trains both critics, but that action group is
excluded from the PPO ratio because the sampled action was not the action that
produced the observed return. Raw and final actions remain available in traces,
and training reports intervened and eligible policy-sample counts. The option
is false by default and requires `policy_credit_assignment: typed_group` so an
intervention in one asset does not unnecessarily mask every action of its
building.

`intervention_distillation_coeff` optionally adds a typed auxiliary loss only
for those masked groups. The target is the final locally feasible decision,
so the shared actor can internalize recurring shield corrections without
claiming that the raw action earned the transition reward. A zero coefficient
is the default. A positive value requires intervention-aware masking; the
projector remains authoritative and the distillation target never performs
community optimization.

For explicit fine-tuning stages, `policy_anchor_reset_on_resume: true` resets
the frozen anchor after loading the actor so the regularizer protects the
resumed checkpoint rather than an older anchor stored inside it. It is false
by default and does not affect ordinary checkpoint restoration.

Deterministic replay can use `actor.deterministic_mode_strategy: expected_signed`
to preserve the signed mean of a charge/idle/discharge policy instead of reducing
it to a hard categorical argmax.  The optional
`deterministic_mode_strategy_by_group_type` mapping overrides that choice for a
typed action family.  This permits, for example, smooth stationary-battery
dispatch while EV sessions retain categorical decisions, without changing the
learned weights or the stochastic policy used during training.
`deterministic_expected_signed_gain_by_group_type` can calibrate the magnitude
of that signed mean for a declared action family; `1.0` preserves the exact
policy mean and the result is always clipped to the port fraction range before
the feasibility layer applies physical and safety constraints.
`deterministic_expected_signed_deadband_by_group_type` can then map a
small signed fraction back to `IDLE`, suppressing low-confidence chatter and
avoidable cycling without changing stochastic training. Both mappings are
typed, deterministic-replay parameters and default to no alteration.
For an `argmax` family,
`deterministic_non_idle_logit_margin_by_group_type` can additionally require a
configured logit advantage over `IDLE` before accepting a learned non-idle
action.  Safety and service feasibility remain authoritative and can still
introduce the minimum necessary action afterwards.

Runtime diagnostics report cumulative per-episode `raw_mode_*_rate` and
`final_mode_*_rate` values for every typed action family. These distinguish
decisions made by the actor from charge/start actions introduced by safety and
feasibility, preventing a shield-only controller from being misreported as a
learned policy improvement.

For sub-hourly datasets, `discount_timebase_seconds` interprets `gamma` and
`gae_lambda` against a physical reference period instead of silently applying
hourly-looking values at every 15-minute transition.  The resolved effective
discounts and runtime step duration are exported in checkpoints and training
diagnostics.  Omitting the option preserves the historical step-based
semantics exactly.

`ev_planning` is an optional deployment-causal auxiliary objective for the EV
actor.  It derives auditable targets only from typed observations available to
the deployed policy: confirmed connection, service deficit, time to departure,
charger capability, the current tariff and explicitly declared price
forecasts before departure.  It never consumes future Simulator state and
never emits a runtime action.  The categorical loss teaches when to choose
`CHARGE_EV` or `IDLE`; a separate continuous loss teaches the requested charge
fraction.  Targets are balanced first by action mode and then by causal reason
so abundant idle instants cannot erase cheap or urgent charging examples.

A small bounded reservoir, configured with
`replay_capacity_per_reason` and `replay_samples_per_reason`, retains auxiliary
examples independently for cheap charging, urgent charging and deliberate
waiting.  This prevents rare EV decisions from being forgotten after an early
charge changes the subsequent on-policy state distribution.  Only the causal
auxiliary loss uses this reservoir: PPO ratios, advantages, critics and rewards
remain strictly on-policy.  Reservoir snapshots retain only policy-visible
typed observations, action groups and agent metadata; compiler evidence and
constraints that the actor never reads are not duplicated. Reservoir contents
and seen counts round-trip in a training checkpoint and are omitted when replay
restoration is disabled.
`ev_actor_charge_ownership_rate`, `ev_projector_charge_takeover_rate`, target
coverage and per-mode recall make the division between learned control and the
safety shield measurable.

`storage_planning` is the corresponding optional causal auxiliary for
stationary batteries. It uses only typed storage SoC/capability, local net
exchange, the current tariff, and explicitly declared tariff forecasts. It
labels local-PV capture, materially cheap charging, materially expensive
import-offset discharge, and deliberate idle decisions. The labels train the
shared neural actor but never issue runtime commands; compiled bounds and the
local feasibility projector remain authoritative. As with EV planning, rare
storage modes use a reason-stratified bounded reservoir that is isolated from
the on-policy PPO rollout and round-trips in training checkpoints. Keeping the
block disabled gives the exact pure TI-MAPPO ablation.

An optional typed behavior-cloning warm start can execute deterministic
`RBCSmartPolicy` actions for complete demonstration episodes and decode those
actions back into the same valid typed action groups used by the actor.  It
pretrains only the shared actor before PPO begins: demonstration transitions
never enter the PPO rollout, the central critic is not trained by the teacher,
and the teacher never mixes actions into an on-policy episode.  After
pretraining, full demonstration snapshots are discarded; checkpoints retain
only the learned actor and an auditable summary of the warm start.  This is an
initialisation strategy, not evidence that TI-MARL outperforms its teacher.

The default `actor.group_context_kind: local` decodes every local action group
from the shared agent latent plus its group type.  The optional
`action_conditioned` variant lets each group query the same typed observation
tokens again before the groups interact.  Four instance-free structural
relations distinguish observations owned by that module, other observations,
local observations and community observations.  It does not embed concrete
building or asset IDs, so adding agents or another instance of a known asset
type still does not resize the actor.  The selected context kind is part of the
checkpoint architecture contract and cannot be changed silently on restore.

SMART demonstrations are strongly mode-imbalanced because `IDLE` is normally
far more common than charge, discharge or start.  The warm start may therefore
apply bounded inverse-frequency weights independently within each action-group
type using `balance_action_modes`, `mode_balance_exponent` and
`max_mode_weight`.  Counts and effective weights are exported as diagnostics.
`balanced_loss_kind: hierarchical_mode_mean` is the stronger alternative: it
averages examples within each mode, modes within each typed action family, and
then the families. This prevents frequent `IDLE` labels from erasing rare but
important charge, discharge and start decisions while keeping each typed
family equally represented.
An optional final `calibration_epochs` phase then optimises the original,
unweighted demonstration likelihood (at `calibration_learning_rate`) to restore
the observed action prior after rare-mode representation learning. Balanced and
calibration losses/batches are reported separately. These settings alter only
actor initialisation; they do not reweight PPO rewards or the subsequent
on-policy objective.

During PPO updates, observation, channel, sensor, agent and action-group sets
from the rollout are packed by stable indices and evaluated in batches. This
preserves the same typed set reductions and hybrid-action densities while
avoiding per-observation device transfers and discarded runtime bundles.
Runtime actor inference also encodes every active agent in one packed call;
static typed identity features are cached without caching values or health.
Update duration and evaluated samples per second are exported with the training
diagnostics.

Initial fail-safe closure also isolates invalid site meters and actuator
channels, blocks charge/EV-V2G/deferrable start during a grid outage, preserves
reliable stationary discharge, prioritises safe EV service under uncertain
SoC/schedule, and falls back to local-only control when community telemetry is
lost.

## Population and topology semantics

- Stable Simulator building order is preserved for actions and rewards.
- A new agent has no predecessor in GAE.
- A removed agent receives an individual terminal transition.
- Surviving agents bootstrap across a topology change.
- A topology change does not resize actor or critic parameters or clear the
  rollout.
- Charger session replacement rebinds the EV identity without retaining a
  stale session entity.

## Persistence

`ti_marl_checkpoint_v1` stores model/optimizer state, the rollout, RNG state,
normalisers/compiler state, resolved configuration, versions and compatibility
hashes. It also pins the backbone and critic kind, so a TI-PPO checkpoint cannot
be loaded accidentally into TI-MAPPO or vice versa. A checkpoint may be
restored into another composition when its composition-independent
compatibility signature matches.

A compiler-version change is rejected by default.  A diagnostic or migration
run must opt in explicitly with `allow_checkpoint_compiler_migration: true`,
and all contract, schema, type-registry and health-rule hashes must still
match.  The option is not an implicit compatibility bypass.

Traces use buffered gzip JSONL chunks and content-addressed snapshot
deduplication. Every transition references complete current/next snapshots
and records Simulator-reported execution. Exported bundles use
`format: ti_marl_torch` and remain `deployable: false` in the first cut. A
separate actor-only deployment handoff contains the TIC contracts, health,
feasibility, normalisation and compatibility signature, never the critic.

## Verification

Focused tests live in:

- `tests/test_ti_marl_compiler.py`;
- `tests/test_ti_marl_policy_runtime.py`;
- `tests/test_ti_marl_runtime_contracts.py`;
- `tests/e2e/test_ti_marl_high_frequency_stress.py`;
- `tests/e2e/test_ti_marl_vertical_slice.py`.

The real-Simulator vertical slice combines multiple buildings and local asset
types, member join/leave, sensor/actuator/asset/community failures, a long
`stuck` event, recovery hysteresis, fixed parameter count, training across
topology changes and command-to-execution trace reconciliation.

Dataset bindings can be replayed without training:

```bash
python scripts/validate_ti_marl_interfaces.py \
  --config /path/to/entity_run.yaml \
  --interfaces-dir /local/path/generated_interfaces \
  --simulator-bindings /local/path/generated_interfaces/technology_bindings/simulator.yaml
```

`softcpsrecsimulator==1.8.0` is the pinned runtime for TI-MARL. The
end-to-end vertical slice runs against the installed package; using an adjacent
Simulator checkout is only a deliberate development override.

Training, development checkpoint selection and confirmation follow
[`notes/ti-marl/experiment-protocol-v1.md`](../../notes/ti-marl/experiment-protocol-v1.md).
The protocol separates neural and Simulator seeds, fingerprints every paired
evaluation surface and prevents confirmation results from influencing model
selection.
