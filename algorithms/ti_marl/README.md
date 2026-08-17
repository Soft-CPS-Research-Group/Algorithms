# TI-MARL v1

`TIMARL` is a first-class Algorithms agent that compiles deployment-neutral
typed runtime frames into a health-aware local control interface.
Its first learning backbone is TI-MAPPO: a shared decentralised actor and a
centralised variable-set critic.

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

`fault_mode` is retained as evidence. It is never treated as a Simulator
health label and there is deliberately no universal `fault_mode → HealthState`
mapping.

## Package map

```text
contracts/   immutable object model, enums and compatibility signatures
compiler/    discovery, binding, health derivation, closure and snapshots
policy/      type-shared relational actor and variable-group decoder
learning/    set critic, stable-ID rollout/GAE and hybrid-action MAPPO
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

Experiment configurations are intentionally not stored here. Campaign YAML,
checkpoints, traces and scorecards remain under ignored local experiment/run
paths.

## Action semantics

The action groups are stationary storage, charger/EV session and
deferrable start. The actor selects one valid categorical port and, where
applicable, a Beta-distributed fraction in `[0, 1]`. The fraction is relative
to the currently compiled port; its dynamic bound is applied exactly once by
the CityLearn codec.

Before encoding, the analytic projector enforces compiled validity,
deferrable must-start and joint local import/export headroom. It performs no
community optimisation. Raw and final bundles plus interventions remain in
the typed trace.

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
hashes. A checkpoint may be restored into another composition when its
composition-independent compatibility signature matches.

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

`softcpsrecsimulator==1.7.0` is the pinned minimum runtime for TI-MARL. The
end-to-end vertical slice runs against the installed package; using an adjacent
Simulator checkout is only a deliberate development override.
