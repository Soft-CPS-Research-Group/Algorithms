# TI-MARL v1

`TIMARL` is a first-class Algorithms agent that compiles the Simulator's
facts-only entity contract into a typed, health-aware local control interface.
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

The three generic versioned inputs live in [`configs/ti_marl`](../../configs/ti_marl):

- `agent_schema_v1.yaml`;
- `type_registry_v1.yaml`;
- `health_rules_v1.yaml`.

Experiment configurations are intentionally not stored here. Campaign YAML,
checkpoints, traces and scorecards remain under ignored local experiment/run
paths.

## Action semantics

The v1 action groups are stationary storage, charger/EV session and
deferrable start. The actor selects one valid categorical port and, where
applicable, a Beta-distributed fraction in `[0, 1]`. The fraction is relative
to the currently compiled port; its dynamic bound is applied exactly once by
the CityLearn codec.

Before encoding, the analytic projector enforces compiled validity,
deferrable must-start and joint local import/export headroom. It performs no
community optimisation. Raw and final bundles plus interventions remain in
the typed trace.

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
`format: ti_marl_torch` and remain `deployable: false` in the first cut.

## Verification

Focused tests live in:

- `tests/test_ti_marl_compiler.py`;
- `tests/test_ti_marl_policy_runtime.py`;
- `tests/e2e/test_ti_marl_vertical_slice.py`.

The real-Simulator vertical slice combines multiple buildings and local asset
types, member join/leave, sensor/actuator/asset/community failures, a long
`stuck` event, recovery hysteresis, fixed parameter count, training across
topology changes and command-to-execution trace reconciliation.

Until `softcpsrecsimulator==1.7.0` is published, that end-to-end test must be
run against the adjacent Simulator checkout via `PYTHONPATH`. Algorithms must
only update its dependency pin after the release package has passed a clean
installation check.
