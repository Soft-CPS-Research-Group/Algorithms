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

The public interface is one versioned, editable document:
[`typed_interface_v1.yaml`](../../configs/ti_marl/typed_interface_v1.yaml).
It has three readable sections:

- `fixed`: agent/module types, relations, dependencies, local constraints and
  shared resources;
- `observations` and `actions`: the ordered network view and controllable
  groups;
- `health`: semantic thresholds, criticality and recovery hysteresis.

The compiler internally derives its former schema/type/health views from this
one file. The split three-path constructor remains temporarily accepted for
old external configurations, but new runs use only:

```yaml
hyperparameters:
  contract_version: ti_marl_v1
  typed_interface_path: configs/ti_marl/typed_interface_v1.yaml
```

The observation list is intentionally hand-editable. `features` fixes the
network slot order; `required_features` turns a Simulator mismatch into a
setup error. Features selected but not marked required keep stable optional
zero slots across compositions.

To pin the observations/actions exposed by a saved or live Simulator contract
into a reviewable copy of the same file:

```bash
python scripts/generate_typed_interface.py \
  --entity-specs /tmp/entity_specs.yaml \
  --output /tmp/my_typed_interface.yaml

python scripts/generate_typed_interface.py \
  --config /path/to/entity_run.yaml \
  --output /tmp/my_typed_interface.yaml
```

Generation does not invent control semantics: it preserves the editable fixed
and health sections and appends an auditable Simulator catalog. The generated
YAML is reloaded and validated before the command succeeds. Every exported
TI-MARL artifact also contains `typed_interface.resolved.yaml` with the live
catalog used by that run.

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

`softcpsrecsimulator==1.7.0` is the pinned minimum runtime for TI-MARL. The
end-to-end vertical slice runs against the installed package; using an adjacent
Simulator checkout is only a deliberate development override.
