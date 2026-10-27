# Agent extension contract

Add numerical logic in an agent package, a lightweight parameter schema, and one
`AgentSpec` in `algorithms/catalogue.py`. No change to the job runner, simulator
factory or project-wide stage union is needed. Programmatic extensions can call
`register_agent(spec)` before validation/execution; the CLI loads the built-in
catalogue, not arbitrary Python paths supplied in YAML.

A schema imports configuration dependencies only. Use `StrictModel` and declare
parameters rather than enabling `extra="allow"`. The registered configuration
symbol must validate the complete stage, including `algorithm`, `count`,
`frozen`, and its agent-owned parameters.

```python
# my_agent_config.py: deliberately independent of the model implementation
from typing import Literal
from pydantic import Field
from algorithms.configuration.model import StrictModel

class Parameters(StrictModel):
    learning_rate: float = Field(gt=0)

class Stage(StrictModel):
    algorithm: Literal['MyAgent']
    count: Literal[1] = 1
    frozen: bool = False
    hyperparameters: Parameters
```

The declaration uses `module:Symbol` references for the schema and implementation.
`AgentCapabilities` records replay/rollout/offline/no learning, raw/encoded inputs,
flat/entity support, dynamic topology, composition restrictions, persistence,
artifact formats, signal kind and dependency extras. An optional registered
`environment_validator(stage, controller)` checks agent-specific dataset/layout
requirements during preflight and before construction in a real job. Do not
advertise a capability solely because a base class has a no-op hook.

## Numerical implementation

Extend `BaseAgent`, which implements the `ExecutionUnit` boundary. Existing agents
continue to implement `predict`, `update`, `export_artifacts` and optional
`attach_environment`, `prepare_training`, checkpoint and episode hooks. The
constructor receives the existing per-stage compatibility view under
`config['algorithm']`; jobs and templates use the canonical `pipeline` format.

The preferred typed entry points are:

- `decide(observations, deterministic, context: StageContext | None)`.
- `learn_transition(transition: Transition, schedule: UpdateSchedule)`.

Default adapters call the existing numerical methods without changing their
update mathematics. `StageContext` carries the manager value, source stage,
signal kind and member IDs. `Transition` carries current/next inputs, reward,
termination and truncation separately, member IDs and simulator step duration.
It distinguishes the action proposed by the runtime-facing policy, the action
submitted after runtime clipping, and an optional physically executed action.
If the simulator does not report the latter, it remains `None`.

`Pipeline` routes stage contexts and observation profiles; `Ensemble` slices
observations, rewards, actions and IDs consistently per member. Typed transitions
retain their duration/info through both composites. Legacy numerical `update`
methods continue to receive submitted actions, matching their prior contract.
An agent may have additional internal safety projections; this boundary does
not rename those internal values or claim they equal physical execution.

Replay buffers, rollout buffers, target-network updates and offline optimizers
remain agent-owned. The core supplies scheduling flags and lifecycle hooks.
Offline `prepare_training` runs once before environment attachment, within the
job's output tree. The four offline agents share validated trainer settings with
their research CLIs. Frozen/evaluation jobs must not request offline optimization.

## Verification before registration

Add an agent-specific numerical test and use the common contract suite to check
registration, valid configuration, supported environment, prediction/update,
checkpoint round trip where supported, and truthful artifact metadata. Choose a
small real job fixture when introducing a new family. Research quality and
serving compatibility are separate acceptance gates.

For entity models, keep the [shared entity contract](../transformer_entity_controller.md)
and the agent-specific [TPPO](../transformer_ppo_spec.md),
[Transformer MATD3](../transformer_matd3_spec.md) or TI-MARL contracts.
Update `configs/tokenizers/fixtures/entity_obs_sample.json` only when the simulator
schema changes, using `scripts/dump_entity_obs_sample.py`, then validate tokenizer
coverage. TI-MARL interfaces are generated against the actual dataset with
`scripts/generate_typed_interfaces.py`; do not reuse an unrelated unit-test fixture.

Evaluation must not change learned normalizers, exploration clocks or replay. Implement `select_evaluation_policy()` only when selecting a validated alternative to the final policy, and return its selection evidence. New exact-episode resume support requires an audited full serializer and the independent-process continuation test before extending the allowlist.
