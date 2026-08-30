# Agents

Guide for developing, registering, and configuring learning agents.

## Overview

Agents live in `algorithms/agents/`, `algorithms/transformer_ppo/`, and
`algorithms/transformer_matd3/`. They extend `BaseAgent`. Infrastructure
(runner, wrapper, tracking, checkpoints) is provided—focus on algorithm logic.

> **Note:** The training loop is handled by the runner. Agents receive already processed observations from the wrapper.

## Delegated Execution

- When suitable subagents are available, delegate each independent deterministic task before executing it in the parent.
- This includes test runs, log extraction, remote status checks, polling, configuration preparation, artifact collection, and routine validation.
- Run independent tasks concurrently when practical. Assign one subagent ownership of each remote run through terminal completion and evidence collection.
- The parent keeps scope, architecture, trade-offs, risk judgments, acceptance decisions, and final synthesis.
- The parent may execute only work needed to unblock delegation or to verify disputed, high-risk, or decision-critical evidence.
- Require concise status and evidence from subagents; do not request or retain raw logs unless needed for diagnosis.

## Base Contract

Extend `algorithms/agents/base_agent.py`:

| Method | Description |
|--------|-------------|
| `predict(observations, deterministic)` | Return actions for current step |
| `update(obs, actions, rewards, next_obs, terminated, truncated, *, update_target_step, global_learning_step, update_step, initial_exploration_done)` | Learning step (respects scheduling flags) |
| `export_artifacts(output_dir, context)` | Save outputs and return manifest metadata |

**Optional:**

| Method | Description |
|--------|-------------|
| `save_checkpoint(output_dir, step)` | Persist training state |
| `load_checkpoint(checkpoint_path)` | Resume from checkpoint |
| `attach_environment(observation_names, action_names, action_space, observation_space, metadata)` | Receive environment metadata |
| `is_initial_exploration_done(global_learning_step)` | Gate warm-up phase |

**Notes:**
- Set `self.use_raw_observations = True` if your agent needs unprocessed observations (see `RuleBasedPolicy`)
- If `resume_training` is enabled in config, the runner calls `load_checkpoint(...)`

## Creating a New Agent

### 1. Implement

```python
from algorithms.agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, config: dict) -> None:
        super().__init__()
        # Read config["algorithm"]["hyperparameters"], etc.

    def predict(self, observations, deterministic=None):
        ...

    def update(self, observations, actions, rewards, next_observations,
               terminated, truncated, *, update_target_step, global_learning_step,
               update_step, initial_exploration_done):
        ...

    def export_artifacts(self, output_dir, context=None):
        return {"model_path": "...", ...}
```

### 2. Register (Required)

In `algorithms/registry.py`:

```python
from algorithms.agents.my_agent import MyAgent

ALGORITHM_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "MADDPG": MADDPG,
    "RuleBasedPolicy": RuleBasedPolicy,
    "MyAgent": MyAgent,  # Add here
}
```

⚠️ **Without registration, the agent cannot be instantiated by the runner.**

### 3. Config & Schema

- `configs/config.yaml` → add parameters under `algorithm.hyperparameters`
- `utils/config_schema.py` → add validation model if needed

## Available Algorithms

| Algorithm | Description |
|-----------|-------------|
| `MADDPG` | Multi-Agent DDPG with replay buffer, actor-critic networks |
| `RuleBasedPolicy` | Heuristic controller for EV charging (uses raw observations) |
| `AgentTransformerPPO` | Entity-interface Transformer PPO with dynamic-topology support and optional auxiliary behavior-cloning loss from separate deterministic `RBCSmartPolicy` demonstrations |
| `AgentTransformerMATD3` | Entity-interface MATD3 with per-building Transformer actors, centralized twin critics, dynamic topology, residual control, and optional behavior cloning |
| `SingleAgentRL` | Schema placeholder only |

The Transformer implementations live in `algorithms/transformer_ppo/` and
`algorithms/transformer_matd3/`.
Use the [shared Transformer/entity contract](docs/transformer_entity_controller.md)
for reusable invariants. Use the [TPPO specification](docs/transformer_ppo_spec.md)
for TPPO. For Transformer MATD3, start with the
[operational guide](docs/transformer_matd3.md), then use the
[technical specification](docs/transformer_matd3_spec.md) and
[ADRs](docs/adr/README.md) when changing architecture.

## Runtime Flow

```
run_experiment.py
    ↓
Validate config (schema)
    ↓
Build env + wrapper + agent
    ↓
Training loop:
    predict() → actions
    update()  ← rewards/obs
    ↓
export_artifacts() → runs/jobs/<job_id>/
```

## Wrapper

`utils/wrapper_citylearn.py` handles:
- Episodes and step management
- Observation encoding (via `configs/encoders/default.json`)
- Update scheduling (respects `update_step`, `update_target_step` flags)
- Metrics tracking (MLflow or JSONL)
- Manifest metadata generation

> Encoders keep training and serving consistent; usually unchanged after initial setup.

## Entity Interface (New Contract)

The shared contract is algorithm-independent. TPPO-specific rules, including
pending-action validation, on-policy flushing, BC, safety, checkpoints, and
ONNX export, are in the TPPO specification.

When `simulator.interface: entity`, the wrapper uses the CityLearn entity contract
instead of legacy flat vectors.

- Input from simulator: entity payload (`tables`, `edges`, `meta`) at `reset/step`.
- Adaptation layer: `utils/entity_adapter.py` converts entity payload to per-agent vectors.
- Actions from agent: still returned as `List[List[float]]` (one vector per agent).
- Output to simulator: wrapper converts agent vectors back into entity action tables.

Where this happens:
- Mode detection: `Wrapper_CityLearn.__init__` (`_entity_interface_mode`).
- Observation conversion: `_apply_entity_layout(...)`.
- Action conversion to simulator payload: `_to_env_actions(...)`.
- Environment metadata for agents (`entity_specs` included): `_attach_model_environment_metadata(...)`.

Dynamic topology notes:
- If `simulator.topology_mode: dynamic`, topology can change during runtime.
- Wrapper rebuilds layout automatically on `topology_version` change.
- Current guardrail: `MADDPG` in `entity+dynamic` raises fail-fast on runtime topology mutation.
  Use `RuleBasedPolicy` (or another dynamic-ready agent) for dynamic topology scenarios.
- `AgentTransformerPPO` supports `entity+dynamic`. Optional behavior cloning collects separate deterministic `RBCSmartPolicy` demonstrations before PPO rollouts and applies an auxiliary actor-only loss; it never changes actor environment actions.
- On topology changes, TPPO rebuilds the Smart teacher. Demonstrations retain their encoded representation and layout signature. BC pretraining groups layout-compatible demonstrations by stored signature and trains every stored compatible signature group with its stored layout, including historical topologies. TPPO fails before PPO when no usable demonstrations exist.

## Remote Experiment Lifecycle

- Job submission is not experiment completion.
- A running job is not experiment completion.
- Completion requires terminal state, artifacts, KPIs, diagnostics, and requested comparisons.
- Estimate the next useful check from observed progress and runtime.
- Do not repeatedly poll unchanged remote state.
- Report only changed state, failures, or requested checkpoints.
- Do not end a delegated goal while required remote work remains.

## Tokenizer Fixture (Entity Interface)

`configs/tokenizers/fixtures/entity_obs_sample.json` is the pinned
simulator-schema snapshot used by `validate_config` and by the tokenizer
unit tests. The 5 hard-fail rules run against this fixture, so a
tokenizer JSON that omits or misclassifies a column fails at
config-load time.

**Regenerate whenever the simulator schema changes** (columns
added/removed/renamed in any entity table, new asset type, adapter
emission order changed):

```bash
python scripts/dump_entity_obs_sample.py \
    --config configs/templates/dynamic/rule_based_entity_dynamic_assets_only_local.yaml \
    --output configs/tokenizers/fixtures/entity_obs_sample.json
```

If the new fixture uncovers uncovered features, `pytest
tests/test_entity_tokenizer_config_schema.py` fails with a rule-1
(coverage) violation — update `configs/tokenizers/entity_default.json`
to classify the new columns (SRO / NFC / CA / excluded), then re-run
tests.

Any entity-mode YAML config works for the dump; the script only reads
`env.entity_specs` (feature names + row ids) and the initial payload's
edge structure.

## Outputs

After training completes, all artifacts are organized in a job-specific directory:

```
runs/jobs/<job_id>/
├── logs/                      # Training logs
├── progress/                  # progress.json updates during training
├── results/                   # Final metrics and KPIs
│   ├── result.json
│   ├── summary.json
│   └── simulation_data/
├── checkpoints/               # Training checkpoints (if enabled)
├── onnx_models/               # Exported ONNX models
├── config.resolved.yaml       # Full resolved configuration
└── artifact_manifest.json     # Metadata for all exported artifacts
```

The manifest (`artifact_manifest.json`) contains metadata returned by `export_artifacts()` and is used for bundle validation and deployment.

## Tests

Run the test suite to verify implementation:

```bash
pytest
```

Coverage includes:
- **Schema validation** - Config structure and types
- **Registry** - Agent registration and instantiation
- **Agent behavior** - MADDPG and RBC logic
- **Wrapper** - Encoding, scheduling, metrics
- **Checkpointing** - Save/resume functionality
- **Manifest** - Artifact metadata generation
- **Bundle validation** - Export contract compliance
