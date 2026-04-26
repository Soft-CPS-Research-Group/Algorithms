# Agents

Guide for developing, registering, and configuring learning agents.

## Overview

Agents live in `algorithms/agents/` and extend `BaseAgent`. Infrastructure (runner, wrapper, tracking, checkpoints) is provided—focus on algorithm logic.

> **Note:** The training loop is handled by the runner. Agents receive already processed observations from the wrapper.

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
| `SingleAgentRL` | Schema placeholder only |

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
