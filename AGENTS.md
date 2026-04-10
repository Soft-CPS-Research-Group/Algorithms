# Agents

Guide for developing, registering, and configuring learning agents.

## Overview

Agents live in `algorithms/agents/` and extend `BaseAgent`. Infrastructure (runner, wrapper, tracking, checkpoints) is provided—focus on algorithm logic.

> **Note:** The training loop is handled by the runner. Agents receive already processed observations from the wrapper.

### TransformerPPO — Design Motivation

The `AgentTransformerPPO` introduces a Transformer backbone to address three design considerations that fixed-topology MLP agents (like MADDPG) cannot handle:

1. **Variable CA count without retraining.** The system must adapt to changes in the number of controllable assets at runtime, supporting variable numbers of CA inputs and outputs. A Transformer processes a variable-length token sequence via self-attention, naturally handling different asset counts.

2. **Strict one-to-one CA input/output mapping.** Each CA's observation features are tokenized into a dedicated token. The actor head produces exactly one action per CA token — the architecture enforces this structurally.

3. **Additional context inputs without spurious outputs.** The system incorporates a single global NFC input (the RL token) and a variable-sized set of SROs (shared read-only context like weather, pricing, time), while keeping all outputs aligned with their respective CAs.

**Thesis goal:** A single model class handles different prosumer configurations (varying numbers and types of controllable assets per building) without retraining from scratch. Weights for shared asset types transfer across topologies, and the self-attention mechanism naturally adapts to different token counts.

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

## TransformerPPO — Validation Phases

Incremental validation plan for `AgentTransformerPPO`. Prove it works on the simplest case first, then scale up.

| Phase | Setup | Goal |
|-------|-------|------|
| **1** | 1 building, fixed topology (no variance in input count) | Baseline: does the agent learn at all? Compare reward curve and KPIs against RBC/MADDPG on the same single building. |
| **2** | 1 building, variable topology (different CA counts across runs) | Core thesis: same model architecture handles different asset configurations. Train on building with 3 CAs, evaluate on building with 1 CA (or vice versa). |
| **3** | 1 building, variable topology + KPI analysis | Assess whether results are good enough or if hyperparameters/architecture need tuning before scaling. |
| **4** | Multiple buildings (scale up) | Full multi-building training. Same as Phase 3 but with more buildings to verify the approach generalizes. |

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
