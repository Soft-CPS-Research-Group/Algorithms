# Training Platform Guide

This document explains how the fixed part of the project is structured, what responsibilities it owns, and how students can extend it with new algorithms. Treat this as the reference manual for anyone onboarding to the codebase.

## 1. Architecture Overview

```
┌─────────────────────────┐      ┌────────────────────────────┐
│ configs/*.yaml          │      │ utils/config_schema.py     │
│ (student-editable)      │ ───▶ │ Pydantic validation        │
└─────────────────────────┘      └──────────────┬─────────────┘
                                                │ validated config
                                        ┌───────▼─────────────────────┐
                                        │ run_experiment.py           │
                                        │ - MLflow, logging, job dirs │
                                        │ - create_agent via registry │
                                        └───────┬─────────────────────┘
                                                │ BaseAgent instance
                                 ┌──────────────▼─────────────┐
                                 │ utils/wrapper_citylearn.py │
                                 │ - wraps CityLearn Env      │
                                 │ - encoding, progress,      │
                                 │   checkpoint managers      │
                                 └──────────────┬─────────────┘
                                                │ observations/actions
                                   ┌────────────▼───────────┐
                                   │ algorithms/agents/*    │
                                   │ (student extensions)   │
                                   └────────────────────────┘
```

Key responsibilities:
- **run_experiment.py** orchestrates a run end-to-end: validates config, prepares directories, starts MLflow (if enabled), instantiates the algorithm class, and writes an artifact manifest after training.
- **Wrapper_CityLearn** hides CityLearn-specific plumbing. It encodes observations, logs metrics, writes progress files, schedules checkpoints, and exposes environment metadata for inference.
- **BaseAgent + registry** define the contract students must fulfil. Algorithms handle model logic, replay buffers, ONNX export, and provide metadata consumed by the manifest.

## 2. Configuration & Schema

All configurations are validated against `utils/config_schema.py` before training begins.

Sections:
- `metadata`: experiment and run identifiers (used in MLflow).
- `runtime`: paths injected at runtime (leave null in files).
- `tracking`: log level, MLflow toggle.
- `checkpointing`: resume/transfer-learning parameters and cadence.
- `simulator`: dataset, schema path, reward function.
- `training`: global training schedule knobs.
- `topology`: **derived** environment dimensions (num agents, observation/action shapes). These remain null in version-controlled configs and are filled by the wrapper.
- `algorithm`: algorithm name and its parameters. The schema currently supports MADDPG and a rule-based placeholder. New algorithms must extend the schema.

### Validation

`validate_config(raw_config)` throws on missing keys/invalid values. Errors list the exact field:
```
Configuration validation failed:
- algorithm -> networks -> actor -> layers: layers must contain at least one hidden dimension
```
This prevents students from launching malformed runs.

## 3. Runtime Workflow

1. **Configuration** – Student clones `configs/config.yaml` or a template under `configs/templates/` and sets algorithm-specific values.
2. **run_experiment** – CLI parses args, validates config, sets runtime paths, and logs the start of the run. If MLflow is disabled, a warning is issued and metrics fall back to local JSONL logs.
3. **Agent instantiation** – `create_agent(config)` looks up `algorithm.name` in `algorithms/registry.py`; the registry maps names to classes.
4. **Wrapper** – handles the CityLearn loop, invoking `BaseAgent.predict`/`update`. It logs metrics and checkpoints using helper classes.
5. **Artifacts** – After training, `agent.export_artifacts` emits ONNX models/metadata, and `build_manifest` creates `artifact_manifest.json` capturing topology, encoders, reward config, and algorithm metadata for inference deployment.

## 4. Checkpointing

Checkpoints are managed by `utils/checkpoint_manager.CheckpointManager`:
- Saves every `checkpoint_interval` steps (post exploration phase) to `<log_dir>/checkpoints/`.
- Optionally logs to MLflow when tracking is enabled.
- Agents must implement `save_checkpoint(output_dir, step)` returning the checkpoint path. The MADDPG example stores PyTorch state dicts and replay buffer state.

**Resuming**: Setting `checkpointing.resume_training: true` reloads model weights/optimizers, but the simulator restarts from the beginning of the episode. The rationale: CityLearn does not expose environment snapshots, so only agent state is restored. Document this for students.

## 5. Metrics and Logging

- **MLflow Enabled**: Metrics are logged every step/episode; KPIs and artifact manifests are uploaded as run artifacts.
- **MLflow Disabled**: Metrics are appended to `<log_dir>/metrics.jsonl` via `LocalMetricsLogger`. The file uses JSONL (one record per line) for easy ingestion.
- `tracking.log_level` controls Loguru output; informative messages (`info`, `debug`) are sprinkled through the runner and wrapper to aid debugging.

## 6. Artifact Manifest

`artifact_manifest.json` (written beside ONNX models) contains:
- Config snapshot (`metadata`, `training`, `topology`, algorithm hyperparameters).
- Environment metadata from `Wrapper_CityLearn.describe_environment()`:
  - `observation_names` and per-observation encoder definitions (type + parameters like `x_min`, `x_max`, `classes`, etc.).
  - `action_bounds` and `action_names` (if provided by CityLearn).
  - Reward function class name and non-private attributes (parameters used during training).
- Agent metadata returned by `BaseAgent.export_artifacts` (e.g., ONNX paths, observation/action dimensions).

This manifest enables a separate inference service to apply the same preprocessing and routing logic that was used during training.

## 7. Extending the Platform with New Algorithms

1. **Implement the Agent**
   - Subclass `BaseAgent` and implement `predict`, `update`, `save_checkpoint`, `export_artifacts` (and optionally `load_checkpoint`).
   - Reuse helper utilities (replay buffers, noise, etc.) as needed.

2. **Register the Agent**
   - Add an entry to `algorithms/registry.py` mapping the algorithm name to the new class.

3. **Extend the Schema**
   - Create a new Pydantic model mirroring the algorithm’s configuration needs.
   - Update `ProjectConfig.algorithm` to accept the new model.
   - Add a template config in `configs/templates/` highlighting the new fields.

4. **Export Metadata**
   - Ensure `export_artifacts` writes whatever the inference service requires. For ONNX-based policies, call `mlflow.log_artifact` if MLflow is active and return a metadata dict describing the exports.

5. **Testing & Documentation**
   - Provide small smoke tests and update documentation to explain how to use the new algorithm.

## 8. Logging Guidelines

- Use `logger.info` for high-level progress (episode completion, training start/end).
- Use `logger.debug` for detailed diagnostics (derived topology, checkpoint events, local metrics writes).
- Reserve `logger.warning` for recoverable issues (e.g., failing to write progress files).
- Reserve `logger.error` for fatal configuration/runtime problems.

## 9. Roadmap / To-Do

- Add unit tests for config validation, wrapper scheduling, checkpoint manager, and manifest generation.
- Extend schema/templates for additional algorithms (single-agent RL, hierarchical RL, rule-based policies).
- Enhance CityLearn wrapper if/when environment state persistence becomes available for true resume.
- Provide ready-to-run notebooks/examples for students.
- Integrate automated CI (e.g., GitHub Actions) to execute the test suite and linting on every contribution.

Once these tasks are completed, the platform can be considered “closed” and students should only modify configurations, algorithm classes, and registry entries.
