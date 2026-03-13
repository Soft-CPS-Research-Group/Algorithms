# Training Platform Guide

This document explains how the fixed part of the project is structured, what responsibilities it owns, and how students can extend it with new algorithms. Treat this as the reference manual for anyone onboarding to the codebase. For a high-level overview and quick commands, start with the repository [README](../README.md).

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
- **Wrapper_CityLearn** hides CityLearn-specific plumbing. It encodes observations, logs metrics, writes progress files, schedules checkpoints, and exposes environment metadata for inference. Encoders are built from `configs/encoders/default.json`.
- **BaseAgent + registry** define the contract students must fulfil. Algorithms handle model logic, replay buffers, ONNX export (using `DEFAULT_ONNX_OPSET`), and provide metadata consumed by the manifest. The `update` method must accept the scheduler-aware signature (`observations`, `actions`, `rewards`, `next_observations`, `terminated`, `truncated`, `update_step`, `update_target_step`, `initial_exploration_done`, `global_learning_step`), and exploration ownership belongs to the algorithm (`BaseAgent.is_initial_exploration_done`).

## 2. Configuration & Schema

All configurations are validated against `utils/config_schema.py` before training begins.

Sections:
- `metadata`: experiment and run identifiers (used in MLflow).
- `runtime`: paths injected at runtime (leave null in files).
- `tracking`: log level, MLflow toggle.
- `checkpointing`: resume/transfer-learning parameters and cadence.
- `simulator`: dataset/schema path, reward function, optional window controls
  (`simulation_start_time_step`, `simulation_end_time_step`,
  `episode_time_steps`), and export controls under `simulator.export`.
- `training`: global update cadence knobs.
- `topology`: **derived** environment dimensions (num agents, observation/action shapes). These remain null in version-controlled configs and are filled by the wrapper.
- `algorithm`: algorithm name and its parameters. The schema currently supports MADDPG and a rule-based placeholder. New algorithms must extend the schema.
- `bundle`: manifest/export options shared by all algorithms (`bundle_version`, `description`, alias map hint, artifact config defaults).

### Validation

`validate_config(raw_config)` throws on missing keys/invalid values. Errors list the exact field:
```
Configuration validation failed:
- algorithm -> networks -> actor -> layers: layers must contain at least one hidden dimension
```
This prevents students from launching malformed runs.

## 3. Runtime Workflow

1. **Configuration** – Student clones `configs/config.yaml` or a template under `configs/templates/` and sets algorithm-specific values.
2. **run_experiment** – CLI parses args, validates config, sets runtime paths, and logs the start of the run. It also writes `jobs/<job_id>/config.resolved.yaml` with runtime-injected fields (for example `runtime.job_id`, `runtime.run_id`) plus derived topology. If enabled, simulator CSV exports are written under `jobs/<job_id>/results/simulation_data/`. If MLflow is disabled, a warning is issued and metrics fall back to local JSONL logs.
3. **Agent instantiation** – `create_agent(config)` looks up `algorithm.name` in `algorithms/registry.py`; the registry maps names to classes.
4. **Wrapper** – handles the CityLearn loop, invoking `BaseAgent.predict`/`update`. It logs metrics and checkpoints using helper classes. Observation encoders are built from `configs/encoders/default.json` so training and serving stay aligned; update the JSON when the simulator exposes new features.
5. **Artifacts** – After training, `agent.export_artifacts` writes ONNX models under `jobs/<job_id>/onnx_models/` (or `policy_agent_<index>.json` for `RuleBasedPolicy`), and `build_manifest` drops `jobs/<job_id>/artifact_manifest.json` capturing topology, encoders, reward config, and algorithm metadata for inference deployment. The training runner validates the bundle contract (`utils/bundle_validator.py`) before finalizing output. See also [`docs/inference_bundle.md`](inference_bundle.md) for the exact bundle served to the API.

## 4. Checkpointing

Checkpoints are managed by `utils/checkpoint_manager.CheckpointManager`:
- Saves every `checkpoint_interval` steps (post exploration phase) to `<log_dir>/checkpoints/`.
- Optionally logs to MLflow when tracking is enabled.
- Agents must implement `save_checkpoint(output_dir, step)` returning the checkpoint path. The MADDPG example stores PyTorch state dicts and replay buffer state.

**Resuming**: Setting `checkpointing.resume_training: true` reloads model weights/optimizers, but the simulator restarts from the beginning of the episode. The rationale: CityLearn does not expose environment snapshots, so only agent state is restored. See [`docs/simulator_limits.md`](simulator_limits.md) for the formal Phase 1 limits/backlog.

## 5. Metrics and Logging

- **MLflow Enabled**: Per-step metrics honour sampled cadence (`tracking.mlflow_step_sample_interval`) and episode summaries are logged.
- **MLflow Disabled**: Metrics are appended to `<log_dir>/metrics.jsonl` via `LocalMetricsLogger`. The file uses JSONL (one record per line) for easy ingestion.
- `tracking.log_level` controls Loguru output; informative messages (`info`, `debug`) are sprinkled through the runner and wrapper to aid debugging.
- `tracking.progress_updates_enabled` and `tracking.progress_update_interval` control progress file writes.
- `tracking.system_metrics_enabled` and `tracking.system_metrics_interval` control CPU/RAM/GPU telemetry (debug-oriented, disabled by default).

## 6. Artifact Manifest

`artifact_manifest.json` (written beside `onnx_models/`) contains:
- Config snapshot (`metadata`, `training`, `topology`, algorithm hyperparameters).
- Environment metadata from `Wrapper_CityLearn.describe_environment()`:
  - `observation_names` and per-observation encoder definitions (type + parameters like `x_min`, `x_max`, `classes`, etc.).
  - `action_bounds`, `action_names`, and optional `action_names_by_agent` for heterogeneous multi-agent setups.
  - Reward function class name and non-private attributes (parameters used during training).
- Agent metadata returned by `BaseAgent.export_artifacts` (e.g., ONNX paths, observation/action dimensions).

This manifest enables a separate inference service to apply the same preprocessing and routing logic that was used during training.

### Manifest Fields in Detail

- `manifest_version`: increment when the structure changes to keep inference code compatible.
- `metadata`: mirrors the config’s experiment name/run name so manifests can be catalogued.
- `topology`: injected by the wrapper; contains agent counts and per-agent observation/action dimensionality.
- `environment`: detailed encoder definitions (type + params), action names/bounds, reward configuration.
- `agent`: metadata returned by the agent with one artifact entry per agent (`agent_index`, `path`, `format`, `config`), plus optional dimensions for ONNX exports.
- `training`: seed and schedule parameters used during training, useful for reproducibility.

## 7. Extending the Platform with New Algorithms

1. **Implement the Agent**
   - Subclass `BaseAgent` and implement `predict`, `update`, `save_checkpoint`, `export_artifacts` (and optionally `load_checkpoint`). The `update` method must accept the full scheduler-aware signature (`observations`, `actions`, `rewards`, `next_observations`, `terminated`, `truncated`, plus keyword-only flags for `update_step`, `update_target_step`, `initial_exploration_done`, `global_learning_step`).
   - Reuse helper utilities (replay buffers, noise, etc.) as needed.

2. **Register the Agent**
   - Add an entry to `algorithms/registry.py` mapping the algorithm name to the new class.

3. **Extend the Schema**
   - Create a new Pydantic model mirroring the algorithm’s configuration needs.
   - Update `ProjectConfig.algorithm` to accept the new model.
   - Add a template config in `configs/templates/` highlighting the new fields.

4. **Export Metadata**
 - Ensure `export_artifacts` writes whatever the inference service requires. For ONNX-based policies, call `mlflow.log_artifact` if MLflow is active and return a metadata dict describing the exports. Use the shared `DEFAULT_ONNX_OPSET` constant so exported graphs stay compatible with the serving stack.
  - For `RuleBasedPolicy`, export `policy_agent_<index>.json` files alongside the manifest so inference can load each agent directly.

5. **Testing & Documentation**
   - Provide small smoke tests and update documentation to explain how to use the new algorithm.

### Worked Example: Adding a New Agent

1. Duplicate `configs/templates/maddpg_local.yaml` and adjust fields for the new algorithm.
2. Implement the agent class under `algorithms/agents/` and ensure it inherits `BaseAgent`.
3. Register the agent name/class mapping in `algorithms/registry.py`.
4. Extend `utils/config_schema.py` with a new Pydantic model if the algorithm introduces unique config fields.
5. Implement `export_artifacts` to export ONNX graphs (or other artefacts) plus metadata.
6. Add unit tests covering key behaviours (config validation edge cases, manifest metadata).
7. Update `docs/platform_guide.md` with any algorithm-specific quirks.

### Troubleshooting

- **Missing artefacts in manifest**: Ensure `export_artifacts` returns a metadata dict and that the files reside under the job directory (manifest root).
- **MLflow disabled but metrics missing**: Confirm `tracking.mlflow_enabled: false` and inspect `<log_dir>/metrics.jsonl`; metrics are appended in JSONL format.
- **Checkpoint resume not restoring simulator state**: Currently only the agent state is persisted; see the roadmap for plans to snapshot the environment.

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
