# Algorithms for Energy Flexibility Optimization

This repository hosts the training side of the energAIze platform. It runs
multi‑agent reinforcement learning experiments on
[CityLearn](https://github.com/intelligent-environments-lab/CityLearn), exports
ONNX policies plus metadata, and feeds the
[`energAIze_inference`](https://github.com/your-org/energAIze_inference) serving
layer. Students extend the *algorithms* while the platform handles
orchestration, logging, and packaging.

## Documentation Map

- 📘 [Training Platform Guide](docs/platform_guide.md): architecture, config
  schema, extension workflow, troubleshooting.
- 📄 [Manifest example](docs/examples/manifest_example.json): sample
  `artifact_manifest.json` emitted after training.
- 🔌 [Inference bundle contract](docs/inference_bundle.md): files the serving API
  expects and how to load them.
- 🧭 [Simulator limits](docs/simulator_limits.md): what is and is not supported in
  this phase (resume, stepping contracts, next backlog).

## Prerequisites

- Python 3.10
- `pip install -r requirements.txt`
- Simulator dependency is installed via `softcpsrecsimulator` (imports remain `citylearn.*`)
- If your environment already had `CityLearn`, run `pip uninstall -y CityLearn` first (or recreate the venv)
- (Optional) `mlflow` if you want the UI locally
- Docker (optional, for containerised runs)

## Quick Start

### Local training

```bash
python run_experiment.py --config configs/config.yaml --job_id dev-run
```

Outputs are written to `./runs/`:

- `runs/jobs/<job_id>/` – logs, checkpoints, ONNX exports, manifest.
- `runs/mlflow/mlruns/` – MLflow tracking store (serve the UI with
  `mlflow ui --backend-store-uri file:./runs/mlflow/mlruns`).

### Inside Docker

The Docker image sets `OPEVA_BASE_DIR=/data`. Mount a volume at `/data` and run
with the same arguments:

```bash
python run_experiment.py --config /data/configs/<experiment>.yaml --job_id <job_id>
```

Artefacts appear under `/data/jobs/<job_id>/`, ready to publish or archive for
inference.

### Deucalion/SIF parity

The same training entrypoint is compatible with the Deucalion worker runtime:

- default worker command: `--config /data/<config_path> --job_id <job_id>`
- keep simulator datasets under `/data/datasets/...` inside the container/SIF
- per-job outputs remain under `/data/jobs/<job_id>/...` in both Docker and SIF

## Repository Layout

- `run_experiment.py` – entrypoint used both locally and in Docker.
- `algorithms/agents/` – algorithm implementations; every class inherits
  `BaseAgent`.
- `algorithms/constants.py` – shared constants (e.g., ONNX opset pin).
- `utils/wrapper_citylearn.py` – wrapper around CityLearn (encoders, logging,
  checkpoints, manifest metadata).
- `configs/` – experiment configs and encoder rules (`configs/encoders/*.json`).
- `reward_function/` – custom reward implementations.
- `tests/` – unit tests for encoders, config validation, etc.
- `docs/` – platform guide, inference bundle spec, manifest example.
- `runs/` – generated at runtime to mirror the `/data` mount.

## Training Flow at a Glance

```
┌────────────────┐       ┌────────────────────────────┐
│ YAML Config(s) │──────▶│ run_experiment.py (entry)  │
└────────────────┘       │ - validates config         │
                         │ - boots MLflow + logging   │
                         └─────────┬──────────────────┘
                                   │
                                   ▼
                        ┌────────────────────────────┐
                        │ Wrapper_CityLearn          │
                        │ - builds encoders from     │
                        │   configs/encoders/*.json  │
                        │ - drives CityLearn loop    │
                        └──────────┬─────────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │                           │
                     ▼                           ▼
        ┌────────────────────────────┐   ┌────────────────────┐
        │ CityLearn Simulator        │   │ BaseAgent (MADDPG) │
        │ datasets/<schema>.json     │   │ algorithms/agents  │
        └────────────┬───────────────┘   └────────┬───────────┘
                     │                            │
                     ▼                            ▼
        ┌────────────────────────────┐   ┌────────────────────┐
        │ Training Artefacts         │   │ MLflow Tracking    │
        │ runs/jobs/<job_id>/        │   │ runs/mlflow/mlruns │
        │ ├─ checkpoints/            │   └────────────────────┘
        │ ├─ onnx_models/            │
        │ └─ artifact_manifest.json  │
        └────────────┬───────────────┘
                     │ exported bundle
                     ▼
        ┌─────────────────────────────────────────────┐
        │ energAIze_inference service (separate repo) │
        │ consumes manifest + ONNX per agent          │
        └─────────────────────────────────────────────┘
```

Use this diagram as the mental model when wiring CI or new algorithms.

## Configuration Essentials

- Base config lives in `configs/config.yaml` (students typically copy this).
- `configs/` keeps a single canonical config file outside templates: `configs/config.yaml`.
- Validation happens in `utils/config_schema.py` (Pydantic models).
- Observation encoders are defined in `configs/encoders/default.json` so training
  and inference stay aligned.
- Templates under `configs/templates/*.yaml` provide algorithm x environment
  starting points:
  `maddpg_local.yaml`,
  `maddpg_deucalion_cpu.yaml`,
  `maddpg_deucalion_gpu.yaml`,
  `rule_based_local.yaml`,
  `rule_based_deucalion_cpu.yaml`,
  `rule_based_deucalion_gpu.yaml`,
  `single_agent_local.yaml`,
  `single_agent_deucalion_cpu.yaml`,
  `single_agent_deucalion_gpu.yaml`.

Important knobs:

- `tracking.mlflow_enabled` – switch between MLflow and JSONL logging.
- `tracking.log_frequency` – legacy step logging control (kept for compatibility).
- `tracking.mlflow_step_sample_interval` – sampled MLflow step metric interval
  (default 10, low-overhead mode).
- `tracking.mlflow_artifacts_profile` – `minimal|curated` artifact policy for
  MLflow uploads (default `minimal`).
- `checkpointing.resume_training` – reload agent state from MLflow (simulator
  still restarts).
- `simulator.simulation_start_time_step`, `simulator.simulation_end_time_step`,
  `simulator.episode_time_steps` – optional overrides for simulation window and
  episode length.
- `simulator.export.*` – CityLearn export controls:
  `mode` (`none|during|end`), KPI CSV toggle, optional fixed session name.
- `bundle.*` – manifest/export metadata (`bundle_version`, `description`,
  `alias_mapping_path`) and default `artifact.config` knobs (for example
  `require_observations_envelope`).
- `training.*` – exploration warm-up, update cadence, target refresh.
- `algorithm.*` – hyperparameters, network sizes, replay buffer type, exploration
  policy. `RuleBasedPolicy` replaces neural network knobs with PV thresholds,
  flexibility buffers, and charger overrides.

## Training Outputs & Inference Hand-off

Every job produces:

- `jobs/<job_id>/logs/<run_id>.log` – Loguru trace.
- `jobs/<job_id>/progress/progress.json` – progress updates for dashboards.
- `jobs/<job_id>/results/result.json` – CityLearn KPI pivot table.
- `jobs/<job_id>/results/simulation_data/<session|timestamp>/` – optional
  simulator CSV exports (`exported_*.csv`, `exported_kpis.csv`) when
  `simulator.export.mode != none` or KPI export is enabled.
- `jobs/<job_id>/checkpoints/` – optional checkpoints.
- `jobs/<job_id>/onnx_models/` – one ONNX actor per agent.
- `jobs/<job_id>/artifact_manifest.json` – metadata consumed by the inference
  repo.
- `jobs/<job_id>/config.resolved.yaml` – runtime-resolved config snapshot
  (auto-populated `runtime.*` and derived topology).
- `jobs/<job_id>/results/summary.json` – lightweight final run summary with
  MLflow identity/URI linkage.
- `jobs/<job_id>/policy_agent_<index>.json` – exported rule-based policy files
  when `RuleBasedPolicy` runs.

Bundle the manifest, ONNX directory, and optional alias map as described in
[`docs/inference_bundle.md`](docs/inference_bundle.md) to deploy a trained agent.

## Extending the Platform (Students)

1. Implement an agent under `algorithms/agents/` inheriting `BaseAgent`.
2. Register it in `algorithms/registry.py`.
3. Honour the full `BaseAgent.update` signature (scheduler flags are
   keyword-only and mandatory).
4. Extend `configs/templates/` and `utils/config_schema.py` if new
   hyperparameters are needed.
5. Implement `export_artifacts` so ONNX + metadata are written under the run
   directory. Use the shared `DEFAULT_ONNX_OPSET` (13).
6. Add unit tests (`pytest`) for new utilities or encoders.

The [Training Platform Guide](docs/platform_guide.md) walks through this process
in detail and contains troubleshooting tips.

## Monitoring & Logging

- MLflow tracking URI resolution priority is:
  `MLFLOW_TRACKING_URI` env -> `runtime.mlflow_uri`/`runtime.tracking_uri` ->
  local `file:.../mlflow/mlruns`.
- MLflow records step metrics using sampled intervals
  (`tracking.mlflow_step_sample_interval`) plus always-on episode/final KPI logs.
- If MLflow is disabled (`tracking.mlflow_enabled: false`), metrics go to
  `<log_dir>/metrics.jsonl` via `LocalMetricsLogger`.
- Wrapper_CityLearn samples CPU/RAM/GPU utilisation every 10 steps.
- Use Loguru levels consistently (`info` for progress, `debug` for verbose state,
  `warning` for recoverable issues).

## Contributing

1. Create a virtual environment and install dependencies (`pip install -r requirements.txt`).
2. Run a smoke test: `python run_experiment.py --config configs/config.yaml --job_id smoke-test`.
3. Execute the test suite: `pytest`.
4. Submit focused PRs with documentation updates when behaviour changes.

## FAQ

- **Can I resume mid-episode?** No. Checkpoint resume reloads the agent state but
  CityLearn restarts at the beginning of the episode. See
  [`docs/simulator_limits.md`](docs/simulator_limits.md) for details and backlog.
- **How do I disable MLflow logging?** Set `tracking.mlflow_enabled: false`.
  Metrics stream to `<log_dir>/metrics.jsonl`.
- **Where do I inspect preprocessing metadata?** Open
  `jobs/<job_id>/artifact_manifest.json`; it lists encoders, action bounds,
  reward configuration, and ONNX artefacts.

## Roadmap

- Capture CityLearn environment state for true mid-episode resume.
- Extend schema/templates for additional algorithm families.
- Provide inference-side manifest validation scripts.
- Publish tutorial notebooks and CI integration examples.

## License

MIT
