# Algorithms for Energy Flexibility Optimization

This repository contains the research code that trains multi-agent reinforcement learning controllers on the [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) simulator. It is the code that runs inside the Docker container managed by the backend orchestrator and also the foundation that students will extend with new agents.

## Repository Layout

- `run_experiment.py` – unified entrypoint used both locally and inside Docker.
- `algorithms/agents/` – agent implementations. Every agent inherits from `BaseAgent`.
- `utils/wrapper_citylearn.py` – glue code between CityLearn and our agents (encoding, logging, checkpoints).
- `configs/` – experiment configuration files.
- `reward_function/` – custom reward definitions.
- `runs/` (created at runtime) – local output directory that mirrors the `/data` mount used in production.

## Running Experiments

### Local development
```bash
python run_experiment.py --config configs/config.yaml --job-id dev-run
```
- Results are stored under `./runs/jobs/dev-run/`.
- MLflow logs go to `./runs/mlflow/mlruns` (open them with `mlflow ui --backend-store-uri file:./runs/mlflow/mlruns`).
- A progress file is updated at `./runs/jobs/dev-run/progress/progress.json` so the orchestrator UI can poll it.

### Inside Docker / orchestrated runs
The container entrypoint invokes `run_experiment.py` directly. The image sets `OPEVA_BASE_DIR=/data`, so the shared volume mounted at `/data` remains the default destination for artefacts. Launch the container with the same arguments as before:
```bash
python run_experiment.py --config /data/configs/<experiment>.yaml --job-id <job-id>
```
All artefacts, results, and logs will appear under `/data/jobs/<job-id>/`.

**Resuming runs.** Setting `checkpointing.resume_training: true` reloads the agent weights and optimiser state from the specified MLflow artefact, but the CityLearn simulator itself starts a fresh episode. In other words, resume picks up the model where it left off while replaying the environment from the beginning. Document this expectation for students so they do not assume simulation state is persisted.

## Configuration

Experiment configuration lives in `configs/config.yaml` and is validated at load
time (`utils/config_schema.py`) so typos or missing fields fail fast. The file is
organised for clarity:

- `metadata`: experiment and run names used for MLflow tracking and logging.
- `runtime`: automatically populated paths (leave these set to `null` in the file).
- `tracking`: logging and MLflow toggles.
- `checkpointing`: resume/transfer-learning controls and checkpoint cadence.
- `simulator`: dataset location and reward function selection.
- `training`: common schedule parameters (seed, exploration boundaries, update cadence).
- `topology`: environment-driven dimensions injected by the wrapper (number of agents/houses, per-agent observation/action sizes, action bounds). These stay `null` in versioned configs but become the canonical source of truth for inference encoders.
- `algorithm`: algorithm-specific hyperparameters, network definitions, replay buffer, and exploration noise configuration. Students can swap this section for other algorithms by following the templates.

Students should copy the base config and edit only the documented fields. Runtime-populated entries (`runtime.*`, derived dimensions) will be filled automatically by `run_experiment.py`.

Validation automatically selects the appropriate schema based on
`algorithm.name` (currently `MADDPG` and a lightweight `RuleBasedPolicy`
example). Additional ready-to-edit examples live in `configs/templates/`
(e.g., `maddpg_example.yaml`, `rule_based_example.yaml`, `single_agent_example.yaml`) so students can start
new algorithms without modifying the canonical base file. Each run emits an
`artifact_manifest.json` alongside the exported ONNX graphs describing topology,
encoders, and reward configuration consumed by the inference service. A deeper
explanation of the platform and extension process lives in
`docs/platform_guide.md`.

## Artefacts and Logging

Every run (local or container) produces the following structure:
```
<base_dir>/
  mlflow/mlruns/           # MLflow experiment store
  jobs/<job_id>/
    job_info.json          # includes mlflow run information
    logs/<run_id>.log      # loguru log file
    progress/progress.json # periodically updated training progress
    results/result.json    # pivoted KPI table from CityLearn
    checkpoints/           # optional training checkpoints
    logs/onnx_models/      # ONNX exports, one per agent
```
Checkpoints are saved every `checkpoint_interval` steps once the exploration warm-up is over. The latest checkpoint file is also logged to MLflow (`checkpoint_artifact` in the config), enabling training to resume from the orchestrator UI.

## Implementing Custom Agents

All agents must inherit from `algorithms.agents.base_agent.BaseAgent` and implement:

- `predict(observations, deterministic)` – return an action vector per agent.
- `update(...)` – consume replay buffer samples and update model parameters.
- `save_checkpoint(output_dir, step)` – persist model/replay buffer state (called automatically by the wrapper).
- `export_artifacts(output_dir)` – export inference artefacts (e.g., ONNX graphs plus metadata).

`utils/wrapper_citylearn.Wrapper_CityLearn` handles encoding observations, scheduling updates, writing progress files, and triggering checkpoints/artefact exports. Encoders are configured in `configs/encoders/default.json`; update that file if the observation set changes so training and inference remain aligned. To add a new algorithm:

1. Create the agent implementation under `algorithms/agents/` and inherit from `BaseAgent`.
2. Register the agent in `algorithms/registry.py` so `run_experiment.py` can instantiate it via configuration (`algorithm.name`).
3. Implement `update` with the full BaseAgent signature (`observations`, `actions`, `rewards`, `next_observations`, `terminated`, `truncated`, plus the scheduling flags passed as keyword-only arguments). A helper mixin can wrap simpler update loops if needed.
4. Extend the config schema if new algorithm-specific parameters are required.
5. Implement `export_artifacts` to emit the artefacts your inference service needs (typically ONNX graphs plus encoder/decoder metadata).

## MLflow & Monitoring

- Every run is wrapped in an MLflow experiment. Metrics such as rewards, losses, and system stats are logged each step/episode.
- Launch the UI locally with `mlflow ui --backend-store-uri file:./runs/mlflow/mlruns`.
- Checkpoints are logged as MLflow artefacts, enabling the backend to resume from the best run or a specific step.
- When MLflow is disabled (`tracking.mlflow_enabled: false`), metrics are written to `<log_dir>/metrics.jsonl` so you still have a structured record of training progress.

## Development Tips for Students

- Keep new code within the existing abstractions (agent, replay buffer, preprocessing). Avoid modifying orchestrator-specific paths.
- Add unit tests for encoders, replay buffers, and any new utility code. The `tests/` folder is ready for expansion.
- Ensure your agent’s ONNX export includes enough metadata (observation order, scaling, action bounds) for the serving project to consume.
- Follow the repository’s logging conventions: use `loguru` for structured logs and rely on MLflow for metrics/artefacts.

## Contributing

1. Create a virtual environment and install dependencies with `pip install -r requirements.txt`.
2. Run `python run_experiment.py --config configs/config.yaml --job-id smoke-test` to verify the setup.
3. Prefer small, well-documented PRs so classmates can review changes easily.

## FAQ

- **Can I resume training mid-episode?** Not yet. `checkpointing.resume_training: true`
  reloads the agent state, but the simulator restarts at the beginning of the episode.
- **How do I disable MLflow logging?** Set `tracking.mlflow_enabled: false`. Metrics will
  then be written to `<log_dir>/metrics.jsonl`.
- **Where can I inspect preprocessing metadata?** Check
  `<log_dir>/artifact_manifest.json`; it contains encoder parameters, reward
  configuration, topology, and ONNX artefacts.

## To-Do

- Capture CityLearn environment state for true mid-episode resume.
- Flesh out schema/templates for additional algorithms (single-agent RL,
  hierarchical, rule-based variants).
- Provide inference-side validation scripts that consume `artifact_manifest.json`.
- Publish example notebooks for students.

## License

This project is licensed under the MIT License.
