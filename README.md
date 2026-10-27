# Algorithms for Energy Flexibility Optimization

Algorithms is the training side of energAIze: configurations and datasets enter a
common job runtime; agents train or evaluate through shared simulator interfaces;
results, KPIs and requested model artifacts leave the job directory.

Start with the [runtime guide](docs/runtime/README.md),
[38-agent catalogue](docs/agent_catalogue.md), and [documentation index](docs/README.md).
The [branch consolidation record](docs/consolidation/README.md) preserves historical
origins. The [stage 2 delivery](docs/runtime/delivery.md) records the common-core
reorganization and its verification.

## Quick start

Python 3.10 and `softcpsrecsimulator==1.8.3` are the tested baseline. Imports from
the simulator remain `citylearn.*`. Use a fresh environment to avoid conflicting
CityLearn distributions.

For configuration validation only:

```bash
python -m venv .venv
. .venv/bin/activate
pip install .
algorithms-job --list-agents
algorithms-job --config configs/config.yaml --validate
```

For CPU jobs without MLflow:

```bash
pip install -r requirements/lock-cpu-py310-linux-x86_64.txt
pip install . --no-deps
algorithms-job --config configs/templates/experimental/transformer_maddpg_smoke.yaml --dry-run
algorithms-job --config configs/templates/experimental/transformer_maddpg_smoke.yaml --profile smoke --job_id smoke-001
```

See [installation profiles](docs/runtime/installation.md) for GPU/platform-specific
wheels, optional tracking/export/offline/solver dependencies and developer tools.
`python run_experiment.py` accepts the same CLI arguments as `algorithms-job`.
Datasets are separate job inputs; local smoke recipes use the bundled repository
dataset, which is not included in the Python wheel.

## Configuration and outputs

Use `schema_version: algorithms.config.v1` and an ordered `pipeline` of agents.
Common initialization, simulator settings, training cadence, tracking, checkpoints
and outputs are configured once. Each stage owns its numerical parameters.
See [configuration, profiles and migration](docs/runtime/configuration.md).

- `train` preserves the configured training schedule.
- `evaluate` runs one deterministic episode from an explicit trained source.
- `smoke` runs a bounded integration check with a deterministic final episode.
- `outputs.model_artifacts` selects `none`, `checkpoint`, `bundle` or `both`;
  omitted/null retains legacy configured behavior.
- `simulator.export` controls KPI/time-series collection. Global export-policy
  corrections remain a separate planned stage.

The output root is `--base-dir`, `OPEVA_BASE_DIR`, or `./runs`, in that order:

```text
runs/jobs/<job_id>/
  config.resolved.yaml
  job_info.json
  logs/<job_id>.log
  progress/progress.json
  results/result.json
  results/summary.json
  results/simulation_data/       # KPIs/time series when requested
  checkpoints/                  # supported/requested training state
  bundle/artifact_manifest.json # when a bundle is requested
  bundle/...                    # agent-specific artifacts and metadata
```

Results include provenance and the output contract version. Supported neural
agents export ONNX; baselines export policy descriptions; experimental agents may
export research Torch artifacts. The common manifest does not by itself certify
an artifact for the inference service. See the [bundle contract](docs/inference_bundle.md)
and each agent's capability declaration.

## Repository organization

| Area | Owner |
| --- | --- |
| `algorithms/configuration/` | Versioned common and agent schemas, migration, profiles |
| `algorithms/catalogue.py`, `registry.py` | Capabilities, lazy registration, composition builder |
| `algorithms/runtime/` | Job, environment, common loop, controller, resources, diagnostics, tracking, checkpoints, outputs |
| `algorithms/agents/`, Transformer, TI-MARL and offline packages | Policies, networks, losses, optimizers and buffers |
| `algorithms/contracts.py`, `execution_unit.py`, `pipeline.py` | Typed interfaces and agent/Pipeline/Ensemble composition |
| `configs/` | Reusable templates, encoder rules and tokenizer resources |
| `reward_function/` | Reward implementations |
| `scripts/` | Research tools, data collection, campaigns and diagnostics |
| `utils/` | Shared utilities and compatibility imports |
| `tests/`, `docs/` | Verification, contracts, decisions and guides |

New agents implement/register their schema, capabilities and numerical logic.
They reuse the job lifecycle. Follow [AGENTS.md](AGENTS.md) and the
[extension guide](docs/runtime/agents.md). Entity agents also follow the
[shared Transformer/entity contract](docs/transformer_entity_controller.md),
[TPPO specification](docs/transformer_ppo_spec.md) or
[Transformer MATD3 guide](docs/transformer_matd3.md).

## Containers and operation

The existing image entry point remains `python run_experiment.py`. Mount the
job volume at `/data` and pass `--config /data/config.yaml --job_id <id>`;
`OPEVA_BASE_DIR=/data` keeps outputs under `/data/jobs/<id>/`.
The [CI workflow](.github/workflows/ci.yml) defines Docker, Jetson, Union and SIF
builds, current image tags and publishing conditions. A successful build does not
replace hardware-specific execution certification. Deucalion remains suspended
operationally; its retained build definition is not authorization to run jobs there.

MLflow is optional in local code. `tracking.mlflow_enabled: false` retains local
JSONL metrics and result files without importing MLflow. Agent checkpoints restore
agent state; they do not promise a mid-episode simulator snapshot. Job outputs,
logs and models have no automatic expiry introduced by this reorganization.

## Verification and contribution

```bash
ruff check
ruff format --check
mypy
pytest
pytest -o addopts='' tests/e2e/test_e2e_consolidated_agents.py tests/e2e/test_common_job_profiles.py
```

See [installation](docs/runtime/installation.md) for test dependencies and
[ownership/provenance](docs/runtime/ownership.md) for contribution boundaries and
the repository's unresolved licensing decision. Submit changes through a PR and
preserve source provenance when consolidating research implementations.

See the [stage-3 reliability contract](docs/runtime/reliability.md) for explicit training/evaluation budgets, data validation, seeds, checkpoint/resume limits, platform support and the measured optional-tracking decision.
