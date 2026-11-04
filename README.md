# Algorithms

Algorithms trains and evaluates energy controllers for energAIze. A job takes a simulator dataset and a YAML configuration, runs the selected controller, and writes evaluation KPIs and any requested model bundle to `runs/jobs/<job_id>/`.

## Run a first job

From the repository root, using Python 3.10 on Linux x86-64:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements/lock-cpu-py310-linux-x86_64.txt
pip install . --no-deps

algorithms-job --list-agents
algorithms-job --config configs/templates/experimental/transformer_maddpg_smoke.yaml --validate
algorithms-job --config configs/templates/experimental/transformer_maddpg_smoke.yaml --dry-run
algorithms-job --config configs/templates/experimental/transformer_maddpg_smoke.yaml --profile smoke --job_id first-job
```

Validation checks the configuration. The dry run also constructs and resets the environment. The smoke profile exercises the job on a bundled dataset. Large datasets are supplied separately. For other platforms or configuration-only installs, see the [installation guide](docs/runtime/installation.md).

## Work on a controller

- **Choose a controller.** Find its registered name and supported interfaces in the [agent catalogue](docs/agent_catalogue.md). The [Core30 guide](docs/baseline_comparison_core30.md) defines the two BAUs, three rule-based controllers and weekly cost oracle used to compare MARL agents.
- **Configure a job.** Start from a [reusable template](configs/templates/README.md). Set the dataset, window, training and evaluation episodes, and requested outputs. [Configuration reference](docs/runtime/configuration.md).
- **Implement an agent.** The agent owns decisions, learning and numerical serialization. The runtime owns the simulator loop, checkpoints and finalization. [Agent development](docs/runtime/agents.md).
- **Evaluate and export.** Use the final deterministic episode for comparisons. The job writes the resolved config, `results/result.json`, requested KPIs and, when enabled, `bundle/artifact_manifest.json`. [Output contract](docs/runtime/exports.md).
- **Run remotely.** The container accepts `--config /data/config.yaml --job_id <id>`. With `OPEVA_BASE_DIR=/data`, results go under `/data/jobs/<id>/`. The experiment CLI supports payload preparation, worker checks, submission and collection. [Platform guide](docs/platform_guide.md).

The [documentation index](docs/README.md) groups the implementation guides and the [lessons from long runs](docs/learning-experience.md). Submit changes through a pull request. Run the relevant tests and static checks defined in [CI](.github/workflows/ci.yml).
