# Common job runtime — stage 2

An Algorithms job receives one versioned configuration and its referenced datasets,
constructs the simulator and execution unit, runs training/evaluation, and writes
results plus the requested model artifacts. Agent implementations own numerical
learning; they do not create a second job lifecycle.

Start with [configuration and migration](configuration.md), [installation](installation.md),
and the [agent extension guide](agents.md). The [38-agent catalogue](../agent_catalogue.md)
records implementations and research limitations. The [stage 2 delivery](delivery.md)
records the acceptance scope and evidence.

```text
run_experiment.py / algorithms-job
  configuration: migration → overrides → profile → schemas → capability checks
  runtime.job: paths / tracking / simulator / model preparation / recovery
    simulator.create_environment + SimulatorAdapter
    SimulationController: observations / topology / scheduling / agent interaction
      loop.run_episodes: one episode and step lifecycle
      resources.ResourceLifecycle: progress, telemetry, watchdog, resource limits
      diagnostics.Diagnostics: shared diagnostics and configured reward shaping
    ExecutionUnit: agent | Pipeline | Ensemble
      predict/decide → simulator.step → update/learn_transition
    checkpoints + outputs: persistence, provenance, manifest and result.json
```

The controller remains a stateful facade because topology and observation layouts
are shared across reset, prediction and updates. Resources and diagnostics are
cohesive mixins of that facade. The extracted loop uses this same state; it does
not keep a second copy. `run_experiment.py`, `utils.config_schema` and
`utils.wrapper_citylearn` remain compatibility imports, delegating to the new owners.

## Entry points

```bash
algorithms-job --list-agents
algorithms-job --config configs/config.yaml --validate
algorithms-job --config configs/templates/experimental/transformer_maddpg_smoke.yaml --dry-run
algorithms-job --config configs/templates/experimental/transformer_maddpg_smoke.yaml --profile smoke --job_id smoke-001
```

`--validate` checks configuration and declared capability combinations without
loading PyTorch, CityLearn or MLflow. `--dry-run` additionally constructs/resets
an environment, checks cardinalities and agent-owned environment validators,
and closes it in a temporary directory. It performs **zero simulator steps and
zero training updates**. It checks local checkpoint/model/dataset paths but does
not download a remote checkpoint or promise its numerical compatibility.
TI-MARL validates its typed interface bindings against the actual simulator
catalogue before training. A test fixture from a different simulator schema is
not an interchangeable production interface.

## Outputs

Every successful job writes `config.resolved.yaml`, `job_info.json`, progress,
`results/result.json` and `results/summary.json` under `runs/jobs/<job_id>/`.
Results include `output_contract_version: algorithms.outputs.v1` and provenance.
KPI and time-series collection continue to follow `simulator.export`.

| `outputs.model_artifacts` | Final artifacts |
| --- | --- |
| `none` | Results/KPIs as configured; no model bundle or job checkpoint |
| `checkpoint` | Results plus a required final checkpoint |
| `bundle` | Results plus validated bundle and manifest |
| `both` | Results, required final checkpoint and bundle |
| omitted / `null` | Legacy configured checkpoint schedule plus bundle |

An explicit checkpoint request fails if no stage supports saving, or if the
completed job does not actually produce the checkpoint. A research artifact is
still a research artifact: the recovered Transformer/offline agents do not become
serving-ready because they share a manifest. ONNX signatures, final-only export
policy and inference interoperability retain their separate later workstreams.
Use a unique job ID for each independent execution; resume by providing a checkpoint.

MLflow is an optional extra and is loaded only when tracking is requested. Local
metrics and results remain available with `tracking.mlflow_enabled: false`.
This stage separates the dependency; it does not claim a new tracking performance
benchmark or remove MLflow from existing published deployment images.

## Research tools and ownership

`run_experiment.py` is the supported job entry point. The offline trainers now
share their parameter definitions with job validation. Their optimization loops
remain agent-owned. Dataset collectors, `run_entity_pipeline.py` and benchmark
scripts remain explicit research/campaign tools: they perform data preparation,
multi-seed experiments or comparisons, not an alternative supported production
job contract. They reuse the simulator input adapter instead of importing the
whole job runner. See [ownership and provenance](ownership.md) and the
[architecture decision](adr-001-common-runtime.md).
