# Training platform guide

The current reference is the [common runtime guide](runtime/README.md). It replaces
the former monolithic runner/wrapper layout and top-level `algorithm` examples.

A job follows one path: validate configuration, create output locations and
optional tracking, construct the simulator, resolve topology, prepare the agent,
attach/resume it, run the common episode loop, and finalize results/artifacts.
`ExecutionUnit` lets that path operate on a single agent, a vertical `Pipeline`
or a horizontal `Ensemble`.

- [Configuration and migration](runtime/configuration.md): schema v1, strict
  fields, overrides, train/evaluate/smoke and output requests.
- [Agent development](runtime/agents.md): lazy declaration, typed contracts,
  capabilities, numerical responsibilities and environment validators.
- [Installation](runtime/installation.md): configuration-only wheel, CPU lock,
  accelerator profiles and optional MLflow/export/offline/solver dependencies.
- [Architecture decision](runtime/adr-001-common-runtime.md): ownership and the
  retained compatibility interfaces.
- [Agent catalogue](agent_catalogue.md): 38 names and implementation limitations.
- [Delivery evidence](runtime/delivery.md): the accepted stage 2 scope.

## Troubleshooting

Use `--validate` for unknown fields and unsupported combinations. Use `--dry-run`
to check dataset/environment compatibility without simulator steps or training.
An error from a registered environment validator belongs to the agent/dataset
contract; do not suppress it and proceed with different dimensions or guessed
observations.

When a job fails, inspect `results/result.json` and `logs/<job_id>.log`.
Progress is in `progress/progress.json`. When tracking is disabled, step metrics
remain under the job's logs as JSONL. A requested bundle is rooted in `bundle/`;
its manifest describes actual agent artifacts. A missing required final
checkpoint is a job failure rather than a completed export.

Checkpoint recovery restores supported agent state. The simulator still restarts
its episode; see [simulator limits](simulator_limits.md). Runtime-only residual
policies and research Torch formats retain their deployment restrictions.

## Entity-specific contracts

The [entity playbook](entity_interface_playbook_pt.md) explains flat/entity inputs.
Use the [shared entity/Transformer contract](transformer_entity_controller.md),
[TPPO specification](transformer_ppo_spec.md), [Transformer MATD3 guide](transformer_matd3.md)
and [ADRs](adr/README.md) for those numerical packages. TI-MARL typed interfaces
must bind to the actual simulator catalogue, including any schema-dependent
feature names.
