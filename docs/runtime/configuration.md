# Configuration v1

The canonical discriminator is `schema_version: algorithms.config.v1`. All 46
tracked YAML configurations have been migrated. [The migration record](template_migration.json)
identifies every file and every removed value. Comments and explicit values have
been preserved where their semantics were already supported.

Common sections are `metadata`, `runtime`, `job`, `outputs`, `tracking`,
`checkpointing`, `simulator`, `training`, `topology`, `execution`, `bundle`, and
`experiment_protocol`. `pipeline` is an ordered, nonempty list of stages. Each
stage's schema is registered alongside its implementation and capabilities.

Unknown common and agent-schema keys are errors. Legacy actor/critic
`exploration.params`, teacher parameters and offline trainer parameters have
explicit validators: a misspelling no longer silently changes an experiment.
Reward kwargs, scenario overlays and artifact metadata remain named extension
maps owned by their respective reward/simulator/export services, not a general
escape hatch for unknown top-level keys.

## Resolution and nulls

1. Read YAML. Reject unsupported versions and ambiguous `algorithm` + `pipeline`.
2. Apply explicit `--set path=value` overrides in their command-line order;
   `--profile` overrides `job.profile`.
3. Migrate versionless syntax/known ignored legacy values, resolve the selected
   profile, validate all schemas and capability combinations.
4. Construct the environment and fill omitted topology dimensions.
5. Record the resolved configuration used by the job.

`--set pipeline.0.hyperparameters.gamma=0.98` addresses a list member.
Values use YAML syntax, including `null`, `false`, integers and lists. An override
is validated by exactly the same schema as the source document. `null` is accepted
only on optional fields. It is not a blanket request to disable a service.
An omitted profile means `train`. An omitted/null model artifact request retains
legacy behavior; select `none` explicitly to disable model artifacts.

Environment overrides retain the existing boundaries: `OPEVA_JOB_NAME` overrides
the recorded run name; the output root comes from `--base-dir`, then
`OPEVA_BASE_DIR`, then `./runs`; `MLFLOW_TRACKING_URI` overrides the configured
tracking URI. Relative dataset/model/checkpoint paths remain relative to the
working directory, preserving existing jobs. Use absolute paths in external
jobs. Missing relative `configs/...` tokenizer assets may resolve to packaged
resources, while an existing local file takes precedence.

## Profiles

- `train`: retain the configured training schedule and episode/export policy.
- `evaluate`: one deterministic episode, frozen stages, no offline optimization
  and no periodic/episode checkpoints. Every trainable stage needs an explicit
  trained source: `checkpointing.resume_training: true` with a checkpoint source,
  or an offline `model_dir`. Freezing a freshly initialized network is not a
  model evaluation.
- `smoke`: two episodes with a deterministic finish and at most five state
  points per episode; MLflow disabled; offline optimization capped at two
  epochs (BC) or two gradient steps (IQL/CQL). It checks integration, not quality.

Profiles override the relevant file settings intentionally. They do not override
an incompatible experiment protocol: use a separate smoke configuration when a
research protocol requires a longer or differently partitioned experiment.

## Migration

```bash
algorithms-job --config old.yaml --migrate-config migrated.yaml
algorithms-job --config migrated.yaml --validate
```

Top-level legacy `algorithm.name` is accepted only by the explicit migration
command. Versionless `pipeline` documents remain supported with migration
warnings. Some old RBC fields were discarded by the previous schema; migration
retains their previous *effective* defaults and reports their removal. To activate
such a parameter deliberately, set it in a v1 document. Known unused PPO/MADDPG
copy-paste parameters and unused CommunityCoordinator weights are also reported
and removed from legacy documents. Unknown keys outside this documented migration
set fail validation; they are never silently deleted.

The new schemas preserve numerical defaults used by existing agents. This is
not an audit of seed propagation, episode boundary algorithms, or benchmark
fairness; those remain the next contract-validation stage.
