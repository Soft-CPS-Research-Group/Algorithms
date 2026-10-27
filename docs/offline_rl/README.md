# Offline policies in the common job

BC, IQL and CQL are recovered from `69316e5` with their original trainers and
research utilities. The current normal entry point is `run_experiment.py`, shared
with online and hierarchical agents.

An offline stage chooses one source. To evaluate an existing policy:

```yaml
pipeline:
  - algorithm: IQLEntityAgent
    hyperparameters:
      model_dir: ./runs/offline/model
      device: cpu
```

To optimize from transition parquets before the same job evaluates and exports:

```yaml
pipeline:
  - algorithm: IQLEntityAgent
    hyperparameters:
      device: cpu
      offline_training:
        dataset_path: ./datasets/offline_rl/collected_entity_transitions
        parameters:
          hidden_layers: [256, 256]
          gradient_steps: 5000
          batch_size: 256
          eval_every_n_steps: 1000
```

Use that stage inside a complete normal config with `metadata`, `simulator` and
`training`. The simulator dataset must match the collection dataset's feature and
action semantics. The transition dataset is distinct from the simulator schema.
Entity groups are inferred from the job topology or explicitly supplied as
`groups: [{obs_dim: 4, action_dim: 2}]`. Training uses `training.seed` and writes
under the job's `offline_training/` directory; evaluation and final exports use
the standard wrapper, results directory and manifest. `CQLEntityAgent` selects the
CQL trainer. Flat `BCAgent`/`IQLAgent` take one parquet file and retain the original
Building_5 schema. BC uses `epochs` rather than `gradient_steps`.

`frozen: true` prevents online updates during simulation; explicit offline
preparation still trains before simulation. Supplying both `model_dir` and
`offline_training` is rejected. Training options are the corresponding
BCTrainingConfig/IQLTrainingConfig/CQLTrainingConfig dataclass fields.

Exports contain policy weights, architecture and normalizers. They can be reloaded
with the corresponding agent's existing loader and are marked non-deployable
`research_torch`. Missing sidecars, unknown observation names, incompatible action
layouts and ambiguous dimension groups fail rather than producing an apparently
valid job. Dataset generalization and scientific model selection are later work.

Historical collectors, benchmarks, analysis and standalone trainer CLIs remain in
`scripts/` for reproducing prior work. Their old full-year timing and KPI claims
are not revalidated by this consolidation. Original reports and plans remain
recoverable from the source commit's archive tag; the current integration evidence
is the common-job and synthetic training test suite.
