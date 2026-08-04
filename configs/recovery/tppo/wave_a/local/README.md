# Local TPPO BC Gates

Run these checks before Wave A server runs:

```bash
scripts/run_tppo_bc_local_checks.sh
```

Set `LOG_DIR` to store local jobs and transcripts elsewhere. Its default is
`runs/local_bc_checks`. Set `PYTHON_BIN` when the project virtual environment
does not provide `python` on `PATH`.

The runner invokes the current experiment entrypoint, `python run_experiment.py`,
with unique canary and smoke job IDs. It runs the canary first, then the smoke.

## Configurations

- `tppo_bc_pretrain_canary.yaml` is a bounded 16-step real dynamic entity
  configuration. It is not synthetic; this corrects the original plan wording.
- `tppo_bc_pretrain_smoke.yaml` is a bounded 192-step real dataset configuration.

Both configurations run three episodes and gate Wave A server runs. Do not start
a Wave A server run if either local gate fails.

## Pass Criteria

For each configuration, the experiment log and local metrics must show:

- completed episode `3/3`;
- every active BC building has positive usable samples and trained batches, recorded as
  `TPPO/behavior_cloning_building_<building>_usable_samples` and
  `TPPO/behavior_cloning_building_<building>_trained_batches` metrics;
- the total `TPPO/behavior_cloning_pretraining_batches` metric is positive;
- no `Skipping behavior-cloning demonstrations` warning;
- the required `logs/<job_id>_stall_watchdog.log` artifact exists and is empty.

The runner reads these metrics from each job's `logs/metrics.jsonl` file. It
fails closed when any metric is missing or zero, or when the required watchdog
artifact is missing or contains output. The empty artifact confirms the wrapper
armed the watchdog without a timeout.
