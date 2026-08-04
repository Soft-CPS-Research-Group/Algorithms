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
- every BC building has usable samples greater than zero;
- the total number of trained batches is greater than zero;
- no `Skipping behavior-cloning demonstrations` warning;
- no watchdog activation or traceback.

The runner fails closed when the log lacks per-building usable-sample evidence or
the positive trained-batch total. This branch currently records aggregate BC
demonstration samples and pretraining epochs, but does not emit the required
per-building usable-sample or trained-batch metrics. The real runs can complete,
but the validation gate will remain blocked until that instrumentation is added.
