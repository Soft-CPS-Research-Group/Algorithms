# Common output lifecycle

The core owns output selection, final-evaluation association, validation and
completion. Agents own mathematical policy and numerical export. All 38 names
use the same service; recursive composition needs no algorithm branches in the
job runner.

## Normal profile

```yaml
training:
  episodes: 10
evaluation:
  episodes: 1
outputs:
  model_artifacts: bundle  # none, checkpoint, bundle, both
simulator:
  export:
    profile: final
    mode: none            # use end when final time series are wanted
    export_kpis_on_episode_end: true
    include_business_as_usual: true
    export_business_as_usual_timeseries: false
bundle:
  verification_samples: 32
  require_deployable: false
```

KPIs and requested time series are enabled only on the last deterministic
evaluation. Without evaluation there is no eligible scorecard. Model output is
independent of KPIs; progress metrics, periodic checkpoints, bundles and series
remain separate. BAU/reference calculation is preserved when its series are off.

Nullable final-episode flags inherit the common flag. The normal profile resolves
both to true, including old configurations with false. Bulk export now requires
`profile: diagnostic`: default maximum three training episodes and 128 MiB. The
byte budget is checked after each export and aborts before the next episode when
exceeded (one export can cross the threshold). Final evaluation is separate.
Diagnostic files use `diagnostic_kpis_ep<N>.csv`; only `exported_kpis.csv` enters
the canonical scorecard.

## Results and failure

`result.json` uses `algorithms.outputs.v2`; `summary.json` uses
`algorithms.summary.v2`. Legacy simulator-export flags remain for old report tools.
Canonical eligibility is in `result.kpis` and `results/kpis.json`
(`algorithms.kpis.v1`). Each metric identifies entity, building/community level,
value, unit, validity and missing reason. Units come from simulator metric suffixes;
undeclared units remain explicit. Blank/non-finite simulator values are not zero;
the exporter does not guess a physical reason absent from the simulator.

The scorecard links evaluated policy, checkpoint selection, seed/window and source
CSV hashes. Bundles copy the scorecard and input/configuration evidence. The result's
`manifest_sha256` identifies the sealed manifest. Training status, export status and
deployment eligibility are independent. Export failure leaves a failed job with
completed training recorded; completion cannot be committed. Reused job IDs retain
the isolated attempts introduced in stage 3.

`results/finalization.json` reports output/checkpoint time, per-episode KPI export
time and byte volumes. Simulator rendering can occur inside `step`; episode-export
timers do not capture all rendering and must not be added to wall time.

## Measured cost

[Raw measurements](export_benchmark.json): 27 local CPU jobs, two selected buildings,
NormalPolicy, two training episodes plus one deterministic evaluation,
24/168/720 hourly steps, three repetitions/profile, tracking/model export disabled.
This is an output-policy benchmark, not production-cluster training performance.

| Steps/episode | Final KPIs bytes / median wall time | Previous final-series profile | All episodes + BAU series |
|---|---:|---:|---:|
| 24 | 33,240 / 0.499 s | 54,418 / 0.507 s | 188,021 / 0.891 s |
| 168 | 33,625 / 1.694 s | 176,958 / 1.823 s | 697,695 / 2.412 s |
| 720 | 33,633 / 6.005 s | 651,432 / 5.998 s | 2,659,417 / 7.847 s |

At 720 steps, final KPIs reduce simulator bytes by 94.8% against final series and
98.7% against all-episode diagnostics. Wall time versus final series is within
noise; versus all-episode export it falls about 23.5%. Final-KPI volume stays
approximately constant with horizon here. The explicit 128 MiB diagnostic budget
is above this measured envelope. Reproduce:
`python -m scripts.benchmark_runtime_exports --output /tmp/export-costs`.

## Adding an agent

Implement `export_artifacts(output_dir, context)` with portable entries. Include
learned preprocessing, normalization and action scaling in the neural graph or a
declared adapter. A teacher, signal, safety projection or temporal state is part
of the policy when prediction uses it. The core captures real evaluation pairs,
checks the ONNX graph/executor, compares actions, seals files and writes the manifest.
Add a real job in `tests/e2e/test_export_contract.py` before declaring support.
See the [bundle contract](../inference_bundle.md) for formats and restrictions.
