# TPPO Status And Next Steps

## Current Status

The TPPO behavior-cloning (BC) data contract and runtime safeguards are
implemented on PR #22. The final local validation passed with `1130 passed,
3 skipped`.

Wave A BC qualification runs completed successfully from
`runs/results2/`:

| Variant | Result |
| --- | --- |
| BC pretraining only | Valid data collection and pretraining; no auxiliary BC loss during PPO. |
| Auxiliary BC, weight 0.42 | Valid data collection and pretraining; strong EV-service improvement with material cost regressions. |

Both runs completed 3/3 episodes. All 17 active buildings had usable BC
samples and trained batches. No skipped demonstrations, watchdog failures, or
runtime errors were reported.

## Observed Trade-Off

Auxiliary BC at weight `0.42` improved EV departure behavior relative to
pretraining only:

- Departure success: `55.5%` to `89.7%`.
- Mean departure SOC deficit: `0.246` to `0.012`.

It also regressed district outcomes:

- Cost: `24,699 EUR` to `28,423 EUR`.
- Grid import: `143,711 kWh` to `164,806 kWh`.
- Emissions: `27,462 kgCO2` to `30,747 kgCO2`.
- Equity metrics worsened.

The current conclusion is that BC is effective for EV service but the `0.42`
auxiliary weight over-regularizes TPPO toward the service-heavy RBC teacher.

## Next Experiment

Run one controlled 3-episode auxiliary-BC variant first. Keep the current
auxiliary configuration unchanged except:

```yaml
behavior_cloning:
  weight: 0.20
  min_weight: 0.0
  decay_start_step: 35041
  decay_steps: 35039
```

Do not change the decay schedule in the same experiment. Episode 1 is the
demonstration-only phase; PPO auxiliary BC starts after pretraining. This run
isolates BC weight as the variable.

Compare it with the completed `weight: 0.0` pretraining-only and `weight:
0.42` auxiliary runs. Qualify it only if it retains the EV-service gain without
the material cost, import, emissions, and equity regressions.

## Deferred Work

Do not run a 10-episode experiment yet. The current auxiliary schedule reaches
zero BC weight at the end of episode 2, so later episodes are plain PPO. The
three-episode logs do not establish that more plain-PPO training improves the
trade-off.

Before any longer run:

- Export per-episode KPIs instead of final-episode-only KPIs.
- Confirm `TPPO/behavior_cloning_effective_weight`, loss, weighted loss, and
  valid-sample metrics are present in `logs/metrics.jsonl`.
- Define a BC schedule that intentionally covers the longer horizon.
