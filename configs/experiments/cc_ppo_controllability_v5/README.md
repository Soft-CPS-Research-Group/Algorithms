# CC-PPO controllability V5

This protocol separates three questions that must not be answered by the same
run:

1. Does the existing frozen PPO react differently when the CC changes only the
   current price or changes current plus forecast prices?
2. Can a time-selective scalar signal outperform a constant multiplier while
   the PPO checkpoint remains untouched?
3. Is a new PPO, trained for a virtual-price interface, needed before another
   learned CC campaign?

## Annual ablations prepared for versioning

| Config | Actor input | Residual base input | Interpretation |
|---|---|---|---|
| `cc_ppo_fixed_0p95_actor_current_only_seed789.yaml` | current price multiplied by 0.95; real forecasts | multiplier 0.95 | explicit current/forecast wedge |
| `cc_ppo_fixed_0p95_actor_current_and_forecasts_seed789.yaml` | current and all forecasts multiplied by 0.95 | multiplier 0.95 | removes that wedge |

Both load the accepted seed-789 checkpoint, keep all 17 PPOs frozen and use
the same annual settlement contract as V4. They are deliberately tagged
`explicit_ood_diagnostic` and are not promotion candidates: the checkpoint was
trained only at the nominal price. Their purpose is causal diagnosis.

Generate the versioned configs again with:

```bash
.venv/bin/python scripts/generate_cc_ppo_controllability_v5.py
```

Use `--smoke --output-dir <ignored-dir>` for functional smokes.

## Temporal probes

`scripts/build_cc_ppo_schedule_probes.py` derives hourly, compressed scalar
schedules from matched annual traces. It creates four causal heuristics:

- discount only when the real current tariff is already cheap versus its
  forecasts;
- discount during community export;
- the union and intersection of those conditions.

When the annual 0.95 trace is provided, it also creates a retrospective block
selector: each block chooses 0.95 only where that separate run cost less than
the neutral run. This is not a mathematical oracle because the two source
traces have different battery states. It is an in-sample hypothesis generator;
only its continuous annual replay counts as evidence.

The schedules keep the actor observation unchanged and control only the
strict-local `SignalAwareRBCSmartLocal` residual base. This isolates temporal
control from actor out-of-distribution effects.

## Gate before price-responsive PPO training

A non-neutral price-conditioned PPO is currently inference-only by design.
Merely randomising the observed virtual price while retaining a reward based on
the unmodified tariff would train the PPO to ignore the coordinator. Before a
new local PPO campaign, the implementation must make the same effective-price
context available to:

- the current and next actor observations;
- the strict-local residual base;
- a strictly per-building economic reward or a price-conditioned local oracle
  teacher.

The leaf must still receive no community observations. The first candidate
will be trained on a held-out multiplier curriculum, frozen, and then tested
under neutral 1.0 before any CC is allowed to control it. Promotion still
requires lower settled annual cost, all hard service/safety gates, and the full
physical/fairness scorecard.
