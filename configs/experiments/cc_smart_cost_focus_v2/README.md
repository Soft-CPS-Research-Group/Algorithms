# CC-SMART cost-focus V2

Annual, settlement-enabled ablation over the exact frozen `SignalAwareRBC`
leaf used by `ppo_cc_settlement_annual_v1`.

The V1 run improved settled cost by EUR 26.72, but the historical reward was
dominated by its peak term.  Its embedded normalizers also did not match the
annual neutral SMART replay.  V2 preserves V1 as a control and tests a reward
whose principal signal is price/cost while keeping peak, ramping, export and
hard violations visible.

## Calibration source

References are measured from annual neutral SMART job
`b0747ffe-5a62-4e68-8218-765deffd4c78`, on the same dataset, full-year window,
frozen leaf and community-market contract:

| Quantity | Statistic | Reference |
|---|---|---:|
| community import | p75 | 5.9362177041 kWh |
| community settled-cost proxy | p90 | 1.5386517323 EUR/step |
| member-retail cost | p90 | 1.5496793514 EUR/step |
| squared import excess | p90 | 9.5067669878 kWh2 |
| import ramp | p90 | 2.5747369863 kWh |
| community export | p90 | 5.3642321795 kWh |

At the neutral baseline, the calibrated settled-focus reward is approximately
78.2% settled cost, 13.9% peak, 7.0% ramping and 0.9% export by cumulative
weighted magnitude.  The hybrid recipe is approximately 82.0% economic
(64.2% settled plus 17.8% member-retail), 11.4% peak, 5.8% ramping and 0.7%
export.  These figures describe scale, not expected KPI improvement.

## Recipes

All recipes use seed 123, price range 0.5--1.3, eight annual episodes and a
336-decision/two-week PPO rollout with gamma 0.995.  Episode 1 collects BC,
episodes 2--7 train PPO, and episode 8 is deterministic evaluation.

| Recipe | Reward | `w_factor` | `w_smoothness` | Purpose |
|---|---|---:|---:|---|
| `legacy_long_control` | historical V1 | 0.30 | 0.10 | isolate extra learning/horizon |
| `settled_focus_regularized` | calibrated cost-first | 0.30 | 0.10 | isolate reward calibration |
| `settled_focus_adaptive` | calibrated cost-first | 0.05 | 0.02 | test less resistance to useful price movement |
| `hybrid_physical_adaptive` | cost-first + 0.25 member-retail | 0.05 | 0.02 | discourage a physical counterfactual regression |

`w_factor` penalizes distance from the neutral multiplier 1.0.
`w_smoothness` penalizes changes between consecutive CC decisions.  The lower
pair is an ablation, not an assumption that regularization is harmful; the
annual scorecard decides whether the added freedom is useful.

Generate the committed annual configs:

```bash
.venv/bin/python scripts/generate_cc_smart_cost_focus_v2.py
```

Generate ignored three-episode BC/PPO/evaluation smokes:

```bash
.venv/bin/python scripts/generate_cc_smart_cost_focus_v2.py \
  --smoke \
  --output-dir runs/local_configs/cc_smart_cost_focus_v2_smokes
```

Do not promote on reward alone.  Compare settled cost first after the hard
service gates, then inspect the member-retail counterfactual, peaks, ramping,
import/export, throughput, emissions and building-level losers.
