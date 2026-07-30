# Local single-agent RL validation

- Campaign: `local_single_agent_rl_validation_20260729`
- Archived: `2026-07-29T19:54:04+01:00`
- Source: `ff5bba3bbf14`, dirty worktree; no immutable image
- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Simulator: `softcpsrecsimulator==1.5.6`
- Window: `0:1023` (1024 x 15 min; 10.67 days)
- Seed: `123`
- Algorithms: strict distributed PPO, strict distributed TD3, RBCSmart
- Gate profile: `phase10_w6_adapted_local_v1`
- Evidence horizon: partial-window diagnostic, not full-year

## Status

Eleven local jobs were attempted: ten finished and one initial RBCSmart job
failed during reward construction because simulator 1.5.6 first supplied
`env_metadata=None`. The lifecycle bug was fixed and the matching RBCSmart rerun
completed. All PPO/TD3 training and deterministic-evaluation jobs completed.

## Headline scorecard

| Run | Cost EUR | EV minimum | EV tolerance | Grid kWh/events | Deferrable missed | Decision |
|---|---:|---:|---:|---:|---:|---|
| RBCSmart matching | 600.66 | 1.000 | 0.987 | 0 / 0 | 0 | `PASS_LEARNING_GATES` |
| PPO demo-replay long | 608.94 | 0.961 | 0.961 | 2.748 / 48 | 0 | reject EV + electrical |
| TD3 balanced long | 598.72 | 0.961 | 0.895 | 0.582 / 10 | 0 | reject EV + electrical |

The strongest short TD3 recipe cost EUR 587.98 with no grid/deferrable/SOC
failure, but EV minimum was only 0.921. Lower cost never overrode a failed gate.
All nine PPO/TD3 diagnostic variants were rejected; only RBCSmart passed.

## Technical findings

- PPO teacher-controlled transitions were incorrectly eligible for the
  on-policy objective; they are now masked from policy loss/entropy/KL.
- Extra BC previously ran before PPO ratios, making `old_log_probs` stale. The
  first on-policy KL fell from 2.99 to 0.0169 after correction.
- Rare EV/deferrable targets required positive/idle weighting. PPO also needed a
  persistent teacher-demonstration replay isolated from its on-policy rollout.
- More training improved both algorithms to EV minimum 0.961, but electrical
  violations appeared. The current recipes are not promotion-ready.

## Evidence and limitations

- Detailed review: `docs/single_agent_rl_audit_20260729_pt.md`
- Raw scorecard: `runs/analysis/single_agent_rl_validation_20260729/scorecard.csv`
- Raw JSON: `runs/analysis/single_agent_rl_validation_20260729/scorecard.json`
- Per-job configs, logs, checkpoints, ONNX and exports:
  `runs/jobs/*single-agent*` and the explicit job IDs in the detailed review.
- Final test suite: `605 passed`, `28 warnings`.

The same window and seed were used adaptively across diagnostics, so this is not
independent statistical validation. No remote, multi-seed, held-out seasonal or
full-year campaign was submitted. Promotion remains blocked until a new recipe
passes the local screen, then held-out windows and seeds 123/456/789.
