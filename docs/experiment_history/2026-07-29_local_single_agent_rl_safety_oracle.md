# Local single-agent RL safety and oracle continuation

- Campaign: `local_single_agent_rl_safety_oracle_20260729`
- Archived: `2026-07-29T23:27:28+01:00`
- Source: `ff5bba3bbf14`, dirty worktree; no immutable image
- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Window: `0:1023` (1023 exported transitions at 15 minutes)
- Seed: `123`
- Algorithms: distributed local PPO, distributed local TD3, fixed-service MILP replay, RBCSmart
- Gate profile: `phase10_w6_adapted_local_v1`
- Evidence horizon: partial-window, adaptive and in-sample oracle-assisted

## Outcome

The local semantic safety layer made both individual learners pass EV,
electrical, deferrable, storage-SOC and outage gates. Without MILP storage
demonstrations, PPO and TD3 remained slightly more expensive than RBCSmart. With
the fixed-service MILP schedule as the storage BC teacher, both passed all gates
and beat RBCSmart on the matching deterministic evaluation.

| Run | Cost EUR | Delta vs RBCSmart | EV minimum/tolerance | Grid kWh/events | Deferrable missed | Decision |
|---|---:|---:|---:|---:|---:|---|
| RBCSmart matching | 600.66 | 0.00 | 1.000 / 0.987 | 0 / 0 | 0 | `PASS_LEARNING_GATES` |
| Fixed-service MILP replay | 521.45 | -79.21 | 1.000 / 0.987 | 0 / 0 | 0 | `PASS_LEARNING_GATES` |
| PPO + MILP storage BC | 562.65 | -38.00 | 1.000 / 0.987 | 0 / 0 | 0 | `PASS_LEARNING_GATES` |
| TD3 + MILP storage BC | 556.30 | -44.36 | 1.000 / 0.987 | 0 / 0 | 0 | `PASS_LEARNING_GATES` |

The simulator replay was EUR 0.86 above the conservative linear schedule. The
certified linear lower bound was EUR 470.21. This is a conditional battery-only
oracle: EV and deferrable service are fixed to RBCSmart and the full simulator
network/phase model is not jointly optimized. It is not a global-optimum
certificate.

## Evidence

- Canonical scorecard:
  `runs/analysis/single_agent_rl_oracle_iteration_20260729/scorecard.csv`
- Canonical JSON:
  `runs/analysis/single_agent_rl_oracle_iteration_20260729/scorecard.json`
- MILP model result and replay schedule:
  `runs/analysis/fixed_service_battery_oracle_20260729/`
- Detailed Portuguese audit: `docs/single_agent_rl_audit_20260729_pt.md`
- Job IDs: `fixed-service-oracle-replay-pilot-20260729`,
  `ppo-oracle-bc-smoke-s123-20260729`,
  `td3-oracle-bc-smoke-s123-20260729`, plus the safe hybrid and rejected
  diagnostic jobs documented in the detailed audit.
- Focused final verification: `86 passed`.
- Full final suite: `646 passed`, `28` known warnings.

## Promotion boundary

These results prove wiring, simulator feasibility and in-window imitation. They
do not establish generalization because the same window was used adaptively and
the storage demonstrations use perfect foresight for it. Promotion requires
training/evaluation separation, held-out seasonal windows and multiple seeds.
Claiming a global optimum additionally requires a joint MILP for stationary
storage, EVs, deferrables and exact network/phase constraints.
