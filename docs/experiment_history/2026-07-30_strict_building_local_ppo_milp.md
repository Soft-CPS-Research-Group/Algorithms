# Strict building-local PPO and individual MILP campaign

> **Provenance correction (2026-07-30):** the encoded PPO observations and
> local reward in this campaign were strict building-local, but the raw
> `RBCSmartPolicy` baseline/service teacher could use community PV/headroom
> fallbacks.  The results below are therefore retained as a legacy local-first
> diagnostic, not as final strict-local evidence.  The corrected campaign uses
> `RBCSmartLocalPolicy` and is recorded in
> `2026-07-30_overnight_full_dataset_campaign.md`.

- Campaign: `strict_building_local_ppo_milp_20260730`
- Archived: `2026-07-30T00:24:24+01:00`
- Source: `ff5bba3bbf14`, dirty worktree; local venv, no immutable image
- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Window: `0:1023` (1023 exported transitions at 15 minutes)
- Seed: `123`
- Market: disabled
- Observation profile: `building_local_v1`
- Reward: `LocalScorecardGuardRewardV2`
- Baseline: strict-local `RBCSmart`
- Gate profile: `building_local_phase10_w6_v1`
- Evidence horizon: in-sample, perfect-foresight-BC-assisted pilot

## Outcome

The prior “individual” campaign still exposed community observations and used
community settlement in the reward. This continuation establishes a genuinely
local retail track: 17 independent learners, no community features, no market
settlement and per-building gates/costs.

| Run | Local cost EUR | Delta vs RBCSmart | Gate pass | Buildings cheaper | Median/worst delta EUR | Gap to MILP replay closed | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| RBCSmart local | 673.45 | 0.00 | 17/17 | baseline | 0.00 / 0.00 | 0.00% | baseline pass |
| 17 individual fixed-service MILP replays | 629.36 | -44.09 | 17/17 | 17/17 | -2.60 / -0.96 | 100.00% | conditional oracle pass |
| PPO local BC smoke v1 | 673.06 | -0.40 | 17/17 | 8/17 | +0.28 / +2.73 | 0.90% | safe but not robustly better |
| PPO local storage BC v2 | 643.14 | -30.31 | 17/17 | 17/17 | -1.73 / -0.63 | 68.75% | promoted assisted pilot |
| PPO autonomous evaluation | 700.30 | +26.85 | 15/17 | 6/17 | +0.34 / +11.45 | -60.90% | `REJECT_LOCAL_GATES` |

The promoted PPO result is a **storage controller assisted in runtime**:
individual MILP schedules provide in-window storage demonstrations and
RBCSmart controls EV/deferrable service. It beats RBCSmart in every building,
not merely in aggregate, but it is not an autonomous/deployable PPO claim.

The separate autonomous evaluation disabled the service teacher on the final
deterministic episode. It was rejected because:

- `Building_1` EV precision was `0.0909`, below the `0.40` gate;
- `Building_15` EV minimum-service rate was `0.8235`, and it also recorded
  `0.01556 kWh` / `1` electrical violation event.

Thus autonomous EV/phase-safe control remains open work.

## Individual and community MILP boundary

The individual solver decomposes the non-aggregated problem into 17 isolated
meters; export from one home cannot offset another home's import. Its current
scope is stationary battery control with EV and deferrable behavior frozen to
RBCSmart:

- optimistic lower bounds summed: EUR `578.9128`;
- conservative model schedules summed: EUR `629.8349`;
- safe CityLearn replay: EUR `629.3619`;
- all 17 solver certificates valid and all 17 replays pass local gates.

At this campaign stage this was not yet the theoretical full-house optimum. The existing community
counterpart is also fixed-service/battery-only and uses joint district netting;
its earlier replay cost was EUR `521.4482`. That value belongs to the community
track and must not be compared as though it were an isolated-house bill.

The subsequently implemented full-house and full-community formulations, claim boundaries and
decomposition test are specified in
`docs/milp_local_community_contract_20260729_pt.md`.

## Implementation and evidence

- Strict local observation profile: `utils/entity_adapter.py`
- Individual decomposition: `algorithms/oracles/individual_building_milp.py`
- Individual solve CLI: `scripts/run_individual_fixed_service_battery_oracle.py`
- Per-building scorecard: `scripts/audit_building_local_behavior.py`
- MILP evidence: `runs/analysis/individual_fixed_service_battery_oracle_20260729/`
- Canonical local scorecard:
  `runs/analysis/building_local_ppo_comparison_20260729/`
- RBCSmart job: `rbcsmart-building-local-nomarket-pilot-20260729`
- Canonical individual replay job:
  `individual-fixed-service-oracle-replay-nomarket-pilot-20260729-r3`
- Assisted PPO job: `ppo-building-local-storage-bc-pilot-s123-20260729`
- Autonomous diagnostic job:
  `ppo-building-local-autonomous-eval-s123-20260729-r2`
- Two earlier autonomous-resume jobs failed before producing KPIs and exposed
  checkpoint path/device bugs; both are fixed and regression-tested.
- Full test suite: `662 passed`, `28` known warnings.

## Promotion boundary

All results reuse the same window adaptively and the MILP teacher has perfect
foresight over it. They establish wiring, simulator feasibility and in-window
opportunity, not generalization. Promotion beyond pilot requires separated
train/evaluation windows, seasonal held-out tests and seeds `123/456/789`.
Autonomous PPO additionally requires all 17 local gates without the runtime
service teacher. A global-optimum claim requires the full EV/V2G/deferrable and
network/phase MILPs plus simulator replay.
