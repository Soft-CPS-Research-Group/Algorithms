# Transformer MATD3 current status

This document is the concise continuation point for Transformer MATD3 work.
Use the [operational guide](transformer_matd3.md),
[technical specification](transformer_matd3_spec.md), and
[ADRs](adr/README.md) for implementation details and invariants.

## Retained recipe

The current validated recipe for the entity dynamic 15-minute pipeline is
the H7 configuration:

- Reward `CostServiceCommunityPrecisionValueRewardV51` with
  `ev_over_service_penalty: 2400.0`.
- Local action safety with
  `local_action_safety_ev_minimum_mode: average`,
  `local_action_safety_headroom_reserve_kw: 0.0`,
  `local_action_safety_protect_ev_service_target: true`.
- Replay BC enabled with `min_weight: 0.04`, `ev_multiplier: 18.0`,
  `storage_multiplier: 0.08`; RBCSmartPolicy as warm-start teacher.
- Online critic trained on executed actions
  ([ADR 0013](adr/0013-executed-action-critic-and-average-ev-minimum.md)).

## Current evidence

The controlled comparison surface uses
`citylearn_three_phase_dynamic_assets_only_demo_15min_parquet`, steps
`4..11003`, the entity interface, dynamic topology (six add/remove events
between steps 5200 and 9200), two training episodes, and one deterministic
evaluation episode.

Default gates are EV departure success ratio `>= 0.90`, cost/BAU `<= 1.00`,
peak-daily/BAU `<= 1.53`, critic TD absolute late `<= 0.15`, and twin-gap
absolute late `<= 0.15`.

| Metric | V51 baseline | H7 (retained) |
|---|---:|---:|
| District cost/BAU | 1.028 | 0.953 |
| District emissions/BAU | 1.125 | 1.014 |
| District peak all-time/BAU | 1.392 | 0.916 |
| District peak daily/BAU | 1.532 | 0.998 |
| District ramping/BAU | 2.013 | 1.532 |
| EV within-tolerance ratio | 0.198 | 0.902 |
| EV departure success ratio | 0.948 | 0.965 |
| EV soc-surplus mean | 0.111 | 0.026 |
| EV soc-deficit mean | 0.002 | 0.002 |
| Critic TD abs late | 0.204 | 0.123 |
| Critic twin-gap abs late | 0.180 | 0.096 |

Three-seed validation of H7 kept cost/BAU coefficient of variation under
1% and EV within-tolerance CV under 6%. Peak-daily CV was 5.4%.

RBC baselines under the same window and reward:

| Metric | RBCSmart | RBCCommunity |
|---|---:|---:|
| Cost/BAU | 0.841 | 0.908 |
| Peak all-time/BAU | 1.229 | 2.171 |
| Peak daily/BAU | 1.160 | 1.572 |
| EV within-tolerance | 0.968 | 0.951 |
| EV departure success | 0.051 | 0.099 |
| EV soc-deficit mean | 0.028 | 0.026 |

RBCSmart and RBCCommunity land cars precisely inside the tolerance band
but below the target SoC. They report high within-tolerance ratios while
meeting the target only 5–10% of departures. RBCCommunity also breaks
peak and ramping. H7 is the only policy in evidence that meets the EV
target and stabilizes the grid at the same time.

The campaign identifiers for this evidence are:

- `tmatd3_dynamic15min_reward_variants_20260911` (V51 baseline)
- `tmatd3_dynamic15min_critic_executed_20260912` (executed-action critic)
- `tmatd3_dynamic15min_passo3_matrix_20260913` (reward and BC sweep)
- `tmatd3_dynamic15min_passo4_safety_bc_20260913` (safety mode sweep,
  H7 identified)
- `tmatd3_dynamic15min_passo5_validate_optimize_20260913` (seed
  robustness and further BC/reward sweeps; nothing beat H7)
- `tmatd3_dynamic15min_rbc_baselines_20260914` (RBC baselines on the
  same 11000-step window)
- `tmatd3_dynamic15min_8mo_final_comparison_20260914` (8-month final
  comparison of H7 against the three RBC policies)

Detailed artifacts remain outside Git under `runs/remote_results/`.
Recollect them from OPEVA by campaign or job identity when they are not
present locally.

## Evidence limits and next work

The 11000-step window covers roughly 4 months of 15-minute data. The
8-month final-comparison campaign extends the same recipe to 23040
steps per episode and spans the full topology-event window. Full-year
performance and cross-year transfer remain out of scope for this
evidence.

Do not chase reward or BC tuning in isolation. Passo 3 and Passo 5
established diminishing returns on within-tolerance once safety mode is
`average`. The candidate follow-ups are:

1. Store raw next observations in `ReplayTransition` and project the
   target action through the safety adapter to close the remaining
   online-target Bellman gap. Requires a checkpoint format bump.
2. Add multi-seed validation as a default gate for future recipe
   promotions. Passo 5 showed peak metrics keep 5–8% seed variance even
   after H7 is applied.
3. Extend the diagnostic suite so `storage_critic_dq_da_*` is measured
   at executed actions after the target-side projection lands.

## Suggested skills

- Use `opeva-runs` for remote lifecycle, collection, and artifact audits.
- Use `opeva-results-reporting` for completed-run comparisons and dashboards.
- Use the repository review workflow before changing learning behavior.
