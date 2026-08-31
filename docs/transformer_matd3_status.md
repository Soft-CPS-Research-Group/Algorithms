# Transformer MATD3 current status

This document is the concise continuation point for Transformer MATD3 work.
Use the [operational guide](transformer_matd3.md),
[technical specification](transformer_matd3_spec.md), and
[ADRs](adr/README.md) for implementation details and invariants.

## Retained implementation

The current branch stabilizes the residual-learning path and its diagnostics:

- Neutral residual components preserve the RBC base action.
- Replay keeps proposed, executed, base, behaviour, and cloning action domains
  distinct. Critics and actor policy-Q learning use proposed actions; executed
  actions remain available for safety and control diagnostics.
- Actor, critic, and target updates advance only after successful updates.
- Delayed actor telemetry remains aligned with its actor-update event.
- Per-building critic gap, TD error, Q-target, gradient, and storage
  sensitivity diagnostics distinguish unavailable values from measured zero.
- Critic loss supports MSE and Huber modes. The validated recipe uses Huber.
- Checkpoint format 6 persists the stabilized replay domains and update
  counters; incompatible format-5 training checkpoints fail explicitly.
- The remote-results collector recovers complete metric streams and simulator
  artifacts when the OPEVA API exposes them.

Per-building reward normalization was removed after evaluation. Global reward
normalization remains the supported mode.

## Current evidence

The controlled comparison surface uses
`citylearn_three_phase_dynamic_assets_only_demo_15min_parquet`, steps
`0..3400`, the entity interface, dynamic topology, four training episodes, and
one deterministic evaluation episode. The retained recipe uses global reward
normalization, unclipped Q targets, Huber critic loss, and storage residual
authority `0.75`.

Default gates are EV minimum feasible rate `>= 0.999`, EV within-tolerance
rate `>= 0.80`, and electrical violation `<= 1e-6 kWh`. Cost is compared only
after these gates pass.

| Candidate | Seed 17 cost delta | Seed 29 cost delta | Decision |
|---|---:|---:|---|
| Global `0.75` reference | `2244.4147 EUR` | `2233.7479 EUR` | Retain |
| Q-target clip `10` | `+17.82 EUR` | `+49.62 EUR` | Reject |
| Per-building normalization | `+1.5518%` | `+1.2967%` | Reject |
| Storage authority `0.375` | `-0.4651%` | `+0.2687%` | Reject |
| Storage authority `0.50` | `-0.2816%` | `+0.4370%` | Reject |

All authority candidates passed the default service and grid gates. Lower
authority reduced storage proposed-to-executed mismatch and safety
interventions, but did not improve learning consistently. Seed 29 regressed
cost, and actor saturation or critic clipping remained above the matched
references. Building 15 service remained stable. Lower losses alone did not
predict better simulator KPIs.

The campaign identifiers are:

- `tmatd3_dynamic15min_qtarget_clip10_20260829`
- `tmatd3_dynamic15min_per_building_reward_norm_20260830`
- `tmatd3_dynamic15min_storage_authority_0375_20260830`
- `tmatd3_dynamic15min_storage_authority_0500_20260830`

Detailed artifacts remain outside Git under `runs/remote_results/`. Recollect
them from OPEVA by campaign or job identity when they are not present locally.

## Evidence limits and next work

The results are short-window evidence. They do not establish full-year
performance, transfer, deployment readiness, or learned cardinality
generalization. Dynamic-layout support proves structural compatibility, not
unchanged performance on unseen entity counts.

Do not run another storage-authority sweep or increase the episode budget from
the current evidence. The next controlled work is:

1. Add per-action-type actor-output quantiles, saturation direction,
   proposed-to-executed deltas, and safety-projection causes. First determine
   whether saturation belongs to storage actions, zero-authority heads, or the
   safety interface. Do not invent a saturation-penalty weight before this.
2. Freeze the retained recipe and evaluate compatible held-out CA and SRO
   cardinalities within the existing buildings, schema, and entity types.
   Require both seeds to pass the default gates, preserve Building 15 service,
   and avoid material mismatch or safety-intervention regressions.

## Suggested skills

- Use `opeva-runs` for remote lifecycle, collection, and artifact audits.
- Use `opeva-results-reporting` for completed-run comparisons and dashboards.
- Use the repository review workflow before changing learning behavior.
