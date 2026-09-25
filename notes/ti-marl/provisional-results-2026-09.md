# TI-MARL provisional evidence and retained lessons

## Status — 2026-09-25

The author has retired this experimental line. The observations below remain
historical evidence and implementation knowledge. They do not select the next
dataset, reward, budget, split, reference or experiment. V22/V23 must not be
continued automatically from the old submission manifests.

## Last locally collected annual comparison

The V21 comparison is a deterministic paired replay of 35,040 15-minute
intervals on the static **17-member** development surface
`ti_marl_static17_full_service_v13_0bc57d9`, using the staged schema in
`ti_marl_static17_full_service_v13_v16_f68361e`. It is not a Core-15, Core-30 or
Premium result. V20 and V21 use the same `0.0495` EV service buffer.

| Metric | V20 reference | V21 candidate |
|---|---:|---:|
| Official settled cost EUR | 20887.196961 | 20894.925778 |
| Feasible EV minimum service | 99.7906% | 99.8604% |
| Feasible EV target tolerance | 98.0809% | 98.1507% |
| Applied electrical violation kWh | 0 | 0 |
| Daily peak indicator | 1.51780 | 1.50963 |
| Ramping indicator | 2.62296 | 2.62576 |
| Solar self-consumption | 72.609% | 72.538% |
| Community import kWh | 124241.184 | 124171.169 |

V21 costs **7.729 EUR more (+0.037%)**. The archived annual acceptance checks
passed matching fingerprints and prices, electrical service, EV minimum and
precision, terminal-energy accounting, and peak/ramping guards. The
`settled_cost_improves` check failed, so the historical decision was **do not
promote**. The exact frozen historical thresholds are retained in
`runs/remote_configs/ti_marl_v21_annual_pending/annual_acceptance_rules.json`;
this outcome is not a pass under a different or future gate profile.

The preceding seasonal comparison reported -0.1655% aggregate cost versus
the paired V20, but the original promotion outcome was also `FAIL`. It records
a post-result amendment to terminal-energy interpretation separately. A
favourable short-window aggregate did not predict annual cost improvement.

## Economic diagnosis retained

The archived annual causal analysis shows that V21 removed EV charging in cheap
periods that often already had community export. Avoiding that consumption had
little or no value under zero-remuneration export settlement. Recovering some
of the service energy in expensive community-import periods increased the
official bill.

The signed net-energy-times-price diagnostic suggested -5.951 EUR, while the
official settlement increased by 7.729 EUR. This is why Algorithms commit
`fdaf764` added a settlement-aware marginal-import option to the EV auxiliary
planner. The code change is retained; the old sweep is not an active research
programme.

Other reusable lessons:

- Use official Simulator settlement for economic decisions and matched
  references for every comparison.
- Distinguish minimum EV service, symmetric target precision and arrival
  above target; they answer different questions.
- Track raw actor intent, feasibility interventions and physically applied
  actions separately. A valid rollout alone does not prove learned safety.
- Preserve terminal stored-energy accounting and declare any post-result
  interpretation amendment.
- Treat deterministic annual replay, seasonal diagnostics, multi-seed
  confirmation and the canonical annual dataset suite as distinct evidence.

## Submitted follow-ups: historical status only

The local V22 interpolation manifest records three annual replay submissions;
the V23 marginal-EV-planning manifest records five training submissions. All
were submitted on 2026-09-04 to Union. Their saved submission status is
`queued`; their current remote status was not checked during this cleanup.
No cancellation, requeue or other remote mutation was performed.

These manifests and their original job identities remain under ignored
`runs/remote_configs/ti_marl_v22_interpolation_pending/` and
`runs/remote_configs/ti_marl_v23_marginal_ev_planning_pending/`. They preserve
provenance, not a queue of work to resume.

## Evidence locations

- `runs/remote_results/ti_marl_v21_annual_confirmation/scorecard.md`
- `runs/remote_results/ti_marl_v21_annual_confirmation/summary.csv`
- `runs/remote_results/ti_marl_v21_annual_confirmation/causal_diagnosis.md`
- `runs/remote_results/ti_marl_v21_headonly_confirmation/scorecard.md`

These local artefacts are intentionally ignored by Git. The separate proposed
thesis-result tables are historical illustrations containing hypothetical
performance numbers; none of those numbers is used as evidence here.
