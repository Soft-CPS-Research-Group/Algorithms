# Phase 10 W6 Local Findings

Updated: 2026-05-31.

## Target

Primary baseline to beat: `RBCSmartPolicy`.

Promotion gates:

- `ev_min_acceptable_feasible_rate >= 0.99`
- no electrical violations
- `ev_within_tolerance_rate >= 0.40`
- cost below the same-window `RBCSmartPolicy`
- battery throughput close to RBCSmart and EV V2G low
- community metrics not materially worse unless cost improves clearly

## Current Window Baselines

Initial comparison window: `win0_0000_2048`, seed `123`.

| recipe | cost EUR | EV min feasible | EV tolerance | battery kWh | EV V2G kWh | read |
|---|---:|---:|---:|---:|---:|---|
| `RBCSmartPolicy` | 4018.05 | 1.0000 | 0.4065 | 5886.2 | 0.7 | target |
| `w6_clone_diagnostic` | 4030.64 | 1.0000 | 0.4479 | 5580.0 | 86.0 | cloning is aligned, but not cheaper |
| `w6_clone_tight_v2g_storage` | 4034.81 | 1.0000 | 0.4127 | 5200.5 | 39.6 | clean, still above cost |
| `w6_clone_cost_nudge` | 3893.79 | 0.9907 | 0.9382 | 8122.0 | 2770.3 | cheap, but abuses EV V2G |
| `w6_clone_cost_gentle_regularized` | 3863.35 | 1.0000 | 0.8671 | 6048.8 | 1257.6 | best cost/EV/battery trade so far, V2G still high |
| `w6_clone_cost_ev_v2g_softwall` | 3845.51 | 0.9892 | 0.8733 | 6966.3 | 138.0 | V2G largely fixed, EV gate missed narrowly |
| `w6_clone_cost_v2g_highclip` | 3830.67 | 0.9985 | 0.9413 | 6312.5 | 1177.3 | highclip helps cost/EV, not V2G |
| `w6_clone_cost_ev_v2g_masswall_gentle` | 3909.14 | 1.0000 | 0.6399 | 5310.4 | 0.0 | first clean pass: beats cost gate, preserves EV, reduces battery/V2G |

`w6_clone_cost_ev_v2g_masswall_gentle` was then run across all four W6A
windows for seed `123`.

| window | cost EUR | delta vs RBCSmart | EV min feasible | EV tolerance | battery kWh | battery ratio | EV V2G kWh | self consumption | read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `win0_0000_2048` | 3909.14 | -108.91 | 1.0000 | 0.6399 | 5310.4 | 0.90 | 0.001 | 0.5080 | pass |
| `win1_2048_4096` | 4646.11 | -25.08 | 1.0000 | 0.6672 | 5840.3 | 1.08 | 0.003 | 0.5076 | pass; battery slightly above RBCSmart |
| `win2_4096_6144` | 5303.67 | -231.64 | 1.0000 | 0.6548 | 5348.6 | 0.98 | 0.003 | 0.4194 | pass |
| `win3_6144_8192` | 2575.61 | -152.04 | 1.0000 | 0.6912 | 5445.2 | 0.92 | 0.010 | 0.4516 | pass |

This is the first recipe that clears the W6A gates on every same-window
comparison for seed `123`: lower cost than RBCSmart, feasible EV service,
better-than-target EV tolerance, negligible EV V2G, no electrical violations,
and battery throughput within a useful range.

Seed `456` was started as a robustness check. The local queue was interrupted
after `win0_0000_2048` because CPU runtime was high and the result was already
enough to validate seed sensitivity on the initial window.

| seed | window | cost EUR | delta vs RBCSmart | EV min feasible | EV tolerance | battery kWh | battery ratio | EV V2G kWh | self consumption | read |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 123 | `win0_0000_2048` | 3909.14 | -108.91 | 1.0000 | 0.6399 | 5310.4 | 0.90 | 0.001 | 0.5080 | pass |
| 456 | `win0_0000_2048` | 3892.82 | -125.23 | 1.0000 | 0.6615 | 5343.3 | 0.91 | 0.000 | 0.5101 | pass |

Reward-level V2G tests on `win0_0000_2048`, seed `123`:

| recipe | cost EUR | delta vs RBCSmart | EV min feasible | EV tolerance | battery kWh | battery ratio | EV V2G kWh | self consumption | read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `w6_clone_cost_ev_v2g_energywall` | 3895.79 | -122.25 | 1.0000 | 0.7975 | 7101.9 | 1.21 | 0.109 | 0.5068 | good EV/cost, too much battery |
| `w6_clone_cost_ev_v2g_energywall_battery_tight` | 3860.09 | -157.96 | 0.9985 | 0.7527 | 5126.5 | 0.87 | 0.013 | 0.5097 | best win0 cost/battery, but EV margin lower |

## Diagnosis

The blocking issue is no longer action mapping or observation encoding. The clone diagnostic tracks RBCSmart closely, and cost-nudge variants learn meaningful cost reductions.

The real failure mode is EV V2G control. The agent finds cost by frequent EV discharge, sometimes while the EV is below required SOC. `softwall` reduced V2G energy strongly, but still produced many tiny negative EV actions and missed the EV feasible gate by a small margin.

The previous actor EV V2G regularizer penalized average negative-action magnitude after filtering to negative samples. That does not strongly penalize the frequency of micro-discharge. W6 now adds `actor_ev_v2g_action_mass_penalty`, which penalizes the negative EV action mass across all EV action slots. This is still a soft actor-loss penalty, not a hard action guard.

Second diagnosis: the reward-side `ev_v2g_service_penalty` only fired when discharge happened while service was at risk. This left EV discharge with no direct penalty when the EV was already above target, so the critic could still value EV V2G as cheap arbitrage. W6 now supports `ev_v2g_discharge_penalty`, an optional per-kWh EV V2G penalty that applies regardless of service risk. The first recipe using it is `w6_clone_cost_ev_v2g_energywall`.

## Operational Finding

Running three local CUDA training jobs concurrently on the RTX 4080 caused `CUDA error: unspecified launch failure`; after that, new PyTorch processes could not initialize CUDA, even though `nvidia-smi` still showed the GPU. Local W6 training should use at most one CUDA training process at a time.

Local queueing is now handled by `scripts/run_phase10_w6_local_queue.py`, which runs selected matrix rows sequentially and clears `CUDA_VISIBLE_DEVICES` by default. Use `--keep-cuda` only when the GPU has been verified healthy.

## Next Experiments

Highest-priority local recipe:

- `w6_clone_cost_ev_v2g_masswall_gentle`

It passed all four W6A windows for seed `123` and confirmed on seed `456`
`win0`. This is enough to prepare W6B promotion while continuing local breadth
tests. Generated promotion matrices:

- `runs/generated_configs/phase10_w6b_masswall_gentle`: MADDPG, 4096-step A100 smoke, seeds 123/456.
- `runs/generated_configs/phase10_w6c_masswall_gentle`: MADDPG + MATD3, full-year, seeds 123/456.

Before remote W6C, run `w6_clone_cost_ev_v2g_energywall_battery_tight` on the
remaining W6A windows and at least seed `456` win0. It is now the most promising
single-window recipe, but `masswall_gentle` remains the safest promotion
candidate because it has already passed all windows for seed `123` and seed
`456` win0 with EV feasible at 1.0.

## 2026-05-31 Follow-Up

The concern that "V2G/battery near zero" was not necessarily good was valid.
Those runs were clean diagnostics, but the better target is controlled
flexibility: use storage/V2G when it creates value, without EV service risk or
runaway throughput.

Two follow-up recipe families were tested on W6A windows:

- `w6_flex_v2g_open_value`: opens EV V2G more aggressively.
- `w6_flex_margin_teacher_storage_tight`: keeps the stronger EV teacher margin
  and tightens stationary storage throughput.

`w6_flex_v2g_open_value` was rejected after `win0_0000_2048`: it produced lower
value than `storage_tight`, raised battery throughput to 1.22x RBCSmart, and
introduced 4.147 kWh EV V2G in the trace, including 1.142 kWh near departure.

`w6_flex_margin_teacher_storage_tight` is now the strongest local W6A candidate
for MADDPG seed `123`.

| window | cost EUR | delta vs RBCSmart | EV min feasible | EV tolerance | battery kWh | battery ratio | EV V2G kWh | read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `win0_0000_2048` | 3869.62 | -148.43 | 1.0000 | 0.7805 | 5534.7 | 0.94 | 0.102 | clean; less cost than masswall, less battery than margin_teacher |
| `win1_2048_4096` | 4571.44 | -99.75 | 0.9984 | 0.7932 | 5938.7 | 1.10 | 0.620 | better cost and much lower battery than margin_teacher |
| `win2_4096_6144` | 5214.88 | -320.43 | 0.9985 | 0.8034 | 5902.6 | 1.08 | 0.137 | best window result so far |
| `win3_6144_8192` | 2510.25 | -217.40 | 1.0000 | 0.7988 | 5829.4 | 0.99 | 0.083 | best window result so far |

Aggregate over the four windows:

| recipe | total delta vs RBCSmart | total battery kWh | avg saving/kWh battery | worst EV min feasible | max battery ratio |
|---|---:|---:|---:|---:|---:|
| `w6_clone_cost_ev_v2g_masswall_gentle` | -517.66 | 21944.5 | 0.02359 | 1.0000 | 1.08 |
| `w6_flex_v2g_margin_teacher` | -735.43 | 26315.8 | 0.02795 | 0.9985 | 1.35 |
| `w6_flex_margin_teacher_storage_tight` | -786.00 | 23205.4 | 0.03387 | 0.9984 | 1.10 |

This is a better tradeoff than both previous candidates: it improves total cost
savings versus `margin_teacher` while reducing battery throughput substantially,
and it beats `masswall_gentle` without dropping below the EV gate.

MATD3 was tested as a win0 comparator for `storage_tight`. It is not currently a
better direction:

| algorithm | cost EUR | delta vs RBCSmart | EV min feasible | EV tolerance | battery ratio | trace EV V2G kWh | read |
|---|---:|---:|---:|---:|---:|---:|---|
| MADDPG | 3869.62 | -148.43 | 1.0000 | 0.7805 | 0.94 | 0.117 | current winner |
| MATD3 | 3953.68 | -64.37 | 1.0000 | 0.6430 | 0.90 | 2.651 | worse cost, worse tolerance, more unsafe/near-departure V2G |

Local operational note: MATD3 also used materially more RAM locally, around
4.4 GB RSS versus roughly 3.1-3.2 GB for MADDPG in comparable W6A runs.

Recommended next promotion candidate:

- MADDPG + `w6_flex_margin_teacher_storage_tight`.

Recommended remote comparison set:

- primary: `w6_flex_margin_teacher_storage_tight`;
- conservative fallback: `w6_clone_cost_ev_v2g_masswall_gentle`;
- skip `w6_flex_v2g_open_value`;
- skip MATD3 unless remote GPU smoke later reveals a large speed/stability
  advantage, which local evidence does not currently support.

If seed `456` fails:

- too expensive: raise actor policy loss slightly, but keep the mass penalty;
- EV gate miss: increase EV BC and teacher anchoring rather than adding a hard action guard;
- V2G returns: keep actor masswall and add reward-level `ev_v2g_discharge_penalty`.

Other candidates worth keeping:

- `w6_clone_cost_gentle_regularized`: good cost and EV, but needs V2G suppression.
- `w6_clone_cost_ev_v2g_softwall`: good cost and V2G, but needs slightly stronger EV anchoring.
- `w6_clone_cost_v2g_highclip`: proves reward clipping matters, but is not enough alone.
- `w6_clone_cost_ev_v2g_energywall`: tests whether moving V2G suppression into the critic/reward is cleaner than actor-only regularization.
