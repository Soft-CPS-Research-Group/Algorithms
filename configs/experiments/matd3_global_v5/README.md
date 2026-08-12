# MATD3 global V5

This campaign follows the annual V4 finding: MATD3 reduced settled cost by
EUR 26.74 against paired SMART, but worsened ramping by 20.27%, daily peak by
0.90% and 9 of 17 member costs. V5 does not promote that model.

## Corrections

- SMART remains the immutable runtime base for EV, deferrable and battery
  actions; MATD3 has residual authority over stationary batteries only.
- The final residual authority is reached at step 12,288 of training year one,
  leaving the remainder of year one and all of year two on-policy at the
  evaluation authority.
- Four causal frames expose recent net exchange and price evolution to the
  actor, which is needed for a ramping objective.
- Community import/peak/ramping penalties use net community exchange, matching
  the physical scorecard boundary.
- Training-year checkpoints are preserved at the exact 35,039-step episode
  boundary. The annual interval of 35,040 used by V4 never selected that
  boundary.
- Replay stores executed action, SMART residual base and behavior-cloning
  target independently.
- During gradual authority growth, cloning targets are projected into the
  action range currently reachable by the residual actor. The projection
  expands with authority, avoiding impossible early losses that merely
  saturate the policy.
- The annual SMART-to-MILP audit found an absolute normalized battery-action
  difference of 0.092 at p50, 0.200 at p90 and 0.306 at p95. V4 exposed only
  about 0.13 effective authority, leaving 42.7% of teacher actions unreachable.
  V5 exposes 0.570 (cost-first), 0.523 (balanced) and 0.450 (ramp-guard), with
  gradual growth and no EV/deferrable authority. The cost-first setting covers
  more than 99% of audited teacher deltas; the remaining extreme tail stays a
  deliberate guard against unconstrained full-action overrides.

## Candidate frontier

- `cost_first_h2`: strongest settlement weighting and light ramp/peak guards.
- `balanced_h2`: joint cost, peak and ramping candidate.
- `ramp_guard_h2`: deliberately stronger physical smoothing ablation.
- `global_scorecard_h2`: fully cooperative critics (team-reward mix 1.0),
  wider storage authority and the joint cost/peak/ramp objective. This is the
  direct test of whether mixed local objectives were blocking a better global
  solution.
- `global_distilled_h2`: the same fully cooperative objective with persistent
  training-only supervision from the promoted physical scorecard schedule.
  Its policy loss is deliberately smaller while behavior cloning decays from
  0.45 to a non-zero 0.06 anchor across the two training years.
- `*_milp_cost_teacher_*`: training-only cost-optimal fixed-service storage
  demonstration for the cost-first candidate.
- `*_milp_scorecard_teacher_*`: training-only cost-constrained demonstration
  that jointly lowers modeled ramping, daily peak and all-time peak for the
  balanced, ramp-guard and fully cooperative candidates.

Every deterministic evaluation remains a causal MATD3 actor over SMART. No
teacher schedule is available to the evaluated actor.

Each candidate has four 4,096-step seasonal pilots and one annual config. All
comparisons use a paired SMART reference with the same seed, window and third
episode realization. Seasonal filters keep only their latest inference
checkpoint; annual candidates retain both training-year snapshots. This
avoids duplicating multi-hundred-MB checkpoint sets across shared workers.
Pilot exploration, SMART phase-out, actor warm-up and residual-authority
growth are scaled to the shorter window: full authority is reached halfway
through training episode one, leaving one and a half training episodes at the
deployment authority. Annual timings remain fixed at 1,024/8,192/12,288
steps as documented above.

## Promotion protocol

First reject seasonal candidates that do not beat paired SMART on settled cost
or have systematic ramp/peak regressions. Run the surviving annual candidate
and replay both preserved training-year checkpoints. Promotion requires:

- lower settled community cost than paired SMART;
- no material electrical or service regression;
- no material regression in daily peak or ramping for the balanced claim;
- member-cost wins reported explicitly (target at least 12/17);
- confirmation on seeds 123, 456 and 789 before freezing a standalone leaf.

The packaged MILP is a conditional fixed-service teacher and comparison
instrument, not a global-optimum certificate for the full problem.

## Demonstrated physical margin

The promoted scorecard teacher was replayed for the full year with the paired
SMART controller, seed and settlement configuration. CityLearn obtained EUR
20,244.08, 124,067.54 kWh import, 21,811.00 kgCO2 gross member emissions, a
0.99560 daily-peak ratio, a 1.08398 all-time-peak ratio and a 0.92556 ramping
ratio. The paired SMART source is EUR 21,957.37, 132,708.21 kWh, 22,407.36
kgCO2, 1.07062, 1.13440 and 2.40401, respectively. All 17 member settled
costs improve. This is evidence of available physical margin, not a MATD3
result and not a full-problem optimum claim.

The replay also produced 0.3337 kWh of electrical violations. That value is
retained in every comparison: it is inside the agreed sub-kWh experimental
tolerance but does not pass the repository's strict zero-violation gate.

The teacher was produced by a physical coordinate heuristic with global
cost/peak/ramp constraints. The exact replay retains neutral
`SignalAwareRBC`, which is behaviorally identical to `RBCSmartPolicy` at
multiplier 1.0; MATD3 uses that plain SMART behavior as its immutable residual
base. CityLearn recomputes EV and deferrable actions under the changed battery
trajectory, so equivalence means the same controller behavior and
configuration, not a byte-identical service trace.

The separate cost-only teacher replayed at EUR 18,515.03 and improved all
17 member costs, but its daily and all-time peak ratios were 1.13337 and
1.17509. It is therefore retained only for the `cost_first_h2` ablation; the
promoted global scorecard teacher is the reference for claims of simultaneous
economic, emissions and physical improvement.
