# ADR 0013: Executed-action critic training and default EV minimum mode

## Status

Accepted.

## Context

Two independent invariants in the runtime pipeline broke expected behavior
during dynamic 15-minute training and evaluation.

1. **Critic input domain.** The online critics regressed against
   `proposed_actions` while the reward and next observation reflected
   `executed_actions` — the actions the environment actually ran after the
   local safety projector. This broke Bellman consistency. Late-training
   diagnostics showed critic TD error and twin-critic disagreement growing
   instead of shrinking, and target Q drifting increasingly negative. Reward
   rescales and higher projection-consistency weights did not close the gap.

2. **EV minimum enforcement mode.** The default
   `local_action_safety_ev_minimum_mode` was `deadline_feasible`. That mode
   computed the energy that cannot be deferred at each step, then combined
   with the charger deadband floor
   (`min_charging_power / max_charging_power`) to force charging near the
   deadline whenever any energy was still needed. Repeated late-window
   activations pushed the SoC above the target within-tolerance band. Across
   14 variants of reward and behavior cloning, the mean surplus-at-departure
   never moved outside the range 0.107–0.112, and departure within-tolerance
   never exceeded 0.24 — while the minimum acceptable SoC ratio stayed above
   0.99. The safety layer, not the reward or the BC anchor, was the binding
   constraint.

The safety layer is not part of the exported ONNX graph. Both defects are
runtime-only behavior; both matter for evaluation numbers and for the
training signal the agent sees.

## Decision

- **Critic online regression on executed actions.** `_learn` now trains
  `critic_1` and `critic_2` on `executed_actions` (the safety-projected
  action) rather than on `proposed_actions`. The target critic still consumes
  the target policy proposal at `s'` because projecting successor actions
  would require raw next observations in replay, which the current schema
  does not carry. The actor loss still evaluates the online critic at the
  actor's own proposal to preserve the policy-improvement gradient. This is a
  minimum-viable Bellman fix.

- **Default `ev_minimum_mode: average`.** The recommended default for the
  Transformer MATD3 and RBC policy recipes is
  `local_action_safety_ev_minimum_mode: average`. The `average` mode reads
  `min_required_action_normalized` from the simulator observation and
  distributes required charging across the remaining time. It is the mode
  under which the policy learns the target-SoC objective without
  adapter-driven overshoot. `deadline_feasible` remains available for cases
  that need bounded intra-step guarantees and can accept overshoot.

## Consequences

- Diagnostic contract. `critic_td_abs`, `critic_gap_abs`, and target Q now
  converge as designed. The late-window TD absolute mean drops from ~0.20
  under the old regime to ~0.12 under the new regime with the same reward
  and safety config.
- Operational KPIs. The Passo 4 comparison recipe (H7:
  `ev_minimum_mode: average`, `headroom_reserve_kw: 0.0`, surplus penalty
  2400, BC replay `min_weight` 0.04) moved district EV within-tolerance
  from 0.20 to 0.90, district cost/BAU from 1.03 to 0.95, and district
  peak-daily/BAU from 1.53 to 1.00, on the same seed and window. Three
  additional seeds validated the same recipe with CV under 6% on cost and
  within-tolerance.
- Bellman residual on the target. Projecting the target proposal would
  reduce the remaining online-vs-target domain gap. It requires an
  additional raw-observation field in `ReplayTransition` and a checkpoint
  format bump. Left as a follow-up.
- Diagnostics that touched proposed actions. The storage-action Q
  sensitivity diagnostic (`storage_critic_dq_da_*`) still evaluates the
  online critic at `proposed_actions`. It is now measured off the training
  distribution. Left unchanged to preserve comparability with earlier runs;
  re-evaluate when the target-side projection is added.

## Alternatives considered and rejected

- **Higher projection-consistency loss weight.** Raising the auxiliary
  actor loss weight from 0.1 to 1.0 did not stop the online critic from
  regressing against off-support actions. It cannot fix Bellman
  inconsistency by regularization alone.
- **Storing target-side raw observations for target-Q projection.**
  Closes the remaining Bellman gap cleanly, but breaks the current replay
  and checkpoint schemas. Not required to hit the operational KPIs on the
  8-month evaluation window.
- **Disabling `protect_ev_service_target`.** Passo 4 H4 showed the target
  cap is not the source of overshoot; disabling it did not move
  surplus-at-departure. Retained.
- **Reducing BC `ev_multiplier` (18.0 → 4.0 / 1.0).** Passo 4 H1 and H2
  showed the EV BC weight is not the binding constraint under
  `deadline_feasible`. Not the singular lever.
- **Enabling `service_teacher` for EV.** RBCSmart landed cars precisely
  below target (deficit ≈ 0.028), which passed within-tolerance but only
  5% met target. Not acceptable as an EV service policy.

## References

- Passo 2 remote runs
  `runs/remote_results/tmatd3_dynamic15min_critic_executed_20260912/`.
- Passo 4 remote runs
  `runs/remote_results/tmatd3_dynamic15min_passo4_safety_bc_20260913/`.
- Passo 5 remote runs
  `runs/remote_results/tmatd3_dynamic15min_passo5_validate_optimize_20260913/`.
- Operational guide section
  `docs/transformer_matd3.md#ev-minimum-enforcement-mode`.
- Schema fields `local_action_safety_ev_minimum_mode` and
  `local_action_safety_headroom_reserve_kw` in `utils/config_schema.py`.
