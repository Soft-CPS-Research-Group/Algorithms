# TPPO BC Correctness Design

## Goal

Make behavior-cloning targets use TPPO's tanh action domain and keep auxiliary samples isolated to their owning building.

## Action Target Normalization

The RBCSmartPolicy teacher returns environment action-space values. TPPO actors produce tanh values before `_affine_action` maps them to environment values. During the demonstration phase, `AgentTransformerPPO.update` will invert this existing affine mapping for each building before calling `record_demonstration`:

`tanh_target = 2 * (environment_action - low) / (high - low) - 1`

The change reuses the already validated per-building `_action_bounds`. `_prepare_action_bounds` rejects non-finite and degenerate (`low >= high`) bounds, so inversion has no zero-span case. The stored BC target will therefore be directly comparable to `torch.tanh(state.actor.mlp(...))`.

## Building Isolation

`BehaviorCloningRegularizer.sample_demonstrations` will accept `building_idx` and read only that building's stored demonstrations before filtering by layout signature. `_run_auxiliary_bc_update` will pass its existing `building_idx`; it will not discard it.

Pretraining remains unchanged because it already groups demonstrations per building.

## Tests

One regression test uses asymmetric, non-unit action bounds. It records a teacher environment action, checks the stored target equals its inverse-affine tanh representation, and verifies zero BC loss against that representation.

One two-building same-layout test records distinct targets, runs the real building-0 auxiliary regularizer path, and verifies the sampled target and resulting loss use only building 0.

## Scope

Only TPPO BC target recording, sampling API/call sites, and focused BC tests change. No policy, environment, or topology behavior changes.
