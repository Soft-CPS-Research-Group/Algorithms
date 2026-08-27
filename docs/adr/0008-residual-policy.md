# ADR-0008 — Residual and warm-start base policy adaptation

Status: accepted
Date: 2026-08-18
Depends on: ADR-0002, ADR-0003, ADR-0005
Related: ADR-0007, ADR-0011

## Context

MATD3 supports residual policies: actor output is added to a base
action produced by a warm-start policy, scaled by a scheduled
authority. The base action is a per-agent flat vector
of length `action_dimension[i]`. Transformer MATD3 emits per-CA
scalars in `action_names[i]` order.

Sub-decisions:

- 8a — residual support in v1.
- 8b — how the base action combines with actor output.
- 8c — what the critic sees under residual.
- 8d — target-policy smoothing under residual.
- 8e — BC-A cloning target under residual.
- 8f — replay storage of base and cloning actions.

## Plain-language

Analogy: pair-programming rollout. Base policy is the senior
developer; the actor is the junior. Final action is
`senior + authority × junior`. Authority is scheduled from low to high
over training.

## Decisions

### 8a — R-YES: preserve residual

Transformer MATD3 supports residual/warm-start policies. Required for
MATD3 experimental continuity.

### 8b — post-actor composition (P1)

The actor produces per-CA raw scalars. Composition applies outside the
neural graph:

```
final_c = base_c + 0.5 * span_c * effective_scale * mask_c * raw_c
```

Where `_c` denotes per-CA index in `action_names[i]` order. The base
action is delivered in the same order by the warm-start policy;
position-`i` alignment holds under the shared contract
defined in `docs/transformer_entity_controller.md`.

### 8c — critic sees the explicit proposed action (F1)

The action injection MLP (ADR-0003c) receives the proposed action per CA.
This is the actor output after residual composition and service-teacher
replacement, before local safety projection. Replay stores both proposed and
executed actions. The executed action is used for environment outcomes and
diagnostics. Legacy MATD3 `critic_action_input_mode` values `final_base_delta`
and `final_base_delta_normalized` are not implemented in v1.

### 8d — target-policy smoothing per-CA

For each target CA action, add Gaussian noise scaled by
`target_policy_noise * span_c * authority_c`, clipped by
`target_policy_noise_clip * span_c * authority_c`, then clip to
`[low_c, high_c]`. This is the legacy MATD3 target-smoothing rule
applied per CA.

### 8e — BC-A cloning target per-CA

`_transition_cloning_actions` and
`_reachable_behavior_cloning_target` apply the legacy residual-authority
behavior per CA.

### 8f — lazy allocation of base/cloning storage

Replay transitions allocate `behavior_actions`, `next_behavior_actions`, and
`cloning_actions` only when the active configuration needs them. This saves memory in
non-residual, non-BC configs.

## Consequences

- The warm-start policy predict path produces per-CA scalars in
  `action_names[i]` order. Attachment and prediction validate count and width.
- Residual authority schedule uses `_residual_action_effective_scale`.
- Residual mask uses `_residual_action_scale_mask` per CA with action-name
  matching.
- ONNX export rejects residual-enabled configs without an explicit
  runtime-only export flag (see ADR-0011).

## Evidence

- `AgentTransformerMATD3._compose_policy_action` implements composition.
- `_residual_action_effective_scale` and `_residual_action_scale_mask`
  implement scheduled authority.
- `_target_action` implements per-CA target smoothing.
- The shared Transformer entity contract defines CA action order.

## Future improvements

- 8c support alternative action domains by
  expanding the injection MLP input from `(d_model + 1)` to
  `(d_model + 3)`.
- 8b base-conditioned actor (P2): inject base action as a CA-token
  feature before the actor. Research change.
