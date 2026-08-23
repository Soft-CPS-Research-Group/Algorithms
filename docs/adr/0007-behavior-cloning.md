# ADR-0007 — Behavior cloning configuration and hard boundaries

Status: accepted
Date: 2026-08-18
Depends on: ADR-0001, ADR-0005, ADR-0006
Related: ADR-0008

## Context

Behavior cloning must be optional with clear boundaries. Two BC
patterns exist:

- BC-A (replay-based) — MATD3-native side loss using
  `cloning_actions` stored per replay transition.
  `algorithms/agents/maddpg_agent.py:3180-3227,4229-4394,4396-4737`.
- BC-B (demonstration-based) — TPPO-style dedicated demonstration
  reservoir with actor-only pretraining and optional auxiliary loss.
  `algorithms/transformer_ppo/behavior_cloning.py:33-666`.

## Plain-language

- BC-A is a continuous side loss during MATD3 updates — like a
  CI job running on every commit.
- BC-B is dedicated pretraining before RL, like language-model
  pretraining on a curated corpus.

Both remain useful; both are opt-in.

## Decisions

### 7a — variants shipped: BC-C hybrid

Both BC-A and BC-B ship in v1. Both default disabled.

### 7b — schema surface: independent enabled flags

Two independent optional blocks under `behavior_cloning`:

```yaml
algorithm:
  behavior_cloning:
    replay_based:            # BC-A
      enabled: false
      teacher: "warm_start"  # "warm_start" | "replay_action" | "external"
      weight: 0.0
      min_weight: 0.0
      decay_start_step: 0
      decay_steps: 0
      ev_multiplier: 1.0
      storage_multiplier: 1.0
      deferrable_multiplier: 1.0
      extra_update_start_step: 0
      extra_update_end_step: 0
      offline_pretrain_steps: 0
    demonstration_based:     # BC-B
      enabled: false
      demonstration_episodes: 1
      max_samples_per_building: 4096
      pretraining_epochs: 4
      batch_size: 64
      weight: 0.0
      min_weight: 0.0
      decay_start_step: 0
      decay_steps: 0
      ev_multiplier: 1.0
      storage_multiplier: 1.0
      teacher:
        policy: "RBCSmartPolicy"
        hyperparameters: {}
```

Absent block or absent sub-block = disabled. Explicit `enabled: true`
required per sub-block.

### 7c — module location: L2

BC-B code moves to `algorithms/transformer_shared/behavior_cloning.py`
in the extraction PR (ADR-0001 updated). BC-A stays in
`algorithms/transformer_matd3/behavior_cloning.py` and reuses
MADDPG-inherited methods.

### 7d — hard boundaries

The following invariants are inviolable and tested:

1. BC never updates critics or their targets.
2. BC never updates value normalizer statistics.
3. BC never mutates replay buffer state.
4. BC never runs when effective weight is 0 (short-circuit before
   forward).
5. BC storage is isolated:
   - BC-A reads `cloning_actions` from replay; no BC code writes to
     replay.
   - BC-B uses a separate reservoir keyed by LayoutSignature.
6. BC teachers are never serialized. They are rebuilt from config on
   `attach_environment`.
7. Absent config block → BC code never executes.
8. BC-B pretraining fails before the first RL update when any building
   has zero usable compatible demonstrations. The unsupported BC-A
   `external` teacher is rejected during configuration validation. A
   supported `warm_start` teacher remains non-blocking when its runtime
   context is unavailable.
9. When both BC subsystems are enabled, they use **separate**
   optimizers, not a shared one.

Enforcement: per-building, per-subsystem BC optimizers cover only
actor + tokenizer + backbone parameters (mirror
`algorithms/transformer_ppo/agent.py:1307-1312`). Concretely:

- `bc_a_optimizer_b` when BC-A is enabled.
- `bc_b_optimizer_b` when BC-B is enabled.

Rationale for separation: Adam moment estimates are per-parameter but
scaled by the gradient signal that feeds them. BC-A and BC-B have
different loss magnitudes, different schedules, and different sampling
distributions. Sharing one optimizer instance couples their moment
estimates and distorts each subsystem's implicit learning-rate
behavior. The cost of two optimizers (moment tensors sized like the
actor stack) is negligible against the correctness gain.

### 7e — BC-A sampling scope: current-signature only

BC-A samples only from the current-signature bucket. Cross-signature
pretraining is BC-B's domain. Rationale: the per-building actor stack
carries only the current layout; cross-signature actor loss would
require a re-tokenization pass under stored layouts, which BC-B
already implements.

## Consequences

- ADR-0001 extraction PR includes BC-B extraction with a shim.
- New `TransformerMATD3StageConfig.behavior_cloning` optional block.
- BC-A reuses `_actor_behavior_cloning_loss`,
  `_actor_behavior_cloning_type_losses`, `_transition_cloning_actions`
  from MADDPG.
- BC-B reuses `BehaviorCloningRegularizer` verbatim from the shared
  package.
- Test suite includes one test per hard-boundary invariant.

## Evidence

- MATD3 BC methods: `maddpg_agent.py:3180-3227,4229-4394,4396-4737`.
- TPPO BC regularizer: `behavior_cloning.py:33-666`.
- Separate BC optimizer pattern:
  `algorithms/transformer_ppo/agent.py:1307-1312`.

## Future improvements

- BC-A cross-signature pretraining, contingent on a "old-layout mode"
  actor forward or a shared re-tokenization pass.
- Single `mode` enum selector if two independent flags prove
  confusing.
- Broaden BC-B teacher registry beyond `RBCSmartPolicy`.
