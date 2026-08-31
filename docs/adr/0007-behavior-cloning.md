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
  The established implementation is in legacy MADDPG/MATD3 helpers.
- BC-B (demonstration-based) — TPPO-style dedicated demonstration
  reservoir with actor-only pretraining and optional auxiliary loss.
  The shared implementation is `BehaviorCloningRegularizer`.

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

BC-B lives in `algorithms/transformer_shared/behavior_cloning.py`.
BC-A lives in `algorithms/transformer_matd3/behavior_cloning.py`, with
controller integration in `algorithms/transformer_matd3/agent.py`.

### 7d — hard boundaries

The following invariants are inviolable and tested:

1. BC never updates critics or their targets.
2. BC never updates reward-normalizer statistics.
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
9. BC-A and BC-B use the same per-building actor optimizer as the policy
   update. Actor-only boundaries remain enforced.

Enforcement: the per-building actor optimizer covers only actor + tokenizer +
backbone parameters. The `bc_a_optimizer` and `bc_b_optimizer` compatibility
fields refer to this optimizer when their subsystem is enabled. Critic
optimizers remain separate.

Rationale: policy, BC-A, and BC-B all update the same actor parameters. One
Adam state prevents independent actor-only moments from competing during the
same training run. Each objective still has its own schedule and metric.

### 7e — BC-A sampling scope: current-signature only

BC-A samples only from the current-signature bucket. Cross-signature
pretraining is BC-B's domain. Rationale: the per-building actor stack
carries only the current layout; cross-signature actor loss would
require a re-tokenization pass under stored layouts, which BC-B
already implements.

## Consequences

- ADR-0001 places BC-B in the shared package. No shim remains.
- `TransformerMATD3StageConfig.behavior_cloning` provides both optional blocks.
- Transformer MATD3 owns its BC-A loss, target, and update helpers.
- BC-B reuses `BehaviorCloningRegularizer` verbatim from the shared
  package.
- Test suite includes one test per hard-boundary invariant.

## Evidence

- Legacy MADDPG/MATD3 provides the replay-BC behavior baseline.
- `BehaviorCloningRegularizer` provides the shared demonstration reservoir.
- Both Transformer agents constrain BC optimizers to actor-stack parameters.

## Future improvements

- BC-A cross-signature pretraining, contingent on a "old-layout mode"
  actor forward or a shared re-tokenization pass.
- Single `mode` enum selector if two independent flags prove
  confusing.
- Broaden BC-B teacher registry beyond `RBCSmartPolicy`.
