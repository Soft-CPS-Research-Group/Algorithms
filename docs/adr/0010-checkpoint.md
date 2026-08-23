# ADR-0010 — Checkpoint format, layout signature validation, replay persistence

Status: accepted
Date: 2026-08-18
Depends on: ADR-0005, ADR-0007, ADR-0008
Related: ADR-0011, ADR-0012

## Context

Two existing checkpoint contracts:

- Legacy MATD3: `checkpoint_version = 2`, modes `full` and `inference`.
- TPPO: `checkpoint_version = 4`, on-policy preflight defer, and strict layout
  signature validation.

Transformer MATD3 combines MATD3's replay/exploration state with
TPPO's per-building neural stack and layout signature validation.

## Plain-language

Analogy: MATD3's checkpoint is a database backup for the trained
system (actor + critic + optimizer + replay). Add TPPO's schema-hash
validation on restore to prevent loading a snapshot into an
incompatible topology.

## Decisions

### 10a — format version: 5

Global monotonic version chain. `checkpoint_version = 5`. Loader
accepts version 5 only; historical versions from MATD3 (2) and TPPO
(≤4) are not cross-loaded.

### 10b — modes: full + inference

Preserve both MATD3 modes.

- Full: all trainable state, replay buffer, exploration, reward
  normalization, RNG, optional BC state.
- Inference: per-building tokenizer + backbone + actor + affine
  bounds; plus `inference_policy_state.exploration_step` for
  residual-authority schedule reconstruction.

Inference-mode load into a non-frozen pipeline stage is rejected, matching the
legacy MATD3 safety rule.

### 10c — signature validation: STRICT

Restore hard-fails on any per-building LayoutSignature (SIG-C)
mismatch. Cross-layout and cross-cardinality restore are rejected.
Runtime topology changes remain compat-aware through
ADR-0002/ADR-0005/ADR-0006.

### 10d — replay persistence: full replay

Full mode persists the replay buffer including all signature buckets
via `get_state()` / `set_state()`. Historical buckets remain
available post-restore for analytics and future extensions. BC-A samples only
the current signature.

### 10e — n-step queue: serialize

The n-step queue (`_n_step_queue`) is serialized in full-mode checkpoints
and restored on load. Bounded payload (max `n_step_returns`
transitions). No preflight defer needed.

### 10f — n-step queue at topology boundary

Runtime topology changes (not restore) can leave queue entries that
span incompatible layouts. The wrapper hands the old observation,
action, and reward but no shape-compatible next-observation across the
transition in `utils/wrapper_citylearn.py`.

Rule: at any topology-change commit (compatible or full rebuild), the
n-step queue is **flushed as truncated**. Every pending entry is
pushed to replay under its stored old-layout signature with
`truncated = True`, using the last observed reward chain and no
bootstrap. The queue is then cleared. A new queue starts under the
new layout.

Rationale: truncation is semantically closer to the environment
boundary than either discard or forced bootstrap. Discard silently
loses data; forced bootstrap requires a shape-compatible next-obs
under the new layout that does not exist.

## Checkpoint payload structure

Header:
- `checkpoint_version: 5`
- `algorithm: "AgentTransformerMATD3"`
- `checkpoint_mode: "full" | "inference"`
- `step: int`
- `num_agents: int`
- `building_names: list[str]`

Per building (full or inference):
- `tokenizer_state_dict_<b>`
- `backbone_state_dict_<b>`
- `actor_state_dict_<b>`
- `layout_signature_<b>: SIG-C tuple`
- `action_names_<b>: tuple[str, ...]`
- `action_bounds_<b>: (low: float[], high: float[])`
- `topology_version_<b>: int`

Per building (full only):
- `tokenizer_target_state_dict_<b>`
- `backbone_target_state_dict_<b>`
- `actor_target_state_dict_<b>`
- For each critic pair `c in {1, 2}`:
  - `critic_<c>_state_dict_<b>`
  - `critic_<c>_target_state_dict_<b>`
  - `critic_<c>_optimizer_state_dict_<b>`
- `actor_optimizer_state_dict_<b>`
- optional `bc_a_optimizer_state_dict_<b>`
- optional `bc_b_optimizer_state_dict_<b>`

Full mode global additions:
- `replay_buffer` (per ADR-0005/ADR-0006 signature-bucketed state)
- `n_step_queue` (serialized)
- `exploration_state: {sigma, exploration_step}`
- `reward_normalization_state: {enabled, count, mean, m2}`
- `rng_state: {python, numpy, torch, torch_cuda}`
- `bc_state` (optional; per ADR-0007)
  - `bc_a_state` when BC-A enabled: side loss decay clock
  - `bc_b_state` when BC-B enabled:
    `BehaviorCloningRegularizer.state_dict()` reservoir + signatures

Inference mode global additions:
- `inference_policy_state: {exploration_step}` for residual
  authority reconstruction.

## Restore validation order

1. Load payload with `torch.load(..., weights_only=False)`.
2. Validate `checkpoint_version == 5`.
3. Validate `algorithm == "AgentTransformerMATD3"`.
4. Validate `checkpoint_mode` consistent with pipeline stage.
5. Validate `num_agents` equals live `num_agents`.
6. Per building:
   - `layout_signature_<b>` equals live signature (SIG-C).
   - `action_names_<b>` equal live action_names.
   - `action_bounds_<b>` equal live bounds within tolerance.
7. Only after every check passes: mutate agent state.

Any mismatch raises before any live state is changed.

## Consequences

- New `save_checkpoint` and `load_checkpoint` on
  `AgentTransformerMATD3`.
- Test coverage includes: version 5 round-trip, signature-mismatch
  rejection, cardinality-mismatch rejection, action-bound mismatch
  rejection, inference-into-non-frozen rejection, n-step queue
  round-trip, BC state round-trip.

## Evidence

- Legacy MATD3 defines the full and inference mode split.
- The TPPO specification defines the earlier strict layout-signature check.
- Transformer MATD3 owns `_n_step_queue` and validates every restored entry.

## Future improvements

- 10c COMPAT restore that reuses type-compatible weights on restore.
  Requires transactional rollback across restore boundary.
