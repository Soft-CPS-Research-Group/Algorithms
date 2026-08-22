# AgentTransformerMATD3 Specification

> Status: **Draft (design phase)**
> Scope: Transformer-based MATD3 on the entity interface with dynamic
> topology support.
> Reviewed `main` commit: `b33e1a88ef7636ed3eef6070ab0ff95b239cda8b`
> Maintainer: Algorithms maintainers

This document specifies `AgentTransformerMATD3` before implementation.
It reuses the shared
[Transformer Entity Controller Contract](transformer_entity_controller.md)
verbatim. It preserves classical MATD3 invariants (twin critics,
delayed actor updates, target-policy smoothing, replay-based off-policy
learning, exploration noise, soft target updates) and replaces the
neural architecture with entity-Transformer components that support
variable observations and actions.

Architecture decisions live in numbered ADRs under
[`docs/adr/`](adr/). This specification aggregates their
consequences; individual rationale and options live in each ADR.

## 1. Applicability

- Interface: `simulator.interface: entity`, including
  `simulator.topology_mode: dynamic`.
- Registered as `AgentTransformerMATD3` in `algorithms/registry.py`.
- Must be the final pipeline stage.
- Requires the shared Transformer/entity contract for all
  observation, layout, and action-name invariants.

Non-goals for v1:

- Cross-building actor weight sharing.
- Joint-attention centralized critic (S1).
- Backbone upgrades (positional encoding, attention masks, masked
  pooling).
- `RBCSmartPolicy` teacher extensions.
- Cross-layout checkpoint restore.

## 2. Motivation

Current MATD3 (`algorithms/agents/matd3_agent.py`) inherits fixed
per-agent observation and action dimensions from MADDPG. Every
neural module is sized at construction time
(`algorithms/agents/maddpg_agent.py:645-696`). This blocks dynamic
topology: any change in the number or type of controllable assets
requires a full rebuild that resets weights to random.

The shared Transformer/entity contract solves this via per-type
weight sharing:

- Per-type tokenizer projections (one `nn.Linear` per declared
  type; `algorithms/transformer_ppo/entity_observation_tokenizer.py:86-91`).
- Per-CA action head applied token-wise
  (`algorithms/transformer_ppo/ppo_components.py:44-45,74-79`).
- Type embeddings for `SRO`, `NFC`, `CA` families
  (`algorithms/transformer_ppo/transformer_backbone.py:44,107`).

When a new charger appears mid-training, it inherits the trained
`charger` projection, the trained backbone, and the trained per-CA
head at zero cost. Replay data for its specific instance accumulates
independently; actor and critic weights do not degrade during the
accumulation window (see ADR-0005).

## 3. Architecture ADR index

| ADR | Decision | Locked |
|---|---|---|
| [0001](adr/0001-shared-package-extraction.md) | Extract algorithm-neutral core to `algorithms/transformer_shared/`; delete all shims by end of plan. | accepted |
| [0002](adr/0002-actor-ownership.md) | Per-building actor stack (Option A). | accepted |
| [0003](adr/0003-centralized-twin-critic.md) | M1 × S2b Deep Sets × (ii) post-tokenizer injection × P1 independent. | accepted |
| [0004](adr/0004-backbone-upgrades.md) | No backbone changes in v1. | accepted (no v1 work) |
| [0005](adr/0005-replay-representation.md) | R1 encoded vectors + SIG-C LayoutSignature; KEEP historical; RESET-FULL on building-count change. | accepted |
| [0006](adr/0006-batching-policy.md) | Signature-bucketed sampling; wait when bucket under-full; expose iterate-all-buckets API. | accepted |
| [0007](adr/0007-behavior-cloning.md) | BC-C hybrid with independent enabled flags; BC-B shared, BC-A MATD3-owned; 8 hard boundaries; BC-A current-signature only. | accepted |
| [0008](adr/0008-residual-policy.md) | Preserve residual policy; post-actor composition; critic sees final only; lazy replay allocation. | accepted |
| [0009](adr/0009-local-price-adapter.md) | Pre-tokenization application with `minmax_space` schema guard. | accepted |
| [0010](adr/0010-checkpoint.md) | Format 5; full and inference modes; STRICT signature validation; full replay persistence; n-step queue serialization. | accepted |
| [0011](adr/0011-onnx-export.md) | Per-building per-topology ONNX; opset 17; runtime-only guards for residual, safety, and price. | accepted |
| [0012](adr/0012-schema-registry-wrapper.md) | Dedicated schema class; classvar dynamic-topology; dropout allowed. | accepted |

Glossary of terms: [`docs/transformer_matd3_glossary.md`](transformer_matd3_glossary.md).

## 4. Per-building state

Each building owns an independent stack:

- `tokenizer: EntityObservationTokenizer` (from
  `algorithms/transformer_shared/`).
- `backbone: TransformerBackbone` (from
  `algorithms/transformer_shared/`).
- `actor: DeterministicActorHead` (new; MATD3-owned).
- `actor_target: DeterministicActorHead`.
- Per critic slot `c in {1, 2}`:
  - `critic_<c>: CentralizedCritic` (new; owns tokenizer + backbone +
    action injection MLP + Deep Sets aggregator + Q head).
  - `critic_<c>_target: CentralizedCritic`.
- `actor_optimizer: torch.optim.Adam`.
- `critic_1_optimizer`, `critic_2_optimizer: torch.optim.Adam`.
- `bc_a_optimizer: Optional[torch.optim.Adam]` when BC-A is enabled;
  covers actor + tokenizer + backbone only.
- `bc_b_optimizer: Optional[torch.optim.Adam]` when BC-B is enabled;
  covers actor + tokenizer + backbone only. Separate from
  `bc_a_optimizer` (see ADR-0007 §7d rule 9).
- `layout: BuildingTokenLayout` (from shared package).
- `topology_version: int`.
- `action_names: tuple[str, ...]`.
- `action_low`, `action_high: np.ndarray[float32]`.
- `layout_signature: LayoutSignature` (SIG-C).

The critic stack (tokenizer + backbone + action injection + Deep Sets
aggregator + Q head) is fully independent from the actor stack (per
ADR-0003 P1).

## 5. Data flows

### 5.1 Observation → action (deterministic)

```
env entity payload {tables, edges, meta}
  → EntityContractAdapter.to_agent_observations
    per-building encoded_observation_b: np.ndarray[float32, obs_dim_b]

for building b:
  optional PriceMultiplierObservationAdapter.transform(encoded_obs_b, ctx)
                                                (ADR-0009, minmax_space only)
  → tokenizer(encoded_obs_b, layout_b)
    sro_tokens_b:  [1, N_sro_b, d_model]
    nfc_token_b:   [1, 1, d_model]
    ca_tokens_b:   [1, N_ca_b, d_model]
  → backbone(sro, nfc, ca)
    ca_embeddings_b: [1, N_ca_b, d_model]
    pooled_b:        [1, d_model]              (not used by actor)
  → actor(ca_embeddings_b)
    raw_action_b: [1, N_ca_b, 1]  (pre-tanh scalar per CA)
  → tanh + affine action bounds
    scaled_action_b: [1, N_ca_b]
  → residual composition (ADR-0008), if enabled
    final_action_b = base_action_b + 0.5 * span * scale * mask * scaled_action_b
  → optional local action safety projection
                                                (CityLearnLocalSafetyAdapter)

Combined agent_actions: list[list[float]], one vector per building.
Wrapper converts via EntityContractAdapter.to_entity_actions.
```

### 5.2 Observation → action (stochastic with exploration)

Identical to 5.1 through the actor + affine step. After the affine
step, per-CA Gaussian noise is added:

```
noise_c ~ N(0, sigma * span_c)   (per CA)
clip noise_c to noise_clip * span_c
scaled_action_c = scaled_action_c + noise_c
clip scaled_action_c to [low_c, high_c]
```

Then residual composition and safety projection proceed as in 5.1.
`sigma` decays after each predict per
`sigma = max(min_sigma, sigma * sigma_decay)`; see
`algorithms/agents/maddpg_agent.py:2881`.

### 5.3 Transition → replay

```
matd3.update(observations, actions, rewards, next_observations,
             terminated, truncated, *, update_target_step,
             global_learning_step, update_step,
             initial_exploration_done)

  base_actions        = _transition_behavior_actions(actions)
  next_base_actions   = _transition_next_behavior_actions(base_actions)
  cloning_actions     = _transition_cloning_actions(actions,
                          base_actions=base_actions)          (BC-A)
  layout_signature    = build_signature(current layout)

  buffer.push(
    encoded_obs: dict[building, np.ndarray[float32, obs_dim_b]],
    next_encoded_obs: dict[building, np.ndarray[float32, obs_dim_b]],
    actions: dict[building, np.ndarray[float32, n_ca_b]],
    behavior_actions: dict[...] | None    (lazy)
    next_behavior_actions: dict[...] | None (lazy)
    cloning_actions: dict[...] | None      (lazy)
    reward: np.ndarray[float32, num_agents],
    terminated: bool | np.ndarray[bool, num_agents],
    truncated:  bool | np.ndarray[bool, num_agents],
    layout_signature,
  )
```

### 5.4 Learning step

```
if not (initial_exploration_done and update_step):
  return
if buffer.bucket_size(current_sig) < batch_size:
  return                                          (ADR-0006 U1)

batch = buffer.sample(current_sig, batch_size)

# Per-agent critic update (twin)
with torch.no_grad():
  # target next actions per building
  for b:
    next_ca_emb_b, _ = target_stack_b(next_obs_b)
    next_raw_b       = actor_target_b(next_ca_emb_b)
    next_scaled_b    = affine(next_raw_b)
    next_action_b    = residual_compose(next_scaled_b, base_next_b)
    next_action_b    = add_target_policy_smoothing_per_ca(next_action_b, b)
                                                   (ADR-0008d)
  # centralized target Q per agent
  for agent i:
    q1_next = critic_1_target_i(all buildings' next_obs,
                                all buildings' next_action)
    q2_next = critic_2_target_i(all buildings' next_obs,
                                all buildings' next_action)
    q_next  = torch.minimum(q1_next, q2_next)     (twin invariant)
    y_i     = r_i + gamma**n * q_next * (1 - done_i)  (ADR-0010e n-step)

# Critic loss
for agent i:
  q1_i = critic_1_i(obs, actions)                 (S2b Deep Sets)
  q2_i = critic_2_i(obs, actions)
  loss_1_i = mse(q1_i, y_i)
  loss_2_i = mse(q2_i, y_i)
  optimize critic_1_i, critic_2_i

# Delayed actor + soft targets
if global_learning_step % actor_update_interval == 0:
  for agent i:
    joint = detached policy actions for all buildings
    joint[i] = actor_i(...)                       (with grad on i)
    joint[i] = residual_compose(joint[i], base_i)
    q_pi_i = critic_1_i(obs, joint)
    actor_loss_i = -q_pi_i.mean()
    optionally + BC-A loss (ADR-0007)
    optimize actor_i
    if update_target_step:
      soft_update(actor_i, actor_target_i, tau)
      soft_update(critic_1_i, critic_1_target_i, tau)
      soft_update(critic_2_i, critic_2_target_i, tau)
```

## 6. Interfaces

### 6.1 Agent public API

```python
class AgentTransformerMATD3(BaseAgent):
    supports_dynamic_topology: bool = True
    requires_final_pipeline_stage: bool = True

    def __init__(self, config: dict) -> None: ...

    def attach_environment(
        self, *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def predict(
        self,
        observations,
        deterministic: bool = False,
        *,
        context: Any = None,
    ) -> List[List[float]]: ...

    def update(
        self,
        observations, actions, rewards, next_observations,
        terminated, truncated,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None: ...

    def snapshot_topology_state(self) -> dict: ...
    def restore_topology_state(self, state: dict) -> None: ...

    def save_checkpoint(self, output_dir: str, step: int) -> str: ...
    def load_checkpoint(self, checkpoint_path: str) -> None: ...

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    def get_diagnostic_metrics(self) -> Dict[str, float]: ...
```

### 6.2 Replay buffer public API

New `SignatureBucketedReplayBuffer` in
`algorithms/transformer_matd3/replay.py`:

```python
class SignatureBucketedReplayBuffer:
    def __init__(self, capacity: int, num_agents: int,
                 batch_size: int) -> None: ...

    def push(self, *, encoded_obs, next_encoded_obs, actions,
             behavior_actions=None, next_behavior_actions=None,
             cloning_actions=None, reward, terminated, truncated,
             layout_signature: LayoutSignature) -> None: ...

    def sample(self, signature: LayoutSignature, k: int) -> Batch: ...
    def signatures(self) -> Iterable[LayoutSignature]: ...
    def bucket_size(self, signature: LayoutSignature) -> int: ...
    def total_size(self) -> int: ...

    def get_state(self) -> dict: ...
    def set_state(self, state: dict) -> None: ...
```

### 6.3 Neural module public API

```python
class DeterministicActorHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None: ...
    def forward(self, ca_embeddings: Tensor,
                deterministic: bool = True) -> Tensor: ...
    # returns pre-tanh scalar means [B, N_ca, 1]

class CentralizedCritic(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, hidden_dim: int,
                 dropout: float) -> None: ...
    def forward(self, per_building_obs: List[Tensor],
                per_building_layouts: List[BuildingTokenLayout],
                per_building_actions: List[Tensor]) -> Tensor: ...
    # returns Q: [B, 1]
```

## 7. Training lifecycle

1. Wrapper attaches environment metadata via `attach_environment`.
   Agent builds per-building stacks, twin critics, and replay buffer.
   Optional BC subsystems attach teacher policies (see ADR-0007).
2. Wrapper calls `predict` at every simulator step. Actor emits
   per-CA actions, residual composition applies if enabled, safety
   projection applies if enabled.
3. Wrapper calls `update` after each step with the executed action.
   Agent computes behavior / cloning actions, forms a layout
   signature, and pushes a transition into the signature-bucketed
   replay buffer.
4. When `update_step=True` and the current-signature bucket has at
   least `batch_size` transitions and initial exploration is done,
   agent runs one MATD3 optimization step:
   - Sample from current-signature bucket.
   - Compute target actions with target-policy smoothing per CA.
   - Update twin critics.
   - If `global_learning_step % actor_update_interval == 0`,
     update actor(s) and apply soft target updates.
   - If BC-A enabled and effective weight > 0, add BC-A loss to the
     actor gradient step (actor-only optimizer).
   - If BC-A extra updates are configured, run them (actor-only).
5. Topology change (detected by wrapper via
   `meta.topology_version`):
   - Wrapper snapshots itself and calls
     `agent.snapshot_topology_state`.
   - Agent rebuilds per-building layout, preserves per-type
     tokenizer weights, actor/critic MLPs, optimizer states, replay
     buckets, and value normalizer statistics.
   - Feature-width drift or new type → hard-fail with restore.
   - New signature bucket is initialized in the replay buffer;
     historical buckets remain (KEEP).
   - N-step queue is flushed as truncated to the old-layout
     signature bucket before the layout swap (ADR-0010 §10f). A new
     queue starts under the new layout.
6. Building-count change → full agent rebuild + full replay reset
   (RESET-FULL).
7. Checkpoint at safe boundary: no preflight defer needed. N-step
   queue is serialized in place.
8. Export at run end (or on demand): one ONNX per building per
   current topology.

## 8. Metrics

Metrics use the `TransformerMATD3/` prefix.

Per learning step (aggregate or per-agent):

- `critic_1_loss_mean`, `critic_2_loss_mean`, `critic_loss_mean`
- `critic_td_abs_mean`, `critic_gap_abs_mean`,
  `critic_grad_norm_mean`
- `q1_expected_mean`, `q2_expected_mean`, `q_min_expected_mean`,
  `q_target_mean`
- `actor_update_performed`, `actor_loss_mean`,
  `actor_policy_loss_mean`, `actor_policy_loss_weight`,
  `actor_policy_q_abs_mean`,
  `actor_grad_norm_mean`
- `reward_raw_mean`, `reward_train_mean`, `reward_train_std`
- `replay_buffer_size`, `replay_bucket_size_current`,
  `replay_bucket_count`
- `n_step_returns`, `n_step_queue_size`
- `target_policy_smoothing`, `target_policy_noise`,
  `target_policy_noise_clip`
- `actor_update_interval`
- `exploration_sigma`, `exploration_step`
- `training_step_time`

BC-A metrics (when enabled):

- `actor_behavior_cloning_loss_mean`,
  `actor_behavior_cloning_effective_weight`
- Per action type: `..._ev_loss_mean`, `..._storage_loss_mean`,
  `..._deferrable_loss_mean`, `..._other_loss_mean`
- `actor_behavior_cloning_extra_updates`,
  `actor_behavior_cloning_extra_loss_mean`,
  `actor_behavior_cloning_extra_grad_norm_mean`

BC-B metrics (when enabled): mirror TPPO BC metrics at
`docs/transformer_ppo_spec.md:337-345` with the
`TransformerMATD3/behavior_cloning_*` prefix.

Local action safety metrics (when enabled):

- `local_action_safety_enabled`,
  `local_action_safety_projections`,
  `local_action_safety_interventions`,
  `local_action_safety_infeasible`,
  `local_action_safety_reason_<reason>`

Local price conditioning metrics (when enabled):

- `local_price_conditioning_enabled`,
  `local_price_context_non_neutral`,
  `local_price_clipping_count`

Diagnostics:

- `residual_policy_enabled`, `residual_action_scale_effective`

## 9. Test strategy

New test modules (paths relative to repository root):

- `tests/test_agent_transformer_matd3.py` (unit + integration)
  - Registry construction; `supports_dynamic_topology = True`.
  - `predict` per-CA output shape and finite range within
    `[low, high]`.
  - Deterministic prediction repeatability.
  - Exploration noise decays; respects `min_sigma`.
  - Twin critic Q1 vs Q2 independence (distinct init seeds → distinct
    initial outputs).
  - Target-policy smoothing applied per-CA, clipped by
    `target_policy_noise_clip * span_c`.
  - Delayed actor update: actor gradient step only when
    `global_learning_step % actor_update_interval == 0`.
  - Soft target update on target actor and both target critics.
  - Local action safety projection becomes the executed action.
  - ONNX export produces one file per building per topology.
  - Export guards raise when residual, safety, or price runtime-only
    flags are missing.

- `tests/test_agent_transformer_matd3_replay.py`
  - Signature-bucketed sampling: batch has homogeneous shape.
  - Under-full bucket skips learning step.
  - `signatures()` and `bucket_size(signature)` API.
  - Historical bucket retained after topology change.
  - Feature-width drift fails the topology transaction atomically:
    no new bucket is created and prior live state is restored.
    Instance-count changes with unchanged type widths do create a
    new bucket.
  - Building-count change triggers full replay reset.

- `tests/test_agent_transformer_matd3_checkpoint.py`
  - Round trip in `full` mode with all state restored.
  - Round trip in `inference` mode restores actors + bounds only.
  - Inference-mode load into a non-frozen pipeline stage raises.
  - Version 5 accepted; other versions rejected.
  - Signature mismatch rejected before any state mutation.
  - `num_agents` mismatch rejected.
  - Action-name and action-bound mismatch rejected.
  - N-step queue round trip.

- `tests/test_agent_transformer_matd3_wrapper_integration.py`
  - Wrapper attaches with entity + dynamic without raising.
  - Predict returns per-building per-CA actions.
  - Runtime topology change: layout rebuild preserves per-type
    tokenizer weights and neural weights.
  - `snapshot_topology_state` and `restore_topology_state` round
    trip.
  - Deferred attach rollback on failure restores wrapper + agent
    atomically.

- `tests/test_agent_transformer_matd3_behavior_cloning.py`
  - BC-A: replay-based side loss changes actor but not critics.
  - BC-A: effective weight zero short-circuits.
  - BC-A: only current-signature transitions sampled.
  - BC-B: pretraining fails when a building has zero usable
    demonstrations.
  - BC-B: pretraining trains historical layout groups by stored
    signature.
  - Hard boundary tests, one per invariant in ADR-0007 §7d.
  - Both BC subsystems disabled by default; enabling one keeps the
    other silent.

- `tests/test_agent_transformer_matd3_residual.py`
  - Residual composition matches per-CA formula.
  - Base action ordering respects `action_names[i]`.
  - Target-policy smoothing scales by `span_c * authority_c`.
  - BC-A `_transition_cloning_actions` with residual_authority
    scope ports per-CA.

- `tests/test_agent_transformer_matd3_price.py`
  - Local price adapter modifies four price feature values.
  - Non-`minmax_space` encoding profile + price enabled → schema
    rejection.

Extend existing tests:

- `tests/test_supports_dynamic_topology.py` (add the new agent).
- `tests/test_bundle_validator.py` and
  `tests/test_artifact_manifest.py` for ONNX manifest keys.
- `tests/test_registry.py` for registry entry.

Traceability: each ADR consequence maps to at least one test.

## 10. Known limits

- Supports entity interface only.
- Cross-layout and cross-cardinality checkpoint restore is rejected.
- Export covers only the current topology.
- Local action safety, local price conditioning, and residual
  policies are external to ONNX.
- Building-count change is a full rebuild plus full replay reset.
- BC-A samples current-signature only in v1.
- Backbone has no positional encoding and no attention mask; batches
  must be layout-homogeneous.
- Historical replay buckets do not evict preferentially; long training
  runs with frequent topology changes may push older transitions out
  of the buffer.

## 11. Future improvements

Rolled up from ADR future-improvements sections. Every entry states
the v1 choice, the alternative, and what would change to promote it.

### 11.1 Cross-building actor generalization (ADR-0002)

- v1: per-building actor stacks (Option A).
- Alternative: shared feature stack or fully shared actor (C or B).
- Pros of alternative: parameter sharing; potentially better
  generalization to unseen building compositions; single ONNX graph.
- Cons: requires backbone upgrades (positional encoding,
  `src_key_padding_mask`, masked pooling) and rework of per-building
  auxiliaries (safety, price adapter, residual base).
- To promote: land ADR-0004 upgrades; add building-id embedding;
  refactor safety and price adapter integration to accept a shared
  actor.

### 11.2 Critic architecture (ADR-0003)

- v1: Deep Sets aggregator over per-building encodes (S2b).
- Alternatives:
  - S2b Transformer aggregator: learned importance weighting;
    risk of starving small-building actors of gradient.
  - S1 joint attention: highest expressiveness; requires backbone
    upgrades.
- To promote: for S2b Transformer, swap the aggregator module. For
  S1, land ADR-0004 upgrades + add building-id embedding.

### 11.3 Critic multiplicity (ADR-0003a)

- v1: M1 per-agent critic pair (2N critics).
- Alternative: M2 shared critic pair (2 critics total).
- Pros of alternative: removes a factor of N from critic compute and
  parameter count.
- To promote: replace `2N` critic instances with a shared pair;
  actor loop remains "hold others fixed at detached values."

### 11.4 Critic feature stack sharing (ADR-0003d)

- v1: P1 independent critic tokenizer + backbone.
- Alternative: P2 reuse actor tokenizer + backbone as detached
  extractor.
- Pros of alternative: fewer parameters.
- Cons: couples critic quality to actor feature quality; unusual
  for TD3.

### 11.5 Backbone upgrades (ADR-0004)

Prerequisite for many other improvements. Add positional encoding,
`src_key_padding_mask`, masked mean pooling, building-id embedding
to the shared backbone. All-in-one PR against the shared package.

### 11.6 Raw-payload replay (ADR-0005a)

- v1: R1 encoded vectors + LayoutSignature.
- Alternative: R2 raw entity payload; tokenizer runs at sample time.
- Pros of alternative: encoder-version-independent replay.
- Cons: larger memory; higher per-batch compute.

### 11.7 Padded + masked batches (ADR-0006a)

- v1: B1 signature-bucketed sampling.
- Alternative: B2 padded batches with attention masks; mixes
  signatures per batch.
- Requires: ADR-0004 upgrades.

### 11.8 Cross-signature BC-A (ADR-0007e)

- v1: BC-A samples current-signature only.
- Alternative: BC-A iterates historical buckets for pretraining.
- Requires: an "old-layout mode" actor forward or a shared
  re-tokenization pass under each stored layout.

### 11.9 Critic action modes with delta channels (ADR-0008c)

- v1: critic sees `final` composed action only.
- Alternative: support `final_base_delta` and
  `final_base_delta_normalized` per current MATD3.
- To promote: widen the injection MLP input from `(d_model + 1)`
  to `(d_model + 3)`; extend sampler to hand base and delta scalars.

### 11.10 Base-conditioned actor (ADR-0008b)

- v1: post-actor residual composition (P1).
- Alternative: inject base action as an additional CA-token feature
  before the actor.
- Pros: actor learns to condition its tweak on the base directly.
- Cons: research change; deviates from the current MATD3 baseline.

### 11.11 Compat checkpoint restore (ADR-0010c)

- v1: STRICT rejection on signature mismatch.
- Alternative: reuse compatible type weights on restore, rebuild
  incompatible parts.
- Cons: requires a transactional rollback across the restore
  boundary; harder to reason about correctness.

### 11.12 ONNX safety and residual baking (ADR-0011)

- v1: safety, residual, price all external to ONNX.
- Alternative: bake safety projections that have a static
  ONNX-representable form.
- Requires: a static contract for the safety projector; residual
  base policies with representable graph forms.

### 11.13 Multi-topology polymorphic ONNX (ADR-0011)

- v1: one ONNX per building per topology.
- Alternative: single polymorphic ONNX.
- Requires: ADR-0002 B/C shared actor + ADR-0004 backbone upgrades.

### 11.14 BC mode enum selector (ADR-0007b)

- v1: two independent `enabled: false` flags.
- Alternative: `behavior_cloning.mode: replay | demonstration | both
  | none`.
- Change only if two flags prove operationally confusing.

## 12. Open prerequisites for planning

These require verification during the planning session (small
explore tasks, not architecture decisions):

1. Confirm that warm-start policy classes registered as
   `warm_start_policy_name` targets can emit per-CA actions in
   `action_names[i]` order after `attach_environment`. Referenced by
   ADR-0008 §Open prerequisite.
2. Confirm the wrapper's dynamic-topology capability check
   (`utils/wrapper_citylearn.py:524-540`;
   `utils/config_schema.py:1298-1313`) uses the `supports_dynamic_topology`
   classvar and not a hard-coded allowlist. Referenced by ADR-0012 §12e.
3. Confirm `PriceMultiplierObservationAdapter` accepts the
   per-building encoded vector produced by
   `EntityContractAdapter.to_agent_encoded_observations` when the
   encoding profile is `minmax_space`. Referenced by ADR-0009.
4. Confirm the shared `RunningValueNormalizer` extraction can proceed
   without breaking TPPO's per-building value normalization. Referenced
   by ADR-0001.

## 13. References

- Shared contract:
  [`docs/transformer_entity_controller.md`](transformer_entity_controller.md)
- TPPO spec (comparator):
  [`docs/transformer_ppo_spec.md`](transformer_ppo_spec.md)
- Glossary:
  [`docs/transformer_matd3_glossary.md`](transformer_matd3_glossary.md)
- ADRs: [`docs/adr/`](adr/)
