# AgentTransformerMATD3 Design

## Objective

Create `AgentTransformerMATD3`: a production-strength MATD3 agent that uses
entity-token Transformer actors and centralized Transformer critics. The goal
is to keep the strongest parts of the existing MATD3 recipe while gaining the
dynamic-topology flexibility already proven by `AgentTransformerPPO`.

The agent supports the `entity` interface and dynamic topology from day one.
It handles assets entering or leaving mid-episode through `topology_version`-
driven layout rebuilds, without relying on flat-vector sentinels or fixed
action dimensions.

## Design Priorities

Every decision in this spec is scored against these priorities, in order:

1. Topology changes work end to end and dynamic input/output shapes are
   handled correctly.
2. MATD3 principles are preserved (twin critics, min-Q target, target policy
   smoothing, delayed actor updates, soft target updates, replay, reward
   normalization).
3. Training and inference performance are respected. Compute and memory
   overheads are made explicit where they matter.

Where a decision creates a known limitation, it is captured in the
`Look If We Need Improvements` section.

## Core Principles

- Deployment is local. Each building exports its own actor model and
  tokenizer/layout metadata.
- Training is centralized. Twin critics may consume community-wide state and
  action context.
- Critics, replay buffers, teacher actions, residual policy state, and BC
  machinery are training-only and are never exported.
- Reuse Transformer/entity concepts from `AgentTransformerPPO`, but keep
  shared Transformer-domain code in a neutral Transformer package (not under
  PPO, not under a generic `utils` bucket).
- Preserve MATD3 strengths: twin critics, target policy smoothing, delayed
  actor updates, warm-start policy, residual-over-teacher policy, behavior
  cloning, reward normalization, diagnostics, and checkpoint/resume.
- All asset-type behavior (BC weighting, residual scales, diagnostics) is
  keyed off CA token `type_name` from the layout. Action-name string matching
  is a debug fallback only.

## Architecture

`AgentTransformerMATD3` is a new standalone `BaseAgent` implementation. It is
registered as `AgentTransformerMATD3` beside `AgentTransformerPPO` and
`MATD3`. It does not inherit from `MATD3` or `MADDPG` because those classes
assume fixed flat dimensions across replay, action processing, export, and
diagnostics.

Each building owns a deployable actor stack:

- `EntityTokenLayoutBuilder` output for the building.
- Entity observation tokenizer.
- Transformer backbone.
- Deterministic actor head that emits one scalar per CA token in the layout
  order (`layout.ca_action_names`).
- Target actor stack.
- Actor optimizer.
- Per-building topology version and layout signature.

Training owns centralized, non-deployable critic state:

- Global critic token packer.
- Two fully independent Transformer critic stacks (critic 1 and critic 2),
  each with its own backbone, type embeddings, and Q head. This preserves
  TD3's overestimation reduction property; a shared backbone would make the
  twin critics correlated and undermine min-Q.
- Two independent target critic stacks mirroring the online critics.
- Critic optimizer(s) (one per critic stack, or a combined optimizer over
  both — implementation choice).
- Replay partitions keyed by topology signature.

The exported artifact boundary is hard: only per-building actor pipelines are
exported to ONNX. `artifact_manifest.json` contains only actor artifacts.
Critics, replay, and teacher/residual/BC state live in the training checkpoint
file and are never referenced by the manifest.

**Controlled building** is defined here as a building whose current layout has
`n_ca >= 1`. Only controlled buildings produce Q-values, actor updates, and
per-building diagnostics. Uncontrolled buildings still contribute observation
tokens to the critic as community context. If a building loses all CAs
(becomes uncontrolled), its actor optimizer and target actor state are frozen
in place; if it later regains CAs with the same token-type feature dimensions,
training resumes from the frozen state without rebuild.

## Package Boundaries

Shared Transformer/entity pieces live in a neutral Transformer-domain package,
not in `AgentTransformerPPO` and not in a generic `utils` bucket. The
following modules are Transformer-domain shared code:

- `algorithms/utils/entity_token_layout.py`
- `algorithms/utils/entity_observation_tokenizer.py`
- `algorithms/utils/transformer_backbone.py`
- `utils/entity_tokenizer_schema.py`
- `configs/tokenizers/entity_default.json`

The implementation should either keep these files where they are (marking them
as neutral Transformer-domain modules and adding no PPO-specific assumptions),
or move them into a clearer neutral package such as
`algorithms/entity_transformers/`. Choose the option that minimizes churn at
implementation time.

Algorithm-specific code stays separate:

- PPO-only logic remains with `agent_transformer_ppo.py` and PPO-specific
  components (`ppo_components.py`, PPO BC regularizer, etc.).
- MATD3-only logic goes into `agent_transformer_matd3.py` and supporting
  MATD3-specific modules, such as deterministic actor heads, action-token
  encoders, global critic packing, twin critic heads, replay partitioning,
  and MATD3 diagnostics.

## Data Flow

The wrapper remains the owner of the simulator entity contract. It converts
the simulator payload into per-building encoded vectors and per-building
action names. `AgentTransformerMATD3` receives those vectors through the
existing `BaseAgent` methods and returns `List[List[float]]` actions in the
same contract used by `AgentTransformerPPO`.

Prediction flow per building `b`:

1. Tokenize the encoded building observation with the cached layout.
2. Run the local Transformer actor backbone.
3. Emit deterministic actor means in `[-1, 1]`, one scalar per CA token.
4. In training mode (`deterministic=False`):
   - Compose with the teacher policy per the residual rules (see below) if
     residual is enabled.
   - During warm-start/phaseout window, optionally blend or replace with
     teacher actions per the exploration rules (see below).
   - Add exploration noise; clip to `[-1, 1]`.
5. Return actions ordered by `layout.ca_action_names[b]`.

Training flow at step `t`:

1. Store the transition with the current topology signature, per-building
   layout summaries, and teacher/base actions.
2. Gate training: skip if `initial_exploration_done` is false and
   `train_during_initial_exploration` is disabled. Skip if
   `len(active_partition) < batch_size`.
3. Otherwise sample from the active-signature partition only.
4. Build global observation/action tokens for all buildings; apply padding
   masks for variable token counts.
5. Compute target actions from target actors, apply residual composition
   with target teacher actions, then apply target policy smoothing in the
   final action space.
6. Update twin critics with the min-Q TD3 target.
7. On delayed-actor-update steps, update each controlled building's actor
   against critic 1, keeping other buildings' actions detached. Critic
   parameters are frozen (excluded from gradient) during actor loss
   computation.
8. Soft-update actor and critic targets on scheduled target-update steps.

The wrapper's per-agent `rewards: List[float]` (see
`utils/wrapper_citylearn.py:1234-1301`) is used as-is. `len(rewards)` must
equal the current number of per-building states; a mismatch is a fail-fast
error. Reward normalization is per building.

Community context reaches the critic through each building's SRO tokens.
No new wrapper hook is introduced.

## Dynamic Topology Lifecycle

The agent uses the same wrapper-driven lifecycle as `AgentTransformerPPO`:

- First `attach_environment(...)` builds all actor layouts, actor stacks,
  critic packing metadata, replay partitions, and teacher/residual/BC
  metadata.
- A repeated attach with identical `observation_names` and `action_names` is
  a no-op.
- A topology change is detected by changed per-building `observation_names`
  or `action_names`, which is the signal produced by the wrapper after any
  `topology_version` increment.

Supported topology changes:

- Asset-count changes within existing buildings (add/remove chargers, storage,
  PV, EVs).
- Feature-count-stable schema (per-type feature dimensions are unchanged).

Unsupported in the first version:

- Building-count changes. If the number of buildings changes on `attach`, or
  a saved checkpoint has a different building count than the current
  environment, the agent fails fast with a clear message. This is captured in
  `Look If We Need Improvements`.

On a supported topology change, per changed building `b`:

1. Rebuild `layouts[b]` from the new observation/action names.
2. Recompute global critic packing metadata and masks for both twin critics.
3. Validate tokenizer coverage, CA action ordering, and feature-count
   stability. Fail fast if a token type's feature count changes.
4. Preserve actor/critic weights and target weights, because per-type
   projections are shared across instances of the same type.
5. Switch active replay sampling to the new topology signature. Old
   partitions are retained read-only subject to eviction rules (see Replay
   Contract).
6. Skip actor/critic updates until the active partition has at least
   `batch_size` transitions.
7. Re-attach the teacher policy with new metadata if residual or BC still
   needs it (see Warm-Start and Residual sections below).

Topology signature is a stable hash of per-building
`(building_id, observation_names, action_names, ca_action_names,
per_type_feature_dims)`. Including building ids and feature dimensions
prevents accidentally reusing incompatible replay or actor state when
buildings are reordered or schema drifts within a stable name set.

## Warm-Start Policy Lifecycle

The warm-start teacher has **two independent roles**:

1. **Exploration replacement/blending** during the early training window.
2. **Residual baseline provider** for the residual policy composition.

These roles have different lifetimes:

### Exploration role (finite)

- Steps `[0, random_exploration_steps)`: teacher provides all actions.
- Steps `[random_exploration_steps,
  random_exploration_steps + warm_start_policy_phaseout_steps)`: teacher
  contribution decays linearly via the existing phaseout probability.
- After the phaseout window: teacher no longer participates in exploration
  blending or action replacement.

### Residual baseline role (indefinite while residual is enabled)

When `residual_policy_enabled` is true, the teacher continues to provide
base actions for the residual composition formula at every step,
independently of the phaseout counter. The teacher policy object remains
alive and attached for as long as residual mode is active.

This matches current MATD3 behavior where
`_current_residual_base_actions()` calls the teacher at every step
regardless of `exploration_step` (`maddpg_agent.py:3034-3044`).

### BC role (finite, scheduled by BC weight decay)

When BC is enabled, the teacher provides target actions for the BC loss
term. This continues as long as `bc_effective_weight > 0`. The teacher
remains alive until both the BC weight has decayed to zero and the residual
role no longer needs it.

### Teacher release rule

The teacher policy is released (set to None, state freed) only when **all
three roles** are inactive: exploration phaseout has ended, residual policy
is disabled or weight is zero, and BC effective weight has decayed to zero.
If any role still needs the teacher, it stays attached.

### Topology-change rules

1. `exploration_step`, `random_exploration_steps`, and
   `warm_start_policy_phaseout_steps` are preserved across topology changes.
   They are never reset.
2. If the teacher is still alive (any role active), re-attach it with new
   `observation_names`/`action_names`/`entity_specs` so it can produce valid
   actions for the new layout. This is a state-refresh, not a schedule
   reset.
3. If the teacher has already been released, it is not re-attached.
4. Teacher-action entries in replay are stored alongside transitions; on
   topology change they are cleared only for the affected buildings' new
   active partition.

## Target Policy Smoothing

Applied to target actor outputs during critic-target computation, in the
**final action space** (post-residual-composition):

1. Compute target actor output for each CA token.
2. If residual is enabled, compose with the teacher base action to get the
   final target action (same formula as inference residual composition).
3. Add Gaussian noise scaled by `target_policy_noise * action_span`,
   clipped elementwise to `[-target_policy_noise_clip * action_span,
   +target_policy_noise_clip * action_span]`.
4. Clip the smoothed target action to action-space bounds (currently
   `[-1, 1]`).

This matches current MATD3 behavior (`maddpg_agent.py:3311-3324`) where
smoothing is applied after full policy composition and clipping, not to the
raw actor latent.

## Residual Policy Composition

Residual composition is enabled only when a warm-start teacher policy is
configured and `residual_policy_enabled` is true. The final action for CA
token `k` on building `b` is:

`action[b, k] = clip(teacher_action[b, k] + 0.5 * action_span[k] *
residual_action_scale * scale_mask[k] * actor_output[b, k],
action_low[k], action_high[k])`

- `actor_output[b, k]` is the tanh-squashed output of the local Transformer
  actor (range `[-1, 1]`).
- `action_span[k] = action_high[k] - action_low[k]` (currently 2.0 for
  `[-1, 1]` bounds).
- `residual_action_scale` follows the existing MATD3 growth schedule.
- `scale_mask[k]` applies per-CA-type multipliers
  (`residual_storage_action_scale_multiplier`,
  `residual_ev_action_scale_multiplier`, etc.) keyed by CA token type.
- Exploration noise is applied to the composed action before the final clip,
  matching current MATD3 behavior.
- When residual is disabled or no teacher is configured, the actor output is
  scaled to action bounds directly: `action[b, k] =
  action_low[k] + 0.5 * (actor_output[b, k] + 1) * action_span[k]`.

## Behavior Cloning

BC for `AgentTransformerMATD3` is **replay-native**. Unlike the on-policy
`BehaviorCloningRegularizer` used by `AgentTransformerPPO` (which indexes
teacher actions by rollout step), MATD3 BC stores teacher/base actions
directly in the replay buffer alongside each transition and reads them back
at sample time.

Implementation approach:

- **Shared utilities** extracted from `BehaviorCloningRegularizer`:
  - Teacher policy building and attach lifecycle (`build_warm_start_policy`).
  - CA-type weight computation (`ca_type_weights` logic).
  - BC effective-weight decay schedule.
- **MATD3-specific BC path**:
  - Teacher actions are stored in replay per transition (no separate buffer).
  - At actor-update time, sampled `teacher_actions` come from the replay
    batch directly.
  - BC loss: weighted MSE between actor-predicted actions and teacher actions
    from replay, weighted per CA token type (EV, storage, other).
  - BC weight decay uses `global_learning_step` and the same
    `decay_start_step`/`decay_steps`/`min_weight` schedule.

- Teacher-action entries in replay use the new topology signature; on
  topology change, affected buildings' entries are invalidated in the old
  partition.
- If BC is enabled, `set_observation_context` and `set_transition_context`
  are implemented so the wrapper can provide raw observations for teacher
  action computation.

## Critic Design

Actors stay local and deployable. Critics are centralized and training-only.

### Token packing

- Input to each critic is a single global token sequence covering every
  building's observation tokens plus action tokens for each CA token. Each
  token carries type-family, per-type, and per-building identity embeddings.
  Padding masks handle variable per-building token counts.

### Action token content

When `critic_action_input_mode` is `final` (default): one action token per
CA token containing the final composed action scalar.

When `critic_action_input_mode` is `final_base_delta` or
`final_base_delta_normalized`: each CA action token carries three scalars —
the final action, the teacher/base action, and the residual delta (optionally
normalized by action span). This matches the current MATD3 15-min residual
recipe (`maddpg_agent.py:2996-3032`).

### Twin critic stacks

Two **fully independent** Transformer critic stacks process the global
sequence. Each stack has:

- Its own Transformer encoder (backbone, type embeddings, layer weights).
- Its own Q head (per-building query token → scalar Q-value per controlled
  building).
- Its own target critic (soft-updated independently).

Independence is required for TD3's overestimation reduction. A shared
backbone makes the twin critics correlated and is explicitly listed as a
future performance optimization in `Look If We Need Improvements`.

### Actor update gradient path

The per-building actor update at building `b` recomputes only building `b`'s
actor output; all other buildings' actions in the global sequence are
detached. Critic 1 parameters are frozen (excluded from gradient
accumulation) during actor loss backward. The actor loss is
`-Q1(s, a_with_b_replaced_by_current_actor)`.

Actor updates iterate over controlled buildings inside a single delayed-
actor-update step. Compute cost is
`O(controlled_buildings * critic_forward)` per actor update; this is an
explicit performance-vs-fidelity choice and is noted in
`Look If We Need Improvements`.

## Replay Contract

Each replay transition stores:

- Encoded per-building observations.
- Encoded per-building next observations.
- Final actions sent to the wrapper.
- Teacher/base actions (for residual/BC; populated via
  `set_transition_context`).
- Next-state teacher/base actions (for target residual composition).
- Per-building rewards.
- Done flag.
- Topology signature.
- Layout summaries needed to reconstruct token batches for that signature.

### Sampling rules

- The sampler returns batches only from the current active topology
  signature.
- If `len(active_partition) < batch_size`, no update is performed this step.
- Old partitions are retained read-only for diagnostics and checkpoint
  continuity but do not contribute gradients.

### Capacity and eviction

A single `replay_capacity` config parameter defines the maximum total
transitions across all partitions combined. When a new transition would
exceed capacity, the oldest transition from the oldest non-active partition
is evicted first; if all non-active partitions are empty, the oldest
transition from the active partition is overwritten (standard ring-buffer
semantics within a partition). Per-partition capacity is not fixed; only the
global total is bounded.

Replay checkpoint state includes all retained partitions (subject to
capacity) plus the active signature.

## Initial Exploration Gating

`is_initial_exploration_done(global_learning_step)` returns True when
`global_learning_step >= end_initial_exploration_time_step` (matching current
MATD3 behavior in `maddpg_agent.py:2028-2029`).

When `initial_exploration_done` is False:

- If `train_during_initial_exploration` is False (default): skip all updates.
- If `train_during_initial_exploration` is True and
  `global_learning_step >= initial_exploration_training_start_step`: allow
  updates.

This gates both critic and actor updates, matching the existing MATD3
contract.

## Diagnostics

Essential parity with current MATD3 diagnostics:

- Replay size, active partition size, partition count, topology-signature
  changes.
- Critic 1 / critic 2 loss, critic Q gap, target Q distribution stats.
- Actor loss, gradient norms, per-building action deviation from teacher.
- Warm-start exploration phaseout probability, residual scale, whether
  teacher is still alive.
- BC loss by CA token type (EV, storage, other), BC effective weight.
- Residual delta magnitude by CA token type.
- Critic action-input mode and whether delta normalization is active.

Metric namespace is `TransformerMATD3/...`. Metrics reference building
identifiers, not fixed agent indices, so they stay meaningful across
topology changes.

## Configuration

Add `TransformerMATD3StageConfig` in `utils/config_schema.py`, alongside
`TransformerPPOStageConfig`. Required sub-blocks:

- `tokenizer_config_path`.
- `transformer_actor` (d_model, nhead, num_layers, dim_feedforward, dropout).
- `transformer_critic` (d_model, nhead, num_layers, dim_feedforward,
  dropout).
- `hyperparameters`: `gamma`, `tau`, `batch_size`, `replay_capacity`,
  `actor_lr`, `critic_lr`, `target_policy_noise`,
  `target_policy_noise_clip`, `actor_update_interval`,
  `critic_action_input_mode`, reward normalization settings.
- `exploration`: existing MATD3 exploration structure, including
  `random_exploration_steps`, `end_initial_exploration_time_step`,
  `train_during_initial_exploration`, and `warm_start_policy` sub-block.
- `residual`: existing MATD3 residual policy structure.
- `behavior_cloning`: weight, min_weight, decay_start_step, decay_steps,
  ev_multiplier, storage_multiplier, warm_start sub-block.
- `diagnostics`: toggles and detail levels.

The class declares `supports_dynamic_topology = True`. Config and runtime
guardrails accept `simulator.topology_mode: dynamic` for
`AgentTransformerMATD3`. The existing dynamic-topology error message for
legacy `MATD3` is unchanged; a new allow-list entry is added for
`AgentTransformerMATD3` only.

## Checkpoint And Export

Checkpoints include:

- Actor, target actor weights per building.
- Critic 1 and critic 2 full stacks (backbone + Q head) and their targets.
- Actor and critic optimizer state.
- Replay partitions and active topology signature.
- Reward normalization state (per building).
- Exploration state (`exploration_step`, `sigma`, teacher activity flags).
- Teacher/residual/BC state (teacher policy weights if stateful, phaseout
  step, BC weight schedule state).
- RNG state.
- Per-building layout signatures and topology versions.

Load behavior:

- Fail fast if the number of buildings changes between saved and current
  environments.
- Fail fast if a token type's feature count changes.
- Otherwise, restore weights and replay partitions and continue.

Export writes only per-building actor ONNX artifacts. Export metadata
includes:

- Building index and building id.
- Actor artifact path.
- Topology version.
- Observation dimension.
- SRO and CA type lists.
- CA action names.
- Action bounds (low, high per CA action).
- Tokenizer config path.
- Dynamic-topology support flag.

`artifact_manifest.json` contains only the actor artifacts above. Critics,
replay, teacher/residual/BC state, and target networks are not in the
manifest.

## Tests And Validation

Unit tests should cover:

- Registry and schema support for `AgentTransformerMATD3`.
- First attach builds layouts and model stacks.
- Repeated attach with identical `observation_names`/`action_names` is a
  no-op.
- Topology-change attach rebuilds changed layouts and global critic packing.
- Building-count change on attach or checkpoint load fails fast.
- Actor output count equals `layout.n_ca`.
- Actor output order matches `layout.ca_action_names`.
- Twin critics are fully independent (no shared parameters between critic 1
  and critic 2).
- Global critic token packing handles variable buildings/assets and masks.
- Critic action tokens carry final/base/delta when configured.
- Replay partitions transitions by topology signature.
- Sampler only returns active-signature transitions.
- Replay eviction respects global capacity across partitions.
- Updates skipped when `len(active_partition) < batch_size`.
- Updates skipped when `initial_exploration_done` is false and
  `train_during_initial_exploration` is disabled.
- Target policy smoothing applies in final action space after residual
  composition.
- Delayed actor updates respect the configured interval.
- Critic parameters are frozen during actor loss backward.
- Teacher stays alive while residual or BC still needs it, even after
  exploration phaseout ends.
- Teacher is released only when all three roles are inactive.
- Teacher is re-attached on topology change only if still alive.
- Warm-start exploration phaseout counter is preserved across topology
  changes.
- Residual composition uses correct action-span scaling formula.
- BC loss reads teacher actions from replay samples (not a rollout buffer).
- BC per-CA-type weighting uses layout token types.
- `set_observation_context` and `set_transition_context` are honored when
  BC/residual are enabled.
- Checkpoint round trip preserves weights, replay active signature, topology
  metadata, and teacher/BC state.
- Checkpoint load rejects incompatible feature schemas and building-count
  changes.
- Export writes actor ONNX artifacts only.
- Export metadata contains layout/action data and action bounds.
- Manifest excludes critic/replay/teacher state.

Integration tests should mirror the dynamic entity smoke coverage used for
`AgentTransformerPPO`, with additional MATD3 assertions:

- An `observation_names` change switches active replay signature.
- Actor and critic target networks survive asset-count changes when feature
  dimensions are stable.
- Critic artifacts are absent from export output.
- Per-building actor artifacts remain independently deployable.
- Warm-start exploration phaseout continues correctly across an in-window
  topology change.
- Teacher residual baseline continues correctly after exploration phaseout
  ends.
- Building-count change fails fast in both attach and checkpoint-load paths.

## Look If We Need Improvements

The following are known limitations or tuning knobs, not part of the initial
scope. Each should be revisited only if verification passes but learning
quality or performance is not good enough.

Design-preserving tuning knobs:

- Preserve per-building actor weights when a building temporarily
  disappears and later returns (cache-and-restore instead of full rebuild).
- Cross-topology replay with masks instead of current-signature-only
  sampling (removes the post-mutation update pause).
- Alternative global critic pooling strategies (asset-level attention with
  building-level summaries).
- Separate action-token encoders per CA type.
- Conservative Q regularization.
- Target critic update cadence tuning.
- Teacher phaseout and residual schedule tuning.
- BC multiplier tuning by asset type.
- Richer diagnostics for attention maps and action-token attribution.
- Joint per-step actor update (single global backward touching all actors)
  instead of the current per-building loop, if the per-building loop
  becomes a training-throughput bottleneck.
- Shared Transformer backbone between twin critics (collapses independence
  but reduces VRAM and compute; measure overestimation bias impact before
  adopting).

Redesign-level changes (would require revisiting the deployment boundary or
per-building actor model):

- Building-count changes at runtime and at checkpoint load (currently
  fail-fast).
- Shared actor weights across buildings with per-building adapters
  (collapses the per-building weight boundary that the deployment story is
  built on).
