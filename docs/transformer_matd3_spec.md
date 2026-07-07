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
- Single Transformer critic backbone shared by both twin critics.
- Two independent Q heads (twin critics) producing one scalar Q per
  controlled building.
- Twin target critics with the same architecture.
- Critic optimizer(s).
- Replay partitions keyed by topology signature.

The exported artifact boundary is hard: only per-building actor pipelines are
exported to ONNX. `artifact_manifest.json` contains only actor artifacts.
Critics, replay, and teacher/residual/BC state live in the training checkpoint
file and are never referenced by the manifest.

**Controlled building** is defined here as a building whose current layout has
`n_ca >= 1`. Only controlled buildings produce Q-values, actor updates, and
per-building diagnostics. Uncontrolled buildings still contribute observation
tokens to the critic as community context.

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
   - Optionally compose with the teacher policy per the warm-start/residual
     rules (see below).
   - Add exploration noise; clip to `[-1, 1]`.
5. Return actions ordered by `layout.ca_action_names[b]`.

Training flow at step `t`:

1. Store the transition with the current topology signature and per-building
   layout summaries.
2. If `len(active_partition) < batch_size`, skip the update this step.
3. Otherwise sample from the active-signature partition only.
4. Build global observation/action tokens for all buildings; apply padding
   masks for variable token counts.
5. Compute target actions from target actors and apply target policy
   smoothing (pre-tanh; see below).
6. Update twin critics with the min-Q TD3 target.
7. On delayed-actor-update steps, update each controlled building's actor
   against the global critic, keeping other buildings' actions detached.
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
2. Recompute global critic packing metadata and masks.
3. Validate tokenizer coverage, CA action ordering, and feature-count
   stability. Fail fast if a token type's feature count changes.
4. Preserve actor/critic weights and target weights, because per-type
   projections are shared across instances of the same type.
5. Switch active replay sampling to the new topology signature. Old
   partitions are retained read-only for diagnostics and checkpoint
   continuity.
6. Skip actor/critic updates until the active partition has at least
   `batch_size` transitions. Do not warm-start critic outputs from
   old-signature critic weights.
7. Apply teacher-policy handling per the warm-start rules below.

## Warm-Start Policy Lifecycle

The behavior mirrors the current MATD3/MADDPG warm-start recipe:

- Steps `[0, random_exploration_steps)`: teacher provides all actions.
- Steps `[random_exploration_steps,
  random_exploration_steps + warm_start_policy_phaseout_steps)`: teacher
  contribution decays linearly via the existing phaseout probability.
- After the phaseout window: teacher is not used and its per-building state
  is released.

Topology-change rules:

1. `exploration_step`, `random_exploration_steps`, and
   `warm_start_policy_phaseout_steps` are preserved across topology changes.
   They are never reset.
2. If the topology change occurs while the teacher is still active (before
   the phaseout window ends), the teacher policy is re-attached with the new
   `observation_names`/`action_names`/`entity_specs` so it can produce valid
   actions for the new layout. This is a state-refresh, not a schedule
   reset.
3. If the topology change occurs after the phaseout window has ended, the
   teacher is not re-attached. Its per-building state has already been
   released and it is not used again.
4. Teacher-action buffers used by BC and residual composition are cleared
   for changed buildings on topology change (aligned with how
   `AgentTransformerPPO`'s BC regularizer clears buffers on rebuild).

## Target Policy Smoothing

Applied to target actor outputs during critic-target computation:

1. Compute target actor pre-tanh means for each CA token.
2. Add Gaussian noise with standard deviation `target_policy_noise`, clipped
   elementwise to `[-target_policy_noise_clip, +target_policy_noise_clip]`.
3. Apply tanh.
4. Clip the final smoothed action to `[-1, 1]`.

This matches TD3's convention of adding noise in the pre-squash space.

## Residual Policy Composition

Residual composition is enabled only when a warm-start teacher policy is
configured. The final action for CA token `k` on building `b` is:

`action[b, k] = clip(teacher_action[b, k] + residual_action_scale *
actor_output[b, k], -1, 1)`

- `actor_output[b, k]` is the tanh-squashed output of the local Transformer
  actor.
- `residual_action_scale` follows the existing MATD3 scale/growth schedule
  (per-CA-type multipliers are applied by CA token type).
- Exploration noise is applied to the composed action before the final clip,
  matching current MATD3 behavior.
- When no teacher is configured, the actor output is used directly as the
  final action and residual composition is disabled.

## Behavior Cloning

Reuses the existing `BehaviorCloningRegularizer` used by
`AgentTransformerPPO` (`algorithms/utils/behavior_cloning.py`).

- BC targets are per CA token, keyed by CA token `type_name` from the
  layout. Type-specific multipliers (EV, storage, etc.) are read from the BC
  config exactly as they are today.
- Teacher-action buffers are cleared per building on topology change.
- BC is a training-only loss term added to the actor objective; it produces
  no runtime action modification (residual composition handles that).
- If BC is enabled in config, `set_observation_context` and
  `set_transition_context` are implemented and populated by the wrapper for
  teacher-aware replay.

## Critic Design

Actors stay local and deployable. Critics are centralized and training-only.

- Input to the critic is a single global token sequence covering every
  building's observation tokens plus one action token per CA token. Each
  token carries type-family, per-type, and per-building identity embeddings.
  Padding masks handle variable per-building token counts.
- One shared Transformer critic backbone processes the global sequence. Two
  independent Q heads read the encoded per-building query token to produce
  the twin Q-values `Q1(s, a)`, `Q2(s, a)` for each controlled building.
- Target critics use the same packing path with target actor actions after
  target policy smoothing.
- The per-building actor update at building `b` recomputes only building
  `b`'s actor output; all other buildings' actions in the global sequence
  are detached. The actor loss is
  `-Q1_global(s, a_with_b_replaced_by_current_actor)`.
- Actor updates iterate over controlled buildings inside a single delayed-
  actor-update step. Compute cost is
  `O(controlled_buildings * critic_forward)` per actor update; this is an
  explicit performance-vs-fidelity choice and is noted in
  `Look If We Need Improvements` for revisiting if it becomes a bottleneck.

## Replay Contract

Each replay transition stores:

- Encoded per-building observations.
- Encoded per-building next observations.
- Final actions sent to the wrapper.
- Teacher/base actions when warm-start or residual behavior is active
  (populated via `set_transition_context`).
- Per-building rewards.
- Done flag.
- Topology signature (a stable hash of per-building
  `(observation_names, action_names)`).
- Layout summaries needed to reconstruct token batches for that signature.

Sampling rules:

- The sampler returns batches only from the current active topology
  signature.
- If `len(active_partition) < batch_size`, no update is performed this step.
- Old partitions are retained read-only for diagnostics and checkpoint
  continuity but do not contribute gradients.

Replay checkpoint state includes all retained partitions plus the active
signature, so resumed training can continue from the correct partition.

## Diagnostics

Essential parity with current MATD3 diagnostics:

- Replay size, active partition size, and topology-signature changes.
- Critic 1 / critic 2 loss, critic Q gap, target Q distribution stats.
- Actor loss, gradient norms, per-building action deviation from teacher.
- Warm-start phaseout probability, whether teacher is still active.
- BC loss by CA token type (EV, storage, other).
- Residual delta magnitude by CA token type.

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
  `target_policy_noise_clip`, `actor_update_interval`, reward normalization
  settings.
- `exploration`: existing MATD3 exploration structure, including
  `random_exploration_steps` and `warm_start_policy` sub-block.
- `residual`: existing MATD3 residual policy structure.
- `behavior_cloning`: reuses the Transformer BC config shape.
- `diagnostics`: toggles and detail levels.

The class declares `supports_dynamic_topology = True`. Config and runtime
guardrails accept `simulator.topology_mode: dynamic` for
`AgentTransformerMATD3`. The existing dynamic-topology error message for
legacy `MATD3` is unchanged; a new allow-list entry is added for
`AgentTransformerMATD3` only.

## Checkpoint And Export

Checkpoints include:

- Actor, target actor, critic backbone, twin Q heads, target critics
  weights.
- Actor and critic optimizer state.
- Replay partitions and active topology signature.
- Reward normalization state (per building).
- Exploration state (`exploration_step`, `sigma`, teacher activity).
- Teacher/residual/BC state.
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
  no-op (the agent-visible signal for topology change).
- Topology-change attach rebuilds changed layouts and global critic
  packing.
- Building-count change on attach or checkpoint load fails fast.
- Actor output count equals `layout.n_ca`.
- Actor output order matches `layout.ca_action_names`.
- Global critic token packing handles variable buildings/assets and masks.
- Replay partitions transitions by topology signature.
- Sampler only returns active-signature transitions.
- Updates skipped when `len(active_partition) < batch_size`.
- Target policy smoothing applies pre-tanh with correct clipping.
- Delayed actor updates respect the configured interval.
- Warm-start phaseout counter is preserved across topology changes.
- Teacher is re-attached only while still active; discarded after phaseout.
- Residual composition uses `teacher + residual_action_scale * actor` with
  correct clipping.
- BC uses `BehaviorCloningRegularizer`; teacher buffers are cleared on
  topology change.
- `set_observation_context` and `set_transition_context` are honored when
  BC/residual are enabled.
- Checkpoint round trip preserves weights, replay active signature, and
  topology metadata.
- Checkpoint load rejects incompatible feature schemas and building-count
  changes.
- Export writes actor ONNX artifacts only.
- Export metadata contains layout/action data required for serving.
- Manifest excludes critic/replay/teacher state.

Integration tests should mirror the dynamic entity smoke coverage used for
`AgentTransformerPPO`, with additional MATD3 assertions:

- An `observation_names` change switches active replay signature.
- Actor and critic target networks survive asset-count changes when feature
  dimensions are stable.
- Critic artifacts are absent from export output.
- Per-building actor artifacts remain independently deployable.
- Warm-start phaseout continues correctly across an in-window topology
  change.
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

Redesign-level changes (would require revisiting the deployment boundary or
per-building actor model):

- Building-count changes at runtime and at checkpoint load (currently
  fail-fast).
- Shared actor weights across buildings with per-building adapters
  (collapses the per-building weight boundary that the deployment story is
  built on).
