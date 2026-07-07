# AgentTransformerMATD3 Design

## Objective

Create `AgentTransformerMATD3`: a production-strength MATD3 agent that uses
entity-token Transformer actors and centralized Transformer critics. The goal is
to keep the strongest parts of the existing MATD3 recipe while gaining the
dynamic-topology flexibility already proven by `AgentTransformerPPO`.

The agent must support the `entity` interface and dynamic topology from day one.
It should handle assets entering or leaving mid-episode through
`topology_version`-driven layout rebuilds, without relying on flat-vector
sentinels or fixed action dimensions.

## Core Principles

- Keep deployment local: each building exports its own actor model and tokenizer
  layout metadata.
- Keep centralized training: twin critics may consume community-wide state and
  action context during learning.
- Do not export critics. Critics, replay buffers, teacher actions, residual
  policy state, and BC machinery are training-only.
- Reuse Transformer/entity concepts from `AgentTransformerPPO`, but keep shared
  Transformer-domain code in neutral modules rather than PPO-specific modules.
- Preserve MATD3 strengths: twin critics, target policy smoothing, delayed actor
  updates, warm-start policy, residual-over-teacher policy, behavior cloning,
  reward normalization, diagnostics, and checkpoint/resume.
- Use layout/action metadata for all asset-type behavior. Avoid fixed flat-index
  assumptions for EV/storage-specific losses, residual scales, or diagnostics.

## Architecture

`AgentTransformerMATD3` is a new standalone `BaseAgent` implementation. It is
registered as `AgentTransformerMATD3` beside `AgentTransformerPPO` and `MATD3`.
It does not inherit from `MATD3` or `MADDPG` because those classes assume fixed
flat dimensions across replay, action processing, export, and diagnostics.

Each building owns a deployable actor stack:

- `EntityTokenLayoutBuilder` output for the building.
- Entity observation tokenizer.
- Transformer backbone.
- Deterministic actor head that emits one scalar per CA token.
- Target actor stack.
- Actor optimizer.
- Per-building topology version and layout signature.

Training owns centralized, non-deployable critic state:

- Global critic token packer for all building observations and actions.
- Twin Transformer critics and twin target critics.
- Q heads that output one scalar Q-value per controlled building.
- Critic optimizers.
- Replay partitions keyed by topology signature.

The exported artifact boundary is hard: only actor pipelines are exported to
ONNX, one per building. The export metadata includes enough layout/action
information for serving to map CA token outputs back to action names.

## Package Boundaries

Shared Transformer/entity pieces should live in a neutral Transformer-domain
package, not in `AgentTransformerPPO` and not as unrelated catch-all utilities.
The implementation should review the current modules:

- `algorithms/utils/entity_token_layout.py`
- `algorithms/utils/entity_observation_tokenizer.py`
- `algorithms/utils/transformer_backbone.py`
- `utils/entity_tokenizer_schema.py`
- `configs/tokenizers/entity_default.json`

Implementation should decide whether to move these modules into a clearer
neutral package such as `algorithms/entity_transformers/` before adding
`AgentTransformerMATD3`. If moving them would create unnecessary churn, the
first implementation may keep file paths stable, but the ownership must still be
treated as Transformer-domain shared code and no PPO-specific assumptions may be
added to those modules.

Algorithm-specific code stays separate:

- PPO-only logic remains with `agent_transformer_ppo.py` and PPO-specific
  components.
- MATD3-only logic goes into `agent_transformer_matd3.py` and supporting
  MATD3-specific modules, such as deterministic actor heads, action-token
  encoders, global critic packing, twin critic heads, replay partitioning, and
  MATD3 diagnostics.

## Data Flow

The wrapper remains the owner of the simulator entity contract. It converts the
simulator payload into per-building encoded vectors and per-building action
names. `AgentTransformerMATD3` receives those vectors through the existing
`BaseAgent` methods and returns `List[List[float]]` actions in the same contract
used by `AgentTransformerPPO`.

Prediction flow per building:

1. Tokenize the encoded building observation with the cached building layout.
2. Run the local Transformer actor backbone.
3. Emit deterministic actor means in `[-1, 1]`, one scalar per CA token.
4. Apply training-time exploration, warm-start phaseout, and residual composition
   when `deterministic=False`.
5. Return actions ordered by `layout.ca_action_names`.

Training flow:

1. Store transition data with topology signature and layout summaries.
2. Sample only from the active topology-signature replay partition.
3. Build global observation/action tokens for all buildings.
4. Compute target actions from target actors and apply target policy smoothing
   per CA action.
5. Train twin centralized critics with the min-Q TD3 target.
6. On delayed actor-update steps, update each local actor through the centralized
   critic objective plus residual/BC regularization terms.
7. Soft-update actor and critic targets on scheduled target-update steps.

## Dynamic Topology Lifecycle

`AgentTransformerMATD3` follows the same wrapper-driven lifecycle as
`AgentTransformerPPO`:

- First `attach_environment(...)` builds all actor layouts, actor stacks, critic
  packing metadata, replay partitions, and teacher/residual/BC metadata.
- A repeated attach with identical `observation_names` and `action_names` is a
  no-op.
- A topology change is detected by changed per-building observation/action
  names after the wrapper observes a `topology_version` increment.

On topology change:

1. Rebuild changed building layouts from the new observation/action names.
2. Recompute global critic packing metadata and masks.
3. Validate tokenizer coverage, CA action ordering, and feature-count stability.
4. Preserve actor/critic weights and target weights when token type feature
   dimensions are unchanged.
5. Fail fast if a token type's feature count changes in a way that invalidates
   existing projection weights.
6. Switch active replay sampling to the new topology signature.

Replay is partitioned by topology signature for correctness. Old partitions may
remain for diagnostics and checkpoint continuity, but the first implementation
trains only from the current active signature. Cross-topology replay with masks
is a possible improvement, not required for the initial design.

## MATD3 Algorithm Scope

The final target is a production-strength Transformer version of the current
best MATD3, not a minimal ablation. Required behavior includes:

- Deterministic local actors.
- Twin centralized critics.
- Min-Q target computation.
- Target policy smoothing.
- Delayed actor updates.
- Soft target updates.
- Replay buffer with topology partitions.
- Reward normalization.
- Exploration noise.
- Warm-start teacher policy support.
- Residual-over-teacher action composition.
- Behavior cloning regularization and offline BC pretraining.
- Checkpoint/resume.
- Essential MATD3 diagnostics.

Existing proven MATD3 logic can be ported from `maddpg_agent.py` and
`matd3_agent.py`, but it should be adapted into smaller Transformer-specific
components rather than inherited wholesale. Any behavior that currently depends
on flat action indices must be rewritten against `BuildingTokenLayout`,
`ca_action_names`, or CA token types.

## Critic Design

Actors stay local and deployable. Critics are centralized and training-only.

The critic input should be a global token sequence containing:

- Observation tokens from every building.
- Action tokens aligned to CA tokens.
- Building identity embeddings.
- Token-family/type embeddings.
- Masks for variable token counts.

Twin critic stacks process this global sequence and produce one scalar Q-value
per controlled building, matching MATD3's per-agent critic objective while using
shared community context. Target critics use the same packing path with target
actor actions and target policy smoothing. This design preserves centralized-
training expressiveness while keeping deployment isolated to local actors.

## Replay Contract

Each replay transition stores:

- Encoded per-building observations.
- Encoded per-building next observations.
- Final actions sent to the wrapper.
- Teacher/base actions when warm-start or residual behavior is active.
- Per-building rewards.
- Done flag.
- Topology signature.
- Layout/action summaries needed to reconstruct token batches.

The sampler returns batches only from the current active topology signature.
This avoids mixing incompatible observation/action dimensions after topology
mutation. Replay checkpoint state includes all partitions plus the active
signature, so resumed training can continue from the correct partition.

## Configuration

Add a new schema stage for `AgentTransformerMATD3`. It should include:

- `tokenizer_config_path`.
- Local actor Transformer config.
- Global critic Transformer config.
- Actor/critic learning rates.
- TD3 hyperparameters: `gamma`, `tau`, target smoothing noise/clip, delayed actor
  update interval, batch size, replay capacity, reward normalization settings.
- Warm-start policy configuration.
- Residual policy configuration.
- Behavior cloning configuration.
- Diagnostics toggles.

Dynamic topology is supported by this agent. The class declares
`supports_dynamic_topology = True`, and config/runtime guardrails should accept
`simulator.topology_mode: dynamic` for `AgentTransformerMATD3`.

## Checkpoint And Export

Checkpoints include:

- Actor, target actor, critic, target critic weights.
- Actor and critic optimizer state.
- Replay partitions and active topology signature.
- Reward normalization state.
- Exploration state.
- Teacher/residual/BC state.
- RNG state.
- Per-building layout signatures and topology versions.

Loading a checkpoint rejects incompatible token feature schemas. Resume across
asset-count changes is allowed only when the active layout can be rebuilt from
the current environment and all token type feature dimensions match the saved
weights.

Export writes only per-building actor ONNX artifacts. Export metadata includes:

- Building index and building id.
- Actor artifact path.
- Topology version.
- Observation dimension.
- SRO and CA type lists.
- CA action names.
- Tokenizer config path.
- Dynamic-topology support flag.

Critics are not exported.

## Tests And Validation

Unit tests should cover:

- Registry and schema support for `AgentTransformerMATD3`.
- First attach builds layouts and model stacks.
- Repeated attach with identical names is a no-op.
- Topology-change attach rebuilds changed layouts and global critic packing.
- Actor output count equals `layout.n_ca`.
- Actor output order matches `layout.ca_action_names`.
- Global critic token packing handles variable buildings/assets and masks.
- Replay partitions transitions by topology signature.
- Sampler only returns active-signature transitions.
- Target policy smoothing applies per CA action and clips correctly.
- Delayed actor updates respect the configured interval.
- Warm-start, residual, and BC behavior use CA token type/action metadata.
- Checkpoint round trip preserves weights, replay active signature, and topology
  metadata.
- Checkpoint load rejects incompatible feature schemas.
- Export writes actor ONNX artifacts only.
- Export metadata contains layout/action data required for serving.

Integration tests should mirror the dynamic entity smoke coverage used for
`AgentTransformerPPO`, with additional MATD3 assertions:

- A `topology_version` increment switches active replay signature.
- Actor and critic target networks survive asset-count changes when feature
  dimensions are stable.
- Critic artifacts are absent from export output.
- Per-building actor artifacts remain independently deployable.

## Look If We Need Improvements

If the first correct implementation trains poorly, investigate these options one
at a time:

- Cross-topology replay with masks instead of current-signature-only sampling.
- Shared actor weights across buildings with per-building adapters.
- Alternative global critic pooling strategies.
- Separate action-token encoders per CA type.
- Conservative Q regularization.
- Target critic update cadence tuning.
- Teacher phaseout and residual schedule tuning.
- BC multiplier tuning by asset type.
- Richer diagnostics for attention maps and action-token attribution.
- Critic architecture variants that combine asset-level attention with
  building-level summaries.

These are not required for the initial implementation. They are explicit
follow-up candidates if verification passes but learning quality is not good
enough.
