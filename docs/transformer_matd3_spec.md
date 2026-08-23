# AgentTransformerMATD3 technical specification

> Status: implemented
>
> Scope: the `AgentTransformerMATD3` implementation at the head of this PR
>
> Maintainers: Algorithms maintainers

This specification defines invariants for maintainers and coding agents. The
[operational guide](transformer_matd3.md) explains configuration and use. The
[ADRs](adr/README.md) preserve decision rationale and rejected alternatives.
The shared [Transformer entity contract](transformer_entity_controller.md)
applies without modification.

Code is authoritative for executable behavior. If code and this specification
diverge, treat the mismatch as a defect and update both in one PR.

## 1. Scope and non-goals

The controller:

- implements off-policy MATD3 over encoded entity observations;
- supports static and dynamic entity topology;
- owns one actor stack and one twin-critic pair per building;
- supports optional residual control, two behavior-cloning paths, local action
  safety, and local price conditioning;
- saves strict format-5 checkpoints;
- exports deterministic per-building ONNX actors.

Current non-goals are cross-building actor weight sharing, mixed-signature
batches, compatible cross-layout checkpoint restore, joint-attention critics,
and ONNX graphs that include runtime-only adapters.

Stable implementation entry points:

- `algorithms/transformer_matd3/agent.py::AgentTransformerMATD3`
- `algorithms/transformer_matd3/components.py`
- `algorithms/transformer_matd3/replay.py::SignatureBucketedReplayBuffer`
- `algorithms/transformer_matd3/types.py::LayoutSignature`
- `utils/config_schema.py::TransformerMATD3StageConfig`

## 2. Required environment contract

The pipeline stage must use `algorithm: AgentTransformerMATD3`, `count: 1`, and
final pipeline position. The simulator must use the entity interface. Dynamic
topology is enabled through `supports_dynamic_topology = True` and transactional
snapshot and restore hooks.

At attachment, the wrapper supplies per-building observation names, action
names, spaces, entity specifications, and building metadata. For building `b`,
action index `i` must always refer to the controllable-asset segment whose
`action_field` equals `action_names[b][i]`.

The tokenizer config partitions encoded features into:

- SRO segments: read-only context tokens;
- one NFC segment: a derived non-flexible-load token;
- CA segments: controllable assets that each produce one action;
- excluded fields: encoded inputs intentionally unused by the model.

## 3. Per-building state

Each building owns an independent online actor stack:

```text
EntityObservationTokenizer
  -> TransformerBackbone
  -> DeterministicActorHead
  -> tanh
  -> affine action bounds
```

It also owns a target copy of the complete actor stack and one actor optimizer.
Actor weights are not shared across buildings. Tokenizer projections and the
actor head are shared across instances of the same entity type within one
building stack.

Each building also owns two independent `CentralizedCritic` instances, their
target copies, and their optimizers. Actor and critic feature stacks do not
share parameters.

Each centralized critic:

1. tokenizes every building with critic-owned projections;
2. processes compatible building shapes in grouped Transformer batches;
3. injects each CA action into its CA embedding through an MLP;
4. combines pooled context and mean action-conditioned CA embeddings;
5. applies a Deep Sets projection and mean over buildings;
6. returns one scalar Q value.

The building mean makes the community aggregation permutation invariant.

## 4. Layout signature and replay

`LayoutSignature` is a tuple of per-building signatures. Each building entry is:

```text
(
  n_sro,
  n_ca,
  ca_action_names,
  segments,
  excluded_feature_names,
  type_feature_widths,
)
```

Each segment signature contains family, type name, optional instance ID,
ordered feature names, and an optional NFC expression. This full signature
guards both tensor shape and feature meaning.

Replay stores encoded current and successor observations, final actions,
per-building rewards, termination flags, truncation flags, and the layout
signature. It allocates optional current and successor base actions only for
residual behavior. It allocates cloning actions only for BC-A.

`SignatureBucketedReplayBuffer` has one global capacity and global FIFO
eviction. It keeps separate buckets per signature. Every learning batch:

- uses the current complete signature;
- contains exactly `batch_size` transitions;
- contains no data from another signature.

An under-full current bucket causes an explicit skipped update. Historical
buckets remain until global FIFO eviction, checkpoint replacement, or a
building-count reset.

## 5. Action paths

### 5.1 Deterministic actor

The actor maps each CA embedding to one scalar, applies `tanh`, then maps
`[-1, 1]` to the attached action bounds. A building with no CA returns an empty
action vector.

### 5.2 Exploration

Before `random_exploration_steps`, prediction samples random bounded actions.
After that boundary, prediction adds configured Gaussian exploration noise to
actor actions and clips to action bounds. Type-specific multipliers may alter
storage and negative EV noise. Exploration sigma decays toward `min_sigma`.

The runner may train only when
`global_learning_step >= end_initial_exploration_time_step`. This learning gate
is independent from the random-action boundary.

### 5.3 Residual composition

When residual control is disabled, the bounded actor output is the final
action. When enabled, the configured warm-start policy supplies a base action
in CA order. The actor supplies a unit correction. The controller scales that
correction by the action span, the scheduled residual authority, and optional
asset-type multipliers, then clips the result to action bounds.

The critic and replay consume final composed actions. Replay also stores current
and successor base actions because target-policy evaluation must reproduce the
same composition.

### 5.4 Runtime adapters

Local price conditioning rewrites a copy of each encoded observation before
tokenization. Current and successor observations use their matching price
contexts. Missing context produces the neutral, unchanged copy. The schema
requires `minmax_space` entity encoding when this path is enabled.

Local action safety consumes raw simulator context and projects the action that
will be executed. It can protect EV minimum or deadline-feasible service,
deferrable must-start behavior, and configured electrical headroom.

Residual composition, price conditioning, and local safety are operational
runtime behavior. They are not part of the actor ONNX graph.

## 6. Learning step

Before replay insertion, the n-step queue computes discounted rewards and done
masks per building. A sampled batch already contains these n-step transitions.
Optional reward normalization applies only to learning targets.

For each actor building `i`:

1. Both target critics evaluate target actions from every building.
2. The target uses the lower of the two Q values.
3. Optional target-policy noise is clipped, added per CA, and action-clipped.
4. Both online critics minimize MSE to the same detached target.
5. On delayed actor steps, actor `i` replaces only its action in the joint
   action list; all other actor actions stay detached.
6. Actor `i` maximizes critic 1's Q. `actor_policy_loss_weight` scales this
   policy term before optional actor-only BC losses are added.
7. Actor and target critic stacks receive soft updates with `tau`.

Critic gradients must not enter actor stacks. One critic's optimizer must not
change the other critic. BC gradients must not enter critics or target stacks.

## 7. Behavior cloning

Both paths default to disabled and own separate actor-only optimizers.

BC-A reads cloning targets from current-signature replay. It supports
`warm_start` and `replay_action`. Configuration rejects the unimplemented
`external` teacher. BC-A can add a weighted actor side loss, scheduled extra
updates, and configured offline pretraining.

BC-B uses `BehaviorCloningRegularizer` and a separate demonstration reservoir
keyed by layout signature. It collects deterministic `RBCSmartPolicy`
demonstrations, pretrains before RL, and can apply an auxiliary actor loss.
Topology attachment rebuilds its teacher. Stored compatible demonstrations keep
their encoded layout representation.

Both paths must satisfy these boundaries:

1. Do not update critics or targets.
2. Do not update reward-normalizer statistics.
3. Do not mutate replay.
4. Short-circuit before actor forward when effective weight is zero.
5. Keep BC-A replay storage and BC-B reservoir storage separate.
6. Rebuild teachers from configuration; never checkpoint teacher objects.
7. Execute no BC code when its path is disabled.
8. Fail before RL when enabled BC-B lacks usable demonstrations.
9. Use separate BC-A and BC-B optimizers.

## 8. Dynamic-topology transaction

`attach_environment` builds candidate layouts and snapshots controller and RNG
state before mutation. Failure restores the full snapshot.

For unchanged building count, the controller permits compatible asset-instance
changes. It preserves neural weights, optimizer state, replay, BC reservoirs,
normalizer state, and clocks. It rejects:

- changed building identity;
- reordered or changed existing segment semantics;
- changed feature names, widths, or NFC expression;
- a type without a compatible existing projection.

A changed building increments its topology version. Before any topology commit,
the controller flushes pending n-step entries as truncated under their stored
old signature.

A building-count change is a full reset. It rebuilds actor and critic state,
replay, BC state, exploration, normalization, and clocks. No learned state is
transferred across this boundary.

## 9. Checkpoint contract

`checkpoint_version` is 5. Only version 5 and algorithm
`AgentTransformerMATD3` are accepted. Checkpoint mode is configured in the
top-level `checkpointing.checkpoint_mode` field.

Full checkpoints contain actor and target stacks, critics and targets, all
optimizers, replay, n-step queue, current signature, topology versions,
exploration state, reward-normalization state, enabled BC state, and Python,
NumPy, PyTorch, and CUDA RNG state.

Inference checkpoints contain actor stacks, signatures, names, bounds,
topology versions, and `exploration_step`. They may load only into a frozen
pipeline stage.

Restore validates the complete payload and prepares replay before mutating live
state. It strictly checks version, algorithm, mode, cardinality, building names,
layout signatures, action names, action bounds, required state keys, n-step
entries, replay state, and enabled BC state. Any apply-time failure rolls back
controller and RNG state.

## 10. ONNX and artifact contract

Export writes one model per building for the current topology using opset 17:

```text
onnx_models/agent_<index>__topology_v<version>.onnx
```

Input is `encoded_obs: [batch, obs_dim]`. Output is
`actions: [batch, n_ca]`. Batch is dynamic; observation and action widths are
fixed for the exported topology.

The graph contains layout slicing, NFC derivation, tokenizer projections,
Transformer backbone, actor head, `tanh`, and affine bounds. It excludes
exploration, target networks, critics, residual composition, local safety, and
price conditioning.

Export rejects enabled runtime-only behavior unless its explicit
`*_runtime_only_export` opt-in is true. Such artifacts set `deployable: false`
and identify every required runtime adapter in manifest metadata.

The returned manifest fragment contains `format`, `artifacts`,
`tokenizer_config_path`, `supports_dynamic_topology`, and `agent_models`.
Per-model metadata includes building identity, topology version, dimensions,
entity types, and CA names. Each artifact's `config` records deployability and
runtime requirements.

## 11. Observability and verification

Training metrics use the `transformer_matd3/` prefix. They report critic and
actor losses, gradient norms, target statistics, replay bucket state, skipped
updates, exploration, BC activity, safety interventions, and optional runtime
profiling.

The focused tests named in the [operational guide](transformer_matd3.md) cover
component shape contracts, critic invariance and independence, replay
validation, BC boundaries, residual actions, topology transactions, checkpoint
atomicity, ONNX parity, schema integration, and templates. The slow end-to-end
test runs the normal entry point through learning, topology mutation,
checkpointing, export, and bundle validation.

## 12. Architecture decision index

| ADR | Locked decision |
|---|---|
| [0001](adr/0001-shared-package-extraction.md) | Shared algorithm-neutral Transformer/entity package |
| [0002](adr/0002-actor-ownership.md) | Independent per-building actor stacks |
| [0003](adr/0003-centralized-twin-critic.md) | Per-building twin critics with Deep Sets community aggregation |
| [0004](adr/0004-backbone-upgrades.md) | No positional or masking upgrades in v1 |
| [0005](adr/0005-replay-representation.md) | Encoded replay with full semantic signatures |
| [0006](adr/0006-batching-policy.md) | Signature-bucketed, homogeneous batches |
| [0007](adr/0007-behavior-cloning.md) | Independent BC-A and BC-B with actor-only boundaries |
| [0008](adr/0008-residual-policy.md) | Post-actor residual composition; critics see final actions |
| [0009](adr/0009-local-price-adapter.md) | Pre-tokenization price conditioning |
| [0010](adr/0010-checkpoint.md) | Strict format-5 full and inference checkpoints |
| [0011](adr/0011-onnx-export.md) | Per-building, per-topology opset-17 actor export |
| [0012](adr/0012-schema-registry-wrapper.md) | Typed schema and capability-based wrapper integration |

## 13. Known limits

- Entity interface only.
- No cross-layout or cross-cardinality checkpoint restore.
- No mixed-signature training batches.
- Current-signature-only BC-A.
- Current-topology-only ONNX export.
- External serving logic required for residual, safety, and price behavior.
- Full learned-state reset when building count changes.
