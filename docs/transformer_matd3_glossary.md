# Transformer MATD3 — Glossary

Alphabetical. Terms are algorithm-specific unless marked
`(shared contract)`.

## Action injection

The small MLP used by the centralized critic to fold a CA action
scalar into its CA embedding after tokenization. See ADR-0003c.

## Actor-only BC

Any behavior-cloning update path that never modifies critic weights,
critic optimizer state, reward-normalizer statistics, or replay buffer
state. See ADR-0007, hard boundary 1.

## Actor policy loss weight

The non-negative multiplier applied to the MATD3 policy objective before BC
losses are added. It does not change critic targets or executed actions.

## Affine action bounds (shared contract)

The per-CA mapping `low + (tanh(x) + 1) * (high - low) / 2` applied to
the actor's pre-tanh scalar to produce a bounded action. Implemented by
`AgentTransformerMATD3._affine_action`.

## Base action

The warm-start policy's proposed per-CA action, delivered in
`action_names` order. Combined with the actor tweak under the residual
policy composition. See ADR-0008.

## Batching (signature-bucketed)

The batching policy where each gradient step samples from exactly one
LayoutSignature bucket. See ADR-0006.

## BC-A (replay-based behavior cloning)

Actor-only weighted MSE using `cloning_actions` stored per replay
transition. Continuous side loss during MATD3 updates. See ADR-0007.

## BC-B (demonstration-based behavior cloning)

Actor-only pretraining on a dedicated reservoir of `RBCSmartPolicy`
demonstrations collected before RL. Optional auxiliary loss during RL.
See ADR-0007.

## BC hard boundaries

The nine invariants in ADR-0007 §7d that constrain every behavior
cloning implementation.

## Building embedding

The per-building `[d_model]` vector consumed by the Deep Sets
aggregator inside a centralized critic. See ADR-0003b.

## CA (controllable asset) (shared contract)

An asset whose action the controller sets. Current CA types are
`storage` and `charger`. See the shared Transformer entity contract.

## CA-position invariant (shared contract)

The rule that action-vector position `i` refers to the CA token whose
`action_field` matches `action_names[b][i]` for building `b`. Every
controller must preserve it. See the shared Transformer entity contract.

## Centralized critic

A critic that consumes every building's observation and every
building's joint actions to output a single scalar Q. See ADR-0003.

## Checkpoint format 5

The Transformer MATD3 checkpoint schema. See ADR-0010.

## Dedicated schema class

The explicit pydantic model `TransformerMATD3StageConfig` that
replaces dict passthrough for MATD3-specific hyperparameters. See
ADR-0012.

## Deep Sets aggregator

A permutation-invariant aggregator over a set of vectors: `Q_out =
MLP_out(mean_i(MLP_in(x_i)))`. Used by the centralized critic. See
ADR-0003b.

## Deployable ONNX

An exported model whose behavior can be reproduced from the graph
alone. Non-deployable when residual, safety, or price require external
runtime state. See ADR-0011.

## Dynamic-topology capability

The class-level `supports_dynamic_topology = True` flag plus the
`snapshot_topology_state` and `restore_topology_state` hooks
required by the wrapper. See ADR-0012.

## Effective scale

The scheduled residual authority at the current training step,
computed by `AgentTransformerMATD3._residual_action_effective_scale`.
See ADR-0008.

## LayoutSignature (SIG-C)

Per-building tuple identifying a layout for compatibility checks:

```
(n_sro, n_ca, ca_action_names, segments, excluded_feature_names,
 type_feature_widths)
```

See ADR-0005b.

## Local price adapter

Pre-tokenization middleware that rewrites price feature values in the
per-building encoded observation vector using a coordinator-supplied
context. See ADR-0009.

## N-step queue serialization

Persisting the pending discounted-return computation state
(`_n_step_queue`) across checkpoints. See ADR-0010e.

## NFC (non-flexible context) (shared contract)

The single per-building token computed from
`non_shiftable_load − solar_generation`. See the shared Transformer
entity contract.

## Per-building stack

A `(tokenizer, backbone, actor_mlp)` triple owned by one building.
Weights are not shared across stacks. See ADR-0002.

## Per-CA head

The MLP applied token-wise to every CA embedding to produce one scalar
action mean. See ADR-0002.

## Residual authority

The per-CA scheduled multiplier that scales the actor's tweak in the
residual composition. See ADR-0008.

## Shared package

`algorithms/transformer_shared/`. Contains algorithm-neutral
Transformer/entity code shared between TPPO and Transformer MATD3.
See ADR-0001.

## Signature bucket

A subset of the replay buffer containing only transitions with a
specific LayoutSignature. See ADR-0006.

## Signature-STRICT restore

A checkpoint restore path that hard-fails on any per-building
LayoutSignature mismatch. See ADR-0010c.

## SRO (shared read-only) (shared contract)

Context tokens that inform the controller but produce no action.
Examples: district time, weather, pricing, building energy, and PV.
See the shared Transformer entity contract.

## Topology-versioned ONNX

Per-topology ONNX file identified by `topology_v<version>` in the
filename. See ADR-0011a.

## Twin critics (algorithm invariant)

Two independent Q-networks per agent whose minimum forms the target-Q
estimate. TD3 invariant. See ADR-0003a.

## Under-full bucket

A signature bucket with fewer than `batch_size` transitions. Training
step is skipped until it refills. See ADR-0006b.
