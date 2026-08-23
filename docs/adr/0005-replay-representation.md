# ADR-0005 — Replay transition representation and layout signature

Status: accepted
Date: 2026-08-18
Depends on: ADR-0002
Related: ADR-0006, ADR-0007, ADR-0008, ADR-0010

## Context

Legacy MATD3 uses `MultiAgentReplayBuffer` with per-agent NumPy arrays
of fixed dimensions. Its shape validation rejects drift, which is the wrong behavior for a
variable-topology agent.

Sub-decisions:

- 5a — transition representation.
- 5b — LayoutSignature contents.
- 5c — validity of transitions from prior layouts.
- 5d — replay behavior on building-count change.

## Plain-language

The replay buffer must retain evidence across topologies without
mixing incompatible shapes into one gradient batch. Analogy: Kafka
topic tagged by Avro schema fingerprint; consumers filter by
fingerprint at read time.

Transferability note: when a new CA instance appears, per-type
tokenizer projections, backbone attention, and the per-CA actor head
are all shared per type. The new asset uses trained weights from step
1. What is missing is *its own* replay evidence until enough
new-topology transitions accumulate. Actor/critic weights do not
degrade during the accumulation window.

## Decisions

### 5a — transition representation: R1

Each transition stores per building:

- `encoded_observation: float32[obs_dim_b]`
- `next_encoded_observation: float32[obs_dim_b]`
- `action: float32[n_ca_b]`
- optional `behavior_actions: float32[n_ca_b]` and
  `next_behavior_actions: float32[n_ca_b]` (lazy; see ADR-0008)
- optional `cloning_actions: float32[n_ca_b]` (lazy; see ADR-0007, 0008)
- `layout_signature: SIG-C tuple` (see 5b)

Community-level per transition:

- `reward: float32[num_agents]` (per-agent scalar)
- `terminated: bool` or `bool[num_agents]`
- `truncated: bool` or `bool[num_agents]`

### 5b — LayoutSignature contents: SIG-C (full + widths)

Per building:

```
(
    n_sro: int,
    n_ca: int,
    ca_action_names: tuple[str, ...],
    segments: tuple[
        tuple[family, type_name, instance_id, feature_names, nfc_expression],
        ...,
    ],
    excluded_feature_names: tuple[str, ...],
    type_feature_widths: tuple[tuple[type_name, in_features], ...],
)
```

`segments` catches semantic drift in ordered feature names and the NFC
expression. `type_feature_widths` catches projection-width drift. SIG-A, the
older TPPO BC signature, does not cover all these cases.

### 5c — historical transitions: KEEP + filter at sample time

Transitions from prior layouts remain in the buffer. The sampler
returns only current-signature transitions to gradient batches
(see ADR-0006). Historical signatures and sizes remain visible to analytics
and future tools. BC-A deliberately samples the current signature only.

Eviction is uniform FIFO across all signature buckets. No preferential
retention of historical.

### 5d — building-count change: RESET-FULL

Building-count change triggers a full agent rebuild under the shared
Transformer entity contract. The replay buffer is
dropped and a new one is instantiated for the new topology.

## Consequences

- Storage layout per transition is variable-width across signatures
  but fixed-width within a signature bucket.
- Batches are formed inside a single signature bucket (ADR-0006).
- Transferable weights carry over topology changes; historical replay
  data remains available but non-mixable.

## Evidence

- `MultiAgentReplayBuffer` validates fixed per-agent dimensions.
- `BehaviorCloningRegularizer` provides the earlier TPPO signature model.
- The shared Transformer entity contract defines building-count changes as a
  full-rebuild boundary.

## Future improvements

- R2 raw-payload replay for encoder-version-independent replay.
  Trade: larger memory and per-batch tokenizer compute.
- 5c PURGE-on-change if buffer memory becomes a binding constraint.
