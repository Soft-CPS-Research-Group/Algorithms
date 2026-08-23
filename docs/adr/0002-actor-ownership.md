# ADR-0002 — Actor ownership model

Status: accepted
Date: 2026-08-18
Depends on: ADR-0001
Related: ADR-0003, ADR-0004, ADR-0011

## Context

The actor turns per-building encoded observations into a bounded action
per CA token. Three ownership models were considered:

- A. One `(tokenizer, backbone, actor MLP)` triple per building.
- B. Fully shared actor across all buildings.
- C. Shared tokenizer + backbone, per-building action MLPs.

B and C require backbone upgrades that the shared package does not
provide (`algorithms/transformer_shared/transformer_backbone.py` has no positional encoding
and no `src_key_padding_mask`). Every MATD3 per-building auxiliary
(action bounds, safety adapter, price adapter, warm-start base
policy) already operates per building.

## Plain-language

Analogy: one team per building vs one company-wide team. Per-building
teams keep per-building context intact and match the ownership TPPO
already uses. Company-wide teams require new coordination
infrastructure (positional labels + attention masks) that we do not
have.

## Decision

Option A: per-building stack. Each building owns its own tokenizer +
backbone + actor MLP. Weights are shared per-type inside a building
(unchanged from TPPO). Weights are not shared across buildings.

Cross-building generalization is not a v1 requirement.

## Consequences

- `AgentTransformerMATD3` owns a list of per-building actor stacks through its
  `_PerBuildingState` records.
- ONNX export is per-building, per-topology (see ADR-0011).
- Backbone upgrades are not required for v1 (see ADR-0004).
- Building-count change is a full rebuild.

## Evidence

- TPPO and Transformer MATD3 both use per-building state records.
- `EntityObservationTokenizer.projections` shares each projection by type
  inside one building stack.
- `TransformerBackbone` has no positional encoding or attention mask.

## Future improvements

- Promote per-building stacks to a shared-backbone or fully-shared
  actor after backbone upgrades land (positional encoding,
  `src_key_padding_mask`, masked pooling). This is strictly additive
  from the per-building baseline; do it only if cross-building
  generalization is required.
