# ADR-0004 — Backbone upgrades

Status: accepted (no v1 work)
Date: 2026-08-18
Depends on: ADR-0002, ADR-0003

## Context

`TransformerBackbone` in `algorithms/transformer_shared/` has
no positional encoding, no `src_key_padding_mask`, and no masked mean
pool. TPPO copes by keeping one backbone per building and flushing
rollouts on layout change.

Options considered during grilling:

- Add positional encodings, attention masks, and masked pooling now.
- Defer these upgrades until an ADR requires them.

## Decision

No backbone changes in v1. The shared `TransformerBackbone` covers all
needs of `AgentTransformerMATD3` under ADR-0002 (per-building actor
stacks) and ADR-0003 (per-building encode + Deep Sets aggregator).

## Consequences

- Actor forward and critic forward use variable `(N_sro, N_ca)`
  per forward call, with batches homogeneous by LayoutSignature
  (see ADR-0005 and ADR-0006).
- If future work adopts joint-attention critic (S1 from ADR-0003) or
  a shared actor (B or C from ADR-0002), this ADR reopens.

## Evidence

- `TransformerBackbone` exposes no positional or padding-mask input.
- TPPO flushes incompatible on-policy state at layout changes.

## Future improvements

- Add sinusoidal or learned positional encoding to the shared
  backbone.
- Accept `src_key_padding_mask` and use masked mean pool.
- Introduce a building-id embedding so joint sequences distinguish
  identical asset types across buildings.
