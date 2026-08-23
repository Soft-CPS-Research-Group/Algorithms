# ADR-0009 — Local price adapter placement

Status: accepted
Date: 2026-08-18
Depends on: ADR-0012
Related: ADR-0011

## Context

`PriceMultiplierObservationAdapter` in
`algorithms/utils/price_multiplier_adapter.py` rewrites four
price feature values in the per-agent encoded observation vector. It
requires:

- 1-D encoded observation vector aligned with `observation_names`.
- Each of the four price feature names appears exactly once.

In entity mode with `minmax_space` encoding, per-building encoded
observation vectors preserve feature order and width under the shared entity
contract.

## Plain-language

A filter middleware that intercepts an observation and edits four
price fields. It is format-sensitive: it must know exactly where the
fields sit. In entity mode with `minmax_space`, positions are stable.

## Decision

Apply the adapter pre-tokenization (P-PRE) — the per-building
encoded observation vector is patched before the tokenizer sees it. This
preserves the established MATD3 adapter placement.

Schema guard (9a): reject the combination
`local_price_conditioning_enabled: true` with any encoding profile
other than `minmax_space`.

## Consequences

- `AgentTransformerMATD3.predict` runs a per-building
  `PriceMultiplierObservationAdapter.transform` before the tokenizer
  forward when `local_price_conditioning_enabled: true`.
- Transition context conditions current and successor encoded observations with
  their matching current and successor price contexts before replay storage.
- The adapter's `require_strict_local=False` mode is used because entity payloads
  may include community features.
- ONNX export gains a runtime-only guard (see ADR-0011): the price
  adapter is external to the graph.

## Evidence

- `PriceMultiplierObservationAdapter` validates unique price-feature positions.
- `EntityContractAdapter` preserves encoded feature order for `minmax_space`.
- Legacy MATD3 applies the same adapter before its policy network.

## Future improvements

- P-POST token-level patching if a non-`minmax_space` profile ever
  needs price conditioning. Requires locating price features in the
  layout metadata.
