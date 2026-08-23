# ADR-0001 — Shared Transformer/entity package extraction

Status: accepted
Date: 2026-08-18
Related: ADR-0007 (BC), ADR-0002..ADR-0012 (depend on shared package availability)

## Context

`AgentTransformerPPO` today owns modules that are algorithm-neutral:
`entity_token_layout.py`, `entity_observation_tokenizer.py`,
`transformer_backbone.py`, and `RunningValueNormalizer` inside
`ppo_components.py`. A new algorithm (Transformer MATD3) needs the same
building blocks. Options: leave modules in place and cross-import, or
extract into a shared package with a documented reuse boundary.

## Plain-language

Analogy: two microservices need the same helper library. Options: (a)
duplicate the code, (b) let service B import from service A, or (c)
extract to a shared library both services depend on. We pick (c) with a
one-time migration and no permanent shims.

## Decision

Extract the algorithm-neutral core into
`algorithms/transformer_shared/`:

- `entity_token_layout.py`
- `entity_observation_tokenizer.py`
- `transformer_backbone.py`
- `value_normalizer.py` (moved from `ppo_components.py`)
- `behavior_cloning.py` (BC-B — see ADR-0007)

Extraction runs in the first PR of the plan. Re-export shims live in
`algorithms/transformer_ppo/` only long enough to keep the extraction
PR small. The MATD3 introduction PR deletes every shim and rewrites
TPPO imports to reference `algorithms/transformer_shared/*` directly.

Test files under `tests/` that today import from
`algorithms/transformer_ppo/*` migrate to
`algorithms/transformer_shared/*` in the MATD3 PR, alongside shim
deletion.

Hard rule: at the end of the plan, `algorithms/transformer_ppo/`
contains no re-export shim files.

## Consequences

- New directory `algorithms/transformer_shared/` with the moved
  modules and an `__init__.py` documenting the algorithm-neutral scope.
- `algorithms/transformer_ppo/ppo_components.py` loses
  `RunningValueNormalizer` (a shim re-export lives at its top during
  the transient period).
- `algorithms/transformer_ppo/behavior_cloning.py` becomes a shim
  during the transient period (see ADR-0007).
- Final state has TPPO importing shared modules directly. Any
  surviving shim after the MATD3 PR merges is a defect.

Implementation status: complete. TPPO and Transformer MATD3 import the shared
package directly. No compatibility shim remains.

## Evidence

- Shared layout, tokenizer, backbone, normalizer, and behavior-cloning modules
  contain no algorithm registry or TPPO policy dependency.
- Import-boundary tests reject cross-algorithm imports and compatibility shims.

## Future improvements

- If a third algorithm consumes the shared package, revisit the shared
  package's public API surface. No changes forecast today.
