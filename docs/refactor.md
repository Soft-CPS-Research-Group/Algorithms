# TransformerPPO Cleanup Refactor

## Context

`gj/plan-c` introduced a complete TransformerPPO pipeline that broadly matches
`docs/spec.md`, but key integration files grew large and now mix multiple
responsibilities:

- `utils/wrapper_citylearn.py`
- `algorithms/agents/transformer_ppo_agent.py`

The main risk is maintainability and safe evolution, not missing functionality.

## Goals

1. Simplify code structure and class responsibilities.
2. Preserve all existing functionality, with `docs/spec.md` as source of truth.
3. Rearrange classes/methods into clearer subdirectories and helper modules.
4. Avoid compatibility shims for internal modules; use direct import updates.

## Non-Goals

- No algorithmic behavior changes.
- No changes to marker semantics, action ordering, topology-change behavior,
  or portability constraints in `ObservationEnricher`.
- No config key or runtime contract changes for `BaseAgent` integration.

## Guardrails

- Keep these invariants unchanged:
  - Marker boundaries are value-based (1000/2000/3000 scheme).
  - Topology changes trigger wrapper handling and agent notification.
  - CA action ordering remains aligned with marker injection order.
  - Enricher remains portable (pure Python, no training-only dependencies).
- Validate with existing Transformer-focused tests and targeted helper tests.

## Two-Phase Plan

### Phase 1: Compatibility-First Extraction

Extract heavy method groups into helper classes while preserving public behavior:

- Wrapper transformer orchestration helpers under `utils/wrapper_transformer/`.
- Agent update/export helpers under `algorithms/agents/transformer_ppo/`.
- Keep `Wrapper_CityLearn` and `AgentTransformerPPO` as orchestration/contract
  owners.

### Phase 2: Direct Rearrangement (No Shims)

- Keep helper classes in their new subdirectories as canonical locations.
- Update imports project-wide to canonical modules.
- Do not add legacy re-exports or shim layers.

## Proposed Structure

- `utils/wrapper_transformer/transformer_observation_coordinator.py`
  - wrapper-side enrichment/topology/encoder-rebuild helper logic.
- `algorithms/agents/transformer_ppo/update_helper.py`
  - done-flag normalization and PPO update loop helper logic.
- `algorithms/agents/transformer_ppo/export_helper.py`
  - ONNX export and dummy-observation construction helper logic.

## Validation Strategy

Primary suite:

- `tests/test_observation_enricher.py`
- `tests/test_observation_tokenizer.py`
- `tests/test_wrapper_transformer.py`
- `tests/test_wrapper_integration_e2e.py`
- `tests/test_agent_transformer_ppo.py`
- `tests/test_e2e_transformer_ppo.py`

Additional targeted tests:

- `tests/test_transformer_refactor_helpers.py`

## Done Criteria

- Existing Transformer behavior is preserved and validated by tests.
- Wrapper/agent files are smaller and delegated to focused helper modules.
- New module layout is in place with direct imports and no compatibility shims.
