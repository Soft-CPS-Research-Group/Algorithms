# TransformerPPO Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify TransformerPPO integration by extracting wrapper/agent helper modules without changing spec-defined behavior.

**Architecture:** Keep `Wrapper_CityLearn` and `AgentTransformerPPO` as runtime contract owners, but delegate transformer-specific orchestration and heavy logic into focused helper modules. Apply direct import updates for new helper locations and validate against Transformer-focused tests.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, CityLearn, pytest

---

## Task 1: Add Failing Tests for New Helper Modules

**Files:**
- Create: `tests/test_transformer_refactor_helpers.py`

- [ ] Add tests that import planned helper modules:
  - `utils.wrapper_transformer.transformer_observation_coordinator`
  - `algorithms.agents.transformer_ppo.export_helper`
- [ ] Run `source .venv/bin/activate && pytest tests/test_transformer_refactor_helpers.py -q`
- [ ] Confirm tests fail with `ModuleNotFoundError` before implementation.

## Task 2: Extract Wrapper Transformer Coordinator

**Files:**
- Create: `utils/wrapper_transformer/transformer_observation_coordinator.py`
- Modify: `utils/wrapper_citylearn.py`

- [ ] Implement transformer-specific wrapper helper functions/classes for:
  - configuration/clear of transformer state
  - enricher setup and marker-registry synchronization
  - topology detection/handling
  - enriched encoder rebuild
  - observation value enrichment
- [ ] Update `Wrapper_CityLearn` methods to delegate to helper logic while preserving public/internal method names used by tests.

## Task 3: Extract Agent Update/State/Export Helpers

**Files:**
- Create: `algorithms/agents/transformer_ppo/__init__.py`
- Create: `algorithms/agents/transformer_ppo/state_helper.py`
- Create: `algorithms/agents/transformer_ppo/update_helper.py`
- Create: `algorithms/agents/transformer_ppo/export_helper.py`
- Modify: `algorithms/agents/transformer_ppo_agent.py`

- [ ] Move marker-registry/per-building state operations into `state_helper.py`.
- [ ] Move done-flag normalization + PPO update loop internals into `update_helper.py`.
- [ ] Move dummy observation and ONNX export internals into `export_helper.py`.
- [ ] Keep agent method signatures unchanged, delegating to helpers.

## Task 4: Direct Import Updates and Cleanup

**Files:**
- Modify: `utils/wrapper_citylearn.py`
- Modify: `algorithms/agents/transformer_ppo_agent.py`
- Modify: `tests/test_transformer_refactor_helpers.py` (if needed)

- [ ] Ensure all new helper references use canonical paths.
- [ ] Avoid compatibility shims/re-exports for these internal helper moves.

## Task 5: Validation

**Files:**
- No new files; execute tests

- [ ] Run targeted refactor tests:
  - `source .venv/bin/activate && pytest tests/test_transformer_refactor_helpers.py -q`
- [ ] Run Transformer-focused suite:
  - `source .venv/bin/activate && pytest tests/test_observation_enricher.py tests/test_observation_tokenizer.py tests/test_wrapper_transformer.py tests/test_wrapper_integration_e2e.py tests/test_agent_transformer_ppo.py tests/test_e2e_transformer_ppo.py -q`
- [ ] If failures appear, fix regressions and re-run until green.

## Task 6: Final Verification Snapshot

**Files:**
- No new files

- [ ] Run a final consolidated verification command for all touched Transformer areas.
- [ ] Capture pass/fail summary and changed-file list for handoff.
