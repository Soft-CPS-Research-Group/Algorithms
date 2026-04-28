# WP01 — Port Reusable Sources from `gj/plan-c` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For all WP plans (v2):** Each task that writes production code MUST follow `superpowers:test-driven-development` (red → verify-red → green → verify-green → refactor → commit). Each WP MUST end with `superpowers:requesting-code-review` before merge.

**Goal:** Bring the reusable v1 ML modules (`TransformerBackbone`, PPO components, transformer_ppo helpers) from branch `gj/plan-c@3e9a6737d7306b04f3516738f86a98ea52106ef5` into this branch as **fresh files** (no v1 commit history), retaining only the parts that v2 reuses verbatim.

**Architecture:** This WP is a clean code transplant. We use `git show gj/plan-c:<path>` to extract single files and write them in place. **No entity-interface logic is added in this WP.** This isolates "did the port itself succeed?" from "did the v2 entity wiring succeed?" so later WPs start from a known-good baseline of pure NN building blocks.

**Tech Stack:** Python 3.11, PyTorch (already in repo), pytest.

**Branch:** `gj/wp01-port-plan-c`
**Base branch:** `main`

---

## Scope

**Files ported verbatim from `gj/plan-c@3e9a673` (extract via `git show gj/plan-c:<path>`):**

- `algorithms/utils/transformer_backbone.py` → same path
- `algorithms/utils/ppo_components.py` → same path
- `algorithms/agents/transformer_ppo/__init__.py` → same path
- `algorithms/agents/transformer_ppo/state_helper.py` → same path
- `algorithms/agents/transformer_ppo/update_helper.py` → same path
- `algorithms/agents/transformer_ppo/export_helper.py` → same path

**Tests ported verbatim:**

- `tests/test_transformer_backbone.py`
- `tests/test_ppo_components.py`
- `tests/test_transformer_refactor_helpers.py`

**Explicitly OUT OF SCOPE (these are v1 marker-based and will be rewritten in WP04/WP05):**

- `algorithms/utils/observation_tokenizer.py` (v1 marker tokenizer — will be replaced by v2 `EntityObservationTokenizer`)
- `algorithms/agents/transformer_ppo_agent.py` (v1 agent — will be replaced by v2 `AgentTransformerPPO`)
- `utils/wrapper_transformer/` (v1 marker coordinator — will be replaced by v2 wrapper hook)
- `configs/templates/transformer_ppo.yaml` (v1 template — WP06 will write a v2 one)
- `configs/tokenizers/default.json` (v1 marker config — WP02 will write `entity_default.json`)
- v1 marker-based tests (`test_observation_tokenizer.py`, `test_agent_transformer_ppo.py`, `test_e2e_transformer_ppo.py`, `test_wrapper_transformer.py`, `test_tokenizer_config_schema.py`)

---

## File Structure

After this WP, the repo will contain:

```
algorithms/
  utils/
    transformer_backbone.py    # NEW (from gj/plan-c)
    ppo_components.py          # NEW (from gj/plan-c)
  agents/
    transformer_ppo/           # NEW package (from gj/plan-c)
      __init__.py
      state_helper.py
      update_helper.py
      export_helper.py
tests/
  test_transformer_backbone.py            # NEW
  test_ppo_components.py                  # NEW
  test_transformer_refactor_helpers.py    # NEW
```

No existing files are modified. No registry entry is added (the v2 agent doesn't exist yet).

---

## Tasks

### Task 1: Create branch and verify base

- [ ] **Step 1: Create the branch from main**

```bash
git checkout main
git pull --ff-only
git checkout -b gj/wp01-port-plan-c
```

- [ ] **Step 2: Verify `gj/plan-c` is reachable at the pinned commit**

```bash
git rev-parse gj/plan-c
# Expected: starts with 3e9a673
git cat-file -e 3e9a6737d7306b04f3516738f86a98ea52106ef5 && echo "OK: pinned commit reachable"
```

If the commit is not reachable: `git fetch origin gj/plan-c` then re-verify. If still not reachable, STOP and ask.

- [ ] **Step 3: Confirm target paths do not yet exist**

```bash
test ! -e algorithms/utils/transformer_backbone.py && echo "OK: backbone absent"
test ! -e algorithms/utils/ppo_components.py && echo "OK: ppo_components absent"
test ! -d algorithms/agents/transformer_ppo && echo "OK: transformer_ppo pkg absent"
```

All three must print "OK". If any path exists, STOP and ask.

---

### Task 2: Port `TransformerBackbone`

**Files:**
- Create: `algorithms/utils/transformer_backbone.py`
- Test: `tests/test_transformer_backbone.py`

This task uses **port-then-verify** rather than RED/GREEN, because the source code already exists on `gj/plan-c` and the tests are also being ported verbatim. The TDD discipline here is: **the ported tests must pass against the ported code without modification**. If they don't, the port is broken.

- [ ] **Step 1: Extract the test file first**

```bash
mkdir -p tests
git show gj/plan-c:tests/test_transformer_backbone.py > tests/test_transformer_backbone.py
```

- [ ] **Step 2: Run the test to confirm it fails (module not yet present)**

```bash
pytest tests/test_transformer_backbone.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'algorithms.utils.transformer_backbone'` (or similar import error).

- [ ] **Step 3: Extract the module file**

```bash
mkdir -p algorithms/utils
git show gj/plan-c:algorithms/utils/transformer_backbone.py > algorithms/utils/transformer_backbone.py
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
pytest tests/test_transformer_backbone.py -v
```
Expected: all tests PASS, exit code 0. If any test fails, STOP — the port is incomplete (missing dependency, e.g. `algorithms/utils/__init__.py` or a sibling helper). Investigate which symbol is missing and either:
  - extract it from `gj/plan-c` (if it is a leaf utility), or
  - STOP and ask.

- [ ] **Step 5: Ensure `algorithms/utils/__init__.py` exists (if needed)**

```bash
test -f algorithms/utils/__init__.py || (touch algorithms/utils/__init__.py && echo "Created __init__.py")
```

- [ ] **Step 6: Commit**

```bash
git add algorithms/utils/transformer_backbone.py algorithms/utils/__init__.py tests/test_transformer_backbone.py
git commit -m "feat(wp01): port TransformerBackbone from gj/plan-c"
```

---

### Task 3: Port `ppo_components`

**Files:**
- Create: `algorithms/utils/ppo_components.py`
- Test: `tests/test_ppo_components.py`

- [ ] **Step 1: Extract test file**

```bash
git show gj/plan-c:tests/test_ppo_components.py > tests/test_ppo_components.py
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_ppo_components.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'algorithms.utils.ppo_components'`.

- [ ] **Step 3: Extract module file**

```bash
git show gj/plan-c:algorithms/utils/ppo_components.py > algorithms/utils/ppo_components.py
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_ppo_components.py -v
```
Expected: all PASS, exit 0.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/ppo_components.py tests/test_ppo_components.py
git commit -m "feat(wp01): port PPO components (ActorHead, CriticHead, RolloutBuffer, compute_ppo_loss) from gj/plan-c"
```

---

### Task 4: Port `transformer_ppo` helper package

**Files:**
- Create: `algorithms/agents/transformer_ppo/__init__.py`
- Create: `algorithms/agents/transformer_ppo/state_helper.py`
- Create: `algorithms/agents/transformer_ppo/update_helper.py`
- Create: `algorithms/agents/transformer_ppo/export_helper.py`
- Test: `tests/test_transformer_refactor_helpers.py`

- [ ] **Step 1: Extract test file**

```bash
git show gj/plan-c:tests/test_transformer_refactor_helpers.py > tests/test_transformer_refactor_helpers.py
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_transformer_refactor_helpers.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'algorithms.agents.transformer_ppo'`.

- [ ] **Step 3: Extract the four package files**

```bash
mkdir -p algorithms/agents/transformer_ppo
git show gj/plan-c:algorithms/agents/transformer_ppo/__init__.py    > algorithms/agents/transformer_ppo/__init__.py
git show gj/plan-c:algorithms/agents/transformer_ppo/state_helper.py  > algorithms/agents/transformer_ppo/state_helper.py
git show gj/plan-c:algorithms/agents/transformer_ppo/update_helper.py > algorithms/agents/transformer_ppo/update_helper.py
git show gj/plan-c:algorithms/agents/transformer_ppo/export_helper.py > algorithms/agents/transformer_ppo/export_helper.py
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_transformer_refactor_helpers.py -v
```
Expected: all PASS, exit 0.

If a helper imports from `algorithms.agents.transformer_ppo_agent` (the v1 agent), the import will fail. In that case:
  1. Identify the offending import.
  2. If the import is only used for type hints, replace with a string literal forward-ref or `TYPE_CHECKING` guard.
  3. If the import is used at runtime, STOP and ask — that helper has a hard dependency on v1 and may not be reusable as-is.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/transformer_ppo/ tests/test_transformer_refactor_helpers.py
git commit -m "feat(wp01): port transformer_ppo helper package (state/update/export) from gj/plan-c"
```

---

### Task 5: Full test sweep + lint

- [ ] **Step 1: Run the entire test suite**

```bash
pytest -x -q
```
Expected: all tests PASS, exit 0. The newly ported tests should be picked up; no pre-existing tests should regress.

- [ ] **Step 2: Run pre-existing lints if configured**

```bash
test -f pyproject.toml && grep -q "\[tool.ruff\]" pyproject.toml && ruff check algorithms/utils/transformer_backbone.py algorithms/utils/ppo_components.py algorithms/agents/transformer_ppo/ || echo "No ruff configured — skip"
```

If lints fail on ported code, fix only style issues (imports, line length). Do **not** modify behavior.

- [ ] **Step 3: Commit any lint fixups**

```bash
git status
# If anything modified:
git add -u
git commit -m "chore(wp01): lint fixups on ported sources"
```

---

## Self-Review Checklist (run before requesting code review)

Run each command and confirm the expected output. Mark each box only after seeing the evidence.

- [ ] **Spec coverage:** This WP implements §2.2 of `docs/specv2.md` (the "files that must be brought across" list, restricted to the verbatim-reused subset). Re-read §2.2 and confirm every file you ported is in the list, and every file in §2.2 marked "reuse verbatim" is ported. The marker tokenizer (`observation_tokenizer.py`) and v1 agent (`transformer_ppo_agent.py`) are intentionally OUT OF SCOPE — verify they are NOT in this WP.

  Run: `git diff --name-only main...HEAD`
  Expected: only the 6 production files + 3 test files + (optionally) `algorithms/utils/__init__.py`.

- [ ] **All ported tests pass on this branch:**
  ```bash
  pytest tests/test_transformer_backbone.py tests/test_ppo_components.py tests/test_transformer_refactor_helpers.py -v
  ```
  Expected: 100% PASS.

- [ ] **Full repo test suite still passes:**
  ```bash
  pytest -x -q
  ```
  Expected: exit 0.

- [ ] **No accidental v1-marker code leaked in:** `grep -RIn "marker" algorithms/utils/transformer_backbone.py algorithms/utils/ppo_components.py algorithms/agents/transformer_ppo/ || echo "OK: no marker references"` — references to "marker" anywhere in the ported files are red flags. Either confirm the reference is harmless (e.g. a docstring), or STOP.

- [ ] **No accidental entity logic added:** This WP must not import `entity_adapter`, must not reference `topology_version`, must not introduce `EntityTokenLayoutBuilder` or `EntityObservationTokenizer`. Verify:
  ```bash
  grep -RIn "entity_adapter\|topology_version\|EntityToken\|EntityObservation" algorithms/utils/transformer_backbone.py algorithms/utils/ppo_components.py algorithms/agents/transformer_ppo/ || echo "OK: no entity refs"
  ```

- [ ] **Branch state:** `git log --oneline main..HEAD` shows 3-4 commits, each scoped to one logical port.

If any check fails, fix and re-run before proceeding.

---

## Code Review

After the self-review checklist passes, invoke `superpowers:requesting-code-review` to dispatch a fresh subagent that reviews the diff against this plan and §2.2 of `docs/specv2.md`. Resolve any blocking findings before opening the PR.

---

## PR Description

```markdown
## Summary
Ports the reusable v1 ML modules — `TransformerBackbone`, PPO components, and the `transformer_ppo` helper package — from `gj/plan-c@3e9a673` onto the v2 branch. No v1 entity-interface logic, marker tokenizer, v1 agent, or v1 templates are included; those are deliberately rebuilt in later v2 work packages. This produces a clean baseline of pure NN building blocks that subsequent WPs (WP04 wiring, WP05 agent) compose into the v2 entity-aware pipeline.

## Key Changes
- Add `algorithms/utils/transformer_backbone.py` (verbatim port).
- Add `algorithms/utils/ppo_components.py` (verbatim port: `ActorHead`, `CriticHead`, `RolloutBuffer`, `compute_ppo_loss`).
- Add `algorithms/agents/transformer_ppo/` package with `state_helper`, `update_helper`, `export_helper`.
- Add three accompanying ported test modules; all pass on this branch.
- No modifications to existing files.
- Registry not yet updated — `AgentTransformerPPO` does not exist yet (WP05).
```
