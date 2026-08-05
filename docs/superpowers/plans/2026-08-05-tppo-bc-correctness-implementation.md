# TPPO BC Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align TPPO BC teacher targets with actor tanh outputs and isolate auxiliary samples by building.

**Architecture:** Invert TPPO's existing per-building affine action scaling before storing teacher demonstrations. Make the regularizer sampling API select a building reservoir before filtering compatible layouts.

**Tech Stack:** Python, NumPy, PyTorch, pytest.

## Global Constraints

- Use `.venv/bin/python -m pytest` for all tests.
- Write and run failing regression tests before production changes.
- Make no unrelated refactors.

---

### Task 1: Normalize teacher targets

**Files:**
- Modify: `algorithms/agents/agent_transformer_ppo.py:545-558`
- Test: `tests/test_agent_transformer_ppo_behavior_cloning.py`

**Interfaces:**
- Consumes: `self._action_bounds[building_idx]` as `(low, high)` tensors shaped `[n_ca, 1]`.
- Produces: BC targets in the actor tanh domain.

- [ ] **Step 1: Write a failing regression test**

Create an agent with asymmetric action-space bounds. Run one demonstration update with an environment-space teacher action. Assert the stored target equals `2 * (teacher - low) / (high - low) - 1`, then assert `demonstration_loss` is zero for that tanh prediction.

- [ ] **Step 2: Verify red**

Run: `.venv/bin/python -m pytest tests/test_agent_transformer_ppo_behavior_cloning.py::test_demo_teacher_actions_are_normalized_to_actor_tanh_space -q`

Expected: FAIL because the stored target remains in environment space.

- [ ] **Step 3: Add minimal normalization**

In the demonstration branch of `AgentTransformerPPO.update`, convert each validated `teacher_action` with:

```python
low, high = self._action_bounds[building_idx]
teacher_tanh_action = 2.0 * (
    teacher_action - low.squeeze(-1).detach().cpu().numpy()
) / (high.squeeze(-1).detach().cpu().numpy() - low.squeeze(-1).detach().cpu().numpy()) - 1.0
```

Pass `teacher_tanh_action.tolist()` to `record_demonstration`.

- [ ] **Step 4: Verify green**

Run the Task 1 test and the BC behavior-cloning suite.

### Task 2: Sample only the owning building

**Files:**
- Modify: `algorithms/utils/behavior_cloning.py:229-237`
- Modify: `algorithms/agents/agent_transformer_ppo.py:1655-1666`
- Modify: `tests/test_agent_transformer_ppo_behavior_cloning.py`
- Modify: call sites in `tests/` for the sampling API.

**Interfaces:**
- Produces: `sample_demonstrations(building_idx: int, layout: BuildingTokenLayout, batch_size: int) -> List[Demonstration]`.

- [ ] **Step 1: Write a failing two-building test**

Record distinct targets for two buildings that share a layout. Run the real building-0 auxiliary BC path and assert its sampled demonstration target and BC loss use only building 0's target.

- [ ] **Step 2: Verify red**

Run: `.venv/bin/python -m pytest tests/test_agent_transformer_ppo_behavior_cloning.py::test_auxiliary_bc_uses_only_owning_building_demonstrations -q`

Expected: FAIL because compatible demonstrations are pooled across buildings.

- [ ] **Step 3: Add minimal building filter**

Select compatible demonstrations from `self._demonstrations.get(building_idx, [])` in `sample_demonstrations`. Pass `building_idx` from `_run_auxiliary_bc_update`. Update test call sites.

- [ ] **Step 4: Verify green**

Run the Task 2 test and both focused BC suites.

### Task 3: Commit and publish

**Files:**
- Modify only files from Tasks 1 and 2.

- [ ] **Step 1: Inspect changes**

Run: `git status --short` and `git diff --check`.

- [ ] **Step 2: Commit**

Run: `git add algorithms/agents/agent_transformer_ppo.py algorithms/utils/behavior_cloning.py tests/test_agent_transformer_ppo_behavior_cloning.py tests/test_behavior_cloning_regularizer.py && git commit -m "fix(bc): align teacher targets and building samples"`

- [ ] **Step 3: Push the PR branch**

Run: `git push origin HEAD:gj/tppo_bclonning`
