# TransformerPPO Behavior Cloning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional Behavior Cloning (BC) and warm-start-action phaseout to `AgentTransformerPPO`, using `RBCCommunityPolicy` as the default teacher, while keeping the existing agent clean and avoiding BC-specific pollution in shared PPO components.

**Architecture:** Implement BC as a `BehaviorCloningRegularizer` collaborator in `algorithms/utils/behavior_cloning.py`. The collaborator owns the teacher policy, teacher-action buffers, BC loss, CA-type weights, weight schedule, phaseout logic, and diagnostics. `AgentTransformerPPO` receives only small lifecycle hook calls, and BC is disabled by omitting the optional `behavior_cloning` config block.

**Tech Stack:** Python, PyTorch, existing entity-interface Transformer PPO modules, existing RBC baseline policies.

---

## File Structure

- Modify: `algorithms/utils/ppo_components.py`
  - Add `Batch.step_indices` as a general-purpose minibatch transition-index field.
- Create: `algorithms/utils/warm_start_policy.py`
  - Shared helper to instantiate and attach warm-start teacher policies.
- Create: `algorithms/utils/behavior_cloning.py`
  - BC collaborator that encapsulates all teacher, phaseout, BC-loss, and diagnostics state.
- Modify: `algorithms/agents/ppo_agents.py`
  - Replace private warm-start-policy construction with the shared helper while preserving behavior.
- Modify: `algorithms/agents/agent_transformer_ppo.py`
  - Add minimal collaborator hooks.
- Modify: `utils/config_schema.py`
  - Add optional `behavior_cloning` schema for `AgentTransformerPPO` pipeline stages.
- Create: `configs/templates/dynamic/transformer_ppo_bc_entity_dynamic.yaml`
  - BC-enabled dynamic entity template.
- Create/modify tests:
  - `tests/test_ppo_components.py`
  - `tests/test_warm_start_policy.py`
  - `tests/test_behavior_cloning_regularizer.py`
  - `tests/test_agent_transformer_ppo_behavior_cloning.py`
  - `tests/test_template_transformer_ppo_bc_entity_dynamic.py`
  - Existing schema and PPO tests as needed.
- Modify docs:
  - `docs/transformer_ppo_spec.md`
  - `AGENTS.md`

---

### Task 1: Worktree Baseline and Plan Artifact

**Files:**
- Create: `docs/superpowers/plans/2026-07-05-tppo-behavior-cloning.md`

- [ ] **Step 1: Verify branch/worktree**

Run: `git branch --show-current && git status --short`

Expected: branch is `gj/tppo_bclonning`; status is clean except the plan artifact being added.

- [ ] **Step 2: Run targeted baseline tests**

Run: `pytest tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_wrapper_integration.py tests/test_ppo_components.py -q`

Expected: all tests pass before feature work begins.

- [ ] **Step 3: Commit plan artifact**

Run: `git add docs/superpowers/plans/2026-07-05-tppo-behavior-cloning.md && git commit -m "docs: plan TransformerPPO behavior cloning"`

---

### Task 2: Add Batch Step Indices

**Files:**
- Modify: `algorithms/utils/ppo_components.py`
- Test: `tests/test_ppo_components.py`

- [ ] **Step 1: Write failing test**

Add a test to `tests/test_ppo_components.py` verifying `RolloutBuffer.get_batches(...)` yields `Batch.step_indices` and that each row maps back to the original stored observation/action.

```python
def test_rollout_buffer_batches_include_original_step_indices():
    import torch
    from algorithms.utils.ppo_components import RolloutBuffer

    buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
    for idx in range(5):
        buffer.add(
            observation=torch.tensor([float(idx)]),
            action=torch.tensor([[float(idx)]]),
            log_prob=torch.tensor([0.0]),
            reward=0.0,
            value=torch.tensor([0.0]),
            done=False,
        )
    buffer.compute_returns_and_advantages(torch.tensor([0.0]))

    seen = []
    for batch in buffer.get_batches(batch_size=2):
        assert batch.step_indices.dtype == torch.long
        assert batch.step_indices.shape[0] == batch.observations.shape[0]
        for row, original_idx in enumerate(batch.step_indices.tolist()):
            assert batch.observations[row, 0].item() == float(original_idx)
            assert batch.actions[row, 0, 0].item() == float(original_idx)
            seen.append(original_idx)

    assert sorted(seen) == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ppo_components.py::test_rollout_buffer_batches_include_original_step_indices -v`

Expected: FAIL because `Batch` has no `step_indices` field.

- [ ] **Step 3: Implement minimal change**

In `algorithms/utils/ppo_components.py`, add `step_indices: torch.Tensor` to the `Batch` dataclass and pass `batch_indices.detach().clone()` in `RolloutBuffer.get_batches(...)`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_ppo_components.py::test_rollout_buffer_batches_include_original_step_indices tests/test_ppo_components.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

Run: `git add algorithms/utils/ppo_components.py tests/test_ppo_components.py && git commit -m "feat(ppo): expose rollout batch step indices"`

---

### Task 3: Shared Warm-Start Policy Helper

**Files:**
- Create: `algorithms/utils/warm_start_policy.py`
- Create: `tests/test_warm_start_policy.py`

- [ ] **Step 1: Write failing tests**

Create tests for:
- Building `RBCCommunityPolicy` by name and attaching it.
- Passing `warm_start_policy_hyperparameters` into the teacher config.
- Unsupported policy names raising a clear `ValueError` listing supported names.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_warm_start_policy.py -v`

Expected: FAIL because `algorithms.utils.warm_start_policy` does not exist.

- [ ] **Step 3: Implement helper**

Create `build_warm_start_policy(...)` with this public shape:

```python
def build_warm_start_policy(
    *,
    owner_name: str,
    policy_name: str,
    policy_hyperparameters: Mapping[str, Any] | None,
    config_template: Mapping[str, Any],
    observation_names: List[List[str]],
    action_names: List[List[str]],
    action_space: List[Any],
    observation_space: List[Any],
    metadata: Optional[Dict[str, Any]],
) -> BaseAgent:
    ...
```

Supported policies: `RuleBasedPolicy`, `RandomPolicy`, `NormalNoBatteryPolicy`, `NormalPolicy`, `RBCBasicPolicy`, `RBCSmartPolicy`, `RBCCommunityPolicy`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_warm_start_policy.py -q`

Expected: all pass.

- [ ] **Step 5: Commit**

Run: `git add algorithms/utils/warm_start_policy.py tests/test_warm_start_policy.py && git commit -m "feat(utils): share warm-start policy builder"`

---

### Task 4: Migrate Existing PPO Warm-Start Initialization

**Files:**
- Modify: `algorithms/agents/ppo_agents.py`
- Test: existing PPO and warm-start policy tests.

- [ ] **Step 1: Write characterization test if missing**

If no existing test covers `_PPOBase._initialize_warm_start_policy`, add one that instantiates `IPPO` or `MAPPO` with `warm_start_policy: RBCSmartPolicy`, calls `attach_environment`, and asserts `_warm_start_policy` is populated.

- [ ] **Step 2: Verify RED only if adding new characterization**

If adding the test before migration, it should pass on current code. This is a characterization refactor exception; do not require failure because behavior already exists.

- [ ] **Step 3: Replace implementation**

In `_PPOBase._initialize_warm_start_policy`, replace the local imports/registry/config-copy/attach logic with `build_warm_start_policy(...)`. Keep method signature and error message owner prefix as `PPO`.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_warm_start_policy.py tests/test_ppo_agents.py -q` if `tests/test_ppo_agents.py` exists; otherwise run the PPO-related tests available in the repo.

- [ ] **Step 5: Commit**

Run: `git add algorithms/agents/ppo_agents.py tests && git commit -m "refactor(ppo): use shared warm-start policy builder"`

---

### Task 5: Config Schema for TransformerPPO BC

**Files:**
- Modify: `utils/config_schema.py`
- Test: schema tests.

- [ ] **Step 1: Write failing tests**

Add tests verifying `TransformerPPOStageConfig` accepts:

```yaml
behavior_cloning:
  enabled: true
  weight: 0.42
  min_weight: 0.24
  decay_start_step: 512
  decay_steps: 3584
  ev_multiplier: 24.0
  storage_multiplier: 0.18
  warm_start:
    policy: RBCCommunityPolicy
    deterministic: true
    noise_scale: 0.0
    phaseout_steps: 6144
    phaseout_mode: blend
    hyperparameters: {}
```

Also add a rejection test for invalid `phaseout_mode`.

- [ ] **Step 2: Verify RED**

Run the schema test selection. Expected failure: extra/unknown `behavior_cloning` field.

- [ ] **Step 3: Implement schema**

Add `TransformerPPOWarmStartConfig` and `TransformerPPOBehaviorCloningConfig`, then add `behavior_cloning: Optional[TransformerPPOBehaviorCloningConfig] = None` to `TransformerPPOStageConfig`.

- [ ] **Step 4: Verify GREEN**

Run schema tests.

- [ ] **Step 5: Commit**

Run: `git add utils/config_schema.py tests && git commit -m "feat(config): add TransformerPPO behavior cloning schema"`

---

### Task 6: BehaviorCloningRegularizer Lifecycle Core

**Files:**
- Create: `algorithms/utils/behavior_cloning.py`
- Create: `tests/test_behavior_cloning_regularizer.py`

- [ ] **Step 1: Write failing tests**

Tests should cover:
- `from_config(...)` returns `None` when config absent or disabled.
- `from_config(...)` returns an instance when enabled and warm_start is present.
- `attach_environment(...)` builds the teacher with the shared helper.
- `record_transition(building_idx)` appends the latest teacher action into a per-building deque.
- `on_buffer_flushed(building_idx)` clears only that building's deque.
- `on_topology_change(...)` clears all deques and re-attaches the teacher.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_behavior_cloning_regularizer.py -v`

Expected: module missing.

- [ ] **Step 3: Implement lifecycle core**

Use one class. Keep dependencies small: standard typing, `collections.deque`, `copy.deepcopy`, `numpy`, `torch`, `build_warm_start_policy`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_behavior_cloning_regularizer.py -q`

Expected: lifecycle tests pass.

- [ ] **Step 5: Commit**

Run: `git add algorithms/utils/behavior_cloning.py tests/test_behavior_cloning_regularizer.py && git commit -m "feat(tppo): add BC regularizer lifecycle core"`

---

### Task 7: BC Loss, Type Weights, Schedule, and Phaseout

**Files:**
- Modify: `algorithms/utils/behavior_cloning.py`
- Test: `tests/test_behavior_cloning_regularizer.py`

- [ ] **Step 1: Write failing tests**

Tests should cover:
- Effective weight schedule: base before start, linear decay, min after end.
- Per-CA-type weights from layout segments: `charger -> ev_multiplier`, `storage -> storage_multiplier`, default `1.0`.
- BC loss is zero when predicted equals teacher.
- BC loss ignores missing/NaN teacher actions.
- `probability` phaseout can return teacher actions.
- `blend` phaseout returns `p * teacher + (1-p) * actor` and decays over predict steps.
- Deterministic prediction disables phaseout but still allows teacher cache for BC.

- [ ] **Step 2: Verify RED**

Run the new tests; expected failures are missing methods.

- [ ] **Step 3: Implement methods**

Add:

```python
effective_weight(global_learning_step: int) -> float
ca_type_weights(layout: BuildingTokenLayout, *, dtype, device) -> torch.Tensor
bc_loss_term(...)
compute_teacher_actions(...)
maybe_phaseout(...)
snapshot_metrics() -> Dict[str, float]
```

BC loss uses finite masks and denominator `clamp(weights.sum(), min=1.0)`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_behavior_cloning_regularizer.py -q`

- [ ] **Step 5: Commit**

Run: `git add algorithms/utils/behavior_cloning.py tests/test_behavior_cloning_regularizer.py && git commit -m "feat(tppo): implement BC loss and warm-start phaseout"`

---

### Task 8: Wire Collaborator Into AgentTransformerPPO

**Files:**
- Modify: `algorithms/agents/agent_transformer_ppo.py`
- Create: `tests/test_agent_transformer_ppo_behavior_cloning.py`

- [ ] **Step 1: Write failing integration tests**

Tests should cover:
- `AgentTransformerPPO(...without behavior_cloning...)` has `_bc is None` and existing predict/update still work.
- With BC config, `_bc` exists after init and its teacher exists after `attach_environment`.
- `predict(...)` computes teacher actions and phaseout can blend.
- `update(...)` calls `record_transition` so teacher deques align with buffer length.
- PPO update adds a finite BC loss and exposes diagnostics.
- Topology change calls `on_topology_change` and flushes teacher deques.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_agent_transformer_ppo_behavior_cloning.py -v`

Expected: missing `_bc` hooks / no behavior.

- [ ] **Step 3: Implement minimal hooks**

Modify `AgentTransformerPPO` only at the collaborator hook points listed in this plan. Avoid duplicating BC logic in the agent.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_agent_transformer_ppo_behavior_cloning.py tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_wrapper_integration.py -q`

- [ ] **Step 5: Commit**

Run: `git add algorithms/agents/agent_transformer_ppo.py tests/test_agent_transformer_ppo_behavior_cloning.py && git commit -m "feat(tppo): wire BC regularizer into TransformerPPO"`

---

### Task 9: BC Template, Smoke Test, and Docs

**Files:**
- Create: `configs/templates/dynamic/transformer_ppo_bc_entity_dynamic.yaml`
- Create: `tests/test_template_transformer_ppo_bc_entity_dynamic.py`
- Modify: `docs/transformer_ppo_spec.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Write template smoke test**

Mirror `tests/test_template_transformer_ppo_entity_dynamic.py` and assert the BC block resolves with `RBCCommunityPolicy`, `phaseout_mode: blend`, `ev_multiplier: 24.0`, `storage_multiplier: 0.18`.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_template_transformer_ppo_bc_entity_dynamic.py -v`

Expected: template file missing.

- [ ] **Step 3: Add template**

Copy the existing TransformerPPO dynamic entity template and add the `behavior_cloning` block. Do not modify the existing non-BC template.

- [ ] **Step 4: Add docs**

Append `docs/transformer_ppo_spec.md` section `13. Behavior Cloning` describing optional BC, teacher policy, phaseout, per-CA-type weights, and deferred residual policy. Add a concise note to `AGENTS.md`.

- [ ] **Step 5: Verify GREEN**

Run: `pytest tests/test_template_transformer_ppo_bc_entity_dynamic.py -q`

- [ ] **Step 6: Commit**

Run: `git add configs/templates/dynamic/transformer_ppo_bc_entity_dynamic.yaml tests/test_template_transformer_ppo_bc_entity_dynamic.py docs/transformer_ppo_spec.md AGENTS.md && git commit -m "feat(tppo): add BC entity-dynamic template and docs"`

---

### Task 10: Final Regression and Review

**Files:**
- All touched files.

- [ ] **Step 1: Run targeted regression**

Run: `pytest tests/test_ppo_components.py tests/test_warm_start_policy.py tests/test_behavior_cloning_regularizer.py tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_wrapper_integration.py tests/test_agent_transformer_ppo_behavior_cloning.py tests/test_template_transformer_ppo_bc_entity_dynamic.py -q`

- [ ] **Step 2: Run full suite**

Run: `pytest -q`

- [ ] **Step 3: Inspect status and diff**

Run: `git status --short && git diff --stat && git log --oneline -10`

- [ ] **Step 4: Final code review**

Review for:
- No BC-specific logic leaked into `ppo_components.py` beyond `Batch.step_indices`.
- `AgentTransformerPPO` only delegates to the collaborator.
- No duplicated warm-start-policy registry.
- Existing non-BC TransformerPPO behavior remains unchanged when config block absent.

- [ ] **Step 5: Report results**

Summarize commits, tests, and any remaining risks.
