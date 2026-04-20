# Plan C Fixes — Address Dummy Implementations

**Goal:** Fix incomplete/dummy implementations found in code review of PR #10

**Context:** PR #10 implements Tasks 1-6 from Plan C. Tests pass but several dummy placeholders need proper implementation before production use.

---

## Issues Summary

| # | Issue | Severity | File:Line |
|---|-------|----------|-----------|
| 1 | `_last_obs` indexing bug | High | transformer_ppo_agent.py:222-223 |
| 2 | Log prob aggregation inconsistency | High | transformer_ppo_agent.py:290,367 |
| 3 | Dummy observation in topology change | High | transformer_ppo_agent.py:488 |
| 4 | Encoder rebuild not implemented | Medium | wrapper_citylearn.py:734 |
| 5 | torch.load security warning | Low | transformer_ppo_agent.py:431 |

---

## Task 1: Fix _last_obs Indexing Bug

**Problem:** predict() assigns scalars instead of per-building indexed storage

**Current code:**
```python
# Line 167-168: initialized as lists
self._last_obs = [None] * self._num_buildings
self._last_actions = [None] * self._num_buildings

# Line 222-223: assigned without index
self._last_obs = obs_tensor
self._last_actions = actions
```

**Fix:**
```python
# Line 222-223: use building index
self._last_obs[b_idx] = obs_tensor
self._last_actions[b_idx] = actions
```

**Test:** Verify multi-building scenario stores separate obs/actions per building

---

## Task 2: Fix Log Prob Aggregation

**Problem:** Inconsistent aggregation (sum vs mean) between update() and _ppo_update()

**Current code:**
```python
# Line 290: update() uses sum
log_prob=self._last_log_probs[b_idx].sum(dim=-1).squeeze()

# Line 367: _ppo_update() uses mean
log_probs_new = log_probs_new.mean(dim=-1)
```

**Decision:** Use `.sum(dim=-1)` consistently (PPO with independent actions)

**Fix:**
```python
# Line 367: change to sum
log_probs_new = log_probs_new.sum(dim=-1)
```

**Reasoning:** Independent CA actions should sum log probs (log product rule)

**Test:** Verify PPO loss computation doesn't explode/vanish

---

## Task 3: Fix on_topology_change Dummy Observation

**Problem:** Hardcoded `dummy_obs = np.zeros(10)` doesn't match actual observation shape

**Current code:**
```python
# Line 488: hardcoded shape
dummy_obs = np.zeros(10)
self._ppo_update(building_idx, dummy_obs)
```

**Options:**

### Option A: Store last observation shape
```python
# In update() after storing transition
self._last_valid_obs[b_idx] = next_observations[b_idx].copy()

# In on_topology_change()
if len(buffer) >= self.minibatch_size:
    last_obs = self._last_valid_obs[building_idx]
    self._ppo_update(building_idx, last_obs)
```

### Option B: Skip GAE bootstrapping
```python
# Modify _ppo_update signature
def _ppo_update(
    self, 
    building_idx: int, 
    last_obs: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Args:
        last_obs: Last observation for GAE bootstrap. 
                  If None, uses zero bootstrap (terminal state).
    """
    if last_obs is None:
        # Zero bootstrap for topology change
        last_value = torch.zeros(1, device=self.device)
    else:
        # Normal bootstrap
        with torch.no_grad():
            # ... existing code
```

**Recommended:** Option B (cleaner, handles edge case properly)

**Test:** Trigger topology change mid-training, verify no crash

---

## Task 4: Implement Encoder Rebuild (Or Document Deferral)

**Problem:** Wrapper detects topology change but encoder rebuild stubbed out

**Current code:**
```python
# Line 734
# Rebuild encoders for this building
# (Implementation depends on existing encoder structure)
```

**Options:**

### Option A: Implement encoder rebuild
Requires understanding existing encoder caching mechanism. Check:
- How encoders currently built (search for encoder setup in wrapper)
- Whether encoder specs cached per building
- How enrichment result (EnrichmentResult.marker_encoder_specs) used

### Option B: Document deferral
If topology change rare in practice (EVs don't connect/disconnect mid-episode), defer to future work:

```python
# Line 734
# TODO(plan-d): Implement encoder rebuild for topology changes
# Currently: encoders built once at init
# Future: rebuild encoders using enrichment.marker_encoder_specs
# For now: topology changes require episode restart
logger.warning(
    f"Topology change detected for building {building_idx}. "
    "Encoder rebuild not implemented - results may be incorrect."
)
```

**Recommended:** Option B for MVP (topology change rare)

**Follow-up:** Create Plan D task if topology change becomes critical

---

## Task 5: Fix torch.load Security Warning

**Problem:** FutureWarning about unsafe pickle loading

**Current code:**
```python
# Line 431
checkpoint = torch.load(checkpoint_path, map_location=self.device)
```

**Fix:**
```python
# Line 431: explicit weights_only for forward compat
checkpoint = torch.load(
    checkpoint_path, 
    map_location=self.device,
    weights_only=False  # Set False explicitly (True blocks optimizer state)
)
```

**Note:** `weights_only=True` prevents loading optimizer state. For full checkpoint restore, need False but acknowledge security implication.

**Test:** Verify checkpoint save/load still works

---

## Implementation Order

1. **Task 1** (5 min) — Simple index fix
2. **Task 2** (5 min) — Change mean to sum  
3. **Task 5** (2 min) — Add weights_only param
4. **Task 3** (15 min) — Implement Option B (skip GAE)
5. **Task 4** (5 min) — Add TODO + warning (Option B)

**Total estimate:** 30 minutes

---

## Testing Plan

After fixes:

```bash
# Run existing tests (should still pass)
pytest tests/test_agent_transformer_ppo.py -v
pytest tests/test_e2e_transformer_ppo.py -v
pytest tests/test_wrapper_transformer.py -v

# Add new test for topology change
# tests/test_agent_transformer_ppo.py::TestAgentTopologyChange::test_topology_change_triggers_update
```

**New test:**
```python
def test_on_topology_change_no_crash(agent_with_env):
    """on_topology_change should handle buffer flush without crash."""
    # Fill buffer past minibatch_size
    for i in range(5):
        obs = np.array([[1001.0, 0.5, 2001.0, 0.5, 0.5, 0.5, 0.5, 3001.0, 100.0]])
        agent_with_env.predict([obs], deterministic=False)
        agent_with_env.update(
            observations=[obs],
            actions=[np.array([[0.5]])],
            rewards=[1.0],
            next_observations=[obs],
            terminated=[False],
            truncated=[False],
            update_target_step=False,
            global_learning_step=i,
            update_step=False,
            initial_exploration_done=True,
        )
    
    # Trigger topology change
    agent_with_env.on_topology_change(building_idx=0)
    
    # Should not crash, buffer should be empty
    assert len(agent_with_env.rollout_buffers[0]) == 0
```

---

## Acceptance Criteria

- [ ] All existing tests pass
- [ ] New topology change test passes
- [ ] No FutureWarnings in test output
- [ ] Multi-building test verifies separate obs storage
- [ ] Code review issues 1-5 resolved

---

## Notes

- **Encoder rebuild**: Deferred to Plan D unless critical path
- **Topology change**: Rare in practice (static building configs), zero-bootstrap acceptable
- **Security**: weights_only=False needed for optimizer state, acceptable for trusted checkpoints
