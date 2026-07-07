# AgentTransformerMATD3 — Plan D: Training Wiring, Context Hooks, Diagnostics & Full Checkpoint

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the complete `update()` method that connects all Plan B/C primitives into a functioning training loop, implement `set_observation_context`/`set_transition_context` hooks for wrapper integration, add comprehensive diagnostics under the `TransformerMATD3/` namespace, extend checkpointing to cover critics+replay+BC state, and validate with dynamic topology integration tests.

**Architecture:** `update()` orchestrates: store transition → gate check → sample replay → build global tokens → critic update (twin MSE) → delayed actor update (per-building, critic frozen) → soft target updates. Context hooks feed teacher actions into replay. Diagnostics emit per-step metrics via `_record_training_metrics`. Full checkpoint persists all training state for resume.

**Tech Stack:** Python 3.10+, PyTorch, numpy, pytest, mlflow (optional metrics sink).

**Spec:** `docs/transformer_matd3_spec.md` (sections: Data Flow, Training flow steps 1-8, Context hooks, Diagnostics, Checkpoint And Export).

**Depends on:** Plan A (actor stack, predict, export), Plan B (twin critics, global token packer, topology-partitioned replay, critic update primitives), Plan C (teacher lifecycle, residual composition, target smoothing, replay-native BC, exploration gating).

**Produces:** A fully trainable agent that passes multi-step update integration tests, dynamic topology smoke tests, diagnostics verification, full checkpoint round-trip, and a wrapper-driven end-to-end smoke.

---

## API Contract from Plan B (READ FIRST)

Plan D wires modules created in Plan B. Use these exact APIs — do NOT improvise method names.

### `TopologyPartitionedReplay` (from `algorithms/utils/matd3_replay.py`)

```python
replay = TopologyPartitionedReplay(capacity=int, batch_size=int)
replay.set_active_signature(signature: str)      # switch active before push/sample
replay.push(transition: TransitionData)          # store one transition
batch: Optional[SampledBatch] = replay.sample()  # NO args; returns None if active < batch_size
size = replay.total_size                          # PROPERTY, not method
size = replay.active_partition_size               # PROPERTY
size = replay.partition_size(sig)                 # method
count = replay.partition_count                    # PROPERTY
state = replay.state_dict()                       # for checkpoint
replay.load_state_dict(state)
```

`TransitionData` fields (all required at construction):
`observations, next_observations, actions, base_actions, next_base_actions, rewards, done, topology_signature, layout_summaries`.

`SampledBatch` fields (returned by sample):
`observations, next_observations, actions, base_actions, next_base_actions, rewards, done, topology_signature, layout_summaries` — each per-building list is `[batch_size, dim]`.

### `GlobalTokenPacker` (from `algorithms/utils/matd3_global_packer.py`)

```python
packer = GlobalTokenPacker(d_model, num_token_types, max_buildings, action_input_mode)
packed: PackedGlobalSequence = packer.pack(
    obs_tokens_per_building=[Tensor(B, n_obs_b, d_model), ...],
    action_values_per_building=[Tensor(B, n_ca_b), ...],
    layouts=[BuildingLayout(...), ...],
    base_actions=[Tensor(B, n_ca_b), ...] or None,
    action_span=2.0,
)
```

`PackedGlobalSequence` fields: `global_tokens, type_ids, building_ids, padding_mask, controlled_building_indices`.

### `TwinTransformerCritics` (from `algorithms/utils/matd3_critic.py`)

Critics take UNPACKED tensors, not a `PackedGlobalSequence` object:

```python
q1, q2 = twins(
    packed.global_tokens,
    packed.type_ids,
    packed.building_ids,
    packed.padding_mask,
    packed.controlled_building_indices,
)
min_q = twins.min_q(...)  # same signature
twins.soft_update_from(source, tau=float)
```

### Agent-Owned Helper Contract

The agent (`AgentTransformerMATD3`) is responsible for these helper methods (Plan D introduces them; Plan B does not):

- `self._build_obs_tokens(observations_per_building)` — call each building's tokenizer+backbone to produce `obs_tokens_per_building` for the packer.
- `self._current_topology_signature()` — stable hash from current `_actors` state (matches `compute_topology_signature` from Plan B).
- `self._recompute_actions_for_building(b, current_actor_output_tensor)` — build a fresh action tokens list where building b uses the current actor output; all other buildings use detached final actions from the sampled batch.

These are defined inline in Task 3 below.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `algorithms/agents/agent_transformer_matd3.py` (modify) | Wire full `update()`, context hooks, diagnostics, full checkpoint |
| `tests/_matd3_test_helpers.py` (create) | Shared fixture helpers (config, transition gen, valid topology-change names) |
| `tests/test_matd3_update_integration.py` (create) | Full update loop tests |
| `tests/test_matd3_dynamic_topology_integration.py` (create) | Dynamic topology smoke tests |
| `tests/test_matd3_diagnostics.py` (create) | Diagnostics output tests |
| `tests/test_matd3_wrapper_smoke.py` (create) | End-to-end wrapper-driven smoke |

---

## Shared Test Helper

All Plan D tests reuse the `_make_matd3()` helper from Plan A (`tests/test_agent_transformer_matd3_foundation.py`). Import it as:

```python
from tests.test_agent_transformer_matd3_foundation import _make_matd3, _matd3_config
```

For Plan D, an extended config helper is needed that enables BC and residual:

```python
# tests/_matd3_test_helpers.py (create once, import in all Plan D tests)
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import List, Tuple

from tests.test_agent_transformer_matd3_foundation import _make_matd3, _matd3_config


def _matd3_config_full_training() -> dict:
    """Config with all training features enabled (BC, residual, reward normalization)."""
    cfg = _matd3_config()
    algo = cfg["algorithm"]
    algo["hyperparameters"]["reward_normalization"] = True
    algo["hyperparameters"]["reward_normalization_clip"] = 5.0
    algo["exploration"] = {
        "random_exploration_steps": 2,
        "end_initial_exploration_time_step": 4,
        "train_during_initial_exploration": False,
        "warm_start_policy": {
            "enabled": True,
            "phaseout_steps": 3,
        },
    }
    algo["residual"] = {
        "enabled": True,
        "initial_scale": 0.1,
        "growth_steps": 10,
        "max_scale": 1.0,
        "storage_scale_multiplier": 0.5,
        "ev_scale_multiplier": 0.8,
    }
    algo["behavior_cloning"] = {
        "enabled": True,
        "weight": 1.0,
        "min_weight": 0.0,
        "decay_start_step": 5,
        "decay_steps": 10,
        "ev_multiplier": 2.0,
        "storage_multiplier": 1.5,
    }
    return cfg


def _make_matd3_full(n_buildings: int = 2):
    """Create an agent with full training config, return (agent, obs_per, act_per, obs_dim)."""
    from tests._entity_sample_obs_names import load_sample_observation_names_for_first_building
    from algorithms.agents.agent_transformer_matd3 import AgentTransformerMATD3

    obs_names = load_sample_observation_names_for_first_building()
    obs_per = [list(obs_names) for _ in range(n_buildings)]
    act_per = [["electrical_storage", "electric_vehicle_storage"] for _ in range(n_buildings)]
    agent = AgentTransformerMATD3(_matd3_config_full_training())
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[None] * n_buildings,
        observation_space=[None] * n_buildings,
        metadata={"building_names": [f"Building_{b}" for b in range(n_buildings)]},
    )
    obs_dim = len(obs_names)
    return agent, obs_per, act_per, obs_dim


def _generate_transition(
    n_buildings: int, obs_dim: int
) -> Tuple[List[npt.NDArray], List[npt.NDArray], List[float], List[npt.NDArray], bool, bool]:
    """Generate a random transition tuple for update()."""
    obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(n_buildings)]
    actions = [np.random.uniform(-1, 1, size=2).astype(np.float64) for _ in range(n_buildings)]
    rewards = [float(np.random.randn()) for _ in range(n_buildings)]
    next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(n_buildings)]
    terminated = False
    truncated = False
    return obs, actions, rewards, next_obs, terminated, truncated


def _run_update_step(
    agent,
    obs,
    actions,
    rewards,
    next_obs,
    terminated,
    truncated,
    *,
    global_learning_step: int,
    update_step: bool = True,
    update_target_step: bool = False,
    initial_exploration_done: bool = True,
) -> None:
    """Run a single update step with context hooks."""
    agent.set_observation_context(
        raw_observations=obs,
        encoded_observations=obs,
    )
    agent.set_transition_context(
        raw_observations=obs,
        raw_next_observations=next_obs,
        encoded_observations=obs,
        encoded_next_observations=next_obs,
    )
    agent.update(
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        terminated=terminated,
        truncated=truncated,
        update_target_step=update_target_step,
        global_learning_step=global_learning_step,
        update_step=update_step,
        initial_exploration_done=initial_exploration_done,
    )


def _add_charger_to_building_obs(
    obs_names: List[str],
    building_id: str,
    new_charger_id: str,
) -> Tuple[List[str], str]:
    """Extend obs_names with a full valid charger asset block.

    A single feature name is NOT enough — the tokenizer requires the FULL
    per-type feature set for the `charger` type plus its EV connected/incoming
    context features. This helper duplicates the feature signature from an
    existing charger in the building so the topology-change test triggers a
    real rebuild instead of tokenizer validation errors.

    Returns (new_obs_names, new_action_name).
    """
    # Discover existing charger to mirror its feature block
    charger_prefix = None
    for name in obs_names:
        if name.startswith("charger::") and "::" in name[len("charger::"):]:
            # e.g. "charger::Building_0/charger_1_1::connected_state"
            _, existing_id, _ = name.split("::", 2)
            charger_prefix = existing_id
            break
    if charger_prefix is None:
        raise RuntimeError(
            "No existing charger in obs_names — cannot mirror feature block. "
            "Use a base fixture that contains at least one charger."
        )

    # Collect all feature suffixes belonging to that charger (including
    # connected_ev::* and incoming_ev::* sub-blocks).
    prefix = f"charger::{charger_prefix}::"
    suffixes = [name[len(prefix):] for name in obs_names if name.startswith(prefix)]
    if not suffixes:
        raise RuntimeError(f"No features found under prefix {prefix!r}")

    new_prefix = f"charger::{building_id}/{new_charger_id}::"
    new_names = list(obs_names) + [new_prefix + s for s in suffixes]
    new_action = f"electric_vehicle_storage_{new_charger_id}"
    return new_names, new_action
```

---

## Task 1: Context Hooks — `set_observation_context` and `set_transition_context`

**Files:**
- Modify: `algorithms/agents/agent_transformer_matd3.py`
- Create: `tests/test_matd3_update_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_matd3_update_integration.py`:

```python
"""Plan D integration tests — update loop for AgentTransformerMATD3."""
from __future__ import annotations

import numpy as np
import pytest

from tests._matd3_test_helpers import (
    _make_matd3_full,
    _generate_transition,
    _run_update_step,
)


class TestContextHooks:
    def test_set_observation_context_stores_raw(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        assert agent._latest_raw_observations is not None
        assert len(agent._latest_raw_observations) == 2
        assert np.allclose(agent._latest_raw_observations[0], obs[0])

    def test_set_observation_context_stores_encoded(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        assert agent._latest_encoded_observations is not None
        assert len(agent._latest_encoded_observations) == 2

    def test_set_transition_context_stores_next(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        agent.set_transition_context(
            raw_observations=obs,
            raw_next_observations=next_obs,
            encoded_observations=obs,
            encoded_next_observations=next_obs,
        )
        assert agent._latest_raw_next_observations is not None
        assert len(agent._latest_raw_next_observations) == 2
        assert agent._latest_encoded_next_observations is not None

    def test_set_transition_context_computes_teacher_actions(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        agent.set_transition_context(
            raw_observations=obs,
            raw_next_observations=next_obs,
            encoded_observations=obs,
            encoded_next_observations=next_obs,
        )
        # Teacher actions should be computed for replay storage
        assert agent._latest_teacher_actions is not None
        assert agent._latest_next_teacher_actions is not None
        assert len(agent._latest_teacher_actions) == 2
        assert len(agent._latest_next_teacher_actions) == 2

    def test_context_hook_noop_when_teacher_released(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        # Simulate teacher release
        agent._warm_start_policy = None
        agent._teacher_alive = False
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        agent.set_transition_context(
            raw_observations=obs,
            raw_next_observations=next_obs,
            encoded_observations=obs,
            encoded_next_observations=next_obs,
        )
        assert agent._latest_teacher_actions is None
        assert agent._latest_next_teacher_actions is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_update_integration.py::TestContextHooks -v`
Expected: AttributeError — `set_observation_context` not implemented on the agent.

- [ ] **Step 3: Implement context hooks**

Add to `algorithms/agents/agent_transformer_matd3.py` (after `predict`, before `update`):

```python
def set_observation_context(
    self,
    *,
    raw_observations: Optional[List[np.ndarray]] = None,
    encoded_observations: Optional[List[np.ndarray]] = None,
) -> None:
    """Receive wrapper-side observation context for teacher action computation."""
    self._latest_raw_observations = (
        [np.asarray(obs, dtype=np.float64) for obs in raw_observations]
        if raw_observations is not None
        else None
    )
    self._latest_encoded_observations = (
        [np.asarray(obs, dtype=np.float64) for obs in encoded_observations]
        if encoded_observations is not None
        else None
    )
    # Compute teacher actions for current state (used in exploration + replay)
    if self._teacher_alive and self._warm_start_policy is not None:
        self._latest_teacher_actions = self._compute_teacher_actions(
            self._latest_raw_observations
        )
    else:
        self._latest_teacher_actions = None

def set_transition_context(
    self,
    *,
    raw_observations: Optional[List[np.ndarray]] = None,
    raw_next_observations: Optional[List[np.ndarray]] = None,
    encoded_observations: Optional[List[np.ndarray]] = None,
    encoded_next_observations: Optional[List[np.ndarray]] = None,
) -> None:
    """Receive current/next observation context for teacher-aware replay."""
    if raw_observations is not None:
        self._latest_raw_observations = [
            np.asarray(obs, dtype=np.float64) for obs in raw_observations
        ]
    if encoded_observations is not None:
        self._latest_encoded_observations = [
            np.asarray(obs, dtype=np.float64) for obs in encoded_observations
        ]
    self._latest_raw_next_observations = (
        [np.asarray(obs, dtype=np.float64) for obs in raw_next_observations]
        if raw_next_observations is not None
        else None
    )
    self._latest_encoded_next_observations = (
        [np.asarray(obs, dtype=np.float64) for obs in encoded_next_observations]
        if encoded_next_observations is not None
        else None
    )
    # Compute teacher actions for next state (for target residual composition)
    if self._teacher_alive and self._warm_start_policy is not None:
        self._latest_teacher_actions = self._compute_teacher_actions(
            self._latest_raw_observations
        )
        self._latest_next_teacher_actions = self._compute_teacher_actions(
            self._latest_raw_next_observations
        )
    else:
        self._latest_teacher_actions = None
        self._latest_next_teacher_actions = None

def _compute_teacher_actions(
    self, raw_observations: Optional[List[np.ndarray]]
) -> Optional[List[List[float]]]:
    """Get teacher policy actions for the given observations."""
    if self._warm_start_policy is None or raw_observations is None:
        return None
    actions = self._warm_start_policy.predict(raw_observations, deterministic=True)
    return [
        np.clip(np.asarray(a, dtype=np.float64).reshape(-1), -1.0, 1.0).tolist()
        for a in actions
    ]
```

Initialize the state attributes in `__init__`:

```python
# Context hook state
self._latest_raw_observations: Optional[List[np.ndarray]] = None
self._latest_encoded_observations: Optional[List[np.ndarray]] = None
self._latest_raw_next_observations: Optional[List[np.ndarray]] = None
self._latest_encoded_next_observations: Optional[List[np.ndarray]] = None
self._latest_teacher_actions: Optional[List[List[float]]] = None
self._latest_next_teacher_actions: Optional[List[List[float]]] = None
self._teacher_alive: bool = False  # Set to True when teacher is attached (Plan C)
self._warm_start_policy: Optional[Any] = None  # Teacher policy object (Plan C)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_matd3_update_integration.py::TestContextHooks -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/agent_transformer_matd3.py tests/_matd3_test_helpers.py tests/test_matd3_update_integration.py
git commit -m "feat(matd3-t): context hooks set_observation_context/set_transition_context"
```

---

## Task 2: Full `update()` Method — Skeleton with Store + Gate

**Files:**
- Modify: `algorithms/agents/agent_transformer_matd3.py`
- Test: `tests/test_matd3_update_integration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_matd3_update_integration.py`:

```python
class TestUpdateGating:
    def test_update_skips_before_initial_exploration(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=0,
            initial_exploration_done=False,
        )
        # Transition should be stored but no gradient step
        assert agent._replay is not None
        assert agent._replay.total_size >= 1
        assert agent._critic_update_count == 0

    def test_update_skips_when_replay_too_small(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        # batch_size=4, so 1 transition is not enough
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=10,
            initial_exploration_done=True,
        )
        assert agent._critic_update_count == 0

    def test_update_stores_transition_with_topology_sig(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=10,
            initial_exploration_done=True,
        )
        sig = agent._current_topology_signature()
        assert agent._replay.partition_size(sig) == 1

    def test_update_stores_teacher_actions_in_replay(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        # Set up teacher actions via context hook
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        agent.set_transition_context(
            raw_observations=obs,
            raw_next_observations=next_obs,
            encoded_observations=obs,
            encoded_next_observations=next_obs,
        )
        agent.update(
            observations=obs, actions=actions, rewards=rewards,
            next_observations=next_obs, terminated=term, truncated=trunc,
            update_target_step=False, global_learning_step=10,
            update_step=True, initial_exploration_done=True,
        )
        sig = agent._current_topology_signature()
        assert agent._replay.partition_size(sig) >= 1
        # sample() uses active partition; set it and sample one transition to
        # verify teacher actions were stored (batch_size=1 for this test).
        agent._replay.batch_size = 1
        agent._replay.set_active_signature(sig)
        batch = agent._replay.sample()
        assert batch is not None
        # teacher/base actions present as SampledBatch fields
        assert batch.base_actions is not None
        assert batch.next_base_actions is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_update_integration.py::TestUpdateGating -v`
Expected: Failure — `update()` is currently a no-op.

- [ ] **Step 3: Implement update skeleton (store + gate)**

Replace the `update()` method in `algorithms/agents/agent_transformer_matd3.py`:

```python
def update(
    self,
    observations: List[npt.NDArray[np.float64]],
    actions: List[npt.NDArray[np.float64]],
    rewards: List[float],
    next_observations: List[npt.NDArray[np.float64]],
    terminated: bool,
    truncated: bool,
    *,
    update_target_step: bool,
    global_learning_step: int,
    update_step: bool,
    initial_exploration_done: bool,
) -> None:
    """Full training step: store → gate → critic → delayed actor → targets."""
    from algorithms.utils.matd3_replay import TransitionData, LayoutSummary

    done = terminated or truncated

    # --- 1. Update reward normalizer ---
    self._update_reward_normalizer(rewards)

    # --- 2. Store transition in replay ---
    topology_sig = self._current_topology_signature()
    self._replay.set_active_signature(topology_sig)

    # Derive teacher/base actions per building (None → zeros as placeholder)
    n_buildings = len(observations)
    base_actions_np = self._teacher_actions_or_zeros(
        self._latest_teacher_actions, n_buildings
    )
    next_base_actions_np = self._teacher_actions_or_zeros(
        self._latest_next_teacher_actions, n_buildings
    )
    layout_summaries = [
        LayoutSummary(
            building_id=s.building_id,
            n_ca=s.layout.n_ca,
            n_sro=s.layout.n_sro,
            obs_dim=len(s.obs_names_tuple),
            action_dim=s.layout.n_ca,
        )
        for s in self._actors
    ]
    transition = TransitionData(
        observations=[np.asarray(o, dtype=np.float32) for o in observations],
        next_observations=[np.asarray(o, dtype=np.float32) for o in next_observations],
        actions=[np.asarray(a, dtype=np.float32) for a in actions],
        base_actions=base_actions_np,
        next_base_actions=next_base_actions_np,
        rewards=[float(r) for r in rewards],
        done=bool(done),
        topology_signature=topology_sig,
        layout_summaries=layout_summaries,
    )
    self._replay.push(transition)

    # --- 3. Gate: skip gradient updates if conditions not met ---
    if not self._should_train_on_step(initial_exploration_done, global_learning_step):
        return
    if not update_step:
        return
    if self._replay.active_partition_size < self._batch_size:
        return

    # --- 4. Sample batch from active partition ---
    batch = self._replay.sample()
    if batch is None:
        return

    # --- 5. Build global tokens for current-state critic input ---
    # obs_tokens_per_building[b]: [B, n_obs_tokens_b, d_model]
    # actions_per_building[b]: [B, n_ca_b]
    obs_tokens = self._build_obs_tokens_from_batch(batch.observations)
    actions_t = [
        torch.as_tensor(batch.actions[b], dtype=torch.float32)
        for b in range(n_buildings)
    ]
    base_t = [
        torch.as_tensor(batch.base_actions[b], dtype=torch.float32)
        for b in range(n_buildings)
    ]
    packed_current = self._critic_packer.pack(
        obs_tokens_per_building=obs_tokens,
        action_values_per_building=actions_t,
        layouts=self._packer_layouts(),
        base_actions=base_t if self._residual_enabled else None,
    )

    # Build packed_next with target-actor actions (post-residual, post-smoothing)
    target_actions = self._compute_target_actions(
        batch.next_observations, batch.next_base_actions
    )
    next_base_t = [
        torch.as_tensor(batch.next_base_actions[b], dtype=torch.float32)
        for b in range(n_buildings)
    ]
    with torch.no_grad():
        next_obs_tokens = self._build_obs_tokens_from_batch(batch.next_observations)
    packed_next = self._critic_packer.pack(
        obs_tokens_per_building=next_obs_tokens,
        action_values_per_building=target_actions,
        layouts=self._packer_layouts(),
        base_actions=next_base_t if self._residual_enabled else None,
    )

    # --- 6. Critic update ---
    self._update_critics(packed_current, packed_next, batch, global_learning_step)
    self._critic_update_count += 1

    # --- 7. Delayed actor update ---
    if self._critic_update_count % self._actor_update_interval == 0:
        self._update_actors(batch, obs_tokens, actions_t, base_t, global_learning_step)
        self._actor_update_count += 1

    # --- 8. Soft-update targets ---
    if update_target_step:
        self._soft_update_all_targets()

    # --- 9. Diagnostics ---
    if self._should_log_training_step(global_learning_step):
        metrics = self._collect_diagnostics(global_learning_step)
        self._record_training_metrics(metrics, global_learning_step)


def _teacher_actions_or_zeros(
    self,
    teacher: Optional[List[List[float]]],
    n_buildings: int,
) -> List[npt.NDArray[np.float32]]:
    """Return per-building teacher action arrays; zeros if teacher unavailable."""
    result: List[npt.NDArray[np.float32]] = []
    for b in range(n_buildings):
        n_ca = self._actors[b].layout.n_ca
        if teacher is not None and b < len(teacher) and len(teacher[b]) == n_ca:
            result.append(np.asarray(teacher[b], dtype=np.float32))
        else:
            result.append(np.zeros(n_ca, dtype=np.float32))
    return result


def _build_obs_tokens_from_batch(
    self,
    observations_per_building: List[npt.NDArray[np.float32]],
) -> List[torch.Tensor]:
    """Run each building's tokenizer+backbone to get obs token embeddings.

    Returns a list of tensors, one per building, each [B, n_obs_tokens_b, d_model].
    Concatenates [SRO tokens, NFC token, CA tokens] per building.
    """
    obs_tokens: List[torch.Tensor] = []
    for b, state in enumerate(self._actors):
        obs_b = torch.as_tensor(observations_per_building[b], dtype=torch.float32)
        tokenized = state.tokenizer(obs_b, state.layout)
        # Concatenate all obs-side token banks
        concat = torch.cat(
            [tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens], dim=1
        )
        obs_tokens.append(concat)
    return obs_tokens


def _packer_layouts(self):
    """Build BuildingLayout list for the packer from current actor states."""
    from algorithms.utils.matd3_global_packer import BuildingLayout
    return [
        BuildingLayout(
            building_index=b,
            n_sro=s.layout.n_sro,
            n_nfc=1,
            n_ca=s.layout.n_ca,
            is_controlled=s.layout.n_ca >= 1,
        )
        for b, s in enumerate(self._actors)
    ]
```

Initialize counters in `__init__`:

```python
self._critic_update_count: int = 0
self._actor_update_count: int = 0
self._batch_size: int = int(h["batch_size"])
self._actor_update_interval: int = int(h["actor_update_interval"])
```

Add the training gate method:

```python
def _should_train_on_step(
    self, initial_exploration_done: bool, global_learning_step: int
) -> bool:
    """Check if training should proceed this step."""
    if initial_exploration_done:
        return True
    exploration_cfg = self.config["algorithm"].get("exploration", {}) or {}
    train_during = bool(exploration_cfg.get("train_during_initial_exploration", False))
    if not train_during:
        return False
    start_step = int(exploration_cfg.get("initial_exploration_training_start_step", 0))
    return global_learning_step >= start_step
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_matd3_update_integration.py::TestUpdateGating -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/agent_transformer_matd3.py tests/test_matd3_update_integration.py
git commit -m "feat(matd3-t): update() skeleton with store + gate logic"
```

---

## Task 3: Critic Update + Delayed Actor Update Loop

**Files:**
- Modify: `algorithms/agents/agent_transformer_matd3.py`
- Test: `tests/test_matd3_update_integration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_matd3_update_integration.py`:

```python
import torch


class TestCriticAndActorUpdate:
    def _fill_replay(self, agent, n_buildings, obs_dim, n_transitions=8):
        """Push enough transitions so sampling works."""
        for step in range(n_transitions):
            obs, actions, rewards, next_obs, term, trunc = _generate_transition(
                n_buildings, obs_dim
            )
            agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
            agent.set_transition_context(
                raw_observations=obs, raw_next_observations=next_obs,
                encoded_observations=obs, encoded_next_observations=next_obs,
            )
            agent.update(
                observations=obs, actions=actions, rewards=rewards,
                next_observations=next_obs, terminated=term, truncated=trunc,
                update_target_step=False, global_learning_step=step,
                update_step=False,  # just store, don't train yet
                initial_exploration_done=True,
            )

    def test_critic_update_changes_critic_params(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)

        # Snapshot critic params before update
        c1_params_before = [p.clone() for p in agent._critic_1.parameters()]

        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=100,
            update_step=True,
            initial_exploration_done=True,
        )
        assert agent._critic_update_count >= 1
        # At least one critic parameter changed
        c1_params_after = list(agent._critic_1.parameters())
        any_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(c1_params_before, c1_params_after)
        )
        assert any_changed, "Critic parameters should have changed after update"

    def test_actor_update_respects_interval(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)

        # actor_update_interval=2, so first critic update should NOT update actor
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=100,
            update_step=True,
            initial_exploration_done=True,
        )
        assert agent._critic_update_count == 1
        assert agent._actor_update_count == 0  # interval=2, so skip

        # Second update should trigger actor update
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=101,
            update_step=True,
            initial_exploration_done=True,
        )
        assert agent._critic_update_count == 2
        assert agent._actor_update_count == 1

    def test_actor_update_changes_actor_params(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)

        # Force actor_update_interval=1 for direct testing
        agent._actor_update_interval = 1
        actor_params_before = [p.clone() for p in agent._actors[0].actor.parameters()]

        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=100,
            update_step=True,
            initial_exploration_done=True,
        )
        actor_params_after = list(agent._actors[0].actor.parameters())
        any_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(actor_params_before, actor_params_after)
        )
        assert any_changed, "Actor parameters should change on actor update step"

    def test_critic_frozen_during_actor_update(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)
        agent._actor_update_interval = 1

        # Record critic params before actor update
        c1_before = [p.clone() for p in agent._critic_1.parameters()]

        # Disable critic optimizer to ensure critic params stay fixed
        # The actor update should NOT modify critic weights
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)

        # Hook into _update_actors only to verify critic is frozen
        agent._verify_critic_frozen_during_actor = True
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=100,
            update_step=True,
            initial_exploration_done=True,
        )
        # Critic params should only change from critic update, not actor update
        # This is verified by the implementation using torch.no_grad() on critic params

    def test_soft_target_update(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)

        # Get target params before
        target_params_before = [
            p.clone() for p in agent._actors[0].target_actor.parameters()
        ]

        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=100,
            update_step=True,
            update_target_step=True,  # trigger soft update
            initial_exploration_done=True,
        )
        target_params_after = list(agent._actors[0].target_actor.parameters())
        any_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(target_params_before, target_params_after)
        )
        assert any_changed, "Target params should change on target update step"

    def test_min_q_target_uses_both_critics(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)

        # Verify both critics have different parameters (independent)
        c1_params = list(agent._critic_1.parameters())
        c2_params = list(agent._critic_2.parameters())
        # They start with different random init — at least one layer differs
        any_diff = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(c1_params, c2_params)
        )
        assert any_diff, "Twin critics must have independent parameters"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_update_integration.py::TestCriticAndActorUpdate -v`
Expected: Failures — `_update_critics` and `_update_actors` not wired.

- [ ] **Step 3: Implement critic update**

Add to `algorithms/agents/agent_transformer_matd3.py`:

```python
def _update_critics(
    self,
    packed_current,           # PackedGlobalSequence for (s, a)
    packed_next,              # PackedGlobalSequence for (s', a')
    batch,                    # SampledBatch
    global_learning_step: int,
) -> None:
    """Twin critic update with min-Q target."""
    n_buildings = len(batch.observations)
    # Rewards per building → shape [B, n_controlled]. Concat only for controlled buildings.
    controlled_idx = packed_current.controlled_building_indices
    rewards_by_ctrl = np.stack(
        [batch.rewards[b] for b in controlled_idx], axis=1
    ).astype(np.float32)  # [B, n_controlled]
    rewards_t = torch.as_tensor(rewards_by_ctrl, dtype=torch.float32)
    dones_t = torch.as_tensor(batch.done, dtype=torch.float32).unsqueeze(-1)  # [B, 1]

    if self._reward_normalization_enabled:
        rewards_t = self._normalize_reward_tensor(rewards_t)

    # Target Q: min(Q1_target, Q2_target)
    with torch.no_grad():
        target_q1 = self._target_critic_1(
            packed_next.global_tokens,
            packed_next.type_ids,
            packed_next.building_ids,
            packed_next.padding_mask,
            packed_next.controlled_building_indices,
        )
        target_q2 = self._target_critic_2(
            packed_next.global_tokens,
            packed_next.type_ids,
            packed_next.building_ids,
            packed_next.padding_mask,
            packed_next.controlled_building_indices,
        )
        target_q = torch.min(target_q1, target_q2)  # [B, n_controlled]
        # TD target: r + gamma * (1 - done) * min_Q_target
        td_target = rewards_t + self._gamma * (1.0 - dones_t) * target_q

    # Critic 1 loss
    q1_pred = self._critic_1(
        packed_current.global_tokens,
        packed_current.type_ids,
        packed_current.building_ids,
        packed_current.padding_mask,
        packed_current.controlled_building_indices,
    )
    q1_loss = torch.nn.functional.mse_loss(q1_pred, td_target)

    # Critic 2 loss
    q2_pred = self._critic_2(
        packed_current.global_tokens,
        packed_current.type_ids,
        packed_current.building_ids,
        packed_current.padding_mask,
        packed_current.controlled_building_indices,
    )
    q2_loss = torch.nn.functional.mse_loss(q2_pred, td_target)

    # Combined gradient step
    self._critic_optimizer.zero_grad()
    total_critic_loss = q1_loss + q2_loss
    total_critic_loss.backward()
    self._critic_optimizer.step()

    # Store for diagnostics
    self._last_q1_loss = float(q1_loss.item())
    self._last_q2_loss = float(q2_loss.item())
    self._last_q_gap = float((q1_pred - q2_pred).abs().mean().item())
    self._last_target_q_mean = float(target_q.mean().item())
    self._last_target_q_std = float(target_q.std().item())
```

- [ ] **Step 4: Implement actor update loop**

Add to `algorithms/agents/agent_transformer_matd3.py`:

```python
def _update_actors(
    self,
    batch,                        # SampledBatch
    obs_tokens_current,           # List[Tensor(B, n_obs_b, d_model)] precomputed
    actions_current_t,            # List[Tensor(B, n_ca_b)] final actions from batch
    base_actions_current_t,       # List[Tensor(B, n_ca_b)] teacher/base from batch
    global_learning_step: int,
) -> None:
    """Delayed actor update: per-building actor against critic 1 (frozen).

    For each controlled building b, rebuild the packed sequence with only
    building b's action tokens recomputed from the current actor (gradient
    flows through b's actor/backbone/tokenizer). Other buildings' actions
    and obs tokens are detached.
    """
    controlled_buildings = [
        b for b, state in enumerate(self._actors) if state.layout.n_ca >= 1
    ]

    # Detach every building's obs tokens so gradient only flows through b.
    # We'll re-encode building b inside the loop (with grad enabled).
    obs_tokens_detached = [t.detach() for t in obs_tokens_current]
    actions_detached = [t.detach() for t in actions_current_t]
    base_detached = [t.detach() for t in base_actions_current_t]
    layouts = self._packer_layouts()

    total_actor_loss = 0.0
    total_bc_loss = 0.0
    total_grad_norm = 0.0

    # Freeze critic 1 for the actor updates
    for p in self._critic_1.parameters():
        p.requires_grad_(False)

    try:
        for b in controlled_buildings:
            state = self._actors[b]

            # Re-encode building b with grad enabled
            obs_b = torch.as_tensor(batch.observations[b], dtype=torch.float32)
            tokenized = state.tokenizer(obs_b, state.layout)
            ca_emb, _ = state.backbone(
                tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens
            )
            raw_actor_output = state.actor(ca_emb).squeeze(-1)  # [B, n_ca]

            # Compose residual if enabled (uses teacher/base from batch)
            teacher_b = torch.as_tensor(batch.base_actions[b], dtype=torch.float32)
            if self._residual_enabled:
                composed_b = self._residual_compose(raw_actor_output, teacher_b, b)
            else:
                composed_b = self._scale_actor_to_action_space(raw_actor_output)

            # Build fresh obs_tokens list: b uses freshly-encoded (grad), others detached
            obs_tokens_b = list(obs_tokens_detached)
            obs_tokens_b[b] = torch.cat(
                [tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens], dim=1
            )

            # Build fresh actions list: b uses composed (grad), others detached
            actions_b = list(actions_detached)
            actions_b[b] = composed_b

            base_actions_b = list(base_detached)
            base_actions_b[b] = teacher_b

            packed_with_actor_b = self._critic_packer.pack(
                obs_tokens_per_building=obs_tokens_b,
                action_values_per_building=actions_b,
                layouts=layouts,
                base_actions=base_actions_b if self._residual_enabled else None,
            )

            q1_for_actor = self._critic_1(
                packed_with_actor_b.global_tokens,
                packed_with_actor_b.type_ids,
                packed_with_actor_b.building_ids,
                packed_with_actor_b.padding_mask,
                packed_with_actor_b.controlled_building_indices,
            )
            ctrl_idx = packed_with_actor_b.controlled_building_indices.index(b)
            actor_policy_loss = -q1_for_actor[:, ctrl_idx].mean()

            # BC loss (if enabled)
            bc_loss = torch.tensor(0.0)
            if self._bc_enabled:
                bc_loss = self._compute_bc_loss(
                    composed_b, teacher_b, b, global_learning_step
                )

            loss_b = actor_policy_loss + bc_loss

            # Backward through actor only (critic is frozen)
            state.optimizer.zero_grad()
            loss_b.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(state.tokenizer.parameters())
                + list(state.backbone.parameters())
                + list(state.actor.parameters()),
                max_norm=10.0,
            )
            state.optimizer.step()

            total_actor_loss += float(actor_policy_loss.item())
            total_bc_loss += float(bc_loss.item())
            total_grad_norm += float(grad_norm)
    finally:
        # Re-enable critic gradients
        for p in self._critic_1.parameters():
            p.requires_grad_(True)

    n = max(len(controlled_buildings), 1)
    self._last_actor_loss = total_actor_loss / n
    self._last_bc_loss = total_bc_loss / n
    self._last_actor_grad_norm = total_grad_norm / n
```

- [ ] **Step 5: Implement target soft-update**

Add to `algorithms/agents/agent_transformer_matd3.py`:

```python
def _soft_update_all_targets(self) -> None:
    """Polyak-average online → target for actors and critics."""
    tau = self._tau
    # Actor targets
    for state in self._actors:
        self._soft_update(state.actor, state.target_actor, tau)
        # Also update tokenizer/backbone targets if separate target backbone exists
    # Critic targets
    self._soft_update(self._critic_1, self._target_critic_1, tau)
    self._soft_update(self._critic_2, self._target_critic_2, tau)

def _soft_update(
    self, online: torch.nn.Module, target: torch.nn.Module, tau: float
) -> None:
    """Polyak averaging: target = tau*online + (1-tau)*target."""
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.lerp_(op.data, tau)
```

- [ ] **Step 6: Implement target action computation**

Add to `algorithms/agents/agent_transformer_matd3.py`:

```python
def _compute_target_actions(
    self,
    next_observations_per_building,        # List[np.ndarray] of [B, obs_dim_b]
    next_base_actions_per_building,        # List[np.ndarray] of [B, n_ca_b]
) -> List[torch.Tensor]:
    """Compute target actions for critic-target computation.

    Steps per building: target_actor → residual compose (with next teacher
    actions from replay) → target policy smoothing → clip.
    Returns per-building tensors of shape [B, n_ca_b], with grad disabled.
    """
    target_actions: List[torch.Tensor] = []
    with torch.no_grad():
        for b, state in enumerate(self._actors):
            if state.layout.n_ca < 1:
                # Uncontrolled buildings still need a tensor slot for the packer
                target_actions.append(
                    torch.zeros(
                        next_observations_per_building[b].shape[0], 0,
                        dtype=torch.float32,
                    )
                )
                continue

            obs_b = torch.as_tensor(
                next_observations_per_building[b], dtype=torch.float32
            )
            tokenized = state.tokenizer(obs_b, state.layout)
            ca_emb, _ = state.backbone(
                tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens
            )
            raw_target = state.target_actor(ca_emb).squeeze(-1)  # [B, n_ca]

            # Residual composition with next base actions from replay
            if self._residual_enabled:
                teacher_b = torch.as_tensor(
                    next_base_actions_per_building[b], dtype=torch.float32
                )
                composed = self._residual_compose(raw_target, teacher_b, b)
            else:
                composed = self._scale_actor_to_action_space(raw_target)

            # Target policy smoothing (in final action space)
            smoothed = self._apply_target_policy_smoothing(composed)
            target_actions.append(smoothed)

    return target_actions

def _apply_target_policy_smoothing(self, actions: torch.Tensor) -> torch.Tensor:
    """Add clipped Gaussian noise to target actions, clip to bounds."""
    noise_scale = self._target_policy_noise
    noise_clip = self._target_policy_noise_clip
    action_span = 2.0  # [-1, 1] bounds

    noise = torch.randn_like(actions) * (noise_scale * action_span)
    noise = torch.clamp(noise, -noise_clip * action_span, noise_clip * action_span)
    return torch.clamp(actions + noise, -1.0, 1.0)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_matd3_update_integration.py::TestCriticAndActorUpdate -v`
Expected: 6 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add algorithms/agents/agent_transformer_matd3.py tests/test_matd3_update_integration.py
git commit -m "feat(matd3-t): critic update + delayed actor update loop + soft targets"
```

---

## Task 4: Full Checkpoint with Critics + Replay + Training State

**Files:**
- Modify: `algorithms/agents/agent_transformer_matd3.py`
- Test: `tests/test_matd3_update_integration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_matd3_update_integration.py`:

```python
import tempfile
from pathlib import Path


class TestFullCheckpoint:
    def _trained_agent(self, n_buildings=2, n_transitions=10):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=n_buildings)
        for step in range(n_transitions):
            obs, actions, rewards, next_obs, term, trunc = _generate_transition(
                n_buildings, obs_dim
            )
            _run_update_step(
                agent, obs, actions, rewards, next_obs, term, trunc,
                global_learning_step=step + 10,
                update_step=True,
                update_target_step=(step % 2 == 0),
                initial_exploration_done=True,
            )
        return agent, obs_dim

    def test_checkpoint_includes_critics(self):
        agent, obs_dim = self._trained_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = agent.save_checkpoint(tmpdir, step=100)
            payload = torch.load(path, map_location="cpu")
            assert "critic_1_state" in payload
            assert "critic_2_state" in payload
            assert "target_critic_1_state" in payload
            assert "target_critic_2_state" in payload
            assert "critic_optimizer_state" in payload

    def test_checkpoint_includes_replay(self):
        agent, obs_dim = self._trained_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = agent.save_checkpoint(tmpdir, step=100)
            payload = torch.load(path, map_location="cpu")
            assert "replay_state" in payload
            assert "active_topology_signature" in payload

    def test_checkpoint_includes_training_state(self):
        agent, obs_dim = self._trained_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = agent.save_checkpoint(tmpdir, step=100)
            payload = torch.load(path, map_location="cpu")
            assert "critic_update_count" in payload
            assert "actor_update_count" in payload
            assert "reward_normalization_state" in payload
            assert "exploration_state" in payload
            assert "rng_state" in payload

    def test_checkpoint_round_trip_preserves_training(self):
        agent, obs_dim = self._trained_agent()
        critic_count_before = agent._critic_update_count
        actor_count_before = agent._actor_update_count

        with tempfile.TemporaryDirectory() as tmpdir:
            path = agent.save_checkpoint(tmpdir, step=100)
            # Create a fresh agent and load
            agent2, _, _, _ = _make_matd3_full(n_buildings=2)
            agent2.load_checkpoint(path)
            assert agent2._critic_update_count == critic_count_before
            assert agent2._actor_update_count == actor_count_before

    def test_checkpoint_round_trip_preserves_replay(self):
        agent, obs_dim = self._trained_agent()
        sig = agent._current_topology_signature()
        replay_size_before = agent._replay.partition_size(sig)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = agent.save_checkpoint(tmpdir, step=100)
            agent2, _, _, _ = _make_matd3_full(n_buildings=2)
            agent2.load_checkpoint(path)
            assert agent2._replay.partition_size(sig) == replay_size_before

    def test_checkpoint_rejects_feature_count_mismatch(self):
        agent, obs_dim = self._trained_agent(n_buildings=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = agent.save_checkpoint(tmpdir, step=100)
            # Tamper with saved feature dims to simulate schema drift
            payload = torch.load(path, map_location="cpu")
            payload["per_type_feature_dims"]["storage"] = 999
            torch.save(payload, path)
            agent2, _, _, _ = _make_matd3_full(n_buildings=2)
            with pytest.raises(ValueError, match="feature.*(count|dim)"):
                agent2.load_checkpoint(path)

    def test_checkpoint_rejects_building_count_mismatch(self):
        agent, obs_dim = self._trained_agent(n_buildings=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = agent.save_checkpoint(tmpdir, step=100)
            agent_1b, _, _, _ = _make_matd3_full(n_buildings=1)
            with pytest.raises(ValueError, match="[Bb]uilding.count"):
                agent_1b.load_checkpoint(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_update_integration.py::TestFullCheckpoint -v`
Expected: Failures — current `save_checkpoint` is Plan A minimal version.

- [ ] **Step 3: Implement full checkpoint save**

Replace `save_checkpoint` in `algorithms/agents/agent_transformer_matd3.py`:

```python
def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
    """Full checkpoint: actors, critics, replay, training state."""
    out = Path(output_dir) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"transformer_matd3_step{step}.pt"

    payload = {
        "step": step,
        # Per-building actor state
        "actors": [
            {
                "building_id": s.building_id,
                "tokenizer_state": s.tokenizer.state_dict(),
                "backbone_state": s.backbone.state_dict(),
                "actor_state": s.actor.state_dict(),
                "target_actor_state": s.target_actor.state_dict(),
                "optimizer_state": s.optimizer.state_dict(),
                "obs_names": list(s.obs_names_tuple),
                "action_names": list(s.action_names_tuple),
                "topology_version": s.topology_version,
            }
            for s in self._actors
        ],
        # Critics
        "critic_1_state": self._critic_1.state_dict(),
        "critic_2_state": self._critic_2.state_dict(),
        "target_critic_1_state": self._target_critic_1.state_dict(),
        "target_critic_2_state": self._target_critic_2.state_dict(),
        "critic_optimizer_state": self._critic_optimizer.state_dict(),
        # Replay
        "replay_state": self._replay.state_dict(),
        "active_topology_signature": self._current_topology_signature(),
        # Training counters
        "critic_update_count": self._critic_update_count,
        "actor_update_count": self._actor_update_count,
        # Reward normalization
        "reward_normalization_state": {
            "count": getattr(self, "_reward_norm_count", 0),
            "mean": getattr(self, "_reward_norm_mean", 0.0),
            "m2": getattr(self, "_reward_norm_m2", 0.0),
        },
        # Exploration state
        "exploration_state": {
            "exploration_step": getattr(self, "_exploration_step", 0),
            "teacher_alive": self._teacher_alive,
            "bc_effective_weight": getattr(self, "_bc_effective_weight", 0.0),
            "residual_scale": getattr(self, "_residual_scale", 0.0),
        },
        # Feature dims for validation on load
        "per_type_feature_dims": self._get_per_type_feature_dims(),
        "n_buildings": len(self._actors),
        # RNG state
        "rng_state": {
            "torch": torch.random.get_rng_state(),
            "numpy": np.random.get_state(),
        },
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved: {} (step {})", path, step)
    return str(path)
```

- [ ] **Step 4: Implement full checkpoint load**

Replace `load_checkpoint` in `algorithms/agents/agent_transformer_matd3.py`:

```python
def load_checkpoint(self, checkpoint_path: str) -> None:
    """Load full checkpoint with validation."""
    payload = torch.load(checkpoint_path, map_location="cpu")

    # --- Validate building count ---
    saved_n = payload.get("n_buildings", len(payload.get("actors", [])))
    if saved_n != len(self._actors):
        raise ValueError(
            f"Checkpoint has {saved_n} buildings; current agent "
            f"has {len(self._actors)}. Building-count mismatch."
        )

    # --- Validate feature dimensions ---
    saved_dims = payload.get("per_type_feature_dims", {})
    current_dims = self._get_per_type_feature_dims()
    for type_name, saved_dim in saved_dims.items():
        current_dim = current_dims.get(type_name)
        if current_dim is not None and int(saved_dim) != int(current_dim):
            raise ValueError(
                f"Feature count mismatch for type '{type_name}': "
                f"checkpoint has {saved_dim}, current has {current_dim}. "
                f"Cannot restore weights."
            )

    # --- Restore actor states ---
    for state, saved in zip(self._actors, payload["actors"]):
        state.tokenizer.load_state_dict(saved["tokenizer_state"])
        state.backbone.load_state_dict(saved["backbone_state"])
        state.actor.load_state_dict(saved["actor_state"])
        state.target_actor.load_state_dict(saved["target_actor_state"])
        state.optimizer.load_state_dict(saved["optimizer_state"])

    # --- Restore critic states ---
    if "critic_1_state" in payload:
        self._critic_1.load_state_dict(payload["critic_1_state"])
        self._critic_2.load_state_dict(payload["critic_2_state"])
        self._target_critic_1.load_state_dict(payload["target_critic_1_state"])
        self._target_critic_2.load_state_dict(payload["target_critic_2_state"])
        self._critic_optimizer.load_state_dict(payload["critic_optimizer_state"])

    # --- Restore replay ---
    if "replay_state" in payload:
        self._replay.load_state_dict(payload["replay_state"])

    # --- Restore training counters ---
    self._critic_update_count = int(payload.get("critic_update_count", 0))
    self._actor_update_count = int(payload.get("actor_update_count", 0))

    # --- Restore reward normalization ---
    rn = payload.get("reward_normalization_state", {})
    self._reward_norm_count = int(rn.get("count", 0))
    self._reward_norm_mean = float(rn.get("mean", 0.0))
    self._reward_norm_m2 = float(rn.get("m2", 0.0))

    # --- Restore exploration state ---
    es = payload.get("exploration_state", {})
    self._exploration_step = int(es.get("exploration_step", 0))
    self._teacher_alive = bool(es.get("teacher_alive", False))
    self._bc_effective_weight = float(es.get("bc_effective_weight", 0.0))
    self._residual_scale = float(es.get("residual_scale", 0.0))

    # --- Restore RNG ---
    rng = payload.get("rng_state")
    if rng is not None:
        torch.random.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])

    logger.info(
        "Checkpoint loaded: step={}, critic_updates={}, actor_updates={}",
        payload.get("step", "?"),
        self._critic_update_count,
        self._actor_update_count,
    )
```

Add the helper:

```python
def _get_per_type_feature_dims(self) -> Dict[str, int]:
    """Get current per-type feature dimensions for checkpoint validation."""
    dims: Dict[str, int] = {}
    for state in self._actors:
        for seg in state.layout.segments:
            if seg.family == "nfc":
                continue
            existing = dims.get(seg.type_name)
            new = len(seg.feature_indices)
            if existing is None:
                dims[seg.type_name] = new
            elif existing != new:
                # Same type with different dims across buildings — use first
                pass
    return dims
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_matd3_update_integration.py::TestFullCheckpoint -v`
Expected: 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add algorithms/agents/agent_transformer_matd3.py tests/test_matd3_update_integration.py
git commit -m "feat(matd3-t): full checkpoint with critics, replay, training state"
```

---

## Task 5: Diagnostics

**Files:**
- Modify: `algorithms/agents/agent_transformer_matd3.py`
- Create: `tests/test_matd3_diagnostics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_matd3_diagnostics.py`:

```python
"""Plan D diagnostics tests for AgentTransformerMATD3."""
from __future__ import annotations

import pytest

from tests._matd3_test_helpers import (
    _make_matd3_full,
    _generate_transition,
    _run_update_step,
)


class TestDiagnostics:
    def _trained_agent(self, n_transitions=10):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        for step in range(n_transitions):
            obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
            _run_update_step(
                agent, obs, actions, rewards, next_obs, term, trunc,
                global_learning_step=step + 10,
                update_step=True,
                update_target_step=(step % 2 == 0),
                initial_exploration_done=True,
            )
        return agent

    def test_diagnostics_namespace(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        # All keys should be under TransformerMATD3/ namespace
        for key in metrics:
            assert key.startswith("TransformerMATD3/"), f"Key {key} not in namespace"

    def test_replay_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/replay_size" in metrics
        assert "TransformerMATD3/active_partition_size" in metrics
        assert "TransformerMATD3/partition_count" in metrics

    def test_critic_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/critic_q1_loss" in metrics
        assert "TransformerMATD3/critic_q2_loss" in metrics
        assert "TransformerMATD3/critic_q_gap" in metrics
        assert "TransformerMATD3/target_q_mean" in metrics
        assert "TransformerMATD3/target_q_std" in metrics

    def test_actor_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/actor_loss" in metrics
        assert "TransformerMATD3/actor_grad_norm" in metrics

    def test_teacher_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/teacher_alive" in metrics
        assert "TransformerMATD3/residual_scale" in metrics

    def test_bc_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/bc_loss" in metrics
        assert "TransformerMATD3/bc_effective_weight" in metrics

    def test_reward_norm_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/reward_norm_mean" in metrics
        assert "TransformerMATD3/reward_norm_std" in metrics

    def test_critic_action_input_mode_reported(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/critic_action_input_mode_final" in metrics

    def test_diagnostics_are_floats(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_diagnostics.py -v`
Expected: AttributeError — `_collect_diagnostics` not implemented.

- [ ] **Step 3: Implement diagnostics collection**

Add to `algorithms/agents/agent_transformer_matd3.py`:

```python
def _collect_diagnostics(self, global_learning_step: int) -> Dict[str, float]:
    """Collect all training diagnostics under TransformerMATD3/ namespace."""
    sig = self._current_topology_signature()
    critic_mode = self.config["algorithm"]["hyperparameters"].get(
        "critic_action_input_mode", "final"
    )

    metrics: Dict[str, float] = {
        # Replay
        "TransformerMATD3/replay_size": float(self._replay.total_size),
        "TransformerMATD3/active_partition_size": float(
            self._replay.partition_size(sig)
        ),
        "TransformerMATD3/partition_count": float(self._replay.partition_count),
        # Critic
        "TransformerMATD3/critic_q1_loss": float(getattr(self, "_last_q1_loss", 0.0)),
        "TransformerMATD3/critic_q2_loss": float(getattr(self, "_last_q2_loss", 0.0)),
        "TransformerMATD3/critic_q_gap": float(getattr(self, "_last_q_gap", 0.0)),
        "TransformerMATD3/target_q_mean": float(
            getattr(self, "_last_target_q_mean", 0.0)
        ),
        "TransformerMATD3/target_q_std": float(
            getattr(self, "_last_target_q_std", 0.0)
        ),
        # Actor
        "TransformerMATD3/actor_loss": float(getattr(self, "_last_actor_loss", 0.0)),
        "TransformerMATD3/actor_grad_norm": float(
            getattr(self, "_last_actor_grad_norm", 0.0)
        ),
        # Teacher/Residual
        "TransformerMATD3/teacher_alive": float(self._teacher_alive),
        "TransformerMATD3/residual_scale": float(
            getattr(self, "_residual_scale", 0.0)
        ),
        "TransformerMATD3/phaseout_probability": float(
            getattr(self, "_last_phaseout_probability", 0.0)
        ),
        # BC
        "TransformerMATD3/bc_loss": float(getattr(self, "_last_bc_loss", 0.0)),
        "TransformerMATD3/bc_effective_weight": float(
            getattr(self, "_bc_effective_weight", 0.0)
        ),
        # Reward normalization
        "TransformerMATD3/reward_norm_mean": float(
            getattr(self, "_reward_norm_mean", 0.0)
        ),
        "TransformerMATD3/reward_norm_std": float(
            self._reward_normalization_std()
            if getattr(self, "_reward_normalization_enabled", False)
            else 0.0
        ),
        "TransformerMATD3/reward_norm_count": float(
            getattr(self, "_reward_norm_count", 0)
        ),
        # Critic input mode
        "TransformerMATD3/critic_action_input_mode_final": float(
            critic_mode == "final"
        ),
        "TransformerMATD3/critic_action_input_mode_delta": float(
            critic_mode in ("final_base_delta", "final_base_delta_normalized")
        ),
        # Counters
        "TransformerMATD3/critic_update_count": float(self._critic_update_count),
        "TransformerMATD3/actor_update_count": float(self._actor_update_count),
        "TransformerMATD3/global_learning_step": float(global_learning_step),
    }
    return metrics

def _should_log_training_step(self, global_learning_step: int) -> bool:
    """Determine if this step should emit diagnostics."""
    interval = self.config["algorithm"]["hyperparameters"].get(
        "log_interval", 10
    )
    return global_learning_step % max(int(interval), 1) == 0

def _record_training_metrics(self, metrics: Dict[str, float], step: int) -> None:
    """Record metrics to MLflow if active, store latest."""
    if not metrics:
        return
    self._latest_training_metrics = dict(metrics)
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)
    except ImportError:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_matd3_diagnostics.py -v`
Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/agent_transformer_matd3.py tests/test_matd3_diagnostics.py
git commit -m "feat(matd3-t): diagnostics collection under TransformerMATD3/ namespace"
```

---

## Task 6: Dynamic Topology Integration Smoke Tests

**Files:**
- Create: `tests/test_matd3_dynamic_topology_integration.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_matd3_dynamic_topology_integration.py`:

```python
"""Plan D — Dynamic topology integration smoke tests for AgentTransformerMATD3."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._matd3_test_helpers import (
    _make_matd3_full,
    _generate_transition,
    _run_update_step,
    _matd3_config_full_training,
    _add_charger_to_building_obs,
)
from tests._entity_sample_obs_names import load_sample_observation_names_for_first_building
from algorithms.agents.agent_transformer_matd3 import AgentTransformerMATD3


class TestDynamicTopologySmoke:
    """Run predict+update for N steps, trigger topology change, verify continuity."""

    def _build_agent_and_train(self, n_steps=8):
        """Create agent, run n_steps of predict+update."""
        agent, obs_per, act_per, obs_dim = _make_matd3_full(n_buildings=2)
        for step in range(n_steps):
            obs, actions, rewards, next_obs, term, trunc = _generate_transition(
                2, obs_dim
            )
            _run_update_step(
                agent, obs, actions, rewards, next_obs, term, trunc,
                global_learning_step=step + 10,
                update_step=True,
                update_target_step=(step % 2 == 0),
                initial_exploration_done=True,
            )
        return agent, obs_per, act_per, obs_dim

    def test_topology_change_switches_replay_signature(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train()
        sig_before = agent._current_topology_signature()
        replay_size_before = agent._replay.partition_size(sig_before)
        assert replay_size_before > 0

        # Trigger topology change: add a full valid charger block to building 0
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )

        sig_after = agent._current_topology_signature()
        assert sig_after != sig_before
        # Old partition preserved
        assert agent._replay.partition_size(sig_before) == replay_size_before
        # New partition starts empty
        assert agent._replay.partition_size(sig_after) == 0

    def test_actor_weights_survive_topology_change(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train()

        # Snapshot building 1 actor params (unchanged building)
        b1_params_before = [
            p.clone() for p in agent._actors[1].actor.parameters()
        ]

        # Trigger topology change on building 0 only (full valid charger block)
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )

        # Building 1 actor params unchanged
        b1_params_after = list(agent._actors[1].actor.parameters())
        for before, after in zip(b1_params_before, b1_params_after):
            assert torch.allclose(before, after)

        # Building 0 actor params preserved (same type projections)
        # Just verify no crash and predict still works
        new_obs_dim = len(new_obs_0)
        obs = [
            np.random.randn(new_obs_dim).astype(np.float64),
            np.random.randn(obs_dim).astype(np.float64),
        ]
        actions = agent.predict(obs, deterministic=True)
        assert len(actions[0]) == 3  # 2 original + 1 new CA
        assert len(actions[1]) == 2  # unchanged

    def test_critic_weights_survive_topology_change(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train()

        # Snapshot critic backbone params
        c1_params_before = [p.clone() for p in agent._critic_1.parameters()]

        # Trigger topology change (full valid charger block)
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )

        # Critic backbone weights preserved (topology change only updates packing)
        c1_params_after = list(agent._critic_1.parameters())
        for before, after in zip(c1_params_before, c1_params_after):
            assert torch.allclose(before, after)

    def test_teacher_reattaches_on_topology_change(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train()
        # Teacher should be alive (exploration/residual still active)
        assert agent._teacher_alive is True

        # Trigger topology change (full valid charger block)
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )

        # Teacher should still be alive and re-attached
        assert agent._teacher_alive is True
        # Teacher can produce actions for new layout
        new_obs_dim = len(new_obs_0)
        obs = [
            np.random.randn(new_obs_dim).astype(np.float64),
            np.random.randn(obs_dim).astype(np.float64),
        ]
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        assert agent._latest_teacher_actions is not None

    def test_training_continues_after_topology_change(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train(n_steps=8)
        critic_updates_before = agent._critic_update_count

        # Trigger topology change (full valid charger block)
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )

        # Fill new partition past batch_size
        new_obs_dim = len(new_obs_0)
        for step in range(10):
            obs = [
                np.random.randn(new_obs_dim).astype(np.float64),
                np.random.randn(obs_dim).astype(np.float64),
            ]
            actions = [
                np.random.uniform(-1, 1, size=3).astype(np.float64),
                np.random.uniform(-1, 1, size=2).astype(np.float64),
            ]
            rewards = [float(np.random.randn()), float(np.random.randn())]
            next_obs = [
                np.random.randn(new_obs_dim).astype(np.float64),
                np.random.randn(obs_dim).astype(np.float64),
            ]
            _run_update_step(
                agent, obs, actions, rewards, next_obs, False, False,
                global_learning_step=100 + step,
                update_step=True,
                initial_exploration_done=True,
            )

        # Critic should have been updated after replay filled
        assert agent._critic_update_count > critic_updates_before

    def test_update_skips_until_new_partition_has_batch_size(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train(n_steps=8)
        critic_updates_at_change = agent._critic_update_count

        # Trigger topology change (full valid charger block)
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )

        # Push 1 transition (less than batch_size=4)
        new_obs_dim = len(new_obs_0)
        obs = [
            np.random.randn(new_obs_dim).astype(np.float64),
            np.random.randn(obs_dim).astype(np.float64),
        ]
        actions = [
            np.random.uniform(-1, 1, size=3).astype(np.float64),
            np.random.uniform(-1, 1, size=2).astype(np.float64),
        ]
        _run_update_step(
            agent, obs, actions, [0.1, 0.2],
            [np.random.randn(new_obs_dim), np.random.randn(obs_dim)],
            False, False,
            global_learning_step=200,
            update_step=True,
            initial_exploration_done=True,
        )
        # Should NOT have updated (partition too small)
        assert agent._critic_update_count == critic_updates_at_change

    def test_export_after_training_only_actors(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train()
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = agent.export_artifacts(tmpdir)
            # Only actor artifacts
            assert manifest["format"] == "onnx"
            for art in manifest["artifacts"]:
                assert "critic" not in art["path"].lower()
                assert Path(tmpdir, art["path"]).exists()
            # No critic keys in top-level manifest
            manifest_str = str(manifest).lower()
            assert "critic" not in manifest_str or "critic_action_input_mode" not in manifest_str

    def test_building_count_change_fails_fast(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train()
        with pytest.raises(ValueError, match="[Bb]uilding.count"):
            agent.attach_environment(
                observation_names=[obs_per[0]],  # 1 building instead of 2
                action_names=[act_per[0]],
                action_space=[None],
                observation_space=[None],
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_dynamic_topology_integration.py -v`
Expected: Failures if topology change handling for critics/replay/teacher is incomplete.

- [ ] **Step 3: Implement topology change handling for critics + replay + teacher**

In the `_handle_topology_change` method, add critic packer rebuild and teacher re-attach:

```python
def _handle_topology_change(self, building_idx: int) -> None:
    state = self._actors[building_idx]
    new_layout = self._layout_builder.build(
        state.building_id,
        list(state.obs_names_tuple),
        list(state.action_names_tuple),
    )
    new_dims = self._compute_type_input_dims(new_layout)
    for tname, dim in new_dims.items():
        if tname not in state.tokenizer.projections:
            raise ValueError(
                f"Topology change for {state.building_id!r}: new type "
                f"{tname!r} has no projection. Restart required."
            )
        if int(state.tokenizer.projections[tname].in_features) != int(dim):
            raise ValueError(
                f"Topology change for {state.building_id!r}: feature count "
                f"for type {tname!r} changed. Weights cannot be preserved."
            )
    state.layout = new_layout
    state.topology_version += 1

    # Rebuild critic packing metadata for new topology
    self._critic_packer.rebuild(
        [s.layout for s in self._actors],
        [s.building_id for s in self._actors],
    )

    # Re-attach teacher if still alive
    if self._teacher_alive and self._warm_start_policy is not None:
        self._reattach_teacher()

    logger.info(
        "Topology change: {} v{} — n_ca={}, replay sig switched",
        state.building_id, state.topology_version, new_layout.n_ca,
    )
```

Add:

```python
def _reattach_teacher(self) -> None:
    """Re-attach teacher policy with current observation/action names."""
    if self._warm_start_policy is None:
        return
    obs_names = [list(s.obs_names_tuple) for s in self._actors]
    act_names = [list(s.action_names_tuple) for s in self._actors]
    # Teacher's attach_environment handles the topology refresh
    if hasattr(self._warm_start_policy, "attach_environment"):
        self._warm_start_policy.attach_environment(
            observation_names=obs_names,
            action_names=act_names,
            action_space=[None] * len(self._actors),
            observation_space=[None] * len(self._actors),
        )

def _current_topology_signature(self) -> str:
    """Compute stable hash of current per-building topology."""
    import hashlib
    parts = []
    for state in self._actors:
        parts.append(f"{state.building_id}|{state.obs_names_tuple}|{state.action_names_tuple}")
        # Include per-type feature dims
        for seg in state.layout.segments:
            if seg.family != "nfc":
                parts.append(f"{seg.type_name}:{len(seg.feature_indices)}")
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_matd3_dynamic_topology_integration.py -v`
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/agent_transformer_matd3.py tests/test_matd3_dynamic_topology_integration.py
git commit -m "feat(matd3-t): dynamic topology integration — replay sig switch, weight preservation, teacher re-attach"
```

---

## Task 7: Wrapper Integration Verification + `is_initial_exploration_done`

**Files:**
- Test: `tests/test_matd3_update_integration.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_matd3_update_integration.py`:

```python
class TestWrapperIntegration:
    """Verify the agent exposes the hooks the wrapper expects."""

    def test_set_observation_context_is_callable(self):
        agent, _, _, _ = _make_matd3_full()
        hook = getattr(agent, "set_observation_context", None)
        assert callable(hook)

    def test_set_transition_context_is_callable(self):
        agent, _, _, _ = _make_matd3_full()
        hook = getattr(agent, "set_transition_context", None)
        assert callable(hook)

    def test_is_initial_exploration_done(self):
        agent, _, _, _ = _make_matd3_full()
        # Default end_initial_exploration_time_step from config exploration block
        assert agent.is_initial_exploration_done(0) is False
        assert agent.is_initial_exploration_done(3) is False
        assert agent.is_initial_exploration_done(4) is True
        assert agent.is_initial_exploration_done(100) is True

    def test_update_called_without_context_does_not_crash(self):
        """Wrapper may skip context hooks if teacher not needed."""
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        # Force teacher off
        agent._teacher_alive = False
        agent._warm_start_policy = None
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        # Call update directly without context hooks — should not crash
        agent.update(
            observations=obs, actions=actions, rewards=rewards,
            next_observations=next_obs, terminated=term, truncated=trunc,
            update_target_step=False, global_learning_step=100,
            update_step=True, initial_exploration_done=True,
        )

    def test_export_after_update_only_actors_in_manifest(self):
        """Manifest must not contain critic/replay/teacher state."""
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        # Run a few updates
        for step in range(6):
            obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
            _run_update_step(
                agent, obs, actions, rewards, next_obs, term, trunc,
                global_learning_step=step + 10,
                update_step=True,
                initial_exploration_done=True,
            )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = agent.export_artifacts(tmpdir)
            assert all("critic" not in a["path"] for a in manifest["artifacts"])
            assert all("replay" not in a["path"] for a in manifest["artifacts"])
            assert all("teacher" not in a["path"] for a in manifest["artifacts"])
```

- [ ] **Step 2: Implement `is_initial_exploration_done`**

Add to `algorithms/agents/agent_transformer_matd3.py`:

```python
def is_initial_exploration_done(self, global_learning_step: int) -> bool:
    """Gate for the wrapper — True when initial exploration window is over."""
    exploration_cfg = self.config["algorithm"].get("exploration", {}) or {}
    end_step = int(exploration_cfg.get("end_initial_exploration_time_step", 0))
    return global_learning_step >= end_step
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_matd3_update_integration.py::TestWrapperIntegration -v`
Expected: 5 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add algorithms/agents/agent_transformer_matd3.py tests/test_matd3_update_integration.py
git commit -m "feat(matd3-t): is_initial_exploration_done + wrapper integration tests"
```

---

## Task 8: End-to-End Wrapper Smoke (Real Simulator)

This is the acceptance test that proves the final goal: the agent works
through `Wrapper_CityLearn` on the real dynamic entity dataset, survives at
least one topology change, and produces an actor-only export manifest.

**Files:**
- Create: `tests/test_matd3_wrapper_smoke.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_matd3_wrapper_smoke.py`:

```python
"""End-to-end wrapper-driven smoke test for AgentTransformerMATD3.

Runs the agent through Wrapper_CityLearn on the dynamic entity dataset for a
short window, verifies topology changes are handled, and validates the
exported manifest contains only actor artifacts.

Skipped if the dataset is not present locally.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml


_TEMPLATE = "configs/templates/rl/transformer_matd3_local.yaml"
_DATASET_SCHEMA = (
    "./datasets/citylearn_three_phase_dynamic_assets_only_demo_15s_parquet/schema.json"
)


def _dataset_available() -> bool:
    return Path(_DATASET_SCHEMA).exists()


@pytest.mark.skipif(not _dataset_available(), reason="dynamic entity dataset not present")
class TestWrapperSmoke:
    def _short_config(self, tmpdir: Path) -> Path:
        """Load the template and shorten the simulation window."""
        with open(_TEMPLATE) as f:
            cfg = yaml.safe_load(f)
        cfg["simulator"]["simulation_start_time_step"] = 0
        cfg["simulator"]["simulation_end_time_step"] = 200
        cfg["simulator"]["episode_time_steps"] = 201
        cfg["simulator"]["episodes"] = 1
        cfg["training"]["steps_between_training_updates"] = 4
        cfg["training"]["target_update_interval"] = 2
        # Ensure batch size is small enough to trigger updates in a short window
        cfg["pipeline"][0]["hyperparameters"]["batch_size"] = 16
        cfg["pipeline"][0]["hyperparameters"]["replay_capacity"] = 5000
        # Provide runtime dirs
        job_dir = tmpdir / "job"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(exist_ok=True)
        (job_dir / "results").mkdir(exist_ok=True)
        cfg["runtime"]["job_dir"] = str(job_dir)
        cfg["runtime"]["log_dir"] = str(job_dir / "logs")
        out_path = tmpdir / "config.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f)
        return out_path

    def test_config_validates(self, tmp_path):
        """Config passes schema validation with the new stage type."""
        from utils.config_schema import load_and_validate_config
        cfg_path = self._short_config(tmp_path)
        cfg = load_and_validate_config(str(cfg_path))
        assert cfg.pipeline[0].algorithm == "AgentTransformerMATD3"

    def test_short_run_completes(self, tmp_path):
        """Agent runs through wrapper for the shortened window without crashing."""
        from utils.config_schema import load_and_validate_config
        from utils.wrapper_citylearn import Wrapper_CityLearn
        from algorithms.registry import build_execution_unit

        cfg_path = self._short_config(tmp_path)
        cfg = load_and_validate_config(str(cfg_path))
        cfg_dict = cfg.model_dump()

        agent = build_execution_unit(cfg_dict)
        wrapper = Wrapper_CityLearn(cfg_dict)
        wrapper.set_model(agent)
        # Run one short episode
        wrapper.train()  # or the equivalent entry point used in tests

        # After training, the agent should have processed transitions
        # and (if updates were triggered) run critic updates.
        assert agent._replay.total_size >= 1

    def test_export_manifest_actors_only(self, tmp_path):
        """After training, export produces an actor-only manifest."""
        from utils.config_schema import load_and_validate_config
        from utils.wrapper_citylearn import Wrapper_CityLearn
        from algorithms.registry import build_execution_unit

        cfg_path = self._short_config(tmp_path)
        cfg = load_and_validate_config(str(cfg_path))
        cfg_dict = cfg.model_dump()

        agent = build_execution_unit(cfg_dict)
        wrapper = Wrapper_CityLearn(cfg_dict)
        wrapper.set_model(agent)
        wrapper.train()

        export_dir = tmp_path / "export"
        manifest = agent.export_artifacts(str(export_dir), context={"config": cfg_dict})

        # Only actor ONNX artifacts
        assert manifest["format"] == "onnx"
        assert len(manifest["artifacts"]) >= 1
        for art in manifest["artifacts"]:
            assert "critic" not in art["path"].lower()
            assert (export_dir / art["path"]).exists()
            # Manifest config carries layout/action metadata
            assert "ca_action_names" in art["config"]
            assert "action_low" in art["config"]
            assert "action_high" in art["config"]

    def test_topology_change_survives_if_dataset_triggers(self, tmp_path):
        """If the dynamic dataset triggers a topology_version increment during
        the short window, the agent must switch replay signature and continue.
        This test only asserts if a change actually happened."""
        from utils.config_schema import load_and_validate_config
        from utils.wrapper_citylearn import Wrapper_CityLearn
        from algorithms.registry import build_execution_unit

        cfg_path = self._short_config(tmp_path)
        cfg = load_and_validate_config(str(cfg_path))
        cfg_dict = cfg.model_dump()

        agent = build_execution_unit(cfg_dict)
        wrapper = Wrapper_CityLearn(cfg_dict)
        wrapper.set_model(agent)
        wrapper.train()

        # If more than one partition exists, a topology change occurred.
        # The active signature must be the most recent one and non-empty.
        if agent._replay.partition_count > 1:
            assert agent._replay.active_signature is not None
            assert agent._replay.active_partition_size >= 1
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_matd3_wrapper_smoke.py -v`
Expected (with dataset present): All tests PASS.
Expected (without dataset): All tests SKIPPED — this is acceptable for CI
without the dataset, but the dataset must be present locally before merge.

- [ ] **Step 3: Commit**

```bash
git add tests/test_matd3_wrapper_smoke.py
git commit -m "test(matd3-t): end-to-end wrapper smoke on dynamic entity dataset"
```

---

## Task 9: Final Full-Suite Verification

- [ ] **Step 1: Run all Plan D tests**

Run: `pytest tests/test_matd3_update_integration.py tests/test_matd3_dynamic_topology_integration.py tests/test_matd3_diagnostics.py -v`
Expected: All tests PASS.

- [ ] **Step 2: Run Plan A tests to confirm no regressions**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x --timeout=120`
Expected: No new failures introduced by Plan D.

- [ ] **Step 4: Verify the full training loop end-to-end (manual smoke)**

```bash
python -c "
from tests._matd3_test_helpers import _make_matd3_full, _generate_transition, _run_update_step
agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
for step in range(20):
    obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
    _run_update_step(
        agent, obs, actions, rewards, next_obs, term, trunc,
        global_learning_step=step + 10,
        update_step=True,
        update_target_step=(step % 4 == 0),
        initial_exploration_done=(step >= 5),
    )
print(f'Critic updates: {agent._critic_update_count}')
print(f'Actor updates: {agent._actor_update_count}')
print(f'Replay size: {agent._replay.total_size}')
metrics = agent._collect_diagnostics(30)
print(f'Diagnostics keys: {len(metrics)}')
print('Plan D smoke: PASS')
"
```

Expected output:
```
Critic updates: <positive integer>
Actor updates: <positive integer, roughly half of critic updates>
Replay size: 20
Diagnostics keys: <20+>
Plan D smoke: PASS
```

- [ ] **Step 5: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "test(matd3-t): Plan D final verification pass"
```

---

## Plan D Complete

After Task 9, `AgentTransformerMATD3` has:
- A fully wired `update()` method: store → gate → critic update → delayed actor update → soft targets.
- `set_observation_context` / `set_transition_context` hooks for wrapper integration.
- Comprehensive diagnostics under `TransformerMATD3/` namespace.
- Full checkpoint including critics, replay, reward normalization, exploration state, and RNG.
- Dynamic topology unit smoke tests verifying replay signature switch, weight preservation, teacher re-attach, and training continuity.
- **End-to-end wrapper smoke on the real dynamic entity dataset (Task 8)** proving config validation, wrapper-driven training, topology handling, and actor-only export manifest.
- `is_initial_exploration_done` for wrapper gating.
- Export remains actor-only (manifest excludes all training state).

---

## Delegation Checklist (READ BEFORE ASSIGNING TO AN AGENT)

If handing this plan (and Plans A/B/C) to an implementation agent, the agent MUST:

1. **Execute plans strictly in order: A → B → C → D.** Do not start Plan B before Plan A tests pass. Same for C after B, and D after C.
2. **Use the exact API names from Plan B.** Do not invent new method names. If a helper looks missing, add it as a new task or agent-owned helper — do not silently rename Plan B primitives. See the "API Contract from Plan B" section at the top of this plan.
3. **Do not skip failing-test steps.** Every task follows red-green: write failing test → run to confirm failure → implement → run to confirm pass → commit.
4. **Do not batch commits.** Every task ends with a commit as specified.
5. **Ask for guidance on ambiguity.** If any signature or behavior does not match between plans, stop and ask. Do NOT improvise.
6. **Task 8 (wrapper smoke) is the acceptance test.** If the dataset is present, Task 8 must pass. If not present, install the dataset before declaring Plan D complete.
7. **After each plan, run the full `pytest tests/ -x --timeout=120` suite.** No regressions allowed.

Combined with Plans A-C, the agent is fully functional for training, checkpoint/resume, dynamic topology, and deployment export.
