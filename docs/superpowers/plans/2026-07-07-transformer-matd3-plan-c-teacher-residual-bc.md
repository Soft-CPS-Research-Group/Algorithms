# AgentTransformerMATD3 — Plan C: Teacher Lifecycle, Residual, BC & Exploration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the warm-start teacher lifecycle (3 independent roles), residual policy composition, target policy smoothing, replay-native behavior cloning, and exploration noise/gating — all as standalone utility modules testable in isolation from the agent class.

**Architecture:** Four focused modules — teacher lifecycle manager, residual composition, BC loss, and exploration — each with complete unit tests. These modules will be integrated into `AgentTransformerMATD3` in Plan D.

**Tech Stack:** Python 3.10+, PyTorch, pytest.

**Spec:** `docs/transformer_matd3_spec.md` (sections: Warm-Start Policy Lifecycle, Target Policy Smoothing, Residual Policy Composition, Behavior Cloning, Initial Exploration Gating)

**Depends on:** Plan A (config schema, registry, actor head), Plan B (critic stacks, global token packer, replay partitions).

**Produces:** Four utility modules with full test coverage that implement all teacher/residual/BC/exploration logic. Plan D wires them into the agent's `predict` and `update` methods.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `algorithms/utils/matd3_teacher_lifecycle.py` (create) | Teacher policy lifecycle: 3 roles, attach, release, topology-change |
| `algorithms/utils/matd3_residual.py` (create) | Residual composition formula + target policy smoothing |
| `algorithms/utils/matd3_bc.py` (create) | Replay-native BC loss (weighted MSE with CA-type weights + decay) |
| `algorithms/utils/matd3_exploration.py` (create) | Exploration noise, sigma decay, exploration gating, phaseout |
| `tests/test_matd3_teacher_lifecycle.py` (create) | Teacher lifecycle tests |
| `tests/test_matd3_residual.py` (create) | Residual + smoothing tests |
| `tests/test_matd3_bc.py` (create) | BC loss tests |
| `tests/test_matd3_exploration.py` (create) | Exploration tests |

---

## Task 1: Teacher Lifecycle Manager

**Files:**
- Create: `algorithms/utils/matd3_teacher_lifecycle.py`
- Create: `tests/test_matd3_teacher_lifecycle.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_matd3_teacher_lifecycle.py
"""Tests for MATD3 teacher lifecycle (3 roles: exploration, residual, BC)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from algorithms.utils.matd3_teacher_lifecycle import TeacherLifecycleManager, TeacherRoleState


class TestTeacherRoleState:
    def test_initial_state_all_inactive(self):
        state = TeacherRoleState()
        assert state.exploration_active is False
        assert state.residual_active is False
        assert state.bc_active is False

    def test_any_active(self):
        state = TeacherRoleState(exploration_active=True)
        assert state.any_active() is True
        state2 = TeacherRoleState()
        assert state2.any_active() is False

    def test_all_combinations(self):
        for exp in (True, False):
            for res in (True, False):
                for bc in (True, False):
                    state = TeacherRoleState(
                        exploration_active=exp,
                        residual_active=res,
                        bc_active=bc,
                    )
                    assert state.any_active() == (exp or res or bc)


class TestTeacherLifecycleManager:
    def _make_manager(
        self,
        *,
        exploration_enabled: bool = True,
        residual_enabled: bool = True,
        bc_enabled: bool = True,
        phaseout_steps: int = 100,
        bc_weight: float = 1.0,
        bc_min_weight: float = 0.0,
        bc_decay_start_step: int = 0,
        bc_decay_steps: int = 200,
    ) -> TeacherLifecycleManager:
        return TeacherLifecycleManager(
            exploration_enabled=exploration_enabled,
            residual_enabled=residual_enabled,
            bc_enabled=bc_enabled,
            phaseout_steps=phaseout_steps,
            bc_weight=bc_weight,
            bc_min_weight=bc_min_weight,
            bc_decay_start_step=bc_decay_start_step,
            bc_decay_steps=bc_decay_steps,
            policy_name="RBCCommunityPolicy",
            policy_hyperparameters={},
            config_template={"algorithm": {"name": "dummy", "hyperparameters": {}}},
        )

    def test_teacher_alive_when_any_role_active(self):
        mgr = self._make_manager()
        assert mgr.is_teacher_needed(exploration_step=0, global_learning_step=0) is True

    def test_exploration_role_ends_after_phaseout(self):
        mgr = self._make_manager(
            residual_enabled=False, bc_enabled=False, phaseout_steps=50
        )
        # During phaseout
        assert mgr.is_exploration_role_active(exploration_step=25) is True
        # After phaseout
        assert mgr.is_exploration_role_active(exploration_step=50) is False
        assert mgr.is_exploration_role_active(exploration_step=51) is False

    def test_residual_role_independent_of_exploration(self):
        mgr = self._make_manager(exploration_enabled=False, bc_enabled=False)
        # Residual stays active indefinitely
        assert mgr.is_residual_role_active() is True

    def test_residual_role_inactive_when_disabled(self):
        mgr = self._make_manager(residual_enabled=False)
        assert mgr.is_residual_role_active() is False

    def test_bc_role_active_while_weight_positive(self):
        mgr = self._make_manager(
            exploration_enabled=False,
            residual_enabled=False,
            bc_weight=1.0,
            bc_decay_start_step=0,
            bc_decay_steps=100,
            bc_min_weight=0.0,
        )
        # Before full decay
        assert mgr.is_bc_role_active(global_learning_step=50) is True
        # After full decay (weight reaches 0)
        assert mgr.is_bc_role_active(global_learning_step=100) is False

    def test_bc_role_active_when_min_weight_positive(self):
        mgr = self._make_manager(
            exploration_enabled=False,
            residual_enabled=False,
            bc_weight=1.0,
            bc_min_weight=0.1,
            bc_decay_start_step=0,
            bc_decay_steps=100,
        )
        # Even after full decay, min_weight > 0 keeps role active
        assert mgr.is_bc_role_active(global_learning_step=200) is True

    def test_teacher_released_only_when_all_roles_inactive(self):
        mgr = self._make_manager(
            phaseout_steps=10,
            bc_weight=1.0,
            bc_min_weight=0.0,
            bc_decay_start_step=0,
            bc_decay_steps=50,
        )
        # Exploration done, BC still active, residual active
        assert mgr.is_teacher_needed(exploration_step=10, global_learning_step=30) is True
        # Exploration done, BC done, but residual still active
        assert mgr.is_teacher_needed(exploration_step=10, global_learning_step=50) is True

    def test_teacher_released_when_all_inactive(self):
        mgr = self._make_manager(
            residual_enabled=False,
            phaseout_steps=10,
            bc_weight=1.0,
            bc_min_weight=0.0,
            bc_decay_start_step=0,
            bc_decay_steps=50,
        )
        # All roles inactive
        assert mgr.is_teacher_needed(exploration_step=10, global_learning_step=50) is False

    def test_get_role_state_snapshot(self):
        mgr = self._make_manager(phaseout_steps=10)
        state = mgr.get_role_state(exploration_step=5, global_learning_step=0)
        assert state.exploration_active is True
        assert state.residual_active is True
        assert state.bc_active is True

    @patch("algorithms.utils.matd3_teacher_lifecycle.build_warm_start_policy")
    def test_attach_builds_teacher(self, mock_build):
        mock_policy = MagicMock()
        mock_build.return_value = mock_policy
        mgr = self._make_manager()
        mgr.attach(
            observation_names=[["obs1", "obs2"]],
            action_names=[["act1"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        assert mgr.teacher_policy is mock_policy
        mock_build.assert_called_once()

    @patch("algorithms.utils.matd3_teacher_lifecycle.build_warm_start_policy")
    def test_release_frees_teacher(self, mock_build):
        mock_build.return_value = MagicMock()
        mgr = self._make_manager()
        mgr.attach(
            observation_names=[["obs1"]],
            action_names=[["act1"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        assert mgr.teacher_policy is not None
        mgr.release()
        assert mgr.teacher_policy is None

    @patch("algorithms.utils.matd3_teacher_lifecycle.build_warm_start_policy")
    def test_try_release_only_when_all_roles_done(self, mock_build):
        mock_build.return_value = MagicMock()
        mgr = self._make_manager(
            residual_enabled=False,
            phaseout_steps=10,
            bc_weight=1.0,
            bc_min_weight=0.0,
            bc_decay_start_step=0,
            bc_decay_steps=50,
        )
        mgr.attach(
            observation_names=[["obs1"]],
            action_names=[["act1"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        # Still needed
        released = mgr.try_release(exploration_step=5, global_learning_step=20)
        assert released is False
        assert mgr.teacher_policy is not None
        # Now all done
        released = mgr.try_release(exploration_step=10, global_learning_step=50)
        assert released is True
        assert mgr.teacher_policy is None

    @patch("algorithms.utils.matd3_teacher_lifecycle.build_warm_start_policy")
    def test_already_released_teacher_not_reattached(self, mock_build):
        mock_build.return_value = MagicMock()
        mgr = self._make_manager(
            residual_enabled=False,
            bc_enabled=False,
            phaseout_steps=5,
        )
        mgr.attach(
            observation_names=[["obs1"]],
            action_names=[["act1"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        mgr.try_release(exploration_step=5, global_learning_step=999)
        assert mgr.teacher_policy is None
        # Should NOT re-attach
        assert mock_build.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_teacher_lifecycle.py -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Implement the teacher lifecycle manager**

Create `algorithms/utils/matd3_teacher_lifecycle.py`:

```python
"""Teacher policy lifecycle for AgentTransformerMATD3.

Manages three independent roles:
1. Exploration replacement/blending (finite, ends after phaseout_steps).
2. Residual baseline provider (indefinite while residual_policy_enabled).
3. BC target provider (finite, ends when bc_effective_weight decays to 0).

Teacher is released ONLY when ALL THREE roles are inactive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from algorithms.utils.warm_start_policy import build_warm_start_policy


@dataclass
class TeacherRoleState:
    """Snapshot of which teacher roles are currently active."""

    exploration_active: bool = False
    residual_active: bool = False
    bc_active: bool = False

    def any_active(self) -> bool:
        return self.exploration_active or self.residual_active or self.bc_active


class TeacherLifecycleManager:
    """Manages teacher policy attach/release across 3 independent roles.

    Parameters
    ----------
    exploration_enabled : bool
        Whether exploration replacement/blending uses the teacher.
    residual_enabled : bool
        Whether residual composition uses the teacher as baseline.
    bc_enabled : bool
        Whether behavior cloning uses teacher actions as targets.
    phaseout_steps : int
        Steps after which exploration role ends (finite).
    bc_weight : float
        Initial BC weight.
    bc_min_weight : float
        Floor for BC weight decay. If > 0, BC role stays active indefinitely.
    bc_decay_start_step : int
        Global learning step at which BC decay begins.
    bc_decay_steps : int
        Steps over which BC weight decays from initial to min_weight.
    policy_name : str
        Name of the teacher policy class.
    policy_hyperparameters : Mapping
        Hyperparameters for teacher policy construction.
    config_template : Mapping
        Config template for building the teacher policy.
    """

    def __init__(
        self,
        *,
        exploration_enabled: bool,
        residual_enabled: bool,
        bc_enabled: bool,
        phaseout_steps: int,
        bc_weight: float,
        bc_min_weight: float,
        bc_decay_start_step: int,
        bc_decay_steps: int,
        policy_name: str,
        policy_hyperparameters: Mapping[str, Any],
        config_template: Mapping[str, Any],
    ) -> None:
        self.exploration_enabled = exploration_enabled
        self.residual_enabled = residual_enabled
        self.bc_enabled = bc_enabled
        self.phaseout_steps = max(0, phaseout_steps)
        self.bc_weight = bc_weight
        self.bc_min_weight = bc_min_weight
        self.bc_decay_start_step = bc_decay_start_step
        self.bc_decay_steps = max(0, bc_decay_steps)
        self.policy_name = policy_name
        self.policy_hyperparameters = dict(policy_hyperparameters)
        self.config_template = dict(config_template)

        self.teacher_policy: Optional[Any] = None
        self._released: bool = False

    def is_exploration_role_active(self, exploration_step: int) -> bool:
        """Exploration role is active while step < phaseout_steps."""
        if not self.exploration_enabled:
            return False
        return exploration_step < self.phaseout_steps

    def is_residual_role_active(self) -> bool:
        """Residual role is indefinitely active while enabled."""
        return self.residual_enabled

    def is_bc_role_active(self, global_learning_step: int) -> bool:
        """BC role is active while effective weight > 0."""
        if not self.bc_enabled:
            return False
        return self._bc_effective_weight(global_learning_step) > 0.0

    def _bc_effective_weight(self, global_learning_step: int) -> float:
        """Compute current BC effective weight using linear decay."""
        if self.bc_weight <= 0.0:
            return 0.0
        if global_learning_step < self.bc_decay_start_step:
            return self.bc_weight
        if self.bc_decay_steps <= 0:
            return self.bc_weight

        progress = min(
            max(
                float(global_learning_step - self.bc_decay_start_step)
                / float(self.bc_decay_steps),
                0.0,
            ),
            1.0,
        )
        target = self.bc_min_weight
        return float(self.bc_weight + (target - self.bc_weight) * progress)

    def is_teacher_needed(
        self, *, exploration_step: int, global_learning_step: int
    ) -> bool:
        """True if any role still requires the teacher."""
        state = self.get_role_state(
            exploration_step=exploration_step,
            global_learning_step=global_learning_step,
        )
        return state.any_active()

    def get_role_state(
        self, *, exploration_step: int, global_learning_step: int
    ) -> TeacherRoleState:
        """Snapshot current role activity."""
        return TeacherRoleState(
            exploration_active=self.is_exploration_role_active(exploration_step),
            residual_active=self.is_residual_role_active(),
            bc_active=self.is_bc_role_active(global_learning_step),
        )

    def attach(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Build and attach the teacher policy."""
        if self._released:
            return
        self.teacher_policy = build_warm_start_policy(
            owner_name="AgentTransformerMATD3",
            policy_name=self.policy_name,
            policy_hyperparameters=self.policy_hyperparameters,
            config_template=self.config_template,
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )

    def reattach(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Re-attach teacher on topology change (if still alive)."""
        if self._released or self.teacher_policy is None:
            return
        self.teacher_policy = build_warm_start_policy(
            owner_name="AgentTransformerMATD3",
            policy_name=self.policy_name,
            policy_hyperparameters=self.policy_hyperparameters,
            config_template=self.config_template,
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )

    def release(self) -> None:
        """Release teacher policy and mark as permanently released."""
        self.teacher_policy = None
        self._released = True

    def try_release(
        self, *, exploration_step: int, global_learning_step: int
    ) -> bool:
        """Release teacher if all roles are inactive. Returns True if released."""
        if self._released:
            return True
        if self.is_teacher_needed(
            exploration_step=exploration_step,
            global_learning_step=global_learning_step,
        ):
            return False
        self.release()
        return True

    @property
    def is_alive(self) -> bool:
        """True if teacher is currently attached and not released."""
        return self.teacher_policy is not None and not self._released


__all__ = ["TeacherLifecycleManager", "TeacherRoleState"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matd3_teacher_lifecycle.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/matd3_teacher_lifecycle.py tests/test_matd3_teacher_lifecycle.py
git commit -m "feat(matd3-t): add teacher lifecycle manager with 3-role release logic"
```

---

## Task 2: Residual Policy Composition

**Files:**
- Create: `algorithms/utils/matd3_residual.py`
- Create: `tests/test_matd3_residual.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_matd3_residual.py
"""Tests for MATD3 residual policy composition."""
from __future__ import annotations

import torch
import pytest

from algorithms.utils.matd3_residual import (
    compose_residual_actions,
    scale_direct_actions,
    build_ca_type_scale_mask,
)


class TestResidualComposition:
    def test_basic_formula(self):
        """action = clip(teacher + 0.5 * span * scale * mask * actor, low, high)"""
        teacher = torch.tensor([[0.5, -0.3]])  # [B=1, K=2]
        actor = torch.tensor([[0.4, -0.6]])    # tanh output in [-1,1]
        span = torch.tensor([2.0, 2.0])        # high - low
        low = torch.tensor([-1.0, -1.0])
        high = torch.tensor([1.0, 1.0])
        scale = 0.5
        mask = torch.tensor([1.0, 1.0])

        result = compose_residual_actions(
            teacher_actions=teacher,
            actor_outputs=actor,
            action_span=span,
            action_low=low,
            action_high=high,
            residual_action_scale=scale,
            scale_mask=mask,
        )
        # Manual: 0.5 + 0.5 * 2.0 * 0.5 * 1.0 * 0.4 = 0.5 + 0.2 = 0.7
        #        -0.3 + 0.5 * 2.0 * 0.5 * 1.0 * (-0.6) = -0.3 - 0.3 = -0.6
        expected = torch.tensor([[0.7, -0.6]])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_clipping_at_bounds(self):
        """Result must be clipped to [low, high]."""
        teacher = torch.tensor([[0.9]])
        actor = torch.tensor([[1.0]])
        span = torch.tensor([2.0])
        low = torch.tensor([-1.0])
        high = torch.tensor([1.0])
        scale = 1.0
        mask = torch.tensor([1.0])

        result = compose_residual_actions(
            teacher_actions=teacher,
            actor_outputs=actor,
            action_span=span,
            action_low=low,
            action_high=high,
            residual_action_scale=scale,
            scale_mask=mask,
        )
        # 0.9 + 0.5 * 2.0 * 1.0 * 1.0 * 1.0 = 0.9 + 1.0 = 1.9 -> clipped to 1.0
        assert result.item() == pytest.approx(1.0)

    def test_per_ca_type_mask(self):
        """Different mask values for different CA types."""
        teacher = torch.tensor([[0.0, 0.0, 0.0]])
        actor = torch.tensor([[1.0, 1.0, 1.0]])
        span = torch.tensor([2.0, 2.0, 2.0])
        low = torch.tensor([-1.0, -1.0, -1.0])
        high = torch.tensor([1.0, 1.0, 1.0])
        scale = 1.0
        mask = torch.tensor([0.5, 1.0, 0.25])  # storage, charger, other

        result = compose_residual_actions(
            teacher_actions=teacher,
            actor_outputs=actor,
            action_span=span,
            action_low=low,
            action_high=high,
            residual_action_scale=scale,
            scale_mask=mask,
        )
        # 0 + 0.5 * 2 * 1.0 * mask_k * 1.0 = mask_k
        expected = torch.tensor([[0.5, 1.0, 0.25]])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_batch_dimension(self):
        """Works with batch > 1."""
        B, K = 4, 3
        teacher = torch.zeros(B, K)
        actor = torch.ones(B, K) * 0.5
        span = torch.full((K,), 2.0)
        low = torch.full((K,), -1.0)
        high = torch.full((K,), 1.0)
        scale = 0.4
        mask = torch.ones(K)

        result = compose_residual_actions(
            teacher_actions=teacher,
            actor_outputs=actor,
            action_span=span,
            action_low=low,
            action_high=high,
            residual_action_scale=scale,
            scale_mask=mask,
        )
        assert result.shape == (B, K)
        # 0 + 0.5 * 2.0 * 0.4 * 1.0 * 0.5 = 0.2
        assert torch.allclose(result, torch.full((B, K), 0.2), atol=1e-6)

    def test_zero_scale_returns_teacher(self):
        """With scale=0, output equals teacher (clipped)."""
        teacher = torch.tensor([[0.3, -0.7]])
        actor = torch.tensor([[0.9, -0.9]])
        span = torch.tensor([2.0, 2.0])
        low = torch.tensor([-1.0, -1.0])
        high = torch.tensor([1.0, 1.0])

        result = compose_residual_actions(
            teacher_actions=teacher,
            actor_outputs=actor,
            action_span=span,
            action_low=low,
            action_high=high,
            residual_action_scale=0.0,
            scale_mask=torch.ones(2),
        )
        assert torch.allclose(result, teacher, atol=1e-6)


class TestDirectScaling:
    def test_direct_scaling_formula(self):
        """action = low + 0.5 * (actor + 1) * span"""
        actor = torch.tensor([[0.0, 1.0, -1.0]])  # midpoint, max, min
        span = torch.tensor([2.0, 2.0, 2.0])
        low = torch.tensor([-1.0, -1.0, -1.0])
        high = torch.tensor([1.0, 1.0, 1.0])

        result = scale_direct_actions(
            actor_outputs=actor,
            action_span=span,
            action_low=low,
            action_high=high,
        )
        # actor=0  -> -1 + 0.5*(0+1)*2 = -1 + 1 = 0
        # actor=1  -> -1 + 0.5*(1+1)*2 = -1 + 2 = 1
        # actor=-1 -> -1 + 0.5*(-1+1)*2 = -1 + 0 = -1
        expected = torch.tensor([[0.0, 1.0, -1.0]])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_asymmetric_bounds(self):
        """Correctly handles non-symmetric bounds."""
        actor = torch.tensor([[0.0]])  # midpoint
        span = torch.tensor([4.0])    # high=3, low=-1
        low = torch.tensor([-1.0])
        high = torch.tensor([3.0])

        result = scale_direct_actions(
            actor_outputs=actor,
            action_span=span,
            action_low=low,
            action_high=high,
        )
        # -1 + 0.5 * (0+1) * 4 = -1 + 2 = 1
        assert result.item() == pytest.approx(1.0)


class TestBuildCATypeScaleMask:
    def test_default_multipliers(self):
        """All types get 1.0 with default multipliers."""
        ca_type_names = ["storage", "charger", "pv"]
        mask = build_ca_type_scale_mask(
            ca_type_names=ca_type_names,
            storage_multiplier=1.0,
            ev_multiplier=1.0,
        )
        assert mask.shape == (3,)
        assert torch.allclose(mask, torch.ones(3))

    def test_custom_multipliers(self):
        """Per-type multipliers applied correctly."""
        ca_type_names = ["storage", "charger", "charger", "pv"]
        mask = build_ca_type_scale_mask(
            ca_type_names=ca_type_names,
            storage_multiplier=0.5,
            ev_multiplier=2.0,
        )
        expected = torch.tensor([0.5, 2.0, 2.0, 1.0])
        assert torch.allclose(mask, expected)

    def test_empty_list(self):
        """Empty CA list returns empty tensor."""
        mask = build_ca_type_scale_mask(
            ca_type_names=[],
            storage_multiplier=0.5,
            ev_multiplier=2.0,
        )
        assert mask.shape == (0,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_residual.py -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Implement the residual module**

Create `algorithms/utils/matd3_residual.py`:

```python
"""Residual policy composition and direct action scaling for AgentTransformerMATD3.

Implements the residual composition formula:
    action[b,k] = clip(teacher[b,k] + 0.5 * span[k] * scale * mask[k] * actor[b,k],
                        low[k], high[k])

And the direct (no-teacher) scaling:
    action[b,k] = low[k] + 0.5 * (actor[b,k] + 1) * span[k]
"""
from __future__ import annotations

from typing import List

import torch


def compose_residual_actions(
    *,
    teacher_actions: torch.Tensor,
    actor_outputs: torch.Tensor,
    action_span: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    residual_action_scale: float,
    scale_mask: torch.Tensor,
) -> torch.Tensor:
    """Compose final actions using residual formula.

    Parameters
    ----------
    teacher_actions : Tensor [B, K]
        Teacher base actions for each CA token.
    actor_outputs : Tensor [B, K]
        Actor tanh outputs in [-1, 1].
    action_span : Tensor [K]
        Per-action span (high - low).
    action_low : Tensor [K]
        Per-action lower bound.
    action_high : Tensor [K]
        Per-action upper bound.
    residual_action_scale : float
        Current residual scale from growth schedule.
    scale_mask : Tensor [K]
        Per-CA-type multipliers.

    Returns
    -------
    Tensor [B, K]
        Final actions clipped to [low, high].
    """
    residual = 0.5 * action_span * residual_action_scale * scale_mask * actor_outputs
    composed = teacher_actions + residual
    return torch.clamp(composed, min=action_low, max=action_high)


def scale_direct_actions(
    *,
    actor_outputs: torch.Tensor,
    action_span: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    """Scale actor outputs directly when no teacher is present.

    Parameters
    ----------
    actor_outputs : Tensor [B, K]
        Actor tanh outputs in [-1, 1].
    action_span : Tensor [K]
        Per-action span (high - low).
    action_low : Tensor [K]
        Per-action lower bound.
    action_high : Tensor [K]
        Per-action upper bound.

    Returns
    -------
    Tensor [B, K]
        Actions in [low, high].
    """
    return action_low + 0.5 * (actor_outputs + 1.0) * action_span


def build_ca_type_scale_mask(
    *,
    ca_type_names: List[str],
    storage_multiplier: float,
    ev_multiplier: float,
) -> torch.Tensor:
    """Build per-CA-token scale mask from type names.

    Parameters
    ----------
    ca_type_names : list of str
        Type name for each CA token (e.g., "storage", "charger", "pv").
    storage_multiplier : float
        Scale multiplier for storage-type CA tokens.
    ev_multiplier : float
        Scale multiplier for charger/EV-type CA tokens.

    Returns
    -------
    Tensor [K]
        Per-token multiplier values.
    """
    multipliers: List[float] = []
    for type_name in ca_type_names:
        name_lower = type_name.lower()
        if name_lower == "storage":
            multipliers.append(storage_multiplier)
        elif name_lower == "charger":
            multipliers.append(ev_multiplier)
        else:
            multipliers.append(1.0)
    return torch.tensor(multipliers, dtype=torch.float32)


__all__ = [
    "compose_residual_actions",
    "scale_direct_actions",
    "build_ca_type_scale_mask",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matd3_residual.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/matd3_residual.py tests/test_matd3_residual.py
git commit -m "feat(matd3-t): add residual composition and direct action scaling"
```

---

## Task 3: Target Policy Smoothing

**Files:**
- Modify: `algorithms/utils/matd3_residual.py`
- Modify: `tests/test_matd3_residual.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_matd3_residual.py`:

```python
from algorithms.utils.matd3_residual import apply_target_policy_smoothing


class TestTargetPolicySmoothing:
    def test_no_noise_when_disabled(self):
        """Zero noise scale means output equals input (clipped)."""
        actions = torch.tensor([[0.5, -0.3]])
        span = torch.tensor([2.0, 2.0])
        low = torch.tensor([-1.0, -1.0])
        high = torch.tensor([1.0, 1.0])

        result = apply_target_policy_smoothing(
            target_actions=actions,
            action_span=span,
            action_low=low,
            action_high=high,
            target_policy_noise=0.0,
            target_policy_noise_clip=0.5,
        )
        assert torch.allclose(result, actions)

    def test_output_within_bounds(self):
        """Smoothed actions must stay within [low, high]."""
        torch.manual_seed(42)
        B, K = 100, 5
        actions = torch.rand(B, K) * 2 - 1  # [-1, 1]
        span = torch.full((K,), 2.0)
        low = torch.full((K,), -1.0)
        high = torch.full((K,), 1.0)

        result = apply_target_policy_smoothing(
            target_actions=actions,
            action_span=span,
            action_low=low,
            action_high=high,
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,
        )
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_noise_clip_respected(self):
        """Noise magnitude is bounded by noise_clip * span."""
        torch.manual_seed(0)
        B, K = 1000, 3
        actions = torch.zeros(B, K)
        span = torch.tensor([2.0, 2.0, 2.0])
        low = torch.full((K,), -10.0)  # Wide bounds so clipping doesn't mask noise clip
        high = torch.full((K,), 10.0)
        noise_clip = 0.3

        result = apply_target_policy_smoothing(
            target_actions=actions,
            action_span=span,
            action_low=low,
            action_high=high,
            target_policy_noise=1.0,  # Large noise
            target_policy_noise_clip=noise_clip,
        )
        # Maximum deviation = noise_clip * span = 0.3 * 2.0 = 0.6
        deviation = (result - actions).abs()
        assert deviation.max() <= noise_clip * span.max() + 1e-6

    def test_post_residual_application(self):
        """Smoothing applied AFTER residual composition."""
        torch.manual_seed(7)
        # First compose residual
        teacher = torch.tensor([[0.5, -0.2]])
        actor = torch.tensor([[0.3, 0.1]])
        span = torch.tensor([2.0, 2.0])
        low = torch.tensor([-1.0, -1.0])
        high = torch.tensor([1.0, 1.0])
        mask = torch.ones(2)

        composed = compose_residual_actions(
            teacher_actions=teacher,
            actor_outputs=actor,
            action_span=span,
            action_low=low,
            action_high=high,
            residual_action_scale=0.5,
            scale_mask=mask,
        )
        # Then apply smoothing
        smoothed = apply_target_policy_smoothing(
            target_actions=composed,
            action_span=span,
            action_low=low,
            action_high=high,
            target_policy_noise=0.1,
            target_policy_noise_clip=0.3,
        )
        # Result differs from composed (noise added) but stays in bounds
        assert smoothed.shape == composed.shape
        assert smoothed.min() >= -1.0
        assert smoothed.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_residual.py::TestTargetPolicySmoothing -v`
Expected: ImportError — `apply_target_policy_smoothing` not found.

- [ ] **Step 3: Implement target policy smoothing**

Add to `algorithms/utils/matd3_residual.py`:

```python
def apply_target_policy_smoothing(
    *,
    target_actions: torch.Tensor,
    action_span: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    target_policy_noise: float,
    target_policy_noise_clip: float,
) -> torch.Tensor:
    """Apply target policy smoothing in final action space.

    Adds clipped Gaussian noise scaled by action span, then clips to bounds.
    Applied AFTER residual composition (in final action space).

    Parameters
    ----------
    target_actions : Tensor [B, K]
        Target actions (post-residual-composition).
    action_span : Tensor [K]
        Per-action span (high - low).
    action_low : Tensor [K]
        Per-action lower bound.
    action_high : Tensor [K]
        Per-action upper bound.
    target_policy_noise : float
        Noise standard deviation scale factor.
    target_policy_noise_clip : float
        Clip factor for noise magnitude.

    Returns
    -------
    Tensor [B, K]
        Smoothed target actions clipped to [low, high].
    """
    if target_policy_noise <= 0.0:
        return torch.clamp(target_actions, min=action_low, max=action_high)

    noise = torch.randn_like(target_actions) * (target_policy_noise * action_span)
    if target_policy_noise_clip > 0.0:
        clip_bound = target_policy_noise_clip * action_span
        noise = torch.clamp(noise, min=-clip_bound, max=clip_bound)
    smoothed = target_actions + noise
    return torch.clamp(smoothed, min=action_low, max=action_high)
```

Update `__all__` to include `"apply_target_policy_smoothing"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matd3_residual.py -v`
Expected: All tests PASS (both residual and smoothing).

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/matd3_residual.py tests/test_matd3_residual.py
git commit -m "feat(matd3-t): add target policy smoothing in final action space"
```

---

## Task 4: Replay-Native Behavior Cloning Loss

**Files:**
- Create: `algorithms/utils/matd3_bc.py`
- Create: `tests/test_matd3_bc.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_matd3_bc.py
"""Tests for MATD3 replay-native behavior cloning loss."""
from __future__ import annotations

import torch
import pytest

from algorithms.utils.matd3_bc import (
    compute_bc_loss,
    compute_bc_effective_weight,
    compute_ca_type_weights,
)


class TestBCEffectiveWeight:
    def test_before_decay_start(self):
        w = compute_bc_effective_weight(
            global_learning_step=50,
            initial_weight=1.0,
            min_weight=0.0,
            decay_start_step=100,
            decay_steps=200,
        )
        assert w == pytest.approx(1.0)

    def test_at_decay_start(self):
        w = compute_bc_effective_weight(
            global_learning_step=100,
            initial_weight=1.0,
            min_weight=0.0,
            decay_start_step=100,
            decay_steps=200,
        )
        assert w == pytest.approx(1.0)

    def test_midway_through_decay(self):
        w = compute_bc_effective_weight(
            global_learning_step=200,
            initial_weight=1.0,
            min_weight=0.0,
            decay_start_step=100,
            decay_steps=200,
        )
        assert w == pytest.approx(0.5)

    def test_after_full_decay(self):
        w = compute_bc_effective_weight(
            global_learning_step=300,
            initial_weight=1.0,
            min_weight=0.0,
            decay_start_step=100,
            decay_steps=200,
        )
        assert w == pytest.approx(0.0)

    def test_respects_min_weight(self):
        w = compute_bc_effective_weight(
            global_learning_step=9999,
            initial_weight=1.0,
            min_weight=0.2,
            decay_start_step=0,
            decay_steps=100,
        )
        assert w == pytest.approx(0.2)

    def test_zero_initial_weight(self):
        w = compute_bc_effective_weight(
            global_learning_step=50,
            initial_weight=0.0,
            min_weight=0.0,
            decay_start_step=0,
            decay_steps=100,
        )
        assert w == pytest.approx(0.0)

    def test_zero_decay_steps(self):
        """No decay steps means weight stays at initial."""
        w = compute_bc_effective_weight(
            global_learning_step=9999,
            initial_weight=0.5,
            min_weight=0.0,
            decay_start_step=0,
            decay_steps=0,
        )
        assert w == pytest.approx(0.5)


class TestCATypeWeights:
    def test_default_all_ones(self):
        weights = compute_ca_type_weights(
            ca_type_names=["storage", "charger", "pv"],
            ev_multiplier=1.0,
            storage_multiplier=1.0,
        )
        assert torch.allclose(weights, torch.ones(3))

    def test_custom_multipliers(self):
        weights = compute_ca_type_weights(
            ca_type_names=["storage", "charger", "charger", "pv", "storage"],
            ev_multiplier=3.0,
            storage_multiplier=0.5,
        )
        expected = torch.tensor([0.5, 3.0, 3.0, 1.0, 0.5])
        assert torch.allclose(weights, expected)

    def test_empty_returns_empty(self):
        weights = compute_ca_type_weights(
            ca_type_names=[],
            ev_multiplier=2.0,
            storage_multiplier=0.5,
        )
        assert weights.shape == (0,)


class TestBCLoss:
    def test_basic_mse_loss(self):
        """BC loss = weighted MSE between actor and teacher."""
        actor_actions = torch.tensor([[0.5, 0.3], [0.2, -0.1]])  # [B=2, K=2]
        teacher_actions = torch.tensor([[0.4, 0.3], [0.0, -0.1]])  # [B=2, K=2]
        ca_type_weights = torch.tensor([1.0, 1.0])

        loss = compute_bc_loss(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            ca_type_weights=ca_type_weights,
            effective_weight=1.0,
        )
        # MSE = mean((0.1)^2, 0, (0.2)^2, 0) = (0.01 + 0 + 0.04 + 0) / 4 = 0.0125
        # With uniform weights: sum(weighted_sq) / sum(weights_expanded)
        # weighted_sq: [[0.01, 0], [0.04, 0]] -> sum = 0.05
        # weights expanded: [[1,1],[1,1]] -> sum = 4
        # raw_loss = 0.05 / 4 = 0.0125
        expected = 0.0125
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_ca_type_weighting(self):
        """Per-type weights affect loss magnitude."""
        actor_actions = torch.tensor([[1.0, 1.0]])
        teacher_actions = torch.tensor([[0.0, 0.0]])
        # Storage gets 0.5 weight, charger gets 2.0
        ca_type_weights = torch.tensor([0.5, 2.0])

        loss = compute_bc_loss(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            ca_type_weights=ca_type_weights,
            effective_weight=1.0,
        )
        # error^2 = [1.0, 1.0], weighted = [0.5, 2.0]
        # sum = 2.5, denominator = (0.5 + 2.0) * 1 = 2.5
        # raw_loss = 2.5 / 2.5 = 1.0
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_effective_weight_multiplied(self):
        """Loss is multiplied by effective_weight."""
        actor_actions = torch.tensor([[0.5]])
        teacher_actions = torch.tensor([[0.0]])
        ca_type_weights = torch.tensor([1.0])

        loss_full = compute_bc_loss(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            ca_type_weights=ca_type_weights,
            effective_weight=1.0,
        )
        loss_half = compute_bc_loss(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            ca_type_weights=ca_type_weights,
            effective_weight=0.5,
        )
        assert loss_half.item() == pytest.approx(loss_full.item() * 0.5, abs=1e-7)

    def test_zero_weight_returns_zero(self):
        """Zero effective weight means zero loss."""
        actor_actions = torch.tensor([[0.9, -0.8]])
        teacher_actions = torch.tensor([[0.0, 0.0]])
        ca_type_weights = torch.tensor([1.0, 1.0])

        loss = compute_bc_loss(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            ca_type_weights=ca_type_weights,
            effective_weight=0.0,
        )
        assert loss.item() == 0.0

    def test_gradient_flows_to_actor(self):
        """BC loss supports gradient computation for actor parameters."""
        actor_actions = torch.tensor([[0.5, 0.3]], requires_grad=True)
        teacher_actions = torch.tensor([[0.0, 0.0]])
        ca_type_weights = torch.tensor([1.0, 1.0])

        loss = compute_bc_loss(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            ca_type_weights=ca_type_weights,
            effective_weight=1.0,
        )
        loss.backward()
        assert actor_actions.grad is not None
        assert actor_actions.grad.shape == (1, 2)

    def test_batch_independence(self):
        """Each sample in batch contributes independently."""
        # Single sample
        actor_single = torch.tensor([[1.0, 0.0]])
        teacher_single = torch.tensor([[0.0, 0.0]])
        weights = torch.tensor([1.0, 1.0])

        loss_single = compute_bc_loss(
            actor_actions=actor_single,
            teacher_actions=teacher_single,
            ca_type_weights=weights,
            effective_weight=1.0,
        )
        # Two identical samples — same loss
        actor_double = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        teacher_double = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        loss_double = compute_bc_loss(
            actor_actions=actor_double,
            teacher_actions=teacher_double,
            ca_type_weights=weights,
            effective_weight=1.0,
        )
        assert loss_single.item() == pytest.approx(loss_double.item(), abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_bc.py -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Implement the BC module**

Create `algorithms/utils/matd3_bc.py`:

```python
"""Replay-native behavior cloning for AgentTransformerMATD3.

Unlike the on-policy BehaviorCloningRegularizer used by AgentTransformerPPO,
this module computes BC loss from teacher actions stored directly in the
replay buffer. Teacher actions are sampled alongside transitions — no
separate rollout buffer required.

Shared utilities with BehaviorCloningRegularizer:
- BC effective-weight decay schedule (extracted here as a pure function).
- CA-type weight computation (extracted here as a pure function).
"""
from __future__ import annotations

from typing import List

import torch


def compute_bc_effective_weight(
    *,
    global_learning_step: int,
    initial_weight: float,
    min_weight: float,
    decay_start_step: int,
    decay_steps: int,
) -> float:
    """Compute current BC effective weight using linear decay.

    Parameters
    ----------
    global_learning_step : int
        Current global learning step.
    initial_weight : float
        Starting BC weight.
    min_weight : float
        Floor value (weight never goes below this).
    decay_start_step : int
        Step at which decay begins.
    decay_steps : int
        Number of steps over which to decay from initial to min.

    Returns
    -------
    float
        Current effective weight in [min_weight, initial_weight].
    """
    if initial_weight <= 0.0:
        return 0.0
    if global_learning_step < decay_start_step:
        return float(initial_weight)
    if decay_steps <= 0:
        return float(initial_weight)

    progress = min(
        max(
            float(global_learning_step - decay_start_step) / float(decay_steps),
            0.0,
        ),
        1.0,
    )
    return float(initial_weight + (min_weight - initial_weight) * progress)


def compute_ca_type_weights(
    *,
    ca_type_names: List[str],
    ev_multiplier: float,
    storage_multiplier: float,
) -> torch.Tensor:
    """Build per-CA-token weights based on type names.

    Parameters
    ----------
    ca_type_names : list of str
        Type name for each CA token (e.g., "storage", "charger", "pv").
    ev_multiplier : float
        Weight multiplier for charger/EV-type tokens.
    storage_multiplier : float
        Weight multiplier for storage-type tokens.

    Returns
    -------
    Tensor [K]
        Per-token BC loss weights.
    """
    weights: List[float] = []
    for type_name in ca_type_names:
        name_lower = type_name.lower()
        if name_lower == "storage":
            weights.append(storage_multiplier)
        elif name_lower == "charger":
            weights.append(ev_multiplier)
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_bc_loss(
    *,
    actor_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    ca_type_weights: torch.Tensor,
    effective_weight: float,
) -> torch.Tensor:
    """Compute weighted MSE BC loss from replay-sampled teacher actions.

    Parameters
    ----------
    actor_actions : Tensor [B, K]
        Actor-predicted actions for the sampled batch.
    teacher_actions : Tensor [B, K]
        Teacher actions from replay (stored alongside transitions).
    ca_type_weights : Tensor [K]
        Per-CA-token type weights.
    effective_weight : float
        Current effective BC weight (after decay).

    Returns
    -------
    Tensor (scalar)
        Weighted BC loss: effective_weight * (sum(type_weights * sq_error) / sum(weights_expanded)).
    """
    if effective_weight <= 0.0:
        return actor_actions.new_tensor(0.0)

    squared_error = (actor_actions - teacher_actions).pow(2)
    # Broadcast weights [K] -> [1, K] -> [B, K]
    weights_expanded = ca_type_weights.view(1, -1).expand_as(squared_error)
    weighted_error = squared_error * weights_expanded
    denominator = weights_expanded.sum().clamp_min(1.0)
    raw_loss = weighted_error.sum() / denominator
    return raw_loss * effective_weight


__all__ = [
    "compute_bc_loss",
    "compute_bc_effective_weight",
    "compute_ca_type_weights",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matd3_bc.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/matd3_bc.py tests/test_matd3_bc.py
git commit -m "feat(matd3-t): add replay-native BC loss with per-CA-type weighting"
```

---

## Task 5: Exploration Noise and Gating

**Files:**
- Create: `algorithms/utils/matd3_exploration.py`
- Create: `tests/test_matd3_exploration.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_matd3_exploration.py
"""Tests for MATD3 exploration noise, gating, and phaseout."""
from __future__ import annotations

import torch
import pytest

from algorithms.utils.matd3_exploration import (
    ExplorationConfig,
    add_exploration_noise,
    compute_sigma,
    is_initial_exploration_done,
    should_train_on_step,
    compute_phaseout_probability,
    apply_exploration_phaseout,
)


class TestInitialExplorationGating:
    def test_not_done_before_threshold(self):
        assert is_initial_exploration_done(
            global_learning_step=99,
            end_initial_exploration_time_step=100,
        ) is False

    def test_done_at_threshold(self):
        assert is_initial_exploration_done(
            global_learning_step=100,
            end_initial_exploration_time_step=100,
        ) is True

    def test_done_after_threshold(self):
        assert is_initial_exploration_done(
            global_learning_step=200,
            end_initial_exploration_time_step=100,
        ) is True


class TestShouldTrainOnStep:
    def test_trains_when_exploration_done(self):
        assert should_train_on_step(
            initial_exploration_done=True,
            train_during_initial_exploration=False,
            global_learning_step=0,
            initial_exploration_training_start_step=0,
        ) is True

    def test_skips_during_exploration_when_disabled(self):
        assert should_train_on_step(
            initial_exploration_done=False,
            train_during_initial_exploration=False,
            global_learning_step=50,
            initial_exploration_training_start_step=0,
        ) is False

    def test_trains_during_exploration_when_enabled_and_past_start(self):
        assert should_train_on_step(
            initial_exploration_done=False,
            train_during_initial_exploration=True,
            global_learning_step=50,
            initial_exploration_training_start_step=30,
        ) is True

    def test_skips_during_exploration_before_start_step(self):
        assert should_train_on_step(
            initial_exploration_done=False,
            train_during_initial_exploration=True,
            global_learning_step=20,
            initial_exploration_training_start_step=30,
        ) is False


class TestSigmaDecay:
    def test_initial_sigma(self):
        sigma = compute_sigma(
            exploration_step=0,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=1000,
        )
        assert sigma == pytest.approx(0.3)

    def test_final_sigma(self):
        sigma = compute_sigma(
            exploration_step=1000,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=1000,
        )
        assert sigma == pytest.approx(0.05)

    def test_midway_sigma(self):
        sigma = compute_sigma(
            exploration_step=500,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=1000,
        )
        # Linear: 0.3 + (0.05 - 0.3) * 0.5 = 0.3 - 0.125 = 0.175
        assert sigma == pytest.approx(0.175)

    def test_past_decay_stays_at_final(self):
        sigma = compute_sigma(
            exploration_step=5000,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=1000,
        )
        assert sigma == pytest.approx(0.05)

    def test_zero_decay_steps_returns_initial(self):
        sigma = compute_sigma(
            exploration_step=100,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=0,
        )
        assert sigma == pytest.approx(0.3)


class TestExplorationNoise:
    def test_output_shape_preserved(self):
        torch.manual_seed(42)
        actions = torch.tensor([[0.5, -0.3, 0.0]])
        span = torch.tensor([2.0, 2.0, 2.0])
        low = torch.tensor([-1.0, -1.0, -1.0])
        high = torch.tensor([1.0, 1.0, 1.0])

        noisy = add_exploration_noise(
            actions=actions,
            action_span=span,
            action_low=low,
            action_high=high,
            sigma=0.2,
            noise_clip=0.5,
        )
        assert noisy.shape == actions.shape

    def test_output_within_bounds(self):
        torch.manual_seed(0)
        B, K = 100, 4
        actions = torch.rand(B, K) * 2 - 1
        span = torch.full((K,), 2.0)
        low = torch.full((K,), -1.0)
        high = torch.full((K,), 1.0)

        noisy = add_exploration_noise(
            actions=actions,
            action_span=span,
            action_low=low,
            action_high=high,
            sigma=0.5,
            noise_clip=1.0,
        )
        assert noisy.min() >= -1.0
        assert noisy.max() <= 1.0

    def test_zero_sigma_no_change(self):
        actions = torch.tensor([[0.5, -0.3]])
        span = torch.tensor([2.0, 2.0])
        low = torch.tensor([-1.0, -1.0])
        high = torch.tensor([1.0, 1.0])

        noisy = add_exploration_noise(
            actions=actions,
            action_span=span,
            action_low=low,
            action_high=high,
            sigma=0.0,
            noise_clip=0.5,
        )
        assert torch.allclose(noisy, actions)

    def test_noise_clip_bounds_noise(self):
        """Noise magnitude bounded by noise_clip * span."""
        torch.manual_seed(1)
        B, K = 1000, 2
        actions = torch.zeros(B, K)
        span = torch.tensor([2.0, 2.0])
        low = torch.full((K,), -10.0)
        high = torch.full((K,), 10.0)
        noise_clip = 0.25

        noisy = add_exploration_noise(
            actions=actions,
            action_span=span,
            action_low=low,
            action_high=high,
            sigma=5.0,  # Very large sigma
            noise_clip=noise_clip,
        )
        deviation = (noisy - actions).abs()
        assert deviation.max() <= noise_clip * span.max() + 1e-6


class TestPhaseoutProbability:
    def test_full_probability_at_start(self):
        p = compute_phaseout_probability(
            exploration_step=0,
            phaseout_steps=100,
        )
        assert p == pytest.approx(1.0)

    def test_zero_probability_at_end(self):
        p = compute_phaseout_probability(
            exploration_step=100,
            phaseout_steps=100,
        )
        assert p == pytest.approx(0.0)

    def test_linear_decay_midpoint(self):
        p = compute_phaseout_probability(
            exploration_step=50,
            phaseout_steps=100,
        )
        assert p == pytest.approx(0.5)

    def test_zero_phaseout_steps_returns_zero(self):
        p = compute_phaseout_probability(
            exploration_step=0,
            phaseout_steps=0,
        )
        assert p == pytest.approx(0.0)


class TestExplorationPhaseout:
    def test_deterministic_skips_phaseout(self):
        actor_actions = torch.tensor([[0.5, -0.3]])
        teacher_actions = torch.tensor([[0.0, 0.0]])
        result = apply_exploration_phaseout(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            phaseout_probability=0.9,
            mode="blend",
            deterministic=True,
        )
        assert torch.allclose(result, actor_actions)

    def test_blend_mode_interpolates(self):
        actor_actions = torch.tensor([[1.0, 0.0]])
        teacher_actions = torch.tensor([[0.0, 1.0]])
        result = apply_exploration_phaseout(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            phaseout_probability=0.5,
            mode="blend",
            deterministic=False,
        )
        # blend: 0.5 * teacher + 0.5 * actor
        expected = torch.tensor([[0.5, 0.5]])
        assert torch.allclose(result, expected)

    def test_blend_full_teacher(self):
        actor_actions = torch.tensor([[1.0]])
        teacher_actions = torch.tensor([[-1.0]])
        result = apply_exploration_phaseout(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            phaseout_probability=1.0,
            mode="blend",
            deterministic=False,
        )
        assert torch.allclose(result, teacher_actions)

    def test_blend_zero_probability_returns_actor(self):
        actor_actions = torch.tensor([[0.7, -0.2]])
        teacher_actions = torch.tensor([[0.0, 0.0]])
        result = apply_exploration_phaseout(
            actor_actions=actor_actions,
            teacher_actions=teacher_actions,
            phaseout_probability=0.0,
            mode="blend",
            deterministic=False,
        )
        assert torch.allclose(result, actor_actions)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_exploration.py -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Implement the exploration module**

Create `algorithms/utils/matd3_exploration.py`:

```python
"""Exploration noise, sigma decay, gating, and phaseout for AgentTransformerMATD3.

Implements:
- Initial exploration gating (is_initial_exploration_done, should_train_on_step).
- Sigma decay schedule for Gaussian exploration noise.
- Exploration noise with optional clip.
- Phaseout probability computation and application (blend/replace modes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ExplorationConfig:
    """Configuration container for exploration parameters."""

    sigma_initial: float = 0.3
    sigma_final: float = 0.05
    sigma_decay_steps: int = 10000
    noise_clip: float = 0.5
    end_initial_exploration_time_step: int = 0
    train_during_initial_exploration: bool = False
    initial_exploration_training_start_step: int = 0
    random_exploration_steps: int = 0
    phaseout_steps: int = 0
    phaseout_mode: str = "blend"


def is_initial_exploration_done(
    *,
    global_learning_step: int,
    end_initial_exploration_time_step: int,
) -> bool:
    """True when global_learning_step >= end_initial_exploration_time_step."""
    return global_learning_step >= end_initial_exploration_time_step


def should_train_on_step(
    *,
    initial_exploration_done: bool,
    train_during_initial_exploration: bool,
    global_learning_step: int,
    initial_exploration_training_start_step: int,
) -> bool:
    """Determine whether training updates should happen this step.

    Parameters
    ----------
    initial_exploration_done : bool
        Whether initial exploration period has ended.
    train_during_initial_exploration : bool
        Config flag allowing training during exploration.
    global_learning_step : int
        Current global step.
    initial_exploration_training_start_step : int
        Step at which training is allowed during exploration.

    Returns
    -------
    bool
        True if training should proceed.
    """
    if initial_exploration_done:
        return True
    if not train_during_initial_exploration:
        return False
    return global_learning_step >= initial_exploration_training_start_step


def compute_sigma(
    *,
    exploration_step: int,
    sigma_initial: float,
    sigma_final: float,
    sigma_decay_steps: int,
) -> float:
    """Compute current exploration sigma using linear decay.

    Parameters
    ----------
    exploration_step : int
        Current exploration step counter.
    sigma_initial : float
        Starting sigma value.
    sigma_final : float
        Final (floor) sigma value.
    sigma_decay_steps : int
        Steps over which sigma decays linearly.

    Returns
    -------
    float
        Current sigma in [sigma_final, sigma_initial].
    """
    if sigma_decay_steps <= 0:
        return sigma_initial
    progress = min(max(float(exploration_step) / float(sigma_decay_steps), 0.0), 1.0)
    return float(sigma_initial + (sigma_final - sigma_initial) * progress)


def add_exploration_noise(
    *,
    actions: torch.Tensor,
    action_span: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    sigma: float,
    noise_clip: float,
) -> torch.Tensor:
    """Add clipped Gaussian exploration noise to actions.

    Parameters
    ----------
    actions : Tensor [B, K]
        Actions to add noise to.
    action_span : Tensor [K]
        Per-action span (high - low).
    action_low : Tensor [K]
        Per-action lower bound.
    action_high : Tensor [K]
        Per-action upper bound.
    sigma : float
        Noise standard deviation (scaled by span).
    noise_clip : float
        Maximum noise magnitude as fraction of span.

    Returns
    -------
    Tensor [B, K]
        Noisy actions clipped to [low, high].
    """
    if sigma <= 0.0:
        return actions

    noise = torch.randn_like(actions) * (sigma * action_span)
    if noise_clip > 0.0:
        clip_bound = noise_clip * action_span
        noise = torch.clamp(noise, min=-clip_bound, max=clip_bound)
    noisy = actions + noise
    return torch.clamp(noisy, min=action_low, max=action_high)


def compute_phaseout_probability(
    *,
    exploration_step: int,
    phaseout_steps: int,
) -> float:
    """Compute teacher phaseout probability (linear decay from 1 to 0).

    Parameters
    ----------
    exploration_step : int
        Current exploration step (counts from phaseout start).
    phaseout_steps : int
        Total steps in phaseout window.

    Returns
    -------
    float
        Probability in [0, 1]. 1.0 = full teacher, 0.0 = no teacher.
    """
    if phaseout_steps <= 0:
        return 0.0
    return max(0.0, 1.0 - float(exploration_step) / float(phaseout_steps))


def apply_exploration_phaseout(
    *,
    actor_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    phaseout_probability: float,
    mode: str,
    deterministic: bool,
) -> torch.Tensor:
    """Apply exploration phaseout blending or replacement.

    Parameters
    ----------
    actor_actions : Tensor [B, K]
        Actor-predicted actions.
    teacher_actions : Tensor [B, K]
        Teacher-predicted actions.
    phaseout_probability : float
        Current teacher influence probability.
    mode : str
        "blend" for linear interpolation, "replace" for stochastic replacement.
    deterministic : bool
        If True, skip phaseout entirely (return actor actions).

    Returns
    -------
    Tensor [B, K]
        Phased-out actions.
    """
    if deterministic or phaseout_probability <= 0.0:
        return actor_actions

    if mode == "blend":
        return phaseout_probability * teacher_actions + (1.0 - phaseout_probability) * actor_actions

    # "replace" mode: stochastic per-batch-element replacement
    if phaseout_probability >= 1.0:
        return teacher_actions

    mask = torch.rand(actor_actions.shape[0], 1, device=actor_actions.device) < phaseout_probability
    return torch.where(mask.expand_as(actor_actions), teacher_actions, actor_actions)


__all__ = [
    "ExplorationConfig",
    "add_exploration_noise",
    "compute_sigma",
    "is_initial_exploration_done",
    "should_train_on_step",
    "compute_phaseout_probability",
    "apply_exploration_phaseout",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matd3_exploration.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/matd3_exploration.py tests/test_matd3_exploration.py
git commit -m "feat(matd3-t): add exploration noise, sigma decay, gating, and phaseout"
```

---

## Task 6: Teacher Topology-Change Integration

**Files:**
- Modify: `tests/test_matd3_teacher_lifecycle.py`
- Verify: `algorithms/utils/matd3_teacher_lifecycle.py`

- [ ] **Step 1: Write topology-change tests**

Append to `tests/test_matd3_teacher_lifecycle.py`:

```python
class TestTeacherTopologyChange:
    """Teacher lifecycle on topology changes."""

    @patch("algorithms.utils.matd3_teacher_lifecycle.build_warm_start_policy")
    def test_reattach_preserves_counters(self, mock_build):
        """Exploration/phaseout counters are NOT reset on topology change."""
        mock_build.return_value = MagicMock()
        mgr = TeacherLifecycleManager(
            exploration_enabled=True,
            residual_enabled=True,
            bc_enabled=False,
            phaseout_steps=100,
            bc_weight=0.0,
            bc_min_weight=0.0,
            bc_decay_start_step=0,
            bc_decay_steps=0,
            policy_name="RBCCommunityPolicy",
            policy_hyperparameters={},
            config_template={"algorithm": {"name": "dummy", "hyperparameters": {}}},
        )
        mgr.attach(
            observation_names=[["obs1"]],
            action_names=[["act1"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        # Simulate topology change with teacher alive
        mgr.reattach(
            observation_names=[["obs1", "obs2"]],
            action_names=[["act1", "act2"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        # Teacher is still alive — rebuild happened
        assert mgr.teacher_policy is not None
        assert mock_build.call_count == 2  # attach + reattach

    @patch("algorithms.utils.matd3_teacher_lifecycle.build_warm_start_policy")
    def test_reattach_skipped_if_released(self, mock_build):
        """Released teacher is NOT re-attached on topology change."""
        mock_build.return_value = MagicMock()
        mgr = TeacherLifecycleManager(
            exploration_enabled=False,
            residual_enabled=False,
            bc_enabled=False,
            phaseout_steps=0,
            bc_weight=0.0,
            bc_min_weight=0.0,
            bc_decay_start_step=0,
            bc_decay_steps=0,
            policy_name="RBCCommunityPolicy",
            policy_hyperparameters={},
            config_template={"algorithm": {"name": "dummy", "hyperparameters": {}}},
        )
        mgr.attach(
            observation_names=[["obs1"]],
            action_names=[["act1"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        mgr.release()  # Explicitly release
        mgr.reattach(
            observation_names=[["obs1", "obs2"]],
            action_names=[["act1", "act2"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        # Should NOT have rebuilt
        assert mgr.teacher_policy is None
        assert mock_build.call_count == 1  # Only the initial attach

    @patch("algorithms.utils.matd3_teacher_lifecycle.build_warm_start_policy")
    def test_reattach_with_residual_active_after_exploration_done(self, mock_build):
        """Teacher survives topology change when residual still active."""
        mock_build.return_value = MagicMock()
        mgr = TeacherLifecycleManager(
            exploration_enabled=True,
            residual_enabled=True,
            bc_enabled=False,
            phaseout_steps=10,
            bc_weight=0.0,
            bc_min_weight=0.0,
            bc_decay_start_step=0,
            bc_decay_steps=0,
            policy_name="RBCCommunityPolicy",
            policy_hyperparameters={},
            config_template={"algorithm": {"name": "dummy", "hyperparameters": {}}},
        )
        mgr.attach(
            observation_names=[["obs1"]],
            action_names=[["act1"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        # Exploration is done, but residual keeps teacher alive
        assert mgr.is_exploration_role_active(exploration_step=10) is False
        assert mgr.is_residual_role_active() is True
        assert mgr.is_teacher_needed(exploration_step=10, global_learning_step=999) is True

        # Topology change — teacher should be re-attached
        mgr.reattach(
            observation_names=[["obs_new"]],
            action_names=[["act_new"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        assert mgr.teacher_policy is not None
        assert mock_build.call_count == 2

    @patch("algorithms.utils.matd3_teacher_lifecycle.build_warm_start_policy")
    def test_is_alive_property(self, mock_build):
        """is_alive reflects current state correctly."""
        mock_build.return_value = MagicMock()
        mgr = TeacherLifecycleManager(
            exploration_enabled=True,
            residual_enabled=False,
            bc_enabled=False,
            phaseout_steps=5,
            bc_weight=0.0,
            bc_min_weight=0.0,
            bc_decay_start_step=0,
            bc_decay_steps=0,
            policy_name="RBCCommunityPolicy",
            policy_hyperparameters={},
            config_template={"algorithm": {"name": "dummy", "hyperparameters": {}}},
        )
        assert mgr.is_alive is False  # Not yet attached
        mgr.attach(
            observation_names=[["obs1"]],
            action_names=[["act1"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata=None,
        )
        assert mgr.is_alive is True
        mgr.release()
        assert mgr.is_alive is False
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_matd3_teacher_lifecycle.py::TestTeacherTopologyChange -v`
Expected: All tests PASS (implementation already handles this via `reattach`).

- [ ] **Step 3: Verify no regression on full lifecycle tests**

Run: `pytest tests/test_matd3_teacher_lifecycle.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_matd3_teacher_lifecycle.py
git commit -m "test(matd3-t): add teacher topology-change integration tests"
```

---

## Task 7: End-to-End Composition Verification

**Files:**
- Modify: `tests/test_matd3_residual.py`

- [ ] **Step 1: Write integration-level tests that chain all modules**

Append to `tests/test_matd3_residual.py`:

```python
from algorithms.utils.matd3_exploration import (
    add_exploration_noise,
    compute_sigma,
    apply_exploration_phaseout,
    compute_phaseout_probability,
)
from algorithms.utils.matd3_bc import compute_bc_loss, compute_bc_effective_weight, compute_ca_type_weights


class TestFullPipelineComposition:
    """Integration tests verifying correct ordering of operations."""

    def test_predict_with_residual_and_noise(self):
        """Simulate predict flow: residual compose -> phaseout -> noise."""
        torch.manual_seed(42)
        B, K = 2, 3
        # Actor output
        actor_out = torch.tanh(torch.randn(B, K))
        # Teacher actions
        teacher_actions = torch.tensor([[0.3, -0.2, 0.1], [0.0, 0.5, -0.3]])
        span = torch.full((K,), 2.0)
        low = torch.full((K,), -1.0)
        high = torch.full((K,), 1.0)
        mask = torch.tensor([0.5, 1.0, 1.0])  # storage, charger, other

        # Step 1: Residual composition
        composed = compose_residual_actions(
            teacher_actions=teacher_actions,
            actor_outputs=actor_out,
            action_span=span,
            action_low=low,
            action_high=high,
            residual_action_scale=0.3,
            scale_mask=mask,
        )
        assert composed.min() >= -1.0
        assert composed.max() <= 1.0

        # Step 2: Phaseout blending (early training)
        phaseout_prob = compute_phaseout_probability(
            exploration_step=25, phaseout_steps=100
        )
        assert phaseout_prob == pytest.approx(0.75)
        blended = apply_exploration_phaseout(
            actor_actions=composed,
            teacher_actions=teacher_actions,
            phaseout_probability=phaseout_prob,
            mode="blend",
            deterministic=False,
        )
        assert blended.shape == (B, K)

        # Step 3: Exploration noise
        sigma = compute_sigma(
            exploration_step=500,
            sigma_initial=0.3,
            sigma_final=0.05,
            sigma_decay_steps=2000,
        )
        noisy = add_exploration_noise(
            actions=blended,
            action_span=span,
            action_low=low,
            action_high=high,
            sigma=sigma,
            noise_clip=0.5,
        )
        assert noisy.min() >= -1.0
        assert noisy.max() <= 1.0

    def test_target_smoothing_post_residual(self):
        """Simulate target computation: target actor -> residual -> smooth."""
        torch.manual_seed(7)
        B, K = 4, 2
        target_actor_out = torch.tanh(torch.randn(B, K))
        teacher = torch.zeros(B, K)
        span = torch.full((K,), 2.0)
        low = torch.full((K,), -1.0)
        high = torch.full((K,), 1.0)
        mask = torch.ones(K)

        # Residual in target space
        target_composed = compose_residual_actions(
            teacher_actions=teacher,
            actor_outputs=target_actor_out,
            action_span=span,
            action_low=low,
            action_high=high,
            residual_action_scale=0.5,
            scale_mask=mask,
        )
        # Apply target smoothing
        smoothed = apply_target_policy_smoothing(
            target_actions=target_composed,
            action_span=span,
            action_low=low,
            action_high=high,
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,
        )
        assert smoothed.shape == (B, K)
        assert smoothed.min() >= -1.0
        assert smoothed.max() <= 1.0

    def test_bc_loss_at_actor_update(self):
        """Simulate actor update: actor output -> BC loss from replay teacher."""
        torch.manual_seed(99)
        B, K = 8, 4
        actor_actions = torch.tanh(torch.randn(B, K, requires_grad=True))
        teacher_from_replay = torch.rand(B, K) * 2 - 1
        ca_types = ["storage", "charger", "charger", "pv"]
        ca_weights = compute_ca_type_weights(
            ca_type_names=ca_types,
            ev_multiplier=2.0,
            storage_multiplier=0.5,
        )
        eff_weight = compute_bc_effective_weight(
            global_learning_step=150,
            initial_weight=1.0,
            min_weight=0.0,
            decay_start_step=100,
            decay_steps=200,
        )
        assert eff_weight == pytest.approx(0.75)

        bc_loss = compute_bc_loss(
            actor_actions=actor_actions,
            teacher_actions=teacher_from_replay,
            ca_type_weights=ca_weights,
            effective_weight=eff_weight,
        )
        assert bc_loss.item() > 0.0
        bc_loss.backward()
        assert actor_actions.grad is not None

    def test_direct_scaling_without_teacher(self):
        """When teacher is released, use direct scaling."""
        B, K = 3, 2
        actor_out = torch.tanh(torch.randn(B, K))
        span = torch.tensor([2.0, 2.0])
        low = torch.tensor([-1.0, -1.0])
        high = torch.tensor([1.0, 1.0])

        direct = scale_direct_actions(
            actor_outputs=actor_out,
            action_span=span,
            action_low=low,
            action_high=high,
        )
        # Must be within bounds
        assert direct.min() >= -1.0
        assert direct.max() <= 1.0
        # Monotonic mapping from actor space
        assert direct.shape == (B, K)
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_matd3_residual.py::TestFullPipelineComposition -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_matd3_residual.py
git commit -m "test(matd3-t): add end-to-end composition verification tests"
```

---

## Task 8: Final Verification

- [ ] **Step 1: Run all Plan C tests**

```bash
pytest tests/test_matd3_teacher_lifecycle.py tests/test_matd3_residual.py tests/test_matd3_bc.py tests/test_matd3_exploration.py -v
```

Expected: All tests PASS.

- [ ] **Step 2: Run existing tests to confirm no regressions**

```bash
pytest tests/ -x --timeout=120
```

Expected: No new failures.

- [ ] **Step 3: Verify imports**

```bash
python -c "
from algorithms.utils.matd3_teacher_lifecycle import TeacherLifecycleManager, TeacherRoleState
from algorithms.utils.matd3_residual import compose_residual_actions, scale_direct_actions, build_ca_type_scale_mask, apply_target_policy_smoothing
from algorithms.utils.matd3_bc import compute_bc_loss, compute_bc_effective_weight, compute_ca_type_weights
from algorithms.utils.matd3_exploration import ExplorationConfig, add_exploration_noise, compute_sigma, is_initial_exploration_done, should_train_on_step, compute_phaseout_probability, apply_exploration_phaseout
print('All Plan C modules importable')
"
```

Expected: `All Plan C modules importable`

- [ ] **Step 4: Final commit (if any cleanup needed)**

```bash
git add -A
git status  # Verify only Plan C files
git commit -m "feat(matd3-t): Plan C complete — teacher lifecycle, residual, BC, exploration"
```

---

## Plan C Complete

**Delivered:**
- `algorithms/utils/matd3_teacher_lifecycle.py` — Teacher lifecycle with 3-role release logic
- `algorithms/utils/matd3_residual.py` — Residual composition + target policy smoothing
- `algorithms/utils/matd3_bc.py` — Replay-native BC loss with per-CA-type weighting
- `algorithms/utils/matd3_exploration.py` — Exploration noise, sigma decay, gating, phaseout
- Full test coverage: lifecycle states, formulas, boundary conditions, gradient flow, pipeline composition

**Ready for Plan D:** Wire these modules into `AgentTransformerMATD3.predict()` and `AgentTransformerMATD3.update()`, connecting teacher actions to replay storage and BC loss to the actor update loop.
