"""Tests for MATD3 teacher lifecycle (3 roles: exploration, residual, BC)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

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
        mgr = self._make_manager(residual_enabled=False, bc_enabled=False, phaseout_steps=50)
        assert mgr.is_exploration_role_active(exploration_step=25) is True
        assert mgr.is_exploration_role_active(exploration_step=50) is False
        assert mgr.is_exploration_role_active(exploration_step=51) is False

    def test_residual_role_independent_of_exploration(self):
        mgr = self._make_manager(exploration_enabled=False, bc_enabled=False)
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
        assert mgr.is_bc_role_active(global_learning_step=50) is True
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
        assert mgr.is_bc_role_active(global_learning_step=200) is True

    def test_teacher_released_only_when_all_roles_inactive(self):
        mgr = self._make_manager(
            phaseout_steps=10,
            bc_weight=1.0,
            bc_min_weight=0.0,
            bc_decay_start_step=0,
            bc_decay_steps=50,
        )
        assert mgr.is_teacher_needed(exploration_step=10, global_learning_step=30) is True
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
        released = mgr.try_release(exploration_step=5, global_learning_step=20)
        assert released is False
        assert mgr.teacher_policy is not None
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
        assert mock_build.call_count == 1
