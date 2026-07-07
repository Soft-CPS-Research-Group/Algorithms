"""Teacher policy lifecycle for AgentTransformerMATD3."""
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
    """Manages teacher attach/release across exploration, residual, and BC roles."""

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
        self._released = False

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
        return float(self.bc_weight + (self.bc_min_weight - self.bc_weight) * progress)

    def is_teacher_needed(self, *, exploration_step: int, global_learning_step: int) -> bool:
        """True if any role still requires the teacher."""
        return self.get_role_state(
            exploration_step=exploration_step,
            global_learning_step=global_learning_step,
        ).any_active()

    def get_role_state(self, *, exploration_step: int, global_learning_step: int) -> TeacherRoleState:
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
        """Re-attach teacher on topology change if still alive."""
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

    def try_release(self, *, exploration_step: int, global_learning_step: int) -> bool:
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
