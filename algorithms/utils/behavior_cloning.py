"""Behavior-cloning support utilities for Transformer PPO."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

from algorithms.utils.warm_start_policy import build_warm_start_policy


class BehaviorCloningRegularizer:
    """Lifecycle and teacher-action buffer core for behavior cloning."""

    def __init__(
        self,
        *,
        enabled: bool,
        weight: float,
        min_weight: float,
        decay_start_step: int,
        decay_steps: int,
        ev_multiplier: float,
        storage_multiplier: float,
        policy: str,
        deterministic: bool,
        noise_scale: float,
        phaseout_steps: int,
        phaseout_mode: str,
        hyperparameters: Mapping[str, Any],
        agent_config_template: Mapping[str, Any],
    ) -> None:
        self.enabled = enabled
        self.weight = weight
        self.min_weight = min_weight
        self.decay_start_step = decay_start_step
        self.decay_steps = decay_steps
        self.ev_multiplier = ev_multiplier
        self.storage_multiplier = storage_multiplier

        self.policy = policy
        self.deterministic = deterministic
        self.noise_scale = noise_scale
        self.phaseout_steps = phaseout_steps
        self.phaseout_mode = phaseout_mode
        self.hyperparameters = deepcopy(dict(hyperparameters))
        self.agent_config_template = deepcopy(dict(agent_config_template))

        self.teacher_policy = None
        self.teacher_action_buffers: List[List[Optional[List[float]]]] = []
        self.latest_teacher_actions: Optional[List[List[float]]] = None

    @classmethod
    def from_config(
        cls,
        algorithm_cfg: Mapping[str, Any],
        agent_config_template: Mapping[str, Any],
    ) -> Optional["BehaviorCloningRegularizer"]:
        behavior_cloning = algorithm_cfg.get("behavior_cloning")
        if not isinstance(behavior_cloning, Mapping):
            return None
        if not bool(behavior_cloning.get("enabled", False)):
            return None

        warm_start = behavior_cloning.get("warm_start")
        if not isinstance(warm_start, Mapping):
            return None

        policy = warm_start.get("policy")
        if not policy:
            return None

        return cls(
            enabled=True,
            weight=float(behavior_cloning.get("weight", 0.0)),
            min_weight=float(behavior_cloning.get("min_weight", 0.0)),
            decay_start_step=int(behavior_cloning.get("decay_start_step", 0)),
            decay_steps=int(behavior_cloning.get("decay_steps", 0)),
            ev_multiplier=float(behavior_cloning.get("ev_multiplier", 1.0)),
            storage_multiplier=float(behavior_cloning.get("storage_multiplier", 1.0)),
            policy=str(policy),
            deterministic=bool(warm_start.get("deterministic", True)),
            noise_scale=float(warm_start.get("noise_scale", 0.0)),
            phaseout_steps=int(warm_start.get("phaseout_steps", 0)),
            phaseout_mode=str(warm_start.get("phaseout_mode", "probability")),
            hyperparameters=warm_start.get("hyperparameters") or {},
            agent_config_template=agent_config_template,
        )

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        self.teacher_policy = build_warm_start_policy(
            owner_name="AgentTransformerPPO",
            policy_name=self.policy,
            policy_hyperparameters=self.hyperparameters,
            config_template=self.agent_config_template,
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )
        self.teacher_action_buffers = [[] for _ in observation_names]

    def set_latest_teacher_actions(
        self, actions: Optional[List[List[float]]]
    ) -> None:
        if actions is None:
            self.latest_teacher_actions = None
            return
        self.latest_teacher_actions = [
            [float(value) for value in building_actions]
            for building_actions in actions
        ]

    def record_transition(self, building_idx: int) -> None:
        buffer = self._buffer_for(building_idx)
        if (
            self.latest_teacher_actions is None
            or building_idx >= len(self.latest_teacher_actions)
        ):
            buffer.append(None)
            return
        buffer.append(list(self.latest_teacher_actions[building_idx]))

    def teacher_action_for(
        self, building_idx: int, step_idx: int
    ) -> Optional[List[float]]:
        if building_idx < 0 or building_idx >= len(self.teacher_action_buffers):
            return None
        buffer = self.teacher_action_buffers[building_idx]
        if step_idx < 0 or step_idx >= len(buffer):
            return None
        action = buffer[step_idx]
        if action is None:
            return None
        return list(action)

    def on_buffer_flushed(self, building_idx: int) -> None:
        if 0 <= building_idx < len(self.teacher_action_buffers):
            self.teacher_action_buffers[building_idx].clear()

    def on_topology_change(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        self.teacher_action_buffers = []
        self.latest_teacher_actions = None
        self.attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )

    def snapshot_metrics(self) -> Dict[str, float]:
        return {
            "behavior_cloning_teacher_enabled": float(self.teacher_policy is not None),
            "behavior_cloning_latest_teacher_available": float(
                self.latest_teacher_actions is not None
            ),
            "behavior_cloning_teacher_buffer_size": float(
                sum(len(buffer) for buffer in self.teacher_action_buffers)
            ),
        }

    def _buffer_for(self, building_idx: int) -> List[Optional[List[float]]]:
        if building_idx < 0 or building_idx >= len(self.teacher_action_buffers):
            raise IndexError(f"Building index {building_idx} is out of range.")
        return self.teacher_action_buffers[building_idx]


__all__ = ["BehaviorCloningRegularizer"]
