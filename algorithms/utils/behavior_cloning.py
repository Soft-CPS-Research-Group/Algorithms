"""Behavior-cloning support utilities for Transformer PPO."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from algorithms.utils.entity_token_layout import BuildingTokenLayout
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
        self.phaseout_step = 0
        self._latest_bc_effective_weight = 0.0
        self._latest_bc_loss = 0.0
        self._latest_bc_weighted_loss = 0.0
        self._latest_bc_valid_samples = 0.0
        self._latest_phaseout_probability = 0.0
        self._latest_phaseout_used = False

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

    def compute_teacher_actions(
        self, raw_or_encoded_observations: List[np.ndarray]
    ) -> Optional[List[List[float]]]:
        if self.teacher_policy is None:
            self.set_latest_teacher_actions(None)
            return None

        actions = self.teacher_policy.predict(
            raw_or_encoded_observations,
            deterministic=self.deterministic,
        )
        normalized = self._copy_actions(actions)
        if self.noise_scale > 0.0:
            normalized = self._add_teacher_noise(normalized)
        self.set_latest_teacher_actions(normalized)
        return self._copy_actions(self.latest_teacher_actions or [])

    def effective_weight(self, global_learning_step: int) -> float:
        if self.weight <= 0.0:
            return 0.0
        if global_learning_step < self.decay_start_step:
            return float(self.weight)
        if self.decay_steps <= 0:
            return float(self.weight)

        target = min(float(self.weight), float(self.min_weight))
        progress = min(
            max(
                float(global_learning_step - self.decay_start_step)
                / float(self.decay_steps),
                0.0,
            ),
            1.0,
        )
        return float(self.weight + (target - self.weight) * progress)

    def ca_type_weights(
        self,
        layout: BuildingTokenLayout,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        weights: List[float] = []
        for segment in layout.segments:
            if segment.family != "ca":
                continue
            if segment.type_name == "charger":
                weights.append(float(self.ev_multiplier))
            elif segment.type_name == "storage":
                weights.append(float(self.storage_multiplier))
            else:
                weights.append(1.0)
        return torch.tensor(weights[: layout.n_ca], dtype=dtype, device=device)

    def bc_loss_term(
        self,
        *,
        building_idx: int,
        layout: BuildingTokenLayout,
        predicted_means: torch.Tensor,
        step_indices: torch.Tensor,
        global_learning_step: int,
    ) -> torch.Tensor:
        effective_weight = self.effective_weight(global_learning_step)
        self._latest_bc_effective_weight = float(effective_weight)
        if effective_weight <= 0.0:
            self._set_bc_loss_diagnostics(0.0, 0.0, 0.0)
            return predicted_means.new_tensor(0.0)

        valid_batch_indices: List[int] = []
        teacher_rows: List[List[float]] = []
        for batch_idx, step_idx in enumerate(
            step_indices.detach().cpu().view(-1).tolist()
        ):
            teacher_action = self.teacher_action_for(building_idx, int(step_idx))
            if teacher_action is None or len(teacher_action) != layout.n_ca:
                continue
            teacher_array = np.asarray(teacher_action, dtype=np.float32)
            if (
                teacher_array.shape != (layout.n_ca,)
                or not np.isfinite(teacher_array).all()
            ):
                continue
            valid_batch_indices.append(batch_idx)
            teacher_rows.append(teacher_array.astype(np.float32).tolist())

        valid_samples = float(len(valid_batch_indices))
        if not valid_batch_indices:
            self._set_bc_loss_diagnostics(0.0, 0.0, 0.0)
            return predicted_means.new_tensor(0.0)

        index = torch.as_tensor(
            valid_batch_indices,
            dtype=torch.long,
            device=predicted_means.device,
        )
        predicted = predicted_means.index_select(0, index).squeeze(-1)
        teacher = predicted_means.new_tensor(teacher_rows)
        ca_weights = self.ca_type_weights(
            layout,
            dtype=predicted_means.dtype,
            device=predicted_means.device,
        ).view(1, -1)
        weighted_error = (predicted - teacher).pow(2) * ca_weights
        denominator = ca_weights.expand_as(predicted).sum().clamp_min(1.0)
        raw_loss = weighted_error.sum() / denominator
        weighted_loss = raw_loss * float(effective_weight)
        self._set_bc_loss_diagnostics(
            float(raw_loss.detach().cpu().item()),
            float(weighted_loss.detach().cpu().item()),
            valid_samples,
        )
        return weighted_loss

    def maybe_phaseout(
        self,
        actor_actions: List[List[float]],
        deterministic: bool,
    ) -> List[List[float]]:
        self._latest_phaseout_probability = 0.0
        self._latest_phaseout_used = False
        if deterministic:
            return actor_actions

        self.phaseout_step += 1
        if self.latest_teacher_actions is None or self.phaseout_steps <= 0:
            return actor_actions

        probability = max(
            0.0,
            1.0 - float(self.phaseout_step) / float(self.phaseout_steps),
        )
        self._latest_phaseout_probability = probability
        if probability <= 0.0:
            return actor_actions

        if self.phaseout_mode == "blend":
            blended = self._blend_actions(
                actor_actions,
                self.latest_teacher_actions,
                probability,
            )
            self._latest_phaseout_used = blended is not actor_actions
            return blended

        if random.random() < probability:
            actions = self._replace_compatible_actions(
                actor_actions,
                self.latest_teacher_actions,
            )
            self._latest_phaseout_used = actions is not actor_actions
            return actions
        return actor_actions

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
            "behavior_cloning_effective_weight": float(self._latest_bc_effective_weight),
            "behavior_cloning_loss": float(self._latest_bc_loss),
            "behavior_cloning_weighted_loss": float(self._latest_bc_weighted_loss),
            "behavior_cloning_valid_samples": float(self._latest_bc_valid_samples),
            "behavior_cloning_phaseout_probability": float(
                self._latest_phaseout_probability
            ),
            "behavior_cloning_phaseout_used": float(self._latest_phaseout_used),
        }

    def _buffer_for(self, building_idx: int) -> List[Optional[List[float]]]:
        if building_idx < 0 or building_idx >= len(self.teacher_action_buffers):
            raise IndexError(f"Building index {building_idx} is out of range.")
        return self.teacher_action_buffers[building_idx]

    def _add_teacher_noise(self, actions: List[List[float]]) -> List[List[float]]:
        noisy: List[List[float]] = []
        for building_actions in actions:
            values = np.asarray(building_actions, dtype=np.float32)
            noise = np.random.normal(0.0, self.noise_scale, size=values.shape)
            noisy.append(
                np.clip(values + noise, -1.0, 1.0).astype(np.float32).tolist()
            )
        return noisy

    def _blend_actions(
        self,
        actor_actions: List[List[float]],
        teacher_actions: List[List[float]],
        probability: float,
    ) -> List[List[float]]:
        blended: List[List[float]] = []
        used_teacher = False
        for building_idx, actor_building_actions in enumerate(actor_actions):
            if building_idx >= len(teacher_actions):
                blended.append(actor_building_actions)
                continue
            actor = np.asarray(actor_building_actions, dtype=np.float32)
            teacher = np.asarray(teacher_actions[building_idx], dtype=np.float32)
            if actor.shape != teacher.shape:
                blended.append(actor_building_actions)
                continue
            values = probability * teacher + (1.0 - probability) * actor
            blended.append(np.clip(values, -1.0, 1.0).astype(np.float32).tolist())
            used_teacher = True
        if not used_teacher:
            return actor_actions
        return blended

    def _replace_compatible_actions(
        self,
        actor_actions: List[List[float]],
        teacher_actions: List[List[float]],
    ) -> List[List[float]]:
        replaced: List[List[float]] = []
        used_teacher = False
        for building_idx, actor_building_actions in enumerate(actor_actions):
            if building_idx >= len(teacher_actions):
                replaced.append(actor_building_actions)
                continue
            teacher_building_actions = teacher_actions[building_idx]
            if len(teacher_building_actions) != len(actor_building_actions):
                replaced.append(actor_building_actions)
                continue
            replaced.append([float(value) for value in teacher_building_actions])
            used_teacher = True
        if not used_teacher:
            return actor_actions
        return replaced

    def _set_bc_loss_diagnostics(
        self,
        raw_loss: float,
        weighted_loss: float,
        valid_samples: float,
    ) -> None:
        self._latest_bc_loss = float(raw_loss)
        self._latest_bc_weighted_loss = float(weighted_loss)
        self._latest_bc_valid_samples = float(valid_samples)

    @staticmethod
    def _copy_actions(actions: List[List[float]]) -> List[List[float]]:
        return [
            [float(value) for value in building_actions]
            for building_actions in actions
        ]


__all__ = ["BehaviorCloningRegularizer"]
