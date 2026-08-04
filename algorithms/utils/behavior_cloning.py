"""Demonstration storage and losses for Transformer PPO behavior cloning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch

from algorithms.utils.entity_token_layout import BuildingTokenLayout
from algorithms.utils.warm_start_policy import build_warm_start_policy


@dataclass(frozen=True)
class Demonstration:
    """One immutable encoded observation and its teacher action target."""

    observation: np.ndarray
    encoded_length: int
    layout: BuildingTokenLayout
    layout_signature: Tuple[Any, ...]
    target: np.ndarray


class BehaviorCloningRegularizer:
    """Own teacher demonstrations and calculate demonstration-only BC losses."""

    def __init__(
        self,
        *,
        demonstration_episodes: int,
        max_samples_per_building: int,
        pretraining_epochs: int,
        batch_size: int,
        weight: float,
        min_weight: float,
        decay_start_step: int,
        decay_steps: int,
        ev_multiplier: float,
        storage_multiplier: float,
        policy: str,
        deterministic: bool,
        hyperparameters: Mapping[str, Any],
        agent_config_template: Mapping[str, Any],
        config_dict: Mapping[str, Any],
    ) -> None:
        self.demonstration_episodes = demonstration_episodes
        self.max_samples_per_building = max_samples_per_building
        self.pretraining_epochs = pretraining_epochs
        self.batch_size = batch_size
        self.weight = weight
        self.min_weight = min_weight
        self.decay_start_step = decay_start_step
        self.decay_steps = decay_steps
        self.ev_multiplier = ev_multiplier
        self.storage_multiplier = storage_multiplier
        self.policy = policy
        self.deterministic = deterministic
        self.hyperparameters = deepcopy(dict(hyperparameters))
        self.agent_config_template = deepcopy(dict(agent_config_template))
        self.config_dict = deepcopy(dict(config_dict))
        self.teacher_policy = None
        self._demonstrations: Dict[int, List[Demonstration]] = {}
        self._seen_per_building: Dict[int, int] = {}
        self._rng = Random(0)
        self._latest_bc_effective_weight = 0.0
        self._latest_bc_loss = 0.0
        self._latest_bc_weighted_loss = 0.0
        self._latest_bc_valid_samples = 0.0
        self._latest_pretraining_epochs = 0.0
        self._latest_incompatible_demonstration_samples = 0.0

    @classmethod
    def from_config(
        cls,
        algorithm_cfg: Mapping[str, Any],
        agent_config_template: Mapping[str, Any],
    ) -> Optional["BehaviorCloningRegularizer"]:
        config = algorithm_cfg.get("behavior_cloning")
        if not isinstance(config, Mapping) or not bool(config.get("enabled", False)):
            return None
        teacher = config.get("teacher")
        if not isinstance(teacher, Mapping) or not teacher.get("policy"):
            return None
        return cls(
            demonstration_episodes=int(config.get("demonstration_episodes", 0)),
            max_samples_per_building=int(config.get("max_samples_per_building", 4096)),
            pretraining_epochs=int(config.get("pretraining_epochs", 4)),
            batch_size=int(config.get("batch_size", 64)),
            weight=float(config.get("weight", 0.0)),
            min_weight=float(config.get("min_weight", 0.0)),
            decay_start_step=int(config.get("decay_start_step", 0)),
            decay_steps=int(config.get("decay_steps", 0)),
            ev_multiplier=float(config.get("ev_multiplier", 1.0)),
            storage_multiplier=float(config.get("storage_multiplier", 1.0)),
            policy=str(teacher["policy"]),
            deterministic=bool(teacher.get("deterministic", True)),
            hyperparameters=teacher.get("hyperparameters") or {},
            agent_config_template=agent_config_template,
            config_dict=config,
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
        self.teacher_policy = self._build_teacher_policy(
            observation_names, action_names, action_space, observation_space, metadata
        )

    def on_topology_change(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
        changed_buildings: Optional[Iterable[int]] = None,
    ) -> None:
        self.attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )

    def compute_teacher_actions(
        self, raw_or_encoded_observations: List[np.ndarray]
    ) -> List[List[float]]:
        if self.teacher_policy is None:
            raise RuntimeError("Behavior-cloning teacher is not attached.")
        return self._copy_actions(
            self.teacher_policy.predict(
                raw_or_encoded_observations, deterministic=self.deterministic
            )
        )

    @staticmethod
    def layout_signature(layout: BuildingTokenLayout) -> Tuple[Any, ...]:
        return (
            layout.n_sro,
            layout.n_ca,
            layout.ca_action_names,
            tuple(
                (
                    segment.family,
                    segment.type_name,
                    segment.instance_id,
                    segment.feature_indices,
                    segment.feature_names,
                )
                for segment in layout.segments
            ),
        )

    @property
    def demonstrations_by_signature(
        self,
    ) -> Dict[Tuple[Any, ...], Tuple[Demonstration, ...]]:
        grouped: Dict[Tuple[Any, ...], List[Demonstration]] = {}
        for demos in self._demonstrations.values():
            for demo in demos:
                grouped.setdefault(demo.layout_signature, []).append(demo)
        return {signature: tuple(demos) for signature, demos in grouped.items()}

    def demonstrations_for_building_by_signature(
        self, building_idx: int
    ) -> Dict[Tuple[Any, ...], Tuple[Demonstration, ...]]:
        grouped: Dict[Tuple[Any, ...], List[Demonstration]] = {}
        for demo in self._demonstrations.get(building_idx, []):
            grouped.setdefault(demo.layout_signature, []).append(demo)
        return {signature: tuple(demos) for signature, demos in grouped.items()}

    def record_demonstration(
        self,
        building_idx: int,
        observation: np.ndarray,
        layout: BuildingTokenLayout,
        target: List[float],
    ) -> None:
        copied_observation = np.asarray(observation, dtype=np.float32).copy()
        copied_target = np.asarray(target, dtype=np.float32).copy()
        if copied_target.shape != (layout.n_ca,) or not np.isfinite(copied_target).all():
            return
        copied_observation.setflags(write=False)
        copied_target.setflags(write=False)
        demo = Demonstration(
            observation=copied_observation,
            encoded_length=int(copied_observation.shape[0]),
            layout=deepcopy(layout),
            layout_signature=self.layout_signature(layout),
            target=copied_target,
        )
        demos = self._demonstrations.setdefault(building_idx, [])
        seen = self._seen_per_building.get(building_idx, 0) + 1
        self._seen_per_building[building_idx] = seen
        if len(demos) < self.max_samples_per_building:
            demos.append(demo)
            return
        replacement = self._rng.randrange(seen)
        if replacement < self.max_samples_per_building:
            demos[replacement] = demo

    def demonstration_count(self, building_idx: Optional[int] = None) -> int:
        if building_idx is None:
            return sum(len(demos) for demos in self._demonstrations.values())
        return len(self._demonstrations.get(building_idx, []))

    def sample_demonstrations(
        self, layout: BuildingTokenLayout, batch_size: int
    ) -> List[Demonstration]:
        compatible = self.demonstrations_by_signature.get(
            self.layout_signature(layout), ()
        )
        if len(compatible) <= batch_size:
            return list(compatible)
        return self._rng.sample(compatible, batch_size)

    def effective_weight(self, global_learning_step: int) -> float:
        if self.weight <= 0.0:
            return 0.0
        if global_learning_step < self.decay_start_step or self.decay_steps <= 0:
            return self.weight
        target = min(self.weight, self.min_weight)
        progress = min((global_learning_step - self.decay_start_step) / self.decay_steps, 1.0)
        return self.weight + (target - self.weight) * progress

    def ca_type_weights(
        self, layout: BuildingTokenLayout, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        weights = [
            self.ev_multiplier if segment.type_name == "charger" else
            self.storage_multiplier if segment.type_name == "storage" else 1.0
            for segment in layout.segments if segment.family == "ca"
        ]
        return torch.tensor(weights, dtype=dtype, device=device)

    def demonstration_loss(
        self,
        *,
        layout: BuildingTokenLayout,
        demonstrations: List[Demonstration],
        predicted_means: torch.Tensor,
        global_learning_step: int,
        apply_weight: bool = True,
    ) -> torch.Tensor:
        effective_weight = self.effective_weight(global_learning_step) if apply_weight else 1.0
        self._latest_bc_effective_weight = effective_weight if apply_weight else 1.0
        if not demonstrations or (apply_weight and effective_weight <= 0.0):
            self._set_loss_metrics(0.0, 0.0, 0.0)
            return predicted_means.new_tensor(0.0)
        targets = predicted_means.new_tensor(np.stack([demo.target for demo in demonstrations]))
        weights = self.ca_type_weights(
            layout, dtype=predicted_means.dtype, device=predicted_means.device
        ).view(1, -1)
        error = (predicted_means.squeeze(-1) - targets).pow(2) * weights
        raw_loss = error.sum() / weights.expand_as(error).sum().clamp_min(1.0)
        weighted_loss = raw_loss * effective_weight
        self._set_loss_metrics(
            float(raw_loss.detach().cpu()),
            float(weighted_loss.detach().cpu()),
            float(len(demonstrations)),
        )
        return weighted_loss

    def set_pretraining_epochs(self, epochs: int) -> None:
        self._latest_pretraining_epochs = float(epochs)

    def set_incompatible_demonstration_samples(self, samples: int) -> None:
        self._latest_incompatible_demonstration_samples = float(samples)

    def snapshot_metrics(self) -> Dict[str, float]:
        return {
            "behavior_cloning_teacher_enabled": float(self.teacher_policy is not None),
            "behavior_cloning_demonstration_samples": float(self.demonstration_count()),
            "behavior_cloning_effective_weight": self._latest_bc_effective_weight,
            "behavior_cloning_loss": self._latest_bc_loss,
            "behavior_cloning_weighted_loss": self._latest_bc_weighted_loss,
            "behavior_cloning_valid_samples": self._latest_bc_valid_samples,
            "behavior_cloning_pretraining_epochs": self._latest_pretraining_epochs,
            "behavior_cloning_incompatible_demonstration_samples": (
                self._latest_incompatible_demonstration_samples
            ),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return training state without serializing the live teacher policy."""
        return {
            "demonstrations": deepcopy(self._demonstrations),
            "seen_per_building": dict(self._seen_per_building),
            "rng_state": self._rng.getstate(),
            "latest_bc_effective_weight": self._latest_bc_effective_weight,
            "latest_bc_loss": self._latest_bc_loss,
            "latest_bc_weighted_loss": self._latest_bc_weighted_loss,
            "latest_bc_valid_samples": self._latest_bc_valid_samples,
            "latest_pretraining_epochs": self._latest_pretraining_epochs,
            "latest_incompatible_demonstration_samples": (
                self._latest_incompatible_demonstration_samples
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore training state after the attached teacher has been rebuilt."""
        self.validate_state_dict(state)
        self._demonstrations = deepcopy(state["demonstrations"])
        self._seen_per_building = dict(state["seen_per_building"])
        self._rng.setstate(state["rng_state"])
        self._latest_bc_effective_weight = float(state["latest_bc_effective_weight"])
        self._latest_bc_loss = float(state["latest_bc_loss"])
        self._latest_bc_weighted_loss = float(state["latest_bc_weighted_loss"])
        self._latest_bc_valid_samples = float(state["latest_bc_valid_samples"])
        self._latest_pretraining_epochs = float(state["latest_pretraining_epochs"])
        self._latest_incompatible_demonstration_samples = float(
            state["latest_incompatible_demonstration_samples"]
        )

    @staticmethod
    def validate_state_dict(state: Mapping[str, Any]) -> None:
        """Reject stored demonstrations from before the encoded-length contract."""
        for demonstrations in state["demonstrations"].values():
            for demonstration in demonstrations:
                if not hasattr(demonstration, "encoded_length"):
                    raise RuntimeError(
                        "Checkpoint predates BC data contract. Re-collect demonstrations "
                        "under the current representation before resuming."
                    )

    def _build_teacher_policy(self, observation_names, action_names, action_space, observation_space, metadata):
        return build_warm_start_policy(
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

    def _set_loss_metrics(self, raw_loss: float, weighted_loss: float, samples: float) -> None:
        self._latest_bc_loss = raw_loss
        self._latest_bc_weighted_loss = weighted_loss
        self._latest_bc_valid_samples = samples

    @staticmethod
    def _copy_actions(actions: List[List[float]]) -> List[List[float]]:
        return [[float(value) for value in row] for row in actions]


__all__ = ["BehaviorCloningRegularizer", "Demonstration"]
