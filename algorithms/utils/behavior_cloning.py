"""Demonstration storage and losses for Transformer PPO behavior cloning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from random import Random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch

from algorithms.utils.entity_token_layout import (
    BuildingTokenLayout,
    NfcExpression,
    TokenSegment,
)
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
        self._rejected_at_record = 0

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
            layout.excluded_feature_names,
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
        if copied_observation.shape != (self.full_representation_width(layout),):
            self._rejected_at_record += 1
            return
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
            "behavior_cloning_rejected_at_record": float(self._rejected_at_record),
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
            "rejected_at_record": self._rejected_at_record,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore training state after the attached teacher has been rebuilt."""
        self.validate_state_dict(
            state, max_samples_per_building=self.max_samples_per_building
        )
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
        self._rejected_at_record = int(state.get("rejected_at_record", 0))

    @staticmethod
    def validate_state_dict(
        state: Mapping[str, Any], *, max_samples_per_building: Optional[int] = None
    ) -> None:
        """Reject persisted state incompatible with the BC restore contract."""
        if not isinstance(state, Mapping):
            raise RuntimeError("Checkpoint BC state must be a mapping.")
        required_keys = (
            "demonstrations",
            "seen_per_building",
            "rng_state",
            "latest_bc_effective_weight",
            "latest_bc_loss",
            "latest_bc_weighted_loss",
            "latest_bc_valid_samples",
            "latest_pretraining_epochs",
            "latest_incompatible_demonstration_samples",
        )
        for key in required_keys:
            if key not in state:
                raise RuntimeError(f"Checkpoint BC state missing required key {key!r}.")
        demonstrations_by_building = state["demonstrations"]
        seen_per_building = state["seen_per_building"]
        if not isinstance(demonstrations_by_building, Mapping):
            raise RuntimeError("Checkpoint BC state has invalid demonstrations mapping.")
        if not isinstance(seen_per_building, Mapping):
            raise RuntimeError("Checkpoint BC state has invalid seen_per_building mapping.")
        if set(demonstrations_by_building) != set(seen_per_building):
            raise RuntimeError(
                "Checkpoint BC state has inconsistent reservoir building keys."
            )
        for building_idx, demonstrations in demonstrations_by_building.items():
            if (
                not isinstance(building_idx, Integral)
                or isinstance(building_idx, bool)
                or building_idx < 0
                or not isinstance(demonstrations, list)
            ):
                raise RuntimeError("Checkpoint BC state has invalid demonstrations entry.")
            if (
                max_samples_per_building is not None
                and len(demonstrations) > max_samples_per_building
            ):
                raise RuntimeError(
                    "Checkpoint BC state exceeds the reservoir sample capacity."
                )
            for demonstration in demonstrations:
                if not hasattr(demonstration, "encoded_length"):
                    raise RuntimeError(
                        "Checkpoint predates BC data contract. Re-collect demonstrations "
                        "under the current representation before resuming."
                    )
                if demonstration.observation.shape != (demonstration.encoded_length,):
                    raise RuntimeError(
                        "Checkpoint behavior-cloning demonstration observation shape "
                        f"{demonstration.observation.shape} does not match "
                        f"encoded_length {demonstration.encoded_length}."
                    )
                BehaviorCloningRegularizer._validate_layout(
                    demonstration.layout, demonstration.encoded_length
                )
                expected_observation_shape = (
                    BehaviorCloningRegularizer.full_representation_width(
                        demonstration.layout
                    ),
                )
                if demonstration.observation.shape != expected_observation_shape:
                    raise RuntimeError(
                        "Checkpoint behavior-cloning demonstration observation shape "
                        f"{demonstration.observation.shape} does not match stored "
                        f"layout width {expected_observation_shape}."
                    )
                if demonstration.encoded_length != expected_observation_shape[0]:
                    raise RuntimeError(
                        "Checkpoint behavior-cloning demonstration encoded_length "
                        f"{demonstration.encoded_length} does not match stored layout "
                        f"width {expected_observation_shape[0]}."
                    )
                if demonstration.target.shape != (demonstration.layout.n_ca,):
                    raise RuntimeError(
                        "Checkpoint behavior-cloning demonstration target shape "
                        f"{demonstration.target.shape} does not match stored layout "
                        f"CA count {(demonstration.layout.n_ca,)}."
                    )
                if not (
                    np.isfinite(demonstration.observation).all()
                    and np.isfinite(demonstration.target).all()
                ):
                    raise RuntimeError(
                        "Checkpoint behavior-cloning demonstration contains non-finite "
                        "observation or target values."
                    )
                if demonstration.layout_signature != BehaviorCloningRegularizer.layout_signature(
                    demonstration.layout
                ):
                    raise RuntimeError(
                        "Checkpoint behavior-cloning demonstration layout_signature "
                        "does not match its stored layout."
                    )
        for building_idx, seen in seen_per_building.items():
            if (
                not isinstance(building_idx, Integral)
                or isinstance(building_idx, bool)
                or building_idx < 0
                or not isinstance(seen, Integral)
                or isinstance(seen, bool)
                or seen < 0
            ):
                raise RuntimeError("Checkpoint BC state has invalid seen_per_building entry.")
            if seen < len(demonstrations_by_building[building_idx]):
                raise RuntimeError(
                    "Checkpoint BC state has reservoir seen count below stored "
                    "demonstrations."
                )
        try:
            Random().setstate(state["rng_state"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Checkpoint BC state has invalid sampler RNG state.") from exc
        numeric_keys = (
            "latest_bc_effective_weight",
            "latest_bc_loss",
            "latest_bc_weighted_loss",
            "latest_bc_valid_samples",
            "latest_pretraining_epochs",
            "latest_incompatible_demonstration_samples",
        )
        for key in numeric_keys:
            value = state[key]
            if (
                not isinstance(value, (Integral, float, np.floating))
                or isinstance(value, bool)
                or not np.isfinite(value)
            ):
                raise RuntimeError(f"Checkpoint BC state has invalid numeric field {key!r}.")
        if "rejected_at_record" in state:
            rejected_at_record = state["rejected_at_record"]
            if (
                not isinstance(rejected_at_record, Integral)
                or isinstance(rejected_at_record, bool)
                or rejected_at_record < 0
            ):
                raise RuntimeError(
                    "Checkpoint BC state has invalid rejection counter."
                )

    @staticmethod
    def _validate_layout(layout: BuildingTokenLayout, encoded_length: int) -> None:
        """Reject serialized layouts that would fail during tokenizer use."""
        if not isinstance(layout, BuildingTokenLayout):
            raise RuntimeError("Checkpoint contains invalid BC layout: wrong layout type.")
        if not isinstance(layout.building_id, str) or not layout.building_id:
            raise RuntimeError("Checkpoint contains invalid BC layout: invalid building_id.")
        if (
            not isinstance(layout.n_sro, Integral)
            or isinstance(layout.n_sro, bool)
            or layout.n_sro < 0
            or not isinstance(layout.n_ca, Integral)
            or isinstance(layout.n_ca, bool)
            or layout.n_ca < 0
        ):
            raise RuntimeError("Checkpoint contains invalid BC layout: invalid token counts.")
        if not isinstance(layout.segments, tuple):
            raise RuntimeError("Checkpoint contains invalid BC layout: segments must be a tuple.")
        if (
            not isinstance(layout.ca_action_names, tuple)
            or len(layout.ca_action_names) != layout.n_ca
            or any(not isinstance(name, str) or not name for name in layout.ca_action_names)
        ):
            raise RuntimeError("Checkpoint contains invalid BC layout: invalid CA action names.")
        if (
            not isinstance(layout.excluded_feature_names, tuple)
            or any(not isinstance(name, str) for name in layout.excluded_feature_names)
        ):
            raise RuntimeError("Checkpoint contains invalid BC layout: invalid excluded features.")

        seen_indices = set()
        n_sro = 0
        n_ca = 0
        n_nfc = 0
        before_nfc = True
        for segment in layout.segments:
            if not isinstance(segment, TokenSegment):
                raise RuntimeError("Checkpoint contains invalid BC layout: invalid segment.")
            if segment.family not in {"sro", "nfc", "ca"}:
                raise RuntimeError("Checkpoint contains invalid BC layout: invalid segment family.")
            if segment.family == "sro" and not before_nfc:
                raise RuntimeError("Checkpoint contains invalid BC layout: invalid segment order.")
            if segment.family == "nfc" and not before_nfc:
                raise RuntimeError("Checkpoint contains invalid BC layout: invalid segment order.")
            if segment.family == "ca" and before_nfc:
                raise RuntimeError("Checkpoint contains invalid BC layout: invalid segment order.")
            if not isinstance(segment.type_name, str) or not segment.type_name:
                raise RuntimeError("Checkpoint contains invalid BC layout: invalid segment type.")
            if segment.instance_id is not None and not isinstance(segment.instance_id, str):
                raise RuntimeError("Checkpoint contains invalid BC layout: invalid segment instance.")
            if (
                not isinstance(segment.feature_indices, tuple)
                or not segment.feature_indices
                or not isinstance(segment.feature_names, tuple)
                or len(segment.feature_names) != len(segment.feature_indices)
                or any(not isinstance(name, str) for name in segment.feature_names)
            ):
                raise RuntimeError("Checkpoint contains invalid BC layout: invalid segment features.")
            for index in segment.feature_indices:
                if (
                    not isinstance(index, Integral)
                    or isinstance(index, bool)
                    or index < 0
                    or index >= encoded_length
                    or index in seen_indices
                ):
                    raise RuntimeError(
                        "Checkpoint contains invalid BC layout: feature index is out of bounds "
                        "or duplicated."
                    )
                seen_indices.add(index)
            if segment.family == "nfc":
                n_nfc += 1
                before_nfc = False
                expression = segment.derived
                if (
                    not isinstance(expression, NfcExpression)
                    or expression.op != "subtract"
                    or len(segment.feature_indices) != 2
                    or not isinstance(expression.left_index_in_segment, Integral)
                    or isinstance(expression.left_index_in_segment, bool)
                    or not isinstance(expression.right_index_in_segment, Integral)
                    or isinstance(expression.right_index_in_segment, bool)
                    or expression.left_index_in_segment != 0
                    or expression.right_index_in_segment != 1
                ):
                    raise RuntimeError("Checkpoint contains invalid BC layout: invalid NFC segment.")
            elif segment.family == "sro":
                n_sro += 1
                if segment.derived is not None:
                    raise RuntimeError("Checkpoint contains invalid BC layout: invalid SRO segment.")
            else:
                n_ca += 1
                if segment.derived is not None:
                    raise RuntimeError("Checkpoint contains invalid BC layout: invalid CA segment.")
        if n_nfc != 1 or n_sro != layout.n_sro or n_ca != layout.n_ca:
            raise RuntimeError("Checkpoint contains invalid BC layout: segment counts disagree.")

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
    def full_representation_width(layout: BuildingTokenLayout) -> int:
        """Return full encoded width, not the last tokenizer-selected index.

        The wrapper preserves excluded features in the encoded observation, so
        selected indices can have gaps and excluded names can follow all tokens.
        """
        selected_feature_count = sum(
            len(segment.feature_indices) for segment in layout.segments
        )
        return selected_feature_count + len(layout.excluded_feature_names)

    @staticmethod
    def _copy_actions(actions: List[List[float]]) -> List[List[float]]:
        return [[float(value) for value in row] for row in actions]


__all__ = ["BehaviorCloningRegularizer", "Demonstration"]
