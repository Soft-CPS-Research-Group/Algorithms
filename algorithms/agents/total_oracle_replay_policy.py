"""Strict replay of complete perfect-foresight electricity schedules.

The policy deliberately has no rule-based fallback.  Every attached action
must have one semantic schedule series in the same building-local action
group.  Power schedules are converted to CityLearn's normalized actions by
using raw asset power observations, schedule metadata, or the explicit
attached-metadata fallback documented by :attr:`POWER_METADATA_KEY`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from algorithms.agents.base_agent import BaseAgent
from algorithms.oracles import SemanticActionSeries, SemanticSchedule


_TOLERANCE = 1.0e-8
_ACTION_BOUND_TOLERANCE = 1.0e-6
_DEADBAND_NUDGE_KW = 1.0e-6


class TotalOracleReplayPolicy(BaseAgent):
    """Replay all storage, EV/V2G and deferrable decisions without a teacher.

    The optional attached-environment metadata fallback is::

        {
          "total_oracle_replay": {
            "action_power_limits_kw": {
              "Building_1": {
                "electrical_storage": {"nominal_power_kw": 5.0},
                "electric_vehicle_storage_charger_1_1": {
                  "max_charging_power_kw": 11.0,
                  "max_discharging_power_kw": 7.2
                }
              }
            }
          }
        }

    EV limits emitted in ``schedule.metadata["action_power_limits_kw"]`` are
    also understood.  Raw observations take precedence over attached metadata,
    which takes precedence over schedule metadata.  The replay horizon is
    consumed once: unlike the older fixed-service evaluation policy, it never
    wraps silently into a second episode.
    """

    _use_raw_observations: bool = True
    POWER_METADATA_KEY = "total_oracle_replay.action_power_limits_kw"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        hyperparameters = dict(
            (config.get("algorithm", {}).get("hyperparameters") or {})
        )
        raw_path = str(hyperparameters.get("schedule_path") or "").strip()
        if not raw_path:
            raise ValueError("TotalOracleReplayPolicy requires schedule_path.")

        self.schedule_path = Path(raw_path).resolve()
        self.allow_attached_action_subset = bool(
            hyperparameters.get("allow_attached_action_subset", False)
        )
        self.repeat_schedule_for_training = bool(
            hyperparameters.get("repeat_schedule_for_training", False)
        )
        payload = self.schedule_path.read_bytes()
        self.schedule_sha256 = hashlib.sha256(payload).hexdigest()
        self.schedule = SemanticSchedule.from_json(payload.decode("utf-8"))
        self._series: Dict[tuple[str, str], SemanticActionSeries] = {
            (item.building_id, item.action_name): item for item in self.schedule.series
        }
        schedule_limits = self.schedule.metadata.get("action_power_limits_kw") or {}
        if not isinstance(schedule_limits, Mapping):
            raise ValueError("schedule.metadata.action_power_limits_kw must be a mapping.")
        self._schedule_power_limits: Mapping[str, Any] = schedule_limits
        self._schedule_call_count = 0

        self._attached = False
        self._building_names: List[str] = []
        self._observation_indices: List[Dict[str, int]] = []
        self._action_names: List[List[str]] = []
        self._action_bounds: List[tuple[np.ndarray, np.ndarray]] = []
        self._action_kinds: Dict[tuple[str, str], str] = {}
        self._power_observation_names: Dict[tuple[str, str, str], str] = {}
        self._power_metadata: Dict[tuple[str, str, str], float] = {}

    @staticmethod
    def _action_kind(action_name: str) -> Optional[str]:
        name = str(action_name)
        if "deferrable_appliance" in name or name.endswith("::start"):
            return "deferrable"
        if "electric_vehicle_storage" in name or name.startswith("charger::"):
            return "ev"
        if name == "electrical_storage" or (
            name.startswith("storage::") and name.endswith("::electrical_storage")
        ):
            return "storage"
        return None

    @staticmethod
    def _series_requires_direction(series: SemanticActionSeries, direction: str) -> bool:
        values = np.asarray(series.values, dtype=np.float64)
        if direction == "charge":
            return bool(np.any(values > _TOLERANCE))
        if direction == "discharge":
            return bool(np.any(values < -_TOLERANCE))
        raise ValueError(f"Unknown schedule direction {direction!r}.")

    @staticmethod
    def _metadata_limits(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        namespace = metadata.get("total_oracle_replay") or {}
        if not isinstance(namespace, Mapping):
            raise ValueError("metadata.total_oracle_replay must be a mapping.")
        limits = namespace.get("action_power_limits_kw") or {}
        if not isinstance(limits, Mapping):
            raise ValueError(
                "metadata.total_oracle_replay.action_power_limits_kw must be a mapping."
            )
        return limits

    def _nudge_ev_deadband_target(
        self,
        *,
        building_id: str,
        action_name: str,
        target: float,
    ) -> float:
        """Keep an exact MILP deadband command above CityLearn float residue."""
        if abs(target) <= _TOLERANCE:
            return target
        building = self._schedule_power_limits.get(building_id) or {}
        action = building.get(action_name) if isinstance(building, Mapping) else None
        if not isinstance(action, Mapping):
            return target
        field = (
            "min_charging_power_kw"
            if target > 0.0
            else "min_discharging_power_kw"
        )
        try:
            minimum = float(action.get(field, 0.0) or 0.0)
        except (TypeError, ValueError):
            return target
        magnitude = abs(target)
        if minimum > 0.0 and abs(magnitude - minimum) <= _DEADBAND_NUDGE_KW:
            return math.copysign(minimum + _DEADBAND_NUDGE_KW, target)
        return target

    @staticmethod
    def _metadata_value(
        limits: Mapping[str, Any],
        schedule_limits: Mapping[str, Any],
        building_id: str,
        action_name: str,
        field_names: Sequence[str],
    ) -> Optional[float]:
        building = limits.get(building_id) or {}
        if not isinstance(building, Mapping):
            raise ValueError(
                f"Power metadata for building {building_id!r} must be a mapping."
            )
        action = building.get(action_name) or {}
        if not isinstance(action, Mapping):
            raise ValueError(
                f"Power metadata for {(building_id, action_name)!r} must be a mapping."
            )
        schedule_building = schedule_limits.get(building_id) or {}
        if schedule_building and not isinstance(schedule_building, Mapping):
            raise ValueError(
                f"Schedule power metadata for building {building_id!r} must be a mapping."
            )
        nested_schedule_action = (
            schedule_building.get(action_name) or {}
            if isinstance(schedule_building, Mapping)
            else {}
        )
        flat_schedule_action = schedule_limits.get(action_name) or {}
        for source, label in (
            (action, "attached"),
            (nested_schedule_action, "schedule"),
            (flat_schedule_action, "schedule"),
        ):
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"{label.title()} power metadata for "
                    f"{(building_id, action_name)!r} must be a mapping."
                )
            for field_name in field_names:
                if field_name not in source:
                    continue
                try:
                    value = float(source[field_name])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{label.title()} power metadata field {field_name!r} for "
                        f"{(building_id, action_name)!r} must be numeric."
                    ) from error
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"{label.title()} power metadata field {field_name!r} for "
                        f"{(building_id, action_name)!r} must be finite and > 0."
                    )
                return value
        return None

    @staticmethod
    def _entity_prefix(action_name: str, kind: str) -> Optional[str]:
        name = str(action_name)
        if kind == "storage" and name.startswith("storage::") and "::" in name:
            return name.rsplit("::", 1)[0]
        if kind == "ev":
            if name.startswith("charger::") and "::" in name:
                return name.rsplit("::", 1)[0]
            if "::electric_vehicle_storage" in name:
                reference = name.rsplit("::", 1)[0]
                return reference if reference.startswith("charger::") else f"charger::{reference}"
        return None

    @classmethod
    def _power_observation_candidates(
        cls,
        observation_names: Sequence[str],
        *,
        building_id: str,
        action_name: str,
        kind: str,
        feature_name: str,
    ) -> List[str]:
        exact: List[str] = []
        prefix = cls._entity_prefix(action_name, kind)
        if prefix is not None:
            exact.append(f"{prefix}::{feature_name}")

        if kind == "storage":
            exact.append(f"storage::{building_id}/{action_name}::{feature_name}")
            if action_name == "electrical_storage":
                exact.append(
                    f"storage::{building_id}/electrical_storage::{feature_name}"
                )
            namespace = "storage::"
        else:
            legacy_prefix = "electric_vehicle_storage_"
            if action_name.startswith(legacy_prefix):
                charger_id = action_name[len(legacy_prefix) :]
                exact.append(f"charger::{building_id}/{charger_id}::{feature_name}")
            namespace = "charger::"

        names = [str(item) for item in observation_names]
        matches = [name for name in exact if name in names]
        if matches:
            return list(dict.fromkeys(matches))
        return [
            name
            for name in names
            if name.startswith(namespace) and name.endswith(f"::{feature_name}")
        ]

    def _bind_power_source(
        self,
        *,
        group_index: int,
        building_id: str,
        action_name: str,
        kind: str,
        source_name: str,
        observation_feature: str,
        metadata_fields: Sequence[str],
        metadata_limits: Mapping[str, Any],
        required: bool,
    ) -> None:
        if not required:
            return
        observation_names = list(self._observation_indices[group_index])
        candidates = self._power_observation_candidates(
            observation_names,
            building_id=building_id,
            action_name=action_name,
            kind=kind,
            feature_name=observation_feature,
        )
        key = building_id, action_name, source_name
        if len(candidates) == 1:
            self._power_observation_names[key] = candidates[0]
            return
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous raw power observations for {(building_id, action_name)!r}: "
                f"{candidates}. Use an asset-namespaced action/observation contract."
            )

        metadata_value = self._metadata_value(
            metadata_limits,
            self._schedule_power_limits,
            building_id,
            action_name,
            metadata_fields,
        )
        if metadata_value is not None:
            self._power_metadata[key] = metadata_value
            return

        field = metadata_fields[0]
        raise ValueError(
            f"Missing raw observation ending in {observation_feature!r} for "
            f"{(building_id, action_name)!r}. Supply raw asset power observations or "
            f"metadata['total_oracle_replay']['action_power_limits_kw']"
            f"[{building_id!r}][{action_name!r}][{field!r}] (EV schedules may also "
            "embed action_power_limits_kw)."
        )

    @staticmethod
    def _validate_series_contract(series: SemanticActionSeries, kind: str) -> None:
        if kind in {"storage", "ev"}:
            if series.unit != "kW" or series.positive_direction != "charge":
                raise ValueError(
                    f"Power series {(series.building_id, series.action_name)!r} must use "
                    "unit='kW' and positive_direction='charge'."
                )
        elif kind == "deferrable":
            supported = {
                ("normalized_action", "start"),
                ("binary_start", "start_cycle"),
            }
            if (series.unit, series.positive_direction) not in supported:
                raise ValueError(
                    f"Deferrable series {(series.building_id, series.action_name)!r} must "
                    "use normalized_action/start or binary_start/start_cycle semantics."
                )

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        del observation_space
        metadata = dict(metadata or {})
        building_names = [str(item) for item in metadata.get("building_names") or ()]
        group_count = len(action_names)
        if not (
            len(observation_names) == group_count
            and len(action_space) == group_count
            and len(building_names) == group_count
        ):
            raise ValueError(
                "Total oracle replay requires one observation/action/space group and one "
                "stable building id per local agent."
            )
        if len(set(building_names)) != len(building_names):
            raise ValueError("Total oracle replay requires unique building ids per action group.")

        seconds = metadata.get("seconds_per_time_step")
        try:
            timestep_hours = float(seconds) / 3600.0
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Total oracle replay requires numeric seconds_per_time_step metadata."
            ) from error
        if not math.isfinite(timestep_hours) or not np.isclose(
            timestep_hours, self.schedule.timestep_hours
        ):
            raise ValueError(
                "Total oracle schedule timestep does not match the attached environment: "
                f"{self.schedule.timestep_hours} != {timestep_hours}."
            )

        self._building_names = building_names
        self._observation_indices = [
            {str(name): index for index, name in enumerate(group)}
            for group in observation_names
        ]
        self._action_names = [[str(name) for name in group] for group in action_names]
        self._action_bounds = []
        self._action_kinds.clear()
        self._power_observation_names.clear()
        self._power_metadata.clear()

        attached_keys: set[tuple[str, str]] = set()
        for group_index, (building_id, names, space) in enumerate(
            zip(building_names, self._action_names, action_space)
        ):
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate action names in group {building_id!r}.")
            if not hasattr(space, "low") or not hasattr(space, "high"):
                raise ValueError(f"Action space for {building_id!r} must expose low/high.")
            low = np.asarray(space.low, dtype=np.float64).reshape(-1)
            high = np.asarray(space.high, dtype=np.float64).reshape(-1)
            if low.size != len(names) or high.size != len(names):
                raise ValueError(
                    f"Action bounds for {building_id!r} do not match its action names."
                )
            if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)) or np.any(low > high):
                raise ValueError(f"Action bounds for {building_id!r} are invalid.")
            self._action_bounds.append((low.copy(), high.copy()))

            for action_name in names:
                key = building_id, action_name
                attached_keys.add(key)
                series = self._series.get(key)
                if series is None:
                    raise ValueError(
                        f"Total oracle schedule has no series for attached local action {key!r}."
                    )
                kind = self._action_kind(action_name)
                if kind is None:
                    raise ValueError(
                        f"Total oracle replay does not recognize attached action {key!r}."
                    )
                self._validate_series_contract(series, kind)
                self._action_kinds[key] = kind

        orphaned = sorted(set(self._series) - attached_keys)
        if orphaned and not self.allow_attached_action_subset:
            raise ValueError(
                "Total oracle schedule contains series outside the attached building-local "
                f"action groups: {orphaned}."
            )

        metadata_limits = self._metadata_limits(metadata)
        for group_index, (building_id, names) in enumerate(
            zip(self._building_names, self._action_names)
        ):
            for action_name in names:
                key = building_id, action_name
                kind = self._action_kinds[key]
                series = self._series[key]
                if kind == "storage":
                    self._bind_power_source(
                        group_index=group_index,
                        building_id=building_id,
                        action_name=action_name,
                        kind=kind,
                        source_name="nominal",
                        observation_feature="nominal_power_kw",
                        metadata_fields=("nominal_power_kw", "nominal_power"),
                        metadata_limits=metadata_limits,
                        required=bool(np.any(np.abs(series.values) > _TOLERANCE)),
                    )
                elif kind == "ev":
                    self._bind_power_source(
                        group_index=group_index,
                        building_id=building_id,
                        action_name=action_name,
                        kind=kind,
                        source_name="charge",
                        observation_feature="max_charging_power_kw",
                        metadata_fields=(
                            "max_charging_power_kw",
                            "max_charge_power_kw",
                            "max_charge_kw",
                        ),
                        metadata_limits=metadata_limits,
                        required=self._series_requires_direction(series, "charge"),
                    )
                    self._bind_power_source(
                        group_index=group_index,
                        building_id=building_id,
                        action_name=action_name,
                        kind=kind,
                        source_name="discharge",
                        observation_feature="max_discharging_power_kw",
                        metadata_fields=(
                            "max_discharging_power_kw",
                            "max_discharge_power_kw",
                            "max_discharge_kw",
                        ),
                        metadata_limits=metadata_limits,
                        required=self._series_requires_direction(series, "discharge"),
                    )

        self._schedule_call_count = 0
        self._attached = True

    def _power_scale(
        self,
        *,
        group_index: int,
        building_id: str,
        action_name: str,
        source_name: str,
        observation: np.ndarray,
    ) -> float:
        key = building_id, action_name, source_name
        observation_name = self._power_observation_names.get(key)
        if observation_name is not None:
            index = self._observation_indices[group_index][observation_name]
            if index >= observation.size:
                raise ValueError(
                    f"Observation {observation_name!r} is absent from the runtime vector."
                )
            value = float(observation[index])
        elif key in self._power_metadata:
            value = self._power_metadata[key]
        else:
            raise ValueError(
                f"No {source_name} power scale is bound for {(building_id, action_name)!r}."
            )
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Invalid {source_name} power scale {value!r} for "
                f"{(building_id, action_name)!r}."
            )
        return value

    @staticmethod
    def _within_bounds(value: float, low: float, high: float, key: tuple[str, str]) -> float:
        if value < low - _ACTION_BOUND_TOLERANCE or value > high + _ACTION_BOUND_TOLERANCE:
            raise ValueError(
                f"Oracle action {value} for {key!r} exceeds attached bounds [{low}, {high}]."
            )
        return float(np.clip(value, low, high))

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        del deterministic, context
        if (
            self._schedule_call_count >= self.schedule.horizon
            and not self.repeat_schedule_for_training
        ):
            raise RuntimeError(
                f"Total oracle replay horizon {self.schedule.horizon} is exhausted; "
                "the schedule does not wrap."
            )
        schedule_step = self._schedule_call_count
        if self.repeat_schedule_for_training:
            schedule_step %= self.schedule.horizon
        actions = self.predict_at_step(observations, schedule_step=schedule_step)
        self._schedule_call_count += 1
        return actions

    def predict_at_step(
        self,
        observations: List[np.ndarray],
        *,
        schedule_step: int,
        deterministic: bool | None = None,
        context: Any = None,
    ) -> List[List[float]]:
        del deterministic, context
        if not self._attached:
            raise RuntimeError("TotalOracleReplayPolicy must be attached before prediction.")
        step = int(schedule_step)
        if self.repeat_schedule_for_training:
            step %= self.schedule.horizon
        if step < 0 or step >= self.schedule.horizon:
            raise IndexError(
                f"schedule_step {step} is outside [0, {self.schedule.horizon})."
            )
        if len(observations) != len(self._building_names):
            raise ValueError(
                "Runtime observation groups do not match attached building-local groups."
            )

        actions: List[List[float]] = []
        for group_index, (building_id, names, raw_observation) in enumerate(
            zip(self._building_names, self._action_names, observations)
        ):
            observation = np.asarray(raw_observation, dtype=np.float64).reshape(-1)
            low, high = self._action_bounds[group_index]
            group_actions: List[float] = []
            for action_index, action_name in enumerate(names):
                key = building_id, action_name
                target = float(self._series[key].values[step])
                kind = self._action_kinds[key]
                if kind == "deferrable" or abs(target) <= _TOLERANCE:
                    normalized = target
                elif kind == "storage":
                    normalized = target / self._power_scale(
                        group_index=group_index,
                        building_id=building_id,
                        action_name=action_name,
                        source_name="nominal",
                        observation=observation,
                    )
                elif kind == "ev":
                    target = self._nudge_ev_deadband_target(
                        building_id=building_id,
                        action_name=action_name,
                        target=target,
                    )
                    direction = "charge" if target > 0.0 else "discharge"
                    normalized = target / self._power_scale(
                        group_index=group_index,
                        building_id=building_id,
                        action_name=action_name,
                        source_name=direction,
                        observation=observation,
                    )
                else:  # pragma: no cover - guarded by attach_environment.
                    raise AssertionError(f"Unexpected action kind {kind!r}.")
                group_actions.append(
                    self._within_bounds(
                        normalized,
                        float(low[action_index]),
                        float(high[action_index]),
                        key,
                    )
                )
            actions.append(group_actions)

        self.actions = actions
        return actions

    def update(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        del (
            observations,
            actions,
            rewards,
            next_observations,
            terminated,
            truncated,
            update_target_step,
            global_learning_step,
            update_step,
            initial_exploration_done,
        )

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del context
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        shared = {
            "policy_type": "total_oracle_replay_policy",
            "semantic_policy_format": "semantic_schedule_replay",
            "oracle_schedule": {
                "problem_id": self.schedule.problem_id,
                "path": str(self.schedule_path),
                "sha256": self.schedule_sha256,
                "horizon": self.schedule.horizon,
                "perfect_foresight": True,
                "deployable": False,
                "wraps": self.repeat_schedule_for_training,
                "repeat_schedule_for_training": self.repeat_schedule_for_training,
                "allow_attached_action_subset": self.allow_attached_action_subset,
            },
            "controls": ["stationary_storage", "ev_v2g", "deferrable_start"],
            "service_teacher": None,
        }
        artifacts = []
        for agent_index, (building_id, action_names) in enumerate(
            zip(self._building_names, self._action_names)
        ):
            policy_path = destination / f"policy_agent_{agent_index}.json"
            policy_path.write_text(
                json.dumps(
                    {
                        **shared,
                        "agent_index": agent_index,
                        "building_id": building_id,
                        "action_names": action_names,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            artifacts.append(
                {
                    "agent_index": agent_index,
                    "path": policy_path.name,
                    "format": "rule_based",
                    "config": {"use_preprocessor": False},
                }
            )
        return {
            "format": "rule_based",
            "parameters": shared,
            "artifacts": artifacts,
        }
