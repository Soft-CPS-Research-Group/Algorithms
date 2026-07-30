"""Replay a semantic MILP battery schedule with strict-local RBC service control."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from algorithms.agents.baseline_policies import RBCSmartLocalPolicy
from algorithms.oracles import SemanticSchedule
from algorithms.utils.citylearn_local_action_safety import (
    CityLearnLocalSafetyAdapter,
    CityLearnSafetyConfig,
    preserve_teacher_service_with_storage_fallback,
)


class FixedServiceOracleReplayPolicy(RBCSmartLocalPolicy):
    """Strict-local RBC EV/deferrable service plus foresight BESS replay.

    This policy is an evaluation instrument, not a deployable online policy:
    its stationary-storage actions come from a perfect-foresight schedule.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        hyper = dict((config.get("algorithm", {}).get("hyperparameters") or {}))
        raw_path = str(hyper.get("schedule_path") or "").strip()
        if not raw_path:
            raise ValueError("FixedServiceOracleReplayPolicy requires schedule_path.")
        self.schedule_path = Path(raw_path).resolve()
        payload = self.schedule_path.read_bytes()
        self.schedule_sha256 = hashlib.sha256(payload).hexdigest()
        self.schedule = SemanticSchedule.from_json(payload.decode("utf-8"))
        if any(series.unit != "kW" for series in self.schedule.series):
            raise ValueError("Oracle replay currently requires all schedule series in kW.")
        self._schedule_series = {
            (series.building_id, series.action_name): np.asarray(series.values, dtype=np.float64)
            for series in self.schedule.series
        }
        self._schedule_call_count = 0
        self.local_action_safety_enabled = bool(
            hyper.get("local_action_safety_enabled", True)
        )
        self.local_action_safety_headroom_reserve_kw = max(
            0.0,
            float(hyper.get("local_action_safety_headroom_reserve_kw", 0.0) or 0.0),
        )
        self._safety_adapters: List[CityLearnLocalSafetyAdapter] = []
        self._building_names: List[str] = []

    def _policy_type(self) -> str:
        return "fixed_service_oracle_replay_policy"

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )
        metadata = dict(metadata or {})
        self._building_names = [str(item) for item in metadata.get("building_names", ())]
        if len(self._building_names) != len(action_names):
            raise ValueError("Oracle replay requires one stable building id per action group.")
        expected_step_hours = float(metadata.get("seconds_per_time_step") or 0.0) / 3600.0
        if not np.isclose(expected_step_hours, self.schedule.timestep_hours):
            raise ValueError(
                "Oracle schedule timestep does not match the attached environment: "
                f"{self.schedule.timestep_hours} != {expected_step_hours}."
            )
        for building_id, names in zip(self._building_names, action_names):
            if "electrical_storage" in names and (
                building_id,
                "electrical_storage",
            ) not in self._schedule_series:
                raise ValueError(f"Oracle schedule has no stationary battery series for {building_id}.")

        self._safety_adapters = []
        if self.local_action_safety_enabled:
            for index, (names, labels, space) in enumerate(
                zip(observation_names, action_names, action_space)
            ):
                self._safety_adapters.append(
                    CityLearnLocalSafetyAdapter(
                        observation_names=names,
                        action_names=labels,
                        action_low=space.low,
                        action_high=space.high,
                        metadata={**metadata, "building_names": [self._building_names[index]]},
                        config=CityLearnSafetyConfig(
                            protect_ev_minimum=False,
                            allow_discretionary_deferrable_start=True,
                            headroom_reserve_kw=self.local_action_safety_headroom_reserve_kw,
                        ),
                    )
                )

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        schedule_step = self._schedule_call_count % self.schedule.horizon
        self._schedule_call_count += 1
        return self.predict_at_step(
            observations,
            schedule_step=schedule_step,
            deterministic=deterministic,
            context=context,
        )

    def predict_at_step(
        self,
        observations: List[np.ndarray],
        *,
        schedule_step: int,
        deterministic: bool | None = None,
        context: Any = None,
    ) -> List[List[float]]:
        """Replay an explicit step without mutating the standalone counter."""

        actions = super().predict(observations, deterministic=deterministic, context=context)
        schedule_step = int(schedule_step) % self.schedule.horizon
        for agent_index, (building_id, names, observation) in enumerate(
            zip(self._building_names, self._action_labels, observations)
        ):
            if "electrical_storage" not in names:
                continue
            action_index = names.index("electrical_storage")
            obs_map = self._obs_index[agent_index]
            nominal_name = f"storage::{building_id}/electrical_storage::nominal_power_kw"
            nominal_index = obs_map.get(nominal_name)
            if nominal_index is None:
                raise ValueError(f"Missing raw observation {nominal_name!r} for oracle replay.")
            nominal_power = float(np.asarray(observation).reshape(-1)[nominal_index])
            if not np.isfinite(nominal_power) or nominal_power <= 0.0:
                raise ValueError(f"Invalid nominal stationary-battery power for {building_id}.")
            target_power = self._schedule_series[(building_id, "electrical_storage")][
                schedule_step
            ]
            low, high = self._get_action_bounds(agent_index, action_index)
            actions[agent_index][action_index] = float(
                np.clip(target_power / nominal_power, low, high)
            )

        if self.local_action_safety_enabled:
            projected: List[List[float]] = []
            for labels, adapter, observation, proposed in zip(
                self._action_labels,
                self._safety_adapters,
                observations,
                actions,
            ):
                result = adapter.project(observation, proposed)
                projected.append(
                    preserve_teacher_service_with_storage_fallback(
                        action_names=labels,
                        teacher_merged_actions=proposed,
                        projected_actions=result.executed_actions,
                    )
                )
            actions = projected
        self.actions = actions
        return actions

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = super().export_artifacts(output_dir, context)
        metadata.setdefault("parameters", {})["oracle_schedule"] = {
            "problem_id": self.schedule.problem_id,
            "path": str(self.schedule_path),
            "sha256": self.schedule_sha256,
            "horizon": self.schedule.horizon,
            "perfect_foresight": True,
            "deployable": False,
        }
        metadata["parameters"]["service_teacher"] = {
            "algorithm": "RBCSmartLocalPolicy",
            "observation_scope": "building_plus_public_exogenous",
            "blocked_observation_token": "community",
        }
        return metadata
