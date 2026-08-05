"""Replay a complete individual total-home MILP schedule without a teacher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from algorithms.agents.base_agent import BaseAgent
from algorithms.oracles import SemanticSchedule
from algorithms.utils.citylearn_local_action_safety import (
    CityLearnLocalSafetyAdapter,
    CityLearnSafetyConfig,
)


class TotalHomeOracleReplayPolicy(BaseAgent):
    """Replay storage, EV/V2G and deferrable decisions from one MILP.

    This policy does not call an RBC or service teacher.  The optional local
    safety layer is a deterministic feasibility projector and its intervention
    count is exported explicitly.
    """

    _use_raw_observations = True

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        hyper = dict((config.get("algorithm", {}).get("hyperparameters") or {}))
        raw_path = str(hyper.get("schedule_path") or "").strip()
        if not raw_path:
            raise ValueError("TotalHomeOracleReplayPolicy requires schedule_path.")
        self.schedule_path = Path(raw_path).resolve()
        payload = self.schedule_path.read_bytes()
        self.schedule_sha256 = hashlib.sha256(payload).hexdigest()
        self.schedule = SemanticSchedule.from_json(payload.decode("utf-8"))
        self._series = {
            (series.building_id, series.action_name): series
            for series in self.schedule.series
        }
        self._power_limits = dict(self.schedule.metadata.get("action_power_limits_kw") or {})
        self.local_action_safety_enabled = bool(hyper.get("local_action_safety_enabled", True))
        self.local_action_safety_fail_on_infeasible = bool(
            hyper.get("local_action_safety_fail_on_infeasible", False)
        )
        self.local_action_safety_protect_ev_minimum = bool(
            hyper.get("local_action_safety_protect_ev_minimum", True)
        )
        self.local_action_safety_ev_minimum_mode = str(
            hyper.get("local_action_safety_ev_minimum_mode", "average")
        ).strip()
        self.local_action_safety_protect_ev_service_target = bool(
            hyper.get("local_action_safety_protect_ev_service_target", False)
        )
        self.local_action_safety_protect_deferrable_must_start = bool(
            hyper.get("local_action_safety_protect_deferrable_must_start", True)
        )
        self.local_action_safety_allow_discretionary_deferrable_start = bool(
            hyper.get(
                "local_action_safety_allow_discretionary_deferrable_start",
                True,
            )
        )
        self.repeat_schedule_for_training = bool(
            hyper.get("repeat_schedule_for_training", False)
        )
        self.local_action_safety_headroom_reserve_kw = max(
            float(hyper.get("local_action_safety_headroom_reserve_kw", 0.0) or 0.0),
            0.0,
        )
        self._step = 0
        self._building_names: list[str] = []
        self._action_names: list[list[str]] = []
        self._action_low: list[np.ndarray] = []
        self._action_high: list[np.ndarray] = []
        self._safety_adapters: list[CityLearnLocalSafetyAdapter] = []
        self._projection_step_count = 0
        self._projection_action_count = 0
        self._maximum_projection_delta = 0.0
        self.actions: list[list[float]] = []

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
        self._building_names = [str(value) for value in metadata.get("building_names", ())]
        if len(self._building_names) != len(action_names):
            raise ValueError("Total-home replay requires one stable building id per action group.")
        expected_hours = float(metadata.get("seconds_per_time_step") or 0.0) / 3600.0
        if not np.isclose(expected_hours, self.schedule.timestep_hours, atol=1.0e-12, rtol=0.0):
            raise ValueError("Total-home schedule timestep does not match the environment.")
        self._action_names = [list(map(str, names)) for names in action_names]
        self._action_low = [np.asarray(space.low, dtype=np.float64) for space in action_space]
        self._action_high = [np.asarray(space.high, dtype=np.float64) for space in action_space]

        expected_keys = set(self._series)
        environment_keys = {
            (building, action)
            for building, names in zip(self._building_names, self._action_names)
            for action in names
        }
        missing = environment_keys - expected_keys
        unexpected = expected_keys - environment_keys
        if missing or unexpected:
            raise ValueError(
                "Total-home schedule/environment action mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )

        self._safety_adapters = []
        if self.local_action_safety_enabled:
            for index, (names, labels, space) in enumerate(
                zip(observation_names, self._action_names, action_space)
            ):
                self._safety_adapters.append(
                    CityLearnLocalSafetyAdapter(
                        observation_names=names,
                        action_names=labels,
                        action_low=space.low,
                        action_high=space.high,
                        metadata={**metadata, "building_names": [self._building_names[index]]},
                        config=CityLearnSafetyConfig(
                            fail_on_infeasible=self.local_action_safety_fail_on_infeasible,
                            protect_ev_minimum=self.local_action_safety_protect_ev_minimum,
                            ev_minimum_mode=self.local_action_safety_ev_minimum_mode,
                            protect_ev_service_target=(
                                self.local_action_safety_protect_ev_service_target
                            ),
                            protect_deferrable_must_start=(
                                self.local_action_safety_protect_deferrable_must_start
                            ),
                            allow_discretionary_deferrable_start=(
                                self.local_action_safety_allow_discretionary_deferrable_start
                            ),
                            headroom_reserve_kw=self.local_action_safety_headroom_reserve_kw,
                        ),
                    )
                )

    def _normalized_value(self, action_name: str, unit: str, value: float) -> float:
        if unit == "binary_start":
            return float(value)
        if unit != "kW":
            raise ValueError(f"Unsupported total-home schedule unit {unit!r}.")
        limits = self._power_limits.get(action_name) or {}
        maximum = limits.get("max_charge_kw") if value >= 0.0 else limits.get("max_discharge_kw")
        try:
            maximum = float(maximum)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Missing power limit for scheduled action {action_name!r}.") from error
        if maximum <= 0.0:
            if abs(value) <= 1.0e-10:
                return 0.0
            raise ValueError(f"Scheduled action {action_name!r} uses unavailable power direction.")
        return float(value / maximum)

    def predict_at_step(
        self,
        observations: List[np.ndarray],
        *,
        schedule_step: int,
        deterministic: bool | None = None,
        context: Any = None,
    ) -> List[List[float]]:
        del deterministic, context
        schedule_step = int(schedule_step)
        if self.repeat_schedule_for_training:
            schedule_step %= self.schedule.horizon
        if not 0 <= schedule_step < self.schedule.horizon:
            raise IndexError("Total-home schedule step is outside the finite replay horizon.")
        proposed: list[list[float]] = []
        for building, names, low, high in zip(
            self._building_names,
            self._action_names,
            self._action_low,
            self._action_high,
        ):
            values = []
            for index, action_name in enumerate(names):
                series = self._series[(building, action_name)]
                normalized = self._normalized_value(
                    action_name,
                    series.unit,
                    float(series.values[schedule_step]),
                )
                values.append(float(np.clip(normalized, low[index], high[index])))
            proposed.append(values)

        if not self.local_action_safety_enabled:
            self.actions = proposed
            return proposed
        executed: list[list[float]] = []
        changed_step = False
        for adapter, observation, values in zip(self._safety_adapters, observations, proposed):
            projected = list(adapter.project(observation, values).executed_actions)
            deltas = np.abs(np.asarray(projected) - np.asarray(values))
            changed = deltas > 1.0e-9
            changed_step |= bool(np.any(changed))
            self._projection_action_count += int(np.count_nonzero(changed))
            self._maximum_projection_delta = max(
                self._maximum_projection_delta,
                float(np.max(deltas, initial=0.0)),
            )
            executed.append([float(value) for value in projected])
        self._projection_step_count += int(changed_step)
        self.actions = executed
        return executed

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        del deterministic, context
        actions = self.predict_at_step(observations, schedule_step=self._step)
        self._step += 1
        return actions

    def update(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        terminated,
        truncated,
        **kwargs,
    ) -> None:
        del observations, actions, rewards, next_observations, terminated, truncated, kwargs

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del context
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        shared = {
            "policy_type": "total_home_oracle_replay",
            "schedule_path": str(self.schedule_path),
            "schedule_sha256": self.schedule_sha256,
            "schedule_problem_id": self.schedule.problem_id,
            "schedule_horizon": self.schedule.horizon,
            "perfect_foresight": True,
            "deployable": False,
            "service_teacher_used": False,
            "repeat_schedule_for_training": self.repeat_schedule_for_training,
            "local_action_safety_enabled": self.local_action_safety_enabled,
            "local_action_safety_fail_on_infeasible": (
                self.local_action_safety_fail_on_infeasible
            ),
            "local_action_safety_protect_ev_minimum": (
                self.local_action_safety_protect_ev_minimum
            ),
            "local_action_safety_ev_minimum_mode": (
                self.local_action_safety_ev_minimum_mode
            ),
            "local_action_safety_protect_ev_service_target": (
                self.local_action_safety_protect_ev_service_target
            ),
            "local_action_safety_protect_deferrable_must_start": (
                self.local_action_safety_protect_deferrable_must_start
            ),
            "local_action_safety_allow_discretionary_deferrable_start": (
                self.local_action_safety_allow_discretionary_deferrable_start
            ),
            "local_action_safety_headroom_reserve_kw": (
                self.local_action_safety_headroom_reserve_kw
            ),
            "safety_projection_step_count": self._projection_step_count,
            "safety_projection_action_count": self._projection_action_count,
            "maximum_safety_projection_delta": self._maximum_projection_delta,
        }
        artifacts = []
        for agent_index, (building, action_names) in enumerate(
            zip(self._building_names, self._action_names)
        ):
            policy_path = destination / f"policy_agent_{agent_index}.json"
            policy = {
                **shared,
                "agent_index": agent_index,
                "building_id": building,
                "action_names": action_names,
            }
            policy_path.write_text(
                json.dumps(policy, indent=2, sort_keys=True) + "\n",
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
            "artifacts": artifacts,
            "parameters": shared,
        }


__all__ = ["TotalHomeOracleReplayPolicy"]
