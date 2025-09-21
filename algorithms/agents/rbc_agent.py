"""Rule-based controller tailored for electric vehicle charging."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from algorithms.agents.base_agent import BaseAgent


@dataclass(slots=True)
class ChargerInfo:
    charger_id: str
    max_power: float
    min_power: float
    max_discharge_power: float
    min_discharge_power: float
    efficiency: float
    capacity: float
    ev_name: Optional[str] = None


class RuleBasedPolicy(BaseAgent):
    """Simple heuristic controller that prioritises PV utilisation while respecting EV requirements."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.use_raw_observations = True

        self._config = config
        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})

        self.pv_charge_threshold: float = float(hyper.get("pv_charge_threshold", 0.0))
        self.flexibility_hours: float = float(hyper.get("flexibility_hours", 3.0))
        self.emergency_hours: float = float(hyper.get("emergency_hours", 1.0))
        self.pv_preferred_charge_rate: float = float(hyper.get("pv_preferred_charge_rate", 0.6))
        self.flex_trickle_charge: float = float(hyper.get("flex_trickle_charge", 0.0))
        self.min_charge_rate: float = float(hyper.get("min_charge_rate", 0.0))
        self.emergency_charge_rate: float = float(hyper.get("emergency_charge_rate", 1.0))
        self.energy_epsilon: float = float(hyper.get("energy_epsilon", 1e-3))
        self.non_flexible_chargers: set[str] = set(hyper.get("non_flexible_chargers", []) or [])
        self.default_capacity: float = float(hyper.get("default_capacity_kwh", 60.0))

        dataset_path = Path(config.get("simulator", {}).get("dataset_path", ""))
        self._dataset_path = dataset_path
        self._dataset_info = self._load_dataset_info(dataset_path) if dataset_path.exists() else None

        self._obs_index: List[Dict[str, int]] = []
        self._action_labels: List[List[str]] = []
        self._action_bounds: List[Dict[str, List[float]]] = []
        self._agent_buildings: Dict[int, str] = {}
        self._ev_action_mapping: Dict[int, List[Optional[ChargerInfo]]] = {}

        self.step_hours: float = 1.0

    # ---------------------------------------------------------------------
    # BaseAgent hooks
    # ---------------------------------------------------------------------
    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._obs_index = [
            {name: idx for idx, name in enumerate(names)} for names in observation_names
        ]
        self._action_labels = [list(names) for names in action_names]

        self._action_bounds = []
        for space in action_space:
            if hasattr(space, "low") and hasattr(space, "high"):
                low = np.asarray(space.low, dtype=float).flatten().tolist()
                high = np.asarray(space.high, dtype=float).flatten().tolist()
            else:
                low = []
                high = []
            self._action_bounds.append({"low": low, "high": high})

        if metadata:
            seconds = metadata.get("seconds_per_time_step")
            if seconds:
                try:
                    self.step_hours = float(seconds) / 3600.0
                except (TypeError, ValueError):
                    self.step_hours = 1.0
            building_names = metadata.get("building_names")
        else:
            building_names = None

        if building_names:
            self._agent_buildings = {
                idx: name for idx, name in enumerate(building_names[: len(observation_names)])
            }
        elif self._dataset_info:
            ordered = self._dataset_info["building_order"]
            self._agent_buildings = {
                idx: ordered[idx] if idx < len(ordered) else ordered[-1]
                for idx in range(len(observation_names))
            }
        else:
            self._agent_buildings = {idx: f"agent_{idx}" for idx in range(len(observation_names))}

        self._build_ev_action_mapping(action_names)

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        actions: List[List[float]] = []

        for agent_idx, obs in enumerate(observations):
            obs = np.asarray(obs, dtype=float)
            obs_map = self._obs_index[agent_idx] if agent_idx < len(self._obs_index) else {}
            action_names = self._action_labels[agent_idx] if agent_idx < len(self._action_labels) else []
            agent_actions: List[float] = []

            for action_position, action_name in enumerate(action_names):
                bounds = self._get_action_bounds(agent_idx, action_position)

                if "electric_vehicle" in action_name:
                    charger_info = self._get_charger_info(agent_idx, action_position)
                    normalised = self._compute_ev_action(
                        agent_idx,
                        obs,
                        obs_map,
                        charger_info,
                        bounds,
                    )
                    high = bounds[1]
                    value = normalised * high if high > 1.0 else normalised
                else:
                    value = 0.0

                value = max(bounds[0], min(bounds[1], value))
                agent_actions.append(value)

            actions.append(agent_actions)

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
        # Rule-based agent is stateless for training purposes.
        return None

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        policy = {
            "policy_type": "rule_based_ev",
            "version": 1,
            "dataset_schema": str(self._dataset_path) if self._dataset_path else None,
            "hyperparameters": {
                "pv_charge_threshold": self.pv_charge_threshold,
                "flexibility_hours": self.flexibility_hours,
                "emergency_hours": self.emergency_hours,
                "pv_preferred_charge_rate": self.pv_preferred_charge_rate,
                "flex_trickle_charge": self.flex_trickle_charge,
                "min_charge_rate": self.min_charge_rate,
                "emergency_charge_rate": self.emergency_charge_rate,
                "energy_epsilon": self.energy_epsilon,
                "non_flexible_chargers": sorted(self.non_flexible_chargers),
                "default_capacity_kwh": self.default_capacity,
                "step_hours": self.step_hours,
            },
            "charger_mapping": self._serialise_charger_mapping(),
        }

        output_path = Path(output_dir) / "rbc_policy.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(policy, handle, indent=2)

        return {
            "format": "rbc",
            "parameters": {
                "hyperparameters": policy["hyperparameters"],
            },
            "artifacts": [
                {
                    "type": "policy_data",
                    "path": str(output_path.relative_to(Path(output_dir))),
                }
            ],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _serialise_charger_mapping(self) -> Dict[str, Any]:
        if not self._dataset_info:
            return {}

        serialisable: Dict[str, List[Dict[str, Any]]] = {}
        for building, chargers in self._dataset_info["charger_info"].items():
            serialisable[building] = [
                {
                    "charger_id": info.charger_id,
                    "max_power": info.max_power,
                    "min_power": info.min_power,
                    "max_discharge_power": info.max_discharge_power,
                    "min_discharge_power": info.min_discharge_power,
                    "efficiency": info.efficiency,
                    "capacity": info.capacity,
                    "ev_name": info.ev_name,
                }
                for info in chargers
            ]

        return serialisable

    def _build_ev_action_mapping(self, action_names: List[List[str]]) -> None:
        self._ev_action_mapping.clear()
        if not self._dataset_info:
            for agent_idx, names in enumerate(action_names):
                ev_slots = [name for name in names if "electric_vehicle" in name]
                self._ev_action_mapping[agent_idx] = [None for _ in ev_slots]
            return

        building_chargers = self._dataset_info["charger_info"]
        for agent_idx, names in enumerate(action_names):
            building = self._agent_buildings.get(agent_idx)
            available = building_chargers.get(building, [])
            available_sorted = sorted(available, key=lambda info: info.charger_id)
            mapping: List[Optional[ChargerInfo]] = []
            ev_slots = [name for name in names if "electric_vehicle" in name]
            for pos, _ in enumerate(ev_slots):
                charger_info = available_sorted[pos] if pos < len(available_sorted) else None
                mapping.append(charger_info)
            self._ev_action_mapping[agent_idx] = mapping

    def _get_charger_info(
        self,
        agent_idx: int,
        ev_action_position: int,
    ) -> Optional[ChargerInfo]:
        mapping = self._ev_action_mapping.get(agent_idx)
        if not mapping or ev_action_position >= len(mapping):
            return None
        return mapping[ev_action_position]

    def _get_action_bounds(self, agent_idx: int, action_position: int) -> Sequence[float]:
        if agent_idx >= len(self._action_bounds):
            return (0.0, 1.0)
        bounds = self._action_bounds[agent_idx]
        lows = bounds.get("low") or []
        highs = bounds.get("high") or []
        low = lows[action_position] if action_position < len(lows) else 0.0
        high = highs[action_position] if action_position < len(highs) else 1.0
        if math.isnan(low):
            low = 0.0
        if math.isnan(high) or high == 0:
            high = 1.0
        return (float(low), float(high))

    def _compute_ev_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        charger_info: Optional[ChargerInfo],
        bounds: Sequence[float],
    ) -> float:
        if charger_info is None:
            capacity = self.default_capacity
            max_power = 7.4
        else:
            capacity = charger_info.capacity or self.default_capacity
            max_power = charger_info.max_power or 7.4

        current_soc = self._get_value(obs, obs_map, "electric_vehicle_soc", default=0.0)
        required_soc = self._get_value(
            obs,
            obs_map,
            "electric_vehicle_required_soc_departure",
            default=current_soc,
        )
        charger_state = self._get_value(obs, obs_map, "electric_vehicle_charger_state", default=0.0)

        if charger_state <= 0.0:
            return 0.0

        soc_gap_pct = max(0.0, required_soc - current_soc)
        energy_needed = (soc_gap_pct / 100.0) * capacity
        if energy_needed <= self.energy_epsilon:
            return 0.0

        hour = self._get_value(obs, obs_map, "hour", default=0.0)
        minute = self._get_value(obs, obs_map, "minute", default=0.0)
        current_time = (hour % 24.0) + minute / 60.0

        departure_time = self._get_value(
            obs,
            obs_map,
            "electric_vehicle_departure_time",
            default=current_time + self.flexibility_hours,
        )
        departure_time = departure_time % 24.0
        time_to_departure = departure_time - current_time
        if time_to_departure <= 0:
            time_to_departure += 24.0

        time_to_departure = max(time_to_departure, self.step_hours)

        flex_flag = self._get_value(obs, obs_map, "electric_vehicle_is_flexible", default=1.0)
        is_flexible = flex_flag > 0.5 and (not charger_info or charger_info.charger_id not in self.non_flexible_chargers)

        solar_generation = max(
            0.0,
            self._get_value(obs, obs_map, "solar_generation", default=0.0),
        )
        if solar_generation == 0.0:
            solar_generation = max(0.0, self._get_value(obs, obs_map, "electricity_generation", default=0.0))

        required_power = energy_needed / time_to_departure
        if max_power <= 0:
            max_power = 7.4

        normalised = min(1.0, required_power / max_power)

        if solar_generation >= self.pv_charge_threshold:
            normalised = max(normalised, self.pv_preferred_charge_rate)

        if time_to_departure <= self.emergency_hours or not is_flexible:
            normalised = max(normalised, self.emergency_charge_rate)

        elif is_flexible and solar_generation < self.pv_charge_threshold and time_to_departure > self.flexibility_hours:
            normalised = max(self.flex_trickle_charge, min(normalised, self.flex_trickle_charge))

        if normalised > 0:
            normalised = max(normalised, self.min_charge_rate)

        low, high = bounds
        normalised = float(np.clip(normalised, 0.0 if low <= 0 else low, 1.0 if high >= 1 else high))
        return normalised

    def _get_value(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        name: str,
        default: float = 0.0,
    ) -> float:
        idx = obs_map.get(name)
        if idx is None or idx >= len(obs):
            return default
        value = obs[idx]
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return default
            value = value[0]
        try:
            if math.isnan(value):
                return default
        except TypeError:
            pass
        return float(value)

    def _load_dataset_info(self, dataset_path: Path) -> Optional[Dict[str, Any]]:
        if not dataset_path.exists():
            return None

        with dataset_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)

        building_order = list(schema.get("buildings", {}).keys())
        charger_info: Dict[str, List[ChargerInfo]] = {
            name: [] for name in building_order
        }
        charger_to_building: Dict[str, str] = {}

        for building_name, data in schema.get("buildings", {}).items():
            for charger_id, charger_data in (data.get("chargers", {}) or {}).items():
                attributes = charger_data.get("attributes", {})
                info = ChargerInfo(
                    charger_id=charger_id,
                    max_power=float(attributes.get("max_charging_power", attributes.get("nominal_power", 0.0) or 0.0)),
                    min_power=float(attributes.get("min_charging_power", 0.0) or 0.0),
                    max_discharge_power=float(attributes.get("max_discharging_power", 0.0) or 0.0),
                    min_discharge_power=float(attributes.get("min_discharging_power", 0.0) or 0.0),
                    efficiency=float(attributes.get("efficiency", 1.0) or 1.0),
                    capacity=0.0,
                )
                charger_info.setdefault(building_name, []).append(info)
                charger_to_building[charger_id] = building_name

        root_dir = dataset_path.parent
        for ev_name, ev_def in schema.get("electric_vehicles_def", {}).items():
            battery_attrs = (ev_def.get("battery", {}) or {}).get("attributes", {})
            capacity = float(battery_attrs.get("capacity", self.default_capacity))
            csv_file = ev_def.get("energy_simulation")
            charger_id: Optional[str] = None
            if csv_file:
                csv_path = root_dir / csv_file
                try:
                    with csv_path.open("r", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        first_row = next(reader)
                        charger_id = (first_row.get("charger") or "").strip()
                except (FileNotFoundError, StopIteration):
                    charger_id = None

            if charger_id and charger_id in charger_to_building:
                building = charger_to_building[charger_id]
                for info in charger_info.get(building, []):
                    if info.charger_id == charger_id:
                        info.capacity = capacity
                        info.ev_name = ev_name
                        break

        # Ensure deterministic ordering of chargers per building
        for building, chargers in charger_info.items():
            charger_info[building] = sorted(chargers, key=lambda info: info.charger_id)

        return {
            "building_order": building_order,
            "charger_info": charger_info,
        }
