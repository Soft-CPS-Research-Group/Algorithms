"""Rule-based controller tailored for electric vehicle charging."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlflow
import numpy as np

from algorithms.agents.base_agent import BaseAgent
from utils.artifact_config_builder import build_auto_artifact_config


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
    phase_connection: Optional[str] = None


@dataclass(slots=True)
class StorageInfo:
    nominal_power: float
    capacity: float = 0.0
    phase_connection: Optional[str] = None


class RuleBasedPolicy(BaseAgent):
    """Simple heuristic controller that prioritises PV utilisation while respecting EV requirements."""

    _use_raw_observations: bool = True

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

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
        self.control_storage: bool = bool(hyper.get("control_storage", True))
        self.control_evs: bool = bool(hyper.get("control_evs", True))
        self.control_deferrables: bool = bool(hyper.get("control_deferrables", True))
        self.deferrable_start_action: float = float(hyper.get("deferrable_start_action", 1.0))
        self.deferrable_urgency_threshold: float = float(hyper.get("deferrable_urgency_threshold", 0.75))
        self.deferrable_slack_threshold: float = float(hyper.get("deferrable_slack_threshold", 0.25))
        self.deferrable_priority_threshold: float = float(hyper.get("deferrable_priority_threshold", 0.5))

        dataset_path = Path(config.get("simulator", {}).get("dataset_path", ""))
        self._dataset_path = dataset_path
        self._dataset_info = self._load_dataset_info(dataset_path) if dataset_path.is_file() else None

        self._obs_index: List[Dict[str, int]] = []
        self._action_labels: List[List[str]] = []
        self._action_bounds: List[Dict[str, List[float]]] = []
        self._agent_buildings: Dict[int, str] = {}
        self._ev_action_mapping: Dict[int, List[Optional[ChargerInfo]]] = {}
        self._ev_action_position_mapping: Dict[int, Dict[int, Optional[ChargerInfo]]] = {}

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

                if self.control_storage and self._is_storage_action_name(action_name):
                    value = self._compute_storage_action(agent_idx, obs, obs_map, action_name, bounds)
                    value = self._apply_storage_dynamic_headroom_limit(
                        value,
                        obs,
                        obs_map,
                        self._get_storage_info(agent_idx),
                        bounds,
                    )
                    value = self._apply_storage_soc_limit(value, obs, obs_map, bounds)
                elif "electric_vehicle" in action_name and self.control_evs:
                    charger_info = self._get_charger_info(agent_idx, action_position)
                    normalised = self._compute_ev_action(
                        agent_idx,
                        obs,
                        obs_map,
                        charger_info,
                        bounds,
                        action_name,
                    )
                    high = bounds[1]
                    value = normalised * high if high > 1.0 else normalised
                    value = self._apply_ev_dynamic_headroom_limit(
                        value,
                        obs,
                        obs_map,
                        charger_info,
                        bounds,
                    )
                elif self.control_deferrables and self._is_deferrable_action_name(action_name, obs_map):
                    value = self._compute_deferrable_action(obs, obs_map, action_name, bounds)
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
        context = context or {}
        bundle_cfg = ((context.get("config") or {}).get("bundle") or {})
        require_observations_envelope = bool(bundle_cfg.get("require_observations_envelope", False))
        global_artifact_config = dict(bundle_cfg.get("artifact_config") or {})
        raw_per_agent_config = bundle_cfg.get("per_agent_artifact_config") or {}
        per_agent_artifact_config = (
            raw_per_agent_config if isinstance(raw_per_agent_config, dict) else {}
        )

        hyperparameters = {
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
            "control_storage": self.control_storage,
            "control_evs": self.control_evs,
            "control_deferrables": self.control_deferrables,
            "deferrable_start_action": self.deferrable_start_action,
            "deferrable_urgency_threshold": self.deferrable_urgency_threshold,
            "deferrable_slack_threshold": self.deferrable_slack_threshold,
            "deferrable_priority_threshold": self.deferrable_priority_threshold,
            "step_hours": self.step_hours,
        }
        for attr in (
            "seed",
            "storage_min_soc",
            "storage_max_soc",
            "storage_target_soc",
            "storage_charge_rate",
            "storage_discharge_rate",
            "price_charge_rate",
            "price_discharge_rate",
            "pv_charge_rate",
            "peak_discharge_rate",
            "storage_price_charge_soc_ceiling",
            "storage_price_discharge_soc_floor",
            "storage_peak_discharge_soc_floor",
            "normal_storage_discharge_import_threshold_kw",
            "ev_normal_charge_rate",
            "ev_normal_target_soc",
            "ev_price_charge_rate",
            "ev_pv_charge_rate",
            "ev_v2g_discharge_rate",
            "allow_v2g",
            "pv_surplus_threshold_kw",
            "import_peak_threshold_kw",
            "low_headroom_threshold_kw",
            "ev_v2g_reserve_soc",
            "ev_service_margin_rate",
            "ev_service_floor_rate",
            "ev_service_lookahead_hours",
            "ev_service_target_soc",
            "ev_deadline_buffer_hours",
            "ev_v2g_min_departure_hours",
            "ev_v2g_service_margin_soc",
            "ev_v2g_buffer_soc",
            "deferrable_safety_margin_steps",
            "deferrable_import_block_threshold_kw",
            "deferrable_community_import_block_threshold_kw",
        ):
            if hasattr(self, attr):
                hyperparameters[attr] = getattr(self, attr)

        export_root = Path(output_dir)
        export_root.mkdir(parents=True, exist_ok=True)
        agent_action_labels = self._action_labels or [[]]

        metadata: Dict[str, Any] = {
            "format": "rule_based",
            "parameters": {"hyperparameters": hyperparameters},
            "artifacts": [],
        }

        for agent_index, action_names in enumerate(agent_action_labels):
            default_actions = {str(name): 0.0 for name in action_names}
            policy = {
                "policy_type": self._policy_type(),
                "version": 1,
                "agent_index": agent_index,
                "dataset_schema": str(self._dataset_path) if self._dataset_path else None,
                "action_names": [str(name) for name in action_names],
                "default_actions": default_actions,
                "rules": [],
                "hyperparameters": hyperparameters,
                "charger_mapping": self._serialise_charger_mapping(),
            }

            output_path = export_root / f"policy_agent_{agent_index}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(policy, handle, indent=2)
            if mlflow.active_run():
                mlflow.log_artifact(str(output_path), artifact_path="model")

            raw_agent_override = (
                per_agent_artifact_config.get(str(agent_index))
                if str(agent_index) in per_agent_artifact_config
                else per_agent_artifact_config.get(agent_index)
            )
            agent_override = raw_agent_override if isinstance(raw_agent_override, dict) else {}
            auto_artifact_config = build_auto_artifact_config(context=context, agent_index=agent_index)
            artifact_config = {"use_preprocessor": False}
            artifact_config.update(auto_artifact_config)
            artifact_config.update(global_artifact_config)
            artifact_config.update(agent_override)
            if require_observations_envelope:
                artifact_config["require_observations_envelope"] = True

            metadata["artifacts"].append(
                {
                    "agent_index": agent_index,
                    "path": str(output_path.relative_to(export_root)),
                    "format": "rule_based",
                    "config": artifact_config,
                }
            )

        return metadata

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _policy_type(self) -> str:
        if self.control_evs and self.control_deferrables:
            return "rule_based_ev_deferrable"
        if self.control_evs:
            return "rule_based_ev_only"
        if self.control_deferrables:
            return "rule_based_deferrable_only"
        return "zero_action"

    @staticmethod
    def _is_storage_action_name(action_name: str) -> bool:
        return str(action_name or "") == "electrical_storage"

    def _compute_storage_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        action_name: str,
        bounds: Sequence[float],
    ) -> float:
        del agent_idx, obs, obs_map, action_name
        low, high = bounds
        return float(np.clip(0.0, low, high))

    def _apply_ev_dynamic_headroom_limit(
        self,
        value: float,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        charger_info: Optional[ChargerInfo],
        bounds: Sequence[float],
    ) -> float:
        if charger_info is None or not charger_info.phase_connection or value == 0.0:
            return value

        export = value < 0.0
        available_kw = self._available_dynamic_headroom_kw(
            obs,
            obs_map,
            charger_info.phase_connection,
            export=export,
        )
        if available_kw is None or not math.isfinite(available_kw):
            return value

        available_kw = max(0.0, available_kw)
        current_power_kw = self._current_charger_power_kw(obs, obs_map, charger_info)
        low, high = bounds
        if value > 0.0:
            available_kw += max(0.0, current_power_kw)
            power_scale = max(charger_info.max_power or 0.0, 1.0e-6)
            limit = available_kw / power_scale if abs(high) <= 1.0 else available_kw
            return min(value, max(0.0, limit))

        available_kw += max(0.0, -current_power_kw)
        power_scale = max(charger_info.max_discharge_power or charger_info.max_power or 0.0, 1.0e-6)
        limit = available_kw / power_scale if abs(low) <= 1.0 else available_kw
        return max(value, -max(0.0, limit))

    def _apply_storage_dynamic_headroom_limit(
        self,
        value: float,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        storage_info: Optional[StorageInfo],
        bounds: Sequence[float],
    ) -> float:
        if storage_info is None or not storage_info.phase_connection or value == 0.0:
            return value

        export = value < 0.0
        available_kw = self._available_dynamic_headroom_kw(
            obs,
            obs_map,
            storage_info.phase_connection,
            export=export,
        )
        if available_kw is None or not math.isfinite(available_kw):
            return value

        available_kw = max(0.0, available_kw)
        low, high = bounds
        power_scale = max(storage_info.nominal_power, 1.0e-6)
        if value > 0.0:
            if available_kw <= self.energy_epsilon:
                available_kw = self._local_pv_surplus_kw(obs, obs_map)
            limit = available_kw / power_scale if abs(high) <= 1.0 else available_kw
            return min(value, max(0.0, limit))

        limit = available_kw / power_scale if abs(low) <= 1.0 else available_kw
        return max(value, -max(0.0, limit))

    def _apply_storage_soc_limit(
        self,
        value: float,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        bounds: Sequence[float],
    ) -> float:
        """Avoid issuing storage commands beyond observed SOC limits.

        This is a baseline-policy guard only. The learning agents still see the
        observations/reward and the simulator remains the source of truth for
        physical constraints.
        """

        if value == 0.0:
            return float(np.clip(0.0, bounds[0], bounds[1]))

        soc = self._storage_soc_ratio(obs, obs_map, default=float("nan"))
        if math.isnan(soc):
            return float(np.clip(value, bounds[0], bounds[1]))

        min_soc, max_soc = self._observed_storage_soc_limits(obs, obs_map)
        if value > 0.0 and soc >= max_soc - self.energy_epsilon:
            return float(np.clip(0.0, bounds[0], bounds[1]))
        if value < 0.0 and soc <= min_soc + self.energy_epsilon:
            return float(np.clip(0.0, bounds[0], bounds[1]))
        return float(np.clip(value, bounds[0], bounds[1]))

    def _storage_soc_ratio(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        *,
        default: float = 0.0,
    ) -> float:
        value = self._first_available_observation_value(
            obs,
            obs_map,
            ("electrical_storage_soc_ratio", "electrical_storage_soc"),
            suffixes=("::soc", "::soc_ratio"),
            default=default,
        )
        if not math.isnan(value) and abs(value) > 1.5:
            value /= 100.0
        return float(np.clip(value, 0.0, 1.0)) if not math.isnan(value) else float("nan")

    def _observed_storage_soc_limits(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
    ) -> Tuple[float, float]:
        def storage_value(names: Sequence[str], suffix: str, default: float) -> float:
            for name in names:
                value = self._get_value(obs, obs_map, name, default=float("nan"))
                if not math.isnan(value):
                    return value

            storage_suffix = f"::{suffix}"
            for raw_name in obs_map:
                raw = str(raw_name)
                if raw.startswith("storage::") and raw.endswith(storage_suffix):
                    value = self._get_value(obs, obs_map, raw, default=float("nan"))
                    if not math.isnan(value):
                        return value

            return default

        min_soc = storage_value(
            ("electrical_storage_soc_min_ratio",),
            "soc_min_ratio",
            0.0,
        )
        max_soc = storage_value(
            ("electrical_storage_soc_max_ratio",),
            "soc_max_ratio",
            1.0,
        )
        if abs(min_soc) > 1.5:
            min_soc /= 100.0
        if abs(max_soc) > 1.5:
            max_soc /= 100.0
        min_soc = float(np.clip(min_soc, 0.0, 1.0))
        max_soc = float(np.clip(max_soc, 0.0, 1.0))
        if min_soc > max_soc:
            return max_soc, min_soc
        return min_soc, max_soc

    def _local_pv_surplus_kw(self, obs: np.ndarray, obs_map: Dict[str, int]) -> float:
        pv_power = self._first_available_observation_value(
            obs,
            obs_map,
            ("pv_power_kw", "solar_generation"),
            suffixes=("::generation_power_kw",),
        )
        load_power = self._first_available_observation_value(
            obs,
            obs_map,
            ("load_power_kw", "non_shiftable_load"),
        )
        return max(0.0, pv_power - load_power)

    def _first_available_observation_value(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        names: Sequence[str],
        *,
        suffixes: Sequence[str] = (),
        default: float = 0.0,
    ) -> float:
        for name in names:
            value = self._get_value(obs, obs_map, name, default=float("nan"))
            if not math.isnan(value):
                return value

        for raw_name in obs_map:
            if any(str(raw_name).endswith(suffix) for suffix in suffixes):
                value = self._get_value(obs, obs_map, str(raw_name), default=float("nan"))
                if not math.isnan(value):
                    return value

        return default

    def _available_dynamic_headroom_kw(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        phase_connection: Optional[str],
        *,
        export: bool,
    ) -> Optional[float]:
        candidates: List[float] = []
        building_name = "charging_building_export_headroom_kw" if export else "charging_building_headroom_kw"
        building_headroom = self._get_value(obs, obs_map, building_name, default=float("nan"))
        if not math.isnan(building_headroom) and building_headroom >= 0.0:
            candidates.append(building_headroom)

        phases = self._phase_connection_to_phases(phase_connection)
        phase_values: List[float] = []
        suffix = "export_headroom_kw" if export else "headroom_kw"
        for phase in phases:
            phase_value = self._get_value(
                obs,
                obs_map,
                f"charging_phase_{phase}_{suffix}",
                default=float("nan"),
            )
            if not math.isnan(phase_value) and phase_value >= 0.0:
                phase_values.append(phase_value)

        if phase_values:
            multiplier = float(len(phases)) if len(phases) > 1 else 1.0
            candidates.append(min(phase_values) * multiplier)

        if not candidates:
            return None
        return min(candidates)

    def _current_charger_power_kw(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        charger_info: Optional[ChargerInfo],
    ) -> float:
        """Return the currently applied charger power from raw observations, when available."""
        if charger_info is None or not charger_info.charger_id:
            return 0.0

        charger_id = str(charger_info.charger_id)
        feature_order = ("applied_power_kw", "commanded_power_kw")
        for feature in feature_order:
            suffixes = (
                f"/{charger_id}::{feature}",
                f"{charger_id}::{feature}",
                f"{charger_id}_{feature}",
            )
            for name in obs_map:
                raw_name = str(name)
                if any(raw_name.endswith(suffix) for suffix in suffixes):
                    return self._get_value(obs, obs_map, raw_name, default=0.0)

        return 0.0

    def _get_storage_info(self, agent_idx: int) -> Optional[StorageInfo]:
        if not self._dataset_info:
            return None
        building = self._agent_buildings.get(agent_idx)
        return self._dataset_info.get("storage_info", {}).get(building)

    @staticmethod
    def _phase_connection_to_phases(phase_connection: Optional[str]) -> Tuple[str, ...]:
        if not phase_connection:
            return ()

        raw = str(phase_connection).strip().upper().replace("-", "_").replace(" ", "_")
        if raw in {"ALL_PHASES", "THREE_PHASE", "THREE_PHASES", "3_PHASE", "3_PHASES", "3PHASE", "BALANCED"}:
            return ("L1", "L2", "L3")
        if raw in {"L1", "L2", "L3"}:
            return (raw,)

        phases = tuple(phase for phase in ("L1", "L2", "L3") if phase in raw)
        return phases

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
                    "phase_connection": info.phase_connection,
                }
                for info in chargers
            ]

        return serialisable

    def _build_ev_action_mapping(self, action_names: List[List[str]]) -> None:
        self._ev_action_mapping.clear()
        self._ev_action_position_mapping.clear()
        if not self._dataset_info:
            for agent_idx, names in enumerate(action_names):
                ev_slots = [name for name in names if "electric_vehicle" in name]
                self._ev_action_mapping[agent_idx] = [None for _ in ev_slots]
                self._ev_action_position_mapping[agent_idx] = {
                    pos: None for pos, name in enumerate(names) if "electric_vehicle" in name
                }
            return

        building_chargers = self._dataset_info["charger_info"]
        for agent_idx, names in enumerate(action_names):
            building = self._agent_buildings.get(agent_idx)
            available = building_chargers.get(building, [])
            available_sorted = sorted(available, key=lambda info: info.charger_id)
            mapping: List[Optional[ChargerInfo]] = []
            position_mapping: Dict[int, Optional[ChargerInfo]] = {}
            ev_slots = [
                (position, name)
                for position, name in enumerate(names)
                if "electric_vehicle" in name
            ]
            for slot_index, (position, _) in enumerate(ev_slots):
                charger_info = available_sorted[slot_index] if slot_index < len(available_sorted) else None
                mapping.append(charger_info)
                position_mapping[position] = charger_info
            self._ev_action_mapping[agent_idx] = mapping
            self._ev_action_position_mapping[agent_idx] = position_mapping

    def _get_charger_info(
        self,
        agent_idx: int,
        ev_action_position: int,
    ) -> Optional[ChargerInfo]:
        position_mapping = self._ev_action_position_mapping.get(agent_idx)
        if position_mapping and ev_action_position in position_mapping:
            return position_mapping[ev_action_position]

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
        action_name: str = "",
    ) -> float:
        charger_id = self._resolve_ev_charger_id(action_name, charger_info)
        building_name = self._agent_buildings.get(agent_idx)
        if charger_info is None:
            capacity = self.default_capacity
            max_power = 7.4
        else:
            capacity = charger_info.capacity or self.default_capacity
            max_power = charger_info.max_power or 7.4

        capacity = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="battery_capacity",
            generic_names=["connected_electric_vehicle_at_charger_battery_capacity"],
            default=capacity,
        )

        current_soc = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="soc",
            generic_names=["electric_vehicle_soc", "connected_electric_vehicle_at_charger_soc"],
            default=0.0,
        )
        required_soc = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="required_soc_departure",
            generic_names=[
                "electric_vehicle_required_soc_departure",
                "connected_electric_vehicle_at_charger_required_soc_departure",
            ],
            default=current_soc,
        )
        charger_state = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="connected_state",
            generic_names=[
                "electric_vehicle_charger_state",
                "electric_vehicle_charger_connected_state",
            ],
            default=0.0,
        )

        if charger_state <= 0.0:
            return 0.0

        soc_gap_fraction = self._soc_gap_fraction(current_soc, required_soc)
        energy_needed = soc_gap_fraction * capacity
        if energy_needed <= self.energy_epsilon:
            return 0.0

        time_to_departure = self._get_ev_departure_hours(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            default=self.flexibility_hours,
        )
        time_to_departure = max(time_to_departure, self.step_hours)

        flex_flag = self._get_value(obs, obs_map, "electric_vehicle_is_flexible", default=1.0)
        is_flexible = flex_flag > 0.5 and (not charger_info or charger_info.charger_id not in self.non_flexible_chargers)

        solar_generation = self._get_value(obs, obs_map, "pv_power_kw", default=float("nan"))
        if math.isnan(solar_generation):
            solar_generation = self._get_value(obs, obs_map, "solar_generation", default=float("nan"))
        if math.isnan(solar_generation):
            solar_generation = self._get_value(obs, obs_map, "electricity_generation", default=0.0)
        solar_generation = max(0.0, solar_generation)

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

    @staticmethod
    def _resolve_ev_charger_id(
        action_name: str,
        charger_info: Optional[ChargerInfo],
    ) -> Optional[str]:
        if charger_info is not None and charger_info.charger_id:
            return charger_info.charger_id

        raw = str(action_name or "")
        prefix = "electric_vehicle_storage_"
        if raw.startswith(prefix) and len(raw) > len(prefix):
            return raw[len(prefix) :]
        return None

    def _get_ev_value(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        *,
        charger_id: Optional[str],
        building_name: Optional[str] = None,
        feature: str,
        generic_names: Sequence[str],
        default: float = 0.0,
    ) -> float:
        candidates: List[str] = []
        if charger_id and building_name:
            entity_prefix = f"charger::{building_name}/{charger_id}::"
            if feature == "connected_state":
                candidates.append(f"{entity_prefix}connected_state")
            elif feature == "incoming_state":
                candidates.append(f"{entity_prefix}incoming_state")
            elif feature == "soc":
                candidates.extend(
                    [
                        f"{entity_prefix}connected_ev_soc",
                        f"{entity_prefix}connected_ev::soc",
                    ]
                )
            elif feature == "required_soc_departure":
                candidates.append(f"{entity_prefix}connected_ev_required_soc_departure")
            elif feature == "battery_capacity":
                candidates.extend(
                    [
                        f"{entity_prefix}connected_ev_battery_capacity_kwh",
                        f"{entity_prefix}connected_ev::battery_capacity_kwh",
                    ]
                )
            else:
                candidates.append(f"{entity_prefix}{feature}")

        if charger_id:
            if feature in {"connected_state", "incoming_state"}:
                candidates.append(f"electric_vehicle_charger_{charger_id}_{feature}")
            else:
                candidates.append(f"connected_electric_vehicle_at_charger_{charger_id}_{feature}")
                candidates.append(f"incoming_electric_vehicle_at_charger_{charger_id}_{feature}")
        candidates.extend(generic_names)

        for name in candidates:
            if name in obs_map:
                return self._get_value(obs, obs_map, name, default=default)
        return default

    def _get_ev_departure_hours(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        *,
        charger_id: Optional[str],
        building_name: Optional[str],
        default: float,
    ) -> float:
        if charger_id and building_name:
            entity_prefix = f"charger::{building_name}/{charger_id}::"
            hours_name = f"{entity_prefix}hours_until_departure"
            if hours_name in obs_map:
                value = self._get_value(obs, obs_map, hours_name, default=float("nan"))
                if not math.isnan(value) and value >= 0.0:
                    return value

            steps_name = f"{entity_prefix}connected_ev_departure_time_step"
            if steps_name in obs_map:
                value = self._get_value(obs, obs_map, steps_name, default=float("nan"))
                if not math.isnan(value) and value >= 0.0:
                    return value * self.step_hours

        if charger_id:
            value = self._get_ev_value(
                obs,
                obs_map,
                charger_id=charger_id,
                building_name=building_name,
                feature="departure_time",
                generic_names=[],
                default=float("nan"),
            )
            if not math.isnan(value):
                return max(value, 0.0) * self.step_hours

        hour = self._get_value(obs, obs_map, "hour", default=0.0)
        minute = self._get_value(obs, obs_map, "minute", default=0.0)
        current_time = (hour % 24.0) + minute / 60.0
        departure_time = self._get_value(
            obs,
            obs_map,
            "electric_vehicle_departure_time",
            default=current_time + default,
        )
        if "hour" not in obs_map and ("district__hour" in obs_map or any(name.startswith("charger::") for name in obs_map)):
            return max(departure_time, 0.0) * self.step_hours

        departure_time = departure_time % 24.0
        time_to_departure = departure_time - current_time
        if time_to_departure <= 0:
            time_to_departure += 24.0
        return time_to_departure

    @staticmethod
    def _soc_gap_fraction(current_soc: float, required_soc: float) -> float:
        gap = max(required_soc - current_soc, 0.0)
        if max(abs(current_soc), abs(required_soc)) > 1.5:
            gap /= 100.0
        return gap

    def _is_deferrable_action_name(self, action_name: str, obs_map: Dict[str, int]) -> bool:
        name = str(action_name or "")
        if name.startswith("deferrable_appliance") or name.endswith("::start"):
            return True
        if name == "start" and self._deferrable_observation_prefixes(obs_map):
            return True
        return False

    def _compute_deferrable_action(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        action_name: str,
        bounds: Sequence[float],
    ) -> float:
        prefix = self._resolve_deferrable_observation_prefix(action_name, obs_map)
        if prefix is None:
            return 0.0

        pending = self._get_deferrable_value(obs, obs_map, prefix, "pending", default=0.0)
        running = self._get_deferrable_value(obs, obs_map, prefix, "running", default=0.0)
        can_start = self._get_deferrable_value(obs, obs_map, prefix, "can_start", default=0.0)
        deadline_missed = self._get_deferrable_value(obs, obs_map, prefix, "deadline_missed", default=0.0)

        if pending <= 0.5 or running > 0.5 or can_start <= 0.5 or deadline_missed > 0.5:
            return 0.0

        urgency = self._get_deferrable_value(obs, obs_map, prefix, "urgency_ratio", default=0.0)
        slack = self._get_deferrable_value(obs, obs_map, prefix, "slack_ratio", default=1.0)
        priority = self._get_deferrable_value(obs, obs_map, prefix, "priority", default=0.0)

        should_start = (
            urgency >= self.deferrable_urgency_threshold
            or slack <= self.deferrable_slack_threshold
            or priority >= self.deferrable_priority_threshold
        )
        if not should_start:
            return 0.0

        low, high = bounds
        return float(np.clip(self.deferrable_start_action, low, high))

    @staticmethod
    def _deferrable_observation_prefixes(obs_map: Dict[str, int]) -> List[str]:
        prefixes: List[str] = []
        seen: set[str] = set()
        for name in obs_map:
            prefix = RuleBasedPolicy._parse_deferrable_observation_prefix(name)
            if prefix is None or prefix in seen:
                continue
            seen.add(prefix)
            prefixes.append(prefix)
        return prefixes

    @staticmethod
    def _parse_deferrable_observation_prefix(name: str) -> Optional[str]:
        raw = str(name or "")
        for feature in ("pending", "can_start", "running", "deadline_missed"):
            entity_suffix = f"::{feature}"
            if raw.startswith("deferrable_appliance::") and raw.endswith(entity_suffix):
                return raw[: -len(entity_suffix)]

            flat_suffix = f"_{feature}"
            if raw.startswith("deferrable_appliance_") and raw.endswith(flat_suffix):
                return raw[: -len(flat_suffix)]

        return None

    def _resolve_deferrable_observation_prefix(
        self,
        action_name: str,
        obs_map: Dict[str, int],
    ) -> Optional[str]:
        prefixes = self._deferrable_observation_prefixes(obs_map)
        if not prefixes:
            return None
        if len(prefixes) == 1:
            return prefixes[0]

        action_tokens = self._deferrable_action_tokens(action_name)
        for prefix in prefixes:
            prefix_tokens = self._deferrable_prefix_tokens(prefix)
            if action_tokens & prefix_tokens:
                return prefix

        return None

    @staticmethod
    def _deferrable_action_tokens(action_name: str) -> set[str]:
        raw = str(action_name or "").strip()
        tokens = {raw} if raw else set()

        candidates = [raw]
        if raw.startswith("deferrable_appliance::"):
            candidates.append(raw[len("deferrable_appliance::") :])
        if raw.startswith("deferrable_appliance_"):
            candidates.append(raw[len("deferrable_appliance_") :])
        if raw.startswith("start_"):
            candidates.append(raw[len("start_") :])
        if raw.endswith("::start"):
            candidates.append(raw[: -len("::start")])

        for candidate in list(candidates):
            if not candidate:
                continue
            tokens.add(candidate)
            if candidate.startswith("deferrable_appliance::"):
                tokens.add(candidate[len("deferrable_appliance::") :])
            if candidate.startswith("deferrable_appliance_"):
                tokens.add(candidate[len("deferrable_appliance_") :])
            if candidate.endswith("::start"):
                tokens.add(candidate[: -len("::start")])
            if "/" in candidate:
                tokens.add(candidate.split("/", 1)[1])

        return {token for token in tokens if token}

    @staticmethod
    def _deferrable_prefix_tokens(prefix: str) -> set[str]:
        raw = str(prefix or "").strip()
        tokens = {raw} if raw else set()

        if raw.startswith("deferrable_appliance::"):
            normalized = raw[len("deferrable_appliance::") :]
            tokens.add(normalized)
        elif raw.startswith("deferrable_appliance_"):
            normalized = raw[len("deferrable_appliance_") :]
            tokens.add(normalized)
        else:
            normalized = raw

        if "/" in normalized:
            tokens.add(normalized.split("/", 1)[1])

        return {token for token in tokens if token}

    def _get_deferrable_value(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        prefix: str,
        feature: str,
        default: float = 0.0,
    ) -> float:
        for name in (f"{prefix}::{feature}", f"{prefix}_{feature}"):
            if name in obs_map:
                return self._get_value(obs, obs_map, name, default=default)
        return default

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
        storage_info: Dict[str, StorageInfo] = {}
        charger_to_building: Dict[str, str] = {}

        for building_name, data in schema.get("buildings", {}).items():
            storage_data = data.get("electrical_storage") or {}
            storage_attrs = storage_data.get("attributes", {}) if isinstance(storage_data, dict) else {}
            if storage_attrs:
                storage_info[building_name] = StorageInfo(
                    nominal_power=float(storage_attrs.get("nominal_power", storage_attrs.get("power", 0.0)) or 0.0),
                    capacity=float(storage_attrs.get("capacity", storage_attrs.get("energy_capacity", 0.0)) or 0.0),
                    phase_connection=storage_attrs.get("phase_connection"),
                )

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
                    phase_connection=attributes.get("phase_connection"),
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
            "storage_info": storage_info,
        }
