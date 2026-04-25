"""Utilities to adapt CityLearn entity payloads to per-building agent views."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from gymnasium import spaces
from loguru import logger
import numpy as np


class EntityContractAdapter:
    """Convert entity observations/actions to algorithm-friendly per-building structures."""

    _LEGACY_CHARGER_ALIASES = {
        "electric_vehicle_charger_state": "connected_state",
        "electric_vehicle_soc": "connected_ev_soc",
        "electric_vehicle_required_soc_departure": "connected_ev_required_soc_departure",
        "electric_vehicle_departure_time": "connected_ev_departure_time_step",
    }

    def __init__(self, env: Any, *, normalization_enabled: bool, clip: bool):
        self.env = env
        self.normalization_enabled = bool(normalization_enabled)
        self.clip = bool(clip)
        self.topology_version: Optional[int] = None

        self._building_ids: List[str] = []
        self._district_features: List[str] = []
        self._building_features: List[str] = []
        self._charger_features: List[str] = []
        self._storage_features: List[str] = []
        self._pv_features: List[str] = []
        self._ev_features: List[str] = []
        self._charger_ids: List[str] = []
        self._storage_ids: List[str] = []
        self._pv_ids: List[str] = []

        self._building_action_features: List[str] = []
        self._charger_action_features: List[str] = []
        self._building_action_col_by_name: Dict[str, int] = {}
        self._charger_action_col_by_name: Dict[str, int] = {}
        self._charger_row_by_id: Dict[str, int] = {}

        self._latest_observation_origins: List[List[Tuple[str, str]]] = []
        self._warned_invalid_bounds: set[str] = set()

    @staticmethod
    def _safe_scalar(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if np.isnan(parsed) or np.isinf(parsed):
            return float(default)
        return parsed

    @staticmethod
    def _as_2d(array_like: Any) -> np.ndarray:
        arr = np.asarray(array_like, dtype=np.float64)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    @staticmethod
    def _table_spaces_from_observation_space(observation_space: Any) -> Mapping[str, Any]:
        if hasattr(observation_space, "spaces"):
            tables = observation_space.spaces.get("tables")
            if hasattr(tables, "spaces"):
                return tables.spaces
        return {}

    @staticmethod
    def _edge_targets(edge_array: Any, source_index: int) -> List[int]:
        arr = np.asarray(edge_array, dtype=np.int64)
        if arr.size == 0:
            return []
        if arr.ndim != 2 or arr.shape[1] < 2:
            return []

        rows = arr[arr[:, 0] == int(source_index)]
        if rows.size == 0:
            return []
        return sorted(int(v) for v in rows[:, 1].tolist())

    @staticmethod
    def _charger_ev_row_map(
        pair_array: Any,
        mask_array: Any,
    ) -> Dict[int, int]:
        pairs = np.asarray(pair_array, dtype=np.int64)
        mask = np.asarray(mask_array, dtype=np.float64)
        mapping: Dict[int, int] = {}

        if pairs.ndim != 2 or pairs.shape[1] < 2:
            return mapping

        for idx in range(min(pairs.shape[0], mask.shape[0])):
            if mask[idx] > 0.5:
                mapping[int(pairs[idx, 0])] = int(pairs[idx, 1])

        return mapping

    @staticmethod
    def _parse_topology_version(observation_payload: Mapping[str, Any]) -> int:
        meta = observation_payload.get("meta", {}) if isinstance(observation_payload, Mapping) else {}
        try:
            return int(meta.get("topology_version", 0))
        except (TypeError, ValueError):
            return 0

    def needs_rebuild(self, observation_payload: Mapping[str, Any]) -> bool:
        current_version = self._parse_topology_version(observation_payload)
        return self.topology_version is None or current_version != self.topology_version

    def rebuild_layout(self, observation_payload: Mapping[str, Any]) -> None:
        specs = self.env.entity_specs
        table_specs = specs.get("tables", {})
        action_specs = specs.get("actions", {})

        self._building_ids = list(table_specs.get("building", {}).get("ids", []))
        self._district_features = list(table_specs.get("district", {}).get("features", []))
        self._building_features = list(table_specs.get("building", {}).get("features", []))
        self._charger_features = list(table_specs.get("charger", {}).get("features", []))
        self._storage_features = list(table_specs.get("storage", {}).get("features", []))
        self._pv_features = list(table_specs.get("pv", {}).get("features", []))
        self._ev_features = list(table_specs.get("ev", {}).get("features", []))

        self._charger_ids = list(table_specs.get("charger", {}).get("ids", []))
        self._storage_ids = list(table_specs.get("storage", {}).get("ids", []))
        self._pv_ids = list(table_specs.get("pv", {}).get("ids", []))
        self._charger_row_by_id = {entity_id: row for row, entity_id in enumerate(self._charger_ids)}

        self._building_action_features = list(action_specs.get("building", {}).get("features", []))
        self._charger_action_features = list(action_specs.get("charger", {}).get("features", []))
        self._building_action_col_by_name = {
            name: idx for idx, name in enumerate(self._building_action_features)
        }
        self._charger_action_col_by_name = {
            name: idx for idx, name in enumerate(self._charger_action_features)
        }

        self.topology_version = self._parse_topology_version(observation_payload)

    def to_agent_observations(
        self,
        observation_payload: Mapping[str, Any],
    ) -> Tuple[List[np.ndarray], List[List[str]], List[spaces.Box]]:
        if self.needs_rebuild(observation_payload):
            self.rebuild_layout(observation_payload)

        tables = observation_payload.get("tables", {})
        edges = observation_payload.get("edges", {})
        table_spaces = self._table_spaces_from_observation_space(self.env.observation_space)

        district_table = self._as_2d(tables.get("district", []))
        building_table = self._as_2d(tables.get("building", []))
        charger_table = self._as_2d(tables.get("charger", []))
        storage_table = self._as_2d(tables.get("storage", []))
        pv_table = self._as_2d(tables.get("pv", []))
        ev_table = self._as_2d(tables.get("ev", []))

        district_space = table_spaces.get("district")
        building_space = table_spaces.get("building")
        charger_space = table_spaces.get("charger")
        storage_space = table_spaces.get("storage")
        pv_space = table_spaces.get("pv")
        ev_space = table_spaces.get("ev")

        district_low = self._as_2d(getattr(district_space, "low", np.zeros_like(district_table)))
        district_high = self._as_2d(getattr(district_space, "high", np.zeros_like(district_table)))
        building_low = self._as_2d(getattr(building_space, "low", np.zeros_like(building_table)))
        building_high = self._as_2d(getattr(building_space, "high", np.zeros_like(building_table)))
        charger_low = self._as_2d(getattr(charger_space, "low", np.zeros_like(charger_table)))
        charger_high = self._as_2d(getattr(charger_space, "high", np.zeros_like(charger_table)))
        storage_low = self._as_2d(getattr(storage_space, "low", np.zeros_like(storage_table)))
        storage_high = self._as_2d(getattr(storage_space, "high", np.zeros_like(storage_table)))
        pv_low = self._as_2d(getattr(pv_space, "low", np.zeros_like(pv_table)))
        pv_high = self._as_2d(getattr(pv_space, "high", np.zeros_like(pv_table)))
        ev_low = self._as_2d(getattr(ev_space, "low", np.zeros_like(ev_table)))
        ev_high = self._as_2d(getattr(ev_space, "high", np.zeros_like(ev_table)))

        charger_connected_ev = self._charger_ev_row_map(
            edges.get("charger_to_ev_connected", []),
            edges.get("charger_to_ev_connected_mask", []),
        )
        charger_incoming_ev = self._charger_ev_row_map(
            edges.get("charger_to_ev_incoming", []),
            edges.get("charger_to_ev_incoming_mask", []),
        )

        observations: List[np.ndarray] = []
        observation_names: List[List[str]] = []
        observation_spaces: List[spaces.Box] = []
        observation_origins: List[List[Tuple[str, str]]] = []

        charger_feature_col = {name: idx for idx, name in enumerate(self._charger_features)}

        for building_index, building_id in enumerate(self._building_ids):
            names: List[str] = []
            values: List[float] = []
            lows: List[float] = []
            highs: List[float] = []
            origins: List[Tuple[str, str]] = []

            def add_feature(name: str, value: float, low: float, high: float, origin: Tuple[str, str]):
                names.append(str(name))
                values.append(self._safe_scalar(value, 0.0))
                lows.append(self._safe_scalar(low, -1.0e6))
                highs.append(self._safe_scalar(high, 1.0e6))
                origins.append(origin)

            for col, feature in enumerate(self._district_features):
                add_feature(
                    f"district__{feature}",
                    district_table[0, col] if district_table.shape[1] > col else 0.0,
                    district_low[0, col] if district_low.shape[1] > col else -1.0e6,
                    district_high[0, col] if district_high.shape[1] > col else 1.0e6,
                    ("district", feature),
                )

            for col, feature in enumerate(self._building_features):
                add_feature(
                    feature,
                    building_table[building_index, col] if building_table.shape[1] > col and building_table.shape[0] > building_index else 0.0,
                    building_low[building_index, col] if building_low.shape[1] > col and building_low.shape[0] > building_index else -1.0e6,
                    building_high[building_index, col] if building_high.shape[1] > col and building_high.shape[0] > building_index else 1.0e6,
                    ("building", feature),
                )

            building_chargers = self._edge_targets(edges.get("building_to_charger", []), building_index)
            building_storages = self._edge_targets(edges.get("building_to_storage", []), building_index)
            building_pvs = self._edge_targets(edges.get("building_to_pv", []), building_index)

            for row in building_storages:
                storage_id = self._storage_ids[row] if row < len(self._storage_ids) else f"storage_{row}"
                for col, feature in enumerate(self._storage_features):
                    add_feature(
                        f"storage::{storage_id}::{feature}",
                        storage_table[row, col] if storage_table.shape[0] > row and storage_table.shape[1] > col else 0.0,
                        storage_low[row, col] if storage_low.shape[0] > row and storage_low.shape[1] > col else -1.0e6,
                        storage_high[row, col] if storage_high.shape[0] > row and storage_high.shape[1] > col else 1.0e6,
                        ("storage", feature),
                    )

            for row in building_pvs:
                pv_id = self._pv_ids[row] if row < len(self._pv_ids) else f"pv_{row}"
                for col, feature in enumerate(self._pv_features):
                    add_feature(
                        f"pv::{pv_id}::{feature}",
                        pv_table[row, col] if pv_table.shape[0] > row and pv_table.shape[1] > col else 0.0,
                        pv_low[row, col] if pv_low.shape[0] > row and pv_low.shape[1] > col else -1.0e6,
                        pv_high[row, col] if pv_high.shape[0] > row and pv_high.shape[1] > col else 1.0e6,
                        ("pv", feature),
                    )

            for row in building_chargers:
                charger_id = self._charger_ids[row] if row < len(self._charger_ids) else f"charger_{row}"
                for col, feature in enumerate(self._charger_features):
                    add_feature(
                        f"charger::{charger_id}::{feature}",
                        charger_table[row, col] if charger_table.shape[0] > row and charger_table.shape[1] > col else 0.0,
                        charger_low[row, col] if charger_low.shape[0] > row and charger_low.shape[1] > col else -1.0e6,
                        charger_high[row, col] if charger_high.shape[0] > row and charger_high.shape[1] > col else 1.0e6,
                        ("charger", feature),
                    )

                def add_ev_context_features(context_label: str, ev_row: Optional[int]) -> None:
                    fallback_row = row if ev_table.shape[0] > row else 0
                    for col, feature in enumerate(self._ev_features):
                        value = 0.0
                        low_value = -1.0e6
                        high_value = 1.0e6

                        if ev_row is not None and ev_table.shape[0] > ev_row and ev_table.shape[1] > col:
                            value = ev_table[ev_row, col]
                        if ev_low.shape[0] > fallback_row and ev_low.shape[1] > col:
                            low_value = ev_low[fallback_row, col]
                        if ev_high.shape[0] > fallback_row and ev_high.shape[1] > col:
                            high_value = ev_high[fallback_row, col]

                        add_feature(
                            f"charger::{charger_id}::{context_label}::{feature}",
                            value,
                            low_value,
                            high_value,
                            ("ev", feature),
                        )

                connected_ev_row = charger_connected_ev.get(row)
                add_ev_context_features("connected_ev", connected_ev_row)

                incoming_ev_row = charger_incoming_ev.get(row)
                add_ev_context_features("incoming_ev", incoming_ev_row)

            # Operational counters per building.
            add_feature("active_chargers_count", float(len(building_chargers)), 0.0, float(max(len(self._charger_ids), 1)), ("meta", "active_chargers_count"))
            add_feature("active_storages_count", float(len(building_storages)), 0.0, float(max(len(self._storage_ids), 1)), ("meta", "active_storages_count"))
            add_feature("active_pvs_count", float(len(building_pvs)), 0.0, float(max(len(self._pv_ids), 1)), ("meta", "active_pvs_count"))

            # RBC compatibility aliases from first connected charger if available.
            if building_chargers:
                first_row = building_chargers[0]
                for legacy_name, source_feature in self._LEGACY_CHARGER_ALIASES.items():
                    source_col = charger_feature_col.get(source_feature)
                    if source_col is None:
                        continue
                    add_feature(
                        legacy_name,
                        charger_table[first_row, source_col] if charger_table.shape[0] > first_row and charger_table.shape[1] > source_col else 0.0,
                        charger_low[first_row, source_col] if charger_low.shape[0] > first_row and charger_low.shape[1] > source_col else 0.0,
                        charger_high[first_row, source_col] if charger_high.shape[0] > first_row and charger_high.shape[1] > source_col else 1.0,
                        ("charger", source_feature),
                    )
            else:
                add_feature("electric_vehicle_charger_state", 0.0, 0.0, 1.0, ("charger", "connected_state"))
                add_feature("electric_vehicle_soc", 0.0, 0.0, 100.0, ("charger", "connected_ev_soc"))
                add_feature("electric_vehicle_required_soc_departure", 0.0, 0.0, 100.0, ("charger", "connected_ev_required_soc_departure"))
                add_feature("electric_vehicle_departure_time", 0.0, 0.0, 24.0, ("charger", "connected_ev_departure_time_step"))

            add_feature("electric_vehicle_is_flexible", 1.0, 0.0, 1.0, ("meta", "electric_vehicle_is_flexible"))

            if "minute" not in names and "minutes" in names:
                minute_idx = names.index("minutes")
                add_feature("minute", values[minute_idx], lows[minute_idx], highs[minute_idx], origins[minute_idx])

            if "solar_generation" not in names and "pv_power_kw" in names:
                pv_idx = names.index("pv_power_kw")
                add_feature("solar_generation", values[pv_idx], lows[pv_idx], highs[pv_idx], origins[pv_idx])

            obs_vector = np.asarray(values, dtype=np.float64)
            low_vector = np.asarray(lows, dtype=np.float64)
            high_vector = np.asarray(highs, dtype=np.float64)

            observations.append(obs_vector)
            observation_names.append(names)
            observation_spaces.append(
                spaces.Box(
                    low=low_vector.astype(np.float32),
                    high=high_vector.astype(np.float32),
                    dtype=np.float32,
                )
            )
            observation_origins.append(origins)

        self._latest_observation_origins = observation_origins
        return observations, observation_names, observation_spaces

    def normalize_observation(
        self,
        agent_index: int,
        observation: Sequence[float],
        observation_names: Sequence[str],
        observation_space: spaces.Box,
    ) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float64)

        if not self.normalization_enabled:
            return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        low = np.asarray(observation_space.low, dtype=np.float64)
        high = np.asarray(observation_space.high, dtype=np.float64)
        denominator = high - low

        valid = np.isfinite(low) & np.isfinite(high) & np.isfinite(denominator) & (denominator > 0.0)
        normalized = values.copy()

        if np.any(valid):
            normalized[valid] = (values[valid] - low[valid]) / denominator[valid]
            if self.clip:
                normalized[valid] = np.clip(normalized[valid], 0.0, 1.0)

        if np.any(~valid):
            origins = self._latest_observation_origins[agent_index] if agent_index < len(self._latest_observation_origins) else []
            for idx in np.where(~valid)[0].tolist():
                if idx < len(origins):
                    warning_key = f"{origins[idx][0]}.{origins[idx][1]}"
                elif idx < len(observation_names):
                    warning_key = str(observation_names[idx])
                else:
                    warning_key = f"feature_{idx}"

                if warning_key in self._warned_invalid_bounds:
                    continue

                self._warned_invalid_bounds.add(warning_key)
                logger.warning(
                    "Entity normalization fallback: invalid bounds for feature '{}' (passthrough applied).",
                    warning_key,
                )

        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _charger_rows_by_building(charger_ids: Sequence[str]) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        for row, charger_id in enumerate(charger_ids):
            if not isinstance(charger_id, str) or "/" not in charger_id:
                continue
            building_name = charger_id.split("/", 1)[0]
            mapping.setdefault(building_name, []).append(row)
        return mapping

    def _match_charger_row(
        self,
        *,
        building_name: str,
        candidate: str,
        allowed_rows: Sequence[int],
    ) -> Optional[int]:
        raw = str(candidate or "").strip()
        if not raw:
            return None

        normalized = raw
        if normalized.startswith("charger::"):
            normalized = normalized[len("charger::") :]

        candidates: List[str] = [normalized]
        if "/" not in normalized:
            candidates.insert(0, f"{building_name}/{normalized}")
        elif normalized.startswith(f"{building_name}/"):
            suffix = normalized[len(building_name) + 1 :]
            if suffix:
                candidates.append(suffix)

        allowed = set(int(row) for row in allowed_rows)
        for charger_id in candidates:
            row = self._charger_row_by_id.get(charger_id)
            if row is None:
                continue
            if allowed and row not in allowed:
                continue
            return int(row)

        return None

    def _resolve_charger_action_target(
        self,
        *,
        action_name: str,
        building_name: str,
        building_charger_rows: Sequence[int],
    ) -> Tuple[Optional[int], Optional[str]]:
        action = str(action_name or "").strip()
        if not action or not self._charger_action_features:
            return None, None

        if action in self._charger_action_col_by_name:
            if len(building_charger_rows) == 1:
                return int(building_charger_rows[0]), action
            return None, None

        for feature in self._charger_action_features:
            prefix = f"{feature}_"
            if action.startswith(prefix):
                row = self._match_charger_row(
                    building_name=building_name,
                    candidate=action[len(prefix) :],
                    allowed_rows=building_charger_rows,
                )
                if row is not None:
                    return row, feature

            suffix_token = f"::{feature}"
            if action.endswith(suffix_token):
                charger_ref = action[: -len(suffix_token)]
                if charger_ref.startswith("charger::"):
                    charger_ref = charger_ref[len("charger::") :]

                row = self._match_charger_row(
                    building_name=building_name,
                    candidate=charger_ref,
                    allowed_rows=building_charger_rows,
                )
                if row is not None:
                    return row, feature

        if len(building_charger_rows) == 1:
            for feature in self._charger_action_features:
                if feature in action:
                    return int(building_charger_rows[0]), feature

        return None, None

    def to_entity_actions(
        self,
        actions: Sequence[Sequence[float]],
        action_names: Sequence[Sequence[str]],
    ) -> Mapping[str, Any]:
        specs = self.env.entity_specs
        action_specs = specs.get("actions", {})
        building_ids = list(action_specs.get("building", {}).get("ids", []))
        charger_ids = list(action_specs.get("charger", {}).get("ids", []))

        self._building_action_features = list(action_specs.get("building", {}).get("features", []))
        self._charger_action_features = list(action_specs.get("charger", {}).get("features", []))
        self._building_action_col_by_name = {
            name: idx for idx, name in enumerate(self._building_action_features)
        }
        self._charger_action_col_by_name = {
            name: idx for idx, name in enumerate(self._charger_action_features)
        }
        self._charger_row_by_id = {entity_id: row for row, entity_id in enumerate(charger_ids)}
        charger_rows_by_building = self._charger_rows_by_building(charger_ids)

        building_table = np.zeros(
            (len(building_ids), len(self._building_action_features)),
            dtype=np.float32,
        )
        charger_table = np.zeros(
            (len(charger_ids), len(self._charger_action_features)),
            dtype=np.float32,
        )

        for building_index, building_actions in enumerate(actions):
            if building_index >= len(action_names) or building_index >= len(building_ids):
                continue

            names = action_names[building_index]
            building_name = building_ids[building_index]
            building_charger_rows = charger_rows_by_building.get(building_name, [])

            for position, action_name in enumerate(names):
                value = 0.0
                if position < len(building_actions):
                    value = self._safe_scalar(building_actions[position], 0.0)

                if action_name in self._building_action_col_by_name:
                    col = self._building_action_col_by_name[action_name]
                    building_table[building_index, col] = value
                    continue

                charger_row, charger_feature = self._resolve_charger_action_target(
                    action_name=str(action_name),
                    building_name=building_name,
                    building_charger_rows=building_charger_rows,
                )
                if charger_row is None or charger_feature is None:
                    continue

                charger_col = self._charger_action_col_by_name.get(charger_feature)
                if charger_col is None:
                    continue
                charger_table[charger_row, charger_col] = value

        return {
            "tables": {
                "building": building_table,
                "charger": charger_table,
            }
        }
