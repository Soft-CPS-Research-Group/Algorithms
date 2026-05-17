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

    def __init__(
        self,
        env: Any,
        *,
        normalization_enabled: bool,
        clip: bool,
        encoding_profile: str = "minmax_space",
    ):
        self.env = env
        self.normalization_enabled = bool(normalization_enabled)
        self.clip = bool(clip)
        self.encoding_profile = str(encoding_profile or "minmax_space").strip().lower()
        if self.encoding_profile not in {"minmax_space", "maddpg_v1", "maddpg_v2_compact"}:
            logger.warning(
                "Unsupported entity encoding profile '{}'; falling back to 'minmax_space'.",
                self.encoding_profile,
            )
            self.encoding_profile = "minmax_space"
        self.seconds_per_time_step = self._safe_scalar(
            getattr(getattr(env, "unwrapped", env), "seconds_per_time_step", 3600.0),
            default=3600.0,
        )
        if self.seconds_per_time_step <= 0.0:
            self.seconds_per_time_step = 3600.0
        self.steps_per_day = max(1.0, 86400.0 / self.seconds_per_time_step)
        self.topology_version: Optional[int] = None

        self._building_ids: List[str] = []
        self._district_features: List[str] = []
        self._building_features: List[str] = []
        self._charger_features: List[str] = []
        self._storage_features: List[str] = []
        self._pv_features: List[str] = []
        self._ev_features: List[str] = []
        self._deferrable_features: List[str] = []
        self._charger_ids: List[str] = []
        self._storage_ids: List[str] = []
        self._pv_ids: List[str] = []
        self._deferrable_ids: List[str] = []

        self._building_action_features: List[str] = []
        self._charger_action_features: List[str] = []
        self._deferrable_action_features: List[str] = []
        self._building_action_col_by_name: Dict[str, int] = {}
        self._charger_action_col_by_name: Dict[str, int] = {}
        self._deferrable_action_col_by_name: Dict[str, int] = {}
        self._charger_row_by_id: Dict[str, int] = {}
        self._deferrable_row_by_id: Dict[str, int] = {}

        self._latest_observation_origins: List[List[Tuple[str, str]]] = []
        self._latest_encoded_observation_names: List[List[str]] = []
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
        self._deferrable_features = list(table_specs.get("deferrable_appliance", {}).get("features", []))

        self._charger_ids = list(table_specs.get("charger", {}).get("ids", []))
        self._storage_ids = list(table_specs.get("storage", {}).get("ids", []))
        self._pv_ids = list(table_specs.get("pv", {}).get("ids", []))
        self._deferrable_ids = list(table_specs.get("deferrable_appliance", {}).get("ids", []))
        self._charger_row_by_id = {entity_id: row for row, entity_id in enumerate(self._charger_ids)}
        self._deferrable_row_by_id = {entity_id: row for row, entity_id in enumerate(self._deferrable_ids)}

        self._building_action_features = list(action_specs.get("building", {}).get("features", []))
        self._charger_action_features = list(action_specs.get("charger", {}).get("features", []))
        self._deferrable_action_features = list(action_specs.get("deferrable_appliance", {}).get("features", []))
        self._building_action_col_by_name = {
            name: idx for idx, name in enumerate(self._building_action_features)
        }
        self._charger_action_col_by_name = {
            name: idx for idx, name in enumerate(self._charger_action_features)
        }
        self._deferrable_action_col_by_name = {
            name: idx for idx, name in enumerate(self._deferrable_action_features)
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
        deferrable_table = self._as_2d(tables.get("deferrable_appliance", []))

        district_space = table_spaces.get("district")
        building_space = table_spaces.get("building")
        charger_space = table_spaces.get("charger")
        storage_space = table_spaces.get("storage")
        pv_space = table_spaces.get("pv")
        ev_space = table_spaces.get("ev")
        deferrable_space = table_spaces.get("deferrable_appliance")

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
        deferrable_low = self._as_2d(getattr(deferrable_space, "low", np.zeros_like(deferrable_table)))
        deferrable_high = self._as_2d(getattr(deferrable_space, "high", np.zeros_like(deferrable_table)))

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
            building_deferrables = self._edge_targets(
                edges.get("building_to_deferrable_appliance", []),
                building_index,
            )

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

            for row in building_deferrables:
                appliance_id = (
                    self._deferrable_ids[row]
                    if row < len(self._deferrable_ids)
                    else f"deferrable_appliance_{row}"
                )
                for col, feature in enumerate(self._deferrable_features):
                    add_feature(
                        f"deferrable_appliance::{appliance_id}::{feature}",
                        deferrable_table[row, col] if deferrable_table.shape[0] > row and deferrable_table.shape[1] > col else 0.0,
                        deferrable_low[row, col] if deferrable_low.shape[0] > row and deferrable_low.shape[1] > col else -1.0e6,
                        deferrable_high[row, col] if deferrable_high.shape[0] > row and deferrable_high.shape[1] > col else 1.0e6,
                        ("deferrable_appliance", feature),
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
            add_feature(
                "active_deferrable_appliances_count",
                float(len(building_deferrables)),
                0.0,
                float(max(len(self._deferrable_ids), 1)),
                ("meta", "active_deferrable_appliances_count"),
            )

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

        if self.encoding_profile in {"maddpg_v1", "maddpg_v2_compact"}:
            encoded, encoded_names = self._encode_maddpg_v1(
                observation=values,
                observation_names=observation_names,
                observation_space=observation_space,
            )
            if self.encoding_profile == "maddpg_v2_compact":
                keep_indices = [
                    idx
                    for idx, name in enumerate(encoded_names)
                    if self._is_maddpg_v2_compact_feature(name)
                ]
                encoded = encoded[keep_indices]
                encoded_names = [encoded_names[idx] for idx in keep_indices]
            while len(self._latest_encoded_observation_names) <= agent_index:
                self._latest_encoded_observation_names.append([])
            self._latest_encoded_observation_names[agent_index] = encoded_names
            return encoded

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

    def encoded_observation_names(self, observation_names: Sequence[Sequence[str]]) -> List[List[str]]:
        if self.encoding_profile not in {"maddpg_v1", "maddpg_v2_compact"}:
            return [[str(name) for name in group] for group in observation_names]

        encoded_names: List[List[str]] = []
        for group in observation_names:
            names = self._maddpg_v1_encoded_names(group)
            if self.encoding_profile == "maddpg_v2_compact":
                names = [name for name in names if self._is_maddpg_v2_compact_feature(name)]
            encoded_names.append(names)
        return encoded_names

    def _encode_maddpg_v1(
        self,
        *,
        observation: np.ndarray,
        observation_names: Sequence[str],
        observation_space: spaces.Box,
    ) -> Tuple[np.ndarray, List[str]]:
        low = np.asarray(observation_space.low, dtype=np.float64)
        high = np.asarray(observation_space.high, dtype=np.float64)
        values_by_name = {
            str(name): self._safe_scalar(observation[index], 0.0)
            for index, name in enumerate(observation_names)
        }
        low_by_name = {
            str(name): self._safe_scalar(low[index], -1.0e6)
            for index, name in enumerate(observation_names)
        }
        high_by_name = {
            str(name): self._safe_scalar(high[index], 1.0e6)
            for index, name in enumerate(observation_names)
        }

        encoded: List[float] = []
        encoded_names: List[str] = []
        emitted_time_of_day = False

        def append(name: str, value: float) -> None:
            encoded_names.append(name)
            encoded.append(float(value))

        for index, raw_name in enumerate(observation_names):
            name = str(raw_name)
            value = self._safe_scalar(observation[index], 0.0)
            lo = self._safe_scalar(low[index], -1.0e6)
            hi = self._safe_scalar(high[index], 1.0e6)

            # These aliases are generated only for legacy RBC compatibility.
            # The profile keeps the simulator/entity EV observations instead.
            if name.startswith("electric_vehicle_"):
                continue
            if name in {"minute", "solar_generation"}:
                continue

            if name.startswith("district__"):
                feature = name.split("__", 1)[1]
                if feature == "topology_version":
                    continue

                if feature == "month":
                    sin_value, cos_value = self._cyclic_pair(value, period=12.0, offset=1.0)
                    append("district__month_sin", sin_value)
                    append("district__month_cos", cos_value)
                    continue

                if feature == "day_type":
                    sin_value, cos_value = self._cyclic_pair(value, period=7.0, offset=1.0)
                    append("district__day_type_sin", sin_value)
                    append("district__day_type_cos", cos_value)
                    append("district__is_weekend", 1.0 if int(round(value)) in {6, 7} else 0.0)
                    continue

                if feature in {"hour", "minutes", "seconds"}:
                    if not emitted_time_of_day:
                        time_sin, time_cos = self._time_of_day_pair(values_by_name)
                        append("district__time_of_day_sin", time_sin)
                        append("district__time_of_day_cos", time_cos)
                        emitted_time_of_day = True
                    continue

            if self._is_binary_feature(name):
                append(name, self._binary(value))
                continue

            if self._is_soc_feature(name):
                append(name, self._soc_fraction(value))
                if name.endswith("connected_ev_required_soc_departure"):
                    soc_name = self._replace_last_feature(name, "connected_ev_soc")
                    soc = self._soc_fraction(values_by_name.get(soc_name, value))
                    required = self._soc_fraction(value)
                    append(self._replace_last_feature(name, "connected_ev_soc_deficit"), max(required - soc, 0.0))
                if name.endswith("incoming_ev_required_soc_departure"):
                    soc_name = self._replace_last_feature(name, "incoming_ev_estimated_soc_arrival")
                    soc = self._soc_fraction(values_by_name.get(soc_name, value))
                    required = self._soc_fraction(value)
                    append(self._replace_last_feature(name, "incoming_ev_soc_deficit"), max(required - soc, 0.0))
                continue

            if name.endswith("connected_ev_departure_time_step"):
                hours = self._steps_until_to_hours(value)
                append(self._replace_last_feature(name, "connected_ev_hours_until_departure"), self._hours_24(hours))
                append(self._replace_last_feature(name, "connected_ev_departure_available"), 1.0 if value >= 0.0 else 0.0)
                append(self._replace_last_feature(name, "connected_ev_departure_urgency_24h"), self._urgency_24(hours))
                continue

            if name.endswith("incoming_ev_estimated_arrival_time_step"):
                hours = self._steps_until_to_hours(value)
                append(self._replace_last_feature(name, "incoming_ev_hours_until_arrival"), self._hours_24(hours))
                append(self._replace_last_feature(name, "incoming_ev_arrival_available"), 1.0 if value >= 0.0 else 0.0)
                append(self._replace_last_feature(name, "incoming_ev_arrival_urgency_24h"), self._urgency_24(hours))
                continue

            if name.endswith("incoming_ev_departure_time_step"):
                hours = self._steps_until_to_hours(value)
                append(self._replace_last_feature(name, "incoming_ev_hours_until_departure_from_time_step"), self._hours_24(hours))
                append(self._replace_last_feature(name, "incoming_ev_departure_available"), 1.0 if value >= 0.0 else 0.0)
                append(self._replace_last_feature(name, "incoming_ev_departure_urgency_24h"), self._urgency_24(hours))
                continue

            if name.startswith("deferrable_appliance::") and name.endswith("_time_step"):
                available = 1.0 if value >= 0.0 else 0.0
                sin_value, cos_value = self._step_time_of_day_pair(value)
                base = name.removesuffix("_time_step")
                append(f"{base}_time_of_day_sin", sin_value)
                append(f"{base}_time_of_day_cos", cos_value)
                append(f"{base}_available", available)
                continue

            if "hours_until_" in name:
                append(f"{name}_24h", self._hours_24(value))
                continue

            if name.endswith("slack_steps") or name.endswith("cycle_duration_steps") or name.endswith("remaining_duration_steps"):
                append(f"{name}_day_ratio", self._steps_day_ratio(value))
                continue

            if name.endswith("remaining_energy_kwh"):
                cycle_energy = values_by_name.get(self._replace_last_feature(name, "cycle_energy_kwh"), 0.0)
                append(f"{name}_cycle_ratio", self._safe_ratio(value, cycle_energy, default=0.0))
                continue

            if name.endswith("current_step_energy_kwh"):
                cycle_energy = values_by_name.get(self._replace_last_feature(name, "cycle_energy_kwh"), 0.0)
                append(f"{name}_cycle_ratio", self._safe_ratio(value, cycle_energy, default=0.0))
                continue

            if name.endswith("cycle_peak_step_offset_ratio"):
                append(name, float(np.clip(value, -1.0, 1.0)))
                continue

            if name.endswith("_ratio") or name.endswith("priority"):
                append(name, 0.0 if value < 0.0 else float(np.clip(value, 0.0, 1.0)))
                continue

            if name.endswith("last_charged_kwh") or name.endswith("electricity_consumption_kwh") or name == "net_electricity_consumption":
                append(name, self._signed_by_bounds(value, lo, hi))
                continue

            if "headroom_kw" in name:
                append(name, self._headroom_ratio(name=name, value=value, low=lo, high=hi))
                continue

            if name.endswith("constraint_violation_kwh"):
                append(name, self._positive_by_high(value, hi, fallback=0.5))
                continue

            if name.endswith("_kwh_step"):
                append(
                    name,
                    self._step_energy_ratio(
                        name=name,
                        value=value,
                        low=lo,
                        high=hi,
                        values_by_name=values_by_name,
                    ),
                )
                continue

            if (
                name.endswith("energy_to_required_soc_kwh")
                or name.endswith("energy_to_full_kwh")
                or name.endswith("energy_available_kwh")
            ):
                append(
                    name,
                    self._energy_capacity_ratio(
                        name=name,
                        value=value,
                        high=hi,
                        values_by_name=values_by_name,
                    ),
                )
                continue

            if "battery_capacity_kwh" in name or name.endswith("capacity_kwh"):
                append(name, self._positive_by_high(value, hi, fallback=100.0))
                continue

            if (
                name.endswith("nominal_power_kw")
                or name.endswith("max_charging_power_kw")
                or name.endswith("max_discharging_power_kw")
            ):
                append(name, self._positive_by_high(value, hi, fallback=22.0))
                continue

            if name.endswith("_power_kw") or name.endswith("slack_kw"):
                append(
                    name,
                    self._power_ratio(
                        name=name,
                        value=value,
                        low=lo,
                        high=hi,
                        values_by_name=values_by_name,
                    ),
                )
                continue

            if "relative_humidity" in name:
                append(name, float(np.clip(value / 100.0, 0.0, 1.0)))
                continue

            if "solar_irradiance" in name:
                append(name, float(np.clip(value / 1000.0, 0.0, 1.0)))
                continue

            append(name, self._minmax(value, lo, hi))

        return np.asarray(encoded, dtype=np.float64), encoded_names

    def _maddpg_v1_encoded_names(self, observation_names: Sequence[str]) -> List[str]:
        names: List[str] = []
        emitted_time_of_day = False

        def append(name: str) -> None:
            names.append(name)

        for raw_name in observation_names:
            name = str(raw_name)

            if name.startswith("electric_vehicle_"):
                continue
            if name in {"minute", "solar_generation"}:
                continue

            if name.startswith("district__"):
                feature = name.split("__", 1)[1]
                if feature == "topology_version":
                    continue

                if feature == "month":
                    append("district__month_sin")
                    append("district__month_cos")
                    continue
                if feature == "day_type":
                    append("district__day_type_sin")
                    append("district__day_type_cos")
                    append("district__is_weekend")
                    continue
                if feature in {"hour", "minutes", "seconds"}:
                    if not emitted_time_of_day:
                        append("district__time_of_day_sin")
                        append("district__time_of_day_cos")
                        emitted_time_of_day = True
                    continue

            if self._is_soc_feature(name):
                append(name)
                if name.endswith("connected_ev_required_soc_departure"):
                    append(self._replace_last_feature(name, "connected_ev_soc_deficit"))
                if name.endswith("incoming_ev_required_soc_departure"):
                    append(self._replace_last_feature(name, "incoming_ev_soc_deficit"))
                continue

            if name.endswith("connected_ev_departure_time_step"):
                append(self._replace_last_feature(name, "connected_ev_hours_until_departure"))
                append(self._replace_last_feature(name, "connected_ev_departure_available"))
                append(self._replace_last_feature(name, "connected_ev_departure_urgency_24h"))
                continue

            if name.endswith("incoming_ev_estimated_arrival_time_step"):
                append(self._replace_last_feature(name, "incoming_ev_hours_until_arrival"))
                append(self._replace_last_feature(name, "incoming_ev_arrival_available"))
                append(self._replace_last_feature(name, "incoming_ev_arrival_urgency_24h"))
                continue

            if name.endswith("incoming_ev_departure_time_step"):
                append(self._replace_last_feature(name, "incoming_ev_hours_until_departure_from_time_step"))
                append(self._replace_last_feature(name, "incoming_ev_departure_available"))
                append(self._replace_last_feature(name, "incoming_ev_departure_urgency_24h"))
                continue

            if name.startswith("deferrable_appliance::") and name.endswith("_time_step"):
                base = name.removesuffix("_time_step")
                append(f"{base}_time_of_day_sin")
                append(f"{base}_time_of_day_cos")
                append(f"{base}_available")
                continue

            if "hours_until_" in name:
                append(f"{name}_24h")
                continue

            if name.endswith("slack_steps") or name.endswith("cycle_duration_steps") or name.endswith("remaining_duration_steps"):
                append(f"{name}_day_ratio")
                continue

            if name.endswith("remaining_energy_kwh") or name.endswith("current_step_energy_kwh"):
                append(f"{name}_cycle_ratio")
                continue

            append(name)

        return names

    @classmethod
    def _is_maddpg_v2_compact_feature(cls, name: str) -> bool:
        """Keep the compact MADDPG observation set.

        v2 keeps decision-relevant local/global state and removes semantic
        duplicates introduced by dense entity bundles, especially power/energy
        pairs, legacy topology counters, and equivalent EV timing features.
        """

        feature = cls._feature_tail_from_encoded_name(name)

        if name.startswith("electric_vehicle_"):
            return False

        if name.startswith("district__"):
            return cls._is_maddpg_v2_compact_district_feature(feature)

        if feature in {
            "active_chargers_count",
            "active_storages_count",
            "active_pvs_count",
            "active_deferrable_appliances_count",
            "electricity_consumption_kwh",
            "electrical_storage_soc",
            "electrical_storage_soc_ratio",
            "net_electricity_consumption",
            "non_shiftable_load",
            "pv_power_kw",
            "bess_power_kw",
            "bess_energy_kwh_step",
        }:
            return False

        if cls._is_redundant_energy_step_feature(feature):
            return False

        if name.startswith("pv::"):
            return feature in {
                "generation_capacity_factor_ratio",
                "generation_power_kw",
                "installed_power_kw",
            }

        if name.startswith("storage::"):
            return feature in {
                "soc",
                "soc_min_ratio",
                "energy_available_kwh",
                "energy_to_full_kwh",
                "capacity_kwh",
                "degraded_capacity_kwh",
                "nominal_power_kw",
                "max_charge_power_kw",
                "max_discharge_power_kw",
                "min_charge_power_kw",
                "min_discharge_power_kw",
                "current_efficiency_ratio",
                "round_trip_efficiency_ratio",
                "phase_connection_L1",
                "phase_connection_L2",
                "phase_connection_L3",
            }

        if name.startswith("charger::"):
            return cls._is_maddpg_v2_compact_charger_feature(feature)

        if name.startswith("deferrable_appliance::"):
            return cls._is_maddpg_v2_compact_deferrable_feature(feature)

        return True

    @staticmethod
    def _feature_tail_from_encoded_name(name: str) -> str:
        if "::" in name:
            return name.split("::")[-1]
        if name.startswith("district__"):
            return name.split("__", 1)[1]
        return name

    @staticmethod
    def _is_redundant_energy_step_feature(feature: str) -> bool:
        return (
            feature.endswith("_energy_kwh_step")
            or "_energy_prev_" in feature
            or feature.endswith("_energy_prev_1_kwh_step")
            or feature.endswith("_energy_prev_3_mean_kwh_step")
            or feature.endswith("current_step_energy_kwh_cycle_ratio")
        )

    @staticmethod
    def _is_maddpg_v2_compact_district_feature(feature: str) -> bool:
        if feature in {
            "active_buildings_count",
            "active_chargers_count",
            "active_evs_count",
            "outdoor_relative_humidity",
            "outdoor_relative_humidity_predicted_1",
            "outdoor_relative_humidity_predicted_2",
            "outdoor_relative_humidity_predicted_3",
        }:
            return False

        if "_energy_kwh_step" in feature or "_prev_" in feature:
            return False

        return feature in {
            "month_sin",
            "month_cos",
            "day_type_sin",
            "day_type_cos",
            "is_weekend",
            "time_of_day_sin",
            "time_of_day_cos",
            "carbon_intensity",
            "electricity_pricing",
            "electricity_pricing_predicted_1",
            "electricity_pricing_predicted_2",
            "electricity_pricing_predicted_3",
            "outdoor_dry_bulb_temperature",
            "outdoor_dry_bulb_temperature_predicted_1",
            "outdoor_dry_bulb_temperature_predicted_2",
            "outdoor_dry_bulb_temperature_predicted_3",
            "diffuse_solar_irradiance",
            "diffuse_solar_irradiance_predicted_1",
            "diffuse_solar_irradiance_predicted_2",
            "diffuse_solar_irradiance_predicted_3",
            "direct_solar_irradiance",
            "direct_solar_irradiance_predicted_1",
            "direct_solar_irradiance_predicted_2",
            "direct_solar_irradiance_predicted_3",
            "community_net_power_kw",
            "community_import_power_kw",
            "community_export_power_kw",
            "community_pv_power_kw",
            "community_ev_power_kw",
            "community_bess_power_kw",
            "community_building_headroom_kw",
            "community_building_export_headroom_kw",
            "community_phase_headroom_kw",
            "community_phase_export_headroom_kw",
        }

    @staticmethod
    def _is_maddpg_v2_compact_charger_feature(feature: str) -> bool:
        if feature in {
            "last_charged_kwh",
            "commanded_power_kw",
            "applied_energy_kwh_step",
            "avg_power_to_departure_kw",
            "time_until_departure_ratio",
            "hours_until_departure_24h",
            "incoming_ev_time_until_departure_ratio",
            "incoming_ev_hours_until_departure_24h",
            "connected_ev_hours_until_departure",
            "incoming_ev_hours_until_arrival",
            "incoming_ev_hours_until_departure_from_time_step",
            "connected_ev::soc_ratio",
            "connected_ev::soc_max_ratio",
            "connected_ev::soc_min_ratio",
            "connected_ev::depth_of_discharge_ratio",
            "connected_ev::energy_available_kwh",
            "connected_ev::energy_to_full_kwh",
            "incoming_ev::soc_ratio",
            "incoming_ev::soc_max_ratio",
            "incoming_ev::soc_min_ratio",
            "incoming_ev::depth_of_discharge_ratio",
            "incoming_ev::energy_available_kwh",
            "incoming_ev::energy_to_full_kwh",
        }:
            return False

        return feature in {
            "connected_state",
            "incoming_state",
            "applied_power_kw",
            "charging_slack_kw",
            "charging_priority_ratio",
            "required_average_power_kw",
            "energy_to_required_soc_kwh",
            "connected_ev_soc",
            "connected_ev_required_soc_departure",
            "connected_ev_soc_deficit",
            "connected_ev_departure_available",
            "connected_ev_departure_urgency_24h",
            "connected_ev_battery_capacity_kwh",
            "connected_ev::soc",
            "connected_ev::battery_capacity_kwh",
            "incoming_ev_estimated_soc_arrival",
            "incoming_ev_required_soc_departure",
            "incoming_ev_soc_deficit",
            "incoming_ev_arrival_available",
            "incoming_ev_arrival_urgency_24h",
            "incoming_ev_departure_available",
            "incoming_ev_departure_urgency_24h",
            "incoming_ev::soc",
            "incoming_ev::battery_capacity_kwh",
            "max_charging_power_kw",
            "max_discharging_power_kw",
            "min_charging_power_kw",
            "min_discharging_power_kw",
            "charger_efficiency_ratio",
            "charge_efficiency_at_max_ratio",
            "discharge_efficiency_at_max_ratio",
            "phase_connection_L1",
            "phase_connection_L2",
            "phase_connection_L3",
        }

    @staticmethod
    def _is_maddpg_v2_compact_deferrable_feature(feature: str) -> bool:
        if feature in {
            "earliest_start_time_of_day_sin",
            "earliest_start_time_of_day_cos",
            "earliest_start_available",
            "latest_start_time_of_day_sin",
            "latest_start_time_of_day_cos",
            "latest_start_available",
            "deadline_time_of_day_sin",
            "deadline_time_of_day_cos",
            "deadline_available",
            "slack_steps_day_ratio",
            "cycle_energy_kwh",
            "current_step_energy_kwh_cycle_ratio",
        }:
            return False

        return feature in {
            "pending",
            "running",
            "can_start",
            "must_run",
            "deadline_missed",
            "priority",
            "urgency_ratio",
            "slack_ratio",
            "hours_until_deadline_24h",
            "hours_until_latest_start_24h",
            "remaining_duration_steps_day_ratio",
            "remaining_energy_kwh_cycle_ratio",
            "remaining_average_power_kw",
            "current_step_power_kw",
            "cycle_duration_steps_day_ratio",
            "cycle_average_power_kw",
            "cycle_peak_power_kw",
            "cycle_load_factor_ratio",
            "cycle_peak_step_offset_ratio",
        }

    @staticmethod
    def _replace_last_feature(name: str, replacement: str) -> str:
        if "::" in name:
            parts = name.split("::")
            parts[-1] = replacement
            return "::".join(parts)

        prefixes = (
            "connected_electric_vehicle_at_charger_",
            "incoming_electric_vehicle_at_charger_",
            "electric_vehicle_",
        )
        for prefix in prefixes:
            if name.startswith(prefix):
                return f"{prefix}{replacement}"
        return replacement

    @staticmethod
    def _cyclic_pair(value: float, *, period: float, offset: float = 0.0) -> Tuple[float, float]:
        angle = 2.0 * np.pi * ((float(value) - offset) % period) / period
        return float(np.sin(angle)), float(np.cos(angle))

    def _time_of_day_pair(self, values_by_name: Mapping[str, float]) -> Tuple[float, float]:
        hour = self._safe_scalar(values_by_name.get("district__hour", 0.0), 0.0)
        minute = self._safe_scalar(values_by_name.get("district__minutes", 0.0), 0.0)
        second = self._safe_scalar(values_by_name.get("district__seconds", 0.0), 0.0)
        seconds_of_day = (hour % 24.0) * 3600.0 + minute * 60.0 + second
        return self._cyclic_pair(seconds_of_day, period=86400.0)

    def _step_time_of_day_pair(self, value: float) -> Tuple[float, float]:
        if value < 0.0:
            return 0.0, 0.0
        step_of_day = value % self.steps_per_day
        return self._cyclic_pair(step_of_day, period=self.steps_per_day)

    def _steps_until_to_hours(self, value: float) -> float:
        if value < 0.0:
            return 24.0
        return value * self.seconds_per_time_step / 3600.0

    @staticmethod
    def _hours_24(value: float) -> float:
        return float(np.clip(value, 0.0, 24.0) / 24.0)

    @staticmethod
    def _urgency_24(value: float) -> float:
        return float(1.0 - np.clip(value, 0.0, 24.0) / 24.0)

    def _steps_day_ratio(self, value: float) -> float:
        if value < 0.0:
            return 0.0
        return float(np.clip(value / self.steps_per_day, 0.0, 1.0))

    @staticmethod
    def _binary(value: float) -> float:
        return 1.0 if value > 0.5 else 0.0

    @staticmethod
    def _is_binary_feature(name: str) -> bool:
        binary_tokens = (
            "connected_state",
            "incoming_state",
            "pending",
            "running",
            "can_start",
            "deadline_missed",
            "must_run",
            "is_flexible",
        )
        if any(token in name for token in binary_tokens):
            return True
        if name.endswith("_available"):
            return True
        return "one_hot" in name or name.endswith("_L1") or name.endswith("_L2") or name.endswith("_L3")

    @staticmethod
    def _is_soc_feature(name: str) -> bool:
        soc_tokens = (
            "connected_ev_soc",
            "connected_ev_required_soc_departure",
            "incoming_ev_estimated_soc_arrival",
            "incoming_ev_required_soc_departure",
            "::soc",
            "electric_vehicle_soc",
            "electric_vehicle_required_soc_departure",
        )
        return any(token in name for token in soc_tokens)

    @staticmethod
    def _soc_fraction(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.5:
            value /= 100.0
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float, *, default: float = 0.0) -> float:
        if denominator <= 0.0:
            return float(default)
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    @staticmethod
    def _minmax(value: float, low: float, high: float) -> float:
        denominator = high - low
        if not np.isfinite(low) or not np.isfinite(high) or not np.isfinite(denominator) or denominator <= 0.0:
            return 0.0
        return float(np.clip((value - low) / denominator, 0.0, 1.0))

    @staticmethod
    def _positive_by_high(value: float, high: float, *, fallback: float) -> float:
        scale = high if np.isfinite(high) and 0.0 < high < 1.0e5 else fallback
        return float(np.clip(value / scale, 0.0, 1.0))

    @staticmethod
    def _reasonable_bound_scale(low: float, high: float, *, signed: bool, max_abs: float) -> Optional[float]:
        if not np.isfinite(low) or not np.isfinite(high):
            return None
        if max(abs(low), abs(high)) >= max_abs:
            return None
        if signed:
            scale = max(abs(low), abs(high), 1.0)
        else:
            scale = max(high, 1.0)
        return float(scale) if scale > 0.0 else None

    @staticmethod
    def _is_signed_power_name(name: str) -> bool:
        signed_tokens = (
            "net_power_kw",
            "bess_power_kw",
            "storage_power_kw",
            "commanded_power_kw",
            "applied_power_kw",
            "slack_kw",
        )
        return any(token in name for token in signed_tokens)

    @staticmethod
    def _is_signed_step_energy_name(name: str) -> bool:
        signed_tokens = (
            "net_energy",
            "bess_energy",
            "storage_energy",
            "applied_energy",
        )
        return any(token in name for token in signed_tokens)

    def _power_scale(
        self,
        *,
        name: str,
        low: float,
        high: float,
        values_by_name: Mapping[str, float],
        signed: bool,
    ) -> float:
        bounded = self._reasonable_bound_scale(low, high, signed=signed, max_abs=1.0e4)
        if bounded is not None:
            return bounded

        if name.startswith("district__community_"):
            active_buildings = self._safe_scalar(
                values_by_name.get("district__active_buildings_count", 0.0),
                0.0,
            )
            return max(active_buildings, 1.0) * 25.0

        if name.startswith("charger::"):
            max_charge = self._safe_scalar(
                values_by_name.get(self._replace_last_feature(name, "max_charging_power_kw"), 0.0),
                0.0,
            )
            max_discharge = self._safe_scalar(
                values_by_name.get(self._replace_last_feature(name, "max_discharging_power_kw"), 0.0),
                0.0,
            )
            return max(max_charge, max_discharge, 22.0)

        if name.startswith("storage::"):
            nominal = self._safe_scalar(
                values_by_name.get(self._replace_last_feature(name, "nominal_power_kw"), 0.0),
                0.0,
            )
            max_charge = self._safe_scalar(
                values_by_name.get(self._replace_last_feature(name, "max_charge_power_kw"), 0.0),
                0.0,
            )
            max_discharge = self._safe_scalar(
                values_by_name.get(self._replace_last_feature(name, "max_discharge_power_kw"), 0.0),
                0.0,
            )
            return max(nominal, max_charge, max_discharge, 5.0)

        if name.startswith("pv::"):
            installed = self._safe_scalar(
                values_by_name.get(self._replace_last_feature(name, "installed_power_kw"), 0.0),
                0.0,
            )
            return max(installed, 1.0)

        return 25.0

    def _power_ratio(
        self,
        *,
        name: str,
        value: float,
        low: float,
        high: float,
        values_by_name: Mapping[str, float],
    ) -> float:
        signed = self._is_signed_power_name(name)
        scale = self._power_scale(
            name=name,
            low=low,
            high=high,
            values_by_name=values_by_name,
            signed=signed,
        )
        if signed:
            return float(np.clip(value / scale, -1.0, 1.0))
        return float(np.clip(value / scale, 0.0, 1.0))

    def _step_energy_ratio(
        self,
        *,
        name: str,
        value: float,
        low: float,
        high: float,
        values_by_name: Mapping[str, float],
    ) -> float:
        signed = self._is_signed_step_energy_name(name)
        bounded = self._reasonable_bound_scale(low, high, signed=signed, max_abs=1.0e4)
        if bounded is not None:
            scale = bounded
        else:
            power_scale = self._power_scale(
                name=name,
                low=low,
                high=high,
                values_by_name=values_by_name,
                signed=signed,
            )
            scale = max(power_scale * self.seconds_per_time_step / 3600.0, 1.0e-6)

        if signed:
            return float(np.clip(value / scale, -1.0, 1.0))
        return float(np.clip(value / scale, 0.0, 1.0))

    def _energy_capacity_ratio(
        self,
        *,
        name: str,
        value: float,
        high: float,
        values_by_name: Mapping[str, float],
    ) -> float:
        capacity_candidates = (
            self._replace_last_feature(name, "battery_capacity_kwh"),
            self._replace_last_feature(name, "connected_ev_battery_capacity_kwh"),
            self._replace_last_feature(name, "capacity_kwh"),
        )
        for candidate in capacity_candidates:
            capacity = self._safe_scalar(values_by_name.get(candidate, 0.0), 0.0)
            if capacity > 0.0:
                return float(np.clip(value / capacity, 0.0, 1.0))

        return self._positive_by_high(value, high, fallback=100.0)

    @staticmethod
    def _signed_by_bounds(value: float, low: float, high: float) -> float:
        finite_bounds = np.isfinite(low) and np.isfinite(high) and max(abs(low), abs(high)) < 1.0e5
        scale = max(abs(low), abs(high), 1.0) if finite_bounds else max(abs(value), 1.0)
        return float(np.clip(value / scale, -1.0, 1.0))

    @staticmethod
    def _headroom_ratio(*, name: str, value: float, low: float, high: float) -> float:
        finite_bounds = np.isfinite(low) and np.isfinite(high) and max(abs(low), abs(high)) < 1.0e5
        if finite_bounds:
            scale = max(abs(low), abs(high), 1.0)
        elif "phase_" in name:
            scale = 25.0
        else:
            scale = 100.0
        return float(np.clip(value / scale, -1.0, 1.0))

    @staticmethod
    def _charger_rows_by_building(charger_ids: Sequence[str]) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        for row, charger_id in enumerate(charger_ids):
            if not isinstance(charger_id, str) or "/" not in charger_id:
                continue
            building_name = charger_id.split("/", 1)[0]
            mapping.setdefault(building_name, []).append(row)
        return mapping

    @staticmethod
    def _entity_rows_by_building(entity_ids: Sequence[str]) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        for row, entity_id in enumerate(entity_ids):
            if not isinstance(entity_id, str) or "/" not in entity_id:
                continue
            building_name = entity_id.split("/", 1)[0]
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

    def _match_deferrable_row(
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
        for prefix in ("deferrable_appliance::", "deferrable_appliance_"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]

        candidates: List[str] = [normalized]
        if "/" not in normalized:
            candidates.insert(0, f"{building_name}/{normalized}")
        elif normalized.startswith(f"{building_name}/"):
            suffix = normalized[len(building_name) + 1 :]
            if suffix:
                candidates.append(suffix)

        allowed = set(int(row) for row in allowed_rows)
        for appliance_id in candidates:
            row = self._deferrable_row_by_id.get(appliance_id)
            if row is None:
                continue
            if allowed and row not in allowed:
                continue
            return int(row)

        return None

    def _resolve_deferrable_action_target(
        self,
        *,
        action_name: str,
        building_name: str,
        building_deferrable_rows: Sequence[int],
    ) -> Tuple[Optional[int], Optional[str]]:
        action = str(action_name or "").strip()
        if not action or not self._deferrable_action_features:
            return None, None

        if action in self._deferrable_action_col_by_name:
            if len(building_deferrable_rows) == 1:
                return int(building_deferrable_rows[0]), action
            return None, None

        for feature in self._deferrable_action_features:
            prefix = f"{feature}_"
            if action.startswith(prefix):
                row = self._match_deferrable_row(
                    building_name=building_name,
                    candidate=action[len(prefix) :],
                    allowed_rows=building_deferrable_rows,
                )
                if row is not None:
                    return row, feature

            namespace_prefix = "deferrable_appliance_"
            if action.startswith(namespace_prefix):
                row = self._match_deferrable_row(
                    building_name=building_name,
                    candidate=action[len(namespace_prefix) :],
                    allowed_rows=building_deferrable_rows,
                )
                if row is not None:
                    return row, feature

            suffix_token = f"::{feature}"
            if action.endswith(suffix_token):
                appliance_ref = action[: -len(suffix_token)]
                if appliance_ref.startswith("deferrable_appliance::"):
                    appliance_ref = appliance_ref[len("deferrable_appliance::") :]

                row = self._match_deferrable_row(
                    building_name=building_name,
                    candidate=appliance_ref,
                    allowed_rows=building_deferrable_rows,
                )
                if row is not None:
                    return row, feature

        if len(building_deferrable_rows) == 1:
            for feature in self._deferrable_action_features:
                if feature in action or "deferrable_appliance" in action:
                    return int(building_deferrable_rows[0]), feature

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
        deferrable_ids = list(action_specs.get("deferrable_appliance", {}).get("ids", []))

        self._building_action_features = list(action_specs.get("building", {}).get("features", []))
        self._charger_action_features = list(action_specs.get("charger", {}).get("features", []))
        self._deferrable_action_features = list(action_specs.get("deferrable_appliance", {}).get("features", []))
        self._building_action_col_by_name = {
            name: idx for idx, name in enumerate(self._building_action_features)
        }
        self._charger_action_col_by_name = {
            name: idx for idx, name in enumerate(self._charger_action_features)
        }
        self._deferrable_action_col_by_name = {
            name: idx for idx, name in enumerate(self._deferrable_action_features)
        }
        self._charger_row_by_id = {entity_id: row for row, entity_id in enumerate(charger_ids)}
        self._deferrable_row_by_id = {entity_id: row for row, entity_id in enumerate(deferrable_ids)}
        charger_rows_by_building = self._charger_rows_by_building(charger_ids)
        deferrable_rows_by_building = self._entity_rows_by_building(deferrable_ids)

        building_table = np.zeros(
            (len(building_ids), len(self._building_action_features)),
            dtype=np.float32,
        )
        charger_table = np.zeros(
            (len(charger_ids), len(self._charger_action_features)),
            dtype=np.float32,
        )
        deferrable_table = np.zeros(
            (len(deferrable_ids), len(self._deferrable_action_features)),
            dtype=np.float32,
        )

        for building_index, building_actions in enumerate(actions):
            if building_index >= len(action_names) or building_index >= len(building_ids):
                continue

            names = action_names[building_index]
            building_name = building_ids[building_index]
            building_charger_rows = charger_rows_by_building.get(building_name, [])
            building_deferrable_rows = deferrable_rows_by_building.get(building_name, [])

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
                continue

            for position, action_name in enumerate(names):
                value = 0.0
                if position < len(building_actions):
                    value = self._safe_scalar(building_actions[position], 0.0)

                deferrable_row, deferrable_feature = self._resolve_deferrable_action_target(
                    action_name=str(action_name),
                    building_name=building_name,
                    building_deferrable_rows=building_deferrable_rows,
                )
                if deferrable_row is None or deferrable_feature is None:
                    continue

                deferrable_col = self._deferrable_action_col_by_name.get(deferrable_feature)
                if deferrable_col is None:
                    continue
                deferrable_table[deferrable_row, deferrable_col] = value

        return {
            "tables": {
                "building": building_table,
                "charger": charger_table,
                "deferrable_appliance": deferrable_table,
            }
        }
