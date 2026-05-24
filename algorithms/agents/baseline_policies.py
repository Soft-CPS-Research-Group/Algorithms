"""Baseline policies for non-learning comparisons."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from algorithms.agents.rbc_agent import ChargerInfo, RuleBasedPolicy


class RandomPolicy(RuleBasedPolicy):
    """Random controller that samples every available action within its bounds."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})
        seed = hyper.get("seed", config.get("training", {}).get("seed", 22))
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        actions: List[List[float]] = []

        for agent_idx, action_names in enumerate(self._action_labels):
            obs = (
                np.asarray(observations[agent_idx], dtype=float)
                if agent_idx < len(observations)
                else np.asarray([], dtype=float)
            )
            obs_map = self._obs_index[agent_idx] if agent_idx < len(self._obs_index) else {}
            agent_actions: List[float] = []
            for action_position, action_name in enumerate(action_names):
                low, high = self._get_action_bounds(agent_idx, action_position)
                if deterministic:
                    value = 0.5 * (low + high)
                else:
                    value = float(self._rng.uniform(low, high))
                if self._is_storage_action_name(str(action_name)):
                    value = self._apply_storage_dynamic_headroom_limit(
                        value,
                        obs,
                        obs_map,
                        self._get_storage_info(agent_idx),
                        (low, high),
                    )
                    value = self._apply_storage_soc_limit(
                        value,
                        obs,
                        obs_map,
                        (low, high),
                    )
                elif "electric_vehicle" in str(action_name):
                    charger_info = self._get_charger_info(agent_idx, action_position)
                    value = self._apply_ev_dynamic_headroom_limit(
                        value,
                        obs,
                        obs_map,
                        charger_info,
                        (low, high),
                    )
                agent_actions.append(float(np.clip(value, low, high)))
            actions.append(agent_actions)

        self.actions = actions
        return actions

    def _policy_type(self) -> str:
        return "random_policy"


class _OperationalBaselinePolicy(RuleBasedPolicy):
    """Shared helpers for deterministic non-learning baselines."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})
        self.storage_min_soc: float = float(hyper.get("storage_min_soc", 0.20))
        self.storage_max_soc: float = float(hyper.get("storage_max_soc", 0.90))
        self.storage_target_soc: float = float(hyper.get("storage_target_soc", 0.50))
        self.storage_charge_rate: float = float(hyper.get("storage_charge_rate", 0.35))
        self.storage_discharge_rate: float = float(hyper.get("storage_discharge_rate", 0.35))
        self.price_charge_rate: float = float(hyper.get("price_charge_rate", 0.60))
        self.price_discharge_rate: float = float(hyper.get("price_discharge_rate", 0.45))
        self.pv_charge_rate: float = float(hyper.get("pv_charge_rate", 0.75))
        self.peak_discharge_rate: float = float(hyper.get("peak_discharge_rate", 0.65))
        self.pv_surplus_threshold_kw: float = float(hyper.get("pv_surplus_threshold_kw", 0.25))
        self.storage_price_charge_soc_ceiling: float = float(
            hyper.get("storage_price_charge_soc_ceiling", self.storage_max_soc)
        )
        self.storage_price_discharge_soc_floor: float = float(
            hyper.get("storage_price_discharge_soc_floor", self.storage_min_soc)
        )
        self.storage_peak_discharge_soc_floor: float = float(
            hyper.get("storage_peak_discharge_soc_floor", self.storage_min_soc)
        )
        self.normal_storage_discharge_import_threshold_kw: float = float(
            hyper.get("normal_storage_discharge_import_threshold_kw", self.pv_surplus_threshold_kw)
        )
        self.ev_normal_charge_rate: float = float(hyper.get("ev_normal_charge_rate", 1.0))
        self.ev_normal_target_soc: float = float(hyper.get("ev_normal_target_soc", 1.0))
        self.ev_price_charge_rate: float = float(hyper.get("ev_price_charge_rate", 0.70))
        self.ev_pv_charge_rate: float = float(hyper.get("ev_pv_charge_rate", 0.85))
        self.ev_v2g_discharge_rate: float = float(hyper.get("ev_v2g_discharge_rate", 0.30))
        self.allow_v2g: bool = bool(hyper.get("allow_v2g", False))
        self.import_peak_threshold_kw: float = float(hyper.get("import_peak_threshold_kw", 7.0))
        self.low_headroom_threshold_kw: float = float(hyper.get("low_headroom_threshold_kw", 2.0))
        self.ev_community_charge_rate: float = float(hyper.get("ev_community_charge_rate", self.ev_pv_charge_rate))
        self.community_v2g_discharge_rate: float = float(
            hyper.get("community_v2g_discharge_rate", self.ev_v2g_discharge_rate)
        )
        self.community_storage_charge_rate: float = float(hyper.get("community_storage_charge_rate", self.pv_charge_rate))
        self.community_storage_discharge_rate: float = float(
            hyper.get("community_storage_discharge_rate", self.peak_discharge_rate)
        )
        self.community_surplus_threshold_kw: float = float(
            hyper.get("community_surplus_threshold_kw", self.pv_surplus_threshold_kw)
        )
        self.community_import_threshold_kw: float = float(
            hyper.get("community_import_threshold_kw", self.import_peak_threshold_kw)
        )
        self.ev_v2g_reserve_soc: float = float(hyper.get("ev_v2g_reserve_soc", 0.15))
        self.ev_service_margin_rate: float = max(0.0, float(hyper.get("ev_service_margin_rate", 0.05)))
        self.ev_service_floor_rate: float = max(0.0, float(hyper.get("ev_service_floor_rate", 0.25)))
        self.ev_service_lookahead_hours: float = max(
            0.0,
            float(hyper.get("ev_service_lookahead_hours", 4.0)),
        )
        self.ev_service_target_soc: float = float(
            np.clip(float(hyper.get("ev_service_target_soc", 0.0) or 0.0), 0.0, 1.0)
        )
        self.ev_deadline_buffer_hours: float = max(0.0, float(hyper.get("ev_deadline_buffer_hours", 0.25)))
        self.ev_v2g_min_departure_hours: float = max(0.0, float(hyper.get("ev_v2g_min_departure_hours", 2.0)))
        self.ev_v2g_service_margin_soc: float = max(0.0, float(hyper.get("ev_v2g_service_margin_soc", 0.05)))
        self.deferrable_safety_margin_steps: float = max(
            0.0,
            float(hyper.get("deferrable_safety_margin_steps", 1.0)),
        )
        self._obs_match_cache: Dict[int, Dict[tuple, List[tuple[str, int]]]] = {}

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
        self._obs_match_cache = {}

    def _policy_type(self) -> str:
        return self.__class__.__name__

    def _cached_observation_matches(
        self,
        obs_map: Mapping[str, int],
        *,
        suffixes: Sequence[str] = (),
        include_tokens: Sequence[str] = (),
        exclude_tokens: Sequence[str] = (),
        prefix: str = "",
    ) -> List[tuple[str, int]]:
        """Return stable observation name/index matches for a static layout."""
        include = tuple(str(token).lower() for token in include_tokens)
        exclude = tuple(str(token).lower() for token in exclude_tokens)
        suffix_key = tuple(str(suffix) for suffix in suffixes)
        cache_key = (suffix_key, include, exclude, str(prefix))
        map_cache = self._obs_match_cache.setdefault(id(obs_map), {})
        cached = map_cache.get(cache_key)
        if cached is not None:
            return cached

        matches: List[tuple[str, int]] = []
        for raw_name, index in obs_map.items():
            name = str(raw_name)
            if prefix and not name.startswith(prefix):
                continue
            if suffix_key and not any(name.endswith(suffix) for suffix in suffix_key):
                continue
            lowered = name.lower()
            if include and not all(token in lowered for token in include):
                continue
            if exclude and any(token in lowered for token in exclude):
                continue
            matches.append((name, int(index)))

        map_cache[cache_key] = matches
        return matches

    def _get_first_value(
        self,
        obs: np.ndarray,
        obs_map: Mapping[str, int],
        names: Sequence[str],
        *,
        suffixes: Sequence[str] = (),
        default: float = 0.0,
    ) -> float:
        for name in names:
            value = self._get_value(obs, obs_map, name, default=float("nan"))
            if not math.isnan(value):
                return value

        if suffixes:
            for name, index in self._cached_observation_matches(obs_map, suffixes=suffixes):
                if index >= len(obs):
                    continue
                value = obs[index]
                try:
                    if math.isnan(value):
                        continue
                except TypeError:
                    pass
                return float(value)

        return default

    def _get_hour(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        return self._get_first_value(obs, obs_map, ["hour", "district__hour"], default=0.0) % 24.0

    def _get_price_context(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> Dict[str, float | bool]:
        price = self._get_first_value(
            obs,
            obs_map,
            ["electricity_pricing", "district__electricity_pricing"],
            default=float("nan"),
        )
        forecasts = [
            self._get_first_value(
                obs,
                obs_map,
                [f"electricity_pricing_predicted_{idx}", f"district__electricity_pricing_predicted_{idx}"],
                default=float("nan"),
            )
            for idx in (1, 2, 3)
        ]
        for _name, index in self._cached_observation_matches(obs_map, include_tokens=("forecast_price",)):
            if index >= len(obs):
                continue
            value = obs[index]
            if not math.isnan(value):
                forecasts.append(float(value))
        valid = [value for value in forecasts if not math.isnan(value)]

        if math.isnan(price):
            hour = self._get_hour(obs, obs_map)
            cheap = hour <= 7.0 or hour >= 22.0
            expensive = 17.0 <= hour <= 21.0
            return {"price": 0.0, "cheap": cheap, "expensive": expensive}

        if not valid:
            return {"price": price, "cheap": False, "expensive": False}

        forecast_mean = float(np.mean(valid))
        forecast_min = float(np.min(valid))
        forecast_max = float(np.max(valid))
        spread = max(forecast_max - forecast_min, abs(forecast_mean) * 0.05, 1.0e-9)
        cheap = price <= forecast_mean - 0.20 * spread or price <= forecast_min + 0.10 * spread
        expensive = price >= forecast_mean + 0.20 * spread or price >= forecast_max - 0.10 * spread
        return {"price": price, "cheap": cheap, "expensive": expensive}

    def _forecast_values(
        self,
        obs: np.ndarray,
        obs_map: Mapping[str, int],
        include_tokens: Sequence[str],
        *,
        exclude_tokens: Sequence[str] = (),
    ) -> List[float]:
        values: List[float] = []
        for _name, index in self._cached_observation_matches(
            obs_map,
            include_tokens=include_tokens,
            exclude_tokens=exclude_tokens,
        ):
            if index >= len(obs):
                continue
            value = obs[index]
            if math.isfinite(value):
                values.append(float(value))
        return values

    def _max_forecast_value(
        self,
        obs: np.ndarray,
        obs_map: Mapping[str, int],
        include_tokens: Sequence[str],
        *,
        exclude_tokens: Sequence[str] = (),
        default: float = 0.0,
    ) -> float:
        values = self._forecast_values(obs, obs_map, include_tokens, exclude_tokens=exclude_tokens)
        return max(values) if values else default

    def _min_forecast_value(
        self,
        obs: np.ndarray,
        obs_map: Mapping[str, int],
        include_tokens: Sequence[str],
        *,
        exclude_tokens: Sequence[str] = (),
        default: float = float("inf"),
    ) -> float:
        values = self._forecast_values(obs, obs_map, include_tokens, exclude_tokens=exclude_tokens)
        return min(values) if values else default

    def _forecast_local_surplus_kw(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        candidates = self._forecast_values(
            obs,
            obs_map,
            ["forecast_pv_surplus_mean"],
            exclude_tokens=["community"],
        )
        candidates.extend(
            self._forecast_values(obs, obs_map, ["forecast_export"], exclude_tokens=["community"])
        )
        net_values = self._forecast_values(obs, obs_map, ["forecast_net"], exclude_tokens=["community"])
        candidates.extend(-value for value in net_values if value < 0.0)
        positive = [max(0.0, value) for value in candidates]
        return max(positive) if positive else 0.0

    def _forecast_local_import_peak_kw(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        candidates = self._forecast_values(obs, obs_map, ["forecast_import"], exclude_tokens=["community"])
        net_values = self._forecast_values(obs, obs_map, ["forecast_net"], exclude_tokens=["community"])
        candidates.extend(value for value in net_values if value > 0.0)
        positive = [max(0.0, value) for value in candidates]
        return max(positive) if positive else 0.0

    def _forecast_local_headroom_min_kw(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        return self._min_forecast_value(
            obs,
            obs_map,
            ["forecast_headroom"],
            exclude_tokens=["community"],
            default=float("inf"),
        )

    def _forecast_community_surplus_kw(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        candidates = self._forecast_values(obs, obs_map, ["forecast_community_pv_surplus"])
        candidates.extend(self._forecast_values(obs, obs_map, ["forecast_community_export"]))
        net_values = self._forecast_values(obs, obs_map, ["forecast_community_net"])
        candidates.extend(-value for value in net_values if value < 0.0)
        positive = [max(0.0, value) for value in candidates]
        return max(positive) if positive else 0.0

    def _forecast_community_import_peak_kw(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        candidates = self._forecast_values(obs, obs_map, ["forecast_community_import"])
        net_values = self._forecast_values(obs, obs_map, ["forecast_community_net"])
        candidates.extend(value for value in net_values if value > 0.0)
        positive = [max(0.0, value) for value in candidates]
        return max(positive) if positive else 0.0

    def _forecast_community_headroom_min_kw(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        return self._min_forecast_value(
            obs,
            obs_map,
            ["forecast_community_headroom"],
            default=float("inf"),
        )

    def _get_storage_soc(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        value = self._storage_soc_ratio(np.asarray(obs, dtype=float), obs_map, default=self.storage_target_soc)
        if math.isnan(value):
            return float(np.clip(self.storage_target_soc, 0.0, 1.0))
        return value

    def _storage_strategy_limits(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> tuple[float, float]:
        observed_min, observed_max = self._observed_storage_soc_limits(np.asarray(obs, dtype=float), obs_map)
        min_soc = max(self.storage_min_soc, observed_min)
        max_soc = min(self.storage_max_soc, observed_max)
        if min_soc > max_soc:
            return observed_min, observed_max
        return min_soc, max_soc

    def _get_pv_power(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        return max(
            0.0,
            self._get_first_value(
                obs,
                obs_map,
                ["pv_power_kw", "solar_generation"],
                suffixes=("::generation_power_kw",),
                default=0.0,
            ),
        )

    def _get_load_power(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        return max(
            0.0,
            self._get_first_value(
                obs,
                obs_map,
                ["load_power_kw", "non_shiftable_load"],
                default=0.0,
            ),
        )

    def _get_import_power(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        import_power = self._get_first_value(obs, obs_map, ["import_power_kw"], default=float("nan"))
        if not math.isnan(import_power):
            return max(0.0, import_power)
        net_power = self._get_first_value(obs, obs_map, ["net_power_kw", "net_electricity_consumption"], default=0.0)
        return max(0.0, net_power)

    def _get_headroom_kw(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        return self._get_first_value(
            obs,
            obs_map,
            [
                "charging_building_headroom_kw",
                "building_import_headroom_kw",
                "phase_headroom_kw",
                "district__community_building_headroom_kw",
                "district__community_phase_headroom_kw",
            ],
            default=float("inf"),
        )

    def _get_community_import_power(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        import_power = self._get_first_value(
            obs,
            obs_map,
            ["district__community_import_power_kw", "community_import_power_kw"],
            default=float("nan"),
        )
        if not math.isnan(import_power):
            return max(0.0, import_power)
        net_power = self._get_first_value(
            obs,
            obs_map,
            ["district__community_net_power_kw", "community_net_power_kw"],
            default=float("nan"),
        )
        if not math.isnan(net_power):
            return max(0.0, net_power)
        return 0.0

    def _get_community_export_power(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        export_power = self._get_first_value(
            obs,
            obs_map,
            ["district__community_export_power_kw", "community_export_power_kw"],
            default=float("nan"),
        )
        if not math.isnan(export_power):
            return max(0.0, export_power)
        net_power = self._get_first_value(
            obs,
            obs_map,
            ["district__community_net_power_kw", "community_net_power_kw"],
            default=float("nan"),
        )
        if not math.isnan(net_power):
            return max(0.0, -net_power)
        return 0.0

    def _get_community_pv_power(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        return max(
            0.0,
            self._get_first_value(
                obs,
                obs_map,
                ["district__community_pv_power_kw", "community_pv_power_kw"],
                default=0.0,
            ),
        )

    def _get_community_headroom_kw(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        return self._get_first_value(
            obs,
            obs_map,
            [
                "district__community_building_headroom_kw",
                "district__community_phase_headroom_kw",
                "community_building_headroom_kw",
                "community_phase_headroom_kw",
            ],
            default=float("inf"),
        )

    def _community_surplus_kw(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        export_power = self._get_community_export_power(obs, obs_map)
        if export_power > self.energy_epsilon:
            return export_power
        pv_power = self._get_community_pv_power(obs, obs_map)
        import_power = self._get_community_import_power(obs, obs_map)
        return max(0.0, pv_power - import_power)

    def _is_community_stressed(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> bool:
        headroom = self._get_community_headroom_kw(obs, obs_map)
        if math.isfinite(headroom) and headroom <= self.low_headroom_threshold_kw:
            return True
        return self._get_community_import_power(obs, obs_map) >= self.community_import_threshold_kw

    def _is_community_stressed_or_forecast_stressed(
        self,
        obs: np.ndarray,
        obs_map: Mapping[str, int],
    ) -> bool:
        if self._is_community_stressed(obs, obs_map):
            return True
        headroom = self._forecast_community_headroom_min_kw(obs, obs_map)
        if math.isfinite(headroom) and headroom <= self.low_headroom_threshold_kw:
            return True
        return self._forecast_community_import_peak_kw(obs, obs_map) >= self.community_import_threshold_kw

    def _pv_surplus_kw(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> float:
        return self._get_pv_power(obs, obs_map) - self._get_load_power(obs, obs_map)

    def _is_grid_stressed(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> bool:
        if self._has_low_headroom(obs, obs_map):
            return True
        return self._get_import_power(obs, obs_map) >= self.import_peak_threshold_kw

    def _is_grid_stressed_or_forecast_stressed(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> bool:
        if self._is_grid_stressed(obs, obs_map):
            return True
        headroom = self._forecast_local_headroom_min_kw(obs, obs_map)
        if math.isfinite(headroom) and headroom <= self.low_headroom_threshold_kw:
            return True
        return self._forecast_local_import_peak_kw(obs, obs_map) >= self.import_peak_threshold_kw

    def _has_low_headroom(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> bool:
        headroom = self._get_headroom_kw(obs, obs_map)
        return bool(math.isfinite(headroom) and headroom <= self.low_headroom_threshold_kw)

    def _has_positive_charge_headroom(self, obs: np.ndarray, obs_map: Mapping[str, int]) -> bool:
        headroom = self._get_headroom_kw(obs, obs_map)
        return not math.isfinite(headroom) or headroom > self.energy_epsilon

    def _clip_non_v2g_charge(self, value: float, bounds: Sequence[float]) -> float:
        low, high = bounds
        return float(np.clip(max(value, 0.0), max(0.0, low), max(0.0, high)))

    def _clip_storage_action(
        self,
        value: float,
        obs: np.ndarray,
        obs_map: Mapping[str, int],
        bounds: Sequence[float],
    ) -> float:
        if abs(value) <= self.energy_epsilon:
            return float(np.clip(0.0, bounds[0], bounds[1]))

        if value > 0.0:
            can_charge = self._get_first_storage_value(obs, obs_map, "can_charge")
            if not math.isnan(can_charge) and can_charge <= 0.5:
                return float(np.clip(0.0, bounds[0], bounds[1]))

            available = self._get_first_storage_value(obs, obs_map, "available_charge_action_normalized")
            if not math.isnan(available):
                value = min(value, max(0.0, available))

        if value < 0.0:
            can_discharge = self._get_first_storage_value(obs, obs_map, "can_discharge")
            if not math.isnan(can_discharge) and can_discharge <= 0.5:
                return float(np.clip(0.0, bounds[0], bounds[1]))

            available = self._get_first_storage_value(obs, obs_map, "available_discharge_action_normalized")
            if not math.isnan(available):
                value = -min(abs(value), max(0.0, available))

        return float(np.clip(value, bounds[0], bounds[1]))

    def _get_first_storage_value(
        self,
        obs: np.ndarray,
        obs_map: Mapping[str, int],
        feature: str,
    ) -> float:
        suffix = f"::{feature}"
        for _name, index in self._cached_observation_matches(
            obs_map,
            suffixes=(suffix,),
            prefix="storage::",
        ):
            if index >= len(obs):
                continue
            value = obs[index]
            if not math.isnan(value):
                return float(value)
        return float("nan")

    def _connected_ev_context(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Mapping[str, int],
        charger_info: Optional[ChargerInfo],
        action_name: str,
    ) -> Dict[str, float | bool | None]:
        charger_id = self._resolve_ev_charger_id(action_name, charger_info)
        building_name = self._agent_buildings.get(agent_idx)
        state = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="connected_state",
            generic_names=["electric_vehicle_charger_state", "electric_vehicle_charger_connected_state"],
            default=0.0,
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
        fallback_capacity = self.default_capacity if charger_info is None else charger_info.capacity or self.default_capacity
        capacity = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="battery_capacity",
            generic_names=["connected_electric_vehicle_at_charger_battery_capacity"],
            default=fallback_capacity,
        )
        if abs(current_soc) > 1.5:
            current_soc /= 100.0
        if abs(required_soc) > 1.5:
            required_soc /= 100.0
        hours = self._get_ev_departure_hours(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            default=self.flexibility_hours,
        )
        max_power = 7.4 if charger_info is None else charger_info.max_power or 7.4
        max_discharge_power = (
            max_power
            if charger_info is None
            else charger_info.max_discharge_power or charger_info.max_power or 7.4
        )
        can_charge = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="can_charge",
            generic_names=[],
            default=float("nan"),
        )
        can_discharge = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="can_discharge",
            generic_names=[],
            default=float("nan"),
        )
        available_charge_action = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="available_charge_action_normalized",
            generic_names=[],
            default=float("nan"),
        )
        available_discharge_action = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="available_discharge_action_normalized",
            generic_names=[],
            default=float("nan"),
        )
        min_required_action = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="min_required_action_normalized",
            generic_names=[],
            default=float("nan"),
        )
        departure_feasibility = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="departure_feasibility_ratio",
            generic_names=[],
            default=float("nan"),
        )
        departure_margin = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="departure_energy_margin_kwh",
            generic_names=[],
            default=float("nan"),
        )
        return {
            "connected": state > 0.0,
            "soc": float(np.clip(current_soc, 0.0, 1.0)),
            "required_soc": float(np.clip(required_soc, 0.0, 1.0)),
            "gap": max(required_soc - current_soc, 0.0),
            "hours_to_departure": max(hours, self.step_hours),
            "max_power": max(max_power, 1.0e-6),
            "max_discharge_power": max(float(max_discharge_power), 1.0e-6),
            "capacity": max(float(capacity), 1.0e-6),
            "charger_id": charger_id,
            "can_charge": can_charge,
            "can_discharge": can_discharge,
            "available_charge_action": available_charge_action,
            "available_discharge_action": available_discharge_action,
            "min_required_action": min_required_action,
            "departure_feasibility": departure_feasibility,
            "departure_margin": departure_margin,
        }

    def _ev_service_gap(self, ctx: Mapping[str, float | bool | None]) -> float:
        target_soc = max(float(ctx["required_soc"]), self.ev_service_target_soc)
        return max(target_soc - float(ctx["soc"]), 0.0)

    @staticmethod
    def _ctx_float(ctx: Mapping[str, float | bool | None], key: str) -> float:
        value = ctx.get(key)
        if value is None:
            return float("nan")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return parsed if math.isfinite(parsed) else float("nan")

    def _required_ev_charge_rate(self, ctx: Mapping[str, float | bool | None], *, margin_rate: float = 0.0) -> float:
        if not ctx.get("connected"):
            return 0.0

        simulator_required = self._ctx_float(ctx, "min_required_action")
        service_gap = self._ev_service_gap(ctx)
        if service_gap <= self.energy_epsilon and (math.isnan(simulator_required) or simulator_required <= 0.0):
            return 0.0

        hours = max(float(ctx["hours_to_departure"]) - self.ev_deadline_buffer_hours, self.step_hours)
        required_rate = service_gap * float(ctx["capacity"]) / (hours * float(ctx["max_power"]))
        if not math.isnan(simulator_required):
            required_rate = max(required_rate, simulator_required)
        return float(np.clip(required_rate + max(0.0, margin_rate), 0.0, 1.0))

    def _safe_ev_charge_rate(
        self,
        ctx: Mapping[str, float | bool | None],
        requested_rate: float,
        bounds: Sequence[float],
    ) -> float:
        if not ctx.get("connected"):
            return 0.0

        can_charge = self._ctx_float(ctx, "can_charge")
        if not math.isnan(can_charge) and can_charge <= 0.5:
            return 0.0

        available = self._ctx_float(ctx, "available_charge_action")
        if not math.isnan(available):
            requested_rate = min(requested_rate, max(0.0, available))

        return self._clip_non_v2g_charge(requested_rate, bounds)

    def _minimum_ev_service_charge_rate(self, ctx: Mapping[str, float | bool | None]) -> float:
        required_rate = self._required_ev_charge_rate(ctx, margin_rate=self.ev_service_margin_rate)
        if required_rate <= self.energy_epsilon:
            return 0.0

        departure_margin = self._ctx_float(ctx, "departure_margin")
        departure_feasibility = self._ctx_float(ctx, "departure_feasibility")
        hours = float(ctx["hours_to_departure"])
        if (not math.isnan(departure_margin) and departure_margin < 0.0) or (
            not math.isnan(departure_feasibility) and departure_feasibility < 0.75
        ):
            return max(required_rate, self.emergency_charge_rate)
        if not math.isnan(departure_feasibility) and departure_feasibility < 1.0:
            return max(required_rate, self.ev_service_floor_rate)
        if hours <= self.emergency_hours or required_rate >= 0.80:
            return max(required_rate, self.emergency_charge_rate)
        if hours <= self.ev_service_lookahead_hours:
            return max(required_rate, self.ev_service_floor_rate)
        return max(required_rate, self.flex_trickle_charge)

    def _safe_ev_v2g_rate(
        self,
        ctx: Mapping[str, float | bool | None],
        *,
        requested_rate: Optional[float] = None,
    ) -> float:
        if not ctx.get("connected"):
            return 0.0
        can_discharge = self._ctx_float(ctx, "can_discharge")
        if not math.isnan(can_discharge) and can_discharge <= 0.5:
            return 0.0

        hours = float(ctx["hours_to_departure"])
        if hours <= self.ev_v2g_min_departure_hours:
            return 0.0

        target_soc = max(float(ctx["required_soc"]), self.ev_service_target_soc)
        reserve_floor = target_soc + self.ev_v2g_reserve_soc + self.ev_v2g_service_margin_soc
        available_soc = float(ctx["soc"]) - reserve_floor
        if available_soc <= self.energy_epsilon:
            return 0.0

        max_discharge_power = float(ctx.get("max_discharge_power") or ctx["max_power"])
        available_rate = available_soc * float(ctx["capacity"]) / (self.step_hours * max_discharge_power)
        simulator_available = self._ctx_float(ctx, "available_discharge_action")
        if not math.isnan(simulator_available):
            available_rate = min(available_rate, max(0.0, simulator_available))
        desired_rate = self.ev_v2g_discharge_rate if requested_rate is None else requested_rate
        return float(np.clip(min(max(0.0, desired_rate), available_rate), 0.0, 1.0))

    def _deferrable_is_startable(
        self,
        obs: np.ndarray,
        obs_map: Mapping[str, int],
        prefix: str,
    ) -> bool:
        pending = self._get_deferrable_value(obs, obs_map, prefix, "pending", default=0.0)
        running = self._get_deferrable_value(obs, obs_map, prefix, "running", default=0.0)
        can_start = self._get_deferrable_value(obs, obs_map, prefix, "can_start", default=0.0)
        deadline_missed = self._get_deferrable_value(obs, obs_map, prefix, "deadline_missed", default=0.0)
        return pending > 0.5 and running <= 0.5 and can_start > 0.5 and deadline_missed <= 0.5

    def _deferrable_must_start_now(
        self,
        obs: np.ndarray,
        obs_map: Mapping[str, int],
        prefix: str,
    ) -> bool:
        must_start_now = self._get_deferrable_value(obs, obs_map, prefix, "must_start_now", default=0.0)
        if must_start_now > 0.5:
            return True

        must_run = self._get_deferrable_value(obs, obs_map, prefix, "must_run", default=0.0)
        if must_run > 0.5:
            return True

        slack_steps = self._get_deferrable_value(obs, obs_map, prefix, "slack_steps", default=float("nan"))
        if not math.isnan(slack_steps) and slack_steps >= 0.0:
            return slack_steps <= self.deferrable_safety_margin_steps

        hours_until_latest = self._get_deferrable_value(
            obs,
            obs_map,
            prefix,
            "hours_until_latest_start",
            default=float("nan"),
        )
        if not math.isnan(hours_until_latest) and hours_until_latest >= 0.0:
            return hours_until_latest <= self.deferrable_safety_margin_steps * self.step_hours

        hours_until_deadline = self._get_deferrable_value(
            obs,
            obs_map,
            prefix,
            "hours_until_deadline",
            default=float("nan"),
        )
        remaining_steps = self._get_deferrable_value(
            obs,
            obs_map,
            prefix,
            "remaining_duration_steps",
            default=float("nan"),
        )
        if math.isnan(remaining_steps) or remaining_steps <= 0.0:
            remaining_steps = self._get_deferrable_value(
                obs,
                obs_map,
                prefix,
                "cycle_duration_steps",
                default=float("nan"),
            )
        if (
            not math.isnan(hours_until_deadline)
            and hours_until_deadline >= 0.0
            and not math.isnan(remaining_steps)
            and remaining_steps > 0.0
        ):
            required_hours = (remaining_steps + self.deferrable_safety_margin_steps) * self.step_hours
            return hours_until_deadline <= required_hours

        return False

    def _deferrable_start_value(self, bounds: Sequence[float]) -> float:
        return float(np.clip(self.deferrable_start_action, bounds[0], bounds[1]))


class NormalPolicy(_OperationalBaselinePolicy):
    """Day-to-day behaviour: immediate EV charge, earliest deferrable start, simple self-consumption BESS."""

    def _policy_type(self) -> str:
        return "normal_policy"

    def _compute_ev_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        charger_info: Optional[ChargerInfo],
        bounds: Sequence[float],
        action_name: str = "",
    ) -> float:
        ctx = self._connected_ev_context(agent_idx, obs, obs_map, charger_info, action_name)
        target_soc = max(float(ctx["required_soc"]), float(np.clip(self.ev_normal_target_soc, 0.0, 1.0)))
        gap = max(target_soc - float(ctx["soc"]), 0.0)
        if not ctx["connected"] or gap <= self.energy_epsilon:
            return 0.0
        return self._safe_ev_charge_rate(ctx, self.ev_normal_charge_rate, bounds)

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
        if pending > 0.5 and running <= 0.5 and can_start > 0.5 and deadline_missed <= 0.5:
            return float(np.clip(self.deferrable_start_action, bounds[0], bounds[1]))
        return 0.0

    def _compute_storage_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        action_name: str,
        bounds: Sequence[float],
    ) -> float:
        del agent_idx, action_name
        soc = self._get_storage_soc(obs, obs_map)
        min_soc, max_soc = self._storage_strategy_limits(obs, obs_map)
        surplus = self._pv_surplus_kw(obs, obs_map)
        import_power = self._get_import_power(obs, obs_map)
        low, high = bounds

        if surplus > self.pv_surplus_threshold_kw and soc < max_soc:
            return self._clip_storage_action(self.storage_charge_rate, obs, obs_map, bounds)
        if import_power > self.normal_storage_discharge_import_threshold_kw and soc > min_soc:
            return self._clip_storage_action(-self.storage_discharge_rate, obs, obs_map, bounds)
        return float(np.clip(0.0, low, high))


class NormalNoBatteryPolicy(NormalPolicy):
    """Day-to-day behaviour without home battery control."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.control_storage = False

    def _policy_type(self) -> str:
        return "normal_no_battery_policy"

    def _compute_storage_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        action_name: str,
        bounds: Sequence[float],
    ) -> float:
        del agent_idx, obs, obs_map, action_name
        return float(np.clip(0.0, bounds[0], bounds[1]))


class RBCBasicPolicy(_OperationalBaselinePolicy):
    """Minimal intelligent controller using current/near-term price and urgency."""

    def _policy_type(self) -> str:
        return "rbc_basic_policy"

    def _compute_ev_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        charger_info: Optional[ChargerInfo],
        bounds: Sequence[float],
        action_name: str = "",
    ) -> float:
        ctx = self._connected_ev_context(agent_idx, obs, obs_map, charger_info, action_name)
        if not ctx["connected"]:
            return 0.0

        service_rate = self._minimum_ev_service_charge_rate(ctx)
        if service_rate <= self.energy_epsilon and self._ev_service_gap(ctx) <= self.energy_epsilon:
            return 0.0
        price_ctx = self._get_price_context(obs, obs_map)

        if bool(price_ctx["cheap"]):
            return self._safe_ev_charge_rate(ctx, max(service_rate, self.ev_price_charge_rate), bounds)
        return self._safe_ev_charge_rate(ctx, service_rate, bounds)

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
        if not self._deferrable_is_startable(obs, obs_map, prefix):
            return 0.0

        urgency = self._get_deferrable_value(obs, obs_map, prefix, "urgency_ratio", default=0.0)
        slack = self._get_deferrable_value(obs, obs_map, prefix, "slack_ratio", default=1.0)
        price_ctx = self._get_price_context(obs, obs_map)
        if (
            self._deferrable_must_start_now(obs, obs_map, prefix)
            or urgency >= self.deferrable_urgency_threshold
            or slack <= self.deferrable_slack_threshold
            or bool(price_ctx["cheap"])
        ):
            return self._deferrable_start_value(bounds)
        return 0.0

    def _compute_storage_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        action_name: str,
        bounds: Sequence[float],
    ) -> float:
        del agent_idx, action_name
        soc = self._get_storage_soc(obs, obs_map)
        min_soc, max_soc = self._storage_strategy_limits(obs, obs_map)
        price_ctx = self._get_price_context(obs, obs_map)
        low, high = bounds
        if bool(price_ctx["cheap"]) and soc < min(max_soc, self.storage_price_charge_soc_ceiling):
            return self._clip_storage_action(self.price_charge_rate, obs, obs_map, bounds)
        if bool(price_ctx["expensive"]) and soc > max(min_soc, self.storage_price_discharge_soc_floor):
            return self._clip_storage_action(-self.price_discharge_rate, obs, obs_map, bounds)
        return float(np.clip(0.0, low, high))


class RBCSmartPolicy(RBCBasicPolicy):
    """Solar, price and peak-aware heuristic with conservative optional V2G."""

    def _policy_type(self) -> str:
        return "rbc_smart_policy"

    def _compute_ev_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        charger_info: Optional[ChargerInfo],
        bounds: Sequence[float],
        action_name: str = "",
    ) -> float:
        ctx = self._connected_ev_context(agent_idx, obs, obs_map, charger_info, action_name)
        if not ctx["connected"]:
            return 0.0
        service_gap = self._ev_service_gap(ctx)
        service_rate = self._minimum_ev_service_charge_rate(ctx)
        price_ctx = self._get_price_context(obs, obs_map)
        surplus = self._pv_surplus_kw(obs, obs_map)
        forecast_surplus = self._forecast_local_surplus_kw(obs, obs_map)
        stressed = self._is_grid_stressed(obs, obs_map)
        stressed_or_forecast = self._is_grid_stressed_or_forecast_stressed(obs, obs_map)
        low, high = bounds
        safe_v2g_rate = self._safe_ev_v2g_rate(ctx)

        if (
            self.allow_v2g
            and low < 0.0
            and safe_v2g_rate > self.energy_epsilon
            and (stressed_or_forecast or bool(price_ctx["expensive"]))
        ):
            return float(np.clip(-safe_v2g_rate, low, high))

        if service_gap <= self.energy_epsilon and service_rate <= self.energy_epsilon:
            return 0.0
        if surplus > self.pv_surplus_threshold_kw and not stressed:
            return self._safe_ev_charge_rate(ctx, max(service_rate, self.ev_pv_charge_rate), bounds)
        if forecast_surplus > self.pv_surplus_threshold_kw and not stressed_or_forecast:
            return self._safe_ev_charge_rate(ctx, max(service_rate, self.ev_pv_charge_rate), bounds)
        if bool(price_ctx["cheap"]) and not stressed_or_forecast:
            return self._safe_ev_charge_rate(ctx, max(service_rate, self.ev_price_charge_rate), bounds)
        return self._safe_ev_charge_rate(ctx, service_rate, bounds)

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
        if not self._deferrable_is_startable(obs, obs_map, prefix):
            return 0.0

        urgency = self._get_deferrable_value(obs, obs_map, prefix, "urgency_ratio", default=0.0)
        slack = self._get_deferrable_value(obs, obs_map, prefix, "slack_ratio", default=1.0)
        price_ctx = self._get_price_context(obs, obs_map)
        surplus = self._pv_surplus_kw(obs, obs_map)
        forecast_surplus = self._forecast_local_surplus_kw(obs, obs_map)
        stressed_or_forecast = self._is_grid_stressed_or_forecast_stressed(obs, obs_map)

        should_start = (
            self._deferrable_must_start_now(obs, obs_map, prefix)
            or urgency >= self.deferrable_urgency_threshold
            or slack <= self.deferrable_slack_threshold
            or (surplus > self.pv_surplus_threshold_kw and not stressed_or_forecast)
            or (forecast_surplus > self.pv_surplus_threshold_kw and not stressed_or_forecast)
            or (bool(price_ctx["cheap"]) and not stressed_or_forecast)
        )
        if should_start:
            return self._deferrable_start_value(bounds)
        return 0.0

    def _compute_storage_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        action_name: str,
        bounds: Sequence[float],
    ) -> float:
        del agent_idx, action_name
        soc = self._get_storage_soc(obs, obs_map)
        min_soc, max_soc = self._storage_strategy_limits(obs, obs_map)
        surplus = self._pv_surplus_kw(obs, obs_map)
        forecast_surplus = self._forecast_local_surplus_kw(obs, obs_map)
        price_ctx = self._get_price_context(obs, obs_map)
        import_power = self._get_import_power(obs, obs_map)
        low_headroom = self._has_low_headroom(obs, obs_map)
        stressed = low_headroom or import_power >= self.import_peak_threshold_kw
        stressed_or_forecast = self._is_grid_stressed_or_forecast_stressed(obs, obs_map)
        forecast_headroom = self._forecast_local_headroom_min_kw(obs, obs_map)
        forecast_stressed = (
            (math.isfinite(forecast_headroom) and forecast_headroom <= self.low_headroom_threshold_kw)
            or self._forecast_local_import_peak_kw(obs, obs_map) >= self.import_peak_threshold_kw
        )
        low, high = bounds
        price_charge_ceiling = min(max_soc, self.storage_price_charge_soc_ceiling)
        price_charge_allowed = (
            self.price_charge_rate > 0.0
            and soc < price_charge_ceiling
            and bool(price_ctx["cheap"])
        )

        if soc < max_soc and surplus > self.pv_surplus_threshold_kw:
            charge_rate = self.pv_charge_rate
            if price_charge_allowed:
                charge_rate = max(charge_rate, self.price_charge_rate)
            if charge_rate > 0.0:
                return self._clip_storage_action(charge_rate, obs, obs_map, bounds)
        if soc < max_soc and forecast_surplus > self.pv_surplus_threshold_kw and not stressed_or_forecast:
            charge_rate = self.pv_charge_rate
            if price_charge_allowed:
                charge_rate = max(charge_rate, self.price_charge_rate)
            if charge_rate > 0.0:
                return self._clip_storage_action(charge_rate, obs, obs_map, bounds)
        if (
            forecast_stressed
            and not bool(price_ctx["cheap"])
            and soc > max(min_soc, self.storage_peak_discharge_soc_floor)
        ):
            return self._clip_storage_action(-self.peak_discharge_rate, obs, obs_map, bounds)
        if (
            bool(price_ctx["expensive"])
            and (stressed or import_power >= self.import_peak_threshold_kw)
            and soc > max(min_soc, self.storage_peak_discharge_soc_floor)
        ):
            return self._clip_storage_action(-self.peak_discharge_rate, obs, obs_map, bounds)
        if bool(price_ctx["expensive"]) and soc > max(min_soc, self.storage_price_discharge_soc_floor):
            return self._clip_storage_action(-self.price_discharge_rate, obs, obs_map, bounds)
        if (
            price_charge_allowed
        ):
            return self._clip_storage_action(self.price_charge_rate, obs, obs_map, bounds)
        return float(np.clip(0.0, low, high))


class RBCCommunityPolicy(RBCSmartPolicy):
    """Community-aware heuristic that prioritises REC surplus and peak relief."""

    def _policy_type(self) -> str:
        return "rbc_community_policy"

    def _compute_ev_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        charger_info: Optional[ChargerInfo],
        bounds: Sequence[float],
        action_name: str = "",
    ) -> float:
        ctx = self._connected_ev_context(agent_idx, obs, obs_map, charger_info, action_name)
        if not ctx["connected"]:
            return 0.0

        service_gap = self._ev_service_gap(ctx)
        service_rate = self._minimum_ev_service_charge_rate(ctx)
        price_ctx = self._get_price_context(obs, obs_map)
        local_surplus = self._pv_surplus_kw(obs, obs_map)
        community_surplus = self._community_surplus_kw(obs, obs_map)
        forecast_community_surplus = self._forecast_community_surplus_kw(obs, obs_map)
        community_stressed = self._is_community_stressed(obs, obs_map)
        community_stressed_or_forecast = self._is_community_stressed_or_forecast_stressed(obs, obs_map)
        local_stressed = self._is_grid_stressed(obs, obs_map)
        low, high = bounds
        safe_v2g_rate = self._safe_ev_v2g_rate(
            ctx,
            requested_rate=self.community_v2g_discharge_rate,
        )

        if (
            self.allow_v2g
            and low < 0.0
            and safe_v2g_rate > self.energy_epsilon
            and (community_stressed_or_forecast or bool(price_ctx["expensive"]))
        ):
            return float(np.clip(-safe_v2g_rate, low, high))

        if service_gap <= self.energy_epsilon and service_rate <= self.energy_epsilon:
            return 0.0

        if local_surplus > self.pv_surplus_threshold_kw and not local_stressed:
            return self._safe_ev_charge_rate(ctx, max(service_rate, self.ev_pv_charge_rate), bounds)
        if community_surplus > self.community_surplus_threshold_kw and not community_stressed:
            return self._safe_ev_charge_rate(ctx, max(service_rate, self.ev_community_charge_rate), bounds)
        if forecast_community_surplus > self.community_surplus_threshold_kw and not community_stressed_or_forecast:
            return self._safe_ev_charge_rate(ctx, max(service_rate, self.ev_community_charge_rate), bounds)
        if bool(price_ctx["cheap"]) and not community_stressed_or_forecast:
            return self._safe_ev_charge_rate(ctx, max(service_rate, self.ev_price_charge_rate), bounds)
        return self._safe_ev_charge_rate(ctx, service_rate, bounds)

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
        if not self._deferrable_is_startable(obs, obs_map, prefix):
            return 0.0

        urgency = self._get_deferrable_value(obs, obs_map, prefix, "urgency_ratio", default=0.0)
        slack = self._get_deferrable_value(obs, obs_map, prefix, "slack_ratio", default=1.0)
        price_ctx = self._get_price_context(obs, obs_map)
        local_surplus = self._pv_surplus_kw(obs, obs_map)
        community_surplus = self._community_surplus_kw(obs, obs_map)
        forecast_community_surplus = self._forecast_community_surplus_kw(obs, obs_map)
        community_stressed_or_forecast = self._is_community_stressed_or_forecast_stressed(obs, obs_map)

        should_start = (
            self._deferrable_must_start_now(obs, obs_map, prefix)
            or urgency >= self.deferrable_urgency_threshold
            or slack <= self.deferrable_slack_threshold
            or (local_surplus > self.pv_surplus_threshold_kw and not community_stressed_or_forecast)
            or (community_surplus > self.community_surplus_threshold_kw and not community_stressed_or_forecast)
            or (
                forecast_community_surplus > self.community_surplus_threshold_kw
                and not community_stressed_or_forecast
            )
            or (bool(price_ctx["cheap"]) and not community_stressed_or_forecast)
        )
        if should_start:
            return self._deferrable_start_value(bounds)
        return 0.0

    def _compute_storage_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        action_name: str,
        bounds: Sequence[float],
    ) -> float:
        del agent_idx, action_name
        soc = self._get_storage_soc(obs, obs_map)
        min_soc, max_soc = self._storage_strategy_limits(obs, obs_map)
        local_surplus = self._pv_surplus_kw(obs, obs_map)
        community_surplus = self._community_surplus_kw(obs, obs_map)
        forecast_community_surplus = self._forecast_community_surplus_kw(obs, obs_map)
        price_ctx = self._get_price_context(obs, obs_map)
        community_stressed = self._is_community_stressed(obs, obs_map)
        community_stressed_or_forecast = self._is_community_stressed_or_forecast_stressed(obs, obs_map)
        forecast_community_headroom = self._forecast_community_headroom_min_kw(obs, obs_map)
        forecast_community_stressed = (
            (
                math.isfinite(forecast_community_headroom)
                and forecast_community_headroom <= self.low_headroom_threshold_kw
            )
            or self._forecast_community_import_peak_kw(obs, obs_map) >= self.community_import_threshold_kw
        )
        import_power = self._get_import_power(obs, obs_map)
        low, high = bounds
        price_charge_ceiling = min(max_soc, self.storage_price_charge_soc_ceiling)
        price_charge_allowed = (
            self.price_charge_rate > 0.0
            and soc < price_charge_ceiling
            and bool(price_ctx["cheap"])
            and not community_stressed
        )

        if soc < max_soc and local_surplus > self.pv_surplus_threshold_kw:
            charge_rate = self.pv_charge_rate
            if price_charge_allowed:
                charge_rate = max(charge_rate, self.price_charge_rate)
            return self._clip_storage_action(charge_rate, obs, obs_map, bounds)

        if (
            soc < max_soc
            and community_surplus > self.community_surplus_threshold_kw
            and not community_stressed
        ):
            charge_rate = self.community_storage_charge_rate
            if price_charge_allowed:
                charge_rate = max(charge_rate, self.price_charge_rate)
            return self._clip_storage_action(charge_rate, obs, obs_map, bounds)

        if (
            soc < max_soc
            and forecast_community_surplus > self.community_surplus_threshold_kw
            and not community_stressed_or_forecast
        ):
            charge_rate = self.community_storage_charge_rate
            if price_charge_allowed:
                charge_rate = max(charge_rate, self.price_charge_rate)
            return self._clip_storage_action(charge_rate, obs, obs_map, bounds)

        if community_stressed and soc > max(min_soc, self.storage_peak_discharge_soc_floor):
            return self._clip_storage_action(-self.community_storage_discharge_rate, obs, obs_map, bounds)

        if (
            forecast_community_stressed
            and not bool(price_ctx["cheap"])
            and soc > max(min_soc, self.storage_peak_discharge_soc_floor)
        ):
            return self._clip_storage_action(-self.community_storage_discharge_rate, obs, obs_map, bounds)

        if (
            bool(price_ctx["expensive"])
            and import_power >= self.import_peak_threshold_kw
            and soc > max(min_soc, self.storage_peak_discharge_soc_floor)
        ):
            return self._clip_storage_action(-self.peak_discharge_rate, obs, obs_map, bounds)

        if bool(price_ctx["expensive"]) and soc > max(min_soc, self.storage_price_discharge_soc_floor):
            return self._clip_storage_action(-self.price_discharge_rate, obs, obs_map, bounds)

        if price_charge_allowed:
            return self._clip_storage_action(self.price_charge_rate, obs, obs_map, bounds)
        return float(np.clip(0.0, low, high))
