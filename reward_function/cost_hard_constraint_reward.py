"""Cost-focused reward with strong penalties for hard operational constraints."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Union

from citylearn.reward_function import RewardFunction


class CostHardConstraintReward(RewardFunction):
    """Encourage low cost while strongly discouraging EV departure and grid constraint violations."""

    required_observation_names = (
        "net_electricity_consumption",
        "electricity_pricing",
        "power_outage",
        "charging_constraint_violation_kwh",
        "electrical_service_violation_kwh",
        "electrical_service_violation",
        "service_violation_kwh",
        "service_violation",
        "electrical_storage_soc",
        "electrical_storage_soc_ratio",
        "electrical_storage_soc_min_ratio",
        "electrical_storage_min_soc_ratio",
        "electrical_storage_soc_max_ratio",
        "electrical_storage_max_soc_ratio",
        "electrical_storage_electricity_consumption",
        "electrical_storage_electricity_consumption_kwh",
        "electric_vehicles_chargers_dict",
        "deferrable_appliances_dict",
        "washing_machines_dict",
        "charging_building_headroom_kw",
        "charging_phase_L1_headroom_kw",
        "charging_phase_L2_headroom_kw",
        "charging_phase_L3_headroom_kw",
    )

    SERVICE_VIOLATION_KEYS = (
        "charging_constraint_violation_kwh",
        "electrical_service_violation_kwh",
        "electrical_service_violation",
        "service_violation_kwh",
        "service_violation",
    )

    BATTERY_THROUGHPUT_KEYS = (
        "electrical_storage_electricity_consumption",
        "electrical_storage_electricity_consumption_kwh",
    )

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        export_credit_ratio: float = 0.8,
        local_cost_weight: float = 1.0,
        grid_violation_penalty: float = 60.0,
        power_outage_penalty: float = 120.0,
        ev_departure_window_hours: float = 1.0,
        ev_departure_service_tolerance: float = 0.05,
        ev_over_service_tolerance: float = 0.05,
        ev_over_service_penalty: float = 0.0,
        ev_connected_deficit_penalty: float = 0.0,
        ev_connected_deficit_exponent: float = 1.0,
        ev_schedule_deficit_penalty: float = 0.0,
        ev_schedule_deficit_exponent: float = 1.0,
        ev_schedule_deficit_cap_soc: Optional[float] = None,
        ev_use_effective_charging_power_for_schedule: bool = False,
        ev_departure_window_penalty_mode: str = "shortfall",
        ev_departure_window_shortfall_cap_soc: Optional[float] = None,
        ev_v2g_service_penalty: float = 0.0,
        ev_departure_deficit_penalty: float = 120.0,
        ev_departure_missed_penalty: float = 250.0,
        ev_default_charging_power_kw: float = 7.4,
        ev_default_battery_capacity_kwh: float = 75.0,
        battery_soc_min: float = 0.05,
        battery_soc_max: float = 0.95,
        use_observed_storage_soc_limits: bool = True,
        battery_soc_violation_penalty: float = 30.0,
        battery_throughput_penalty: float = 0.0,
        deferrable_deadline_missed_penalty: float = 0.0,
        deferrable_urgency_penalty: float = 0.0,
        community_import_penalty: float = 0.0,
        community_peak_import_penalty: float = 0.0,
        community_export_penalty: float = 0.0,
        community_penalty_divide_by_agents: bool = False,
        community_settlement_cost_weight: float = 0.0,
        community_local_price_ratio: float = 0.8,
        community_grid_export_price: float = 0.0,
        scale_state_penalties_by_time_step: bool = False,
        state_penalty_reference_seconds: float = 3600.0,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.export_credit_ratio = float(export_credit_ratio)
        self.local_cost_weight = float(local_cost_weight)
        self.grid_violation_penalty = float(grid_violation_penalty)
        self.power_outage_penalty = float(power_outage_penalty)
        self.ev_departure_window_hours = float(ev_departure_window_hours)
        self.ev_departure_service_tolerance = max(float(ev_departure_service_tolerance), 0.0)
        self.ev_over_service_tolerance = max(float(ev_over_service_tolerance), 0.0)
        self.ev_over_service_penalty = float(ev_over_service_penalty)
        self.ev_connected_deficit_penalty = float(ev_connected_deficit_penalty)
        self.ev_connected_deficit_exponent = max(float(ev_connected_deficit_exponent), 0.1)
        self.ev_schedule_deficit_penalty = float(ev_schedule_deficit_penalty)
        self.ev_schedule_deficit_exponent = max(float(ev_schedule_deficit_exponent), 0.1)
        self.ev_schedule_deficit_cap_soc = self._optional_non_negative_float(ev_schedule_deficit_cap_soc)
        self.ev_use_effective_charging_power_for_schedule = bool(ev_use_effective_charging_power_for_schedule)
        self.ev_departure_window_penalty_mode = str(ev_departure_window_penalty_mode or "shortfall").lower()
        self.ev_departure_window_shortfall_cap_soc = self._optional_non_negative_float(
            ev_departure_window_shortfall_cap_soc
        )
        self.ev_v2g_service_penalty = float(ev_v2g_service_penalty)
        self.ev_departure_deficit_penalty = float(ev_departure_deficit_penalty)
        self.ev_departure_missed_penalty = float(ev_departure_missed_penalty)
        self.ev_default_charging_power_kw = max(float(ev_default_charging_power_kw), 1.0e-6)
        self.ev_default_battery_capacity_kwh = max(float(ev_default_battery_capacity_kwh), 1.0e-6)
        self.battery_soc_min = float(battery_soc_min)
        self.battery_soc_max = float(battery_soc_max)
        self.use_observed_storage_soc_limits = bool(use_observed_storage_soc_limits)
        self.battery_soc_violation_penalty = float(battery_soc_violation_penalty)
        self.battery_throughput_penalty = float(battery_throughput_penalty)
        self.deferrable_deadline_missed_penalty = float(deferrable_deadline_missed_penalty)
        self.deferrable_urgency_penalty = float(deferrable_urgency_penalty)
        self.community_import_penalty = float(community_import_penalty)
        self.community_peak_import_penalty = float(community_peak_import_penalty)
        self.community_export_penalty = float(community_export_penalty)
        self.community_penalty_divide_by_agents = bool(community_penalty_divide_by_agents)
        self.community_settlement_cost_weight = float(community_settlement_cost_weight)
        self.community_local_price_ratio = min(max(float(community_local_price_ratio), 0.0), 1.0)
        self.community_grid_export_price = float(community_grid_export_price)
        self.scale_state_penalties_by_time_step = bool(scale_state_penalties_by_time_step)
        self.state_penalty_reference_seconds = max(float(state_penalty_reference_seconds), 1.0)
        self.seconds_per_time_step = self._safe_float(
            env_metadata.get("seconds_per_time_step") if isinstance(env_metadata, Mapping) else None,
            default=self.state_penalty_reference_seconds,
        )
        self.state_penalty_scale = (
            max(self.seconds_per_time_step, 1.0) / self.state_penalty_reference_seconds
            if self.scale_state_penalties_by_time_step
            else 1.0
        )
        self._charger_phase_map = self._build_charger_phase_map(env_metadata)
        self._charger_static_import_limit_kw_map = self._build_charger_static_import_limit_kw_map(env_metadata)
        self.last_components_by_agent: List[Mapping[str, float]] = []
        self.last_community_components: Mapping[str, float] = {}

    @staticmethod
    def _build_charger_phase_map(env_metadata: Mapping[str, Any]) -> Mapping[str, str]:
        if not isinstance(env_metadata, Mapping):
            return {}

        phase_map: dict[str, str] = {}
        buildings = env_metadata.get("buildings")
        if not isinstance(buildings, list):
            return phase_map

        for building in buildings:
            if not isinstance(building, Mapping):
                continue
            building_name = str(building.get("name") or "")
            charger_phase_map = building.get("charger_phase_map")
            if not isinstance(charger_phase_map, Mapping):
                continue
            for charger_id, phase_connection in charger_phase_map.items():
                if phase_connection in (None, ""):
                    continue
                charger_key = str(charger_id)
                phase_map[charger_key] = str(phase_connection)
                if building_name:
                    phase_map[f"{building_name}/{charger_key}"] = str(phase_connection)

        return phase_map

    @classmethod
    def _build_charger_static_import_limit_kw_map(cls, env_metadata: Mapping[str, Any]) -> Mapping[str, float]:
        if not isinstance(env_metadata, Mapping):
            return {}

        buildings = env_metadata.get("buildings")
        if not isinstance(buildings, list):
            return {}

        limit_map: dict[str, float] = {}
        for building in buildings:
            if not isinstance(building, Mapping):
                continue

            building_name = str(building.get("name") or "")
            charger_phase_map = building.get("charger_phase_map")
            electrical_service = building.get("electrical_service")
            if not isinstance(charger_phase_map, Mapping) or not isinstance(electrical_service, Mapping):
                continue

            limits = electrical_service.get("limits")
            if not isinstance(limits, Mapping):
                continue

            total = limits.get("total")
            total_limit = cls._safe_float(
                total.get("import_kw") if isinstance(total, Mapping) else None,
                default=float("nan"),
            )
            per_phase = limits.get("per_phase")
            if not isinstance(per_phase, Mapping):
                per_phase = {}

            for charger_id, phase_connection in charger_phase_map.items():
                phases = cls._phase_connection_to_phases(phase_connection)
                candidates: List[float] = []
                if phases:
                    phase_limits = []
                    for phase in phases:
                        phase_limits_mapping = per_phase.get(phase)
                        phase_limit = cls._safe_float(
                            phase_limits_mapping.get("import_kw") if isinstance(phase_limits_mapping, Mapping) else None,
                            default=float("nan"),
                        )
                        if phase_limit == phase_limit and phase_limit >= 0.0:
                            phase_limits.append(phase_limit)
                    if phase_limits:
                        candidates.append(min(phase_limits) * float(len(phases)))

                if total_limit == total_limit and total_limit >= 0.0:
                    candidates.append(total_limit)

                if not candidates:
                    continue

                charger_key = str(charger_id)
                limit = min(candidates)
                limit_map[charger_key] = limit
                if building_name:
                    limit_map[f"{building_name}/{charger_key}"] = limit

        return limit_map

    def _refresh_charger_phase_map(self) -> None:
        if not self.ev_use_effective_charging_power_for_schedule:
            return

        metadata = getattr(self, "env_metadata", None)
        phase_map = self._build_charger_phase_map(metadata)
        if phase_map:
            self._charger_phase_map = phase_map

        static_limit_map = self._build_charger_static_import_limit_kw_map(metadata)
        if static_limit_map:
            self._charger_static_import_limit_kw_map = static_limit_map

    def _refresh_state_penalty_scale(self) -> None:
        if not self.scale_state_penalties_by_time_step:
            self.state_penalty_scale = 1.0
            return

        metadata = getattr(self, "env_metadata", None)
        metadata_seconds = None
        if isinstance(metadata, Mapping):
            metadata_seconds = metadata.get("seconds_per_time_step")
        self.seconds_per_time_step = self._safe_float(
            metadata_seconds,
            default=self.seconds_per_time_step,
        )
        self.state_penalty_scale = max(self.seconds_per_time_step, 1.0) / self.state_penalty_reference_seconds

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if parsed != parsed or parsed in (float("inf"), float("-inf")):
            return default
        return parsed

    @staticmethod
    def _optional_non_negative_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed != parsed:
            return None
        return max(parsed, 0.0)

    @staticmethod
    def _cap_non_negative(value: float, cap: Optional[float]) -> float:
        if value != value:
            return 0.0
        value = max(float(value), 0.0)
        if cap is None:
            return value
        return min(value, max(float(cap), 0.0))

    def _extract_first(self, observation: Mapping[str, Union[int, float]], candidates: Iterable[str], default: float = 0.0) -> float:
        for key in candidates:
            if key in observation:
                return self._safe_float(observation.get(key), default=default)
        return default

    def _has_invalid_departure_marker(
        self,
        observation: Mapping[str, Union[int, float]],
        candidates: Iterable[str],
    ) -> bool:
        """Return true when the simulator uses a negative departure sentinel.

        Entity EV observations can represent an unknown/no-scheduled departure
        with `*_departure_time_step = -1` while `hours_until_departure` is 0.
        That must not be treated as a missed departure at every step.
        """
        for key in candidates:
            if key not in observation:
                continue
            value = self._safe_float(observation.get(key), default=float("inf"))
            if value < 0.0:
                return True
        return False

    def _has_known_departure_marker(
        self,
        observation: Mapping[str, Union[int, float]],
        candidates: Iterable[str],
    ) -> bool:
        for key in candidates:
            if key not in observation:
                continue
            value = self._safe_float(observation.get(key), default=float("-inf"))
            if value >= 0.0:
                return True
        return False

    def _sanitize_hours_until_departure(
        self,
        hours_until_departure: float,
        observation: Mapping[str, Union[int, float]],
        departure_marker_candidates: Iterable[str],
    ) -> float:
        if self._has_invalid_departure_marker(observation, departure_marker_candidates):
            return float("inf")
        return hours_until_departure

    @staticmethod
    def _is_storage_observation_key(key: str) -> bool:
        normalized = key.lower()
        if "storage" not in normalized:
            return False
        return not any(token in normalized for token in ("electric_vehicle", "connected_ev", "charger"))

    def _extract_storage_value(
        self,
        observation: Mapping[str, Union[int, float]],
        exact_candidates: Iterable[str],
        suffix_candidates: Iterable[str],
        default: float = float("nan"),
    ) -> float:
        for key in exact_candidates:
            if key in observation:
                return self._safe_float(observation.get(key), default=default)

        suffixes = tuple(suffix_candidates)
        for raw_key, value in observation.items():
            key = str(raw_key)
            if self._is_storage_observation_key(key) and any(key.endswith(suffix) for suffix in suffixes):
                return self._safe_float(value, default=default)

        return default

    def _cost_term(self, observation: Mapping[str, Union[int, float]]) -> float:
        return self._cost_components(observation)["local_cost_reward"]

    def _cost_components(self, observation: Mapping[str, Union[int, float]]) -> Mapping[str, float]:
        net_consumption = self._safe_float(observation.get("net_electricity_consumption"), default=0.0)
        electricity_price = max(self._safe_float(observation.get("electricity_pricing"), default=0.0), 0.0)

        import_cost = max(net_consumption, 0.0) * electricity_price
        export_credit = max(-net_consumption, 0.0) * electricity_price * self.export_credit_ratio

        return {
            "net_electricity_consumption": net_consumption,
            "electricity_price": electricity_price,
            "local_import_energy": max(net_consumption, 0.0),
            "local_export_energy": max(-net_consumption, 0.0),
            "local_import_cost": import_cost,
            "local_export_credit": export_credit,
            "local_cost_reward": -(import_cost - export_credit),
        }

    @staticmethod
    def _allocate_weighted_import_share(
        imports: List[float],
        traded_energy: float,
        weights: Optional[List[float]] = None,
    ) -> List[float]:
        allocations = [0.0 for _ in imports]
        remaining = max(float(traded_energy), 0.0)
        weights = list(weights) if weights is not None else [1.0 for _ in imports]
        weights = [max(float(weight), 0.0) for weight in weights]
        eps = 1.0e-9

        while remaining > eps:
            needs = [max(imported - allocated, 0.0) for imported, allocated in zip(imports, allocations)]
            active_indexes = [index for index, need in enumerate(needs) if need > eps]
            if not active_indexes:
                break

            active_weight_sum = sum(weights[index] for index in active_indexes)
            granted = [0.0 for _ in imports]
            for index in active_indexes:
                if active_weight_sum <= eps:
                    share = remaining / float(len(active_indexes))
                else:
                    share = remaining * (weights[index] / active_weight_sum)
                granted[index] = min(share, needs[index])

            granted_total = sum(granted)
            if granted_total <= eps:
                break

            allocations = [allocated + grant for allocated, grant in zip(allocations, granted)]
            remaining -= granted_total

        return allocations

    def _community_settlement_components(
        self,
        observations: List[Mapping[str, Union[int, float]]],
    ) -> tuple[List[Mapping[str, float]], Mapping[str, float]]:
        imports = [
            max(self._safe_float(observation.get("net_electricity_consumption"), default=0.0), 0.0)
            for observation in observations
        ]
        exports = [
            max(-self._safe_float(observation.get("net_electricity_consumption"), default=0.0), 0.0)
            for observation in observations
        ]
        prices = [
            max(self._safe_float(observation.get("electricity_pricing"), default=0.0), 0.0)
            for observation in observations
        ]

        total_import = sum(imports)
        total_export = sum(exports)
        traded_energy = min(total_import, total_export)
        local_imports = (
            self._allocate_weighted_import_share(imports, traded_energy)
            if total_import > 0.0 and traded_energy > 0.0
            else [0.0 for _ in imports]
        )
        local_exports = (
            [exported * (traded_energy / total_export) for exported in exports]
            if total_export > 0.0 and traded_energy > 0.0
            else [0.0 for _ in exports]
        )

        rows: List[Mapping[str, float]] = []
        total_cost = 0.0
        total_grid_import = 0.0
        total_grid_export = 0.0
        for imported, exported, price, local_import, local_export in zip(
            imports,
            exports,
            prices,
            local_imports,
            local_exports,
        ):
            local_price = self.community_local_price_ratio * price
            grid_import = max(imported - local_import, 0.0)
            grid_export = max(exported - local_export, 0.0)
            settlement_cost = (
                grid_import * price
                + local_import * local_price
                - local_export * local_price
                - grid_export * self.community_grid_export_price
            )
            settlement_reward = -settlement_cost * self.community_settlement_cost_weight
            total_cost += settlement_cost
            total_grid_import += grid_import
            total_grid_export += grid_export
            rows.append(
                {
                    "community_settlement_cost": settlement_cost,
                    "community_settlement_reward": settlement_reward,
                    "community_local_import_energy": local_import,
                    "community_local_export_energy": local_export,
                    "community_grid_import_energy": grid_import,
                    "community_grid_export_energy": grid_export,
                    "community_local_price": local_price,
                    "community_settlement_cost_weight": self.community_settlement_cost_weight,
                }
            )

        totals = {
            "community_settlement_cost_total": total_cost,
            "community_local_traded_energy": traded_energy,
            "community_grid_import_after_settlement": total_grid_import,
            "community_grid_export_after_settlement": total_grid_export,
            "community_local_price_ratio": self.community_local_price_ratio,
            "community_grid_export_price": self.community_grid_export_price,
            "community_settlement_cost_weight": self.community_settlement_cost_weight,
        }
        return rows, totals

    def _battery_safety_components(self, observation: Mapping[str, Union[int, float]]) -> Mapping[str, float]:
        storage_soc = self._extract_storage_value(
            observation,
            exact_candidates=("electrical_storage_soc", "electrical_storage_soc_ratio"),
            suffix_candidates=("::soc", "::soc_ratio", "_soc", "_soc_ratio"),
        )
        min_limit = self.battery_soc_min
        max_limit = self.battery_soc_max
        if self.use_observed_storage_soc_limits:
            min_limit = self._extract_storage_value(
                observation,
                exact_candidates=("electrical_storage_soc_min_ratio", "electrical_storage_min_soc_ratio"),
                suffix_candidates=("::soc_min_ratio", "::min_soc_ratio", "_soc_min_ratio", "_min_soc_ratio"),
                default=min_limit,
            )
            max_limit = self._extract_storage_value(
                observation,
                exact_candidates=("electrical_storage_soc_max_ratio", "electrical_storage_max_soc_ratio"),
                suffix_candidates=("::soc_max_ratio", "::max_soc_ratio", "_soc_max_ratio", "_max_soc_ratio"),
                default=max_limit,
            )

        if min_limit == min_limit and max_limit == max_limit and min_limit > max_limit:
            min_limit, max_limit = max_limit, min_limit

        if storage_soc != storage_soc:  # NaN-safe check
            below = 0.0
            above = 0.0
            soc_penalty = 0.0
        else:
            below = max(min_limit - storage_soc, 0.0)
            above = max(storage_soc - max_limit, 0.0)
            soc_penalty = (below + above) * self.battery_soc_violation_penalty * self.state_penalty_scale

        throughput = abs(self._extract_first(observation, self.BATTERY_THROUGHPUT_KEYS, default=0.0))
        throughput_penalty = throughput * self.battery_throughput_penalty
        return {
            "storage_soc": 0.0 if storage_soc != storage_soc else storage_soc,
            "battery_soc_min_limit": min_limit,
            "battery_soc_max_limit": max_limit,
            "battery_soc_below_limit": below,
            "battery_soc_above_limit": above,
            "battery_soc_violation_penalty_amount": soc_penalty,
            "battery_throughput": throughput,
            "battery_throughput_penalty": throughput_penalty,
            "battery_safety_penalty": soc_penalty + throughput_penalty,
        }

    def _battery_safety_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
        return self._battery_safety_components(observation)["battery_safety_penalty"]

    @staticmethod
    def _sum_components(rows: Iterable[Mapping[str, float]]) -> Mapping[str, float]:
        total: dict[str, float] = {}
        for row in rows:
            for key, value in row.items():
                total[key] = total.get(key, 0.0) + float(value)
        return total

    def _single_ev_departure_components(
        self,
        *,
        connected: bool,
        soc: float,
        required_soc: float,
        hours_until_departure: float,
        battery_capacity_kwh: float,
        max_charging_power_kw: float,
        effective_max_charging_power_kw: Optional[float] = None,
        charged_energy_kwh: float = 0.0,
        priority: float = 1.0,
        allow_departure_missed_penalty: bool = True,
    ) -> Mapping[str, float]:
        if not connected:
            return {
                "ev_connected_count": 0.0,
                "ev_soc_deficit_sum": 0.0,
                "ev_soc_service_target_sum": 0.0,
                "ev_soc_shortfall_beyond_tolerance_sum": 0.0,
                "ev_soc_strict_gap_within_tolerance_sum": 0.0,
                "ev_soc_surplus_sum": 0.0,
                "ev_soc_over_service_sum": 0.0,
                "ev_soc_absolute_error_sum": 0.0,
                "ev_over_service_penalty_amount": 0.0,
                "ev_dense_deficit_penalty": 0.0,
                "ev_nominal_max_charging_power_kw_sum": 0.0,
                "ev_effective_max_charging_power_kw_sum": 0.0,
                "ev_remaining_charge_capacity_soc_sum": 0.0,
                "ev_schedule_min_soc_required_sum": 0.0,
                "ev_schedule_soc_deficit_sum": 0.0,
                "ev_schedule_penalty_deficit_sum": 0.0,
                "ev_schedule_deficit_penalty": 0.0,
                "ev_v2g_discharge_kwh_sum": 0.0,
                "ev_v2g_service_risk_sum": 0.0,
                "ev_v2g_service_abuse_penalty": 0.0,
                "ev_departure_window_count": 0.0,
                "ev_departure_window_penalty_shortfall_sum": 0.0,
                "ev_departure_window_penalty": 0.0,
                "ev_departure_missed_count": 0.0,
                "ev_departure_missed_penalty_amount": 0.0,
                "ev_priority_weight_sum": 0.0,
                "ev_service_penalty": 0.0,
            }

        soc = self._soc_ratio(soc)
        required_soc = self._soc_ratio(required_soc)
        priority_weight = self._priority_weight(priority)
        capacity_kwh = self._positive_or_default(battery_capacity_kwh, self.ev_default_battery_capacity_kwh)
        max_charge_kw = self._positive_or_default(max_charging_power_kw, self.ev_default_charging_power_kw)
        schedule_max_charge_kw = self._positive_or_default(
            effective_max_charging_power_kw
            if effective_max_charging_power_kw is not None
            else max_charge_kw,
            max_charge_kw,
        )
        available_hours = max(hours_until_departure, 0.0) if hours_until_departure == hours_until_departure else 0.0
        service_target_soc = max(required_soc - self.ev_departure_service_tolerance, 0.0)
        max_soc_gain_remaining = schedule_max_charge_kw * available_hours / capacity_kwh
        schedule_min_soc = max(service_target_soc - max_soc_gain_remaining, 0.0)
        schedule_deficit = max(schedule_min_soc - soc, 0.0)
        schedule_penalty_deficit = self._cap_non_negative(
            schedule_deficit,
            self.ev_schedule_deficit_cap_soc,
        )
        schedule_penalty = (
            (schedule_penalty_deficit ** self.ev_schedule_deficit_exponent)
            * self.ev_schedule_deficit_penalty
            * priority_weight
            * self.state_penalty_scale
        )

        deficit = max(required_soc - soc, 0.0)
        service_shortfall = max(service_target_soc - soc, 0.0)
        strict_gap_within_tolerance = max(deficit - service_shortfall, 0.0)
        surplus = max(soc - required_soc, 0.0)
        over_service = max(soc - required_soc - self.ev_over_service_tolerance, 0.0)
        absolute_error = abs(soc - required_soc)
        over_service_penalty = (
            over_service
            * self.ev_over_service_penalty
            * priority_weight
            * self.state_penalty_scale
        )
        dense_penalty = (
            self._dense_ev_deficit_penalty(deficit, hours_until_departure)
            * priority_weight
            * self.state_penalty_scale
        )
        discharge_kwh = max(-self._safe_float(charged_energy_kwh, default=0.0), 0.0)
        v2g_service_risk = service_shortfall + schedule_deficit
        v2g_penalty = self._ev_v2g_service_penalty(
            discharge_kwh=discharge_kwh,
            service_risk=v2g_service_risk,
            priority_weight=priority_weight,
            hours_until_departure=hours_until_departure,
        )
        window_penalty = 0.0
        missed_penalty = 0.0
        departure_window_count = 0.0
        missed_count = 0.0
        if hours_until_departure <= self.ev_departure_window_hours:
            departure_window_count = 1.0
            if self.ev_departure_window_penalty_mode in {"schedule", "schedule_deficit", "unrecoverable"}:
                window_shortfall = schedule_deficit
            else:
                window_shortfall = service_shortfall
            window_penalty_shortfall = self._cap_non_negative(
                window_shortfall,
                self.ev_departure_window_shortfall_cap_soc,
            )
            window_penalty = (
                window_penalty_shortfall
                * self.ev_departure_deficit_penalty
                * priority_weight
                * self.state_penalty_scale
            )
        else:
            window_penalty_shortfall = 0.0
        if allow_departure_missed_penalty and hours_until_departure <= 0.0 and service_shortfall > 0.0:
            missed_count = 1.0
            missed_penalty = self.ev_departure_missed_penalty * priority_weight

        return {
            "ev_connected_count": 1.0,
            "ev_soc_deficit_sum": deficit,
            "ev_soc_service_target_sum": service_target_soc,
            "ev_soc_shortfall_beyond_tolerance_sum": service_shortfall,
            "ev_soc_strict_gap_within_tolerance_sum": strict_gap_within_tolerance,
            "ev_soc_surplus_sum": surplus,
            "ev_soc_over_service_sum": over_service,
            "ev_soc_absolute_error_sum": absolute_error,
            "ev_over_service_penalty_amount": over_service_penalty,
            "ev_dense_deficit_penalty": dense_penalty,
            "ev_nominal_max_charging_power_kw_sum": max_charge_kw,
            "ev_effective_max_charging_power_kw_sum": schedule_max_charge_kw,
            "ev_remaining_charge_capacity_soc_sum": max_soc_gain_remaining,
            "ev_schedule_min_soc_required_sum": schedule_min_soc,
            "ev_schedule_soc_deficit_sum": schedule_deficit,
            "ev_schedule_penalty_deficit_sum": schedule_penalty_deficit,
            "ev_schedule_deficit_penalty": schedule_penalty,
            "ev_v2g_discharge_kwh_sum": discharge_kwh,
            "ev_v2g_service_risk_sum": v2g_service_risk,
            "ev_v2g_service_abuse_penalty": v2g_penalty,
            "ev_departure_window_count": departure_window_count,
            "ev_departure_window_penalty_shortfall_sum": window_penalty_shortfall,
            "ev_departure_window_penalty": window_penalty,
            "ev_departure_missed_count": missed_count,
            "ev_departure_missed_penalty_amount": missed_penalty,
            "ev_priority_weight_sum": priority_weight,
            "ev_service_penalty": (
                dense_penalty
                + schedule_penalty
                + v2g_penalty
                + window_penalty
                + missed_penalty
                + over_service_penalty
            ),
        }

    @staticmethod
    def _phase_connection_to_phases(phase_connection: Any) -> tuple[str, ...]:
        if phase_connection is None:
            return ()
        raw = str(phase_connection).strip().upper().replace("-", "_").replace(" ", "_")
        if not raw:
            return ()
        if raw in {"ALL", "ALL_PHASES", "THREE_PHASE", "THREE_PHASES", "3P", "L1_L2_L3"}:
            return ("L1", "L2", "L3")
        phases = tuple(phase for phase in ("L1", "L2", "L3") if phase in raw)
        return phases

    def _charger_phase_connection(self, charger_info: Mapping[str, Any]) -> str:
        for key in ("phase_connection", "phase", "connected_phase"):
            value = charger_info.get(key)
            if value not in (None, ""):
                return str(value)

        charger_id = str(charger_info.get("charger_id") or charger_info.get("id") or "")
        if charger_id:
            phase_connection = self._charger_phase_map.get(charger_id)
            if phase_connection:
                return str(phase_connection)

        phases = [
            phase
            for phase in ("L1", "L2", "L3")
            if self._safe_float(charger_info.get(f"phase_connection_{phase}"), default=0.0) > 0.5
        ]
        return "_".join(phases)

    def _effective_ev_charging_power_kw(
        self,
        observation: Mapping[str, Union[int, float]],
        charger_info: Mapping[str, Any],
        nominal_max_charging_power_kw: float,
    ) -> float:
        nominal = self._positive_or_default(
            nominal_max_charging_power_kw,
            self.ev_default_charging_power_kw,
        )
        if not self.ev_use_effective_charging_power_for_schedule:
            return nominal

        phases = self._phase_connection_to_phases(self._charger_phase_connection(charger_info))
        charger_id = str(charger_info.get("charger_id") or charger_info.get("id") or "")
        if not phases and charger_id:
            phases = tuple(
                phase
                for phase in ("L1", "L2", "L3")
                if any(
                    charger_id in str(key)
                    and str(key).endswith(f"phase_connection_{phase}")
                    and self._safe_float(value, default=0.0) > 0.5
                    for key, value in observation.items()
                )
            )
        if not phases:
            return nominal

        current_power = max(
            self._extract_first(
                charger_info,
                (
                    "applied_power_kw",
                    "commanded_power_kw",
                    "current_power_kw",
                    "charging_power_kw",
                ),
                default=0.0,
            ),
            0.0,
        )
        if current_power <= 0.0 and charger_id:
            for feature in ("applied_power_kw", "commanded_power_kw", "current_power_kw", "charging_power_kw"):
                for key, value in observation.items():
                    if charger_id in str(key) and str(key).endswith(feature):
                        current_power = max(self._safe_float(value, default=0.0), 0.0)
                        break
                if current_power > 0.0:
                    break
        candidates: List[float] = []

        static_import_limit = self._charger_static_import_limit_kw_map.get(charger_id) if charger_id else None
        if static_import_limit is not None:
            static_import_limit = self._safe_float(static_import_limit, default=float("nan"))
            if static_import_limit == static_import_limit and static_import_limit >= 0.0:
                candidates.append(static_import_limit)

        building_headroom = self._extract_first(
            observation,
            ("charging_building_headroom_kw",),
            default=float("nan"),
        )
        include_observed_headroom = current_power > 0.0 or not candidates
        if include_observed_headroom and building_headroom == building_headroom and building_headroom >= 0.0:
            candidates.append(building_headroom + current_power)

        phase_headrooms: List[float] = []
        if include_observed_headroom:
            for phase in phases:
                phase_headroom = self._extract_first(
                    observation,
                    (f"charging_phase_{phase}_headroom_kw",),
                    default=float("nan"),
                )
                if phase_headroom == phase_headroom and phase_headroom >= 0.0:
                    phase_headrooms.append(phase_headroom)
        if include_observed_headroom and phase_headrooms:
            candidates.append(min(phase_headrooms) * float(len(phases)) + current_power)

        if not candidates:
            return nominal

        effective_limit = min(candidates)
        return min(nominal, max(effective_limit, 0.0))

    def _ev_departure_components(self, observation: Mapping[str, Union[int, float]]) -> Mapping[str, float]:
        ev_chargers = observation.get("electric_vehicles_chargers_dict")
        if isinstance(ev_chargers, Mapping):
            rows = []
            for charger_id, charger_info in ev_chargers.items():
                if not isinstance(charger_info, Mapping):
                    continue
                charger_info = {**charger_info, "charger_id": charger_info.get("charger_id", charger_id)}
                connected = bool(charger_info.get("connected", False))
                soc = self._safe_float(charger_info.get("battery_soc"), default=0.0)
                required_soc = self._safe_float(charger_info.get("required_soc"), default=soc)
                hours_until_departure = self._safe_float(charger_info.get("hours_until_departure"), default=float("inf"))
                departure_marker_candidates = (
                    "departure_time",
                    "departure_time_step",
                    "connected_ev_departure_time",
                    "connected_ev_departure_time_step",
                    "electric_vehicle_departure_time",
                    "electric_vehicle_departure_time_step",
                )
                hours_until_departure = self._sanitize_hours_until_departure(
                    hours_until_departure,
                    charger_info,
                    departure_marker_candidates,
                )
                allow_missed_penalty = self._has_known_departure_marker(
                    charger_info,
                    departure_marker_candidates,
                )
                battery_capacity = self._extract_first(
                    charger_info,
                    (
                        "battery_capacity",
                        "battery_capacity_kwh",
                        "connected_ev_battery_capacity_kwh",
                    ),
                    default=self.ev_default_battery_capacity_kwh,
                )
                max_charging_power = self._extract_first(
                    charger_info,
                    (
                        "max_charging_power_kw",
                        "max_charging_power",
                        "max_power",
                        "nominal_power",
                    ),
                    default=self.ev_default_charging_power_kw,
                )
                effective_max_charging_power = self._effective_ev_charging_power_kw(
                    observation,
                    charger_info,
                    max_charging_power,
                )
                priority = self._extract_first(
                    charger_info,
                    ("priority", "charging_priority_ratio"),
                    default=1.0,
                )
                charged_energy = self._extract_first(
                    charger_info,
                    (
                        "last_charged_kwh",
                        "applied_energy_kwh_step",
                        "charging_action_kwh",
                        "charged_energy_kwh",
                    ),
                    default=0.0,
                )
                rows.append(
                    self._single_ev_departure_components(
                        connected=connected,
                        soc=soc,
                        required_soc=required_soc,
                        hours_until_departure=hours_until_departure,
                        battery_capacity_kwh=battery_capacity,
                        max_charging_power_kw=max_charging_power,
                        effective_max_charging_power_kw=effective_max_charging_power,
                        charged_energy_kwh=charged_energy,
                        priority=priority,
                        allow_departure_missed_penalty=allow_missed_penalty,
                    )
                )
            if rows:
                return self._sum_components(rows)
            return self._single_ev_departure_components(
                connected=False,
                soc=0.0,
                required_soc=0.0,
                hours_until_departure=float("inf"),
                battery_capacity_kwh=self.ev_default_battery_capacity_kwh,
                max_charging_power_kw=self.ev_default_charging_power_kw,
            )

        charger_ids = self._flat_ev_charger_ids(observation)
        if charger_ids:
            rows = []
            for charger_id in charger_ids:
                connected = self._safe_float(
                    observation.get(f"electric_vehicle_charger_{charger_id}_connected_state"),
                    default=0.0,
                )
                soc = self._safe_float(
                    observation.get(f"connected_electric_vehicle_at_charger_{charger_id}_soc"),
                    default=0.0,
                )
                required_soc = self._safe_float(
                    observation.get(f"connected_electric_vehicle_at_charger_{charger_id}_required_soc_departure"),
                    default=soc,
                )
                hours_until_departure = self._safe_float(
                    observation.get(f"connected_electric_vehicle_at_charger_{charger_id}_departure_time"),
                    default=float("inf"),
                )
                departure_marker_candidates = (
                    f"connected_electric_vehicle_at_charger_{charger_id}_departure_time",
                    f"connected_electric_vehicle_at_charger_{charger_id}_departure_time_step",
                    f"electric_vehicle_charger_{charger_id}_connected_ev_departure_time",
                    f"electric_vehicle_charger_{charger_id}_connected_ev_departure_time_step",
                    f"electric_vehicle_charger_{charger_id}_departure_time",
                    f"electric_vehicle_charger_{charger_id}_departure_time_step",
                )
                hours_until_departure = self._sanitize_hours_until_departure(
                    hours_until_departure,
                    observation,
                    departure_marker_candidates,
                )
                allow_missed_penalty = self._has_known_departure_marker(
                    observation,
                    departure_marker_candidates,
                )
                battery_capacity = self._extract_first(
                    observation,
                    (
                        f"connected_electric_vehicle_at_charger_{charger_id}_battery_capacity",
                        f"connected_electric_vehicle_at_charger_{charger_id}_battery_capacity_kwh",
                        f"electric_vehicle_charger_{charger_id}_connected_ev_battery_capacity_kwh",
                    ),
                    default=self.ev_default_battery_capacity_kwh,
                )
                max_charging_power = self._extract_first(
                    observation,
                    (
                        f"electric_vehicle_charger_{charger_id}_max_charging_power_kw",
                        f"electric_vehicle_charger_{charger_id}_max_charging_power",
                        f"electric_vehicle_charger_{charger_id}_max_power",
                        f"electric_vehicle_charger_{charger_id}_nominal_power",
                    ),
                    default=self.ev_default_charging_power_kw,
                )
                charger_info = {
                    "charger_id": charger_id,
                    "phase_connection_L1": observation.get(f"electric_vehicle_charger_{charger_id}_phase_connection_L1"),
                    "phase_connection_L2": observation.get(f"electric_vehicle_charger_{charger_id}_phase_connection_L2"),
                    "phase_connection_L3": observation.get(f"electric_vehicle_charger_{charger_id}_phase_connection_L3"),
                    "applied_power_kw": observation.get(f"electric_vehicle_charger_{charger_id}_applied_power_kw"),
                    "commanded_power_kw": observation.get(f"electric_vehicle_charger_{charger_id}_commanded_power_kw"),
                }
                effective_max_charging_power = self._effective_ev_charging_power_kw(
                    observation,
                    charger_info,
                    max_charging_power,
                )
                priority = self._extract_first(
                    observation,
                    (
                        f"electric_vehicle_charger_{charger_id}_charging_priority_ratio",
                        f"connected_electric_vehicle_at_charger_{charger_id}_priority",
                    ),
                    default=1.0,
                )
                charged_energy = self._extract_first(
                    observation,
                    (
                        f"electric_vehicle_charger_{charger_id}_last_charged_kwh",
                        f"electric_vehicle_charger_{charger_id}_applied_energy_kwh_step",
                        f"electric_vehicle_charger_{charger_id}_charging_action_kwh",
                        f"connected_electric_vehicle_at_charger_{charger_id}_last_charged_kwh",
                    ),
                    default=0.0,
                )
                rows.append(
                    self._single_ev_departure_components(
                        connected=connected > 0.5,
                        soc=soc,
                        required_soc=required_soc,
                        hours_until_departure=hours_until_departure,
                        battery_capacity_kwh=battery_capacity,
                        max_charging_power_kw=max_charging_power,
                        effective_max_charging_power_kw=effective_max_charging_power,
                        charged_energy_kwh=charged_energy,
                        priority=priority,
                        allow_departure_missed_penalty=allow_missed_penalty,
                    )
                )
            return self._sum_components(rows)

        connected = self._extract_first(
            observation,
            ("electric_vehicle_charger_connected_state", "electric_vehicle_charger_state"),
            default=0.0,
        )
        soc = self._extract_first(
            observation,
            ("connected_electric_vehicle_at_charger_soc", "electric_vehicle_soc"),
            default=0.0,
        )
        required_soc = self._extract_first(
            observation,
            (
                "connected_electric_vehicle_at_charger_required_soc_departure",
                "electric_vehicle_required_soc_departure",
                "required_soc",
            ),
            default=soc,
        )
        hours_until_departure = self._extract_first(
            observation,
            (
                "connected_electric_vehicle_at_charger_departure_time",
                "electric_vehicle_hours_until_departure",
                "hours_until_departure",
            ),
            default=float("inf"),
        )
        departure_marker_candidates = (
            "connected_electric_vehicle_at_charger_departure_time",
            "connected_electric_vehicle_at_charger_departure_time_step",
            "electric_vehicle_charger_connected_ev_departure_time",
            "electric_vehicle_charger_connected_ev_departure_time_step",
            "electric_vehicle_departure_time",
            "electric_vehicle_departure_time_step",
            "departure_time_step",
        )
        hours_until_departure = self._sanitize_hours_until_departure(
            hours_until_departure,
            observation,
            departure_marker_candidates,
        )
        allow_missed_penalty = self._has_known_departure_marker(
            observation,
            departure_marker_candidates,
        )
        battery_capacity = self._extract_first(
            observation,
            (
                "connected_electric_vehicle_at_charger_battery_capacity",
                "connected_electric_vehicle_at_charger_battery_capacity_kwh",
                "electric_vehicle_battery_capacity",
                "electric_vehicle_battery_capacity_kwh",
            ),
            default=self.ev_default_battery_capacity_kwh,
        )
        max_charging_power = self._extract_first(
            observation,
            (
                "electric_vehicle_charger_max_charging_power_kw",
                "electric_vehicle_charger_max_charging_power",
                "electric_vehicle_charger_max_power",
                "electric_vehicle_charger_nominal_power",
            ),
            default=self.ev_default_charging_power_kw,
        )
        effective_max_charging_power = self._effective_ev_charging_power_kw(
            observation,
            {},
            max_charging_power,
        )
        priority = self._extract_first(
            observation,
            ("electric_vehicle_charger_charging_priority_ratio", "connected_electric_vehicle_at_charger_priority"),
            default=1.0,
        )
        charged_energy = self._extract_first(
            observation,
            (
                "electric_vehicle_charger_last_charged_kwh",
                "electric_vehicle_charger_applied_energy_kwh_step",
                "electric_vehicle_charger_charging_action_kwh",
                "connected_electric_vehicle_at_charger_last_charged_kwh",
            ),
            default=0.0,
        )
        return self._single_ev_departure_components(
            connected=connected > 0.5,
            soc=soc,
            required_soc=required_soc,
            hours_until_departure=hours_until_departure,
            battery_capacity_kwh=battery_capacity,
            max_charging_power_kw=max_charging_power,
            effective_max_charging_power_kw=effective_max_charging_power,
            charged_energy_kwh=charged_energy,
            priority=priority,
            allow_departure_missed_penalty=allow_missed_penalty,
        )

    def _ev_departure_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
        return self._ev_departure_components(observation)["ev_service_penalty"]

    def _legacy_ev_departure_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
        ev_chargers = observation.get("electric_vehicles_chargers_dict")
        if not isinstance(ev_chargers, Mapping):
            return self._flat_ev_departure_penalty(observation)

        penalty = 0.0
        for charger_info in ev_chargers.values():
            if not isinstance(charger_info, Mapping):
                continue

            connected = bool(charger_info.get("connected", False))
            if not connected:
                continue

            soc = self._safe_float(charger_info.get("battery_soc"), default=0.0)
            required_soc = self._safe_float(charger_info.get("required_soc"), default=soc)
            hours_until_departure = self._safe_float(charger_info.get("hours_until_departure"), default=float("inf"))

            deficit = self._soc_deficit_fraction(soc=soc, required_soc=required_soc)
            service_shortfall = self._soc_shortfall_beyond_tolerance_fraction(
                soc=soc,
                required_soc=required_soc,
            )
            penalty += self._dense_ev_deficit_penalty(deficit, hours_until_departure) * self.state_penalty_scale
            if hours_until_departure <= self.ev_departure_window_hours:
                penalty += service_shortfall * self.ev_departure_deficit_penalty * self.state_penalty_scale
            if hours_until_departure <= 0.0 and service_shortfall > 0.0:
                penalty += self.ev_departure_missed_penalty * self.state_penalty_scale

        return penalty

    def _flat_ev_departure_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
        charger_ids = self._flat_ev_charger_ids(observation)
        if not charger_ids:
            return self._generic_flat_ev_departure_penalty(observation)

        penalty = 0.0
        for charger_id in charger_ids:
            connected = self._safe_float(
                observation.get(f"electric_vehicle_charger_{charger_id}_connected_state"),
                default=0.0,
            )
            if connected <= 0.5:
                continue

            soc = self._safe_float(
                observation.get(f"connected_electric_vehicle_at_charger_{charger_id}_soc"),
                default=0.0,
            )
            required_soc = self._safe_float(
                observation.get(f"connected_electric_vehicle_at_charger_{charger_id}_required_soc_departure"),
                default=soc,
            )
            hours_until_departure = self._safe_float(
                observation.get(f"connected_electric_vehicle_at_charger_{charger_id}_departure_time"),
                default=float("inf"),
            )
            departure_marker_candidates = (
                f"connected_electric_vehicle_at_charger_{charger_id}_departure_time",
                f"connected_electric_vehicle_at_charger_{charger_id}_departure_time_step",
                f"electric_vehicle_charger_{charger_id}_connected_ev_departure_time",
                f"electric_vehicle_charger_{charger_id}_connected_ev_departure_time_step",
                f"electric_vehicle_charger_{charger_id}_departure_time",
                f"electric_vehicle_charger_{charger_id}_departure_time_step",
            )
            hours_until_departure = self._sanitize_hours_until_departure(
                hours_until_departure,
                observation,
                departure_marker_candidates,
            )
            allow_missed_penalty = self._has_known_departure_marker(
                observation,
                departure_marker_candidates,
            )
            deficit = self._soc_deficit_fraction(soc=soc, required_soc=required_soc)
            service_shortfall = self._soc_shortfall_beyond_tolerance_fraction(
                soc=soc,
                required_soc=required_soc,
            )
            penalty += self._dense_ev_deficit_penalty(deficit, hours_until_departure) * self.state_penalty_scale
            if hours_until_departure <= self.ev_departure_window_hours:
                penalty += service_shortfall * self.ev_departure_deficit_penalty * self.state_penalty_scale
            if allow_missed_penalty and hours_until_departure <= 0.0 and service_shortfall > 0.0:
                penalty += self.ev_departure_missed_penalty * self.state_penalty_scale

        return penalty

    def _generic_flat_ev_departure_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
        connected = self._extract_first(
            observation,
            ("electric_vehicle_charger_connected_state", "electric_vehicle_charger_state"),
            default=0.0,
        )
        if connected <= 0.5:
            return 0.0

        soc = self._extract_first(
            observation,
            ("connected_electric_vehicle_at_charger_soc", "electric_vehicle_soc"),
            default=0.0,
        )
        required_soc = self._extract_first(
            observation,
            (
                "connected_electric_vehicle_at_charger_required_soc_departure",
                "electric_vehicle_required_soc_departure",
                "required_soc",
            ),
            default=soc,
        )
        hours_until_departure = self._extract_first(
            observation,
            (
                "connected_electric_vehicle_at_charger_departure_time",
                "electric_vehicle_hours_until_departure",
                "hours_until_departure",
            ),
            default=float("inf"),
        )
        departure_marker_candidates = (
            "connected_electric_vehicle_at_charger_departure_time",
            "connected_electric_vehicle_at_charger_departure_time_step",
            "electric_vehicle_charger_connected_ev_departure_time",
            "electric_vehicle_charger_connected_ev_departure_time_step",
            "electric_vehicle_departure_time",
            "electric_vehicle_departure_time_step",
            "departure_time_step",
        )
        hours_until_departure = self._sanitize_hours_until_departure(
            hours_until_departure,
            observation,
            departure_marker_candidates,
        )
        allow_missed_penalty = self._has_known_departure_marker(
            observation,
            departure_marker_candidates,
        )

        deficit = self._soc_deficit_fraction(soc=soc, required_soc=required_soc)
        service_shortfall = self._soc_shortfall_beyond_tolerance_fraction(
            soc=soc,
            required_soc=required_soc,
        )
        penalty = self._dense_ev_deficit_penalty(deficit, hours_until_departure) * self.state_penalty_scale
        if hours_until_departure <= self.ev_departure_window_hours:
            penalty += service_shortfall * self.ev_departure_deficit_penalty * self.state_penalty_scale
        if allow_missed_penalty and hours_until_departure <= 0.0 and service_shortfall > 0.0:
            penalty += self.ev_departure_missed_penalty * self.state_penalty_scale
        return penalty

    @staticmethod
    def _flat_ev_charger_ids(observation: Mapping[str, Union[int, float]]) -> List[str]:
        prefix = "connected_electric_vehicle_at_charger_"
        suffix = "_soc"
        charger_ids: List[str] = []
        seen: set[str] = set()
        for key in observation:
            raw = str(key)
            if not raw.startswith(prefix) or not raw.endswith(suffix):
                continue
            charger_id = raw[len(prefix) : -len(suffix)]
            if not charger_id or charger_id in seen:
                continue
            seen.add(charger_id)
            charger_ids.append(charger_id)
        return charger_ids

    @staticmethod
    def _soc_deficit_fraction(*, soc: float, required_soc: float) -> float:
        if soc != soc or required_soc != required_soc:
            return 0.0
        deficit = max(required_soc - soc, 0.0)
        if max(abs(soc), abs(required_soc)) > 1.5:
            deficit /= 100.0
        return deficit

    def _soc_shortfall_beyond_tolerance_fraction(self, *, soc: float, required_soc: float) -> float:
        if soc != soc or required_soc != required_soc:
            return 0.0
        if max(abs(soc), abs(required_soc)) > 1.5:
            soc /= 100.0
            required_soc /= 100.0
        service_target_soc = max(required_soc - self.ev_departure_service_tolerance, 0.0)
        return max(service_target_soc - soc, 0.0)

    @staticmethod
    def _soc_ratio(value: float) -> float:
        if value != value:
            return 0.0
        if abs(value) > 1.5:
            value /= 100.0
        return min(max(value, 0.0), 1.0)

    @staticmethod
    def _positive_or_default(value: float, default: float) -> float:
        if value != value or value <= 0.0:
            return default
        return value

    @staticmethod
    def _priority_weight(value: float) -> float:
        if value != value:
            return 1.0
        return min(max(value, 0.1), 5.0)

    def _dense_ev_deficit_penalty(self, deficit: float, hours_until_departure: Optional[float]) -> float:
        if deficit <= 0.0 or self.ev_connected_deficit_penalty <= 0.0:
            return 0.0
        if hours_until_departure is None or hours_until_departure != hours_until_departure:
            hours_until_departure = self.ev_departure_window_hours
        urgency_denominator = max(float(hours_until_departure), self.ev_departure_window_hours, 1.0)
        shaped_deficit = deficit ** self.ev_connected_deficit_exponent
        return (shaped_deficit * self.ev_connected_deficit_penalty) / urgency_denominator

    def _ev_v2g_service_penalty(
        self,
        *,
        discharge_kwh: float,
        service_risk: float,
        priority_weight: float,
        hours_until_departure: Optional[float],
    ) -> float:
        if self.ev_v2g_service_penalty <= 0.0 or discharge_kwh <= 0.0 or service_risk <= 0.0:
            return 0.0
        if hours_until_departure is None or hours_until_departure != hours_until_departure:
            return 0.0
        if hours_until_departure == float("inf"):
            return 0.0

        window = max(self.ev_departure_window_hours, 1.0e-6)
        urgency = 1.0 + max(window - max(hours_until_departure, 0.0), 0.0) / window
        return discharge_kwh * self.ev_v2g_service_penalty * priority_weight * (1.0 + service_risk) * urgency

    def _deferrable_service_components(self, observation: Mapping[str, Union[int, float]]) -> Mapping[str, float]:
        if self.deferrable_deadline_missed_penalty <= 0.0 and self.deferrable_urgency_penalty <= 0.0:
            return {
                "deferrable_pending_count": 0.0,
                "deferrable_running_count": 0.0,
                "deferrable_can_start_count": 0.0,
                "deferrable_deadline_missed_count": 0.0,
                "deferrable_deadline_missed_penalty_amount": 0.0,
                "deferrable_urgency_penalty_amount": 0.0,
                "deferrable_service_penalty": 0.0,
            }

        appliances = observation.get("deferrable_appliances_dict")
        if isinstance(appliances, Mapping):
            rows = []
            for appliance_info in appliances.values():
                if isinstance(appliance_info, Mapping):
                    rows.append(self._single_deferrable_service_components(appliance_info))
            if rows:
                return self._sum_components(rows)
            return self._empty_deferrable_service_components()

        return self._flat_deferrable_service_components(observation)

    def _deferrable_service_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
        return self._deferrable_service_components(observation)["deferrable_service_penalty"]

    def _single_deferrable_service_components(self, appliance_info: Mapping[str, Any]) -> Mapping[str, float]:
        pending = self._safe_float(appliance_info.get("pending"), default=0.0)
        running = self._safe_float(appliance_info.get("running"), default=0.0)
        can_start = self._safe_float(appliance_info.get("can_start"), default=1.0)
        deadline_missed = max(self._safe_float(appliance_info.get("deadline_missed"), default=0.0), 0.0)
        urgency = min(max(self._safe_float(appliance_info.get("urgency_ratio"), default=0.0), 0.0), 1.0)
        priority = max(self._safe_float(appliance_info.get("priority"), default=1.0), 0.1)
        remaining_energy = max(self._safe_float(appliance_info.get("remaining_energy_kwh"), default=0.0), 0.0)
        cycle_energy = max(self._safe_float(appliance_info.get("cycle_energy_kwh"), default=0.0), 0.0)
        remaining_ratio = min(remaining_energy / cycle_energy, 1.0) if cycle_energy > 0.0 else float(remaining_energy > 0.0)

        deadline_penalty = deadline_missed * self.deferrable_deadline_missed_penalty * priority * self.state_penalty_scale
        urgency_penalty = 0.0
        if pending > 0.5 and running <= 0.5 and can_start > 0.5:
            urgency_penalty = urgency * remaining_ratio * self.deferrable_urgency_penalty * priority * self.state_penalty_scale
        return {
            "deferrable_pending_count": float(pending > 0.5),
            "deferrable_running_count": float(running > 0.5),
            "deferrable_can_start_count": float(can_start > 0.5),
            "deferrable_deadline_missed_count": float(deadline_missed > 0.0),
            "deferrable_deadline_missed_penalty_amount": deadline_penalty,
            "deferrable_urgency_penalty_amount": urgency_penalty,
            "deferrable_service_penalty": deadline_penalty + urgency_penalty,
        }

    @staticmethod
    def _empty_deferrable_service_components() -> Mapping[str, float]:
        return {
            "deferrable_pending_count": 0.0,
            "deferrable_running_count": 0.0,
            "deferrable_can_start_count": 0.0,
            "deferrable_deadline_missed_count": 0.0,
            "deferrable_deadline_missed_penalty_amount": 0.0,
            "deferrable_urgency_penalty_amount": 0.0,
            "deferrable_service_penalty": 0.0,
        }

    def _single_deferrable_service_penalty(self, appliance_info: Mapping[str, Any]) -> float:
        return self._single_deferrable_service_components(appliance_info)["deferrable_service_penalty"]

    def _flat_deferrable_service_components(self, observation: Mapping[str, Union[int, float]]) -> Mapping[str, float]:
        rows = []
        for appliance_id in self._flat_deferrable_appliance_ids(observation):
            prefix = f"deferrable_appliance_{appliance_id}_"
            appliance_info = {
                "pending": observation.get(f"{prefix}pending"),
                "running": observation.get(f"{prefix}running"),
                "can_start": observation.get(f"{prefix}can_start"),
                "deadline_missed": observation.get(f"{prefix}deadline_missed"),
                "urgency_ratio": observation.get(f"{prefix}urgency_ratio"),
                "priority": observation.get(f"{prefix}priority"),
                "remaining_energy_kwh": observation.get(f"{prefix}remaining_energy_kwh"),
                "cycle_energy_kwh": observation.get(f"{prefix}cycle_energy_kwh"),
            }
            rows.append(self._single_deferrable_service_components(appliance_info))
        return self._sum_components(rows) if rows else self._empty_deferrable_service_components()

    def _flat_deferrable_service_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
        return self._flat_deferrable_service_components(observation)["deferrable_service_penalty"]

    @staticmethod
    def _flat_deferrable_appliance_ids(observation: Mapping[str, Union[int, float]]) -> List[str]:
        prefix = "deferrable_appliance_"
        suffix = "_pending"
        appliance_ids: List[str] = []
        seen: set[str] = set()
        for key in observation:
            raw = str(key)
            if not raw.startswith(prefix) or not raw.endswith(suffix):
                continue
            appliance_id = raw[len(prefix) : -len(suffix)]
            if not appliance_id or appliance_id in seen:
                continue
            seen.add(appliance_id)
            appliance_ids.append(appliance_id)
        return appliance_ids

    def _hard_constraint_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
        return self._hard_constraint_components(observation)["hard_constraint_penalty"]

    def _hard_constraint_components(self, observation: Mapping[str, Union[int, float]]) -> Mapping[str, float]:
        service_violation = max(self._extract_first(observation, self.SERVICE_VIOLATION_KEYS, default=0.0), 0.0)
        power_outage = max(self._safe_float(observation.get("power_outage"), default=0.0), 0.0)
        service_violation_penalty = service_violation * self.grid_violation_penalty
        power_outage_penalty = power_outage * self.power_outage_penalty * self.state_penalty_scale
        battery_components = dict(self._battery_safety_components(observation))
        battery_safety_penalty = battery_components["battery_safety_penalty"]
        ev_components = dict(self._ev_departure_components(observation))
        ev_service_penalty = ev_components["ev_service_penalty"]
        deferrable_components = dict(self._deferrable_service_components(observation))
        deferrable_service_penalty = deferrable_components["deferrable_service_penalty"]
        hard_constraint_penalty = (
            service_violation_penalty
            + power_outage_penalty
            + battery_safety_penalty
            + ev_service_penalty
            + deferrable_service_penalty
        )

        return {
            "service_violation": service_violation,
            "service_violation_penalty": service_violation_penalty,
            "power_outage": power_outage,
            "power_outage_penalty": power_outage_penalty,
            **battery_components,
            **ev_components,
            **deferrable_components,
            "hard_constraint_penalty": hard_constraint_penalty,
            "state_penalty_scale": self.state_penalty_scale,
        }

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        self._refresh_state_penalty_scale()
        self._refresh_charger_phase_map()

        component_rows: List[Mapping[str, float]] = []
        per_building_rewards = []
        settlement_rows, settlement_totals = self._community_settlement_components(observations)
        for index, observation in enumerate(observations):
            cost_components = dict(self._cost_components(observation))
            hard_components = dict(self._hard_constraint_components(observation))
            settlement_components = dict(settlement_rows[index]) if index < len(settlement_rows) else {}
            local_cost_weighted_reward = cost_components["local_cost_reward"] * self.local_cost_weight
            reward = (
                local_cost_weighted_reward
                + settlement_components.get("community_settlement_reward", 0.0)
                - hard_components["hard_constraint_penalty"]
            )
            component = {
                **cost_components,
                "local_cost_weight": self.local_cost_weight,
                "local_cost_weighted_reward": local_cost_weighted_reward,
                **settlement_components,
                **hard_components,
                "community_import_penalty": 0.0,
                "community_import_linear_penalty": 0.0,
                "community_peak_import_penalty": 0.0,
                "community_export_penalty": 0.0,
                "community_shared_penalty": 0.0,
                "reward_total": reward,
            }
            component_rows.append(component)
            per_building_rewards.append(reward)

        community_import = sum(
            max(self._safe_float(obs.get("net_electricity_consumption"), default=0.0), 0.0)
            for obs in observations
        )
        community_net_consumption = sum(
            self._safe_float(obs.get("net_electricity_consumption"), default=0.0)
            for obs in observations
        )
        community_export = max(-community_net_consumption, 0.0)
        agent_count = max(len(observations), 1)
        community_import_linear_penalty = 0.0
        community_peak_import_penalty = 0.0
        community_export_penalty = 0.0
        shared_penalty = 0.0
        shared_penalty_per_agent = 0.0

        if self.community_import_penalty > 0.0:
            community_import_linear_penalty = community_import * self.community_import_penalty

        if self.community_peak_import_penalty > 0.0:
            community_peak_import_penalty = (community_import**2) * self.community_peak_import_penalty

        if self.community_export_penalty > 0.0:
            community_export_penalty = community_export * self.community_export_penalty

        shared_penalty = community_import_linear_penalty + community_peak_import_penalty + community_export_penalty
        if shared_penalty > 0.0:
            shared_penalty_per_agent = shared_penalty / agent_count if self.community_penalty_divide_by_agents else shared_penalty
            per_building_rewards = [reward - shared_penalty_per_agent for reward in per_building_rewards]
            component_rows = [
                {
                    **row,
                    "community_import_penalty": shared_penalty_per_agent,
                    "community_import_linear_penalty": community_import_linear_penalty,
                    "community_peak_import_penalty": community_peak_import_penalty,
                    "community_export_penalty": community_export_penalty,
                    "community_shared_penalty": shared_penalty_per_agent,
                    "reward_total": row["reward_total"] - shared_penalty_per_agent,
                }
                for row in component_rows
            ]

        self.last_components_by_agent = component_rows
        self.last_community_components = {
            "community_import_energy": community_import,
            "community_export_energy": community_export,
            "community_net_consumption": community_net_consumption,
            "community_import_penalty": shared_penalty,
            "community_import_linear_penalty": community_import_linear_penalty,
            "community_peak_import_penalty": community_peak_import_penalty,
            "community_export_penalty": community_export_penalty,
            "community_shared_penalty": shared_penalty,
            "community_shared_penalty_per_agent": shared_penalty_per_agent,
            "community_penalty_divide_by_agents": float(self.community_penalty_divide_by_agents),
            **settlement_totals,
        }

        if self.central_agent:
            return [sum(per_building_rewards)]

        return per_building_rewards

    def get_last_components(self) -> Mapping[str, Any]:
        return {
            "per_agent": [dict(row) for row in self.last_components_by_agent],
            "community": dict(self.last_community_components),
        }


class CostServiceGuardRewardV2(CostHardConstraintReward):
    """Named reward profile that protects EV service before optimizing cost.

    This is the first MADDPG-oriented profile that fixed the failure mode where
    the agent reduced cost by using V2G from EVs that were still below their
    service target.
    """

    DEFAULT_KWARGS = {
        "ev_departure_window_hours": 4.0,
        "ev_connected_deficit_penalty": 120.0,
        "ev_schedule_deficit_penalty": 480.0,
        "ev_departure_deficit_penalty": 480.0,
        "ev_departure_missed_penalty": 1000.0,
        "ev_v2g_service_penalty": 200.0,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCostBalancedRewardV3(CostHardConstraintReward):
    """Named reward profile for Phase 6F.1 cost tuning under EV service gates."""

    DEFAULT_KWARGS = {
        "ev_departure_window_hours": 3.0,
        "ev_connected_deficit_penalty": 70.0,
        "ev_schedule_deficit_penalty": 320.0,
        "ev_departure_deficit_penalty": 420.0,
        "ev_departure_missed_penalty": 1000.0,
        "ev_v2g_service_penalty": 140.0,
        "battery_throughput_penalty": 0.002,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityBandRewardV4(CostHardConstraintReward):
    """Phase 6F.2 profile aligned with community settlement and EV target bands.

    The main service gate remains the minimum acceptable EV departure. This
    profile adds two cost-oriented signals that the previous profiles missed:
    community-market settlement cost and a soft penalty for charging connected
    EVs beyond the requested SOC band.
    """

    DEFAULT_KWARGS = {
        "local_cost_weight": 0.0,
        "export_credit_ratio": 0.0,
        "community_settlement_cost_weight": 1.0,
        "community_local_price_ratio": 0.8,
        "community_grid_export_price": 0.0,
        "community_import_penalty": 0.0,
        "community_peak_import_penalty": 0.0005,
        "community_export_penalty": 0.0,
        "community_penalty_divide_by_agents": True,
        "ev_departure_window_hours": 3.0,
        "ev_departure_service_tolerance": 0.05,
        "ev_over_service_tolerance": 0.05,
        "ev_over_service_penalty": 40.0,
        "ev_connected_deficit_penalty": 80.0,
        "ev_schedule_deficit_penalty": 360.0,
        "ev_departure_deficit_penalty": 420.0,
        "ev_departure_missed_penalty": 1000.0,
        "ev_v2g_service_penalty": 180.0,
        "battery_throughput_penalty": 0.05,
        "deferrable_deadline_missed_penalty": 100.0,
        "deferrable_urgency_penalty": 10.0,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityStorageBandRewardV41(CostHardConstraintReward):
    """Phase 6F.3 profile: V4 with stronger storage discipline and EV band cost."""

    DEFAULT_KWARGS = {
        **CostServiceCommunityBandRewardV4.DEFAULT_KWARGS,
        "ev_over_service_tolerance": 0.02,
        "ev_over_service_penalty": 120.0,
        "ev_connected_deficit_penalty": 90.0,
        "ev_schedule_deficit_penalty": 420.0,
        "ev_departure_deficit_penalty": 500.0,
        "ev_v2g_service_penalty": 220.0,
        "battery_throughput_penalty": 1.0,
        "community_peak_import_penalty": 0.0008,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityServiceBandRewardV42(CostHardConstraintReward):
    """Phase 6F.3 profile: V41 with service-first EV target-band pressure.

    V41 disciplined storage but still allowed a cost-improving policy that
    under-served infeasible EV departures and over-served the feasible one.
    V42 keeps community settlement cost but raises the dense/schedule/departure
    EV terms and the over-service term, so cost improvements must not come
    primarily from ignoring EV service quality.
    """

    DEFAULT_KWARGS = {
        **CostServiceCommunityStorageBandRewardV41.DEFAULT_KWARGS,
        "community_settlement_cost_weight": 0.8,
        "ev_over_service_tolerance": 0.02,
        "ev_over_service_penalty": 350.0,
        "ev_connected_deficit_penalty": 180.0,
        "ev_schedule_deficit_penalty": 900.0,
        "ev_departure_deficit_penalty": 1200.0,
        "ev_departure_missed_penalty": 2000.0,
        "ev_v2g_service_penalty": 400.0,
        "battery_throughput_penalty": 1.2,
        "community_peak_import_penalty": 0.0008,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityBatteryValueRewardV43(CostHardConstraintReward):
    """Phase 6F battery-value profile.

    V42 made EV service much safer, but its named-profile defaults also fell
    back to an artificial storage SOC minimum of 5% and a very high throughput
    cost. That made an empty stationary battery look unsafe and made useful
    community arbitrage too expensive relative to the settlement price signal.

    V43 keeps the V42 EV-service pressure, restores physical storage fallback
    bounds, scales dense state penalties by the simulator time step, and leaves
    only a light throughput cost to discourage pure churning. When the
    simulator exposes per-storage SOC limits, those observed limits still
    override the physical fallback.
    """

    DEFAULT_KWARGS = {
        **CostServiceCommunityServiceBandRewardV42.DEFAULT_KWARGS,
        "battery_soc_min": 0.0,
        "battery_soc_max": 1.0,
        "use_observed_storage_soc_limits": True,
        "battery_throughput_penalty": 0.02,
        "scale_state_penalties_by_time_step": True,
        "state_penalty_reference_seconds": 3600.0,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunitySmoothServiceRewardV44(CostHardConstraintReward):
    """Phase 6F.4 profile: smoother dense EV service with hard terminal safety.

    V43 made the reward physically safer, but the dense EV service terms still
    dominated long 15-second episodes, especially Building 15. V44 keeps strong
    departure and V2G protection, but shapes dense connected/schedule deficits
    quadratically so early recoverable deficits do not swamp the cost signal.
    """

    DEFAULT_KWARGS = {
        **CostServiceCommunityBatteryValueRewardV43.DEFAULT_KWARGS,
        "community_settlement_cost_weight": 0.9,
        "ev_departure_window_hours": 3.0,
        "ev_over_service_tolerance": 0.03,
        "ev_over_service_penalty": 220.0,
        "ev_connected_deficit_penalty": 130.0,
        "ev_connected_deficit_exponent": 2.0,
        "ev_schedule_deficit_penalty": 620.0,
        "ev_schedule_deficit_exponent": 2.0,
        "ev_departure_deficit_penalty": 1500.0,
        "ev_departure_missed_penalty": 2500.0,
        "ev_v2g_service_penalty": 600.0,
        "battery_throughput_penalty": 0.01,
        "community_peak_import_penalty": 0.0006,
        "scale_state_penalties_by_time_step": True,
        "state_penalty_reference_seconds": 3600.0,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityFeasibleServiceRewardV45(CostHardConstraintReward):
    """Phase 6F.5 profile: EV service pressure based on feasible charge power.

    V44 showed that Building 15 can be phase-headroom limited: chargers are
    commanded positively, but the physical per-phase limits cap useful power
    below the charger nominal rating. V45 keeps the same service-first intent,
    but computes schedule pressure from effective available charging power
    when phase/headroom observations are present. Departure-window pressure is
    applied to unrecoverable schedule deficit instead of the whole SOC gap.
    """

    DEFAULT_KWARGS = {
        **CostServiceCommunitySmoothServiceRewardV44.DEFAULT_KWARGS,
        "ev_use_effective_charging_power_for_schedule": True,
        "ev_departure_window_penalty_mode": "schedule_deficit",
        "ev_schedule_deficit_penalty": 720.0,
        "ev_departure_deficit_penalty": 900.0,
        "ev_departure_missed_penalty": 2500.0,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityFeasiblePrecisionRewardV46(CostHardConstraintReward):
    """Phase 6H.6 profile: protect EV precision without letting infeasible gaps dominate.

    V45/6H.5 reached the minimum acceptable feasible EV gate, but still missed
    the feasible SOC band in one over-service event and kept the critic pinned
    near the negative target clip. V46 keeps the same physical feasibility
    model, caps only the training penalty for unrecoverable schedule shortfall,
    and raises over-service pressure. Raw deficit/surplus components remain
    uncapped for diagnostics and KPI comparison.
    """

    DEFAULT_KWARGS = {
        **CostServiceCommunityFeasibleServiceRewardV45.DEFAULT_KWARGS,
        "ev_over_service_tolerance": 0.02,
        "ev_over_service_penalty": 420.0,
        "ev_schedule_deficit_cap_soc": 0.08,
        "ev_departure_window_shortfall_cap_soc": 0.08,
        "ev_departure_deficit_penalty": 760.0,
        "ev_schedule_deficit_penalty": 620.0,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityFeasiblePrecisionRewardV47(CostHardConstraintReward):
    """Phase 6H.7 profile: keep V46 feasible service and push down EV over-service.

    V46 became the best cost milestone while preserving the feasible minimum
    EV gate, but its remaining feasible SOC error was over-service. V47 keeps
    the V46 deficit caps and feasibility model unchanged, then tightens the
    upper target band and increases the over-service penalty. This avoids
    teaching the agent to under-charge infeasible EVs while making "too full"
    departures less attractive.
    """

    DEFAULT_KWARGS = {
        **CostServiceCommunityFeasiblePrecisionRewardV46.DEFAULT_KWARGS,
        "ev_over_service_tolerance": 0.01,
        "ev_over_service_penalty": 760.0,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityStorageValueRewardV49(CostHardConstraintReward):
    """Phase 6 reward profile: give storage/V2G room when it creates community value.

    V48-style configs kept EV service stable, but early comparisons suggested
    that storage could look worse than no-storage if cycling was over-penalized
    relative to the community settlement signal. V49 keeps the feasible EV
    model from V46, lowers pure throughput cost, and slightly raises community
    settlement/peak pressure so storage can be rewarded when it reduces grid
    import or peaks. Physical SOC limits still come from simulator observations
    when present.
    """

    DEFAULT_KWARGS = {
        **CostServiceCommunityFeasiblePrecisionRewardV46.DEFAULT_KWARGS,
        "community_settlement_cost_weight": 1.15,
        "community_peak_import_penalty": 0.0010,
        "community_export_penalty": 0.0001,
        "battery_throughput_penalty": 0.003,
        "battery_soc_min": 0.0,
        "battery_soc_max": 1.0,
        "use_observed_storage_soc_limits": True,
        "ev_v2g_service_penalty": 700.0,
        "scale_state_penalties_by_time_step": True,
        "state_penalty_reference_seconds": 3600.0,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityDeadlineValueRewardV50(CostHardConstraintReward):
    """Phase 6 reward profile: stronger EV deadline shaping with bounded pressure.

    The EV target should not be enforced as a hard rule at every connected step,
    but the agent must see the consequence of being below the recoverable
    schedule before the departure event. V50 keeps V46's effective charging
    power model, extends the warning window, and raises schedule/departure
    pressure while capping unrecoverable deficits so infeasible departures do
    not dominate all cost/community learning.
    """

    DEFAULT_KWARGS = {
        **CostServiceCommunityFeasiblePrecisionRewardV46.DEFAULT_KWARGS,
        "community_settlement_cost_weight": 1.0,
        "community_peak_import_penalty": 0.0008,
        "ev_departure_window_hours": 6.0,
        "ev_connected_deficit_penalty": 95.0,
        "ev_connected_deficit_exponent": 2.0,
        "ev_schedule_deficit_penalty": 820.0,
        "ev_schedule_deficit_exponent": 1.5,
        "ev_schedule_deficit_cap_soc": 0.10,
        "ev_departure_window_shortfall_cap_soc": 0.10,
        "ev_departure_deficit_penalty": 980.0,
        "ev_departure_missed_penalty": 3000.0,
        "ev_over_service_tolerance": 0.02,
        "ev_over_service_penalty": 380.0,
        "ev_v2g_service_penalty": 760.0,
        "battery_throughput_penalty": 0.006,
        "scale_state_penalties_by_time_step": True,
        "state_penalty_reference_seconds": 3600.0,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityPrecisionValueRewardV51(CostHardConstraintReward):
    """Phase 6 reward profile: balance EV minimum service with target precision.

    V50 improves minimum acceptable departure service, but the short 2022 scout
    run showed it can over-serve EVs and miss the within-tolerance KPI. V51 keeps
    the useful deadline pressure, raises the over-service term, and gives the
    community/cost signal slightly more room so the policy does not learn that
    "always charge more" is the only safe response.
    """

    DEFAULT_KWARGS = {
        **CostServiceCommunityDeadlineValueRewardV50.DEFAULT_KWARGS,
        "community_settlement_cost_weight": 1.08,
        "community_peak_import_penalty": 0.0008,
        "ev_connected_deficit_penalty": 80.0,
        "ev_connected_deficit_exponent": 2.0,
        "ev_schedule_deficit_penalty": 760.0,
        "ev_schedule_deficit_exponent": 1.45,
        "ev_schedule_deficit_cap_soc": 0.08,
        "ev_departure_window_shortfall_cap_soc": 0.08,
        "ev_departure_deficit_penalty": 920.0,
        "ev_departure_missed_penalty": 2800.0,
        "ev_over_service_tolerance": 0.03,
        "ev_over_service_penalty": 1200.0,
        "ev_v2g_service_penalty": 720.0,
        "battery_throughput_penalty": 0.004,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)


class CostServiceCommunityPeakDeadlineRewardV52(CostHardConstraintReward):
    """Phase 6 reward profile: EV deadline service plus community peak reduction.

    V52 keeps the useful EV deadline pressure from V50, avoids V51's excessive
    retreat from charging, and makes the community objective more explicit:
    settlement cost, squared community import peaks and leftover grid export all
    matter. The storage throughput term stays low enough that a battery can be
    useful for peak shaving, while SOC limits remain delegated to simulator
    observations when they exist.
    """

    DEFAULT_KWARGS = {
        **CostServiceCommunityDeadlineValueRewardV50.DEFAULT_KWARGS,
        "community_settlement_cost_weight": 1.12,
        "community_peak_import_penalty": 0.0014,
        "community_export_penalty": 0.00035,
        "ev_connected_deficit_penalty": 92.0,
        "ev_connected_deficit_exponent": 2.0,
        "ev_schedule_deficit_penalty": 820.0,
        "ev_schedule_deficit_exponent": 1.45,
        "ev_schedule_deficit_cap_soc": 0.09,
        "ev_departure_window_shortfall_cap_soc": 0.09,
        "ev_departure_deficit_penalty": 980.0,
        "ev_departure_missed_penalty": 3000.0,
        "ev_over_service_tolerance": 0.035,
        "ev_over_service_penalty": 720.0,
        "ev_v2g_service_penalty": 740.0,
        "battery_throughput_penalty": 0.0035,
        "battery_soc_min": 0.0,
        "battery_soc_max": 1.0,
        "use_observed_storage_soc_limits": True,
    }

    def __init__(self, env_metadata: Mapping[str, Any], **kwargs: Any) -> None:
        params = dict(self.DEFAULT_KWARGS)
        params.update(kwargs)
        super().__init__(env_metadata, **params)
