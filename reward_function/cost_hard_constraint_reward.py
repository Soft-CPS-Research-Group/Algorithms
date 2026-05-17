"""Cost-focused reward with strong penalties for hard operational constraints."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Union

from citylearn.reward_function import RewardFunction


class CostHardConstraintReward(RewardFunction):
    """Encourage low cost while strongly discouraging EV departure and grid constraint violations."""

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
        grid_violation_penalty: float = 60.0,
        power_outage_penalty: float = 120.0,
        ev_departure_window_hours: float = 1.0,
        ev_departure_service_tolerance: float = 0.05,
        ev_connected_deficit_penalty: float = 0.0,
        ev_schedule_deficit_penalty: float = 0.0,
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
        scale_state_penalties_by_time_step: bool = False,
        state_penalty_reference_seconds: float = 3600.0,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.export_credit_ratio = float(export_credit_ratio)
        self.grid_violation_penalty = float(grid_violation_penalty)
        self.power_outage_penalty = float(power_outage_penalty)
        self.ev_departure_window_hours = float(ev_departure_window_hours)
        self.ev_departure_service_tolerance = max(float(ev_departure_service_tolerance), 0.0)
        self.ev_connected_deficit_penalty = float(ev_connected_deficit_penalty)
        self.ev_schedule_deficit_penalty = float(ev_schedule_deficit_penalty)
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
        self.last_components_by_agent: List[Mapping[str, float]] = []
        self.last_community_components: Mapping[str, float] = {}

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

    def _extract_first(self, observation: Mapping[str, Union[int, float]], candidates: Iterable[str], default: float = 0.0) -> float:
        for key in candidates:
            if key in observation:
                return self._safe_float(observation.get(key), default=default)
        return default

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
        priority: float = 1.0,
    ) -> Mapping[str, float]:
        if not connected:
            return {
                "ev_connected_count": 0.0,
                "ev_soc_deficit_sum": 0.0,
                "ev_soc_service_target_sum": 0.0,
                "ev_soc_shortfall_beyond_tolerance_sum": 0.0,
                "ev_soc_strict_gap_within_tolerance_sum": 0.0,
                "ev_soc_surplus_sum": 0.0,
                "ev_soc_absolute_error_sum": 0.0,
                "ev_dense_deficit_penalty": 0.0,
                "ev_schedule_min_soc_required_sum": 0.0,
                "ev_schedule_soc_deficit_sum": 0.0,
                "ev_schedule_deficit_penalty": 0.0,
                "ev_departure_window_count": 0.0,
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
        available_hours = max(hours_until_departure, 0.0) if hours_until_departure == hours_until_departure else 0.0
        service_target_soc = max(required_soc - self.ev_departure_service_tolerance, 0.0)
        max_soc_gain_remaining = max_charge_kw * available_hours / capacity_kwh
        schedule_min_soc = max(service_target_soc - max_soc_gain_remaining, 0.0)
        schedule_deficit = max(schedule_min_soc - soc, 0.0)
        schedule_penalty = (
            schedule_deficit
            * self.ev_schedule_deficit_penalty
            * priority_weight
            * self.state_penalty_scale
        )

        deficit = max(required_soc - soc, 0.0)
        service_shortfall = max(service_target_soc - soc, 0.0)
        strict_gap_within_tolerance = max(deficit - service_shortfall, 0.0)
        surplus = max(soc - required_soc, 0.0)
        absolute_error = abs(soc - required_soc)
        dense_penalty = (
            self._dense_ev_deficit_penalty(deficit, hours_until_departure)
            * priority_weight
            * self.state_penalty_scale
        )
        window_penalty = 0.0
        missed_penalty = 0.0
        departure_window_count = 0.0
        missed_count = 0.0
        if hours_until_departure <= self.ev_departure_window_hours:
            departure_window_count = 1.0
            window_penalty = (
                service_shortfall
                * self.ev_departure_deficit_penalty
                * priority_weight
                * self.state_penalty_scale
            )
        if hours_until_departure <= 0.0 and service_shortfall > 0.0:
            missed_count = 1.0
            missed_penalty = self.ev_departure_missed_penalty * priority_weight

        return {
            "ev_connected_count": 1.0,
            "ev_soc_deficit_sum": deficit,
            "ev_soc_service_target_sum": service_target_soc,
            "ev_soc_shortfall_beyond_tolerance_sum": service_shortfall,
            "ev_soc_strict_gap_within_tolerance_sum": strict_gap_within_tolerance,
            "ev_soc_surplus_sum": surplus,
            "ev_soc_absolute_error_sum": absolute_error,
            "ev_dense_deficit_penalty": dense_penalty,
            "ev_schedule_min_soc_required_sum": schedule_min_soc,
            "ev_schedule_soc_deficit_sum": schedule_deficit,
            "ev_schedule_deficit_penalty": schedule_penalty,
            "ev_departure_window_count": departure_window_count,
            "ev_departure_window_penalty": window_penalty,
            "ev_departure_missed_count": missed_count,
            "ev_departure_missed_penalty_amount": missed_penalty,
            "ev_priority_weight_sum": priority_weight,
            "ev_service_penalty": dense_penalty + schedule_penalty + window_penalty + missed_penalty,
        }

    def _ev_departure_components(self, observation: Mapping[str, Union[int, float]]) -> Mapping[str, float]:
        ev_chargers = observation.get("electric_vehicles_chargers_dict")
        if isinstance(ev_chargers, Mapping):
            rows = []
            for charger_info in ev_chargers.values():
                if not isinstance(charger_info, Mapping):
                    continue
                connected = bool(charger_info.get("connected", False))
                soc = self._safe_float(charger_info.get("battery_soc"), default=0.0)
                required_soc = self._safe_float(charger_info.get("required_soc"), default=soc)
                hours_until_departure = self._safe_float(charger_info.get("hours_until_departure"), default=float("inf"))
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
                priority = self._extract_first(
                    charger_info,
                    ("priority", "charging_priority_ratio"),
                    default=1.0,
                )
                rows.append(
                    self._single_ev_departure_components(
                        connected=connected,
                        soc=soc,
                        required_soc=required_soc,
                        hours_until_departure=hours_until_departure,
                        battery_capacity_kwh=battery_capacity,
                        max_charging_power_kw=max_charging_power,
                        priority=priority,
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
                priority = self._extract_first(
                    observation,
                    (
                        f"electric_vehicle_charger_{charger_id}_charging_priority_ratio",
                        f"connected_electric_vehicle_at_charger_{charger_id}_priority",
                    ),
                    default=1.0,
                )
                rows.append(
                    self._single_ev_departure_components(
                        connected=connected > 0.5,
                        soc=soc,
                        required_soc=required_soc,
                        hours_until_departure=hours_until_departure,
                        battery_capacity_kwh=battery_capacity,
                        max_charging_power_kw=max_charging_power,
                        priority=priority,
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
        priority = self._extract_first(
            observation,
            ("electric_vehicle_charger_charging_priority_ratio", "connected_electric_vehicle_at_charger_priority"),
            default=1.0,
        )
        return self._single_ev_departure_components(
            connected=connected > 0.5,
            soc=soc,
            required_soc=required_soc,
            hours_until_departure=hours_until_departure,
            battery_capacity_kwh=battery_capacity,
            max_charging_power_kw=max_charging_power,
            priority=priority,
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

        deficit = self._soc_deficit_fraction(soc=soc, required_soc=required_soc)
        service_shortfall = self._soc_shortfall_beyond_tolerance_fraction(
            soc=soc,
            required_soc=required_soc,
        )
        penalty = self._dense_ev_deficit_penalty(deficit, hours_until_departure) * self.state_penalty_scale
        if hours_until_departure <= self.ev_departure_window_hours:
            penalty += service_shortfall * self.ev_departure_deficit_penalty * self.state_penalty_scale
        if hours_until_departure <= 0.0 and service_shortfall > 0.0:
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
        return (deficit * self.ev_connected_deficit_penalty) / urgency_denominator

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

        component_rows: List[Mapping[str, float]] = []
        per_building_rewards = []
        for observation in observations:
            cost_components = dict(self._cost_components(observation))
            hard_components = dict(self._hard_constraint_components(observation))
            reward = cost_components["local_cost_reward"] - hard_components["hard_constraint_penalty"]
            component = {
                **cost_components,
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
        }

        if self.central_agent:
            return [sum(per_building_rewards)]

        return per_building_rewards

    def get_last_components(self) -> Mapping[str, Any]:
        return {
            "per_agent": [dict(row) for row in self.last_components_by_agent],
            "community": dict(self.last_community_components),
        }
