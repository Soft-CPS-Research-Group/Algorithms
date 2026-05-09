"""Cost-focused reward with strong penalties for hard operational constraints."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Union

from citylearn.reward_function import RewardFunction


class CostHardConstraintReward(RewardFunction):
    """Encourage low cost while strongly discouraging EV departure and grid constraint violations."""

    SERVICE_VIOLATION_KEYS = (
        "electrical_service_violation_kwh",
        "electrical_service_violation",
        "service_violation_kwh",
        "service_violation",
    )

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        export_credit_ratio: float = 0.8,
        grid_violation_penalty: float = 60.0,
        power_outage_penalty: float = 120.0,
        ev_departure_window_hours: float = 1.0,
        ev_connected_deficit_penalty: float = 0.0,
        ev_departure_deficit_penalty: float = 120.0,
        ev_departure_missed_penalty: float = 250.0,
        battery_soc_min: float = 0.05,
        battery_soc_max: float = 0.95,
        battery_soc_violation_penalty: float = 30.0,
        community_import_penalty: float = 0.0,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.export_credit_ratio = float(export_credit_ratio)
        self.grid_violation_penalty = float(grid_violation_penalty)
        self.power_outage_penalty = float(power_outage_penalty)
        self.ev_departure_window_hours = float(ev_departure_window_hours)
        self.ev_connected_deficit_penalty = float(ev_connected_deficit_penalty)
        self.ev_departure_deficit_penalty = float(ev_departure_deficit_penalty)
        self.ev_departure_missed_penalty = float(ev_departure_missed_penalty)
        self.battery_soc_min = float(battery_soc_min)
        self.battery_soc_max = float(battery_soc_max)
        self.battery_soc_violation_penalty = float(battery_soc_violation_penalty)
        self.community_import_penalty = float(community_import_penalty)

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

    def _cost_term(self, observation: Mapping[str, Union[int, float]]) -> float:
        net_consumption = self._safe_float(observation.get("net_electricity_consumption"), default=0.0)
        electricity_price = max(self._safe_float(observation.get("electricity_pricing"), default=0.0), 0.0)

        import_cost = max(net_consumption, 0.0) * electricity_price
        export_credit = max(-net_consumption, 0.0) * electricity_price * self.export_credit_ratio

        return -(import_cost - export_credit)

    def _battery_safety_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
        storage_soc = self._safe_float(observation.get("electrical_storage_soc"), default=float("nan"))
        if storage_soc != storage_soc:  # NaN-safe check
            return 0.0

        below = max(self.battery_soc_min - storage_soc, 0.0)
        above = max(storage_soc - self.battery_soc_max, 0.0)
        violation = below + above
        return violation * self.battery_soc_violation_penalty

    def _ev_departure_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
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

            deficit = max(required_soc - soc, 0.0)
            penalty += self._dense_ev_deficit_penalty(deficit, hours_until_departure)
            if hours_until_departure <= self.ev_departure_window_hours:
                penalty += deficit * self.ev_departure_deficit_penalty
            if hours_until_departure <= 0.0 and deficit > 0.0:
                penalty += self.ev_departure_missed_penalty

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
            penalty += self._dense_ev_deficit_penalty(deficit, hours_until_departure)
            if hours_until_departure <= self.ev_departure_window_hours:
                penalty += deficit * self.ev_departure_deficit_penalty
            if hours_until_departure <= 0.0 and deficit > 0.0:
                penalty += self.ev_departure_missed_penalty

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
        penalty = self._dense_ev_deficit_penalty(deficit, hours_until_departure)
        if hours_until_departure <= self.ev_departure_window_hours:
            penalty += deficit * self.ev_departure_deficit_penalty
        if hours_until_departure <= 0.0 and deficit > 0.0:
            penalty += self.ev_departure_missed_penalty
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

    def _dense_ev_deficit_penalty(self, deficit: float, hours_until_departure: Optional[float]) -> float:
        if deficit <= 0.0 or self.ev_connected_deficit_penalty <= 0.0:
            return 0.0
        if hours_until_departure is None or hours_until_departure != hours_until_departure:
            hours_until_departure = self.ev_departure_window_hours
        urgency_denominator = max(float(hours_until_departure), self.ev_departure_window_hours, 1.0)
        return (deficit * self.ev_connected_deficit_penalty) / urgency_denominator

    def _hard_constraint_penalty(self, observation: Mapping[str, Union[int, float]]) -> float:
        service_violation = max(self._extract_first(observation, self.SERVICE_VIOLATION_KEYS, default=0.0), 0.0)
        power_outage = max(self._safe_float(observation.get("power_outage"), default=0.0), 0.0)

        return (
            (service_violation * self.grid_violation_penalty)
            + (power_outage * self.power_outage_penalty)
            + self._battery_safety_penalty(observation)
            + self._ev_departure_penalty(observation)
        )

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        per_building_rewards = [
            self._cost_term(observation) - self._hard_constraint_penalty(observation)
            for observation in observations
        ]

        if self.community_import_penalty > 0.0:
            community_import = sum(max(self._safe_float(obs.get("net_electricity_consumption"), default=0.0), 0.0) for obs in observations)
            shared_penalty = community_import * self.community_import_penalty
            per_building_rewards = [reward - shared_penalty for reward in per_building_rewards]

        if self.central_agent:
            return [sum(per_building_rewards)]

        return per_building_rewards
