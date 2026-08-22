"""Deployment-causal EV scheduling targets for the TI-MARL actor.

The feasibility projector is a safety shield: it may prevent an infeasible EV
departure, but its interventions are not an economic charging policy.  This
module builds an independent, typed and auditable training signal from the
information that is available to the deployed actor (connection, SoC/service
need, time to departure, charger capability, and price forecasts).

The resulting targets are used only as an auxiliary actor objective.  They do
not bypass PPO, do not issue runtime commands, and do not depend on CityLearn
names or future simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from algorithms.ti_marl.contracts.models import (
    ActionDecision,
    ActionGroupInstance,
    InterfaceSnapshot,
    ObservationPart,
)


_EXPLICIT_PRICE_HORIZON = re.compile(
    r"(?:forecast_)?price_next_(?P<amount>\d+)(?P<unit>m|h)$"
)


@dataclass(frozen=True)
class EVPlanningTarget:
    """One supervised target for an EV action group."""

    agent_id: str
    group_id: str
    decision: ActionDecision
    reason: str
    current_price: float
    future_price_min: float
    required_duty_ratio: float


class CausalEVPlanner:
    """Label proactive EV actions from typed, deployment-available evidence.

    This is deliberately a small receding-horizon rule rather than a perfect
    foresight oracle.  It charges at high power when the current tariff is one
    of the cheapest known opportunities before departure, waits when a cheaper
    deployable forecast is available and there is slack, and charges whenever
    waiting would make the service target unsafe.
    """

    def __init__(
        self,
        *,
        charge_fraction: float = 0.95,
        service_tolerance_ratio: float = 0.05,
        price_tie_tolerance: float = 1.0e-6,
        urgency_duty_ratio: float = 0.85,
        minimum_price_spread: float = 0.0,
    ) -> None:
        if not 0.0 < float(charge_fraction) < 1.0:
            raise ValueError("EV planning charge_fraction must lie in (0, 1)")
        if not 0.0 <= float(service_tolerance_ratio) <= 0.5:
            raise ValueError(
                "EV planning service_tolerance_ratio must lie in [0, 0.5]"
            )
        if not 0.0 < float(urgency_duty_ratio) <= 1.0:
            raise ValueError("EV planning urgency_duty_ratio must lie in (0, 1]")
        if float(price_tie_tolerance) < 0.0 or float(minimum_price_spread) < 0.0:
            raise ValueError("EV planning price tolerances must be non-negative")
        self.charge_fraction = float(charge_fraction)
        self.service_tolerance_ratio = float(service_tolerance_ratio)
        self.price_tie_tolerance = float(price_tie_tolerance)
        self.urgency_duty_ratio = float(urgency_duty_ratio)
        self.minimum_price_spread = float(minimum_price_spread)

    def targets(
        self,
        snapshot: InterfaceSnapshot,
        *,
        seconds_per_time_step: float,
    ) -> Tuple[EVPlanningTarget, ...]:
        results: list[EVPlanningTarget] = []
        for agent_id in snapshot.agent_ids:
            parts = snapshot.parts_for(agent_id)
            for group in snapshot.groups_for(agent_id):
                if group.group_type != "ev_session":
                    continue
                target = self._target_for_group(
                    agent_id,
                    group,
                    parts,
                    seconds_per_time_step=max(float(seconds_per_time_step), 1.0e-9),
                )
                if target is not None:
                    results.append(target)
        return tuple(results)

    def configuration(self) -> Mapping[str, float | str]:
        return {
            "kind": "causal_cheapest_feasible_slot_v1",
            "charge_fraction": self.charge_fraction,
            "service_tolerance_ratio": self.service_tolerance_ratio,
            "price_tie_tolerance": self.price_tie_tolerance,
            "urgency_duty_ratio": self.urgency_duty_ratio,
            "minimum_price_spread": self.minimum_price_spread,
        }

    def _target_for_group(
        self,
        agent_id: str,
        group: ActionGroupInstance,
        parts: Sequence[ObservationPart],
        *,
        seconds_per_time_step: float,
    ) -> Optional[EVPlanningTarget]:
        modes = tuple(port.mode for port in group.ports)
        charge_port = next(
            (
                port
                for port in group.ports
                if port.mode == "CHARGE_EV" and port.valid and group.enabled
            ),
            None,
        )
        if charge_port is None or "IDLE" not in modes:
            return None

        values = self._group_values(group, parts)
        if values.get("connected_state", 0.0) <= 0.5:
            return None
        hours_until_departure = values.get("hours_until_departure")
        if hours_until_departure is None or hours_until_departure < 0.0:
            return None

        available_power = max(float(group.max_charge_power_kw), 0.0) * max(
            float(charge_port.upper_bound), 0.0
        )
        available_power = min(
            available_power,
            max(values.get("available_charge_power_kw", available_power), 0.0),
        )
        if available_power <= 1.0e-9:
            return None

        energy_needed = self._energy_needed(values)
        if energy_needed is None or energy_needed <= 1.0e-9:
            return None
        efficiency = float(
            np.clip(
                values.get(
                    "charge_efficiency_at_max_ratio",
                    values.get("charger_efficiency_ratio", 1.0),
                ),
                1.0e-3,
                1.0,
            )
        )
        remaining_hours = max(float(hours_until_departure), 0.0)
        required_average_power = energy_needed / max(
            efficiency * max(remaining_hours, seconds_per_time_step / 3600.0),
            1.0e-9,
        )
        required_duty_ratio = float(
            np.clip(required_average_power / available_power, 0.0, 1.0)
        )

        current_price = self._current_price(parts)
        future_prices = self._future_prices(parts, remaining_hours)
        if current_price is None or not future_prices:
            # Price-free targets would merely reproduce the service shield.
            # Leave those samples to PPO and the explicit safety projection.
            return None

        step_hours = seconds_per_time_step / 3600.0
        future_capacity = available_power * efficiency * max(
            remaining_hours - step_hours,
            0.0,
        )
        urgent = (
            remaining_hours <= step_hours + 1.0e-9
            or energy_needed > future_capacity + 1.0e-9
            or required_duty_ratio >= self.urgency_duty_ratio
        )
        future_min = min(future_prices)
        price_spread = future_min - current_price
        cheap_now = (
            current_price
            <= future_min + self.price_tie_tolerance
            and price_spread + self.price_tie_tolerance
            >= self.minimum_price_spread
        )
        if urgent or cheap_now:
            mode = "CHARGE_EV"
            fraction = self.charge_fraction
            reason = "service_urgent" if urgent else "cheapest_forecast_opportunity"
        else:
            mode = "IDLE"
            fraction = 0.0
            reason = "cheaper_forecast_with_service_slack"
        return EVPlanningTarget(
            agent_id=agent_id,
            group_id=group.group_id,
            decision=ActionDecision(
                group_id=group.group_id,
                mode=mode,
                fraction=fraction,
                mode_index=modes.index(mode),
            ),
            reason=reason,
            current_price=float(current_price),
            future_price_min=float(future_min),
            required_duty_ratio=required_duty_ratio,
        )

    def _energy_needed(self, values: Mapping[str, float]) -> Optional[float]:
        direct = values.get("energy_to_required_soc_kwh")
        if direct is not None and np.isfinite(direct):
            return max(float(direct), 0.0)
        soc = values.get("connected_ev_soc")
        required_soc = values.get("connected_ev_required_soc_departure")
        capacity = values.get("connected_ev_battery_capacity_kwh")
        if any(value is None or not np.isfinite(value) for value in (soc, required_soc, capacity)):
            return None
        target_soc = max(float(required_soc) - self.service_tolerance_ratio, 0.0)
        return max(target_soc - float(soc), 0.0) * max(float(capacity), 0.0)

    @staticmethod
    def _group_values(
        group: ActionGroupInstance,
        parts: Sequence[ObservationPart],
    ) -> Mapping[str, float]:
        values: dict[str, float] = {}
        group_suffix = group.module_id.rsplit("_", 1)[-1]
        for part in parts:
            sensor_suffix = part.sensor_id.rsplit("_", 1)[-1]
            related = part.sensor_id == group.module_id or (
                part.sensor_type == "ev_session" and sensor_suffix == group_suffix
            )
            if not related or not part.valid or len(part.values) != 1:
                continue
            value = float(part.values[0])
            if np.isfinite(value):
                values[part.observation_id] = value
        return values

    @staticmethod
    def _current_price(parts: Sequence[ObservationPart]) -> Optional[float]:
        candidates = [
            float(part.values[0])
            for part in parts
            if part.semantic_type == "market_price"
            and part.valid
            and len(part.values) == 1
            and np.isfinite(float(part.values[0]))
        ]
        return candidates[0] if candidates else None

    @staticmethod
    def _future_prices(
        parts: Sequence[ObservationPart],
        hours_until_departure: float,
    ) -> Tuple[float, ...]:
        values: list[tuple[float, float]] = []
        for part in parts:
            if (
                part.semantic_type != "market_price_forecast"
                or not part.valid
                or len(part.values) != 1
            ):
                continue
            match = _EXPLICIT_PRICE_HORIZON.search(part.observation_id.lower())
            if match is None:
                continue
            amount = float(match.group("amount"))
            horizon_hours = amount / 60.0 if match.group("unit") == "m" else amount
            value = float(part.values[0])
            if (
                horizon_hours <= hours_until_departure + 1.0e-9
                and np.isfinite(value)
            ):
                values.append((horizon_hours, value))
        return tuple(value for _horizon, value in sorted(values))
