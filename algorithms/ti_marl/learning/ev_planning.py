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
    ActionPortInstance,
    InterfaceSnapshot,
    ObservationPart,
)


_EXPLICIT_PRICE_HORIZON = re.compile(
    r"(?:forecast_)?price_next_(?P<amount>\d+)(?P<unit>m|h)$"
)
_EXPLICIT_COMMUNITY_NET_HORIZON = re.compile(
    r"forecast_community_net_next_(?P<amount>\d+)(?P<unit>m|h)_kw$"
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
    current_opportunity_value: float = float("nan")
    future_opportunity_value_min: float = float("nan")


class CausalEVPlanner:
    """Label proactive EV actions from typed, deployment-available evidence.

    This is deliberately a small receding-horizon rule rather than a perfect
    foresight oracle.  It charges at high power when the current tariff is one
    of the cheapest known opportunities before departure, waits when a cheaper
    deployable forecast is available and there is slack, charges whenever
    waiting would make the service target unsafe, and uses service-safe V2G
    only to offset local demand during an expensive tariff opportunity.
    """

    def __init__(
        self,
        *,
        charge_fraction: float = 0.95,
        discharge_fraction: float = 0.50,
        service_tolerance_ratio: float = 0.05,
        v2g_service_margin_ratio: float = 0.05,
        price_tie_tolerance: float = 1.0e-6,
        urgency_duty_ratio: float = 0.85,
        minimum_price_spread: float = 0.0,
        minimum_v2g_price_spread: float = 0.01,
        minimum_v2g_departure_hours: float = 1.0,
        v2g_avoided_import_value_ratio: float = 1.0,
        v2g_minimum_profit_margin_eur_per_kwh: float = 0.01,
        v2g_degradation_cost_eur_per_kwh: float = 0.0,
        opportunity_value_kind: str = "tariff",
    ) -> None:
        if not 0.0 < float(charge_fraction) < 1.0:
            raise ValueError("EV planning charge_fraction must lie in (0, 1)")
        if not 0.0 < float(discharge_fraction) < 1.0:
            raise ValueError("EV planning discharge_fraction must lie in (0, 1)")
        if not 0.0 <= float(service_tolerance_ratio) <= 0.5:
            raise ValueError(
                "EV planning service_tolerance_ratio must lie in [0, 0.5]"
            )
        if not 0.0 <= float(v2g_service_margin_ratio) <= 0.5:
            raise ValueError(
                "EV planning v2g_service_margin_ratio must lie in [0, 0.5]"
            )
        if not 0.0 < float(urgency_duty_ratio) <= 1.0:
            raise ValueError("EV planning urgency_duty_ratio must lie in (0, 1]")
        if (
            float(price_tie_tolerance) < 0.0
            or float(minimum_price_spread) < 0.0
            or float(minimum_v2g_price_spread) < 0.0
            or float(minimum_v2g_departure_hours) < 0.0
            or float(v2g_minimum_profit_margin_eur_per_kwh) < 0.0
            or float(v2g_degradation_cost_eur_per_kwh) < 0.0
        ):
            raise ValueError("EV planning price tolerances must be non-negative")
        if not 0.0 <= float(v2g_avoided_import_value_ratio) <= 1.0:
            raise ValueError(
                "EV planning V2G avoided-import value ratio must lie in [0, 1]"
            )
        if opportunity_value_kind not in {
            "tariff",
            "community_marginal_import",
        }:
            raise ValueError(
                "EV planning opportunity_value_kind must be 'tariff' or "
                "'community_marginal_import'"
            )
        self.charge_fraction = float(charge_fraction)
        self.discharge_fraction = float(discharge_fraction)
        self.service_tolerance_ratio = float(service_tolerance_ratio)
        self.v2g_service_margin_ratio = float(v2g_service_margin_ratio)
        self.price_tie_tolerance = float(price_tie_tolerance)
        self.urgency_duty_ratio = float(urgency_duty_ratio)
        self.minimum_price_spread = float(minimum_price_spread)
        self.minimum_v2g_price_spread = float(minimum_v2g_price_spread)
        self.minimum_v2g_departure_hours = float(minimum_v2g_departure_hours)
        self.v2g_avoided_import_value_ratio = float(
            v2g_avoided_import_value_ratio
        )
        self.v2g_minimum_profit_margin_eur_per_kwh = float(
            v2g_minimum_profit_margin_eur_per_kwh
        )
        self.v2g_degradation_cost_eur_per_kwh = float(
            v2g_degradation_cost_eur_per_kwh
        )
        self.opportunity_value_kind = str(opportunity_value_kind)

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
            "kind": "causal_bidirectional_service_capped_v2",
            "charge_fraction": self.charge_fraction,
            "discharge_fraction": self.discharge_fraction,
            "service_tolerance_ratio": self.service_tolerance_ratio,
            "v2g_service_margin_ratio": self.v2g_service_margin_ratio,
            "price_tie_tolerance": self.price_tie_tolerance,
            "urgency_duty_ratio": self.urgency_duty_ratio,
            "minimum_price_spread": self.minimum_price_spread,
            "minimum_v2g_price_spread": self.minimum_v2g_price_spread,
            "minimum_v2g_departure_hours": self.minimum_v2g_departure_hours,
            "v2g_avoided_import_value_ratio": (
                self.v2g_avoided_import_value_ratio
            ),
            "v2g_minimum_profit_margin_eur_per_kwh": (
                self.v2g_minimum_profit_margin_eur_per_kwh
            ),
            "v2g_degradation_cost_eur_per_kwh": (
                self.v2g_degradation_cost_eur_per_kwh
            ),
            "opportunity_value_kind": self.opportunity_value_kind,
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
        discharge_port = next(
            (
                port
                for port in group.ports
                if port.mode == "DISCHARGE_EV" and port.valid and group.enabled
            ),
            None,
        )
        if "IDLE" not in modes:
            return None

        values = self._group_values(group, parts)
        if values.get("connected_state", 0.0) <= 0.5:
            return None
        hours_until_departure = values.get("hours_until_departure")
        if hours_until_departure is None or hours_until_departure < 0.0:
            return None

        energy_needed = self._energy_needed(values)
        if energy_needed is None:
            return None
        current_price = self._current_price(parts)
        future_price_points = self._future_price_points(
            parts,
            float(hours_until_departure),
        )
        future_prices = tuple(value for _horizon, value in future_price_points)
        if energy_needed <= 1.0e-9:
            v2g_target = self._v2g_target(
                agent_id=agent_id,
                group=group,
                discharge_port=discharge_port,
                modes=modes,
                parts=parts,
                values=values,
                hours_until_departure=float(hours_until_departure),
                current_price=current_price,
                future_prices=future_prices,
                seconds_per_time_step=seconds_per_time_step,
            )
            if v2g_target is not None:
                return v2g_target
            return self._idle_target(
                agent_id=agent_id,
                group=group,
                modes=modes,
                reason="service_target_satisfied",
                current_price=current_price,
                future_prices=future_prices,
                required_duty_ratio=0.0,
            )
        if charge_port is None:
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
        energy_limited_fraction = energy_needed / max(
            available_power * efficiency * step_hours,
            1.0e-9,
        )
        economic_fraction = float(
            np.clip(
                max(
                    min(self.charge_fraction, energy_limited_fraction),
                    float(charge_port.lower_bound)
                    / max(float(charge_port.upper_bound), 1.0e-9),
                ),
                0.0,
                1.0,
            )
        )
        planned_charge_power_kw = available_power * economic_fraction
        future_min = min(future_prices)
        current_opportunity_value = float(current_price)
        future_opportunity_values = future_prices
        if self.opportunity_value_kind == "community_marginal_import":
            current_community_net = self._community_net_power(parts)
            future_community_net = dict(
                self._future_community_net_points(
                    parts,
                    float(hours_until_departure),
                )
            )
            paired_future_values = tuple(
                self._marginal_import_value(
                    price=future_price,
                    community_net_power_kw=future_community_net[horizon],
                    incremental_power_kw=planned_charge_power_kw,
                )
                for horizon, future_price in future_price_points
                if horizon in future_community_net
            )
            if current_community_net is None or not paired_future_values:
                return None
            current_opportunity_value = self._marginal_import_value(
                price=float(current_price),
                community_net_power_kw=current_community_net,
                incremental_power_kw=planned_charge_power_kw,
            )
            future_opportunity_values = paired_future_values
        future_opportunity_min = min(future_opportunity_values)
        price_spread = future_opportunity_min - current_opportunity_value
        cheap_now = (
            current_opportunity_value
            <= future_opportunity_min + self.price_tie_tolerance
            and price_spread + self.price_tie_tolerance
            >= self.minimum_price_spread
        )
        if urgent or cheap_now:
            mode = "CHARGE_EV"
            # A fixed economic charge fraction is useful at an ordinary cheap
            # opportunity, but it must never teach less power than the average
            # duty already required to reach the service target.  The safety
            # projector used to hide this mismatch by increasing urgent
            # actions after the actor, leaving the learned EV head dependent on
            # repeated takeovers.
            target_fraction = (
                max(self.charge_fraction, required_duty_ratio)
                if urgent
                else self.charge_fraction
            )
            fraction = float(
                np.clip(
                    max(
                        min(target_fraction, energy_limited_fraction),
                        float(charge_port.lower_bound)
                        / max(float(charge_port.upper_bound), 1.0e-9),
                    ),
                    0.0,
                    1.0,
                )
            )
            if urgent:
                reason = "service_urgent"
            elif self.opportunity_value_kind == "community_marginal_import":
                reason = "community_marginal_cheapest_opportunity"
            else:
                reason = "cheapest_forecast_opportunity"
        else:
            mode = "IDLE"
            fraction = 0.0
            reason = (
                "community_marginal_cheaper_forecast_with_service_slack"
                if self.opportunity_value_kind == "community_marginal_import"
                else "cheaper_forecast_with_service_slack"
            )
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
            current_opportunity_value=float(current_opportunity_value),
            future_opportunity_value_min=float(future_opportunity_min),
        )

    def _v2g_target(
        self,
        *,
        agent_id: str,
        group: ActionGroupInstance,
        discharge_port: ActionPortInstance | None,
        modes: Sequence[str],
        parts: Sequence[ObservationPart],
        values: Mapping[str, float],
        hours_until_departure: float,
        current_price: Optional[float],
        future_prices: Sequence[float],
        seconds_per_time_step: float,
    ) -> Optional[EVPlanningTarget]:
        """Return a demand-capped V2G target without spending service energy."""

        if (
            discharge_port is None
            or current_price is None
            or not future_prices
            or hours_until_departure + 1.0e-9
            < self.minimum_v2g_departure_hours
        ):
            return None
        future_min = min(future_prices)

        soc = values.get("connected_ev_soc")
        required_soc = values.get("connected_ev_required_soc_departure")
        capacity = values.get("connected_ev_battery_capacity_kwh")
        if any(
            value is None or not np.isfinite(value)
            for value in (soc, required_soc, capacity)
        ):
            return None
        service_floor = max(
            float(required_soc) + self.v2g_service_margin_ratio,
            float(values.get("connected_ev_soc_min_ratio", 0.0)),
        )
        surplus_energy = (
            max(float(soc) - min(service_floor, 1.0), 0.0)
            * max(float(capacity), 0.0)
        )
        if surplus_energy <= 1.0e-9:
            return None

        available_power = max(float(group.max_discharge_power_kw), 0.0) * max(
            float(discharge_port.upper_bound), 0.0
        )
        available_power = min(
            available_power,
            max(
                values.get("available_discharge_power_kw", available_power),
                0.0,
            ),
        )
        local_net_demand = self._local_inflexible_demand(
            parts,
            seconds_per_time_step=seconds_per_time_step,
        )
        if available_power <= 1.0e-9 or local_net_demand <= 1.0e-9:
            return None
        discharge_efficiency = float(
            np.clip(
                values.get(
                    "discharge_efficiency_at_max_ratio",
                    values.get("charger_efficiency_ratio", 1.0),
                ),
                1.0e-3,
                1.0,
            )
        )
        charge_efficiency = float(
            np.clip(
                values.get(
                    "charge_efficiency_at_max_ratio",
                    values.get("charger_efficiency_ratio", 1.0),
                ),
                1.0e-3,
                1.0,
            )
        )
        replacement_cost = future_min / max(
            charge_efficiency * discharge_efficiency,
            1.0e-3,
        )
        net_margin = (
            float(current_price) * self.v2g_avoided_import_value_ratio
            - replacement_cost
            - self.v2g_degradation_cost_eur_per_kwh
        )
        required_margin = max(
            self.minimum_v2g_price_spread,
            self.v2g_minimum_profit_margin_eur_per_kwh,
        )
        if net_margin + self.price_tie_tolerance < required_margin:
            return None
        step_hours = seconds_per_time_step / 3600.0
        surplus_fraction = surplus_energy * discharge_efficiency / max(
            available_power * step_hours,
            1.0e-9,
        )
        demand_fraction = local_net_demand / available_power
        fraction = float(
            np.clip(
                min(
                    self.discharge_fraction,
                    surplus_fraction,
                    demand_fraction,
                ),
                0.0,
                1.0,
            )
        )
        minimum_fraction = float(discharge_port.lower_bound) / max(
            float(discharge_port.upper_bound),
            1.0e-9,
        )
        if fraction <= 1.0e-9 or fraction + 1.0e-9 < minimum_fraction:
            return None
        return EVPlanningTarget(
            agent_id=agent_id,
            group_id=group.group_id,
            decision=ActionDecision(
                group_id=group.group_id,
                mode="DISCHARGE_EV",
                fraction=fraction,
                mode_index=modes.index("DISCHARGE_EV"),
            ),
            reason="expensive_import_with_service_surplus",
            current_price=float(current_price),
            future_price_min=float(future_min),
            required_duty_ratio=0.0,
        )

    @staticmethod
    def _idle_target(
        *,
        agent_id: str,
        group: ActionGroupInstance,
        modes: Sequence[str],
        reason: str,
        current_price: Optional[float],
        future_prices: Sequence[float],
        required_duty_ratio: float,
    ) -> EVPlanningTarget:
        return EVPlanningTarget(
            agent_id=agent_id,
            group_id=group.group_id,
            decision=ActionDecision(
                group_id=group.group_id,
                mode="IDLE",
                fraction=0.0,
                mode_index=modes.index("IDLE"),
            ),
            reason=reason,
            current_price=(
                float(current_price) if current_price is not None else float("nan")
            ),
            future_price_min=(
                float(min(future_prices)) if future_prices else float("nan")
            ),
            required_duty_ratio=float(required_duty_ratio),
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
    def _local_inflexible_demand(
        parts: Sequence[ObservationPart],
        *,
        seconds_per_time_step: float,
    ) -> float:
        """Return demand that cannot be fabricated by flexible actions."""

        def value(observation_ids: tuple[str, ...]) -> Optional[float]:
            candidates = [
                float(part.values[0])
                for part in parts
                if part.observation_id in observation_ids
                and part.scope == "local"
                and part.valid
                and len(part.values) == 1
                and np.isfinite(float(part.values[0]))
            ]
            return candidates[-1] if candidates else None

        step_hours = max(float(seconds_per_time_step) / 3600.0, 1.0e-9)
        load_power_kw = value(("non_shiftable_load_power_kw",))
        if load_power_kw is None:
            load_energy_kwh = value(
                ("non_shiftable_load", "load_energy_kwh_step")
            )
            if load_energy_kwh is not None:
                load_power_kw = load_energy_kwh / step_hours

        pv_power_kw = value(("pv_power_kw",))
        if pv_power_kw is None:
            pv_energy_kwh = value(
                ("solar_generation", "pv_energy_kwh_step")
            )
            if pv_energy_kwh is not None:
                pv_power_kw = pv_energy_kwh / step_hours

        if load_power_kw is not None:
            return max(
                float(load_power_kw) - max(float(pv_power_kw or 0.0), 0.0),
                0.0,
            )

        return 0.0

    @staticmethod
    def _future_prices(
        parts: Sequence[ObservationPart],
        hours_until_departure: float,
    ) -> Tuple[float, ...]:
        return tuple(
            value
            for _horizon, value in CausalEVPlanner._future_price_points(
                parts,
                hours_until_departure,
            )
        )

    @staticmethod
    def _future_price_points(
        parts: Sequence[ObservationPart],
        hours_until_departure: float,
    ) -> Tuple[tuple[float, float], ...]:
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
        return tuple(sorted(values))

    @staticmethod
    def _community_net_power(
        parts: Sequence[ObservationPart],
    ) -> Optional[float]:
        candidates = [
            float(part.values[0])
            for part in parts
            if part.scope == "community"
            and part.observation_id == "community_net_power_kw"
            and part.valid
            and len(part.values) == 1
            and np.isfinite(float(part.values[0]))
        ]
        return candidates[-1] if candidates else None

    @staticmethod
    def _future_community_net_points(
        parts: Sequence[ObservationPart],
        hours_until_departure: float,
    ) -> Tuple[tuple[float, float], ...]:
        values: list[tuple[float, float]] = []
        for part in parts:
            if (
                part.scope != "community"
                or not part.valid
                or len(part.values) != 1
            ):
                continue
            match = _EXPLICIT_COMMUNITY_NET_HORIZON.search(
                part.observation_id.lower()
            )
            if match is None:
                continue
            amount = float(match.group("amount"))
            horizon_hours = (
                amount / 60.0 if match.group("unit") == "m" else amount
            )
            value = float(part.values[0])
            if (
                horizon_hours <= hours_until_departure + 1.0e-9
                and np.isfinite(value)
            ):
                values.append((horizon_hours, value))
        return tuple(sorted(values))

    @staticmethod
    def _marginal_import_value(
        *,
        price: float,
        community_net_power_kw: float,
        incremental_power_kw: float,
    ) -> float:
        """Return settlement cost per incremental kWh of flexible demand.

        Community grid export has zero value on the frozen experimental
        surface.  Flexible demand inside an existing export therefore has
        zero marginal grid cost; only the portion crossing into positive
        community import is valued at the retail tariff.
        """

        increment = max(float(incremental_power_kw), 0.0)
        if increment <= 1.0e-9:
            return max(float(price), 0.0)
        before = max(float(community_net_power_kw), 0.0)
        after = max(float(community_net_power_kw) + increment, 0.0)
        imported_fraction = np.clip((after - before) / increment, 0.0, 1.0)
        return max(float(price), 0.0) * float(imported_fraction)
