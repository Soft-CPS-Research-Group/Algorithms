"""Analytic joint projection over one building's action groups."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from algorithms.ti_marl.contracts.models import (
    ActionDecision,
    ActionGroupInstance,
    InterfaceSnapshot,
    LocalActionBundle,
    ObservationPart,
)


_EXPLICIT_PRICE_HORIZON = re.compile(
    r"(?:next|forecast)[_-]?(?P<amount>\d+(?:\.\d+)?)(?P<unit>m|h)(?:_|$)"
)


class AnalyticLocalProjector:
    """Project typed bundles without performing community optimization."""

    def __init__(
        self,
        *,
        enforce_ev_service: bool = True,
        ev_service_margin_ratio: float = 0.05,
        ev_service_strategy: str = "average",
        ev_service_tolerance_ratio: float = 0.05,
        ev_service_jit_buffer_seconds: float = 0.0,
        ev_service_jit_minimum_average_fraction: float = 0.0,
        enforce_ev_discharge_reserve: bool = True,
        ev_v2g_reserve_margin_ratio: float = 0.0,
        enforce_ev_economic_guard: bool = True,
        ev_v2g_avoided_import_value_ratio: float = 1.0,
        ev_v2g_minimum_profit_margin_eur_per_kwh: float = 0.01,
        ev_v2g_degradation_cost_eur_per_kwh: float = 0.0,
        ev_v2g_require_local_demand: bool = True,
        headroom_reserve_kw: float = 0.0,
        deferrable_service_margin_seconds: float = 0.0,
    ) -> None:
        self.enforce_ev_service = bool(enforce_ev_service)
        self.ev_service_margin_ratio = max(float(ev_service_margin_ratio), 0.0)
        if ev_service_strategy not in {
            "average",
            "minimum_average",
            "just_in_time",
        }:
            raise ValueError(
                "ev_service_strategy must be one of "
                "{'average', 'minimum_average', 'just_in_time'}"
            )
        self.ev_service_strategy = str(ev_service_strategy)
        self.ev_service_tolerance_ratio = max(
            float(ev_service_tolerance_ratio), 0.0
        )
        self.ev_service_jit_buffer_seconds = max(
            float(ev_service_jit_buffer_seconds), 0.0
        )
        self.ev_service_jit_minimum_average_fraction = float(
            np.clip(ev_service_jit_minimum_average_fraction, 0.0, 1.0)
        )
        self.enforce_ev_discharge_reserve = bool(
            enforce_ev_discharge_reserve
        )
        self.ev_v2g_reserve_margin_ratio = max(
            float(ev_v2g_reserve_margin_ratio), 0.0
        )
        self.enforce_ev_economic_guard = bool(enforce_ev_economic_guard)
        self.ev_v2g_avoided_import_value_ratio = float(
            np.clip(ev_v2g_avoided_import_value_ratio, 0.0, 1.0)
        )
        self.ev_v2g_minimum_profit_margin_eur_per_kwh = max(
            float(ev_v2g_minimum_profit_margin_eur_per_kwh),
            0.0,
        )
        self.ev_v2g_degradation_cost_eur_per_kwh = max(
            float(ev_v2g_degradation_cost_eur_per_kwh),
            0.0,
        )
        self.ev_v2g_require_local_demand = bool(
            ev_v2g_require_local_demand
        )
        self.headroom_reserve_kw = max(float(headroom_reserve_kw), 0.0)
        self.deferrable_service_margin_seconds = max(
            float(deferrable_service_margin_seconds), 0.0
        )
        self.seconds_per_time_step = 3600.0

    def set_seconds_per_time_step(self, seconds: float) -> None:
        seconds = float(seconds)
        if not np.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("seconds_per_time_step must be finite and positive")
        self.seconds_per_time_step = seconds

    def configuration(self) -> Mapping[str, object]:
        return {
            "kind": "analytic_projection",
            "enforce_ev_service": self.enforce_ev_service,
            "ev_service_margin_ratio": self.ev_service_margin_ratio,
            "ev_service_strategy": self.ev_service_strategy,
            "ev_service_tolerance_ratio": self.ev_service_tolerance_ratio,
            "ev_service_jit_buffer_seconds": (
                self.ev_service_jit_buffer_seconds
            ),
            "ev_service_jit_minimum_average_fraction": (
                self.ev_service_jit_minimum_average_fraction
            ),
            "enforce_ev_discharge_reserve": (
                self.enforce_ev_discharge_reserve
            ),
            "ev_v2g_reserve_margin_ratio": self.ev_v2g_reserve_margin_ratio,
            "enforce_ev_economic_guard": self.enforce_ev_economic_guard,
            "ev_v2g_avoided_import_value_ratio": (
                self.ev_v2g_avoided_import_value_ratio
            ),
            "ev_v2g_minimum_profit_margin_eur_per_kwh": (
                self.ev_v2g_minimum_profit_margin_eur_per_kwh
            ),
            "ev_v2g_degradation_cost_eur_per_kwh": (
                self.ev_v2g_degradation_cost_eur_per_kwh
            ),
            "ev_v2g_require_local_demand": self.ev_v2g_require_local_demand,
            "headroom_reserve_kw": self.headroom_reserve_kw,
            "deferrable_service_margin_seconds": (
                self.deferrable_service_margin_seconds
            ),
            "seconds_per_time_step": self.seconds_per_time_step,
        }

    def project(
        self,
        snapshot: InterfaceSnapshot,
        bundles: Iterable[LocalActionBundle],
    ) -> Tuple[LocalActionBundle, ...]:
        groups = {group.group_id: group for group in snapshot.action_groups}
        return tuple(self._project_agent(snapshot, bundle, groups) for bundle in bundles)

    def _project_agent(
        self,
        snapshot: InterfaceSnapshot,
        bundle: LocalActionBundle,
        groups: Mapping[str, ActionGroupInstance],
    ) -> LocalActionBundle:
        interventions = list(bundle.interventions)
        decisions: Dict[str, ActionDecision] = {}
        for decision in bundle.decisions:
            group = groups.get(decision.group_id)
            if group is None:
                interventions.append(
                    self._intervention(decision.group_id, "removed_group", decision.mode, "IDLE", decision.fraction, 0.0)
                )
                continue
            if group.forced_mode is not None:
                forced_port = next(
                    (item for item in group.ports if item.mode == group.forced_mode),
                    None,
                )
                forced_fraction = float(group.forced_fraction or 0.0)
                if forced_port is None or not forced_port.valid:
                    forced_mode = "IDLE"
                    forced_fraction = 0.0
                    forced_index = 0
                else:
                    forced_mode = group.forced_mode
                    forced_index = next(
                        (
                            index
                            for index, item in enumerate(group.ports)
                            if item.mode == forced_mode
                        ),
                        0,
                    )
                decisions[group.group_id] = replace(
                    decision,
                    mode=forced_mode,
                    fraction=forced_fraction,
                    mode_index=forced_index,
                )
                interventions.append(
                    self._intervention(
                        group.group_id,
                        group.fallback_reason or "typed_failsafe",
                        decision.mode,
                        forced_mode,
                        decision.fraction,
                        forced_fraction,
                    )
                )
                continue
            port = next((item for item in group.ports if item.mode == decision.mode), None)
            if not group.enabled or port is None or not port.valid:
                decisions[group.group_id] = replace(decision, mode="IDLE", fraction=0.0, mode_index=0)
                interventions.append(
                    self._intervention(group.group_id, "invalid_port_fallback", decision.mode, "IDLE", decision.fraction, 0.0)
                )
                continue
            # ``fraction`` is relative to the *currently compiled* port.  The
            # runtime bound itself is applied once by the CityLearn codec.
            # Clipping the fraction to ``port.upper_bound`` here would apply
            # that contraction a second time.
            bounded = float(np.clip(decision.fraction, 0.0, 1.0))
            if decision.mode == "IDLE":
                bounded = 0.0
            elif float(port.lower_bound) > 0.0:
                # Port bounds are normalized physical magnitudes. Since the
                # learned fraction is relative to the currently available
                # upper bound, convert the typed minimum into that domain.
                bounded = max(
                    bounded,
                    float(port.lower_bound) / max(float(port.upper_bound), 1.0e-9),
                )
            if abs(bounded - decision.fraction) > 1.0e-9:
                interventions.append(
                    self._intervention(
                        group.group_id,
                        (
                            "typed_minimum_power"
                            if bounded > float(np.clip(decision.fraction, 0.0, 1.0))
                            else "fraction_domain"
                        ),
                        decision.mode,
                        decision.mode,
                        decision.fraction,
                        bounded,
                    )
                )
            decisions[group.group_id] = replace(decision, fraction=bounded)

        deferrable_service_floors = self._enforce_deferrable_must_start(
            snapshot,
            bundle.agent_id,
            groups,
            decisions,
            interventions,
        )
        ev_service_floors = self._enforce_ev_service(
            snapshot,
            bundle.agent_id,
            groups,
            decisions,
            interventions,
        )
        self._enforce_ev_economic_guard(
            snapshot,
            bundle.agent_id,
            groups,
            decisions,
            interventions,
        )
        self._scale_direction(
            snapshot,
            bundle.agent_id,
            groups,
            decisions,
            interventions,
            direction="charge",
            constraint_types=(
                "charging_headroom_kw",
                "charging_phase_headroom_kw",
            ),
            minimum_power_by_group={
                **ev_service_floors,
                **deferrable_service_floors,
            },
        )
        self._scale_direction(
            snapshot,
            bundle.agent_id,
            groups,
            decisions,
            interventions,
            direction="discharge",
            constraint_types=(
                "export_headroom_kw",
                "export_phase_headroom_kw",
            ),
            minimum_power_by_group={},
        )
        self._drop_actions_below_typed_minimum(
            groups,
            decisions,
            interventions,
        )
        ordered = tuple(decisions[key] for key in sorted(decisions))
        return LocalActionBundle(
            agent_id=bundle.agent_id,
            decisions=ordered,
            interventions=tuple(interventions),
        )

    def _drop_actions_below_typed_minimum(
        self,
        groups: Mapping[str, ActionGroupInstance],
        decisions: Dict[str, ActionDecision],
        interventions: list[Mapping[str, object]],
    ) -> None:
        """Never emit a continuous command below its typed operating floor."""

        for group_id, decision in tuple(decisions.items()):
            if decision.mode in {"IDLE", "START"}:
                continue
            group = groups[group_id]
            port = next(
                (item for item in group.ports if item.mode == decision.mode),
                None,
            )
            if port is None or float(port.lower_bound) <= 0.0:
                continue
            applied_magnitude = max(float(decision.fraction), 0.0) * max(
                float(port.upper_bound),
                0.0,
            )
            if applied_magnitude + 1.0e-9 >= float(port.lower_bound):
                continue
            decisions[group_id] = replace(
                decision,
                mode="IDLE",
                fraction=0.0,
                mode_index=0,
            )
            interventions.append(
                self._intervention(
                    group_id,
                    "typed_minimum_power_unavailable",
                    decision.mode,
                    "IDLE",
                    decision.fraction,
                    0.0,
                )
            )

    def _enforce_ev_service(
        self,
        snapshot: InterfaceSnapshot,
        agent_id: str,
        groups: Mapping[str, ActionGroupInstance],
        decisions: Dict[str, ActionDecision],
        interventions: list[Mapping[str, object]],
    ) -> Mapping[str, tuple[float, float]]:
        """Reserve the local input power needed to remain on the EV schedule."""

        if not self.enforce_ev_service:
            return {}
        floors: dict[str, tuple[float, float]] = {}
        for group_id, decision in tuple(decisions.items()):
            group = groups[group_id]
            if group.group_type != "ev_session":
                continue
            values = {
                part.observation_id: float(part.values[0])
                for part in snapshot.parts_for(agent_id)
                if part.sensor_id == group.module_id
                and part.valid
                and len(part.values) == 1
                and np.isfinite(float(part.values[0]))
            }
            connected = values.get("connected_state", 0.0) > 0.5
            hours_until_departure = values.get("hours_until_departure", -1.0)
            required_average_power = values.get(
                "required_average_power_kw",
                values.get("avg_power_to_departure_kw", 0.0),
            )
            if not connected:
                continue
            if self.enforce_ev_discharge_reserve:
                self._limit_ev_discharge_to_service_reserve(
                    group,
                    decision,
                    values,
                    decisions,
                    interventions,
                )
            decision = decisions[group_id]
            charge_port = next(
                (port for port in group.ports if port.mode == "CHARGE_EV" and port.valid),
                None,
            )
            if charge_port is None:
                continue
            available_power = (
                max(group.max_charge_power_kw, 0.0)
                * max(float(charge_port.upper_bound), 0.0)
            )
            if available_power <= 1.0e-9:
                continue
            efficiency = values.get(
                "charge_efficiency_at_max_ratio",
                values.get("charger_efficiency_ratio", 1.0),
            )
            efficiency = float(np.clip(efficiency, 1.0e-3, 1.0))
            requested_floor = self._ev_service_requested_floor(
                values,
                available_power=available_power,
                efficiency=efficiency,
                hours_until_departure=hours_until_departure,
                required_average_power=required_average_power,
            )
            if requested_floor <= 1.0e-9:
                continue
            minimum_power = min(max(requested_floor, 0.0), available_power)
            minimum_fraction = minimum_power / available_power
            current_port = next(
                (port for port in group.ports if port.mode == decision.mode),
                None,
            )
            current_power = (
                max(group.max_charge_power_kw, 0.0)
                * max(float(current_port.upper_bound), 0.0)
                * max(float(decision.fraction), 0.0)
                if decision.mode == "CHARGE_EV" and current_port is not None
                else 0.0
            )
            floors[group_id] = (minimum_power, hours_until_departure)
            if current_power + 1.0e-9 < minimum_power:
                mode_index = next(
                    (
                        index
                        for index, port in enumerate(group.ports)
                        if port.mode == "CHARGE_EV"
                    ),
                    0,
                )
                decisions[group_id] = replace(
                    decision,
                    mode="CHARGE_EV",
                    fraction=minimum_fraction,
                    mode_index=mode_index,
                )
                interventions.append(
                    self._intervention(
                        group_id,
                        "ev_service_minimum_charge",
                        decision.mode,
                        "CHARGE_EV",
                        decision.fraction,
                        minimum_fraction,
                    )
                )
            if requested_floor > available_power + 1.0e-9:
                interventions.append(
                    self._intervention(
                        group_id,
                        "ev_service_capacity_limited",
                        "CHARGE_EV",
                        "CHARGE_EV",
                        requested_floor / max(available_power, 1.0e-9),
                        1.0,
                    )
                )
        return floors

    def _limit_ev_discharge_to_service_reserve(
        self,
        group: ActionGroupInstance,
        decision: ActionDecision,
        values: Mapping[str, float],
        decisions: Dict[str, ActionDecision],
        interventions: list[Mapping[str, object]],
    ) -> None:
        """Keep the post-action EV SoC above its declared service reserve.

        The previous guard only noticed an EV that was already below its
        schedule before the action. A large, physically valid discharge could
        therefore cross the reserve in one time step. This bound is expressed
        in grid-side power and includes the discharge efficiency.
        """

        if decision.mode != "DISCHARGE_EV":
            return
        required = (
            values.get("connected_ev_soc"),
            values.get("connected_ev_required_soc_departure"),
            values.get("connected_ev_battery_capacity_kwh"),
        )
        if any(
            value is None or not np.isfinite(float(value))
            for value in required
        ) or float(required[2]) <= 0.0:
            self._set_idle(
                group,
                decision,
                decisions,
                interventions,
                "ev_service_reserve_unknown",
            )
            return

        soc, required_soc, capacity_kwh = (float(value) for value in required)
        minimum_soc = max(float(values.get("connected_ev_soc_min_ratio", 0.0)), 0.0)
        service_floor = max(
            minimum_soc,
            float(required_soc) - self.ev_service_tolerance_ratio
            + self.ev_v2g_reserve_margin_ratio,
        )
        service_floor = float(np.clip(service_floor, 0.0, 1.0))
        surplus_battery_kwh = max(soc - service_floor, 0.0) * capacity_kwh
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
        step_hours = self.seconds_per_time_step / 3600.0
        maximum_grid_power_kw = (
            surplus_battery_kwh * discharge_efficiency
            / max(step_hours, 1.0e-9)
        )
        port = next(
            (item for item in group.ports if item.mode == "DISCHARGE_EV"),
            None,
        )
        available_power_kw = (
            max(float(group.max_discharge_power_kw), 0.0)
            * (0.0 if port is None else max(float(port.upper_bound), 0.0))
        )
        maximum_fraction = float(
            np.clip(
                maximum_grid_power_kw / max(available_power_kw, 1.0e-9),
                0.0,
                1.0,
            )
        )
        if maximum_fraction <= 1.0e-9:
            self._set_idle(
                group,
                decision,
                decisions,
                interventions,
                "ev_service_discharge_reserve",
            )
            return
        if decision.fraction <= maximum_fraction + 1.0e-9:
            return
        decisions[group.group_id] = replace(
            decision,
            fraction=maximum_fraction,
        )
        interventions.append(
            self._intervention(
                group.group_id,
                "ev_service_discharge_reserve",
                decision.mode,
                decision.mode,
                decision.fraction,
                maximum_fraction,
            )
        )

    def _enforce_ev_economic_guard(
        self,
        snapshot: InterfaceSnapshot,
        agent_id: str,
        groups: Mapping[str, ActionGroupInstance],
        decisions: Dict[str, ActionDecision],
        interventions: list[Mapping[str, object]],
    ) -> None:
        """Reject locally unprofitable or unusable EV discharges.

        This is a causal validity guard, not a community optimizer: it only
        uses observations available to the local actor and never allocates
        energy between agents. Flexible charging is deliberately excluded
        from the demand budget so one EV cannot justify discharging merely
        because another local EV is charging in the same decision.
        """

        if not self.enforce_ev_economic_guard:
            return
        parts = snapshot.parts_for(agent_id)
        current_price = self._current_price(parts)
        local_demand_kw = self._local_inflexible_demand(parts)
        other_discharge_kw = sum(
            self._decision_power_kw(groups[group_id], decision, "discharge")
            for group_id, decision in decisions.items()
            if groups[group_id].group_type != "ev_session"
        )
        remaining_local_demand_kw = max(
            local_demand_kw - other_discharge_kw,
            0.0,
        )
        accepted_ev_discharge_kw = 0.0

        for group_id in sorted(decisions):
            decision = decisions[group_id]
            group = groups[group_id]
            if group.group_type != "ev_session" or decision.mode != "DISCHARGE_EV":
                continue
            values = self._module_scalar_values(parts, group.module_id)
            hours_until_departure = float(values.get("hours_until_departure", -1.0))
            if (
                not np.isfinite(hours_until_departure)
                or hours_until_departure < 0.0
            ):
                self._set_idle(
                    group,
                    decision,
                    decisions,
                    interventions,
                    "ev_v2g_schedule_unknown",
                )
                continue
            future_prices = self._future_prices(parts, hours_until_departure)
            if current_price is None or not future_prices:
                self._set_idle(
                    group,
                    decision,
                    decisions,
                    interventions,
                    "ev_v2g_price_unknown",
                )
                continue
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
            replacement_cost = min(future_prices) / max(
                charge_efficiency * discharge_efficiency,
                1.0e-3,
            )
            net_margin = (
                float(current_price) * self.ev_v2g_avoided_import_value_ratio
                - replacement_cost
                - self.ev_v2g_degradation_cost_eur_per_kwh
            )
            if (
                net_margin + 1.0e-9
                < self.ev_v2g_minimum_profit_margin_eur_per_kwh
            ):
                self._set_idle(
                    group,
                    decision,
                    decisions,
                    interventions,
                    "ev_v2g_unprofitable",
                )
                continue

            requested_power_kw = self._decision_power_kw(
                group,
                decision,
                "discharge",
            )
            if not self.ev_v2g_require_local_demand:
                accepted_ev_discharge_kw += requested_power_kw
                continue
            available_demand_kw = max(
                remaining_local_demand_kw - accepted_ev_discharge_kw,
                0.0,
            )
            if available_demand_kw <= 1.0e-9:
                self._set_idle(
                    group,
                    decision,
                    decisions,
                    interventions,
                    "ev_v2g_no_local_demand",
                )
                continue
            if requested_power_kw > available_demand_kw + 1.0e-9:
                updated_fraction = decision.fraction * (
                    available_demand_kw / max(requested_power_kw, 1.0e-9)
                )
                decisions[group_id] = replace(
                    decision,
                    fraction=updated_fraction,
                )
                interventions.append(
                    self._intervention(
                        group_id,
                        "ev_v2g_local_demand_cap",
                        decision.mode,
                        decision.mode,
                        decision.fraction,
                        updated_fraction,
                    )
                )
                accepted_ev_discharge_kw += available_demand_kw
            else:
                accepted_ev_discharge_kw += requested_power_kw

    @staticmethod
    def _module_scalar_values(
        parts: Sequence[ObservationPart],
        module_id: str,
    ) -> Mapping[str, float]:
        return {
            part.observation_id: float(part.values[0])
            for part in parts
            if part.sensor_id == module_id
            and part.valid
            and len(part.values) == 1
            and np.isfinite(float(part.values[0]))
        }

    @staticmethod
    def _current_price(parts: Sequence[ObservationPart]) -> Optional[float]:
        values = [
            float(part.values[0])
            for part in parts
            if part.semantic_type == "market_price"
            and part.valid
            and len(part.values) == 1
            and np.isfinite(float(part.values[0]))
        ]
        return values[0] if values else None

    def _local_inflexible_demand(
        self,
        parts: Sequence[ObservationPart],
    ) -> float:
        """Return local demand that cannot be created by flexible actions.

        ``net_power_kw`` may already contain the previous EV or stationary
        storage action. Require explicit non-shiftable load (with optional PV)
        so a charger cannot create its own apparent demand. Missing evidence
        fails closed instead of treating flexible net load as useful demand.
        """

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

        step_hours = self.seconds_per_time_step / 3600.0
        load_power_kw = value(("non_shiftable_load_power_kw",))
        if load_power_kw is None:
            load_energy_kwh = value(
                ("non_shiftable_load", "load_energy_kwh_step")
            )
            if load_energy_kwh is not None:
                load_power_kw = load_energy_kwh / max(step_hours, 1.0e-9)

        pv_power_kw = value(("pv_power_kw",))
        if pv_power_kw is None:
            pv_energy_kwh = value(
                ("solar_generation", "pv_energy_kwh_step")
            )
            if pv_energy_kwh is not None:
                pv_power_kw = pv_energy_kwh / max(step_hours, 1.0e-9)

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
            price = float(part.values[0])
            if (
                np.isfinite(price)
                and price >= 0.0
                and (
                    hours_until_departure < 0.0
                    or horizon_hours <= hours_until_departure + 1.0e-9
                )
            ):
                values.append((horizon_hours, price))
        return tuple(price for _horizon, price in sorted(values))

    @staticmethod
    def _decision_power_kw(
        group: ActionGroupInstance,
        decision: ActionDecision,
        direction: str,
    ) -> float:
        expected = (
            decision.mode.startswith("CHARGE_")
            if direction == "charge"
            else decision.mode.startswith("DISCHARGE_")
        )
        if not expected:
            return 0.0
        port = next(
            (item for item in group.ports if item.mode == decision.mode),
            None,
        )
        rated = (
            group.max_charge_power_kw
            if direction == "charge"
            else group.max_discharge_power_kw
        )
        return (
            max(float(rated), 0.0)
            * max(float(decision.fraction), 0.0)
            * (0.0 if port is None else max(float(port.upper_bound), 0.0))
        )

    def _set_idle(
        self,
        group: ActionGroupInstance,
        decision: ActionDecision,
        decisions: Dict[str, ActionDecision],
        interventions: list[Mapping[str, object]],
        reason: str,
    ) -> None:
        decisions[group.group_id] = replace(
            decision,
            mode="IDLE",
            fraction=0.0,
            mode_index=0,
        )
        interventions.append(
            self._intervention(
                group.group_id,
                reason,
                decision.mode,
                "IDLE",
                decision.fraction,
                0.0,
            )
        )

    def _ev_service_requested_floor(
        self,
        values: Mapping[str, float],
        *,
        available_power: float,
        efficiency: float,
        hours_until_departure: float,
        required_average_power: float,
    ) -> float:
        soc = values.get("connected_ev_soc")
        required_soc = values.get("connected_ev_required_soc_departure")
        capacity_kwh = values.get("connected_ev_battery_capacity_kwh")
        complete_state = (
            soc is not None
            and required_soc is not None
            and capacity_kwh is not None
            and np.isfinite(soc)
            and np.isfinite(required_soc)
            and np.isfinite(capacity_kwh)
            and capacity_kwh > 0.0
        )
        if self.ev_service_strategy in {"minimum_average", "just_in_time"}:
            if (
                complete_state
                and hours_until_departure >= 0.0
            ):
                target_soc = max(
                    float(required_soc) - self.ev_service_tolerance_ratio,
                    0.0,
                )
                energy_needed_kwh = (
                    max(target_soc - float(soc), 0.0) * float(capacity_kwh)
                )
                if energy_needed_kwh <= 1.0e-9:
                    return 0.0
                if hours_until_departure <= 0.0:
                    return float(available_power)
                if self.ev_service_strategy == "minimum_average":
                    return (
                        energy_needed_kwh
                        / max(efficiency * hours_until_departure, 1.0e-9)
                        * (1.0 + self.ev_service_margin_ratio)
                    )
                step_hours = self.seconds_per_time_step / 3600.0
                buffer_hours = self.ev_service_jit_buffer_seconds / 3600.0
                future_hours = max(
                    float(hours_until_departure) - step_hours - buffer_hours,
                    0.0,
                )
                future_delivery_kwh = (
                    max(float(available_power), 0.0)
                    * efficiency
                    * future_hours
                )
                required_now_kwh = max(
                    energy_needed_kwh - future_delivery_kwh,
                    0.0,
                )
                requested = required_now_kwh / max(
                    efficiency * step_hours,
                    1.0e-9,
                )
                minimum_average = (
                    energy_needed_kwh
                    / max(efficiency * hours_until_departure, 1.0e-9)
                )
                smoothed_floor = (
                    minimum_average
                    * self.ev_service_jit_minimum_average_fraction
                )
                return max(requested, smoothed_floor) * (
                    1.0 + self.ev_service_margin_ratio
                )

        if hours_until_departure <= 0.0 or required_average_power <= 1.0e-9:
            return 0.0
        return (
            required_average_power
            / efficiency
            * (1.0 + self.ev_service_margin_ratio)
        )

    def _enforce_deferrable_must_start(
        self,
        snapshot: InterfaceSnapshot,
        agent_id: str,
        groups: Mapping[str, ActionGroupInstance],
        decisions: Dict[str, ActionDecision],
        interventions: list[Mapping[str, object]],
    ) -> Mapping[str, tuple[float, float]]:
        """Force due cycles and expose their indivisible headroom requests.

        The returned deadline is time-to-latest-start, rather than the final
        cycle deadline.  This lets the joint projector order an imminent
        binary START alongside EV service floors without silently giving all
        EV reservations precedence.
        """

        floors: dict[str, tuple[float, float]] = {}
        for group_id, decision in tuple(decisions.items()):
            group = groups[group_id]
            if group.group_type != "deferrable":
                continue
            module_parts = [
                part
                for part in snapshot.parts_for(agent_id)
                if part.sensor_id == group.module_id
            ]
            if not module_parts:
                continue
            values = {
                feature: value
                for part in module_parts
                if part.valid
                for feature, value in zip(part.feature_names, part.values)
            }
            slack_steps = values.get("slack_steps", 1.0)
            service_margin_steps = (
                self.deferrable_service_margin_seconds
                / max(self.seconds_per_time_step, 1.0e-9)
            )
            must_start = (
                values.get("pending", 0.0) > 0.5
                and values.get("can_start", 0.0) > 0.5
                and slack_steps <= service_margin_steps
            )
            start_port = next((port for port in group.ports if port.mode == "START" and port.valid), None)
            if must_start and start_port is not None and decision.mode != "START":
                decisions[group_id] = replace(decision, mode="START", fraction=1.0, mode_index=1)
                interventions.append(
                    self._intervention(
                        group_id,
                        (
                            "deferrable_service_margin_start"
                            if slack_steps > 0.0
                            else "deferrable_must_start"
                        ),
                        decision.mode,
                        "START",
                        decision.fraction,
                        1.0,
                    )
                )
            if must_start and start_port is not None:
                time_to_latest_start_hours = (
                    max(float(slack_steps), 0.0)
                    * self.seconds_per_time_step
                    / 3600.0
                )
                floors[group_id] = (
                    max(float(group.activation_power_kw), 0.0),
                    time_to_latest_start_hours,
                )
                continue
            schedule_parts = [
                part
                for part in module_parts
                if any(
                    token in part.observation_id.lower()
                    for token in ("deadline", "latest_start", "slack")
                )
            ]
            unknown_deadline = bool(schedule_parts) and not any(
                part.valid for part in schedule_parts
            )
            urgent_without_deadline = (
                values.get("must_run", 0.0) > 0.5
                and values.get("pending", 0.0) > 0.5
                and values.get("can_start", 0.0) > 0.5
                and values.get("running", 0.0) <= 0.5
                and values.get("last_start_requested", 0.0) <= 0.5
                and values.get("last_start_applied", 0.0) <= 0.5
                and unknown_deadline
            )
            if urgent_without_deadline and start_port is not None:
                decisions[group_id] = replace(
                    decision,
                    mode="START",
                    fraction=1.0,
                    mode_index=1,
                )
                interventions.append(
                    self._intervention(
                        group_id,
                        "required_deferrable_unknown_deadline",
                        decision.mode,
                        "START",
                        decision.fraction,
                        1.0,
                    )
                )
                floors[group_id] = (
                    max(float(group.activation_power_kw), 0.0),
                    0.0,
                )
        return floors

    def _scale_direction(
        self,
        snapshot: InterfaceSnapshot,
        agent_id: str,
        groups: Mapping[str, ActionGroupInstance],
        decisions: Dict[str, ActionDecision],
        interventions: list[Mapping[str, object]],
        *,
        direction: str,
        constraint_types: tuple[str, ...],
        minimum_power_by_group: Mapping[str, tuple[float, float]],
    ) -> None:
        constraints = sorted(
            (
                item
                for item in snapshot.constraints
                if item.owner_agent_id == agent_id
                and item.constraint_type in constraint_types
                and item.active
            ),
            key=lambda item: (
                constraint_types.index(item.constraint_type),
                item.constraint_id,
            ),
        )
        for constraint in constraints:
            self._scale_constraint(
                constraint,
                groups,
                decisions,
                interventions,
                direction=direction,
                minimum_power_by_group=minimum_power_by_group,
            )

    def _scale_constraint(
        self,
        constraint,
        groups: Mapping[str, ActionGroupInstance],
        decisions: Dict[str, ActionDecision],
        interventions: list[Mapping[str, object]],
        *,
        direction: str,
        minimum_power_by_group: Mapping[str, tuple[float, float]],
    ) -> None:
        if constraint is None or constraint.upper_bound is None or not np.isfinite(constraint.upper_bound):
            return
        member_ids = set(constraint.member_group_ids)
        coefficients = dict(constraint.member_group_coefficients)
        selected = []
        total_power = 0.0
        for group_id, decision in decisions.items():
            if member_ids and group_id not in member_ids:
                continue
            is_start = direction == "charge" and decision.mode == "START"
            expected = is_start or (
                decision.mode.startswith("CHARGE_")
                if direction == "charge"
                else decision.mode.startswith("DISCHARGE_")
            )
            if not expected:
                continue
            group = groups[group_id]
            rated = (
                group.activation_power_kw
                if is_start
                else (
                    group.max_charge_power_kw
                    if direction == "charge"
                    else group.max_discharge_power_kw
                )
            )
            port = next((item for item in group.ports if item.mode == decision.mode), None)
            available_fraction = (
                1.0
                if is_start and port is not None and port.valid
                else (0.0 if port is None else max(float(port.upper_bound), 0.0))
            )
            power = (
                max(rated, 0.0)
                * (1.0 if is_start else max(decision.fraction, 0.0))
                * available_fraction
            )
            coefficient = max(float(coefficients.get(group_id, 1.0)), 0.0)
            if coefficient <= 0.0:
                continue
            total_power += power * coefficient
            selected.append(
                (
                    group_id,
                    decision,
                    rated,
                    available_fraction,
                    coefficient,
                    is_start,
                )
            )
        # Runtime headroom describes the service state observed before the
        # next command.  A small configured reserve absorbs changes in the
        # uncontrollable base load between observation and execution, so the
        # downstream adapter does not need to clip otherwise feasible-looking
        # commands.  The default is zero for backward compatibility.
        limit = max(
            float(constraint.upper_bound) - self.headroom_reserve_kw,
            0.0,
        )
        if total_power <= limit + 1.0e-9 or total_power <= 0.0:
            return
        allocated_floors: dict[str, float] = {}
        remaining = limit
        current_power = {
            group_id: max(rated, 0.0)
            * (1.0 if is_start else max(decision.fraction, 0.0))
            * available_fraction
            for (
                group_id,
                decision,
                rated,
                available_fraction,
                _coefficient,
                is_start,
            ) in selected
        }
        selected_ids = set(current_power)
        for group_id, (requested, _deadline) in sorted(
            minimum_power_by_group.items(),
            key=lambda item: (item[1][1], item[0]),
        ):
            if group_id not in selected_ids:
                continue
            coefficient = max(float(coefficients.get(group_id, 1.0)), 0.0)
            requested_group_power = min(
                max(float(requested), 0.0),
                current_power[group_id],
            )
            requested_resource = requested_group_power * coefficient
            group = groups[group_id]
            is_binary_start = decisions[group_id].mode == "START"
            if is_binary_start:
                if requested_resource > remaining + 1.0e-9:
                    decision = decisions[group_id]
                    decisions[group_id] = replace(
                        decision,
                        mode="IDLE",
                        fraction=0.0,
                        mode_index=0,
                    )
                    current_power[group_id] = 0.0
                    interventions.append(
                        self._intervention(
                            group_id,
                            "deferrable_headroom_limited",
                            "START",
                            "IDLE",
                            decision.fraction,
                            0.0,
                        )
                    )
                    continue
                allocated_floors[group_id] = requested_group_power
                remaining = max(remaining - requested_resource, 0.0)
                continue
            allocated_resource = min(requested_resource, remaining)
            allocated_group_power = allocated_resource / max(
                coefficient, 1.0e-9
            )
            allocated_floors[group_id] = allocated_group_power
            remaining = max(remaining - allocated_resource, 0.0)
            if allocated_resource + 1.0e-9 < requested_resource:
                decision = decisions[group_id]
                port = next(
                    (item for item in group.ports if item.mode == decision.mode),
                    None,
                )
                capacity = (
                    max(group.max_charge_power_kw, 0.0)
                    * (0.0 if port is None else max(float(port.upper_bound), 0.0))
                )
                interventions.append(
                    self._intervention(
                        group_id,
                        "ev_service_headroom_limited",
                        decision.mode,
                        decision.mode,
                        decision.fraction,
                        allocated_group_power / max(capacity, 1.0e-9),
                    )
                )

        # START is a binary demand, not a fractionally scalable charging
        # command.  Admit it only when the remaining service headroom can
        # carry its full first-step power; otherwise defer it safely.
        for (
            group_id,
            decision,
            _rated,
            _available_fraction,
            coefficient,
            is_start,
        ) in sorted(selected, key=lambda item: item[0]):
            if not is_start:
                continue
            if group_id in allocated_floors:
                continue
            requested_resource = current_power[group_id] * coefficient
            if requested_resource <= remaining + 1.0e-9:
                allocated_floors[group_id] = current_power[group_id]
                remaining = max(remaining - requested_resource, 0.0)
                continue
            decisions[group_id] = replace(
                decision,
                mode="IDLE",
                fraction=0.0,
                mode_index=0,
            )
            current_power[group_id] = 0.0
            interventions.append(
                self._intervention(
                    group_id,
                    "deferrable_headroom_limited",
                    "START",
                    "IDLE",
                    decision.fraction,
                    0.0,
                )
            )

        flexible_total = 0.0
        requested_power: dict[str, float] = {}
        for (
            group_id,
            decision,
            rated,
            available_fraction,
            coefficient,
            is_start,
        ) in selected:
            power = current_power[group_id]
            requested_power[group_id] = power
            if is_start:
                continue
            flexible_total += (
                max(power - allocated_floors.get(group_id, 0.0), 0.0)
                * coefficient
            )
        flexible_scale = (
            min(remaining / flexible_total, 1.0)
            if flexible_total > 1.0e-9
            else 0.0
        )
        for (
            group_id,
            decision,
            rated,
            available_fraction,
            _coefficient,
            is_start,
        ) in selected:
            if is_start:
                continue
            floor = allocated_floors.get(group_id, 0.0)
            flexible = max(requested_power[group_id] - floor, 0.0)
            updated_power = floor + flexible * flexible_scale
            capacity = max(rated, 0.0) * available_fraction
            updated = updated_power / max(capacity, 1.0e-9)
            decisions[group_id] = replace(decision, fraction=updated)
            if abs(updated - decision.fraction) > 1.0e-9:
                interventions.append(
                    self._intervention(group_id, constraint.constraint_id, decision.mode, decision.mode, decision.fraction, updated)
                )

    def assert_feasible(self, snapshot: InterfaceSnapshot, bundles: Iterable[LocalActionBundle]) -> None:
        groups = {group.group_id: group for group in snapshot.action_groups}
        for bundle in bundles:
            for constraint in (
                item
                for item in snapshot.constraints
                if item.owner_agent_id == bundle.agent_id and item.active
            ):
                if constraint.constraint_type in {
                    "charging_headroom_kw",
                    "charging_phase_headroom_kw",
                }:
                    direction = "charge"
                elif constraint.constraint_type in {
                    "export_headroom_kw",
                    "export_phase_headroom_kw",
                }:
                    direction = "discharge"
                else:
                    continue
                limit = constraint.upper_bound
                if limit is None or not np.isfinite(limit):
                    continue
                member_ids = set(constraint.member_group_ids)
                coefficients = dict(constraint.member_group_coefficients)
                total = 0.0
                for decision in bundle.decisions:
                    if member_ids and decision.group_id not in member_ids:
                        continue
                    is_start = direction == "charge" and decision.mode == "START"
                    if not is_start and (
                        (
                            direction == "charge"
                            and not decision.mode.startswith("CHARGE_")
                        )
                        or (
                            direction == "discharge"
                            and not decision.mode.startswith("DISCHARGE_")
                        )
                    ):
                        continue
                    group = groups[decision.group_id]
                    rated = (
                        group.activation_power_kw
                        if is_start
                        else (
                            group.max_charge_power_kw
                            if direction == "charge"
                            else group.max_discharge_power_kw
                        )
                    )
                    port = next(
                        (item for item in group.ports if item.mode == decision.mode),
                        None,
                    )
                    if (
                        not is_start
                        and port is not None
                        and decision.fraction * float(port.upper_bound)
                        + 1.0e-9
                        < float(port.lower_bound)
                    ):
                        raise AssertionError(
                            f"Projected {decision.mode} magnitude is below the "
                            f"typed minimum for {decision.group_id}"
                        )
                    available_fraction = (
                        1.0
                        if is_start and port is not None and port.valid
                        else (
                            0.0
                            if port is None
                            else max(float(port.upper_bound), 0.0)
                        )
                    )
                    total += (
                        rated
                        * (1.0 if is_start else decision.fraction)
                        * available_fraction
                        * max(float(coefficients.get(decision.group_id, 1.0)), 0.0)
                    )
                effective_limit = max(
                    float(limit) - self.headroom_reserve_kw,
                    0.0,
                )
                if total > effective_limit + 1.0e-6:
                    raise AssertionError(
                        f"Projected {direction} power {total} exceeds "
                        f"{bundle.agent_id} effective limit {effective_limit}"
                    )

    @staticmethod
    def _intervention(
        group_id: str,
        reason: str,
        raw_mode: str,
        final_mode: str,
        raw_fraction: float,
        final_fraction: float,
    ) -> Mapping[str, object]:
        return {
            "group_id": group_id,
            "reason": reason,
            "raw_mode": raw_mode,
            "final_mode": final_mode,
            "raw_fraction": float(raw_fraction),
            "final_fraction": float(final_fraction),
            "magnitude": abs(float(final_fraction) - float(raw_fraction)),
        }
