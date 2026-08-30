"""Deployment-causal stationary-storage targets for the TI-MARL actor.

The learned actor remains the runtime controller.  This module only supplies
an auditable auxiliary training signal built from typed inputs that are
available at deployment: storage state/capability, local net exchange, current
tariff, and declared tariff forecasts.  It never reads future simulator state
and never bypasses the actor or local feasibility projector.
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


@dataclass(frozen=True)
class StoragePlanningTarget:
    """One supervised target for a stationary-storage action group."""

    agent_id: str
    group_id: str
    decision: ActionDecision
    reason: str
    current_price: float
    future_price_min: float
    future_price_max: float
    storage_soc: float


class CausalStoragePlanner:
    """Label safe price/PV opportunities from typed deployment evidence.

    The planner is intentionally myopic and conservative.  It stores local PV
    surplus, charges when the present tariff is materially below a declared
    future tariff, discharges only against positive local demand when the
    present tariff is materially above the forecast, and otherwise labels
    ``IDLE``.  Local bounds and the feasibility layer remain authoritative.
    """

    def __init__(
        self,
        *,
        charge_fraction: float = 0.55,
        discharge_fraction: float = 0.45,
        minimum_soc_ratio: float = 0.20,
        maximum_soc_ratio: float = 0.90,
        price_tie_tolerance: float = 1.0e-6,
        minimum_price_spread: float = 0.01,
        pv_surplus_threshold_kw: float = 0.25,
        import_threshold_kw: float = 0.25,
    ) -> None:
        for name, value in {
            "charge_fraction": charge_fraction,
            "discharge_fraction": discharge_fraction,
        }.items():
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"Storage planning {name} must lie in (0, 1)")
        if not 0.0 <= float(minimum_soc_ratio) < float(maximum_soc_ratio) <= 1.0:
            raise ValueError(
                "Storage planning SoC ratios must satisfy 0 <= min < max <= 1"
            )
        if any(
            float(value) < 0.0
            for value in (
                price_tie_tolerance,
                minimum_price_spread,
                pv_surplus_threshold_kw,
                import_threshold_kw,
            )
        ):
            raise ValueError("Storage planning tolerances must be non-negative")
        self.charge_fraction = float(charge_fraction)
        self.discharge_fraction = float(discharge_fraction)
        self.minimum_soc_ratio = float(minimum_soc_ratio)
        self.maximum_soc_ratio = float(maximum_soc_ratio)
        self.price_tie_tolerance = float(price_tie_tolerance)
        self.minimum_price_spread = float(minimum_price_spread)
        self.pv_surplus_threshold_kw = float(pv_surplus_threshold_kw)
        self.import_threshold_kw = float(import_threshold_kw)

    def targets(
        self,
        snapshot: InterfaceSnapshot,
        *,
        seconds_per_time_step: float,
    ) -> Tuple[StoragePlanningTarget, ...]:
        del seconds_per_time_step
        results: list[StoragePlanningTarget] = []
        for agent_id in snapshot.agent_ids:
            parts = snapshot.parts_for(agent_id)
            for group in snapshot.groups_for(agent_id):
                if group.group_type != "stationary_storage":
                    continue
                target = self._target_for_group(agent_id, group, parts)
                if target is not None:
                    results.append(target)
        return tuple(results)

    def configuration(self) -> Mapping[str, float | str]:
        return {
            "kind": "causal_local_storage_opportunity_v1",
            "charge_fraction": self.charge_fraction,
            "discharge_fraction": self.discharge_fraction,
            "minimum_soc_ratio": self.minimum_soc_ratio,
            "maximum_soc_ratio": self.maximum_soc_ratio,
            "price_tie_tolerance": self.price_tie_tolerance,
            "minimum_price_spread": self.minimum_price_spread,
            "pv_surplus_threshold_kw": self.pv_surplus_threshold_kw,
            "import_threshold_kw": self.import_threshold_kw,
        }

    def _target_for_group(
        self,
        agent_id: str,
        group: ActionGroupInstance,
        parts: Sequence[ObservationPart],
    ) -> Optional[StoragePlanningTarget]:
        if not group.enabled:
            return None
        modes = tuple(port.mode for port in group.ports)
        if "IDLE" not in modes:
            return None
        charge_port = self._valid_port(group, "CHARGE_STATIONARY")
        discharge_port = self._valid_port(group, "DISCHARGE_STATIONARY")
        values = self._group_values(group, parts)
        soc = values.get("soc")
        if soc is None:
            soc = values.get("state_of_charge")
        if soc is None or not np.isfinite(soc):
            return None
        soc = float(soc)

        current_price = self._current_price(parts)
        future_prices = self._future_prices(parts)
        future_min = min(future_prices) if future_prices else float("nan")
        future_max = max(future_prices) if future_prices else float("nan")
        local_net_demand = self._local_net_demand(parts)

        mode = "IDLE"
        fraction = 0.0
        reason = "no_material_opportunity"
        if (
            charge_port is not None
            and soc < self.maximum_soc_ratio - 1.0e-9
            and local_net_demand < -self.pv_surplus_threshold_kw
        ):
            available_power = self._available_power(
                group,
                charge_port,
                values,
                charge=True,
            )
            if available_power > 1.0e-9:
                mode = "CHARGE_STATIONARY"
                fraction = min(
                    self.charge_fraction,
                    -local_net_demand / available_power,
                )
                reason = "local_pv_surplus"
        elif (
            charge_port is not None
            and soc < self.maximum_soc_ratio - 1.0e-9
            and current_price is not None
            and future_prices
            and future_max - current_price + self.price_tie_tolerance
            >= self.minimum_price_spread
            and current_price <= future_min + self.price_tie_tolerance
        ):
            mode = "CHARGE_STATIONARY"
            fraction = self.charge_fraction
            reason = "cheapest_forecast_opportunity"
        elif (
            discharge_port is not None
            and soc > self.minimum_soc_ratio + 1.0e-9
            and local_net_demand > self.import_threshold_kw
            and current_price is not None
            and future_prices
            and current_price - future_min + self.price_tie_tolerance
            >= self.minimum_price_spread
            and current_price >= future_max - self.price_tie_tolerance
        ):
            available_power = self._available_power(
                group,
                discharge_port,
                values,
                charge=False,
            )
            if available_power > 1.0e-9:
                mode = "DISCHARGE_STATIONARY"
                fraction = min(
                    self.discharge_fraction,
                    local_net_demand / available_power,
                )
                reason = "highest_forecast_import_offset"

        if mode != "IDLE":
            port = charge_port if mode == "CHARGE_STATIONARY" else discharge_port
            assert port is not None
            fraction = float(
                np.clip(
                    max(
                        fraction,
                        float(port.lower_bound)
                        / max(float(port.upper_bound), 1.0e-9),
                    ),
                    0.0,
                    1.0,
                )
            )
        return StoragePlanningTarget(
            agent_id=agent_id,
            group_id=group.group_id,
            decision=ActionDecision(
                group_id=group.group_id,
                mode=mode,
                fraction=fraction,
                mode_index=modes.index(mode),
            ),
            reason=reason,
            current_price=(
                float(current_price) if current_price is not None else float("nan")
            ),
            future_price_min=float(future_min),
            future_price_max=float(future_max),
            storage_soc=soc,
        )

    @staticmethod
    def _valid_port(
        group: ActionGroupInstance,
        mode: str,
    ) -> ActionPortInstance | None:
        return next(
            (port for port in group.ports if port.mode == mode and port.valid),
            None,
        )

    @staticmethod
    def _group_values(
        group: ActionGroupInstance,
        parts: Sequence[ObservationPart],
    ) -> Mapping[str, float]:
        values: dict[str, float] = {}
        for part in parts:
            if (
                part.sensor_id != group.module_id
                or not part.valid
                or len(part.values) != 1
            ):
                continue
            value = float(part.values[0])
            if np.isfinite(value):
                values[part.observation_id] = value
        return values

    @staticmethod
    def _available_power(
        group: ActionGroupInstance,
        port: ActionPortInstance,
        values: Mapping[str, float],
        *,
        charge: bool,
    ) -> float:
        group_power = (
            group.max_charge_power_kw if charge else group.max_discharge_power_kw
        )
        name = "available_charge_power_kw" if charge else "available_discharge_power_kw"
        return min(
            max(float(group_power), 0.0) * max(float(port.upper_bound), 0.0),
            max(float(values.get(name, group_power)), 0.0),
        )

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
    def _local_net_demand(parts: Sequence[ObservationPart]) -> float:
        candidates = [
            float(part.values[0])
            for part in parts
            if part.observation_id == "net_power_kw"
            and part.scope == "local"
            and part.valid
            and len(part.values) == 1
            and np.isfinite(float(part.values[0]))
        ]
        return candidates[-1] if candidates else 0.0

    @staticmethod
    def _future_prices(parts: Sequence[ObservationPart]) -> Tuple[float, ...]:
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
            if np.isfinite(value):
                values.append((horizon_hours, value))
        return tuple(value for _horizon, value in sorted(values))
