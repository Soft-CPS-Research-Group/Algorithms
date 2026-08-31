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
        price_regime_kind: str = "strict_extrema",
        forecast_mean_margin_fraction: float = 0.20,
        forecast_edge_margin_fraction: float = 0.10,
        forecast_spread_floor_ratio: float = 0.05,
        scale_price_fraction_by_opportunity: bool = False,
        minimum_price_fraction_scale: float = 0.50,
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
        if price_regime_kind not in {"strict_extrema", "relative_forecast"}:
            raise ValueError(
                "Storage planning price_regime_kind must be "
                "'strict_extrema' or 'relative_forecast'"
            )
        for name, value in {
            "forecast_mean_margin_fraction": forecast_mean_margin_fraction,
            "forecast_edge_margin_fraction": forecast_edge_margin_fraction,
            "minimum_price_fraction_scale": minimum_price_fraction_scale,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Storage planning {name} must lie in [0, 1]")
        if float(forecast_spread_floor_ratio) < 0.0:
            raise ValueError(
                "Storage planning forecast_spread_floor_ratio must be non-negative"
            )
        self.charge_fraction = float(charge_fraction)
        self.discharge_fraction = float(discharge_fraction)
        self.minimum_soc_ratio = float(minimum_soc_ratio)
        self.maximum_soc_ratio = float(maximum_soc_ratio)
        self.price_tie_tolerance = float(price_tie_tolerance)
        self.minimum_price_spread = float(minimum_price_spread)
        self.pv_surplus_threshold_kw = float(pv_surplus_threshold_kw)
        self.import_threshold_kw = float(import_threshold_kw)
        self.price_regime_kind = str(price_regime_kind)
        self.forecast_mean_margin_fraction = float(
            forecast_mean_margin_fraction
        )
        self.forecast_edge_margin_fraction = float(
            forecast_edge_margin_fraction
        )
        self.forecast_spread_floor_ratio = float(forecast_spread_floor_ratio)
        self.scale_price_fraction_by_opportunity = bool(
            scale_price_fraction_by_opportunity
        )
        self.minimum_price_fraction_scale = float(
            minimum_price_fraction_scale
        )

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

    def configuration(self) -> Mapping[str, float | str | bool]:
        return {
            "kind": "causal_local_storage_opportunity_v2",
            "charge_fraction": self.charge_fraction,
            "discharge_fraction": self.discharge_fraction,
            "minimum_soc_ratio": self.minimum_soc_ratio,
            "maximum_soc_ratio": self.maximum_soc_ratio,
            "price_tie_tolerance": self.price_tie_tolerance,
            "minimum_price_spread": self.minimum_price_spread,
            "pv_surplus_threshold_kw": self.pv_surplus_threshold_kw,
            "import_threshold_kw": self.import_threshold_kw,
            "price_regime_kind": self.price_regime_kind,
            "forecast_mean_margin_fraction": (
                self.forecast_mean_margin_fraction
            ),
            "forecast_edge_margin_fraction": (
                self.forecast_edge_margin_fraction
            ),
            "forecast_spread_floor_ratio": self.forecast_spread_floor_ratio,
            "scale_price_fraction_by_opportunity": (
                self.scale_price_fraction_by_opportunity
            ),
            "minimum_price_fraction_scale": (
                self.minimum_price_fraction_scale
            ),
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
        cheap_now, expensive_now, charge_score, discharge_score = (
            self._price_regime(current_price, future_prices)
        )

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
            and cheap_now
        ):
            mode = "CHARGE_STATIONARY"
            fraction = self._price_fraction(
                self.charge_fraction,
                charge_score,
            )
            reason = (
                "cheapest_forecast_opportunity"
                if self.price_regime_kind == "strict_extrema"
                else "cheap_relative_forecast"
            )
        elif (
            discharge_port is not None
            and soc > self.minimum_soc_ratio + 1.0e-9
            and local_net_demand > self.import_threshold_kw
            and current_price is not None
            and future_prices
            and current_price - future_min + self.price_tie_tolerance
            >= self.minimum_price_spread
            and expensive_now
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
                    self._price_fraction(
                        self.discharge_fraction,
                        discharge_score,
                    ),
                    local_net_demand / available_power,
                )
                reason = (
                    "highest_forecast_import_offset"
                    if self.price_regime_kind == "strict_extrema"
                    else "expensive_relative_import_offset"
                )

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

    def _price_regime(
        self,
        current_price: Optional[float],
        future_prices: Sequence[float],
    ) -> tuple[bool, bool, float, float]:
        """Classify the current tariff using only declared causal forecasts.

        ``strict_extrema`` preserves the original v15 behaviour.  The
        ``relative_forecast`` regime recognizes a wider cheap/expensive band
        around the forecast distribution, while the independent absolute
        ``minimum_price_spread`` gate in :meth:`_target_for_group` still
        rejects economically immaterial opportunities.
        """

        if current_price is None or not future_prices:
            return False, False, 0.0, 0.0
        current = float(current_price)
        forecast = np.asarray(tuple(future_prices), dtype=np.float64)
        future_min = float(np.min(forecast))
        future_max = float(np.max(forecast))
        future_mean = float(np.mean(forecast))
        spread = max(
            future_max - future_min,
            abs(future_mean) * self.forecast_spread_floor_ratio,
            1.0e-9,
        )
        if self.price_regime_kind == "strict_extrema":
            cheap = current <= future_min + self.price_tie_tolerance
            expensive = current >= future_max - self.price_tie_tolerance
        else:
            cheap = (
                current
                <= future_mean
                - self.forecast_mean_margin_fraction * spread
                + self.price_tie_tolerance
                or current
                <= future_min
                + self.forecast_edge_margin_fraction * spread
                + self.price_tie_tolerance
            )
            expensive = (
                current
                >= future_mean
                + self.forecast_mean_margin_fraction * spread
                - self.price_tie_tolerance
                or current
                >= future_max
                - self.forecast_edge_margin_fraction * spread
                - self.price_tie_tolerance
            )
        charge_score = float(np.clip((future_max - current) / spread, 0.0, 1.0))
        discharge_score = float(
            np.clip((current - future_min) / spread, 0.0, 1.0)
        )
        return bool(cheap), bool(expensive), charge_score, discharge_score

    def _price_fraction(self, maximum: float, opportunity_score: float) -> float:
        if not self.scale_price_fraction_by_opportunity:
            return float(maximum)
        scale = self.minimum_price_fraction_scale + (
            1.0 - self.minimum_price_fraction_scale
        ) * float(np.clip(opportunity_score, 0.0, 1.0))
        return float(maximum) * scale

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
