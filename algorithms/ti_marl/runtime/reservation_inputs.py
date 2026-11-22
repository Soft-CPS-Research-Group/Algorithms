"""Causal numerical inputs for the TI-MARL service reservation.

An interval limit is expressed at the electrical boundary in kW. Energy is the
stored (battery-side) quantity. Asset ratings, efficiencies and wiring come from the declared equipment profile.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

from algorithms.ti_marl.contracts.enums import HealthState

from .reservation_solver import ReservationUnavailable


@dataclass(frozen=True)
class PowerMode:
    name: str
    direction: int  # +1 charging, -1 discharging, 0 idle or programme start
    min_kw: float
    rating_kw: float
    current_max_kw: float
    available_now: bool = True
    forced_fraction: float | None = None
    available_later: bool = True


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float
    energy_kwh: float
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    hourly_loss: float = 0.0
    discharge_reserve_kwh: float = 0.0
    # Connector-side rating at SoC <= 90% for the Portugal EV taper.
    taper_rating_kw: float | None = None
    taper_start_soc: float = 0.9
    taper_at_full: float = 0.25


@dataclass(frozen=True)
class Equipment:
    key: str
    modes: tuple[PowerMode, ...]
    battery: Battery | None = None
    deadline_seconds: float | None = None
    target_kwh: float | None = None
    programme_kw: tuple[float, ...] = ()
    programme_step_seconds: float = 900.0
    earliest_seconds: float = 0.0
    running: bool = False
    unconfirmed_import_kw: float = 0.0
    unconfirmed_export_kw: float = 0.0
    unconfirmed_service: bool = False
    measured_request: float | None = None
    # None denotes a fully specified programme. A peak alone gives no
    # guaranteed absorption of solar power in an individual future slot.
    programme_min_kw: tuple[float, ...] | None = None
    # A measured maximum for any later slot of this already-running programme.
    running_tail_max_kw: float | None = None


@dataclass(frozen=True)
class PowerLimit:
    key: str
    coefficients: Mapping[str, float]
    # Signed power for import/export, charging gauge for shared charger limits.
    direction: str
    current_kw: float
    future_kw: tuple[float, ...] = ()
    constant: bool = False
    scope: str | None = None

    def at(self, t, offset=0.0):
        if self.constant or t <= 0:
            return self.current_kw
        index = int((t + offset) // 900)
        return self.future_kw[index] if index < len(self.future_kw) else 0.0


@dataclass(frozen=True)
class SiteInputs:
    equipment: tuple[Equipment, ...]
    limits: tuple[PowerLimit, ...]
    diagnostics: tuple[str, ...] = ()


def physical_reading(part):
    """Return a fresh and measured scalar for certifying a physical bound."""
    return (
        part.valid
        and len(part.values) == 1
        and math.isfinite(part.values[0])
        and getattr(part, "health", HealthState.HEALTHY) == HealthState.HEALTHY
        and not getattr(part, "estimated", False)
        and getattr(part, "age_seconds", 0.0) == 0
    )


def _parts_by_sensor(snapshot, agent):
    result = {}
    for part in snapshot.parts_for(agent):
        if physical_reading(part):
            result.setdefault(part.sensor_id, {})[part.observation_id] = float(part.values[0])
    return result


def _running_tail_max_kw(values, steps, dt, first_kw, peak_kw):
    """Cap each future nonnegative slot by the total energy left after this one."""
    remaining = values.get("remaining_energy_kwh")
    current = values.get("current_step_energy_kwh")
    if steps < 2 or first_kw is None or remaining is None or current is None:
        return None
    hours = dt / 3600.0
    # The two native meter readings are stored as floats. Round their
    # difference upwards before treating it as a physical upper bound.
    slack = 1e-6 * max(1.0, abs(remaining), abs(current), peak_kw * hours)
    if (
        not 0 <= current <= remaining
        or abs(current - first_kw * hours) > slack
        or remaining > steps * peak_kw * hours + slack
    ):
        return None
    upper = (remaining - current + slack) / hours
    return upper if upper < peak_kw else None


def _profiles(group, values, dt, tolerance, diagnostics):
    physics = dict(group.reservation_profile)
    charge = group.max_charge_power_kw
    discharge = group.max_discharge_power_kw
    modes = []
    for port in group.ports:
        direction = (
            1
            if port.mode.startswith("CHARGE_")
            else -1
            if port.mode.startswith("DISCHARGE_")
            else 0
        )
        scale = charge if direction == 1 else discharge if direction == -1 else 0.0
        forced = group.forced_mode
        available = port.valid and (group.enabled or port.mode == "IDLE")
        if forced is not None:
            available = available and port.mode == forced
        modes.append(
            PowerMode(
                port.mode,
                direction,
                scale * port.lower_bound,
                scale,
                scale * port.upper_bound,
                available,
                group.forced_fraction if forced == port.mode else None,
                group.enabled
                and forced is None
                and all(
                    reason in {"runtime_bound_below_typed_minimum", "unknown_or_zero_runtime_bound"}
                    for reason in port.invalid_reasons
                ),
            )
        )
    kwargs = {}
    is_ev = group.group_type in {"charger", "ev_charger"} or any("EV" in m.name for m in modes)
    if group.group_type == "deferrable":
        pending = values.get("pending", 0.0) > 0.5
        running = values.get("running", 0.0) > 0.5
        if pending or running:
            steps = int(
                values.get("remaining_duration_steps" if running else "cycle_duration_steps", 0.0)
            )
            peak = values.get("cycle_peak_power_kw", 0.0)
            until = values.get("hours_until_deadline")
            if steps < 1 or until is None or peak <= 0:
                raise ReservationUnavailable(f"{group.group_id}: incomplete active programme")
            # The public interface exposes the first power and a peak, rather than
            # the complete shape. An upper envelope protects each occupied slot.
            first_key = "current_step_power_kw" if running else "start_power_kw"
            # SoftCPS returns zero preview power outside the start window.
            # There is then no measurement of the first sample of a future start.
            first = values.get(first_key) if running or values.get("can_start", 0) > 0.5 else None
            # SoftCPS names the last occupied interval as the deadline.
            # The reservation uses the exclusive end of that interval. A
            # separately declared latest start can close the window earlier.
            finish = max(0.0, until * 3600.0) + dt
            latest = values.get("hours_until_latest_start")
            if not running and latest is not None and latest >= 0.0:
                finish = min(finish, latest * 3600.0 + steps * dt)
            kwargs.update(
                programme_kw=(peak if first is None else first,) + (peak,) * (steps - 1),
                programme_min_kw=(0.0 if first is None else first,) + (0.0,) * (steps - 1),
                programme_step_seconds=dt,
                deadline_seconds=finish,
                earliest_seconds=max(0.0, values.get("hours_until_earliest_start", 0.0) * 3600.0),
                running=running,
            )
            if running and not group.output_unconfirmed:
                kwargs["running_tail_max_kw"] = _running_tail_max_kw(values, steps, dt, first, peak)
            if steps > 1:
                diagnostics.append(f"{group.group_id}:programme_peak_envelope")
    elif is_ev:
        connected = values.get("connected_state", 0.0) > 0.5
        if (
            "connected_state" not in values
            and group.enabled
            and any(m.direction and m.available_now for m in modes)
        ):
            raise ReservationUnavailable(f"{group.group_id}: unknown connection state")
        if connected and not group.output_unconfirmed:
            keys = (
                "connected_ev_battery_capacity_kwh",
                "connected_ev_soc",
                "connected_ev_required_soc_departure",
                "hours_until_departure",
            )
            if any(k not in values or values[k] < 0.0 for k in keys):
                raise ReservationUnavailable(f"{group.group_id}: incomplete connected EV session")
            capacity, soc, target, hours = (values[k] for k in keys)
            eta = values.get(
                "charge_efficiency_at_max_ratio", values.get("charger_efficiency_ratio")
            )
            battery_rating = values.get(
                "connected_ev_battery_nominal_power_kw", physics.get("ev_battery_rating_kw")
            )
            if (
                battery_rating is None
                or eta is None
                or not 0 < eta <= 1.0
                or not {
                    "battery_efficiency",
                    "hourly_loss",
                    "ev_taper_start_soc",
                    "ev_taper_at_full",
                }
                <= physics.keys()
            ):
                raise ReservationUnavailable(
                    f"{group.group_id}: missing declared EV reservation profile"
                )
            if "connected_ev_battery_nominal_power_kw" not in values:
                diagnostics.append(f"{group.group_id}:fleet_lower_bound_on_ev_rating")
            battery_eta = math.sqrt(physics["battery_efficiency"])
            kwargs.update(
                battery=Battery(
                    capacity,
                    soc * capacity,
                    eta * battery_eta,
                    values.get("discharge_efficiency_at_max_ratio", eta) * battery_eta,
                    physics["hourly_loss"],
                    capacity * max(0.0, values.get("connected_ev_soc_min_ratio", 0.0)),
                    battery_rating / eta,
                    physics["ev_taper_start_soc"],
                    physics["ev_taper_at_full"],
                ),
                deadline_seconds=hours * 3600.0,
                target_kwh=max(0.0, target - tolerance) * capacity,
            )
    elif any(m.direction for m in modes) and not group.output_unconfirmed:
        capacity = values.get("degraded_capacity_kwh", values.get("capacity_kwh"))
        soc = values.get("electrical_storage_soc_ratio", values.get("soc"))
        if capacity is None or soc is None:
            raise ReservationUnavailable(f"{group.group_id}: missing storage capacity/SoC")
        if not {"battery_efficiency", "hourly_loss"} <= physics.keys():
            raise ReservationUnavailable(
                f"{group.group_id}: missing declared storage reservation profile"
            )
        eta = math.sqrt(physics["battery_efficiency"])
        kwargs["battery"] = Battery(
            capacity,
            soc * capacity,
            eta,
            eta,
            physics["hourly_loss"],
            capacity * values.get("soc_min_ratio", 0.1),
        )
    if group.output_unconfirmed:
        # Unknown physical output is external to the proposed IDLE. Use the
        # rating, even when the live capability of the unavailable port is zero.
        if group.group_type == "deferrable":
            if not kwargs.get("running"):
                peak = values.get("cycle_peak_power_kw")
                if peak is None or peak <= 0.0:
                    raise ReservationUnavailable(
                        f"{group.group_id}: unknown programme peak during failure"
                    )
                kwargs["unconfirmed_import_kw"] = peak
        elif not is_ev or values.get("connected_state", 1.0) > 0.5:
            if charge <= 0 and discharge <= 0:
                raise ReservationUnavailable(
                    f"{group.group_id}: unknown power rating during failure"
                )
            kwargs.update(unconfirmed_import_kw=charge, unconfirmed_export_kw=discharge)
        if is_ev and values.get("connected_state", 1.0) > 0.5:
            kwargs["unconfirmed_service"] = True
            soc = values.get("connected_ev_soc")
            target = values.get("connected_ev_required_soc_departure")
            capacity = values.get("connected_ev_battery_capacity_kwh")
            if (
                soc is not None
                and target is not None
                and capacity is not None
                and 0 <= soc <= 1
                and 0 <= target <= 1
                and capacity > 0
            ):
                kwargs["measured_request"] = max(0.0, target - tolerance - soc) * capacity
            hours = values.get("hours_until_departure")
            if hours is not None and hours >= 0.0:
                kwargs["deadline_seconds"] = hours * 3600.0
            diagnostics.append(f"{group.group_id}:service_delivery_unconfirmed")
        diagnostics.append(f"{group.group_id}:unconfirmed_physical_output")
    if kwargs.get("deadline_seconds") is not None:
        seconds = kwargs["deadline_seconds"]
        # The wire uses float32 hours. Remove sub-millisecond noise at an exact
        # declared command boundary (e.g. 1/60 h), without rounding real deadlines.
        if abs(seconds - round(seconds)) < 0.01:
            kwargs["deadline_seconds"] = float(round(seconds))
    return Equipment(group.group_id, tuple(modes), **kwargs)


def read_site_inputs(snapshot, agent, dt, tolerance, envelopes=None):
    by_sensor = _parts_by_sensor(snapshot, agent)
    diagnostics = []
    groups = snapshot.groups_for(agent)
    units = tuple(
        _profiles(g, by_sensor.get(g.module_id, {}), dt, tolerance, diagnostics) for g in groups
    )
    unit_by_key = {u.key: u for u in units}
    if envelopes is None:
        envelopes = dict(getattr(snapshot, "reservation_envelopes", {}) or {})
        envelopes.update(
            {
                p.observation_id: p.values
                for p in snapshot.parts_for(agent)
                if p.valid and p.semantic_type == "reservation_headroom" and p.unit == "kW"
            }
        )
    limits = []
    for limit in snapshot.constraints:
        if not limit.active or limit.owner_agent_id != agent:
            continue
        if limit.upper_bound is None or not math.isfinite(limit.upper_bound):
            raise ReservationUnavailable(f"{limit.constraint_id}: unavailable upper bound")
        kind = limit.constraint_type
        if kind.startswith("export"):
            direction = "export"
        elif kind.startswith("charging") or kind == "group_import_limit_kw":
            direction = "import"
        else:
            raise ReservationUnavailable(f"Unsupported active reservation constraint {kind!r}")
        members = limit.member_group_ids or tuple(unit.key for unit in units)
        coeff = dict(limit.member_group_coefficients) or dict.fromkeys(members, 1.0)
        constant = kind == "group_import_limit_kw"
        future = tuple(envelopes.get(limit.constraint_id, ()))
        if any(not math.isfinite(v) for v in future):
            raise ReservationUnavailable(f"{limit.constraint_id}: nonfinite capacity forecast")
        if not future and not constant:
            diagnostics.append(f"{limit.constraint_id}:future_flexible_headroom_reserved_at_zero")
        current = limit.upper_bound
        if limit.nameplate_kw is not None:
            phase_match = re.search(r"_phase_(L[123])_", limit.constraint_id)
            meter = f"charging_phase_{phase_match[1]}_power_kw" if phase_match else "net_power_kw"
            net = by_sensor.get("self", {}).get(meter)
            if net is None:
                raise ReservationUnavailable(
                    f"{limit.constraint_id}: missing net meter for signed limit"
                )
            previous = 0.0
            for group in groups:
                if group.group_id not in coeff or group.group_type == "deferrable":
                    continue
                power = by_sensor.get(group.module_id, {}).get("last_applied_power_kw")
                if power is None and group.output_unconfirmed:
                    # Reconstruct the worst background when the last command is
                    # also unknown. Net = background + old physical output.
                    unit = unit_by_key[group.group_id]
                    power = (
                        -unit.unconfirmed_export_kw
                        if direction == "import"
                        else unit.unconfirmed_import_kw
                    )
                if power is None:
                    raise ReservationUnavailable(
                        f"{limit.constraint_id}: missing previous power of {group.group_id}"
                    )
                previous += coeff[group.group_id] * power
            # Subtract old signed commands once before replacing them. Legacy
            # directional headroom can otherwise allow too much import just
            # after a battery discharge, or too much export after charging.
            sign = 1.0 if direction == "import" else -1.0
            current = limit.nameplate_kw - sign * (net - previous)
        scope = "group" if constant else "phase" if "_phase_" in kind else "site"
        limits.append(
            PowerLimit(limit.constraint_id, coeff, direction, current, future, constant, scope)
        )
    return SiteInputs(units, tuple(limits), tuple(diagnostics))
