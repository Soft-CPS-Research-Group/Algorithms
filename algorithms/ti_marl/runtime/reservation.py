"""Conditional construction of a local action (TI-MARL §6.6).

The MILP proves feasibility of mode prefixes. The power pass freezes a
canonical integer witness, and optimises two linear objectives per parameter.
State and solver ownership are local to one physical site and decision.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from time import monotonic

import numpy as np

from algorithms.ti_marl.contracts.models import ActionDecision, LocalActionBundle
from algorithms.ti_marl.contracts.profile_registry import ACTION_PROFILES

from .reservation_inputs import SiteInputs, read_site_inputs
from .reservation_solver import LinearModel, ReservationInfeasible, ReservationUnavailable


def _mode_index(name):
    for profile in ACTION_PROFILES.values():
        modes = profile["modes"]
        if name in modes:
            return modes.index(name)
    raise ReservationUnavailable(f"Unknown typed action mode {name}")


def group_order_for(snapshot, agent_id, order_seed=0):
    """Stable-key permutation; independent of Python hash and input row order."""

    def key(group):
        material = f"{order_seed}\0{snapshot.time_step}\0{group.group_id}".encode()
        return sha256(material).digest(), group.group_id

    return tuple(g.group_id for g in sorted(snapshot.groups_for(agent_id), key=key))


def _sum_terms(target, terms, factor=1.0):
    for col, coefficient in terms.items():
        target[col] = target.get(col, 0.0) + factor * coefficient


class _UnavoidableForecastExport(ReservationInfeasible):
    """No joint schedule can absorb the projected export over the affected intervals."""

    def __init__(self, at_seconds, limit_key):
        super().__init__(f"Forecast export exceeds controllable absorption at {limit_key}")
        self.at_seconds = at_seconds
        self.limit_key = limit_key


class SiteReservation:
    @classmethod
    def from_snapshot(
        cls,
        snapshot,
        agent_id,
        *,
        step_seconds=900.0,
        order_seed=0,
        tolerance_soc=0.05,
        departure_margin_seconds=0.0,
        reserve_programmes_in_certified_window=False,
        budget_seconds=5.0,
        envelopes=None,
    ):
        inputs = read_site_inputs(snapshot, agent_id, step_seconds, tolerance_soc, envelopes)
        return cls.plan(
            inputs,
            step_seconds=step_seconds,
            departure_margin_seconds=departure_margin_seconds,
            reserve_programmes_in_certified_window=reserve_programmes_in_certified_window,
            group_order=group_order_for(snapshot, agent_id, order_seed),
            budget_seconds=budget_seconds,
            block_offset_seconds=getattr(snapshot, "clock_seconds", snapshot.timestamp_seconds)
            % 900.0,
        )

    @classmethod
    def plan(cls, inputs, *, budget_seconds=5.0, **options):
        """Prefer a full witness. Reassess before an unserviceable forecast export.

        Every retained row keeps its original physical bound. A new programme
        must finish inside the certified window. A held command or running
        programme may never be truncated to obtain a witness.
        """
        end_time = monotonic() + budget_seconds
        alert = None
        remaining = budget_seconds
        while True:
            try:
                return cls(inputs, budget_seconds=remaining, _forecast_alert=alert, **options)
            except _UnavoidableForecastExport as conflict:
                # Each retry ends before a conflict strictly inside the last horizon.
                alert = conflict
                remaining = end_time - monotonic()
                if remaining <= 0:
                    raise ReservationUnavailable(
                        "Site reservation time budget expired"
                    ) from conflict

    def __init__(
        self,
        inputs: SiteInputs,
        *,
        step_seconds=900.0,
        group_order=None,
        budget_seconds=5.0,
        block_offset_seconds=0.0,
        departure_margin_seconds=0.0,
        reserve_programmes_in_certified_window=False,
        _forecast_alert=None,
    ):
        if not 0 < step_seconds <= 86400 or not math.isfinite(step_seconds):
            raise ValueError("Invalid physical control interval")
        self.units = {g.key: g for g in inputs.equipment}
        self.group_order = tuple(sorted(self.units) if group_order is None else group_order)
        if len(self.group_order) != len(self.units) or set(self.group_order) != self.units.keys():
            raise ValueError("Invalid site construction order")
        self.limits = inputs.limits
        self.notes = inputs.diagnostics
        self.model = LinearModel(budget_seconds)
        self.modes = {}
        self.controls = {}
        self.binary = {}
        self.energy = {}
        self.starts = {}
        self.coverage = {}
        self.protected = {}
        self.early = {}
        self.deviation = {}
        self.fixed_modes = {}
        self.fixed_fractions = {}
        self.mode_cache = {}
        self.interval_cache = {}
        self.witness = None
        self.ledger = {}
        self.service_status = {}
        self.calendar = ()
        self.flow = defaultdict(dict)
        self.programmes = defaultdict(dict)
        self.block_offset = float(block_offset_seconds) % 900.0
        self.dt = step_seconds
        if (
            not math.isfinite(departure_margin_seconds)
            or not 0 <= departure_margin_seconds <= 86400
        ):
            raise ValueError("Experimental EV departure margin must be in [0, 86400] seconds")
        self.departure_margin = departure_margin_seconds
        self.early_reserve = {}
        self.edges = self._edges(step_seconds)
        self.forecast_alert = _forecast_alert
        self.reserve_programmes_in_certified_window = reserve_programmes_in_certified_window
        if _forecast_alert is not None:
            cut = _forecast_alert.at_seconds
            held = max(
                [self.dt]
                + [
                    len(u.programme_kw) * u.programme_step_seconds
                    for u in inputs.equipment
                    if u.running
                ]
            )
            if cut < held or cut not in self.edges[:-1]:
                raise ReservationInfeasible("Forecast conflict within a committed interval")
            self.edges = self.edges[: self.edges.index(cut) + 1]
        self.n = len(self.edges) - 1
        for unit in inputs.equipment:
            self._build_equipment(unit)
        forecast_rows = {}
        for limit in inputs.limits:
            self._build_limit(limit, forecast_rows)
        # Service optimisation fixes additional bounds. The diagnostic must
        # start from the original physical problem, before those priorities.
        initial_bounds = tuple(self.model.bounds) if forecast_rows else ()
        initial_rows = len(self.model.rows)
        try:
            self._prioritise()
        except ReservationInfeasible:
            if forecast_rows and self._forecast_causes_conflict(
                forecast_rows, initial_bounds, initial_rows
            ):
                raise _UnavoidableForecastExport(*min(forecast_rows.values()))
            raise
        self._original_problem = self.model.verification_problem()
        self._initial_witness = self.witness.copy()

    def verified_fallback(self, agent_id):
        """Complete action certified before sampling, independent of the failed prefix."""
        x = self._initial_witness
        if not self._original_problem.verify(x):
            raise ReservationUnavailable("Original complete witness no longer verifies")
        decisions = []
        for key in self.group_order:
            name = self._witness_mode(x, key, 0)
            col = self.controls.get((key, 0, name))
            fraction = (
                float(name == "START")
                if col is None
                else float(x[col]) / self.power_scale_kw(key, name)
            )
            decisions.append(ActionDecision(key, name, fraction, _mode_index(name)))
        return LocalActionBundle(
            agent_id, tuple(decisions), ({"reason": "reservation_supervised_fallback"},)
        )

    def _witness_mode(self, x, key, interval):
        for name in self.modes[key]:
            bit = self.binary.get((key, interval, name))
            if bit is not None and x[bit] > 0.5:
                return name
        return "IDLE"

    def _edges(self, step):
        # Retain a short first decision and every known deadline. Future
        # planning remains on quarter-hour blocks, even at a 15-second cadence.
        horizon = step
        extra = set()
        for u in self.units.values():
            deadline = u.deadline_seconds
            if deadline is not None and 0 < deadline <= 86400:
                horizon = max(horizon, deadline)
                if deadline >= step:
                    extra.add(deadline)
                    if u.target_kwh is not None and self.departure_margin:
                        early = deadline - self.departure_margin
                        if early > step:
                            extra.add(early)
            duration = len(u.programme_kw) * u.programme_step_seconds
            if duration and u.earliest_seconds <= step:
                horizon = max(horizon, min(86400.0, duration))
        return sorted(
            {0.0, step, horizon}
            | extra
            | {
                t - self.block_offset
                for t in range(900, math.ceil(horizon + self.block_offset), 900)
                if step < t - self.block_offset < horizon
            }
        )

    def _build_equipment(self, unit):
        key = unit.key
        m = self.model
        modes = {mode.name: mode for mode in unit.modes}
        if "IDLE" not in modes:
            raise ReservationUnavailable(f"{key}: missing IDLE mode")
        self.modes[key] = modes
        self.deviation[key] = m.var(0.0, 2.0)
        if unit.battery:
            b = unit.battery
            if not (
                b.capacity_kwh > 0
                and 0 <= b.energy_kwh <= b.capacity_kwh + 1e-6
                and 0 < b.charge_efficiency <= 1
                and 0 < b.discharge_efficiency <= 1
                and 0 <= b.hourly_loss < 1
            ):
                raise ReservationUnavailable(f"{key}: invalid battery profile")
            energy = min(b.energy_kwh, b.capacity_kwh)
            self.energy[key, 0] = m.var(energy, energy)
        starts = []
        for t in range(self.n):
            lo, hi = self.edges[t : t + 2]
            switches = {}
            charge, discharge = {}, {}
            for name, mode in modes.items():
                if name == "IDLE":
                    continue
                now_ok = mode.available_later if t else mode.available_now
                if name == "START":
                    duration = len(unit.programme_kw) * unit.programme_step_seconds
                    allowed = (
                        now_ok
                        and not unit.running
                        and bool(unit.programme_kw)
                        and lo >= unit.earliest_seconds
                        and lo + duration
                        <= min(unit.deadline_seconds or 0.0, self.edges[-1]) + 1e-8
                    )
                    if not allowed:
                        continue
                    bit = m.var(binary=True)
                    self.starts[key, t] = bit
                    starts.append(bit)
                    self._add_programme(unit, lo, bit)
                elif mode.direction:
                    if not unit.battery or not now_ok:
                        continue
                    if unit.deadline_seconds is not None and hi > unit.deadline_seconds + 1e-8:
                        continue
                    upper = mode.current_max_kw if t == 0 else mode.rating_kw
                    lower = mode.min_kw
                    if t == 0 and mode.forced_fraction is not None:
                        lower = upper = mode.forced_fraction * mode.current_max_kw
                    if upper <= 0.0 or upper + 1e-9 < lower:
                        continue
                    bit = m.var(binary=True)
                    power = m.var(0.0, upper)
                    m.row({power: 1.0, bit: -upper}, hi=0.0)
                    m.row({power: 1.0, bit: -lower}, lo=0.0)
                    self.controls[key, t, name] = power
                    self.flow[key, t][power] = float(mode.direction)
                    if mode.direction == 1:
                        charge[power] = 1.0
                    else:
                        discharge[power] = 1.0
                else:
                    raise ReservationUnavailable(f"{key}: unknown mode {name}")
                self.binary[key, t, name] = bit
                switches[bit] = 1.0
            m.row(switches, lo=1 if t == 0 and not modes["IDLE"].available_now else 0, hi=1.0)
            if unit.battery:
                self._energy_balance(unit, t, (hi - lo) / 3600.0, charge, discharge)
        if starts:
            m.row(dict.fromkeys(starts, 1.0), hi=1.0)
        if unit.running:
            self._add_programme(unit, 0.0, None)
        self._service(unit)

    def _add_programme(self, unit, start, bit):
        # Import needs a peak, export needs a guaranteed minimum. A block
        # that overlaps only part of a cycle has zero guaranteed absorption.
        lower = unit.programme_kw if unit.programme_min_kw is None else unit.programme_min_kw
        if len(lower) != len(unit.programme_kw) or any(
            not 0 <= lo <= hi for lo, hi in zip(lower, unit.programme_kw)
        ):
            raise ReservationUnavailable(f"{unit.key}: invalid programme power envelope")
        end = start + len(lower) * unit.programme_step_seconds
        for t, (lo, hi) in enumerate(zip(self.edges, self.edges[1:])):
            if hi <= start or lo >= end or (t == 0 and bit is None):
                # Running power is already included in the live net meter.
                continue
            first = max(0, math.floor((lo - start) / unit.programme_step_seconds))
            stop = min(len(lower), math.ceil((hi - start) / unit.programme_step_seconds))
            self.programmes["import", unit.key, t][bit] = max(unit.programme_kw[first:stop])
            self.programmes["export", unit.key, t][bit] = (
                min(lower[first:stop]) if start <= lo and hi <= end else 0.0
            )

    def _energy_balance(self, unit, t, hours, charge, discharge):
        key, b = unit.key, unit.battery
        before = self.energy[key, t]
        keep = (1.0 - b.hourly_loss) ** hours
        after = self.model.var(0.0, keep * b.capacity_kwh)
        self.energy[key, t + 1] = after
        balance = {after: 1.0, before: -keep}
        _sum_terms(balance, charge, -keep * hours * b.charge_efficiency)
        _sum_terms(balance, discharge, keep * hours / b.discharge_efficiency)
        self.model.row(balance, lo=0.0, hi=0.0)
        if b.discharge_reserve_kwh > 0 and discharge:
            reserve = {after: 1.0}
            for mode in unit.modes:
                bit = self.binary.get((key, t, mode.name))
                if bit is not None and mode.direction == -1:
                    reserve[bit] = -keep * b.discharge_reserve_kwh
            self.model.row(reserve, lo=0.0)
        if b.taper_rating_kw is not None and charge:
            rate = b.taper_rating_kw
            self.model.row(charge, hi=rate)
            slope = (1.0 - b.taper_at_full) / (1.0 - b.taper_start_soc)
            self.model.row(
                {**charge, before: slope * rate / b.capacity_kwh},
                hi=(1.0 + slope * b.taper_start_soc) * rate,
            )

    def _service(self, unit):
        deadline = unit.deadline_seconds
        if unit.unconfirmed_service:
            # Preserve measured obligation when available. Missing telemetry
            # leaves its quantity unknown, without declaring a service deficit.
            self.service_status[unit.key] = "actuation_unconfirmed"
            self.ledger[unit.key] = unit.measured_request
            return
        if unit.running:
            self.service_status[unit.key] = "running_programme"
            self.ledger[unit.key] = 1.0
            return
        if deadline is None:
            return
        if unit.battery is not None and unit.target_kwh is not None:
            requested = max(0.0, unit.target_kwh - unit.battery.energy_kwh)
        elif unit.programme_kw:
            requested = 1.0
        else:
            return
        self.ledger[unit.key] = requested
        service_status = "in_reservation"
        if deadline > self.edges[-1]:
            if not (
                self.reserve_programmes_in_certified_window
                and self.forecast_alert is not None
                and unit.programme_kw
                and any(key == unit.key for key, _ in self.starts)
            ):
                self.service_status[unit.key] = "beyond_reservation_horizon"
                return
            # Consider only complete starts constructed inside the certified horizon.
            # The original user deadline still governs priority between devices.
            service_status = "in_certified_window"
        if deadline < self.dt - 1e-8:
            # A single setpoint must be executable throughout its control
            # interval. No part of a later interval counts towards this service.
            self.service_status[unit.key] = "deadline_before_next_command"
            if requested > 0.0:
                self.protected[unit.key] = 0.0
            return
        self.service_status[unit.key] = service_status
        if unit.battery and unit.target_kwh is not None:
            end = self.edges.index(deadline)
            initial = min(unit.battery.energy_kwh, unit.target_kwh)
            deficit = max(0.0, unit.target_kwh - unit.battery.energy_kwh)
            if deficit < 1e-9:
                self.model.row({self.energy[unit.key, end]: 1.0}, lo=unit.target_kwh)
                return
            frac = self.model.var()
            self.model.row({self.energy[unit.key, end]: 1.0, frac: -deficit}, lo=initial)
        elif unit.programme_kw:
            frac = self.model.var(binary=True)
            bits = {bit: 1.0 for (key, _), bit in self.starts.items() if key == unit.key}
            self.model.row({**bits, frac: -1.0}, lo=0.0, hi=0.0)
        else:
            return
        self.coverage[unit.key] = frac

    def _build_limit(self, limit, forecast_rows):
        if limit.direction not in {"import", "export"}:
            raise ReservationUnavailable(f"Unknown limit direction {limit.direction}")
        sign = 1.0 if limit.direction == "import" else -1.0
        for t in range(self.n):
            terms = {}
            operating = 0.0
            for key, factor in limit.coefficients.items():
                if key not in self.units:
                    raise ReservationUnavailable(f"Unknown member {key} of {limit.key}")
                _sum_terms(terms, self.flow[key, t], sign * factor)
                unit = self.units[key]
                unknown = unit.unconfirmed_import_kw if sign > 0 else unit.unconfirmed_export_kw
                operating += factor * unknown
                for bit, power in self.programmes[limit.direction, key, t].items():
                    if bit is None:
                        operating += sign * factor * power
                    else:
                        terms[bit] = terms.get(bit, 0.0) + sign * factor * power
            lo, hi = self.edges[t : t + 2]
            bound = limit.at(lo, self.block_offset)
            if t == 0 and not limit.constant:
                # A held command can span a civil quarter even at subminute
                # cadence. It must fit every quarter over which it is applied.
                for boundary in range(900, math.ceil(hi + self.block_offset), 900):
                    bound = min(bound, limit.at(boundary - self.block_offset, self.block_offset))
            upper = bound - operating
            self.model.row(terms, hi=upper)
            if (
                t > 0
                and bound < 0.0
                and limit.direction == "export"
                and not limit.constant
                and limit.scope in {"site", "phase"}
            ):
                forecast_rows[len(self.model.rows) - 1] = (lo, limit.key)

    def _forecast_causes_conflict(self, forecast_rows, initial_bounds, initial_rows):
        """Isolate the cause; no command from this diagnostic model is dispatched."""
        probe = LinearModel(self.model.remaining_seconds())
        integers = set(self.model.integer)
        for col, (lower, upper) in enumerate(initial_bounds):
            probe.var(lower, upper, binary=col in integers)
        for row in range(initial_rows):
            if row not in forecast_rows:
                probe.row(self.model.rows[row], *self.model.limits[row])
        try:
            probe.solve()
        except ReservationInfeasible:
            return False
        # Reinstating exactly these original rows must prove the full physical
        # problem infeasible, independent of the service optimisation. This
        # also detects consecutive solar slots competing for a single start.
        for row in forecast_rows:
            probe.row(self.model.rows[row], *self.model.limits[row])
        try:
            probe.solve()
        except ReservationInfeasible:
            return True
        return False

    def _fair_service_groups(self, fractions):
        cohorts = defaultdict(list)
        for key in sorted(fractions):
            cohorts[self.units[key].deadline_seconds].append(key)
        groups = []
        for _, cohort in sorted(cohorts.items()):
            fair = self.model.var()
            for key in cohort:
                self.model.row({fractions[key]: 1.0, fair: -1.0}, lo=0.0)
            groups.append((cohort, fair))
        return groups

    def _maximise_levels(self, fractions, groups):
        for col in fractions.values():
            self.model.bound(col, 1.0, 1.0)
        try:
            self.witness = self.model.solve()
            return dict.fromkeys(fractions, 1.0)
        except ReservationInfeasible:
            if not fractions:
                raise
        for col in fractions.values():
            self.model.bound(col, 0.0, 1.0)
        result = {}
        for cohort, fair in groups:
            x = self.model.solve({fair: -1.0})
            self.model.bound(fair, max(0.0, x[fair] - 1e-9), 1.0)
            cost = {fractions[key]: -1.0 for key in cohort}
            x = self.model.solve(cost, known_feasible=x)
            self.model.row(dict.fromkeys(cost, 1.0), lo=sum(x[col] for col in cost) - 1e-9)
            for key in cohort:
                col = fractions[key]
                x = self.model.solve({col: -1.0}, known_feasible=x)
                level = float(np.clip(x[col], 0.0, 1.0))
                if col in self.model.integer:
                    level = round(level)
                result[key] = level
                self.model.bound(col, level, level)
        self.witness = self.model.solve()
        return result

    def _energy_at(self, unit, seconds):
        left = int(np.searchsorted(self.edges, seconds, side="right")) - 1
        if seconds == self.edges[left]:
            return {self.energy[unit.key, left]: 1.0}
        # Between two boundaries the command is held constant. Apply the
        # same transfer-then-standing-loss convention as the full interval.
        duration = self.edges[left + 1] - self.edges[left]
        elapsed = seconds - self.edges[left]
        ratio = elapsed / duration
        keep = 1.0 - unit.battery.hourly_loss
        return {
            self.energy[unit.key, left]: (1.0 - ratio) * keep ** (elapsed / 3600),
            self.energy[unit.key, left + 1]: ratio * keep ** ((elapsed - duration) / 3600),
        }

    def _prioritise(self):
        eligible = {
            key: self.model.var()
            for key, unit in self.units.items()
            if self.departure_margin
            and unit.battery is not None
            and unit.target_kwh is not None
            and self.service_status.get(key) == "in_reservation"
        }
        nominal_groups = self._fair_service_groups(self.coverage)
        buffer_groups = self._fair_service_groups(eligible)
        self.protected.update(self._maximise_levels(self.coverage, nominal_groups))
        if not eligible:
            return
        # All real deadline protections are now fixed. Within that same set,
        # maximise the fraction of the protected departure energy already
        # present at D - margin. Zero early energy is always physically legal.
        for key, col in eligible.items():
            unit = self.units[key]
            target = min(unit.target_kwh, unit.battery.energy_kwh) + self.ledger[
                key
            ] * self.protected.get(key, 1.0)
            early = max(0.0, unit.deadline_seconds - self.departure_margin)
            row = self._energy_at(unit, early)
            row[col] = -target
            self.model.row(row, lo=0.0)
            self.early_reserve[key] = {"at_seconds": early, "target_kwh": target}
        levels = self._maximise_levels(eligible, buffer_groups)
        for key, level in levels.items():
            record = self.early_reserve[key]
            record["protected_kwh"] = record["target_kwh"] * level
            record["shortfall_at_margin_kwh"] = record["target_kwh"] * (1.0 - level)

    def _fix_mode(self, key, name):
        old = {}
        for other in self.modes[key]:
            bit = self.binary.get((key, 0, other))
            if bit is None:
                continue
            old[bit] = self.model.bounds[bit]
            value = float(other == name)
            self.model.bound(bit, value, value)
        return old

    def equipment_fraction_bounds(self, key, name):
        """V0's individual port limits, before any multi-device reconciliation."""
        mode = self.modes[key][name]
        if name in {"IDLE", "START"}:
            return (0.0, 0.0) if name == "IDLE" else (1.0, 1.0)
        unit = self.units[key]
        if not mode.available_now or unit.battery is None or mode.current_max_kw <= 0:
            return (1.0, 0.0)
        b = unit.battery
        upper_kw = mode.current_max_kw
        if mode.direction == 1:
            upper_kw = min(
                upper_kw, (b.capacity_kwh - b.energy_kwh) * 3600.0 / (self.dt * b.charge_efficiency)
            )
            if b.taper_rating_kw is not None:
                taper = 1.0 - (1.0 - b.taper_at_full) * max(
                    b.energy_kwh / b.capacity_kwh - b.taper_start_soc, 0.0
                ) / (1.0 - b.taper_start_soc)
                upper_kw = min(upper_kw, b.taper_rating_kw * taper)
        else:
            upper_kw = min(
                upper_kw,
                max(0.0, b.energy_kwh - b.discharge_reserve_kwh)
                * 3600.0
                * b.discharge_efficiency
                / self.dt,
            )
        lower = mode.min_kw / mode.current_max_kw
        upper = upper_kw / mode.current_max_kw
        if mode.forced_fraction is not None:
            lower, upper = max(lower, mode.forced_fraction), min(upper, mode.forced_fraction)
        return lower, upper

    def equipment_modes(self, key):
        allowed = []
        for name, mode in self.modes[key].items():
            if not mode.available_now:
                continue
            if name == "IDLE":
                allowed.append(name)
            elif (key, 0, name) in self.binary:
                lo, hi = self.equipment_fraction_bounds(key, name)
                if hi > 0 and lo <= hi:
                    allowed.append(name)
        return tuple(allowed)

    def admissible_modes(self, key):
        self.model.remaining_seconds()
        if key in self.mode_cache:
            return self.mode_cache[key]
        if key != self.group_order[len(self.fixed_modes)]:
            raise ValueError("Modes must follow the site decision order")
        allowed = []
        for name, mode in self.modes[key].items():
            if not mode.available_now or (name != "IDLE" and (key, 0, name) not in self.binary):
                continue
            old = self._fix_mode(key, name)
            try:
                if not self.model.verify(self.witness):
                    self.witness = self.model.solve()
                allowed.append(name)
            except ReservationInfeasible:
                pass
            finally:
                for col, (lo, hi) in old.items():
                    self.model.bound(col, lo, hi)
        if not allowed:
            raise ReservationUnavailable(f"{key}: no independently verified feasible mode")
        self.mode_cache[key] = tuple(allowed)
        return self.mode_cache[key]

    def select_mode(self, key, mode):
        if key in self.fixed_modes or mode not in self.admissible_modes(key):
            raise ValueError(f"Invalid mode {key}:{mode}")
        self.fixed_modes[key] = mode
        self._fix_mode(key, mode)
        if len(self.fixed_modes) == len(self.units):
            self._fix_witness()

    def _fix_witness(self):
        # Earliest programme completion uses its start index. A connected
        # unidirectional EV has monotone stored energy until departure, making
        # binary search on the completion boundary exact.
        completion_bounds = {}
        for key in sorted(self.coverage, key=lambda k: (self.units[k].deadline_seconds, k)):
            if self.protected[key] <= 1e-8:
                continue
            u = self.units[key]
            if u.programme_kw:
                costs = {bit: float(t) for (g, t), bit in self.starts.items() if g == key}
                self.witness = self.model.solve(costs)
                for (group, t), bit in self.starts.items():
                    if group != key:
                        continue
                    value = round(self.witness[bit])
                    self.model.bound(bit, value, value)
                    if value:
                        self.early[key] = (
                            self.edges[t] + len(u.programme_kw) * u.programme_step_seconds
                        )
            else:
                col, bounds = self._fix_earliest_energy(key)
                completion_bounds[col] = bounds
        # Positional base-3 chunks implement exact lexicographic tie breaking
        # without ill-conditioned powers across the full reservation.
        groups = []
        for t in range(1, self.n):
            for key in sorted(self.units):
                ranks = {}
                for name, mode in self.modes[key].items():
                    bit = self.binary.get((key, t, name))
                    if bit is not None:
                        ranks[bit] = 2 if mode.direction == -1 else 1
                if ranks:
                    groups.append(ranks)
        for start in range(0, len(groups), 8):
            chunk = groups[start : start + 8]
            costs = {
                bit: rank * 3 ** (len(chunk) - i - 1)
                for i, ranks in enumerate(chunk)
                for bit, rank in ranks.items()
            }
            if not self.model.verify(self.witness) or any(self.witness[j] > 0.5 for j in costs):
                self.witness = self.model.solve(costs)
            for bit in costs:
                value = round(self.witness[bit])
                self.model.bound(bit, value, value)
        if not self.model.verify(self.witness):
            self.witness = self.model.solve()
        # Earliest-completion restrictions select a discrete calendar. After
        # fixing that calendar, power can move within its whole service polytope.
        for col, (lo, hi) in completion_bounds.items():
            self.model.bound(col, lo, hi)
        # All discrete modes have now been fixed; the power pass is a true LP.
        self.model.relax_fixed_integers()
        # The sparse calendar determines all future modes. Interval indices
        # refer to the recorded physical boundaries; omitted modes are IDLE.
        self.calendar = tuple(
            (key, t, name)
            for (key, t, name), bit in self.binary.items()
            if t > 0 and self.witness[bit] > 0.5
        )

    def _fix_earliest_energy(self, key):
        unit = self.units[key]
        target = unit.battery.energy_kwh + self.ledger[key] * self.protected[key]
        end = self.edges.index(unit.deadline_seconds)

        def possible(idx):
            col = self.energy[key, idx]
            lo, hi = self.model.bounds[col]
            if target > hi + 1e-8:
                return False
            self.model.bound(col, max(lo, min(target, hi)), hi)
            try:
                if not self.model.verify(self.witness):
                    self.witness = self.model.solve()
                return True
            except ReservationInfeasible:
                return False
            finally:
                self.model.bound(col, lo, hi)

        if unit.battery.hourly_loss or any(m.direction == -1 for m in unit.modes):
            first = next((t for t in range(1, end + 1) if possible(t)), None)
            if first is None:
                raise ReservationUnavailable("Earliest completion could not be verified")
        else:
            lo, hi = 1, end
            while lo < hi:
                mid = (lo + hi) // 2
                if possible(mid):
                    hi = mid
                else:
                    lo = mid + 1
            first = lo
        col = self.energy[key, first]
        lower, upper = self.model.bounds[col]
        self.model.bound(col, max(lower, min(upper, target)), upper)
        self.early[key] = self.edges[first]
        return col, (lower, upper)

    def remaining_margin_kw(self, key):
        """Site and tightest downstream import margins in the group's kW.

        Includes selected powers and fixed programme starts. Unknown limits
        have no numeric feature; they are represented by None in the policy.
        """
        full_site, downstream = [], []
        for limit in self.limits:
            factor = limit.coefficients.get(key, 0.0)
            if limit.direction != "import" or factor <= 0.0:
                continue
            used = 0.0
            for member, weight in limit.coefficients.items():
                mode = self.fixed_modes.get(member)
                unit = self.units[member]
                used += weight * unit.unconfirmed_import_kw
                if mode == "START" and unit.programme_kw:
                    used += weight * unit.programme_kw[0]
                elif mode and member in self.fixed_fractions:
                    profile = self.modes[member][mode]
                    used += (
                        weight
                        * profile.direction
                        * profile.current_max_kw
                        * self.fixed_fractions[member]
                    )
            bucket = (
                full_site
                if limit.scope == "site"
                or (
                    limit.scope is None
                    and set(limit.coefficients) == set(self.units)
                    and all(weight == 1 for weight in limit.coefficients.values())
                )
                else downstream
            )
            bucket.append((limit.current_kw - used) / factor)
        return min(full_site, default=None), min(downstream, default=None)

    def power_scale_kw(self, key, mode):
        return self.modes[key][mode].current_max_kw

    def fraction_bounds(self, key):
        self.model.remaining_seconds()
        if len(self.fixed_modes) != len(self.units):
            raise ValueError("Select every mode before constructing continuous parameters")
        if key in self.interval_cache:
            return self.interval_cache[key]
        mode = self.fixed_modes[key]
        if mode in {"START", "IDLE"}:
            value = float(mode == "START")
            return value, value
        power = self.controls[key, 0, mode]
        scale = self.power_scale_kw(key, mode)
        lower = float(self.model.solve({power: 1.0})[power]) / scale
        self.witness = self.model.solve({power: -1.0})
        upper = float(self.witness[power]) / scale
        if lower > upper + 1e-7:
            raise ReservationUnavailable("Inconsistent continuous domain")
        interval = float(np.clip(min(lower, upper), 0.0, 1.0)), float(np.clip(upper, 0.0, 1.0))
        if (interval[1] - interval[0]) * scale <= 1e-7:
            midpoint = sum(interval) / 2.0
            interval = midpoint, midpoint
        self.interval_cache[key] = interval
        return interval

    def select_fraction(self, key, fraction):
        if key in self.fixed_fractions:
            raise ValueError(f"Duplicate parameter for {key}")
        lo, hi = self.fraction_bounds(key)
        if not math.isfinite(fraction) or fraction < lo - 1e-8 or fraction > hi + 1e-8:
            raise ValueError(f"Fraction {fraction} outside the domain {lo, hi}")
        mode = self.fixed_modes[key]
        if mode not in {"IDLE", "START"}:
            col = self.controls[key, 0, mode]
            p = min(hi, max(lo, fraction)) * self.power_scale_kw(key, mode)
            self.model.bound(col, p, p)
        self.fixed_fractions[key] = fraction
        self.interval_cache.clear()
        if len(self.fixed_fractions) == len(self.units) and not self.model.verify(self.witness):
            self.witness = self.model.solve()
        self.model.remaining_seconds()

    @property
    def diagnostics(self):
        return {
            "reservation_notes": self.notes,
            "forecast_export_alert": (
                {
                    "at_seconds": self.forecast_alert.at_seconds,
                    "limit": self.forecast_alert.limit_key,
                }
                if self.forecast_alert is not None
                else None
            ),
            "requested_service": dict(self.ledger),
            "protected_fractions": dict(self.protected),
            "departure_margin_seconds": self.departure_margin,
            "ev_early_reserve": self.early_reserve,
            "service_status": dict(self.service_status),
            "witness_calendar": self.calendar,
            "witness_setpoints_kw": tuple(
                (group, t, mode, float(self.witness[col]))
                for (group, t, mode), col in self.controls.items()
                if self.witness is not None and self.witness[col] > 1e-8
            ),
            "interval_boundaries_seconds": tuple(self.edges),
            "earliest_completion_seconds": dict(self.early),
            "solves": self.model.solves,
            "verified": self.witness is not None and self.model.verify(self.witness),
            "horizon_seconds": self.edges[-1],
        }

    def reconcile(self, raw):
        """Minimum altered modes, then minimum normalised power change (V0)."""
        if self.fixed_modes:
            raise ValueError("Reconciliation requires an empty site prefix")
        by_key = {decision.group_id: decision for decision in raw.decisions}
        if by_key.keys() != self.units.keys():
            raise ValueError("Missing or extraneous action groups in site proposal")
        change_cost = {}
        distance = {}
        for key, proposal in by_key.items():
            if proposal.mode == "IDLE":
                change_cost.update(
                    {bit: 1.0 for (g, t, _), bit in self.binary.items() if g == key and t == 0}
                )
            elif (key, 0, proposal.mode) in self.binary:
                change_cost[self.binary[key, 0, proposal.mode]] = -1.0
            dev = self.deviation[key]
            command = {}
            for name, m in self.modes[key].items():
                col = self.controls.get((key, 0, name))
                if col is not None:
                    command[col] = m.direction / m.current_max_kw
                if name == "START" and (key, 0, name) in self.binary:
                    command[self.binary[key, 0, name]] = 1.0
            m = self.modes[key].get(proposal.mode)
            target = (
                proposal.fraction * m.direction
                if m and m.direction
                else float(proposal.mode == "START")
            )
            self.model.row({**command, dev: -1.0}, hi=target)
            self.model.row({**command, dev: 1.0}, lo=target)
            distance[dev] = 1.0
        x = self.model.solve(change_cost)
        optimum = round(sum(x[j] * v for j, v in change_cost.items()))
        self.model.row(change_cost, lo=optimum, hi=optimum)
        x = self.model.solve(distance)
        self.model.row(distance, hi=sum(x[col] for col in distance) + 1e-9)
        decisions = []
        for key in sorted(self.units):
            # Resolve equivalent repairs by operational key and mode order.
            candidates = self.modes[key]
            ranking = {
                self.binary[key, 0, n]: rank
                for rank, n in enumerate(candidates, 1)
                if (key, 0, n) in self.binary
            }
            x = self.model.solve(ranking)
            name = next(
                (
                    n
                    for n in candidates
                    if (key, 0, n) in self.binary and x[self.binary[key, 0, n]] > 0.5
                ),
                "IDLE",
            )
            self._fix_mode(key, name)
            self.fixed_modes[key] = name
            if name in {"IDLE", "START"}:
                fraction = float(name == "START")
            else:
                col = self.controls[key, 0, name]
                x = self.model.solve({col: 1.0})
                fraction = float(x[col]) / self.power_scale_kw(key, name)
                self.model.bound(col, x[col], x[col])
            old = by_key[key]
            # mode_index is the actor's categorical vocabulary index. A
            # reconciled bundle is an applied command, never a PPO sample.
            decisions.append(
                replace(
                    old,
                    mode=name,
                    fraction=fraction,
                    mode_index=old.mode_index if old.mode == name else _mode_index(name),
                )
            )
        self.witness = self.model.solve()
        self.model.remaining_seconds()
        return replace(raw, decisions=tuple(decisions))
