"""Causal operating inputs for the thesis controllers, in physical units."""

from __future__ import annotations

import json
import math
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping

import numpy as np

from algorithms.ti_marl.contracts.models import InterfaceSnapshot
from algorithms.ti_marl.runtime.passive_export import PassiveExportFeedback

BLOCK_SECONDS = 900
HORIZON = 96
CHANNELS = ("price_eur_per_kwh", "non_shiftable_kw", "solar_kw")
# Quality of each forecast value, supplied to both policies.
MISSING, ESTIMATED, PUBLISHED, PERSISTENCE = 0, 1, 2, 3
_NATIVE_POINT = re.compile(r"forecast_(price|pv)_next_(\d+)(m|h)(?:_kw)?$")


@dataclass(frozen=True)
class SiteOutlook:
    site: str
    # Each tuple is ordered by target quarter-hour; a zero with quality
    # MISSING is distinct from a forecast of zero.
    values: tuple[tuple[float, ...], ...]
    quality: tuple[tuple[int, ...], ...]
    issued_at_seconds: float | None = None
    received_at_seconds: float | None = None
    calibration_error_kw: tuple[float, float] | None = None
    # Demand and PV, in that order. A residual is measured minus forecast.
    residual_block_start_seconds: float | None = None
    prediction_as_of_seconds: float | None = None
    residual_kw: tuple[float | None, float | None] = (None, None)
    max_abs_residual_kw: tuple[float, float] = (0.0, 0.0)
    # Indexed by lead minus one. A missing entry keeps the independent channel bound.
    passive_export_error_kw: tuple[float | None, ...] = ()
    historical_export_upper_kw: tuple[float | None, ...] = ()


@dataclass(frozen=True)
class OperatingSnapshot(InterfaceSnapshot):
    outlooks: tuple[SiteOutlook, ...] = ()
    incentive: tuple[float, ...] = ()
    measured_peak_kw: float | None = None
    clock_seconds: float = 0.0
    reservation_envelopes: Mapping[str, tuple[float, ...]] = field(default_factory=dict)


def samples_for(snapshot, site):
    """Scalar, valid readings for this site, including its authorised context."""
    return {
        part.observation_id: float(part.values[0])
        for part in snapshot.parts_for(site)
        if part.valid and len(part.values) == 1 and math.isfinite(part.values[0])
    }


class IssuedForecastArchive:
    """Replay only deliveries received by the simulated decision time."""

    def __init__(self, path):
        self.path = Path(path)
        self.latest = {}
        pending = {}
        # Index delivery order once. Keep only the currently visible curves
        # in memory, including when replaying a full year across many sites.
        with self.path.open("rb") as source:
            offset = 0
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    record = json.loads(line)
                    issue = float(record["issued_at_seconds"])
                    receipt = float(record["received_at_seconds"])
                    if not (math.isfinite(issue) and math.isfinite(receipt) and receipt >= issue):
                        raise ValueError(f"Invalid forecast timestamps on line {line_number}")
                    pending.setdefault(record["site"], []).append((receipt, issue, offset))
                offset += len(line)
        self.timelines = {}
        for site, records in pending.items():
            times, offsets = [], []
            newest_issue = -math.inf
            for receipt, issue, offset in sorted(records):
                if issue >= newest_issue:
                    newest_issue = issue
                    times.append(receipt)
                    offsets.append(offset)
            self.timelines[site] = (times, offsets)

    def at(self, now):
        result = {}
        reads = []
        for site, (times, offsets) in self.timelines.items():
            index = bisect_right(times, now) - 1
            if index < 0:
                continue
            offset = offsets[index]
            cached = self.latest.get(site)
            if cached is not None and cached[0] == offset:
                result[site] = cached[1]
            else:
                reads.append((offset, site))
        if reads:
            with self.path.open("rb") as source:
                for offset, site in sorted(reads):
                    source.seek(offset)
                    result[site] = json.loads(source.readline())
                    self.latest[site] = (offset, result[site])
        return result


class ForecastInputs:
    """Use issued curves first, then the simulator's causal points and persistence.

    The load channel is strictly non-shiftable. The simulator's aggregate
    forecast_load includes deferrable programmes and is not a substitute.
    """

    def __init__(self, step_seconds: float, *, metered_export_reserve=False):
        if not math.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        self.step_seconds = step_seconds
        self.history = {}
        self.feedback = defaultdict(PhysicalForecastFeedback)
        self.export_feedback = (
            defaultdict(PassiveExportFeedback)
            if metered_export_reserve and step_seconds == BLOCK_SECONDS
            else None
        )
        self.historical_export_feedback = {}
        self.last_timestamp = None

    def snapshot(
        self, base, issued: Mapping, now: float, *, error_kw=None, pv_lag_steps: int | None = 0
    ) -> OperatingSnapshot:
        block = math.floor(now / BLOCK_SECONDS)
        if self.last_timestamp is not None and now < self.last_timestamp:
            raise ValueError("Forecast clock went backwards within an episode")
        new_instant = now != self.last_timestamp
        self.last_timestamp = now
        if block - HORIZON - 1 in self.history:
            del self.history[block - HORIZON - 1]
        # Following a gap, drop any other expired measurements.
        if len(self.history) > HORIZON + 1:
            self.history = {b: h for b, h in self.history.items() if b >= block - HORIZON}
        current_bin = self.history.setdefault(block, {})
        # A sub-quarter-hour decision sees two partial delivery blocks at the
        # edges of its rolling day. Keep both of them for the physical solver.
        slot_count = HORIZON + int(now % BLOCK_SECONDS != 0)
        outlooks = []
        for site in base.agent_ids:
            obs = samples_for(base, site)
            load = obs.get("non_shiftable_load")
            load_kw = None if load is None else load * 3600 / self.step_seconds
            if new_instant and load_kw is not None:
                total, count = current_bin.get(site, (0.0, 0))
                current_bin[site] = (total + load_kw, count + 1)
            values = np.zeros((3, slot_count))
            quality = np.zeros((3, slot_count), dtype=np.int8)
            for channel, current in enumerate(
                (obs.get("electricity_pricing"), load_kw, obs.get("pv_power_kw"))
            ):
                if current is not None:
                    values[channel].fill(current)
                    quality[channel].fill(PERSISTENCE)
            for slot in range(slot_count):
                historical = self.history.get(block + slot - HORIZON, {}).get(site)
                if historical and slot and block + slot - HORIZON < block:
                    values[1, slot] = historical[0] / historical[1]
                    quality[1, slot] = ESTIMATED
            for name, point in obs.items():
                match = _NATIVE_POINT.fullmatch(name)
                if match is None:
                    continue
                seconds = int(match[2]) * (60 if match[3] == "m" else 3600)
                slot = math.floor((now + seconds) / BLOCK_SECONDS) - block
                if 0 <= slot < slot_count:
                    channel = 0 if match[1] == "price" else 2
                    values[channel, slot] = point
                    quality[channel, slot] = ESTIMATED
            record = issued.get(site)
            issue = receipt = error = None
            if record is not None:
                self._overlay(record, now, block, values, quality)
                if float(record["received_at_seconds"]) <= now:
                    issue = float(record["issued_at_seconds"])
                    receipt = float(record["received_at_seconds"])
                    if "calibration_error_kw" in record:
                        error = tuple(float(v) for v in record["calibration_error_kw"])
                        stop = float(record["calibration_end_seconds"])
                        if not (
                            len(error) == 2
                            and all(math.isfinite(v) and v >= 0 for v in error)
                            and math.isfinite(stop)
                            and stop <= issue
                        ):
                            raise ValueError("Invalid or non-causal reservation calibration")
            if error_kw is not None:
                if self.export_feedback is not None and error is not None:
                    error = tuple(max(error_kw, e) for e in error)
                else:
                    error = (error_kw, error_kw)
            feedback = self.feedback[site]
            if new_instant:
                # Only a fresh local meter can validate the forecast. The
                # policy can still consume stale estimates with a quality flag.
                measured = {
                    p.observation_id: float(p.values[0])
                    for p in base.parts_for(site)
                    if p.owner_agent_id == site
                    and p.sensor_id == "self"
                    and p.observation_id in {"non_shiftable_load", "pv_power_kw"}
                    and p.valid
                    and not p.estimated
                    and p.age_seconds == 0
                    and len(p.values) == 1
                    and math.isfinite(p.values[0])
                    and p.values[0] >= 0
                }
                measured_load = measured.get("non_shiftable_load")
                actual = (
                    None if measured_load is None else measured_load * 3600 / self.step_seconds,
                    measured.get("pv_power_kw"),
                )
                upcoming = tuple(
                    float(values[c, 1]) if quality[c, 1] in (ESTIMATED, PUBLISHED) else None
                    for c in (1, 2)
                )
                feedback.observe(now, self.step_seconds, actual, upcoming, error, pv_lag_steps)
            export_errors = ()
            historical_upper = ()
            if self.export_feedback is not None and now == block * BLOCK_SECONDS:
                paired = self.export_feedback[site]
                eligible = self._paired_slots(record, now, paired.LEADS)
                if new_instant:
                    predicted = np.where(
                        eligible, values[2, paired.LEADS] - values[1, paired.LEADS], np.nan
                    )
                    paired.observe(block, feedback.completed_passive_export_kw, predicted)
                floors = feedback.calibration if error is not None else None
                export_errors = tuple(
                    margin if present else None
                    for margin, present in zip(paired.margins(floors), eligible)
                )
                if site in self.historical_export_feedback or (
                    record is not None and "passive_export_upper_kw" in record
                ):
                    if site not in self.historical_export_feedback:
                        self.historical_export_feedback[site] = PassiveExportFeedback()
                    history = self.historical_export_feedback[site]
                    upper = self._historical_export(record, now, eligible)
                    if new_instant:
                        history.observe(block, feedback.completed_passive_export_kw, upper)
                    # The issued estimate already retains all historical same-quarter
                    # misses. Add the live same-lead miss or configured floor on top.
                    historical_upper = tuple(
                        float(value + margin) if np.isfinite(value) and margin is not None else None
                        for value, margin in zip(upper, history.margins(floors))
                    )
            outlooks.append(
                SiteOutlook(
                    site,
                    tuple(map(tuple, values.tolist())),
                    tuple(map(tuple, quality.tolist())),
                    issue,
                    receipt,
                    feedback.effective_margin(error),
                    feedback.residual_block,
                    feedback.residual_as_of,
                    feedback.residual,
                    feedback.max_abs,
                    export_errors,
                    historical_upper,
                )
            )
        attrs = {f.name: getattr(base, f.name) for f in fields(InterfaceSnapshot)}
        return OperatingSnapshot(**attrs, outlooks=tuple(outlooks), clock_seconds=now)

    @staticmethod
    def _historical_export(record, now, eligible):
        result = np.full(len(eligible), np.nan)
        if not any(eligible) or record is None or "passive_export_upper_kw" not in record:
            return result
        cutoff = record.get("passive_export_history_end_seconds")
        curve = record["passive_export_upper_kw"]
        if (
            not isinstance(cutoff, (float, int))
            or not 0 <= cutoff <= now
            or not isinstance(curve, (list, tuple))
            or len(curve) != len(eligible)
        ):
            return result
        for i, value in enumerate(curve):
            if eligible[i] and isinstance(value, (float, int)) and math.isfinite(value):
                result[i] = value
        return result

    @staticmethod
    def _paired_slots(record, now, leads):
        # Only a fresh, explicitly issued pair can calibrate the joint meter.
        if (
            record is None
            or any(
                record[k] != now
                for k in ("issued_at_seconds", "received_at_seconds", "valid_from_seconds")
            )
            or record["interval_seconds"] != BLOCK_SECONDS
        ):
            return np.zeros(len(leads), dtype=bool)
        demand, solar = (record.get(key, ()) for key in CHANNELS[1:])
        return np.array(
            [
                lead < min(len(demand), len(solar))
                and demand[lead] is not None
                and solar[lead] is not None
                for lead in leads
            ]
        )

    @staticmethod
    def _overlay(record, now, block, values, quality):
        """Times are in seconds on the episode's declared physical clock."""
        if not isinstance(record, Mapping):
            raise ValueError("Expected a forecast record for each supplied site")
        issue, receipt, start, period = (
            float(record[key])
            for key in (
                "issued_at_seconds",
                "received_at_seconds",
                "valid_from_seconds",
                "interval_seconds",
            )
        )
        if (
            not all(map(math.isfinite, (issue, receipt, start, period)))
            or period <= 0
            or receipt < issue
        ):
            raise ValueError("Invalid forecast interval or publication timestamps")
        if receipt > now:
            return
        for channel, key in enumerate(CHANNELS):
            series = record.get(key, ())
            published = record.get("published_price", ())
            for slot in range(values.shape[1]):
                target = max(now, (block + slot) * BLOCK_SECONDS)
                index = math.floor((target - start) / period)
                if index < 0 or index >= len(series) or series[index] is None:
                    continue
                number = float(series[index])
                if not math.isfinite(number) or (channel and number < 0):
                    raise ValueError(f"Invalid forecast for {key}")
                values[channel, slot] = number
                firm = channel == 0 and index < len(published) and published[index] is True
                quality[channel, slot] = PUBLISHED if firm else ESTIMATED


class PhysicalForecastFeedback:
    """Compare an ex-ante forecast with a complete metered quarter at one site."""

    def __init__(self):
        self.block = None
        self.forecast = self.next_forecast = (None, None)
        self.as_of = self.next_as_of = None
        self.residual_block = self.residual_as_of = None
        self.residual = (None, None)
        self.max_abs = (0.0, 0.0)
        self.completed_passive_export_kw = None
        self.calibration = (0.0, 0.0)

    def _add_meter(self, channel, start, seconds, power):
        if self.block is None or power is None:
            return
        # Only complete, contiguous meter intervals can contribute to one
        # physical quarter. A missing reading or a gap invalidates its mean.
        end = start + seconds
        if end <= (self.block + 1) * BLOCK_SECONDS and math.isclose(
            self.cursor[channel], start, rel_tol=0.0, abs_tol=1e-6
        ):
            self.area[channel] += power * seconds
            self.cursor[channel] = end

    def observe(self, now, seconds, actual, upcoming, calibration, pv_lag_steps):
        # SoftCPS exposes PV from its endogenous interval, which commonly
        # ends at this observation. Count the final sample before closing it.
        if pv_lag_steps == 1:
            self._add_meter(1, now - seconds, seconds, actual[1])
        block = math.floor(now / BLOCK_SECONDS)
        start = block * BLOCK_SECONDS
        if self.block != block:
            self.completed_passive_export_kw = None
            if self.block is not None:
                end = (self.block + 1) * BLOCK_SECONDS
                if block == self.block + 1 and all(
                    math.isclose(cursor, end, rel_tol=0.0, abs_tol=1e-6) for cursor in self.cursor
                ):
                    self.completed_passive_export_kw = (self.area[1] - self.area[0]) / BLOCK_SECONDS
                self.residual_block = self.block * BLOCK_SECONDS
                self.residual_as_of = self.as_of
                self.residual = tuple(
                    area / BLOCK_SECONDS - predicted
                    if math.isclose(cursor, end, rel_tol=0.0, abs_tol=1e-6)
                    and predicted is not None
                    else None
                    for area, cursor, predicted in zip(self.area, self.cursor, self.forecast)
                )
                self.max_abs = tuple(
                    max(previous, abs(error)) if error is not None else previous
                    for previous, error in zip(self.max_abs, self.residual)
                )
            consecutive = self.block is not None and block == self.block + 1
            self.forecast = self.next_forecast if consecutive else (None, None)
            self.as_of = self.next_as_of if consecutive else None
            self.block = block
            self.area = [0.0, 0.0]
            self.cursor = [start, start]
        self._add_meter(0, now, seconds, actual[0])
        if pv_lag_steps == 0:
            self._add_meter(1, now, seconds, actual[1])
        self.next_forecast = upcoming
        self.next_as_of = now
        if calibration is not None:
            self.calibration = tuple(map(max, self.calibration, calibration))

    def effective_margin(self, calibration):
        if calibration is None:
            return None
        return tuple(map(max, self.calibration, self.max_abs))


def reservation_envelopes(snapshot, registry, error_kw):
    """Bounds conditional on the declared net-load error and equipment ratings.

    A configured scalar bounds each channel separately. Without calibration
    we leave future limits unknown. On a
    phase with unknown baseload distribution, assign all demand (or all PV for
    export) to this phase, and credit no offset from the opposite flow.
    """
    curves = {curve.site: curve for curve in snapshot.outlooks}
    result = {}
    for constraint in snapshot.constraints:
        site = constraint.owner_agent_id
        spec = registry.for_agent(site).constraints
        curve = curves[site]
        margin = curve.calibration_error_kw
        if error_kw is not None:
            floor = margin or (error_kw, error_kw)
            margin = tuple(
                max(error_kw, old, observed)
                for old, observed in zip(floor, curve.max_abs_residual_kw)
            )
        if margin is None:
            continue
        phase = re.search(r"_phase_(L[123])_", constraint.constraint_id)
        export = constraint.constraint_type.startswith("export")
        if constraint.constraint_type not in {
            "charging_headroom_kw",
            "export_headroom_kw",
            "charging_phase_headroom_kw",
            "export_phase_headroom_kw",
        }:
            continue
        side = "export" if export else "import"
        name = ("phase_" if phase else "grid_") + side
        rating = spec.get(name, {}).get("max")
        if phase:
            rating = rating.get(phase[1]) if isinstance(rating, Mapping) else None
        if rating is None:
            continue
        bounds = []
        for load, pv, lq, pq in zip(
            curve.values[1], curve.values[2], curve.quality[1], curve.quality[2]
        ):
            # Flat extrapolations are useful policy inputs but cannot silently
            # become a calibrated lower bound on demand or future solar.
            if lq in (MISSING, PERSISTENCE) and not export:
                break
            if pq in (MISSING, PERSISTENCE) and export:
                break
            # Independent physical envelopes cannot cross zero. Split passive
            # demand and PV by their declared electrical connection if available.
            load_lo = max(0.0, load - margin[0]) if lq not in (MISSING, PERSISTENCE) else 0.0
            pv_lo = max(0.0, pv - margin[1]) if pq not in (MISSING, PERSISTENCE) else 0.0
            load_hi, pv_hi = load + margin[0], pv + margin[1]
            share = 1.0
            if phase:
                weights = spec.get("background_phase_weights", {})
                share = weights.get(phase[1]) if isinstance(weights, Mapping) else None
            if export:
                background = pv_hi if share is None else share * (pv_hi - load_lo)
                lead = len(bounds)
                if share is not None and 1 <= lead <= len(curve.passive_export_error_kw):
                    paired = curve.passive_export_error_kw[lead - 1]
                    if paired is not None and lq == pq == ESTIMATED and background > rating:
                        # If only the independent extremes imply overload, a measured
                        # joint certificate can admit net-consuming programmes. Give
                        # discretionary export no additional capacity.
                        upper = pv - load + paired
                        if lead <= len(curve.historical_export_upper_kw):
                            historical = curve.historical_export_upper_kw[lead - 1]
                            if historical is not None:
                                upper = min(upper, historical)
                        if share * upper <= rating:
                            background = float(rating)
            else:
                background = load_hi if share is None else share * (load_hi - pv_lo)
            bounds.append(float(rating) - background)
        result[constraint.constraint_id] = tuple(bounds)
    return result
