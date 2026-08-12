"""Redistribute a district battery schedule across equivalent local assets.

The district fixed-service MILP may aggregate equivalent batteries to keep the
annual optimization small. Expanding the resulting power proportionally is
lossless for district net exchange, but it is indifferent to simultaneous
member import and export. CityLearn's emissions KPI is based on gross member
imports, so that otherwise harmless allocation can regress emissions.

This module keeps the aggregate battery power exactly fixed and solves small
six-hour linear programs that allocate it among the physical batteries. The
objective minimizes carbon-weighted gross member import. Equivalent state of
charge is restored at every window boundary, making every window at least as
feasible as the original proportional expansion and preventing hidden inter-window
energy borrowing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_array

from algorithms.oracles.perfect_foresight_milp import (
    PerfectForesightProblem,
    SemanticActionSeries,
    SemanticSchedule,
)


_NUMERICAL_FEASIBILITY_TOLERANCE_KWH = 1.0e-6


@dataclass(frozen=True)
class LocalDispatchRedistributionResult:
    schedule: SemanticSchedule
    source_gross_import_kwh: float
    redistributed_gross_import_kwh: float
    source_emissions_kgco2: float
    redistributed_emissions_kgco2: float
    source_battery_throughput_kwh: float
    redistributed_battery_throughput_kwh: float
    maximum_aggregate_power_error_kw: float
    window_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_gross_import_kwh": self.source_gross_import_kwh,
            "redistributed_gross_import_kwh": self.redistributed_gross_import_kwh,
            "source_emissions_kgco2": self.source_emissions_kgco2,
            "redistributed_emissions_kgco2": self.redistributed_emissions_kgco2,
            "source_battery_throughput_kwh": self.source_battery_throughput_kwh,
            "redistributed_battery_throughput_kwh": (
                self.redistributed_battery_throughput_kwh
            ),
            "maximum_aggregate_power_error_kw": self.maximum_aggregate_power_error_kw,
            "window_count": self.window_count,
            "schedule": self.schedule.to_dict(),
        }


def _equivalent_battery_signature(problem: PerfectForesightProblem) -> tuple[Any, ...]:
    if not problem.batteries:
        raise ValueError("Local dispatch redistribution requires physical batteries.")
    signatures = {
        (
            battery.action_name,
            battery.initial_energy_kwh,
            battery.final_energy_min_kwh,
            *battery.conservative.to_dict().values(),
        )
        for battery in problem.batteries
    }
    if len(signatures) != 1:
        raise ValueError(
            "Local dispatch redistribution currently requires exactly equivalent "
            "physical batteries."
        )
    return next(iter(signatures))


def _source_power_matrix(
    problem: PerfectForesightProblem,
    schedule: SemanticSchedule,
) -> np.ndarray:
    if schedule.horizon != problem.horizon:
        raise ValueError("Schedule and fixed-service problem horizons do not match.")
    if not math.isclose(schedule.timestep_hours, problem.timestep_hours):
        raise ValueError("Schedule and fixed-service problem timesteps do not match.")
    by_key = {
        (series.building_id, series.action_name): np.asarray(
            series.values, dtype=np.float64
        )
        for series in schedule.series
    }
    rows = []
    for battery in problem.batteries:
        key = (battery.building_id, battery.action_name)
        if key not in by_key:
            raise ValueError(f"Schedule is missing physical battery series {key!r}.")
        values = by_key[key]
        if values.shape != (problem.horizon,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid physical battery series for {key!r}.")
        rows.append(values)
    return np.stack(rows)


def _aggregate_soc_trajectory(
    problem: PerfectForesightProblem,
    aggregate_energy_kwh: np.ndarray,
) -> np.ndarray:
    batteries = problem.batteries
    model = batteries[0].conservative
    state = np.empty(problem.horizon + 1, dtype=np.float64)
    state[0] = sum(battery.initial_energy_kwh for battery in batteries)
    for step, energy in enumerate(aggregate_energy_kwh):
        state[step + 1] = state[step] + (
            model.charge_efficiency * energy
            if energy >= 0.0
            else energy / model.discharge_efficiency
        )
    total_capacity = sum(battery.conservative.capacity_kwh for battery in batteries)
    tolerance = 1.0e-5 * max(total_capacity, 1.0)
    if np.min(state) < -tolerance or np.max(state) > total_capacity + tolerance:
        raise ValueError("Aggregate source schedule violates conservative storage bounds.")
    # Preserve the raw trajectory here.  Solver schedules can carry residuals
    # around 1e-7 kWh at an empty/full boundary.  Clipping each weekly boundary
    # independently would change the energy balance while the aggregate action
    # equality remains exact, making an otherwise feasible redistribution LP
    # inconsistent.  The window model admits the same tiny numerical margin.
    return state


def _solve_window(
    *,
    base_net_load_kwh: np.ndarray,
    aggregate_energy_kwh: np.ndarray,
    carbon_intensity: np.ndarray,
    initial_soc_kwh: np.ndarray,
    terminal_soc_kwh: np.ndarray,
    capacity_kwh: float,
    max_charge_kwh: float,
    max_discharge_kwh: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    time_limit_seconds: float | None,
    feasibility_tolerance_kwh: float = _NUMERICAL_FEASIBILITY_TOLERANCE_KWH,
) -> tuple[np.ndarray, np.ndarray]:
    n_batteries, horizon = base_net_load_kwh.shape
    charge_start = 0
    discharge_start = n_batteries * horizon
    soc_start = discharge_start + n_batteries * horizon
    import_start = soc_start + n_batteries * (horizon + 1)
    size = import_start + n_batteries * horizon

    def charge(asset: int, step: int) -> int:
        return charge_start + asset * horizon + step

    def discharge(asset: int, step: int) -> int:
        return discharge_start + asset * horizon + step

    def soc(asset: int, step: int) -> int:
        return soc_start + asset * (horizon + 1) + step

    def gross_import(asset: int, step: int) -> int:
        return import_start + asset * horizon + step

    objective = np.zeros(size, dtype=np.float64)
    for asset in range(n_batteries):
        start = gross_import(asset, 0)
        objective[start : start + horizon] = carbon_intensity + 1.0e-6
        # Break otherwise indifferent solutions toward minimum cycling. This
        # also prevents simultaneous charge/discharge in the same battery.
        objective[
            charge(asset, 0) : charge(asset, 0) + horizon
        ] = 1.0e-5
        objective[
            discharge(asset, 0) : discharge(asset, 0) + horizon
        ] = 1.0e-5

    lower = np.zeros(size, dtype=np.float64)
    upper = np.full(size, np.inf, dtype=np.float64)
    for asset in range(n_batteries):
        for step in range(horizon):
            upper[charge(asset, step)] = (
                max_charge_kwh + feasibility_tolerance_kwh
            )
            upper[discharge(asset, step)] = (
                max_discharge_kwh + feasibility_tolerance_kwh
            )
        for step in range(horizon + 1):
            lower[soc(asset, step)] = -feasibility_tolerance_kwh
            upper[soc(asset, step)] = capacity_kwh + feasibility_tolerance_kwh
        lower[soc(asset, 0)] = upper[soc(asset, 0)] = initial_soc_kwh[asset]
        # Aggregate action sums and aggregate SOC are mathematically
        # equivalent, but their floating-point summation order can differ by
        # ~1e-14.  An exact terminal bound lets HiGHS' presolver turn that
        # harmless residual into an infeasible equality system.
        lower[soc(asset, horizon)] = (
            terminal_soc_kwh[asset] - feasibility_tolerance_kwh
        )
        upper[soc(asset, horizon)] = (
            terminal_soc_kwh[asset] + feasibility_tolerance_kwh
        )

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_values: list[float] = []
    eq_rhs: list[float] = []

    def add_eq(entries: Sequence[tuple[int, float]], rhs: float) -> None:
        row = len(eq_rhs)
        for column, value in entries:
            if value != 0.0:
                eq_rows.append(row)
                eq_cols.append(column)
                eq_values.append(float(value))
        eq_rhs.append(float(rhs))

    for asset in range(n_batteries):
        for step in range(horizon):
            add_eq(
                [
                    (soc(asset, step + 1), 1.0),
                    (soc(asset, step), -1.0),
                    (charge(asset, step), -charge_efficiency),
                    (discharge(asset, step), 1.0 / discharge_efficiency),
                ],
                0.0,
            )
    for step, aggregate in enumerate(aggregate_energy_kwh):
        add_eq(
            [
                entry
                for asset in range(n_batteries)
                for entry in (
                    (charge(asset, step), 1.0),
                    (discharge(asset, step), -1.0),
                )
            ],
            float(aggregate),
        )

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_values: list[float] = []
    ub_rhs: list[float] = []
    for asset in range(n_batteries):
        for step in range(horizon):
            row = len(ub_rhs)
            # gross_import >= base + charge - discharge
            ub_rows.extend((row, row, row))
            ub_cols.extend(
                (
                    charge(asset, step),
                    discharge(asset, step),
                    gross_import(asset, step),
                )
            )
            ub_values.extend((1.0, -1.0, -1.0))
            ub_rhs.append(-float(base_net_load_kwh[asset, step]))

    equality = coo_array(
        (
            np.asarray(eq_values, dtype=np.float64),
            (np.asarray(eq_rows, dtype=np.int64), np.asarray(eq_cols, dtype=np.int64)),
        ),
        shape=(len(eq_rhs), size),
    ).tocsc()
    inequality = coo_array(
        (
            np.asarray(ub_values, dtype=np.float64),
            (np.asarray(ub_rows, dtype=np.int64), np.asarray(ub_cols, dtype=np.int64)),
        ),
        shape=(len(ub_rhs), size),
    ).tocsc()
    options: dict[str, Any] = {}
    if time_limit_seconds is not None:
        options["time_limit"] = float(time_limit_seconds)
    raw = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.asarray(ub_rhs, dtype=np.float64),
        A_eq=equality,
        b_eq=np.asarray(eq_rhs, dtype=np.float64),
        bounds=np.column_stack((lower, upper)),
        method="highs",
        options=options,
    )
    if not raw.success or raw.x is None:
        raise RuntimeError(f"Local battery dispatch LP failed: {raw.message}")
    vector = np.asarray(raw.x, dtype=np.float64)
    charge_values = np.stack(
        [
            vector[charge(asset, 0) : charge(asset, 0) + horizon]
            for asset in range(n_batteries)
        ]
    )
    discharge_values = np.stack(
        [
            vector[discharge(asset, 0) : discharge(asset, 0) + horizon]
            for asset in range(n_batteries)
        ]
    )
    action_values = charge_values - discharge_values
    terminal_values = np.asarray(
        [vector[soc(asset, horizon)] for asset in range(n_batteries)],
        dtype=np.float64,
    )
    return action_values, terminal_values


def redistribute_equivalent_battery_schedule(
    problem: PerfectForesightProblem,
    schedule: SemanticSchedule,
    carbon_intensity_kgco2_per_kwh: Sequence[float],
    *,
    window_steps: int = 24,
    time_limit_seconds_per_window: float | None = 60.0,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> LocalDispatchRedistributionResult:
    """Keep district battery power fixed while minimizing gross local import."""

    _equivalent_battery_signature(problem)
    if window_steps <= 0:
        raise ValueError("window_steps must be > 0.")
    carbon = np.asarray(carbon_intensity_kgco2_per_kwh, dtype=np.float64)
    if carbon.shape != (problem.horizon,) or not np.all(np.isfinite(carbon)):
        raise ValueError("Carbon intensity must be finite and match the horizon.")
    if np.any(carbon < 0.0):
        raise ValueError("Carbon intensity must be non-negative.")

    source_power = _source_power_matrix(problem, schedule)
    aggregate_energy = np.sum(source_power, axis=0) * problem.timestep_hours
    aggregate_soc = _aggregate_soc_trajectory(problem, aggregate_energy)
    batteries = problem.batteries
    model = batteries[0].conservative
    n_batteries = len(batteries)
    current_soc = np.asarray(
        [battery.initial_energy_kwh for battery in batteries], dtype=np.float64
    )
    allocated_energy = np.zeros((n_batteries, problem.horizon), dtype=np.float64)
    window_count = 0
    for start in range(0, problem.horizon, window_steps):
        end = min(start + window_steps, problem.horizon)
        terminal_total = aggregate_soc[end]
        terminal = np.full(n_batteries, terminal_total / n_batteries, dtype=np.float64)
        try:
            window_actions, solved_terminal_soc = _solve_window(
                base_net_load_kwh=problem.base_net_load_kwh[:, start:end],
                aggregate_energy_kwh=aggregate_energy[start:end],
                carbon_intensity=carbon[start:end],
                initial_soc_kwh=current_soc,
                terminal_soc_kwh=terminal,
                capacity_kwh=model.capacity_kwh,
                max_charge_kwh=model.max_charge_kw * problem.timestep_hours,
                max_discharge_kwh=model.max_discharge_kw * problem.timestep_hours,
                charge_efficiency=model.charge_efficiency,
                discharge_efficiency=model.discharge_efficiency,
                time_limit_seconds=time_limit_seconds_per_window,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Local battery dispatch failed for window [{start}, {end}): {exc}"
            ) from exc
        allocated_energy[:, start:end] = window_actions
        terminal_error = float(np.max(np.abs(solved_terminal_soc - terminal)))
        # HiGHS also applies its own primal feasibility tolerance when it
        # returns a point on our 1e-6-kWh terminal band.
        if terminal_error > 2.0 * _NUMERICAL_FEASIBILITY_TOLERANCE_KWH:
            raise RuntimeError(
                "Local battery dispatch exceeded its terminal SOC tolerance: "
                f"{terminal_error:.6g} kWh."
            )
        # Carry the state that is implied by the emitted actions. Replacing it
        # with the canonical target hid up to 1e-6 kWh at every boundary; over
        # a year those tiny discontinuities accumulated into a millikWh-scale
        # trajectory mismatch. The following window retains the same terminal
        # band, so HiGHS can absorb its own one-window residual without
        # inventing energy between windows.
        current_soc = solved_terminal_soc
        window_count += 1
        if progress_callback is not None:
            progress_callback(window_count, end, problem.horizon)

    allocated_power = allocated_energy / problem.timestep_hours
    aggregate_error = np.max(np.abs(np.sum(allocated_power, axis=0) - np.sum(source_power, axis=0)))
    tolerance = 1.0e-5
    if aggregate_error > tolerance:
        raise RuntimeError(
            f"Redistributed schedule changed aggregate power by {aggregate_error:.6g} kW."
        )
    source_net = problem.base_net_load_kwh + source_power * problem.timestep_hours
    redistributed_net = problem.base_net_load_kwh + allocated_energy
    source_import = np.maximum(source_net, 0.0)
    redistributed_import = np.maximum(redistributed_net, 0.0)

    series = tuple(
        SemanticActionSeries(
            building_id=battery.building_id,
            action_name=battery.action_name,
            values=tuple(float(value) for value in allocated_power[index]),
        )
        for index, battery in enumerate(batteries)
    )
    redistributed_schedule = SemanticSchedule(
        problem_id=f"{schedule.problem_id}-local-carbon-dispatch",
        horizon=schedule.horizon,
        timestep_hours=schedule.timestep_hours,
        series=series,
        metadata={
            **dict(schedule.metadata),
            "local_dispatch_redistribution": "six_hour_carbon_weighted_gross_import_lp",
            "aggregate_power_preserved": True,
            "window_steps": int(window_steps),
            "physical_series_count": len(series),
            "requires_citylearn_replay": True,
        },
    )
    return LocalDispatchRedistributionResult(
        schedule=redistributed_schedule,
        source_gross_import_kwh=float(np.sum(source_import)),
        redistributed_gross_import_kwh=float(np.sum(redistributed_import)),
        source_emissions_kgco2=float(np.sum(source_import * carbon.reshape(1, -1))),
        redistributed_emissions_kgco2=float(
            np.sum(redistributed_import * carbon.reshape(1, -1))
        ),
        source_battery_throughput_kwh=float(
            np.sum(np.abs(source_power)) * problem.timestep_hours
        ),
        redistributed_battery_throughput_kwh=float(
            np.sum(np.abs(allocated_energy))
        ),
        maximum_aggregate_power_error_kw=float(aggregate_error),
        window_count=window_count,
    )


__all__ = [
    "LocalDispatchRedistributionResult",
    "redistribute_equivalent_battery_schedule",
]
