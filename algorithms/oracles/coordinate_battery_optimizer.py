"""Scalable physical-battery coordinate descent for the fixed-service oracle.

The monolithic annual gross-member LP contains one import epigraph for every
building and time step and can reach a time limit before producing a feasible
primal.  This module starts from an already feasible physical schedule and
optimizes one battery at a time while holding every other battery fixed.  Each
subproblem retains the *global* district cost, ramp and peak constraints, but
has only one controllable battery.  Accepted updates monotonically reduce the
true carbon-weighted gross member import.

This is a feasible coordinate-descent heuristic, not a global-optimality
certificate.  Its output still requires CityLearn replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from algorithms.oracles.perfect_foresight_milp import (
    PerfectForesightProblem,
    SemanticActionSeries,
    SemanticSchedule,
    SolveOptions,
)
from algorithms.oracles.scorecard_battery_milp import (
    ScorecardShapingOptions,
    solve_scorecard_battery_schedule,
)


@dataclass(frozen=True)
class CoordinateDispatchMetrics:
    community_cost_eur: float
    total_import_kwh: float
    gross_member_import_kwh: float
    community_emissions_kgco2: float
    mean_absolute_ramp_kwh: float
    mean_daily_peak_import_kwh: float
    all_time_peak_import_kwh: float
    battery_throughput_kwh: float

    def to_dict(self) -> dict[str, float]:
        return {
            "community_cost_eur": self.community_cost_eur,
            "total_import_kwh": self.total_import_kwh,
            "gross_member_import_kwh": self.gross_member_import_kwh,
            "community_emissions_kgco2": self.community_emissions_kgco2,
            "mean_absolute_ramp_kwh": self.mean_absolute_ramp_kwh,
            "mean_daily_peak_import_kwh": self.mean_daily_peak_import_kwh,
            "all_time_peak_import_kwh": self.all_time_peak_import_kwh,
            "battery_throughput_kwh": self.battery_throughput_kwh,
        }


@dataclass(frozen=True)
class CoordinateDispatchResult:
    schedule: SemanticSchedule
    initial_metrics: CoordinateDispatchMetrics
    final_metrics: CoordinateDispatchMetrics
    completed_sweeps: int
    accepted_updates: int
    attempted_updates: int
    projected_candidate_updates: int
    accepted_projection_correction_kwh: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_metrics": self.initial_metrics.to_dict(),
            "final_metrics": self.final_metrics.to_dict(),
            "completed_sweeps": self.completed_sweeps,
            "accepted_updates": self.accepted_updates,
            "attempted_updates": self.attempted_updates,
            "projected_candidate_updates": self.projected_candidate_updates,
            "accepted_projection_correction_kwh": (
                self.accepted_projection_correction_kwh
            ),
            "global_optimum_claim": False,
            "requires_citylearn_replay": True,
            "schedule": self.schedule.to_dict(),
        }


def _schedule_power_matrix(
    problem: PerfectForesightProblem,
    schedule: SemanticSchedule,
) -> np.ndarray:
    if schedule.horizon != problem.horizon or not np.isclose(
        schedule.timestep_hours,
        problem.timestep_hours,
    ):
        raise ValueError("Coordinate schedule does not match the problem horizon.")
    by_key = {
        (series.building_id, series.action_name): np.asarray(
            series.values,
            dtype=np.float64,
        )
        for series in schedule.series
    }
    rows = []
    for battery in problem.batteries:
        key = (battery.building_id, battery.action_name)
        values = by_key.get(key)
        if values is None:
            raise ValueError(f"Coordinate schedule is missing {key!r}.")
        if values.shape != (problem.horizon,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Coordinate schedule has an invalid series for {key!r}.")
        rows.append(values)
    return np.stack(rows)


def _metrics(
    problem: PerfectForesightProblem,
    power_kw: np.ndarray,
    carbon: np.ndarray,
) -> CoordinateDispatchMetrics:
    energy = power_kw * problem.timestep_hours
    member_net = np.asarray(problem.base_net_load_kwh, dtype=np.float64) + energy
    district_net = np.sum(member_net, axis=0)
    district_import = np.clip(district_net, 0.0, None)
    member_import = np.clip(member_net, 0.0, None)
    steps_per_day = max(int(round(24.0 / problem.timestep_hours)), 1)
    daily_peaks = [
        float(np.max(district_import[start : min(start + steps_per_day, problem.horizon)]))
        for start in range(0, problem.horizon, steps_per_day)
    ]
    return CoordinateDispatchMetrics(
        community_cost_eur=float(np.dot(problem.price_eur_per_kwh, district_import)),
        total_import_kwh=float(np.sum(district_import)),
        gross_member_import_kwh=float(np.sum(member_import)),
        community_emissions_kgco2=float(
            np.sum(member_import * carbon.reshape(1, -1))
        ),
        mean_absolute_ramp_kwh=(
            0.0
            if problem.horizon <= 1
            else float(np.mean(np.abs(np.diff(district_net))))
        ),
        mean_daily_peak_import_kwh=float(np.mean(daily_peaks)),
        all_time_peak_import_kwh=float(np.max(district_import)),
        battery_throughput_kwh=float(np.sum(np.abs(energy))),
    )


def _project_signed_power_to_battery(
    problem: PerfectForesightProblem,
    battery_index: int,
    power_kw: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Make a signed trajectory replayable by the conservative battery model."""

    battery = problem.batteries[battery_index]
    model = battery.conservative
    requested_energy = np.asarray(power_kw, dtype=np.float64) * problem.timestep_hours
    projected_energy = np.zeros(problem.horizon, dtype=np.float64)
    soc = float(battery.initial_energy_kwh)
    charge_limit = model.max_charge_kw * problem.timestep_hours
    discharge_limit = model.max_discharge_kw * problem.timestep_hours
    for step, requested in enumerate(requested_energy):
        if requested >= 0.0:
            accepted = min(
                requested,
                charge_limit,
                max((model.capacity_kwh - soc) / model.charge_efficiency, 0.0),
            )
            soc += model.charge_efficiency * accepted
        else:
            discharge = min(
                -requested,
                discharge_limit,
                max(soc * model.discharge_efficiency, 0.0),
            )
            accepted = -discharge
            soc -= discharge / model.discharge_efficiency
        projected_energy[step] = accepted
    if soc + 1.0e-7 < battery.final_energy_min_kwh:
        raise RuntimeError("Projected coordinate schedule misses terminal battery energy.")
    correction = float(np.sum(np.abs(projected_energy - requested_energy)))
    return projected_energy / problem.timestep_hours, correction


def optimize_physical_battery_schedule_coordinate_descent(
    problem: PerfectForesightProblem,
    initial_schedule: SemanticSchedule,
    carbon_intensity_kgco2_per_kwh: Sequence[float],
    shaping: ScorecardShapingOptions,
    solve_options: Optional[SolveOptions] = None,
    *,
    max_sweeps: int = 2,
    minimum_sweep_improvement_kgco2: float = 0.01,
    progress_callback: Optional[
        Callable[[int, int, CoordinateDispatchMetrics, bool], None]
    ] = None,
) -> CoordinateDispatchResult:
    """Reduce gross emissions monotonically under global scorecard limits."""

    if max_sweeps <= 0:
        raise ValueError("max_sweeps must be > 0.")
    if minimum_sweep_improvement_kgco2 < 0.0:
        raise ValueError("minimum_sweep_improvement_kgco2 must be >= 0.")
    if shaping.emissions_accounting != "gross_member_import":
        raise ValueError("Coordinate descent requires gross_member_import accounting.")
    if shaping.emissions_weight <= 0.0:
        raise ValueError("Coordinate descent requires a positive emissions weight.")
    if len(problem.batteries) != len(problem.building_ids):
        raise ValueError("Coordinate descent requires one physical battery per building.")
    battery_buildings = tuple(battery.building_id for battery in problem.batteries)
    if battery_buildings != tuple(problem.building_ids):
        raise ValueError("Physical batteries must follow the building order.")

    carbon = np.asarray(carbon_intensity_kgco2_per_kwh, dtype=np.float64)
    if carbon.shape != (problem.horizon,) or not np.all(np.isfinite(carbon)):
        raise ValueError("Carbon intensity must be finite and match the horizon.")
    if np.any(carbon < 0.0):
        raise ValueError("Carbon intensity must be non-negative.")

    solve_options = solve_options or SolveOptions()
    power = _schedule_power_matrix(problem, initial_schedule).copy()
    initial_metrics = _metrics(problem, power, carbon)
    current_metrics = initial_metrics
    accepted_updates = 0
    attempted_updates = 0
    projected_candidate_updates = 0
    accepted_projection_correction_kwh = 0.0
    completed_sweeps = 0

    for sweep in range(max_sweeps):
        emissions_before_sweep = current_metrics.community_emissions_kgco2
        for battery_index, battery in enumerate(problem.batteries):
            attempted_updates += 1
            energy = power * problem.timestep_hours
            fixed_other_net = np.sum(
                problem.base_net_load_kwh + energy,
                axis=0,
            ) - (
                problem.base_net_load_kwh[battery_index] + energy[battery_index]
            )
            subproblem = PerfectForesightProblem(
                problem_id=(
                    f"{problem.problem_id}-coordinate-s{sweep + 1}-"
                    f"{battery.building_id}"
                ),
                timestep_hours=problem.timestep_hours,
                building_ids=(battery.building_id, "__fixed_other_members__"),
                price_eur_per_kwh=problem.price_eur_per_kwh,
                base_net_load_kwh=np.stack(
                    (problem.base_net_load_kwh[battery_index], fixed_other_net)
                ),
                batteries=(battery,),
                metadata={
                    **dict(problem.metadata),
                    "coordinate_battery": battery.building_id,
                    "coordinate_sweep": sweep + 1,
                },
            )
            warm = SemanticSchedule(
                problem_id=subproblem.problem_id,
                horizon=problem.horizon,
                timestep_hours=problem.timestep_hours,
                series=(
                    SemanticActionSeries(
                        building_id=battery.building_id,
                        action_name=battery.action_name,
                        values=tuple(float(value) for value in power[battery_index]),
                    ),
                ),
                metadata={"coordinate_warm_start": True},
            )
            result = solve_scorecard_battery_schedule(
                subproblem,
                shaping,
                solve_options,
                carbon_intensity_kgco2_per_kwh=carbon,
                initial_schedule=warm,
            )
            accepted = False
            if result.schedule is not None:
                candidate = np.asarray(
                    result.schedule.series[0].values,
                    dtype=np.float64,
                )
                projection_correction = 0.0
                if (
                    result.simultaneous_charge_discharge_kwh is None
                    or result.simultaneous_charge_discharge_kwh > 1.0e-7
                ):
                    candidate, projection_correction = (
                        _project_signed_power_to_battery(
                            problem,
                            battery_index,
                            candidate,
                        )
                    )
                    projected_candidate_updates += 1
                candidate_power = power.copy()
                candidate_power[battery_index] = candidate
                candidate_metrics = _metrics(problem, candidate_power, carbon)
                tolerance = 1.0e-5
                physical_limits_hold = (
                    candidate_metrics.community_cost_eur
                    <= shaping.community_cost_limit_eur + tolerance
                    and (
                        shaping.mean_absolute_ramp_limit_kwh is None
                        or candidate_metrics.mean_absolute_ramp_kwh
                        <= shaping.mean_absolute_ramp_limit_kwh + tolerance
                    )
                    and (
                        shaping.mean_daily_peak_import_limit_kwh is None
                        or candidate_metrics.mean_daily_peak_import_kwh
                        <= shaping.mean_daily_peak_import_limit_kwh + tolerance
                    )
                    and (
                        shaping.all_time_peak_import_limit_kwh is None
                        or candidate_metrics.all_time_peak_import_kwh
                        <= shaping.all_time_peak_import_limit_kwh + tolerance
                    )
                )
                if (
                    physical_limits_hold
                    and candidate_metrics.community_emissions_kgco2
                    <= current_metrics.community_emissions_kgco2 + 1.0e-8
                ):
                    accepted = bool(
                        np.max(np.abs(candidate - power[battery_index])) > 1.0e-9
                    )
                    power = candidate_power
                    current_metrics = candidate_metrics
                    if accepted:
                        accepted_updates += 1
                        accepted_projection_correction_kwh += (
                            projection_correction
                        )
            if progress_callback is not None:
                progress_callback(
                    sweep + 1,
                    battery_index + 1,
                    current_metrics,
                    accepted,
                )

        completed_sweeps += 1
        if (
            emissions_before_sweep - current_metrics.community_emissions_kgco2
            < minimum_sweep_improvement_kgco2
        ):
            break

    series = tuple(
        SemanticActionSeries(
            building_id=battery.building_id,
            action_name=battery.action_name,
            values=tuple(float(value) for value in power[index]),
        )
        for index, battery in enumerate(problem.batteries)
    )
    schedule = SemanticSchedule(
        problem_id=f"{problem.problem_id}-physical-coordinate-carbon",
        horizon=problem.horizon,
        timestep_hours=problem.timestep_hours,
        series=series,
        metadata={
            **dict(initial_schedule.metadata),
            "optimizer": "physical_battery_coordinate_descent",
            "global_constraints_preserved_per_update": True,
            "gross_member_emissions_monotonic": True,
            "relaxed_candidates_projected_to_physical_battery": True,
            "global_optimum_claim": False,
            "requires_citylearn_replay": True,
        },
    )
    return CoordinateDispatchResult(
        schedule=schedule,
        initial_metrics=initial_metrics,
        final_metrics=current_metrics,
        completed_sweeps=completed_sweeps,
        accepted_updates=accepted_updates,
        attempted_updates=attempted_updates,
        projected_candidate_updates=projected_candidate_updates,
        accepted_projection_correction_kwh=(
            accepted_projection_correction_kwh
        ),
    )


__all__ = [
    "CoordinateDispatchMetrics",
    "CoordinateDispatchResult",
    "optimize_physical_battery_schedule_coordinate_descent",
]
