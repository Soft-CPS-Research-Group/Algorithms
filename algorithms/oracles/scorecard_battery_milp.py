"""Cost-constrained scorecard shaping for the fixed-service battery oracle.

The economic oracle in :mod:`perfect_foresight_milp` deliberately minimizes
only import cost so that its lower/upper-bound certificate remains easy to
interpret.  A cost-optimal trajectory is not necessarily the best teaching
signal for a controller that must also improve ramping and peaks.  This module
therefore solves a second, explicitly different problem:

* keep modeled community import cost below a caller-provided ceiling; and
* minimize a weighted combination of net-exchange ramping, mean daily peak,
  all-time peak and carbon-weighted imports; and
* optionally enforce explicit limits for each physical KPI.

The result is a perfect-foresight, fixed-service battery demonstration.  It is
not a certificate for the complete CityLearn problem and still requires an
exact simulator replay before promotion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_array

from algorithms.oracles.perfect_foresight_milp import (
    BatteryTrajectory,
    PerfectForesightProblem,
    SemanticActionSeries,
    SemanticSchedule,
    SolveOptions,
    _STATUS_NAMES,
    _milp_primal_only,
    _optional_finite,
    _solver_options,
)


def _finite_non_negative(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}.")
    return parsed


@dataclass(frozen=True)
class ScorecardShapingOptions:
    """Objective and economic guardrail for the shaped demonstration."""

    community_cost_limit_eur: float
    ramping_weight: float = 1.0
    daily_peak_weight: float = 1.0
    all_time_peak_weight: float = 0.25
    emissions_weight: float = 0.0
    throughput_tiebreaker: float = 1.0e-8
    import_cost_tiebreaker: float = 1.0e-8
    community_emissions_limit_kgco2: Optional[float] = None
    mean_absolute_ramp_limit_kwh: Optional[float] = None
    mean_daily_peak_import_limit_kwh: Optional[float] = None
    all_time_peak_import_limit_kwh: Optional[float] = None
    enforce_exclusive_battery_direction: bool = True
    emissions_accounting: str = "district_net_import"

    def __post_init__(self) -> None:
        cost_limit = _finite_non_negative(
            "community_cost_limit_eur", self.community_cost_limit_eur
        )
        if cost_limit <= 0.0:
            raise ValueError("community_cost_limit_eur must be > 0.")
        object.__setattr__(self, "community_cost_limit_eur", cost_limit)
        for name in (
            "ramping_weight",
            "daily_peak_weight",
            "all_time_peak_weight",
            "emissions_weight",
            "throughput_tiebreaker",
            "import_cost_tiebreaker",
        ):
            object.__setattr__(self, name, _finite_non_negative(name, getattr(self, name)))
        if (
            self.ramping_weight <= 0.0
            and self.daily_peak_weight <= 0.0
            and self.all_time_peak_weight <= 0.0
            and self.emissions_weight <= 0.0
        ):
            raise ValueError("At least one physical scorecard weight must be > 0.")
        for name in (
            "community_emissions_limit_kgco2",
            "mean_absolute_ramp_limit_kwh",
            "mean_daily_peak_import_limit_kwh",
            "all_time_peak_import_limit_kwh",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_non_negative(name, value))
        accounting = str(self.emissions_accounting).strip().lower()
        if accounting not in {"district_net_import", "gross_member_import"}:
            raise ValueError(
                "emissions_accounting must be 'district_net_import' or "
                "'gross_member_import'."
            )
        object.__setattr__(self, "emissions_accounting", accounting)

    def to_dict(self) -> dict[str, Any]:
        return {
            "community_cost_limit_eur": self.community_cost_limit_eur,
            "ramping_weight": self.ramping_weight,
            "daily_peak_weight": self.daily_peak_weight,
            "all_time_peak_weight": self.all_time_peak_weight,
            "emissions_weight": self.emissions_weight,
            "throughput_tiebreaker": self.throughput_tiebreaker,
            "import_cost_tiebreaker": self.import_cost_tiebreaker,
            "community_emissions_limit_kgco2": (
                self.community_emissions_limit_kgco2
            ),
            "mean_absolute_ramp_limit_kwh": self.mean_absolute_ramp_limit_kwh,
            "mean_daily_peak_import_limit_kwh": (
                self.mean_daily_peak_import_limit_kwh
            ),
            "all_time_peak_import_limit_kwh": self.all_time_peak_import_limit_kwh,
            "enforce_exclusive_battery_direction": bool(
                self.enforce_exclusive_battery_direction
            ),
            "emissions_accounting": self.emissions_accounting,
        }


@dataclass(frozen=True)
class ScorecardSolverInfo:
    status: str
    status_code: int
    optimal: bool
    has_solution: bool
    message: str
    shaped_objective: Optional[float]
    dual_bound: Optional[float]
    mip_gap: Optional[float]
    candidate_primal_feasible: bool
    initial_primal_feasible: bool
    selected_primal_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "status_code": self.status_code,
            "optimal": self.optimal,
            "has_solution": self.has_solution,
            "message": self.message,
            "shaped_objective": self.shaped_objective,
            "dual_bound": self.dual_bound,
            "mip_gap": self.mip_gap,
            "candidate_primal_feasible": self.candidate_primal_feasible,
            "initial_primal_feasible": self.initial_primal_feasible,
            "selected_primal_source": self.selected_primal_source,
        }


@dataclass(frozen=True)
class ScorecardBatteryResult:
    problem_id: str
    options: ScorecardShapingOptions
    solver: ScorecardSolverInfo
    community_cost_eur: Optional[float]
    total_import_kwh: Optional[float]
    mean_absolute_ramp_kwh: Optional[float]
    mean_daily_peak_import_kwh: Optional[float]
    all_time_peak_import_kwh: Optional[float]
    community_emissions_kgco2: Optional[float]
    gross_member_import_kwh: Optional[float]
    simultaneous_charge_discharge_kwh: Optional[float]
    battery_trajectories: tuple[BatteryTrajectory, ...]
    schedule: Optional[SemanticSchedule]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "formulation": "cost_constrained_scorecard_battery_milp",
            "global_optimum_claim": False,
            "options": self.options.to_dict(),
            "solver": self.solver.to_dict(),
            "community_cost_eur": self.community_cost_eur,
            "total_import_kwh": self.total_import_kwh,
            "mean_absolute_ramp_kwh": self.mean_absolute_ramp_kwh,
            "mean_daily_peak_import_kwh": self.mean_daily_peak_import_kwh,
            "all_time_peak_import_kwh": self.all_time_peak_import_kwh,
            "community_emissions_kgco2": self.community_emissions_kgco2,
            "gross_member_import_kwh": self.gross_member_import_kwh,
            "simultaneous_charge_discharge_kwh": (
                self.simultaneous_charge_discharge_kwh
            ),
            "battery_trajectories": [
                item.to_dict() for item in self.battery_trajectories
            ],
            "schedule": None if self.schedule is None else self.schedule.to_dict(),
            "guarantee": (
                "The cost ceiling and physical objective apply to the supplied "
                "conservative linear fixed-service battery model. CityLearn "
                "feasibility and KPIs require simulator replay."
            ),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class _Layout:
    n_batteries: int
    horizon: int
    n_days: int
    charge_start: int
    discharge_start: int
    soc_start: int
    import_start: int
    net_start: int
    direction_start: int
    ramp_start: int
    daily_peak_start: int
    all_time_peak: int
    member_import_start: int
    size: int

    @classmethod
    def build(
        cls,
        n_batteries: int,
        n_buildings: int,
        horizon: int,
        n_days: int,
        *,
        include_direction: bool,
        include_member_import: bool,
    ) -> "_Layout":
        cursor = 0
        charge_start = cursor
        cursor += n_batteries * horizon
        discharge_start = cursor
        cursor += n_batteries * horizon
        soc_start = cursor
        cursor += n_batteries * (horizon + 1)
        import_start = cursor
        cursor += horizon
        net_start = cursor
        cursor += horizon
        direction_start = cursor
        if include_direction:
            cursor += n_batteries * horizon
        ramp_start = cursor
        cursor += max(horizon - 1, 0)
        daily_peak_start = cursor
        cursor += n_days
        all_time_peak = cursor
        cursor += 1
        member_import_start = cursor
        if include_member_import:
            cursor += n_buildings * horizon
        return cls(
            n_batteries=n_batteries,
            horizon=horizon,
            n_days=n_days,
            charge_start=charge_start,
            discharge_start=discharge_start,
            soc_start=soc_start,
            import_start=import_start,
            net_start=net_start,
            direction_start=direction_start,
            ramp_start=ramp_start,
            daily_peak_start=daily_peak_start,
            all_time_peak=all_time_peak,
            member_import_start=member_import_start,
            size=cursor,
        )

    def charge(self, battery: int, step: int) -> int:
        return self.charge_start + battery * self.horizon + step

    def discharge(self, battery: int, step: int) -> int:
        return self.discharge_start + battery * self.horizon + step

    def soc(self, battery: int, step: int) -> int:
        return self.soc_start + battery * (self.horizon + 1) + step

    def grid_import(self, step: int) -> int:
        return self.import_start + step

    def net(self, step: int) -> int:
        return self.net_start + step

    def direction(self, battery: int, step: int) -> int:
        return self.direction_start + battery * self.horizon + step

    def ramp(self, step: int) -> int:
        if step <= 0:
            raise ValueError("Ramp variables start at time step 1.")
        return self.ramp_start + step - 1

    def daily_peak(self, day: int) -> int:
        return self.daily_peak_start + day

    def member_import(self, building: int, step: int) -> int:
        return self.member_import_start + building * self.horizon + step


def _initial_primal_from_schedule(
    problem: PerfectForesightProblem,
    shaping: ScorecardShapingOptions,
    layout: _Layout,
    schedule: SemanticSchedule,
    *,
    steps_per_day: int,
    include_member_import: bool,
) -> np.ndarray:
    """Lift a semantic physical schedule into every auxiliary LP variable."""

    if schedule.horizon != problem.horizon or not math.isclose(
        schedule.timestep_hours,
        problem.timestep_hours,
    ):
        raise ValueError("Initial scorecard schedule does not match the problem horizon.")
    schedule_by_key = {
        (series.building_id, series.action_name): np.asarray(
            series.values,
            dtype=np.float64,
        )
        for series in schedule.series
    }
    vector = np.zeros(layout.size, dtype=np.float64)
    building_index = {
        building_id: index
        for index, building_id in enumerate(problem.building_ids)
    }
    member_net = np.asarray(problem.base_net_load_kwh, dtype=np.float64).copy()
    district_net = np.sum(member_net, axis=0)

    for battery_index, battery in enumerate(problem.batteries):
        key = (battery.building_id, battery.action_name)
        if key not in schedule_by_key:
            raise ValueError(f"Initial scorecard schedule is missing {key!r}.")
        power_kw = schedule_by_key[key]
        if power_kw.shape != (problem.horizon,) or not np.all(np.isfinite(power_kw)):
            raise ValueError(f"Initial scorecard schedule has an invalid series for {key!r}.")
        signed_energy = power_kw * problem.timestep_hours
        charge = np.clip(signed_energy, 0.0, None)
        discharge = np.clip(-signed_energy, 0.0, None)
        vector[
            layout.charge(battery_index, 0) :
            layout.charge(battery_index, 0) + problem.horizon
        ] = charge
        vector[
            layout.discharge(battery_index, 0) :
            layout.discharge(battery_index, 0) + problem.horizon
        ] = discharge
        model = battery.conservative
        soc = np.empty(problem.horizon + 1, dtype=np.float64)
        soc[0] = battery.initial_energy_kwh
        for step in range(problem.horizon):
            soc[step + 1] = (
                soc[step]
                + model.charge_efficiency * charge[step]
                - discharge[step] / model.discharge_efficiency
            )
        vector[
            layout.soc(battery_index, 0) :
            layout.soc(battery_index, 0) + problem.horizon + 1
        ] = soc
        if shaping.enforce_exclusive_battery_direction:
            vector[
                layout.direction(battery_index, 0) :
                layout.direction(battery_index, 0) + problem.horizon
            ] = (charge > 1.0e-9).astype(np.float64)
        building = building_index[battery.building_id]
        member_net[building] += charge - discharge
        district_net += charge - discharge

    district_import = np.clip(district_net, 0.0, None)
    vector[layout.import_start : layout.net_start] = district_import
    vector[layout.net_start : layout.direction_start] = district_net
    if problem.horizon > 1:
        vector[layout.ramp_start : layout.daily_peak_start] = np.abs(
            np.diff(district_net)
        )
    daily_peaks = [
        float(np.max(district_import[start : min(start + steps_per_day, problem.horizon)]))
        for start in range(0, problem.horizon, steps_per_day)
    ]
    vector[layout.daily_peak_start : layout.all_time_peak] = daily_peaks
    vector[layout.all_time_peak] = float(np.max(district_import))
    if include_member_import:
        vector[layout.member_import_start :] = np.clip(
            member_net,
            0.0,
            None,
        ).reshape(-1)
    return vector


def solve_scorecard_battery_schedule(
    problem: PerfectForesightProblem,
    shaping: ScorecardShapingOptions,
    solve_options: Optional[SolveOptions] = None,
    *,
    carbon_intensity_kgco2_per_kwh: Optional[Sequence[float]] = None,
    initial_schedule: Optional[SemanticSchedule] = None,
) -> ScorecardBatteryResult:
    """Solve a conservative battery schedule under an explicit cost ceiling."""

    solve_options = solve_options or SolveOptions()
    horizon = problem.horizon
    batteries = problem.batteries
    steps_per_day = max(int(round(24.0 / problem.timestep_hours)), 1)
    n_days = int(math.ceil(horizon / steps_per_day))
    include_member_import = bool(
        shaping.emissions_accounting == "gross_member_import"
        and (
            shaping.emissions_weight > 0.0
            or shaping.community_emissions_limit_kgco2 is not None
        )
    )
    layout = _Layout.build(
        len(batteries),
        len(problem.building_ids),
        horizon,
        n_days,
        include_direction=shaping.enforce_exclusive_battery_direction,
        include_member_import=include_member_import,
    )
    carbon = None
    if carbon_intensity_kgco2_per_kwh is not None:
        carbon = np.asarray(carbon_intensity_kgco2_per_kwh, dtype=np.float64)
        if carbon.shape != (horizon,) or not np.all(np.isfinite(carbon)):
            raise ValueError("Carbon intensity must be finite and match the horizon.")
        if np.any(carbon < 0.0):
            raise ValueError("Carbon intensity must be non-negative.")
    if (
        shaping.emissions_weight > 0.0
        or shaping.community_emissions_limit_kgco2 is not None
    ) and carbon is None:
        raise ValueError(
            "Carbon intensity is required for an emissions objective or limit."
        )

    objective = np.zeros(layout.size, dtype=np.float64)
    if horizon > 1:
        objective[layout.ramp_start : layout.daily_peak_start] = (
            shaping.ramping_weight / float(horizon - 1)
        )
    objective[layout.daily_peak_start : layout.all_time_peak] = (
        shaping.daily_peak_weight / float(n_days)
    )
    objective[layout.all_time_peak] = shaping.all_time_peak_weight
    objective[layout.import_start : layout.net_start] = (
        shaping.import_cost_tiebreaker * problem.price_eur_per_kwh
    )
    if shaping.emissions_weight > 0.0 and carbon is not None:
        if shaping.emissions_accounting == "district_net_import":
            objective[layout.import_start : layout.net_start] += (
                shaping.emissions_weight * carbon / float(horizon)
            )
        else:
            if not include_member_import:  # pragma: no cover - guarded above
                raise RuntimeError("Gross-emissions objective lacks import variables.")
            for building in range(len(problem.building_ids)):
                start = layout.member_import(building, 0)
                objective[start : start + horizon] = (
                    shaping.emissions_weight * carbon / float(horizon)
                )
    if shaping.throughput_tiebreaker > 0.0:
        objective[layout.charge_start : layout.discharge_start] = (
            shaping.throughput_tiebreaker
        )
        objective[layout.discharge_start : layout.soc_start] = (
            shaping.throughput_tiebreaker
        )

    lower_bounds = np.zeros(layout.size, dtype=np.float64)
    upper_bounds = np.full(layout.size, np.inf, dtype=np.float64)
    integrality = np.zeros(layout.size, dtype=np.int32)
    lower_bounds[layout.net_start : layout.direction_start] = -np.inf

    for battery_index, battery in enumerate(batteries):
        model = battery.conservative
        charge_limit = model.max_charge_kw * problem.timestep_hours
        discharge_limit = model.max_discharge_kw * problem.timestep_hours
        for time_step in range(horizon):
            upper_bounds[layout.charge(battery_index, time_step)] = charge_limit
            upper_bounds[layout.discharge(battery_index, time_step)] = discharge_limit
            if shaping.enforce_exclusive_battery_direction:
                direction = layout.direction(battery_index, time_step)
                upper_bounds[direction] = 1.0
                integrality[direction] = 1
        for time_step in range(horizon + 1):
            upper_bounds[layout.soc(battery_index, time_step)] = model.capacity_kwh
        initial = layout.soc(battery_index, 0)
        lower_bounds[initial] = battery.initial_energy_kwh
        upper_bounds[initial] = battery.initial_energy_kwh
        lower_bounds[layout.soc(battery_index, horizon)] = battery.final_energy_min_kwh

    row_indices: list[int] = []
    col_indices: list[int] = []
    coefficients: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_row(entries: Sequence[tuple[int, float]], lower: float, upper: float) -> None:
        row = len(constraint_lower)
        for column, coefficient in entries:
            if coefficient != 0.0:
                row_indices.append(row)
                col_indices.append(column)
                coefficients.append(float(coefficient))
        constraint_lower.append(float(lower))
        constraint_upper.append(float(upper))

    for battery_index, battery in enumerate(batteries):
        model = battery.conservative
        charge_limit = model.max_charge_kw * problem.timestep_hours
        discharge_limit = model.max_discharge_kw * problem.timestep_hours
        for time_step in range(horizon):
            add_row(
                [
                    (layout.soc(battery_index, time_step + 1), 1.0),
                    (layout.soc(battery_index, time_step), -1.0),
                    (
                        layout.charge(battery_index, time_step),
                        -model.charge_efficiency,
                    ),
                    (
                        layout.discharge(battery_index, time_step),
                        1.0 / model.discharge_efficiency,
                    ),
                ],
                0.0,
                0.0,
            )
            if shaping.enforce_exclusive_battery_direction:
                direction = layout.direction(battery_index, time_step)
                add_row(
                    [
                        (layout.charge(battery_index, time_step), 1.0),
                        (direction, -charge_limit),
                    ],
                    -np.inf,
                    0.0,
                )
                add_row(
                    [
                        (layout.discharge(battery_index, time_step), 1.0),
                        (direction, discharge_limit),
                    ],
                    -np.inf,
                    discharge_limit,
                )

    district_base = np.sum(problem.base_net_load_kwh, axis=0)
    building_index = {
        building_id: index
        for index, building_id in enumerate(problem.building_ids)
    }
    batteries_by_building: dict[int, list[int]] = {
        index: [] for index in range(len(problem.building_ids))
    }
    for battery_index, battery in enumerate(batteries):
        batteries_by_building[building_index[battery.building_id]].append(
            battery_index
        )
    for time_step in range(horizon):
        net_entries: list[tuple[int, float]] = [(layout.net(time_step), 1.0)]
        for battery_index in range(len(batteries)):
            net_entries.append((layout.charge(battery_index, time_step), -1.0))
            net_entries.append((layout.discharge(battery_index, time_step), 1.0))
        add_row(net_entries, float(district_base[time_step]), float(district_base[time_step]))

        # Modeled import dominates positive net exchange. The positive import
        # tiebreaker selects the exact positive part without introducing a
        # second grid-direction binary variable.
        add_row(
            [
                (layout.grid_import(time_step), 1.0),
                (layout.net(time_step), -1.0),
            ],
            0.0,
            np.inf,
        )
        day = time_step // steps_per_day
        add_row(
            [
                (layout.daily_peak(day), 1.0),
                (layout.net(time_step), -1.0),
            ],
            0.0,
            np.inf,
        )
        add_row(
            [
                (layout.all_time_peak, 1.0),
                (layout.net(time_step), -1.0),
            ],
            0.0,
            np.inf,
        )
        if time_step > 0:
            add_row(
                [
                    (layout.ramp(time_step), 1.0),
                    (layout.net(time_step), -1.0),
                    (layout.net(time_step - 1), 1.0),
                ],
                0.0,
                np.inf,
            )
            add_row(
                [
                    (layout.ramp(time_step), 1.0),
                    (layout.net(time_step), 1.0),
                    (layout.net(time_step - 1), -1.0),
                ],
                0.0,
                np.inf,
            )

        if include_member_import:
            for building in range(len(problem.building_ids)):
                member_entries = [
                    (layout.member_import(building, time_step), 1.0)
                ]
                for battery_index in batteries_by_building[building]:
                    member_entries.extend(
                        (
                            (layout.charge(battery_index, time_step), -1.0),
                            (layout.discharge(battery_index, time_step), 1.0),
                        )
                    )
                add_row(
                    member_entries,
                    float(problem.base_net_load_kwh[building, time_step]),
                    np.inf,
                )

    add_row(
        [
            (layout.grid_import(time_step), float(problem.price_eur_per_kwh[time_step]))
            for time_step in range(horizon)
        ],
        -np.inf,
        shaping.community_cost_limit_eur,
    )
    if shaping.community_emissions_limit_kgco2 is not None and carbon is not None:
        if shaping.emissions_accounting == "district_net_import":
            emissions_entries = [
                (layout.grid_import(time_step), float(carbon[time_step]))
                for time_step in range(horizon)
            ]
        else:
            if not include_member_import:  # pragma: no cover - guarded above
                raise RuntimeError("Gross-emissions ceiling lacks import variables.")
            emissions_entries = [
                (layout.member_import(building, time_step), float(carbon[time_step]))
                for building in range(len(problem.building_ids))
                for time_step in range(horizon)
            ]
        add_row(
            emissions_entries,
            -np.inf,
            shaping.community_emissions_limit_kgco2,
        )
    if shaping.mean_absolute_ramp_limit_kwh is not None and horizon > 1:
        add_row(
            [
                (layout.ramp(time_step), 1.0)
                for time_step in range(1, horizon)
            ],
            -np.inf,
            shaping.mean_absolute_ramp_limit_kwh * float(horizon - 1),
        )
    if shaping.mean_daily_peak_import_limit_kwh is not None:
        add_row(
            [
                (layout.daily_peak(day), 1.0)
                for day in range(n_days)
            ],
            -np.inf,
            shaping.mean_daily_peak_import_limit_kwh * float(n_days),
        )
    if shaping.all_time_peak_import_limit_kwh is not None:
        add_row(
            [(layout.all_time_peak, 1.0)],
            -np.inf,
            shaping.all_time_peak_import_limit_kwh,
        )

    constraint_matrix = coo_array(
        (
            np.asarray(coefficients, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(col_indices, dtype=np.int64),
            ),
        ),
        shape=(len(constraint_lower), layout.size),
    ).tocsc()
    initial_vector = None
    if initial_schedule is not None:
        initial_vector = _initial_primal_from_schedule(
            problem,
            shaping,
            layout,
            initial_schedule,
            steps_per_day=steps_per_day,
            include_member_import=include_member_import,
        )
        tolerance = 1.0e-5
        lower_violation = lower_bounds - initial_vector
        upper_violation = initial_vector - upper_bounds
        maximum_bound_violation = float(
            max(np.max(lower_violation), np.max(upper_violation))
        )
        if maximum_bound_violation > tolerance:
            raise ValueError(
                "Initial scorecard schedule violates variable bounds by "
                f"{maximum_bound_violation:.9g}."
            )
        initial_lhs = np.asarray(constraint_matrix @ initial_vector).reshape(-1)
        initial_lower = np.asarray(constraint_lower, dtype=np.float64)
        initial_upper = np.asarray(constraint_upper, dtype=np.float64)
        maximum_constraint_violation = float(
            max(
                np.max(initial_lower - initial_lhs),
                np.max(initial_lhs - initial_upper),
            )
        )
        if maximum_constraint_violation > tolerance:
            raise ValueError(
                "Initial scorecard schedule violates model constraints by "
                f"{maximum_constraint_violation:.9g}."
            )
    raw = _milp_primal_only(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            constraint_matrix,
            np.asarray(constraint_lower, dtype=np.float64),
            np.asarray(constraint_upper, dtype=np.float64),
        ),
        options=_solver_options(solve_options),
        x0=initial_vector,
    )

    status_code = int(raw.status)
    vector = getattr(raw, "x", None)
    has_solution = vector is not None and np.all(np.isfinite(vector))
    solver = ScorecardSolverInfo(
        status=_STATUS_NAMES.get(status_code, "unknown"),
        status_code=status_code,
        optimal=status_code == 0,
        has_solution=bool(has_solution),
        message=str(raw.message),
        shaped_objective=_optional_finite(getattr(raw, "fun", None)),
        dual_bound=_optional_finite(getattr(raw, "mip_dual_bound", None)),
        mip_gap=_optional_finite(getattr(raw, "mip_gap", None)),
        candidate_primal_feasible=bool(
            getattr(raw, "candidate_primal_feasible", vector is not None)
        ),
        initial_primal_feasible=bool(
            getattr(raw, "initial_primal_feasible", False)
        ),
        selected_primal_source=str(
            getattr(raw, "selected_primal_source", "solver" if vector is not None else "none")
        ),
    )
    if not has_solution:
        return ScorecardBatteryResult(
            problem_id=problem.problem_id,
            options=shaping,
            solver=solver,
            community_cost_eur=None,
            total_import_kwh=None,
            mean_absolute_ramp_kwh=None,
            mean_daily_peak_import_kwh=None,
            all_time_peak_import_kwh=None,
            community_emissions_kgco2=None,
            gross_member_import_kwh=None,
            simultaneous_charge_discharge_kwh=None,
            battery_trajectories=(),
            schedule=None,
        )

    vector = np.asarray(vector, dtype=np.float64)
    net = vector[layout.net_start : layout.direction_start]
    actual_import = np.clip(net, 0.0, None)
    community_cost = float(np.dot(problem.price_eur_per_kwh, actual_import))
    # Reconstruct physical member imports from the selected battery flows.
    # The auxiliary epigraph variables are exact when gross emissions appear
    # in the objective/ceiling, but otherwise HiGHS is free to return any
    # larger feasible value; those slack values must never leak into reported
    # scorecard metrics.
    member_net = np.asarray(problem.base_net_load_kwh, dtype=np.float64).copy()
    for battery_index, battery in enumerate(batteries):
        member_index = building_index[battery.building_id]
        charge = vector[
            layout.charge(battery_index, 0) :
            layout.charge(battery_index, 0) + horizon
        ]
        discharge = vector[
            layout.discharge(battery_index, 0) :
            layout.discharge(battery_index, 0) + horizon
        ]
        member_net[member_index] += charge - discharge
    member_import = np.clip(member_net, 0.0, None)
    gross_member_import = float(np.sum(member_import))
    if carbon is None:
        community_emissions = None
    elif shaping.emissions_accounting == "district_net_import":
        community_emissions = float(np.dot(carbon, actual_import))
    else:
        community_emissions = float(
            np.sum(member_import * carbon.reshape(1, -1))
        )
    tolerance = 1.0e-6 * max(1.0, shaping.community_cost_limit_eur)
    if community_cost > shaping.community_cost_limit_eur + tolerance:
        raise RuntimeError("Scorecard oracle returned a schedule above its cost ceiling.")
    if (
        shaping.community_emissions_limit_kgco2 is not None
        and community_emissions is not None
        and community_emissions
        > shaping.community_emissions_limit_kgco2
        + 1.0e-6 * max(1.0, shaping.community_emissions_limit_kgco2)
    ):
        raise RuntimeError("Scorecard oracle returned a schedule above its emissions ceiling.")

    daily_peaks = [
        float(np.max(actual_import[start : min(start + steps_per_day, horizon)]))
        for start in range(0, horizon, steps_per_day)
    ]
    mean_ramp = 0.0 if horizon <= 1 else float(np.mean(np.abs(np.diff(net))))
    mean_daily_peak = float(np.mean(daily_peaks))
    all_time_peak = float(np.max(actual_import))
    physical_tolerance = 1.0e-6
    if (
        shaping.mean_absolute_ramp_limit_kwh is not None
        and mean_ramp
        > shaping.mean_absolute_ramp_limit_kwh + physical_tolerance
    ):
        raise RuntimeError("Scorecard oracle returned a schedule above its ramp limit.")
    if (
        shaping.mean_daily_peak_import_limit_kwh is not None
        and mean_daily_peak
        > shaping.mean_daily_peak_import_limit_kwh + physical_tolerance
    ):
        raise RuntimeError(
            "Scorecard oracle returned a schedule above its mean daily-peak limit."
        )
    if (
        shaping.all_time_peak_import_limit_kwh is not None
        and all_time_peak
        > shaping.all_time_peak_import_limit_kwh + physical_tolerance
    ):
        raise RuntimeError(
            "Scorecard oracle returned a schedule above its all-time peak limit."
        )
    trajectories: list[BatteryTrajectory] = []
    schedule_series: list[SemanticActionSeries] = []
    simultaneous_charge_discharge_kwh = 0.0
    for battery_index, battery in enumerate(batteries):
        charge = vector[
            layout.charge(battery_index, 0) : layout.charge(battery_index, 0) + horizon
        ]
        discharge = vector[
            layout.discharge(battery_index, 0) : layout.discharge(battery_index, 0) + horizon
        ]
        soc = vector[
            layout.soc(battery_index, 0) : layout.soc(battery_index, 0) + horizon + 1
        ]
        simultaneous_charge_discharge_kwh += float(
            np.sum(np.minimum(np.clip(charge, 0.0, None), np.clip(discharge, 0.0, None)))
        )
        trajectories.append(
            BatteryTrajectory(
                building_id=battery.building_id,
                action_name=battery.action_name,
                charge_kwh=tuple(float(value) for value in charge),
                discharge_kwh=tuple(float(value) for value in discharge),
                state_of_charge_kwh=tuple(float(value) for value in soc),
            )
        )
        signed_power = (charge - discharge) / problem.timestep_hours
        signed_power[np.abs(signed_power) < 1.0e-8] = 0.0
        schedule_series.append(
            SemanticActionSeries(
                building_id=battery.building_id,
                action_name=battery.action_name,
                values=tuple(float(value) for value in signed_power),
            )
        )

    schedule = SemanticSchedule(
        problem_id=problem.problem_id,
        horizon=horizon,
        timestep_hours=problem.timestep_hours,
        series=tuple(schedule_series),
        metadata={
            "formulation": "cost_constrained_scorecard_battery_milp",
            "cost_semantics": "district_import_with_zero_export_credit",
            "scorecard_boundary": "community_net_exchange",
            "shaping_options": shaping.to_dict(),
            "perfect_foresight": True,
            "fixed_service": True,
            "requires_citylearn_replay": True,
        },
    )
    return ScorecardBatteryResult(
        problem_id=problem.problem_id,
        options=shaping,
        solver=solver,
        community_cost_eur=community_cost,
        total_import_kwh=float(np.sum(actual_import)),
        mean_absolute_ramp_kwh=mean_ramp,
        mean_daily_peak_import_kwh=mean_daily_peak,
        all_time_peak_import_kwh=all_time_peak,
        community_emissions_kgco2=community_emissions,
        gross_member_import_kwh=gross_member_import,
        simultaneous_charge_discharge_kwh=simultaneous_charge_discharge_kwh,
        battery_trajectories=tuple(trajectories),
        schedule=schedule,
    )
