"""Perfect-foresight mixed-integer model for one complete local home.

Unlike the fixed-service battery oracle, this formulation makes stationary
storage, connected EV/V2G power, and deferrable-cycle start times decisions in
the same optimization.  Optional total and per-phase service limits are hard
constraints.  The objective is the building's own grid-import cost with zero
export credit; there is no community netting or community observation.

The device physics are deliberately explicit linear approximations.  An
optimal result is a certificate for this model, not automatically for
CityLearn's nonlinear battery curves.  A simulator replay remains mandatory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from algorithms.oracles.perfect_foresight_milp import (
    SemanticActionSeries,
    SemanticSchedule,
    SolveOptions,
)


EPS = 1.0e-8


def _finite(name: str, value: float, *, minimum: float | None = None) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        suffix = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must be {suffix}; got {value!r}.")
    return parsed


def _efficiency(name: str, value: float) -> float:
    parsed = _finite(name, value)
    if not 0.0 < parsed <= 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return parsed


@dataclass(frozen=True)
class LinearStorageSpec:
    capacity_kwh: float
    initial_energy_kwh: float
    final_energy_min_kwh: float
    minimum_energy_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    action_name: str = "electrical_storage"
    phase_connection: str | None = None

    def __post_init__(self) -> None:
        capacity = _finite("capacity_kwh", self.capacity_kwh, minimum=0.0)
        minimum = _finite("minimum_energy_kwh", self.minimum_energy_kwh, minimum=0.0)
        initial = _finite("initial_energy_kwh", self.initial_energy_kwh, minimum=0.0)
        final = _finite("final_energy_min_kwh", self.final_energy_min_kwh, minimum=0.0)
        if minimum > capacity + EPS or initial < minimum - EPS or initial > capacity + EPS:
            raise ValueError("Storage initial/minimum energy is outside capacity bounds.")
        if final < minimum - EPS or final > capacity + EPS:
            raise ValueError("Storage final minimum energy is outside capacity bounds.")
        if not str(self.action_name).strip():
            raise ValueError("action_name must be non-empty.")
        object.__setattr__(self, "capacity_kwh", capacity)
        object.__setattr__(self, "minimum_energy_kwh", minimum)
        object.__setattr__(self, "initial_energy_kwh", initial)
        object.__setattr__(self, "final_energy_min_kwh", final)
        object.__setattr__(self, "max_charge_kw", _finite("max_charge_kw", self.max_charge_kw, minimum=0.0))
        object.__setattr__(self, "max_discharge_kw", _finite("max_discharge_kw", self.max_discharge_kw, minimum=0.0))
        object.__setattr__(self, "charge_efficiency", _efficiency("charge_efficiency", self.charge_efficiency))
        object.__setattr__(self, "discharge_efficiency", _efficiency("discharge_efficiency", self.discharge_efficiency))


@dataclass(frozen=True)
class EVSessionSpec:
    session_id: str
    action_name: str
    electric_vehicle_id: str
    start_time_step: int
    end_time_step: int
    capacity_kwh: float
    initial_energy_kwh: float
    required_departure_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    min_charge_kw: float = 0.0
    min_discharge_kw: float = 0.0
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    phase_connection: str | None = None
    allow_departure_shortfall: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        start, end = int(self.start_time_step), int(self.end_time_step)
        if start < 0 or end <= start:
            raise ValueError("EV session must satisfy 0 <= start_time_step < end_time_step.")
        capacity = _finite("EV capacity_kwh", self.capacity_kwh, minimum=0.0)
        minimum = _finite("EV minimum_energy_kwh", self.minimum_energy_kwh, minimum=0.0)
        initial = _finite("EV initial_energy_kwh", self.initial_energy_kwh, minimum=0.0)
        required = _finite(
            "EV required_departure_energy_kwh",
            self.required_departure_energy_kwh,
            minimum=0.0,
        )
        if minimum > capacity + EPS or not minimum - EPS <= initial <= capacity + EPS:
            raise ValueError("EV initial/minimum energy is outside capacity bounds.")
        if not minimum - EPS <= required <= capacity + EPS:
            raise ValueError("EV departure requirement is outside capacity bounds.")
        max_charge = _finite("EV max_charge_kw", self.max_charge_kw, minimum=0.0)
        max_discharge = _finite("EV max_discharge_kw", self.max_discharge_kw, minimum=0.0)
        min_charge = _finite("EV min_charge_kw", self.min_charge_kw, minimum=0.0)
        min_discharge = _finite("EV min_discharge_kw", self.min_discharge_kw, minimum=0.0)
        if min_charge > max_charge + EPS or min_discharge > max_discharge + EPS:
            raise ValueError("EV minimum charger power cannot exceed maximum power.")
        for name in ("session_id", "action_name", "electric_vehicle_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        object.__setattr__(self, "start_time_step", start)
        object.__setattr__(self, "end_time_step", end)
        object.__setattr__(self, "capacity_kwh", capacity)
        object.__setattr__(self, "minimum_energy_kwh", minimum)
        object.__setattr__(self, "initial_energy_kwh", initial)
        object.__setattr__(self, "required_departure_energy_kwh", required)
        object.__setattr__(self, "max_charge_kw", max_charge)
        object.__setattr__(self, "max_discharge_kw", max_discharge)
        object.__setattr__(self, "min_charge_kw", min_charge)
        object.__setattr__(self, "min_discharge_kw", min_discharge)
        object.__setattr__(self, "charge_efficiency", _efficiency("EV charge_efficiency", self.charge_efficiency))
        object.__setattr__(self, "discharge_efficiency", _efficiency("EV discharge_efficiency", self.discharge_efficiency))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class DeferrableCycleSpec:
    cycle_id: str
    action_name: str
    earliest_start_time_step: int
    latest_start_time_step: int
    load_profile_kwh: tuple[float, ...]
    must_run: bool = True

    def __post_init__(self) -> None:
        earliest, latest = int(self.earliest_start_time_step), int(self.latest_start_time_step)
        profile = tuple(_finite("deferrable profile energy", value, minimum=0.0) for value in self.load_profile_kwh)
        if earliest < 0 or latest < earliest:
            raise ValueError("Deferrable start window is invalid.")
        if not profile:
            raise ValueError("Deferrable load profile must not be empty.")
        if not str(self.cycle_id).strip() or not str(self.action_name).strip():
            raise ValueError("Deferrable cycle_id and action_name must be non-empty.")
        object.__setattr__(self, "earliest_start_time_step", earliest)
        object.__setattr__(self, "latest_start_time_step", latest)
        object.__setattr__(self, "load_profile_kwh", profile)


@dataclass(frozen=True)
class ElectricalServiceSpec:
    mode: str = "single_phase"
    default_split: str = "balanced"
    total_import_limit_kw: float | None = None
    total_export_limit_kw: float | None = None
    per_phase_import_limit_kw: Mapping[str, float | None] = field(default_factory=dict)
    per_phase_export_limit_kw: Mapping[str, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        split = str(self.default_split).strip().lower()
        if mode not in {"single_phase", "three_phase"}:
            raise ValueError("Electrical service mode must be single_phase or three_phase.")
        if split not in {"balanced", "l1", "l2", "l3"}:
            raise ValueError("Electrical service default_split is invalid.")
        if mode == "single_phase" and split not in {"balanced", "l1"}:
            raise ValueError("Single-phase service can only use balanced/L1 splitting.")

        def limit(name: str, value: float | None) -> float | None:
            return None if value is None else _finite(name, value, minimum=0.0)

        phases = ("L1",) if mode == "single_phase" else ("L1", "L2", "L3")
        import_limits = {phase: limit(f"{phase} import limit", self.per_phase_import_limit_kw.get(phase)) for phase in phases}
        export_limits = {phase: limit(f"{phase} export limit", self.per_phase_export_limit_kw.get(phase)) for phase in phases}
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "default_split", split)
        object.__setattr__(self, "total_import_limit_kw", limit("total import limit", self.total_import_limit_kw))
        object.__setattr__(self, "total_export_limit_kw", limit("total export limit", self.total_export_limit_kw))
        object.__setattr__(self, "per_phase_import_limit_kw", import_limits)
        object.__setattr__(self, "per_phase_export_limit_kw", export_limits)

    @property
    def phases(self) -> tuple[str, ...]:
        return ("L1",) if self.mode == "single_phase" else ("L1", "L2", "L3")

    def split(self, connection: str | None) -> Mapping[str, float]:
        if self.mode == "single_phase":
            return {"L1": 1.0}
        normalized = None if connection is None else str(connection).strip().upper()
        if normalized in self.phases:
            return {phase: float(phase == normalized) for phase in self.phases}
        if normalized == "ALL_PHASES" or self.default_split == "balanced":
            return {phase: 1.0 / 3.0 for phase in self.phases}
        target = self.default_split.upper()
        return {phase: float(phase == target) for phase in self.phases}


@dataclass(frozen=True)
class TotalHomeProblem:
    problem_id: str
    building_id: str
    timestep_hours: float
    price_eur_per_kwh: np.ndarray
    base_net_load_kwh: np.ndarray
    stationary_storage: LinearStorageSpec | None = None
    ev_sessions: tuple[EVSessionSpec, ...] = ()
    deferrable_cycles: tuple[DeferrableCycleSpec, ...] = ()
    electrical_service: ElectricalServiceSpec | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.problem_id).strip() or not str(self.building_id).strip():
            raise ValueError("problem_id and building_id must be non-empty.")
        step_hours = _finite("timestep_hours", self.timestep_hours)
        if step_hours <= 0.0:
            raise ValueError("timestep_hours must be > 0.")
        prices = np.asarray(self.price_eur_per_kwh, dtype=np.float64)
        base = np.asarray(self.base_net_load_kwh, dtype=np.float64)
        if prices.ndim != 1 or prices.size == 0 or base.shape != prices.shape:
            raise ValueError("Prices and base net load must be aligned non-empty 1-D arrays.")
        if not np.all(np.isfinite(prices)) or np.any(prices < 0.0) or not np.all(np.isfinite(base)):
            raise ValueError("Prices/base load must be finite and prices non-negative.")
        sessions = tuple(self.ev_sessions)
        ids = set()
        occupied: dict[str, set[int]] = {}
        for session in sessions:
            if session.end_time_step > prices.size:
                raise ValueError(f"EV session {session.session_id!r} exceeds problem horizon.")
            if session.session_id in ids:
                raise ValueError(f"Duplicate EV session id {session.session_id!r}.")
            ids.add(session.session_id)
            steps = set(range(session.start_time_step, session.end_time_step))
            if occupied.setdefault(session.action_name, set()).intersection(steps):
                raise ValueError(f"Overlapping EV sessions for action {session.action_name!r}.")
            occupied[session.action_name].update(steps)
        cycles = tuple(self.deferrable_cycles)
        for cycle in cycles:
            if cycle.latest_start_time_step + len(cycle.load_profile_kwh) > prices.size:
                raise ValueError(f"Deferrable cycle {cycle.cycle_id!r} can exceed problem horizon.")
        prices = prices.copy()
        base = base.copy()
        prices.setflags(write=False)
        base.setflags(write=False)
        object.__setattr__(self, "timestep_hours", step_hours)
        object.__setattr__(self, "price_eur_per_kwh", prices)
        object.__setattr__(self, "base_net_load_kwh", base)
        object.__setattr__(self, "ev_sessions", sessions)
        object.__setattr__(self, "deferrable_cycles", cycles)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def horizon(self) -> int:
        return int(self.price_eur_per_kwh.size)


@dataclass(frozen=True)
class TotalHomeSolution:
    status: str
    status_code: int
    optimal: bool
    has_solution: bool
    message: str
    objective_eur: float | None
    objective_lower_bound_eur: float | None
    mip_gap: float | None
    schedule: SemanticSchedule | None
    grid_import_kw: tuple[float, ...]
    grid_net_power_kw: tuple[float, ...]
    stationary_energy_kwh: tuple[float, ...]
    ev_departure_energy_kwh: Mapping[str, float]
    ev_departure_shortfall_kwh: Mapping[str, float]
    deferrable_start_time_step: Mapping[str, int]
    diagnostics: Mapping[str, Any]


class _ModelBuilder:
    def __init__(self) -> None:
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.objective: list[float] = []
        self.rows: list[dict[int, float]] = []
        self.row_lower: list[float] = []
        self.row_upper: list[float] = []

    def variables(
        self,
        count: int,
        *,
        lower: float | Sequence[float] = 0.0,
        upper: float | Sequence[float] = np.inf,
        integer: bool = False,
        objective: float | Sequence[float] = 0.0,
    ) -> np.ndarray:
        start = len(self.lower)
        count = int(count)

        def values(raw: float | Sequence[float]) -> list[float]:
            array = np.broadcast_to(np.asarray(raw, dtype=np.float64), (count,))
            return array.tolist()

        self.lower.extend(values(lower))
        self.upper.extend(values(upper))
        self.integrality.extend([1 if integer else 0] * count)
        self.objective.extend(values(objective))
        return np.arange(start, start + count, dtype=np.int64)

    def constraint(
        self,
        coefficients: Mapping[int, float],
        *,
        lower: float = -np.inf,
        upper: float = np.inf,
    ) -> None:
        cleaned = {int(k): float(v) for k, v in coefficients.items() if abs(float(v)) > 0.0}
        self.rows.append(cleaned)
        self.row_lower.append(float(lower))
        self.row_upper.append(float(upper))

    @staticmethod
    def add(coefs: dict[int, float], index: int, value: float) -> None:
        coefs[int(index)] = coefs.get(int(index), 0.0) + float(value)

    def scipy_inputs(self) -> tuple[np.ndarray, np.ndarray, Bounds, LinearConstraint]:
        row_indices: list[int] = []
        col_indices: list[int] = []
        data: list[float] = []
        for row_index, row in enumerate(self.rows):
            for col_index, value in row.items():
                row_indices.append(row_index)
                col_indices.append(col_index)
                data.append(value)
        matrix = coo_array(
            (data, (row_indices, col_indices)),
            shape=(len(self.rows), len(self.lower)),
        ).tocsc()
        return (
            np.asarray(self.objective, dtype=np.float64),
            np.asarray(self.integrality, dtype=np.int32),
            Bounds(np.asarray(self.lower), np.asarray(self.upper)),
            LinearConstraint(matrix, np.asarray(self.row_lower), np.asarray(self.row_upper)),
        )


def _status(code: int, has_solution: bool) -> str:
    if code == 0:
        return "optimal"
    if code == 1 and has_solution:
        return "limit_with_incumbent"
    if code == 1:
        return "limit_without_incumbent"
    if code == 2:
        return "infeasible"
    if code == 3:
        return "unbounded"
    return "solver_error"


def solve_total_home_milp(
    problem: TotalHomeProblem,
    options: SolveOptions | None = None,
) -> TotalHomeSolution:
    """Solve the complete local-home linear MILP."""

    options = options or SolveOptions()
    model = _ModelBuilder()
    horizon = problem.horizon
    step_hours = problem.timestep_hours

    grid_import = model.variables(
        horizon,
        objective=problem.price_eur_per_kwh * step_hours,
    )
    net_coefficients: list[dict[int, float]] = [dict() for _ in range(horizon)]
    phase_coefficients: dict[str, list[dict[int, float]]] = {}
    service = problem.electrical_service
    if service is not None:
        phase_coefficients = {phase: [dict() for _ in range(horizon)] for phase in service.phases}

    storage_vars: dict[str, np.ndarray] = {}
    storage = problem.stationary_storage
    if storage is not None:
        charge = model.variables(horizon, upper=storage.max_charge_kw)
        discharge = model.variables(horizon, upper=storage.max_discharge_kw)
        mode = model.variables(horizon, upper=1.0, integer=True)
        energy = model.variables(
            horizon + 1,
            lower=storage.minimum_energy_kwh,
            upper=storage.capacity_kwh,
        )
        model.constraint({energy[0]: 1.0}, lower=storage.initial_energy_kwh, upper=storage.initial_energy_kwh)
        model.constraint({energy[-1]: 1.0}, lower=storage.final_energy_min_kwh)
        for t in range(horizon):
            model.constraint({charge[t]: 1.0, mode[t]: -storage.max_charge_kw}, upper=0.0)
            model.constraint({discharge[t]: 1.0, mode[t]: storage.max_discharge_kw}, upper=storage.max_discharge_kw)
            model.constraint(
                {
                    energy[t + 1]: 1.0,
                    energy[t]: -1.0,
                    charge[t]: -storage.charge_efficiency * step_hours,
                    discharge[t]: step_hours / storage.discharge_efficiency,
                },
                lower=0.0,
                upper=0.0,
            )
            _ModelBuilder.add(net_coefficients[t], charge[t], 1.0)
            _ModelBuilder.add(net_coefficients[t], discharge[t], -1.0)
            if service is not None:
                for phase, share in service.split(storage.phase_connection).items():
                    _ModelBuilder.add(phase_coefficients[phase][t], charge[t], share)
                    _ModelBuilder.add(phase_coefficients[phase][t], discharge[t], -share)
        storage_vars = {"charge": charge, "discharge": discharge, "energy": energy}

    ev_vars: dict[str, dict[str, np.ndarray]] = {}
    ev_shortfall_vars: dict[str, int] = {}
    for session in problem.ev_sessions:
        count = session.end_time_step - session.start_time_step
        charge = model.variables(count, upper=session.max_charge_kw)
        discharge = model.variables(count, upper=session.max_discharge_kw)
        mode = model.variables(count, upper=1.0, integer=True)
        energy = model.variables(
            count + 1,
            lower=session.minimum_energy_kwh,
            upper=session.capacity_kwh,
        )
        model.constraint({energy[0]: 1.0}, lower=session.initial_energy_kwh, upper=session.initial_energy_kwh)
        if session.allow_departure_shortfall:
            shortfall = int(model.variables(1, upper=session.required_departure_energy_kwh)[0])
            model.constraint(
                {energy[-1]: 1.0, shortfall: 1.0},
                lower=session.required_departure_energy_kwh,
            )
            ev_shortfall_vars[session.session_id] = shortfall
        else:
            model.constraint({energy[-1]: 1.0}, lower=session.required_departure_energy_kwh)
        for local_t, t in enumerate(range(session.start_time_step, session.end_time_step)):
            model.constraint({charge[local_t]: 1.0, mode[local_t]: -session.max_charge_kw}, upper=0.0)
            model.constraint({charge[local_t]: -1.0, mode[local_t]: session.min_charge_kw}, upper=0.0)
            model.constraint(
                {discharge[local_t]: 1.0, mode[local_t]: session.max_discharge_kw},
                upper=session.max_discharge_kw,
            )
            if session.min_discharge_kw > 0.0:
                model.constraint(
                    {discharge[local_t]: -1.0, mode[local_t]: -session.min_discharge_kw},
                    upper=-session.min_discharge_kw,
                )
            model.constraint(
                {
                    energy[local_t + 1]: 1.0,
                    energy[local_t]: -1.0,
                    charge[local_t]: -session.charge_efficiency * step_hours,
                    discharge[local_t]: step_hours / session.discharge_efficiency,
                },
                lower=0.0,
                upper=0.0,
            )
            _ModelBuilder.add(net_coefficients[t], charge[local_t], 1.0)
            _ModelBuilder.add(net_coefficients[t], discharge[local_t], -1.0)
            if service is not None:
                for phase, share in service.split(session.phase_connection).items():
                    _ModelBuilder.add(phase_coefficients[phase][t], charge[local_t], share)
                    _ModelBuilder.add(phase_coefficients[phase][t], discharge[local_t], -share)
        ev_vars[session.session_id] = {"charge": charge, "discharge": discharge, "energy": energy}

    cycle_vars: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cycle in problem.deferrable_cycles:
        starts = np.arange(cycle.earliest_start_time_step, cycle.latest_start_time_step + 1)
        choices = model.variables(len(starts), upper=1.0, integer=True)
        sum_row = {int(index): 1.0 for index in choices}
        if cycle.must_run:
            model.constraint(sum_row, lower=1.0, upper=1.0)
        else:
            model.constraint(sum_row, upper=1.0)
        for choice, start in zip(choices, starts):
            for offset, energy_kwh in enumerate(cycle.load_profile_kwh):
                t = int(start + offset)
                power_kw = energy_kwh / step_hours
                _ModelBuilder.add(net_coefficients[t], choice, power_kw)
                if service is not None:
                    for phase, share in service.split(None).items():
                        _ModelBuilder.add(phase_coefficients[phase][t], choice, power_kw * share)
        cycle_vars[cycle.cycle_id] = (choices, starts)

    base_kw = problem.base_net_load_kwh / step_hours
    base_phase_share = service.split(None) if service is not None else {}
    for t in range(horizon):
        import_row = dict(net_coefficients[t])
        _ModelBuilder.add(import_row, grid_import[t], -1.0)
        model.constraint(import_row, upper=-float(base_kw[t]))
        if service is None:
            continue
        total_row = dict(net_coefficients[t])
        if service.total_import_limit_kw is not None:
            model.constraint(total_row, upper=service.total_import_limit_kw - float(base_kw[t]))
        if service.total_export_limit_kw is not None:
            model.constraint(total_row, lower=-service.total_export_limit_kw - float(base_kw[t]))
        for phase in service.phases:
            phase_base = float(base_kw[t]) * float(base_phase_share[phase])
            phase_row = phase_coefficients[phase][t]
            import_limit = service.per_phase_import_limit_kw.get(phase)
            export_limit = service.per_phase_export_limit_kw.get(phase)
            if import_limit is not None:
                model.constraint(phase_row, upper=float(import_limit) - phase_base)
            if export_limit is not None:
                model.constraint(phase_row, lower=-float(export_limit) - phase_base)

    solver_options: dict[str, Any] = {
        "presolve": bool(options.presolve),
        "disp": bool(options.display_solver_output),
    }
    if options.time_limit_seconds is not None:
        solver_options["time_limit"] = float(options.time_limit_seconds)
    if options.mip_relative_gap is not None:
        solver_options["mip_rel_gap"] = float(options.mip_relative_gap)
    if options.node_limit is not None:
        solver_options["node_limit"] = int(options.node_limit)
    cost_objective = np.asarray(model.objective, dtype=np.float64).copy()

    def run_solver() -> Any:
        objective, integrality, bounds, constraints = model.scipy_inputs()
        return milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=solver_options,
        )

    service_stage: dict[str, Any] = {"used": False}
    if ev_shortfall_vars:
        model.objective = [0.0] * len(model.objective)
        for shortfall in ev_shortfall_vars.values():
            model.objective[shortfall] = 1.0
        service_raw = run_solver()
        service_vector = None if service_raw.x is None else np.asarray(service_raw.x, dtype=np.float64)
        if service_vector is not None and np.all(np.isfinite(service_vector)):
            minimum_shortfall = float(sum(service_vector[index] for index in ev_shortfall_vars.values()))
            model.constraint(
                {index: 1.0 for index in ev_shortfall_vars.values()},
                upper=minimum_shortfall + 1.0e-7,
            )
            model.objective = cost_objective.tolist()
            raw = run_solver()
            service_stage = {
                "used": True,
                "status": _status(int(service_raw.status), True),
                "minimum_total_shortfall_kwh": minimum_shortfall,
            }
        else:
            raw = service_raw
            service_stage = {
                "used": True,
                "status": _status(int(service_raw.status), False),
                "minimum_total_shortfall_kwh": None,
            }
    else:
        raw = run_solver()
    vector = None if raw.x is None else np.asarray(raw.x, dtype=np.float64)
    has_solution = vector is not None and np.all(np.isfinite(vector))
    status_code = int(raw.status)
    status = _status(status_code, has_solution)
    if not has_solution:
        return TotalHomeSolution(
            status=status,
            status_code=status_code,
            optimal=False,
            has_solution=False,
            message=str(raw.message),
            objective_eur=None,
            objective_lower_bound_eur=getattr(raw, "mip_dual_bound", None),
            mip_gap=getattr(raw, "mip_gap", None),
            schedule=None,
            grid_import_kw=(),
            grid_net_power_kw=(),
            stationary_energy_kwh=(),
            ev_departure_energy_kwh={},
            ev_departure_shortfall_kwh={},
            deferrable_start_time_step={},
            diagnostics={
                "variable_count": len(model.lower),
                "constraint_count": len(model.rows),
                "service_lexicographic_stage": service_stage,
            },
        )

    net_power = base_kw.copy()
    for t, coefficients in enumerate(net_coefficients):
        net_power[t] += sum(value * vector[index] for index, value in coefficients.items())
    service_violation_kw = 0.0
    service_peaks: dict[str, float] = {}
    if service is not None:
        if service.total_import_limit_kw is not None:
            service_violation_kw = max(
                service_violation_kw,
                float(np.max(net_power - service.total_import_limit_kw)),
            )
        if service.total_export_limit_kw is not None:
            service_violation_kw = max(
                service_violation_kw,
                float(np.max(-net_power - service.total_export_limit_kw)),
            )
        for phase in service.phases:
            phase_power = base_kw * float(base_phase_share[phase])
            phase_power = phase_power + np.asarray(
                [
                    sum(value * vector[index] for index, value in phase_coefficients[phase][t].items())
                    for t in range(horizon)
                ],
                dtype=np.float64,
            )
            service_peaks[f"{phase}_import_peak_kw"] = float(np.max(np.clip(phase_power, 0.0, None)))
            service_peaks[f"{phase}_export_peak_kw"] = float(np.max(np.clip(-phase_power, 0.0, None)))
            import_limit = service.per_phase_import_limit_kw.get(phase)
            export_limit = service.per_phase_export_limit_kw.get(phase)
            if import_limit is not None:
                service_violation_kw = max(service_violation_kw, float(np.max(phase_power - import_limit)))
            if export_limit is not None:
                service_violation_kw = max(service_violation_kw, float(np.max(-phase_power - export_limit)))
    service_violation_kw = max(service_violation_kw, 0.0)
    schedule_series: list[SemanticActionSeries] = []
    storage_energy: tuple[float, ...] = ()
    if storage is not None:
        storage_power = vector[storage_vars["charge"]] - vector[storage_vars["discharge"]]
        storage_energy = tuple(float(value) for value in vector[storage_vars["energy"]])
        schedule_series.append(
            SemanticActionSeries(
                building_id=problem.building_id,
                action_name=storage.action_name,
                values=tuple(float(value) for value in storage_power),
            )
        )

    ev_power_by_action: dict[str, np.ndarray] = {}
    ev_departures: dict[str, float] = {}
    ev_shortfalls: dict[str, float] = {}
    for session in problem.ev_sessions:
        variables = ev_vars[session.session_id]
        power = vector[variables["charge"]] - vector[variables["discharge"]]
        full = ev_power_by_action.setdefault(session.action_name, np.zeros(horizon, dtype=np.float64))
        full[session.start_time_step : session.end_time_step] += power
        ev_departures[session.session_id] = float(vector[variables["energy"]][-1])
        ev_shortfalls[session.session_id] = (
            float(vector[ev_shortfall_vars[session.session_id]])
            if session.session_id in ev_shortfall_vars
            else max(session.required_departure_energy_kwh - ev_departures[session.session_id], 0.0)
        )
    for action_name, values in ev_power_by_action.items():
        schedule_series.append(
            SemanticActionSeries(
                building_id=problem.building_id,
                action_name=action_name,
                values=tuple(float(value) for value in values),
            )
        )

    deferrable_starts: dict[str, int] = {}
    deferrable_actions: dict[str, np.ndarray] = {}
    for cycle in problem.deferrable_cycles:
        choices, starts = cycle_vars[cycle.cycle_id]
        selected = np.flatnonzero(vector[choices] > 0.5)
        if selected.size:
            start = int(starts[int(selected[0])])
            deferrable_starts[cycle.cycle_id] = start
            deferrable_actions.setdefault(cycle.action_name, np.zeros(horizon))[start] = 1.0
    for action_name, values in deferrable_actions.items():
        schedule_series.append(
            SemanticActionSeries(
                building_id=problem.building_id,
                action_name=action_name,
                values=tuple(float(value) for value in values),
                unit="binary_start",
                positive_direction="start_cycle",
            )
        )

    schedule = SemanticSchedule(
        problem_id=problem.problem_id,
        horizon=horizon,
        timestep_hours=step_hours,
        series=tuple(schedule_series),
        metadata={
            **dict(problem.metadata),
            "scope": "individual_total_home_linear_milp",
            "decisions": ["stationary_storage", "ev_v2g", "deferrable_start", "electrical_service"],
            "cost_semantics": "individual_import_with_zero_export_credit",
            "global_community_optimum_claim": False,
            "requires_citylearn_replay": True,
            "action_power_limits_kw": {
                **(
                    {
                        storage.action_name: {
                            "max_charge_kw": storage.max_charge_kw,
                            "max_discharge_kw": storage.max_discharge_kw,
                        }
                    }
                    if storage is not None
                    else {}
                ),
                **{
                    action_name: {
                        "max_charge_kw": max(
                            session.max_charge_kw
                            for session in problem.ev_sessions
                            if session.action_name == action_name
                        ),
                        "max_discharge_kw": max(
                            session.max_discharge_kw
                            for session in problem.ev_sessions
                            if session.action_name == action_name
                        ),
                    }
                    for action_name in dict.fromkeys(
                        session.action_name for session in problem.ev_sessions
                    )
                },
            },
        },
    )
    lower_bound = getattr(raw, "mip_dual_bound", None)
    gap = getattr(raw, "mip_gap", None)
    return TotalHomeSolution(
        status=status,
        status_code=status_code,
        optimal=status_code == 0,
        has_solution=True,
        message=str(raw.message),
        objective_eur=float(np.dot(cost_objective, vector)),
        objective_lower_bound_eur=None if lower_bound is None else float(lower_bound),
        mip_gap=None if gap is None else float(gap),
        schedule=schedule,
        grid_import_kw=tuple(float(value) for value in vector[grid_import]),
        grid_net_power_kw=tuple(float(value) for value in net_power),
        stationary_energy_kwh=storage_energy,
        ev_departure_energy_kwh=ev_departures,
        ev_departure_shortfall_kwh=ev_shortfalls,
        deferrable_start_time_step=deferrable_starts,
        diagnostics={
            "variable_count": len(model.lower),
            "binary_variable_count": int(sum(model.integrality)),
            "constraint_count": len(model.rows),
            "maximum_grid_import_identity_error_kw": float(
                np.max(np.abs(vector[grid_import] - np.clip(net_power, 0.0, None)))
            ),
            "service_lexicographic_stage": service_stage,
            "total_ev_departure_shortfall_kwh": float(sum(ev_shortfalls.values())),
            "maximum_electrical_service_violation_kw": service_violation_kw,
            "electrical_service_peaks_kw": service_peaks,
        },
    )


__all__ = [
    "DeferrableCycleSpec",
    "EVSessionSpec",
    "ElectricalServiceSpec",
    "LinearStorageSpec",
    "TotalHomeProblem",
    "TotalHomeSolution",
    "solve_total_home_milp",
]
