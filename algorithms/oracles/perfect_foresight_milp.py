"""Sparse bounded perfect-foresight oracle.

This module deliberately separates two mathematical objects:

* an optimistic linear relaxation, whose optimum is a lower bound for the
  conservative model; and
* a conservative mixed-integer model that produces a single signed battery
  action per asset and time step.

The lower-bound proof is structural.  Optimistic battery capacity and power
must dominate their conservative equivalents, optimistic efficiencies must be
at least as high, charge/discharge complementarity is relaxed, and a
non-negative energy-spill variable is added.  Every conservative trajectory
can therefore be embedded in the optimistic formulation using the same grid
exchange and state of charge.

The conservative result is feasible for the *linear battery model supplied to
this module*.  It is not claimed to be feasible for CityLearn's nonlinear
battery implementation until an external simulator replay validates it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp
from scipy.sparse import coo_array, csc_array

try:  # SciPy 1.15.x exposes its bundled HiGHS bindings here.
    from scipy.optimize._highspy import _core as _highs_core
    from scipy.optimize._milp import _milp_iv
except ImportError:  # pragma: no cover - compatibility fallback for other SciPy builds.
    _highs_core = None
    _milp_iv = None


_NUMERICAL_TOLERANCE = 1.0e-8


def _finite_non_negative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}.")
    return value


def _efficiency(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1]; got {value!r}.")
    return value


@dataclass(frozen=True)
class BatteryModel:
    """Linear grid-side battery parameters for one formulation."""

    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacity_kwh", _finite_non_negative("capacity_kwh", self.capacity_kwh))
        object.__setattr__(self, "max_charge_kw", _finite_non_negative("max_charge_kw", self.max_charge_kw))
        object.__setattr__(self, "max_discharge_kw", _finite_non_negative("max_discharge_kw", self.max_discharge_kw))
        object.__setattr__(
            self,
            "charge_efficiency",
            _efficiency("charge_efficiency", self.charge_efficiency),
        )
        object.__setattr__(
            self,
            "discharge_efficiency",
            _efficiency("discharge_efficiency", self.discharge_efficiency),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "capacity_kwh": self.capacity_kwh,
            "max_charge_kw": self.max_charge_kw,
            "max_discharge_kw": self.max_discharge_kw,
            "charge_efficiency": self.charge_efficiency,
            "discharge_efficiency": self.discharge_efficiency,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BatteryModel":
        return cls(**dict(payload))


@dataclass(frozen=True)
class BatteryAsset:
    """Semantic battery identity and its optimistic/conservative models."""

    building_id: str
    action_name: str
    initial_energy_kwh: float
    final_energy_min_kwh: float
    optimistic: BatteryModel
    conservative: BatteryModel

    def __post_init__(self) -> None:
        if not str(self.building_id).strip():
            raise ValueError("building_id must be a non-empty string.")
        if not str(self.action_name).strip():
            raise ValueError("action_name must be a non-empty string.")

        initial = _finite_non_negative("initial_energy_kwh", self.initial_energy_kwh)
        final_min = _finite_non_negative("final_energy_min_kwh", self.final_energy_min_kwh)
        object.__setattr__(self, "building_id", str(self.building_id))
        object.__setattr__(self, "action_name", str(self.action_name))
        object.__setattr__(self, "initial_energy_kwh", initial)
        object.__setattr__(self, "final_energy_min_kwh", final_min)

        if initial > self.conservative.capacity_kwh + _NUMERICAL_TOLERANCE:
            raise ValueError("initial_energy_kwh exceeds conservative capacity.")
        if final_min > self.conservative.capacity_kwh + _NUMERICAL_TOLERANCE:
            raise ValueError("final_energy_min_kwh exceeds conservative capacity.")

        dominance_checks = (
            ("capacity_kwh", self.optimistic.capacity_kwh, self.conservative.capacity_kwh),
            ("max_charge_kw", self.optimistic.max_charge_kw, self.conservative.max_charge_kw),
            ("max_discharge_kw", self.optimistic.max_discharge_kw, self.conservative.max_discharge_kw),
            (
                "charge_efficiency",
                self.optimistic.charge_efficiency,
                self.conservative.charge_efficiency,
            ),
            (
                "discharge_efficiency",
                self.optimistic.discharge_efficiency,
                self.conservative.discharge_efficiency,
            ),
        )
        for name, optimistic_value, conservative_value in dominance_checks:
            if optimistic_value + _NUMERICAL_TOLERANCE < conservative_value:
                raise ValueError(
                    f"optimistic {name} must be >= conservative {name} to preserve the lower-bound proof."
                )

    @property
    def semantic_key(self) -> tuple[str, str]:
        return self.building_id, self.action_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "action_name": self.action_name,
            "initial_energy_kwh": self.initial_energy_kwh,
            "final_energy_min_kwh": self.final_energy_min_kwh,
            "optimistic": self.optimistic.to_dict(),
            "conservative": self.conservative.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BatteryAsset":
        data = dict(payload)
        data["optimistic"] = BatteryModel.from_dict(data["optimistic"])
        data["conservative"] = BatteryModel.from_dict(data["conservative"])
        return cls(**data)


@dataclass(frozen=True)
class PerfectForesightProblem:
    """Exogenous district data and controllable battery assets.

    ``base_net_load_kwh`` is shaped ``(n_buildings, horizon)`` and already
    includes uncontrollable demand minus generation.  Positive values import
    energy; negative values export energy.  Export has zero credit in this MVP.
    """

    problem_id: str
    timestep_hours: float
    building_ids: tuple[str, ...]
    price_eur_per_kwh: np.ndarray
    base_net_load_kwh: np.ndarray
    batteries: tuple[BatteryAsset, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.problem_id).strip():
            raise ValueError("problem_id must be a non-empty string.")
        timestep_hours = float(self.timestep_hours)
        if not math.isfinite(timestep_hours) or timestep_hours <= 0.0:
            raise ValueError("timestep_hours must be finite and > 0.")

        building_ids = tuple(str(value) for value in self.building_ids)
        if not building_ids or any(not value.strip() for value in building_ids):
            raise ValueError("building_ids must contain at least one non-empty identifier.")
        if len(set(building_ids)) != len(building_ids):
            raise ValueError("building_ids must be unique.")

        prices = np.asarray(self.price_eur_per_kwh, dtype=np.float64)
        base = np.asarray(self.base_net_load_kwh, dtype=np.float64)
        if prices.ndim != 1 or prices.size == 0:
            raise ValueError("price_eur_per_kwh must be a non-empty 1-D array.")
        if base.shape != (len(building_ids), prices.size):
            raise ValueError(
                "base_net_load_kwh must have shape "
                f"({len(building_ids)}, {prices.size}); got {base.shape}."
            )
        if not np.all(np.isfinite(prices)) or np.any(prices < 0.0):
            raise ValueError("price_eur_per_kwh must be finite and non-negative for the zero-export-credit model.")
        if not np.all(np.isfinite(base)):
            raise ValueError("base_net_load_kwh must contain only finite values.")

        batteries = tuple(self.batteries)
        building_set = set(building_ids)
        semantic_keys: set[tuple[str, str]] = set()
        for battery in batteries:
            if battery.building_id not in building_set:
                raise ValueError(f"Battery building_id {battery.building_id!r} is not present in building_ids.")
            if battery.semantic_key in semantic_keys:
                raise ValueError(f"Duplicate semantic battery key: {battery.semantic_key!r}.")
            semantic_keys.add(battery.semantic_key)

        prices = prices.copy()
        base = base.copy()
        prices.setflags(write=False)
        base.setflags(write=False)
        object.__setattr__(self, "problem_id", str(self.problem_id))
        object.__setattr__(self, "timestep_hours", timestep_hours)
        object.__setattr__(self, "building_ids", building_ids)
        object.__setattr__(self, "price_eur_per_kwh", prices)
        object.__setattr__(self, "base_net_load_kwh", base)
        object.__setattr__(self, "batteries", batteries)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def horizon(self) -> int:
        return int(self.price_eur_per_kwh.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "timestep_hours": self.timestep_hours,
            "building_ids": list(self.building_ids),
            "price_eur_per_kwh": self.price_eur_per_kwh.tolist(),
            "base_net_load_kwh": self.base_net_load_kwh.tolist(),
            "batteries": [battery.to_dict() for battery in self.batteries],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PerfectForesightProblem":
        data = dict(payload)
        data["building_ids"] = tuple(data["building_ids"])
        data["price_eur_per_kwh"] = np.asarray(data["price_eur_per_kwh"], dtype=np.float64)
        data["base_net_load_kwh"] = np.asarray(data["base_net_load_kwh"], dtype=np.float64)
        data["batteries"] = tuple(BatteryAsset.from_dict(item) for item in data.get("batteries", ()))
        return cls(**data)


@dataclass(frozen=True)
class SolveOptions:
    """HiGHS options and the conservative lexicographic tie-breaker."""

    time_limit_seconds: Optional[float] = None
    mip_relative_gap: Optional[float] = None
    node_limit: Optional[int] = None
    presolve: bool = True
    solver: str = "choose"
    display_solver_output: bool = False
    throughput_tiebreaker_eur_per_kwh: float = 1.0e-9
    lexicographic_shortfall_tolerance_kwh: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.time_limit_seconds is not None and float(self.time_limit_seconds) <= 0.0:
            raise ValueError("time_limit_seconds must be > 0 when provided.")
        if self.mip_relative_gap is not None and float(self.mip_relative_gap) < 0.0:
            raise ValueError("mip_relative_gap must be >= 0 when provided.")
        if self.node_limit is not None and int(self.node_limit) < 0:
            raise ValueError("node_limit must be >= 0 when provided.")
        solver = str(self.solver).strip().lower()
        if solver not in {"choose", "simplex", "ipm"}:
            raise ValueError("solver must be one of: choose, simplex, ipm.")
        object.__setattr__(self, "solver", solver)
        _finite_non_negative(
            "throughput_tiebreaker_eur_per_kwh",
            self.throughput_tiebreaker_eur_per_kwh,
        )
        _finite_non_negative(
            "lexicographic_shortfall_tolerance_kwh",
            self.lexicographic_shortfall_tolerance_kwh,
        )


@dataclass(frozen=True)
class SemanticActionSeries:
    """A schedule series identified independently of action-vector position."""

    building_id: str
    action_name: str
    values: tuple[float, ...]
    unit: str = "kW"
    positive_direction: str = "charge"

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "action_name": self.action_name,
            "values": list(self.values),
            "unit": self.unit,
            "positive_direction": self.positive_direction,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticActionSeries":
        data = dict(payload)
        data["values"] = tuple(float(value) for value in data["values"])
        return cls(**data)


@dataclass(frozen=True)
class SemanticSchedule:
    """Replay-oriented schedule with stable semantic action identifiers."""

    problem_id: str
    horizon: int
    timestep_hours: float
    series: tuple[SemanticActionSeries, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0.")
        keys: set[tuple[str, str]] = set()
        for item in self.series:
            key = item.building_id, item.action_name
            if key in keys:
                raise ValueError(f"Duplicate schedule series: {key!r}.")
            if len(item.values) != self.horizon:
                raise ValueError(
                    f"Schedule series {key!r} has {len(item.values)} values; expected {self.horizon}."
                )
            if not all(math.isfinite(value) for value in item.values):
                raise ValueError(f"Schedule series {key!r} contains non-finite values.")
            keys.add(key)
        object.__setattr__(self, "series", tuple(self.series))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "horizon": self.horizon,
            "timestep_hours": self.timestep_hours,
            "series": [item.to_dict() for item in self.series],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticSchedule":
        data = dict(payload)
        data["series"] = tuple(SemanticActionSeries.from_dict(item) for item in data["series"])
        return cls(**data)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "SemanticSchedule":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class SolverInfo:
    status: str
    status_code: int
    optimal: bool
    has_solution: bool
    message: str
    solver_objective_eur: Optional[float]
    dual_bound_eur: Optional[float]
    mip_gap: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "status_code": self.status_code,
            "optimal": self.optimal,
            "has_solution": self.has_solution,
            "message": self.message,
            "solver_objective_eur": self.solver_objective_eur,
            "dual_bound_eur": self.dual_bound_eur,
            "mip_gap": self.mip_gap,
        }


@dataclass(frozen=True)
class BatteryTrajectory:
    building_id: str
    action_name: str
    charge_kwh: tuple[float, ...]
    discharge_kwh: tuple[float, ...]
    state_of_charge_kwh: tuple[float, ...]
    spill_kwh: Optional[tuple[float, ...]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "action_name": self.action_name,
            "charge_kwh": list(self.charge_kwh),
            "discharge_kwh": list(self.discharge_kwh),
            "state_of_charge_kwh": list(self.state_of_charge_kwh),
            "spill_kwh": None if self.spill_kwh is None else list(self.spill_kwh),
        }


@dataclass(frozen=True)
class FormulationResult:
    formulation: str
    solver: SolverInfo
    community_cost_eur: Optional[float]
    grid_import_kwh: Optional[tuple[float, ...]]
    battery_trajectories: tuple[BatteryTrajectory, ...]
    schedule: Optional[SemanticSchedule]

    def to_dict(self) -> dict[str, Any]:
        return {
            "formulation": self.formulation,
            "solver": self.solver.to_dict(),
            "community_cost_eur": self.community_cost_eur,
            "grid_import_kwh": None if self.grid_import_kwh is None else list(self.grid_import_kwh),
            "battery_trajectories": [item.to_dict() for item in self.battery_trajectories],
            "schedule": None if self.schedule is None else self.schedule.to_dict(),
        }


@dataclass(frozen=True)
class BoundedOracleResult:
    """Lower/upper certificate for the supplied pair of linear models."""

    problem_id: str
    lower: FormulationResult
    conservative: FormulationResult
    certified_lower_bound_eur: Optional[float]
    model_feasible_upper_bound_eur: Optional[float]
    absolute_gap_eur: Optional[float]
    relative_gap: Optional[float]
    certificate_valid: bool
    guarantee: str = (
        "Bounds apply to the supplied optimistic/conservative linear models. "
        "The upper schedule requires exact simulator replay before it can be called CityLearn-feasible."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "certified_lower_bound_eur": self.certified_lower_bound_eur,
            "model_feasible_upper_bound_eur": self.model_feasible_upper_bound_eur,
            "absolute_gap_eur": self.absolute_gap_eur,
            "relative_gap": self.relative_gap,
            "certificate_valid": self.certificate_valid,
            "guarantee": self.guarantee,
            "lower": self.lower.to_dict(),
            "conservative": self.conservative.to_dict(),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class _VariableLayout:
    n_batteries: int
    horizon: int
    include_spill: bool
    include_direction: bool
    charge_start: int
    discharge_start: int
    soc_start: int
    grid_start: int
    spill_start: Optional[int]
    direction_start: Optional[int]
    size: int

    @classmethod
    def build(cls, n_batteries: int, horizon: int, *, lower_bound: bool) -> "_VariableLayout":
        cursor = 0
        charge_start = cursor
        cursor += n_batteries * horizon
        discharge_start = cursor
        cursor += n_batteries * horizon
        soc_start = cursor
        cursor += n_batteries * (horizon + 1)
        grid_start = cursor
        cursor += horizon
        spill_start: Optional[int] = None
        direction_start: Optional[int] = None
        if lower_bound:
            spill_start = cursor
            cursor += n_batteries * horizon
        else:
            direction_start = cursor
            cursor += n_batteries * horizon
        return cls(
            n_batteries=n_batteries,
            horizon=horizon,
            include_spill=lower_bound,
            include_direction=not lower_bound,
            charge_start=charge_start,
            discharge_start=discharge_start,
            soc_start=soc_start,
            grid_start=grid_start,
            spill_start=spill_start,
            direction_start=direction_start,
            size=cursor,
        )

    def charge(self, battery: int, time_step: int) -> int:
        return self.charge_start + battery * self.horizon + time_step

    def discharge(self, battery: int, time_step: int) -> int:
        return self.discharge_start + battery * self.horizon + time_step

    def soc(self, battery: int, time_step: int) -> int:
        return self.soc_start + battery * (self.horizon + 1) + time_step

    def grid(self, time_step: int) -> int:
        return self.grid_start + time_step

    def spill(self, battery: int, time_step: int) -> int:
        assert self.spill_start is not None
        return self.spill_start + battery * self.horizon + time_step

    def direction(self, battery: int, time_step: int) -> int:
        assert self.direction_start is not None
        return self.direction_start + battery * self.horizon + time_step


_STATUS_NAMES = {
    0: "optimal",
    1: "limit_reached",
    2: "infeasible",
    3: "unbounded",
    4: "solver_error",
}


def _optional_finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _solver_options(options: SolveOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "presolve": bool(options.presolve),
        "disp": bool(options.display_solver_output),
    }
    if options.solver != "choose":
        payload["solver"] = options.solver
    if options.time_limit_seconds is not None:
        payload["time_limit"] = float(options.time_limit_seconds)
    if options.mip_relative_gap is not None:
        payload["mip_rel_gap"] = float(options.mip_relative_gap)
    if options.node_limit is not None:
        payload["node_limit"] = int(options.node_limit)
    return payload


def _milp_primal_only(
    *,
    c: np.ndarray,
    integrality: np.ndarray,
    bounds: Bounds,
    constraints: LinearConstraint,
    options: Mapping[str, Any],
    x0: Optional[np.ndarray] = None,
) -> OptimizeResult:
    """Run bundled HiGHS without extracting unused LP basis marginals.

    SciPy 1.15.1's public :func:`scipy.optimize.milp` wrapper iterates over
    every column through pybind after HiGHS has solved the model in order to
    construct bound marginals.  ``milp`` does not expose those marginals, and
    on an annual 15-minute oracle that conversion takes hours even though the
    LP itself takes seconds.  The oracle only needs the primal solution,
    objective and MIP certificate, so use the same bundled solver while
    intentionally omitting that discarded basis conversion.

    Builds of SciPy without the bundled private bindings retain the public
    fallback.  The project pins SciPy 1.15.1 on the x86 experiment workers.
    """

    if _highs_core is None or _milp_iv is None:
        return milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=dict(options),
        )

    prepared_options = dict(options)
    # ``solver`` is a valid native HiGHS option but not listed by SciPy's
    # public MILP validator. Route it explicitly to avoid a misleading
    # unrecognized-option warning on the private fast path.
    solver_choice = prepared_options.pop("solver", None)
    prepared = _milp_iv(
        c,
        integrality,
        bounds,
        constraints,
        prepared_options,
    )
    (
        objective,
        variable_types,
        lower_bounds,
        upper_bounds,
        indptr,
        indices,
        data,
        constraint_lower,
        constraint_upper,
        highs_option_values,
    ) = prepared
    if solver_choice is not None:
        highs_option_values["solver"] = solver_choice

    model = _highs_core.HighsLp()
    model.num_col_ = int(objective.size)
    model.num_row_ = int(constraint_upper.size)
    model.a_matrix_.num_col_ = model.num_col_
    model.a_matrix_.num_row_ = model.num_row_
    model.a_matrix_.format_ = _highs_core.MatrixFormat.kColwise
    model.col_cost_ = objective
    model.col_lower_ = lower_bounds
    model.col_upper_ = upper_bounds
    model.row_lower_ = constraint_lower
    model.row_upper_ = constraint_upper
    model.a_matrix_.start_ = indptr
    model.a_matrix_.index_ = indices
    model.a_matrix_.value_ = data

    is_mip = bool(np.any(variable_types))
    if is_mip:
        model.integrality_ = [
            _highs_core.HighsVarType(int(value)) for value in variable_types
        ]

    solver = _highs_core._Highs()
    highs_options = _highs_core.HighsOptions()
    for key, value in highs_option_values.items():
        if value is None:
            continue
        if key in {"presolve", "parallel"} and isinstance(value, bool):
            value = "on" if value else "off"
        setattr(highs_options, key, value)

    option_status = solver.passOptions(highs_options)
    if option_status == _highs_core.HighsStatus.kError:
        return OptimizeResult(
            status=4,
            message="HiGHS rejected solver options.",
            x=None,
            fun=None,
            mip_dual_bound=None,
            mip_gap=None,
        )
    if solver.passModel(model) == _highs_core.HighsStatus.kError:
        return OptimizeResult(
            status=4,
            message="HiGHS rejected the oracle model.",
            x=None,
            fun=None,
            mip_dual_bound=None,
            mip_gap=None,
        )
    if x0 is not None:
        initial = np.asarray(x0, dtype=np.float64).reshape(-1)
        if initial.shape != (model.num_col_,) or not np.all(np.isfinite(initial)):
            return OptimizeResult(
                status=4,
                message="Invalid initial primal vector.",
                x=None,
                fun=None,
                mip_dual_bound=None,
                mip_gap=None,
            )
        warm_status = solver.setSolution(
            model.num_col_,
            np.arange(model.num_col_, dtype=np.int32),
            initial,
        )
        if warm_status == _highs_core.HighsStatus.kError:
            return OptimizeResult(
                status=4,
                message="HiGHS rejected the initial primal vector.",
                x=None,
                fun=None,
                mip_dual_bound=None,
                mip_gap=None,
            )
    if solver.run() == _highs_core.HighsStatus.kError:
        return OptimizeResult(
            status=4,
            message="HiGHS failed while solving the oracle model.",
            x=None,
            fun=None,
            mip_dual_bound=None,
            mip_gap=None,
        )

    model_status = solver.getModelStatus()
    message = solver.modelStatusToString(model_status)
    if model_status == _highs_core.HighsModelStatus.kOptimal:
        status = 0
    elif model_status in {
        _highs_core.HighsModelStatus.kTimeLimit,
        _highs_core.HighsModelStatus.kIterationLimit,
        _highs_core.HighsModelStatus.kSolutionLimit,
        _highs_core.HighsModelStatus.kObjectiveBound,
        _highs_core.HighsModelStatus.kObjectiveTarget,
    }:
        status = 1
    elif model_status == _highs_core.HighsModelStatus.kInfeasible:
        status = 2
    elif model_status == _highs_core.HighsModelStatus.kUnbounded:
        status = 3
    else:
        status = 4

    info = solver.getInfo()
    objective_value = _optional_finite(info.objective_function_value)
    # HiGHS can hold a valid primal point when either an LP or MIP reaches a
    # limit.  The old code accepted that point only for MIPs, discarding large
    # continuous-oracle solutions even when HiGHS reported a finite objective.
    candidate_solution = solver.getSolution() if status in {0, 1} else None
    candidate_vector = (
        None
        if candidate_solution is None
        or not getattr(candidate_solution, "value_valid", False)
        else np.asarray(candidate_solution.col_value, dtype=np.float64)
    )
    matrix = csc_array(
        (data, indices, indptr),
        shape=(int(constraint_upper.size), int(objective.size)),
    )

    def primal_feasible(vector: Optional[np.ndarray]) -> bool:
        if (
            vector is None
            or vector.shape != objective.shape
            or not np.all(np.isfinite(vector))
        ):
            return False
        finite_lower = np.isfinite(lower_bounds)
        finite_upper = np.isfinite(upper_bounds)
        lower_tolerance = 1.0e-5 + 1.0e-8 * np.maximum(
            1.0,
            np.abs(lower_bounds[finite_lower]),
        )
        upper_tolerance = 1.0e-5 + 1.0e-8 * np.maximum(
            1.0,
            np.abs(upper_bounds[finite_upper]),
        )
        if np.any(
            vector[finite_lower]
            < lower_bounds[finite_lower] - lower_tolerance
        ) or np.any(
            vector[finite_upper]
            > upper_bounds[finite_upper] + upper_tolerance
        ):
            return False
        integer_columns = np.flatnonzero(np.asarray(variable_types) != 0)
        if integer_columns.size and np.any(
            np.abs(vector[integer_columns] - np.rint(vector[integer_columns]))
            > 1.0e-5
        ):
            return False
        lhs = np.asarray(matrix @ vector).reshape(-1)
        finite_row_lower = np.isfinite(constraint_lower)
        finite_row_upper = np.isfinite(constraint_upper)
        row_lower_tolerance = 1.0e-5 + 1.0e-8 * np.maximum(
            1.0,
            np.abs(constraint_lower[finite_row_lower]),
        )
        row_upper_tolerance = 1.0e-5 + 1.0e-8 * np.maximum(
            1.0,
            np.abs(constraint_upper[finite_row_upper]),
        )
        return bool(
            not np.any(
                lhs[finite_row_lower]
                < constraint_lower[finite_row_lower]
                - row_lower_tolerance
            )
            and not np.any(
                lhs[finite_row_upper]
                > constraint_upper[finite_row_upper]
                + row_upper_tolerance
            )
        )

    initial_vector = None if x0 is None else np.asarray(x0, dtype=np.float64)
    candidate_primal_feasible = primal_feasible(candidate_vector)
    initial_primal_feasible = primal_feasible(initial_vector)
    feasible_candidates = []
    if candidate_primal_feasible:
        feasible_candidates.append(("solver", candidate_vector))
    if initial_primal_feasible:
        feasible_candidates.append(("initial", initial_vector))
    selected = (
        min(
            feasible_candidates,
            key=lambda item: float(np.dot(objective, item[1])),
        )
        if feasible_candidates
        else None
    )
    selected_primal_source = "none" if selected is None else selected[0]
    x = None if selected is None else selected[1]
    if x is not None:
        objective_value = float(np.dot(objective, x))
    return OptimizeResult(
        status=status,
        message=message,
        x=x,
        fun=objective_value,
        mip_dual_bound=(
            _optional_finite(info.mip_dual_bound) if is_mip else None
        ),
        mip_gap=_optional_finite(info.mip_gap) if is_mip else None,
        candidate_primal_feasible=candidate_primal_feasible,
        initial_primal_feasible=initial_primal_feasible,
        selected_primal_source=selected_primal_source,
    )


def _build_and_solve(
    problem: PerfectForesightProblem,
    *,
    lower_bound: bool,
    options: SolveOptions,
) -> FormulationResult:
    horizon = problem.horizon
    batteries = problem.batteries
    layout = _VariableLayout.build(len(batteries), horizon, lower_bound=lower_bound)

    objective = np.zeros(layout.size, dtype=np.float64)
    objective[layout.grid_start : layout.grid_start + horizon] = problem.price_eur_per_kwh
    if not lower_bound and options.throughput_tiebreaker_eur_per_kwh > 0.0:
        tie = float(options.throughput_tiebreaker_eur_per_kwh)
        objective[layout.charge_start : layout.discharge_start] = tie
        objective[layout.discharge_start : layout.soc_start] = tie

    lower_bounds = np.zeros(layout.size, dtype=np.float64)
    upper_bounds = np.full(layout.size, np.inf, dtype=np.float64)
    integrality = np.zeros(layout.size, dtype=np.int32)

    for battery_index, battery in enumerate(batteries):
        model = battery.optimistic if lower_bound else battery.conservative
        charge_limit = model.max_charge_kw * problem.timestep_hours
        discharge_limit = model.max_discharge_kw * problem.timestep_hours
        for time_step in range(horizon):
            upper_bounds[layout.charge(battery_index, time_step)] = charge_limit
            upper_bounds[layout.discharge(battery_index, time_step)] = discharge_limit
            if not lower_bound:
                direction_index = layout.direction(battery_index, time_step)
                upper_bounds[direction_index] = 1.0
                integrality[direction_index] = 1
        for time_step in range(horizon + 1):
            upper_bounds[layout.soc(battery_index, time_step)] = model.capacity_kwh
        initial_index = layout.soc(battery_index, 0)
        lower_bounds[initial_index] = battery.initial_energy_kwh
        upper_bounds[initial_index] = battery.initial_energy_kwh
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
        model = battery.optimistic if lower_bound else battery.conservative
        charge_limit = model.max_charge_kw * problem.timestep_hours
        discharge_limit = model.max_discharge_kw * problem.timestep_hours
        for time_step in range(horizon):
            dynamics = [
                (layout.soc(battery_index, time_step + 1), 1.0),
                (layout.soc(battery_index, time_step), -1.0),
                (layout.charge(battery_index, time_step), -model.charge_efficiency),
                (layout.discharge(battery_index, time_step), 1.0 / model.discharge_efficiency),
            ]
            if lower_bound:
                dynamics.append((layout.spill(battery_index, time_step), 1.0))
            add_row(dynamics, 0.0, 0.0)

            if not lower_bound:
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
    for time_step in range(horizon):
        entries: list[tuple[int, float]] = [(layout.grid(time_step), 1.0)]
        for battery_index in range(len(batteries)):
            entries.append((layout.charge(battery_index, time_step), -1.0))
            entries.append((layout.discharge(battery_index, time_step), 1.0))
        add_row(entries, float(district_base[time_step]), np.inf)

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
    constraints = LinearConstraint(
        constraint_matrix,
        np.asarray(constraint_lower, dtype=np.float64),
        np.asarray(constraint_upper, dtype=np.float64),
    )

    raw_result = _milp_primal_only(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options=_solver_options(options),
    )

    status_code = int(raw_result.status)
    solution = getattr(raw_result, "x", None)
    has_solution = solution is not None and np.all(np.isfinite(solution))
    solver_objective = _optional_finite(getattr(raw_result, "fun", None))
    dual_bound = _optional_finite(getattr(raw_result, "mip_dual_bound", None))
    if status_code == 0 and dual_bound is None:
        dual_bound = solver_objective
    solver_info = SolverInfo(
        status=_STATUS_NAMES.get(status_code, "unknown"),
        status_code=status_code,
        optimal=status_code == 0,
        has_solution=bool(has_solution),
        message=str(raw_result.message),
        solver_objective_eur=solver_objective,
        dual_bound_eur=dual_bound,
        mip_gap=_optional_finite(getattr(raw_result, "mip_gap", None)),
    )

    if not has_solution:
        return FormulationResult(
            formulation="optimistic_lower_bound_lp" if lower_bound else "conservative_schedule_milp",
            solver=solver_info,
            community_cost_eur=None,
            grid_import_kwh=None,
            battery_trajectories=(),
            schedule=None,
        )

    solution = np.asarray(solution, dtype=np.float64)
    grid = solution[layout.grid_start : layout.grid_start + horizon]
    official_cost = float(np.dot(problem.price_eur_per_kwh, grid))
    trajectories: list[BatteryTrajectory] = []
    schedule_series: list[SemanticActionSeries] = []
    for battery_index, battery in enumerate(batteries):
        charge = solution[
            layout.charge(battery_index, 0) : layout.charge(battery_index, 0) + horizon
        ]
        discharge = solution[
            layout.discharge(battery_index, 0) : layout.discharge(battery_index, 0) + horizon
        ]
        soc = solution[layout.soc(battery_index, 0) : layout.soc(battery_index, 0) + horizon + 1]
        spill: Optional[np.ndarray]
        if lower_bound:
            spill = solution[
                layout.spill(battery_index, 0) : layout.spill(battery_index, 0) + horizon
            ]
        else:
            spill = None
        trajectories.append(
            BatteryTrajectory(
                building_id=battery.building_id,
                action_name=battery.action_name,
                charge_kwh=tuple(float(value) for value in charge),
                discharge_kwh=tuple(float(value) for value in discharge),
                state_of_charge_kwh=tuple(float(value) for value in soc),
                spill_kwh=None if spill is None else tuple(float(value) for value in spill),
            )
        )
        if not lower_bound:
            signed_power = (charge - discharge) / problem.timestep_hours
            signed_power[np.abs(signed_power) < _NUMERICAL_TOLERANCE] = 0.0
            schedule_series.append(
                SemanticActionSeries(
                    building_id=battery.building_id,
                    action_name=battery.action_name,
                    values=tuple(float(value) for value in signed_power),
                )
            )

    schedule = None
    if not lower_bound:
        schedule = SemanticSchedule(
            problem_id=problem.problem_id,
            horizon=horizon,
            timestep_hours=problem.timestep_hours,
            series=tuple(schedule_series),
            metadata={
                "formulation": "conservative_linear_milp",
                "cost_semantics": "district_import_with_zero_export_credit",
                "requires_citylearn_replay": True,
            },
        )

    return FormulationResult(
        formulation="optimistic_lower_bound_lp" if lower_bound else "conservative_schedule_milp",
        solver=solver_info,
        community_cost_eur=official_cost,
        grid_import_kwh=tuple(float(value) for value in grid),
        battery_trajectories=tuple(trajectories),
        schedule=schedule,
    )


def solve_optimistic_lower_bound(
    problem: PerfectForesightProblem,
    options: Optional[SolveOptions] = None,
) -> FormulationResult:
    """Solve the optimistic LP relaxation."""

    return _build_and_solve(problem, lower_bound=True, options=options or SolveOptions())


def solve_conservative_schedule(
    problem: PerfectForesightProblem,
    options: Optional[SolveOptions] = None,
) -> FormulationResult:
    """Solve the conservative MILP and emit a semantic signed-power schedule."""

    return _build_and_solve(problem, lower_bound=False, options=options or SolveOptions())


def solve_bounded_oracle(
    problem: PerfectForesightProblem,
    options: Optional[SolveOptions] = None,
) -> BoundedOracleResult:
    """Solve both formulations and construct their explicit bound certificate."""

    options = options or SolveOptions()
    lower = solve_optimistic_lower_bound(problem, options)
    conservative = solve_conservative_schedule(problem, options)

    certified_lower: Optional[float] = None
    if lower.solver.dual_bound_eur is not None:
        certified_lower = lower.solver.dual_bound_eur
    elif lower.solver.optimal:
        certified_lower = lower.community_cost_eur

    upper = conservative.community_cost_eur if conservative.solver.has_solution else None
    gap: Optional[float] = None
    relative_gap: Optional[float] = None
    certificate_valid = certified_lower is not None and upper is not None
    if certificate_valid:
        assert certified_lower is not None and upper is not None
        raw_gap = upper - certified_lower
        tolerance = _NUMERICAL_TOLERANCE * max(1.0, abs(upper), abs(certified_lower))
        if raw_gap < -tolerance:
            raise RuntimeError(
                "Optimistic lower bound exceeds the conservative feasible cost; "
                "the dominance contract or model construction is inconsistent."
            )
        gap = max(raw_gap, 0.0)
        relative_gap = 0.0 if gap == 0.0 else gap / max(abs(upper), _NUMERICAL_TOLERANCE)

    return BoundedOracleResult(
        problem_id=problem.problem_id,
        lower=lower,
        conservative=conservative,
        certified_lower_bound_eur=certified_lower,
        model_feasible_upper_bound_eur=upper,
        absolute_gap_eur=gap,
        relative_gap=relative_gap,
        certificate_valid=certificate_valid,
    )
