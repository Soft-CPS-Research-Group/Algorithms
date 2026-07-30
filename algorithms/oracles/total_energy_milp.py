"""Perfect-foresight MILP for complete local and community electricity control.

Unlike :mod:`citylearn_fixed_service`, this formulation does not freeze EV or
deferrable-appliance service.  It jointly schedules stationary storage, EV
charging/V2G sessions and fixed-profile deferrable cycles while enforcing
building import/export and phase limits.

Two settlement semantics are supported:

``individual``
    Every building pays for its own positive grid import.  Export has no
    credit.  This is the correct theoretical comparator for building-local
    PPO/TD3 policies.

``community``
    Positive import is calculated after netting all building exchanges.  The
    same physical building and phase limits still apply.  This is the joint
    comparator for future community controllers and MARL policies.

The relaxed formulation is a structural lower bound for the supplied linear
model.  The mixed-integer schedule is a model-feasible upper candidate and
must still be replayed in CityLearn before it is called simulator-feasible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from algorithms.oracles.perfect_foresight_milp import (
    SemanticActionSeries,
    SemanticSchedule,
    SolveOptions,
    SolverInfo,
)


_TOLERANCE = 1.0e-8
_PHASES = ("L1", "L2", "L3")
_STATUS_NAMES = {
    0: "optimal",
    1: "limit_reached",
    2: "infeasible",
    3: "unbounded",
    4: "solver_error",
}


def _finite(name: str, value: Any, *, minimum: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric; got {value!r}.") from error
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        suffix = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must be {suffix}; got {value!r}.")
    return parsed


def _efficiency(name: str, value: Any) -> float:
    parsed = _finite(name, value, minimum=0.0)
    if parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in (0, 1]; got {value!r}.")
    return parsed


def _phase_connection(value: str) -> str:
    normalized = str(value).strip()
    if normalized not in {*_PHASES, "all_phases"}:
        raise ValueError(
            f"phase_connection must be one of L1/L2/L3/all_phases; got {value!r}."
        )
    return normalized


@dataclass(frozen=True)
class StorageAsset:
    """Linear grid-side model for a stationary battery."""

    building_id: str
    action_name: str
    capacity_kwh: float
    initial_energy_kwh: float
    final_energy_min_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    loss_coefficient: float = 0.0
    minimum_energy_kwh: float = 0.0
    phase_connection: str = "all_phases"

    def __post_init__(self) -> None:
        if not str(self.building_id).strip() or not str(self.action_name).strip():
            raise ValueError("Storage building_id and action_name must be non-empty.")
        capacity = _finite("capacity_kwh", self.capacity_kwh, minimum=0.0)
        minimum = _finite("minimum_energy_kwh", self.minimum_energy_kwh, minimum=0.0)
        initial = _finite("initial_energy_kwh", self.initial_energy_kwh, minimum=0.0)
        final_min = _finite(
            "final_energy_min_kwh", self.final_energy_min_kwh, minimum=0.0
        )
        if minimum > capacity + _TOLERANCE:
            raise ValueError("minimum_energy_kwh exceeds capacity_kwh.")
        if not minimum - _TOLERANCE <= initial <= capacity + _TOLERANCE:
            raise ValueError("initial_energy_kwh is outside battery energy bounds.")
        if not minimum - _TOLERANCE <= final_min <= capacity + _TOLERANCE:
            raise ValueError("final_energy_min_kwh is outside battery energy bounds.")
        object.__setattr__(self, "building_id", str(self.building_id))
        object.__setattr__(self, "action_name", str(self.action_name))
        object.__setattr__(self, "capacity_kwh", capacity)
        object.__setattr__(self, "minimum_energy_kwh", minimum)
        object.__setattr__(self, "initial_energy_kwh", initial)
        object.__setattr__(self, "final_energy_min_kwh", final_min)
        object.__setattr__(
            self, "max_charge_kw", _finite("max_charge_kw", self.max_charge_kw, minimum=0.0)
        )
        object.__setattr__(
            self,
            "max_discharge_kw",
            _finite("max_discharge_kw", self.max_discharge_kw, minimum=0.0),
        )
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
        loss = _finite("loss_coefficient", self.loss_coefficient, minimum=0.0)
        if loss > 1.0:
            raise ValueError("loss_coefficient must be <= 1.")
        object.__setattr__(self, "loss_coefficient", loss)
        object.__setattr__(self, "phase_connection", _phase_connection(self.phase_connection))

    def to_dict(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class EVSession:
    """One contiguous charger connection with an arrival and departure target."""

    session_id: str
    building_id: str
    action_name: str
    start_time_step: int
    end_time_step: int
    capacity_kwh: float
    initial_energy_kwh: float
    required_final_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    min_charge_kw: float = 0.0
    min_discharge_kw: float = 0.0
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    loss_coefficient: float = 0.0
    phase_connection: str = "all_phases"
    allow_departure_shortfall: bool = False

    def __post_init__(self) -> None:
        if not str(self.session_id).strip():
            raise ValueError("session_id must be non-empty.")
        if not str(self.building_id).strip() or not str(self.action_name).strip():
            raise ValueError("EV building_id and action_name must be non-empty.")
        start = int(self.start_time_step)
        end = int(self.end_time_step)
        if start < 0 or end < start:
            raise ValueError("EV session must satisfy 0 <= start_time_step <= end_time_step.")
        capacity = _finite("capacity_kwh", self.capacity_kwh, minimum=0.0)
        minimum = _finite("minimum_energy_kwh", self.minimum_energy_kwh, minimum=0.0)
        initial = _finite("initial_energy_kwh", self.initial_energy_kwh, minimum=0.0)
        required = _finite(
            "required_final_energy_kwh", self.required_final_energy_kwh, minimum=0.0
        )
        if minimum > capacity + _TOLERANCE:
            raise ValueError("EV minimum_energy_kwh exceeds capacity_kwh.")
        if not minimum - _TOLERANCE <= initial <= capacity + _TOLERANCE:
            raise ValueError("EV initial_energy_kwh is outside its energy bounds.")
        if not minimum - _TOLERANCE <= required <= capacity + _TOLERANCE:
            raise ValueError("EV required_final_energy_kwh is outside its energy bounds.")
        max_charge = _finite("max_charge_kw", self.max_charge_kw, minimum=0.0)
        max_discharge = _finite("max_discharge_kw", self.max_discharge_kw, minimum=0.0)
        min_charge = _finite("min_charge_kw", self.min_charge_kw, minimum=0.0)
        min_discharge = _finite("min_discharge_kw", self.min_discharge_kw, minimum=0.0)
        if min_charge > max_charge + _TOLERANCE:
            raise ValueError("EV min_charge_kw exceeds max_charge_kw.")
        if min_discharge > max_discharge + _TOLERANCE:
            raise ValueError("EV min_discharge_kw exceeds max_discharge_kw.")
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "building_id", str(self.building_id))
        object.__setattr__(self, "action_name", str(self.action_name))
        object.__setattr__(self, "start_time_step", start)
        object.__setattr__(self, "end_time_step", end)
        object.__setattr__(self, "capacity_kwh", capacity)
        object.__setattr__(self, "minimum_energy_kwh", minimum)
        object.__setattr__(self, "initial_energy_kwh", initial)
        object.__setattr__(self, "required_final_energy_kwh", required)
        object.__setattr__(self, "max_charge_kw", max_charge)
        object.__setattr__(self, "max_discharge_kw", max_discharge)
        object.__setattr__(self, "min_charge_kw", min_charge)
        object.__setattr__(self, "min_discharge_kw", min_discharge)
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
        loss = _finite("loss_coefficient", self.loss_coefficient, minimum=0.0)
        if loss > 1.0:
            raise ValueError("loss_coefficient must be <= 1.")
        object.__setattr__(self, "loss_coefficient", loss)
        object.__setattr__(self, "phase_connection", _phase_connection(self.phase_connection))
        object.__setattr__(
            self, "allow_departure_shortfall", bool(self.allow_departure_shortfall)
        )

    @property
    def duration(self) -> int:
        return self.end_time_step - self.start_time_step + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class DeferrableCycle:
    """A fixed-profile appliance cycle and its feasible start window."""

    cycle_id: str
    building_id: str
    action_name: str
    earliest_start_time_step: int
    latest_start_time_step: int
    load_profile_kwh: tuple[float, ...]
    must_run: bool = True
    phase_connection: str = "all_phases"

    def __post_init__(self) -> None:
        if not str(self.cycle_id).strip():
            raise ValueError("cycle_id must be non-empty.")
        if not str(self.building_id).strip() or not str(self.action_name).strip():
            raise ValueError("Deferrable building_id and action_name must be non-empty.")
        earliest = int(self.earliest_start_time_step)
        latest = int(self.latest_start_time_step)
        if earliest < 0 or latest < earliest:
            raise ValueError("Deferrable cycle must satisfy 0 <= earliest <= latest.")
        profile = tuple(
            _finite("load_profile_kwh", value, minimum=0.0)
            for value in self.load_profile_kwh
        )
        if not profile:
            raise ValueError("Deferrable load_profile_kwh must not be empty.")
        object.__setattr__(self, "cycle_id", str(self.cycle_id))
        object.__setattr__(self, "building_id", str(self.building_id))
        object.__setattr__(self, "action_name", str(self.action_name))
        object.__setattr__(self, "earliest_start_time_step", earliest)
        object.__setattr__(self, "latest_start_time_step", latest)
        object.__setattr__(self, "load_profile_kwh", profile)
        object.__setattr__(self, "must_run", bool(self.must_run))
        object.__setattr__(self, "phase_connection", _phase_connection(self.phase_connection))

    def to_dict(self) -> dict[str, Any]:
        return {
            field_name: list(getattr(self, field_name))
            if field_name == "load_profile_kwh"
            else getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class ElectricalService:
    """Signed building and per-phase power limits in kW."""

    building_id: str
    total_import_kw: Optional[float] = None
    total_export_kw: Optional[float] = None
    phase_import_kw: Mapping[str, Optional[float]] = field(default_factory=dict)
    phase_export_kw: Mapping[str, Optional[float]] = field(default_factory=dict)
    default_split: str = "balanced"

    def __post_init__(self) -> None:
        if not str(self.building_id).strip():
            raise ValueError("ElectricalService building_id must be non-empty.")
        split = str(self.default_split).strip().lower()
        if split not in {"balanced", "l1", "l2", "l3"}:
            raise ValueError("default_split must be balanced or L1/L2/L3.")

        def normalize_limits(
            name: str, values: Mapping[str, Optional[float]]
        ) -> dict[str, Optional[float]]:
            unknown = set(values) - set(_PHASES)
            if unknown:
                raise ValueError(f"{name} contains unknown phases: {sorted(unknown)}.")
            return {
                phase: None
                if values.get(phase) is None
                else _finite(f"{name}.{phase}", values[phase], minimum=0.0)
                for phase in _PHASES
            }

        object.__setattr__(self, "building_id", str(self.building_id))
        object.__setattr__(self, "default_split", split)
        object.__setattr__(
            self,
            "total_import_kw",
            None
            if self.total_import_kw is None
            else _finite("total_import_kw", self.total_import_kw, minimum=0.0),
        )
        object.__setattr__(
            self,
            "total_export_kw",
            None
            if self.total_export_kw is None
            else _finite("total_export_kw", self.total_export_kw, minimum=0.0),
        )
        object.__setattr__(
            self, "phase_import_kw", normalize_limits("phase_import_kw", self.phase_import_kw)
        )
        object.__setattr__(
            self, "phase_export_kw", normalize_limits("phase_export_kw", self.phase_export_kw)
        )

    def fractions(self, connection: Optional[str]) -> Mapping[str, float]:
        if connection in _PHASES:
            return {phase: 1.0 if phase == connection else 0.0 for phase in _PHASES}
        if connection == "all_phases" or self.default_split == "balanced":
            return {phase: 1.0 / 3.0 for phase in _PHASES}
        selected = self.default_split.upper()
        return {phase: 1.0 if phase == selected else 0.0 for phase in _PHASES}

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "total_import_kw": self.total_import_kw,
            "total_export_kw": self.total_export_kw,
            "phase_import_kw": dict(self.phase_import_kw),
            "phase_export_kw": dict(self.phase_export_kw),
            "default_split": self.default_split,
        }


@dataclass(frozen=True)
class TotalEnergyProblem:
    """Complete flexible-electricity optimization problem."""

    problem_id: str
    timestep_hours: float
    building_ids: tuple[str, ...]
    price_eur_per_kwh: np.ndarray
    base_net_load_kwh: np.ndarray
    settlement: Literal["individual", "community"] = "individual"
    stationary_storage: tuple[StorageAsset, ...] = ()
    ev_sessions: tuple[EVSession, ...] = ()
    deferrable_cycles: tuple[DeferrableCycle, ...] = ()
    electrical_services: tuple[ElectricalService, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.problem_id).strip():
            raise ValueError("problem_id must be non-empty.")
        timestep = _finite("timestep_hours", self.timestep_hours, minimum=0.0)
        if timestep <= 0.0:
            raise ValueError("timestep_hours must be > 0.")
        buildings = tuple(str(value) for value in self.building_ids)
        if not buildings or any(not value.strip() for value in buildings):
            raise ValueError("building_ids must contain non-empty identifiers.")
        if len(set(buildings)) != len(buildings):
            raise ValueError("building_ids must be unique.")
        prices = np.asarray(self.price_eur_per_kwh, dtype=np.float64)
        base = np.asarray(self.base_net_load_kwh, dtype=np.float64)
        if prices.ndim != 1 or prices.size == 0:
            raise ValueError("price_eur_per_kwh must be a non-empty 1-D array.")
        if np.any(~np.isfinite(prices)) or np.any(prices < 0.0):
            raise ValueError("price_eur_per_kwh must be finite and non-negative.")
        if base.shape != (len(buildings), prices.size) or np.any(~np.isfinite(base)):
            raise ValueError(
                "base_net_load_kwh must be finite with shape "
                f"({len(buildings)}, {prices.size}); got {base.shape}."
            )
        if self.settlement not in {"individual", "community"}:
            raise ValueError("settlement must be 'individual' or 'community'.")
        building_set = set(buildings)
        for asset in (*self.stationary_storage, *self.ev_sessions, *self.deferrable_cycles):
            if asset.building_id not in building_set:
                raise ValueError(f"Unknown flexible-asset building_id {asset.building_id!r}.")
        storage_keys = [
            (asset.building_id, asset.action_name) for asset in self.stationary_storage
        ]
        if len(set(storage_keys)) != len(storage_keys):
            raise ValueError("Stationary-storage semantic keys must be unique.")
        session_ids = [session.session_id for session in self.ev_sessions]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("EV session_id values must be unique.")
        sessions_by_action: dict[tuple[str, str], list[EVSession]] = {}
        for session in self.ev_sessions:
            sessions_by_action.setdefault(
                (session.building_id, session.action_name), []
            ).append(session)
        for semantic_key, sessions in sessions_by_action.items():
            ordered = sorted(sessions, key=lambda item: item.start_time_step)
            if any(
                current.start_time_step <= previous.end_time_step
                for previous, current in zip(ordered, ordered[1:])
            ):
                raise ValueError(f"Overlapping EV sessions for {semantic_key!r}.")
        cycle_ids = [cycle.cycle_id for cycle in self.deferrable_cycles]
        if len(set(cycle_ids)) != len(cycle_ids):
            raise ValueError("Deferrable cycle_id values must be unique.")
        for session in self.ev_sessions:
            if session.end_time_step >= prices.size:
                raise ValueError(f"EV session {session.session_id!r} exceeds the horizon.")
        for cycle in self.deferrable_cycles:
            if cycle.latest_start_time_step + len(cycle.load_profile_kwh) > prices.size:
                raise ValueError(f"Deferrable cycle {cycle.cycle_id!r} exceeds the horizon.")
        services = tuple(self.electrical_services)
        service_ids = [service.building_id for service in services]
        if len(set(service_ids)) != len(service_ids):
            raise ValueError("Electrical service building_id values must be unique.")
        if any(item not in building_set for item in service_ids):
            raise ValueError("Electrical service refers to an unknown building.")
        prices = prices.copy()
        base = base.copy()
        prices.setflags(write=False)
        base.setflags(write=False)
        object.__setattr__(self, "problem_id", str(self.problem_id))
        object.__setattr__(self, "timestep_hours", timestep)
        object.__setattr__(self, "building_ids", buildings)
        object.__setattr__(self, "price_eur_per_kwh", prices)
        object.__setattr__(self, "base_net_load_kwh", base)
        object.__setattr__(self, "stationary_storage", tuple(self.stationary_storage))
        object.__setattr__(self, "ev_sessions", tuple(self.ev_sessions))
        object.__setattr__(self, "deferrable_cycles", tuple(self.deferrable_cycles))
        object.__setattr__(self, "electrical_services", services)
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
            "settlement": self.settlement,
            "stationary_storage": [item.to_dict() for item in self.stationary_storage],
            "ev_sessions": [item.to_dict() for item in self.ev_sessions],
            "deferrable_cycles": [item.to_dict() for item in self.deferrable_cycles],
            "electrical_services": [item.to_dict() for item in self.electrical_services],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TotalEnergyProblem":
        data = dict(payload)
        data["building_ids"] = tuple(data["building_ids"])
        data["price_eur_per_kwh"] = np.asarray(
            data["price_eur_per_kwh"], dtype=np.float64
        )
        data["base_net_load_kwh"] = np.asarray(
            data["base_net_load_kwh"], dtype=np.float64
        )
        data["stationary_storage"] = tuple(
            StorageAsset(**item) for item in data.get("stationary_storage", ())
        )
        data["ev_sessions"] = tuple(
            EVSession(**item) for item in data.get("ev_sessions", ())
        )
        data["deferrable_cycles"] = tuple(
            DeferrableCycle(
                **{
                    **item,
                    "load_profile_kwh": tuple(item["load_profile_kwh"]),
                }
            )
            for item in data.get("deferrable_cycles", ())
        )
        data["electrical_services"] = tuple(
            ElectricalService(**item) for item in data.get("electrical_services", ())
        )
        return cls(**data)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "TotalEnergyProblem":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class TotalEnergyResult:
    formulation: str
    solver: SolverInfo
    cost_eur: Optional[float]
    grid_import_kwh: Optional[tuple[tuple[float, ...], ...]]
    building_net_load_kwh: Optional[tuple[tuple[float, ...], ...]]
    schedule: Optional[SemanticSchedule]
    selected_deferrable_starts: Mapping[str, Optional[int]] = field(default_factory=dict)
    ev_final_energy_kwh: Mapping[str, float] = field(default_factory=dict)
    ev_departure_shortfall_kwh: Mapping[str, float] = field(default_factory=dict)
    minimum_total_ev_shortfall_kwh: Optional[float] = None
    realized_total_ev_shortfall_kwh: Optional[float] = None
    lexicographic_shortfall_tolerance_kwh: Optional[float] = None
    minimum_ev_shortfall_by_building_kwh: Mapping[str, float] = field(
        default_factory=dict
    )
    realized_ev_shortfall_by_building_kwh: Mapping[str, float] = field(
        default_factory=dict
    )
    lexicographic_shortfall_cap_by_building_kwh: Mapping[str, float] = field(
        default_factory=dict
    )
    service_phase_status: Optional[str] = None
    service_phase_optimal: Optional[bool] = None
    service_phase_shortfall_incumbent_kwh: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "formulation": self.formulation,
            "solver": self.solver.to_dict(),
            "cost_eur": self.cost_eur,
            "grid_import_kwh": None
            if self.grid_import_kwh is None
            else [list(row) for row in self.grid_import_kwh],
            "building_net_load_kwh": None
            if self.building_net_load_kwh is None
            else [list(row) for row in self.building_net_load_kwh],
            "schedule": None if self.schedule is None else self.schedule.to_dict(),
            "selected_deferrable_starts": dict(self.selected_deferrable_starts),
            "ev_final_energy_kwh": dict(self.ev_final_energy_kwh),
            "ev_departure_shortfall_kwh": dict(
                self.ev_departure_shortfall_kwh
            ),
            "minimum_total_ev_shortfall_kwh": self.minimum_total_ev_shortfall_kwh,
            "realized_total_ev_shortfall_kwh": self.realized_total_ev_shortfall_kwh,
            "lexicographic_shortfall_tolerance_kwh": (
                self.lexicographic_shortfall_tolerance_kwh
            ),
            "minimum_ev_shortfall_by_building_kwh": dict(
                self.minimum_ev_shortfall_by_building_kwh
            ),
            "realized_ev_shortfall_by_building_kwh": dict(
                self.realized_ev_shortfall_by_building_kwh
            ),
            "lexicographic_shortfall_cap_by_building_kwh": dict(
                self.lexicographic_shortfall_cap_by_building_kwh
            ),
            "service_phase_status": self.service_phase_status,
            "service_phase_optimal": self.service_phase_optimal,
            "service_phase_shortfall_incumbent_kwh": (
                self.service_phase_shortfall_incumbent_kwh
            ),
        }


@dataclass(frozen=True)
class TotalEnergyBoundedResult:
    problem_id: str
    lower: TotalEnergyResult
    conservative: TotalEnergyResult
    certified_lower_bound_eur: Optional[float]
    model_feasible_upper_bound_eur: Optional[float]
    absolute_gap_eur: Optional[float]
    relative_gap: Optional[float]
    certificate_valid: bool
    guarantee: str = (
        "Bounds apply to the supplied linear total-energy model. The mixed-integer schedule "
        "requires exact CityLearn replay before it is called simulator-feasible."
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
class _StorageVars:
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class _EVVars:
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    charge_on: np.ndarray
    discharge_on: np.ndarray
    shortfall: Optional[int]


@dataclass(frozen=True)
class _CycleVars:
    starts: tuple[int, ...]
    variables: np.ndarray


@dataclass(frozen=True)
class _Layout:
    size: int
    storage: tuple[_StorageVars, ...]
    ev: tuple[_EVVars, ...]
    cycles: tuple[_CycleVars, ...]
    grid: np.ndarray

    @classmethod
    def build(cls, problem: TotalEnergyProblem) -> "_Layout":
        cursor = 0

        def block(size: int) -> np.ndarray:
            nonlocal cursor
            result = np.arange(cursor, cursor + size, dtype=np.int64)
            cursor += size
            return result

        storage = []
        for _ in problem.stationary_storage:
            storage.append(
                _StorageVars(
                    charge=block(problem.horizon),
                    discharge=block(problem.horizon),
                    soc=block(problem.horizon + 1),
                    direction=block(problem.horizon),
                )
            )
        ev = []
        for session in problem.ev_sessions:
            ev.append(
                _EVVars(
                    charge=block(session.duration),
                    discharge=block(session.duration),
                    soc=block(session.duration + 1),
                    charge_on=block(session.duration),
                    discharge_on=block(session.duration),
                    shortfall=(int(block(1)[0]) if session.allow_departure_shortfall else None),
                )
            )
        cycles = []
        for cycle in problem.deferrable_cycles:
            starts = tuple(
                range(cycle.earliest_start_time_step, cycle.latest_start_time_step + 1)
            )
            cycles.append(_CycleVars(starts=starts, variables=block(len(starts))))
        grid_rows = len(problem.building_ids) if problem.settlement == "individual" else 1
        grid = block(grid_rows * problem.horizon).reshape(grid_rows, problem.horizon)
        return cls(
            size=cursor,
            storage=tuple(storage),
            ev=tuple(ev),
            cycles=tuple(cycles),
            grid=grid,
        )


def _solver_options(options: SolveOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "presolve": bool(options.presolve),
        "disp": bool(options.display_solver_output),
    }
    if options.time_limit_seconds is not None:
        payload["time_limit"] = float(options.time_limit_seconds)
    if options.mip_relative_gap is not None:
        payload["mip_rel_gap"] = float(options.mip_relative_gap)
    if options.node_limit is not None:
        payload["node_limit"] = int(options.node_limit)
    return payload


def _optional_finite(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _solve(
    problem: TotalEnergyProblem,
    *,
    relaxed: bool,
    options: SolveOptions,
) -> TotalEnergyResult:
    layout = _Layout.build(problem)
    horizon = problem.horizon
    building_index = {name: index for index, name in enumerate(problem.building_ids)}
    service_by_building = {
        service.building_id: service for service in problem.electrical_services
    }

    objective = np.zeros(layout.size, dtype=np.float64)
    for row in layout.grid:
        objective[row] = problem.price_eur_per_kwh
    if not relaxed and options.throughput_tiebreaker_eur_per_kwh > 0.0:
        tie = float(options.throughput_tiebreaker_eur_per_kwh)
        for variables in layout.storage:
            objective[variables.charge] = tie
            objective[variables.discharge] = tie
        for variables in layout.ev:
            objective[variables.charge] = tie
            objective[variables.discharge] = tie

    lower_bounds = np.zeros(layout.size, dtype=np.float64)
    upper_bounds = np.full(layout.size, np.inf, dtype=np.float64)
    integrality = np.zeros(layout.size, dtype=np.int32)
    ev_shortfall_variables: list[int] = []
    ev_shortfall_variables_by_building: dict[str, list[int]] = {}

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_row(
        entries: Sequence[tuple[int, float]], lower: float, upper: float
    ) -> None:
        row = len(constraint_lower)
        for column, coefficient in entries:
            if coefficient != 0.0:
                row_indices.append(row)
                column_indices.append(int(column))
                coefficients.append(float(coefficient))
        constraint_lower.append(float(lower))
        constraint_upper.append(float(upper))

    # Per-building/time signed control-energy coefficients.  These are reused
    # by settlement and physical service constraints.
    controls: dict[tuple[int, int], list[tuple[int, float, str]]] = {
        (building, time_step): []
        for building in range(len(problem.building_ids))
        for time_step in range(horizon)
    }

    for asset_index, (asset, variables) in enumerate(
        zip(problem.stationary_storage, layout.storage)
    ):
        del asset_index
        charge_limit = asset.max_charge_kw * problem.timestep_hours
        discharge_limit = asset.max_discharge_kw * problem.timestep_hours
        upper_bounds[variables.charge] = charge_limit
        upper_bounds[variables.discharge] = discharge_limit
        upper_bounds[variables.soc] = asset.capacity_kwh
        lower_bounds[variables.soc] = asset.minimum_energy_kwh
        lower_bounds[variables.soc[0]] = asset.initial_energy_kwh
        upper_bounds[variables.soc[0]] = asset.initial_energy_kwh
        lower_bounds[variables.soc[-1]] = max(
            lower_bounds[variables.soc[-1]], asset.final_energy_min_kwh
        )
        upper_bounds[variables.direction] = 1.0
        if not relaxed:
            integrality[variables.direction] = 1
        building = building_index[asset.building_id]
        for time_step in range(horizon):
            retention = 1.0 if time_step == 0 else 1.0 - asset.loss_coefficient
            add_row(
                [
                    (variables.soc[time_step + 1], 1.0),
                    (variables.soc[time_step], -retention),
                    (variables.charge[time_step], -asset.charge_efficiency),
                    (
                        variables.discharge[time_step],
                        1.0 / asset.discharge_efficiency,
                    ),
                ],
                0.0,
                0.0,
            )
            add_row(
                [
                    (variables.charge[time_step], 1.0),
                    (variables.direction[time_step], -charge_limit),
                ],
                -np.inf,
                0.0,
            )
            add_row(
                [
                    (variables.discharge[time_step], 1.0),
                    (variables.direction[time_step], discharge_limit),
                ],
                -np.inf,
                discharge_limit,
            )
            controls[(building, time_step)].extend(
                [
                    (variables.charge[time_step], 1.0, asset.phase_connection),
                    (variables.discharge[time_step], -1.0, asset.phase_connection),
                ]
            )

    for session, variables in zip(problem.ev_sessions, layout.ev):
        charge_limit = session.max_charge_kw * problem.timestep_hours
        discharge_limit = session.max_discharge_kw * problem.timestep_hours
        min_charge = session.min_charge_kw * problem.timestep_hours
        min_discharge = session.min_discharge_kw * problem.timestep_hours
        upper_bounds[variables.charge] = charge_limit
        upper_bounds[variables.discharge] = discharge_limit
        upper_bounds[variables.soc] = session.capacity_kwh
        lower_bounds[variables.soc] = session.minimum_energy_kwh
        lower_bounds[variables.soc[0]] = session.initial_energy_kwh
        upper_bounds[variables.soc[0]] = session.initial_energy_kwh
        if variables.shortfall is None:
            lower_bounds[variables.soc[-1]] = max(
                lower_bounds[variables.soc[-1]], session.required_final_energy_kwh
            )
        else:
            upper_bounds[variables.shortfall] = max(
                session.required_final_energy_kwh - session.minimum_energy_kwh,
                0.0,
            )
            ev_shortfall_variables.append(variables.shortfall)
            ev_shortfall_variables_by_building.setdefault(
                session.building_id, []
            ).append(variables.shortfall)
            add_row(
                [
                    (variables.soc[-1], 1.0),
                    (variables.shortfall, 1.0),
                ],
                session.required_final_energy_kwh,
                np.inf,
            )
        upper_bounds[variables.charge_on] = 1.0
        upper_bounds[variables.discharge_on] = 1.0
        if not relaxed:
            integrality[variables.charge_on] = 1
            integrality[variables.discharge_on] = 1
        building = building_index[session.building_id]
        for offset in range(session.duration):
            time_step = session.start_time_step + offset
            retention = 1.0 if offset == 0 else 1.0 - session.loss_coefficient
            add_row(
                [
                    (variables.soc[offset + 1], 1.0),
                    (variables.soc[offset], -retention),
                    (variables.charge[offset], -session.charge_efficiency),
                    (
                        variables.discharge[offset],
                        1.0 / session.discharge_efficiency,
                    ),
                ],
                0.0,
                0.0,
            )
            add_row(
                [
                    (variables.charge[offset], 1.0),
                    (variables.charge_on[offset], -charge_limit),
                ],
                -np.inf,
                0.0,
            )
            add_row(
                [
                    (variables.charge[offset], -1.0),
                    (variables.charge_on[offset], min_charge),
                ],
                -np.inf,
                0.0,
            )
            add_row(
                [
                    (variables.discharge[offset], 1.0),
                    (variables.discharge_on[offset], -discharge_limit),
                ],
                -np.inf,
                0.0,
            )
            add_row(
                [
                    (variables.discharge[offset], -1.0),
                    (variables.discharge_on[offset], min_discharge),
                ],
                -np.inf,
                0.0,
            )
            add_row(
                [
                    (variables.charge_on[offset], 1.0),
                    (variables.discharge_on[offset], 1.0),
                ],
                -np.inf,
                1.0,
            )
            controls[(building, time_step)].extend(
                [
                    (variables.charge[offset], 1.0, session.phase_connection),
                    (variables.discharge[offset], -1.0, session.phase_connection),
                ]
            )

    deferrable_start_exclusion: dict[tuple[str, str, int], list[int]] = {}
    for cycle, variables in zip(problem.deferrable_cycles, layout.cycles):
        upper_bounds[variables.variables] = 1.0
        if not relaxed:
            integrality[variables.variables] = 1
        add_row(
            [(variable, 1.0) for variable in variables.variables],
            1.0 if cycle.must_run else 0.0,
            1.0,
        )
        building = building_index[cycle.building_id]
        for start, variable in zip(variables.starts, variables.variables):
            deferrable_start_exclusion.setdefault(
                (cycle.building_id, cycle.action_name, start), []
            ).append(int(variable))
            for offset, energy in enumerate(cycle.load_profile_kwh):
                controls[(building, start + offset)].append(
                    (variable, energy, cycle.phase_connection)
                )
    for variables in deferrable_start_exclusion.values():
        if len(variables) > 1:
            add_row([(variable, 1.0) for variable in variables], -np.inf, 1.0)

    # Settlement import epigraph.
    if problem.settlement == "individual":
        for building in range(len(problem.building_ids)):
            for time_step in range(horizon):
                entries = [(layout.grid[building, time_step], 1.0)]
                entries.extend(
                    (column, -coefficient)
                    for column, coefficient, _ in controls[(building, time_step)]
                )
                add_row(
                    entries,
                    float(problem.base_net_load_kwh[building, time_step]),
                    np.inf,
                )
    else:
        for time_step in range(horizon):
            entries = [(layout.grid[0, time_step], 1.0)]
            for building in range(len(problem.building_ids)):
                entries.extend(
                    (column, -coefficient)
                    for column, coefficient, _ in controls[(building, time_step)]
                )
            add_row(entries, float(np.sum(problem.base_net_load_kwh[:, time_step])), np.inf)

    # Physical signed service envelopes, including base load and phase split.
    for building_id, service in service_by_building.items():
        building = building_index[building_id]
        base_fractions = service.fractions(None)
        for time_step in range(horizon):
            entries = [
                (column, coefficient)
                for column, coefficient, _ in controls[(building, time_step)]
            ]
            base = float(problem.base_net_load_kwh[building, time_step])
            lower = (
                -np.inf
                if service.total_export_kw is None
                else -service.total_export_kw * problem.timestep_hours - base
            )
            upper = (
                np.inf
                if service.total_import_kw is None
                else service.total_import_kw * problem.timestep_hours - base
            )
            if np.isfinite(lower) or np.isfinite(upper):
                add_row(entries, lower, upper)

            for phase in _PHASES:
                phase_entries = []
                for column, coefficient, connection in controls[(building, time_step)]:
                    fraction = service.fractions(connection)[phase]
                    if fraction:
                        phase_entries.append((column, coefficient * fraction))
                phase_base = base * base_fractions[phase]
                phase_export = service.phase_export_kw.get(phase)
                phase_import = service.phase_import_kw.get(phase)
                phase_lower = (
                    -np.inf
                    if phase_export is None
                    else -phase_export * problem.timestep_hours - phase_base
                )
                phase_upper = (
                    np.inf
                    if phase_import is None
                    else phase_import * problem.timestep_hours - phase_base
                )
                if np.isfinite(phase_lower) or np.isfinite(phase_upper):
                    add_row(phase_entries, phase_lower, phase_upper)

    matrix = coo_array(
        (
            np.asarray(coefficients, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(column_indices, dtype=np.int64),
            ),
        ),
        shape=(len(constraint_lower), layout.size),
    ).tocsc()
    base_constraints = LinearConstraint(
        matrix,
        np.asarray(constraint_lower, dtype=np.float64),
        np.asarray(constraint_upper, dtype=np.float64),
    )
    solver_options = _solver_options(options)
    bounds = Bounds(lower_bounds, upper_bounds)

    def run_solver(
        solver_objective: np.ndarray,
        *,
        extra_constraint: Optional[LinearConstraint] = None,
    ) -> Any:
        constraints: Any = (
            base_constraints
            if extra_constraint is None
            else (base_constraints, extra_constraint)
        )
        return milp(
            c=solver_objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=solver_options,
        )

    minimum_total_shortfall: Optional[float] = None
    lexicographic_shortfall_tolerance: Optional[float] = None
    minimum_shortfall_by_building: dict[str, float] = {}
    shortfall_cap_by_building: dict[str, float] = {}
    service_phase_status: Optional[str] = None
    service_phase_optimal: Optional[bool] = None
    service_phase_shortfall_incumbent: Optional[float] = None
    if ev_shortfall_variables:
        service_objective = np.zeros(layout.size, dtype=np.float64)
        service_objective[ev_shortfall_variables] = 1.0
        service_raw = run_solver(service_objective)
        service_status_code = int(service_raw.status)
        service_phase_status = _STATUS_NAMES.get(service_status_code, "unknown")
        service_phase_optimal = service_status_code == 0
        service_solution = getattr(service_raw, "x", None)
        service_has_solution = service_solution is not None and np.all(
            np.isfinite(service_solution)
        )
        if service_has_solution:
            service_solution = np.asarray(service_solution, dtype=np.float64)
            service_phase_shortfall_incumbent = float(
                np.sum(service_solution[ev_shortfall_variables])
            )
        if service_has_solution and service_phase_optimal:
            minimum_total_shortfall = service_phase_shortfall_incumbent
            lexicographic_shortfall_tolerance = float(
                options.lexicographic_shortfall_tolerance_kwh
            )

            # The service-only feasible set factorizes by building: community
            # settlement contributes an import epigraph but no physical
            # cross-building constraint.  Consequently, every globally
            # service-optimal solution must attain the independently minimal
            # aggregate shortfall of each building.  Keeping one cap per
            # building is therefore equivalent to the global lexicographic
            # cap when the tolerance is zero, while eliminating the large
            # cross-building shortfall degeneracy in the economic phase.
            for building_id, variables in ev_shortfall_variables_by_building.items():
                value = float(np.sum(service_solution[variables]))
                minimum_shortfall_by_building[building_id] = (
                    0.0 if value <= 1.0e-7 else value
                )

            positive_total = sum(
                value
                for value in minimum_shortfall_by_building.values()
                if value > 0.0
            )
            for building_id, minimum in minimum_shortfall_by_building.items():
                # A numerical tolerance must never introduce missed service
                # at a building whose targets are all fully attainable.  For
                # positive minima, distribute the single global tolerance
                # proportionally, so the building caps still add to exactly
                # minimum_total_shortfall + tolerance (up to round-off).
                share = (
                    lexicographic_shortfall_tolerance * minimum / positive_total
                    if positive_total > 0.0 and minimum > 0.0
                    else 0.0
                )
                shortfall_cap_by_building[building_id] = minimum + share

            service_row_indices: list[int] = []
            service_column_indices: list[int] = []
            service_coefficients: list[float] = []
            service_upper: list[float] = []
            for row, (building_id, variables) in enumerate(
                ev_shortfall_variables_by_building.items()
            ):
                service_row_indices.extend([row] * len(variables))
                service_column_indices.extend(variables)
                service_coefficients.extend([1.0] * len(variables))
                service_upper.append(shortfall_cap_by_building[building_id])
            service_rows = coo_array(
                (
                    np.asarray(service_coefficients, dtype=np.float64),
                    (
                        np.asarray(service_row_indices, dtype=np.int64),
                        np.asarray(service_column_indices, dtype=np.int64),
                    ),
                ),
                shape=(len(service_upper), layout.size),
            ).tocsc()
            service_constraint = LinearConstraint(
                service_rows,
                np.full(len(service_upper), -np.inf, dtype=np.float64),
                np.asarray(service_upper, dtype=np.float64),
            )
            raw = run_solver(objective, extra_constraint=service_constraint)
        else:
            # An incumbent from an unfinished service solve is not a proven
            # lexicographic optimum.  Returning it with its non-optimal solver
            # status is honest and avoids a misleading "optimal" economic
            # solve under an uncertified service cap.
            raw = service_raw
    else:
        raw = run_solver(objective)
    status_code = int(raw.status)
    solution = getattr(raw, "x", None)
    has_solution = solution is not None and np.all(np.isfinite(solution))
    objective_value = _optional_finite(getattr(raw, "fun", None))
    dual_bound = _optional_finite(getattr(raw, "mip_dual_bound", None))
    if status_code == 0 and dual_bound is None:
        dual_bound = objective_value
    solver = SolverInfo(
        status=_STATUS_NAMES.get(status_code, "unknown"),
        status_code=status_code,
        optimal=status_code == 0,
        has_solution=bool(has_solution),
        message=str(raw.message),
        solver_objective_eur=objective_value,
        dual_bound_eur=dual_bound,
        mip_gap=_optional_finite(getattr(raw, "mip_gap", None)),
    )
    if not has_solution:
        return TotalEnergyResult(
            formulation="total_energy_lp_relaxation" if relaxed else "total_energy_milp",
            solver=solver,
            cost_eur=None,
            grid_import_kwh=None,
            building_net_load_kwh=None,
            schedule=None,
            minimum_total_ev_shortfall_kwh=minimum_total_shortfall,
            lexicographic_shortfall_tolerance_kwh=(
                lexicographic_shortfall_tolerance
            ),
            minimum_ev_shortfall_by_building_kwh=minimum_shortfall_by_building,
            lexicographic_shortfall_cap_by_building_kwh=(
                shortfall_cap_by_building
            ),
            service_phase_status=service_phase_status,
            service_phase_optimal=service_phase_optimal,
            service_phase_shortfall_incumbent_kwh=(
                service_phase_shortfall_incumbent
            ),
        )

    solution = np.asarray(solution, dtype=np.float64)
    realized_total_shortfall = (
        float(np.sum(solution[ev_shortfall_variables]))
        if ev_shortfall_variables
        else None
    )
    realized_shortfall_by_building = {
        building_id: float(np.sum(solution[variables]))
        for building_id, variables in ev_shortfall_variables_by_building.items()
    }
    grid = solution[layout.grid]
    official_cost = float(np.sum(grid * problem.price_eur_per_kwh[None, :]))
    building_net = np.asarray(problem.base_net_load_kwh, dtype=np.float64).copy()
    for (building, time_step), entries in controls.items():
        building_net[building, time_step] += sum(
            solution[column] * coefficient for column, coefficient, _ in entries
        )

    if relaxed:
        return TotalEnergyResult(
            formulation="total_energy_lp_relaxation",
            solver=solver,
            cost_eur=official_cost,
            grid_import_kwh=tuple(tuple(float(value) for value in row) for row in grid),
            building_net_load_kwh=tuple(
                tuple(float(value) for value in row) for row in building_net
            ),
            schedule=None,
            minimum_total_ev_shortfall_kwh=minimum_total_shortfall,
            realized_total_ev_shortfall_kwh=realized_total_shortfall,
            lexicographic_shortfall_tolerance_kwh=(
                lexicographic_shortfall_tolerance
            ),
            minimum_ev_shortfall_by_building_kwh=minimum_shortfall_by_building,
            realized_ev_shortfall_by_building_kwh=(
                realized_shortfall_by_building
            ),
            lexicographic_shortfall_cap_by_building_kwh=(
                shortfall_cap_by_building
            ),
            service_phase_status=service_phase_status,
            service_phase_optimal=service_phase_optimal,
            service_phase_shortfall_incumbent_kwh=(
                service_phase_shortfall_incumbent
            ),
        )

    semantic_values: dict[tuple[str, str], np.ndarray] = {}
    semantic_metadata: dict[tuple[str, str], tuple[str, str]] = {}
    for asset, variables in zip(problem.stationary_storage, layout.storage):
        key = (asset.building_id, asset.action_name)
        values = (
            solution[variables.charge] - solution[variables.discharge]
        ) / problem.timestep_hours
        semantic_values[key] = values.copy()
        semantic_metadata[key] = ("kW", "charge")
    ev_final: dict[str, float] = {}
    ev_shortfall: dict[str, float] = {}
    for session, variables in zip(problem.ev_sessions, layout.ev):
        key = (session.building_id, session.action_name)
        values = semantic_values.setdefault(key, np.zeros(horizon, dtype=np.float64))
        values[session.start_time_step : session.end_time_step + 1] += (
            solution[variables.charge] - solution[variables.discharge]
        ) / problem.timestep_hours
        semantic_metadata[key] = ("kW", "charge")
        ev_final[session.session_id] = float(solution[variables.soc[-1]])
        ev_shortfall[session.session_id] = (
            float(solution[variables.shortfall])
            if variables.shortfall is not None
            else max(
                session.required_final_energy_kwh
                - float(solution[variables.soc[-1]]),
                0.0,
            )
        )
    selected_starts: dict[str, Optional[int]] = {}
    for cycle, variables in zip(problem.deferrable_cycles, layout.cycles):
        key = (cycle.building_id, cycle.action_name)
        values = semantic_values.setdefault(key, np.zeros(horizon, dtype=np.float64))
        chosen: Optional[int] = None
        for start, variable in zip(variables.starts, variables.variables):
            if solution[variable] > 0.5:
                values[start] = 1.0
                chosen = int(start)
        selected_starts[cycle.cycle_id] = chosen
        semantic_metadata[key] = ("normalized_action", "start")
    series = []
    for key in sorted(semantic_values):
        values = semantic_values[key]
        values[np.abs(values) < _TOLERANCE] = 0.0
        unit, direction = semantic_metadata[key]
        series.append(
            SemanticActionSeries(
                building_id=key[0],
                action_name=key[1],
                values=tuple(float(value) for value in values),
                unit=unit,
                positive_direction=direction,
            )
        )
    action_power_limits_kw: dict[str, dict[str, dict[str, float]]] = {}
    for asset in problem.stationary_storage:
        action_power_limits_kw.setdefault(asset.building_id, {})[
            asset.action_name
        ] = {
            "nominal_power_kw": float(
                max(asset.max_charge_kw, asset.max_discharge_kw)
            ),
            "max_charging_power_kw": float(asset.max_charge_kw),
            "max_discharging_power_kw": float(asset.max_discharge_kw),
        }
    for session in problem.ev_sessions:
        action_limits = action_power_limits_kw.setdefault(
            session.building_id, {}
        ).setdefault(
            session.action_name,
            {
                "max_charging_power_kw": 0.0,
                "max_discharging_power_kw": 0.0,
                "min_charging_power_kw": 0.0,
                "min_discharging_power_kw": 0.0,
            },
        )
        action_limits["max_charging_power_kw"] = max(
            action_limits["max_charging_power_kw"], float(session.max_charge_kw)
        )
        action_limits["max_discharging_power_kw"] = max(
            action_limits["max_discharging_power_kw"],
            float(session.max_discharge_kw),
        )
        action_limits["min_charging_power_kw"] = max(
            action_limits["min_charging_power_kw"], float(session.min_charge_kw)
        )
        action_limits["min_discharging_power_kw"] = max(
            action_limits["min_discharging_power_kw"],
            float(session.min_discharge_kw),
        )

    schedule = SemanticSchedule(
        problem_id=problem.problem_id,
        horizon=horizon,
        timestep_hours=problem.timestep_hours,
        series=tuple(series),
        metadata={
            "formulation": "total_energy_milp",
            "settlement": problem.settlement,
            "minimum_total_ev_shortfall_kwh": minimum_total_shortfall,
            "realized_total_ev_shortfall_kwh": realized_total_shortfall,
            "lexicographic_shortfall_tolerance_kwh": (
                lexicographic_shortfall_tolerance
            ),
            "minimum_ev_shortfall_by_building_kwh": (
                minimum_shortfall_by_building
            ),
            "realized_ev_shortfall_by_building_kwh": (
                realized_shortfall_by_building
            ),
            "lexicographic_shortfall_cap_by_building_kwh": (
                shortfall_cap_by_building
            ),
            "service_phase_status": service_phase_status,
            "service_phase_optimal": service_phase_optimal,
            "service_phase_shortfall_incumbent_kwh": (
                service_phase_shortfall_incumbent
            ),
            "includes_stationary_storage": bool(problem.stationary_storage),
            "includes_ev_v2g": bool(problem.ev_sessions),
            "includes_deferrable_appliances": bool(problem.deferrable_cycles),
            "includes_electrical_service": bool(problem.electrical_services),
            "requires_citylearn_replay": True,
            "action_power_limits_kw": action_power_limits_kw,
            **dict(problem.metadata),
        },
    )
    return TotalEnergyResult(
        formulation="total_energy_milp",
        solver=solver,
        cost_eur=official_cost,
        grid_import_kwh=tuple(tuple(float(value) for value in row) for row in grid),
        building_net_load_kwh=tuple(
            tuple(float(value) for value in row) for row in building_net
        ),
        schedule=schedule,
        selected_deferrable_starts=selected_starts,
        ev_final_energy_kwh=ev_final,
        ev_departure_shortfall_kwh=ev_shortfall,
        minimum_total_ev_shortfall_kwh=minimum_total_shortfall,
        realized_total_ev_shortfall_kwh=realized_total_shortfall,
        lexicographic_shortfall_tolerance_kwh=(
            lexicographic_shortfall_tolerance
        ),
        minimum_ev_shortfall_by_building_kwh=minimum_shortfall_by_building,
        realized_ev_shortfall_by_building_kwh=realized_shortfall_by_building,
        lexicographic_shortfall_cap_by_building_kwh=(
            shortfall_cap_by_building
        ),
        service_phase_status=service_phase_status,
        service_phase_optimal=service_phase_optimal,
        service_phase_shortfall_incumbent_kwh=(
            service_phase_shortfall_incumbent
        ),
    )


def solve_total_energy_relaxation(
    problem: TotalEnergyProblem,
    options: Optional[SolveOptions] = None,
) -> TotalEnergyResult:
    """Solve the continuous relaxation used as a structural lower bound."""

    return _solve(problem, relaxed=True, options=options or SolveOptions())


def solve_total_energy_schedule(
    problem: TotalEnergyProblem,
    options: Optional[SolveOptions] = None,
) -> TotalEnergyResult:
    """Solve the complete mixed-integer schedule."""

    return _solve(problem, relaxed=False, options=options or SolveOptions())


def solve_bounded_total_energy_oracle(
    problem: TotalEnergyProblem,
    options: Optional[SolveOptions] = None,
) -> TotalEnergyBoundedResult:
    """Return a lower/upper certificate for the supplied total-energy model."""

    options = options or SolveOptions()
    lower = solve_total_energy_relaxation(problem, options)
    conservative = solve_total_energy_schedule(problem, options)
    certified_lower = lower.solver.dual_bound_eur
    if certified_lower is None and lower.solver.optimal:
        certified_lower = lower.cost_eur
    upper = conservative.cost_eur if conservative.solver.has_solution else None
    valid = (
        certified_lower is not None
        and upper is not None
        and lower.service_phase_optimal is not False
        and conservative.service_phase_optimal is not False
    )
    if valid:
        lower_service = lower.minimum_total_ev_shortfall_kwh
        upper_service = conservative.minimum_total_ev_shortfall_kwh
        if lower_service is None and upper_service is None:
            pass
        elif lower_service is None or upper_service is None:
            valid = False
        elif not math.isclose(
            lower_service, upper_service, rel_tol=0.0, abs_tol=1.0e-6
        ):
            valid = False
    absolute_gap: Optional[float] = None
    relative_gap: Optional[float] = None
    if valid:
        assert certified_lower is not None and upper is not None
        raw_gap = upper - certified_lower
        tolerance = _TOLERANCE * max(1.0, abs(upper), abs(certified_lower))
        if raw_gap < -tolerance:
            raise RuntimeError("Total-energy lower bound exceeds its feasible upper candidate.")
        absolute_gap = max(raw_gap, 0.0)
        relative_gap = absolute_gap / max(abs(upper), _TOLERANCE)
    return TotalEnergyBoundedResult(
        problem_id=problem.problem_id,
        lower=lower,
        conservative=conservative,
        certified_lower_bound_eur=certified_lower,
        model_feasible_upper_bound_eur=upper,
        absolute_gap_eur=absolute_gap,
        relative_gap=relative_gap,
        certificate_valid=valid,
    )
