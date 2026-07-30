"""Exact building decomposition for the individual total-energy oracle.

The :class:`~algorithms.oracles.total_energy_milp.TotalEnergyProblem` model is
separable by building when ``settlement == "individual"``: the objective uses
one import meter per building and every flexible asset and electrical-service
constraint belongs to exactly one building.  This module makes that structure
explicit, solves one smaller problem per building and recombines the results
using semantic action identities rather than action-vector positions.

Community settlement is deliberately rejected.  Its net import variable
couples buildings, so a sum of isolated solves is not a community optimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

from algorithms.oracles.perfect_foresight_milp import (
    SemanticActionSeries,
    SemanticSchedule,
    SolveOptions,
    SolverInfo,
)
from algorithms.oracles.total_energy_milp import (
    TotalEnergyBoundedResult,
    TotalEnergyProblem,
    TotalEnergyResult,
    solve_bounded_total_energy_oracle,
    solve_total_energy_relaxation,
    solve_total_energy_schedule,
)


_NUMERICAL_TOLERANCE = 1.0e-8
DecompositionMode = Literal["bounded", "relaxation", "schedule"]
BuildingResult = Union[TotalEnergyResult, TotalEnergyBoundedResult]


def validate_individual_total_energy_separability(
    problem: TotalEnergyProblem,
) -> None:
    """Fail closed if ``problem`` cannot be separated exactly by building.

    The current total-energy problem schema has only one cross-building term:
    the single net-import meter used by community settlement.  Asset ownership
    and electrical-service ownership are already validated by
    :class:`TotalEnergyProblem` itself.
    """

    if problem.settlement != "individual":
        raise ValueError(
            "Individual total-energy decomposition requires "
            "settlement='individual'; community netting is non-separable."
        )


def split_individual_total_energy_problems(
    problem: TotalEnergyProblem,
) -> tuple[TotalEnergyProblem, ...]:
    """Return one identity-preserving total-energy problem per building."""

    validate_individual_total_energy_separability(problem)
    subproblems: list[TotalEnergyProblem] = []
    for building_index, building_id in enumerate(problem.building_ids):
        metadata = {
            **dict(problem.metadata),
            "source_problem_id": problem.problem_id,
            "individual_building_id": building_id,
            "decomposition": "exact_individual_settlement_by_building",
            "cost_semantics": "individual_import_with_zero_export_credit",
            "community_optimum_claim": False,
        }
        subproblems.append(
            TotalEnergyProblem(
                problem_id=f"{problem.problem_id}::individual::{building_id}",
                timestep_hours=problem.timestep_hours,
                building_ids=(building_id,),
                price_eur_per_kwh=problem.price_eur_per_kwh,
                base_net_load_kwh=problem.base_net_load_kwh[
                    building_index : building_index + 1, :
                ],
                settlement="individual",
                stationary_storage=tuple(
                    item
                    for item in problem.stationary_storage
                    if item.building_id == building_id
                ),
                ev_sessions=tuple(
                    item
                    for item in problem.ev_sessions
                    if item.building_id == building_id
                ),
                deferrable_cycles=tuple(
                    item
                    for item in problem.deferrable_cycles
                    if item.building_id == building_id
                ),
                electrical_services=tuple(
                    item
                    for item in problem.electrical_services
                    if item.building_id == building_id
                ),
                metadata=metadata,
            )
        )
    return tuple(subproblems)


@dataclass(frozen=True)
class IndividualTotalEnergyBuildingSolution:
    """One isolated building solve, retained for auditability."""

    building_id: str
    subproblem_id: str
    result: BuildingResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "subproblem_id": self.subproblem_id,
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class IndividualTotalEnergyDecompositionResult:
    """Per-building evidence plus the semantically recombined result."""

    source_problem_id: str
    mode: DecompositionMode
    buildings: tuple[IndividualTotalEnergyBuildingSolution, ...]
    combined: BuildingResult
    guarantee: str = (
        "This is an exact structural decomposition of the supplied "
        "individual-settlement total-energy model: costs and constraints are "
        "local to each building and the combined objective is their sum. Solver "
        "optimality and bounds remain those reported per building. It is not a "
        "community-settlement optimum, and a mixed-integer schedule still "
        "requires exact CityLearn replay before it is simulator-feasible."
    )

    @property
    def schedule(self) -> Optional[SemanticSchedule]:
        if isinstance(self.combined, TotalEnergyBoundedResult):
            return self.combined.conservative.schedule
        return self.combined.schedule

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_problem_id": self.source_problem_id,
            "mode": self.mode,
            "decomposition": "exact_individual_settlement_by_building",
            "community_optimum_claim": False,
            "guarantee": self.guarantee,
            "combined": self.combined.to_dict(),
            "buildings": [item.to_dict() for item in self.buildings],
        }


def _expected_action_keys(
    problem: TotalEnergyProblem,
) -> tuple[tuple[str, str], ...]:
    keys: set[tuple[str, str]] = set()
    for item in (
        *problem.stationary_storage,
        *problem.ev_sessions,
        *problem.deferrable_cycles,
    ):
        keys.add((item.building_id, item.action_name))
    building_order = {name: index for index, name in enumerate(problem.building_ids)}
    return tuple(sorted(keys, key=lambda key: (building_order[key[0]], key[1])))


def _sum_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _aggregate_soft_service_value(
    problem: TotalEnergyProblem,
    results_by_building: dict[str, TotalEnergyResult],
    attribute: str,
) -> Optional[float]:
    soft_buildings = {
        session.building_id
        for session in problem.ev_sessions
        if session.allow_departure_shortfall
    }
    if not soft_buildings:
        return None
    values = [getattr(results_by_building[name], attribute) for name in soft_buildings]
    return _sum_optional(values)


def _aggregate_solver_info(
    results: Sequence[TotalEnergyResult],
) -> SolverInfo:
    all_optimal = all(item.solver.optimal for item in results)
    all_have_solution = all(item.solver.has_solution for item in results)
    first_failure = next(
        (item.solver for item in results if not item.solver.has_solution), None
    )
    if all_optimal:
        status, status_code = "optimal", 0
    elif all_have_solution:
        status, status_code = "limit_reached", 1
    elif first_failure is not None:
        status, status_code = first_failure.status, first_failure.status_code
    else:  # Defensive only; a valid problem always contains at least one building.
        status, status_code = "unknown", 4

    objective = _sum_optional(
        [item.solver.solver_objective_eur for item in results]
    )
    dual_bound = _sum_optional([item.solver.dual_bound_eur for item in results])
    mip_gap: Optional[float] = None
    if objective is not None and dual_bound is not None:
        mip_gap = max(objective - dual_bound, 0.0) / max(
            abs(objective), _NUMERICAL_TOLERANCE
        )
    return SolverInfo(
        status=status,
        status_code=status_code,
        optimal=all_optimal,
        has_solution=all_have_solution,
        message=(
            "Independent-building decomposition: "
            + ", ".join(item.solver.status for item in results)
        ),
        solver_objective_eur=objective,
        dual_bound_eur=dual_bound,
        mip_gap=mip_gap,
    )


def _combine_schedule(
    problem: TotalEnergyProblem,
    results_by_building: dict[str, TotalEnergyResult],
    *,
    minimum_shortfall: Optional[float],
    realized_shortfall: Optional[float],
    aggregate_shortfall_tolerance: Optional[float],
) -> Optional[SemanticSchedule]:
    schedules = [results_by_building[name].schedule for name in problem.building_ids]
    if any(schedule is None for schedule in schedules):
        return None

    expected_keys = _expected_action_keys(problem)
    expected_set = set(expected_keys)
    series_by_key: dict[tuple[str, str], SemanticActionSeries] = {}
    action_limits: dict[str, dict[str, dict[str, float]]] = {}
    for building_id, schedule in zip(problem.building_ids, schedules):
        assert schedule is not None
        if schedule.horizon != problem.horizon or not math.isclose(
            schedule.timestep_hours,
            problem.timestep_hours,
            rel_tol=0.0,
            abs_tol=_NUMERICAL_TOLERANCE,
        ):
            raise ValueError(
                f"Individual schedule for {building_id!r} does not match the source horizon."
            )
        for item in schedule.series:
            key = item.building_id, item.action_name
            if item.building_id != building_id:
                raise ValueError(
                    f"Individual schedule for {building_id!r} contains action {key!r}."
                )
            if key not in expected_set:
                raise ValueError(f"Unexpected semantic action series {key!r}.")
            if key in series_by_key:
                raise ValueError(f"Duplicate semantic action series {key!r}.")
            series_by_key[key] = item

        raw_limits = schedule.metadata.get("action_power_limits_kw", {})
        for owner, owner_limits in raw_limits.items():
            if owner != building_id:
                raise ValueError(
                    f"Action metadata for {building_id!r} contains owner {owner!r}."
                )
            action_limits[owner] = {
                str(action): {str(key): float(value) for key, value in limits.items()}
                for action, limits in owner_limits.items()
            }

    missing = [key for key in expected_keys if key not in series_by_key]
    if missing:
        raise ValueError(f"Missing semantic action series: {missing!r}.")

    return SemanticSchedule(
        problem_id=f"{problem.problem_id}::individual::combined",
        horizon=problem.horizon,
        timestep_hours=problem.timestep_hours,
        series=tuple(series_by_key[key] for key in expected_keys),
        metadata={
            **dict(problem.metadata),
            "formulation": "total_energy_milp",
            "settlement": "individual",
            "decomposition": "exact_individual_settlement_by_building",
            "source_problem_id": problem.problem_id,
            "community_optimum_claim": False,
            "minimum_total_ev_shortfall_kwh": minimum_shortfall,
            "realized_total_ev_shortfall_kwh": realized_shortfall,
            "lexicographic_shortfall_tolerance_kwh": (
                aggregate_shortfall_tolerance
            ),
            "per_building_lexicographic_shortfall_tolerance": True,
            "includes_stationary_storage": bool(problem.stationary_storage),
            "includes_ev_v2g": bool(problem.ev_sessions),
            "includes_deferrable_appliances": bool(problem.deferrable_cycles),
            "includes_electrical_service": bool(problem.electrical_services),
            "requires_citylearn_replay": True,
            "action_power_limits_kw": action_limits,
        },
    )


def combine_individual_total_energy_results(
    problem: TotalEnergyProblem,
    building_solutions: Sequence[IndividualTotalEnergyBuildingSolution],
) -> TotalEnergyResult:
    """Recombine isolated LP or MILP results into source building order."""

    validate_individual_total_energy_separability(problem)
    if any(not isinstance(item.result, TotalEnergyResult) for item in building_solutions):
        raise TypeError("Expected TotalEnergyResult values for result recombination.")
    results_by_building: dict[str, TotalEnergyResult] = {}
    for item in building_solutions:
        if item.building_id not in problem.building_ids:
            raise ValueError(f"Unexpected building result {item.building_id!r}.")
        if item.building_id in results_by_building:
            raise ValueError(f"Duplicate building result {item.building_id!r}.")
        assert isinstance(item.result, TotalEnergyResult)
        results_by_building[item.building_id] = item.result
    missing = [name for name in problem.building_ids if name not in results_by_building]
    if missing:
        raise ValueError(f"Missing individual building results: {missing!r}.")

    ordered = [results_by_building[name] for name in problem.building_ids]
    formulations = {item.formulation for item in ordered}
    if len(formulations) != 1:
        raise ValueError("Cannot combine mixed LP-relaxation and MILP results.")
    formulation = next(iter(formulations))
    solver = _aggregate_solver_info(ordered)
    cost = _sum_optional([item.cost_eur for item in ordered])

    grid_import: Optional[tuple[tuple[float, ...], ...]] = None
    building_net: Optional[tuple[tuple[float, ...], ...]] = None
    if solver.has_solution:
        grid_rows: list[tuple[float, ...]] = []
        net_rows: list[tuple[float, ...]] = []
        for building_id, item in zip(problem.building_ids, ordered):
            if item.grid_import_kwh is None or len(item.grid_import_kwh) != 1:
                raise ValueError(
                    f"Individual grid result for {building_id!r} must contain one row."
                )
            if item.building_net_load_kwh is None or len(item.building_net_load_kwh) != 1:
                raise ValueError(
                    f"Individual net-load result for {building_id!r} must contain one row."
                )
            grid_rows.append(tuple(item.grid_import_kwh[0]))
            net_rows.append(tuple(item.building_net_load_kwh[0]))
        grid_import = tuple(grid_rows)
        building_net = tuple(net_rows)

    minimum_shortfall = _aggregate_soft_service_value(
        problem, results_by_building, "minimum_total_ev_shortfall_kwh"
    )
    realized_shortfall = _aggregate_soft_service_value(
        problem, results_by_building, "realized_total_ev_shortfall_kwh"
    )
    aggregate_tolerance = _aggregate_soft_service_value(
        problem, results_by_building, "lexicographic_shortfall_tolerance_kwh"
    )
    schedule = None
    if formulation == "total_energy_milp":
        schedule = _combine_schedule(
            problem,
            results_by_building,
            minimum_shortfall=minimum_shortfall,
            realized_shortfall=realized_shortfall,
            aggregate_shortfall_tolerance=aggregate_tolerance,
        )

    selected_starts: dict[str, Optional[int]] = {}
    ev_final: dict[str, float] = {}
    ev_shortfall: dict[str, float] = {}
    for item in ordered:
        for target, source, label in (
            (selected_starts, item.selected_deferrable_starts, "cycle"),
            (ev_final, item.ev_final_energy_kwh, "EV session"),
            (ev_shortfall, item.ev_departure_shortfall_kwh, "EV session"),
        ):
            overlap = set(target) & set(source)
            if overlap:
                raise ValueError(f"Duplicate {label} identifiers: {sorted(overlap)!r}.")
            target.update(source)

    return TotalEnergyResult(
        formulation=formulation,
        solver=solver,
        cost_eur=cost,
        grid_import_kwh=grid_import,
        building_net_load_kwh=building_net,
        schedule=schedule,
        selected_deferrable_starts=selected_starts,
        ev_final_energy_kwh=ev_final,
        ev_departure_shortfall_kwh=ev_shortfall,
        minimum_total_ev_shortfall_kwh=minimum_shortfall,
        realized_total_ev_shortfall_kwh=realized_shortfall,
        lexicographic_shortfall_tolerance_kwh=aggregate_tolerance,
    )


def _combine_bounded_results(
    problem: TotalEnergyProblem,
    building_solutions: Sequence[IndividualTotalEnergyBuildingSolution],
) -> TotalEnergyBoundedResult:
    bounded: list[TotalEnergyBoundedResult] = []
    for item in building_solutions:
        if not isinstance(item.result, TotalEnergyBoundedResult):
            raise TypeError("Expected bounded results for bounded recombination.")
        bounded.append(item.result)
    lower_solutions = tuple(
        IndividualTotalEnergyBuildingSolution(
            building_id=item.building_id,
            subproblem_id=item.subproblem_id,
            result=result.lower,
        )
        for item, result in zip(building_solutions, bounded)
    )
    upper_solutions = tuple(
        IndividualTotalEnergyBuildingSolution(
            building_id=item.building_id,
            subproblem_id=item.subproblem_id,
            result=result.conservative,
        )
        for item, result in zip(building_solutions, bounded)
    )
    lower = combine_individual_total_energy_results(problem, lower_solutions)
    conservative = combine_individual_total_energy_results(problem, upper_solutions)
    certified_lower = _sum_optional(
        [item.certified_lower_bound_eur for item in bounded]
    )
    upper = _sum_optional([item.model_feasible_upper_bound_eur for item in bounded])
    valid = (
        all(item.certificate_valid for item in bounded)
        and certified_lower is not None
        and upper is not None
    )
    absolute_gap: Optional[float] = None
    relative_gap: Optional[float] = None
    if valid:
        assert certified_lower is not None and upper is not None
        raw_gap = upper - certified_lower
        tolerance = _NUMERICAL_TOLERANCE * max(
            1.0, abs(upper), abs(certified_lower)
        )
        if raw_gap < -tolerance:
            raise RuntimeError(
                "Combined individual lower bound exceeds the combined upper candidate."
            )
        absolute_gap = max(raw_gap, 0.0)
        relative_gap = absolute_gap / max(abs(upper), _NUMERICAL_TOLERANCE)

    return TotalEnergyBoundedResult(
        problem_id=problem.problem_id,
        lower=lower,
        conservative=conservative,
        certified_lower_bound_eur=certified_lower,
        model_feasible_upper_bound_eur=upper,
        absolute_gap_eur=absolute_gap,
        relative_gap=relative_gap,
        certificate_valid=valid,
        guarantee=(
            "Bounds are exact sums of isolated building bounds for the supplied "
            "individual-settlement total-energy model. They do not certify a "
            "community-settlement optimum. The combined mixed-integer schedule "
            "requires exact CityLearn replay before it is simulator-feasible."
        ),
    )


def solve_decomposed_individual_total_energy(
    problem: TotalEnergyProblem,
    *,
    mode: DecompositionMode = "bounded",
    options: Optional[SolveOptions] = None,
) -> IndividualTotalEnergyDecompositionResult:
    """Solve all buildings independently and retain complete audit evidence."""

    validate_individual_total_energy_separability(problem)
    if mode not in {"bounded", "relaxation", "schedule"}:
        raise ValueError(f"Unknown decomposition mode {mode!r}.")
    if options is None:
        # Zero preserves the exact lexicographic service optimum.  Giving every
        # isolated building the usual numerical slack would multiply that
        # allowance by the number of buildings with soft EV targets.
        options = SolveOptions(
            throughput_tiebreaker_eur_per_kwh=0.0,
            lexicographic_shortfall_tolerance_kwh=0.0,
        )
    elif options.lexicographic_shortfall_tolerance_kwh != 0.0:
        raise ValueError(
            "Exact individual decomposition requires "
            "lexicographic_shortfall_tolerance_kwh=0. A per-building non-zero "
            "tolerance would change the aggregate feasible set."
        )
    elif options.throughput_tiebreaker_eur_per_kwh != 0.0:
        raise ValueError(
            "Exact individual decomposition requires "
            "throughput_tiebreaker_eur_per_kwh=0 so the LP lower bound and "
            "mixed-integer upper candidate use the same economic objective."
        )
    building_solutions: list[IndividualTotalEnergyBuildingSolution] = []
    for subproblem in split_individual_total_energy_problems(problem):
        if mode == "bounded":
            result: BuildingResult = solve_bounded_total_energy_oracle(
                subproblem, options
            )
        elif mode == "relaxation":
            result = solve_total_energy_relaxation(subproblem, options)
        else:
            result = solve_total_energy_schedule(subproblem, options)
        building_solutions.append(
            IndividualTotalEnergyBuildingSolution(
                building_id=subproblem.building_ids[0],
                subproblem_id=subproblem.problem_id,
                result=result,
            )
        )

    frozen_solutions = tuple(building_solutions)
    combined: BuildingResult
    if mode == "bounded":
        combined = _combine_bounded_results(problem, frozen_solutions)
    else:
        combined = combine_individual_total_energy_results(
            problem, frozen_solutions
        )
    return IndividualTotalEnergyDecompositionResult(
        source_problem_id=problem.problem_id,
        mode=mode,
        buildings=frozen_solutions,
        combined=combined,
    )


__all__ = [
    "IndividualTotalEnergyBuildingSolution",
    "IndividualTotalEnergyDecompositionResult",
    "combine_individual_total_energy_results",
    "solve_decomposed_individual_total_energy",
    "split_individual_total_energy_problems",
    "validate_individual_total_energy_separability",
]
