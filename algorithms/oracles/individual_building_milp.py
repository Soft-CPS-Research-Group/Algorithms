"""Independent-building decomposition for the fixed-service battery oracle.

This module solves one :class:`PerfectForesightProblem` per building and then
combines the semantic battery schedules.  Each building is metered separately:
exports from one building cannot offset imports in another building.

The scope is intentionally narrow.  These are perfect-foresight, fixed-service,
battery-only bounds for the supplied linear models.  They are not full-home or
global community optima, and conservative schedules still require simulator
replay before being called CityLearn-feasible.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from algorithms.oracles.perfect_foresight_milp import (
    BoundedOracleResult,
    PerfectForesightProblem,
    SemanticActionSeries,
    SemanticSchedule,
    SolveOptions,
    solve_bounded_oracle,
)


_NUMERICAL_TOLERANCE = 1.0e-8


def _reject_aggregated_batteries(problem: PerfectForesightProblem) -> None:
    """Fail closed when fixed-service metadata describes aggregate assets."""

    physical_count = problem.metadata.get("physical_battery_count")
    oracle_count = problem.metadata.get("oracle_battery_group_count")
    if physical_count is not None and int(physical_count) != len(problem.batteries):
        raise ValueError(
            "Individual-building decomposition requires a non-aggregated problem; "
            f"metadata reports {physical_count} physical batteries but the problem "
            f"contains {len(problem.batteries)} oracle assets."
        )
    if oracle_count is not None and int(oracle_count) != len(problem.batteries):
        raise ValueError(
            "Individual-building decomposition requires a non-aggregated problem; "
            "oracle_battery_group_count does not match the battery asset count."
        )

    for group in problem.metadata.get("battery_groups", ()):
        members = tuple(group.get("members", ()))
        if len(members) != 1:
            raise ValueError(
                "Individual-building decomposition requires a non-aggregated problem; "
                "a battery group contains multiple physical members."
            )

        member = members[0]
        oracle_key = (
            str(group.get("oracle_building_id", "")),
            str(group.get("oracle_action_name", "")),
        )
        member_key = (
            str(member.get("building_id", "")),
            str(member.get("action_name", "")),
        )
        if oracle_key != member_key:
            raise ValueError(
                "Individual-building decomposition requires identity-preserving "
                "battery groups; oracle and physical semantic keys differ."
            )


def split_individual_building_problems(
    problem: PerfectForesightProblem,
) -> tuple[PerfectForesightProblem, ...]:
    """Split a non-aggregated problem into one isolated problem per building.

    The original building order is retained.  Every battery is assigned by its
    semantic ``building_id`` and appears in exactly one subproblem.
    """

    _reject_aggregated_batteries(problem)
    subproblems: list[PerfectForesightProblem] = []
    for building_index, building_id in enumerate(problem.building_ids):
        batteries = tuple(
            battery for battery in problem.batteries if battery.building_id == building_id
        )
        metadata = {
            **dict(problem.metadata),
            "scope": "individual_building_fixed_service_battery_only",
            "global_optimum_claim": False,
            "source_problem_id": problem.problem_id,
            "individual_building_id": building_id,
            "cost_semantics": "individual_import_with_zero_export_credit",
        }
        subproblems.append(
            PerfectForesightProblem(
                problem_id=f"{problem.problem_id}::individual::{building_id}",
                timestep_hours=problem.timestep_hours,
                building_ids=(building_id,),
                price_eur_per_kwh=problem.price_eur_per_kwh,
                base_net_load_kwh=problem.base_net_load_kwh[
                    building_index : building_index + 1, :
                ],
                batteries=batteries,
                metadata=metadata,
            )
        )
    return tuple(subproblems)


def combine_individual_building_schedules(
    problem: PerfectForesightProblem,
    schedules: Sequence[SemanticSchedule],
) -> SemanticSchedule:
    """Combine complete per-building schedules in original semantic order.

    This function validates horizon, timestep, building ownership, missing
    actions and unexpected actions before combining.  It never maps schedules
    by vector position.
    """

    expected_keys = tuple(battery.semantic_key for battery in problem.batteries)
    expected_key_set = set(expected_keys)
    series_by_key: dict[tuple[str, str], SemanticActionSeries] = {}
    seen_buildings: set[str] = set()

    for schedule in schedules:
        if schedule.horizon != problem.horizon:
            raise ValueError(
                f"Schedule {schedule.problem_id!r} horizon {schedule.horizon} does "
                f"not match source horizon {problem.horizon}."
            )
        if not math.isclose(
            schedule.timestep_hours,
            problem.timestep_hours,
            rel_tol=0.0,
            abs_tol=_NUMERICAL_TOLERANCE,
        ):
            raise ValueError(
                f"Schedule {schedule.problem_id!r} timestep does not match the source problem."
            )

        schedule_buildings = {item.building_id for item in schedule.series}
        if len(schedule_buildings) > 1:
            raise ValueError(
                f"Schedule {schedule.problem_id!r} contains more than one building."
            )
        if schedule_buildings & seen_buildings:
            duplicate = sorted(schedule_buildings & seen_buildings)[0]
            raise ValueError(f"Multiple individual schedules supplied for building {duplicate!r}.")
        seen_buildings.update(schedule_buildings)

        for item in schedule.series:
            key = item.building_id, item.action_name
            if key not in expected_key_set:
                raise ValueError(f"Unexpected semantic action series: {key!r}.")
            if key in series_by_key:
                raise ValueError(f"Duplicate semantic action series: {key!r}.")
            series_by_key[key] = item

    missing = [key for key in expected_keys if key not in series_by_key]
    if missing:
        raise ValueError(f"Missing semantic action series: {missing!r}.")

    return SemanticSchedule(
        problem_id=f"{problem.problem_id}::individual::combined",
        horizon=problem.horizon,
        timestep_hours=problem.timestep_hours,
        series=tuple(series_by_key[key] for key in expected_keys),
        metadata={
            "scope": "individual_building_fixed_service_battery_only",
            "global_optimum_claim": False,
            "source_problem_id": problem.problem_id,
            "cost_semantics": "sum_of_individual_imports_with_zero_export_credit",
            "requires_citylearn_replay": True,
        },
    )


@dataclass(frozen=True)
class IndividualBuildingOracleSolution:
    """Bound certificate for one isolated building."""

    building_id: str
    result: BoundedOracleResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "building_id": self.building_id,
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class IndividualBuildingOracleResult:
    """Aggregate certificate and schedule for independent building solves."""

    source_problem_id: str
    buildings: tuple[IndividualBuildingOracleSolution, ...]
    combined_schedule: Optional[SemanticSchedule]
    combined_grid_import_kwh: Optional[tuple[float, ...]]
    certified_lower_bound_eur: Optional[float]
    model_feasible_upper_bound_eur: Optional[float]
    absolute_gap_eur: Optional[float]
    relative_gap: Optional[float]
    certificate_valid: bool
    guarantee: str = (
        "Bounds are the sum of isolated, fixed-service, battery-only building "
        "problems with zero export credit. They are not full-home or global "
        "community optima. The combined schedule requires simulator replay."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_problem_id": self.source_problem_id,
            "certified_lower_bound_eur": self.certified_lower_bound_eur,
            "model_feasible_upper_bound_eur": self.model_feasible_upper_bound_eur,
            "absolute_gap_eur": self.absolute_gap_eur,
            "relative_gap": self.relative_gap,
            "certificate_valid": self.certificate_valid,
            "guarantee": self.guarantee,
            "combined_grid_import_kwh": (
                None
                if self.combined_grid_import_kwh is None
                else list(self.combined_grid_import_kwh)
            ),
            "combined_schedule": (
                None if self.combined_schedule is None else self.combined_schedule.to_dict()
            ),
            "buildings": [item.to_dict() for item in self.buildings],
        }


def _sum_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def solve_individual_building_oracles(
    problem: PerfectForesightProblem,
    options: Optional[SolveOptions] = None,
    *,
    max_workers: int = 1,
) -> IndividualBuildingOracleResult:
    """Solve and combine independent fixed-service battery-only building bounds."""

    subproblems = split_individual_building_problems(problem)
    worker_count = int(max_workers)
    if worker_count < 1:
        raise ValueError("max_workers must be at least 1.")

    if worker_count == 1 or len(subproblems) == 1:
        bounded_results = tuple(
            solve_bounded_oracle(subproblem, options) for subproblem in subproblems
        )
    else:
        # Each building is mathematically independent.  A process pool avoids
        # serializing the HiGHS solves behind Python's GIL and, unlike solver
        # threads, keeps one explicit time limit per building/formulation.
        # HiGHS may already have initialized native thread state in callers
        # that perform a serial solve before this function.  ``spawn`` keeps
        # that state out of workers and avoids fork-after-solver deadlocks.
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(subproblems)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            bounded_results = tuple(
                pool.map(
                    solve_bounded_oracle,
                    subproblems,
                    [options] * len(subproblems),
                )
            )

    building_solutions = tuple(
        IndividualBuildingOracleSolution(
            building_id=subproblem.building_ids[0],
            result=bounded,
        )
        for subproblem, bounded in zip(subproblems, bounded_results)
    )

    lower = _sum_optional(
        [item.result.certified_lower_bound_eur for item in building_solutions]
    )
    upper = _sum_optional(
        [item.result.model_feasible_upper_bound_eur for item in building_solutions]
    )
    certificate_valid = (
        all(item.result.certificate_valid for item in building_solutions)
        and lower is not None
        and upper is not None
    )

    gap: Optional[float] = None
    relative_gap: Optional[float] = None
    if certificate_valid:
        assert lower is not None and upper is not None
        raw_gap = upper - lower
        tolerance = _NUMERICAL_TOLERANCE * max(1.0, abs(upper), abs(lower))
        if raw_gap < -tolerance:
            raise RuntimeError(
                "Combined individual lower bound exceeds the combined feasible cost."
            )
        gap = max(raw_gap, 0.0)
        relative_gap = 0.0 if gap == 0.0 else gap / max(abs(upper), _NUMERICAL_TOLERANCE)

    schedules = [item.result.conservative.schedule for item in building_solutions]
    combined_schedule: Optional[SemanticSchedule] = None
    if all(schedule is not None for schedule in schedules):
        combined_schedule = combine_individual_building_schedules(
            problem,
            [schedule for schedule in schedules if schedule is not None],
        )

    grid_imports = [item.result.conservative.grid_import_kwh for item in building_solutions]
    combined_grid_import: Optional[tuple[float, ...]] = None
    if all(values is not None for values in grid_imports):
        grid_array = np.sum(
            np.asarray([values for values in grid_imports if values is not None], dtype=np.float64),
            axis=0,
        )
        combined_grid_import = tuple(float(value) for value in grid_array)

    return IndividualBuildingOracleResult(
        source_problem_id=problem.problem_id,
        buildings=building_solutions,
        combined_schedule=combined_schedule,
        combined_grid_import_kwh=combined_grid_import,
        certified_lower_bound_eur=lower,
        model_feasible_upper_bound_eur=upper,
        absolute_gap_eur=gap,
        relative_gap=relative_gap,
        certificate_valid=certificate_valid,
    )


__all__ = [
    "IndividualBuildingOracleResult",
    "IndividualBuildingOracleSolution",
    "combine_individual_building_schedules",
    "solve_individual_building_oracles",
    "split_individual_building_problems",
]
