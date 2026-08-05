#!/usr/bin/env python3
"""Isolate total-energy LP infeasibility by building and asset class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles import (
    SolveOptions,
    TotalEnergyProblem,
    solve_total_energy_relaxation,
)


def _subproblem(
    problem: TotalEnergyProblem,
    building_id: str,
    *,
    include_storage: bool,
    include_ev: bool,
    include_deferrable: bool,
    include_service: bool,
) -> TotalEnergyProblem:
    building_index = problem.building_ids.index(building_id)
    return TotalEnergyProblem(
        problem_id=f"{problem.problem_id}::{building_id}",
        timestep_hours=problem.timestep_hours,
        building_ids=(building_id,),
        price_eur_per_kwh=problem.price_eur_per_kwh,
        base_net_load_kwh=problem.base_net_load_kwh[building_index : building_index + 1],
        settlement="individual",
        stationary_storage=tuple(
            item
            for item in problem.stationary_storage
            if include_storage and item.building_id == building_id
        ),
        ev_sessions=tuple(
            item
            for item in problem.ev_sessions
            if include_ev and item.building_id == building_id
        ),
        deferrable_cycles=tuple(
            item
            for item in problem.deferrable_cycles
            if include_deferrable and item.building_id == building_id
        ),
        electrical_services=tuple(
            item
            for item in problem.electrical_services
            if include_service and item.building_id == building_id
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=60.0)
    args = parser.parse_args()
    problem = TotalEnergyProblem.from_json(args.problem.read_text(encoding="utf-8"))
    scenarios = {
        "base_only": (False, False, False, False),
        "storage": (True, False, False, False),
        "ev": (False, True, False, False),
        "deferrable": (False, False, True, False),
        "all_without_service": (True, True, True, False),
        "all_with_service": (True, True, True, True),
    }
    rows = []
    for building_id in problem.building_ids:
        for scenario, flags in scenarios.items():
            subproblem = _subproblem(
                problem,
                building_id,
                include_storage=flags[0],
                include_ev=flags[1],
                include_deferrable=flags[2],
                include_service=flags[3],
            )
            result = solve_total_energy_relaxation(
                subproblem,
                SolveOptions(time_limit_seconds=args.time_limit_seconds),
            )
            rows.append(
                {
                    "building_id": building_id,
                    "scenario": scenario,
                    "status": result.solver.status,
                    "has_solution": result.solver.has_solution,
                    "cost_eur": result.cost_eur,
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
            if not result.solver.has_solution:
                break
    print(json.dumps({"summary": rows}, indent=2, sort_keys=True))
    return 0 if all(row["has_solution"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
