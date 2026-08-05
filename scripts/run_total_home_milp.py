#!/usr/bin/env python3
"""Build and solve one closed-window CityLearn total-home MILP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles import (
    SolveOptions,
    build_citylearn_total_home_problem,
    solve_total_home_milp,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument("--building", required=True)
    parser.add_argument("--start-time-step", type=int, required=True)
    parser.add_argument("--end-time-step", type=int, required=True, help="Exclusive source end.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1.0e-6)
    parser.add_argument(
        "--hard-ev-targets",
        action="store_true",
        help="Reject physically impossible EV targets instead of minimizing unavoidable shortfall first.",
    )
    parser.add_argument(
        "--ev-departure-soc-margin",
        type=float,
        default=0.0,
        help="Conservative SOC margin added to departure targets for replay calibration.",
    )
    args = parser.parse_args()

    built = build_citylearn_total_home_problem(
        schema_path=args.schema_path,
        building_id=args.building,
        start_time_step=args.start_time_step,
        end_time_step=args.end_time_step,
        allow_physically_infeasible_ev_shortfall=not args.hard_ev_targets,
        ev_departure_soc_margin=args.ev_departure_soc_margin,
    )
    problem = built.problem
    result = solve_total_home_milp(
        problem,
        SolveOptions(
            time_limit_seconds=args.time_limit_seconds,
            mip_relative_gap=args.mip_relative_gap,
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    problem_summary = {
        "problem_id": problem.problem_id,
        "building_id": problem.building_id,
        "horizon": problem.horizon,
        "timestep_hours": problem.timestep_hours,
        "ev_session_count": built.ev_session_count,
        "deferrable_cycle_count": built.deferrable_cycle_count,
        "stationary_storage_present": problem.stationary_storage is not None,
        "electrical_service_present": problem.electrical_service is not None,
        "metadata": dict(problem.metadata),
    }
    solution_summary = {
        "status": result.status,
        "status_code": result.status_code,
        "optimal": result.optimal,
        "has_solution": result.has_solution,
        "message": result.message,
        "objective_eur": result.objective_eur,
        "objective_lower_bound_eur": result.objective_lower_bound_eur,
        "mip_gap": result.mip_gap,
        "ev_departure_energy_kwh": dict(result.ev_departure_energy_kwh),
        "ev_departure_shortfall_kwh": dict(result.ev_departure_shortfall_kwh),
        "deferrable_start_time_step": dict(result.deferrable_start_time_step),
        "diagnostics": dict(result.diagnostics),
        "claim_scope": "optimal_for_individual_total_home_linear_model" if result.optimal else "incumbent_or_no_solution",
        "citylearn_replay_required": True,
        "community_optimum_claim": False,
    }
    (args.output_dir / "problem_summary.json").write_text(
        json.dumps(problem_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "solution.json").write_text(
        json.dumps(solution_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.schedule is not None:
        (args.output_dir / "schedule.json").write_text(
            result.schedule.to_json(indent=2) + "\n",
            encoding="utf-8",
        )
        with (args.output_dir / "trajectory.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_step", "grid_import_kw", "grid_net_power_kw"])
            writer.writerows(
                zip(range(problem.horizon), result.grid_import_kw, result.grid_net_power_kw)
            )
    print(json.dumps(solution_summary, indent=2, sort_keys=True))
    return 0 if result.has_solution else 2


if __name__ == "__main__":
    raise SystemExit(main())
