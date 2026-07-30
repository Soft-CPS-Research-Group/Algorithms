#!/usr/bin/env python3
"""Solve a serialized complete individual/community energy MILP problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles import (
    SolveOptions,
    TotalEnergyBoundedResult,
    TotalEnergyProblem,
    TotalEnergyResult,
    solve_decomposed_individual_total_energy,
    solve_bounded_total_energy_oracle,
    solve_total_energy_relaxation,
    solve_total_energy_schedule,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("bounded", "relaxation", "schedule"),
        default="bounded",
    )
    parser.add_argument("--time-limit-seconds", type=float)
    parser.add_argument("--mip-relative-gap", type=float)
    parser.add_argument("--node-limit", type=int)
    parser.add_argument("--display-solver-output", action="store_true")
    parser.add_argument(
        "--decompose-individual",
        action="store_true",
        help=(
            "Solve settlement=individual as one exact isolated model per "
            "building and retain per-building results. Community settlement "
            "is rejected."
        ),
    )
    parser.add_argument(
        "--lexicographic-shortfall-tolerance-kwh",
        type=float,
        default=None,
        help=(
            "Allowed numerical slack above the minimum aggregate EV departure "
            "shortfall in the cost stage (default: 0.001 kWh for a joint solve; "
            "0 for exact individual decomposition). A non-zero value is "
            "rejected with --decompose-individual because it would be applied "
            "once per building."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    problem_path = args.problem.resolve()
    problem = TotalEnergyProblem.from_json(problem_path.read_text(encoding="utf-8"))
    lexicographic_shortfall_tolerance_kwh = (
        (0.0 if args.decompose_individual else 1.0e-3)
        if args.lexicographic_shortfall_tolerance_kwh is None
        else args.lexicographic_shortfall_tolerance_kwh
    )
    options = SolveOptions(
        time_limit_seconds=args.time_limit_seconds,
        mip_relative_gap=args.mip_relative_gap,
        node_limit=args.node_limit,
        display_solver_output=args.display_solver_output,
        throughput_tiebreaker_eur_per_kwh=(
            0.0 if args.decompose_individual else 1.0e-9
        ),
        lexicographic_shortfall_tolerance_kwh=(
            lexicographic_shortfall_tolerance_kwh
        ),
    )
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "problem.json").write_text(
        problem.to_json() + "\n", encoding="utf-8"
    )

    decomposition = None
    if args.decompose_individual:
        decomposition = solve_decomposed_individual_total_energy(
            problem, mode=args.mode, options=options
        )
        result = decomposition.combined
        payload = decomposition.to_dict()
        schedule = decomposition.schedule
    elif args.mode == "bounded":
        result = solve_bounded_total_energy_oracle(problem, options)
        payload = result.to_dict()
        schedule = result.conservative.schedule
    else:
        result = (
            solve_total_energy_relaxation(problem, options)
            if args.mode == "relaxation"
            else solve_total_energy_schedule(problem, options)
        )
        payload = result.to_dict()
        schedule = result.schedule

    if args.mode == "bounded":
        assert isinstance(result, TotalEnergyBoundedResult)
        summary = {
            "problem_id": problem.problem_id,
            "settlement": problem.settlement,
            "horizon": problem.horizon,
            "mode": args.mode,
            "certificate_valid": result.certificate_valid,
            "certified_lower_bound_eur": result.certified_lower_bound_eur,
            "model_feasible_upper_bound_eur": result.model_feasible_upper_bound_eur,
            "relative_gap": result.relative_gap,
            "conservative_status": result.conservative.solver.status,
            "minimum_total_ev_shortfall_kwh": (
                result.conservative.minimum_total_ev_shortfall_kwh
            ),
            "realized_total_ev_shortfall_kwh": (
                result.conservative.realized_total_ev_shortfall_kwh
            ),
            "lexicographic_shortfall_tolerance_kwh": (
                lexicographic_shortfall_tolerance_kwh
            ),
            "minimum_ev_shortfall_by_building_kwh": dict(
                result.conservative.minimum_ev_shortfall_by_building_kwh
            ),
            "realized_ev_shortfall_by_building_kwh": dict(
                result.conservative.realized_ev_shortfall_by_building_kwh
            ),
            "lexicographic_shortfall_cap_by_building_kwh": dict(
                result.conservative.lexicographic_shortfall_cap_by_building_kwh
            ),
            "service_phase_status": result.conservative.service_phase_status,
            "service_phase_optimal": result.conservative.service_phase_optimal,
        }
    else:
        assert isinstance(result, TotalEnergyResult)
        summary = {
            "problem_id": problem.problem_id,
            "settlement": problem.settlement,
            "horizon": problem.horizon,
            "mode": args.mode,
            "status": result.solver.status,
            "has_solution": result.solver.has_solution,
            "cost_eur": result.cost_eur,
            "minimum_total_ev_shortfall_kwh": (
                result.minimum_total_ev_shortfall_kwh
            ),
            "realized_total_ev_shortfall_kwh": (
                result.realized_total_ev_shortfall_kwh
            ),
            "lexicographic_shortfall_tolerance_kwh": (
                result.lexicographic_shortfall_tolerance_kwh
            ),
            "minimum_ev_shortfall_by_building_kwh": dict(
                result.minimum_ev_shortfall_by_building_kwh
            ),
            "realized_ev_shortfall_by_building_kwh": dict(
                result.realized_ev_shortfall_by_building_kwh
            ),
            "lexicographic_shortfall_cap_by_building_kwh": dict(
                result.lexicographic_shortfall_cap_by_building_kwh
            ),
            "service_phase_status": result.service_phase_status,
            "service_phase_optimal": result.service_phase_optimal,
        }
    if decomposition is not None:
        summary.update(
            {
                "decomposition": "exact_individual_settlement_by_building",
                "community_optimum_claim": False,
                "building_count": len(decomposition.buildings),
                "buildings": [
                    {
                        "building_id": item.building_id,
                        "subproblem_id": item.subproblem_id,
                        **(
                            {
                                "certificate_valid": item.result.certificate_valid,
                                "certified_lower_bound_eur": (
                                    item.result.certified_lower_bound_eur
                                ),
                                "model_feasible_upper_bound_eur": (
                                    item.result.model_feasible_upper_bound_eur
                                ),
                                "lower_status": item.result.lower.solver.status,
                                "conservative_status": (
                                    item.result.conservative.solver.status
                                ),
                                "minimum_total_ev_shortfall_kwh": (
                                    item.result.conservative.minimum_total_ev_shortfall_kwh
                                ),
                                "realized_total_ev_shortfall_kwh": (
                                    item.result.conservative.realized_total_ev_shortfall_kwh
                                ),
                            }
                            if isinstance(item.result, TotalEnergyBoundedResult)
                            else {
                                "status": item.result.solver.status,
                                "has_solution": item.result.solver.has_solution,
                                "cost_eur": item.result.cost_eur,
                                "minimum_total_ev_shortfall_kwh": (
                                    item.result.minimum_total_ev_shortfall_kwh
                                ),
                                "realized_total_ev_shortfall_kwh": (
                                    item.result.realized_total_ev_shortfall_kwh
                                ),
                            }
                        ),
                    }
                    for item in decomposition.buildings
                ],
                "guarantee": decomposition.guarantee,
            }
        )
    (output_directory / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if schedule is not None:
        (output_directory / "replay_schedule.json").write_text(
            schedule.to_json() + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if schedule is not None or args.mode == "relaxation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
