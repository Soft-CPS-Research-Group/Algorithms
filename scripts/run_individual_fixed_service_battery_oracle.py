#!/usr/bin/env python3
"""Solve one isolated fixed-service stationary-battery MILP per building."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles import (
    SolveOptions,
    build_fixed_service_battery_problem,
    solve_individual_building_oracles,
)


def _building_rows(result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result.buildings:
        bounded = item.result
        rows.append(
            {
                "building": item.building_id,
                "scope": "individual_building_fixed_service_battery_only",
                "global_optimum_claim": False,
                "certified_lower_bound_eur": bounded.certified_lower_bound_eur,
                "model_feasible_upper_bound_eur": bounded.model_feasible_upper_bound_eur,
                "absolute_gap_eur": bounded.absolute_gap_eur,
                "relative_gap": bounded.relative_gap,
                "certificate_valid": bounded.certificate_valid,
                "lower_status": bounded.lower.solver.status,
                "lower_optimal": bounded.lower.solver.optimal,
                "conservative_status": bounded.conservative.solver.status,
                "conservative_optimal": bounded.conservative.solver.optimal,
                "conservative_mip_gap": bounded.conservative.solver.mip_gap,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--simulation-data", type=Path, required=True)
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--time-limit-seconds", type=float, default=120.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1.0e-4)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of independent building solves to run concurrently.",
    )
    args = parser.parse_args(argv)

    built = build_fixed_service_battery_problem(
        schema_path=args.schema,
        simulation_data_directory=args.simulation_data,
        problem_id=args.problem_id,
        episode=args.episode,
        aggregate_equivalent_batteries=False,
    )
    result = solve_individual_building_oracles(
        built.problem,
        SolveOptions(
            time_limit_seconds=args.time_limit_seconds,
            mip_relative_gap=args.mip_relative_gap,
        ),
        max_workers=args.max_workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "problem.json").write_text(
        json.dumps(built.problem.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.combined_schedule is not None:
        (args.output_dir / "replay_schedule.json").write_text(
            result.combined_schedule.to_json(indent=2) + "\n",
            encoding="utf-8",
        )

    rows = _building_rows(result)
    _write_csv(args.output_dir / "building_summary.csv", rows)
    summary = {
        "problem_id": built.problem.problem_id,
        "scope": "individual_building_fixed_service_battery_only",
        "global_optimum_claim": False,
        "full_home_optimum_claim": False,
        "certificate_valid": result.certificate_valid,
        "certified_lower_bound_eur_sum": result.certified_lower_bound_eur,
        "model_feasible_upper_bound_eur_sum": result.model_feasible_upper_bound_eur,
        "absolute_gap_eur_sum": result.absolute_gap_eur,
        "relative_gap": result.relative_gap,
        "building_count": len(result.buildings),
        "guarantee": result.guarantee,
        "diagnostics": built.diagnostics.to_dict(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.certificate_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
