#!/usr/bin/env python3
"""Build and solve the conditional CityLearn stationary-battery oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles import (
    SolveOptions,
    expand_aggregated_battery_schedule,
    solve_bounded_oracle,
)
from algorithms.oracles.citylearn_fixed_service import build_fixed_service_battery_problem


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--simulation-data", type=Path, required=True)
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--time-limit-seconds", type=float, default=120.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1.0e-4)
    args = parser.parse_args(argv)

    built = build_fixed_service_battery_problem(
        schema_path=args.schema,
        simulation_data_directory=args.simulation_data,
        problem_id=args.problem_id,
        episode=args.episode,
    )
    result = solve_bounded_oracle(
        built.problem,
        SolveOptions(
            time_limit_seconds=args.time_limit_seconds,
            mip_relative_gap=args.mip_relative_gap,
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "problem.json").write_text(
        json.dumps(built.problem.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "result.json").write_text(result.to_json(indent=2) + "\n", encoding="utf-8")
    if result.conservative.schedule is not None:
        replay_schedule = expand_aggregated_battery_schedule(
            result.conservative.schedule,
            built.problem.metadata,
        )
        (args.output_dir / "replay_schedule.json").write_text(
            replay_schedule.to_json(indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "problem_id": built.problem.problem_id,
        "scope": built.problem.metadata["scope"],
        "global_optimum_claim": False,
        "diagnostics": built.diagnostics.to_dict(),
        "lower_bound_eur": result.certified_lower_bound_eur,
        "conservative_linear_cost_eur": result.model_feasible_upper_bound_eur,
        "linear_model_certificate_valid": result.certificate_valid,
        "conservative_schedule_requires_citylearn_replay": True,
        "physical_battery_count": built.problem.metadata["physical_battery_count"],
        "oracle_battery_group_count": built.problem.metadata["oracle_battery_group_count"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.certificate_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
