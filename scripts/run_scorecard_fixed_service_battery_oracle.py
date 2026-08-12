#!/usr/bin/env python3
"""Build a cost-constrained peak/ramp battery teaching schedule."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles import (
    ScorecardShapingOptions,
    SemanticSchedule,
    SolveOptions,
    build_fixed_service_battery_problem,
    expand_aggregated_battery_schedule,
    solve_scorecard_battery_schedule,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--simulation-data", type=Path, required=True)
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--cost-limit-eur", type=float, required=True)
    parser.add_argument("--ramping-weight", type=float, default=1.0)
    parser.add_argument("--daily-peak-weight", type=float, default=1.0)
    parser.add_argument("--all-time-peak-weight", type=float, default=0.25)
    parser.add_argument("--emissions-weight", type=float, default=0.0)
    parser.add_argument("--carbon-intensity", type=Path)
    parser.add_argument(
        "--initial-schedule",
        type=Path,
        help="Optional feasible semantic battery schedule used as a HiGHS warm start.",
    )
    parser.add_argument("--emissions-limit-kgco2", type=float)
    parser.add_argument("--mean-ramp-limit-kwh", type=float)
    parser.add_argument("--mean-daily-peak-limit-kwh", type=float)
    parser.add_argument("--all-time-peak-limit-kwh", type=float)
    parser.add_argument(
        "--relax-battery-direction",
        action="store_true",
        help=(
            "Solve the continuous relaxation. Positive throughput tie-breaking "
            "still discourages simultaneous charge and discharge."
        ),
    )
    parser.add_argument(
        "--physical-batteries",
        action="store_true",
        help="Optimize every building battery instead of an equivalent aggregate.",
    )
    parser.add_argument(
        "--emissions-accounting",
        choices=("district_net_import", "gross_member_import"),
        default="district_net_import",
    )
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1.0e-4)
    parser.add_argument(
        "--solver",
        choices=("choose", "simplex", "ipm"),
        default="choose",
        help="HiGHS LP algorithm; IPM is useful for the annual physical model.",
    )
    args = parser.parse_args(argv)

    built = build_fixed_service_battery_problem(
        schema_path=args.schema,
        simulation_data_directory=args.simulation_data,
        problem_id=args.problem_id,
        episode=args.episode,
        aggregate_equivalent_batteries=not args.physical_batteries,
    )
    shaping = ScorecardShapingOptions(
        community_cost_limit_eur=args.cost_limit_eur,
        ramping_weight=args.ramping_weight,
        daily_peak_weight=args.daily_peak_weight,
        all_time_peak_weight=args.all_time_peak_weight,
        emissions_weight=args.emissions_weight,
        community_emissions_limit_kgco2=args.emissions_limit_kgco2,
        mean_absolute_ramp_limit_kwh=args.mean_ramp_limit_kwh,
        mean_daily_peak_import_limit_kwh=args.mean_daily_peak_limit_kwh,
        all_time_peak_import_limit_kwh=args.all_time_peak_limit_kwh,
        enforce_exclusive_battery_direction=not args.relax_battery_direction,
        emissions_accounting=args.emissions_accounting,
    )
    carbon = None
    if args.carbon_intensity is not None:
        carbon_frame = pd.read_parquet(args.carbon_intensity)
        if "carbon_intensity" not in carbon_frame:
            raise ValueError("Carbon-intensity parquet lacks 'carbon_intensity'.")
        carbon = carbon_frame["carbon_intensity"].to_numpy(dtype=float)[
            : built.problem.horizon
        ]
    initial_schedule = None
    if args.initial_schedule is not None:
        raw_schedule = args.initial_schedule.read_bytes()
        if args.initial_schedule.suffix == ".gz":
            raw_schedule = gzip.decompress(raw_schedule)
        initial_schedule = SemanticSchedule.from_json(
            raw_schedule.decode("utf-8")
        )
    result = solve_scorecard_battery_schedule(
        built.problem,
        shaping,
        SolveOptions(
            time_limit_seconds=args.time_limit_seconds,
            mip_relative_gap=args.mip_relative_gap,
            solver=args.solver,
        ),
        carbon_intensity_kgco2_per_kwh=carbon,
        initial_schedule=initial_schedule,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "problem.json").write_text(
        json.dumps(built.problem.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "result.json").write_text(
        result.to_json(indent=2) + "\n", encoding="utf-8"
    )
    if result.schedule is not None:
        replay = expand_aggregated_battery_schedule(
            result.schedule,
            built.problem.metadata,
        )
        (args.output_dir / "replay_schedule.json").write_text(
            replay.to_json(indent=2) + "\n", encoding="utf-8"
        )

    summary = {
        "problem_id": built.problem.problem_id,
        "scope": "conditional_fixed_service_stationary_battery_scorecard_shaping",
        "global_optimum_claim": False,
        "diagnostics": built.diagnostics.to_dict(),
        "options": shaping.to_dict(),
        "solver": result.solver.to_dict(),
        "linear_model_community_cost_eur": result.community_cost_eur,
        "linear_model_total_import_kwh": result.total_import_kwh,
        "linear_model_mean_absolute_ramp_kwh": result.mean_absolute_ramp_kwh,
        "linear_model_mean_daily_peak_import_kwh": (
            result.mean_daily_peak_import_kwh
        ),
        "linear_model_all_time_peak_import_kwh": (
            result.all_time_peak_import_kwh
        ),
        "linear_model_community_emissions_kgco2": (
            result.community_emissions_kgco2
        ),
        "linear_model_gross_member_import_kwh": result.gross_member_import_kwh,
        "linear_model_simultaneous_charge_discharge_kwh": (
            result.simultaneous_charge_discharge_kwh
        ),
        "requires_citylearn_replay": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.solver.has_solution else 2


if __name__ == "__main__":
    raise SystemExit(main())
