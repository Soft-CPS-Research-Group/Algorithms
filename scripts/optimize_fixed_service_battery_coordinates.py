#!/usr/bin/env python3
"""Improve a feasible physical battery schedule by coordinate descent."""

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
    optimize_physical_battery_schedule_coordinate_descent,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--simulation-data", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--initial-schedule", type=Path, required=True)
    parser.add_argument("--carbon-intensity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cost-limit-eur", type=float, required=True)
    parser.add_argument("--mean-ramp-limit-kwh", type=float)
    parser.add_argument("--mean-daily-peak-limit-kwh", type=float)
    parser.add_argument("--all-time-peak-limit-kwh", type=float)
    parser.add_argument("--max-sweeps", type=int, default=2)
    parser.add_argument("--throughput-tiebreaker", type=float, default=1.0e-8)
    parser.add_argument("--relax-battery-direction", action="store_true")
    parser.add_argument("--time-limit-seconds-per-building", type=float, default=30.0)
    parser.add_argument(
        "--solver",
        choices=("choose", "simplex", "ipm"),
        default="simplex",
    )
    args = parser.parse_args(argv)

    built = build_fixed_service_battery_problem(
        schema_path=args.schema,
        simulation_data_directory=args.simulation_data,
        problem_id="fixed-service-physical-coordinate-carbon",
        episode=args.episode,
        aggregate_equivalent_batteries=False,
    )
    raw_schedule = args.initial_schedule.read_bytes()
    if args.initial_schedule.suffix == ".gz":
        raw_schedule = gzip.decompress(raw_schedule)
    initial_schedule = SemanticSchedule.from_json(raw_schedule.decode("utf-8"))
    carbon_frame = pd.read_parquet(args.carbon_intensity)
    if "carbon_intensity" not in carbon_frame:
        raise ValueError("Carbon-intensity parquet lacks 'carbon_intensity'.")
    carbon = carbon_frame["carbon_intensity"].to_numpy(dtype=float)[
        : built.problem.horizon
    ]
    shaping = ScorecardShapingOptions(
        community_cost_limit_eur=args.cost_limit_eur,
        ramping_weight=0.0,
        daily_peak_weight=0.0,
        all_time_peak_weight=0.0,
        emissions_weight=1.0,
        mean_absolute_ramp_limit_kwh=args.mean_ramp_limit_kwh,
        mean_daily_peak_import_limit_kwh=args.mean_daily_peak_limit_kwh,
        all_time_peak_import_limit_kwh=args.all_time_peak_limit_kwh,
        throughput_tiebreaker=args.throughput_tiebreaker,
        enforce_exclusive_battery_direction=not args.relax_battery_direction,
        emissions_accounting="gross_member_import",
    )
    result = optimize_physical_battery_schedule_coordinate_descent(
        built.problem,
        initial_schedule,
        carbon,
        shaping,
        SolveOptions(
            time_limit_seconds=args.time_limit_seconds_per_building,
            solver=args.solver,
        ),
        max_sweeps=args.max_sweeps,
        progress_callback=lambda sweep, battery, metrics, accepted: print(
            f"coordinate progress: sweep={sweep} battery={battery}/"
            f"{len(built.problem.batteries)} accepted={int(accepted)} "
            f"emissions={metrics.community_emissions_kgco2:.6f} "
            f"cost={metrics.community_cost_eur:.6f}",
            flush=True,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = result.schedule.to_json(indent=None).encode("utf-8")
    args.output.write_bytes(
        gzip.compress(encoded, compresslevel=9, mtime=0)
        if args.output.suffix == ".gz"
        else encoded
    )
    summary = result.to_dict()
    summary.pop("schedule", None)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
