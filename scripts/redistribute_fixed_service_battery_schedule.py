#!/usr/bin/env python3
"""Redistribute an aggregate-expanded battery schedule for local emissions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles import (
    SemanticSchedule,
    build_fixed_service_battery_problem,
    redistribute_equivalent_battery_schedule,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--simulation-data", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--carbon-intensity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--window-steps", type=int, default=24)
    args = parser.parse_args()

    built = build_fixed_service_battery_problem(
        schema_path=args.schema,
        simulation_data_directory=args.simulation_data,
        problem_id="local-carbon-dispatch-source",
        episode=args.episode,
        aggregate_equivalent_batteries=False,
    )
    raw_schedule = args.schedule.read_bytes()
    if args.schedule.suffix == ".gz":
        raw_schedule = gzip.decompress(raw_schedule)
    schedule = SemanticSchedule.from_json(raw_schedule.decode("utf-8"))
    carbon_frame = pd.read_parquet(args.carbon_intensity)
    if "carbon_intensity" not in carbon_frame:
        raise ValueError("Carbon-intensity parquet lacks 'carbon_intensity'.")
    carbon = carbon_frame["carbon_intensity"].to_numpy(dtype=float)[
        : built.problem.horizon
    ]
    result = redistribute_equivalent_battery_schedule(
        built.problem,
        schedule,
        carbon,
        window_steps=args.window_steps,
        progress_callback=lambda window, end, horizon: (
            print(
                f"redistribution progress: window={window} "
                f"steps={end}/{horizon}",
                flush=True,
            )
            if window % 25 == 0 or end == horizon
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = result.schedule.to_json(indent=None).encode("utf-8")
    args.output.write_bytes(
        gzip.compress(encoded, compresslevel=9, mtime=0)
        if args.output.suffix == ".gz"
        else encoded
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary = result.to_dict()
    summary.pop("schedule", None)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
