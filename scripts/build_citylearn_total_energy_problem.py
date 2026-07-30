#!/usr/bin/env python3
"""Extract a serialized total-energy MILP problem from a CityLearn dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles import build_citylearn_total_energy_problem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--start-time-step", type=int, required=True)
    parser.add_argument("--end-time-step", type=int, required=True)
    parser.add_argument(
        "--settlement", choices=("individual", "community"), required=True
    )
    parser.add_argument(
        "--building",
        action="append",
        dest="buildings",
        help="Repeat to select buildings; omit to include all schema buildings.",
    )
    parser.add_argument("--electrical-service-reserve-kw", type=float, default=0.1)
    parser.add_argument("--problem-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    built = build_citylearn_total_energy_problem(
        schema_path=args.schema,
        start_time_step=args.start_time_step,
        end_time_step=args.end_time_step,
        problem_id=args.problem_id,
        settlement=args.settlement,
        building_ids=args.buildings,
        electrical_service_reserve_kw=args.electrical_service_reserve_kw,
    )
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "problem.json").write_text(
        built.problem.to_json() + "\n", encoding="utf-8"
    )
    summary = {
        "problem_id": built.problem.problem_id,
        "settlement": built.problem.settlement,
        "horizon": built.problem.horizon,
        "diagnostics": built.diagnostics.to_dict(),
        "problem_path": str(output_directory / "problem.json"),
    }
    (output_directory / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
