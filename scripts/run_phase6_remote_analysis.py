#!/usr/bin/env python3
"""Collect remote Phase 6 jobs and build aggregate plus per-building reports."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.collect_remote_results import DEFAULT_SERVER


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)  # noqa: S603 - local helper invoking repo scripts with explicit args.


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.environ.get("OPEVA_SERVER", DEFAULT_SERVER))
    parser.add_argument("--job-id", action="append", default=[], help="Remote orchestrator job id. Can be repeated.")
    parser.add_argument(
        "--jobs-file",
        type=Path,
        action="append",
        default=[],
        help="Text/CSV/JSON file with job ids. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-lines", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Do not call the orchestrator; build reports from an existing output-dir/summary.csv.",
    )
    parser.add_argument("--ev-feasible-min", type=float, default=0.999)
    parser.add_argument("--ev-within-min", type=float, default=0.80)
    parser.add_argument("--cost-near-pct", type=float, default=5.0)
    parser.add_argument("--max-grid-violation-kwh", type=float, default=1e-6)
    parser.add_argument("--battery-throughput-ratio-warn", type=float, default=3.0)
    parser.add_argument("--peak-ratio-warn", type=float, default=1.05)
    parser.add_argument("--solar-self-consumption-warn", type=float, default=0.50)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    scripts = _script_dir()
    python = sys.executable

    if not args.skip_collect:
        collect_cmd = [
            python,
            str(scripts / "collect_remote_results.py"),
            "--server",
            str(args.server),
            "--output-dir",
            str(args.output_dir),
            "--tail-lines",
            str(args.tail_lines),
            "--timeout",
            str(args.timeout),
        ]
        for job_id in args.job_id:
            collect_cmd.extend(["--job-id", str(job_id)])
        for jobs_file in args.jobs_file:
            collect_cmd.extend(["--jobs-file", str(jobs_file)])
        _run(collect_cmd)

    summary_csv = args.output_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"Missing summary CSV: {summary_csv}", file=sys.stderr)
        return 2

    common_gate_args = [
        "--ev-feasible-min",
        str(args.ev_feasible_min),
        "--ev-within-min",
        str(args.ev_within_min),
        "--max-grid-violation-kwh",
        str(args.max_grid_violation_kwh),
        "--battery-throughput-ratio-warn",
        str(args.battery_throughput_ratio_warn),
    ]

    _run(
        [
            python,
            str(scripts / "build_phase6_remote_scorecard.py"),
            "--summary-csv",
            str(summary_csv),
            "--output-dir",
            str(args.output_dir),
            "--cost-near-pct",
            str(args.cost_near_pct),
            "--peak-ratio-warn",
            str(args.peak_ratio_warn),
            *common_gate_args,
        ]
    )
    _run(
        [
            python,
            str(scripts / "build_remote_job_report.py"),
            "--results-dir",
            str(args.output_dir),
            "--summary-csv",
            str(summary_csv),
        ]
    )
    _run(
        [
            python,
            str(scripts / "build_phase6_building_scorecard.py"),
            "--results-dir",
            str(args.output_dir),
            "--output-dir",
            str(args.output_dir),
            "--solar-self-consumption-warn",
            str(args.solar_self_consumption_warn),
            *common_gate_args,
        ]
    )

    print("Generated:")
    for name in (
        "summary.csv",
        "scorecard.csv",
        "scorecard.md",
        "job_report.csv",
        "job_report.md",
        "building_scorecard.csv",
        "building_scorecard.md",
    ):
        print(args.output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
