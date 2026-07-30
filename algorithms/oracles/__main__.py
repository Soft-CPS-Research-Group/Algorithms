"""Minimal JSON CLI for the bounded perfect-foresight core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from algorithms.oracles.perfect_foresight_milp import (
    PerfectForesightProblem,
    SolveOptions,
    solve_bounded_oracle,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve a bounded perfect-foresight battery problem.")
    parser.add_argument("problem_json", type=Path, help="JSON file matching PerfectForesightProblem.to_dict().")
    parser.add_argument("--output", type=Path, help="Optional result JSON path; stdout is used otherwise.")
    parser.add_argument("--time-limit-seconds", type=float)
    parser.add_argument("--mip-relative-gap", type=float)
    args = parser.parse_args(argv)

    problem = PerfectForesightProblem.from_dict(
        json.loads(args.problem_json.read_text(encoding="utf-8"))
    )
    result = solve_bounded_oracle(
        problem,
        SolveOptions(
            time_limit_seconds=args.time_limit_seconds,
            mip_relative_gap=args.mip_relative_gap,
        ),
    )
    payload = result.to_json(indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0 if result.certificate_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
