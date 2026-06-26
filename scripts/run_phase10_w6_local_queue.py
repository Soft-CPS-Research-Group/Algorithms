#!/usr/bin/env python3
"""Run selected Phase 10 W6 configs locally in a sequential queue."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matches(value: str, allowed: set[str]) -> bool:
    return not allowed or value in allowed


def _is_completed(base_dir: Path, job_id: str) -> bool:
    result_path = base_dir / "jobs" / job_id / "results" / "result.json"
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return result.get("status") == "completed"


def _select_rows(
    rows: Iterable[dict[str, str]],
    *,
    recipes: set[str],
    windows: set[str],
    seeds: set[str],
    algorithms: set[str],
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if _matches(row.get("recipe", ""), recipes)
        and _matches(row.get("window", ""), windows)
        and _matches(row.get("seed", ""), seeds)
        and _matches(row.get("algorithm", ""), algorithms)
    ]
    return sorted(selected, key=lambda row: (row.get("window", ""), row.get("recipe", ""), row.get("seed", "")))


def _split_csv(values: list[str]) -> set[str]:
    parsed: set[str] = set()
    for value in values:
        parsed.update(item.strip() for item in value.split(",") if item.strip())
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path, help="Path to generated run_matrix.csv.")
    parser.add_argument("--base-dir", required=True, type=Path, help="Base output directory for local runs.")
    parser.add_argument("--recipe", action="append", default=[], help="Recipe name to run. Repeat or comma-separate.")
    parser.add_argument("--window", action="append", default=[], help="Window name to run. Repeat or comma-separate.")
    parser.add_argument("--seed", action="append", default=[], help="Seed to run. Repeat or comma-separate.")
    parser.add_argument("--algorithm", action="append", default=[], help="Algorithm name to run. Repeat or comma-separate.")
    parser.add_argument("--force", action="store_true", help="Run even if result.json is already completed.")
    parser.add_argument("--keep-cuda", action="store_true", help="Do not clear CUDA_VISIBLE_DEVICES for child runs.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop at the first failed child run.")
    args = parser.parse_args()

    recipes = _split_csv(args.recipe)
    windows = _split_csv(args.window)
    seeds = _split_csv(args.seed)
    algorithms = _split_csv(args.algorithm)
    rows = _select_rows(
        _read_matrix(args.matrix),
        recipes=recipes,
        windows=windows,
        seeds=seeds,
        algorithms=algorithms,
    )
    if not rows:
        raise SystemExit("No rows matched the requested filters.")

    log_dir = args.base_dir / "driver_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    env = os.environ.copy()
    if not args.keep_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""

    print(f"Selected {len(rows)} queued W6 runs.", flush=True)
    for row in rows:
        job_id = row["job_id"]
        if _is_completed(args.base_dir, job_id) and not args.force:
            print(f"SKIP completed {job_id}", flush=True)
            continue

        config_path = row["config_path"]
        log_path = log_dir / f"{job_id}.log"
        print(f"RUN {job_id}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.run(
                [
                    sys.executable,
                    "run_experiment.py",
                    "--config",
                    config_path,
                    "--job_id",
                    job_id,
                    "--base-dir",
                    str(args.base_dir),
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        print(f"DONE {job_id} rc={proc.returncode}", flush=True)
        if proc.returncode != 0:
            failures.append(job_id)
            print(f"LOG {log_path}", flush=True)
            if args.stop_on_failure:
                break

    if failures:
        print("FAILED " + ",".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
