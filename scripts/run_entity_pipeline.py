"""Unified entity-pipeline CLI: probe → collect → train → benchmark.

Chains collect_rbcsmart_dataset → train_iql_entity / train_cql_entity →
benchmark_entity_agents into one command. Any CityLearn dataset can be used
by overriding --schema.

Usage examples
--------------
# Full pipeline, 15s parquet dataset (default):
.venv/bin/python -m scripts.run_entity_pipeline --output runs/pipeline_001

# Hourly dataset, Building_5 group only, quick test:
.venv/bin/python -m scripts.run_entity_pipeline \\
    --schema datasets/citylearn_three_phase_electrical_service_demo/schema.json \\
    --buildings Building_5 \\
    --steps collect,train-iql,benchmark \\
    --train-seeds 22,23 --val-seeds 31 --gradient-steps 5000 \\
    --eval-seeds 200,201 \\
    --no-offline

# Just rerun benchmark on existing models:
.venv/bin/python -m scripts.run_entity_pipeline \\
    --output runs/pipeline_001 \\
    --steps benchmark \\
    --eval-seeds 200,201,202,203,204,205,206,207,208,209
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

DEFAULT_SCHEMA = str(
    REPO_ROOT / "datasets" / "citylearn_three_phase_electrical_service_demo_15s_parquet" / "schema.json"
)
DEFAULT_TRAIN_SEEDS = "22,23,24,25,26,27,28,29,30"
DEFAULT_VAL_SEEDS = "31"
DEFAULT_COLLECT_SEEDS = "22,23,24,25,26,27,28,29,30,31"
DEFAULT_EVAL_SEEDS = ",".join(str(s) for s in range(200, 210))
DEFAULT_GRADIENT_STEPS = 50_000
ALL_STEPS = "collect,train-iql,train-cql,benchmark"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help="Path to CityLearn schema.json. Determines time resolution and topology.",
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        type=Path,
        help="Base output directory. Sub-dirs data/, models-iql/, models-cql/, benchmark/ created here.",
    )
    p.add_argument(
        "--steps",
        default=ALL_STEPS,
        help="Comma-separated steps: collect, train-iql, train-cql, benchmark. Default: all.",
    )
    p.add_argument(
        "--buildings",
        default=None,
        help="Comma-separated building names, e.g. Building_5,Building_1. "
             "Restricts which agent groups are trained and benchmarked. Default: all.",
    )
    p.add_argument(
        "--algorithm", default="both", choices=["iql", "cql", "both"],
        help="Which algorithm(s) to train. Default: both.",
    )
    p.add_argument("--collect-seeds", default=DEFAULT_COLLECT_SEEDS)
    p.add_argument("--collect-episodes", type=int, default=1)
    p.add_argument("--train-seeds", default=DEFAULT_TRAIN_SEEDS)
    p.add_argument("--val-seeds", default=DEFAULT_VAL_SEEDS)
    p.add_argument("--eval-seeds", default=DEFAULT_EVAL_SEEDS)
    p.add_argument("--gradient-steps", type=int, default=DEFAULT_GRADIENT_STEPS)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--no-offline",
        dest="offline",
        action="store_false",
        default=True,
        help="Disable offline=True for CSV-based datasets (e.g. hourly dataset).",
    )
    return p


def _run(cmd: List[str]) -> None:
    """Run a subprocess command, exiting on failure."""
    print(f"\n[pipeline] $ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[pipeline] ERROR: command exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def _resolve_groups_arg(schema: str, buildings: Optional[str]) -> Optional[str]:
    """Derive obs_dim:action_dim group keys from building names via env probe."""
    if buildings is None:
        return None
    from algorithms.offline_rl.entity_schema import probe_agent_groups, buildings_to_group_keys
    groups = probe_agent_groups(schema)
    requested = [b.strip() for b in buildings.split(",")]
    keys = buildings_to_group_keys(requested, groups)
    if not keys:
        print(f"[pipeline] ERROR: no groups found for buildings={requested}", file=sys.stderr)
        print(f"[pipeline] Available groups: {[g.group_key for g in groups]}", file=sys.stderr)
        sys.exit(1)
    # Convert group keys (obs627_act1) to obs_dim:action_dim format for --groups arg
    result = []
    for g in groups:
        if g.group_key in keys:
            result.append(f"{g.obs_dim}:{g.action_dim}")
    return ",".join(result)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    steps = {s.strip() for s in args.steps.split(",")}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    data_dir = output / "data"
    iql_dir = output / "models-iql"
    cql_dir = output / "models-cql"
    bench_dir = output / "benchmark"

    offline_flag = [] if args.offline else ["--no-offline"]
    groups_str = _resolve_groups_arg(args.schema, args.buildings)
    groups_flag = ["--groups", groups_str] if groups_str else []

    print(f"[pipeline] schema:    {args.schema}")
    print(f"[pipeline] output:    {output}")
    print(f"[pipeline] steps:     {sorted(steps)}")
    print(f"[pipeline] buildings: {args.buildings or 'all'}")
    print(f"[pipeline] algorithm: {args.algorithm}")

    # --- Collect ---
    if "collect" in steps:
        collect_seeds = [s.strip() for s in args.collect_seeds.split(",")]
        _run([
            PYTHON, "-m", "scripts.collect_rbcsmart_dataset",
            "--schema", args.schema,
            "--seeds", *collect_seeds,
            "--episodes", str(args.collect_episodes),
            "--output-dir", str(data_dir),
            *offline_flag,
        ])

    # --- Train IQL ---
    if "train-iql" in steps and args.algorithm in ("iql", "both"):
        _run([
            PYTHON, "-m", "scripts.train_iql_entity",
            "--data-dir", str(data_dir),
            "--output", str(iql_dir),
            "--seeds", args.train_seeds,
            "--val-seeds", args.val_seeds,
            "--gradient-steps", str(args.gradient_steps),
            "--device", args.device,
            *groups_flag,
        ])

    # --- Train CQL ---
    if "train-cql" in steps and args.algorithm in ("cql", "both"):
        _run([
            PYTHON, "-m", "scripts.train_cql_entity",
            "--data-dir", str(data_dir),
            "--output", str(cql_dir),
            "--seeds", args.train_seeds,
            "--val-seeds", args.val_seeds,
            "--gradient-steps", str(args.gradient_steps),
            "--device", args.device,
            *groups_flag,
        ])

    # --- Benchmark ---
    if "benchmark" in steps:
        bench_dir.mkdir(parents=True, exist_ok=True)
        bench_cmd = [
            PYTHON, "-m", "scripts.benchmark_entity_agents",
            "--schema", args.schema,
            "--eval-seeds", args.eval_seeds,
            "--output", str(bench_dir / "results.json"),
            "--device", args.device,
            *offline_flag,
        ]
        if "train-iql" in steps or (iql_dir / "all_groups_summary.json").exists():
            bench_cmd += ["--iql-root", str(iql_dir)]
        if "train-cql" in steps or (cql_dir / "all_groups_summary.json").exists():
            bench_cmd += ["--cql-root", str(cql_dir)]
        _run(bench_cmd)

    print(f"\n[pipeline] All steps complete. Results in {output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
