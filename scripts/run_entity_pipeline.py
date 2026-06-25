"""Unified entity-pipeline CLI: probe → collect → train → benchmark → analyze.

Chains collect_rbcsmart_dataset → train_iql_entity / train_cql_entity →
benchmark_entity_agents → analyze_entity_dataset into one command.  Any
CityLearn dataset can be used by overriding --schema.

Resume & idempotency
--------------------
Each stage writes a ``runs/<output>/.{stage}.done`` sentinel on success.
A subsequent run will SKIP any stage whose sentinel exists.  Use
``--force STAGE[,STAGE...]`` (or ``--force all``) to bypass sentinels and
forward the underlying script's own force flag:

* ``--force collect``           → ``--no-skip-existing`` to collector
* ``--force train-iql``         → ``--force`` to IQL trainer
* ``--force train-cql``         → ``--force`` to CQL trainer
* ``--force feature-analysis``  → ``--force`` to analyzer
* ``--force benchmark``         → no sub-flag (no internal sentinel to clear)

Per-run progress is written to ``runs/<output>/status.json`` after every
stage transition (``running`` → ``done``/``skipped``/``failed``).

Usage examples
--------------
# Full pipeline, 15-min parquet dataset (default):
.venv/bin/python -m scripts.run_entity_pipeline --output runs/pipeline_001

# Hourly dataset, Building_5 group only, quick test:
.venv/bin/python -m scripts.run_entity_pipeline \\
    --schema datasets/citylearn_three_phase_electrical_service_demo/schema.json \\
    --buildings Building_5 \\
    --steps collect,train-iql,benchmark \\
    --train-seeds 22,23 --val-seeds 31 --gradient-steps 5000 \\
    --eval-seeds 200,201 \\
    --no-offline

# Resume run; force re-collection only:
.venv/bin/python -m scripts.run_entity_pipeline \\
    --output runs/pipeline_001 --force collect

# Just rerun benchmark on existing models:
.venv/bin/python -m scripts.run_entity_pipeline \\
    --output runs/pipeline_001 \\
    --steps benchmark --force benchmark \\
    --eval-seeds 200,201,202,203,204,205,206,207,208,209
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

DEFAULT_SCHEMA = str(
    REPO_ROOT / "datasets" / "citylearn_three_phase_electrical_service_demo_15min_parquet" / "schema.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "offline_iql_cql_initiative_15min"
DEFAULT_TRAIN_SEEDS = "22,23,24,25,26,27,28,29,30"
DEFAULT_VAL_SEEDS = "31"
DEFAULT_COLLECT_SEEDS = "22,23,24,25,26,27,28,29,30,31"
DEFAULT_EVAL_SEEDS = ",".join(str(s) for s in range(200, 210))
DEFAULT_GRADIENT_STEPS = 150_000
DEFAULT_CQL_ALPHA = 0.2
DEFAULT_HIDDEN_LAYERS = "256,256"
DEFAULT_CHECKPOINT_EVERY = 5000

# Canonical stage order (also drives the default --steps value).
STAGES: List[str] = ["collect", "train-iql", "train-cql", "benchmark", "feature-analysis"]
ALL_STEPS = ",".join(STAGES)


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
        default=DEFAULT_OUTPUT,
        type=Path,
        help=(
            "Base output directory. Sub-dirs data/, models-iql/, models-cql/, "
            "benchmark/ and data/feature_analysis/ are created here. "
            f"Default: {DEFAULT_OUTPUT}"
        ),
    )
    p.add_argument(
        "--steps",
        default=ALL_STEPS,
        help=(
            "Comma-separated steps: collect, train-iql, train-cql, benchmark, "
            "feature-analysis. Default: all."
        ),
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
    p.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help=(
            "Override per-episode length forwarded to the collector. "
            "When unset, the collector falls back to schema steps-per-day "
            "(24 hourly, 96 for 15-min, 5760 for 15-sec). Set to the full "
            "annual horizon (e.g. 35040 for 15-min) to collect 1 year per "
            "seed and avoid overfitting on 150k gradient steps."
        ),
    )
    p.add_argument("--train-seeds", default=DEFAULT_TRAIN_SEEDS)
    p.add_argument("--val-seeds", default=DEFAULT_VAL_SEEDS)
    p.add_argument("--eval-seeds", default=DEFAULT_EVAL_SEEDS)
    p.add_argument("--gradient-steps", type=int, default=DEFAULT_GRADIENT_STEPS)
    p.add_argument(
        "--hidden-layers",
        default=DEFAULT_HIDDEN_LAYERS,
        help=(
            "Comma-separated MLP hidden widths for actor/critic, forwarded to both "
            f"train_iql_entity and train_cql_entity. Default: {DEFAULT_HIDDEN_LAYERS}."
        ),
    )
    p.add_argument(
        "--cql-alpha",
        type=float,
        default=DEFAULT_CQL_ALPHA,
        help=(
            "CQL conservatism weight (alpha). Only forwarded to train_cql_entity. "
            f"Default: {DEFAULT_CQL_ALPHA}."
        ),
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help=(
            "Persist a within-seed checkpoint every N gradient steps. "
            "Forwarded to train_iql_entity and train_cql_entity "
            "(IQL/CQL TrainingConfig.checkpoint_every_n_steps). "
            f"Default: {DEFAULT_CHECKPOINT_EVERY}."
        ),
    )
    p.add_argument(
        "--force",
        default=None,
        help=(
            "Comma-separated stages to force-rerun, bypassing .{stage}.done "
            "sentinels under the run output dir. Use 'all' to force every "
            "stage. Sub-flags forwarded: collect→--no-skip-existing, "
            "train-iql/train-cql/feature-analysis→--force."
        ),
    )
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
    """Derive obs_dim:action_dim group keys by probing the schema.

    Always probes the schema (independent of --buildings) so the orchestrator
    passes the actually-present groups to trainers. Otherwise trainers fall
    back to the stale hardcoded AGENT_GROUPS constant in
    ``algorithms/offline_rl/entity_schema.py`` and crash with
    'No rows for group (obs_dim=627, action_dim=1)' when the schema's
    encoder produces different obs_dims (e.g. 163/225/257/287).

    Returns a comma-separated 'obs_dim:action_dim' string; when ``buildings``
    is set, the result is filtered to groups containing those buildings.
    """
    from algorithms.offline_rl.entity_schema import probe_agent_groups, buildings_to_group_keys
    groups = probe_agent_groups(schema)
    if not groups:
        return None
    if buildings is None:
        # Default: pass ALL groups present in the schema.
        return ",".join(f"{g.obs_dim}:{g.action_dim}" for g in groups)
    # Filter to groups containing the requested buildings.
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


# ---------------------------------------------------------------------------
# Phase 5 — sentinels, --force parsing, status.json
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Wall-clock UTC ISO timestamp (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sentinel_path(output: Path, stage: str) -> Path:
    return output / f".{stage}.done"


def _sentinel_exists(output: Path, stage: str) -> bool:
    return _sentinel_path(output, stage).exists()


def _write_sentinel(output: Path, stage: str, *, duration_seconds: float) -> None:
    payload = {
        "stage": stage,
        "completed_at": _now_iso(),
        "duration_seconds": float(duration_seconds),
    }
    _sentinel_path(output, stage).write_text(json.dumps(payload, indent=2))


def _parse_force(force_arg: Optional[str]) -> Set[str]:
    """Parse --force value into a set of stage names.

    ``--force all`` expands to every stage in ``STAGES``.
    Empty / None → empty set.
    """
    if force_arg is None or not force_arg.strip():
        return set()
    forced = {s.strip() for s in force_arg.split(",") if s.strip()}
    if "all" in forced:
        return set(STAGES)
    return forced


def _read_status(output: Path) -> dict:
    path = output / "status.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"stages": {}}


def _write_status(output: Path, status: dict) -> None:
    """Atomic status.json write (``.tmp`` → rename)."""
    path = output / "status.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(path)


def _update_status(output: Path, stage: str, **fields) -> None:
    """Merge ``fields`` into ``status.json[stages][stage]`` and persist."""
    status = _read_status(output)
    status.setdefault("stages", {})
    entry = status["stages"].get(stage, {})
    entry.update(fields)
    status["stages"][stage] = entry
    _write_status(output, status)


def _run_stage(
    output: Path,
    stage: str,
    *,
    forced: Set[str],
    cmd: List[str],
) -> str:
    """Run one pipeline stage with sentinel-skip + status.json tracking.

    Returns the resulting status string: ``done`` or ``skipped``.
    On subprocess failure, ``_run`` calls ``sys.exit`` directly.
    """
    is_forced = stage in forced
    sentinel = _sentinel_path(output, stage)

    if sentinel.exists() and not is_forced:
        print(
            f"[pipeline] {stage}: SKIP (.{stage}.done exists; "
            f"use --force {stage} to rerun)",
            flush=True,
        )
        _update_status(
            output, stage,
            status="skipped",
            skipped_at=_now_iso(),
        )
        return "skipped"

    if is_forced and sentinel.exists():
        print(f"[pipeline] {stage}: --force present; removing sentinel", flush=True)
        sentinel.unlink()

    _update_status(output, stage, status="running", started_at=_now_iso())
    t0 = time.time()
    try:
        _run(cmd)
    except SystemExit:
        # Subprocess failed (or was killed): record terminal "failed" status
        # so the viewer can surface it instead of stale "running", then
        # propagate the exit so callers observe the failure.  No sentinel
        # is written — on re-invocation the orchestrator will re-run this
        # stage, and the trainer's own checkpoint_latest.pt resume kicks in.
        duration = time.time() - t0
        _update_status(
            output, stage,
            status="failed",
            duration_seconds=duration,
            failed_at=_now_iso(),
        )
        raise
    duration = time.time() - t0
    _write_sentinel(output, stage, duration_seconds=duration)
    _update_status(
        output, stage,
        status="done",
        duration_seconds=duration,
        completed_at=_now_iso(),
    )
    return "done"


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    steps = {s.strip() for s in args.steps.split(",")}
    forced = _parse_force(args.force)
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
    print(f"[pipeline] forced:    {sorted(forced) if forced else 'none'}")
    print(f"[pipeline] buildings: {args.buildings or 'all'}")
    print(f"[pipeline] algorithm: {args.algorithm}")

    # --- Collect ---
    if "collect" in steps:
        collect_seeds = [s.strip() for s in args.collect_seeds.split(",")]
        cmd = [
            PYTHON, "-m", "scripts.collect_rbcsmart_dataset",
            "--schema", args.schema,
            "--seeds", *collect_seeds,
            "--episodes", str(args.collect_episodes),
            "--output-dir", str(data_dir),
            *offline_flag,
        ]
        if args.episode_steps is not None:
            cmd.extend(["--episode-steps", str(args.episode_steps)])
        if "collect" in forced:
            cmd.append("--no-skip-existing")
        _run_stage(output, "collect", forced=forced, cmd=cmd)

    # --- Train IQL ---
    if "train-iql" in steps and args.algorithm in ("iql", "both"):
        cmd = [
            PYTHON, "-m", "scripts.train_iql_entity",
            "--data-dir", str(data_dir),
            "--output", str(iql_dir),
            "--seeds", args.train_seeds,
            "--val-seeds", args.val_seeds,
            "--gradient-steps", str(args.gradient_steps),
            "--hidden-layers", args.hidden_layers,
            "--checkpoint-every", str(args.checkpoint_every),
            "--device", args.device,
            *groups_flag,
        ]
        if "train-iql" in forced:
            cmd.append("--force")
        _run_stage(output, "train-iql", forced=forced, cmd=cmd)

    # --- Train CQL ---
    if "train-cql" in steps and args.algorithm in ("cql", "both"):
        cmd = [
            PYTHON, "-m", "scripts.train_cql_entity",
            "--data-dir", str(data_dir),
            "--output", str(cql_dir),
            "--seeds", args.train_seeds,
            "--val-seeds", args.val_seeds,
            "--gradient-steps", str(args.gradient_steps),
            "--hidden-layers", args.hidden_layers,
            "--cql-alpha", str(args.cql_alpha),
            "--checkpoint-every", str(args.checkpoint_every),
            "--device", args.device,
            *groups_flag,
        ]
        if "train-cql" in forced:
            cmd.append("--force")
        _run_stage(output, "train-cql", forced=forced, cmd=cmd)

    # --- Benchmark ---
    if "benchmark" in steps:
        bench_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON, "-m", "scripts.benchmark_entity_agents",
            "--schema", args.schema,
            "--eval-seeds", args.eval_seeds,
            "--output", str(bench_dir / "results.json"),
            "--device", args.device,
            *offline_flag,
        ]
        if "train-iql" in steps or (iql_dir / "all_groups_summary.json").exists():
            cmd += ["--iql-root", str(iql_dir)]
        if "train-cql" in steps or (cql_dir / "all_groups_summary.json").exists():
            cmd += ["--cql-root", str(cql_dir)]
        # benchmark has no internal sentinel; --force benchmark just bypasses
        # the orchestrator-level sentinel via _run_stage().
        _run_stage(output, "benchmark", forced=forced, cmd=cmd)

    # --- Feature analysis ---
    if "feature-analysis" in steps:
        cmd = [
            PYTHON, "-m", "scripts.analyze_entity_dataset",
            "--data-dir", str(data_dir),
            "--output-dir", str(output),
        ]
        if "feature-analysis" in forced:
            cmd.append("--force")
        _run_stage(output, "feature-analysis", forced=forced, cmd=cmd)

    print(f"\n[pipeline] All steps complete. Results in {output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
