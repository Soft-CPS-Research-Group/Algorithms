"""Collect an offline-RL dataset by rolling out IQLAgent.

Output (per seed):
  datasets/offline_rl/iql/seed_<N>.parquet

Per-directory:
  datasets/offline_rl/iql/manifest.json
  datasets/offline_rl/iql/kpi_summary.csv
  datasets/offline_rl/iql/sample_first_1000.csv

Design notes
------------
* The behaviour policy is **IQLAgent** controlling Building 5 + OfflineRBC for
  the other 16 buildings — same multi-agent setup as collect_rbc_dataset.py.
* A single IQL checkpoint (best seed from run-001: seed_101) is used for all
  10 collection seeds (32..41). Collection seeds are disjoint from RBC seeds
  (22..31) and eval seeds (200..209).
* Both action columns (ev_charger, electrical_storage) are expected to have
  non-zero variance — IQL controls both dims unlike RBC which leaves
  electrical_storage constant.
* `reward_env` is the env-returned reward for Building 5. `reward` is filled
  with NaN here and computed later by apply_reward.py.
* Collector fails fast on schema mismatch or zero-variance action columns.

Usage
-----
  .venv/bin/python -m scripts.collect_iql_dataset
  .venv/bin/python -m scripts.collect_iql_dataset \\
      --checkpoint runs/offline_iql/run-001/seed_101 \\
      --seeds 32 33 34 35 36 37 38 39 40 41
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Quiet wrapper logs before importing it
from loguru import logger as _loguru_logger  # noqa: E402

_loguru_logger.remove()
_loguru_logger.add(sys.stderr, level="WARNING")

from algorithms.offline_rl import schema as S  # noqa: E402
from algorithms.offline_rl.iql_agent import IQLAgent  # noqa: E402
from scripts._benchmark_common import (  # noqa: E402
    DATASET_SCHEMA,
    clip_actions,
    extract_kpis,
    make_env,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "offline_rl" / "iql"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "runs" / "offline_iql" / "run-001" / "seed_101"
DEFAULT_SEEDS: List[int] = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41]

SAFETY_MAX_STEPS = 9000  # guard against runaway loops


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _build_iql(checkpoint_dir: Path) -> IQLAgent:
    """Load IQLAgent from a seed checkpoint directory."""
    return IQLAgent.from_seed_dir(checkpoint_dir)


# ---------------------------------------------------------------------------
# Per-seed rollout
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def collect_seed(seed: int, *, checkpoint_dir: Path) -> Dict[str, Any]:
    """Run one full IQL rollout, return rows + per-seed KPIs."""
    env = make_env(seed)
    obs_list, _ = env.reset()

    # Validate B5 obs/action names against the schema before doing work.
    S.validate_observation_names(env.observation_names[S.TARGET_BUILDING_INDEX])
    S.validate_action_names(env.action_names[S.TARGET_BUILDING_INDEX])

    agent = _build_iql(checkpoint_dir)
    agent.attach_environment(
        observation_names=env.observation_names,
        action_names=env.action_names,
        action_space=env.action_space,
        observation_space=env.observation_space,
        metadata={
            "seconds_per_time_step": getattr(env, "seconds_per_time_step", None),
            "building_names": [b.name for b in env.buildings],
            "interface": "flat",
            "topology_mode": "static",
            "topology_version": 0,
        },
    )

    rows: List[Dict[str, Any]] = []
    n_agents = len(env.action_names)
    b5 = S.TARGET_BUILDING_INDEX
    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        actions = agent.predict(obs_list, deterministic=True)
        if not isinstance(actions, list) or len(actions) != n_agents:
            raise RuntimeError(
                f"agent.predict returned shape {len(actions) if hasattr(actions,'__len__') else '?'}, "
                f"expected {n_agents}"
            )
        actions = clip_actions(actions, env.action_space)
        next_obs_list, rewards, terminated, truncated, _ = env.step(actions)

        # Build one row for Building 5 only.
        b5_obs = obs_list[b5]
        b5_action = actions[b5]
        b5_next_obs = next_obs_list[b5]
        b5_reward = float(rewards[b5])

        row: Dict[str, Any] = {
            "episode": 0,
            "timestep": step,
            "seed": int(seed),
            "policy_mode": "behaviour",
        }
        for name, val in zip(S.OBSERVATION_NAMES, b5_obs):
            row[S.obs_column(name)] = val
        for name, val in zip(S.ACTION_NAMES, b5_action):
            row[S.action_column(name)] = float(val)
        row["reward_env"] = b5_reward
        row["reward"] = float("nan")  # filled by apply_reward.py later
        for name, val in zip(S.OBSERVATION_NAMES, b5_next_obs):
            row[S.next_obs_column(name)] = val
        row["terminated"] = int(bool(terminated))
        row["truncated"] = int(bool(truncated))
        rows.append(row)

        obs_list = next_obs_list
        step += 1
        if step >= SAFETY_MAX_STEPS:
            raise RuntimeError(
                f"Rollout exceeded safety cap of {SAFETY_MAX_STEPS} steps "
                f"(env did not terminate). seed={seed}"
            )
        if step % 1000 == 0:
            print(f"  [seed={seed}] step {step}", flush=True)

    kpi_df = env.evaluate()
    district = extract_kpis(kpi_df, level="district")
    building = extract_kpis(kpi_df, level="building", name=S.TARGET_BUILDING_NAME)

    print(f"  [seed={seed}] episode done in {step} steps", flush=True)
    return {
        "seed": seed,
        "rows": rows,
        "n_steps": step,
        "district_kpis": district,
        "building_kpis": building,
    }


# ---------------------------------------------------------------------------
# Writers + validators (reuse logic from collect_rbc_dataset)
# ---------------------------------------------------------------------------


def _rows_to_table(rows: Sequence[Dict[str, Any]]) -> pa.Table:
    """Convert rows to an Arrow Table that strictly matches the schema."""
    df = pd.DataFrame(rows)
    expected_cols = S.all_columns()
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    if missing or extra:
        raise S.SchemaError(
            f"DataFrame columns mismatch.\n  missing: {missing}\n  extra: {extra}"
        )
    df = df[expected_cols]  # enforce order
    return pa.Table.from_pandas(df, schema=S.build_arrow_schema(), preserve_index=False)


def write_seed_parquet(out_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    table = _rows_to_table(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")


def _action_stats(table: pa.Table) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in S.ACTION_COLUMNS:
        arr = np.asarray(table.column(col).to_numpy(), dtype=np.float64)
        out[col] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


def _reward_env_stats(table: pa.Table) -> Dict[str, float]:
    arr = np.asarray(table.column("reward_env").to_numpy(), dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def assert_action_variance(
    stats: Dict[str, Dict[str, float]],
    *,
    tol: float = 1e-6,
    expected_zero_variance: Sequence[str] = (),
) -> None:
    """Fail fast on zero-variance action columns.

    IQL controls both action dims so no columns are expected to be constant.
    Pass column names in ``expected_zero_variance`` if a specific policy does
    legitimately leave a column constant (for compatibility with callers from
    collect_rbc_dataset).
    """
    expected_set = set(expected_zero_variance)
    bad: List[str] = []
    warned: List[str] = []
    for col, s in stats.items():
        if s["std"] < tol:
            if col in expected_set:
                warned.append(col)
            else:
                bad.append(col)
    if warned:
        print(
            "[collect] note: zero-variance action columns (expected for this "
            "behaviour policy):\n"
            + "\n".join(f"  {c} (std={stats[c]['std']:.2e})" for c in warned),
            flush=True,
        )
    if bad:
        raise S.SchemaError(
            "Zero-variance action columns detected (the M2 failure mode):\n"
            + "\n".join(f"  {c}: std={stats[c]['std']:.3e}" for c in bad)
            + "\n(if this is intentional for this policy, add the column to "
            "`expected_zero_variance`.)"
        )


# ---------------------------------------------------------------------------
# Manifest + KPI summary
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    out_dir: Path,
    *,
    checkpoint_dir: Path,
    seeds: Sequence[int],
    seed_files: Dict[int, Path],
    aggregated_action_stats: Dict[str, Dict[str, float]],
    aggregated_reward_stats: Dict[str, float],
    n_steps_total: int,
) -> Path:
    env_config_hash = _hash_file(REPO_ROOT / DATASET_SCHEMA.lstrip("./"))

    manifest = {
        "behaviour_policy": "iql",
        "behaviour_policy_class": "algorithms.offline_rl.iql_agent.IQLAgent",
        "behaviour_policy_checkpoint": str(checkpoint_dir),
        "schema_hash": S.schema_hash(),
        "env_config_hash": f"sha256:{env_config_hash}",
        "citylearn_dataset": DATASET_SCHEMA,
        "building": S.TARGET_BUILDING_NAME,
        "building_index": S.TARGET_BUILDING_INDEX,
        "seeds": [int(s) for s in seeds],
        "seed_files": {str(s): str(p.name) for s, p in seed_files.items()},
        "n_seeds": len(seeds),
        "n_steps_total": int(n_steps_total),
        "action_stats_aggregated": aggregated_action_stats,
        "reward_env_stats_aggregated": aggregated_reward_stats,
        "kpi_summary_path": "kpi_summary.csv",
        "sample_csv_path": "sample_first_1000.csv",
        "collected_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "code_git_sha": _git_sha(),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def write_kpi_summary(out_dir: Path, runs: Sequence[Dict[str, Any]]) -> Path:
    kpi_keys_district = sorted({k for r in runs for k in r["district_kpis"]})
    kpi_keys_building = sorted({k for r in runs for k in r["building_kpis"]})
    rows = []
    for r in runs:
        row: Dict[str, Any] = {"seed": r["seed"], "n_steps": r["n_steps"]}
        for k in kpi_keys_district:
            row[f"district.{k}"] = r["district_kpis"].get(k, math.nan)
        for k in kpi_keys_building:
            row[f"building.{k}"] = r["building_kpis"].get(k, math.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = out_dir / "kpi_summary.csv"
    df.to_csv(path, index=False)
    return path


def write_sample_csv(out_dir: Path, table: pa.Table, n: int = 1000) -> Path:
    df = table.slice(0, n).to_pandas()
    path = out_dir / "sample_first_1000.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _aggregate_action_stats(per_seed: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in S.ACTION_COLUMNS:
        means = [s[col]["mean"] for s in per_seed.values()]
        stds = [s[col]["std"] for s in per_seed.values()]
        mins = [s[col]["min"] for s in per_seed.values()]
        maxs = [s[col]["max"] for s in per_seed.values()]
        out[col] = {
            "mean_of_means": float(np.mean(means)),
            "mean_of_stds": float(np.mean(stds)),
            "min_overall": float(np.min(mins)),
            "max_overall": float(np.max(maxs)),
        }
    return out


def _aggregate_reward_stats(per_seed: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    means = [s["mean"] for s in per_seed.values()]
    stds = [s["std"] for s in per_seed.values()]
    return {
        "mean_of_means": float(np.mean(means)),
        "mean_of_stds": float(np.mean(stds)),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"IQL seed checkpoint dir (must contain policy.pt etc.). Default: {DEFAULT_CHECKPOINT_DIR}",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"Env seeds. Default: {DEFAULT_SEEDS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a single seed (32) and skip overwrite protection.",
    )
    args = parser.parse_args(argv)

    checkpoint_dir: Path = args.checkpoint
    seeds = args.seeds if args.seeds is not None else (DEFAULT_SEEDS[:1] if args.smoke else DEFAULT_SEEDS)
    out_dir: Path = args.output_dir

    if not checkpoint_dir.exists():
        print(f"[collect_iql] ERROR: checkpoint not found: {checkpoint_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[collect_iql] writing to: {out_dir}")
    print(f"[collect_iql] checkpoint: {checkpoint_dir}")
    print(f"[collect_iql] seeds: {seeds}")
    print(f"[collect_iql] schema_hash: {S.schema_hash()[:16]}…")

    runs: List[Dict[str, Any]] = []
    seed_files: Dict[int, Path] = {}
    per_seed_action_stats: Dict[int, Dict[str, Dict[str, float]]] = {}
    per_seed_reward_stats: Dict[int, Dict[str, float]] = {}
    n_steps_total = 0
    first_table: pa.Table | None = None

    for seed in seeds:
        out_path = out_dir / f"seed_{seed}.parquet"
        if out_path.exists() and not args.smoke:
            print(f"[collect_iql] skipping existing {out_path.name}")
            table = pq.read_table(out_path)
        else:
            run = collect_seed(seed, checkpoint_dir=checkpoint_dir)
            runs.append(run)
            write_seed_parquet(out_path, run["rows"])
            table = pq.read_table(out_path)

        seed_files[seed] = out_path
        astats = _action_stats(table)
        rstats = _reward_env_stats(table)
        assert_action_variance(astats)  # IQL: no expected_zero_variance
        per_seed_action_stats[seed] = astats
        per_seed_reward_stats[seed] = rstats
        n_steps_total += table.num_rows
        if first_table is None:
            first_table = table
        print(
            f"[collect_iql] seed={seed} rows={table.num_rows} "
            f"action_stds={[round(astats[c]['std'], 4) for c in S.ACTION_COLUMNS]} "
            f"reward_env=mean={rstats['mean']:.3f}"
        )

    # If no runs were collected (all skipped), still emit manifest using cached rows.
    if not runs:
        kpi_summary_path = out_dir / "kpi_summary.csv"
        if not kpi_summary_path.exists():
            print("[collect_iql] no fresh runs and kpi_summary.csv missing — re-running rollouts")
            runs = [collect_seed(s, checkpoint_dir=checkpoint_dir) for s in seeds]

    if runs:
        write_kpi_summary(out_dir, runs)

    if first_table is not None:
        write_sample_csv(out_dir, first_table)

    agg_action = _aggregate_action_stats(per_seed_action_stats)
    agg_reward = _aggregate_reward_stats(per_seed_reward_stats)
    manifest_path = write_manifest(
        out_dir,
        checkpoint_dir=checkpoint_dir,
        seeds=seeds,
        seed_files=seed_files,
        aggregated_action_stats=agg_action,
        aggregated_reward_stats=agg_reward,
        n_steps_total=n_steps_total,
    )

    print(f"\n[collect_iql] manifest: {manifest_path}")
    print(f"[collect_iql] total transitions: {n_steps_total}")
    print(f"[collect_iql] action stats (per-seed mean of std):")
    for col in S.ACTION_COLUMNS:
        print(f"  {col}: std≈{agg_action[col]['mean_of_stds']:.4f} "
              f"range=[{agg_action[col]['min_overall']:.3f}, {agg_action[col]['max_overall']:.3f}]")
    print(f"[collect_iql] reward_env: mean≈{agg_reward['mean_of_means']:.3f}, "
          f"std≈{agg_reward['mean_of_stds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
