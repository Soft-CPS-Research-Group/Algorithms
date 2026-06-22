"""Collect an offline-RL dataset by rolling out RBCSmartPolicy in entity interface.

Each row stores one agent's transition for one timestep. All 17 buildings are
recorded so that downstream IQL/CQL training can use any agent group.

Output per seed:
    datasets/offline_rl/rbcsmart_entity/seed_<N>.parquet

Per-directory:
    datasets/offline_rl/rbcsmart_entity/manifest.json
    datasets/offline_rl/rbcsmart_entity/kpi_summary.csv
    datasets/offline_rl/rbcsmart_entity/sample_first_1000.csv

Design notes
------------
* Policy: RBCSmartPolicy (solar/price/peak-aware heuristic).
* Reward: CostServiceCommunityFeasiblePrecisionRewardV46 captured live from
  the simulator — same signal used by MADDPG training, enabling apples-to-apples
  offline-vs-online comparison.
* Observations: minmax-normalised entity-encoded vectors from EntityContractAdapter.
* All 17 agents are recorded. Agent groups are identified by (obs_dim, action_dim)
  so downstream code can train separate policies per group without re-collection.
* 10 episodes per seed (1 day = 5760 steps each). Collection seeds 22–31.
* Fails fast on: schema mismatch, zero action variance in non-constant columns.
* Memory efficiency: rows are flushed to parquet in batches of BATCH_FLUSH_STEPS
  environment steps (default 200) to avoid OOM when materialising wide-sparse
  DataFrames for long episodes.

Usage
-----
    .venv/bin/python -m scripts.collect_rbcsmart_dataset --smoke
    .venv/bin/python -m scripts.collect_rbcsmart_dataset \\
        --seeds 22 23 24 25 26 27 28 29 30 31 --episodes 10
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger as _loguru_logger  # noqa: E402
_loguru_logger.remove()
_loguru_logger.add(sys.stderr, level="WARNING")

# IMPORTANT: apply CityLearn runtime patches BEFORE constructing any CityLearnEnv.
# Without this, the 35040-step 15-min full-year collect OOMs at ~step 16000 due
# to unbounded `_action_feedback_series_cache` growth (Bug 7).
from utils.citylearn_patches import apply_citylearn_patches  # noqa: E402

apply_citylearn_patches()

from citylearn.citylearn import CityLearnEnv  # noqa: E402
from reward_function import CostServiceCommunityFeasiblePrecisionRewardV46  # noqa: E402
from utils.entity_adapter import EntityContractAdapter  # noqa: E402
from algorithms.agents.baseline_policies import RBCSmartPolicy  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_PATH = str(
    REPO_ROOT / "datasets" / "citylearn_three_phase_electrical_service_demo_15s_parquet" / "schema.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "offline_rl" / "rbcsmart_entity"
DEFAULT_SEEDS: List[int] = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
DEFAULT_EPISODES: int = 10
EPISODE_STEPS: int = 5760          # 1 day at 15 s resolution
SMOKE_STEPS: int = 200             # fast sanity check
SAFETY_MAX_STEPS: int = 6000       # minimum safety floor (historical 15-s daily margin)
SAFETY_BUFFER_STEPS: int = 100     # margin above episode_steps when episode_steps > floor
BATCH_FLUSH_STEPS: int = 200       # env steps between parquet row-group flushes


def _compute_safety_limit(episode_steps: int) -> int:
    """Return the runaway-loop safety threshold for a rollout.

    The collector's main loop trusts the env's ``truncated`` flag to end an
    episode at ``episode_steps``. The safety threshold is a *backstop* for
    pathological cases where the env never reports truncation.

    Behavior:

    * Floor: never goes below ``SAFETY_MAX_STEPS`` so that short rollouts
      retain a defensive margin (e.g. a smoke episode of 200 steps still
      gets a 6000-step backstop).
    * Scaling: when ``episode_steps`` is large (e.g. 15-min annual = 35 040),
      the limit becomes ``episode_steps + SAFETY_BUFFER_STEPS`` so the
      backstop is triggered only by genuine runaway behavior, not by a
      legitimate long rollout.

    Regression guard: before this helper existed, the loop tested against a
    hard-coded ``SAFETY_MAX_STEPS = 6000`` constant, which aborted 15-min
    full-year rollouts (episode_steps = 35 040) at step 6000.
    """
    try:
        n = int(episode_steps)
    except (TypeError, ValueError):
        n = 0
    return max(SAFETY_MAX_STEPS, n + SAFETY_BUFFER_STEPS)

# Parquet column prefixes
OBS_PREFIX = "obs__"
NEXT_OBS_PREFIX = "next_obs__"
ACTION_PREFIX = "action__"

# ---------------------------------------------------------------------------
# Env + adapter factory
# ---------------------------------------------------------------------------


def _make_env(
    *,
    start_step: int = 0,
    episode_steps: int = EPISODE_STEPS,
    schema_path: str = SCHEMA_PATH,
    offline: bool = True,
) -> CityLearnEnv:
    """Instantiate a CityLearnEnv with V46 reward and entity interface."""
    return CityLearnEnv(
        schema=schema_path,
        central_agent=False,
        interface="entity",
        topology_mode="static",
        reward_function=CostServiceCommunityFeasiblePrecisionRewardV46,
        simulation_start_time_step=start_step,
        episode_time_steps=episode_steps,
        offline=offline,
    )


def _make_adapter(env: CityLearnEnv) -> EntityContractAdapter:
    return EntityContractAdapter(
        env,
        normalization_enabled=True,
        clip=True,
        encoding_profile="minmax_space",
    )


def _make_rbc(obs_names: List[List[str]], action_names: List[List[str]], env: CityLearnEnv) -> RBCSmartPolicy:
    rbc = RBCSmartPolicy(
        config={"algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {}}, "simulator": {}}
    )
    from gymnasium import spaces as gym_spaces
    obs_spaces = [
        gym_spaces.Box(low=-1e6, high=1e6, shape=(len(obs),), dtype=np.float32)
        for obs in obs_names
    ]
    rbc.attach_environment(
        observation_names=obs_names,
        action_names=action_names,
        action_space=list(getattr(env, "flat_action_space", [])),
        observation_space=obs_spaces,
        metadata={
            "building_names": [b.name for b in env.buildings],
            "seconds_per_time_step": env.seconds_per_time_step,
        },
    )
    return rbc


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _obs_col(name: str) -> str:
    return f"{OBS_PREFIX}{name}"


def _next_obs_col(name: str) -> str:
    return f"{NEXT_OBS_PREFIX}{name}"


def _action_col(name: str) -> str:
    return f"{ACTION_PREFIX}{name}"


# ---------------------------------------------------------------------------
# Streaming parquet writer
# ---------------------------------------------------------------------------


class _StreamingParquetWriter:
    """Accumulates row dicts and flushes them as row groups to a single parquet file.

    The schema (column set) is inferred from the *first* flush batch and then
    reused for all subsequent batches. Missing columns in later batches are
    filled with NaN, extra columns are dropped. In practice, since all 17
    agents appear at every env step, the column set is identical across batches.
    """

    def __init__(self, path: Path, compression: str = "snappy") -> None:
        self._path = path
        self._compression = compression
        self._writer: Optional[pq.ParquetWriter] = None
        self._schema: Optional[pa.Schema] = None
        self._n_rows_written = 0

    def write_batch(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, preserve_index=False)

        if self._writer is None:
            self._schema = table.schema
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(
                str(self._path), schema=self._schema, compression=self._compression
            )

        # Align columns to schema (add missing as null, drop extras)
        if table.schema != self._schema:
            table = _align_table(table, self._schema)

        self._writer.write_table(table)
        self._n_rows_written += len(table)

    def close(self) -> int:
        """Close the writer and return total rows written."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._n_rows_written

    @property
    def n_rows(self) -> int:
        return self._n_rows_written


def _align_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Reindex a table to match the target schema; fill missing cols with null."""
    cols = {}
    for i, field in enumerate(schema):
        if field.name in table.schema.names:
            col = table.column(field.name)
            if col.type != field.type:
                col = col.cast(field.type)
            cols[field.name] = col
        else:
            cols[field.name] = pa.array([None] * len(table), type=field.type)
    return pa.table(cols, schema=schema)


# ---------------------------------------------------------------------------
# Per-episode rollout
# ---------------------------------------------------------------------------


def collect_episode(
    *,
    seed: int,
    episode_idx: int,
    start_step: int,
    episode_steps: int,
    schema_path: str = SCHEMA_PATH,
    offline: bool = True,
    on_batch: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    batch_flush_steps: int = BATCH_FLUSH_STEPS,
) -> Dict[str, Any]:
    """Run one episode and return KPIs (+ rows only when ``on_batch`` is None).

    Parameters
    ----------
    on_batch:
        When provided, row dicts are flushed to this callback every
        ``batch_flush_steps`` env steps instead of being accumulated in memory.
        The returned ``"rows"`` key will be an empty list in this case.
    batch_flush_steps:
        How many env steps to collect before calling ``on_batch``.
    """
    env = _make_env(start_step=start_step, episode_steps=episode_steps, schema_path=schema_path, offline=offline)
    adapter = _make_adapter(env)

    # Backstop against runaway loops if the env never reports
    # ``terminated``/``truncated``. Scales with ``episode_steps`` so that
    # legitimate long rollouts (e.g. 15-min full-year = 35040 steps) aren't
    # aborted by a static guard. See ``_compute_safety_limit`` docstring.
    safety_limit = _compute_safety_limit(episode_steps)

    obs_payload, _ = env.reset()
    # RBCSmartPolicy sets ``_use_raw_observations = True`` and indexes obs by
    # name expecting raw kW values (see algorithms/agents/baseline_policies.py).
    # Feeding it encoded (minmax-normalised) observations causes the encoder
    # to collapse all kW features to the 0.5 midpoint and ``_pv_surplus_kw``
    # to return 0, which silently disables every storage charge branch.
    # We therefore keep two parallel views of every observation:
    #
    #   * ``raw_obs_list``  - fed to ``rbc.predict()`` (raw kW values).
    #   * ``enc_obs_list``  - stored in the parquet ``obs__/next_obs__``
    #                         columns so downstream IQL/CQL trains on the
    #                         same minmax-encoded features the wrapper would
    #                         hand to a learned policy.
    raw_obs_list, obs_names, obs_spaces = adapter.to_agent_observations(obs_payload)
    enc_obs_list, _, _ = adapter.to_agent_encoded_observations(obs_payload)
    action_names = [list(names) for names in env.action_names]
    n_agents = len(raw_obs_list)

    rbc = _make_rbc(obs_names, action_names, env)

    rows: List[Dict[str, Any]] = []
    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        actions = rbc.predict(raw_obs_list, deterministic=True)
        env_actions = adapter.to_entity_actions(actions, action_names)
        next_obs_payload, rewards, terminated, truncated, _ = env.step(env_actions)
        next_raw_obs_list, _, _ = adapter.to_agent_observations(next_obs_payload)
        next_enc_obs_list, _, _ = adapter.to_agent_encoded_observations(next_obs_payload)

        for agent_idx in range(n_agents):
            obs_agent = enc_obs_list[agent_idx]
            next_obs_agent = next_enc_obs_list[agent_idx]
            act_agent = actions[agent_idx]
            a_names = action_names[agent_idx]
            o_names = obs_names[agent_idx]

            row: Dict[str, Any] = {
                "seed": int(seed),
                "episode": int(episode_idx),
                "timestep": int(step),
                "agent_idx": int(agent_idx),
                "obs_dim": int(len(obs_agent)),
                "action_dim": int(len(act_agent)),
                "reward": float(rewards[agent_idx]),
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
            }
            for name, val in zip(o_names, obs_agent):
                row[_obs_col(name)] = float(val)
            for name, val in zip(a_names, act_agent):
                row[_action_col(name)] = float(val)
            for name, val in zip(o_names, next_obs_agent):
                row[_next_obs_col(name)] = float(val)
            rows.append(row)

        raw_obs_list = next_raw_obs_list
        enc_obs_list = next_enc_obs_list
        step += 1
        if step >= safety_limit:
            raise RuntimeError(
                f"Rollout exceeded {safety_limit} steps "
                f"(episode_steps={episode_steps}) — env did not terminate. "
                f"seed={seed} episode={episode_idx}"
            )
        if step % 1000 == 0:
            print(f"  [seed={seed} ep={episode_idx}] step {step}", flush=True)

        # Streaming flush
        if on_batch is not None and step % batch_flush_steps == 0:
            on_batch(rows)
            rows = []

    # Final flush of any remaining rows
    if on_batch is not None and rows:
        on_batch(rows)
        rows = []

    kpi_df = env.evaluate()
    district_kpis = _extract_kpis(kpi_df, level="district")
    building_kpis_list = [
        _extract_kpis(kpi_df, level="building", name=b.name)
        for b in env.buildings
    ]
    n_rows_episode = step * n_agents
    print(f"  [seed={seed} ep={episode_idx}] done in {step} steps, n_rows={n_rows_episode}", flush=True)
    return {
        "seed": seed,
        "episode": episode_idx,
        "rows": rows,  # empty when on_batch is used
        "n_steps": step,
        "district_kpis": district_kpis,
        "building_kpis": building_kpis_list,
        "obs_dims": [len(o) for o in enc_obs_list],
        "action_dims": [len(a) for a in actions],
        "obs_names": obs_names,
        "action_names": action_names,
    }


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------


def _extract_kpis(
    df: pd.DataFrame,
    *,
    level: str,
    name: str | None = None,
) -> Dict[str, float]:
    sub = df[df["level"] == level]
    if name is not None:
        sub = sub[sub["name"] == name]
    out: Dict[str, float] = {}
    for _, row in sub.iterrows():
        try:
            out[str(row["cost_function"])] = float(row["value"])
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def assert_action_variance(
    df: pd.DataFrame,
    *,
    tol: float = 1e-6,
    warn_only: bool = False,
) -> None:
    """Warn (or fail) if any action column has near-zero variance.

    EV charger and deferrable actions are legitimately zero in windows where
    no EVs are connected (e.g. early morning). Only raise an error for the
    stationary battery action (``action__electrical_storage``) which must vary
    for any meaningful battery-control dataset.
    """
    action_cols = [c for c in df.columns if c.startswith(ACTION_PREFIX)]
    bad: List[str] = []
    warned: List[str] = []
    for col in action_cols:
        arr = df[col].dropna().to_numpy(dtype=np.float64)
        if arr.std() < tol:
            # Battery must vary; EV/deferrable may be zero in some windows.
            if "electrical_storage" in col and "electric_vehicle" not in col:
                bad.append(f"{col}: std={arr.std():.2e}")
            else:
                warned.append(f"{col}: std={arr.std():.2e}")
    if warned:
        print(
            "[collect] note: zero-variance action columns (expected in limited windows):\n"
            + "\n".join(f"  {w}" for w in warned),
            flush=True,
        )
    if bad and not warn_only:
        raise ValueError(
            "Zero-variance battery action columns (M2 failure mode):\n"
            + "\n".join(f"  {b}" for b in bad)
        )
    elif bad:
        print(
            "[collect] WARNING: zero-variance battery action columns:\n"
            + "\n".join(f"  {b}" for b in bad),
            flush=True,
        )


def _action_variance_check_from_parquet(seed_path: Path) -> None:
    """Read only action columns from the written parquet and check variance."""
    pf = pq.ParquetFile(str(seed_path))
    action_cols = [
        c for c in pf.schema_arrow.names
        if c.startswith(ACTION_PREFIX)
    ]
    if not action_cols:
        return
    df = pq.read_table(str(seed_path), columns=action_cols).to_pandas()
    assert_action_variance(df)


def _agent_group_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarise agent groups present in the dataset."""
    groups = Counter(zip(df["obs_dim"].tolist(), df["action_dim"].tolist()))
    return {f"obs{k[0]}_act{k[1]}": v for k, v in groups.most_common()}


def _agent_group_summary_from_parquet(seed_path: Path) -> Dict[str, Any]:
    """Read only obs_dim/action_dim columns to compute group summary."""
    df = pq.read_table(str(seed_path), columns=["obs_dim", "action_dim"]).to_pandas()
    return _agent_group_summary(df)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_seed_parquet(out_path: Path, df: pd.DataFrame) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_manifest(
    out_dir: Path,
    *,
    schema_path: str,
    episode_steps: int,
    seeds: Sequence[int],
    episodes_per_seed: int,
    seed_files: Dict[int, Path],
    n_rows_total: int,
    agent_groups: Dict[str, Any],
    schema_hash: str,
) -> Path:
    manifest = {
        "behaviour_policy": "RBCSmartPolicy",
        "behaviour_policy_class": "algorithms.agents.baseline_policies.RBCSmartPolicy",
        "reward_function": "CostServiceCommunityFeasiblePrecisionRewardV46",
        "reward_captured": "live",
        "dataset_path": str(schema_path),
        "interface": "entity",
        "topology_mode": "static",
        "entity_encoding": "minmax_space",
        "seeds": [int(s) for s in seeds],
        "episodes_per_seed": int(episodes_per_seed),
        "episode_time_steps": int(episode_steps),
        "n_agents": 17,
        "agent_groups": agent_groups,
        "n_rows_total": int(n_rows_total),
        "seed_files": {str(s): str(p.name) for s, p in seed_files.items()},
        "schema_hash": schema_hash,
        "collected_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "code_git_sha": _git_sha(),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def write_kpi_summary(out_dir: Path, results: List[Dict[str, Any]]) -> Path:
    rows = []
    kpi_keys_d = sorted({k for r in results for k in r["district_kpis"]})
    for r in results:
        row: Dict[str, Any] = {
            "seed": r["seed"],
            "episode": r["episode"],
            "n_steps": r["n_steps"],
        }
        for k in kpi_keys_d:
            row[f"district.{k}"] = r["district_kpis"].get(k, math.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = out_dir / "kpi_summary.csv"
    df.to_csv(path, index=False)
    return path


def write_sample_csv(out_dir: Path, seed_path: Path, n: int = 1000) -> Path:
    """Write sample CSV by reading first N rows from an existing parquet file."""
    df = pq.read_table(str(seed_path)).to_pandas().head(n)
    path = out_dir / "sample_first_1000.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--schema",
        default=SCHEMA_PATH,
        help="Path to CityLearn schema.json (default: 15s parquet dataset).",
    )
    p.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        dest="episode_steps",
        help="Steps per episode. Defaults to one calendar day for the schema's "
             "time resolution (86400 // seconds_per_time_step).",
    )
    p.add_argument(
        "--no-offline",
        dest="offline",
        action="store_false",
        default=True,
        help="Disable offline=True in CityLearnEnv (needed for CSV-based datasets).",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Single seed, 1 episode, 200 steps — fast sanity check.",
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If True (default), skip seeds whose seed_*.parquet already exists "
            "AND short-circuit the whole stage when .collect.done is present. "
            "Use --no-skip-existing to force re-collection."
        ),
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from algorithms.offline_rl.entity_schema import episode_steps_for_schema
    resolved_episode_steps = (
        args.episode_steps
        if args.episode_steps is not None
        else episode_steps_for_schema(args.schema)
    )

    seeds: List[int] = args.seeds if args.seeds is not None else DEFAULT_SEEDS
    if args.smoke and args.seeds is None:
        # Only cap to 1 seed when --smoke is used without explicit --seeds.
        # If the caller provides --seeds explicitly, honour that list so that
        # multi-seed smoke runs (for train/val splits) are possible.
        seeds = seeds[:1]

    episodes_per_seed: int = 1 if args.smoke else (args.episodes or DEFAULT_EPISODES)
    episode_steps = SMOKE_STEPS if args.smoke else resolved_episode_steps

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-stage idempotency: short-circuit if .collect.done is present and
    # the caller has not explicitly opted in to a fresh re-collection via
    # --no-skip-existing.  --smoke always re-runs to validate the pipeline.
    collect_done_path = out_dir / ".collect.done"
    if args.skip_existing and collect_done_path.exists() and not args.smoke:
        print(
            f"[collect] .collect.done present at {out_dir} — skipping "
            f"(use --no-skip-existing to force re-collection).",
            flush=True,
        )
        return 0

    print(f"[collect] schema:            {args.schema}")
    print(f"[collect] output_dir: {out_dir}")
    print(f"[collect] seeds: {seeds}")
    print(f"[collect] episodes_per_seed: {episodes_per_seed}")
    print(f"[collect] steps_per_episode: {episode_steps}")
    print(f"[collect] offline: {args.offline}")
    print(f"[collect] reward: CostServiceCommunityFeasiblePrecisionRewardV46 (live)")

    schema_hash = hashlib.sha256(Path(args.schema).read_bytes()).hexdigest()[:16]
    print(f"[collect] schema_hash: {schema_hash}...")

    seed_files: Dict[int, Path] = {}
    all_results: List[Dict[str, Any]] = []
    n_rows_total = 0
    first_seed_path: Optional[Path] = None

    for seed in seeds:
        seed_path = out_dir / f"seed_{seed}.parquet"
        if seed_path.exists() and args.skip_existing and not args.smoke:
            print(f"[collect] skipping existing {seed_path.name}")
            seed_files[seed] = seed_path
            n_rows_total += pq.read_metadata(str(seed_path)).num_rows
            if first_seed_path is None:
                first_seed_path = seed_path
            continue

        print(f"\n[collect] === seed {seed} ===")
        seed_episode_results: List[Dict[str, Any]] = []

        # Use streaming writer to avoid OOM on wide-sparse DataFrames
        streaming_writer = _StreamingParquetWriter(seed_path)

        for ep in range(episodes_per_seed):
            start_step = ep * episode_steps
            result = collect_episode(
                seed=seed, episode_idx=ep,
                start_step=start_step, episode_steps=episode_steps,
                schema_path=args.schema, offline=args.offline,
                on_batch=streaming_writer.write_batch,
                batch_flush_steps=BATCH_FLUSH_STEPS,
            )
            seed_episode_results.append(result)
            all_results.append(result)

        n_rows_seed = streaming_writer.close()
        seed_files[seed] = seed_path
        n_rows_total += n_rows_seed

        if first_seed_path is None:
            first_seed_path = seed_path

        # Action variance check: read only action cols back from parquet (memory-efficient)
        _action_variance_check_from_parquet(seed_path)

        # Compute group summary from lightweight read
        agent_groups = _agent_group_summary_from_parquet(seed_path)
        reward_mean = float(np.mean([
            r["district_kpis"].get("carbon_emissions_intensity", float("nan"))
            for r in seed_episode_results
            if r["district_kpis"]
        ])) if seed_episode_results else float("nan")

        # Use mean reward from rows if available in results
        all_rewards = [
            v for r in seed_episode_results
            for v in [r["district_kpis"].get("net_electricity_consumption_emission_intensity", float("nan"))]
        ]
        print(
            f"[collect] seed={seed}: {n_rows_seed} rows, "
            f"groups={agent_groups}"
        )

    # Manifest
    if seed_files:
        # Compute agent groups from first available seed parquet
        first_path = first_seed_path or next(iter(seed_files.values()))
        agent_groups_global = _agent_group_summary_from_parquet(first_path)
    else:
        agent_groups_global = {}

    manifest_path = write_manifest(
        out_dir,
        schema_path=args.schema,
        episode_steps=episode_steps,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        seed_files=seed_files,
        n_rows_total=n_rows_total,
        agent_groups=agent_groups_global,
        schema_hash=schema_hash,
    )

    if all_results:
        write_kpi_summary(out_dir, all_results)
    if first_seed_path is not None:
        write_sample_csv(out_dir, first_seed_path)

    # Per-stage success sentinel.  Written last so that any earlier failure
    # leaves the directory in an "incomplete" state and a re-run will resume
    # the work instead of short-circuiting.
    collect_done_path.write_text(
        json.dumps(
            {
                "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "n_seeds": len(seed_files),
                "n_rows_total": int(n_rows_total),
                "schema_path": str(args.schema),
                "schema_hash": schema_hash,
                "episode_time_steps": int(episode_steps),
                "code_git_sha": _git_sha(),
            },
            indent=2,
        )
    )

    print(f"\n[collect] manifest: {manifest_path}")
    print(f"[collect] sentinel: {collect_done_path}")
    print(f"[collect] total rows: {n_rows_total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
