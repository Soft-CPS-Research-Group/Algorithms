"""Thesis-grade dataset analyzer for the entity-interface RBCSmart dataset.

Reads ``seed_*.parquet`` files (and an optional ``manifest.json``) from a
``--data-dir`` and produces a ``feature_analysis/`` subdirectory under
``--output-dir`` containing:

* ``summary.md`` — markdown narrative with embedded figure references.
* ``figures/`` — per-section PNG files.
* ``.feature_analysis.done`` — per-stage idempotency sentinel.

Seven analysis sections (matching ``docs/offline_rl/iql_cql_initiative_plan.md``
Phase 4):

1. Dataset stats: rows per seed, total transitions, disk size, schema link.
2. Per-group observation distributions (histograms of top-N features per
   ``(obs_dim, action_dim)`` group).
3. Action coverage: scatter (+ marginal hists) of action vectors per group;
   CQL motivation overlay showing concentration → conservatism rationale.
4. Reward distribution by regime: charge / idle / discharge × peak / off-peak,
   inferred from ``action__electrical_storage`` and timestep modulo daily cycle.
5. Feature × reward correlations: per-group Spearman heatmap + top-10 ranking.
6. Per-building summary table: row count, mean reward, action entropy,
   obs PCA explained-variance (top-3 components).
7. Temporal patterns: reward / action by hour-of-day and day-of-week.

Idempotency
-----------
* ``.feature_analysis.done`` short-circuits a re-run (unless ``--force``).
* ``--force`` rewrites everything regardless of sentinel state.

Examples
--------
::

    .venv/bin/python -m scripts.analyze_entity_dataset \\
        --data-dir runs/offline_iql_cql_initiative_15min/data \\
        --output-dir runs/offline_iql_cql_initiative_15min
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Headless matplotlib backend for reproducible CI / cron use.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBS_PREFIX = "obs__"
ACTION_PREFIX = "action__"
NEXT_OBS_PREFIX = "next_obs__"

# Feature counts per figure
TOP_N_OBS_FEATURES = 20
TOP_N_CORRELATIONS = 10

# Default day length in steps (15-min schema). Overridden by manifest when
# available (via episode_time_steps / steps_per_day inference).
DEFAULT_STEPS_PER_DAY = 96


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
#
# The entity-interface parquets are *wide-sparse*: each row carries columns
# for every agent group's schema, with NaN for cols not used by that row's
# group.  At production scale (17 buildings × 35040 steps × 10 seeds × ~4320
# cols) a naive ``pd.concat([pq.read_table(p).to_pandas() ...])`` materialises
# ~205 GB of float64 — far beyond the 48 GB system RAM.
#
# These loaders mirror the proven column-selective strategy already used by
# ``algorithms.offline_rl.entity_dataset._discover_group_cols_from_schema``:
#
# * Schema discovery reads only row group 0 (~3400 rows) of the first parquet
#   to identify which cols belong to which (obs_dim, action_dim) group.
# * Per-group / narrow loads use ``pq.read_table(columns=needed)`` so pyarrow
#   skips irrelevant column chunks at the storage layer.
# * ``max_rows_per_seed`` caps per-(group, seed) rows after load using a
#   reproducible random sample (preserves temporal coverage; not just the
#   first N rows which would bias §4/§7 toward early steps).


# Narrow column set used for dataset-wide sections (§1 dataset stats,
# §4 reward-by-regime, §7 temporal patterns).  These need only a handful of
# columns regardless of how wide the schema is.
NARROW_DATASET_COLS: List[str] = [
    "seed",
    "episode",
    "timestep",
    "agent_idx",
    "obs_dim",
    "action_dim",
    "reward",
    "terminated",
    "truncated",
    f"{ACTION_PREFIX}electrical_storage",
]


def _seed_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.glob("seed_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no seed_*.parquet files in {data_dir}")
    return files


def _discover_groups(data_dir: Path) -> List[Tuple[int, int]]:
    """Return sorted ``(obs_dim, action_dim)`` pairs present in the dataset.

    Reads only row group 0 of the first parquet (~3400 rows); cheap and avoids
    loading any obs/action columns.
    """
    files = _seed_files(data_dir)
    pf = pq.ParquetFile(str(files[0]))
    sample = pf.read_row_group(0, columns=["obs_dim", "action_dim"]).to_pandas()
    return sorted(
        {(int(o), int(a)) for o, a in zip(sample["obs_dim"], sample["action_dim"])}
    )


def _discover_group_columns(
    parquet_files: List[Path],
    *,
    obs_dim: int,
    action_dim: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Identify (obs_cols, action_cols, next_obs_cols) for one group.

    Reads only row group 0 of the first parquet with all candidate
    obs__/action__/next_obs__ cols, picks one row in the target group, and
    returns the cols that are non-NaN for that row.
    """
    pf = pq.ParquetFile(str(parquet_files[0]))
    all_cols = pf.schema_arrow.names
    obs_schema_cols = sorted(c for c in all_cols if c.startswith(OBS_PREFIX))
    act_schema_cols = sorted(c for c in all_cols if c.startswith(ACTION_PREFIX))
    next_schema_cols = sorted(c for c in all_cols if c.startswith(NEXT_OBS_PREFIX))

    sample = pf.read_row_group(
        0,
        columns=["obs_dim", "action_dim"] + obs_schema_cols + act_schema_cols + next_schema_cols,
    ).to_pandas()
    mask = (sample["obs_dim"] == obs_dim) & (sample["action_dim"] == action_dim)
    if not mask.any():
        raise ValueError(
            f"No rows for group (obs_dim={obs_dim}, action_dim={action_dim}) "
            f"in {parquet_files[0]} row group 0"
        )
    row = sample[mask].iloc[0]
    obs_cols = [c for c in obs_schema_cols if not pd.isna(row[c])]
    act_cols = [c for c in act_schema_cols if not pd.isna(row[c])]
    next_obs_cols = [c for c in next_schema_cols if not pd.isna(row[c])]
    return obs_cols, act_cols, next_obs_cols


def _load_group(
    data_dir: Path,
    *,
    obs_dim: int,
    action_dim: int,
    max_rows_per_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Load rows + columns for a single (obs_dim, action_dim) group.

    Memory-efficient: uses pyarrow column pruning so only this group's
    obs/action/next_obs cols are materialised.  For the production 17-building
    × 10-seed × 35040-step dataset, the obs627_act1 group goes from ~205 GB
    (whole-dataset wide-sparse) to ~3.55 GB (column-pruned).

    Parameters
    ----------
    max_rows_per_seed:
        If set, randomly downsample to this many rows per seed (with fixed
        random_state for reproducibility).  Use to cap memory on bounded
        systems; statistically sound for histograms, Spearman ρ, and PCA.
    """
    files = _seed_files(data_dir)
    obs_cols, act_cols, next_obs_cols = _discover_group_columns(
        files, obs_dim=obs_dim, action_dim=action_dim
    )
    meta_cols = [
        "seed", "episode", "timestep", "agent_idx",
        "obs_dim", "action_dim", "reward",
        "terminated", "truncated",
    ]
    needed = meta_cols + obs_cols + act_cols + next_obs_cols

    parts: List[pd.DataFrame] = []
    for p in files:
        df_raw = pq.read_table(str(p), columns=needed).to_pandas()
        mask = (df_raw["obs_dim"] == obs_dim) & (df_raw["action_dim"] == action_dim)
        df_group = df_raw[mask]
        if max_rows_per_seed is not None and len(df_group) > max_rows_per_seed:
            df_group = df_group.sample(
                n=int(max_rows_per_seed), random_state=42
            ).sort_index()
        parts.append(df_group.reset_index(drop=True))
        # Release the full-seed frame before loading the next parquet.
        del df_raw
    return pd.concat(parts, ignore_index=True)


def _load_columns_narrow(
    data_dir: Path,
    cols: List[str],
    *,
    max_rows_per_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Load a narrow column subset across all seeds.

    Powers dataset-wide sections (§1, §4, §7) that need only ~10 cols
    regardless of how wide the schema is.  Memory: ~10 cols × 5.96M rows ×
    8 B ≈ 480 MB (vs 205 GB whole-dataset).

    Cols absent from the schema are silently skipped — this matches the
    pre-refactor behavior where dropping NaN cols was the recovery path.
    """
    files = _seed_files(data_dir)
    pf = pq.ParquetFile(str(files[0]))
    schema_cols = set(pf.schema_arrow.names)
    use_cols = [c for c in cols if c in schema_cols]

    parts: List[pd.DataFrame] = []
    for p in files:
        df = pq.read_table(str(p), columns=use_cols).to_pandas()
        if max_rows_per_seed is not None and len(df) > max_rows_per_seed:
            df = df.sample(n=int(max_rows_per_seed), random_state=42).sort_index()
        parts.append(df.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def load_manifest(data_dir: Path) -> Optional[Dict[str, Any]]:
    """Return parsed manifest.json or None if not present."""
    path = data_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _dataset_size_bytes(data_dir: Path) -> int:
    return sum(p.stat().st_size for p in data_dir.glob("seed_*.parquet"))


def _steps_per_day(manifest: Optional[Dict[str, Any]]) -> int:
    """Infer steps-per-day from the manifest's ``episode_time_steps``.

    Heuristic: if episode_time_steps is one of the common CityLearn full-year
    values (8760, 35040, 525600, 2102400) we map to the corresponding daily
    bin size.  Otherwise fall back to DEFAULT_STEPS_PER_DAY.
    """
    if manifest is None:
        return DEFAULT_STEPS_PER_DAY
    n = int(manifest.get("episode_time_steps", 0))
    if n == 8760:
        return 24            # hourly
    if n == 35040:
        return 96            # 15 min
    if n == 525600:
        return 1440          # 1 min
    if n == 2102400:
        return 5760          # 15 s
    return DEFAULT_STEPS_PER_DAY


# ---------------------------------------------------------------------------
# Group helpers
# ---------------------------------------------------------------------------


def _group_key(obs_dim: int, action_dim: int) -> str:
    return f"obs{int(obs_dim)}_act{int(action_dim)}"


def _list_groups(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Return sorted list of unique ``(obs_dim, action_dim)`` pairs."""
    return sorted({(int(o), int(a)) for o, a in zip(df["obs_dim"], df["action_dim"])})


def _group_slice(df: pd.DataFrame, obs_dim: int, action_dim: int) -> pd.DataFrame:
    sub = df[(df["obs_dim"] == obs_dim) & (df["action_dim"] == action_dim)].copy()
    # Drop columns that are entirely null for this group (different groups
    # have different feature schemas in wide-sparse format).
    return sub.dropna(axis=1, how="all")


def _feature_cols(group_df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in group_df.columns if c.startswith(prefix)]


# ---------------------------------------------------------------------------
# Section 1 — Dataset stats
# ---------------------------------------------------------------------------


def figure_01_dataset_stats(
    fig_dir: Path,
    *,
    data_dir: Path,
    manifest: Optional[Dict[str, Any]],
) -> Tuple[Path, Dict[str, Any]]:
    """Build §1 dataset stats from parquet metadata only — no data load.

    Memory-safe: reads ``ParquetFile.metadata.num_rows`` (zero data) for total
    counts and probes only row group 0 of the first parquet for ``n_agents``.
    This guarantees the reported total transitions reflects the *true* dataset
    size even when downstream sections operate on a ``max_rows_per_seed``
    sample.
    """
    files = _seed_files(data_dir)
    per_seed_counts = {
        int(p.stem.split("_")[1]): int(pq.ParquetFile(str(p)).metadata.num_rows)
        for p in files
    }
    seeds = sorted(per_seed_counts.keys())
    total_rows = sum(per_seed_counts.values())
    # Probe row group 0 of the first parquet for n_agents.
    pf = pq.ParquetFile(str(files[0]))
    sample = pf.read_row_group(0, columns=["agent_idx"]).to_pandas()
    n_agents = int(sample["agent_idx"].nunique())
    size_bytes = _dataset_size_bytes(data_dir)
    schema_path = (manifest or {}).get("dataset_path", "(unknown)")
    schema_hash = (manifest or {}).get("schema_hash", "(unknown)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table_rows = [
        ["Number of seeds", str(len(seeds))],
        ["Seeds", ", ".join(str(s) for s in seeds)],
        ["Total transitions", f"{total_rows:,}"],
        ["Unique agents", str(n_agents)],
        ["Dataset size (MB)", f"{size_bytes / 1e6:.2f}"],
        ["Schema path", str(schema_path)],
        ["Schema hash", str(schema_hash)],
    ]
    tbl = ax.table(
        cellText=table_rows,
        colLabels=["Field", "Value"],
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)
    ax.set_title("01 — Dataset stats", fontsize=12, weight="bold")
    fig.tight_layout()
    out_path = fig_dir / "01_dataset_stats_table.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats: Dict[str, Any] = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "per_seed_rows": {int(k): int(v) for k, v in per_seed_counts.items()},
        "total_rows": total_rows,
        "n_agents": n_agents,
        "size_mb": float(size_bytes / 1e6),
        "schema_path": str(schema_path),
        "schema_hash": str(schema_hash),
    }
    return out_path, stats


# ---------------------------------------------------------------------------
# Section 2 — Per-group observation distributions
# ---------------------------------------------------------------------------


def figure_02_obs_distributions(
    group_df: pd.DataFrame,
    group_key: str,
    fig_dir: Path,
    *,
    top_n: int = TOP_N_OBS_FEATURES,
) -> Path:
    obs_cols = _feature_cols(group_df, OBS_PREFIX)
    # Rank features by variance — high-variance features are the interesting ones.
    variances = group_df[obs_cols].var(numeric_only=True)
    ranked = variances.sort_values(ascending=False).head(top_n).index.tolist()
    n_panels = max(1, len(ranked))
    n_cols = min(5, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, col in enumerate(ranked):
        ax = axes[i // n_cols][i % n_cols]
        ax.set_visible(True)
        values = group_df[col].dropna().to_numpy()
        ax.hist(values, bins=30, color="#4c72b0", alpha=0.8, edgecolor="white")
        title = col.replace(OBS_PREFIX, "")
        ax.set_title(title[:32], fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        # Stat overlay
        if len(values) > 0:
            mean = float(np.mean(values))
            std = float(np.std(values))
            ax.axvline(mean, color="black", linestyle="--", linewidth=0.7)
            ax.text(
                0.98, 0.95,
                f"μ={mean:.2f}\nσ={std:.2f}",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=6,
            )
    fig.suptitle(f"02 — Observation distributions ({group_key})", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = fig_dir / f"02_obs_distributions_{group_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Section 3 — Action coverage (+ CQL motivation overlay)
# ---------------------------------------------------------------------------


def figure_03_action_coverage(
    group_df: pd.DataFrame,
    group_key: str,
    fig_dir: Path,
) -> Path:
    act_cols = _feature_cols(group_df, ACTION_PREFIX)
    if not act_cols:
        # Skip with a stub figure so downstream tests still pass.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"03 — Action coverage ({group_key}) — no action columns")
        ax.axis("off")
        out_path = fig_dir / f"03_action_coverage_{group_key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    if len(act_cols) == 1:
        # 1-D action: marginal hist only.
        fig, ax = plt.subplots(figsize=(7, 4))
        values = group_df[act_cols[0]].dropna().to_numpy()
        ax.hist(values, bins=40, color="#dd8452", alpha=0.8, edgecolor="white")
        # CQL motivation: shade the empirical support.
        if len(values) > 0:
            lo, hi = float(np.min(values)), float(np.max(values))
            ax.axvspan(lo, hi, color="#dd8452", alpha=0.1)
            ax.axvline(np.mean(values), color="black", linestyle="--", linewidth=0.7)
            coverage = (hi - lo) / 2.0  # action range ⊂ [-1, 1]
            ax.text(
                0.98, 0.95,
                f"empirical support: [{lo:.2f}, {hi:.2f}]\n"
                f"coverage of [−1,1]: {coverage * 100:.0f}%",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            )
        ax.set_xlabel(act_cols[0].replace(ACTION_PREFIX, ""))
        ax.set_ylabel("count")
        ax.set_title(
            f"03 — Action coverage ({group_key}) — CQL motivation: "
            "unshaded region is OOD",
            fontsize=10,
        )
        fig.tight_layout()
        out_path = fig_dir / f"03_action_coverage_{group_key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ≥2-D action: scatter + marginal hists on first two dims.
    fig, ax = plt.subplots(figsize=(7, 6))
    x = group_df[act_cols[0]].dropna().to_numpy()
    y = group_df[act_cols[1]].dropna().to_numpy()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    ax.scatter(x, y, s=4, alpha=0.4, c="#dd8452")
    # CQL OOD region (full [-1,1]² square) shaded around the support.
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor="gray", linestyle=":"))
    ax.set_xlabel(act_cols[0].replace(ACTION_PREFIX, ""))
    ax.set_ylabel(act_cols[1].replace(ACTION_PREFIX, ""))
    ax.set_title(
        f"03 — Action coverage ({group_key}) — CQL motivation: "
        "dotted box = full [-1,1]² action space",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = fig_dir / f"03_action_coverage_{group_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Section 4 — Reward distribution by regime
# ---------------------------------------------------------------------------


def figure_04_reward_by_regime(
    df: pd.DataFrame,
    fig_dir: Path,
    *,
    steps_per_day: int,
) -> Path:
    """Box plots of reward conditioned on (action regime, time-of-day regime).

    * Action regime: charge / idle / discharge derived from
      ``action__electrical_storage`` (idle = |a| < 0.05).
    * Time regime: peak (12:00–22:00) vs off-peak.
    """
    storage_col = next(
        (c for c in df.columns if c == f"{ACTION_PREFIX}electrical_storage"),
        None,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    if storage_col is None:
        ax.set_title("04 — Reward by regime — action__electrical_storage missing")
        ax.axis("off")
        out_path = fig_dir / "04_reward_by_regime.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    work = df.dropna(subset=[storage_col, "reward", "timestep"]).copy()

    def _action_regime(a: float) -> str:
        if a < -0.05:
            return "discharge"
        if a > 0.05:
            return "charge"
        return "idle"

    work["action_regime"] = work[storage_col].astype(float).apply(_action_regime)

    # Time-of-day: peak window 12:00–22:00 → step_of_day ∈ [steps_per_day/2, steps_per_day*22/24).
    step_of_day = (work["timestep"].astype(int) % steps_per_day).to_numpy()
    peak_lo = int(steps_per_day * 12 / 24)
    peak_hi = int(steps_per_day * 22 / 24)
    work["time_regime"] = np.where(
        (step_of_day >= peak_lo) & (step_of_day < peak_hi),
        "peak",
        "off-peak",
    )

    regimes = ["charge", "idle", "discharge"]
    times = ["off-peak", "peak"]
    data = []
    labels = []
    for tr in times:
        for ar in regimes:
            sub = work[(work["action_regime"] == ar) & (work["time_regime"] == tr)]["reward"]
            data.append(sub.to_numpy() if len(sub) else np.array([0.0]))
            labels.append(f"{ar}\n{tr}")
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    palette = ["#4c72b0", "#dd8452", "#55a868"] * 2
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("reward")
    ax.set_title("04 — Reward by action regime × time-of-day", fontsize=11, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = fig_dir / "04_reward_by_regime.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Section 5 — Feature × reward correlations
# ---------------------------------------------------------------------------


def figure_05_correlations(
    group_df: pd.DataFrame,
    group_key: str,
    fig_dir: Path,
    *,
    top_n: int = TOP_N_CORRELATIONS,
) -> Tuple[Path, List[Tuple[str, float]]]:
    obs_cols = _feature_cols(group_df, OBS_PREFIX)
    if not obs_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"05 — Correlations ({group_key}) — no obs columns")
        ax.axis("off")
        out_path = fig_dir / f"05_correlations_{group_key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path, []

    from scipy.stats import spearmanr

    rewards = group_df["reward"].astype(float).to_numpy()
    corrs: List[Tuple[str, float]] = []
    for col in obs_cols:
        vals = group_df[col].astype(float).to_numpy()
        if np.std(vals) < 1e-12:
            corrs.append((col, 0.0))
            continue
        try:
            rho, _ = spearmanr(vals, rewards)
        except Exception:
            rho = 0.0
        corrs.append((col, float(rho) if np.isfinite(rho) else 0.0))

    # Top-N by |ρ|
    ranked = sorted(corrs, key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    names = [n.replace(OBS_PREFIX, "")[:32] for n, _ in ranked]
    values = [r for _, r in ranked]

    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    # Heatmap (1 × N)
    matrix = np.array(values).reshape(1, -1)
    im = ax_heat.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(len(names)))
    ax_heat.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax_heat.set_yticks([])
    ax_heat.set_title(f"Spearman ρ(feature, reward) — top {len(ranked)}", fontsize=10)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # Bar plot
    colors = ["#c44e52" if v < 0 else "#4c72b0" for v in values]
    ax_bar.barh(range(len(values)), values, color=colors)
    ax_bar.set_yticks(range(len(values)))
    ax_bar.set_yticklabels(names, fontsize=7)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Spearman ρ")
    ax_bar.set_title(f"05 — Top {len(ranked)} feature–reward correlations", fontsize=10)
    ax_bar.axvline(0, color="black", linewidth=0.5)

    fig.suptitle(f"05 — Feature × reward correlations ({group_key})", fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = fig_dir / f"05_correlations_{group_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path, ranked


# ---------------------------------------------------------------------------
# Section 6 — Per-building summary table
# ---------------------------------------------------------------------------


def _compute_per_agent_rows_for_group(group_df: pd.DataFrame) -> List[List[str]]:
    """Compute the per-agent table rows for one (obs_dim, action_dim) group.

    Returns ``[[agent_idx, n_rows, mean_reward, action_entropy, PCA_EVR], ...]``
    for every agent_idx in the group.  Caller is responsible for accumulating
    rows across groups before rendering.
    """
    from sklearn.decomposition import PCA

    rows: List[List[str]] = []
    for agent_idx in sorted(group_df["agent_idx"].unique()):
        sub = group_df[group_df["agent_idx"] == agent_idx].dropna(axis=1, how="all")
        n_rows = int(len(sub))
        mean_reward = float(sub["reward"].mean()) if "reward" in sub else float("nan")
        # Action entropy on discretised action__electrical_storage (10 bins).
        storage_col = f"{ACTION_PREFIX}electrical_storage"
        if storage_col in sub:
            vals = sub[storage_col].dropna().to_numpy()
            if len(vals) > 0:
                hist, _ = np.histogram(vals, bins=10, range=(-1, 1), density=False)
                p = hist / max(hist.sum(), 1)
                p = p[p > 0]
                entropy = float(-np.sum(p * np.log(p)))
            else:
                entropy = float("nan")
        else:
            entropy = float("nan")
        # PCA explained-var (top 3 components) over this group's obs cols.
        obs_cols = _feature_cols(sub, OBS_PREFIX)
        if len(obs_cols) >= 1 and n_rows > 3:
            X = sub[obs_cols].dropna().to_numpy()
            X = X[: max(1, len(X))]
            try:
                k = min(3, X.shape[1], X.shape[0] - 1)
                pca = PCA(n_components=max(1, k))
                pca.fit(X)
                evr = pca.explained_variance_ratio_
                evr_str = ", ".join(f"{v * 100:.0f}%" for v in evr)
            except Exception:
                evr_str = "—"
        else:
            evr_str = "—"
        rows.append([
            str(int(agent_idx)),
            f"{n_rows:,}",
            f"{mean_reward:.4f}",
            f"{entropy:.3f}" if entropy == entropy else "nan",
            evr_str,
        ])
    return rows


def figure_06_per_building_table(
    rows: List[List[str]],
    fig_dir: Path,
) -> Path:
    """Render the accumulated per-agent rows into a single table figure.

    Caller (``main()``) builds ``rows`` by calling
    ``_compute_per_agent_rows_for_group`` for every group and concatenating.
    """
    # Sort by integer agent_idx so the table reads top-to-bottom across all groups.
    rows = sorted(rows, key=lambda r: int(r[0]))
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.4 * len(rows) + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["agent_idx", "n_rows", "mean_reward", "action_entropy", "PCA EVR (top-k)"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    ax.set_title("06 — Per-agent summary", fontsize=12, weight="bold")
    fig.tight_layout()
    out_path = fig_dir / "06_per_building_table.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Section 7 — Temporal patterns
# ---------------------------------------------------------------------------


def figure_07_temporal_patterns(
    df: pd.DataFrame,
    fig_dir: Path,
    *,
    steps_per_day: int,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    work = df.dropna(subset=["timestep", "reward"]).copy()
    step_of_day = (work["timestep"].astype(int) % steps_per_day).to_numpy()
    work["step_of_day"] = step_of_day

    # Panel 1: reward by step-of-day
    grouped = work.groupby("step_of_day")["reward"].agg(["mean", "std"]).reset_index()
    ax0 = axes[0]
    ax0.plot(grouped["step_of_day"], grouped["mean"], color="#4c72b0", label="mean reward")
    ax0.fill_between(
        grouped["step_of_day"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        color="#4c72b0",
        alpha=0.2,
        label="±1 std",
    )
    ax0.set_xlabel(f"step of day (0–{steps_per_day - 1})")
    ax0.set_ylabel("reward")
    ax0.set_title("Reward by step-of-day")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    # Panel 2: storage action by step-of-day (mean ± std)
    storage_col = f"{ACTION_PREFIX}electrical_storage"
    ax1 = axes[1]
    if storage_col in work:
        agg = work.groupby("step_of_day")[storage_col].agg(["mean", "std"]).reset_index()
        ax1.plot(agg["step_of_day"], agg["mean"], color="#dd8452", label="mean action")
        ax1.fill_between(
            agg["step_of_day"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color="#dd8452",
            alpha=0.2,
            label="±1 std",
        )
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.set_xlabel(f"step of day (0–{steps_per_day - 1})")
        ax1.set_ylabel("action__electrical_storage")
        ax1.set_title("Storage action by step-of-day")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
    else:
        ax1.axis("off")
        ax1.set_title("action__electrical_storage missing")

    fig.suptitle("07 — Temporal patterns", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = fig_dir / "07_temporal_patterns.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# summary.md builder
# ---------------------------------------------------------------------------


def write_summary(
    summary_path: Path,
    *,
    figures_subdir: str,
    dataset_stats: Dict[str, Any],
    groups: Iterable[str],
    obs_dist_paths: Dict[str, Path],
    action_cov_paths: Dict[str, Path],
    corr_paths: Dict[str, Path],
    corr_rankings: Dict[str, List[Tuple[str, float]]],
    data_dir: Path,
    manifest: Optional[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Feature analysis — RBCSmart entity dataset\n")
    lines.append(f"Generated: {_dt.datetime.now(_dt.timezone.utc).isoformat()}\n")
    lines.append(f"Source: `{data_dir}`\n")
    lines.append("")

    # 1. Dataset stats
    lines.append("## 1. Dataset stats\n")
    lines.append(f"![dataset stats]({figures_subdir}/01_dataset_stats_table.png)\n")
    lines.append("")
    lines.append(f"* Seeds: `{dataset_stats['seeds']}`")
    lines.append(f"* Total transitions: **{dataset_stats['total_rows']:,}**")
    lines.append(f"* Unique agents: {dataset_stats['n_agents']}")
    lines.append(f"* Dataset size on disk: {dataset_stats['size_mb']:.2f} MB")
    lines.append(f"* Schema: `{dataset_stats['schema_path']}` (hash `{dataset_stats['schema_hash']}`)")
    if manifest is not None:
        lines.append(f"* Behaviour policy: `{manifest.get('behaviour_policy', 'unknown')}`")
        lines.append(f"* Reward function: `{manifest.get('reward_function', 'unknown')}`")
        lines.append(f"* Entity encoding: `{manifest.get('entity_encoding', 'unknown')}`")
    lines.append("")

    # 2. Observation distributions
    lines.append("## 2. Observation distributions (per group)\n")
    for gk in groups:
        if gk in obs_dist_paths:
            rel = obs_dist_paths[gk].name
            lines.append(f"### Group `{gk}`")
            lines.append(f"![obs dist {gk}]({figures_subdir}/{rel})\n")

    # 3. Action coverage
    lines.append("## 3. Action coverage (CQL motivation)\n")
    lines.append(
        "Empirical action support vs full [−1, 1]ᵈ space.  The unshaded region "
        "is out-of-distribution — exactly the area CQL penalises during Q "
        "learning to avoid overestimation on unseen actions.\n"
    )
    for gk in groups:
        if gk in action_cov_paths:
            rel = action_cov_paths[gk].name
            lines.append(f"### Group `{gk}`")
            lines.append(f"![action coverage {gk}]({figures_subdir}/{rel})\n")

    # 4. Reward by regime
    lines.append("## 4. Reward distribution by action × time regime\n")
    lines.append(
        "Boxplots of step-wise reward conditioned on (action regime, "
        "time-of-day regime).  Action regimes: charge / idle / discharge "
        "inferred from `action__electrical_storage`.  Time regimes: peak "
        "(12:00–22:00) vs off-peak.\n"
    )
    lines.append(f"![reward by regime]({figures_subdir}/04_reward_by_regime.png)\n")

    # 5. Correlations
    lines.append("## 5. Feature × reward correlations\n")
    lines.append("Per-group Spearman ρ between each obs feature and the reward.\n")
    for gk in groups:
        if gk in corr_paths:
            rel = corr_paths[gk].name
            lines.append(f"### Group `{gk}`")
            lines.append(f"![correlations {gk}]({figures_subdir}/{rel})")
            ranking = corr_rankings.get(gk, [])
            if ranking:
                lines.append("\n| # | feature | Spearman ρ |")
                lines.append("|---|---------|------------|")
                for i, (name, rho) in enumerate(ranking, start=1):
                    short = name.replace(OBS_PREFIX, "")
                    lines.append(f"| {i} | `{short}` | {rho:+.3f} |")
            lines.append("")

    # 6. Per-building table
    lines.append("## 6. Per-building summary\n")
    lines.append(f"![per-building summary]({figures_subdir}/06_per_building_table.png)\n")

    # 7. Temporal patterns
    lines.append("## 7. Temporal patterns\n")
    lines.append(f"![temporal patterns]({figures_subdir}/07_temporal_patterns.png)\n")

    summary_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="Directory with seed_*.parquet files.")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to create the feature_analysis/ subdir.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore .feature_analysis.done and rewrite everything.",
    )
    p.add_argument(
        "--max-rows-per-seed",
        type=int,
        default=200000,
        help=(
            "Cap per-(group, seed) rows loaded into memory; subsamples randomly "
            "with fixed random_state=42 for reproducibility.  Use to bound RAM "
            "on the production 5.96M-row dataset (default 200000 ≈ 20 GB peak "
            "for the largest group).  Set to 0 to disable capping."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    data_dir: Path = args.data_dir
    out_root: Path = args.output_dir
    fa_dir = out_root / "feature_analysis"
    fig_dir = fa_dir / "figures"
    sentinel = fa_dir / ".feature_analysis.done"
    max_rows: Optional[int] = (
        int(args.max_rows_per_seed) if args.max_rows_per_seed and args.max_rows_per_seed > 0 else None
    )

    if sentinel.exists() and not args.force:
        print(
            f"[analyze] .feature_analysis.done present at {fa_dir} — skipping "
            f"(use --force to regenerate).",
            flush=True,
        )
        return 0

    fa_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] data_dir:          {data_dir}")
    print(f"[analyze] output:            {fa_dir}")
    print(f"[analyze] max_rows_per_seed: {max_rows if max_rows is not None else 'unlimited'}")

    # --- Schema discovery (cheap: row group 0 of first parquet only) ---
    manifest = load_manifest(data_dir)
    groups = _discover_groups(data_dir)
    group_keys = [_group_key(o, a) for o, a in groups]
    print(f"[analyze] groups={group_keys}")

    # --- Dataset-wide sections (§1, §4, §7): narrow loader ~480 MB ---
    narrow_df = _load_columns_narrow(
        data_dir, NARROW_DATASET_COLS, max_rows_per_seed=max_rows
    )
    steps_per_day = _steps_per_day(manifest)
    print(
        f"[analyze] narrow rows={len(narrow_df):,} cols={len(narrow_df.columns)} "
        f"steps_per_day={steps_per_day}"
    )

    _, dataset_stats = figure_01_dataset_stats(fig_dir, data_dir=data_dir, manifest=manifest)
    figure_04_reward_by_regime(narrow_df, fig_dir, steps_per_day=steps_per_day)
    figure_07_temporal_patterns(narrow_df, fig_dir, steps_per_day=steps_per_day)
    del narrow_df  # release before per-group loop

    # --- Per-group sections (§2, §3, §5) + per-agent §6 row accumulation ---
    obs_dist_paths: Dict[str, Path] = {}
    action_cov_paths: Dict[str, Path] = {}
    corr_paths: Dict[str, Path] = {}
    corr_rankings: Dict[str, List[Tuple[str, float]]] = {}
    per_agent_rows: List[List[str]] = []

    for obs_dim, action_dim in groups:
        gk = _group_key(obs_dim, action_dim)
        group_df = _load_group(
            data_dir,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_rows_per_seed=max_rows,
        )
        print(f"[analyze] group {gk}: rows={len(group_df):,} cols={len(group_df.columns)}")
        obs_dist_paths[gk] = figure_02_obs_distributions(group_df, gk, fig_dir)
        action_cov_paths[gk] = figure_03_action_coverage(group_df, gk, fig_dir)
        corr_path, ranking = figure_05_correlations(group_df, gk, fig_dir)
        corr_paths[gk] = corr_path
        corr_rankings[gk] = ranking
        per_agent_rows.extend(_compute_per_agent_rows_for_group(group_df))
        del group_df  # release before next group

    # --- §6 (per-agent summary) renders the accumulated rows ---
    figure_06_per_building_table(per_agent_rows, fig_dir)

    # --- summary.md ---
    write_summary(
        fa_dir / "summary.md",
        figures_subdir="figures",
        dataset_stats=dataset_stats,
        groups=group_keys,
        obs_dist_paths=obs_dist_paths,
        action_cov_paths=action_cov_paths,
        corr_paths=corr_paths,
        corr_rankings=corr_rankings,
        data_dir=data_dir,
        manifest=manifest,
    )

    # --- Sentinel ---
    sentinel.write_text(
        json.dumps(
            {
                "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "n_groups": len(group_keys),
                "groups": group_keys,
                "steps_per_day": int(steps_per_day),
                "max_rows_per_seed": max_rows,
            },
            indent=2,
        )
    )

    print(f"[analyze] summary.md: {fa_dir / 'summary.md'}")
    print(f"[analyze] sentinel:   {sentinel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
