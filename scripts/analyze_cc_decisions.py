"""Analyse and visualise Community Coordinator decision traces.

Loads per-episode CC decision CSVs from an MLflow run and produces:

  1. Per-building o1 signals over time   — do buildings get different signals?
  2. Community context over time         — price, import, PV, carbon
  3. Price vs mean o1 correlation        — does CC charge cheap / discharge expensive?
  4. Cross-building differentiation      — o1_std per step (key thesis metric)
  5. Building heatmap                    — o1 per building across a single episode
  6. Summary statistics table            — printed to stdout

Usage
-----
  # Analyse a specific MLflow run (all episodes):
  python scripts/analyze_cc_decisions.py --run-id <mlflow_run_id>

  # Analyse a specific episode only:
  python scripts/analyze_cc_decisions.py --run-id <mlflow_run_id> --episode 5

  # Load from a local folder containing the CSVs directly:
  python scripts/analyze_cc_decisions.py --csv-dir /path/to/decision_traces/

  # Save plots to disk instead of displaying:
  python scripts/analyze_cc_decisions.py --run-id <run_id> --output-dir ./plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe default; overridden below if interactive

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-id",  help="MLflow run ID to load decision_traces artifacts from.")
    src.add_argument("--csv-dir", help="Local directory containing cc_decisions_ep*.csv files.")

    p.add_argument("--episode",    type=int, default=None, help="Single episode to analyse. Default: last episode.")
    p.add_argument("--output-dir", default=None,            help="Save plots here instead of showing interactively.")
    p.add_argument("--tracking-uri", default=None,          help="MLflow tracking URI. Uses MLFLOW_TRACKING_URI env var if unset.")
    p.add_argument("--top-n-buildings", type=int, default=6, help="Max buildings shown on the per-building o1 plot (default 6).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_from_mlflow(run_id: str, tracking_uri: Optional[str]) -> List[pd.DataFrame]:
    """Download all cc_decisions CSVs for a run, return list of DataFrames sorted by episode."""
    import mlflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id, path="decision_traces")
    except Exception as e:
        sys.exit(f"Cannot list artifacts for run {run_id}: {e}")

    csvs = sorted([a.path for a in artifacts if a.path.endswith(".csv")])
    if not csvs:
        sys.exit(f"No cc_decisions CSVs found under decision_traces/ for run {run_id}.")

    tmp_dir = Path("/tmp/cc_analysis")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for path in csvs:
        local = client.download_artifacts(run_id, path, str(tmp_dir))
        df = pd.read_csv(local)
        # Extract episode number from filename for labelling.
        ep_str = Path(path).stem.split("ep")[1].split("_")[0] if "ep" in Path(path).stem else "?"
        df["_episode"] = int(ep_str) if ep_str.isdigit() else len(dfs) + 1
        dfs.append(df)

    print(f"Loaded {len(dfs)} episodes from run {run_id}.")
    return dfs


def _load_from_dir(csv_dir: str) -> List[pd.DataFrame]:
    """Load all cc_decisions CSVs from a local directory."""
    directory = Path(csv_dir)
    csvs = sorted(directory.glob("cc_decisions_ep*.csv"))
    if not csvs:
        sys.exit(f"No cc_decisions_ep*.csv files found in {csv_dir}.")

    dfs = []
    for path in csvs:
        df = pd.read_csv(path)
        ep_str = path.stem.split("ep")[1].split("_")[0] if "ep" in path.stem else str(len(dfs) + 1)
        df["_episode"] = int(ep_str) if ep_str.isdigit() else len(dfs) + 1
        dfs.append(df)

    print(f"Loaded {len(dfs)} episodes from {csv_dir}.")
    return dfs


def _select_episode(dfs: List[pd.DataFrame], episode: Optional[int]) -> pd.DataFrame:
    if episode is None:
        df = dfs[-1]
        print(f"Using last episode: {df['_episode'].iloc[0]}.")
    else:
        matches = [d for d in dfs if d["_episode"].iloc[0] == episode]
        if not matches:
            available = [d["_episode"].iloc[0] for d in dfs]
            sys.exit(f"Episode {episode} not found. Available: {available}")
        df = matches[0]
        print(f"Using episode {episode}.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _building_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("o1_b")],
                  key=lambda c: int(c[4:]))


def _save_or_show(fig: plt.Figure, output_dir: Optional[Path], name: str) -> None:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — Per-building o1 signals
# ---------------------------------------------------------------------------

def plot_per_building_o1(df: pd.DataFrame, top_n: int, output_dir: Optional[Path]) -> None:
    cols = _building_cols(df)
    if not cols:
        print("  [skip] No o1_b* columns found.")
        return

    # Pick top_n buildings with highest signal variance (most interesting).
    variances = {c: df[c].var() for c in cols}
    selected = sorted(variances, key=variances.get, reverse=True)[:top_n]
    remaining = [c for c in cols if c not in selected]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1, ax2 = axes

    # Top: selected buildings
    for col in selected:
        b_idx = col[4:]
        ax1.plot(df["cc_step"], df[col], label=f"Building {b_idx}", alpha=0.85, linewidth=1.0)
    ax1.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("o1 signal")
    ax1.set_title(f"Per-building CC signal — episode {df['_episode'].iloc[0]} "
                  f"(top {top_n} by variance)")
    ax1.legend(ncol=3, fontsize=8)
    ax1.set_ylim(-1.1, 1.1)

    # Bottom: price overlay
    if "price" in df.columns:
        ax2.plot(df["cc_step"], df["price"], color="black", linewidth=1.0, label="Price")
        ax2.set_ylabel("Price ($/kWh)")
        ax2.set_xlabel("CC step")
        ax2.set_title("Electricity price")
        ax2.legend()

    fig.tight_layout()
    _save_or_show(fig, output_dir, f"ep{df['_episode'].iloc[0]}_per_building_o1.png")


# ---------------------------------------------------------------------------
# Plot 2 — Community context
# ---------------------------------------------------------------------------

def plot_community_context(df: pd.DataFrame, output_dir: Optional[Path]) -> None:
    ctx_cols = {
        "price":               "Price ($/kWh)",
        "community_import_kw": "Community import (kW)",
        "community_pv_kw":     "Community PV (kW)",
        "community_net_kw":    "Community net (kW)",
        "carbon_intensity":    "Carbon intensity",
        "active_evs":          "Active EVs",
    }
    available = [(c, lbl) for c, lbl in ctx_cols.items() if c in df.columns]
    if not available:
        print("  [skip] No community context columns found.")
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, available):
        ax.plot(df["cc_step"], df[col], linewidth=1.0)
        ax.set_ylabel(label, fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    axes[-1].set_xlabel("CC step")
    fig.suptitle(f"Community context — episode {df['_episode'].iloc[0]}", fontsize=11)
    fig.tight_layout()
    _save_or_show(fig, output_dir, f"ep{df['_episode'].iloc[0]}_community_context.png")


# ---------------------------------------------------------------------------
# Plot 3 — Price vs mean o1 (arbitrage learning)
# ---------------------------------------------------------------------------

def plot_price_vs_o1(df: pd.DataFrame, output_dir: Optional[Path]) -> None:
    if "price" not in df.columns or "o1_mean" not in df.columns:
        print("  [skip] price or o1_mean column missing.")
        return

    corr = df["price"].corr(df["o1_mean"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(df["cc_step"], df["o1_mean"], color="steelblue", linewidth=1.0, label="o1 mean")
    axes[0].axhline(0, color="grey", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("o1 mean")
    axes[0].legend()

    axes[1].plot(df["cc_step"], df["price"], color="darkorange", linewidth=1.0, label="Price")
    axes[1].set_ylabel("Price ($/kWh)")
    axes[1].set_xlabel("CC step")
    axes[1].legend()

    fig.suptitle(
        f"Price vs CC mean signal — episode {df['_episode'].iloc[0]}\n"
        f"corr(price, o1_mean) = {corr:+.3f}  "
        f"(want NEGATIVE: charge cheap, discharge expensive)",
        fontsize=10,
    )
    fig.tight_layout()
    _save_or_show(fig, output_dir, f"ep{df['_episode'].iloc[0]}_price_vs_o1.png")
    print(f"  corr(price, o1_mean) = {corr:+.3f}")


# ---------------------------------------------------------------------------
# Plot 4 — Cross-building differentiation over time
# ---------------------------------------------------------------------------

def plot_differentiation(df: pd.DataFrame, output_dir: Optional[Path]) -> None:
    if "o1_std" not in df.columns:
        print("  [skip] o1_std column missing.")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["cc_step"], df["o1_std"], color="purple", linewidth=1.0)
    ax.axhline(df["o1_std"].mean(), color="purple", linewidth=0.8, linestyle="--",
               label=f"mean = {df['o1_std'].mean():.3f}")
    ax.fill_between(df["cc_step"], 0, df["o1_std"], alpha=0.15, color="purple")
    ax.set_ylabel("o1 std across buildings")
    ax.set_xlabel("CC step")
    ax.set_title(
        f"Cross-building differentiation — episode {df['_episode'].iloc[0]}\n"
        f"std > 0 = CC assigns different signals to different buildings "
        f"(thesis key metric)"
    )
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, output_dir, f"ep{df['_episode'].iloc[0]}_differentiation.png")
    print(f"  mean cross-building std = {df['o1_std'].mean():.4f}  "
          f"(0 = broadcast, >0 = genuine coordination)")


# ---------------------------------------------------------------------------
# Plot 5 — Building heatmap
# ---------------------------------------------------------------------------

def plot_building_heatmap(df: pd.DataFrame, output_dir: Optional[Path]) -> None:
    cols = _building_cols(df)
    if not cols:
        print("  [skip] No o1_b* columns for heatmap.")
        return

    matrix = df[cols].T.values   # (N_buildings, T_steps)
    b_labels = [f"B{c[4:]}" for c in cols]

    fig, ax = plt.subplots(figsize=(16, max(4, len(cols) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="o1 signal")
    ax.set_yticks(range(len(b_labels)))
    ax.set_yticklabels(b_labels, fontsize=7)
    ax.set_xlabel("CC step")
    ax.set_title(
        f"CC signal heatmap — episode {df['_episode'].iloc[0]}\n"
        f"Green = charge (+1), Red = discharge (−1)"
    )
    fig.tight_layout()
    _save_or_show(fig, output_dir, f"ep{df['_episode'].iloc[0]}_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 6 — Learning curve across episodes (o1_std + corr(price,o1))
# ---------------------------------------------------------------------------

def plot_learning_curve(dfs: List[pd.DataFrame], output_dir: Optional[Path]) -> None:
    if len(dfs) < 2:
        print("  [skip] Only one episode — no learning curve.")
        return

    episodes, corrs, stds = [], [], []
    for df in dfs:
        ep = df["_episode"].iloc[0]
        episodes.append(ep)
        if "price" in df.columns and "o1_mean" in df.columns:
            corrs.append(df["price"].corr(df["o1_mean"]))
        else:
            corrs.append(float("nan"))
        stds.append(df["o1_std"].mean() if "o1_std" in df.columns else float("nan"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(episodes, corrs, marker="o", markersize=4, color="darkorange")
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("corr(price, o1_mean)")
    ax1.set_title("Arbitrage learning — want negative (charge cheap, discharge expensive)")

    ax2.plot(episodes, stds, marker="o", markersize=4, color="purple")
    ax2.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("mean o1_std across buildings")
    ax2.set_xlabel("Episode")
    ax2.set_title("Cross-building differentiation — want > 0")

    fig.suptitle("CC learning curves across episodes", fontsize=11)
    fig.tight_layout()
    _save_or_show(fig, output_dir, "learning_curves.png")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    cols = _building_cols(df)
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Episode:        {df['_episode'].iloc[0]}")
    print(f"  CC steps:       {len(df)}")
    if "price" in df.columns and "o1_mean" in df.columns:
        print(f"  corr(price,o1): {df['price'].corr(df['o1_mean']):+.3f}  (target: negative)")
    if "o1_std" in df.columns:
        print(f"  mean o1_std:    {df['o1_std'].mean():.4f}  (target: > 0)")
        print(f"  max  o1_std:    {df['o1_std'].max():.4f}")
    if cols:
        per_building = df[cols].mean().rename(lambda c: f"B{c[4:]}")
        print(f"\n  Mean o1 per building:")
        print(per_building.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("─────────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Allow interactive display if no output-dir.
    if args.output_dir is None:
        matplotlib.use("TkAgg" if sys.platform != "darwin" else "MacOSX")
        import importlib; importlib.reload(plt)

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.run_id:
        dfs = _load_from_mlflow(args.run_id, args.tracking_uri)
    else:
        dfs = _load_from_dir(args.csv_dir)

    # Episode-level analysis.
    df = _select_episode(dfs, args.episode)
    print_summary(df)

    print("Generating plots…")
    plot_per_building_o1(df, args.top_n_buildings, output_dir)
    plot_community_context(df, output_dir)
    plot_price_vs_o1(df, output_dir)
    plot_differentiation(df, output_dir)
    plot_building_heatmap(df, output_dir)

    # Cross-episode learning curves (uses all episodes).
    plot_learning_curve(dfs, output_dir)

    if output_dir:
        print(f"\nAll plots saved to {output_dir}/")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
