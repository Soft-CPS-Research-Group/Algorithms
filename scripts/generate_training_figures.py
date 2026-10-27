"""Generate training-curve figures for IQL and CQL entity training runs.

Reads metrics.jsonl files (one per seed per agent group) from the
offline_iql_entity and offline_cql_entity run directories and produces:

  fig_training_loss_<group>.png   – v_loss, q_loss, policy_loss (mean ± std across seeds)
  fig_training_valmse_all.png     – val_policy_mse for IQL vs CQL, all groups
  fig_training_cql_penalty.png    – CQL-specific cql_penalty curve, all groups

Outputs are written to --output-dir (default: thesis/ch5/assets).

Usage:
    python scripts/generate_training_figures.py \
        --iql-root .worktrees/entity-iql/runs/offline_iql_entity/run-001 \
        --cql-root .worktrees/entity-iql/runs/offline_cql_entity/run-001 \
        --output-dir /path/to/thesis/ch5/assets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_group_metrics(run_root: Path, group_key: str) -> List[Dict]:
    """Load all metrics.jsonl records for a group, across all seeds."""
    group_dir = run_root / group_key
    if not group_dir.exists():
        return []
    records: List[Dict] = []
    for seed_dir in sorted(group_dir.iterdir()):
        mfile = seed_dir / "metrics.jsonl"
        if mfile.exists():
            seed_records = [json.loads(l) for l in mfile.read_text().splitlines() if l.strip()]
            records.append(seed_records)
    return records  # list of per-seed record lists


def _mean_std(per_seed: List[List[Dict]], key: str):
    """Return (steps, mean, std) arrays for a given metric key."""
    if not per_seed:
        return np.array([]), np.array([]), np.array([])
    steps = np.array([r["step"] for r in per_seed[0]])
    values = np.array([[r[key] for r in seed_records] for seed_records in per_seed])
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return steps, mean, std


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_ALGO_COLORS = {"IQL": "#2196F3", "CQL": "#F44336"}
_GROUP_LABELS = {
    "obs627_act1": "Group A (obs=627, act=1) — 10 bldgs",
    "obs706_act2": "Group B (obs=706, act=2) — 5 bldgs",
    "obs749_act3": "Group C (obs=749, act=3) — B1",
    "obs785_act3": "Group D (obs=785, act=3) — B15",
}


def _plot_loss_curves(
    iql_root: Path | None,
    cql_root: Path | None,
    group_key: str,
    output_dir: Path,
) -> Path:
    """One figure with 3 loss subplots (v_loss, q_loss, policy_loss) for one group."""
    loss_keys = ["v_loss", "q_loss", "policy_loss"]
    loss_labels = ["Value loss", "Q-function loss", "Policy loss"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    group_label = _GROUP_LABELS.get(group_key, group_key)
    fig.suptitle(f"Training loss — {group_label}", fontsize=12)

    for ax, key, label in zip(axes, loss_keys, loss_labels):
        for algo, root in [("IQL", iql_root), ("CQL", cql_root)]:
            if root is None:
                continue
            per_seed = _load_group_metrics(root, group_key)
            if not per_seed:
                continue
            steps, mean, std = _mean_std(per_seed, key)
            color = _ALGO_COLORS[algo]
            ax.plot(steps, mean, label=algo, color=color, linewidth=1.8)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.18, color=color)
        ax.set_xlabel("Gradient step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    fig.tight_layout()
    out_path = output_dir / f"fig_training_loss_{group_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return out_path


def _plot_val_mse(
    iql_root: Path | None,
    cql_root: Path | None,
    groups: List[str],
    output_dir: Path,
) -> Path:
    """Single figure: val_policy_mse for IQL vs CQL across all groups."""
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Validation policy MSE — IQL vs CQL", fontsize=12)

    for ax, group_key in zip(axes, groups):
        for algo, root in [("IQL", iql_root), ("CQL", cql_root)]:
            if root is None:
                continue
            per_seed = _load_group_metrics(root, group_key)
            if not per_seed:
                continue
            steps, mean, std = _mean_std(per_seed, "val_policy_mse")
            color = _ALGO_COLORS[algo]
            ax.plot(steps, mean, label=algo, color=color, linewidth=1.8)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.18, color=color)
        short = group_key.replace("obs", "obs=").replace("_act", ", act=")
        ax.set_title(short, fontsize=9)
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Validation MSE")
        ax.legend(fontsize=8)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    fig.tight_layout()
    out_path = output_dir / "fig_training_valmse_all.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return out_path


def _plot_cql_penalty(
    cql_root: Path | None,
    groups: List[str],
    output_dir: Path,
) -> Path | None:
    """CQL-specific: cql_penalty across all groups."""
    if cql_root is None:
        return None
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), sharey=False)
    if len(groups) == 1:
        axes = [axes]
    fig.suptitle("CQL conservative penalty", fontsize=12)

    for ax, group_key in zip(axes, groups):
        per_seed = _load_group_metrics(cql_root, group_key)
        if not per_seed:
            ax.set_visible(False)
            continue
        # check key exists
        if "cql_penalty" not in per_seed[0][0]:
            ax.set_visible(False)
            continue
        steps, mean, std = _mean_std(per_seed, "cql_penalty")
        color = _ALGO_COLORS["CQL"]
        ax.plot(steps, mean, color=color, linewidth=1.8)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.18, color=color)
        short = group_key.replace("obs", "obs=").replace("_act", ", act=")
        ax.set_title(short, fontsize=9)
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("CQL penalty")
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    fig.tight_layout()
    out_path = output_dir / "fig_training_cql_penalty.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iql-root", type=Path,
                   default=Path(".worktrees/entity-iql/runs/offline_iql_entity/run-001"),
                   help="Root dir of IQL entity training run.")
    p.add_argument("--cql-root", type=Path,
                   default=Path(".worktrees/entity-iql/runs/offline_cql_entity/run-001"),
                   help="Root dir of CQL entity training run.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("thesis/ch5/assets"),
                   help="Destination for generated PNG files.")
    p.add_argument("--groups", nargs="+",
                   default=["obs627_act1", "obs706_act2", "obs749_act3", "obs785_act3"],
                   help="Agent group keys to process.")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    iql_root = args.iql_root if args.iql_root.exists() else None
    cql_root = args.cql_root if args.cql_root.exists() else None

    if iql_root is None and cql_root is None:
        print("ERROR: neither --iql-root nor --cql-root found on disk.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[training-figures] iql_root = {iql_root}")
    print(f"[training-figures] cql_root = {cql_root}")
    print(f"[training-figures] output   = {args.output_dir}")
    print()

    # Filter to groups that actually have data
    active_groups = []
    for g in args.groups:
        has_iql = iql_root is not None and (iql_root / g).exists()
        has_cql = cql_root is not None and (cql_root / g).exists()
        if has_iql or has_cql:
            active_groups.append(g)
    print(f"[training-figures] active groups: {active_groups}\n")

    print("Generating per-group loss curves...")
    for g in active_groups:
        _plot_loss_curves(iql_root, cql_root, g, args.output_dir)

    print("\nGenerating validation MSE comparison...")
    _plot_val_mse(iql_root, cql_root, active_groups, args.output_dir)

    print("\nGenerating CQL penalty curves...")
    _plot_cql_penalty(cql_root, active_groups, args.output_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
