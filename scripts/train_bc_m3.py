"""CLI driver for M3 multi-seed BC training.

Trains one BC policy per seed on a single building's offline CSV, persists
per-seed artifacts under ``runs/offline_bc_m3/<run_id>/seed_<n>/``, and
writes aggregated ``multi_seed_summary.json`` + ``seeds_index.json``.

Example
-------
.. code-block:: bash

    python scripts/train_bc_m3.py \
        --dataset datasets/offline_rl/m2/offline_dataset_agent_4.csv \
        --output  runs/offline_bc_m3/bc-m3-v1 \
        --seeds   22,23,24 \
        --epochs  50

Defaults match the M3 plan:
  - val_episodes_mode = "random:2"
  - action_target     = "action_clean"
  - dropout=0.1, weight_decay=1e-5, gradient_clip_norm=1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.offline.bc_trainer import (  # noqa: E402
    BCTrainingConfig,
    train_bc_multi_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the offline RL CSV (single building).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root for per-seed artifacts.",
    )
    parser.add_argument(
        "--seeds",
        default="22,23,24",
        help="Comma-separated training seeds (default: 22,23,24).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--val-episodes-mode",
        default="random:2",
        help="'last:N' or 'random:N' (clean episodes only).",
    )
    parser.add_argument(
        "--action-target",
        default="action_clean",
        choices=["action", "action_clean"],
    )
    parser.add_argument(
        "--hidden-layers",
        default="256,256",
        help="Comma-separated hidden layer sizes.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mlflow-disable", action="store_true")
    parser.add_argument("--mlflow-experiment", default="offline_bc_m3")
    parser.add_argument("--mlflow-run-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    hidden = [int(h) for h in args.hidden_layers.split(",") if h.strip()]

    config = BCTrainingConfig(
        hidden_layers=hidden,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        gradient_clip_norm=args.gradient_clip_norm,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_episodes_mode=args.val_episodes_mode,
        action_target=args.action_target,
        device=args.device,
        mlflow_enabled=not args.mlflow_disable,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
    )

    summary = train_bc_multi_seed(
        csv_path=args.dataset,
        output_root=args.output,
        seeds=seeds,
        config_template=config,
    )

    print("\n=== Multi-seed training complete ===")
    print(f"  output_root  : {summary.output_root}")
    print(f"  best_seed    : {summary.best_seed}")
    print(f"  best_val_loss: {summary.best_val_loss:.6f}")
    print(f"  duration     : {summary.duration_seconds:.1f}s")
    print(f"  summary      : {summary.summary_path}")
    print(f"  seeds_index  : {summary.seeds_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
