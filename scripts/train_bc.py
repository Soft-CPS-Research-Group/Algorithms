"""Train BC on the RBC dataset.

Multi-seed driver around ``algorithms.offline_rl.bc_trainer``.

Inputs
------
* ``--dataset`` — Parquet matching the offline-RL schema. Defaults to
  ``datasets/offline_rl/derived/rbc_with_reward.parquet``.
* ``--output`` — root dir for per-seed artefacts.
* ``--seeds`` — comma-separated training seeds (default 100,101,102,103,104).

Outputs
-------
For each seed, ``<output>/seed_<N>/`` containing ``policy.pt``,
``obs_standardiser.npz``, ``architecture.json``, ``metrics.jsonl``,
``seed_summary.json``. Aggregated ``multi_seed_summary.json`` and
``seeds_index.json`` at ``<output>/``.

Examples
--------
Smoke-run (1 seed, 5 epochs, tiny net)::

    .venv/bin/python -m scripts.train_bc \\
        --output runs/offline_bc/smoke \\
        --seeds 100 \\
        --epochs 5 \\
        --hidden-layers 64,64

Full run (5 seeds, 50 epochs, default arch)::

    .venv/bin/python -m scripts.train_bc \\
        --output runs/offline_bc/run-001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.offline_rl.bc_trainer import (  # noqa: E402
    BCTrainingConfig,
    train_multi_seed,
)


DEFAULT_DATASET = (
    REPO_ROOT
    / "datasets"
    / "offline_rl"
    / "derived"
    / "rbc_with_reward.parquet"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--seeds",
        default="100,101,102,103,104",
        help="Comma-separated training seeds.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gradient-clip-norm", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--hidden-layers", default="256,256")
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    hidden = [int(h) for h in args.hidden_layers.split(",") if h.strip()]
    config = BCTrainingConfig(
        hidden_layers=hidden,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        device=args.device,
        num_workers=args.num_workers,
    )
    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")

    print(f"[train_bc] dataset={args.dataset}")
    print(f"[train_bc] output={args.output}")
    print(f"[train_bc] seeds={seeds}")
    print(f"[train_bc] config={config}")

    summary = train_multi_seed(
        args.dataset,
        args.output,
        seeds=seeds,
        config=config,
    )
    print("\n=== BC multi-seed training complete ===")
    print(f"  output_root           : {args.output}")
    print(f"  n_seeds               : {summary['n_seeds']}")
    print(f"  final_val_mse mean±std: {summary['final_val_mse_mean']:.6f} ± {summary['final_val_mse_std']:.6f}")
    print(f"  best_val_mse  mean±std: {summary['best_val_mse_mean']:.6f} ± {summary['best_val_mse_std']:.6f}")
    print(f"  duration              : {summary['duration_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
