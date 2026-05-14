"""Train IQL on the RBC dataset.

Multi-seed driver around ``algorithms.offline_rl.iql_trainer``.

Inputs
------
* ``--dataset`` — Parquet matching the offline-RL schema.
* ``--output`` — root dir for per-seed artefacts.
* ``--seeds`` — comma-separated training seeds.

Outputs
-------
For each seed, ``<output>/seed_<N>/`` containing ``policy.pt``, ``q1.pt``,
``q2.pt``, ``value.pt``, ``obs_standardiser.npz``, ``architecture.json``,
``metrics.jsonl``, ``seed_summary.json``. Aggregated ``multi_seed_summary.json``
and ``seeds_index.json`` at ``<output>/``.

Examples
--------
Smoke (1 seed, 200 steps, tiny net)::

    .venv/bin/python -m scripts.train_iql \\
        --output runs/offline_iql/smoke \\
        --seeds 100 \\
        --gradient-steps 200 \\
        --hidden-layers 64,64 \\
        --eval-every 50

Full (5 seeds, 150k steps)::

    nohup .venv/bin/python -m scripts.train_iql \\
        --output runs/offline_iql/run-001 \\
        --seeds 100,101,102,103,104 \\
        --gradient-steps 150000 \\
        > runs/offline_iql/run-001/train.log 2>&1 &
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.offline_rl.iql_trainer import (  # noqa: E402
    IQLTrainingConfig,
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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", default="100,101,102,103,104")
    # Architecture
    p.add_argument("--hidden-layers", default="256,256")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-std-init", type=float, default=math.log(0.1))
    # IQL hyperparams
    p.add_argument("--tau-expectile", type=float, default=0.7)
    p.add_argument("--beta-advantage", type=float, default=3.0)
    p.add_argument("--advantage-clip", type=float, default=100.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau-target", type=float, default=0.005)
    # Optimisation
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gradient-clip-norm", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gradient-steps", type=int, default=150_000)
    p.add_argument("--eval-every", type=int, default=2_500)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    hidden = [int(h) for h in args.hidden_layers.split(",") if h.strip()]
    config = IQLTrainingConfig(
        hidden_layers=hidden,
        dropout=args.dropout,
        log_std_init=args.log_std_init,
        tau_expectile=args.tau_expectile,
        beta_advantage=args.beta_advantage,
        advantage_clip=args.advantage_clip,
        gamma=args.gamma,
        tau_target=args.tau_target,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
        eval_every_n_steps=args.eval_every,
        val_fraction=args.val_fraction,
        device=args.device,
    )
    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[train_iql] dataset={args.dataset}")
    print(f"[train_iql] output={args.output}")
    print(f"[train_iql] seeds={seeds}")
    print(f"[train_iql] config={config}")

    summary = train_multi_seed(args.dataset, args.output, seeds=seeds, config=config)
    print("\n=== IQL multi-seed training complete ===")
    print(f"  output_root              : {args.output}")
    print(f"  n_seeds                  : {summary['n_seeds']}")
    print(
        f"  best_val_policy_mse mean±std: "
        f"{summary['best_val_policy_mse_mean']:.6f} ± "
        f"{summary['best_val_policy_mse_std']:.6f}"
    )
    print(f"  duration                 : {summary['duration_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
