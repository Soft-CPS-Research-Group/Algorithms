"""Train IQL on the entity-interface RBCSmart dataset.

Multi-seed, multi-group driver around
``algorithms.offline_rl.iql_entity_trainer``.

Each agent group (identified by obs_dim × action_dim) gets its own set of
policy, Q-networks, and value network.  All four groups are trained by default;
use ``--groups`` to restrict to a subset.

Inputs
------
* ``--data-dir``  — Directory with ``seed_*.parquet`` files from
  ``scripts/collect_rbcsmart_dataset.py``.
* ``--output``    — Root dir for per-group / per-seed artefacts.
* ``--seeds``     — Comma-separated training seeds (default: 22,23,24,25,26).
* ``--val-seeds`` — Comma-separated seeds to hold out for validation
                    (default: last seed in ``--seeds``).

Outputs
-------
``<output>/<group_key>/seed_<N>/`` containing ``policy.pt``, ``q1.pt``,
``q2.pt``, ``value.pt``, ``obs_standardiser.npz``, ``architecture.json``,
``metrics.jsonl``, ``seed_summary.json``.

Aggregated per group: ``multi_seed_summary.json``, ``seeds_index.json``.
All-groups: ``all_groups_summary.json`` at ``<output>/``.

Examples
--------
Smoke (1 seed, 200 steps, tiny net, one group)::

    .venv/bin/python -m scripts.train_iql_entity \\
        --data-dir datasets/offline_rl/rbcsmart_entity \\
        --output runs/offline_iql_entity/smoke \\
        --seeds 22 \\
        --groups 706:2 \\
        --gradient-steps 200 \\
        --hidden-layers 64,64 \\
        --eval-every 50

Full (5 seeds, 150k steps, all groups)::

    nohup .venv/bin/python -m scripts.train_iql_entity \\
        --data-dir datasets/offline_rl/rbcsmart_entity \\
        --output runs/offline_iql_entity/run-001 \\
        --seeds 22,23,24,25,26 \\
        --val-seeds 26 \\
        --gradient-steps 150000 \\
        > runs/offline_iql_entity/run-001/train.log 2>&1 &
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.offline_rl.entity_schema import AGENT_GROUPS  # noqa: E402
from algorithms.offline_rl.iql_entity_trainer import (  # noqa: E402
    train_all_groups,
    train_entity_multi_seed,
)
from algorithms.offline_rl.iql_trainer import IQLTrainingConfig  # noqa: E402


def _parse_groups(raw: str) -> List[Tuple[int, int]]:
    """Parse ``'627:1,706:2'`` → ``[(627, 1), (706, 2)]``."""
    result = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"invalid group spec '{token}'; expected 'obs_dim:action_dim'"
            )
        result.append((int(parts[0]), int(parts[1])))
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    default_data = REPO_ROOT / "datasets" / "offline_rl" / "rbcsmart_entity"
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=Path, default=default_data)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--seeds",
        default="22,23,24,25,26",
        help="Comma-separated training seeds (default: 22,23,24,25,26)",
    )
    p.add_argument(
        "--val-seeds",
        default=None,
        help="Comma-separated validation seeds (default: last seed in --seeds)",
    )
    p.add_argument(
        "--groups",
        default=None,
        help=(
            "Comma-separated obs_dim:action_dim pairs to train "
            "(default: all four groups). E.g. '706:2,627:1'"
        ),
    )
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
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=5_000,
        help=(
            "Persist a within-seed checkpoint every N gradient steps. "
            "Forwarded to IQLTrainingConfig.checkpoint_every_n_steps. "
            "Default: 5000."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Ignore each seed's seed.done / checkpoint_latest.pt sentinels "
            "and retrain from scratch. Forwarded to "
            "train_all_groups / train_entity_multi_seed."
        ),
    )
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    seeds: List[int] = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("[train_iql_entity] ERROR: --seeds is empty", file=sys.stderr)
        return 1

    val_seeds: Optional[List[int]]
    if args.val_seeds is not None:
        val_seeds = [int(s) for s in args.val_seeds.split(",") if s.strip()]
    else:
        val_seeds = [seeds[-1]]  # hold out last seed by default

    groups: Optional[List[Tuple[int, int]]]
    if args.groups is not None:
        groups = _parse_groups(args.groups)
    else:
        groups = list(AGENT_GROUPS)

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
        checkpoint_every_n_steps=args.checkpoint_every,
        device=args.device,
    )

    if not args.data_dir.exists():
        print(
            f"[train_iql_entity] ERROR: data-dir not found: {args.data_dir}",
            file=sys.stderr,
        )
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[train_iql_entity] data_dir   = {args.data_dir}")
    print(f"[train_iql_entity] output     = {args.output}")
    print(f"[train_iql_entity] seeds      = {seeds}")
    print(f"[train_iql_entity] val_seeds  = {val_seeds}")
    print(f"[train_iql_entity] groups     = {groups}")
    print(f"[train_iql_entity] config     = {config}")

    if len(groups) == len(AGENT_GROUPS) and groups == list(AGENT_GROUPS):
        # Convenience: use train_all_groups (writes all_groups_summary.json)
        results = train_all_groups(
            data_dir=args.data_dir,
            output_root=args.output,
            seeds=seeds,
            val_seeds=val_seeds,
            config=config,
            groups=groups,
            force=args.force,
        )
    else:
        # Subset of groups: run each independently
        results = {}
        for obs_dim, action_dim in groups:
            group_key = f"obs{obs_dim}_act{action_dim}"
            group_out = args.output / group_key
            print(f"\n[train_iql_entity] === group {group_key} ===", flush=True)
            agg = train_entity_multi_seed(
                data_dir=args.data_dir,
                output_root=group_out,
                obs_dim=obs_dim,
                action_dim=action_dim,
                seeds=seeds,
                val_seeds=val_seeds,
                config=config,
                force=args.force,
            )
            results[group_key] = agg

    print("\n=== IQL entity training complete ===")
    for group_key, agg in results.items():
        mse_mean = agg.get("best_val_policy_mse_mean", float("nan"))
        mse_std = agg.get("best_val_policy_mse_std", float("nan"))
        duration = agg.get("duration_seconds", float("nan"))
        print(
            f"  {group_key:20s}  val_mse={mse_mean:.6f} ± {mse_std:.6f}  "
            f"duration={duration:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
