"""Train CQL on the entity-interface RBCSmart dataset.

Multi-seed, multi-group driver around
``algorithms.offline_rl.cql_entity_trainer``.

Identical interface to ``train_iql_entity.py`` with additional
``--cql-alpha`` and ``--cql-n-random-actions`` flags.

Examples
--------
Smoke (1 seed, 200 steps, tiny net, one group)::

    .venv/bin/python -m scripts.train_cql_entity \\
        --data-dir datasets/offline_rl/rbcsmart_entity \\
        --output runs/offline_cql_entity/smoke \\
        --seeds 22 \\
        --val-seeds 23 \\
        --groups 706:2 \\
        --gradient-steps 200 \\
        --hidden-layers 64,64 \\
        --eval-every 50

Full (5 seeds, 150k steps, all groups)::

    nohup .venv/bin/python -m scripts.train_cql_entity \\
        --data-dir datasets/offline_rl/rbcsmart_entity \\
        --output runs/offline_cql_entity/run-001 \\
        --seeds 22,23,24,25,26 \\
        --val-seeds 26 \\
        --gradient-steps 150000 \\
        > runs/offline_cql_entity/run-001/train.log 2>&1 &
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

from algorithms.offline_rl.cql_entity_trainer import (  # noqa: E402
    CQLTrainingConfig,
    train_all_groups,
    train_cql_multi_seed,
)
from algorithms.offline_rl.entity_schema import AGENT_GROUPS  # noqa: E402


def _parse_groups(raw: str) -> List[Tuple[int, int]]:
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


def parse_args() -> argparse.Namespace:
    default_data = REPO_ROOT / "datasets" / "offline_rl" / "rbcsmart_entity"
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=Path, default=default_data)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", default="22,23,24,25,26")
    p.add_argument("--val-seeds", default=None)
    p.add_argument("--groups", default=None)
    # Architecture
    p.add_argument("--hidden-layers", default="256,256")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-std-init", type=float, default=math.log(0.1))
    # IQL/CQL shared hyperparams
    p.add_argument("--tau-expectile", type=float, default=0.7)
    p.add_argument("--beta-advantage", type=float, default=3.0)
    p.add_argument("--advantage-clip", type=float, default=100.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau-target", type=float, default=0.005)
    # CQL-specific
    p.add_argument("--cql-alpha", type=float, default=0.2,
                   help="Weight of the conservative Q penalty (default 0.2)")
    p.add_argument("--cql-n-random-actions", type=int, default=10,
                   help="Random actions per state for CQL logsumexp (default 10)")
    p.add_argument("--cql-min-q-weight", type=float, default=0.0,
                   help="Lower bound for CQL penalty (0 = disabled)")
    # Optimisation
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gradient-clip-norm", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gradient-steps", type=int, default=150_000)
    p.add_argument("--eval-every", type=int, default=2_500)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    seeds: List[int] = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("[train_cql_entity] ERROR: --seeds is empty", file=sys.stderr)
        return 1

    val_seeds: Optional[List[int]]
    if args.val_seeds is not None:
        val_seeds = [int(s) for s in args.val_seeds.split(",") if s.strip()]
    else:
        val_seeds = [seeds[-1]]

    groups: Optional[List[Tuple[int, int]]]
    if args.groups is not None:
        groups = _parse_groups(args.groups)
    else:
        groups = list(AGENT_GROUPS)

    hidden = [int(h) for h in args.hidden_layers.split(",") if h.strip()]
    config = CQLTrainingConfig(
        hidden_layers=hidden,
        dropout=args.dropout,
        log_std_init=args.log_std_init,
        tau_expectile=args.tau_expectile,
        beta_advantage=args.beta_advantage,
        advantage_clip=args.advantage_clip,
        gamma=args.gamma,
        tau_target=args.tau_target,
        cql_alpha=args.cql_alpha,
        cql_n_random_actions=args.cql_n_random_actions,
        cql_min_q_weight=args.cql_min_q_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
        eval_every_n_steps=args.eval_every,
        device=args.device,
    )

    if not args.data_dir.exists():
        print(
            f"[train_cql_entity] ERROR: data-dir not found: {args.data_dir}",
            file=sys.stderr,
        )
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[train_cql_entity] data_dir   = {args.data_dir}")
    print(f"[train_cql_entity] output     = {args.output}")
    print(f"[train_cql_entity] seeds      = {seeds}")
    print(f"[train_cql_entity] val_seeds  = {val_seeds}")
    print(f"[train_cql_entity] groups     = {groups}")
    print(f"[train_cql_entity] cql_alpha  = {args.cql_alpha}")
    print(f"[train_cql_entity] config     = {config}")

    if len(groups) == len(AGENT_GROUPS) and groups == list(AGENT_GROUPS):
        results = train_all_groups(
            data_dir=args.data_dir,
            output_root=args.output,
            seeds=seeds,
            val_seeds=val_seeds,
            config=config,
            groups=groups,
        )
    else:
        results = {}
        for obs_dim, action_dim in groups:
            group_key = f"obs{obs_dim}_act{action_dim}"
            group_out = args.output / group_key
            print(f"\n[train_cql_entity] === group {group_key} ===", flush=True)
            agg = train_cql_multi_seed(
                data_dir=args.data_dir,
                output_root=group_out,
                obs_dim=obs_dim,
                action_dim=action_dim,
                seeds=seeds,
                val_seeds=val_seeds,
                config=config,
            )
            results[group_key] = agg

    print("\n=== CQL entity training complete ===")
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
