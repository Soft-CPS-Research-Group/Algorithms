"""Apply frozen reward weights to a raw parquet dataset (apply-only variant).

This script applies the frozen ``reward_weights.json`` to any collection of
seed_*.parquet files **without re-fitting** — weights are loaded as-is.

It is the Step 6 counterpart to ``calibrate_reward.py`` (which fits weights
from RBC rollouts). Use this script whenever you want to stamp a new behaviour
dataset with the same reward yardstick as the RBC data, so that run-001 and
run-002 are comparable.

Usage
-----
  # Apply frozen weights to the IQL raw dataset:
  .venv/bin/python -m scripts.apply_reward \\
      --input datasets/offline_rl/iql/seed_32.parquet \\
             datasets/offline_rl/iql/seed_33.parquet \\
      --weights datasets/offline_rl/derived/reward_weights.json \\
      --output datasets/offline_rl/iql_derived/iql_with_reward.parquet

  # Glob shorthand (shell expands):
  .venv/bin/python -m scripts.apply_reward \\
      --input datasets/offline_rl/iql/seed_*.parquet \\
      --weights datasets/offline_rl/derived/reward_weights.json \\
      --output datasets/offline_rl/iql_derived/iql_with_reward.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.offline_rl import reward as RW  # noqa: E402

DEFAULT_WEIGHTS_PATH = REPO_ROOT / "datasets" / "offline_rl" / "derived" / "reward_weights.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "datasets" / "offline_rl" / "iql_derived" / "iql_with_reward.parquet"


# ---------------------------------------------------------------------------
# Core function (importable by tests)
# ---------------------------------------------------------------------------


def apply_reward(
    *,
    input_paths: Sequence[Path],
    weights_path: Path,
    output_path: Path,
) -> None:
    """Load parquets, apply frozen weights, write augmented output.

    Each input parquet's ``reward`` column is overwritten with the value
    computed by ``RW.compute_reward_vectorised``. All other columns are
    preserved unchanged.

    Parameters
    ----------
    input_paths:
        One or more parquet files. They are concatenated in the order given.
    weights_path:
        Path to a ``reward_weights.json`` produced by ``calibrate_reward.py``
        (or compatible format: ``{"weights": {...}}``.
    output_path:
        Destination parquet. Parent directories are created as needed.
    """
    weights = RW.load_weights(weights_path)

    parts: List[pd.DataFrame] = []
    for p in input_paths:
        df = pd.read_parquet(p)
        reward_array, _ = RW.compute_reward_vectorised(df, weights=weights)
        df = df.copy()
        df["reward"] = reward_array
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Input parquet file(s). Multiple paths are concatenated.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Frozen reward weights JSON. Default: {DEFAULT_WEIGHTS_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output parquet path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    args = parser.parse_args(argv)

    input_paths: List[Path] = args.input
    missing = [p for p in input_paths if not p.exists()]
    if missing:
        print(f"[apply_reward] ERROR: input file(s) not found: {missing}", file=sys.stderr)
        return 1

    if not args.weights.exists():
        print(f"[apply_reward] ERROR: weights file not found: {args.weights}", file=sys.stderr)
        return 1

    weights = RW.load_weights(args.weights)
    print(f"[apply_reward] weights: {weights}")
    print(f"[apply_reward] inputs : {[str(p) for p in input_paths]}")
    print(f"[apply_reward] output : {args.output}")

    apply_reward(
        input_paths=input_paths,
        weights_path=args.weights,
        output_path=args.output,
    )

    out_df = pd.read_parquet(args.output)
    n_non_finite = int(np.sum(~np.isfinite(out_df["reward"].to_numpy())))
    print(f"[apply_reward] rows written : {len(out_df)}")
    print(f"[apply_reward] reward mean  : {out_df['reward'].mean():.4f}")
    print(f"[apply_reward] reward std   : {out_df['reward'].std():.4f}")
    print(f"[apply_reward] non-finite   : {n_non_finite}")

    if n_non_finite > 0:
        print(f"[apply_reward] WARNING: {n_non_finite} non-finite reward values!", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
