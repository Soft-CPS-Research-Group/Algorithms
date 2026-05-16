"""Standalone feature analysis for the Building 5 offline RL dataset."""
from __future__ import annotations
from pathlib import Path

DATA_PATH   = Path("datasets/offline_rl/derived/rbc_with_reward.parquet")
OUTPUT_DIR  = Path("docs/offline_rl/feature_analysis")
FIGURES_DIR = OUTPUT_DIR / "figures"
DOC_PATH    = OUTPUT_DIR / "feature_analysis.md"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
