"""Standalone feature analysis for the Building 5 offline RL dataset."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH   = Path("datasets/offline_rl/derived/rbc_with_reward.parquet")
OUTPUT_DIR  = Path("docs/offline_rl/feature_analysis")
FIGURES_DIR = OUTPUT_DIR / "figures"
DOC_PATH    = OUTPUT_DIR / "feature_analysis.md"


def _section_overview(df: pd.DataFrame) -> tuple[str, str]:
    n_rows, n_cols = df.shape
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    act_cols = [c for c in df.columns if c.startswith("action_")]
    missing = df.isnull().sum().sum()
    storage_var = df["action_electrical_storage"].var() if "action_electrical_storage" in df.columns else None
    storage_note = (
        f"`action_electrical_storage` variance = {storage_var:.6f} — **constant zero; excluded from analysis.**"
        if storage_var is not None and storage_var < 1e-9
        else f"`action_electrical_storage` variance = {storage_var:.6f}"
    )
    reward_min = df["reward"].min()
    reward_max = df["reward"].max()
    reward_mean = df["reward"].mean()
    reward_std = df["reward"].std()

    md = f"""## 1. Dataset overview

| Property | Value |
|---|---|
| Rows | {n_rows:,} |
| Columns | {n_cols} |
| Observation features | {len(obs_cols)} |
| Action dimensions | {len(act_cols)} |
| Missing values | {missing} |
| Reward range | [{reward_min:.3f}, {reward_max:.3f}] |
| Reward mean ± std | {reward_mean:.3f} ± {reward_std:.3f} |

{storage_note}
"""
    return "", md


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
