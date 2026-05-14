"""Tests for scripts/collect_iql_dataset.py.

Written before implementation (TDD / RED phase).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from algorithms.offline_rl import schema as S

REPO_ROOT = Path(__file__).resolve().parents[2]
IQL_CHECKPOINT_SEED_101 = REPO_ROOT / "runs" / "offline_iql" / "run-001" / "seed_101"
FROZEN_WEIGHTS = REPO_ROOT / "datasets" / "offline_rl" / "derived" / "reward_weights.json"


# ---------------------------------------------------------------------------
# 1. Default seeds are disjoint from RBC and eval seeds
# ---------------------------------------------------------------------------


def test_default_seeds_disjoint() -> None:
    """Seeds 32..41 must not overlap with RBC (22..31) or eval (200..209)."""
    from scripts.collect_iql_dataset import DEFAULT_SEEDS

    rbc_seeds = set(range(22, 32))
    eval_seeds = set(range(200, 210))
    iql_seeds = set(DEFAULT_SEEDS)

    assert len(iql_seeds) == 10
    assert iql_seeds == set(range(32, 42))
    assert iql_seeds.isdisjoint(rbc_seeds), "IQL seeds overlap with RBC seeds"
    assert iql_seeds.isdisjoint(eval_seeds), "IQL seeds overlap with eval seeds"


# ---------------------------------------------------------------------------
# 2. _build_iql returns IQLAgent loaded from a checkpoint dir
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not IQL_CHECKPOINT_SEED_101.exists(),
    reason="IQL run-001/seed_101 checkpoint not present",
)
def test_build_iql_returns_iql_agent() -> None:
    from algorithms.offline_rl.iql_agent import IQLAgent
    from scripts.collect_iql_dataset import _build_iql

    agent = _build_iql(IQL_CHECKPOINT_SEED_101)
    assert isinstance(agent, IQLAgent)


# ---------------------------------------------------------------------------
# 3. collect_seed produces valid schema rows and both actions have variance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not IQL_CHECKPOINT_SEED_101.exists(),
    reason="IQL run-001/seed_101 checkpoint not present",
)
def test_collect_seed_schema_and_action_variance() -> None:
    """IQL controls both actions; neither should be constant across a full rollout."""
    from scripts.collect_iql_dataset import collect_seed

    result = collect_seed(32, checkpoint_dir=IQL_CHECKPOINT_SEED_101)
    rows = result["rows"]
    assert len(rows) > 0

    df = pd.DataFrame(rows)
    # All schema columns must be present.
    for col in S.all_columns():
        assert col in df.columns, f"missing column: {col}"

    # Both action columns must have non-zero variance.
    for col in S.ACTION_COLUMNS:
        std = float(df[col].std())
        assert std > 1e-6, (
            f"IQL action column {col!r} has zero variance (std={std:.3e}); "
            "policy may be degenerate."
        )


# ---------------------------------------------------------------------------
# 4. assert_action_variance passes for IQL (no expected_zero_variance needed)
# ---------------------------------------------------------------------------


def test_assert_action_variance_passes_for_iql_actions() -> None:
    """IQL uses both action dims; assert_action_variance should not raise."""
    from scripts.collect_iql_dataset import assert_action_variance

    stats = {
        S.action_column("ev_charger"): {"mean": 0.1, "std": 0.5, "min": -1.0, "max": 1.0},
        S.action_column("electrical_storage"): {"mean": 0.05, "std": 0.3, "min": -1.0, "max": 1.0},
    }
    # Should not raise — both actions have variance.
    assert_action_variance(stats)


def test_assert_action_variance_raises_on_zero_std() -> None:
    """Zero-variance action column without expected_zero_variance must raise SchemaError."""
    from scripts.collect_iql_dataset import assert_action_variance

    stats = {
        S.action_column("ev_charger"): {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
        S.action_column("electrical_storage"): {"mean": 0.1, "std": 0.4, "min": -1.0, "max": 1.0},
    }
    with pytest.raises(S.SchemaError):
        assert_action_variance(stats)


# ---------------------------------------------------------------------------
# 5. Manifest written with behaviour_policy == "iql"
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not IQL_CHECKPOINT_SEED_101.exists(),
    reason="IQL run-001/seed_101 checkpoint not present",
)
def test_write_manifest_has_iql_policy(tmp_path: Path) -> None:
    from scripts.collect_iql_dataset import write_manifest

    seed_path = tmp_path / "seed_32.parquet"
    seed_path.touch()
    manifest_path = write_manifest(
        tmp_path,
        checkpoint_dir=IQL_CHECKPOINT_SEED_101,
        seeds=[32],
        seed_files={32: seed_path},
        aggregated_action_stats={
            col: {"mean_of_means": 0.0, "mean_of_stds": 0.3, "min_overall": -1.0, "max_overall": 1.0}
            for col in S.ACTION_COLUMNS
        },
        aggregated_reward_stats={"mean_of_means": -1.0, "mean_of_stds": 0.5},
        n_steps_total=8760,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["behaviour_policy"] == "iql"
    assert manifest["building"] == S.TARGET_BUILDING_NAME
    assert manifest["seeds"] == [32]
    assert manifest["n_steps_total"] == 8760
