"""Tests for scripts/apply_reward.py.

Written before implementation (TDD / RED phase).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algorithms.offline_rl import reward as RW
from algorithms.offline_rl import schema as S

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_WEIGHTS_PATH = REPO_ROOT / "datasets" / "offline_rl" / "derived" / "reward_weights.json"
RBC_PARQUET = REPO_ROOT / "datasets" / "offline_rl" / "derived" / "rbc_with_reward.parquet"


# ---------------------------------------------------------------------------
# Helpers: build a minimal valid parquet fixture from the RBC dataset
# ---------------------------------------------------------------------------


def _make_minimal_parquet(tmp_path: Path, n_rows: int = 50) -> Path:
    """Write a tiny seed_32.parquet with all schema columns (reward=NaN)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    if RBC_PARQUET.exists():
        df = pd.read_parquet(RBC_PARQUET).head(n_rows).copy()
        df["reward"] = np.nan
        df["seed"] = 32
        df["timestep"] = np.arange(n_rows)
        p = tmp_path / "seed_32.parquet"
        df.to_parquet(p, index=False)
        return p
    # Minimal synthetic fixture — all numeric zeros, with valid column names.
    cols = S.all_columns()
    data = {c: np.zeros(n_rows, dtype=np.float64) for c in cols}
    data["episode"] = np.zeros(n_rows, dtype=np.int64)
    data["timestep"] = np.arange(n_rows, dtype=np.int64)
    data["seed"] = np.full(n_rows, 32, dtype=np.int64)
    data["policy_mode"] = ["behaviour"] * n_rows
    data["reward_env"] = np.zeros(n_rows)
    data["reward"] = np.full(n_rows, np.nan)
    data["terminated"] = np.zeros(n_rows, dtype=np.int64)
    data["truncated"] = np.zeros(n_rows, dtype=np.int64)
    df = pd.DataFrame(data)
    p = tmp_path / "seed_32.parquet"
    df.to_parquet(p, index=False)
    return p


def _make_weights_json(tmp_path: Path, weights: dict | None = None) -> Path:
    if weights is None:
        weights = dict(RW.DEFAULT_WEIGHTS)
    data = {
        "weights": weights,
        "term_names": list(RW.TERM_NAMES),
        "peak_window_hours": RW.PEAK_WINDOW_HOURS,
    }
    p = tmp_path / "reward_weights.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# 1. apply_reward adds finite reward column to output parquet
# ---------------------------------------------------------------------------


def test_apply_reward_adds_finite_reward_column(tmp_path: Path) -> None:
    from scripts.apply_reward import apply_reward

    parquet_path = _make_minimal_parquet(tmp_path / "in")
    weights_path = _make_weights_json(tmp_path)
    out_path = tmp_path / "out.parquet"

    apply_reward(
        input_paths=[parquet_path],
        weights_path=weights_path,
        output_path=out_path,
    )

    assert out_path.exists(), "output parquet not written"
    df = pd.read_parquet(out_path)
    assert "reward" in df.columns
    assert np.all(np.isfinite(df["reward"].to_numpy())), "reward column contains non-finite values"


# ---------------------------------------------------------------------------
# 2. Reward values match direct compute_reward_vectorised call
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RBC_PARQUET.exists(), reason="RBC parquet not present")
def test_apply_reward_matches_compute_reward_vectorised(tmp_path: Path) -> None:
    from scripts.apply_reward import apply_reward

    parquet_path = _make_minimal_parquet(tmp_path / "in")
    weights_path = _make_weights_json(tmp_path)
    out_path = tmp_path / "out.parquet"

    apply_reward(
        input_paths=[parquet_path],
        weights_path=weights_path,
        output_path=out_path,
    )

    df_in = pd.read_parquet(parquet_path)
    df_out = pd.read_parquet(out_path)
    weights = RW.load_weights(weights_path)
    expected_reward, _ = RW.compute_reward_vectorised(df_in, weights=weights)

    np.testing.assert_allclose(
        df_out["reward"].to_numpy(),
        expected_reward,
        rtol=1e-6,
        err_msg="apply_reward reward != compute_reward_vectorised",
    )


# ---------------------------------------------------------------------------
# 3. apply_reward loads frozen weights from JSON (no re-fitting)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FROZEN_WEIGHTS_PATH.exists(), reason="frozen weights file not present")
def test_apply_reward_loads_frozen_weights(tmp_path: Path) -> None:
    """Calling apply_reward with the frozen weights path produces finite output."""
    from scripts.apply_reward import apply_reward

    parquet_path = _make_minimal_parquet(tmp_path / "in")
    out_path = tmp_path / "out.parquet"

    apply_reward(
        input_paths=[parquet_path],
        weights_path=FROZEN_WEIGHTS_PATH,
        output_path=out_path,
    )

    df = pd.read_parquet(out_path)
    assert np.all(np.isfinite(df["reward"].to_numpy()))


# ---------------------------------------------------------------------------
# 4. CLI smoke: python -m scripts.apply_reward runs without error
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RBC_PARQUET.exists(), reason="RBC parquet not present")
def test_apply_reward_cli_smoke(tmp_path: Path) -> None:
    from scripts.apply_reward import main

    parquet_path = _make_minimal_parquet(tmp_path / "in")
    weights_path = _make_weights_json(tmp_path)
    out_path = tmp_path / "out.parquet"

    rc = main([
        "--input", str(parquet_path),
        "--weights", str(weights_path),
        "--output", str(out_path),
    ])

    assert rc == 0
    assert out_path.exists()


# ---------------------------------------------------------------------------
# 5. Multiple input parquets are concatenated correctly
# ---------------------------------------------------------------------------


def test_apply_reward_concatenates_multiple_inputs(tmp_path: Path) -> None:
    from scripts.apply_reward import apply_reward

    p1 = _make_minimal_parquet(tmp_path / "in1", n_rows=20)
    # Second parquet with different seed
    in2_dir = tmp_path / "in2"
    in2_dir.mkdir(parents=True, exist_ok=True)
    df2 = pd.read_parquet(p1).copy()
    df2["seed"] = 33
    p2 = in2_dir / "seed_33.parquet"
    df2.to_parquet(p2, index=False)

    weights_path = _make_weights_json(tmp_path)
    out_path = tmp_path / "combined.parquet"

    apply_reward(
        input_paths=[p1, p2],
        weights_path=weights_path,
        output_path=out_path,
    )

    df_out = pd.read_parquet(out_path)
    assert len(df_out) == 40, f"Expected 40 rows (20+20), got {len(df_out)}"
    assert np.all(np.isfinite(df_out["reward"].to_numpy()))
