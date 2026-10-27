"""Tests for entity_schema.py and entity_dataset.py."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from algorithms.offline_rl.entity_schema import (
    ACTION_PREFIX,
    META_COLUMNS,
    NEXT_OBS_PREFIX,
    OBS_PREFIX,
    AgentGroupSpec,
    AGENT_GROUPS,
    action_columns,
    action_name_from_col,
    next_obs_col,
    obs_columns,
    obs_name_from_col,
)
from algorithms.offline_rl.entity_dataset import (
    EntityOfflineDataset,
    _discover_group_columns,
    load_entity_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(
    *,
    n_steps: int = 20,
    seed: int = 1,
    episode: int = 0,
    groups: List[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """Build a minimal wide-format DataFrame like collect_rbcsmart_dataset produces."""
    if groups is None:
        groups = [(4, 1), (6, 2)]

    rng = np.random.default_rng(seed * 1000 + episode)
    rows = []
    for step in range(n_steps):
        for agent_idx, (obs_d, act_d) in enumerate(groups):
            obs_names = [f"obs_feat_{i}" for i in range(obs_d)]
            act_names = [f"act_{i}" for i in range(act_d)]

            row = {
                "seed": seed,
                "episode": episode,
                "timestep": step,
                "agent_idx": agent_idx,
                "obs_dim": obs_d,
                "action_dim": act_d,
                "reward": float(rng.normal(-0.02, 0.01)),
                "terminated": 0,
                "truncated": int(step == n_steps - 1),
            }
            # Obs columns for this group
            obs_vals = rng.uniform(-1.0, 1.0, size=obs_d)
            for name, val in zip(obs_names, obs_vals):
                row[f"{OBS_PREFIX}{name}"] = float(val)
            # Action columns for this group
            act_vals = rng.uniform(0.0, 1.0, size=act_d)
            for name, val in zip(act_names, act_vals):
                row[f"{ACTION_PREFIX}{name}"] = float(val)
            # Next-obs columns
            next_obs_vals = rng.uniform(-1.0, 1.0, size=obs_d)
            for name, val in zip(obs_names, next_obs_vals):
                row[f"{NEXT_OBS_PREFIX}{name}"] = float(val)
            rows.append(row)

    df = pd.DataFrame(rows)
    # Fill NaN for columns belonging to other groups
    df = df.fillna(value=float("nan"))
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


# ---------------------------------------------------------------------------
# entity_schema tests
# ---------------------------------------------------------------------------


def test_obs_col_prefix_constant():
    assert OBS_PREFIX == "obs__"


def test_action_col_prefix_constant():
    assert ACTION_PREFIX == "action__"


def test_next_obs_col_prefix_constant():
    assert NEXT_OBS_PREFIX == "next_obs__"


def test_meta_columns_tuple():
    assert "seed" in META_COLUMNS
    assert "reward" in META_COLUMNS
    assert "agent_idx" in META_COLUMNS


def test_agent_groups_known():
    assert (627, 1) in AGENT_GROUPS
    assert (706, 2) in AGENT_GROUPS
    assert (749, 3) in AGENT_GROUPS
    assert (785, 3) in AGENT_GROUPS


def test_obs_columns_filter():
    cols = ["obs__a", "action__b", "next_obs__c", "seed", "obs__d"]
    result = obs_columns(cols)
    assert result == ["obs__a", "obs__d"]


def test_action_columns_filter():
    cols = ["obs__a", "action__b", "action__c", "next_obs__x"]
    result = action_columns(cols)
    assert result == ["action__b", "action__c"]


def test_obs_name_from_col_strips_prefix():
    assert obs_name_from_col("obs__district__hour") == "district__hour"


def test_action_name_from_col_strips_prefix():
    assert action_name_from_col("action__electrical_storage") == "electrical_storage"


def test_next_obs_col_converts_prefix():
    assert next_obs_col("obs__district__hour") == "next_obs__district__hour"


def test_agent_group_spec_key():
    spec = AgentGroupSpec(obs_dim=706, action_dim=2, obs_names=[], action_names=[])
    assert spec.group_key == "obs706_act2"


def test_agent_group_spec_immutable():
    spec = AgentGroupSpec(obs_dim=627, action_dim=1)
    with pytest.raises(Exception):
        spec.obs_dim = 999


# ---------------------------------------------------------------------------
# entity_dataset tests
# ---------------------------------------------------------------------------


def test_discover_group_columns_correct():
    df = _make_synthetic_df(groups=[(4, 1), (6, 2)])
    obs_cols, act_cols = _discover_group_columns(df, obs_dim=4, action_dim=1)
    assert len(obs_cols) == 4
    assert len(act_cols) == 1
    assert all(c.startswith(OBS_PREFIX) for c in obs_cols)
    assert all(c.startswith(ACTION_PREFIX) for c in act_cols)


def test_discover_group_columns_second_group():
    df = _make_synthetic_df(groups=[(4, 1), (6, 2)])
    obs_cols, act_cols = _discover_group_columns(df, obs_dim=6, action_dim=2)
    assert len(obs_cols) == 6
    assert len(act_cols) == 2


def test_discover_group_columns_raises_on_missing_group():
    df = _make_synthetic_df(groups=[(4, 1)])
    with pytest.raises(ValueError, match="No rows found"):
        _discover_group_columns(df, obs_dim=999, action_dim=1)


def test_entity_offline_dataset_len():
    n = 50
    obs_d, act_d = 4, 1
    obs = torch.randn(n, obs_d)
    acts = torch.randn(n, act_d)
    rewards = torch.randn(n)
    next_obs = torch.randn(n, obs_d)
    dones = torch.zeros(n)
    ds = EntityOfflineDataset(obs, acts, rewards, next_obs, dones)
    assert len(ds) == n


def test_entity_offline_dataset_getitem_shapes():
    n, obs_d, act_d = 30, 6, 2
    obs = torch.randn(n, obs_d)
    acts = torch.randn(n, act_d)
    rewards = torch.randn(n)
    next_obs = torch.randn(n, obs_d)
    dones = torch.zeros(n)
    ds = EntityOfflineDataset(obs, acts, rewards, next_obs, dones)
    o, a, r, no, d = ds[0]
    assert o.shape == (obs_d,)
    assert a.shape == (act_d,)
    assert r.shape == ()
    assert no.shape == (obs_d,)
    assert d.shape == ()


def test_entity_offline_dataset_dim_properties():
    obs_d, act_d = 4, 2
    ds = EntityOfflineDataset(
        obs=torch.randn(10, obs_d),
        actions=torch.randn(10, act_d),
        rewards=torch.randn(10),
        next_obs=torch.randn(10, obs_d),
        dones=torch.zeros(10),
    )
    assert ds.obs_dim == obs_d
    assert ds.action_dim == act_d


def test_load_entity_dataset_returns_correct_shapes():
    groups = [(4, 1), (6, 2)]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Two seeds (1 and 2), val = seed 2
        for s in [1, 2]:
            df = _make_synthetic_df(seed=s, n_steps=20, groups=groups)
            _write_parquet(df, tmp / f"seed_{s}.parquet")

        train_ds, val_ds, spec, standardiser = load_entity_dataset(
            tmp, obs_dim=4, action_dim=1, val_seeds=[2]
        )

    # seed 1 has 20 steps, 1 agent in group (4,1) → 20 train rows
    assert len(train_ds) == 20
    # seed 2 has 20 steps, 1 agent in group (4,1) → 20 val rows
    assert len(val_ds) == 20
    assert train_ds.obs_dim == 4
    assert train_ds.action_dim == 1
    assert spec.obs_dim == 4
    assert spec.action_dim == 1
    assert len(spec.obs_names) == 4
    assert len(spec.action_names) == 1


def test_load_entity_dataset_train_val_split_by_seed():
    groups = [(4, 1)]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for s in [10, 20, 30]:
            df = _make_synthetic_df(seed=s, n_steps=10, groups=groups)
            _write_parquet(df, tmp / f"seed_{s}.parquet")

        train_ds, val_ds, _, _ = load_entity_dataset(
            tmp, obs_dim=4, action_dim=1, val_seeds=[30]
        )
    # 2 seeds in train (20 rows each), 1 in val (10 rows)
    assert len(train_ds) == 20
    assert len(val_ds) == 10


def test_load_entity_dataset_standardiser_fit_on_train():
    """Standardiser mean should match training obs mean, not combined mean."""
    groups = [(4, 1)]
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Seed 1 obs mean ≈ 0.5; seed 2 obs mean ≈ -0.5
        for s, lo, hi in [(1, 0.0, 1.0), (2, -1.0, 0.0)]:
            rows = []
            for step in range(20):
                obs_vals = rng.uniform(lo, hi, size=4)
                row = {
                    "seed": s, "episode": 0, "timestep": step, "agent_idx": 0,
                    "obs_dim": 4, "action_dim": 1,
                    "reward": -0.01, "terminated": 0, "truncated": int(step == 19),
                }
                for i, v in enumerate(obs_vals):
                    row[f"{OBS_PREFIX}obs_feat_{i}"] = float(v)
                    row[f"{NEXT_OBS_PREFIX}obs_feat_{i}"] = float(v)
                row[f"{ACTION_PREFIX}act_0"] = float(rng.uniform(0, 1))
                rows.append(row)
            df = pd.DataFrame(rows)
            _write_parquet(df, tmp / f"seed_{s}.parquet")

        _, _, _, std = load_entity_dataset(
            tmp, obs_dim=4, action_dim=1, val_seeds=[2], standardise=True
        )
    # Standardiser should be fit on seed 1 only (training split)
    assert std.mean[0] == pytest.approx(0.5, abs=0.15)


def test_load_entity_dataset_no_files_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="No seed_"):
            load_entity_dataset(Path(tmpdir), obs_dim=4, action_dim=1)


def test_load_entity_dataset_wrong_group_raises():
    groups = [(4, 1)]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        df = _make_synthetic_df(seed=1, groups=groups, n_steps=5)
        _write_parquet(df, tmp / "seed_1.parquet")
        with pytest.raises(ValueError):
            load_entity_dataset(tmp, obs_dim=999, action_dim=1)


def test_load_entity_dataset_reward_scale():
    groups = [(4, 1)]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for s in [1, 2]:
            df = _make_synthetic_df(seed=s, n_steps=10, groups=groups)
            _write_parquet(df, tmp / f"seed_{s}.parquet")

        train_no_scale, _, _, _ = load_entity_dataset(
            tmp, obs_dim=4, action_dim=1, val_seeds=[2], reward_scale=1.0, standardise=False
        )
        train_scaled, _, _, _ = load_entity_dataset(
            tmp, obs_dim=4, action_dim=1, val_seeds=[2], reward_scale=10.0, standardise=False
        )

    # Scaled rewards should be 10x unscaled
    _, _, r_base, _, _ = train_no_scale[0]
    _, _, r_scaled, _, _ = train_scaled[0]
    assert r_scaled.item() == pytest.approx(r_base.item() * 10.0, rel=1e-5)
