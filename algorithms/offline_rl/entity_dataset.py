"""Dataset loader for the entity-interface offline-RL dataset.

Each call to ``EntityOfflineDataset`` loads all parquet files from a directory,
filters rows to a single agent group (identified by ``obs_dim`` and
``action_dim``), and wraps them in a ``torch.utils.data.Dataset`` suitable for
IQL / CQL training.

Observation standardisation
----------------------------
A per-feature ``ObservationStandardiser`` is fit on the *training* split only
and applied to both splits.  This prevents test-set leakage.

Train / val split
-----------------
Split is by *seed*, not by row.  One or more seeds are held out as the
validation set — this ensures the val agent never saw a training seed's
environment draw.

Usage example
-------------
::

    from algorithms.offline_rl.entity_dataset import EntityOfflineDataset, load_entity_dataset

    train_ds, val_ds, spec = load_entity_dataset(
        data_dir=Path("datasets/offline_rl/rbcsmart_entity"),
        obs_dim=706,
        action_dim=2,
        val_seeds=[31],
    )
    obs, actions, rewards, next_obs, dones = train_ds[0]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from algorithms.offline_rl.bc_dataset import ObservationStandardiser
from algorithms.offline_rl.entity_schema import (
    ACTION_PREFIX,
    NEXT_OBS_PREFIX,
    OBS_PREFIX,
    AgentGroupSpec,
    action_name_from_col,
    obs_name_from_col,
)

try:
    import pyarrow.parquet as pq
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError("pyarrow and pandas are required for EntityOfflineDataset.") from e


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _discover_group_columns(
    df,
    *,
    obs_dim: int,
    action_dim: int,
) -> Tuple[List[str], List[str]]:
    """Return (obs_cols, action_cols) for a given group.

    Takes the first row belonging to this group and finds all non-NaN obs
    and action columns — those are exactly the columns for this group.
    """
    mask = (df["obs_dim"] == obs_dim) & (df["action_dim"] == action_dim)
    if not mask.any():
        raise ValueError(
            f"No rows found for group (obs_dim={obs_dim}, action_dim={action_dim}). "
            "Check your data_dir and group parameters."
        )
    sample = df[mask].iloc[0]

    all_obs = sorted(c for c in df.columns if c.startswith(OBS_PREFIX))
    all_act = sorted(c for c in df.columns if c.startswith(ACTION_PREFIX))

    obs_cols = [c for c in all_obs if not pd.isna(sample[c])]
    act_cols = [c for c in all_act if not pd.isna(sample[c])]

    if len(obs_cols) != obs_dim:
        raise ValueError(
            f"Expected {obs_dim} obs columns for group ({obs_dim},{action_dim}), "
            f"got {len(obs_cols)}."
        )
    if len(act_cols) != action_dim:
        raise ValueError(
            f"Expected {action_dim} action columns for group ({obs_dim},{action_dim}), "
            f"got {len(act_cols)}."
        )
    return obs_cols, act_cols


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------


class EntityOfflineDataset(Dataset):
    """A ``Dataset`` over transitions for a single agent group.

    Parameters
    ----------
    obs:
        Float32 tensor ``(N, obs_dim)``.
    actions:
        Float32 tensor ``(N, action_dim)``.
    rewards:
        Float32 tensor ``(N,)``.
    next_obs:
        Float32 tensor ``(N, obs_dim)``.
    dones:
        Float32 tensor ``(N,)``.  1.0 if terminal OR truncated.
    """

    def __init__(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        super().__init__()
        n = obs.shape[0]
        assert actions.shape[0] == n
        assert rewards.shape[0] == n
        assert next_obs.shape[0] == n
        assert dones.shape[0] == n
        self._obs = obs
        self._actions = actions
        self._rewards = rewards
        self._next_obs = next_obs
        self._dones = dones

    def __len__(self) -> int:
        return self._obs.shape[0]

    def __getitem__(self, idx: int):
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_obs[idx],
            self._dones[idx],
        )

    @property
    def obs_dim(self) -> int:
        return int(self._obs.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self._actions.shape[1])


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_entity_dataset(
    data_dir: Path,
    *,
    obs_dim: int,
    action_dim: int,
    val_seeds: Optional[Sequence[int]] = None,
    reward_scale: float = 1.0,
    standardise: bool = True,
    device: str = "cpu",
) -> Tuple[EntityOfflineDataset, EntityOfflineDataset, AgentGroupSpec, ObservationStandardiser]:
    """Load parquet files and return (train_dataset, val_dataset, spec, standardiser).

    Parameters
    ----------
    data_dir:
        Directory containing ``seed_N.parquet`` files.
    obs_dim, action_dim:
        Target agent group.
    val_seeds:
        Seeds to hold out as validation.  Default: last seed in directory.
    reward_scale:
        Multiply rewards by this factor (e.g. to normalise magnitude).
    standardise:
        If True, fit ``ObservationStandardiser`` on training obs and apply to both.
    device:
        Torch device for returned tensors.
    """
    data_dir = Path(data_dir)
    parquet_files = sorted(data_dir.glob("seed_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No seed_*.parquet files found in {data_dir}")

    # Load all seeds into one DataFrame, keeping only needed columns + meta
    dfs = []
    all_seeds = []
    for pf in parquet_files:
        seed = int(pf.stem.split("_")[1])
        all_seeds.append(seed)
        df_raw = pd.read_parquet(pf)
        mask = (df_raw["obs_dim"] == obs_dim) & (df_raw["action_dim"] == action_dim)
        dfs.append(df_raw[mask].copy())

    combined = pd.concat(dfs, ignore_index=True)
    if combined.empty:
        raise ValueError(
            f"No rows found for group (obs_dim={obs_dim}, action_dim={action_dim}) "
            f"in {data_dir}"
        )

    obs_cols, act_cols = _discover_group_columns(combined, obs_dim=obs_dim, action_dim=action_dim)
    next_obs_cols = [NEXT_OBS_PREFIX + c[len(OBS_PREFIX):] for c in obs_cols]
    missing_next = [c for c in next_obs_cols if c not in combined.columns]
    if missing_next:
        raise ValueError(f"Missing next_obs columns: {missing_next[:5]}...")

    obs_names = [obs_name_from_col(c) for c in obs_cols]
    action_names = [action_name_from_col(c) for c in act_cols]
    spec = AgentGroupSpec(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_names=obs_names,
        action_names=action_names,
    )

    # Train / val split by seed
    if val_seeds is None:
        val_seeds_list = [max(all_seeds)]
    else:
        val_seeds_list = list(val_seeds)

    train_mask = ~combined["seed"].isin(val_seeds_list)
    val_mask = combined["seed"].isin(val_seeds_list)

    if not train_mask.any():
        raise ValueError(f"Training split is empty. val_seeds={val_seeds_list}, all_seeds={all_seeds}")
    if not val_mask.any():
        raise ValueError(f"Validation split is empty. val_seeds={val_seeds_list}, all_seeds={all_seeds}")

    train_df = combined[train_mask].reset_index(drop=True)
    val_df = combined[val_mask].reset_index(drop=True)

    # Extract arrays — fill NaN with 0.
    # Entity observations are sparse: features absent in a given timestep
    # (e.g. EV charger when no EV is present) are stored as NaN in the wide
    # parquet.  Zero-filling is semantically correct: "feature absent → 0".
    # The same applies to EV-charger action columns (NaN → no-charge = 0).
    def _to_array(df, cols):
        return df[cols].fillna(0.0).to_numpy(dtype=np.float32)

    train_obs = _to_array(train_df, obs_cols)
    train_acts = _to_array(train_df, act_cols)
    train_rewards = train_df["reward"].to_numpy(dtype=np.float32) * float(reward_scale)
    train_next_obs = _to_array(train_df, next_obs_cols)
    train_dones = np.maximum(
        train_df["terminated"].to_numpy(dtype=np.float32),
        train_df["truncated"].to_numpy(dtype=np.float32),
    )

    val_obs = _to_array(val_df, obs_cols)
    val_acts = _to_array(val_df, act_cols)
    val_rewards = val_df["reward"].to_numpy(dtype=np.float32) * float(reward_scale)
    val_next_obs = _to_array(val_df, next_obs_cols)
    val_dones = np.maximum(
        val_df["terminated"].to_numpy(dtype=np.float32),
        val_df["truncated"].to_numpy(dtype=np.float32),
    )

    # Observation standardisation (fit on train only)
    standardiser = ObservationStandardiser.fit(
        np.zeros((1, obs_dim), dtype=np.float32),
        feature_names=obs_names,
    )
    if standardise:
        standardiser = ObservationStandardiser.fit(train_obs, feature_names=obs_names)
        train_obs = standardiser.transform(train_obs)
        val_obs = standardiser.transform(val_obs)
        train_next_obs = standardiser.transform(train_next_obs)
        val_next_obs = standardiser.transform(val_next_obs)

    def _mk(arr) -> torch.Tensor:
        return torch.from_numpy(arr).to(device)

    train_ds = EntityOfflineDataset(
        obs=_mk(train_obs),
        actions=_mk(train_acts),
        rewards=_mk(train_rewards),
        next_obs=_mk(train_next_obs),
        dones=_mk(train_dones),
    )
    val_ds = EntityOfflineDataset(
        obs=_mk(val_obs),
        actions=_mk(val_acts),
        rewards=_mk(val_rewards),
        next_obs=_mk(val_next_obs),
        dones=_mk(val_dones),
    )
    return train_ds, val_ds, spec, standardiser
