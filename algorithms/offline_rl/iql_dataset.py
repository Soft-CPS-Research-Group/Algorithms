"""IQL dataset: in-memory (s, a, r, s', done) tensors.

Reads ``datasets/offline_rl/derived/rbc_with_reward.parquet`` (or any parquet
matching the offline-RL schema), extracts the 35-D obs / next_obs, 2-D action,
scalar reward, and ``terminated`` flag for Building 5. Performs a row-level
90/10 train/val split and fits an ``ObservationStandardiser`` on the train
slice only — applied to **both** ``s`` and ``s'``.

Asserts ``truncated`` is identically zero across the parquet (this dataset
runs full annual episodes; truncation isn't used). Fail-fast at load time so
trainer code can ignore the column.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from algorithms.offline_rl import schema as S
from algorithms.offline_rl.bc_dataset import ObservationStandardiser


# ---------------------------------------------------------------------------
# Parquet → numpy
# ---------------------------------------------------------------------------


def load_iql_arrays(
    parquet_path: Path,
) -> Tuple[
    np.ndarray,  # obs (N, 35)
    np.ndarray,  # action (N, 2)
    np.ndarray,  # reward (N,)
    np.ndarray,  # next_obs (N, 35)
    np.ndarray,  # done (N,) float32 in {0, 1}
    List[str],   # obs feature names
    List[str],   # action feature names
]:
    df = pd.read_parquet(parquet_path)

    missing = [
        c for c in (
            list(S.OBS_COLUMNS)
            + list(S.NEXT_OBS_COLUMNS)
            + list(S.ACTION_COLUMNS)
            + ["reward", "terminated", "truncated"]
        )
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"parquet missing schema columns: {missing}")

    truncated = df["truncated"].to_numpy()
    if int(np.asarray(truncated, dtype=np.int64).sum()) != 0:
        raise ValueError(
            "IQL dataset expects truncated ≡ 0 across rows; got non-zero values"
        )

    obs = df[list(S.OBS_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    next_obs = df[list(S.NEXT_OBS_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    action = df[list(S.ACTION_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    reward = df["reward"].to_numpy(dtype=np.float32, copy=True)
    done = df["terminated"].to_numpy(dtype=np.float32, copy=True)

    for arr, name in (
        (obs, "obs"),
        (next_obs, "next_obs"),
        (action, "action"),
        (reward, "reward"),
        (done, "terminated"),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"non-finite values in {name} column(s)")

    return (
        obs,
        action,
        reward,
        next_obs,
        done,
        list(S.OBSERVATION_NAMES),
        list(S.ACTION_NAMES),
    )


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------


class IQLDataset(Dataset):
    """Holds (s, a, r, s', done) tensors in memory.

    All tensors are ``float32``. The dataset (~88k rows × ~73 floats × 4 B
    ≈ 26 MB) easily fits in RAM.
    """

    def __init__(
        self,
        obs_std: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs_std: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        n = obs_std.shape[0]
        for arr, name in (
            (actions, "actions"),
            (rewards, "rewards"),
            (next_obs_std, "next_obs_std"),
            (dones, "dones"),
        ):
            if arr.shape[0] != n:
                raise ValueError(
                    f"{name} row count {arr.shape[0]} != obs row count {n}"
                )
        self._obs = torch.from_numpy(np.ascontiguousarray(obs_std, dtype=np.float32))
        self._act = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))
        self._rew = torch.from_numpy(np.ascontiguousarray(rewards, dtype=np.float32))
        self._next_obs = torch.from_numpy(
            np.ascontiguousarray(next_obs_std, dtype=np.float32)
        )
        self._done = torch.from_numpy(np.ascontiguousarray(dones, dtype=np.float32))

    def __len__(self) -> int:
        return int(self._obs.shape[0])

    def __getitem__(self, idx: int):
        return (
            self._obs[idx],
            self._act[idx],
            self._rew[idx],
            self._next_obs[idx],
            self._done[idx],
        )

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform random batch sample (with replacement)."""
        n = self._obs.shape[0]
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        idx = torch.randint(0, n, (int(batch_size),), generator=generator)
        return (
            self._obs[idx],
            self._act[idx],
            self._rew[idx],
            self._next_obs[idx],
            self._done[idx],
        )


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IQLSplit:
    train: IQLDataset
    val: IQLDataset
    standardiser: ObservationStandardiser
    train_indices: np.ndarray
    val_indices: np.ndarray
    obs_feature_names: List[str]
    action_feature_names: List[str]


def make_iql_split(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    obs_feature_names: Sequence[str],
    action_feature_names: Sequence[str],
    *,
    val_fraction: float = 0.1,
    seed: int,
    eps: float = 1e-6,
) -> IQLSplit:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n = obs.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 rows to split, got {n}")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, n - 1)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    standardiser = ObservationStandardiser.fit(
        obs[train_idx], obs_feature_names, eps=eps
    )
    train_obs = standardiser.transform(obs[train_idx])
    val_obs = standardiser.transform(obs[val_idx])
    train_next = standardiser.transform(next_obs[train_idx])
    val_next = standardiser.transform(next_obs[val_idx])

    return IQLSplit(
        train=IQLDataset(
            train_obs,
            actions[train_idx],
            rewards[train_idx],
            train_next,
            dones[train_idx],
        ),
        val=IQLDataset(
            val_obs,
            actions[val_idx],
            rewards[val_idx],
            val_next,
            dones[val_idx],
        ),
        standardiser=standardiser,
        train_indices=train_idx,
        val_indices=val_idx,
        obs_feature_names=list(obs_feature_names),
        action_feature_names=list(action_feature_names),
    )


def load_iql_split(
    parquet_path: Path,
    *,
    val_fraction: float = 0.1,
    seed: int,
) -> IQLSplit:
    obs, action, reward, next_obs, done, obs_names, act_names = load_iql_arrays(
        parquet_path
    )
    return make_iql_split(
        obs,
        action,
        reward,
        next_obs,
        done,
        obs_names,
        act_names,
        val_fraction=val_fraction,
        seed=seed,
    )
