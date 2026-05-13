"""BC dataset utilities.

Reads ``datasets/offline_rl/derived/rbc_with_reward.parquet`` (or any
parquet matching the offline-RL schema), extracts the 35-D obs and 2-D action vectors
for Building 5, performs a per-seed 90/10 row-level train/val split, and
fits an observation standardiser **on the training subset only**.

Targets are the raw RBC actions in their native ``[0, 1]``-ish range. The
policy's tanh head outputs in ``[-1, 1]`` and the env clips actions to its
own bounds, so no action standardisation is needed.

Standalone — does not import from ``algorithms/offline/``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from algorithms.offline_rl import schema as S


# ---------------------------------------------------------------------------
# Standardiser
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ObservationStandardiser:
    """Per-feature mean/std normaliser fitted on training observations.

    ``std`` is floored at ``eps`` to prevent divide-by-zero for constant
    features (e.g. ``electric_vehicle_charger_state`` for buildings without
    chargers — not the case for B5, but defensive).
    """

    mean: np.ndarray
    std: np.ndarray
    feature_names: List[str]
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(self.std, dtype=np.float32).reshape(-1)
        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"mean/std shape mismatch: {self.mean.shape} vs {self.std.shape}"
            )
        if len(self.feature_names) != self.mean.shape[0]:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} != mean dim {self.mean.shape[0]}"
            )
        # Floor std.
        self.std = np.maximum(self.std, self.eps).astype(np.float32)

    @classmethod
    def fit(
        cls, obs: np.ndarray, feature_names: Sequence[str], *, eps: float = 1e-6
    ) -> "ObservationStandardiser":
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim != 2:
            raise ValueError(f"obs must be 2-D, got shape {obs.shape}")
        if obs.shape[1] != len(feature_names):
            raise ValueError(
                f"obs has {obs.shape[1]} features, names give {len(feature_names)}"
            )
        return cls(
            mean=obs.mean(axis=0).astype(np.float32),
            std=obs.std(axis=0).astype(np.float32),
            feature_names=list(feature_names),
            eps=float(eps),
        )

    def transform(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            return ((obs - self.mean) / self.std).astype(np.float32)
        if obs.ndim == 2:
            return ((obs - self.mean[None, :]) / self.std[None, :]).astype(np.float32)
        raise ValueError(f"obs must be 1-D or 2-D, got shape {obs.shape}")

    # --- I/O ---------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            feature_names=np.asarray(self.feature_names),
            eps=np.asarray([self.eps], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> "ObservationStandardiser":
        data = np.load(path, allow_pickle=False)
        return cls(
            mean=data["mean"],
            std=data["std"],
            feature_names=[str(n) for n in data["feature_names"]],
            eps=float(data["eps"][0]),
        )


# ---------------------------------------------------------------------------
# Parquet → numpy
# ---------------------------------------------------------------------------


def load_obs_action_arrays(
    parquet_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load (obs, action) arrays from a offline-RL schema parquet.

    Returns
    -------
    obs : (N, 35) float32
    action : (N, 2) float32
    obs_feature_names : list of 35 names (from schema.OBSERVATION_NAMES)
    action_feature_names : list of 2 names (from schema.ACTION_NAMES)
    """
    df = pd.read_parquet(parquet_path)
    missing_obs = [c for c in S.OBS_COLUMNS if c not in df.columns]
    missing_act = [c for c in S.ACTION_COLUMNS if c not in df.columns]
    if missing_obs or missing_act:
        raise ValueError(
            f"parquet missing schema columns:\n  obs: {missing_obs}\n  act: {missing_act}"
        )
    obs = df[list(S.OBS_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    action = df[list(S.ACTION_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    if not np.all(np.isfinite(obs)):
        raise ValueError("non-finite values in obs columns")
    if not np.all(np.isfinite(action)):
        raise ValueError("non-finite values in action columns")
    return obs, action, list(S.OBSERVATION_NAMES), list(S.ACTION_NAMES)


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------


class BCDataset(Dataset):
    """Torch dataset over already-standardised obs and raw actions.

    Holds tensors directly (no on-the-fly conversion). Intended for in-memory
    training; the dataset (~88k rows × 70 floats × 4 bytes ≈ 25 MB) easily
    fits.
    """

    def __init__(self, obs_std: np.ndarray, actions: np.ndarray) -> None:
        if obs_std.shape[0] != actions.shape[0]:
            raise ValueError(
                f"obs/action row count mismatch: {obs_std.shape[0]} vs {actions.shape[0]}"
            )
        self._obs = torch.from_numpy(np.ascontiguousarray(obs_std, dtype=np.float32))
        self._act = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))

    def __len__(self) -> int:
        return int(self._obs.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._obs[idx], self._act[idx]


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BCDatasetSplit:
    """Result of a 90/10 (or arbitrary fraction) row-level split."""

    train: BCDataset
    val: BCDataset
    standardiser: ObservationStandardiser
    train_indices: np.ndarray  # (n_train,)
    val_indices: np.ndarray    # (n_val,)
    obs_feature_names: List[str]
    action_feature_names: List[str]


def make_split(
    obs: np.ndarray,
    actions: np.ndarray,
    obs_feature_names: Sequence[str],
    action_feature_names: Sequence[str],
    *,
    val_fraction: float = 0.1,
    seed: int,
    eps: float = 1e-6,
) -> BCDatasetSplit:
    """Random row-level split. Standardiser is fit on the *train* subset only.

    Determinism: with a given ``seed``, ``train_indices`` and ``val_indices``
    are reproducible.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n = obs.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 rows to split, got {n}")
    if actions.shape[0] != n:
        raise ValueError(
            f"obs/action row count mismatch: {n} vs {actions.shape[0]}"
        )

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, n - 1)  # leave at least one row for train
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    train_obs_raw = obs[train_idx]
    val_obs_raw = obs[val_idx]
    train_act = actions[train_idx]
    val_act = actions[val_idx]

    standardiser = ObservationStandardiser.fit(
        train_obs_raw, obs_feature_names, eps=eps
    )
    train_obs_std = standardiser.transform(train_obs_raw)
    val_obs_std = standardiser.transform(val_obs_raw)

    return BCDatasetSplit(
        train=BCDataset(train_obs_std, train_act),
        val=BCDataset(val_obs_std, val_act),
        standardiser=standardiser,
        train_indices=train_idx,
        val_indices=val_idx,
        obs_feature_names=list(obs_feature_names),
        action_feature_names=list(action_feature_names),
    )


# ---------------------------------------------------------------------------
# Convenience: load + split in one call
# ---------------------------------------------------------------------------


def load_and_split(
    parquet_path: Path,
    *,
    val_fraction: float = 0.1,
    seed: int,
) -> BCDatasetSplit:
    obs, action, obs_names, act_names = load_obs_action_arrays(parquet_path)
    return make_split(
        obs,
        action,
        obs_names,
        act_names,
        val_fraction=val_fraction,
        seed=seed,
    )
