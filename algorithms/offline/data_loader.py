"""CSV → PyTorch dataloader pipeline for the offline RL dataset.

Responsibilities:

* Load ``offline_dataset.csv`` produced by :class:`EVDataCollectionRBC`.
* Split rows by **episode** (no shuffling across episode boundaries) so the
  validation set tests true generalisation to a fresh year.
* Compute per-feature mean / std on the **training set only** and apply the
  same statistics to the validation set.
* Return PyTorch ``DataLoader`` objects + the standardisation statistics
  needed for inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class NormalizationStats:
    """Per-feature mean/std computed from the training subset.

    The same statistics are saved alongside the model so inference can apply
    the **exact** same transform to raw CityLearn observations.
    """

    feature_names: List[str]
    mean: List[float]
    std: List[float]
    action_names: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean,
            "std": self.std,
            "action_names": self.action_names,
        }


@dataclass
class DatasetBundle:
    """Everything :mod:`bc_trainer` needs after loading the CSV."""

    train_loader: DataLoader
    val_loader: DataLoader
    stats: NormalizationStats
    train_size: int
    val_size: int
    train_episodes: List[int]
    val_episodes: List[int]
    obs_dim: int
    action_dim: int


def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    obs_cols = [c for c in df.columns if c.startswith("obs_") and not c.startswith("obs_next")]
    action_cols = [c for c in df.columns if c.startswith("action_")]
    if not obs_cols:
        raise ValueError("Dataset has no obs_* columns.")
    if not action_cols:
        raise ValueError("Dataset has no action_* columns.")
    return obs_cols, action_cols


def _split_by_episode(
    df: pd.DataFrame, val_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """Hold out the **last** ``val_fraction`` of episodes for validation.

    Splitting by episode (rather than by row) prevents leakage — adjacent
    timesteps are nearly identical so a random row split would let the model
    memorise instead of generalise.
    """
    episodes = sorted(df["episode"].unique().tolist())
    n_val = max(1, int(round(len(episodes) * val_fraction)))
    val_episodes = episodes[-n_val:]
    train_episodes = [ep for ep in episodes if ep not in val_episodes]

    train_df = df[df["episode"].isin(train_episodes)].reset_index(drop=True)
    val_df = df[df["episode"].isin(val_episodes)].reset_index(drop=True)
    return train_df, val_df, train_episodes, val_episodes


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / safe_std


def load_dataset(
    csv_path: str | Path,
    *,
    batch_size: int = 256,
    val_fraction: float = 0.20,
    num_workers: int = 0,
    seed: int = 22,
) -> DatasetBundle:
    """Load the CSV, split by episode, standardise features, and build dataloaders."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Offline dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        raise ValueError("Dataset is missing the 'episode' column.")

    obs_cols, action_cols = _split_columns(df)
    train_df, val_df, train_eps, val_eps = _split_by_episode(df, val_fraction)

    train_obs = train_df[obs_cols].to_numpy(dtype=np.float32)
    train_act = train_df[action_cols].to_numpy(dtype=np.float32)
    val_obs = val_df[obs_cols].to_numpy(dtype=np.float32)
    val_act = val_df[action_cols].to_numpy(dtype=np.float32)

    # Standardisation stats computed from training subset ONLY.
    mean = train_obs.mean(axis=0)
    std = train_obs.std(axis=0)

    train_obs_norm = _standardize(train_obs, mean, std).astype(np.float32)
    val_obs_norm = _standardize(val_obs, mean, std).astype(np.float32)

    # PyTorch tensors and DataLoaders.
    generator = torch.Generator().manual_seed(seed)
    train_ds = TensorDataset(torch.from_numpy(train_obs_norm), torch.from_numpy(train_act))
    val_ds = TensorDataset(torch.from_numpy(val_obs_norm), torch.from_numpy(val_act))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,           # within-training shuffling is fine — episode boundary is preserved.
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    stats = NormalizationStats(
        feature_names=[c[len("obs_"):] for c in obs_cols],
        mean=mean.astype(float).tolist(),
        std=std.astype(float).tolist(),
        action_names=[c[len("action_"):] for c in action_cols],
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        stats=stats,
        train_size=len(train_ds),
        val_size=len(val_ds),
        train_episodes=train_eps,
        val_episodes=val_eps,
        obs_dim=len(obs_cols),
        action_dim=len(action_cols),
    )
