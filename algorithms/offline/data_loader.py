"""CSV → PyTorch dataloader pipeline for the offline RL dataset (M2/M3).

Responsibilities:

* Load an offline dataset CSV produced by the M2 ``DistrictDataCollectionRBC``
  (or the legacy M1 ``EVDataCollectionRBC`` — both column shapes are
  supported).
* Split rows by **episode**:
    - ``last:N``    — hold out the last ``N`` episodes (legacy M1 behaviour).
    - ``random:N``  — hold out ``N`` *clean* episodes selected deterministically
      from ``seed`` (M3 default; noisy episodes always stay in train).
* Compute per-feature mean / std on the **training set only** and apply the
  same statistics to the validation set.
* Choose the action target column family (``"action"`` for the executed
  action, ``"action_clean"`` for the RBC's intended action — recommended on
  M2 noisy episodes).
* Return PyTorch ``DataLoader`` objects + the standardisation statistics
  needed for inference.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    action_target: str
    policy_mode_summary: Dict[str, Dict[str, int]] = field(default_factory=dict)
    topology_versions: List[int] = field(default_factory=list)


def _split_columns(
    df: pd.DataFrame, action_target: str
) -> Tuple[List[str], List[str]]:
    """Identify obs and action columns.

    ``obs_*`` columns must NOT match ``next_obs_*``. The legacy M1 regex used
    a ``not c.startswith("obs_next")`` clause that never matched the CSV's
    actual ``next_obs_*`` prefix — we fix that here (M1 polish #9).
    """
    obs_cols = [
        c for c in df.columns
        if c.startswith("obs_") and not c.startswith("next_obs_")
    ]

    if action_target == "action":
        # Executed (post-clip, possibly noisy) action.
        action_cols = [
            c for c in df.columns
            if c.startswith("action_") and not c.startswith("action_clean_")
        ]
    elif action_target == "action_clean":
        action_cols = [c for c in df.columns if c.startswith("action_clean_")]
    else:
        raise ValueError(
            f"action_target must be 'action' or 'action_clean', got {action_target!r}"
        )

    if not obs_cols:
        raise ValueError("Dataset has no obs_* columns.")
    if not action_cols:
        raise ValueError(
            f"Dataset has no {action_target}_* columns. "
            "Hint: M1 datasets only have 'action_*'; pass action_target='action'."
        )
    return obs_cols, action_cols


def _select_val_episodes(
    df: pd.DataFrame,
    *,
    val_episodes_mode: str,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Pick which episode indices go to validation vs train.

    Modes:
      * ``last:N``   — last ``N`` episodes, no shuffling (M1 behaviour).
      * ``random:N`` — ``N`` random *clean* episodes (requires ``policy_mode``
        column with values ``clean``/``noisy``); deterministic from ``seed``.
    """
    episodes_all = sorted(int(e) for e in df["episode"].unique().tolist())

    if ":" not in val_episodes_mode:
        raise ValueError(
            f"val_episodes_mode must be 'last:N' or 'random:N', got {val_episodes_mode!r}"
        )
    mode, n_str = val_episodes_mode.split(":", 1)
    n_val = max(1, int(n_str))

    if mode == "last":
        val_eps = episodes_all[-n_val:]
        train_eps = [e for e in episodes_all if e not in val_eps]
        return train_eps, val_eps

    if mode == "random":
        if "policy_mode" not in df.columns:
            raise ValueError(
                "val_episodes_mode='random:N' requires the 'policy_mode' column "
                "(produced by DistrictDataCollectionRBC, M2+)."
            )
        clean_eps = sorted(
            int(e) for e in df.loc[df["policy_mode"] == "clean", "episode"].unique()
        )
        if len(clean_eps) < n_val:
            raise ValueError(
                f"Need >= {n_val} clean episodes for val, found {len(clean_eps)}."
            )
        rng = random.Random(seed)
        val_eps = sorted(rng.sample(clean_eps, n_val))
        train_eps = [e for e in episodes_all if e not in val_eps]
        return train_eps, val_eps

    raise ValueError(f"Unknown val_episodes_mode prefix: {mode!r}")


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / safe_std


def _summarize_policy_modes(df: pd.DataFrame, episodes: List[int]) -> Dict[str, int]:
    if "policy_mode" not in df.columns:
        return {}
    sub = df[df["episode"].isin(episodes)]
    return {k: int(v) for k, v in sub["policy_mode"].value_counts().to_dict().items()}


def load_dataset(
    csv_path: str | Path,
    *,
    batch_size: int = 256,
    val_fraction: Optional[float] = None,        # legacy; ignored when val_episodes_mode is set
    val_episodes_mode: str = "last:1",            # M1 default to preserve compatibility
    action_target: str = "action",                # M1 default; M3 callers pass "action_clean"
    num_workers: int = 0,
    seed: int = 22,
) -> DatasetBundle:
    """Load the CSV, split by episode, standardise features, and build dataloaders.

    Backwards-compatible with the M1 calling convention: if neither
    ``val_episodes_mode`` nor ``val_fraction`` is changed from the defaults,
    behaviour matches the original loader closely enough for legacy tests.
    Pass ``val_episodes_mode="random:2"`` and ``action_target="action_clean"``
    for the M3 BC training default.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Offline dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        raise ValueError("Dataset is missing the 'episode' column.")

    obs_cols, action_cols = _split_columns(df, action_target=action_target)

    # Backwards-compat shim: if caller still passes val_fraction, convert it to
    # a "last:N" split (M1 semantics).
    if val_fraction is not None and val_episodes_mode == "last:1":
        episodes_all = sorted(df["episode"].unique().tolist())
        n_val = max(1, int(round(len(episodes_all) * float(val_fraction))))
        val_episodes_mode = f"last:{n_val}"

    train_eps, val_eps = _select_val_episodes(
        df, val_episodes_mode=val_episodes_mode, seed=seed
    )

    train_df = df[df["episode"].isin(train_eps)].reset_index(drop=True)
    val_df = df[df["episode"].isin(val_eps)].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"Empty split: train_rows={len(train_df)}, val_rows={len(val_df)}; "
            f"train_eps={train_eps}, val_eps={val_eps}"
        )

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
        shuffle=True,
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

    # M3 expects feature/action names to use the action_clean prefix when
    # action_target == "action_clean" so inference can rebuild the column
    # mapping without ambiguity.
    if action_target == "action_clean":
        action_name_strip = len("action_clean_")
    else:
        action_name_strip = len("action_")

    stats = NormalizationStats(
        feature_names=[c[len("obs_"):] for c in obs_cols],
        mean=mean.astype(float).tolist(),
        std=std.astype(float).tolist(),
        action_names=[c[action_name_strip:] for c in action_cols],
    )

    policy_mode_summary = {
        "train": _summarize_policy_modes(df, train_eps),
        "val": _summarize_policy_modes(df, val_eps),
    }
    topology_versions = (
        sorted(int(v) for v in df["topology_version"].unique().tolist())
        if "topology_version" in df.columns
        else []
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
        action_target=action_target,
        policy_mode_summary=policy_mode_summary,
        topology_versions=topology_versions,
    )
