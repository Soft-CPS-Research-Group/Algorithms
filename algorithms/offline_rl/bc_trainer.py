"""BC trainer.

Single-seed training loop + multi-seed driver. MSE loss across both action
dims; per-dim MSE is also tracked (essential for the sanity story —
``action_electrical_storage`` is constant 0 and dominates aggregate MSE,
so we always need to inspect the EV-dim MSE separately).

Per-seed artefacts written under ``output_root/seed_<N>/``:

  * ``policy.pt``           — torch state_dict
  * ``obs_standardiser.npz``
  * ``metrics.jsonl``       — one line per epoch: train/val MSE + per-dim
  * ``seed_summary.json``   — final stats, best epoch, training time
  * ``architecture.json``   — net shape

Aggregated under ``output_root/``:

  * ``multi_seed_summary.json`` — per-seed best/final, mean ± std
  * ``seeds_index.json``        — seed → seed_dir map
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from algorithms.offline_rl import schema as S
from algorithms.offline_rl.bc_dataset import (
    BCDatasetSplit,
    load_and_split,
)
from algorithms.offline_rl.bc_policy import BCPolicy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BCTrainingConfig:
    hidden_layers: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    batch_size: int = 256
    epochs: int = 50
    val_fraction: float = 0.1
    device: str = "cpu"
    num_workers: int = 0  # in-memory dataset; workers add overhead
    log_every_n_epochs: int = 1


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _per_dim_mse(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    diff = (pred - target) ** 2
    return [float(diff[:, i].mean().item()) for i in range(diff.shape[1])]


def _evaluate(
    policy: BCPolicy,
    loader: DataLoader,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    policy.eval()
    losses: List[float] = []
    per_dim_sums: Optional[np.ndarray] = None
    n_total = 0
    with torch.no_grad():
        for obs, action in loader:
            obs = obs.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            pred = policy(obs)
            mse = torch.mean((pred - action) ** 2).item()
            losses.append(mse * obs.shape[0])
            per_dim = ((pred - action) ** 2).sum(dim=0).cpu().numpy()
            per_dim_sums = per_dim if per_dim_sums is None else per_dim_sums + per_dim
            n_total += obs.shape[0]
    if n_total == 0:
        return {"mse": float("nan"), "per_dim_mse": []}
    avg = float(sum(losses) / n_total)
    per_dim = (per_dim_sums / n_total).tolist() if per_dim_sums is not None else []
    return {"mse": avg, "per_dim_mse": [float(v) for v in per_dim]}


def train_single_seed(
    parquet_path: Path,
    output_dir: Path,
    *,
    seed: int,
    config: BCTrainingConfig,
) -> Dict[str, Any]:
    """Train one BC model for one seed. Returns summary dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determinism (within reason for CPU runs).
    torch.manual_seed(seed)
    np.random.seed(seed)

    split: BCDatasetSplit = load_and_split(
        parquet_path, val_fraction=config.val_fraction, seed=seed
    )
    obs_dim = split.standardiser.mean.shape[0]
    action_dim = len(split.action_feature_names)

    device = _resolve_device(config.device)
    policy = BCPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    ).to(device)
    optimiser = torch.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = DataLoader(
        split.train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        split.val,
        batch_size=max(config.batch_size, 1024),
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("")  # truncate

    best_val: float = float("inf")
    best_epoch: int = -1
    best_per_dim: List[float] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    t0 = time.time()
    last_train_summary: Dict[str, Any] = {}

    for epoch in range(config.epochs):
        policy.train()
        train_losses: List[float] = []
        train_per_dim_sums: Optional[np.ndarray] = None
        n_total = 0
        for obs, action in train_loader:
            obs = obs.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            pred = policy(obs)
            loss = nn.functional.mse_loss(pred, action)
            optimiser.zero_grad()
            loss.backward()
            if config.gradient_clip_norm and config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), config.gradient_clip_norm
                )
            optimiser.step()
            train_losses.append(float(loss.item()) * obs.shape[0])
            with torch.no_grad():
                pd_sum = ((pred - action) ** 2).sum(dim=0).cpu().numpy()
            train_per_dim_sums = (
                pd_sum if train_per_dim_sums is None else train_per_dim_sums + pd_sum
            )
            n_total += obs.shape[0]

        train_mse = float(sum(train_losses) / max(n_total, 1))
        train_per_dim = (
            (train_per_dim_sums / max(n_total, 1)).tolist()
            if train_per_dim_sums is not None
            else []
        )
        val_summary = _evaluate(policy, val_loader, device=device)

        record = {
            "epoch": epoch,
            "train_mse": train_mse,
            "train_per_dim_mse": [float(v) for v in train_per_dim],
            "val_mse": val_summary["mse"],
            "val_per_dim_mse": val_summary["per_dim_mse"],
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        last_train_summary = record

        if val_summary["mse"] < best_val:
            best_val = float(val_summary["mse"])
            best_epoch = epoch
            best_per_dim = list(val_summary["per_dim_mse"])
            # Snapshot best-epoch weights on CPU so subsequent training
            # iterations don't mutate the saved tensors.
            best_state = {
                k: v.detach().cpu().clone() for k, v in policy.state_dict().items()
            }

    duration = time.time() - t0

    # Persist best-epoch weights as policy.pt. Falls back to current weights
    # only in the pathological case where val never improved (e.g. epochs=0).
    if best_state is not None:
        torch.save(best_state, output_dir / "policy.pt")
    else:
        torch.save(policy.state_dict(), output_dir / "policy.pt")
    split.standardiser.save(output_dir / "obs_standardiser.npz")
    arch = policy.architecture_summary()
    arch["action_feature_names"] = list(split.action_feature_names)
    arch["obs_feature_names"] = list(split.obs_feature_names)
    (output_dir / "architecture.json").write_text(json.dumps(arch, indent=2))

    summary: Dict[str, Any] = {
        "seed": int(seed),
        "output_dir": str(output_dir),
        "n_train": int(split.train_indices.shape[0]),
        "n_val": int(split.val_indices.shape[0]),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "epochs": int(config.epochs),
        "duration_seconds": float(duration),
        "final_train_mse": float(last_train_summary.get("train_mse", float("nan"))),
        "final_train_per_dim_mse": list(
            last_train_summary.get("train_per_dim_mse", [])
        ),
        "final_val_mse": float(last_train_summary.get("val_mse", float("nan"))),
        "final_val_per_dim_mse": list(
            last_train_summary.get("val_per_dim_mse", [])
        ),
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "best_val_per_dim_mse": list(best_per_dim),
        "config": dataclasses.asdict(config),
        "action_feature_names": list(split.action_feature_names),
    }
    (output_dir / "seed_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------------


def train_multi_seed(
    parquet_path: Path,
    output_root: Path,
    *,
    seeds: Sequence[int],
    config: BCTrainingConfig,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    parquet_path = Path(parquet_path)

    seed_summaries: List[Dict[str, Any]] = []
    seeds_index: Dict[str, str] = {}
    t0 = time.time()
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        print(f"[bc] training seed={seed} → {seed_dir}", flush=True)
        summary = train_single_seed(
            parquet_path,
            seed_dir,
            seed=int(seed),
            config=config,
        )
        seed_summaries.append(summary)
        seeds_index[str(int(seed))] = str(seed_dir)
        print(
            f"  seed={seed} final_val_mse={summary['final_val_mse']:.6f} "
            f"per_dim={summary['final_val_per_dim_mse']} "
            f"best_epoch={summary['best_epoch']} duration={summary['duration_seconds']:.1f}s",
            flush=True,
        )

    duration = time.time() - t0

    final_vals = [s["final_val_mse"] for s in seed_summaries]
    best_vals = [s["best_val_mse"] for s in seed_summaries]
    aggregate = {
        "n_seeds": len(seed_summaries),
        "seeds": [int(s["seed"]) for s in seed_summaries],
        "final_val_mse_mean": float(np.mean(final_vals)) if final_vals else float("nan"),
        "final_val_mse_std": float(np.std(final_vals, ddof=0)) if final_vals else float("nan"),
        "best_val_mse_mean": float(np.mean(best_vals)) if best_vals else float("nan"),
        "best_val_mse_std": float(np.std(best_vals, ddof=0)) if best_vals else float("nan"),
        "duration_seconds": float(duration),
        "parquet_path": str(parquet_path),
        "config": dataclasses.asdict(config),
        "per_seed": seed_summaries,
    }
    (output_root / "multi_seed_summary.json").write_text(json.dumps(aggregate, indent=2))
    (output_root / "seeds_index.json").write_text(json.dumps(seeds_index, indent=2))
    return aggregate
