"""Behavioral Cloning training loop (M3).

Algorithm: minimise ``MSE(pi(obs), action)`` over the offline dataset using
Adam with optional weight decay, dropout, and gradient clipping. Validation
loss is tracked at the end of every epoch on a held-out *episode-level*
split. Outputs everything an inference agent needs:

* ``model.pth`` — trained policy weights + architecture summary.
* ``normalization_stats.json`` — mean/std per observation feature.
* ``training_metadata.json`` — hyperparameters, dataset path, final losses.
* ``loss_history.json`` — per-epoch train/val losses.
* ``loss_curve.png`` — quick visual sanity check.
* ``val_episodes.json`` — which episode indices were held out (for audit).

A multi-seed driver (:func:`train_bc_multi_seed`) runs :func:`train_bc` once
per seed, writes a ``seed_<n>/`` subdir per run, and produces aggregated
``multi_seed_summary.json`` + ``seeds_index.json`` files. When MLflow is
enabled, a parent run is created under ``offline_bc_m3`` with one nested
child run per seed, mirroring the directory structure.

Optional MLflow tracking is supported via :func:`utils.mlflow_helper`-style
imports (no hard dependency on the helper itself).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # No display required.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from algorithms.offline.bc_policy import BCPolicy
from algorithms.offline.data_loader import DatasetBundle, load_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BCTrainingConfig:
    """Hyperparameters and IO config for a single-seed BC run."""

    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    dropout: float = 0.0
    gradient_clip_norm: Optional[float] = None
    batch_size: int = 256
    epochs: int = 50
    val_fraction: float = 0.20                # legacy; ignored if val_episodes_mode is set
    val_episodes_mode: str = "last:1"         # see data_loader._select_val_episodes
    action_target: str = "action"             # "action" (M1) or "action_clean" (M3)
    seed: int = 22
    device: str = "auto"                       # "auto" | "cpu" | "cuda"
    mlflow_enabled: bool = True
    mlflow_experiment: str = "offline_bc"
    mlflow_run_name: Optional[str] = None
    # Used when train_bc is invoked from train_bc_multi_seed: the trainer
    # will join the parent run as a nested child instead of starting a
    # standalone run.
    mlflow_nested: bool = False

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class BCTrainingResult:
    """Returned by :func:`train_bc`."""

    output_dir: Path
    model_path: Path
    stats_path: Path
    metadata_path: Path
    history_path: Path
    plot_path: Path
    val_episodes_path: Path
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    duration_seconds: float
    seed: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evaluate(model: BCPolicy, loader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for obs, act in loader:
            obs = obs.to(device)
            act = act.to(device)
            pred = model(obs)
            loss = criterion(pred, act)
            total_loss += loss.item() * obs.size(0)
            total_samples += obs.size(0)
    return total_loss / max(total_samples, 1)


def _save_loss_curve(history: Dict[str, List[float]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["train_loss"], label="train", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"], label="validation", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------


def train_bc(
    dataset: DatasetBundle,
    config: BCTrainingConfig,
    output_dir: str | Path,
    *,
    dataset_path: Optional[str] = None,
) -> BCTrainingResult:
    """Train a BC policy on ``dataset`` and write all artifacts to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = config.resolve_device()
    logger.info("BC training device: {} (seed={})", device, config.seed)

    _set_global_seeds(config.seed)

    model = BCPolicy(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()

    logger.info(
        "Model: {} → {} → {} (hidden ReLU, dropout={}, output Tanh, weight_decay={}, grad_clip={})",
        dataset.obs_dim,
        " → ".join(str(h) for h in config.hidden_layers),
        dataset.action_dim,
        config.dropout,
        config.weight_decay,
        config.gradient_clip_norm,
    )
    logger.info(
        "Train samples: {} (eps {}); Val samples: {} (eps {}); action_target={}",
        dataset.train_size,
        dataset.train_episodes,
        dataset.val_size,
        dataset.val_episodes,
        dataset.action_target,
    )

    # Optional MLflow setup -------------------------------------------------
    mlflow_run = None
    if config.mlflow_enabled:
        try:
            import mlflow

            if not config.mlflow_nested:
                mlflow.set_experiment(config.mlflow_experiment)
            mlflow_run = mlflow.start_run(
                run_name=config.mlflow_run_name,
                nested=config.mlflow_nested,
            )
            mlflow.log_params(
                {
                    "hidden_layers": config.hidden_layers,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "dropout": config.dropout,
                    "gradient_clip_norm": config.gradient_clip_norm,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "val_episodes_mode": config.val_episodes_mode,
                    "action_target": config.action_target,
                    "seed": config.seed,
                    "obs_dim": dataset.obs_dim,
                    "action_dim": dataset.action_dim,
                    "train_samples": dataset.train_size,
                    "val_samples": dataset.val_size,
                    "device": str(device),
                }
            )
        except Exception as exc:
            logger.warning("MLflow disabled (init failed): {}", exc)
            mlflow_run = None

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    start = time.time()

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            for obs, act in dataset.train_loader:
                obs = obs.to(device)
                act = act.to(device)

                optimizer.zero_grad()
                pred = model(obs)
                loss = criterion(pred, act)
                loss.backward()
                if config.gradient_clip_norm is not None and config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float(config.gradient_clip_norm)
                    )
                optimizer.step()

                running += loss.item() * obs.size(0)
                seen += obs.size(0)

            train_loss = running / max(seen, 1)
            val_loss = _evaluate(model, dataset.val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            logger.info(
                "[seed={}] Epoch {:>3}/{} | train_loss={:.6f} | val_loss={:.6f}{}",
                config.seed,
                epoch,
                config.epochs,
                train_loss,
                val_loss,
                "  ← new best" if epoch == best_epoch else "",
            )

            if mlflow_run is not None:
                try:
                    import mlflow

                    mlflow.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss},
                        step=epoch,
                    )
                except Exception as exc:
                    logger.warning(
                        "MLflow metric log failed at epoch {}: {}", epoch, exc
                    )

    finally:
        duration = time.time() - start

    # Persist artifacts ------------------------------------------------------
    if best_state_dict is None:
        best_state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    model_path = output_dir / "model.pth"
    torch.save(
        {
            "state_dict": best_state_dict,
            "architecture": model.architecture_summary(),
        },
        model_path,
    )

    stats_path = output_dir / "normalization_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset.stats.to_dict(), fh, indent=2)

    history_path = output_dir / "loss_history.json"
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    plot_path = output_dir / "loss_curve.png"
    _save_loss_curve(
        history,
        plot_path,
        title=f"BC training curves — seed={config.seed}",
    )

    val_episodes_path = output_dir / "val_episodes.json"
    val_episodes_path.write_text(
        json.dumps(
            {
                "seed": config.seed,
                "val_episodes": dataset.val_episodes,
                "train_episodes": dataset.train_episodes,
                "policy_mode_summary": dataset.policy_mode_summary,
                "topology_versions": dataset.topology_versions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata: Dict[str, Any] = {
        "algorithm": "BehavioralCloning",
        "library": "pytorch",
        "torch_version": torch.__version__,
        "device": str(device),
        "dataset_path": dataset_path,
        "dataset": {
            "train_size": dataset.train_size,
            "val_size": dataset.val_size,
            "train_episodes": dataset.train_episodes,
            "val_episodes": dataset.val_episodes,
            "obs_dim": dataset.obs_dim,
            "action_dim": dataset.action_dim,
            "action_target": dataset.action_target,
            "policy_mode_summary": dataset.policy_mode_summary,
            "topology_versions": dataset.topology_versions,
        },
        "hyperparameters": {
            "hidden_layers": config.hidden_layers,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "gradient_clip_norm": config.gradient_clip_norm,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "val_episodes_mode": config.val_episodes_mode,
            "action_target": config.action_target,
            "seed": config.seed,
            "optimizer": "Adam",
            "loss": "MSE",
            "hidden_activation": "relu",
            "output_activation": "tanh",
        },
        "results": {
            "epochs_run": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
            "best_epoch": best_epoch,
            "duration_seconds": duration,
        },
        "artifacts": {
            "model": model_path.name,
            "normalization_stats": stats_path.name,
            "loss_history": history_path.name,
            "loss_curve": plot_path.name,
            "val_episodes": val_episodes_path.name,
        },
    }
    metadata_path = output_dir / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Artifacts written to {}", output_dir)
    logger.info(
        "[seed={}] Best val loss = {:.6f} (epoch {}); duration = {:.1f}s",
        config.seed,
        best_val_loss,
        best_epoch,
        duration,
    )

    if mlflow_run is not None:
        try:
            import mlflow

            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(stats_path))
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact(str(plot_path))
            mlflow.log_artifact(str(metadata_path))
            mlflow.log_artifact(str(val_episodes_path))
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.end_run()
        except Exception as exc:
            logger.warning("MLflow artifact log failed: {}", exc)

    return BCTrainingResult(
        output_dir=output_dir,
        model_path=model_path,
        stats_path=stats_path,
        metadata_path=metadata_path,
        history_path=history_path,
        plot_path=plot_path,
        val_episodes_path=val_episodes_path,
        final_train_loss=history["train_loss"][-1] if history["train_loss"] else float("nan"),
        final_val_loss=history["val_loss"][-1] if history["val_loss"] else float("nan"),
        best_val_loss=best_val_loss if best_val_loss != float("inf") else float("nan"),
        best_epoch=best_epoch,
        epochs_run=len(history["train_loss"]),
        duration_seconds=duration,
        seed=config.seed,
    )


# ---------------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------------


@dataclass
class MultiSeedSummary:
    seeds: List[int]
    per_seed_results: List[BCTrainingResult]
    output_root: Path
    summary_path: Path
    seeds_index_path: Path
    best_seed: int
    best_val_loss: float
    duration_seconds: float


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
    }


def train_bc_multi_seed(
    csv_path: str | Path,
    output_root: str | Path,
    *,
    seeds: Sequence[int],
    config_template: BCTrainingConfig,
) -> MultiSeedSummary:
    """Train one BC policy per seed under ``output_root/seed_<n>/`` and aggregate.

    The data split itself depends on the seed (``random:N`` mode), so each
    run sees a different train/val partition — meaningful multi-seed variance.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not seeds:
        raise ValueError("Need at least one seed.")

    parent_run = None
    if config_template.mlflow_enabled:
        try:
            import mlflow

            mlflow.set_experiment(config_template.mlflow_experiment)
            parent_run_name = (
                config_template.mlflow_run_name
                or f"multi_seed_{output_root.name}"
            )
            parent_run = mlflow.start_run(run_name=parent_run_name)
            mlflow.log_params(
                {
                    "seeds": list(seeds),
                    "n_seeds": len(seeds),
                    "csv_path": str(csv_path),
                    "output_root": str(output_root),
                    "epochs": config_template.epochs,
                    "val_episodes_mode": config_template.val_episodes_mode,
                    "action_target": config_template.action_target,
                    "weight_decay": config_template.weight_decay,
                    "dropout": config_template.dropout,
                }
            )
        except Exception as exc:
            logger.warning("MLflow disabled (parent init failed): {}", exc)
            parent_run = None

    start = time.time()
    per_seed_results: List[BCTrainingResult] = []
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset *per seed* so the random val split actually varies.
        dataset = load_dataset(
            csv_path,
            batch_size=config_template.batch_size,
            val_episodes_mode=config_template.val_episodes_mode,
            action_target=config_template.action_target,
            seed=seed,
        )

        # Clone config with this seed and nested-MLflow flag.
        seed_config = BCTrainingConfig(**asdict(config_template))
        seed_config.seed = int(seed)
        seed_config.mlflow_run_name = f"seed_{seed}"
        seed_config.mlflow_nested = parent_run is not None

        result = train_bc(
            dataset=dataset,
            config=seed_config,
            output_dir=seed_dir,
            dataset_path=str(csv_path),
        )
        per_seed_results.append(result)

    duration = time.time() - start

    # Aggregate ------------------------------------------------------------
    best_idx = int(
        np.argmin(
            [
                r.best_val_loss if not math.isnan(r.best_val_loss) else float("inf")
                for r in per_seed_results
            ]
        )
    )
    best_seed = per_seed_results[best_idx].seed
    best_val_loss = per_seed_results[best_idx].best_val_loss

    summary = {
        "seeds": [int(r.seed) for r in per_seed_results],
        "best_val_loss": {
            **_mean_std([r.best_val_loss for r in per_seed_results]),
            "per_seed": {
                str(r.seed): float(r.best_val_loss) for r in per_seed_results
            },
        },
        "best_epoch": {
            **_mean_std([r.best_epoch for r in per_seed_results]),
            "per_seed": {
                str(r.seed): int(r.best_epoch) for r in per_seed_results
            },
        },
        "final_val_loss": {
            **_mean_std([r.final_val_loss for r in per_seed_results]),
            "per_seed": {
                str(r.seed): float(r.final_val_loss) for r in per_seed_results
            },
        },
        "training_seconds_per_seed": {
            str(r.seed): float(r.duration_seconds) for r in per_seed_results
        },
        "training_seconds_total": float(duration),
        "best_seed": int(best_seed),
        "val_episodes_per_seed": {
            str(r.seed): json.loads(r.val_episodes_path.read_text())["val_episodes"]
            for r in per_seed_results
        },
    }
    summary_path = output_root / "multi_seed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    seeds_index = {
        str(r.seed): {
            "val_loss": float(r.best_val_loss),
            "best_epoch": int(r.best_epoch),
            "model_path": str(r.model_path.relative_to(output_root)),
            "stats_path": str(r.stats_path.relative_to(output_root)),
            "metadata_path": str(r.metadata_path.relative_to(output_root)),
        }
        for r in per_seed_results
    }
    seeds_index_path = output_root / "seeds_index.json"
    seeds_index_path.write_text(json.dumps(seeds_index, indent=2), encoding="utf-8")

    logger.info(
        "Multi-seed training complete: {} seeds, best_seed={} (val_loss={:.6f}), total {:.1f}s",
        len(seeds),
        best_seed,
        best_val_loss,
        duration,
    )

    if parent_run is not None:
        try:
            import mlflow

            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(seeds_index_path))
            mlflow.log_metric("best_seed_val_loss", float(best_val_loss))
            mlflow.log_metric("best_seed", int(best_seed))
            mlflow.log_metric(
                "best_val_loss_mean", float(summary["best_val_loss"]["mean"])
            )
            mlflow.log_metric(
                "best_val_loss_std", float(summary["best_val_loss"]["std"])
            )
            mlflow.end_run()
        except Exception as exc:
            logger.warning("MLflow parent finalisation failed: {}", exc)

    return MultiSeedSummary(
        seeds=[int(r.seed) for r in per_seed_results],
        per_seed_results=per_seed_results,
        output_root=output_root,
        summary_path=summary_path,
        seeds_index_path=seeds_index_path,
        best_seed=int(best_seed),
        best_val_loss=float(best_val_loss),
        duration_seconds=float(duration),
    )
