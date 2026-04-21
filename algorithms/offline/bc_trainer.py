"""Behavioral Cloning training loop.

Algorithm: minimise ``MSE(pi(obs), action)`` over the offline dataset using
Adam. Validation loss is tracked at the end of every epoch on a held-out
*episode-level* split. Outputs everything an inference agent needs:

* ``model.pth`` — trained policy weights + architecture summary.
* ``normalization_stats.json`` — mean/std per observation feature.
* ``training_metadata.json`` — hyperparameters, dataset path, final losses.
* ``loss_history.json`` — per-epoch train/val losses.
* ``loss_curve.png`` — quick visual sanity check.

Optional MLflow tracking is supported via :func:`utils.mlflow_helper`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # No display required.
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from algorithms.offline.bc_policy import BCPolicy
from algorithms.offline.data_loader import DatasetBundle


@dataclass
class BCTrainingConfig:
    """Hyperparameters and IO config for a BC run."""

    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 3e-4
    batch_size: int = 256
    epochs: int = 50
    val_fraction: float = 0.20
    seed: int = 22
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    mlflow_enabled: bool = True
    mlflow_experiment: str = "offline_bc"
    mlflow_run_name: Optional[str] = None

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
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    duration_seconds: float


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


def _save_loss_curve(history: Dict[str, List[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["train_loss"], label="train", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"], label="validation", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Behavioral Cloning — training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


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
    logger.info("BC training device: {}", device)

    # Reproducibility.
    torch.manual_seed(config.seed)

    # Build model + optimiser.
    model = BCPolicy(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        hidden_layers=config.hidden_layers,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    logger.info(
        "Model: {} → {} → {} (hidden ReLU, output Tanh)",
        dataset.obs_dim,
        " → ".join(str(h) for h in config.hidden_layers),
        dataset.action_dim,
    )
    logger.info(
        "Train samples: {} (eps {}); Val samples: {} (eps {})",
        dataset.train_size,
        dataset.train_episodes,
        dataset.val_size,
        dataset.val_episodes,
    )

    # Optional MLflow setup -------------------------------------------------
    mlflow_run = None
    if config.mlflow_enabled:
        try:
            import mlflow

            mlflow.set_experiment(config.mlflow_experiment)
            mlflow_run = mlflow.start_run(run_name=config.mlflow_run_name)
            mlflow.log_params(
                {
                    "hidden_layers": config.hidden_layers,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "val_fraction": config.val_fraction,
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
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            logger.info(
                "Epoch {:>3}/{} | train_loss={:.6f} | val_loss={:.6f}{}",
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
                    logger.warning("MLflow metric log failed at epoch {}: {}", epoch, exc)

    finally:
        duration = time.time() - start

    # Persist artifacts ------------------------------------------------------
    if best_state_dict is None:
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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
    _save_loss_curve(history, plot_path)

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
        },
        "hyperparameters": {
            "hidden_layers": config.hidden_layers,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "val_fraction": config.val_fraction,
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
        },
    }
    metadata_path = output_dir / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Artifacts written to {}", output_dir)
    logger.info(
        "Best val loss = {:.6f} (epoch {}); duration = {:.1f}s",
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
        final_train_loss=history["train_loss"][-1] if history["train_loss"] else float("nan"),
        final_val_loss=history["val_loss"][-1] if history["val_loss"] else float("nan"),
        best_val_loss=best_val_loss if best_val_loss != float("inf") else float("nan"),
        best_epoch=best_epoch,
        epochs_run=len(history["train_loss"]),
        duration_seconds=duration,
    )
