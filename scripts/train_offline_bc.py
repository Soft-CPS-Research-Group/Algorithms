"""CLI entrypoint for Behavioral Cloning training.

Example
-------

.. code-block:: bash

    python scripts/train_offline_bc.py \\
        --config configs/offline_bc/train.yaml \\
        --dataset datasets/offline_rl/offline_dataset.csv \\
        --run_id bc-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

# Ensure the repo root is importable when invoked directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.offline.bc_trainer import BCTrainingConfig, train_bc  # noqa: E402
from algorithms.offline.data_loader import load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline BC policy.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training YAML config.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to the offline dataset CSV. Falls back to dataset.path in the config.",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier; output dir = runs/offline_bc/<run_id>. Defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Override the runs root directory (default: <repo>/runs/offline_bc).",
    )
    return parser.parse_args()


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_config(raw: dict) -> BCTrainingConfig:
    hp = raw.get("hyperparameters", {}) or {}
    tracking = raw.get("tracking", {}) or {}
    return BCTrainingConfig(
        hidden_layers=list(hp.get("hidden_layers", [256, 256])),
        learning_rate=float(hp.get("learning_rate", 3e-4)),
        batch_size=int(hp.get("batch_size", 256)),
        epochs=int(hp.get("epochs", 50)),
        val_fraction=float(hp.get("val_fraction", 0.20)),
        seed=int(hp.get("seed", 22)),
        device=str(hp.get("device", "auto")),
        mlflow_enabled=bool(tracking.get("mlflow_enabled", True)),
        mlflow_experiment=str(tracking.get("mlflow_experiment", "offline_bc")),
        mlflow_run_name=tracking.get("mlflow_run_name"),
    )


def main() -> int:
    args = parse_args()
    raw_config = _load_yaml(args.config)

    dataset_path = args.dataset or (raw_config.get("dataset") or {}).get("path")
    if not dataset_path:
        logger.error("No dataset path provided (use --dataset or dataset.path in the config).")
        return 1
    dataset_path = str(Path(dataset_path).resolve())

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root) if args.output_root else (REPO_ROOT / "runs" / "offline_bc")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to file as well as stdout.
    log_path = output_dir / "training.log"
    logger.add(str(log_path), level="INFO")

    logger.info("Run ID: {}", run_id)
    logger.info("Dataset: {}", dataset_path)
    logger.info("Output dir: {}", output_dir)

    bc_cfg = _build_config(raw_config)

    # Persist a copy of the resolved config alongside the artifacts.
    resolved_path = output_dir / "config.resolved.yaml"
    with resolved_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "run_id": run_id,
                "dataset_path": dataset_path,
                "hyperparameters": {
                    "hidden_layers": bc_cfg.hidden_layers,
                    "learning_rate": bc_cfg.learning_rate,
                    "batch_size": bc_cfg.batch_size,
                    "epochs": bc_cfg.epochs,
                    "val_fraction": bc_cfg.val_fraction,
                    "seed": bc_cfg.seed,
                    "device": bc_cfg.device,
                },
                "tracking": {
                    "mlflow_enabled": bc_cfg.mlflow_enabled,
                    "mlflow_experiment": bc_cfg.mlflow_experiment,
                    "mlflow_run_name": bc_cfg.mlflow_run_name,
                },
            },
            fh,
            sort_keys=False,
        )

    bundle = load_dataset(
        csv_path=dataset_path,
        batch_size=bc_cfg.batch_size,
        val_fraction=bc_cfg.val_fraction,
        seed=bc_cfg.seed,
    )

    result = train_bc(bundle, bc_cfg, output_dir, dataset_path=dataset_path)

    summary = {
        "run_id": run_id,
        "output_dir": str(result.output_dir),
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "final_train_loss": result.final_train_loss,
        "final_val_loss": result.final_val_loss,
        "epochs_run": result.epochs_run,
        "duration_seconds": result.duration_seconds,
    }
    logger.info("Training summary:\n{}", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
