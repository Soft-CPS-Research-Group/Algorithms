"""Shared tracking service extracted from the original job entry point."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

from utils.pipeline_utils import summarise_pipeline_algorithms


class OptionalMLflow:
    """Load tracking only when enabled; disabled jobs need no MLflow install."""

    def active_run(self):
        backend = sys.modules.get("mlflow")
        return backend.active_run() if backend is not None else None

    def __getattr__(self, name):
        try:
            backend = import_module("mlflow")
        except ImportError as error:
            raise RuntimeError(
                "MLflow tracking was requested but is not installed. "
                "Install the tracking extra or set tracking.mlflow_enabled=false."
            ) from error
        return getattr(backend, name)


mlflow = OptionalMLflow()


def _resolve_tracking_uri(runtime: dict[str, Any], fallback_mlflow_dir: Path) -> str:
    """Resolve MLflow tracking URI with env-first precedence."""
    env_uri = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if env_uri:
        return env_uri

    runtime_tracking_uri = (runtime.get("tracking_uri") or runtime.get("mlflow_uri") or "").strip()
    if runtime_tracking_uri:
        return runtime_tracking_uri

    return f"file:{fallback_mlflow_dir}"


def _compute_config_hash(config: dict[str, Any]) -> str:
    """Build a stable hash of the validated config excluding runtime-only volatile fields."""
    payload = json.loads(json.dumps(config))
    runtime = payload.get("runtime") or {}
    for key in (
        "log_dir",
        "job_dir",
        "job_id",
        "run_id",
        "run_name",
        "experiment_id",
        "mlflow_run_url",
    ):
        runtime.pop(key, None)
    payload["runtime"] = runtime
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _discover_git_sha(repo_root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        sha = completed.stdout.strip()
        return sha or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_mlflow_run_url(
    base_url: Optional[str], experiment_id: Optional[str], run_id: Optional[str]
) -> Optional[str]:
    if not base_url or not experiment_id or not run_id:
        return None
    normalized = base_url.rstrip("/")
    return f"{normalized}/#/experiments/{experiment_id}/runs/{run_id}"


def _build_mlflow_tags(
    config: dict[str, Any], *, job_id: str, run_name: str, config_hash: str, git_sha: Optional[str]
) -> dict[str, str]:
    simulator_cfg = config.get("simulator", {})
    dataset_name = (
        simulator_cfg.get("dataset_name") or simulator_cfg.get("dataset_path") or "unknown_dataset"
    )
    algorithm_name = summarise_pipeline_algorithms(config) or "unknown_algorithm"
    tags: dict[str, str] = {
        "opeva.job_id": str(job_id),
        "opeva.algorithm": str(algorithm_name),
        "opeva.dataset": str(dataset_name),
        "opeva.run_name": str(run_name),
        "opeva.config_hash": config_hash,
    }
    if git_sha:
        tags["opeva.git_sha"] = git_sha
    return tags


def _log_curated_artifacts(
    *,
    resolved_config_path: Path,
    manifest_path: Optional[Path],
    result_path: Path,
    summary_path: Path,
) -> None:
    if not mlflow.active_run():
        return
    for artifact in (resolved_config_path, manifest_path, result_path, summary_path):
        if artifact.exists():
            mlflow.log_artifact(str(artifact), artifact_path="artifacts")
