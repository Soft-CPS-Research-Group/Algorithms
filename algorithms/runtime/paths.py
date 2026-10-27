"""Shared paths service extracted from the original job entry point."""

from __future__ import annotations

import json
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from algorithms.runtime.atomic import atomic_json, atomic_text


def _derive_job_id(job_id: Optional[str]) -> str:
    if job_id:
        if job_id in {".", ".."} or "/" in job_id or "\\" in job_id:
            raise ValueError("job_id must be a single directory name, without path separators")
        return job_id
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _prepare_paths(base_dir: Path, job_id: str) -> dict[str, Path]:
    job_dir = base_dir / "jobs" / job_id
    logical_job_dir = job_dir
    job_dir.mkdir(parents=True, exist_ok=True)
    claim = job_dir / ".attempt"
    if (
        claim.exists()
        or (job_dir / "config.resolved.yaml").exists()
        or (job_dir / "results/result.json").exists()
    ):
        job_dir = job_dir / "attempts" / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=False)
        claim = job_dir / ".attempt"
    try:
        claim.touch(exist_ok=False)
    except FileExistsError:
        job_dir = logical_job_dir / "attempts" / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=False)
        (job_dir / ".attempt").touch(exist_ok=False)
    atomic_json(
        logical_job_dir / "latest_attempt.json",
        {"path": str(job_dir.relative_to(logical_job_dir)), "job_id": job_id},
    )
    bundle_dir = job_dir / "bundle"
    results_dir = job_dir / "results"
    progress_dir = job_dir / "progress"
    paths = {
        "job_dir": job_dir,
        "bundle_dir": bundle_dir,
        "log_dir": job_dir / "logs",
        "checkpoints_dir": job_dir / "checkpoints",
        "onnx_dir": bundle_dir / "onnx_models",
        "results_dir": results_dir,
        "simulation_data_dir": results_dir / "simulation_data",
        "progress_dir": progress_dir,
        "result_path": results_dir / "result.json",
        "summary_path": results_dir / "summary.json",
        "progress_path": progress_dir / "progress.json",
        "mlflow_dir": base_dir / "mlflow" / "mlruns",
        "job_info_path": job_dir / "job_info.json",
        "artifact_manifest_path": bundle_dir / "artifact_manifest.json",
        "resolved_config_path": job_dir / "config.resolved.yaml",
    }

    for key, path in paths.items():
        if key.endswith("_dir"):
            path.mkdir(parents=True, exist_ok=True)
        elif key.endswith("_path"):
            path.parent.mkdir(parents=True, exist_ok=True)

    return paths


def _write_resolved_config(config: dict, output_path: Path) -> None:
    """Persist the runtime-resolved configuration for reproducibility."""
    atomic_text(output_path, yaml.safe_dump(config, sort_keys=False))


def _mark_failed_outputs(path_info: dict[str, Path], exc: BaseException) -> None:
    """Persist a terminal failure state for orchestrators and long local runs."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    from algorithms.runtime.lifecycle import JobCancelled

    status = "cancelled" if isinstance(exc, (KeyboardInterrupt, JobCancelled)) else "failed"
    result_payload = {}
    if path_info["result_path"].is_file():
        try:
            result_payload = json.loads(path_info["result_path"].read_text())
        except (ValueError, OSError):
            pass
    if result_payload.get("export_status") == "pending":
        result_payload["export_status"] = "failed"
    result_payload.update(
        {
            "status": status,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "timestamp": timestamp,
        }
    )
    result_payload["traceback"] = traceback.format_exc()
    atomic_json(path_info["result_path"], result_payload)

    progress_payload: dict[str, Any] = {}
    if path_info["progress_path"].exists():
        try:
            with open(path_info["progress_path"], "r", encoding="utf-8") as progress_file:
                progress_payload = json.load(progress_file)
        except (OSError, json.JSONDecodeError):
            progress_payload = {}
    progress_payload.update(
        {
            "status": status,
            "timestamp": timestamp,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    )
    atomic_json(path_info["progress_path"], progress_payload)
