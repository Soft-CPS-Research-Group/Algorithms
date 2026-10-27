"""Typed ownership of one job's configuration, identity and output locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict


class JobPaths(TypedDict):
    job_dir: Path
    bundle_dir: Path
    log_dir: Path
    checkpoints_dir: Path
    onnx_dir: Path
    results_dir: Path
    simulation_data_dir: Path
    progress_dir: Path
    result_path: Path
    summary_path: Path
    progress_path: Path
    mlflow_dir: Path
    job_info_path: Path
    artifact_manifest_path: Path
    resolved_config_path: Path


@dataclass(frozen=True)
class JobContext:
    config: dict[str, Any]
    paths: JobPaths
    job_id: str
    run_id: str
    run_name: str
    experiment_id: str | None
    tracking_uri: str
    mlflow_enabled: bool
    artifact_profile: str


@dataclass(frozen=True)
class OutputPlan:
    bundle: bool
    explicit_checkpoint: bool

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OutputPlan:
        mode = (config.get("outputs") or {}).get("model_artifacts")
        return cls(
            bundle=mode in (None, "bundle", "both"),
            explicit_checkpoint=mode in ("checkpoint", "both"),
        )
