"""Shared tracking service extracted from the original job entry point."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

from utils.pipeline_utils import summarise_pipeline_algorithms


class OptionalMLflow:
    """Load tracking only when enabled; disabled jobs need no MLflow install."""

    policy = None

    def active_run(self):
        if self.policy is not None and (not self.policy["enabled"] or self.policy["failed"]):
            return None
        backend = sys.modules.get("mlflow")
        return backend.active_run() if backend is not None else None

    def metrics_active(self):
        return (
            self.policy["metrics_enabled"]
            if self.policy is not None
            else self.active_run() is not None
        )

    def _backend(self, name, *args, **kwargs):
        started = time.perf_counter()
        try:
            result = getattr(import_module("mlflow"), name)(*args, **kwargs)
            if self.policy is not None:
                self.policy["remote_calls"] += 1
            return result
        except Exception as error:
            if self.policy is None or self.policy["failure_policy"] == "abort":
                raise
            self.policy["failed"] = True
            self.policy["error"] = f"{type(error).__name__}: {error}"
            from loguru import logger

            logger.warning("MLflow failed; continuing with durable local evidence: {}", error)
            return None
        finally:
            if self.policy is not None:
                self.policy["remote_seconds"] += time.perf_counter() - started

    def log_metrics(self, metrics, step=None, **kwargs):
        if self.policy is not None and not self.policy["metrics_enabled"]:
            return
        clean = {}
        for key, value in metrics.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"Non-finite tracking metric: {key}")
            clean[str(key)] = numeric
        policy = self.policy
        if policy is not None:
            record = json.dumps({"step": step, "metrics": clean}, allow_nan=False) + "\n"
            with policy["local_path"].open("a", encoding="utf-8") as handle:
                handle.write(record)
            policy["local_bytes"] += len(record.encode())
            if not policy["enabled"] or policy["failed"]:
                return
            boundary = step is None or any("Episode" in key or "Overall" in key for key in clean)
            if not boundary and int(step) % policy["sample_interval"]:
                return
        return self._backend("log_metrics", clean, step=step, **kwargs)

    def log_metric(self, key, value, step=None, **kwargs):
        return self.log_metrics({key: value}, step=step, **kwargs)

    def _artifact(self, method, path, artifact_path=None, **kwargs):
        policy = self.policy
        if policy is not None:
            if not policy["enabled"] or policy["failed"]:
                return
            label = str(artifact_path or "")
            allowed = policy["profile"] == "full"
            allowed |= policy["profile"] == "curated" and label == "artifacts"
            allowed |= policy["models"] and label in {"model", "models", "onnx"}
            allowed |= policy["checkpoints"] and label.startswith("checkpoints")
            if not allowed:
                return
        return self._backend(method, path, artifact_path=artifact_path, **kwargs)

    def log_artifact(self, path, artifact_path=None, **kwargs):
        return self._artifact("log_artifact", path, artifact_path, **kwargs)

    def log_artifacts(self, path, artifact_path=None, **kwargs):
        return self._artifact("log_artifacts", path, artifact_path, **kwargs)

    def __getattr__(self, name):
        if self.policy is not None and name in {
            "set_tracking_uri",
            "set_experiment",
            "start_run",
            "set_tags",
            "log_params",
            "log_param",
            "end_run",
        }:

            def call(*args, **kwargs):
                if not self.policy["enabled"] or self.policy["failed"]:
                    return None
                return self._backend(name, *args, **kwargs)

            return call
        try:
            backend = import_module("mlflow")
        except ImportError as error:
            raise RuntimeError(
                "MLflow tracking was requested but is not installed. "
                "Install the tracking extra or set tracking.mlflow_enabled=false."
            ) from error
        return getattr(backend, name)


mlflow = OptionalMLflow()


def metrics_active(backend=mlflow):
    hook = getattr(backend, "metrics_active", None)
    return hook() if callable(hook) else backend.active_run() is not None


@contextmanager
def tracking_session(config, paths):
    from algorithms.runtime.atomic import atomic_json

    previous = mlflow.policy
    settings = config.get("tracking") or {}
    paths["log_dir"].mkdir(parents=True, exist_ok=True)
    mlflow.policy = {
        "enabled": bool(settings.get("mlflow_enabled", False)),
        "failed": False,
        "metrics_enabled": settings.get("metrics_enabled", True),
        "failure_policy": settings.get("mlflow_failure_policy", "local"),
        "sample_interval": settings.get("mlflow_step_sample_interval", 10),
        "profile": settings.get("mlflow_artifacts_profile", "minimal"),
        "models": settings.get("mlflow_upload_models", False),
        "checkpoints": settings.get("mlflow_upload_checkpoints", False),
        "local_path": paths["log_dir"] / "metrics.jsonl",
        "remote_calls": 0,
        "remote_seconds": 0.0,
        "local_bytes": 0,
    }
    original_error = None
    try:
        yield mlflow.policy
    except BaseException as error:
        original_error = error
        raise
    finally:
        policy = mlflow.policy
        # A logging failure must not leave an active MLflow run attached to the
        # next job in this process. end_run pops its local context even on error.
        backend = sys.modules.get("mlflow")
        if (
            policy["enabled"]
            and backend is not None
            and callable(getattr(backend, "end_run", None))
        ):
            if backend.active_run() is not None:
                try:
                    backend.end_run(
                        status="FAILED" if original_error or policy["failed"] else "FINISHED"
                    )
                except Exception as error:
                    policy["cleanup_error"] = f"{type(error).__name__}: {error}"
        mlflow.policy = previous
        try:
            atomic_json(
                paths["results_dir"] / "tracking.json",
                {key: value for key, value in policy.items() if key != "local_path"},
            )
        except OSError:
            if original_error is None:
                raise


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
        if artifact is not None and artifact.exists():
            mlflow.log_artifact(str(artifact), artifact_path="artifacts")
