"""Artifact manifest construction helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from loguru import logger


def build_manifest(
    config: Dict[str, Any],
    environment_metadata: Dict[str, Any],
    agent_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine configuration, environment, and agent metadata into a manifest."""
    metadata = dict(config.get("metadata", {}) or {})
    bundle_cfg = dict(config.get("bundle", {}) or {})
    _merge_bundle_metadata(metadata, bundle_cfg)
    normalized_agent_metadata = _normalize_agent_metadata(agent_metadata)

    manifest = {
        "manifest_version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata,
        "simulator": {
            "dataset_name": config.get("simulator", {}).get("dataset_name"),
            "dataset_path": config.get("simulator", {}).get("dataset_path"),
            "central_agent": config.get("simulator", {}).get("central_agent"),
            "reward_function": config.get("simulator", {}).get("reward_function"),
        },
        "training": config.get("training", {}),
        "topology": config.get("topology", {}),
        "pipeline": _summarise_pipeline(config),
        "environment": environment_metadata,
        "agent": normalized_agent_metadata,
    }
    return manifest


def write_manifest(manifest: Dict[str, Any], output_dir: str) -> Path:
    """Persist the manifest to disk and return the path."""
    path = Path(output_dir) / "artifact_manifest.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=_json_default)
    except Exception as exc:
        logger.error("Failed to write artifact manifest: {}", exc)
        raise
    return path


def _summarise_pipeline(config: Dict[str, Any]) -> list:
    """Build the manifest's pipeline summary from the resolved config."""
    pipeline_cfg = config.get("pipeline") or []
    summary = []
    for index, stage in enumerate(pipeline_cfg):
        if not isinstance(stage, dict):
            continue
        summary.append(
            {
                "stage_index": index,
                "algorithm": stage.get("algorithm"),
                "count": int(stage.get("count", 1) or 1),
                "hyperparameters": stage.get("hyperparameters", {}) or {},
            }
        )
    return summary


def _merge_bundle_metadata(metadata: Dict[str, Any], bundle_cfg: Dict[str, Any]) -> None:
    """Backfill bundle metadata keys when provided in the `bundle` config section."""
    if not metadata.get("bundle_version") and bundle_cfg.get("bundle_version"):
        metadata["bundle_version"] = bundle_cfg["bundle_version"]
    if not metadata.get("description") and bundle_cfg.get("description"):
        metadata["description"] = bundle_cfg["description"]
    if not metadata.get("alias_mapping_path") and bundle_cfg.get("alias_mapping_path"):
        metadata["alias_mapping_path"] = bundle_cfg["alias_mapping_path"]


def _normalize_agent_metadata(agent_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the manifest carries canonical artifact entries."""
    metadata = dict(agent_metadata or {})
    artifacts = metadata.get("artifacts") or []
    default_format = metadata.get("format") or "onnx"

    normalized_artifacts = []
    for raw_artifact in artifacts:
        artifact = dict(raw_artifact or {})
        artifact.setdefault("format", default_format)
        artifact["config"] = dict(artifact.get("config") or {})
        normalized_artifacts.append(artifact)

    metadata["format"] = default_format
    metadata["artifacts"] = normalized_artifacts
    return metadata


def _json_default(obj: Any) -> Any:
    """Best-effort conversion of non-serializable objects for JSON dumps."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    try:
        import numpy as np  # Local import to keep optional dependency.

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
