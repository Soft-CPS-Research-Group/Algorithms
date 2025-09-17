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
    manifest = {
        "manifest_version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metadata": config.get("metadata", {}),
        "simulator": {
            "dataset_name": config.get("simulator", {}).get("dataset_name"),
            "dataset_path": config.get("simulator", {}).get("dataset_path"),
            "central_agent": config.get("simulator", {}).get("central_agent"),
            "reward_function": config.get("simulator", {}).get("reward_function"),
        },
        "training": config.get("training", {}),
        "topology": config.get("topology", {}),
        "algorithm": {
            "name": config.get("algorithm", {}).get("name"),
            "hyperparameters": config.get("algorithm", {}).get("hyperparameters", {}),
        },
        "environment": environment_metadata,
        "agent": agent_metadata,
    }
    return manifest


def write_manifest(manifest: Dict[str, Any], output_dir: str) -> Path:
    """Persist the manifest to disk and return the path."""
    path = Path(output_dir) / "artifact_manifest.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    except Exception as exc:
        logger.error("Failed to write artifact manifest: %s", exc)
        raise
    return path
