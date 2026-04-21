"""Validation helpers for training-side bundle contract checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from loguru import logger

SUPPORTED_ARTIFACT_FORMATS = {"onnx", "rule_based", "offline_dataset", "behavioral_cloning"}


class BundleValidationError(ValueError):
    """Raised when a generated bundle does not satisfy the v1 contract."""


def validate_bundle_contract(manifest: Mapping[str, Any], artifacts_root: Path | str) -> None:
    """Validate manifest structure and exported artifact files.

    Raises:
        BundleValidationError: if the manifest/artifacts do not satisfy the v1 contract.
    """
    root = Path(artifacts_root).resolve()
    required_sections = [
        "manifest_version",
        "metadata",
        "simulator",
        "training",
        "topology",
        "algorithm",
        "environment",
        "agent",
    ]

    missing = [section for section in required_sections if section not in manifest]
    if missing:
        raise BundleValidationError(f"Manifest missing required sections: {missing}")

    if manifest.get("manifest_version") != 1:
        raise BundleValidationError(
            f"Unsupported manifest_version={manifest.get('manifest_version')!r}; expected 1"
        )

    metadata = _require_mapping(manifest.get("metadata"), field="metadata")
    for key in ("experiment_name", "run_name"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise BundleValidationError(f"metadata.{key} must be a non-empty string")

    alias_mapping_path = metadata.get("alias_mapping_path")
    if alias_mapping_path:
        alias_path = _resolve_bundle_path(root, alias_mapping_path, field="metadata.alias_mapping_path")
        if not alias_path.exists():
            raise BundleValidationError(
                f"metadata.alias_mapping_path points to a missing file: {alias_mapping_path}"
            )

    environment = _require_mapping(manifest.get("environment"), field="environment")
    _validate_environment(environment)

    topology = _require_mapping(manifest.get("topology"), field="topology")
    num_agents = topology.get("num_agents")
    if num_agents is not None:
        if not isinstance(num_agents, int) or num_agents < 1:
            raise BundleValidationError("topology.num_agents must be a positive integer when provided")

    agent = _require_mapping(manifest.get("agent"), field="agent")
    _validate_agent(agent=agent, root=root, num_agents=num_agents)

    logger.debug("Bundle contract validation succeeded for {}", root)


def _validate_environment(environment: Mapping[str, Any]) -> None:
    observation_names = environment.get("observation_names")
    encoders = environment.get("encoders")

    if not isinstance(observation_names, list) or not observation_names:
        raise BundleValidationError("environment.observation_names must be a non-empty list")
    if not isinstance(encoders, list) or not encoders:
        raise BundleValidationError("environment.encoders must be a non-empty list")

    for index, names in enumerate(observation_names):
        if not isinstance(names, list) or any(not isinstance(item, str) for item in names):
            raise BundleValidationError(f"environment.observation_names[{index}] must be a list[str]")

    action_names_by_agent = environment.get("action_names_by_agent")
    if action_names_by_agent is not None:
        if isinstance(action_names_by_agent, list):
            for idx, values in enumerate(action_names_by_agent):
                if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                    raise BundleValidationError(
                        f"environment.action_names_by_agent[{idx}] must be a list[str]"
                    )
        elif isinstance(action_names_by_agent, dict):
            for key, values in action_names_by_agent.items():
                if not isinstance(key, (str, int)):
                    raise BundleValidationError("environment.action_names_by_agent keys must be str/int")
                if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                    raise BundleValidationError(
                        f"environment.action_names_by_agent[{key!r}] must be a list[str]"
                    )
        else:
            raise BundleValidationError(
                "environment.action_names_by_agent must be either list[list[str]] or dict[str, list[str]]"
            )


def _validate_agent(agent: Mapping[str, Any], root: Path, num_agents: int | None) -> None:
    top_level_format = agent.get("format")
    if top_level_format not in SUPPORTED_ARTIFACT_FORMATS:
        raise BundleValidationError(
            f"agent.format must be one of {sorted(SUPPORTED_ARTIFACT_FORMATS)}, got {top_level_format!r}"
        )

    artifacts = agent.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BundleValidationError("agent.artifacts must be a non-empty list")

    seen_agent_indices: set[int] = set()

    for idx, raw_artifact in enumerate(artifacts):
        artifact = _require_mapping(raw_artifact, field=f"agent.artifacts[{idx}]")
        agent_index = artifact.get("agent_index")
        if not isinstance(agent_index, int) or agent_index < 0:
            raise BundleValidationError(f"agent.artifacts[{idx}].agent_index must be a non-negative integer")

        if agent_index in seen_agent_indices:
            raise BundleValidationError(f"Duplicate agent artifact for agent_index={agent_index}")
        seen_agent_indices.add(agent_index)

        artifact_path_value = artifact.get("path")
        if not isinstance(artifact_path_value, str) or not artifact_path_value.strip():
            raise BundleValidationError(f"agent.artifacts[{idx}].path must be a non-empty string")

        artifact_path = _resolve_bundle_path(root, artifact_path_value, field=f"agent.artifacts[{idx}].path")
        if not artifact_path.exists():
            raise BundleValidationError(f"Artifact file does not exist: {artifact_path_value}")
        if artifact_path.is_dir():
            raise BundleValidationError(f"Artifact path must reference a file, got directory: {artifact_path_value}")

        artifact_format = artifact.get("format") or top_level_format
        if artifact_format not in SUPPORTED_ARTIFACT_FORMATS:
            raise BundleValidationError(
                f"agent.artifacts[{idx}].format must be one of {sorted(SUPPORTED_ARTIFACT_FORMATS)}, "
                f"got {artifact_format!r}"
            )

        artifact_config = artifact.get("config")
        if artifact_config is not None and not isinstance(artifact_config, dict):
            raise BundleValidationError(f"agent.artifacts[{idx}].config must be an object when provided")

        if artifact_format == "onnx":
            if artifact_path.suffix.lower() != ".onnx":
                raise BundleValidationError(f"ONNX artifact must end with .onnx, got {artifact_path_value}")
        elif artifact_format == "rule_based":
            expected_name = f"policy_agent_{agent_index}.json"
            if artifact_path.name != expected_name:
                raise BundleValidationError(
                    f"rule_based artifact for agent {agent_index} must be named {expected_name}, "
                    f"got {artifact_path.name}"
                )
        elif artifact_format == "offline_dataset":
            if artifact_path.suffix.lower() != ".csv":
                raise BundleValidationError(
                    f"offline_dataset artifact must end with .csv, got {artifact_path_value}"
                )

        if num_agents is not None and agent_index >= num_agents:
            raise BundleValidationError(
                f"agent.artifacts[{idx}].agent_index={agent_index} is outside topology.num_agents={num_agents}"
            )

    if (
        num_agents is not None
        and top_level_format not in ("offline_dataset", "behavioral_cloning")
        and len(artifacts) != num_agents
    ):
        raise BundleValidationError(
            "Number of exported artifacts must match topology.num_agents "
            f"(artifacts={len(artifacts)}, num_agents={num_agents})"
        )


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BundleValidationError(f"{field} must be an object")
    return value


def _resolve_bundle_path(root: Path, raw_path: str, *, field: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (root / path).resolve()

    if not path.is_absolute() and resolved != root and root not in resolved.parents:
        raise BundleValidationError(f"{field} points outside bundle root: {raw_path!r}")

    return resolved
