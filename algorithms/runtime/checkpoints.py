"""Shared checkpoints service extracted from the original job entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger

from algorithms.agents.base_agent import BaseAgent
from algorithms.execution_unit import ExecutionUnit
from algorithms.pipeline import Pipeline
from algorithms.runtime.checkpoint_index import discover_legacy, resolve_index
from algorithms.runtime.tracking import mlflow
from utils.experiment_protocol import file_sha256


def _build_checkpoint_artifact_candidates(checkpoint_artifact: str) -> list[str]:
    """Return ordered MLflow artifact paths with legacy fallback."""
    artifact_name = (
        str(checkpoint_artifact or "latest_checkpoint.pth").strip() or "latest_checkpoint.pth"
    )
    candidates = [f"checkpoints/{artifact_name}", artifact_name]
    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduplicated.append(candidate)
            seen.add(candidate)
    return deduplicated


def _agent_supports_checkpoint_loading(agent: ExecutionUnit) -> bool:
    """Return whether the unit overrides the default ``load_checkpoint``."""
    load_checkpoint = getattr(type(agent), "load_checkpoint", None)
    if load_checkpoint is None:
        return False
    return load_checkpoint not in {
        ExecutionUnit.load_checkpoint,
        BaseAgent.load_checkpoint,
    }


def _resolve_best_checkpoint_run_id(
    agent: ExecutionUnit,
    *,
    experiment_name: Optional[str],
) -> Optional[str]:
    """Resolve best checkpoint run id through an agent hook when available."""
    if not experiment_name:
        return None

    get_best_checkpoint = getattr(agent, "get_best_checkpoint", None)
    if not callable(get_best_checkpoint):
        raise RuntimeError(
            "checkpointing.use_best_checkpoint_artifact=true requires the agent to implement "
            "`get_best_checkpoint(experiment_name)`."
        )
    return get_best_checkpoint(experiment_name)


def _download_checkpoint_from_mlflow(
    *,
    checkpoint_run_id: str,
    checkpoint_artifact: str,
    tracking_uri: str,
    download_dir: Path,
) -> Path:
    """Download checkpoint artifact from MLflow using current API with fallback paths."""
    # Explicit reads are independent of this job's logging policy.
    from algorithms.runtime.tracking import OptionalMLflow

    if isinstance(mlflow, OptionalMLflow):
        from importlib import import_module

        import_module("mlflow").set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(tracking_uri)
    artifact_candidates = _build_checkpoint_artifact_candidates(checkpoint_artifact)
    last_error: Optional[Exception] = None

    for artifact_path in artifact_candidates:
        try:
            downloaded = mlflow.artifacts.download_artifacts(
                run_id=checkpoint_run_id,
                artifact_path=artifact_path,
                dst_path=str(download_dir),
            )
            resolved = Path(downloaded)
            logger.info(
                "Resolved checkpoint artifact '{}' from run {} -> {}",
                artifact_path,
                checkpoint_run_id,
                resolved,
            )
            return resolved
        except Exception as exc:  # pragma: no cover - error details are asserted via raised message
            last_error = exc
            logger.warning(
                "Checkpoint artifact '{}' not available in run {}: {}",
                artifact_path,
                checkpoint_run_id,
                exc,
            )

    attempted = ", ".join(artifact_candidates)
    raise RuntimeError(
        f"Could not download checkpoint from MLflow run '{checkpoint_run_id}'. "
        f"Tried artifact paths: {attempted}."
    ) from last_error


def _resolve_local_checkpoint_path(
    *,
    checkpoint_artifact: str,
    checkpoints_dir: Path,
) -> Path:
    """Resolve a local checkpoint path using standard and fallback locations."""
    artifact = (
        str(checkpoint_artifact or "latest_checkpoint.pth").strip() or "latest_checkpoint.pth"
    )
    # Current jobs write to <job>/checkpoints.  The legacy wrapper wrote to
    # <job>/logs/checkpoints, so keep that root readable for old runs.
    checkpoint_roots = [
        checkpoints_dir,
        checkpoints_dir.parent / "logs" / "checkpoints",
    ]
    for root in checkpoint_roots:
        indexed = resolve_index(root) if root.exists() else None
        if indexed:
            return indexed[1]
        legacy = discover_legacy(root, artifact) if root.exists() else root
        if legacy.is_file():
            return legacy
    candidate_paths = [root / artifact for root in checkpoint_roots]
    candidate_paths.append(Path(artifact))
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()

    # Composite single-agent baselines persist one checkpoint per Ensemble
    # member under checkpoints/agent_<index>/.  Resolve the root so the
    # Ensemble can route each file to the matching local learner.
    for checkpoint_root in checkpoint_roots:
        if checkpoint_root.exists() and any(checkpoint_root.glob("agent_*")):
            return checkpoint_root.resolve()

    attempted = ", ".join(str(path) for path in candidate_paths)
    raise RuntimeError(f"Could not resolve local checkpoint path. Tried: {attempted}.")


def _resume_agent_from_checkpoint(
    *,
    agent: ExecutionUnit,
    config: dict[str, Any],
    tracking_uri: str,
    checkpoints_dir: Path,
) -> Optional[Path]:
    """Resume training by loading an agent checkpoint if requested in config."""
    checkpoint_cfg = config.get("checkpointing", {}) or {}
    if not bool(checkpoint_cfg.get("resume_training", False)):
        return None

    if not _agent_supports_checkpoint_loading(agent):
        raise RuntimeError(
            f"resume_training=true but agent '{agent.__class__.__name__}' does not implement load_checkpoint()."
        )

    stage_checkpoint_paths = checkpoint_cfg.get("stage_checkpoint_local_paths") or {}
    if stage_checkpoint_paths:
        if not isinstance(agent, Pipeline):
            raise RuntimeError(
                "checkpointing.stage_checkpoint_local_paths requires a Pipeline execution unit."
            )
        for raw_index, raw_path in sorted(
            stage_checkpoint_paths.items(), key=lambda item: int(item[0])
        ):
            stage_index = int(raw_index)
            checkpoint_path = Path(str(raw_path)).expanduser().resolve()
            if not checkpoint_path.exists():
                raise RuntimeError(
                    "Configured stage checkpoint path does not exist for "
                    f"stage {stage_index}: {checkpoint_path}"
                )
            agent.load_stage_checkpoint(stage_index, str(checkpoint_path))
            logger.info(
                "Pipeline stage {} resumed from checkpoint {}",
                stage_index,
                checkpoint_path,
            )
        return None

    checkpoint_artifact = str(checkpoint_cfg.get("checkpoint_artifact", "latest_checkpoint.pth"))
    checkpoint_run_id = checkpoint_cfg.get("checkpoint_run_id")
    checkpoint_local_path = checkpoint_cfg.get("checkpoint_local_path")
    if bool(checkpoint_cfg.get("use_best_checkpoint_artifact", False)) and not checkpoint_run_id:
        metadata_cfg = config.get("metadata", {}) or {}
        checkpoint_run_id = _resolve_best_checkpoint_run_id(
            agent,
            experiment_name=metadata_cfg.get("experiment_name"),
        )
        logger.info("Resolved best checkpoint run id: {}", checkpoint_run_id)

    if checkpoint_run_id:
        checkpoint_path = _download_checkpoint_from_mlflow(
            checkpoint_run_id=str(checkpoint_run_id),
            checkpoint_artifact=checkpoint_artifact,
            tracking_uri=tracking_uri,
            download_dir=checkpoints_dir,
        )
    elif checkpoint_local_path:
        checkpoint_path = Path(str(checkpoint_local_path)).expanduser().resolve()
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Configured checkpoint_local_path does not exist: {checkpoint_path}"
            )
    else:
        checkpoint_path = _resolve_local_checkpoint_path(
            checkpoint_artifact=checkpoint_artifact,
            checkpoints_dir=checkpoints_dir,
        )

    checkpoint_path = discover_legacy(checkpoint_path, checkpoint_artifact)
    protocol = config.get("experiment_protocol") or {}
    expected_checkpoint_sha256 = protocol.get("selected_checkpoint_sha256")
    if expected_checkpoint_sha256:
        checkpoint_file = checkpoint_path
        if checkpoint_file.is_dir():
            checkpoint_file = checkpoint_file / checkpoint_artifact
        if not checkpoint_file.is_file():
            raise RuntimeError(
                "Cannot verify selected checkpoint hash because the resolved "
                f"checkpoint is not a file: {checkpoint_file}"
            )
        actual_checkpoint_sha256 = file_sha256(checkpoint_file)
        if actual_checkpoint_sha256 != str(expected_checkpoint_sha256):
            raise RuntimeError(
                "Selected checkpoint SHA-256 mismatch: "
                f"expected {expected_checkpoint_sha256}, got {actual_checkpoint_sha256}"
            )

    agent.load_checkpoint(str(checkpoint_path))
    logger.info("Agent '{}' resumed from checkpoint {}", agent.__class__.__name__, checkpoint_path)
    return checkpoint_path
