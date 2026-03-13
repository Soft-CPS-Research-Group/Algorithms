"""Unified entrypoint for running training experiments locally or inside Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mlflow
import yaml
from loguru import logger
from pydantic import ValidationError

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from reward_function.V2G_Reward import V2GPenaltyReward
from algorithms.agents.base_agent import BaseAgent
from algorithms.registry import (
    build_unsupported_algorithm_message,
    create_agent,
    is_algorithm_supported,
)
from utils.helpers import set_default_config
from utils.mlflow_helper import end_mlflow_run, start_mlflow_run
from utils.wrapper_citylearn import Wrapper_CityLearn as Wrapper
from utils.artifact_manifest import build_manifest, write_manifest
from utils.bundle_validator import validate_bundle_contract
from utils.config_schema import validate_config

# Available reward functions keyed by config name
REWARD_FUNCTION_MAP = {
    "V2GPenaltyReward": V2GPenaltyReward,
    "RewardFunction": RewardFunction,
}

DEFAULT_LOCAL_BASE = Path(os.environ.get("OPEVA_BASE_DIR", "./runs"))


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without binding defaults for portability."""
    parser = argparse.ArgumentParser(description="Run a CityLearn training experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--job_id",
        dest="job_id",
        help="Unique ID used to organise output artefacts",
    )
    parser.add_argument(
        "--base-dir",
        help="Base directory for logs/results/mlflow artefacts (defaults to env OPEVA_BASE_DIR or ./runs)",
    )
    return parser


def cli_main(default_base: Optional[str] = None) -> None:
    """Parse CLI arguments and run the experiment."""
    parser = build_argument_parser()
    args = parser.parse_args()

    base_dir = Path(args.base_dir or default_base or DEFAULT_LOCAL_BASE).resolve()
    run_experiment(config_path=args.config, job_id=args.job_id, base_dir=base_dir)


def _derive_job_id(job_id: Optional[str]) -> str:
    if job_id:
        return job_id
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _prepare_paths(base_dir: Path, job_id: str) -> dict[str, Path]:
    job_dir = base_dir / "jobs" / job_id
    results_dir = job_dir / "results"
    progress_dir = job_dir / "progress"
    paths = {
        "job_dir": job_dir,
        "log_dir": job_dir / "logs",
        "checkpoints_dir": job_dir / "checkpoints",
        "onnx_dir": job_dir / "onnx_models",
        "results_dir": results_dir,
        "simulation_data_dir": results_dir / "simulation_data",
        "progress_dir": progress_dir,
        "result_path": results_dir / "result.json",
        "summary_path": results_dir / "summary.json",
        "progress_path": progress_dir / "progress.json",
        "mlflow_dir": base_dir / "mlflow" / "mlruns",
        "job_info_path": job_dir / "job_info.json",
        "artifact_manifest_path": job_dir / "artifact_manifest.json",
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
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


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
    for key in ("log_dir", "job_dir", "job_id", "run_id", "run_name", "experiment_id", "mlflow_run_url"):
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
    except Exception:
        return None


def _build_mlflow_run_url(base_url: Optional[str], experiment_id: Optional[str], run_id: Optional[str]) -> Optional[str]:
    if not base_url or not experiment_id or not run_id:
        return None
    normalized = base_url.rstrip("/")
    return f"{normalized}/#/experiments/{experiment_id}/runs/{run_id}"


def _build_mlflow_tags(config: dict[str, Any], *, job_id: str, run_name: str, config_hash: str, git_sha: Optional[str]) -> dict[str, str]:
    simulator_cfg = config.get("simulator", {})
    dataset_name = simulator_cfg.get("dataset_name") or simulator_cfg.get("dataset_path") or "unknown_dataset"
    algorithm_name = (config.get("algorithm", {}) or {}).get("name", "unknown_algorithm")
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
    manifest_path: Path,
    result_path: Path,
    summary_path: Path,
) -> None:
    if not mlflow.active_run():
        return
    for artifact in (resolved_config_path, manifest_path, result_path, summary_path):
        if artifact.exists():
            mlflow.log_artifact(str(artifact), artifact_path="artifacts")


def _build_checkpoint_artifact_candidates(checkpoint_artifact: str) -> list[str]:
    """Return ordered MLflow artifact paths with legacy fallback."""
    artifact_name = str(checkpoint_artifact or "latest_checkpoint.pth").strip() or "latest_checkpoint.pth"
    candidates = [f"checkpoints/{artifact_name}", artifact_name]
    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduplicated.append(candidate)
            seen.add(candidate)
    return deduplicated


def _agent_supports_checkpoint_loading(agent: BaseAgent) -> bool:
    """Return whether the agent overrides ``BaseAgent.load_checkpoint``."""
    load_checkpoint = getattr(type(agent), "load_checkpoint", None)
    if load_checkpoint is None:
        return False
    return load_checkpoint is not BaseAgent.load_checkpoint


def _resolve_best_checkpoint_run_id(
    agent: BaseAgent,
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
    artifact = str(checkpoint_artifact or "latest_checkpoint.pth").strip() or "latest_checkpoint.pth"
    candidate_paths = [
        checkpoints_dir / artifact,
        Path(artifact),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.resolve()

    attempted = ", ".join(str(path) for path in candidate_paths)
    raise RuntimeError(f"Could not resolve local checkpoint path. Tried: {attempted}.")


def _resume_agent_from_checkpoint(
    *,
    agent: BaseAgent,
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

    checkpoint_artifact = str(checkpoint_cfg.get("checkpoint_artifact", "latest_checkpoint.pth"))
    checkpoint_run_id = checkpoint_cfg.get("checkpoint_run_id")
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
    else:
        checkpoint_path = _resolve_local_checkpoint_path(
            checkpoint_artifact=checkpoint_artifact,
            checkpoints_dir=checkpoints_dir,
        )

    agent.load_checkpoint(str(checkpoint_path))
    logger.info("Agent '{}' resumed from checkpoint {}", agent.__class__.__name__, checkpoint_path)
    return checkpoint_path


def run_experiment(config_path: str, job_id: Optional[str], base_dir: Path) -> None:
    """Execute training with outputs written under ``base_dir``/jobs/<job_id>."""
    job_id = _derive_job_id(job_id)
    base_dir = base_dir.resolve()
    path_info = _prepare_paths(base_dir, job_id)

    with open(config_path, "r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    try:
        config_model = validate_config(raw_config)
    except ValidationError as exc:
        messages = "\n".join(
            f"- {' -> '.join(str(item) for item in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        logger.error("Configuration validation failed:\n{}", messages)
        raise SystemExit(1) from exc

    config = config_model.to_dict()
    algorithm_name = (config.get("algorithm", {}) or {}).get("name")
    if not is_algorithm_supported(algorithm_name):
        message = build_unsupported_algorithm_message(algorithm_name)
        logger.error(message)
        raise ValueError(message)

    metadata = config.get("metadata", {})
    runtime = config.setdefault("runtime", {})
    tracking = config.get("tracking", {})
    artifact_profile = str(tracking.get("mlflow_artifacts_profile", "minimal")).strip().lower() or "minimal"
    mlflow_enabled = bool(tracking.get("mlflow_enabled", True))
    log_level = str(tracking.get("log_level", "INFO")).upper()
    active_log_file = path_info["log_dir"] / f"{job_id}.log"
    file_sink_id = logger.add(str(active_log_file), level=log_level)
    logger.info("Logging bootstrap to {}", active_log_file)

    config_hash = _compute_config_hash(config)
    git_sha = _discover_git_sha(Path(__file__).resolve().parent)
    tracking_uri = _resolve_tracking_uri(runtime, path_info["mlflow_dir"])
    mlflow_ui_base_url = (os.environ.get("MLFLOW_UI_BASE_URL") or "").strip() or None

    if not mlflow_enabled:
        logger.warning("MLflow disabled; metrics will be stored in local JSONL logs only.")

    logger.info(
        "Starting experiment '{}' (job_id={})",
        metadata.get("experiment_name"),
        job_id,
    )

    runtime["log_dir"] = str(path_info["log_dir"])
    runtime["job_dir"] = str(path_info["job_dir"])
    runtime["mlflow_uri"] = tracking_uri
    runtime["tracking_uri"] = tracking_uri
    runtime["job_id"] = job_id

    try:
        start_mlflow_run(config=config)
        run = mlflow.active_run()
        if mlflow_enabled and run is None:
            raise RuntimeError("MLflow run could not be started.")

        experiment_id: Optional[str] = None
        if run is None:
            run_id = f"local-{job_id}"
            run_name = metadata.get("run_name") or run_id
            logger.info("MLflow disabled; using local run id {}", run_id)
        else:
            run_id = run.info.run_id
            run_name = run.info.run_name or metadata.get("run_name") or run_id
            experiment_id = run.info.experiment_id
            logger.info("MLflow run started: name={}, id={}, experiment_id={}", run_name, run_id, experiment_id)
            mlflow.set_tags(
                _build_mlflow_tags(
                    config,
                    job_id=job_id,
                    run_name=run_name,
                    config_hash=config_hash,
                    git_sha=git_sha,
                )
            )

        runtime["run_id"] = run_id
        runtime["run_name"] = run_name
        runtime["experiment_id"] = experiment_id
        runtime["mlflow_run_url"] = _build_mlflow_run_url(mlflow_ui_base_url, experiment_id, run_id if run else None)

        target_log_file = path_info["log_dir"] / f"{run_id}.log"
        if target_log_file != active_log_file:
            try:
                if target_log_file.exists():
                    target_log_file.unlink()
                if active_log_file.exists():
                    active_log_file.replace(target_log_file)
                active_log_file = target_log_file
            except OSError as exc:
                logger.warning(
                    "Could not rename log file {} -> {}: {}",
                    active_log_file,
                    target_log_file,
                    exc,
                )
        logger.info("Logging to {}", active_log_file)

        # Persist job metadata for orchestrators/consumers.
        job_info_path = path_info["job_info_path"]
        job_info = {}
        if job_info_path.exists():
            with open(job_info_path, "r", encoding="utf-8") as job_file:
                job_info = json.load(job_file)
        job_info.update(
            {
                "job_id": job_id,
                "job_dir": str(path_info["job_dir"]),
                "run_id": run_id,
                "run_name": run_name,
                "mlflow_run_id": run_id if mlflow_enabled else None,
                "experiment_id": experiment_id,
                "mlflow_experiment_id": experiment_id,
                "tracking_uri": tracking_uri,
                "mlflow_uri": tracking_uri,
                "mlflow_run_url": runtime["mlflow_run_url"],
                "mlflow_enabled": mlflow_enabled,
            }
        )
        with open(job_info_path, "w", encoding="utf-8") as job_file:
            json.dump(job_info, job_file, indent=2)

        # Materialise expected per-job output files early for orchestration parity.
        simulator_cfg = config.get("simulator", {})
        configured_episode_total = int(simulator_cfg.get("episodes", 1) or 1)
        configured_step_total = simulator_cfg.get("episode_time_steps")
        if not isinstance(configured_step_total, int) or configured_step_total <= 0:
            configured_step_total = None
        configured_global_step_total = (
            configured_episode_total * configured_step_total if configured_step_total is not None else None
        )

        progress_payload: dict[str, Any] = {
            "episode": 0,
            "episode_current": 0,
            "episode_total": configured_episode_total,
            "step": 0,
            "step_current": 0,
            "global_step": 0,
            "status": "initializing",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "progress_pct": 0.0,
        }
        if configured_step_total is not None:
            progress_payload["step_total"] = configured_step_total
        if configured_global_step_total is not None:
            progress_payload["global_step_total"] = configured_global_step_total

        with open(path_info["progress_path"], "w", encoding="utf-8") as progress_file:
            json.dump(progress_payload, progress_file, indent=2)
        with open(path_info["result_path"], "w", encoding="utf-8") as result_file:
            json.dump({"status": "pending"}, result_file, indent=2)

        # Build environment and agent stack.
        reward_key = config["simulator"]["reward_function"]
        reward_cls = REWARD_FUNCTION_MAP.get(reward_key)
        if reward_cls is None:
            raise ValueError(f"Unsupported reward function '{reward_key}'.")

        simulator_cfg = config["simulator"]
        export_cfg = simulator_cfg.get("export", {})
        env_kwargs = {
            "schema": simulator_cfg["dataset_path"],
            "central_agent": simulator_cfg["central_agent"],
            "reward_function": reward_cls,
            "render_mode": export_cfg.get("mode", "none"),
            "export_kpis_on_episode_end": export_cfg.get("export_kpis_on_episode_end", False),
            "render_directory": str(path_info["simulation_data_dir"]),
        }
        if export_cfg.get("session_name"):
            env_kwargs["render_session_name"] = export_cfg["session_name"]

        for key in ("simulation_start_time_step", "simulation_end_time_step", "episode_time_steps"):
            value = simulator_cfg.get(key)
            if value is not None:
                env_kwargs[key] = value

        env = CityLearnEnv(**env_kwargs)

        wrapper = Wrapper(
            env=env,
            config=config,
            job_id=job_id,
            progress_path=str(path_info["progress_path"]),
        )

        # Populate derived dimensions required by MADDPG.
        set_default_config(config, ["topology", "observation_dimensions"], wrapper.observation_dimension)
        set_default_config(config, ["topology", "action_dimensions"], wrapper.action_dimension)
        set_default_config(config, ["topology", "num_agents"], len(wrapper.action_space))
        logger.debug(
            "Topology derived: num_agents={}, obs_dims={}, action_dims={}",
            config.get("topology", {}).get("num_agents"),
            config.get("topology", {}).get("observation_dimensions"),
            config.get("topology", {}).get("action_dimensions"),
        )

        resolved_config_path = path_info["resolved_config_path"]
        _write_resolved_config(config, resolved_config_path)
        logger.info("Resolved runtime config written to {}", resolved_config_path)

        agent = create_agent(config=config)
        wrapper.set_model(agent)
        _resume_agent_from_checkpoint(
            agent=agent,
            config=config,
            tracking_uri=tracking_uri,
            checkpoints_dir=path_info["checkpoints_dir"],
        )

        logger.info("Starting training loop")
        wrapper.learn(episodes=int(simulator_cfg.get("episodes", 1) or 1))

        kpi_metrics = {}
        if export_cfg.get("export_kpis_on_episode_end", False):
            result_payload = {
                "status": "completed",
                "kpi_source": "simulator_export",
                "export_kpis_on_episode_end": True,
                "simulation_data_dir": str(path_info["simulation_data_dir"]),
            }
            logger.info("KPI evaluation delegated to simulator export; skipping explicit env.evaluate().")
        else:
            result_payload = {
                "status": "completed",
                "kpi_source": "disabled",
                "message": (
                    "KPI evaluation is disabled. Enable simulator.export.export_kpis_on_episode_end "
                    "to export KPIs directly from the environment."
                ),
                "simulation_data_dir": str(path_info["simulation_data_dir"]),
            }
            logger.info("KPI evaluation disabled by config; skipping explicit env.evaluate().")

        result_path = path_info["result_path"]
        with open(result_path, "w", encoding="utf-8") as result_file:
            json.dump(result_payload, result_file, indent=2)
        logger.info("Result payload written to {}", result_path)

        artifacts_root = path_info["job_dir"].resolve()
        environment_metadata = wrapper.describe_environment()
        agent_metadata = agent.export_artifacts(
            output_dir=str(artifacts_root),
            context={
                "topology": config.get("topology", {}),
                "environment": environment_metadata,
                "config": config,
            },
        )
        manifest = build_manifest(config, environment_metadata, agent_metadata)
        validate_bundle_contract(manifest, artifacts_root)
        manifest_path = write_manifest(manifest, str(artifacts_root))
        logger.info("Artifact manifest written to {}", manifest_path)

        summary_payload = {
            "job_id": job_id,
            "run_id": run_id,
            "run_name": run_name,
            "experiment_id": experiment_id,
            "tracking_uri": tracking_uri,
            "mlflow_enabled": mlflow_enabled,
            "artifact_profile": artifact_profile,
            "kpi_metric_count": len(kpi_metrics),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        summary_path = path_info["summary_path"]
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary_payload, summary_file, indent=2)

        if mlflow.active_run() and artifact_profile == "curated":
            _log_curated_artifacts(
                resolved_config_path=resolved_config_path,
                manifest_path=Path(manifest_path),
                result_path=result_path,
                summary_path=summary_path,
            )

        logger.info("Experiment complete")
    finally:
        end_mlflow_run()
        try:
            logger.remove(file_sink_id)
        except ValueError:
            pass


if __name__ == "__main__":
    cli_main()
