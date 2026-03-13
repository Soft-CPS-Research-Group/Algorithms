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
from algorithms.registry import create_agent
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

    metadata = config.get("metadata", {})
    runtime = config.setdefault("runtime", {})
    tracking = config.get("tracking", {})
    artifact_profile = str(tracking.get("mlflow_artifacts_profile", "minimal")).strip().lower() or "minimal"
    mlflow_enabled = bool(tracking.get("mlflow_enabled", True))

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
        with open(path_info["progress_path"], "w", encoding="utf-8") as progress_file:
            json.dump(
                {
                    "episode": 0,
                    "step": 0,
                    "global_step": 0,
                    "status": "initializing",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
                progress_file,
                indent=2,
            )
        with open(path_info["result_path"], "w", encoding="utf-8") as result_file:
            json.dump({"status": "pending"}, result_file, indent=2)

        # Configure loguru file sink after run metadata is available.
        log_file = path_info["log_dir"] / f"{run_id}.log"
        log_level = tracking.get("log_level", "INFO").upper()
        logger.add(str(log_file), level=log_level)
        logger.info("Logging to {}", log_file)

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

        logger.info("Starting training loop")
        wrapper.learn()

        logger.info("Evaluating environment KPIs")
        kpis = wrapper.env.evaluate()
        kpis = kpis.pivot(index="cost_function", columns="name", values="value").round(3).dropna(how="all")

        kpi_metrics = {}
        if hasattr(kpis, "stack"):
            for (cost_function, name), value in kpis.stack(dropna=True).items():
                kpi_metrics[f"kpi_{cost_function}_{name}"] = float(value)
        else:
            for kpi_name, kpi_value in kpis.items():
                kpi_metrics[f"kpi_{kpi_name}"] = float(kpi_value)

        if mlflow.active_run():
            for kpi_name, kpi_value in kpi_metrics.items():
                mlflow.log_metric(kpi_name, kpi_value)
        elif getattr(wrapper, "local_metrics_logger", None):
            if kpi_metrics:
                wrapper.local_metrics_logger.log(kpi_metrics, -2)

        result_path = path_info["result_path"]
        with open(result_path, "w", encoding="utf-8") as result_file:
            json.dump(kpis.to_dict(), result_file, indent=2)
        logger.info("KPI summary written to {}", result_path)

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


if __name__ == "__main__":
    cli_main()
