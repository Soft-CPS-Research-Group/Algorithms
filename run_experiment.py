"""Unified entrypoint for running training experiments locally or inside Docker."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

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

    # Inject runtime paths expected by downstream helpers.
    log_dir = path_info["log_dir"]
    mlflow_uri = path_info["mlflow_dir"]

    metadata = config.get("metadata", {})
    runtime = config.setdefault("runtime", {})
    tracking = config.get("tracking", {})

    if not tracking.get("mlflow_enabled", True):
        logger.warning("MLflow disabled; metrics will be stored in local JSONL logs only.")

    logger.info(
        "Starting experiment '{}' (job_id={})",
        metadata.get("experiment_name"),
        job_id,
    )

    runtime["log_dir"] = str(log_dir)
    runtime["job_dir"] = str(path_info["job_dir"])
    runtime["mlflow_uri"] = f"file:{mlflow_uri}"
    runtime["job_id"] = job_id

    try:
        start_mlflow_run(config=config)
        run = mlflow.active_run()
        mlflow_enabled = tracking.get("mlflow_enabled", True)
        if mlflow_enabled and run is None:
            raise RuntimeError("MLflow run could not be started.")

        if run is None:
            run_id = f"local-{job_id}"
            run_name = metadata.get("run_name") or run_id
            logger.info("MLflow disabled; using local run id {}", run_id)
        else:
            run_id = run.info.run_id
            run_name = run.info.run_name
            logger.info("MLflow run started: name={}, id={}", run_name, run_id)

        runtime["run_id"] = run_id
        runtime["run_name"] = run_name

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
                "mlflow_uri": runtime["mlflow_uri"],
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
        log_file = log_dir / f"{run_id}.log"
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
        if mlflow.active_run():
            mlflow.log_artifact(str(resolved_config_path), artifact_path="artifacts")

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
        if mlflow.active_run():
            mlflow.log_artifact(str(manifest_path), artifact_path="artifacts")

        logger.info("Experiment complete")
    finally:
        end_mlflow_run()


if __name__ == "__main__":
    cli_main()
