"""Common result, provenance and model-bundle finalization."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import citylearn
from loguru import logger

from algorithms.runtime.atomic import atomic_json
from algorithms.runtime.context import JobContext, OutputPlan
from algorithms.runtime.simulator import _resolve_agent_observation_dimensions
from algorithms.runtime.tracking import _log_curated_artifacts, mlflow
from utils.artifact_manifest import build_manifest, write_manifest
from utils.bundle_validator import validate_bundle_contract
from utils.experiment_protocol import build_pairing_fingerprint
from utils.training_provenance import build_provenance, digest, file_digest, seal_manifest


def finalize_outputs(
    context: JobContext, wrapper: Any, agent: Any, algorithm_name: str | None
) -> dict[str, Any]:
    started = time.perf_counter()
    config, path_info = context.config, context.paths
    job_id, run_id, run_name = context.job_id, context.run_id, context.run_name
    experiment_id, tracking_uri = context.experiment_id, context.tracking_uri
    mlflow_enabled, artifact_profile = context.mlflow_enabled, context.artifact_profile
    simulator_cfg = config["simulator"]
    export_cfg = dict(simulator_cfg.get("export") or {})
    for key in ("kpis_final_episode_only", "timeseries_final_episode_only"):
        if export_cfg.get(key) is None:
            export_cfg[key] = export_cfg.get("final_episode_only", False)
    resolved_config_path = path_info["resolved_config_path"]
    output_plan = OutputPlan.from_config(config)
    kpi_source = (
        "simulator_export"
        if export_cfg.get("export_kpis_on_episode_end", False)
        else "not_collected"
    )
    if kpi_source == "simulator_export":
        logger.info("KPI extraction delegated to simulator export files.")
    else:
        logger.info(
            "No KPI extraction performed during run; use exported artifacts for ad hoc comparison."
        )

    wrapper_reward_metadata = (
        wrapper.get_wrapper_reward_metadata()
        if hasattr(wrapper, "get_wrapper_reward_metadata")
        else {
            "enabled": bool((simulator_cfg.get("wrapper_reward") or {}).get("enabled", False)),
            "profile": str(
                (simulator_cfg.get("wrapper_reward") or {}).get("profile", "cost_limits_v1")
            ),
            "version": None,
            "clip_enabled": bool(
                (simulator_cfg.get("wrapper_reward") or {}).get("clip_enabled", True)
            ),
            "clip_min": (simulator_cfg.get("wrapper_reward") or {}).get("clip_min", -10.0),
            "clip_max": (simulator_cfg.get("wrapper_reward") or {}).get("clip_max", 10.0),
            "squash": str((simulator_cfg.get("wrapper_reward") or {}).get("squash", "none")),
        }
    )
    environment_metadata = (
        wrapper.describe_environment() if hasattr(wrapper, "describe_environment") else {}
    )
    reward_function_metadata = (
        environment_metadata.get("reward_function", {})
        if isinstance(environment_metadata, dict)
        else {}
    )

    experiment_protocol = config.get("experiment_protocol")
    pairing_fingerprint = (
        build_pairing_fingerprint(
            config,
            simulator_version=str(getattr(citylearn, "__version__", "unknown")),
        )
        if experiment_protocol
        else None
    )
    episode_execution = (
        wrapper.get_episode_execution_history()
        if hasattr(wrapper, "get_episode_execution_history")
        else []
    )
    result_payload = {
        "status": "finalizing",
        "kpi_source": kpi_source,
        "export_kpis_on_episode_end": bool(export_cfg.get("export_kpis_on_episode_end", False)),
        "export_final_episode_only": bool(export_cfg.get("final_episode_only", False)),
        "export_kpis_final_episode_only": bool(
            export_cfg.get("kpis_final_episode_only", export_cfg.get("final_episode_only", False))
        ),
        "export_timeseries_final_episode_only": bool(
            export_cfg.get(
                "timeseries_final_episode_only", export_cfg.get("final_episode_only", False)
            )
        ),
        "export_include_business_as_usual": bool(export_cfg.get("include_business_as_usual", True)),
        "export_business_as_usual_timeseries": bool(
            export_cfg.get("export_business_as_usual_timeseries", True)
        ),
        "simulation_data_dir": str(path_info["simulation_data_dir"]),
        "reward_function": reward_function_metadata,
        "wrapper_reward_enabled": bool(wrapper_reward_metadata.get("enabled", False)),
        "wrapper_reward_profile": wrapper_reward_metadata.get("profile"),
        "wrapper_reward_version": wrapper_reward_metadata.get("version"),
        "experiment_protocol": experiment_protocol,
        "pairing_fingerprint": pairing_fingerprint,
        "episode_execution": episode_execution,
        "evaluated_policy": getattr(wrapper, "evaluated_policy", None),
        "randomness": config.get("runtime", {}).get("randomness"),
        "episode_plan": config.get("runtime", {}).get("episode_plan"),
    }

    result_path = path_info["result_path"]
    artifacts_root = path_info["bundle_dir"].resolve()
    # Dynamic entity mode can change agent cardinality at runtime.
    # Persist final topology before exporting artifacts/manifest.
    final_topology = config.setdefault("topology", {})
    final_topology["observation_dimensions"] = _resolve_agent_observation_dimensions(
        wrapper, algorithm_name
    )
    final_topology["action_dimensions"] = wrapper.action_dimension
    final_topology["num_agents"] = len(wrapper.action_space)

    environment_metadata = wrapper.describe_environment()
    if output_plan.bundle:
        agent_metadata = agent.export_artifacts(
            output_dir=str(artifacts_root),
            context={
                "topology": config.get("topology", {}),
                "environment": environment_metadata,
                "config": config,
            },
        )
        manifest = build_manifest(config, environment_metadata, agent_metadata)
        seal_manifest(manifest, artifacts_root, resolved_config_path, job_id)
        validate_bundle_contract(manifest, artifacts_root)
        manifest_path = write_manifest(manifest, str(artifacts_root))
        logger.info("Artifact manifest written to {}", manifest_path)
        result_payload["provenance"] = manifest["provenance"]
        result_payload["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    else:
        provenance = build_provenance(config, environment_metadata, {"artifacts": []})
        provenance["job_id"] = job_id
        provenance["resolved_config_sha256"] = file_digest(resolved_config_path)
        provenance["identity_sha256"] = digest(provenance)
        manifest = {"provenance": provenance}
        manifest_path = None
        result_payload["provenance"] = provenance
        result_payload["manifest_sha256"] = None
    result_payload["output_contract_version"] = "algorithms.outputs.v1"
    result_payload["model_artifacts"] = (config.get("outputs") or {}).get(
        "model_artifacts"
    ) or "configured"
    atomic_json(result_path, result_payload)
    logger.info("Result payload written to {}", result_path)

    summary_payload = {
        "job_id": job_id,
        "run_id": run_id,
        "run_name": run_name,
        "experiment_id": experiment_id,
        "tracking_uri": tracking_uri,
        "mlflow_enabled": mlflow_enabled,
        "artifact_profile": artifact_profile,
        "provenance": manifest["provenance"],
        "manifest_sha256": result_payload["manifest_sha256"],
        "kpi_metric_count": 0,
        "bundle_dir": str(path_info["bundle_dir"]),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "experiment_protocol": experiment_protocol,
        "pairing_fingerprint_sha256": (
            pairing_fingerprint.get("sha256") if isinstance(pairing_fingerprint, Mapping) else None
        ),
    }
    summary_path = path_info["summary_path"]
    atomic_json(summary_path, summary_payload)

    if mlflow.active_run() and artifact_profile in {"curated", "full"}:
        _log_curated_artifacts(
            resolved_config_path=resolved_config_path,
            manifest_path=Path(manifest_path) if manifest_path is not None else None,
            result_path=result_path,
            summary_path=summary_path,
        )

    atomic_json(
        path_info["results_dir"] / "finalization.json",
        {
            "output_seconds": time.perf_counter() - started,
            "checkpoint_seconds": getattr(
                getattr(wrapper, "checkpoint_manager", None), "total_seconds", 0.0
            ),
        },
    )
    # The caller commits completion after tracking and resource finalization.
    result_payload["status"] = "completed"
    return result_payload


def commit_completion(context, wrapper, result_payload):
    path_info, simulator_cfg = context.paths, context.config["simulator"]
    completed_episodes = int(simulator_cfg.get("episodes", 1) or 1)
    final_step_total = max(int(getattr(wrapper, "episode_time_steps", 0) or 0) - 1, 0)
    final_step = max(0, final_step_total - 1)
    final_global_step = max(0, int(getattr(wrapper, "global_step", 0) or 0))
    final_global_step_total = (
        completed_episodes * final_step_total if final_step_total > 0 else None
    )
    atomic_json(path_info["result_path"], result_payload)
    if getattr(wrapper, "progress_tracker", None) is not None:
        wrapper.progress_tracker.update(
            episode=max(0, completed_episodes - 1),
            step=final_step,
            global_step=final_global_step,
            episode_total=completed_episodes,
            step_total=final_step_total if final_step_total > 0 else None,
            global_step_total=final_global_step_total,
            status="completed",
            force_complete=True,
        )

    return result_payload
