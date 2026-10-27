"""Unified entrypoint for running training experiments locally or inside Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

_STARTUP_TRACE_ENABLED = (
    os.environ.get("OPEVA_STARTUP_TRACE", "1").strip().lower() not in {"0", "false", "no", "off"}
    and os.path.basename(sys.argv[0]) == "run_experiment.py"
)
_STARTUP_TRACE_T0 = time.monotonic()


def _startup_trace(message: str) -> None:
    if not _STARTUP_TRACE_ENABLED:
        return
    elapsed = time.monotonic() - _STARTUP_TRACE_T0
    print(f"[opeva-startup +{elapsed:.3f}s] {message}", file=sys.stderr, flush=True)


_startup_trace("module import started")

import citylearn
import numpy as np
import yaml
from loguru import logger
from pydantic import ValidationError

from algorithms.runtime.tracking import mlflow
from utils.training_provenance import seal_manifest

_startup_trace("third-party imports loaded")

_startup_trace("before CityLearnEnv import")
from citylearn.citylearn import CityLearnEnv

_startup_trace("after CityLearnEnv import")
_startup_trace("before reward registry import")
from reward_function.registry import (
    REWARD_FUNCTION_MAP,
    get_available_reward_function_names,
)

_startup_trace("after reward registry import")
_startup_trace("before base agent import")
from algorithms.agents.base_agent import BaseAgent

_startup_trace("after base agent import")
_startup_trace("before execution unit import")
from algorithms.execution_unit import ExecutionUnit

_startup_trace("after execution unit import")
from algorithms.pipeline import Pipeline

_startup_trace("before algorithms registry import")
from algorithms.registry import (
    ENCODED_OBSERVATION_ALGORITHMS,
    build_execution_unit,
    build_unsupported_algorithm_message,
)

_startup_trace("after algorithms registry import")
_startup_trace("before helper imports")
from utils.helpers import set_default_config
from utils.mlflow_helper import end_mlflow_run, start_mlflow_run
from utils.pipeline_utils import summarise_pipeline_algorithms

_startup_trace("after helper imports")
_startup_trace("before wrapper import")
from algorithms.runtime.controller import SimulationController as Wrapper

_startup_trace("after wrapper import")
_startup_trace("before artifact imports")
from utils.artifact_manifest import build_manifest, write_manifest
from utils.bundle_validator import validate_bundle_contract
from utils.config_schema import validate_config
from utils.experiment_protocol import build_pairing_fingerprint, file_sha256

_startup_trace("after artifact imports")

_startup_trace("project imports loaded")

from algorithms.runtime.checkpoints import (
    _agent_supports_checkpoint_loading,
    _build_checkpoint_artifact_candidates,
    _download_checkpoint_from_mlflow,
    _resolve_best_checkpoint_run_id,
    _resolve_local_checkpoint_path,
    _resume_agent_from_checkpoint,
)
from algorithms.runtime.cli import build_argument_parser, cli_main
from algorithms.runtime.context import JobContext, OutputPlan
from algorithms.runtime.outputs import commit_completion, finalize_outputs
from algorithms.runtime.paths import (
    _derive_job_id,
    _mark_failed_outputs,
    _prepare_paths,
    _write_resolved_config,
)
from algorithms.runtime.preflight import validate_environment_layout
from algorithms.runtime.simulator import (
    _apply_simulation_schema_overlays,
    _community_market_overlay,
    _load_electrical_service_overrides,
    _resolve_agent_observation_dimensions,
    _resolve_citylearn_schema_input,
    _validate_dynamic_entity_schema_input,
    create_environment,
)
from algorithms.runtime.tracking import (
    _build_mlflow_run_url,
    _build_mlflow_tags,
    _compute_config_hash,
    _discover_git_sha,
    _log_curated_artifacts,
    _resolve_tracking_uri,
)

DEFAULT_LOCAL_BASE = Path(os.environ.get("OPEVA_BASE_DIR", "./runs"))


def run_experiment(config_path: str, job_id: Optional[str], base_dir: Path) -> None:
    """Own the complete attempt, including configuration failures and cancellation."""
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "The reliable job runtime requires Python >= 3.10; the Python 3.8 Jetson image remains a legacy platform pending migration."
        )
    from algorithms.runtime.lifecycle import cancellation_signals

    resolved_id = _derive_job_id(job_id)
    paths = _prepare_paths(Path(base_dir).resolve(), resolved_id)
    try:
        with cancellation_signals():
            result = _run_experiment(config_path, resolved_id, Path(base_dir), paths)
        return result
    except BaseException as exc:
        try:
            _mark_failed_outputs(paths, exc)
        except OSError:
            logger.exception("Could not persist attempt failure")
        raise


def _run_experiment(config_path, job_id, base_dir, path_info):
    """Execute training with outputs written under ``base_dir``/jobs/<job_id>."""
    job_id = _derive_job_id(job_id)
    base_dir = base_dir.resolve()
    _startup_trace(f"run_experiment entered job_id={job_id} base_dir={base_dir}")

    if isinstance(config_path, Mapping):
        raw_config = deepcopy(dict(config_path))
    else:
        with open(config_path, "r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}
    _startup_trace("config yaml loaded")
    from algorithms.export.portable import encode
    from algorithms.runtime.atomic import atomic_json

    atomic_json(path_info["job_dir"] / "config.initial.json", encode(raw_config))

    try:
        config_model = validate_config(raw_config)
    except ValidationError as exc:
        messages = "\n".join(
            f"- {' -> '.join(str(item) for item in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        logger.error("Configuration validation failed:\n{}", messages)
        raise SystemExit(1) from exc
    _startup_trace("config validated")

    config = config_model.to_dict()
    from algorithms.runtime.randomness import initialize_job

    random_metadata = initialize_job(config)
    _startup_trace(f"paths prepared job_dir={path_info['job_dir']}")
    pipeline_cfg = config.get("pipeline") or []
    if not pipeline_cfg:
        message = build_unsupported_algorithm_message(None)
        logger.error(message)
        raise ValueError(message)
    # Derive the algorithm name used for observation-dimension resolution.
    # Prefer the first neural stage (needs encoded obs); fall back to first stage.
    algorithm_name: Optional[str] = next(
        (
            stage.get("algorithm")
            for stage in pipeline_cfg
            if isinstance(stage, dict) and stage.get("algorithm") in ENCODED_OBSERVATION_ALGORITHMS
        ),
        pipeline_cfg[0].get("algorithm")
        if pipeline_cfg and isinstance(pipeline_cfg[0], dict)
        else None,
    )

    metadata = config.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    job_name_override = (os.environ.get("OPEVA_JOB_NAME") or "").strip()
    if job_name_override:
        metadata["run_name"] = job_name_override
        config["metadata"] = metadata
    runtime = config.setdefault("runtime", {})
    runtime["randomness"] = random_metadata
    from algorithms.configuration.episodes import episode_plan

    runtime["episode_plan"] = episode_plan(config)
    tracking = config.get("tracking", {})
    artifact_profile = (
        str(tracking.get("mlflow_artifacts_profile", "minimal")).strip().lower() or "minimal"
    )
    mlflow_enabled = bool(tracking.get("mlflow_enabled", False))
    log_level = str(tracking.get("log_level", "INFO")).upper()
    active_log_file = path_info["log_dir"] / f"{job_id}.log"
    # Keep a single canonical log file per job (`<job_id>.log`).
    # The worker already captures container stdout/stderr separately, so
    # removing default sinks avoids duplicate lines in the same file.
    logger.remove()
    file_sink_id = logger.add(str(active_log_file), level=log_level)
    _startup_trace(f"loguru file sink configured log_file={active_log_file}")
    logger.info("Logging bootstrap to {}", active_log_file)

    config_hash = _compute_config_hash(config)
    git_sha = _discover_git_sha(Path(__file__).resolve().parents[2])
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

    env = None
    wrapper = None
    from algorithms.runtime.tracking import tracking_session

    tracking_context = tracking_session(config, path_info)
    tracking_state = tracking_context.__enter__()
    try:
        start_mlflow_run(config=config)
        run = mlflow.active_run()
        if mlflow_enabled and run is None:
            if tracking_state["failed"]:
                mlflow_enabled = False
            else:
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
            logger.info(
                "MLflow run started: name={}, id={}, experiment_id={}",
                run_name,
                run_id,
                experiment_id,
            )
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
        runtime["mlflow_run_url"] = _build_mlflow_run_url(
            mlflow_ui_base_url, experiment_id, run_id if run else None
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
                "bundle_dir": str(path_info["bundle_dir"]),
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
        configured_state_points = simulator_cfg.get("episode_time_steps")
        configured_step_total = (
            configured_state_points - 1 if isinstance(configured_state_points, int) else None
        )
        if not isinstance(configured_step_total, int) or configured_step_total <= 0:
            configured_step_total = None
        configured_global_step_total = (
            configured_episode_total * configured_step_total
            if configured_step_total is not None
            else None
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

        # Environment construction is shared with preflight and research utilities.
        simulator_cfg = config["simulator"]
        env = create_environment(config, path_info["simulation_data_dir"], env_factory=CityLearnEnv)

        wrapper = Wrapper(
            env=env,
            config=config,
            job_id=job_id,
            progress_path=str(path_info["progress_path"]),
        )

        # Populate derived dimensions required by neural agents.
        set_default_config(
            config,
            ["topology", "observation_dimensions"],
            _resolve_agent_observation_dimensions(wrapper, algorithm_name),
        )
        set_default_config(config, ["topology", "action_dimensions"], wrapper.action_dimension)
        set_default_config(config, ["topology", "num_agents"], len(wrapper.action_space))
        logger.debug(
            "Topology derived: num_agents={}, obs_dims={}, action_dims={}",
            config.get("topology", {}).get("num_agents"),
            config.get("topology", {}).get("observation_dimensions"),
            config.get("topology", {}).get("action_dimensions"),
        )
        set_default_config(config, ["simulator", "reward_function_kwargs"], {})
        set_default_config(config, ["simulator", "interface"], "flat")
        set_default_config(config, ["simulator", "topology_mode"], "static")
        set_default_config(
            config,
            ["simulator", "entity_encoding"],
            {
                "enabled": bool(
                    str(config.get("simulator", {}).get("interface", "flat")).strip().lower()
                    == "entity"
                ),
                "normalization": "minmax_space",
                "clip": True,
            },
        )
        set_default_config(
            config,
            ["simulator", "wrapper_reward"],
            {
                "enabled": False,
                "profile": "cost_limits_v1",
                "clip_enabled": True,
                "clip_min": -10.0,
                "clip_max": 10.0,
                "squash": "none",
            },
        )

        resolved_config_path = path_info["resolved_config_path"]
        _write_resolved_config(config, resolved_config_path)
        logger.info("Resolved runtime config written to {}", resolved_config_path)

        validate_environment_layout(config, wrapper)
        agent = build_execution_unit(config=config)
        from algorithms.runtime.resume import leaves

        runtime["randomness"]["agents"] = [
            getattr(unit, "runtime_identity", {}) for _, unit in leaves(agent)
        ]
        prepare_training = getattr(agent, "prepare_training", None)
        if callable(prepare_training):
            from algorithms.runtime.randomness import call_hook

            call_hook(
                agent,
                "prepare_training",
                str(path_info["job_dir"] / "offline_training"),
                context={"config": config},
            )
        wrapper.set_model(agent)
        checkpoint_path = _resume_agent_from_checkpoint(
            agent=agent,
            config=config,
            tracking_uri=tracking_uri,
            checkpoints_dir=path_info["checkpoints_dir"],
        )

        from algorithms.runtime.checkpoint_index import resolve_index
        from algorithms.runtime.resume import capture_runtime, restore_runtime

        if getattr(wrapper, "checkpoint_manager", None) is not None:
            wrapper.checkpoint_manager.runtime_state = lambda: capture_runtime(wrapper)
        if config["checkpointing"].get("resume_mode") == "episode":
            import torch

            source = config["checkpointing"].get("checkpoint_local_path")
            indexed = resolve_index(Path(source)) if source else None
            if indexed is None:
                raise ValueError("Episode resume requires a local indexed checkpoint")
            state = torch.load(
                indexed[2] / "runtime_state.pt", map_location="cpu", weights_only=False
            )
            restore_runtime(wrapper, state)
        if checkpoint_path is not None:
            runtime["loaded_checkpoint"] = str(checkpoint_path)
        _write_resolved_config(config, resolved_config_path)

        logger.info("Starting training loop")
        wrapper.learn(
            episodes=int(simulator_cfg.get("episodes", 1) or 1),
            deterministic_finish=bool(simulator_cfg.get("deterministic_finish", False)),
        )

        # A completed training run must leave a reusable representation of its
        # final state even when the horizon is not an exact checkpoint interval.
        # Without this, a successful run can silently expose an older periodic
        # checkpoint as ``latest_checkpoint.pth``.
        checkpoint_manager = getattr(wrapper, "checkpoint_manager", None)
        save_final_checkpoint = getattr(checkpoint_manager, "save_final", None)
        if getattr(wrapper, "_final_training_checkpoint", None) is not None:
            pass
        elif OutputPlan.from_config(config).explicit_checkpoint:
            if checkpoint_manager is None:
                raise RuntimeError("Checkpoint output requested without a checkpoint service")
            checkpoint_path = checkpoint_manager.save(
                agent, max(0, int(getattr(wrapper, "global_step", 0) or 0))
            )
            if checkpoint_path is None or not Path(checkpoint_path).exists():
                raise RuntimeError("Requested final checkpoint was not produced by the agent")
        elif callable(save_final_checkpoint):
            save_final_checkpoint(
                agent=agent,
                step=max(0, int(getattr(wrapper, "global_step", 0) or 0)),
            )

        context = JobContext(
            config,
            path_info,
            job_id,
            run_id,
            run_name,
            experiment_id,
            tracking_uri,
            mlflow_enabled,
            artifact_profile,
        )
        result_payload = finalize_outputs(context, wrapper, agent, algorithm_name)

        logger.info("Finalizing experiment resources")
    except Exception as exc:
        logger.exception("Experiment failed")
        try:
            _mark_failed_outputs(path_info, exc)
        except OSError:
            logger.exception("Could not persist failure outputs")
        raise
    finally:
        # Run every finalizer even if a previous one fails. Cleanup must never
        # hide the original training/export failure from the orchestrator.
        finalizers = []
        if wrapper is not None:
            for name in ("_cancel_stall_watchdog", "_close_stall_watchdog_file"):
                cleanup = getattr(wrapper, name, None)
                if callable(cleanup):
                    finalizers.append(cleanup)
        if env is not None:
            close_environment = getattr(env, "close", None)
            if callable(close_environment):
                finalizers.append(close_environment)
        finalizers.append(end_mlflow_run)
        for cleanup in finalizers:
            try:
                cleanup()
            except Exception:
                logger.exception("Resource cleanup failed: {}", cleanup)
        try:
            tracking_context.__exit__(*sys.exc_info())
        finally:
            try:
                logger.remove(file_sink_id)
            except ValueError:
                pass
    commit_completion(context, wrapper, result_payload)


if __name__ == "__main__":
    cli_main()
