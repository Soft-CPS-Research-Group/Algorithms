"""Job intentions and capability checks, without loading numerical agents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from algorithms.registry import agent_spec


def resolve_profile(raw: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(raw))
    job = config.get("job") or {}
    if not isinstance(job, dict):
        return config
    profile = job.get("profile", "train")
    simulator = config.setdefault("simulator", {})
    checkpointing = config.setdefault("checkpointing", {})
    outputs = config.get("outputs") or {}
    if (
        not isinstance(outputs, dict)
        or not isinstance(simulator, dict)
        or not isinstance(checkpointing, dict)
    ):
        return config
    if profile == "evaluate":
        simulator["episodes"] = 1
        simulator["deterministic_finish"] = True
        for stage in config.get("pipeline") or []:
            if not isinstance(stage, dict):
                continue
            if not isinstance(stage.get("hyperparameters", {}), dict):
                raise ValueError("Agent hyperparameters must be a mapping")
            if (stage.get("hyperparameters") or {}).get("offline_training"):
                raise ValueError("evaluate jobs cannot request offline_training; provide model_dir")
            stage["frozen"] = True
        checkpointing["checkpoint_interval"] = None
        checkpointing["checkpoint_on_episode_end"] = False
    elif profile == "smoke":
        simulator["episodes"] = 2
        simulator["deterministic_finish"] = True
        horizon = simulator.get("episode_time_steps")
        if isinstance(horizon, int):
            simulator["episode_time_steps"] = max(2, min(horizon, 5))
        elif isinstance(horizon, list) and horizon:
            start, end = horizon[0]
            simulator["episode_time_steps"] = [[start, min(end, start + 4)]] * 2
        else:
            simulator["episode_time_steps"] = 5
        if not isinstance(config.setdefault("tracking", {}), dict):
            raise ValueError("tracking must be a mapping")
        config["tracking"]["mlflow_enabled"] = False
        for stage in config.get("pipeline") or []:
            if not isinstance(stage, dict):
                continue
            if not isinstance(stage.get("hyperparameters", {}), dict):
                raise ValueError("Agent hyperparameters must be a mapping")
            offline = (stage.get("hyperparameters") or {}).get("offline_training")
            if isinstance(offline, dict):
                parameters = offline.setdefault("parameters", {})
                if not isinstance(parameters, dict):
                    raise ValueError("offline_training.parameters must be a mapping")
                length_key = "epochs" if stage.get("algorithm") == "BCAgent" else "gradient_steps"
                parameters[length_key] = min(int(parameters.get(length_key, 2)), 2)
    requested = outputs.get("model_artifacts")
    if requested in {"none", "bundle"}:
        checkpointing["checkpoint_interval"] = None
        checkpointing["checkpoint_on_episode_end"] = False
    return config


def validate_capabilities(project: Any) -> None:
    stages = project.pipeline
    for index, stage in enumerate(stages):
        spec = agent_spec(stage.algorithm)
        cap = spec.capabilities
        if project.simulator.interface not in cap.interfaces:
            raise ValueError(f"{stage.algorithm} requires simulator.interface in {cap.interfaces}")
        if cap.final_stage_only and index != len(stages) - 1:
            raise ValueError(f"{stage.algorithm} must be the final pipeline stage")
        if cap.standalone_only and len(stages) != 1:
            raise ValueError(f"{stage.algorithm} requires a standalone single-stage pipeline")
        if cap.single_controller and stage.count != 1:
            raise ValueError(f"{stage.algorithm} requires count=1")
        if project.job.profile == "evaluate" and cap.learning != "none":
            parameters = stage.model_dump().get("hyperparameters") or {}
            checkpoint = project.checkpointing
            has_source = bool(parameters.get("model_dir")) or bool(
                checkpoint.resume_training
                and (
                    checkpoint.checkpoint_local_path
                    or checkpoint.checkpoint_run_id
                    or index in (checkpoint.stage_checkpoint_local_paths or {})
                )
            )
            if not has_source:
                raise ValueError(
                    f"evaluate requires a trained model source for {stage.algorithm}: "
                    "enable checkpointing.resume_training with a checkpoint source, or provide model_dir"
                )
    requested = project.outputs.model_artifacts
    if requested in {"checkpoint", "both"} and not any(
        agent_spec(stage.algorithm).capabilities.checkpoint_save for stage in stages
    ):
        raise ValueError("Requested checkpoint output, but no stage supports checkpoint saving")
