"""Construction checks through the same simulator and agent interfaces as a job."""

from __future__ import annotations

import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from algorithms.registry import agent_spec, import_symbol


def validate_environment_layout(config: Mapping[str, Any], wrapper: Any) -> list[dict[str, Any]]:
    from algorithms.runtime.simulator import _resolve_agent_observation_dimensions

    reports = []
    cardinality = len(wrapper.action_space)
    for index, stage in enumerate(config["pipeline"]):
        name = stage["algorithm"]
        cap = agent_spec(name).capabilities
        count = stage.get("count", 1)
        if count > 1 and count != cardinality:
            raise ValueError(
                f"pipeline[{index}] count={count} does not match {cardinality} simulator members"
            )
        if cap.single_building and count == 1 and cardinality != 1:
            raise ValueError(
                f"{name} controls one building; use count={cardinality} to create an Ensemble"
            )
        action_dims = list(wrapper.action_dimension)
        validator = agent_spec(name).environment_validator
        if validator:
            import_symbol(validator)(stage, wrapper)
        reports.append(
            {
                "algorithm": name,
                "count": count,
                "observation_dimensions": _resolve_agent_observation_dimensions(wrapper, name),
                "action_dimensions": action_dims,
                "capabilities": asdict(cap),
            }
        )
    return reports


def preflight_job(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read/reset the configured dataset; never step, optimize or publish outputs."""
    from algorithms.configuration.project import validate_config
    from algorithms.runtime.controller import SimulationController
    from algorithms.runtime.simulator import SimulatorAdapter, create_environment

    resolved = validate_config(dict(config)).to_dict()
    checked = deepcopy(resolved)
    checked.setdefault("tracking", {})["mlflow_enabled"] = False
    checked["simulator"]["export"] = {
        **checked["simulator"].get("export", {}),
        "mode": "none",
        "export_kpis_on_episode_end": False,
    }
    checkpoints = checked.get("checkpointing", {})
    local_paths = [
        checkpoints.get("checkpoint_local_path"),
        *(checkpoints.get("stage_checkpoint_local_paths") or {}).values(),
    ]
    for path in local_paths:
        if path and not Path(path).expanduser().exists():
            raise FileNotFoundError(f"Checkpoint source does not exist: {path}")
    for stage in checked["pipeline"]:
        parameters = stage.get("hyperparameters") or {}
        preparation = parameters.get("offline_training") or {}
        for label, path in (
            ("offline dataset", preparation.get("dataset_path")),
            ("offline model", parameters.get("model_dir")),
        ):
            if path and not Path(path).expanduser().exists():
                raise FileNotFoundError(f"Missing {label}: {path}")
    env = None
    with tempfile.TemporaryDirectory(prefix="algorithms-preflight-") as temporary:
        try:
            checked.setdefault("runtime", {}).update({"job_dir": temporary, "log_dir": temporary})
            env = create_environment(checked, temporary)
            wrapper = SimulationController(env, config=checked)
            stages = validate_environment_layout(checked, wrapper)
            return {
                "status": "environment_validated",
                "schema_version": resolved["schema_version"],
                "profile": resolved["job"]["profile"],
                "stages": stages,
                "simulation_steps": 0,
                "training_updates": 0,
                "remote_checkpoint": "not_contacted"
                if checkpoints.get("checkpoint_run_id")
                else None,
            }
        finally:
            if env is not None:
                SimulatorAdapter(env).close()
