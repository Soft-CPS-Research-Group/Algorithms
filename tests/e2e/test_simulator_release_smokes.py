"""Short real-run compatibility smokes for a newly pinned Simulator release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from run_experiment import run_experiment


pytestmark = pytest.mark.slow
REPO_ROOT = Path(__file__).resolve().parents[2]
HORIZON = 16


def _common_smoke_config(template: Path, *, name: str) -> dict[str, Any]:
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["metadata"]["experiment_name"] = "simulator_release_smoke"
    config["metadata"]["run_name"] = name
    config.setdefault("runtime", {})
    tracking = config.setdefault("tracking", {})
    tracking["mlflow_enabled"] = False
    tracking["progress_updates_enabled"] = False
    tracking["system_metrics_enabled"] = False
    tracking["runtime_profiling_enabled"] = False
    checkpointing = config.setdefault("checkpointing", {})
    checkpointing["resume_training"] = False
    checkpointing["checkpoint_interval"] = None
    simulator = config["simulator"]
    simulator["episodes"] = 1
    simulator["deterministic_finish"] = False
    simulator["simulation_start_time_step"] = 0
    simulator["simulation_end_time_step"] = HORIZON - 1
    simulator["episode_time_steps"] = HORIZON
    export = simulator.setdefault("export", {})
    export["mode"] = "end"
    export["export_kpis_on_episode_end"] = True
    export["final_episode_only"] = True
    export["include_business_as_usual"] = False
    export["export_business_as_usual_timeseries"] = False
    export["session_name"] = name
    return config


def _run(tmp_path: Path, config: Mapping[str, Any], *, job_id: str) -> Path:
    config_path = tmp_path / f"{job_id}.yaml"
    config_path.write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )
    run_experiment(config_path=str(config_path), job_id=job_id, base_dir=tmp_path)
    job_dir = tmp_path / "jobs" / job_id
    result = job_dir / "results" / "result.json"
    summary = job_dir / "results" / "summary.json"
    manifest = job_dir / "bundle" / "artifact_manifest.json"
    assert result.is_file()
    assert summary.is_file()
    assert manifest.is_file()
    json.loads(result.read_text(encoding="utf-8"))
    json.loads(summary.read_text(encoding="utf-8"))
    json.loads(manifest.read_text(encoding="utf-8"))
    return job_dir


def test_rbc_smart_static_release_smoke(tmp_path):
    config = _common_smoke_config(
        REPO_ROOT / "configs/templates/baselines/rbc_smart_15min_local.yaml",
        name="rbc-smart-release-smoke",
    )
    _run(tmp_path, config, job_id="rbc-smart")


def test_matd3_static_release_smoke(tmp_path):
    config = _common_smoke_config(
        REPO_ROOT / "configs/templates/rl/matd3_15min_residual_local.yaml",
        name="matd3-release-smoke",
    )
    stage = config["pipeline"][0]
    stage["networks"]["actor"]["layers"] = [32, 32]
    critic = stage["networks"]["critic"]
    critic["layers"] = [32, 16]
    critic["state_layers"] = [32]
    critic["action_layers"] = [16]
    critic["joint_layers"] = [32, 16]
    replay = stage["replay_buffer"]
    replay["capacity"] = 128
    replay["batch_size"] = 8
    exploration = stage["exploration"]["params"]
    exploration["use_amp"] = False
    exploration["end_initial_exploration_time_step"] = 4
    exploration["random_exploration_steps"] = 4
    exploration["warm_start_policy_phaseout_steps"] = 0
    exploration["train_during_initial_exploration"] = True
    exploration["initial_exploration_training_start_step"] = 0
    _run(tmp_path, config, job_id="matd3")


def test_individual_ppo_static_release_smoke(tmp_path):
    config = _common_smoke_config(
        REPO_ROOT
        / "configs/templates/rl/ppo_distributed_local_total_energy_bc_smoke.yaml",
        name="ppo-release-smoke",
    )
    stage = config["pipeline"][0]
    stage["networks"]["actor"]["layers"] = [32, 32]
    stage["networks"]["critic"]["layers"] = [32, 32]
    stage["replay_buffer"]["capacity"] = HORIZON
    stage["replay_buffer"]["batch_size"] = 8
    exploration = stage["exploration"]["params"]
    exploration["rollout_length"] = 8
    exploration["minibatch_size"] = 8
    exploration["ppo_epochs"] = 1
    exploration["actor_behavior_cloning_extra_updates"] = 0
    _run(tmp_path, config, job_id="ppo")
