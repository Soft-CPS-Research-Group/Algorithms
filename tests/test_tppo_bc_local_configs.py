"""Contract for the local TPPO behavior-cloning pretraining canary."""
from __future__ import annotations

from pathlib import Path

import yaml

from utils.config_schema import validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CANARY_PATH = (
    REPO_ROOT
    / "configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_canary.yaml"
)
SMOKE_PATH = (
    REPO_ROOT
    / "configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_smoke.yaml"
)


def test_local_bc_pretrain_canary_has_bounded_cpu_pretraining_contract() -> None:
    with CANARY_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    validate_config(config)

    simulator = config["simulator"]
    stage = config["pipeline"][0]
    behavior_cloning = stage["behavior_cloning"]

    assert simulator["episodes"] == 3
    assert simulator["simulation_start_time_step"] == 0
    assert simulator["simulation_end_time_step"] == 15
    assert simulator["episode_time_steps"] == 16
    assert config["tracking"]["stall_watchdog_enabled"] is True
    assert config["tracking"]["stall_watchdog_timeout_seconds"] == 60.0
    assert config["training"]["steps_between_training_updates"] == 8
    assert stage["hyperparameters"]["require_cuda"] is False
    assert stage["hyperparameters"]["minibatch_size"] == 8
    assert behavior_cloning["demonstration_episodes"] == 1
    assert behavior_cloning["max_samples_per_building"] == 16
    assert behavior_cloning["pretraining_epochs"] == 1
    assert behavior_cloning["batch_size"] == 4


def test_local_bc_pretrain_smoke_has_bounded_cpu_pretraining_contract() -> None:
    with SMOKE_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    validate_config(config)

    simulator = config["simulator"]
    stage = config["pipeline"][0]
    behavior_cloning = stage["behavior_cloning"]

    assert config["metadata"]["run_name"] == "tppo-recovery-wa-tppo-bc-pretrain-local-smoke-s7"
    assert simulator["episodes"] == 3
    assert simulator["simulation_start_time_step"] == 0
    assert simulator["simulation_end_time_step"] == 191
    assert simulator["episode_time_steps"] == 192
    assert config["tracking"]["stall_watchdog_enabled"] is True
    assert config["tracking"]["stall_watchdog_timeout_seconds"] == 120.0
    assert stage["hyperparameters"]["require_cuda"] is False
    assert stage["hyperparameters"]["minibatch_size"] == 16
    assert behavior_cloning["demonstration_episodes"] == 1
    assert behavior_cloning["max_samples_per_building"] == 128
    assert behavior_cloning["pretraining_epochs"] == 1
    assert behavior_cloning["batch_size"] == 16
