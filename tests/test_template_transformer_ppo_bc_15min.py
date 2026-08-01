"""Contract tests for the real 15-minute Transformer-PPO BC smoke template."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
from algorithms.agents import agent_transformer_ppo
from algorithms.registry import build_execution_unit
from utils.config_schema import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    REPO_ROOT
    / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_smoke.yaml"
)
DATASET_PATH = (
    REPO_ROOT
    / "datasets/citylearn_three_phase_dynamic_assets_only_demo_15min_parquet/schema.json"
)
SERVER_TEMPLATE_CASES = (
    (
        "week",
        REPO_ROOT
        / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_week.yaml",
        672,
        672,
    ),
    (
        "month",
        REPO_ROOT
        / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_month.yaml",
        2880,
        2880,
    ),
    (
        "year",
        REPO_ROOT
        / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_year.yaml",
        34816,
        35039,
    ),
)


def _load_template() -> dict:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_server_template(config: dict) -> dict:
    normalized = deepcopy(config)
    normalized["metadata"].pop("experiment_name")
    normalized["metadata"].pop("run_name")
    normalized["tracking"].pop("tags")
    normalized["simulator"]["export"].pop("session_name")
    behavior_cloning = normalized["pipeline"][0]["behavior_cloning"]
    behavior_cloning.pop("decay_steps")
    behavior_cloning["warm_start"].pop("phaseout_steps")
    return normalized


def test_smoke_template_has_valid_15min_bc_contract() -> None:
    config = _load_template()
    validate_config(config)

    metadata = config["metadata"]
    simulator = config["simulator"]
    training = config["training"]
    stage = config["pipeline"][0]
    transformer = stage["transformer"]
    hyperparameters = stage["hyperparameters"]
    behavior_cloning = stage["behavior_cloning"]
    warm_start = behavior_cloning["warm_start"]

    assert "15min" in metadata["experiment_name"].lower()
    assert "smoke" in metadata["experiment_name"].lower()
    assert "15min" in metadata["run_name"].lower()
    assert "smoke" in metadata["run_name"].lower()

    assert (
        simulator["dataset_name"]
        == "citylearn_three_phase_dynamic_assets_only_demo_15min_parquet"
    )
    assert (
        simulator["dataset_path"]
        == "./datasets/citylearn_three_phase_dynamic_assets_only_demo_15min_parquet/schema.json"
    )
    assert simulator["interface"] == "entity"
    assert simulator["topology_mode"] == "dynamic"
    assert simulator["entity_encoding"] == {
        "enabled": True,
        "normalization": "minmax_space",
        "clip": True,
    }
    assert simulator["reward_function"] == "CostHardConstraintReward"
    assert simulator["reward_function_kwargs"] == {}
    assert simulator["episodes"] == 1
    assert simulator["simulation_start_time_step"] == 5184
    assert simulator["simulation_end_time_step"] == 5248
    assert simulator["episode_time_steps"] == 65
    assert simulator["topology_event_time_offset"] == -5184
    assert simulator["export"]["mode"] == "end"
    assert simulator["export"]["final_episode_only"] is True
    assert simulator["export"]["include_business_as_usual"] is False
    assert simulator["export"]["export_business_as_usual_timeseries"] is False

    assert training["steps_between_training_updates"] == 16
    assert training["target_update_interval"] == 0
    assert config["tracking"]["mlflow_enabled"] is False
    assert config["tracking"]["log_level"] == "INFO"
    assert config["tracking"]["mlflow_artifacts_profile"] == "minimal"
    assert config["tracking"]["action_diagnostics_enabled"] is True
    assert config["tracking"]["training_diagnostics_enabled"] is True
    assert config["tracking"]["reward_diagnostics_enabled"] is True
    assert config["checkpointing"]["resume_training"] is False
    assert config["checkpointing"]["checkpoint_interval"] is None

    assert stage["algorithm"] == "AgentTransformerPPO"
    assert transformer["d_model"] == 32
    assert transformer["nhead"] == 4
    assert transformer["num_layers"] == 1
    assert transformer["dim_feedforward"] == 64
    assert hyperparameters["gamma"] == pytest.approx(0.99)
    assert hyperparameters["gae_lambda"] == pytest.approx(0.95)
    assert hyperparameters["clip_eps"] == pytest.approx(0.2)
    assert hyperparameters["minibatch_size"] == 16
    assert hyperparameters["ppo_epochs"] == 1

    assert behavior_cloning["enabled"] is True
    assert behavior_cloning["weight"] == pytest.approx(0.42)
    assert behavior_cloning["min_weight"] == pytest.approx(0.0)
    assert behavior_cloning["decay_start_step"] == 0
    assert behavior_cloning["decay_steps"] == 64
    assert behavior_cloning["ev_multiplier"] == pytest.approx(24.0)
    assert behavior_cloning["storage_multiplier"] == pytest.approx(0.18)
    assert warm_start["policy"] == "RBCCommunityPolicy"
    assert warm_start["deterministic"] is True
    assert warm_start["noise_scale"] == pytest.approx(0.0)
    assert warm_start["phaseout_steps"] == 64
    assert warm_start["phaseout_mode"] == "blend"


def test_smoke_dataset_is_dynamic_15min_and_window_shifts_event_to_local_step() -> None:
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    config = _load_template()
    simulator = config["simulator"]

    assert dataset["seconds_per_time_step"] == 900
    assert dataset["simulation_start_time_step"] == 0
    assert dataset["simulation_end_time_step"] == 35039
    assert dataset["topology_mode"] == "dynamic"
    assert any(event["time_step"] == 5200 for event in dataset["topology_events"])
    source_event = next(
        event for event in dataset["topology_events"] if event["time_step"] == 5200
    )
    assert source_event["time_step"] + simulator["topology_event_time_offset"] == 16


def test_smoke_template_builds_bc_agent_without_environment_attach() -> None:
    agent = build_execution_unit(_load_template())

    assert isinstance(agent, AgentTransformerPPO)
    assert agent._bc is not None
    assert agent.requires_raw_observation_context is True


@pytest.mark.parametrize(
    ("duration_name", "template_path", "bc_decay_steps", "phaseout_steps"),
    SERVER_TEMPLATE_CASES,
    ids=[case[0] for case in SERVER_TEMPLATE_CASES],
)
def test_server_template_has_full_year_15min_bc_contract(
    duration_name: str,
    template_path: Path,
    bc_decay_steps: int,
    phaseout_steps: int,
) -> None:
    config = _load_yaml(template_path)
    validate_config(config)

    metadata = config["metadata"]
    simulator = config["simulator"]
    export = simulator["export"]
    tracking = config["tracking"]
    checkpointing = config["checkpointing"]
    training = config["training"]
    stage = config["pipeline"][0]
    transformer = stage["transformer"]
    hyperparameters = stage["hyperparameters"]
    behavior_cloning = stage["behavior_cloning"]
    warm_start = behavior_cloning["warm_start"]

    for value in (metadata["experiment_name"], metadata["run_name"]):
        assert "15min" in value.lower()
        assert duration_name in value.lower()
    assert tracking["tags"]["teacher_duration"] == duration_name

    assert (
        simulator["dataset_name"]
        == "citylearn_three_phase_dynamic_assets_only_demo_15min_parquet"
    )
    assert (
        simulator["dataset_path"]
        == "./datasets/citylearn_three_phase_dynamic_assets_only_demo_15min_parquet/schema.json"
    )
    assert simulator["interface"] == "entity"
    assert simulator["topology_mode"] == "dynamic"
    assert simulator["entity_encoding"] == {
        "enabled": True,
        "normalization": "minmax_space",
        "clip": True,
    }
    assert simulator["reward_function"] == "CostHardConstraintReward"
    assert simulator["reward_function_kwargs"] == {}
    assert simulator["episodes"] == 1
    assert simulator["simulation_start_time_step"] == 0
    assert simulator["simulation_end_time_step"] == 35039
    assert simulator["episode_time_steps"] == 35040
    assert simulator.get("topology_event_time_offset", 0) == 0

    assert export["mode"] == "end"
    assert export["final_episode_only"] is True
    assert export["kpis_final_episode_only"] is True
    assert export["timeseries_final_episode_only"] is True
    assert export["include_business_as_usual"] is True
    assert export["export_business_as_usual_timeseries"] is False
    assert duration_name in export["session_name"].lower()

    assert tracking["mlflow_enabled"] is True
    assert tracking["log_level"] == "INFO"
    assert tracking["log_frequency"] == 512
    assert tracking["mlflow_step_sample_interval"] == 512
    assert tracking["mlflow_artifacts_profile"] == "minimal"
    assert tracking["progress_updates_enabled"] is True
    assert tracking["progress_update_interval"] == 128
    assert tracking["system_metrics_enabled"] is False
    assert tracking["action_diagnostics_enabled"] is True
    assert tracking["action_diagnostics_detail"] == "summary"
    assert tracking["training_diagnostics_enabled"] is True
    assert tracking["training_diagnostics_detail"] == "summary"
    assert tracking["reward_diagnostics_enabled"] is True
    assert tracking["reward_diagnostics_detail"] == "summary"
    assert tracking["runtime_profiling_enabled"] is True
    assert tracking["runtime_profiling_interval"] == 512
    assert tracking["runtime_profiling_detail"] == "summary"
    assert tracking["max_step_seconds"] == pytest.approx(240.0)
    assert tracking["max_update_seconds"] == pytest.approx(2400.0)
    assert tracking["stall_watchdog_enabled"] is True
    assert tracking["stall_watchdog_timeout_seconds"] == pytest.approx(2400.0)
    assert tracking["resource_guard_enabled"] is True
    assert tracking["max_process_rss_mb"] == pytest.approx(88000.0)
    assert tracking["min_available_ram_mb"] == pytest.approx(2048.0)

    assert checkpointing["resume_training"] is False
    assert checkpointing["checkpoint_artifact"] == "transformer_ppo_checkpoint.pt"
    assert checkpointing["checkpoint_interval"] is None
    assert training == {
        "seed": 7,
        "steps_between_training_updates": 256,
        "target_update_interval": 0,
    }

    assert stage["algorithm"] == "AgentTransformerPPO"
    assert transformer == {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": pytest.approx(0.1),
    }
    assert hyperparameters == {
        "learning_rate": pytest.approx(2.0e-4),
        "gamma": pytest.approx(0.99),
        "gae_lambda": pytest.approx(0.95),
        "clip_eps": pytest.approx(0.2),
        "ppo_epochs": 4,
        "minibatch_size": 64,
        "entropy_coeff": pytest.approx(0.03),
        "value_coeff": pytest.approx(0.5),
        "max_grad_norm": pytest.approx(1.0),
        "require_cuda": True,
    }

    assert behavior_cloning["enabled"] is True
    assert behavior_cloning["weight"] == pytest.approx(0.42)
    assert behavior_cloning["min_weight"] == pytest.approx(0.0)
    assert behavior_cloning["decay_start_step"] == 0
    assert behavior_cloning["decay_steps"] == bc_decay_steps
    assert behavior_cloning["ev_multiplier"] == pytest.approx(24.0)
    assert behavior_cloning["storage_multiplier"] == pytest.approx(0.18)
    assert warm_start == {
        "policy": "RBCCommunityPolicy",
        "deterministic": True,
        "noise_scale": pytest.approx(0.0),
        "phaseout_steps": phaseout_steps,
        "phaseout_mode": "blend",
        "hyperparameters": {},
    }


@pytest.mark.parametrize(
    ("duration_name", "template_path", "bc_decay_steps", "phaseout_steps"),
    SERVER_TEMPLATE_CASES,
    ids=[case[0] for case in SERVER_TEMPLATE_CASES],
)
def test_server_template_builds_agent_with_requested_bc_duration(
    duration_name: str,
    template_path: Path,
    bc_decay_steps: int,
    phaseout_steps: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del duration_name
    monkeypatch.setattr(
        agent_transformer_ppo,
        "_select_torch_device",
        lambda **_: torch.device("cpu"),
    )
    agent = build_execution_unit(_load_yaml(template_path))

    assert isinstance(agent, AgentTransformerPPO)
    assert agent._bc is not None
    assert agent._bc.enabled is True
    assert agent._bc.decay_steps == bc_decay_steps
    assert agent._bc.phaseout_steps == phaseout_steps
    assert agent.requires_raw_observation_context is True


def test_server_templates_differ_only_in_identity_and_teacher_duration() -> None:
    normalized = [
        _normalize_server_template(_load_yaml(template_path))
        for _, template_path, _, _ in SERVER_TEMPLATE_CASES
    ]

    assert normalized[1:] == normalized[:-1]


def test_server_templates_use_full_year_dynamic_15min_dataset_coordinates() -> None:
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    assert dataset["seconds_per_time_step"] == 900
    assert dataset["simulation_start_time_step"] == 0
    assert dataset["simulation_end_time_step"] == 35039
    assert dataset["topology_mode"] == "dynamic"
    assert [event["time_step"] for event in dataset["topology_events"]] == [
        5200,
        6000,
        6800,
        7600,
        8400,
        9200,
    ]

    for _, template_path, _, _ in SERVER_TEMPLATE_CASES:
        simulator = _load_yaml(template_path)["simulator"]
        assert simulator.get("topology_event_time_offset", 0) == 0


def test_year_template_reaches_zero_teacher_influence_within_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    year_path = next(
        template_path
        for duration_name, template_path, _, _ in SERVER_TEMPLATE_CASES
        if duration_name == "year"
    )
    config = _load_yaml(year_path)
    bc = config["pipeline"][0]["behavior_cloning"]
    update_interval = config["training"]["steps_between_training_updates"]
    decision_count = config["simulator"]["episode_time_steps"] - 1
    final_update_step = (decision_count // update_interval) * update_interval

    assert final_update_step == 34816
    assert bc["decay_steps"] == final_update_step
    assert bc["warm_start"]["phaseout_steps"] == decision_count

    monkeypatch.setattr(
        agent_transformer_ppo,
        "_select_torch_device",
        lambda **_: torch.device("cpu"),
    )
    agent = build_execution_unit(config)
    assert agent._bc is not None
    assert agent._bc.effective_weight(final_update_step) == pytest.approx(0.0)

    agent._bc.phaseout_step = decision_count - 1
    agent._bc.set_latest_teacher_actions([[1.0]])
    actor_actions = [[0.0]]
    assert agent._bc.maybe_phaseout(actor_actions, deterministic=False) == actor_actions
    assert agent._bc.snapshot_metrics()["behavior_cloning_phaseout_probability"] == pytest.approx(0.0)
