from pathlib import Path

import yaml

from scripts.generate_phase10_w6_configs import generate_w6_configs
from utils.config_schema import validate_config


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_w6_smoke_local_generates_one_short_config_per_recipe(tmp_path):
    rows = generate_w6_configs(output_dir=tmp_path, stage="w6-smoke-local", seeds=[123])

    assert len(rows) == 4
    for row in rows:
        config = _load(Path(row["config_path"]))
        validate_config(config)
        assert config["simulator"]["episode_time_steps"] == 256
        assert config["simulator"]["simulation_end_time_step"] == 255
        assert config["simulator"]["episodes"] == 1
        assert config["algorithm"]["exploration"]["params"]["random_exploration_steps"] == 64


def test_w6a_local_matrix_generates_guided_window_configs(tmp_path):
    rows = generate_w6_configs(output_dir=tmp_path, stage="w6a-local")

    assert len(rows) == 40
    assert (tmp_path / "run_matrix.csv").exists()
    assert (tmp_path / "README.md").exists()

    primary = next(
        row
        for row in rows
        if row["recipe"] == "w6_ev_only_bc_primary"
        and row["seed"] == 123
        and row["window"] == "win0_0000_2048"
    )
    config = _load(Path(primary["config_path"]))
    validate_config(config)

    simulator = config["simulator"]
    assert simulator["reward_function"] == "CostServiceCommunityFeasiblePrecisionRewardV46"
    assert simulator["reward_function_kwargs"] == {}
    assert simulator["simulation_start_time_step"] == 0
    assert simulator["simulation_end_time_step"] == 2047
    assert simulator["episode_time_steps"] == 2048
    assert simulator["episodes"] == 8
    assert simulator["deterministic_finish"] is True
    assert simulator["export"]["include_business_as_usual"] is False
    assert simulator["export"]["timeseries_final_episode_only"] is True

    algorithm = config["algorithm"]
    assert algorithm["name"] == "MADDPG"
    assert algorithm["replay_buffer"]["class"] == "RewardWeightedMultiAgentReplayBuffer"
    assert algorithm["replay_buffer"]["capacity"] == 200000
    assert algorithm["replay_buffer"]["priority_fraction"] == 0.35
    assert algorithm["replay_buffer"]["priority_mode"] == "negative_reward"
    assert algorithm["replay_buffer"]["observation_event_priority_mode"] == "ev_departure_service"

    exploration = algorithm["exploration"]["params"]
    assert exploration["initial_exploration_strategy"] == "policy"
    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert exploration["warm_start_policy_phaseout_mode"] == "blend"
    assert exploration["warm_start_policy_phaseout_steps"] == 4096
    assert exploration["actor_behavior_cloning_weight"] == 0.06
    assert exploration["actor_behavior_cloning_min_weight"] == 0.006
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 4.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.0
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 5.0
    assert exploration["actor_storage_action_l2_penalty"] == 0.004
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 0.05
    assert exploration["actor_behavior_cloning_extra_updates"] == 1


def test_w6c_full_year_matrix_generates_maddpg_and_matd3_remote_configs(tmp_path):
    rows = generate_w6_configs(output_dir=tmp_path, stage="w6c-full-year")

    assert len(rows) == 8
    matd3 = next(
        row
        for row in rows
        if row["algorithm"] == "MATD3"
        and row["recipe"] == "w6_balanced_bc_storage_light"
        and row["seed"] == 456
    )
    config = _load(Path(matd3["config_path"]))
    validate_config(config)

    assert config["algorithm"]["name"] == "MATD3"
    assert config["algorithm"]["hyperparameters"]["require_cuda"] is True
    assert config["algorithm"]["exploration"]["params"]["target_policy_smoothing"] is True
    assert config["algorithm"]["exploration"]["params"]["actor_update_interval"] == 2
    assert config["simulator"]["episodes"] == 2
    assert config["simulator"]["episode_time_steps"] == 8760
    assert config["simulator"]["deterministic_finish"] is True
    assert config["tracking"]["stall_watchdog_enabled"] is True
    assert config["tracking"]["stall_watchdog_context_interval_steps"] == 64
    assert config["execution"]["deucalion"]["time"] == "12:00:00"
    assert config["execution"]["deucalion"]["mem_gb"] == 96
    assert config["execution"]["deucalion"]["cpus_per_task"] == 4
    assert config["execution"]["deucalion"]["gpus"] == 1
