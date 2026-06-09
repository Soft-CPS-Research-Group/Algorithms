from pathlib import Path

import yaml

from scripts.generate_phase10_w6_configs import generate_w6_configs
from utils.config_schema import validate_config


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _alg(config: dict) -> dict:
    return config["pipeline"][0]


def test_w6_smoke_local_generates_one_short_config_per_recipe(tmp_path):
    rows = generate_w6_configs(output_dir=tmp_path, stage="w6-smoke-local", seeds=[123])

    assert len(rows) == 4
    for row in rows:
        config = _load(Path(row["config_path"]))
        validate_config(config)
        assert config["simulator"]["episode_time_steps"] == 256
        assert config["simulator"]["simulation_end_time_step"] == 255
        assert config["simulator"]["episodes"] == 1
        assert config["simulator"]["deterministic_finish"] is False
        assert _alg(config)["replay_buffer"]["batch_size"] == 64
        assert _alg(config)["replay_buffer"]["capacity"] == 20000
        exploration = _alg(config)["exploration"]["params"]
        assert exploration["random_exploration_steps"] == 64
        assert exploration["n_step_returns"] == 8
        assert exploration["actor_policy_loss_normalization"] is True
        assert exploration["actor_offline_bc_pretrain_steps"] == 8


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
    assert simulator["export"]["include_business_as_usual"] is True
    assert simulator["export"]["export_business_as_usual_timeseries"] is True
    assert simulator["export"]["kpis_final_episode_only"] is True
    assert simulator["export"]["timeseries_final_episode_only"] is True

    algorithm = _alg(config)
    assert algorithm["algorithm"] == "MADDPG"
    assert algorithm["replay_buffer"]["class"] == "RewardWeightedMultiAgentReplayBuffer"
    assert algorithm["replay_buffer"]["capacity"] == 200000
    assert algorithm["replay_buffer"]["priority_fraction"] == 0.35
    assert algorithm["replay_buffer"]["priority_mode"] == "negative_reward"
    assert algorithm["replay_buffer"]["observation_event_priority_mode"] == "ev_departure_service"
    assert algorithm["networks"]["critic"]["class"] == "LateFusionCritic"
    assert algorithm["networks"]["critic"]["state_layers"] == [1024, 512]
    assert algorithm["networks"]["critic"]["action_layers"] == [256]
    assert algorithm["networks"]["critic"]["joint_layers"] == [512, 256]

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
    assert exploration["n_step_returns"] == 8
    assert exploration["n_step_gamma"] == 0.995
    assert exploration["n_step_priority_aggregation"] == "max"
    assert exploration["actor_policy_loss_normalization"] is True
    assert exploration["actor_policy_loss_normalization_max_scale"] == 100.0
    assert exploration["actor_offline_bc_pretrain_steps"] == 64
    assert exploration["actor_offline_bc_pretrain_min_replay"] == 256
    assert config["tracking"]["tags"]["recipe"] == "w6_ev_only_bc_primary"
    assert config["tracking"]["tags"]["window"] == "win0_0000_2048"
    assert config["tracking"]["tags"]["seed"] == 123
    assert config["tracking"]["tags"]["n_step_returns"] == 8
    assert config["tracking"]["tags"]["actor_policy_loss_normalization"] is True


def test_w6a_local_can_generate_clone_improvement_variants(tmp_path):
    rows = generate_w6_configs(
        output_dir=tmp_path,
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_clone_cost_nudge", "w6_clone_tight_v2g_storage"],
        include_baselines=False,
    )

    assert len(rows) == 8

    cost_nudge = next(row for row in rows if row["recipe"] == "w6_clone_cost_nudge")
    cost_config = _load(Path(cost_nudge["config_path"]))
    validate_config(cost_config)
    cost_exploration = _alg(cost_config)["exploration"]["params"]
    assert cost_exploration["actor_behavior_cloning_weight"] == 0.35
    assert cost_exploration["actor_behavior_cloning_min_weight"] == 0.2
    assert cost_exploration["actor_policy_loss_weight"] == 0.05
    assert cost_exploration["actor_policy_loss_warmup_weight"] == 0.01

    tight = next(row for row in rows if row["recipe"] == "w6_clone_tight_v2g_storage")
    tight_config = _load(Path(tight["config_path"]))
    validate_config(tight_config)
    tight_exploration = _alg(tight_config)["exploration"]["params"]
    assert tight_exploration["actor_policy_loss_weight"] == 0.0
    assert tight_exploration["actor_storage_action_l2_penalty"] == 0.01
    assert tight_exploration["actor_ev_v2g_action_l2_penalty"] == 0.15


def test_w6a_local_can_generate_residual_community_recipe(tmp_path):
    rows = generate_w6_configs(
        output_dir=tmp_path,
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_residual_comm_constraint"],
        algorithms=["MATD3"],
        include_baselines=False,
    )

    assert len(rows) == 4
    config = _load(Path(rows[0]["config_path"]))
    validate_config(config)

    assert _alg(config)["algorithm"] == "MATD3"
    assert _alg(config)["networks"]["critic"]["class"] == "LateFusionCritic"
    assert config["simulator"]["reward_function"] == "CostServiceCommunityResidualConstraintRewardV53"
    assert _alg(config)["replay_buffer"]["observation_event_priority_mode"] == "combined"
    exploration = _alg(config)["exploration"]["params"]
    assert exploration["warm_start_policy"] == "RBCCommunityPolicy"
    assert exploration["residual_policy_enabled"] is True
    assert exploration["residual_action_scale"] == 0.08
    assert exploration["residual_action_final_scale"] == 0.28
    assert exploration["actor_residual_delta_l2_penalty"] == 0.02
    assert exploration["n_step_returns"] == 8
    assert exploration["actor_policy_loss_normalization"] is True
    assert config["tracking"]["tags"]["teacher_policy"] == "RBCCommunityPolicy"
    assert config["tracking"]["tags"]["residual_policy"] is True
    assert config["tracking"]["tags"]["actor_residual_delta_l2_penalty"] == 0.02


def test_w6_remote_smoke_can_generate_w7_matd3_residual_repair(tmp_path):
    rows = generate_w6_configs(
        output_dir=tmp_path,
        stage="w6b-remote-smoke",
        seeds=[123],
        recipes=["w7_residual_comm_ev_repair"],
        algorithms=["MATD3"],
        include_baselines=False,
    )

    assert len(rows) == 1
    config = _load(Path(rows[0]["config_path"]))
    validate_config(config)

    assert _alg(config)["algorithm"] == "MATD3"
    assert config["simulator"]["episode_time_steps"] == 4096
    assert config["simulator"]["reward_function"] == "CostServiceCommunityResidualConstraintRewardV53"
    assert config["simulator"]["reward_function_kwargs"]["ev_departure_missed_penalty"] == 4600.0
    assert config["execution"]["deucalion"]["gpus"] == 1
    exploration = _alg(config)["exploration"]["params"]
    assert exploration["warm_start_policy"] == "RBCCommunityPolicy"
    assert exploration["residual_policy_enabled"] is True
    assert exploration["residual_action_final_scale"] == 0.45
    assert exploration["residual_ev_action_scale_multiplier"] == 0.70
    assert exploration["critic_action_input_mode"] == "final_base_delta_normalized"
    assert exploration["actor_policy_loss_weight"] == 0.18
    assert exploration["actor_behavior_cloning_min_weight"] == 0.012
    assert exploration["n_step_returns"] == 12


def test_w6_remote_smoke_can_generate_w7_dense_conservative_and_headed_variants(tmp_path):
    rows = generate_w6_configs(
        output_dir=tmp_path,
        stage="w6b-remote-smoke",
        seeds=[123],
        recipes=[
            "w7_residual_comm_ev_dense_conservative",
            "w7_residual_comm_ev_dense_heads",
        ],
        algorithms=["MATD3"],
        include_baselines=False,
    )

    assert len(rows) == 2

    conservative = next(row for row in rows if row["recipe"] == "w7_residual_comm_ev_dense_conservative")
    conservative_config = _load(Path(conservative["config_path"]))
    validate_config(conservative_config)
    assert conservative_config["simulator"]["reward_function"] == "CostServiceCommunityDenseEVResidualRewardV54"
    assert conservative_config["simulator"]["reward_function_kwargs"]["ev_schedule_deficit_penalty"] == 1650.0
    assert _alg(conservative_config)["networks"]["actor"]["class"] == "Actor"
    conservative_exploration = _alg(conservative_config)["exploration"]["params"]
    assert conservative_exploration["residual_action_final_scale"] == 0.20
    assert conservative_exploration["residual_ev_action_scale_multiplier"] == 0.32
    assert conservative_exploration["actor_residual_delta_l2_penalty"] == 0.06
    assert conservative_exploration["n_step_returns"] == 16

    headed = next(row for row in rows if row["recipe"] == "w7_residual_comm_ev_dense_heads")
    headed_config = _load(Path(headed["config_path"]))
    validate_config(headed_config)
    assert _alg(headed_config)["networks"]["actor"]["class"] == "SemanticMultiHeadActor"
    assert _alg(headed_config)["networks"]["actor"]["head_layers"] == [64]
    assert headed_config["tracking"]["tags"]["actor_class"] == "SemanticMultiHeadActor"
    headed_exploration = _alg(headed_config)["exploration"]["params"]
    assert headed_exploration["actor_community_context_enabled"] is True
    assert headed_exploration["actor_frame_stack_steps"] == 3
    assert headed_exploration["actor_auxiliary_loss_weight"] == 0.020
    assert headed_exploration["actor_storage_smoothness_l2_penalty"] == 0.010


def test_w6_remote_smoke_can_generate_w7_min_service_context_variant(tmp_path):
    rows = generate_w6_configs(
        output_dir=tmp_path,
        stage="w6b-remote-smoke",
        seeds=[123],
        recipes=["w7_residual_comm_min_service_ctx"],
        algorithms=["MATD3"],
        include_baselines=False,
    )

    assert len(rows) == 1
    config = _load(Path(rows[0]["config_path"]))
    validate_config(config)

    assert config["simulator"]["reward_function"] == "CostServiceCommunityDenseEVResidualRewardV54"
    kwargs = config["simulator"]["reward_function_kwargs"]
    assert kwargs["community_settlement_cost_weight"] == 1.26
    assert kwargs["ev_departure_missed_penalty"] == 5200.0
    assert kwargs["ev_over_service_tolerance"] == 0.05
    assert kwargs["ev_over_service_penalty"] == 620.0

    algorithm = _alg(config)
    assert algorithm["networks"]["actor"]["class"] == "Actor"
    exploration = algorithm["exploration"]["params"]
    assert exploration["warm_start_policy"] == "RBCCommunityPolicy"
    assert exploration["residual_policy_enabled"] is True
    assert exploration["residual_action_final_scale"] == 0.36
    assert exploration["residual_ev_action_scale_multiplier"] == 0.52
    assert exploration["n_step_returns"] == 12
    assert exploration["actor_community_context_enabled"] is True
    assert exploration["actor_frame_stack_steps"] == 2
    assert exploration["actor_auxiliary_loss_weight"] == 0.010
    assert exploration["actor_storage_smoothness_l2_penalty"] == 0.0


def test_w6_remote_smoke_can_generate_w7_heads_clone_diagnostic(tmp_path):
    rows = generate_w6_configs(
        output_dir=tmp_path,
        stage="w6b-remote-smoke",
        seeds=[123],
        recipes=["w7_heads_clone_diagnostic"],
        algorithms=["MATD3"],
        include_baselines=False,
    )

    assert len(rows) == 1
    config = _load(Path(rows[0]["config_path"]))
    validate_config(config)

    algorithm = _alg(config)
    assert algorithm["networks"]["actor"]["class"] == "SemanticMultiHeadActor"
    assert algorithm["networks"]["actor"]["head_layers"] == [64]
    assert config["tracking"]["tags"]["actor_class"] == "SemanticMultiHeadActor"
    assert config["tracking"]["tags"]["residual_policy"] is False

    exploration = algorithm["exploration"]["params"]
    assert exploration["warm_start_policy"] == "RBCCommunityPolicy"
    assert exploration["residual_policy_enabled"] is False
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.650
    assert exploration["actor_behavior_cloning_min_weight"] == 0.500
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 1.0
    assert exploration["actor_community_context_enabled"] is True
    assert exploration["actor_frame_stack_steps"] == 3
    assert exploration["actor_auxiliary_loss_weight"] == 0.020
    assert exploration["n_step_returns"] == 1


def test_w6a_local_can_generate_reward_regularized_variants(tmp_path):
    rows = generate_w6_configs(
        output_dir=tmp_path,
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_clone_cost_nudge_v2g_tight", "w6_clone_cost_v47_precision"],
        include_baselines=False,
    )

    assert len(rows) == 8

    tight = next(row for row in rows if row["recipe"] == "w6_clone_cost_nudge_v2g_tight")
    tight_config = _load(Path(tight["config_path"]))
    validate_config(tight_config)
    assert tight_config["simulator"]["reward_function"] == "CostServiceCommunityFeasiblePrecisionRewardV46"
    assert tight_config["simulator"]["reward_function_kwargs"]["ev_v2g_service_penalty"] == 1200.0
    assert tight_config["simulator"]["reward_function_kwargs"]["battery_throughput_penalty"] == 0.03
    assert _alg(tight_config)["exploration"]["params"]["actor_ev_v2g_action_l2_penalty"] == 0.25

    softwall_rows = generate_w6_configs(
        output_dir=tmp_path / "softwall",
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_clone_cost_ev_v2g_softwall"],
        include_baselines=False,
    )
    softwall_config = _load(Path(softwall_rows[0]["config_path"]))
    validate_config(softwall_config)
    assert _alg(softwall_config)["exploration"]["params"]["actor_ev_v2g_action_l2_penalty"] == 8.0

    highclip_rows = generate_w6_configs(
        output_dir=tmp_path / "highclip",
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_clone_cost_v2g_highclip"],
        include_baselines=False,
    )
    highclip_config = _load(Path(highclip_rows[0]["config_path"]))
    validate_config(highclip_config)
    assert _alg(highclip_config)["exploration"]["params"]["reward_normalization_clip"] == 25.0

    masswall_rows = generate_w6_configs(
        output_dir=tmp_path / "masswall",
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_clone_cost_ev_v2g_masswall_gentle"],
        include_baselines=False,
    )
    masswall_config = _load(Path(masswall_rows[0]["config_path"]))
    validate_config(masswall_config)
    masswall_exploration = _alg(masswall_config)["exploration"]["params"]
    assert masswall_exploration["actor_ev_v2g_action_l2_penalty"] == 4.0
    assert masswall_exploration["actor_ev_v2g_action_mass_penalty"] == 8.0

    energywall_rows = generate_w6_configs(
        output_dir=tmp_path / "energywall",
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_clone_cost_ev_v2g_energywall"],
        include_baselines=False,
    )
    energywall_config = _load(Path(energywall_rows[0]["config_path"]))
    validate_config(energywall_config)
    assert energywall_config["simulator"]["reward_function_kwargs"]["ev_v2g_discharge_penalty"] == 1.0

    tight_energywall_rows = generate_w6_configs(
        output_dir=tmp_path / "tight_energywall",
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_clone_cost_ev_v2g_energywall_battery_tight"],
        include_baselines=False,
    )
    tight_energywall_config = _load(Path(tight_energywall_rows[0]["config_path"]))
    validate_config(tight_energywall_config)
    tight_energywall_exploration = _alg(tight_energywall_config)["exploration"]["params"]
    assert tight_energywall_exploration["actor_storage_action_l2_penalty"] == 0.02
    assert tight_energywall_exploration["actor_ev_v2g_action_mass_penalty"] == 6.0
    assert tight_energywall_config["simulator"]["reward_function_kwargs"]["battery_throughput_penalty"] == 0.05

    flex_rows = generate_w6_configs(
        output_dir=tmp_path / "flex",
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_flex_v2g_safe_value"],
        include_baselines=False,
    )
    flex_config = _load(Path(flex_rows[0]["config_path"]))
    validate_config(flex_config)
    assert flex_config["simulator"]["reward_function"] == "CostServiceCommunityPeakDeadlineRewardV52"
    assert flex_config["simulator"]["reward_function_kwargs"]["community_peak_import_penalty"] == 0.002
    assert _alg(flex_config)["exploration"]["params"]["actor_ev_v2g_action_mass_penalty"] == 0.8

    flex_followup_rows = generate_w6_configs(
        output_dir=tmp_path / "flex_followups",
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_flex_margin_teacher_storage_tight", "w6_flex_v2g_open_value"],
        include_baselines=False,
    )
    storage_tight = next(
        row for row in flex_followup_rows if row["recipe"] == "w6_flex_margin_teacher_storage_tight"
    )
    storage_tight_config = _load(Path(storage_tight["config_path"]))
    validate_config(storage_tight_config)
    assert storage_tight_config["simulator"]["reward_function"] == "CostServiceCommunityFeasiblePrecisionRewardV46"
    assert storage_tight_config["simulator"]["reward_function_kwargs"]["battery_throughput_penalty"] == 0.014
    assert _alg(storage_tight_config)["exploration"]["params"]["actor_storage_action_l2_penalty"] == 0.014

    v2g_open = next(row for row in flex_followup_rows if row["recipe"] == "w6_flex_v2g_open_value")
    v2g_open_config = _load(Path(v2g_open["config_path"]))
    validate_config(v2g_open_config)
    assert v2g_open_config["simulator"]["reward_function"] == "CostServiceCommunityPeakDeadlineRewardV52"
    assert v2g_open_config["simulator"]["reward_function_kwargs"]["ev_v2g_service_penalty"] == 2000.0
    assert _alg(v2g_open_config)["exploration"]["params"]["actor_ev_v2g_action_mass_penalty"] == 0.45

    ev_repair_rows = generate_w6_configs(
        output_dir=tmp_path / "ev_repair",
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_flex_ev_gate_repair_strong_bc"],
        include_baselines=False,
    )
    ev_repair_config = _load(Path(ev_repair_rows[0]["config_path"]))
    validate_config(ev_repair_config)
    ev_repair_exploration = _alg(ev_repair_config)["exploration"]["params"]
    assert ev_repair_exploration["warm_start_policy_phaseout_steps"] == 8192
    assert ev_repair_exploration["actor_behavior_cloning_weight"] == 0.68
    assert ev_repair_exploration["actor_behavior_cloning_min_weight"] == 0.52
    assert ev_repair_exploration["actor_ev_behavior_cloning_multiplier"] == 28.0
    assert ev_repair_exploration["actor_policy_loss_weight"] == 0.01
    assert ev_repair_config["simulator"]["reward_function_kwargs"]["ev_v2g_service_penalty"] == 1800.0
    assert ev_repair_config["simulator"]["reward_function_kwargs"]["battery_throughput_penalty"] == 0.018

    ev_repair_followup_rows = generate_w6_configs(
        output_dir=tmp_path / "ev_repair_followups",
        stage="w6a-local",
        seeds=[123],
        recipes=[
            "w6_flex_ev_gate_repair_mid_bc",
            "w6_flex_ev_gate_repair_cost_push",
            "w6_flex_ev_gate_repair_policy_open",
        ],
        include_baselines=False,
    )
    assert len(ev_repair_followup_rows) == 12
    mid_config = _load(
        Path(next(row for row in ev_repair_followup_rows if row["recipe"] == "w6_flex_ev_gate_repair_mid_bc")["config_path"])
    )
    validate_config(mid_config)
    mid_exploration = _alg(mid_config)["exploration"]["params"]
    assert mid_exploration["actor_behavior_cloning_weight"] == 0.56
    assert mid_exploration["actor_ev_behavior_cloning_multiplier"] == 22.0
    assert mid_exploration["warm_start_policy_phaseout_steps"] == 6144

    cost_push_config = _load(
        Path(next(row for row in ev_repair_followup_rows if row["recipe"] == "w6_flex_ev_gate_repair_cost_push")["config_path"])
    )
    validate_config(cost_push_config)
    assert cost_push_config["simulator"]["reward_function_kwargs"]["community_settlement_cost_weight"] == 1.24
    assert _alg(cost_push_config)["exploration"]["params"]["actor_policy_loss_weight"] == 0.026

    policy_open_config = _load(
        Path(next(row for row in ev_repair_followup_rows if row["recipe"] == "w6_flex_ev_gate_repair_policy_open")["config_path"])
    )
    validate_config(policy_open_config)
    assert _alg(policy_open_config)["exploration"]["params"]["warm_start_policy_phaseout_steps"] == 3072
    assert _alg(policy_open_config)["exploration"]["params"]["actor_behavior_cloning_extra_updates"] == 2

    precision = next(row for row in rows if row["recipe"] == "w6_clone_cost_v47_precision")
    precision_config = _load(Path(precision["config_path"]))
    validate_config(precision_config)
    assert precision_config["simulator"]["reward_function"] == "CostServiceCommunityFeasiblePrecisionRewardV47"
    assert precision_config["tracking"]["tags"]["reward_function"] == "CostServiceCommunityFeasiblePrecisionRewardV47"
    assert precision["reward_function"] == "CostServiceCommunityFeasiblePrecisionRewardV47"


def test_w6c_full_year_matrix_generates_maddpg_and_matd3_remote_configs(tmp_path):
    rows = generate_w6_configs(output_dir=tmp_path, stage="w6c-full-year")

    assert len(rows) == 8
    matd3 = next(
        row
        for row in rows
        if row["algorithm"] == "MATD3"
        and row["recipe"] == "w6_residual_comm_cost_push"
        and row["seed"] == 456
    )
    config = _load(Path(matd3["config_path"]))
    validate_config(config)

    assert _alg(config)["algorithm"] == "MATD3"
    assert _alg(config)["hyperparameters"]["require_cuda"] is True
    exploration = _alg(config)["exploration"]["params"]
    assert exploration["target_policy_smoothing"] is True
    assert exploration["actor_update_interval"] == 2
    assert exploration["n_step_returns"] == 8
    assert exploration["actor_policy_loss_normalization"] is True
    assert exploration["warm_start_policy"] == "RBCCommunityPolicy"
    assert exploration["residual_policy_enabled"] is True
    assert exploration["critic_action_input_mode"] == "final_base_delta_normalized"
    assert exploration["actor_offline_bc_pretrain_steps"] == 128
    assert matd3["teacher_policy"] == "RBCCommunityPolicy"
    assert matd3["critic_action_input_mode"] == "final_base_delta_normalized"
    assert config["simulator"]["episodes"] == 2
    assert config["simulator"]["episode_time_steps"] == 8760
    assert config["simulator"]["deterministic_finish"] is True
    assert config["tracking"]["stall_watchdog_enabled"] is True
    assert config["tracking"]["stall_watchdog_context_interval_steps"] == 64
    assert config["execution"]["deucalion"]["time"] == "12:00:00"
    assert config["execution"]["deucalion"]["mem_gb"] == 96
    assert config["execution"]["deucalion"]["cpus_per_task"] == 4
    assert config["execution"]["deucalion"]["gpus"] == 1


def test_w6a_local_allows_explicit_matd3_comparator(tmp_path):
    rows = generate_w6_configs(
        output_dir=tmp_path,
        stage="w6a-local",
        seeds=[123],
        recipes=["w6_flex_v2g_safe_value"],
        algorithms=["MATD3"],
        include_baselines=False,
    )

    assert len(rows) == 4
    config = _load(Path(rows[0]["config_path"]))
    validate_config(config)
    assert _alg(config)["algorithm"] == "MATD3"
    assert _alg(config)["hyperparameters"]["require_cuda"] is False
    assert _alg(config)["exploration"]["params"]["target_policy_smoothing"] is True
