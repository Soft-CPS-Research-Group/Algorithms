import json
from argparse import Namespace

import yaml

from scripts import run_phase6a_benchmark as phase6a


def test_phase6a_dry_run_generates_configs_and_summary(tmp_path):
    output_dir = tmp_path / "phase6a"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["random", "maddpg"],
        maddpg_variant=["noop_centered"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    payload = phase6a.run_phase6a(args)

    assert payload["output_dir"] == str(output_dir)
    assert len(payload["rows"]) == 2
    assert {row["status"] for row in payload["rows"]} == {"planned"}
    assert (output_dir / "benchmark_summary.csv").is_file()
    assert (output_dir / "benchmark_summary.json").is_file()
    assert (output_dir / "README.md").is_file()

    random_config = output_dir / "generated_configs" / "phase6a_15s_random_random_seed123.yaml"
    maddpg_config = output_dir / "generated_configs" / "phase6a_15s_maddpg_noop_centered_seed123.yaml"
    assert random_config.is_file()
    assert maddpg_config.is_file()

    random_payload = yaml.safe_load(random_config.read_text(encoding="utf-8"))
    maddpg_payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))

    assert random_payload["simulator"]["export"]["mode"] == "none"
    assert random_payload["simulator"]["episode_time_steps"] == 16
    assert maddpg_payload["algorithm"]["name"] == "MADDPG"
    assert maddpg_payload["algorithm"]["replay_buffer"]["batch_size"] == 16
    assert maddpg_payload["algorithm"]["networks"]["actor"]["layers"] == [32]
    assert maddpg_payload["algorithm"]["networks"]["critic"]["layers"] == [64, 32]
    assert (
        maddpg_payload["algorithm"]["exploration"]["params"]["initial_exploration_strategy"]
        == "noop_centered"
    )
    assert maddpg_payload["algorithm"]["exploration"]["params"]["random_exploration_steps"] == 4

    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["settings"]["dry_run"] is True
    assert summary["settings"]["kpi_export"] is False


def test_phase6a_deterministic_finish_is_written_to_generated_config(tmp_path):
    output_dir = tmp_path / "phase6a_deterministic_finish"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["noop_centered"],
        seed=[123],
        episodes=2,
        deterministic_finish=True,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    payload = phase6a.run_phase6a(args)

    maddpg_config = output_dir / "generated_configs" / "phase6a_15s_maddpg_noop_centered_seed123.yaml"
    config_payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))

    assert config_payload["simulator"]["deterministic_finish"] is True
    assert payload["settings"]["deterministic_finish"] is True
    assert summary["settings"]["deterministic_finish"] is True
    assert payload["rows"][0]["deterministic_finish"] is True


def test_phase6a_hyperparameter_override_is_written_to_generated_config(tmp_path):
    output_dir = tmp_path / "phase6a_hyperparameter_override"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["2022"],
        agent=["rbc_smart"],
        maddpg_variant=[],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        checkpoint_interval=None,
        reward_diagnostics_detail="summary",
        reward_function_override=None,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        hyperparameter_override=[
            "price_charge_rate=0.15",
            "allow_v2g=false",
            "storage_price_charge_soc_ceiling=0.75",
        ],
        dry_run=True,
        fail_fast=False,
    )

    payload = phase6a.run_phase6a(args)

    config_path = output_dir / "generated_configs" / "phase6a_2022_rbc_smart_rbc_smart_seed123.yaml"
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    hyperparameters = config_payload["algorithm"]["hyperparameters"]

    assert hyperparameters["price_charge_rate"] == 0.15
    assert hyperparameters["allow_v2g"] is False
    assert hyperparameters["storage_price_charge_soc_ceiling"] == 0.75
    assert payload["settings"]["hyperparameter_overrides"] == args.hyperparameter_override
    assert summary["settings"]["hyperparameter_overrides"] == args.hyperparameter_override


def test_phase6a_maddpg_override_groups_are_written_to_generated_config(tmp_path):
    output_dir = tmp_path / "phase6a_maddpg_overrides"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_teacher_clone_ev_band_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        checkpoint_interval=None,
        reward_diagnostics_detail="summary",
        reward_function_override=None,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        hyperparameter_override=[],
        exploration_override=["actor_policy_loss_weight=0.05"],
        warm_start_hyperparameter_override=["ev_service_floor_rate=0.6"],
        replay_buffer_override=["priority_fraction=0.1"],
        reward_kwarg_override=["ev_over_service_penalty=7.5"],
        dry_run=True,
        fail_fast=False,
    )

    payload = phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_teacher_clone_ev_band_rbc_smart_seed123.yaml"
    )
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    exploration = config_payload["algorithm"]["exploration"]["params"]

    assert exploration["actor_policy_loss_weight"] == 0.05
    assert exploration["warm_start_policy_hyperparameters"]["ev_service_floor_rate"] == 0.6
    assert config_payload["algorithm"]["replay_buffer"]["priority_fraction"] == 0.1
    assert config_payload["simulator"]["reward_function_kwargs"]["ev_over_service_penalty"] == 7.5
    assert payload["settings"]["exploration_overrides"] == args.exploration_override
    assert payload["settings"]["warm_start_hyperparameter_overrides"] == args.warm_start_hyperparameter_override
    assert summary["settings"]["replay_buffer_overrides"] == args.replay_buffer_override
    assert summary["settings"]["reward_kwarg_overrides"] == args.reward_kwarg_override


def test_phase6a_checkpoint_interval_override_is_written_to_generated_config(tmp_path):
    output_dir = tmp_path / "phase6a_checkpoint_interval"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["noop_centered"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        checkpoint_interval=64,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    payload = phase6a.run_phase6a(args)

    maddpg_config = output_dir / "generated_configs" / "phase6a_15s_maddpg_noop_centered_seed123.yaml"
    config_payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))

    assert config_payload["checkpointing"]["checkpoint_interval"] == 64
    assert payload["settings"]["checkpoint_interval"] == 64
    assert summary["settings"]["checkpoint_interval"] == 64


def test_phase6a_ev_priority_bc_variant_sets_reward_and_actor_regularization(tmp_path):
    output_dir = tmp_path / "phase6a_ev_priority"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["ev_priority_bc_warm_rbc_basic"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_ev_priority_bc_warm_rbc_basic_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    reward_kwargs = payload["simulator"]["reward_function_kwargs"]

    assert exploration["initial_exploration_strategy"] == "policy"
    assert exploration["warm_start_policy"] == "RBCBasicPolicy"
    assert exploration["actor_behavior_cloning_weight"] == 0.05
    assert reward_kwargs["ev_departure_window_hours"] == 4.0
    assert reward_kwargs["ev_schedule_deficit_penalty"] == 480.0


def test_phase6a_ev_service_v2g_guard_variant_sets_service_discharge_penalty(tmp_path):
    output_dir = tmp_path / "phase6a_ev_v2g_guard"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["ev_service_v2g_guard_warm_rbc_basic"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_ev_service_v2g_guard_warm_rbc_basic_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    reward_kwargs = payload["simulator"]["reward_function_kwargs"]

    assert exploration["warm_start_policy"] == "RBCBasicPolicy"
    assert exploration["actor_behavior_cloning_weight"] == 0.05
    assert reward_kwargs["ev_v2g_service_penalty"] == 200.0
    assert reward_kwargs["ev_departure_missed_penalty"] == 1000.0


def test_phase6a_ev_service_v2g_guard_prioritized_variant_sets_replay_buffer(tmp_path):
    output_dir = tmp_path / "phase6a_ev_v2g_prioritized"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["ev_service_v2g_guard_prioritized_warm_rbc_basic"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_ev_service_v2g_guard_prioritized_warm_rbc_basic_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    replay_buffer = payload["algorithm"]["replay_buffer"]
    reward_kwargs = payload["simulator"]["reward_function_kwargs"]

    assert replay_buffer["class"] == "RewardWeightedMultiAgentReplayBuffer"
    assert replay_buffer["priority_fraction"] == 0.5
    assert replay_buffer["priority_alpha"] == 0.7
    assert replay_buffer["priority_mode"] == "negative_reward"
    assert replay_buffer["priority_max"] == 100.0
    assert reward_kwargs["ev_v2g_service_penalty"] == 200.0


def test_phase6a_service_guard_v2_variant_uses_named_reward_and_low_priority_replay(tmp_path):
    output_dir = tmp_path / "phase6a_service_guard_v2"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    replay_buffer = payload["algorithm"]["replay_buffer"]
    exploration = payload["algorithm"]["exploration"]["params"]

    assert payload["simulator"]["reward_function"] == "CostServiceGuardRewardV2"
    assert payload["simulator"]["reward_function_kwargs"] == {}
    assert replay_buffer["class"] == "RewardWeightedMultiAgentReplayBuffer"
    assert replay_buffer["priority_fraction"] == 0.25
    assert replay_buffer["priority_mode"] == "negative_reward"
    assert replay_buffer["priority_max"] == 100.0
    assert exploration["actor_behavior_cloning_decay_start_step"] == 4
    assert exploration["actor_behavior_cloning_decay_steps"] == 8


def test_phase6a_cost_balanced_v3_variant_uses_named_reward(tmp_path):
    output_dir = tmp_path / "phase6a_cost_balanced_v3"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]

    assert payload["simulator"]["reward_function"] == "CostServiceCostBalancedRewardV3"
    assert payload["simulator"]["reward_function_kwargs"] == {}
    assert exploration["actor_behavior_cloning_weight"] == 0.04
    assert exploration["actor_behavior_cloning_min_weight"] == 0.01


def test_phase6a_community_band_v4_variant_uses_settlement_reward_and_tiny_priority(tmp_path):
    output_dir = tmp_path / "phase6a_community_band_v4"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    replay_buffer = payload["algorithm"]["replay_buffer"]
    exploration = payload["algorithm"]["exploration"]["params"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityBandRewardV4"
    assert payload["simulator"]["reward_function_kwargs"] == {}
    assert replay_buffer["class"] == "RewardWeightedMultiAgentReplayBuffer"
    assert replay_buffer["priority_fraction"] == 0.15
    assert replay_buffer["priority_mode"] == "negative_reward"
    assert exploration["actor_behavior_cloning_weight"] == 0.03
    assert exploration["actor_behavior_cloning_min_weight"] == 0.005
    assert exploration["actor_behavior_cloning_decay_start_step"] == 4
    assert exploration["actor_behavior_cloning_decay_steps"] == 4


def test_phase6a_community_storage_band_v41_variant_sets_storage_regularization(tmp_path):
    output_dir = tmp_path / "phase6a_community_storage_band_v41"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_storage_band_v41_prioritized_regularized_warm_rbc_basic"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_storage_band_v41_prioritized_regularized_warm_rbc_basic_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    replay_buffer = payload["algorithm"]["replay_buffer"]
    exploration = payload["algorithm"]["exploration"]["params"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityStorageBandRewardV41"
    assert replay_buffer["class"] == "RewardWeightedMultiAgentReplayBuffer"
    assert replay_buffer["priority_fraction"] == 0.15
    assert exploration["actor_behavior_cloning_weight"] == 0.025
    assert exploration["actor_behavior_cloning_min_weight"] == 0.005
    assert exploration["actor_storage_action_l2_penalty"] == 4.0
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 0.020


def test_phase6a_community_service_band_v42_smart_variant_sets_service_reward(tmp_path):
    output_dir = tmp_path / "phase6a_community_service_band_v42"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_service_band_v42_prioritized_regularized_warm_rbc_smart"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_service_band_v42_prioritized_regularized_warm_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    replay_buffer = payload["algorithm"]["replay_buffer"]
    exploration = payload["algorithm"]["exploration"]["params"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityServiceBandRewardV42"
    assert replay_buffer["class"] == "RewardWeightedMultiAgentReplayBuffer"
    assert replay_buffer["priority_fraction"] == 0.20
    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert exploration["actor_behavior_cloning_weight"] == 0.020
    assert exploration["actor_behavior_cloning_min_weight"] == 0.005
    assert exploration["actor_storage_action_l2_penalty"] == 4.0
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 0.040


def test_phase6a_community_service_band_v42_warmtrain_variant_trains_during_warmup(tmp_path):
    output_dir = tmp_path / "phase6a_community_service_band_v42_warmtrain"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_service_band_v42_prioritized_warmtrain_rbc_smart"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_service_band_v42_prioritized_warmtrain_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityServiceBandRewardV42"
    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert exploration["train_during_initial_exploration"] is True
    assert exploration["initial_exploration_training_start_step"] == 0
    assert exploration["actor_behavior_cloning_weight"] == 0.040
    assert exploration["storage_exploration_noise_multiplier"] == 0.25
    assert exploration["ev_negative_exploration_noise_multiplier"] == 0.35
    assert exploration["actor_behavior_cloning_decay_start_step"] == 16
    assert exploration["actor_behavior_cloning_decay_steps"] == 16


def test_phase6a_community_service_band_v42_phaseout_variant_sets_warm_start_phaseout(tmp_path):
    output_dir = tmp_path / "phase6a_community_service_band_v42_phaseout"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_service_band_v42_prioritized_warmtrain_phaseout_rbc_smart"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_service_band_v42_prioritized_warmtrain_phaseout_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]

    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert exploration["train_during_initial_exploration"] is True
    assert exploration["use_amp"] is False
    assert exploration["warm_start_policy_phaseout_steps"] == 32
    assert exploration["warm_start_policy_phaseout_mode"] == "blend"
    assert exploration["storage_exploration_noise_multiplier"] == 0.25
    assert exploration["ev_negative_exploration_noise_multiplier"] == 0.0
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 5.0
    assert exploration["actor_behavior_cloning_weight"] == 0.080
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 8.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.25
    assert exploration["actor_behavior_cloning_min_weight"] == 0.060
    assert exploration["actor_behavior_cloning_decay_steps"] == 96


def test_phase6a_community_battery_value_v43_variant_sets_reward_and_storage_knobs(tmp_path):
    output_dir = tmp_path / "phase6a_community_battery_value_v43"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_battery_value_v43_prioritized_warmtrain_phaseout_rbc_smart"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_battery_value_v43_prioritized_warmtrain_phaseout_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityBatteryValueRewardV43"
    assert payload["simulator"]["reward_function_kwargs"] == {}
    assert replay_buffer["class"] == "RewardWeightedMultiAgentReplayBuffer"
    assert replay_buffer["priority_fraction"] == 0.20
    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert exploration["train_during_initial_exploration"] is True
    assert exploration["use_amp"] is False
    assert exploration["warm_start_policy_phaseout_steps"] == 32
    assert exploration["warm_start_policy_phaseout_mode"] == "blend"
    assert exploration["storage_exploration_noise_multiplier"] == 0.50
    assert exploration["actor_storage_action_l2_penalty"] == 0.50
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 1.0


def test_phase6a_community_smooth_service_v44_variant_sets_stability_knobs(tmp_path):
    output_dir = tmp_path / "phase6a_community_smooth_service_v44"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_smooth_service_v44_stable_teacher_bc_rbc_smart"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    maddpg_config = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_smooth_service_v44_stable_teacher_bc_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunitySmoothServiceRewardV44"
    assert replay_buffer["class"] == "RewardWeightedMultiAgentReplayBuffer"
    assert replay_buffer["priority_fraction"] == 0.25
    assert replay_buffer["priority_max"] == 40.0
    assert payload["algorithm"]["networks"]["critic"]["lr"] == 1.0e-4
    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert exploration["critic_loss"] == "huber"
    assert exploration["critic_target_clip_abs"] == 35.0
    assert exploration["actor_behavior_cloning_source"] == "warm_start_policy"
    assert exploration["actor_behavior_cloning_min_weight"] == 0.100
    assert exploration["actor_behavior_cloning_decay_steps"] == 192
    assert exploration["warm_start_policy_phaseout_steps"] == 96


def test_phase6a_can_force_reward_diagnostic_detail_and_reward_function(tmp_path):
    output_dir = tmp_path / "phase6a_reward_audit"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["rbc_smart"],
        maddpg_variant=[],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        reward_diagnostics_detail="per_agent",
        reward_function_override="CostServiceCommunitySmoothServiceRewardV44",
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = output_dir / "generated_configs" / "phase6a_15s_rbc_smart_rbc_smart_seed123.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))

    assert payload["tracking"]["reward_diagnostics_detail"] == "per_agent"
    assert payload["simulator"]["reward_function"] == "CostServiceCommunitySmoothServiceRewardV44"
    assert payload["simulator"]["reward_function_kwargs"] == {}
    assert summary["settings"]["reward_diagnostics_detail"] == "per_agent"
    assert summary["settings"]["reward_function_override"] == "CostServiceCommunitySmoothServiceRewardV44"


def test_phase6a_community_feasible_service_v45_variant_sets_reward_and_stability_knobs(tmp_path):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_stable_teacher_bc_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_stable_teacher_bc_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasibleServiceRewardV45"
    assert replay_buffer["class"] == "RewardWeightedMultiAgentReplayBuffer"
    assert replay_buffer["priority_fraction"] == 0.25
    assert replay_buffer["priority_max"] == 40.0
    assert replay_buffer["behavior_action_priority_weight"] == 0.5
    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert exploration["critic_loss"] == "huber"
    assert exploration["critic_target_clip_abs"] == 35.0
    assert exploration["actor_behavior_cloning_source"] == "warm_start_policy"
    assert exploration["actor_behavior_cloning_min_weight"] == 0.100


def test_phase6a_community_feasible_service_v45_actor_pretrain_variant_ramps_policy_loss(tmp_path):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_actor_pretrain"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_actor_pretrain_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_actor_pretrain_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasibleServiceRewardV45"
    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert exploration["actor_behavior_cloning_source"] == "warm_start_policy"
    assert exploration["actor_policy_loss_warmup_weight"] == 0.05
    assert exploration["actor_policy_loss_warmup_steps"] == 256
    assert exploration["actor_behavior_cloning_weight"] == 0.180
    assert exploration["actor_behavior_cloning_min_weight"] == 0.120
    assert exploration["actor_behavior_cloning_decay_steps"] == 256
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 16.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 4.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.50


def test_phase6a_community_feasible_service_v45_actor_pretrain_slow_variant_is_conservative(tmp_path):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_actor_pretrain_slow"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_actor_pretrain_slow_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_actor_pretrain_slow_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasibleServiceRewardV45"
    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert exploration["actor_policy_loss_weight"] == 0.60
    assert exploration["actor_policy_loss_warmup_weight"] == 0.02
    assert exploration["actor_policy_loss_warmup_steps"] == 1024
    assert exploration["actor_behavior_cloning_weight"] == 0.250
    assert exploration["actor_behavior_cloning_min_weight"] == 0.200
    assert exploration["actor_behavior_cloning_decay_steps"] == 1024
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 24.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 6.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 1.00
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 16.0


def test_phase6a_community_feasible_service_v45_teacher_clone_variant_disables_policy_loss(tmp_path):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_teacher_clone"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_teacher_clone_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_teacher_clone_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasibleServiceRewardV45"
    assert replay_buffer["behavior_action_priority_weight"] == 2.0
    assert replay_buffer["behavior_action_priority_mode"] == "positive"
    assert replay_buffer["behavior_action_priority_scope"] == "all"
    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_policy_loss_warmup_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.500
    assert exploration["actor_behavior_cloning_min_weight"] == 0.500
    assert exploration["actor_behavior_cloning_decay_steps"] == 0
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 24.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 8.0
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 20.0


def test_phase6a_community_feasible_service_v45_teacher_clone_ev_focus_variant_prioritizes_ev(tmp_path):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_teacher_clone_ev_focus"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]
    teacher_hyperparameters = exploration["warm_start_policy_hyperparameters"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasibleServiceRewardV45"
    assert replay_buffer["behavior_action_priority_weight"] == 3.0
    assert replay_buffer["behavior_action_priority_mode"] == "positive"
    assert replay_buffer["behavior_action_priority_scope"] == "ev"
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.650
    assert exploration["actor_behavior_cloning_min_weight"] == 0.650
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 36.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 16.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.25
    assert exploration["actor_storage_action_l2_penalty"] == 0.30
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 24.0
    assert teacher_hyperparameters["price_charge_rate"] == 0.0
    assert teacher_hyperparameters["storage_price_discharge_soc_floor"] == 0.30
    assert teacher_hyperparameters["storage_peak_discharge_soc_floor"] == 0.30


def test_phase6a_community_feasible_service_v45_learning_teacher_variant_uses_soft_teacher(
    tmp_path,
):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_learning_teacher"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]
    teacher_hyperparameters = exploration["warm_start_policy_hyperparameters"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasibleServiceRewardV45"
    assert exploration["warm_start_policy"] == "RBCSmartPolicy"
    assert replay_buffer["behavior_action_priority_weight"] == 3.0
    assert replay_buffer["behavior_action_priority_scope"] == "ev"
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.650
    assert exploration["actor_behavior_cloning_min_weight"] == 0.650
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 36.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 16.0
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 4.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.25
    assert teacher_hyperparameters["allow_v2g"] is False
    assert teacher_hyperparameters["ev_service_floor_rate"] == 0.25
    assert teacher_hyperparameters["ev_service_lookahead_hours"] == 4.0
    assert teacher_hyperparameters["price_charge_rate"] == 0.60
    assert teacher_hyperparameters["storage_price_discharge_soc_floor"] == 0.20


def test_phase6a_community_feasible_service_v45_learning_teacher_event_variant_focuses_ev_events(
    tmp_path,
):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_learning_teacher_event"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=[
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_event_rbc_smart"
        ],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_teacher_clone_ev_learning_teacher_event_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]
    teacher_hyperparameters = exploration["warm_start_policy_hyperparameters"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasibleServiceRewardV45"
    assert replay_buffer["priority_fraction"] == 0.30
    assert replay_buffer["priority_max"] == 60.0
    assert replay_buffer["behavior_action_priority_weight"] == 4.0
    assert replay_buffer["behavior_action_priority_scope"] == "ev"
    assert replay_buffer["observation_event_priority_weight"] == 6.0
    assert replay_buffer["observation_event_priority_mode"] == "ev_departure_service"
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.700
    assert exploration["actor_behavior_cloning_min_weight"] == 0.700
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 44.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 24.0
    assert exploration["actor_ev_behavior_cloning_positive_target_power"] == 1.15
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 4.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.15
    assert exploration["actor_storage_action_l2_penalty"] == 0.45
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 30.0
    assert exploration["warm_start_policy_phaseout_steps"] == 1024
    assert teacher_hyperparameters["allow_v2g"] is False
    assert teacher_hyperparameters["ev_service_floor_rate"] == 0.25
    assert teacher_hyperparameters["storage_price_discharge_soc_floor"] == 0.30
    assert teacher_hyperparameters["storage_peak_discharge_soc_floor"] == 0.30
    assert teacher_hyperparameters["price_discharge_rate"] == 0.35


def test_phase6a_community_feasible_precision_v46_learning_teacher_variant_caps_critic_pressure(
    tmp_path,
):
    output_dir = tmp_path / "phase6a_community_feasible_precision_v46_learning_teacher"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasiblePrecisionRewardV46"
    assert exploration["critic_target_clip_abs"] == 25.0
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.650
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 36.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 16.0
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 4.0
    assert replay_buffer["priority_fraction"] == 0.20
    assert replay_buffer["priority_max"] == 30.0
    assert replay_buffer["behavior_action_priority_weight"] == 3.0
    assert replay_buffer["behavior_action_priority_scope"] == "ev"


def test_phase6a_community_feasible_precision_v47_learning_teacher_variant_guards_over_service(
    tmp_path,
):
    output_dir = tmp_path / "phase6a_community_feasible_precision_v47_learning_teacher"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_precision_v47_teacher_clone_ev_learning_teacher_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_precision_v47_teacher_clone_ev_learning_teacher_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasiblePrecisionRewardV47"
    assert exploration["critic_target_clip_abs"] == 25.0
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.850
    assert exploration["actor_behavior_cloning_min_weight"] == 0.850
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 64.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 20.0
    assert exploration["actor_ev_behavior_cloning_positive_target_power"] == 1.15
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 16.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.12
    assert exploration["actor_storage_action_l2_penalty"] == 0.45
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 32.0
    assert replay_buffer["priority_fraction"] == 0.20
    assert replay_buffer["priority_max"] == 30.0
    assert replay_buffer["behavior_action_priority_weight"] == 4.0
    assert replay_buffer["behavior_action_priority_scope"] == "ev"


def test_phase6a_community_feasible_precision_v48_learning_teacher_variant_tightens_zero_band(
    tmp_path,
):
    output_dir = tmp_path / "phase6a_community_feasible_precision_v48_learning_teacher"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_precision_v48_zero_band_teacher_clone_ev_learning_teacher_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_precision_v48_zero_band_teacher_clone_ev_learning_teacher_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasiblePrecisionRewardV46"
    assert exploration["critic_target_clip_abs"] == 25.0
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.650
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 40.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 16.0
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 8.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.25
    assert exploration["actor_storage_action_l2_penalty"] == 0.30
    assert replay_buffer["priority_fraction"] == 0.20
    assert replay_buffer["priority_max"] == 30.0
    assert replay_buffer["behavior_action_priority_weight"] == 3.5
    assert replay_buffer["behavior_action_priority_scope"] == "ev"


def test_phase6a_community_storage_value_v49_variant_uses_storage_value_reward(tmp_path):
    output_dir = tmp_path / "phase6a_community_storage_value_v49"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_storage_value_v49_teacher_clone_ev_balanced_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_storage_value_v49_teacher_clone_ev_balanced_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityStorageValueRewardV49"
    assert exploration["critic_target_clip_abs"] == 25.0
    assert exploration["actor_policy_loss_weight"] == 0.10
    assert exploration["actor_behavior_cloning_weight"] == 0.550
    assert exploration["actor_behavior_cloning_min_weight"] == 0.350
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.60
    assert exploration["actor_storage_action_l2_penalty"] == 0.08
    assert replay_buffer["priority_fraction"] == 0.25
    assert replay_buffer["priority_max"] == 35.0
    assert replay_buffer["behavior_action_priority_weight"] == 3.25
    assert replay_buffer["behavior_action_priority_scope"] == "ev"


def test_phase6a_community_deadline_value_v50_variant_uses_deadline_reward(tmp_path):
    output_dir = tmp_path / "phase6a_community_deadline_value_v50"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_deadline_value_v50_teacher_clone_ev_balanced_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_deadline_value_v50_teacher_clone_ev_balanced_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityDeadlineValueRewardV50"
    assert exploration["critic_target_clip_abs"] == 25.0
    assert exploration["actor_policy_loss_weight"] == 0.04
    assert exploration["actor_behavior_cloning_weight"] == 0.700
    assert exploration["actor_behavior_cloning_min_weight"] == 0.600
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 48.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 22.0
    assert replay_buffer["priority_fraction"] == 0.30
    assert replay_buffer["priority_max"] == 45.0
    assert replay_buffer["behavior_action_priority_weight"] == 4.5
    assert replay_buffer["observation_event_priority_mode"] == "ev_departure_service"


def test_phase6a_community_precision_value_v51_variant_uses_precision_reward(tmp_path):
    output_dir = tmp_path / "phase6a_community_precision_value_v51"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_precision_value_v51_teacher_clone_ev_precise_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_precision_value_v51_teacher_clone_ev_precise_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityPrecisionValueRewardV51"
    assert exploration["critic_target_clip_abs"] == 25.0
    assert exploration["actor_policy_loss_weight"] == 0.08
    assert exploration["actor_behavior_cloning_weight"] == 0.620
    assert exploration["actor_behavior_cloning_min_weight"] == 0.420
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 38.0
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 14.0
    assert replay_buffer["priority_fraction"] == 0.30
    assert replay_buffer["priority_max"] == 42.0
    assert replay_buffer["behavior_action_priority_weight"] == 3.5
    assert replay_buffer["observation_event_priority_mode"] == "ev_departure_service"


def test_phase6a_community_peak_deadline_v52_variant_uses_peak_deadline_reward(tmp_path):
    output_dir = tmp_path / "phase6a_community_peak_deadline_v52"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_peak_deadline_v52_teacher_clone_ev_community_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_peak_deadline_v52_teacher_clone_ev_community_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityPeakDeadlineRewardV52"
    assert exploration["critic_target_clip_abs"] == 25.0
    assert exploration["actor_policy_loss_weight"] == 0.065
    assert exploration["actor_behavior_cloning_weight"] == 0.680
    assert exploration["actor_behavior_cloning_min_weight"] == 0.560
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 44.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.35
    assert exploration["storage_exploration_noise_multiplier"] == 0.55
    assert replay_buffer["priority_fraction"] == 0.30
    assert replay_buffer["priority_max"] == 44.0
    assert replay_buffer["behavior_action_priority_weight"] == 3.75
    assert replay_buffer["observation_event_priority_mode"] == "ev_departure_service"


def test_phase6a_community_deadline_zero_guard_v53_variant_uses_deadline_reward(tmp_path):
    output_dir = tmp_path / "phase6a_community_deadline_zero_guard_v53"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_deadline_zero_guard_v53_teacher_clone_ev_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_deadline_zero_guard_v53_teacher_clone_ev_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityDeadlineValueRewardV50"
    assert exploration["actor_policy_loss_weight"] == 0.035
    assert exploration["actor_behavior_cloning_weight"] == 0.780
    assert exploration["actor_behavior_cloning_min_weight"] == 0.700
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 58.0
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 24.0
    assert replay_buffer["priority_fraction"] == 0.35
    assert replay_buffer["priority_max"] == 50.0
    assert replay_buffer["behavior_action_priority_weight"] == 5.0
    assert replay_buffer["observation_event_priority_weight"] == 8.0


def test_phase6a_community_deadline_clone_v54_variant_anchors_to_teacher(tmp_path):
    output_dir = tmp_path / "phase6a_community_deadline_clone_v54"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_deadline_clone_v54_teacher_clone_anchor_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_deadline_clone_v54_teacher_clone_anchor_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityDeadlineValueRewardV50"
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 1.200
    assert exploration["actor_behavior_cloning_min_weight"] == 1.200
    assert exploration["actor_behavior_cloning_decay_steps"] == 0
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 80.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 32.0
    assert exploration["actor_action_l2_penalty"] == 0.0
    assert replay_buffer["priority_fraction"] == 0.40
    assert replay_buffer["priority_max"] == 60.0
    assert replay_buffer["behavior_action_priority_weight"] == 6.0


def test_phase6a_community_peak_deadline_v55_variant_uses_bc_warmup(tmp_path):
    output_dir = tmp_path / "phase6a_community_peak_deadline_v55"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_peak_deadline_v55_teacher_clone_bc_warmup_community_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_peak_deadline_v55_teacher_clone_bc_warmup_community_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityPeakDeadlineRewardV52"
    assert exploration["actor_policy_loss_weight"] == 0.025
    assert exploration["actor_behavior_cloning_weight"] == 1.000
    assert exploration["actor_behavior_cloning_min_weight"] == 0.850
    assert exploration["actor_behavior_cloning_extra_updates"] == 1
    assert exploration["actor_behavior_cloning_extra_update_end_step"] == 16 * 48
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 72.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.35
    assert replay_buffer["priority_fraction"] == 0.35
    assert replay_buffer["priority_max"] == 55.0
    assert replay_buffer["behavior_action_priority_weight"] == 5.5
    assert replay_buffer["observation_event_priority_weight"] == 8.0


def test_phase6a_community_feasible_precision_v56_variant_uses_policy_finetune(tmp_path):
    output_dir = tmp_path / "phase6a_community_feasible_precision_v56"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_precision_v56_teacher_clone_policy_finetune_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_precision_v56_teacher_clone_policy_finetune_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasiblePrecisionRewardV46"
    assert exploration["actor_policy_loss_weight"] == 0.015
    assert exploration["actor_behavior_cloning_weight"] == 0.800
    assert exploration["actor_behavior_cloning_min_weight"] == 0.650
    assert exploration["actor_behavior_cloning_extra_updates"] == 1
    assert exploration["actor_behavior_cloning_extra_update_end_step"] == 16 * 40
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 48.0
    assert exploration["actor_storage_action_l2_penalty"] == 0.20
    assert replay_buffer["priority_fraction"] == 0.25
    assert replay_buffer["priority_max"] == 35.0
    assert replay_buffer["behavior_action_priority_weight"] == 4.0
    assert replay_buffer["observation_event_priority_weight"] == 4.0


def test_phase6a_rbc_smart_warm_start_uses_dataset_specific_teacher_hyperparameters(tmp_path):
    output_dir = tmp_path / "phase6a_rbc_smart_teacher_hyperparameters"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["2022"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_2022_maddpg_community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    teacher_hyperparameters = payload["algorithm"]["exploration"]["params"][
        "warm_start_policy_hyperparameters"
    ]

    assert teacher_hyperparameters["allow_v2g"] is True
    assert teacher_hyperparameters["price_charge_rate"] == 0.15
    assert teacher_hyperparameters["storage_price_charge_soc_ceiling"] == 0.85
    assert teacher_hyperparameters["storage_price_discharge_soc_floor"] == 0.30
    assert teacher_hyperparameters["storage_peak_discharge_soc_floor"] == 0.30


def test_phase6a_community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_variant_keeps_bc_gate(
    tmp_path,
):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasibleServiceRewardV45"
    assert replay_buffer["behavior_action_priority_weight"] == 3.0
    assert replay_buffer["behavior_action_priority_scope"] == "ev"
    assert exploration["actor_policy_loss_weight"] == 0.12
    assert exploration["actor_policy_loss_warmup_weight"] == 0.01
    assert exploration["actor_policy_loss_warmup_start_step"] == 32
    assert exploration["actor_policy_loss_warmup_steps"] == 1536
    assert exploration["actor_behavior_cloning_weight"] == 0.650
    assert exploration["actor_behavior_cloning_min_weight"] == 0.450
    assert exploration["actor_behavior_cloning_decay_steps"] == 1536
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 40.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 18.0
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.35
    assert exploration["actor_storage_action_l2_penalty"] == 0.25
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 28.0
    assert exploration["warm_start_policy_phaseout_steps"] == 1024


def test_phase6a_community_feasible_service_v45_teacher_clone_ev_focus_event_variant_enables_event_replay(
    tmp_path,
):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_teacher_clone_ev_focus_event"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_teacher_clone_ev_focus_event_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_teacher_clone_ev_focus_event_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasibleServiceRewardV45"
    assert replay_buffer["priority_fraction"] == 0.35
    assert replay_buffer["priority_max"] == 60.0
    assert replay_buffer["behavior_action_priority_weight"] == 4.0
    assert replay_buffer["behavior_action_priority_scope"] == "ev"
    assert replay_buffer["observation_event_priority_weight"] == 8.0
    assert replay_buffer["observation_event_priority_mode"] == "ev_departure_service"
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.700
    assert exploration["actor_behavior_cloning_min_weight"] == 0.700
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 44.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 24.0
    assert exploration["actor_ev_behavior_cloning_positive_target_power"] == 1.25
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.20
    assert exploration["actor_storage_action_l2_penalty"] == 0.35
    assert exploration["actor_ev_v2g_action_l2_penalty"] == 30.0
    assert exploration["warm_start_policy_phaseout_steps"] == 768


def test_phase6a_community_feasible_service_v45_teacher_clone_ev_guarded_band_variant_adds_light_stop_weight(
    tmp_path,
):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_teacher_clone_ev_guarded_band"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_teacher_clone_ev_guarded_band_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_teacher_clone_ev_guarded_band_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert replay_buffer["behavior_action_priority_weight"] == 3.0
    assert replay_buffer["behavior_action_priority_scope"] == "ev"
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.650
    assert exploration["actor_behavior_cloning_min_weight"] == 0.650
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 36.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 16.0
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 4.0
    assert exploration["actor_ev_behavior_cloning_zero_target_threshold"] == 0.04
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.25


def test_phase6a_community_feasible_service_v45_teacher_clone_ev_band_variant_weights_ev_stop(tmp_path):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_teacher_clone_ev_band"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_teacher_clone_ev_band_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_teacher_clone_ev_band_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert payload["simulator"]["reward_function"] == "CostServiceCommunityFeasibleServiceRewardV45"
    assert replay_buffer["behavior_action_priority_weight"] == 2.5
    assert replay_buffer["behavior_action_priority_scope"] == "ev"
    assert exploration["actor_policy_loss_weight"] == 0.0
    assert exploration["actor_behavior_cloning_weight"] == 0.600
    assert exploration["actor_behavior_cloning_min_weight"] == 0.600
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 32.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 12.0
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 12.0
    assert exploration["actor_ev_behavior_cloning_zero_target_threshold"] == 0.04
    assert exploration["actor_storage_behavior_cloning_multiplier"] == 0.25


def test_phase6a_community_feasible_service_v45_teacher_clone_ev_balanced_variant_blends_charge_and_stop(tmp_path):
    output_dir = tmp_path / "phase6a_community_feasible_service_v45_teacher_clone_ev_balanced"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["maddpg"],
        maddpg_variant=["community_feasible_service_v45_teacher_clone_ev_balanced_rbc_smart"],
        seed=[123],
        episodes=1,
        deterministic_finish=False,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=16,
        warm_start_phaseout_steps=None,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    phase6a.run_phase6a(args)

    config_path = (
        output_dir
        / "generated_configs"
        / "phase6a_15s_maddpg_community_feasible_service_v45_teacher_clone_ev_balanced_rbc_smart_seed123.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exploration = payload["algorithm"]["exploration"]["params"]
    replay_buffer = payload["algorithm"]["replay_buffer"]

    assert replay_buffer["behavior_action_priority_weight"] == 2.75
    assert replay_buffer["behavior_action_priority_scope"] == "ev"
    assert exploration["actor_behavior_cloning_weight"] == 0.625
    assert exploration["actor_behavior_cloning_min_weight"] == 0.625
    assert exploration["actor_ev_behavior_cloning_multiplier"] == 34.0
    assert exploration["actor_ev_behavior_cloning_positive_target_weight"] == 14.0
    assert exploration["actor_ev_behavior_cloning_zero_target_weight"] == 8.0
    assert exploration["actor_ev_behavior_cloning_zero_target_threshold"] == 0.04
