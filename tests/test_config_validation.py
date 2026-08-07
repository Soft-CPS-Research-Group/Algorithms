import copy
from pathlib import Path

import pytest
import yaml

from utils.config_schema import validate_config


@pytest.fixture
def base_config():
    config_path = Path("configs/config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_validate_config_success(base_config):
    # Should not raise
    validate_config(base_config)


def test_validate_config_accepts_metadata_community_name(base_config):
    config = copy.deepcopy(base_config)
    config["metadata"]["community_name"] = "porto_cluster_a"
    validate_config(config)


def test_validate_config_accepts_building_local_entity_profile(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["entity_encoding"]["profile"] = "building_local_v1"
    validate_config(config)


def test_validate_config_accepts_strict_local_rbc_policy(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"] = [
        {
            "algorithm": "RBCSmartLocalPolicy",
            "count": 1,
            "hyperparameters": {},
        }
    ]
    validate_config(config)


def test_validate_config_rejects_non_leaf_transformer_ppo() -> None:
    config_path = Path("configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["pipeline"].append(
        {"algorithm": "RuleBasedPolicy", "count": 1, "hyperparameters": {}}
    )

    with pytest.raises(ValueError, match="must be the final pipeline stage"):
        validate_config(config)


def test_validate_config_rejects_legacy_algorithm_key(base_config):
    config = copy.deepcopy(base_config)
    config.pop("pipeline", None)
    config["algorithm"] = {
        "name": "RuleBasedPolicy",
        "hyperparameters": {},
    }
    with pytest.raises(ValueError, match="deprecated top-level 'algorithm'"):
        validate_config(config)


def test_validate_config_missing_pipeline(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"] = None
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_empty_pipeline(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"] = []
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_invalid_network_layers(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"][0]["networks"]["actor"]["layers"] = []
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_accepts_ev_stratified_replay_sampling(base_config):
    config = copy.deepcopy(base_config)
    replay = config["pipeline"][0]["replay_buffer"]
    replay["behavior_action_priority_scope"] = "ev"
    replay["behavior_action_stratified_sampling"] = True
    replay["behavior_action_positive_threshold"] = 0.1

    model = validate_config(config)
    parsed = model.pipeline[0].replay_buffer

    assert parsed.behavior_action_stratified_sampling is True
    assert parsed.behavior_action_positive_threshold == pytest.approx(0.1)


def test_validate_config_rejects_ev_stratified_sampling_without_ev_scope(base_config):
    config = copy.deepcopy(base_config)
    replay = config["pipeline"][0]["replay_buffer"]
    replay["behavior_action_priority_scope"] = "all"
    replay["behavior_action_stratified_sampling"] = True

    with pytest.raises(ValueError, match="requires behavior_action_priority_scope='ev'"):
        validate_config(config)


def test_validate_config_accepts_late_fusion_critic_layers(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"][0]["networks"]["critic"] = {
        "class": "LateFusionCritic",
        "layers": [64, 32],
        "state_layers": [64],
        "action_layers": [32],
        "joint_layers": [64, 32],
        "lr": 1.0e-3,
    }
    validate_config(config)


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/templates/cc_local.yaml"),
        Path("configs/templates/hiro_local.yaml"),
    ],
)
def test_validate_config_accepts_hierarchical_templates(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        validate_config(yaml.safe_load(handle))


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/templates/cc_local.yaml"),
        Path("configs/templates/cc_level2_local.yaml"),
    ],
)
def test_cc_templates_use_complete_annual_horizon(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["simulator"]["episode_time_steps"] == 35040


def test_cc_level1_bc_collection_matches_complete_annual_horizon():
    with Path("configs/templates/cc_local.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["pipeline"][0]["hyperparameters"]["bc_collect_steps"] == 8760


def test_fixed_price_schedule_requires_ordered_entries_from_step_zero():
    from utils.config_schema import FixedPriceSignalHyperparameters

    valid = FixedPriceSignalHyperparameters(
        schedule=[
            {"start_step": 0, "multiplier": 1.05},
            {"start_step": 96, "multiplier": 1.025},
        ]
    )
    assert valid.schedule is not None
    assert valid.schedule[1].start_step == 96

    with pytest.raises(ValueError, match="start at step 0"):
        FixedPriceSignalHyperparameters(
            schedule=[{"start_step": 96, "multiplier": 1.025}]
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        FixedPriceSignalHyperparameters(
            schedule=[
                {"start_step": 0, "multiplier": 1.05},
                {"start_step": 0, "multiplier": 1.025},
            ]
        )

    vector = FixedPriceSignalHyperparameters(
        vector_schedule=[
            {"start_step": 0, "multipliers": [1.0, 1.0]},
            {"start_step": 96, "multipliers": [0.7, 1.3]},
        ]
    )
    assert vector.vector_schedule is not None
    assert vector.vector_schedule[1].multipliers == [0.7, 1.3]

    with pytest.raises(ValueError, match="equal widths"):
        FixedPriceSignalHyperparameters(
            vector_schedule=[
                {"start_step": 0, "multipliers": [1.0, 1.0]},
                {"start_step": 96, "multipliers": [0.7]},
            ]
        )

    with pytest.raises(ValueError, match="must not be empty"):
        FixedPriceSignalHyperparameters(vector_schedule=[])

    with pytest.raises(ValueError, match="mutually exclusive"):
        FixedPriceSignalHyperparameters(
            multipliers=[1.0, 1.0],
            vector_schedule=[
                {"start_step": 0, "multipliers": [1.0, 1.0]},
            ],
        )


def test_to_dict_removes_none_network_optional_layers_from_pipeline(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"][0]["networks"]["critic"] = {
        "class": "LateFusionCritic",
        "layers": [64, 32],
        "state_layers": None,
        "action_layers": None,
        "joint_layers": None,
        "lr": 1.0e-3,
    }
    config["pipeline"][0]["networks"]["actor"]["head_layers"] = None

    resolved = validate_config(config).to_dict()

    critic = resolved["pipeline"][0]["networks"]["critic"]
    actor = resolved["pipeline"][0]["networks"]["actor"]
    assert "state_layers" not in critic
    assert "action_layers" not in critic
    assert "joint_layers" not in critic
    assert "head_layers" not in actor


def test_validate_config_accepts_deucalion_execution(base_config):
    config = copy.deepcopy(base_config)
    config["execution"] = {
        "deucalion": {
            "command_mode": "run",
            "gpus": 0,
            "datasets": ["datasets/citylearn_charging_constraints_demo"],
            "required_paths": ["/projects/F202508843CPCAA0/tiagocalof/images/simulator.sif"],
        }
    }
    validate_config(config)


def test_validate_config_rejects_invalid_deucalion_dataset(base_config):
    config = copy.deepcopy(base_config)
    config["execution"] = {
        "deucalion": {
            "datasets": ["/absolute/path/not/allowed"],
        }
    }
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_accepts_bundle_section(base_config):
    config = copy.deepcopy(base_config)
    config["bundle"] = {
        "bundle_version": "2026-03-10-v1",
        "description": "Validation test",
        "alias_mapping_path": "aliases.json",
        "require_observations_envelope": True,
        "artifact_config": {"input_site_key": "site_a"},
        "per_agent_artifact_config": {
            "0": {"input_site_key": "boavista"},
            "1": {"input_site_key": "sao_mamede"},
        },
    }
    validate_config(config)


def test_validate_config_accepts_selected_pipeline_stage_checkpoint(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "CCLevel1",
            "count": 1,
            "hyperparameters": {},
        },
    )
    config["checkpointing"].update(
        {
            "resume_training": True,
            "checkpoint_local_path": None,
            "checkpoint_run_id": None,
            "stage_checkpoint_local_paths": {1: "runs/jobs/local-ppo/checkpoints"},
        }
    )

    parsed = validate_config(config)

    assert parsed.checkpointing.stage_checkpoint_local_paths == {
        1: "runs/jobs/local-ppo/checkpoints"
    }


def test_validate_config_accepts_fixed_neutral_price_signal_manager(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "FixedPriceSignal",
            "count": 1,
            "frozen": True,
            "hyperparameters": {"multiplier": 1.0},
        },
    )

    parsed = validate_config(config)

    assert parsed.pipeline[0].algorithm == "FixedPriceSignal"
    assert parsed.pipeline[0].hyperparameters.multiplier == 1.0


def test_validate_config_accepts_fixed_per_member_price_signal_manager(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "FixedPriceSignal",
            "count": 1,
            "frozen": True,
            "hyperparameters": {"multipliers": [0.9, 1.05]},
        },
    )

    parsed = validate_config(config)

    assert parsed.pipeline[0].hyperparameters.multipliers == [0.9, 1.05]


@pytest.mark.parametrize("multipliers", [[], [0.9, 0.0], [-1.0, 0.9]])
def test_validate_config_rejects_invalid_fixed_price_signal_vector(base_config, multipliers):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "FixedPriceSignal",
            "count": 1,
            "frozen": True,
            "hyperparameters": {"multipliers": multipliers},
        },
    )

    with pytest.raises(Exception, match="multipliers"):
        validate_config(config)


def test_validate_config_rejects_invalid_cc_price_range(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "CCLevel1",
            "count": 1,
            "hyperparameters": {"price_min": 1.0, "price_max": 1.0},
        },
    )

    with pytest.raises(Exception, match="price_max must be greater"):
        validate_config(config)


def test_validate_config_rejects_out_of_range_pipeline_stage_checkpoint(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "CCLevel1",
            "count": 1,
            "hyperparameters": {},
        },
    )
    config["checkpointing"].update(
        {
            "resume_training": True,
            "checkpoint_local_path": None,
            "checkpoint_run_id": None,
            "stage_checkpoint_local_paths": {2: "runs/jobs/missing/checkpoints"},
        }
    )

    with pytest.raises(Exception, match="outside pipeline range"):
        validate_config(config)


def test_validate_config_rejects_invalid_per_agent_artifact_config(base_config):
    config = copy.deepcopy(base_config)
    config["bundle"]["per_agent_artifact_config"] = {"0": ["invalid"]}
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_accepts_simulator_export_and_time_controls(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["progress_updates_enabled"] = False
    config["tracking"]["progress_update_interval"] = 3
    config["tracking"]["system_metrics_enabled"] = True
    config["tracking"]["system_metrics_interval"] = 12
    config["checkpointing"]["require_update_step"] = False
    config["checkpointing"]["require_initial_exploration_done"] = False
    config["simulator"]["simulation_start_time_step"] = 0
    config["simulator"]["simulation_end_time_step"] = 95
    config["simulator"]["episodes"] = 2
    config["simulator"]["episode_time_steps"] = 24
    config["simulator"]["export"] = {
        "mode": "end",
        "export_kpis_on_episode_end": True,
        "final_episode_only": True,
        "include_business_as_usual": False,
        "export_business_as_usual_timeseries": False,
        "kpi_round_decimals": 4,
        "session_name": "session-a",
    }
    validate_config(config)


def test_validate_config_accepts_runtime_safety_guards(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["progress_phase_updates_enabled"] = True
    config["tracking"]["progress_phase_start_step"] = 4500
    config["tracking"]["progress_phase_end_step"] = 5700
    config["tracking"]["max_step_seconds"] = 15.0
    config["tracking"]["stall_watchdog_enabled"] = True
    config["tracking"]["stall_watchdog_timeout_seconds"] = 900.0
    config["tracking"]["stall_watchdog_exit_on_timeout"] = True
    config["tracking"]["stall_watchdog_repeat"] = False
    config["tracking"]["stall_watchdog_traceback_file"] = "logs/stall_watchdog.log"
    config["tracking"]["stall_watchdog_context_interval_steps"] = 64
    config["tracking"]["resource_guard_enabled"] = True
    config["tracking"]["max_process_rss_mb"] = 12000.0
    config["tracking"]["min_available_ram_mb"] = 1024.0
    validate_config(config)


def test_validate_config_accepts_max_update_seconds(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["max_update_seconds"] = 2400.0

    resolved = validate_config(config).to_dict()

    assert resolved["tracking"]["max_update_seconds"] == 2400.0


@pytest.mark.parametrize("value", [0, -1.0])
def test_validate_config_rejects_non_positive_max_update_seconds(base_config, value):
    config = copy.deepcopy(base_config)
    config["tracking"]["max_update_seconds"] = value

    with pytest.raises(Exception):
        validate_config(config)


@pytest.fixture
def transformer_ppo_template_config():
    config_path = Path("configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_validate_config_accepts_transformer_ppo_require_cuda(transformer_ppo_template_config):
    transformer_ppo_template_config["pipeline"][0]["hyperparameters"]["require_cuda"] = True

    resolved = validate_config(transformer_ppo_template_config).to_dict()

    assert resolved["pipeline"][0]["hyperparameters"]["require_cuda"] is True


def test_validate_config_rejects_enabled_transformer_ppo_bc_without_demonstrations(
    transformer_ppo_template_config,
):
    transformer_ppo_template_config["pipeline"][0]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": 0,
    }

    with pytest.raises(ValueError, match="demonstration_episodes.*at least 1"):
        validate_config(transformer_ppo_template_config)


def test_validate_config_accepts_disabled_transformer_ppo_bc_without_demonstrations(
    transformer_ppo_template_config,
):
    transformer_ppo_template_config["pipeline"][0]["behavior_cloning"] = {
        "enabled": False,
        "demonstration_episodes": 0,
    }

    validate_config(transformer_ppo_template_config)


def test_validate_config_accepts_enabled_transformer_ppo_bc_with_demonstrations(
    transformer_ppo_template_config,
):
    transformer_ppo_template_config["pipeline"][0]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": 1,
    }

    validate_config(transformer_ppo_template_config)


def test_validate_config_rejects_invalid_runtime_safety_guards(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["progress_phase_start_step"] = 5700
    config["tracking"]["progress_phase_end_step"] = 4500
    with pytest.raises(Exception):
        validate_config(config)

    config = copy.deepcopy(base_config)
    config["tracking"]["max_step_seconds"] = 0
    with pytest.raises(Exception):
        validate_config(config)

    config = copy.deepcopy(base_config)
    config["tracking"]["stall_watchdog_timeout_seconds"] = 0
    with pytest.raises(Exception):
        validate_config(config)

    config = copy.deepcopy(base_config)
    config["tracking"]["stall_watchdog_context_interval_steps"] = 0
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_accepts_wrapper_reward_overrides(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["wrapper_reward"] = {
        "enabled": True,
        "profile": "cost_limits_v1",
        "clip_enabled": True,
        "clip_min": -5.0,
        "clip_max": 5.0,
        "squash": "tanh",
    }
    validate_config(config)


def test_validate_config_rejects_wrapper_reward_invalid_clip_range(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["wrapper_reward"]["clip_min"] = 1.0
    config["simulator"]["wrapper_reward"]["clip_max"] = -1.0
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_rejects_invalid_simulator_export_mode(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["export"] = {
        "mode": "invalid-mode",
        "export_kpis_on_episode_end": False,
    }
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_rejects_invalid_simulation_window(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["simulation_start_time_step"] = 50
    config["simulator"]["simulation_end_time_step"] = 10
    with pytest.raises(Exception):
        validate_config(config)

    config = copy.deepcopy(base_config)
    config["simulator"]["episodes"] = 0
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_rejects_dynamic_topology_without_entity_interface(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "flat"
    config["simulator"]["topology_mode"] = "dynamic"
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_rejects_maddpg_with_entity_dynamic(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["topology_mode"] = "dynamic"
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_accepts_rule_based_with_entity_dynamic(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["topology_mode"] = "dynamic"
    config["simulator"]["dataset_name"] = "citylearn_three_phase_dynamic_topology_demo_v1"
    config["simulator"]["dataset_path"] = "./datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json"
    config["pipeline"] = [
        {
            "algorithm": "RuleBasedPolicy",
            "count": 1,
            "hyperparameters": {
                "pv_charge_threshold": 0.0,
                "flexibility_hours": 3.0,
                "emergency_hours": 1.0,
                "pv_preferred_charge_rate": 0.6,
                "flex_trickle_charge": 0.0,
                "min_charge_rate": 0.0,
                "emergency_charge_rate": 1.0,
                "energy_epsilon": 1e-3,
                "default_capacity_kwh": 60.0,
                "non_flexible_chargers": [],
            },
            "networks": None,
            "replay_buffer": None,
            "exploration": None,
        }
    ]
    validate_config(config)


def test_validate_config_accepts_signal_aware_rbc_stage(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"] = [
        {
            "algorithm": "SignalAwareRBC",
            "count": 17,
            "hyperparameters": {},
            "networks": None,
            "replay_buffer": None,
            "exploration": None,
        }
    ]

    validate_config(config)


def test_validate_config_rejects_invalid_mlflow_artifacts_profile(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["mlflow_artifacts_profile"] = "all"
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_rejects_invalid_tracking_intervals(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["progress_update_interval"] = 0
    with pytest.raises(Exception):
        validate_config(config)

    config = copy.deepcopy(base_config)
    config["tracking"]["system_metrics_interval"] = 0
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_all_templates():
    # Experimental templates intentionally use placeholder algorithms (e.g.
    # SingleAgentRL) that are not yet runtime-backed.  Exclude them here.
    template_paths = sorted(
        p
        for p in Path("configs/templates").rglob("*.yaml")
        if "experimental" not in p.parts
    )
    assert template_paths, "No template files found under configs/templates"

    for template_path in template_paths:
        with template_path.open("r", encoding="utf-8") as handle:
            template_config = yaml.safe_load(handle)
        validate_config(template_config)


def test_rbc_community_templates_use_community_settlement_objective():
    template_paths = [
        Path("configs/templates/baselines/rbc_community_local.yaml"),
        Path("configs/templates/baselines/rbc_community_2022_all_plus_evs_local.yaml"),
    ]

    for template_path in template_paths:
        with template_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        kwargs = config["simulator"]["reward_function_kwargs"]
        hyper = config["pipeline"][0]["hyperparameters"]

        assert kwargs["local_cost_weight"] == pytest.approx(0.0)
        assert kwargs["community_settlement_cost_weight"] == pytest.approx(1.0)
        assert kwargs["community_local_price_ratio"] == pytest.approx(0.8)
        assert kwargs["community_grid_export_price"] == pytest.approx(0.0)
        assert hyper["community_local_price_ratio"] == pytest.approx(0.8)
        assert hyper["community_grid_export_price"] == pytest.approx(0.0)


def test_rbc_smart_templates_keep_house_level_cost_objective():
    template_paths = [
        Path("configs/templates/baselines/rbc_smart_local.yaml"),
        Path("configs/templates/baselines/rbc_smart_2022_all_plus_evs_local.yaml"),
    ]

    for template_path in template_paths:
        with template_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        kwargs = config["simulator"]["reward_function_kwargs"]

        assert kwargs.get("local_cost_weight", 1.0) == pytest.approx(1.0)
        assert kwargs["community_settlement_cost_weight"] == pytest.approx(0.0)
