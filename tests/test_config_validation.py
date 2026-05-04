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
        "session_name": "session-a",
    }
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
    template_paths = sorted(Path("configs/templates").glob("*.yaml"))
    assert template_paths, "No template files found under configs/templates"

    for template_path in template_paths:
        with template_path.open("r", encoding="utf-8") as handle:
            template_config = yaml.safe_load(handle)
        validate_config(template_config)
