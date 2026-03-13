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


def test_validate_config_missing_algorithm(base_config):
    config = copy.deepcopy(base_config)
    config["algorithm"] = None
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_invalid_network_layers(base_config):
    config = copy.deepcopy(base_config)
    config["algorithm"]["networks"]["actor"]["layers"] = []
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
    }
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
