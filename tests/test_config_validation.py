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


def test_validate_all_templates():
    template_paths = sorted(Path("configs/templates").glob("*.yaml"))
    assert template_paths, "No template files found under configs/templates"

    for template_path in template_paths:
        with template_path.open("r", encoding="utf-8") as handle:
            template_config = yaml.safe_load(handle)
        validate_config(template_config)
