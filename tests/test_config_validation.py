import copy

import pytest

from utils.config_schema import validate_config


@pytest.fixture
def base_config():
    from pathlib import Path
    import yaml

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
