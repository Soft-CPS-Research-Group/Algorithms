from __future__ import annotations

from pathlib import Path

import yaml

from scripts.build_cc_matd3_level1_v1 import (
    CHECKPOINT_PATH,
    DEFAULT_BASE_CONFIG,
    build_configs,
)
from utils.config_schema import validate_config


def _load(path: Path):
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_cc_matd3_level1_configs_freeze_the_checkpointed_leaf(tmp_path) -> None:
    paths = build_configs(DEFAULT_BASE_CONFIG, tmp_path)

    assert len(paths) == 7
    for path in paths:
        config = _load(path)
        leaf = config.pipeline[-1]
        assert leaf.algorithm == "MATD3"
        assert leaf.frozen is True
        assert config.checkpointing.checkpoint_mode == "inference"
        assert config.checkpointing.stage_checkpoint_local_paths[1] == CHECKPOINT_PATH
        assert config.simulator.community_market.enabled is True
        assert config.simulator.episode_time_steps in {96, 35040}
        assert config.tracking.max_step_seconds == 900.0


def test_cc_matd3_level1_neutral_replay_is_an_explicit_gate(tmp_path) -> None:
    paths = build_configs(DEFAULT_BASE_CONFIG, tmp_path)
    neutral_path = next(path for path in paths if "fixed_1p0_seed789" in path.name)
    config = _load(neutral_path)

    assert [stage.algorithm for stage in config.pipeline] == [
        "FixedPriceSignal",
        "MATD3",
    ]
    assert config.pipeline[0].hyperparameters.multiplier == 1.0
    assert config.simulator.episodes == 1
    assert config.tracking.tags["fixed_multiplier"] == "1.0"
    assert config.pipeline[1].exploration.params[
        "local_price_forecast_mode"
    ] == "real_unmodified"
