from __future__ import annotations

from pathlib import Path

import yaml

from scripts.generate_matd3_settlement_recovery_v2 import generate
from utils.config_schema import validate_config


def _load(path: Path):
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_matd3_recovery_config_preserves_annual_science_and_salvage(tmp_path) -> None:
    config = _load(generate(tmp_path))

    assert config.simulator.episodes == 4
    assert config.simulator.deterministic_finish is True
    assert config.simulator.simulation_end_time_step == 35039
    assert config.checkpointing.checkpoint_interval == 35040
    assert config.checkpointing.checkpoint_mode == "inference"
    assert config.pipeline[0].algorithm == "MATD3"
    assert config.pipeline[0].exploration.params[
        "residual_policy_runtime_only_export"
    ] is True
    assert config.tracking.stall_watchdog_enabled is True
    assert config.tracking.runtime_profiling_interval == 512


def test_matd3_recovery_smoke_crosses_two_profiling_boundaries(tmp_path) -> None:
    config = _load(generate(tmp_path, smoke=True))

    assert config.simulator.episodes == 1
    assert config.simulator.episode_time_steps == 1024
    assert config.simulator.simulation_end_time_step == 1023
    assert config.checkpointing.checkpoint_interval == 1024
