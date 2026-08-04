from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.generate_ppo_cc_settlement_smokes import (
    SMOKE_PROTOCOL,
    SMOKE_STEPS,
    generate_smokes,
)
from utils.config_schema import validate_config


EXPECTED_NAMES = {
    "smart_settlement_smoke.yaml",
    "cc_smart_settlement_smoke_seed123.yaml",
    "ppo_settlement_smoke_seed789.yaml",
    "cc_ppo_settlement_smoke_seed789.yaml",
}
ROOT = Path(__file__).resolve().parents[1]


def _load(paths: list[Path]) -> dict[str, dict]:
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in paths
    }


def test_smoke_generator_produces_valid_settlement_configs(tmp_path: Path):
    paths = generate_smokes(tmp_path)
    assert {path.name for path in paths} == EXPECTED_NAMES

    for config in _load(paths).values():
        validate_config(config)
        simulator = config["simulator"]
        export = simulator["export"]
        market = simulator["community_market"]

        assert config["metadata"]["experiment_name"] == SMOKE_PROTOCOL
        assert config["tracking"]["tags"]["protocol"] == SMOKE_PROTOCOL
        assert config["tracking"]["tags"]["evidence"] == "smoke"
        assert config["tracking"]["mlflow_enabled"] is False
        assert simulator["simulation_start_time_step"] == 0
        assert simulator["simulation_end_time_step"] == SMOKE_STEPS - 1
        assert simulator["episode_time_steps"] == SMOKE_STEPS
        assert simulator["deterministic_finish"] is True
        assert market["enabled"] is True
        assert market["local_price_ratio_to_grid_import"] == 0.8
        assert market["intra_community_sell_ratio"] == 0.8
        assert market["grid_export_price"] == 0.0
        assert export["export_kpis_on_episode_end"] is True
        assert export["final_episode_only"] is True
        assert export["include_business_as_usual"] is True


def test_smoke_generator_runs_directly_outside_the_repository(tmp_path: Path):
    output_dir = tmp_path / "generated"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_ppo_cc_settlement_smokes.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert {path.name for path in output_dir.glob("*.yaml")} == EXPECTED_NAMES
    assert str(output_dir / "smart_settlement_smoke.yaml") in completed.stdout


def test_cc_smokes_force_bc_and_one_real_ppo_update(tmp_path: Path):
    configs = _load(generate_smokes(tmp_path))
    learned = (
        configs["cc_smart_settlement_smoke_seed123.yaml"],
        configs["cc_ppo_settlement_smoke_seed789.yaml"],
    )
    neutral = (
        configs["smart_settlement_smoke.yaml"],
        configs["ppo_settlement_smoke_seed789.yaml"],
    )

    for config in learned:
        params = config["pipeline"][0]["hyperparameters"]
        assert config["simulator"]["episodes"] == 3
        assert params["cc_action_interval"] == 4
        assert params["num_steps"] == 96
        assert params["bc_collect_steps"] == 96
        assert params["bc_train_steps"] == 2000
        assert config["checkpointing"]["checkpoint_interval"] == SMOKE_STEPS - 1

    for config in neutral:
        assert config["simulator"]["episodes"] == 1
        assert config["pipeline"][0]["algorithm"] == "FixedPriceSignal"
        assert config["checkpointing"]["checkpoint_interval"] is None


def test_smoke_pairs_keep_byte_equivalent_frozen_leaves(tmp_path: Path):
    configs = _load(generate_smokes(tmp_path))
    assert (
        configs["smart_settlement_smoke.yaml"]["pipeline"][1]
        == configs["cc_smart_settlement_smoke_seed123.yaml"]["pipeline"][1]
    )
    assert (
        configs["ppo_settlement_smoke_seed789.yaml"]["pipeline"][1]
        == configs["cc_ppo_settlement_smoke_seed789.yaml"]["pipeline"][1]
    )


@pytest.mark.parametrize("steps", [0, 384, 389])
def test_smoke_window_must_match_one_canonical_cc_rollout(
    tmp_path: Path,
    steps: int,
):
    with pytest.raises(ValueError):
        generate_smokes(tmp_path, steps=steps)
