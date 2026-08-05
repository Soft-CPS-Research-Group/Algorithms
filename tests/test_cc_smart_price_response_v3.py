from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.generate_cc_smart_price_response_v3 import (
    ADAPTIVE_RECIPE,
    DENSE_RECIPE,
    FIXED_MULTIPLIERS,
    SEED,
    generate,
)
from utils.config_schema import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/cc_smart_price_response_v3"


def _load(name: str) -> dict:
    config = yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))
    validate_config(config)
    return config


@pytest.mark.parametrize("multiplier", FIXED_MULTIPLIERS)
def test_fixed_probes_freeze_the_comparison_surface(multiplier: float):
    name = f"cc_smart_fixed_{multiplier:.1f}.yaml".replace(".", "p", 1)
    config = _load(name)
    simulator = config["simulator"]

    assert simulator["dataset_name"] == (
        "citylearn_three_phase_electrical_service_demo_15min_parquet"
    )
    assert simulator["simulation_start_time_step"] == 0
    assert simulator["simulation_end_time_step"] == 35039
    assert simulator["episodes"] == 1
    assert simulator["community_market"]["enabled"] is True
    assert simulator["community_market"]["local_price_ratio_to_grid_import"] == 0.8
    assert config["pipeline"][0]["algorithm"] == "FixedPriceSignal"
    assert config["pipeline"][0]["hyperparameters"]["multiplier"] == pytest.approx(
        multiplier
    )
    assert config["pipeline"][1]["algorithm"] == "SignalAwareRBC"
    assert config["pipeline"][1]["frozen"] is True


def test_update_dense_probe_preserves_v1_but_increases_real_updates():
    config = _load(f"cc_smart_{DENSE_RECIPE}_seed{SEED}.yaml")
    simulator = config["simulator"]
    params = config["pipeline"][0]["hyperparameters"]

    assert simulator["episodes"] == 8
    assert simulator["reward_function_kwargs"] == {
        "cost_aggregation": "community_net",
        "w_cost": 1.0,
        "w_peak": 0.6,
        "w_ramp": 0.4,
        "w_export": 0.05,
        "w_violation": 2.0,
    }
    assert params["num_steps"] == 96
    assert params["gamma"] == pytest.approx(0.99)
    assert params["mini_batch_size"] == 64
    assert params["w_factor"] == pytest.approx(0.3)
    assert params["w_smoothness"] == pytest.approx(0.1)
    assert config["tracking"]["tags"]["planned_ppo_update_count_approx"] == "547"


def test_post_sweep_incumbent_residual_is_bounded_around_fixed_1p3():
    config = _load(f"cc_smart_{ADAPTIVE_RECIPE}_seed{SEED}.yaml")
    params = config["pipeline"][0]["hyperparameters"]

    assert config["tracking"]["tags"]["selection_basis"] == "post_fixed_sweep"
    assert params["price_min"] == pytest.approx(0.5)
    assert params["price_max"] == pytest.approx(1.3)
    assert params["reference_multiplier"] == pytest.approx(1.3)
    assert params["policy_residual_scale"] == pytest.approx(0.5)
    assert params["w_factor"] == pytest.approx(0.10)
    assert params["w_smoothness"] == pytest.approx(0.05)
    assert params["num_steps"] == 96

    effective_min = params["reference_multiplier"] + params["policy_residual_scale"] * (
        params["price_min"] - params["reference_multiplier"]
    )
    effective_max = params["reference_multiplier"] + params["policy_residual_scale"] * (
        params["price_max"] - params["reference_multiplier"]
    )
    assert effective_min == pytest.approx(0.9)
    assert effective_max == pytest.approx(1.3)


def test_generated_templates_match_committed_templates(tmp_path: Path):
    generated = generate(tmp_path)
    for path in generated:
        committed = CONFIG_ROOT / path.name
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == yaml.safe_load(
            committed.read_text(encoding="utf-8")
        )
