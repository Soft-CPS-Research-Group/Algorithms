from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.generate_cc_smart_cost_focus_v2 import (
    ANNUAL_SMART_REFERENCES,
    RECIPE_NAMES,
    SEED,
    SMOKE_STEPS,
    generate,
    generate_smokes,
)
from utils.config_schema import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/cc_smart_cost_focus_v2"


def _load(recipe_name: str) -> dict:
    path = CONFIG_ROOT / f"cc_smart_{recipe_name}_seed{SEED}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(payload)
    return payload


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_cost_focus_v2_configs_validate_and_freeze_common_contract(recipe_name: str):
    config = _load(recipe_name)
    simulator = config["simulator"]
    manager = config["pipeline"][0]

    assert config["tracking"]["mlflow_enabled"] is False
    assert simulator["dataset_name"] == (
        "citylearn_three_phase_electrical_service_demo_15min_parquet"
    )
    assert simulator["simulation_start_time_step"] == 0
    assert simulator["simulation_end_time_step"] == 35039
    assert simulator["episode_time_steps"] == 35040
    assert simulator["episodes"] == 8
    assert simulator["deterministic_finish"] is True
    assert simulator["community_market"]["enabled"] is True
    assert simulator["community_market"]["local_price_ratio_to_grid_import"] == 0.8
    assert simulator["community_market"]["grid_export_price"] == 0.0
    assert config["training"]["seed"] == SEED

    assert manager["algorithm"] == "CCLevel1"
    params = manager["hyperparameters"]
    assert params["num_steps"] == 336
    assert params["gamma"] == pytest.approx(0.995)
    assert params["mini_batch_size"] == 84
    assert params["price_min"] == 0.5
    assert params["price_max"] == 1.3
    assert params["bc_collect_steps"] == 8760
    assert config["pipeline"][1]["algorithm"] == "SignalAwareRBC"
    assert config["pipeline"][1]["frozen"] is True


@pytest.mark.parametrize(
    ("recipe_name", "member_weight", "w_factor", "w_smoothness"),
    (
        ("settled_focus_regularized", 0.0, 0.30, 0.10),
        ("settled_focus_adaptive", 0.0, 0.05, 0.02),
        ("hybrid_physical_adaptive", 0.25, 0.05, 0.02),
    ),
)
def test_cost_focus_v2_calibrated_recipe_ablation(
    recipe_name: str,
    member_weight: float,
    w_factor: float,
    w_smoothness: float,
):
    config = _load(recipe_name)
    reward = config["simulator"]["reward_function_kwargs"]
    params = config["pipeline"][0]["hyperparameters"]

    assert reward["cost_aggregation"] == "community_net"
    assert reward["w_cost"] == 1.0
    assert reward["w_member_retail_cost"] == member_weight
    assert reward["w_peak"] == 0.15
    assert reward["w_ramp"] == 0.10
    assert reward["w_export"] == 0.02
    for key, expected in ANNUAL_SMART_REFERENCES.items():
        assert reward[key] == pytest.approx(expected)
    assert params["w_factor"] == pytest.approx(w_factor)
    assert params["w_smoothness"] == pytest.approx(w_smoothness)


def test_cost_focus_v2_legacy_control_preserves_v1_reward_and_regularization():
    config = _load("legacy_long_control")
    reward = config["simulator"]["reward_function_kwargs"]
    params = config["pipeline"][0]["hyperparameters"]

    assert reward == {
        "cost_aggregation": "community_net",
        "w_cost": 1.0,
        "w_peak": 0.6,
        "w_ramp": 0.4,
        "w_export": 0.05,
        "w_violation": 2.0,
    }
    assert params["w_factor"] == pytest.approx(0.3)
    assert params["w_smoothness"] == pytest.approx(0.1)


def test_generated_cost_focus_v2_templates_match_committed_templates(tmp_path: Path):
    generated = generate(tmp_path)
    assert {path.name for path in generated} == {
        f"cc_smart_{recipe_name}_seed{SEED}.yaml" for recipe_name in RECIPE_NAMES
    }
    for path in generated:
        committed = CONFIG_ROOT / path.name
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == yaml.safe_load(
            committed.read_text(encoding="utf-8")
        )


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_cost_focus_v2_smokes_cover_bc_one_ppo_update_and_evaluation(
    tmp_path: Path,
    recipe_name: str,
):
    paths = {path.name: path for path in generate_smokes(tmp_path)}
    path = paths[f"cc_smart_{recipe_name}_seed{SEED}.yaml"]
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)

    simulator = config["simulator"]
    params = config["pipeline"][0]["hyperparameters"]
    assert simulator["episodes"] == 3
    assert simulator["simulation_end_time_step"] == SMOKE_STEPS - 1
    assert simulator["episode_time_steps"] == SMOKE_STEPS
    assert params["num_steps"] == 336
    assert params["bc_collect_steps"] == 336
    assert config["tracking"]["tags"]["evidence"] == "smoke"
