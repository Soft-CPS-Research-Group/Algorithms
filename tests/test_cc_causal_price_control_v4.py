from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.generate_cc_causal_price_control_v4 import (
    EXPERIMENT_NAME,
    PPO_FIXED_MULTIPLIERS,
    PPO_SEED,
    SMART_RECIPES,
    SMART_SEED,
    generate,
    generate_smokes,
)
from utils.config_schema import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments" / EXPERIMENT_NAME


def _fixed_name(multiplier: float) -> str:
    return f"cc_ppo_base_price_fixed_{multiplier:.2f}_seed{PPO_SEED}.yaml".replace(
        ".", "p", 1
    )


def _load(name: str) -> dict:
    payload = yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))
    validate_config(payload)
    return payload


@pytest.mark.parametrize("recipe", SMART_RECIPES)
def test_v4_smart_candidates_share_settlement_and_incumbent_contract(recipe: str):
    config = _load(f"cc_smart_{recipe}_seed{SMART_SEED}.yaml")
    simulator = config["simulator"]
    manager = config["pipeline"][0]
    params = manager["hyperparameters"]

    assert simulator["community_market"]["enabled"] is True
    assert simulator["simulation_start_time_step"] == 0
    assert simulator["simulation_end_time_step"] == 35039
    assert simulator["episode_time_steps"] == 35040
    assert simulator["episodes"] == 10
    assert manager["algorithm"] == "CCLevel1"
    assert params["reference_multiplier"] == pytest.approx(1.3)
    assert params["policy_residual_scale"] == pytest.approx(0.5)
    assert params["w_factor"] == pytest.approx(0.01)
    assert params["w_smoothness"] == pytest.approx(0.005)
    assert params["bc_collect_steps"] == 8760
    assert params["bc_train_steps"] == 4000
    assert config["pipeline"][1]["algorithm"] == "SignalAwareRBC"
    assert config["pipeline"][1]["frozen"] is True


def test_v4_smart_temporal_ablation_preserves_seven_day_physical_horizon():
    hourly = _load(f"cc_smart_settled_cost_hourly_seed{SMART_SEED}.yaml")
    quarter = _load(f"cc_smart_settled_cost_15min_seed{SMART_SEED}.yaml")
    hourly_params = hourly["pipeline"][0]["hyperparameters"]
    quarter_params = quarter["pipeline"][0]["hyperparameters"]

    assert hourly_params["cc_action_interval"] == 4
    assert hourly_params["num_steps"] == 168
    assert hourly_params["gamma"] == pytest.approx(0.995)
    assert quarter_params["cc_action_interval"] == 1
    assert quarter_params["num_steps"] == 672
    assert quarter_params["gamma"] == pytest.approx(0.99875)
    assert hourly_params["cc_action_interval"] * hourly_params["num_steps"] == (
        quarter_params["cc_action_interval"] * quarter_params["num_steps"]
    )


def test_v4_smart_reward_ablation_is_cost_first_and_explicit():
    cost = _load(f"cc_smart_settled_cost_15min_seed{SMART_SEED}.yaml")
    cost_peak = _load(f"cc_smart_settled_cost_peak_15min_seed{SMART_SEED}.yaml")

    assert cost["simulator"]["reward_function_kwargs"]["w_cost"] == 1.0
    assert cost["simulator"]["reward_function_kwargs"]["w_peak"] == 0.0
    assert cost["simulator"]["reward_function_kwargs"]["w_ramp"] == 0.0
    assert cost_peak["simulator"]["reward_function_kwargs"]["w_peak"] == 0.05
    assert cost_peak["simulator"]["reward_function_kwargs"]["w_ramp"] == 0.02


@pytest.mark.parametrize("multiplier", PPO_FIXED_MULTIPLIERS)
def test_v4_ppo_fixed_probes_condition_only_the_strict_local_residual_base(
    multiplier: float,
):
    config = _load(_fixed_name(multiplier))
    manager, ppo = config["pipeline"]
    exploration = ppo["exploration"]["params"]

    assert config["simulator"]["community_market"]["enabled"] is True
    assert config["simulator"]["episodes"] == 1
    assert manager["algorithm"] == "FixedPriceSignal"
    assert manager["hyperparameters"]["multiplier"] == pytest.approx(multiplier)
    assert ppo["algorithm"] == "PPO"
    assert ppo["frozen"] is True
    assert exploration["local_price_conditioning_enabled"] is False
    assert exploration["residual_base_policy"] == "SignalAwareRBCSmartLocal"
    assert exploration["residual_base_price_conditioning_enabled"] is True
    assert exploration["residual_base_policy_hyperparameters"][
        "signal_price_charge_rate"
    ] == pytest.approx(0.6)
    assert config["checkpointing"]["stage_checkpoint_local_paths"] == {
        1: "./artifacts/frozen_ppo/annual_v1/seed789"
    }


def test_v4_fixed_1p00_is_the_pre_registered_neutral_control():
    config = _load(_fixed_name(1.0))
    assert config["pipeline"][0]["hyperparameters"]["multiplier"] == 1.0
    assert config["tracking"]["tags"]["cc_price_scope"] == (
        "strict_local_residual_base_only"
    )


def test_generated_v4_templates_match_committed_templates(tmp_path: Path):
    for path in generate(tmp_path):
        committed = CONFIG_ROOT / path.name
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == yaml.safe_load(
            committed.read_text(encoding="utf-8")
        )


def test_v4_smokes_cover_real_manager_update_and_fixed_leaf_inference(tmp_path: Path):
    paths = {path.name: path for path in generate_smokes(tmp_path)}
    smart = yaml.safe_load(
        paths[f"cc_smart_settled_cost_15min_seed{SMART_SEED}.yaml"].read_text(
            encoding="utf-8"
        )
    )
    ppo = yaml.safe_load(paths[_fixed_name(1.3)].read_text(encoding="utf-8"))
    validate_config(smart)
    validate_config(ppo)

    smart_params = smart["pipeline"][0]["hyperparameters"]
    assert smart["simulator"]["episodes"] == 3
    assert smart["simulator"]["episode_time_steps"] == 673
    assert smart_params["bc_collect_steps"] == 672
    assert smart_params["bc_train_steps"] == 2
    assert ppo["simulator"]["episodes"] == 1
    assert ppo["simulator"]["episode_time_steps"] == 385
    assert ppo["tracking"]["tags"]["evidence"] == "functional_smoke"
