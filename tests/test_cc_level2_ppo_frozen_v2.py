from __future__ import annotations

from pathlib import Path

import yaml

from scripts.generate_cc_level2_ppo_frozen_v2 import CC_RECIPES, generate
from utils.config_schema import validate_config


def _load(path: Path):
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_v2_separates_parity_gates_from_cc_only_training(tmp_path: Path) -> None:
    configs = {path.stem: _load(path) for path in generate(tmp_path)}

    assert len(configs) == 2 + len(CC_RECIPES)
    original = configs["cc_l2_ppo_gate_a_original_ppo_neutral_seed789"]
    signal = configs["cc_l2_ppo_gate_b_signal_path_neutral_seed789"]

    assert original.simulator.episodes == 1
    assert signal.simulator.episodes == 1
    assert original.pipeline[0].algorithm == "FixedPriceSignal"
    assert signal.pipeline[0].algorithm == "FixedPriceSignal"
    assert original.pipeline[1].frozen is True
    assert signal.pipeline[1].frozen is True
    assert original.pipeline[1].exploration.params["residual_base_policy"] == (
        "RBCSmartLocalPolicy"
    )
    assert signal.pipeline[1].exploration.params["residual_base_policy"] == (
        "SignalAwareRBCSmartLocal"
    )


def test_v2_trains_only_a_conservative_hourly_cc_level2(tmp_path: Path) -> None:
    configs = {path.stem: _load(path) for path in generate(tmp_path)}

    for name in CC_RECIPES:
        config = configs[f"cc_l2_ppo_{name}"]
        manager, leaf = config.pipeline

        assert config.simulator.episodes == 10
        assert config.simulator.deterministic_finish is True
        assert config.simulator.reward_function == "CCRewardLevel2"
        reward = config.simulator.reward_function_kwargs
        assert reward["cost_aggregation"] == "community_settled"
        assert reward["w_ramp"] > 0.0
        assert reward["w_violation"] == 2.0

        assert manager.algorithm == "CCLevel2"
        assert manager.frozen is False
        assert manager.hyperparameters.price_min == 0.90
        assert manager.hyperparameters.price_max == 1.00
        assert manager.hyperparameters.reference_multipliers == [1.0] * 17
        assert manager.hyperparameters.cc_action_interval == 4
        assert manager.hyperparameters.policy_parameterization == "centered_residual"

        assert leaf.algorithm == "PPO"
        assert leaf.frozen is True
        params = leaf.exploration.params
        assert params["local_price_conditioning_enabled"] is False
        assert params["actor_policy_loss_weight"] == 0.0
        assert params["residual_ev_action_scale_multiplier"] == 0.0
        assert params["residual_base_policy_hyperparameters"]["allow_v2g"] is False
        assert config.checkpointing.fine_tune is False
        assert config.tracking.tags["joint_training"] == "False"


def test_v2_smoke_reaches_real_cc_learning_without_unfreezing_leaf(
    tmp_path: Path,
) -> None:
    configs = {path.stem: _load(path) for path in generate(tmp_path, smoke=True)}

    for name in CC_RECIPES:
        config = configs[f"cc_l2_ppo_{name}"]
        manager, leaf = config.pipeline
        assert config.simulator.episodes == 3
        assert manager.hyperparameters.bc_collect_steps == 96
        assert manager.hyperparameters.bc_train_steps == 4
        assert manager.hyperparameters.num_steps == 96
        assert leaf.frozen is True
        assert config.checkpointing.checkpoint_interval is None
