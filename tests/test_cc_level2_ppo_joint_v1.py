from __future__ import annotations

from pathlib import Path

import yaml

from scripts.generate_cc_level2_ppo_joint_v1 import RECIPES, generate
from utils.config_schema import validate_config


def _load(path: Path):
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_joint_cc_level2_ppo_configs_connect_and_train_both_layers(
    tmp_path: Path,
) -> None:
    configs = [_load(path) for path in generate(tmp_path)]

    assert len(configs) == len(RECIPES)
    for config in configs:
        manager, leaf = config.pipeline
        assert config.simulator.reward_function == "CCRewardLevel2"
        assert config.simulator.reward_function_kwargs["cost_aggregation"] == (
            "community_settled"
        )
        assert manager.algorithm == "CCLevel2"
        assert manager.frozen is False
        assert manager.hyperparameters.policy_parameterization == "centered_residual"
        assert manager.hyperparameters.c_dim == 119
        assert manager.hyperparameters.include_community_headroom is True
        assert manager.hyperparameters.bc_use_physical_teacher_context is True
        assert manager.hyperparameters.bc_target_import == (
            config.simulator.reward_function_kwargs["target_import"]
        )
        assert manager.hyperparameters.bc_reference_peak == (
            config.simulator.reward_function_kwargs["reference_peak"]
        )
        assert manager.hyperparameters.bc_reference_export == (
            config.simulator.reward_function_kwargs["reference_export"]
        )
        assert manager.hyperparameters.bc_pretrain_enabled is True
        assert leaf.algorithm == "PPO"
        assert leaf.frozen is False
        params = leaf.exploration.params
        assert params["local_price_conditioning_enabled"] is True
        assert params["local_price_conditioning_trainable"] is True
        assert params["residual_base_price_conditioning_enabled"] is True
        assert config.tracking.tags["cc_multiplier_policy"] == (
            "learned_per_building"
        )
        assert config.tracking.tags["cc_price_scope"] == (
            "ppo_actor_and_residual_base"
        )
        assert config.tracking.tags["ppo_actor_price_conditioning"] == "True"
        assert "fixed_multiplier" not in config.tracking.tags
        assert config.checkpointing.stage_checkpoint_local_paths[1].endswith(
            "seed789"
        )
        assert config.checkpointing.fine_tune is True


def test_joint_cc_level2_ppo_forecast_and_v2g_ablation_is_exact(
    tmp_path: Path,
) -> None:
    configs = {path.stem: _load(path) for path in generate(tmp_path)}
    current = configs["cc_l2_ppo_joint_current_storage_seed789"].pipeline[1]
    forecasts = configs["cc_l2_ppo_joint_forecasts_storage_seed789"].pipeline[1]
    v2g = configs["cc_l2_ppo_joint_forecasts_v2g_seed789"].pipeline[1]

    assert current.exploration.params["local_price_forecast_mode"] == (
        "real_unmodified"
    )
    assert forecasts.exploration.params["local_price_forecast_mode"] == (
        "persist_current"
    )
    assert forecasts.exploration.params["residual_ev_action_scale_multiplier"] == 0.0
    assert v2g.exploration.params["residual_ev_action_scale_multiplier"] == 0.15
    assert v2g.exploration.params["residual_base_policy_hyperparameters"][
        "allow_v2g"
    ] is True


def test_joint_cc_level2_ppo_smoke_reaches_price_training(tmp_path: Path) -> None:
    for path in generate(tmp_path, smoke=True):
        config = _load(path)
        assert config.simulator.episodes == 3
        assert config.pipeline[0].hyperparameters.bc_collect_steps == 96
        assert config.pipeline[0].hyperparameters.bc_train_steps == 2
        assert config.checkpointing.checkpoint_interval is None
