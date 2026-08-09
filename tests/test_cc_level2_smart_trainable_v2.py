from __future__ import annotations

from pathlib import Path

import yaml

from scripts.generate_cc_level2_smart_trainable_v2 import RECIPES, generate
from utils.config_schema import validate_config


def _load(path: Path):
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_cc_level2_smart_v2_configs_are_trainable_and_matched(tmp_path: Path) -> None:
    paths = generate(tmp_path)

    assert len(paths) == len(RECIPES)
    configs = [_load(path) for path in paths]
    for config in configs:
        manager, leaf = config.pipeline
        assert config.simulator.reward_function == "CCRewardLevel2"
        assert config.simulator.reward_function_kwargs["cost_aggregation"] == (
            "community_settled"
        )
        assert config.simulator.community_market.enabled is True
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
        assert manager.hyperparameters.reference_multipliers == [1.3] * 17
        assert manager.hyperparameters.policy_residual_scale == 0.5
        assert manager.hyperparameters.bc_pretrain_enabled is True
        assert leaf.algorithm == "SignalAwareRBC"
        assert leaf.frozen is True

    scientific = {
        (
            config.simulator.dataset_name,
            config.simulator.simulation_start_time_step,
            config.simulator.simulation_end_time_step,
            config.simulator.episode_time_steps,
            config.simulator.episodes,
        )
        for config in configs
    }
    assert len(scientific) == 1


def test_cc_level2_smart_v2_smokes_exercise_bc_and_ppo(tmp_path: Path) -> None:
    paths = generate(tmp_path, smoke=True)

    for path in paths:
        config = _load(path)
        manager = config.pipeline[0].hyperparameters
        assert config.simulator.episodes == 3
        assert manager.bc_collect_steps == manager.num_steps == 168
        assert manager.bc_train_steps == 2
        assert config.tracking.tags["promotion_eligible"] == "False"
