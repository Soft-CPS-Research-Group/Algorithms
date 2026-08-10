from __future__ import annotations

import yaml

from scripts.generate_matd3_v3_campaign import (
    SMART_CONFIG,
    VARIANTS,
    build_config,
)
from utils.config_schema import validate_config


def _params(config: dict) -> dict:
    return config["pipeline"][0]["exploration"]["params"]


def test_matd3_v3_candidates_validate_and_share_the_annual_surface():
    for variant_name in VARIANTS:
        config = build_config(variant_name=variant_name)
        validate_config(config)
        assert config["simulator"]["episodes"] == 3
        assert config["simulator"]["simulation_start_time_step"] == 0
        assert config["simulator"]["simulation_end_time_step"] == 35039
        assert config["simulator"]["community_market"]["enabled"] is True
        assert config["simulator"]["entity_encoding"]["profile"] == "maddpg_v4_operational"
        assert config["simulator"]["export"]["final_episode_only"] is True
        assert config["pipeline"][0]["algorithm"] == "MATD3"


def test_matd3_v3_uses_the_exact_current_smart_teacher_recipe():
    smart = yaml.safe_load(SMART_CONFIG.read_text(encoding="utf-8"))
    expected = next(
        stage["hyperparameters"]
        for stage in smart["pipeline"]
        if stage["algorithm"] == "SignalAwareRBC"
    )

    config = build_config(variant_name="smart_anchor")

    assert _params(config)["warm_start_policy"] == "RBCSmartPolicy"
    assert _params(config)["warm_start_policy_hyperparameters"] == expected


def test_matd3_v3_variants_progressively_open_cooperation_and_storage():
    anchor = _params(build_config(variant_name="smart_anchor"))
    cooperative = _params(build_config(variant_name="cooperative_team70"))
    storage_open = _params(build_config(variant_name="cooperative_storage_open"))
    scorecard = _params(build_config(variant_name="cooperative_scorecard"))
    cost_first = _params(build_config(variant_name="cooperative_cost_first"))

    assert anchor["critic_team_reward_mix"] == 0.0
    assert cooperative["critic_team_reward_mix"] == 0.70
    assert storage_open["critic_team_reward_mix"] == 0.70
    assert scorecard["critic_team_reward_mix"] == 0.70
    assert cost_first["critic_team_reward_mix"] == 0.85
    assert storage_open["residual_action_final_scale"] > cooperative["residual_action_final_scale"]
    assert cost_first["residual_action_final_scale"] > storage_open["residual_action_final_scale"]
    assert cost_first["actor_storage_behavior_cloning_multiplier"] < storage_open[
        "actor_storage_behavior_cloning_multiplier"
    ]
    assert scorecard["actor_storage_smoothness_l2_penalty"] > storage_open[
        "actor_storage_smoothness_l2_penalty"
    ]


def test_matd3_v3_smoke_preserves_recipe_but_shortens_evidence_horizon():
    smoke = build_config(
        variant_name="cooperative_storage_open",
        smoke_steps=4096,
    )
    validate_config(smoke)

    assert smoke["simulator"]["episodes"] == 1
    assert smoke["simulator"]["simulation_end_time_step"] == 4095
    assert smoke["checkpointing"]["checkpoint_interval"] is None
    assert smoke["tracking"]["tags"]["evidence"] == "functional_smoke"
