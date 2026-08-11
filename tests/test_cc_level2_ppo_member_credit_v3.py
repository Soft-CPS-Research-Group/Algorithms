from __future__ import annotations

from scripts.generate_cc_level2_ppo_member_credit_v3 import (
    VARIANTS,
    build_config,
    build_paired_neutral_config,
)
from utils.config_schema import validate_config


def test_cc_level2_v3_uses_member_credit_without_unfreezing_ppo() -> None:
    for name in VARIANTS:
        config = build_config(name)
        validate_config(config)
        manager, leaf = config["pipeline"]
        reward = config["simulator"]["reward_function_kwargs"]

        assert manager["algorithm"] == "CCLevel2"
        assert manager["frozen"] is False
        assert manager["hyperparameters"]["credit_assignment"] == (
            "member_decomposed"
        )
        assert 0.0 < manager["hyperparameters"]["team_reward_mix"] < 1.0
        assert manager["hyperparameters"]["price_min"] <= 0.60
        assert manager["hyperparameters"]["price_max"] > 1.0
        assert reward["credit_assignment"] == "member_decomposed"
        assert reward["cost_aggregation"] == "community_settled"
        assert leaf["algorithm"] == "PPO"
        assert leaf["frozen"] is True
        assert leaf["exploration"]["params"]["actor_policy_loss_weight"] == 0.0

    schedule = build_config("member_schedule_teacher_hourly")
    assert schedule["pipeline"][0]["hyperparameters"]["bc_teacher_mode"] == (
        "cheap_and_export"
    )


def test_cc_level2_v3_smoke_reaches_member_ppo_update() -> None:
    config = build_config("member_cost_hourly", smoke=True)
    validate_config(config)

    manager = config["pipeline"][0]["hyperparameters"]
    assert config["simulator"]["episodes"] == 3
    assert manager["bc_collect_steps"] == 96
    assert manager["bc_train_steps"] == 4
    assert manager["num_steps"] == 64
    assert manager["credit_assignment"] == "member_decomposed"


def test_cc_level2_v3_pilot_has_matched_neutral_and_real_learning() -> None:
    candidate = build_config("member_cost_hourly", pilot_steps=4096)
    neutral = build_paired_neutral_config(pilot_steps=4096)
    validate_config(candidate)
    validate_config(neutral)

    manager = candidate["pipeline"][0]["hyperparameters"]
    assert candidate["simulator"]["episodes"] == 5
    assert candidate["simulator"]["simulation_end_time_step"] == 4095
    assert manager["bc_collect_steps"] == 1024
    assert manager["num_steps"] == 256
    assert neutral["simulator"]["episodes"] == 1
    assert neutral["simulator"]["simulation_end_time_step"] == 4095
    assert neutral["pipeline"][0]["hyperparameters"]["multiplier"] == 1.0
