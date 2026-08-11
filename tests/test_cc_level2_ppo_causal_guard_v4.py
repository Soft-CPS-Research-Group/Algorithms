from __future__ import annotations

from scripts.generate_cc_level2_ppo_causal_guard_v4 import (
    CAUSAL_VECTOR_INCUMBENT,
    VARIANTS,
    build_causal_incumbent_config,
    build_config,
    build_paired_neutral_config,
)
from utils.config_schema import validate_config


def test_v4_is_causally_gated_and_keeps_frozen_ppo() -> None:
    for name, variant in VARIANTS.items():
        config = build_config(name)
        validate_config(config)
        manager, leaf = config["pipeline"]
        params = manager["hyperparameters"]

        assert manager["algorithm"] == "CCLevel2"
        assert manager["frozen"] is False
        assert params["policy_parameterization"] == "causal_active_only"
        assert params["causal_initial_multiplier"] == 0.90
        assert params["price_min"] == variant["price_min"]
        assert params["price_max"] == 1.0
        assert params["bc_pretrain_enabled"] is False
        assert params["credit_assignment"] == "member_decomposed"
        assert params["causal_use_physical_context"] is True
        assert params["include_community_history"] is True
        assert params["c_dim"] == 122
        assert params["separate_value_encoder"] is True
        assert params["reward_normalization"] == variant.get(
            "reward_normalization", "running_zscore"
        )
        assert config["simulator"]["reward_function_kwargs"][
            "ramp_credit_allocation"
        ] == "causal_net"
        assert config["tracking"]["tags"][
            "ramp_credit_allocation"
        ] == "causal_net"
        assert config["tracking"]["tags"]["training_episodes"] == str(
            int(variant["episodes"]) - 1
        )
        assert config["tracking"]["tags"]["total_episodes"] == str(
            variant["episodes"]
        )
        assert config["tracking"]["tags"]["evaluation_episode_index"] == str(
            variant["episodes"]
        )
        assert config["tracking"]["tags"]["episode_realization_matched"] == "True"
        assert config["tracking"]["tags"]["team_reward_mix"] == str(
            variant["team_reward_mix"]
        )
        assert config["tracking"]["tags"]["ppo_actor_price_conditioning"] == (
            "current_only"
        )
        assert config["tracking"]["tags"]["ppo_price_forecast_conditioning"] == (
            "real_unmodified"
        )
        assert config["tracking"]["tags"]["v2g_enabled"] == "True"
        assert leaf["algorithm"] == "PPO"
        assert leaf["frozen"] is True
        assert leaf["exploration"]["params"]["actor_policy_loss_weight"] == 0.0
        residual = leaf["exploration"]["params"][
            "residual_base_policy_hyperparameters"
        ]
        assert residual["allow_v2g"] is True
        assert residual["signal_price_response_mode"] == "linear_discount"
        assert leaf["exploration"]["params"][
            "local_price_conditioning_enabled"
        ] is True
        assert leaf["exploration"]["params"][
            "local_price_forecast_mode"
        ] == "real_unmodified"


def test_v4_pilot_has_learning_and_exact_neutral_comparator() -> None:
    candidate = build_config("causal_member_cost_hourly", pilot_steps=4096)
    neutral = build_paired_neutral_config(pilot_steps=4096)
    incumbent = build_causal_incumbent_config(pilot_steps=4096)
    validate_config(candidate)
    validate_config(neutral)
    validate_config(incumbent)

    assert candidate["simulator"]["episodes"] == 5
    assert candidate["simulator"]["simulation_end_time_step"] == 4095
    assert candidate["pipeline"][0]["hyperparameters"]["num_steps"] == 256
    assert neutral["simulator"]["episodes"] == 1
    assert neutral["pipeline"][0]["hyperparameters"]["multiplier"] == 1.0
    assert neutral["pipeline"][1]["exploration"]["params"][
        "residual_base_policy_hyperparameters"
    ]["allow_v2g"] is True
    neutral_leaf = neutral["pipeline"][1]["exploration"]["params"]
    candidate_leaf = candidate["pipeline"][1]["exploration"]["params"]
    assert neutral_leaf["local_price_conditioning_enabled"] is True
    assert neutral_leaf["local_price_forecast_mode"] == "real_unmodified"
    assert neutral_leaf["residual_base_price_conditioning_enabled"] is True
    assert neutral_leaf["residual_base_policy_hyperparameters"] == (
        candidate_leaf["residual_base_policy_hyperparameters"]
    )
    assert neutral["pipeline"][1] == candidate["pipeline"][1]
    assert incumbent["pipeline"][0]["algorithm"] == "CausalPriceSignal"
    assert incumbent["pipeline"][0]["hyperparameters"][
        "discount_multiplier"
    ] == 0.90
    assert incumbent["pipeline"][1]["exploration"]["params"][
        "residual_base_policy_hyperparameters"
    ]["allow_v2g"] is True
    assert incumbent["simulator"]["simulation_end_time_step"] == 4095


def test_v4_can_match_neutral_control_to_candidate_episode_realization() -> None:
    neutral = build_paired_neutral_config(pilot_steps=4096, episodes=5)
    validate_config(neutral)

    assert neutral["simulator"]["episodes"] == 5
    assert neutral["simulator"]["deterministic_finish"] is True
    assert neutral["simulator"]["export"]["final_episode_only"] is True
    assert neutral["tracking"]["tags"]["evaluation_episode_index"] == "5"
    assert neutral["tracking"]["tags"]["episode_realization_matched"] == "True"
    assert neutral["simulator"]["export"]["session_name"].endswith(
        "-ep5-pilot4096"
    )


def test_v4_emits_annual_neutral_control_on_candidate_episode_12() -> None:
    neutral = build_paired_neutral_config(episodes=12)
    validate_config(neutral)

    assert neutral["simulator"]["episodes"] == 12
    assert neutral["simulator"]["simulation_end_time_step"] == 35039
    assert neutral["simulator"]["episode_time_steps"] == 35040
    assert neutral["simulator"]["export"]["final_episode_only"] is True
    assert neutral["tracking"]["tags"]["evaluation_episode_index"] == "12"
    assert neutral["tracking"]["tags"]["episode_realization_matched"] == "True"
    assert neutral["tracking"]["tags"]["ppo_actor_price_conditioning"] == (
        "current_only"
    )
    assert neutral["tracking"]["tags"]["ppo_price_forecast_conditioning"] == (
        "real_unmodified"
    )
    assert neutral["tracking"]["tags"]["v2g_enabled"] == "True"
    assert neutral["tracking"]["tags"]["evidence"] == (
        "annual_episode_matched_reference"
    )
    assert "pilot_steps" not in neutral["tracking"]["tags"]
    assert neutral["simulator"]["export"]["session_name"].endswith(
        "-ep12-annual"
    )


def test_v4_vector_incumbent_starts_at_measured_per_building_prices() -> None:
    config = build_config(
        "causal_member_vector_incumbent",
        pilot_steps=4096,
    )
    validate_config(config)
    manager = config["pipeline"][0]["hyperparameters"]

    assert manager["causal_initial_multipliers"] == CAUSAL_VECTOR_INCUMBENT
    assert manager["price_min"] <= min(CAUSAL_VECTOR_INCUMBENT)
    assert manager["reward_normalization"] == "none"
    assert manager["team_reward_mix"] == 0.25

    guarded = build_config(
        "causal_member_vector_guarded",
        pilot_steps=4096,
    )
    validate_config(guarded)
    guarded_manager = guarded["pipeline"][0]["hyperparameters"]
    assert guarded_manager["causal_initial_multipliers"] == CAUSAL_VECTOR_INCUMBENT
    assert guarded_manager["causal_residual_scale"] == 0.20


def test_v4_cost_exploration_crosses_measured_leaf_deadband() -> None:
    hourly = build_config(
        "causal_member_vector_cost_explore_hourly",
        pilot_steps=4096,
    )
    half_hourly = build_config(
        "causal_member_vector_cost_explore_30min",
        pilot_steps=4096,
    )
    validate_config(hourly)
    validate_config(half_hourly)

    hourly_manager = hourly["pipeline"][0]["hyperparameters"]
    half_hourly_manager = half_hourly["pipeline"][0]["hyperparameters"]
    assert hourly_manager["causal_initial_multipliers"] == CAUSAL_VECTOR_INCUMBENT
    assert hourly_manager["causal_residual_scale"] == 1.0
    assert hourly_manager["initial_log_std"] == -1.25
    assert hourly_manager["cc_action_interval"] == 4
    assert half_hourly_manager["cc_action_interval"] == 2
    assert hourly_manager["team_reward_mix"] == 0.10
    assert hourly["simulator"]["reward_function_kwargs"]["w_ramp"] == 0.005

    scorecard = build_config(
        "causal_member_vector_scorecard_explore_hourly",
        pilot_steps=4096,
    )
    validate_config(scorecard)
    scorecard_manager = scorecard["pipeline"][0]["hyperparameters"]
    assert scorecard_manager["initial_log_std"] == -1.25
    assert scorecard_manager["causal_residual_scale"] == 1.0
    assert scorecard_manager["cc_action_interval"] == 4
    assert scorecard_manager["team_reward_mix"] == 0.25
    assert scorecard["simulator"]["reward_function_kwargs"]["w_peak"] == 0.06
    assert scorecard["simulator"]["reward_function_kwargs"]["w_ramp"] == 0.04
