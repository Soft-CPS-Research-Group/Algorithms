from __future__ import annotations

from scripts.generate_matd3_storage_safe_v4 import (
    VARIANTS,
    build_config,
    build_smart_reference_config,
)
from utils.config_schema import validate_config


def test_matd3_v4_freezes_service_authority_and_uses_controllable_reward() -> None:
    for name, variant in VARIANTS.items():
        config = build_config(name)
        validate_config(config)
        params = config["pipeline"][0]["exploration"]["params"]

        assert config["simulator"]["reward_function"] == (
            "CostCommunityStorageResidualRewardV55"
        )
        assert config["simulator"]["episodes"] == variant["training_years"] + 1
        assert config["simulator"]["deterministic_finish"] is True
        assert params["warm_start_policy"] == "RBCSmartPolicy"
        assert params["residual_policy_enabled"] is True
        assert params["residual_storage_action_scale_multiplier"] > 0.0
        assert params["residual_ev_action_scale_multiplier"] == 0.0
        assert params["residual_deferrable_action_scale_multiplier"] == 0.0
        assert params["residual_building_gain_multipliers"] == variant.get(
            "residual_building_gain_multipliers",
            {"Building_15": 0.0},
        )
        assert params["local_action_safety_enabled"] is bool(
            variant.get("local_action_safety_enabled", False)
        )
        assert params["local_action_safety_runtime_only_export"] is True


def test_matd3_v4_compares_training_horizons_and_cooperative_credit() -> None:
    h1 = build_config("storage_cost_h1")
    h2 = build_config("storage_cost_h2")
    h4 = build_config("storage_cost_h4")
    team = build_config("storage_team25_h2")

    assert h1["simulator"]["episodes"] == 2
    assert h2["simulator"]["episodes"] == 3
    assert h4["simulator"]["episodes"] == 5
    assert team["pipeline"][0]["exploration"]["params"][
        "critic_team_reward_mix"
    ] == 0.25

    anchored = build_config("storage_cost_h1")
    medium = build_config("storage_cost_medium_projected_h1")
    wide = build_config("storage_cost_wide_projected_h1")
    smooth = build_config("storage_cost_smooth_projected_h1")
    net_smooth = build_config("storage_net_smooth_projected_h1")
    net_context = build_config("storage_net_context_smooth_projected_h1")
    old_recipe_projected = build_config(
        "storage_context_old_recipe_projected_h1"
    )
    old_recipe_unprojected = build_config(
        "storage_context_old_recipe_unprojected_h1"
    )
    exact_replay = build_config(
        "storage_context_exact_replay_h1",
        smoke_steps=4096,
    )
    exact_extension_h2 = build_config(
        "storage_context_exact_extension_h2",
        smoke_steps=4096,
    )
    net_context_ramp = build_config("storage_net_context_ramp_projected_h1")
    anchored_params = anchored["pipeline"][0]["exploration"]["params"]
    medium_params = medium["pipeline"][0]["exploration"]["params"]
    wide_params = wide["pipeline"][0]["exploration"]["params"]
    assert anchored_params["residual_action_final_scale"] == 0.06
    assert medium_params["residual_action_final_scale"] == 0.10
    assert wide_params["residual_action_final_scale"] == 0.16
    assert medium_params["local_action_safety_enabled"] is True
    assert wide_params["local_action_safety_enabled"] is True
    smooth_params = smooth["pipeline"][0]["exploration"]["params"]
    assert smooth_params["local_action_safety_enabled"] is True
    assert smooth_params["actor_storage_smoothness_l2_penalty"] == 0.01
    assert smooth_params["actor_storage_smoothness_deadband"] == 0.03
    assert smooth["simulator"]["reward_function_kwargs"][
        "community_penalty_use_net_exchange"
    ] is False
    assert net_smooth["simulator"]["reward_function_kwargs"][
        "community_penalty_use_net_exchange"
    ] is True
    assert net_context["pipeline"][0]["exploration"]["params"][
        "actor_community_context_enabled"
    ] is True
    old_recipe_params = old_recipe_projected["pipeline"][0]["exploration"][
        "params"
    ]
    assert old_recipe_params["residual_action_final_scale"] == 0.055
    assert old_recipe_params["residual_building_gain_multipliers"] == {
        "Building_15": 1.0
    }
    assert old_recipe_params["local_action_safety_enabled"] is True
    unprojected_params = old_recipe_unprojected["pipeline"][0]["exploration"][
        "params"
    ]
    assert unprojected_params["residual_action_final_scale"] == 0.055
    assert unprojected_params["local_action_safety_enabled"] is False
    assert unprojected_params["residual_building_gain_multipliers"] == {}
    assert unprojected_params["actor_community_context_enabled"] is True
    exact_params = exact_replay["pipeline"][0]["exploration"]["params"]
    assert exact_params["residual_action_final_scale"] == 0.24
    assert exact_params["residual_action_growth_steps"] == 35040
    assert exact_params["local_action_safety_enabled"] is False
    assert exact_params["residual_building_gain_multipliers"] == {}
    exact_extension_params = exact_extension_h2["pipeline"][0][
        "exploration"
    ]["params"]
    assert exact_extension_h2["simulator"]["episodes"] == 3
    assert exact_extension_params["residual_action_final_scale"] == 0.24
    assert exact_extension_params["residual_action_growth_steps"] == 35040
    assert exact_extension_params["local_action_safety_enabled"] is False
    assert exact_extension_params["residual_building_gain_multipliers"] == {}
    assert net_context_ramp["simulator"]["reward_function_kwargs"][
        "community_ramping_penalty"
    ] == 0.004
    assert medium_params["local_action_safety_headroom_reserve_kw"] == 0.10


def test_matd3_v4_storage_learning_ablations_align_replay_and_credit() -> None:
    team = build_config("storage_net_context_team25_projected_h1")
    replay = build_config("storage_net_context_replay_projected_h1")
    accelerated = build_config("storage_net_context_accelerated_projected_h1")
    accelerated_wide = build_config(
        "storage_net_context_accelerated_wide_projected_h1"
    )
    accelerated_cost = build_config(
        "storage_net_context_accelerated_cost_first_projected_h1"
    )
    temporal = build_config("storage_net_context_temporal_projected_h1")
    guarded = build_config("storage_net_context_b15_guarded_projected_h1")

    assert team["pipeline"][0]["exploration"]["params"][
        "critic_team_reward_mix"
    ] == 0.25
    replay_buffer = replay["pipeline"][0]["replay_buffer"]
    assert replay_buffer["behavior_action_priority_weight"] == 0.0
    assert replay_buffer["observation_event_priority_mode"] == "ev_pv_price_peak"

    accelerated_params = accelerated["pipeline"][0]["exploration"]["params"]
    assert accelerated_params["random_exploration_steps"] == 1024
    assert accelerated_params["train_during_initial_exploration"] is True
    assert accelerated_params["actor_policy_loss_weight"] == 0.20
    assert accelerated["pipeline"][0]["networks"]["actor"]["lr"] == 1.0e-4
    assert accelerated_wide["pipeline"][0]["exploration"]["params"][
        "residual_action_final_scale"
    ] == 0.16
    assert accelerated_cost["simulator"]["reward_function_kwargs"][
        "community_settlement_cost_weight"
    ] == 1.50
    assert accelerated_cost["simulator"]["reward_function_kwargs"][
        "battery_throughput_penalty"
    ] == 0.0001

    temporal_params = temporal["pipeline"][0]["exploration"]["params"]
    assert temporal_params["actor_frame_stack_steps"] == 4
    assert temporal["simulator"]["reward_function_kwargs"][
        "community_ramping_penalty"
    ] == 0.002

    guarded_params = guarded["pipeline"][0]["exploration"]["params"]
    assert guarded_params["residual_building_gain_multipliers"] == {
        "Building_15": 0.25
    }
    assert guarded_params["local_action_safety_headroom_reserve_kw"] == 0.75


def test_matd3_v4_old_recipe_selection_ablations_are_explicit() -> None:
    winners = build_config("storage_context_old_recipe_winners_projected_h1")
    strong = build_config(
        "storage_context_old_recipe_strong_winners_projected_h1"
    )
    h2 = build_config("storage_context_old_recipe_projected_h2")
    h2_smoke = build_config(
        "storage_context_old_recipe_projected_h2",
        smoke_steps=4096,
    )

    winners_gains = winners["pipeline"][0]["exploration"]["params"][
        "residual_building_gain_multipliers"
    ]
    strong_gains = strong["pipeline"][0]["exploration"]["params"][
        "residual_building_gain_multipliers"
    ]
    assert sum(value > 0.0 for value in winners_gains.values()) == 11
    assert sum(value > 0.0 for value in strong_gains.values()) == 7
    assert winners_gains["Building_12"] == 0.0
    assert strong_gains["Building_15"] == 1.0
    assert h2["simulator"]["episodes"] == 3
    assert h2_smoke["simulator"]["episodes"] == 3
    assert h2_smoke["tracking"]["tags"]["training_years"] == "2"


def test_matd3_v4_smoke_contains_training_and_deterministic_replay() -> None:
    smoke = build_config("storage_cost_h1", smoke_steps=4096)
    validate_config(smoke)

    assert smoke["simulator"]["episodes"] == 2
    assert smoke["simulator"]["simulation_end_time_step"] == 4095
    assert smoke["checkpointing"]["checkpoint_interval"] is None
    assert smoke["pipeline"][0]["exploration"]["params"][
        "residual_action_growth_steps"
    ] == 4096


def test_matd3_v4_has_exact_paired_smart_reference() -> None:
    reference = build_smart_reference_config(seed=789, smoke_steps=4096)
    validate_config(reference)

    assert reference["pipeline"][1]["algorithm"] == "SignalAwareRBC"
    assert reference["simulator"]["episodes"] == 1
    assert reference["simulator"]["simulation_end_time_step"] == 4095
    assert reference["simulator"]["community_market"]["enabled"] is True
    assert reference["tracking"]["tags"]["recipe"] == "smart_paired_reference"


def test_matd3_v4_smart_reference_can_match_h1_and_h2_episode_realizations() -> None:
    h1_reference = build_smart_reference_config(
        seed=789,
        smoke_steps=4096,
        episodes=2,
    )
    h2_reference = build_smart_reference_config(
        seed=789,
        smoke_steps=4096,
        episodes=3,
    )
    validate_config(h1_reference)
    validate_config(h2_reference)

    assert h1_reference["simulator"]["episodes"] == 2
    assert h2_reference["simulator"]["episodes"] == 3
    assert h1_reference["simulator"]["export"]["final_episode_only"] is True
    assert h1_reference["tracking"]["tags"]["evaluation_episode_index"] == "2"
    assert h2_reference["tracking"]["tags"]["evaluation_episode_index"] == "3"
    assert h1_reference["tracking"]["tags"]["episode_realization_matched"] == "True"
