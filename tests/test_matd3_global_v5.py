from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts.generate_matd3_global_v5 import (
    REPO_ROOT,
    SEASONAL_START_STEPS,
    VARIANTS,
    build_config,
    build_smart_reference_config,
    generate,
)
from utils.config_schema import validate_config


TEACHER_SCHEDULE = (
    REPO_ROOT
    / "configs"
    / "demonstrations"
    / "community_fixed_service_battery_oracle_annual_v1.json.gz"
)
SCORECARD_TEACHER_SCHEDULE = (
    REPO_ROOT
    / "configs"
    / "demonstrations"
    / "community_fixed_service_battery_global_scorecard_teacher_annual_v5.json.gz"
)

TEACHER_SHA256 = "f40c201d545ea03226ddb97688f3cf694fdab5471f93dd1295cf4fdb4843b425"
SCORECARD_TEACHER_SHA256 = (
    "4524206b39faf54d9484a84e4386f03d21c408f9b1b35987183624cc4ec88912"
)


def test_matd3_v5_corrects_exposure_and_preserves_episode_models() -> None:
    for name in VARIANTS:
        config = build_config(name)
        validate_config(config)
        params = config["pipeline"][0]["exploration"]["params"]

        assert config["simulator"]["episodes"] == 3
        assert config["simulator"]["deterministic_finish"] is True
        assert config["checkpointing"]["checkpoint_interval"] is None
        assert config["checkpointing"]["checkpoint_on_episode_end"] is True
        assert config["checkpointing"]["keep_episode_checkpoints"] is True
        assert params["warm_start_policy"] == "RBCSmartPolicy"
        assert params["warm_start_policy_phaseout_steps"] == 8192
        assert params["residual_action_growth_steps"] == 12288
        assert params["actor_frame_stack_steps"] == 4
        assert params["residual_ev_action_scale_multiplier"] == 0.0
        assert params["residual_deferrable_action_scale_multiplier"] == 0.0
        assert params["actor_behavior_cloning_weight"] == 0.0
        assert config["simulator"]["reward_function_kwargs"][
            "community_penalty_use_net_exchange"
        ] is True


def test_matd3_v5_variants_form_explicit_cost_ramp_frontier() -> None:
    cost = build_config("cost_first_h2")
    balanced = build_config("balanced_h2")
    ramp = build_config("ramp_guard_h2")

    cost_reward = cost["simulator"]["reward_function_kwargs"]
    balanced_reward = balanced["simulator"]["reward_function_kwargs"]
    ramp_reward = ramp["simulator"]["reward_function_kwargs"]
    assert cost_reward["community_settlement_cost_weight"] > balanced_reward[
        "community_settlement_cost_weight"
    ] > ramp_reward["community_settlement_cost_weight"]
    assert cost_reward["community_ramping_penalty"] < balanced_reward[
        "community_ramping_penalty"
    ] < ramp_reward["community_ramping_penalty"]
    effective_authority = []
    for config in (cost, balanced, ramp):
        params = config["pipeline"][0]["exploration"]["params"]
        effective_authority.append(
            params["residual_action_final_scale"]
            * params["residual_storage_action_scale_multiplier"]
        )
    assert effective_authority == [0.57, 0.5225, 0.45]


def test_matd3_v5_global_scorecard_is_fully_cooperative() -> None:
    config = build_config("global_scorecard_h2")
    validate_config(config)

    params = config["pipeline"][0]["exploration"]["params"]
    reward = config["simulator"]["reward_function_kwargs"]
    assert params["critic_team_reward_mix"] == 1.0
    assert reward["community_penalty_use_net_exchange"] is True
    assert reward["community_emissions_penalty"] > 0.0
    assert reward["community_emissions_use_net_exchange"] is False
    assert params["residual_action_final_scale"] * params[
        "residual_storage_action_scale_multiplier"
    ] == 0.57


def test_matd3_v5_global_distillation_keeps_supervision_anchor() -> None:
    config = build_config(
        "global_distilled_h2",
        teacher_schedule=SCORECARD_TEACHER_SCHEDULE,
        teacher_label="scorecard",
    )
    validate_config(config)

    params = config["pipeline"][0]["exploration"]["params"]
    assert params["critic_team_reward_mix"] == 1.0
    assert params["actor_behavior_cloning_weight"] == pytest.approx(0.45)
    assert params["actor_behavior_cloning_min_weight"] == pytest.approx(0.06)
    assert params["actor_behavior_cloning_decay_steps"] > 2 * 35000 - 1024
    assert params["actor_behavior_cloning_extra_update_end_step"] == 35039
    assert params["actor_policy_loss_weight"] < 0.3


def test_matd3_v5_can_use_milp_for_bc_without_replacing_smart_base() -> None:
    assert TEACHER_SCHEDULE.is_file()
    config = build_config("balanced_h2", teacher_schedule=TEACHER_SCHEDULE)
    validate_config(config)
    params = config["pipeline"][0]["exploration"]["params"]

    assert params["warm_start_policy"] == "RBCSmartPolicy"
    assert params["actor_behavior_cloning_source"] == "teacher_policy"
    assert params["actor_behavior_cloning_teacher_policy"] == (
        "FixedServiceOracleReplayPolicy"
    )
    assert params["actor_behavior_cloning_teacher_action_scope"] == (
        "residual_authority"
    )
    assert params[
        "actor_behavior_cloning_clip_target_to_residual_authority"
    ] is True
    assert params["actor_behavior_cloning_teacher_hyperparameters"][
        "schedule_path"
    ] == str(TEACHER_SCHEDULE)
    assert params["actor_behavior_cloning_teacher_hyperparameters"][
        "service_policy"
    ] == "RBCSmartPolicy"
    assert config["tracking"]["tags"]["evaluation_teacher_access"] == "False"


def test_matd3_v5_seasonal_windows_are_paired_and_in_bounds() -> None:
    for start_step in SEASONAL_START_STEPS:
        candidate = build_config(
            "balanced_h2",
            start_step=start_step,
            steps=4096,
        )
        reference = build_smart_reference_config(
            start_step=start_step,
            steps=4096,
        )
        validate_config(candidate)
        validate_config(reference)
        for config in (candidate, reference):
            assert config["simulator"]["simulation_start_time_step"] == start_step
            assert config["simulator"]["simulation_end_time_step"] == (
                start_step + 4095
            )
            assert config["simulator"]["episodes"] == 3
        assert candidate["checkpointing"]["checkpoint_on_episode_end"] is True
        assert candidate["checkpointing"]["keep_episode_checkpoints"] is False
        assert candidate["tracking"]["tags"]["episode_checkpoint_selection"] == (
            "False"
        )
        params = candidate["pipeline"][0]["exploration"]["params"]
        assert params["random_exploration_steps"] == 512
        assert params["warm_start_policy_phaseout_steps"] == 2048
        assert params["residual_action_growth_steps"] == 2048
        assert params["actor_policy_loss_warmup_steps"] == 2048


def test_matd3_v5_short_teacher_windows_reach_final_authority_during_training() -> None:
    config = build_config(
        "global_distilled_h2",
        steps=512,
        teacher_schedule=SCORECARD_TEACHER_SCHEDULE,
        teacher_label="scorecard",
    )
    validate_config(config)

    params = config["pipeline"][0]["exploration"]["params"]
    assert params["random_exploration_steps"] == 64
    assert params["initial_exploration_training_start_step"] == 64
    assert params["warm_start_policy_phaseout_steps"] == 256
    assert params["residual_action_growth_steps"] == 256
    assert params["actor_policy_loss_warmup_steps"] == 256
    assert params["actor_behavior_cloning_decay_start_step"] == 64
    assert params["actor_behavior_cloning_decay_steps"] == 960
    assert params["actor_behavior_cloning_extra_update_end_step"] == 1023


def test_packaged_teacher_path_is_portable_relative_to_repo() -> None:
    relative_path = Path("configs/demonstrations") / TEACHER_SCHEDULE.name
    config = build_config("cost_first_h2", teacher_schedule=relative_path)
    validate_config(config)

    assert config["pipeline"][0]["exploration"]["params"][
        "actor_behavior_cloning_teacher_hyperparameters"
    ]["schedule_path"] == str(relative_path)


def test_packaged_teacher_artifacts_are_pinned_and_complete() -> None:
    for path, expected_sha in (
        (TEACHER_SCHEDULE, TEACHER_SHA256),
        (SCORECARD_TEACHER_SCHEDULE, SCORECARD_TEACHER_SHA256),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        assert payload["horizon"] == 35039
        assert len(payload["series"]) == 17
        assert all(len(row["values"]) == 35039 for row in payload["series"])


def test_generation_routes_cost_and_scorecard_teachers_separately(
    tmp_path: Path,
) -> None:
    outputs = generate(
        tmp_path,
        teacher_schedule=TEACHER_SCHEDULE,
        scorecard_teacher_schedule=SCORECARD_TEACHER_SCHEDULE,
    )
    names = {path.name for path in outputs}

    assert "matd3_v5_cost_first_h2_milp_cost_teacher_annual_seed789.yaml" in names
    assert "matd3_v5_balanced_h2_milp_scorecard_teacher_annual_seed789.yaml" in names
    assert "matd3_v5_ramp_guard_h2_milp_scorecard_teacher_annual_seed789.yaml" in names
    assert (
        "matd3_v5_global_scorecard_h2_milp_scorecard_teacher_annual_seed789.yaml"
        in names
    )
    assert (
        "matd3_v5_global_distilled_h2_milp_scorecard_teacher_annual_seed789.yaml"
        in names
    )
    assert not any("balanced_h2_milp_cost_teacher" in name for name in names)
