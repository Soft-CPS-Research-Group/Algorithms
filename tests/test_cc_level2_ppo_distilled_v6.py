from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from algorithms.agents.cc_level2_agent import CCLevel2Agent
from scripts.generate_cc_level2_bidirectional_map_v6 import (
    NON_NEUTRAL_MULTIPLIERS,
    PROBE_MULTIPLIERS,
    build_probe_config,
    build_pulse_probe_config,
    generate as generate_map,
)
from scripts.generate_cc_level2_ppo_distilled_v6 import (
    ANNUAL_EPISODES,
    PRICE_MAX,
    PRICE_MIN,
    REPO_ROOT,
    SCORECARD_TEACHER,
    VARIANTS,
    build_config,
    build_paired_neutral_config,
    generate,
)
from utils.config_schema import validate_config


def test_v6_configs_are_bidirectional_teacher_distilled_and_safe() -> None:
    neutral = build_paired_neutral_config()
    validate_config(neutral)
    neutral_leaf = neutral["pipeline"][1]["exploration"]["params"]
    assert neutral["pipeline"][0]["algorithm"] == "FixedPriceSignal"
    assert neutral_leaf["residual_base_policy_hyperparameters"][
        "signal_price_response_mode"
    ] == "linear_bidirectional"

    for name, variant in VARIANTS.items():
        config = build_config(name)
        validate_config(config)
        manager, leaf = config["pipeline"]
        params = manager["hyperparameters"]
        residual = leaf["exploration"]["params"][
            "residual_base_policy_hyperparameters"
        ]

        assert config["simulator"]["episodes"] == ANNUAL_EPISODES
        assert manager["algorithm"] == "CCLevel2"
        assert params["price_min"] == pytest.approx(PRICE_MIN)
        assert params["price_max"] == pytest.approx(PRICE_MAX)
        assert params["bc_pretrain_enabled"] is True
        assert params["bc_collection_policy"] == "neutral_label_only"
        assert params["neutral_baseline_enabled"] is True
        assert params["neutral_warmup_episodes"] == 1
        assert params["bc_anchor_weight"] > params["bc_anchor_min_weight"] > 0.0
        assert residual["signal_price_response_mode"] == "linear_bidirectional"
        assert residual["signal_price_charge_reference_multiplier"] == pytest.approx(
            0.85
        )
        assert residual[
            "signal_price_discharge_reference_multiplier"
        ] == pytest.approx(1.15)
        assert config["tracking"]["tags"]["evaluation_teacher_access"] == "False"
        if variant["teacher_path"] is not None:
            assert params["bc_teacher_mode"] == "oracle_storage_schedule"
            assert not Path(params["bc_oracle_schedule_path"]).is_absolute()


def test_oracle_storage_teacher_maps_charge_and_discharge_to_price_direction() -> None:
    config = build_config("milp_scorecard_seed789", horizon=384)
    params = config["pipeline"][0]["hyperparameters"]
    agent = CCLevel2Agent({"algorithm": {"hyperparameters": params}})
    agent._building_names = [f"Building_{index}" for index in range(1, 18)]

    with gzip.open(SCORECARD_TEACHER, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    series = {
        row["building_id"]: np.asarray(row["values"], dtype=np.float64)
        for row in payload["series"]
    }

    # Find one hourly decision with both charging and discharging labels.
    selected_step = None
    expected_power = None
    for step in range(0, payload["horizon"] - 4, 4):
        powers = np.asarray(
            [series[f"Building_{index}"][step : step + 4].mean() for index in range(1, 18)]
        )
        if np.any(powers > 0.05) and np.any(powers < -0.05):
            selected_step = step
            expected_power = powers
            break
    assert selected_step is not None
    assert expected_power is not None

    agent._episode_step_context = selected_step
    multipliers = agent._oracle_storage_teacher_multipliers()
    assert np.all(multipliers[expected_power > 0.05] < 1.0)
    assert np.all(multipliers[expected_power < -0.05] > 1.0)
    assert np.all(multipliers >= PRICE_MIN)
    assert np.all(multipliers <= PRICE_MAX)


def test_neutral_label_collection_never_executes_the_teacher() -> None:
    config = build_config("milp_scorecard_seed789", horizon=384)
    params = dict(config["pipeline"][0]["hyperparameters"])
    params.update(
        {
            "num_buildings": 17,
            "bc_collect_steps": 1,
            "bc_train_steps": 1,
            "bc_train_chunk_steps": 1,
            "bc_progress_interval": 1,
        }
    )
    agent = CCLevel2Agent({"algorithm": {"hyperparameters": params}})
    agent._building_names = [f"Building_{index}" for index in range(1, 18)]
    agent._district_positions = list(range(agent._n_district))
    agent._building_feat_positions = [
        list(range(agent._n_district, agent._n_district + 6))
        for _ in range(17)
    ]
    observations = [np.zeros(agent._c_dim, dtype=np.float32) for _ in range(17)]

    agent.set_episode_context(episode_step=0)
    output = agent.predict(observations, deterministic=False)

    assert output == pytest.approx([1.0] * 17)
    assert agent._bc_pretrain_done is True
    assert agent._bc_anchor_inputs is not None
    assert agent._bc_anchor_targets is not None


def test_bidirectional_map_has_global_curve_and_one_coordinate_probes(tmp_path: Path) -> None:
    paths = generate_map(tmp_path, horizon=384, include_member_probes=True)
    assert len(paths) == len(PROBE_MULTIPLIERS) + 17 * 4

    global_high = build_probe_config(multiplier=1.30, horizon=384)
    validate_config(global_high)
    assert global_high["pipeline"][0]["hyperparameters"]["multipliers"] == [
        1.30
    ] * 17

    member_low = build_probe_config(
        multiplier=0.70,
        building_number=8,
        horizon=384,
    )
    validate_config(member_low)
    values = member_low["pipeline"][0]["hyperparameters"]["multipliers"]
    assert values[7] == pytest.approx(0.70)
    assert sum(value != 1.0 for value in values) == 1


def test_bidirectional_map_has_matched_global_and_member_pulses(
    tmp_path: Path,
) -> None:
    paths = generate_map(
        tmp_path,
        horizon=384,
        include_pulse_probes=True,
        include_member_pulse_probes=True,
    )
    assert len(paths) == (
        len(PROBE_MULTIPLIERS)
        + len(NON_NEUTRAL_MULTIPLIERS)
        + 17 * len(NON_NEUTRAL_MULTIPLIERS)
    )

    global_pulse = build_pulse_probe_config(
        multiplier=1.30,
        pulse_start=96,
        pulse_duration=4,
        horizon=384,
    )
    validate_config(global_pulse)
    schedule = global_pulse["pipeline"][0]["hyperparameters"]["schedule"]
    assert schedule == [
        {"start_step": 0, "multiplier": 1.0},
        {"start_step": 96, "multiplier": 1.30},
        {"start_step": 100, "multiplier": 1.0},
    ]

    member_pulse = build_pulse_probe_config(
        multiplier=0.70,
        building_number=8,
        pulse_start=96,
        pulse_duration=4,
        horizon=384,
    )
    validate_config(member_pulse)
    vector_schedule = member_pulse["pipeline"][0]["hyperparameters"][
        "vector_schedule"
    ]
    assert vector_schedule[0]["multipliers"] == [1.0] * 17
    assert vector_schedule[1]["multipliers"][7] == pytest.approx(0.70)
    assert sum(
        value != 1.0 for value in vector_schedule[1]["multipliers"]
    ) == 1
    assert vector_schedule[2]["multipliers"] == [1.0] * 17


def test_v6_generation_writes_all_candidates(tmp_path: Path) -> None:
    outputs = generate(tmp_path, horizon=384)
    assert len(outputs) == 1 + len(VARIANTS)
    assert all(path.is_file() for path in outputs)


def test_v6_schema_rejects_oracle_teacher_without_schedule() -> None:
    config = build_config("milp_cost_seed456", horizon=384)
    config["pipeline"][0]["hyperparameters"]["bc_oracle_schedule_path"] = None
    with pytest.raises(ValueError, match="bc_oracle_schedule_path"):
        validate_config(config)
