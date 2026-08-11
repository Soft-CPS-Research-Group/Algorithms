from __future__ import annotations

import pytest

from scripts.generate_cc_level2_causal_search_v5 import (
    COORDINATE_DELTA,
    INCUMBENT,
    NUM_BUILDINGS,
    build_config,
    build_vector_config,
    generate_custom_vector,
    generate_refinements,
    probe_names,
    vector_for_probe,
)
from utils.config_schema import validate_config


def test_v5_coordinate_campaign_has_exact_parity_and_two_probes_per_building() -> None:
    names = probe_names()

    assert len(names) == 1 + 2 * NUM_BUILDINGS
    assert vector_for_probe("vector_parity") == [INCUMBENT] * NUM_BUILDINGS
    for building_number in range(1, NUM_BUILDINGS + 1):
        down = vector_for_probe(f"building_{building_number}_down")
        up = vector_for_probe(f"building_{building_number}_up")
        changed = building_number - 1
        assert down[changed] == pytest.approx(INCUMBENT - COORDINATE_DELTA)
        assert up[changed] == pytest.approx(INCUMBENT + COORDINATE_DELTA)
        assert sum(value != INCUMBENT for value in down) == 1
        assert sum(value != INCUMBENT for value in up) == 1


def test_v5_coordinate_probe_is_causal_frozen_and_uses_vector_signal() -> None:
    config = build_config("building_15_down", pilot_steps=4096)
    validate_config(config)
    manager, leaf = config["pipeline"]

    assert manager["algorithm"] == "CausalPriceSignal"
    assert manager["frozen"] is True
    assert len(manager["hyperparameters"]["discount_multipliers"]) == NUM_BUILDINGS
    assert leaf["frozen"] is True
    residual = leaf["exploration"]["params"][
        "residual_base_policy_hyperparameters"
    ]
    leaf_params = leaf["exploration"]["params"]
    assert leaf_params["local_price_conditioning_enabled"] is True
    assert leaf_params["local_price_forecast_mode"] == "real_unmodified"
    assert leaf_params["residual_base_price_conditioning_enabled"] is True
    assert residual["signal_price_response_mode"] == "linear_discount"
    assert residual["signal_price_charge_reference_multiplier"] == pytest.approx(
        INCUMBENT
    )
    assert config["simulator"]["episodes"] == 1
    assert config["simulator"]["simulation_end_time_step"] == 4095
    assert config["simulator"]["community_market"]["enabled"] is True
    assert config["tracking"]["tags"]["uses_future_realized_data"] == "False"


def test_v5_coordinate_probe_rejects_bad_name_and_short_horizon() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        vector_for_probe("random")
    with pytest.raises(ValueError, match="at least 4096"):
        build_config("vector_parity", pilot_steps=1024)


def test_v5_custom_vector_supports_refinement_and_surcharge() -> None:
    values = [INCUMBENT] * NUM_BUILDINGS
    values[14] = 1.05
    config = build_vector_config(
        values,
        label="Building 15 surcharge 1.05",
        pilot_steps=4096,
        cc_action_interval=1,
    )
    validate_config(config)

    assert config["pipeline"][0]["hyperparameters"]["discount_multipliers"] == values
    assert config["tracking"]["tags"]["search_method"] == "vector_refinement"
    assert config["pipeline"][0]["hyperparameters"]["cc_action_interval"] == 1
    assert "building_15_surcharge_1_05" in config["simulator"]["export"][
        "session_name"
    ]


def test_v5_custom_vector_can_match_a_learned_candidate_episode() -> None:
    config = build_vector_config(
        [INCUMBENT] * NUM_BUILDINGS,
        label="matched episode control",
        pilot_steps=4096,
        cc_action_interval=4,
        episodes=5,
    )
    validate_config(config)

    assert config["simulator"]["episodes"] == 5
    assert config["simulator"]["export"]["final_episode_only"] is True
    assert config["tracking"]["tags"]["evaluation_episode_index"] == "5"
    assert config["tracking"]["tags"]["episode_realization_matched"] == "True"
    assert "-ep5-pilot4096" in config["simulator"]["export"]["session_name"]


def test_v5_custom_vector_rejects_bad_shape_range_and_label() -> None:
    with pytest.raises(ValueError, match="17 multipliers"):
        build_vector_config([0.9], label="short")
    with pytest.raises(ValueError, match=r"\[0.5, 1.3\]"):
        build_vector_config([1.31] * NUM_BUILDINGS, label="high")
    with pytest.raises(ValueError, match="label"):
        build_vector_config([0.9] * NUM_BUILDINGS, label="---")


def test_v5_refinement_generator_changes_only_requested_building(tmp_path) -> None:
    paths = generate_refinements(
        tmp_path,
        building_number=15,
        multipliers=[0.825, 1.05],
        pilot_steps=4096,
    )

    assert len(paths) == 2
    for path, expected in zip(paths, [0.825, 1.05]):
        import yaml

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        values = config["pipeline"][0]["hyperparameters"]["discount_multipliers"]
        assert values[14] == pytest.approx(expected)
        assert sum(value != INCUMBENT for value in values) == 1


def test_v5_custom_vector_generator_writes_combined_candidate(tmp_path) -> None:
    values = [INCUMBENT] * NUM_BUILDINGS
    values[1] = 0.85
    values[14] = 0.825

    path = generate_custom_vector(
        tmp_path,
        multipliers=values,
        label="combined safe incumbent",
        pilot_steps=4096,
    )

    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["pipeline"][0]["hyperparameters"]["discount_multipliers"] == values
    assert config["tracking"]["tags"]["recipe"] == "combined_safe_incumbent"
