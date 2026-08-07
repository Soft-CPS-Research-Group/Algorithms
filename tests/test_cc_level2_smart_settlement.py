from __future__ import annotations

from pathlib import Path

import yaml

from scripts.generate_cc_level2_smart_settlement import (
    EMPIRICAL_VECTOR,
    NUM_BUILDINGS,
    generate,
)
from utils.config_schema import validate_config


def test_cc_level2_smart_campaign_configs_are_valid_and_matched(tmp_path: Path) -> None:
    paths = generate(tmp_path)

    assert len(paths) == 3
    configs = {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in paths
    }
    for config in configs.values():
        validate_config(config)
        simulator = config["simulator"]
        market = simulator["community_market"]
        assert simulator["dataset_name"] == (
            "citylearn_three_phase_electrical_service_demo_15min_parquet"
        )
        assert simulator["simulation_start_time_step"] == 0
        assert simulator["simulation_end_time_step"] == 35039
        assert market["enabled"] is True
        assert market["local_price_ratio_to_grid_import"] == 0.8
        assert market["intra_community_sell_ratio"] == 0.8
        assert market["grid_export_price"] == 0.0
        assert config["pipeline"][1]["algorithm"] == "SignalAwareRBC"
        assert config["pipeline"][1]["frozen"] is True

    neutral = configs["cc_l2_smart_neutral_vector"]
    assert neutral["pipeline"][0]["hyperparameters"]["multipliers"] == (
        [1.0] * NUM_BUILDINGS
    )

    empirical = configs["cc_l2_smart_empirical_vector"]
    assert empirical["pipeline"][0]["hyperparameters"]["multipliers"] == (
        EMPIRICAL_VECTOR
    )

    learned = configs["cc_l2_smart_learned_seed123"]
    assert learned["simulator"]["reward_function"] == "CCRewardLevel2"
    assert learned["simulator"]["reward_function_kwargs"]["cost_aggregation"] == (
        "community_settled"
    )
    manager = learned["pipeline"][0]
    assert manager["algorithm"] == "CCLevel2"
    assert manager["hyperparameters"]["reference_multipliers"] == EMPIRICAL_VECTOR
    assert manager["hyperparameters"]["bc_pretrain_enabled"] is False
