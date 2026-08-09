from __future__ import annotations

from pathlib import Path

import yaml

from scripts.generate_cc_ppo_causal_online_v5p3b import RECIPES, generate
from utils.config_schema import validate_config


def _load(path: Path):
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_v5p3b_is_causal_matched_and_settled(tmp_path: Path) -> None:
    configs = [_load(path) for path in generate(tmp_path)]

    assert len(configs) == len(RECIPES)
    for config in configs:
        manager, leaf = config.pipeline
        assert config.simulator.community_market.enabled is True
        assert manager.algorithm == "CausalPriceSignal"
        assert manager.frozen is True
        assert manager.hyperparameters.discount_multiplier == 0.90
        assert leaf.algorithm == "PPO"
        assert leaf.frozen is True
        assert leaf.exploration.params["local_price_conditioning_enabled"] is False
        assert leaf.exploration.params[
            "residual_base_price_conditioning_enabled"
        ] is True
        assert config.tracking.tags["causal_online"] == "True"
        assert config.tracking.tags["uses_future_realized_data"] == "False"


def test_v5p3b_contains_balanced_cost_and_density_ablations(tmp_path: Path) -> None:
    configs = {path.stem: _load(path) for path in generate(tmp_path)}
    balanced = configs["cc_ppo_causal_online_hourly_balanced_seed789"]
    hourly = configs["cc_ppo_causal_online_hourly_cost_seed789"]
    dense = configs["cc_ppo_causal_online_15min_cost_seed789"]

    def charge(config):
        return config.pipeline[1].exploration.params[
            "residual_base_policy_hyperparameters"
        ]["signal_price_charge_rate"]

    assert balanced.pipeline[0].hyperparameters.cc_action_interval == 4
    assert charge(balanced) == 0.45
    assert hourly.pipeline[0].hyperparameters.cc_action_interval == 4
    assert charge(hourly) == 0.60
    assert dense.pipeline[0].hyperparameters.cc_action_interval == 1
    assert charge(dense) == 0.60
