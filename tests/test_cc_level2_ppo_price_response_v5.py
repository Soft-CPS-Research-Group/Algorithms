from __future__ import annotations

import pytest

from scripts.generate_cc_level2_price_response_v5 import RECIPES, build_config
from utils.config_schema import validate_config


@pytest.mark.parametrize("name", RECIPES)
def test_price_response_recipe_is_causal_frozen_and_valid(name: str) -> None:
    config = build_config(name, pilot_steps=4096)
    validate_config(config)
    manager, leaf = config["pipeline"]
    params = leaf["exploration"]["params"]

    assert manager["algorithm"] == "CausalPriceSignal"
    assert manager["frozen"] is True
    assert manager["hyperparameters"]["discount_multipliers"] == [0.9] * 17
    assert leaf["algorithm"] == "PPO"
    assert leaf["frozen"] is True
    assert params["local_price_conditioning_enabled"] is RECIPES[name][
        "local_price_conditioning_enabled"
    ]
    assert params["residual_base_price_conditioning_enabled"] is RECIPES[name][
        "residual_base_price_conditioning_enabled"
    ]
    assert params["local_price_forecast_mode"] == RECIPES[name][
        "local_price_forecast_mode"
    ]
    assert config["simulator"]["community_market"]["enabled"] is True
    assert config["tracking"]["tags"]["uses_future_realized_data"] == "False"


def test_unknown_price_response_recipe_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown PPO price-response"):
        build_config("mystery")
