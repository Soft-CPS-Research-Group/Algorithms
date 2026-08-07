from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.build_cc_ppo_online_v6 import PROFILES, build_configs
from utils.config_schema import validate_config


BASE = Path(
    "configs/experiments/cc_causal_price_control_v4/"
    "cc_ppo_base_price_fixed_1p00_seed789.yaml"
)


def test_v6_builds_three_causal_online_candidates(tmp_path: Path) -> None:
    outputs = build_configs(
        base_config=BASE,
        output_dir=tmp_path,
        signal_price_charge_rate=0.45,
        episodes=6,
    )

    assert len(outputs) == len(PROFILES) == 3
    for output in outputs:
        config = yaml.safe_load(output.read_text(encoding="utf-8"))
        validate_config(config)

        simulator = config["simulator"]
        assert simulator["episodes"] == 6
        assert simulator["deterministic_finish"] is True
        assert simulator["reward_function_kwargs"]["cost_aggregation"] == (
            "community_settled"
        )
        assert simulator["community_market"]["enabled"] is True

        manager, leaf = config["pipeline"]
        assert manager["algorithm"] == "CCLevel1"
        assert manager["frozen"] is False
        assert manager["hyperparameters"]["price_min"] == pytest.approx(0.5)
        assert manager["hyperparameters"]["price_max"] == pytest.approx(1.3)
        assert leaf["algorithm"] == "PPO"
        assert leaf["frozen"] is True
        params = leaf["exploration"]["params"]
        assert params["local_price_conditioning_enabled"] is False
        assert params["residual_base_price_conditioning_enabled"] is True
        assert params["residual_base_policy"] == "SignalAwareRBCSmartLocal"
        assert params["residual_base_policy_hyperparameters"][
            "signal_price_charge_rate"
        ] == pytest.approx(0.45)


def test_v6_smoke_contract_keeps_train_then_clean_eval(tmp_path: Path) -> None:
    outputs = build_configs(
        base_config=BASE,
        output_dir=tmp_path,
        signal_price_charge_rate=0.30,
        episodes=3,
        smoke_steps=96,
    )

    for output in outputs:
        config = yaml.safe_load(output.read_text(encoding="utf-8"))
        validate_config(config)
        simulator = config["simulator"]
        assert simulator["simulation_end_time_step"] == 95
        assert simulator["episode_time_steps"] == 96
        assert simulator["episodes"] == 3
        manager = config["pipeline"][0]["hyperparameters"]
        assert manager["bc_collect_steps"] == 24
        assert manager["bc_train_steps"] == 16


@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_v6_rejects_invalid_signal_rate(tmp_path: Path, rate: float) -> None:
    with pytest.raises(ValueError, match="signal_price_charge_rate"):
        build_configs(
            base_config=BASE,
            output_dir=tmp_path,
            signal_price_charge_rate=rate,
        )
