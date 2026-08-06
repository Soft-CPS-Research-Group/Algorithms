from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.build_cc_ppo_schedule_probes import (
    _compress_schedule,
    _probe_config,
    derive_schedule_masks,
)
from scripts.generate_cc_ppo_controllability_v5 import (
    EXPERIMENT_NAME,
    FORECAST_MODES,
    PPO_SEED,
    actor_price_probe,
    generate,
)
from utils.config_schema import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments" / EXPERIMENT_NAME


def _label(mode: str) -> str:
    return (
        "actor_current_only"
        if mode == "real_unmodified"
        else "actor_current_and_forecasts"
    )


@pytest.mark.parametrize("forecast_mode", FORECAST_MODES)
def test_v5_actor_price_path_ablation_is_frozen_explicit_ood(
    forecast_mode: str,
):
    config = actor_price_probe(forecast_mode)
    validate_config(config)
    manager, ppo = config["pipeline"]
    exploration = ppo["exploration"]["params"]

    assert config["simulator"]["community_market"]["enabled"] is True
    assert config["simulator"]["episode_time_steps"] == 35040
    assert manager["algorithm"] == "FixedPriceSignal"
    assert manager["hyperparameters"]["multiplier"] == pytest.approx(0.95)
    assert ppo["frozen"] is True
    assert exploration["local_price_conditioning_enabled"] is True
    assert exploration["local_price_forecast_mode"] == forecast_mode
    assert exploration["residual_base_policy"] == "SignalAwareRBCSmartLocal"
    assert exploration["residual_base_price_conditioning_enabled"] is True
    assert config["tracking"]["tags"]["promotion_eligible"] == "False"
    assert config["tracking"]["tags"]["inference_distribution"] == (
        "explicit_ood_diagnostic"
    )


def test_generated_v5_templates_match_committed_templates(tmp_path: Path):
    for path in generate(tmp_path):
        committed = CONFIG_ROOT / path.name
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == yaml.safe_load(
            committed.read_text(encoding="utf-8")
        )


@pytest.mark.parametrize("forecast_mode", FORECAST_MODES)
def test_v5_smoke_uses_the_registered_short_functional_window(
    tmp_path: Path,
    forecast_mode: str,
):
    paths = {path.name: path for path in generate(tmp_path, smoke=True)}
    name = f"cc_ppo_fixed_0p95_{_label(forecast_mode)}_seed{PPO_SEED}.yaml"
    config = yaml.safe_load(paths[name].read_text(encoding="utf-8"))
    validate_config(config)

    assert config["simulator"]["simulation_start_time_step"] == 0
    assert config["simulator"]["simulation_end_time_step"] == 384
    assert config["simulator"]["episode_time_steps"] == 385
    assert config["tracking"]["tags"]["evidence"] == "functional_smoke"


def test_schedule_masks_separate_tariff_export_and_retrospective_hypotheses():
    pricing_fields = [
        "timestamp",
        "electricity_pricing-$/kWh",
        "electricity_pricing_predicted_1-$/kWh",
        "electricity_pricing_predicted_2-$/kWh",
        "electricity_pricing_predicted_3-$/kWh",
    ]
    timestamps = [f"t{index}" for index in range(4)]
    pricing_rows = [
        {
            "timestamp": timestamp,
            "electricity_pricing-$/kWh": str(current),
            "electricity_pricing_predicted_1-$/kWh": "2",
            "electricity_pricing_predicted_2-$/kWh": "2",
            "electricity_pricing_predicted_3-$/kWh": "2",
        }
        for timestamp, current in zip(timestamps, (1, 1, 3, 3))
    ]
    community_fields = ["timestamp", "Net Electricity Consumption-kWh", "Price-$"]
    neutral_rows = [
        {
            "timestamp": timestamp,
            "Net Electricity Consumption-kWh": str(net),
            "Price-$": "10",
        }
        for timestamp, net in zip(timestamps, (1, -1, 1, -1))
    ]
    discount_rows = [
        {
            "timestamp": timestamp,
            "Net Electricity Consumption-kWh": str(net),
            "Price-$": str(cost),
        }
        for timestamp, net, cost in zip(
            timestamps,
            (1, -1, 1, -1),
            (9, 9, 11, 11),
        )
    ]

    masks = derive_schedule_masks(
        pricing_fieldnames=pricing_fields,
        pricing_rows=pricing_rows,
        neutral_fieldnames=community_fields,
        neutral_rows=neutral_rows,
        discount_fieldnames=community_fields,
        discount_rows=discount_rows,
        block_steps=2,
        activation_fraction=0.5,
    )

    assert masks["native_cheap"] == [True, False]
    assert masks["community_export"] == [True, True]
    assert masks["cheap_or_export"] == [True, True]
    assert masks["cheap_and_export"] == [True, False]
    assert masks["retrospective_cost"] == [True, False]
    assert _compress_schedule(
        masks["retrospective_cost"],
        block_steps=2,
        discount=0.95,
        neutral=1.0,
    ) == [
        {"start_step": 0, "multiplier": 0.95},
        {"start_step": 2, "multiplier": 1.0},
    ]


@pytest.mark.parametrize("charge_rate", (0.3, 0.45, 0.6))
def test_temporal_probe_records_and_applies_signal_charge_rate(charge_rate: float):
    config = _probe_config(
        recipe="community_export",
        schedule=[{"start_step": 0, "multiplier": 0.95}],
        discount=0.95,
        block_steps=4,
        signal_price_charge_rate=charge_rate,
        smoke_transitions=None,
    )
    validate_config(config)

    tags = config["tracking"]["tags"]
    residual = config["pipeline"][1]["exploration"]["params"][
        "residual_base_policy_hyperparameters"
    ]
    assert float(tags["signal_price_charge_rate"]) == pytest.approx(charge_rate)
    assert residual["signal_price_charge_rate"] == pytest.approx(charge_rate)
    assert f"charge-{charge_rate:.2f}".replace(".", "p") in config["simulator"][
        "export"
    ]["session_name"]


@pytest.mark.parametrize("charge_rate", (-0.01, 1.01))
def test_temporal_probe_rejects_invalid_signal_charge_rate(charge_rate: float):
    with pytest.raises(ValueError, match="signal_price_charge_rate"):
        _probe_config(
            recipe="community_export",
            schedule=[{"start_step": 0, "multiplier": 0.95}],
            discount=0.95,
            block_steps=4,
            signal_price_charge_rate=charge_rate,
            smoke_transitions=None,
        )
