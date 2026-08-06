#!/usr/bin/env python3
"""Generate the deployable causal-online CC over the frozen PPO leaf."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.generate_cc_causal_price_control_v4 import (
        PPO_SEED,
        REPO_ROOT,
        ppo_fixed_recipe,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from generate_cc_causal_price_control_v4 import (  # type: ignore[no-redef]
        PPO_SEED,
        REPO_ROOT,
        ppo_fixed_recipe,
    )


EXPERIMENT_NAME = "cc_ppo_causal_online_v5p3"
CHARGE_RATES = (0.45, 0.60)
DISCOUNT_MULTIPLIER = 0.95
NEUTRAL_MULTIPLIER = 1.0
CC_ACTION_INTERVAL = 4
SMOKE_TRANSITIONS = 384


def causal_online_recipe(charge_rate: float) -> dict[str, Any]:
    if charge_rate not in CHARGE_RATES:
        raise ValueError(f"Unsupported V5.3 charge rate: {charge_rate}")
    config = copy.deepcopy(ppo_fixed_recipe(1.0))
    label = f"{charge_rate:.2f}".replace(".", "p")
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-PPO causal online CAE charge {charge_rate:.2f} seed {PPO_SEED}",
            "description": (
                "Deployable causal Level-1 price rule over the frozen PPO leaf. "
                "At each hourly decision it uses only the current pre-action "
                "tariff, its three forecasts and observed community export; it "
                "never reads an annual schedule or a next observation."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "controller": "cc_ppo_causal_online",
            "recipe": "current_cheap_and_current_export",
            "causal_online": "True",
            "trace_derived": "False",
            "uses_future_realized_data": "False",
            "promotion_eligible": "True",
            "in_sample_diagnostic": "False",
            "cc_action_interval": str(CC_ACTION_INTERVAL),
            "discount_multiplier": str(DISCOUNT_MULTIPLIER),
            "signal_price_charge_rate": str(charge_rate),
            "cc_observation_contract": "current_pre_action_only",
        }
    )
    config["pipeline"][0] = {
        "algorithm": "CausalPriceSignal",
        "count": 1,
        "frozen": True,
        "hyperparameters": {
            "neutral_multiplier": NEUTRAL_MULTIPLIER,
            "discount_multiplier": DISCOUNT_MULTIPLIER,
            "cc_action_interval": CC_ACTION_INTERVAL,
            "community_export_threshold_kw": 1.0e-9,
            "forecast_mean_margin": 0.20,
            "forecast_min_margin": 0.10,
            "spread_floor_ratio": 0.05,
        },
    }
    config["pipeline"][1]["exploration"]["params"][
        "residual_base_policy_hyperparameters"
    ]["signal_price_charge_rate"] = float(charge_rate)
    config["simulator"]["export"]["session_name"] = (
        f"cc-ppo-v5p3-causal-online-cae-charge-{label}-seed{PPO_SEED}"
    )
    config["checkpointing"]["checkpoint_interval"] = None
    return config


def _derive_smoke(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    smoke["metadata"]["run_name"] += " [functional smoke]"
    smoke["metadata"]["description"] += (
        " Short functional smoke only; never use as performance evidence."
    )
    smoke["tracking"]["tags"]["evidence"] = "functional_smoke"
    smoke["tracking"]["tags"]["promotion_eligible"] = "False"
    smoke["simulator"]["episodes"] = 1
    smoke["simulator"]["simulation_start_time_step"] = 0
    smoke["simulator"]["simulation_end_time_step"] = SMOKE_TRANSITIONS
    smoke["simulator"]["episode_time_steps"] = SMOKE_TRANSITIONS + 1
    smoke["simulator"]["export"]["session_name"] += "-smoke"
    smoke["checkpointing"]["checkpoint_interval"] = None
    return smoke


def _name(charge_rate: float) -> str:
    label = f"{charge_rate:.2f}".replace(".", "p")
    return f"cc_ppo_causal_online_cae_charge_{label}_seed{PPO_SEED}.yaml"


def generate(output_dir: Path, *, smoke: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for charge_rate in CHARGE_RATES:
        config = causal_online_recipe(charge_rate)
        if smoke:
            config = _derive_smoke(config)
        path = output_dir / _name(charge_rate)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / EXPERIMENT_NAME,
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for path in generate(args.output_dir, smoke=args.smoke):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
