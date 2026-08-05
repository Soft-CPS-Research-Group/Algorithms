#!/usr/bin/env python3
"""Generate the causal CC-SMART and corrected CC-PPO price campaign V4."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.generate_cc_smart_cost_focus_v2 import ANNUAL_SMART_REFERENCES
    from scripts.generate_ppo_cc_settlement_templates import (
        REPO_ROOT,
        _ppo_configs,
        _smart_configs,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from generate_cc_smart_cost_focus_v2 import ANNUAL_SMART_REFERENCES
    from generate_ppo_cc_settlement_templates import (
        REPO_ROOT,
        _ppo_configs,
        _smart_configs,
    )


EXPERIMENT_NAME = "cc_causal_price_control_v4"
SMART_SEED = 123
PPO_SEED = 789
PPO_FIXED_MULTIPLIERS = (0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30)
SMART_RECIPES = (
    "settled_cost_hourly",
    "settled_cost_15min",
    "settled_cost_peak_15min",
)
FIXED_SMOKE_TRANSITIONS = 384


def _tag(config: dict[str, Any], *, controller: str, recipe: str) -> None:
    config["metadata"]["experiment_name"] = EXPERIMENT_NAME
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "controller": controller,
            "recipe": recipe,
            "settlement": "enabled",
            "evidence_horizon": "full_year",
            "leaf_frozen": "True",
        }
    )


def _settled_reward(*, peak_weight: float, ramp_weight: float) -> dict[str, Any]:
    return {
        "cost_aggregation": "community_net",
        "w_cost": 1.0,
        "w_member_retail_cost": 0.0,
        "w_peak": float(peak_weight),
        "w_ramp": float(ramp_weight),
        "w_export": 0.0,
        "w_violation": 2.0,
        **ANNUAL_SMART_REFERENCES,
    }


def smart_recipe(recipe: str) -> dict[str, Any]:
    if recipe not in SMART_RECIPES:
        raise ValueError(f"Unknown V4 SMART recipe: {recipe}")

    _, config = _smart_configs()
    config = copy.deepcopy(config)
    _tag(config, controller="cc_smart", recipe=recipe)
    config["metadata"].update(
        {
            "run_name": f"CC-SMART causal V4 {recipe} seed {SMART_SEED}",
            "description": (
                "Post-response-sweep CC-SMART candidate centred on fixed 1.3; "
                "optimises settlement cost with an explicit temporal-control ablation."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "cc_seed": str(SMART_SEED),
            "selection_basis": "post_fixed_sweep_1p3",
            "training_episodes": "10",
        }
    )
    config["training"]["seed"] = SMART_SEED
    config["simulator"]["episodes"] = 10
    config["simulator"]["export"]["session_name"] = (
        f"cc-causal-v4-smart-{recipe}-seed{SMART_SEED}"
    )
    config["simulator"]["reward_function_kwargs"] = _settled_reward(
        peak_weight=0.05 if recipe == "settled_cost_peak_15min" else 0.0,
        ramp_weight=0.02 if recipe == "settled_cost_peak_15min" else 0.0,
    )
    config["checkpointing"]["checkpoint_interval"] = 35040

    params = config["pipeline"][0]["hyperparameters"]
    fifteen_minute = recipe != "settled_cost_hourly"
    params.update(
        {
            "cc_action_interval": 1 if fifteen_minute else 4,
            # Seven physical days in both temporal-control variants.
            "num_steps": 672 if fifteen_minute else 168,
            # 0.99875 per 15 minutes is approximately 0.995 per hour.
            "gamma": 0.99875 if fifteen_minute else 0.995,
            "mini_batch_size": 96 if fifteen_minute else 84,
            "ent_coef": 0.002,
            "reference_multiplier": 1.3,
            "policy_residual_scale": 0.5,
            # Cost is the primary objective. These remain non-zero only to
            # suppress economically meaningless high-frequency jitter.
            "w_factor": 0.01,
            "w_smoothness": 0.005,
            "bc_collect_steps": 8760,
            "bc_train_steps": 4000,
            "bc_w_cost": 1.0,
            "bc_w_peak": 0.05 if recipe == "settled_cost_peak_15min" else 0.0,
            "bc_w_ramp": 0.02 if recipe == "settled_cost_peak_15min" else 0.0,
            "bc_w_export": 0.0,
            "bc_w_violation": 2.0,
            "bc_w_headroom": 0.0,
            "bc_reference_ramping": ANNUAL_SMART_REFERENCES[
                "reference_ramping"
            ],
        }
    )
    config["tracking"]["tags"].update(
        {
            "cc_action_interval": str(params["cc_action_interval"]),
            "ppo_horizon_cc_decisions": str(params["num_steps"]),
            "effective_price_min": "0.9",
            "effective_price_max": "1.3",
        }
    )
    return config


def ppo_fixed_recipe(multiplier: float) -> dict[str, Any]:
    if multiplier not in PPO_FIXED_MULTIPLIERS:
        raise ValueError(f"Unsupported V4 PPO probe multiplier: {multiplier}")

    ppo, _ = _ppo_configs()
    config = copy.deepcopy(ppo)
    recipe = f"base_only_fixed_{multiplier:.2f}".replace(".", "p")
    _tag(config, controller="cc_ppo_fixed_probe", recipe=recipe)
    config["metadata"].update(
        {
            "run_name": (
                f"CC-PPO corrected residual-base fixed {multiplier:.2f} seed {PPO_SEED}"
            ),
            "description": (
                "Annual causal response probe: the multiplier controls only the "
                "strict-local SMART residual base; the frozen PPO actor receives "
                "its original encoded observation unchanged."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "ppo_seed": str(PPO_SEED),
            "fixed_multiplier": str(multiplier),
            "cc_price_scope": "strict_local_residual_base_only",
            "ppo_actor_price_conditioning": "False",
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"cc-causal-v4-ppo-base-only-fixed-{multiplier:.2f}-seed{PPO_SEED}"
        .replace(".", "p")
    )
    config["pipeline"][0]["hyperparameters"]["multiplier"] = float(multiplier)
    exploration = config["pipeline"][1]["exploration"]["params"]
    exploration.update(
        {
            "local_price_conditioning_enabled": False,
            "residual_base_policy": "SignalAwareRBCSmartLocal",
            "residual_base_price_conditioning_enabled": True,
        }
    )
    exploration["residual_base_policy_hyperparameters"][
        "signal_price_charge_rate"
    ] = 0.6
    return config


def _fixed_name(multiplier: float) -> str:
    return f"cc_ppo_base_price_fixed_{multiplier:.2f}_seed{PPO_SEED}.yaml".replace(
        ".", "p", 1
    )


def generate(output_dir: Path) -> list[Path]:
    payloads: dict[str, dict[str, Any]] = {
        f"cc_smart_{recipe}_seed{SMART_SEED}.yaml": smart_recipe(recipe)
        for recipe in SMART_RECIPES
    }
    payloads.update(
        {
            _fixed_name(multiplier): ppo_fixed_recipe(multiplier)
            for multiplier in PPO_FIXED_MULTIPLIERS
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, payload in payloads.items():
        path = output_dir / filename
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def _derive_smoke(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    manager = smoke["pipeline"][0]
    if manager["algorithm"] == "CCLevel1":
        params = manager["hyperparameters"]
        transitions = int(params["num_steps"]) * int(params["cc_action_interval"])
        smoke["simulator"]["episodes"] = 3
        params["bc_collect_steps"] = int(params["num_steps"])
        params["bc_train_steps"] = 2
        smoke["checkpointing"]["checkpoint_interval"] = transitions
    else:
        transitions = FIXED_SMOKE_TRANSITIONS
        smoke["simulator"]["episodes"] = 1
        smoke["checkpointing"]["checkpoint_interval"] = None

    smoke["metadata"]["run_name"] += " [functional smoke]"
    smoke["metadata"]["description"] += (
        " Short functional smoke only; never use as performance evidence."
    )
    smoke["tracking"]["tags"]["evidence"] = "functional_smoke"
    smoke["simulator"]["simulation_start_time_step"] = 0
    smoke["simulator"]["simulation_end_time_step"] = transitions
    smoke["simulator"]["episode_time_steps"] = transitions + 1
    smoke["simulator"]["export"]["session_name"] += "-smoke"
    return smoke


def generate_smokes(output_dir: Path) -> list[Path]:
    payloads: dict[str, dict[str, Any]] = {
        f"cc_smart_{recipe}_seed{SMART_SEED}.yaml": _derive_smoke(
            smart_recipe(recipe)
        )
        for recipe in SMART_RECIPES
    }
    payloads.update(
        {
            _fixed_name(multiplier): _derive_smoke(ppo_fixed_recipe(multiplier))
            for multiplier in PPO_FIXED_MULTIPLIERS
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, payload in payloads.items():
        path = output_dir / filename
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "configs/experiments" / EXPERIMENT_NAME,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate short functional smokes instead of annual evidence configs.",
    )
    args = parser.parse_args()
    writer = generate_smokes if args.smoke else generate
    for path in writer(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
