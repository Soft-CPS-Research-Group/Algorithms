#!/usr/bin/env python3
"""Generate stronger deployable V5.3 causal-online CC-PPO candidates.

The first V5.3 configs used a 0.95 discount even though the matched V5.2
ablation identified 0.90 as the useful causal intervention.  V5.3b keeps the
same current-observation-only rule and tests the matched cost/balanced charge
rates plus a 15-minute decision-density ablation.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.generate_cc_ppo_causal_online_v5p3 import (
        PPO_SEED,
        REPO_ROOT,
        causal_online_recipe,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from generate_cc_ppo_causal_online_v5p3 import (
        PPO_SEED,
        REPO_ROOT,
        causal_online_recipe,
    )


EXPERIMENT_NAME = "cc_ppo_causal_online_v5p3b"
RECIPES = {
    "hourly_balanced": {"charge_rate": 0.45, "interval": 4},
    "hourly_cost": {"charge_rate": 0.60, "interval": 4},
    "15min_cost": {"charge_rate": 0.60, "interval": 1},
}
DISCOUNT_MULTIPLIER = 0.90


def recipe(name: str) -> dict[str, Any]:
    if name not in RECIPES:
        raise ValueError(f"Unknown V5.3b recipe: {name}")
    variant = RECIPES[name]
    charge_rate = float(variant["charge_rate"])
    interval = int(variant["interval"])
    config = copy.deepcopy(causal_online_recipe(charge_rate))
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-PPO causal online V5.3b {name} seed {PPO_SEED}",
            "description": (
                "Deployable current-observation-only cheap-and-export rule. "
                "Uses the 0.90 intervention identified by V5.2 without reading "
                "an annual trace, a future outcome or community state in the PPO leaf."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": name,
            "discount_multiplier": str(DISCOUNT_MULTIPLIER),
            "signal_price_charge_rate": str(charge_rate),
            "cc_action_interval": str(interval),
            "selection_basis": "matched_v5p2_discount_then_causal_replay",
        }
    )
    manager = config["pipeline"][0]["hyperparameters"]
    manager.update(
        {
            "discount_multiplier": DISCOUNT_MULTIPLIER,
            "cc_action_interval": interval,
        }
    )
    config["pipeline"][1]["exploration"]["params"][
        "residual_base_policy_hyperparameters"
    ]["signal_price_charge_rate"] = charge_rate
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{name}-seed{PPO_SEED}"
    )
    return config


def derive_smoke(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    transitions = 384
    smoke["metadata"]["run_name"] += " [functional smoke]"
    smoke["tracking"]["tags"].update(
        {"evidence": "functional_smoke", "promotion_eligible": "False"}
    )
    smoke["simulator"].update(
        {
            "episodes": 1,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": transitions,
            "episode_time_steps": transitions + 1,
        }
    )
    smoke["simulator"]["export"]["session_name"] += "-smoke"
    return smoke


def generate(output_dir: Path, *, smoke: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in RECIPES:
        config = recipe(name)
        if smoke:
            config = derive_smoke(config)
        path = output_dir / f"cc_ppo_causal_online_{name}_seed{PPO_SEED}.yaml"
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
