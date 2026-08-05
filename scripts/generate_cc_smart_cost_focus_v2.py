#!/usr/bin/env python3
"""Generate the annual CC-SMART cost-focus V2 ablation campaign."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.generate_ppo_cc_settlement_templates import REPO_ROOT, _smart_configs
    from scripts.generate_ppo_cc_settlement_smokes import _derive_smoke
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from generate_ppo_cc_settlement_templates import REPO_ROOT, _smart_configs
    from generate_ppo_cc_settlement_smokes import _derive_smoke


EXPERIMENT_NAME = "cc_smart_cost_focus_v2"
SEED = 123
# 1,345 rows produce 1,344 transitions = one 336-decision CC rollout.
SMOKE_STEPS = 1345

# P75/P90 references measured on the matching annual neutral SMART replay
# b0747ffe-5a62-4e68-8218-765deffd4c78.  Keeping the source explicit prevents
# the stale hard-coded references in the historical V1 recipe from silently
# changing the relative importance of cost, peak and ramping.
ANNUAL_SMART_REFERENCES = {
    "target_import": 5.936217704103814,
    "reference_cost": 1.5386517322861484,
    "reference_member_retail_cost": 1.5496793513624267,
    "reference_peak": 9.506766987795396,
    "reference_ramping": 2.57473698630929,
    "reference_export": 5.364232179522515,
}

RECIPE_NAMES = (
    "legacy_long_control",
    "settled_focus_regularized",
    "settled_focus_adaptive",
    "hybrid_physical_adaptive",
)


def _cost_focus_reward(*, member_retail_weight: float) -> dict[str, Any]:
    return {
        "cost_aggregation": "community_net",
        "w_cost": 1.0,
        "w_member_retail_cost": float(member_retail_weight),
        "w_peak": 0.15,
        "w_ramp": 0.10,
        "w_export": 0.02,
        "w_violation": 2.0,
        **ANNUAL_SMART_REFERENCES,
    }


def _configure_long_horizon(config: dict[str, Any]) -> None:
    config["simulator"]["episodes"] = 8
    config["checkpointing"]["checkpoint_interval"] = 35040

    params = config["pipeline"][0]["hyperparameters"]
    params.update(
        {
            # One CC decision per hour: 336 decisions represent two weeks.
            "num_steps": 336,
            "gamma": 0.995,
            "mini_batch_size": 84,
        }
    )


def _configure_cost_focus_bc(config: dict[str, Any]) -> None:
    params = config["pipeline"][0]["hyperparameters"]
    params.update(
        {
            "bc_w_cost": 1.0,
            "bc_w_peak": 0.15,
            "bc_w_ramp": 0.10,
            "bc_w_export": 0.02,
            "bc_w_violation": 2.0,
            "bc_w_headroom": 0.5,
            "bc_reference_ramping": ANNUAL_SMART_REFERENCES[
                "reference_ramping"
            ],
        }
    )


def _recipe(recipe_name: str) -> dict[str, Any]:
    if recipe_name not in RECIPE_NAMES:
        raise ValueError(f"Unknown V2 recipe: {recipe_name}")

    _, config = _smart_configs()
    config = copy.deepcopy(config)
    _configure_long_horizon(config)

    descriptions = {
        "legacy_long_control": (
            "Historical V1 reward and regularization with the V2 training "
            "budget and two-week PPO horizon."
        ),
        "settled_focus_regularized": (
            "Calibrated settled-cost-first reward with historical action "
            "regularization."
        ),
        "settled_focus_adaptive": (
            "Calibrated settled-cost-first reward with moderate action "
            "regularization."
        ),
        "hybrid_physical_adaptive": (
            "Calibrated settled-cost-first reward plus a smaller member-retail "
            "counterfactual term and moderate action regularization."
        ),
    }
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-SMART cost focus V2 {recipe_name} seed {SEED}",
            "description": descriptions[recipe_name],
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "controller": "cc_smart",
            "recipe": recipe_name,
            "cc_seed": str(SEED),
            "training_episodes": "8",
            "ppo_horizon_cc_decisions": "336",
            "evidence_horizon": "full_year",
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"cc-smart-cost-focus-v2-{recipe_name}-seed{SEED}"
    )
    config["training"]["seed"] = SEED

    if recipe_name == "legacy_long_control":
        return config

    member_retail_weight = 0.25 if recipe_name == "hybrid_physical_adaptive" else 0.0
    config["simulator"]["reward_function_kwargs"] = _cost_focus_reward(
        member_retail_weight=member_retail_weight
    )
    _configure_cost_focus_bc(config)

    params = config["pipeline"][0]["hyperparameters"]
    if recipe_name == "settled_focus_regularized":
        params.update({"w_factor": 0.30, "w_smoothness": 0.10})
    else:
        params.update({"w_factor": 0.05, "w_smoothness": 0.02})

    return config


def generate(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for recipe_name in RECIPE_NAMES:
        path = output_dir / f"cc_smart_{recipe_name}_seed{SEED}.yaml"
        path.write_text(
            yaml.safe_dump(_recipe(recipe_name), sort_keys=False),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def generate_smokes(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for recipe_name in RECIPE_NAMES:
        path = output_dir / f"cc_smart_{recipe_name}_seed{SEED}.yaml"
        path.write_text(
            yaml.safe_dump(
                _derive_smoke(_recipe(recipe_name), steps=SMOKE_STEPS),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
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
        help="Generate three-episode BC + PPO smokes instead of annual configs.",
    )
    args = parser.parse_args()
    writer = generate_smokes if args.smoke else generate
    for path in writer(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
