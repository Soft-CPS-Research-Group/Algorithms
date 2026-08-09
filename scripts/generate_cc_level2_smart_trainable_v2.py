#!/usr/bin/env python3
"""Generate the trainable, settlement-aligned CC-L2 over SMART campaign V2.

V1 accidentally initialized its absolute tanh policy at the upper price bound,
where the actor gradient was effectively saturated.  V2 uses the centered
residual parameterization: raw action zero is exactly the measured 1.30
incumbent, while negative actions retain a useful gradient toward 0.90.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.generate_cc_level2_smart_settlement import learned_recipe
    from scripts.generate_cc_smart_cost_focus_v2 import ANNUAL_SMART_REFERENCES
    from scripts.generate_ppo_cc_settlement_templates import REPO_ROOT
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from generate_cc_level2_smart_settlement import learned_recipe
    from generate_cc_smart_cost_focus_v2 import ANNUAL_SMART_REFERENCES
    from generate_ppo_cc_settlement_templates import REPO_ROOT


EXPERIMENT_NAME = "cc_level2_smart_trainable_v2"
RECIPES = {
    "cost_bc_seed123": {"seed": 123, "w_peak": 0.0, "w_export": 0.0},
    "cost_bc_seed456": {"seed": 456, "w_peak": 0.0, "w_export": 0.0},
    "scorecard_bc_seed123": {"seed": 123, "w_peak": 0.05, "w_export": 0.01},
}


def recipe(name: str) -> dict[str, Any]:
    if name not in RECIPES:
        raise ValueError(f"Unknown CC-L2 SMART V2 recipe: {name}")
    variant = RECIPES[name]
    seed = int(variant["seed"])
    config = copy.deepcopy(learned_recipe())
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-L2 SMART trainable V2 {name}",
            "description": (
                "Settlement-aligned per-building CC over frozen SMART. Uses a "
                "centered residual policy to preserve gradient at the measured "
                "1.30 incumbent and a causal BC warm start before PPO updates."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": name,
            "cc_seed": str(seed),
            "policy_parameterization": "centered_residual",
            "v1_failure_mode": "absolute_tanh_saturated_at_upper_bound",
            "reference_source": "matched_fixed_1p3_incumbent",
            "promotion_eligible": "True",
        }
    )
    config["training"]["seed"] = seed
    config["simulator"]["episodes"] = 10
    config["simulator"]["reward_function_kwargs"] = {
        "cost_aggregation": "community_settled",
        "w_cost": 1.0,
        "w_peak": float(variant["w_peak"]),
        "w_export": float(variant["w_export"]),
        "w_ev": 0.5,
        "urgency_horizon": 4.0,
        "target_import": ANNUAL_SMART_REFERENCES["target_import"],
        "reference_cost": ANNUAL_SMART_REFERENCES[
            "reference_member_retail_cost"
        ],
        "reference_peak": ANNUAL_SMART_REFERENCES["reference_peak"],
        "reference_export": ANNUAL_SMART_REFERENCES["reference_export"],
    }
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{name}"
    )

    manager = config["pipeline"][0]["hyperparameters"]
    manager.update(
        {
            "c_dim": 119,
            "hidden_dims": [128, 128],
            "price_min": 0.5,
            "price_max": 1.3,
            "reference_multipliers": [1.3] * 17,
            "policy_residual_scale": 0.5,
            "policy_parameterization": "centered_residual",
            "include_community_headroom": True,
            "cc_action_interval": 1,
            "num_steps": 672,
            "lr": 5.0e-5,
            "gamma": 0.99875,
            "gae_lambda": 0.95,
            "num_epochs": 4,
            "mini_batch_size": 96,
            "clip_coef": 0.15,
            "vf_coef": 0.25,
            "ent_coef": 0.002,
            "target_kl": 0.03,
            "initial_log_std": -2.5,
            "w_factor": 0.005,
            "w_smoothness": 0.002,
            "bc_pretrain_enabled": True,
            "bc_use_physical_teacher_context": True,
            "bc_collect_steps": 8760,
            "bc_train_steps": 4000,
            "bc_lr": 0.001,
            "bc_w_cost": 1.0,
            "bc_w_peak": float(variant["w_peak"]),
            "bc_w_export": float(variant["w_export"]),
            "bc_w_ev": 0.5,
            "bc_mult_scale": 1.0,
            "bc_target_import": ANNUAL_SMART_REFERENCES["target_import"],
            "bc_reference_peak": ANNUAL_SMART_REFERENCES["reference_peak"],
            "bc_reference_export": ANNUAL_SMART_REFERENCES[
                "reference_export"
            ],
        }
    )
    return config


def derive_smoke(config: dict[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(config)
    transitions = 672
    smoke["metadata"]["run_name"] += " [functional smoke]"
    smoke["tracking"]["tags"].update(
        {"evidence": "functional_smoke", "promotion_eligible": "False"}
    )
    smoke["simulator"].update(
        {
            "episodes": 3,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": transitions,
            "episode_time_steps": transitions + 1,
        }
    )
    smoke["simulator"]["export"]["session_name"] += "-smoke"
    manager = smoke["pipeline"][0]["hyperparameters"]
    manager.update(
        {
            "num_steps": 168,
            "bc_collect_steps": 168,
            "bc_train_steps": 2,
            "num_epochs": 1,
        }
    )
    smoke["checkpointing"]["checkpoint_interval"] = None
    return smoke


def generate(output_dir: Path, *, smoke: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in RECIPES:
        config = recipe(name)
        if smoke:
            config = derive_smoke(config)
        path = output_dir / f"cc_l2_smart_{name}.yaml"
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
