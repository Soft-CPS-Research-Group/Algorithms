#!/usr/bin/env python3
"""Generate joint price-conditioned PPO + learned CC-L2 candidates.

The frozen seed-789 PPO was trained only at the native tariff.  Applying a
learned vector of virtual prices directly is therefore out-of-distribution.
These recipes restore that checkpoint, make price conditioning trainable, use
CC-L2 behaviour cloning as a causal warm-up, and then update both layers under
the exact settled community objective.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.generate_cc_causal_price_control_v4 import ppo_fixed_recipe
    from scripts.generate_cc_level2_smart_settlement import learned_recipe
    from scripts.generate_cc_smart_cost_focus_v2 import ANNUAL_SMART_REFERENCES
    from scripts.generate_ppo_cc_settlement_templates import PPO_SEED, REPO_ROOT
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from generate_cc_causal_price_control_v4 import ppo_fixed_recipe
    from generate_cc_level2_smart_settlement import learned_recipe
    from generate_cc_smart_cost_focus_v2 import ANNUAL_SMART_REFERENCES
    from generate_ppo_cc_settlement_templates import PPO_SEED, REPO_ROOT


EXPERIMENT_NAME = "cc_level2_ppo_joint_v1"
RECIPES = {
    "current_storage": {
        "forecast_mode": "real_unmodified",
        "allow_v2g": False,
        "ev_residual_scale": 0.0,
    },
    "forecasts_storage": {
        "forecast_mode": "persist_current",
        "allow_v2g": False,
        "ev_residual_scale": 0.0,
    },
    "forecasts_v2g": {
        "forecast_mode": "persist_current",
        "allow_v2g": True,
        "ev_residual_scale": 0.15,
    },
}


def recipe(name: str) -> dict[str, Any]:
    if name not in RECIPES:
        raise ValueError(f"Unknown CC-L2 PPO recipe: {name}")
    variant = RECIPES[name]
    config = copy.deepcopy(ppo_fixed_recipe(1.0))
    manager = copy.deepcopy(learned_recipe()["pipeline"][0])

    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-L2 joint price-conditioned PPO {name} seed {PPO_SEED}",
            "description": (
                "Joint causal training of a per-building price coordinator and "
                "the restored seed-789 local PPO leaves. The CC receives community "
                "state; every PPO remains building-local and receives community "
                "information only through its effective price."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "controller": "cc_level2_joint_ppo",
            "recipe": name,
            "cc_level": "2",
            "leaf_frozen": "False",
            "leaf_community_blind": "True",
            "joint_training": "True",
            "ppo_price_conditioning_trainable": "True",
            "ppo_price_forecast_mode": str(variant["forecast_mode"]),
            "v2g_enabled": str(bool(variant["allow_v2g"])),
            "settlement": "enabled",
            "promotion_requires_paired_neutral_replay": "True",
            "cc_multiplier_policy": "learned_per_building",
            "cc_price_scope": "ppo_actor_and_residual_base",
            "ppo_actor_price_conditioning": "True",
        }
    )
    # ``ppo_fixed_recipe`` contributes tags describing its fixed, frozen
    # predecessor.  They are false for this jointly trained CC-L2 campaign and
    # must not survive into MLflow or the archived experiment record.
    config["tracking"]["tags"].pop("fixed_multiplier", None)
    config["checkpointing"].update(
        {
            "fine_tune": True,
            "restore_optimizers": False,
            "restore_replay_buffer": False,
            "restore_exploration_state": False,
            "restore_reward_normalizer": False,
            "reset_replay_buffer": True,
            "checkpoint_interval": 35040,
        }
    )
    config["simulator"].update(
        {
            "reward_function": "CCRewardLevel2",
            "reward_function_kwargs": {
                "cost_aggregation": "community_settled",
                "w_cost": 1.0,
                "w_peak": 0.05,
                "w_export": 0.01,
                "w_ev": 0.75 if variant["allow_v2g"] else 0.5,
                "urgency_horizon": 4.0,
                "target_import": ANNUAL_SMART_REFERENCES["target_import"],
                "reference_cost": ANNUAL_SMART_REFERENCES[
                    "reference_member_retail_cost"
                ],
                "reference_peak": ANNUAL_SMART_REFERENCES["reference_peak"],
                "reference_export": ANNUAL_SMART_REFERENCES["reference_export"],
            },
            "episodes": 6,
            "deterministic_finish": True,
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{name}-seed{PPO_SEED}"
    )

    manager["frozen"] = False
    manager_hyper = manager["hyperparameters"]
    manager_hyper.update(
        {
            "c_dim": 119,
            "hidden_dims": [128, 128],
            "price_min": 0.5,
            "price_max": 1.3,
            "reference_multipliers": [1.0] * 17,
            "policy_residual_scale": 1.0,
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
            "bc_w_peak": 0.05,
            "bc_w_export": 0.01,
            "bc_w_ev": 0.75 if variant["allow_v2g"] else 0.5,
            "bc_mult_scale": 1.0,
            "bc_target_import": ANNUAL_SMART_REFERENCES["target_import"],
            "bc_reference_peak": ANNUAL_SMART_REFERENCES["reference_peak"],
            "bc_reference_export": ANNUAL_SMART_REFERENCES[
                "reference_export"
            ],
        }
    )

    leaf = config["pipeline"][1]
    leaf["frozen"] = False
    exploration = leaf["exploration"]["params"]
    exploration.update(
        {
            "local_price_conditioning_enabled": True,
            "local_price_conditioning_trainable": True,
            "local_price_forecast_mode": str(variant["forecast_mode"]),
            "residual_base_policy": "SignalAwareRBCSmartLocal",
            "residual_base_price_conditioning_enabled": True,
            # Fine-tune cautiously from the already strong frozen checkpoint.
            "actor_policy_loss_weight": 0.25,
            "actor_behavior_cloning_weight": 0.02,
            "actor_behavior_cloning_min_weight": 0.0,
            "actor_behavior_cloning_decay_start_step": 0,
            "actor_behavior_cloning_decay_steps": 70080,
            "residual_ev_action_scale_multiplier": float(
                variant["ev_residual_scale"]
            ),
            "actor_ev_v2g_action_l2_penalty": (
                0.001 if variant["allow_v2g"] else 0.0
            ),
        }
    )
    exploration["residual_base_policy_hyperparameters"].update(
        {
            "signal_price_charge_rate": 0.6,
            "allow_v2g": bool(variant["allow_v2g"]),
            "ev_v2g_discharge_rate": 0.18,
            "ev_v2g_reserve_soc": 0.15,
            "ev_v2g_min_departure_hours": 3.0,
            "ev_v2g_service_margin_soc": 0.05,
        }
    )
    config["pipeline"] = [manager, leaf]
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
            "num_steps": 96,
            "bc_collect_steps": 96,
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
        path = output_dir / f"cc_l2_ppo_joint_{name}_seed{PPO_SEED}.yaml"
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
