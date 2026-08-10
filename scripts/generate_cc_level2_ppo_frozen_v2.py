#!/usr/bin/env python3
"""Generate the conservative CC-L2 campaign over the frozen PPO seed 789.

The rejected V1 campaign fine-tuned every local PPO from the first transition
using one duplicated community reward. It therefore changed both layers at
once and destroyed the strong local battery policy. V2 keeps the audited PPO
immutable, validates the neutral signal path separately, then trains only a
bounded per-building coordinator.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.build_cc_ppo_online_v6 import NEUTRAL_REFERENCES
    from scripts.generate_cc_causal_price_control_v4 import ppo_fixed_recipe
    from scripts.generate_cc_level2_smart_settlement import (
        learned_recipe as smart_learned_recipe,
    )
    from scripts.generate_ppo_cc_settlement_templates import (
        PPO_SEED,
        REPO_ROOT,
        _ppo_configs,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from build_cc_ppo_online_v6 import NEUTRAL_REFERENCES
    from generate_cc_causal_price_control_v4 import ppo_fixed_recipe
    from generate_cc_level2_smart_settlement import (
        learned_recipe as smart_learned_recipe,
    )
    from generate_ppo_cc_settlement_templates import PPO_SEED, REPO_ROOT, _ppo_configs


EXPERIMENT_NAME = "cc_level2_ppo_frozen_v2"
NUM_BUILDINGS = 17
CC_RECIPES = {
    "cost_seed123": {
        "seed": 123,
        "w_peak": 0.03,
        "w_ramp": 0.02,
        "w_export": 0.01,
        "bc_mult_scale": 0.12,
    },
    "scorecard_seed456": {
        "seed": 456,
        "w_peak": 0.15,
        "w_ramp": 0.10,
        "w_export": 0.02,
        "bc_mult_scale": 0.08,
    },
}


def _common_tags(config: dict[str, Any], *, recipe: str) -> None:
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": recipe,
            "cc_level": "2",
            "settlement": "enabled",
            "leaf_frozen": "True",
            "leaf_community_blind": "True",
            "joint_training": "False",
            "v2g_enabled": "False",
            "v1_decision": "rejected_catastrophic_leaf_drift",
            "promotion_requires_paired_neutral_replay": "True",
        }
    )


def original_neutral_recipe() -> dict[str, Any]:
    """Replay the exact original PPO composite without a price-aware base."""

    ppo, _ = _ppo_configs()
    config = copy.deepcopy(ppo)
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-L2 V2 gate A original PPO neutral seed {PPO_SEED}",
            "description": (
                "Gate A: exact annual replay of the original frozen PPO composite. "
                "This is the immutable performance floor for every CC-L2 candidate."
            ),
        }
    )
    _common_tags(config, recipe="gate_a_original_ppo_neutral")
    config["tracking"]["tags"].update(
        {
            "controller": "ppo_neutral_reference",
            "cc_price_scope": "none",
            "promotion_eligible": "False",
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-gate-a-original-neutral-seed{PPO_SEED}"
    )
    return config


def signal_path_neutral_recipe() -> dict[str, Any]:
    """Validate that the controllable signal path is neutral at multiplier 1."""

    config = copy.deepcopy(ppo_fixed_recipe(1.0))
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-L2 V2 gate B signal-path neutral seed {PPO_SEED}",
            "description": (
                "Gate B: neutral multiplier replay over the frozen PPO with the "
                "strict-local signal-aware residual base. It must reproduce Gate A "
                "before learned per-building prices are interpreted."
            ),
        }
    )
    _common_tags(config, recipe="gate_b_signal_path_neutral")
    config["tracking"]["tags"].update(
        {
            "controller": "ppo_neutral_signal_path",
            "cc_price_scope": "strict_local_residual_base_only",
            "promotion_eligible": "False",
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-gate-b-signal-neutral-seed{PPO_SEED}"
    )
    leaf_params = config["pipeline"][1]["exploration"]["params"]
    leaf_params["residual_ev_action_scale_multiplier"] = 0.0
    leaf_params["residual_base_policy_hyperparameters"]["allow_v2g"] = False
    return config


def cc_recipe(name: str) -> dict[str, Any]:
    if name not in CC_RECIPES:
        raise ValueError(f"Unknown frozen CC-L2 PPO V2 recipe: {name}")

    variant = CC_RECIPES[name]
    seed = int(variant["seed"])
    config = copy.deepcopy(signal_path_neutral_recipe())
    manager = copy.deepcopy(smart_learned_recipe()["pipeline"][0])

    config["metadata"].update(
        {
            "run_name": f"CC-L2 V2 frozen PPO {name}",
            "description": (
                "Per-building coordinator trained against exact community settlement "
                "while all seventeen PPO leaves remain deterministic and immutable. "
                "The coordinator may only discount within [0.90, 1.00] and acts hourly."
            ),
        }
    )
    _common_tags(config, recipe=name)
    config["tracking"]["tags"].update(
        {
            "controller": "trainable_cc_level2_over_frozen_ppo",
            "cc_seed": str(seed),
            "cc_price_scope": "strict_local_residual_base_only",
            "price_range": "0.90_1.00",
            "cc_action_interval": "4",
            "training_episodes": "10",
            "promotion_eligible": "False",
        }
    )
    config["training"]["seed"] = seed
    config["simulator"].update(
        {
            "reward_function": "CCRewardLevel2",
            "reward_function_kwargs": {
                "cost_aggregation": "community_settled",
                "community_local_price_ratio": 0.8,
                "community_grid_export_price": 0.0,
                "w_cost": 1.0,
                "w_peak": float(variant["w_peak"]),
                "w_ramp": float(variant["w_ramp"]),
                "w_export": float(variant["w_export"]),
                "w_violation": 2.0,
                "w_ev": 0.5,
                "urgency_horizon": 4.0,
                "target_import": NEUTRAL_REFERENCES["target_import"],
                "reference_cost": NEUTRAL_REFERENCES["reference_cost"],
                "reference_peak": NEUTRAL_REFERENCES["reference_peak"],
                "reference_ramping": NEUTRAL_REFERENCES["reference_ramping"],
                "reference_export": NEUTRAL_REFERENCES["reference_export"],
                "reference_violation": 1.0,
            },
            "episodes": 10,
            "deterministic_finish": True,
        }
    )
    config["simulator"]["export"]["session_name"] = f"{EXPERIMENT_NAME}-{name}"

    manager["frozen"] = False
    manager["hyperparameters"].update(
        {
            "c_dim": 119,
            "num_buildings": NUM_BUILDINGS,
            "hidden_dims": [128, 128],
            "price_min": 0.90,
            "price_max": 1.00,
            "reference_multipliers": [1.0] * NUM_BUILDINGS,
            "policy_residual_scale": 1.0,
            "policy_parameterization": "centered_residual",
            "include_community_headroom": True,
            "cc_action_interval": 4,
            "num_steps": 336,
            "lr": 2.5e-5,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "num_epochs": 4,
            "mini_batch_size": 84,
            "clip_coef": 0.10,
            "vf_coef": 0.25,
            "ent_coef": 0.001,
            "target_kl": 0.01,
            "initial_log_std": -3.0,
            "w_factor": 0.002,
            "w_smoothness": 0.005,
            "bc_pretrain_enabled": True,
            "bc_use_physical_teacher_context": True,
            "bc_collect_steps": 8760,
            "bc_train_steps": 4000,
            "bc_lr": 5.0e-4,
            "bc_w_cost": 1.0,
            "bc_w_peak": float(variant["w_peak"]),
            "bc_w_export": float(variant["w_export"]),
            "bc_w_soc": 0.05,
            "bc_w_net": 0.05,
            "bc_w_ev": 0.5,
            "bc_mult_scale": float(variant["bc_mult_scale"]),
            "bc_target_import": NEUTRAL_REFERENCES["target_import"],
            "bc_reference_peak": NEUTRAL_REFERENCES["reference_peak"],
            "bc_reference_export": NEUTRAL_REFERENCES["reference_export"],
        }
    )

    leaf = config["pipeline"][1]
    leaf["frozen"] = True
    leaf_params = leaf["exploration"]["params"]
    leaf_params.update(
        {
            "local_price_conditioning_enabled": False,
            "local_price_forecast_mode": "real_unmodified",
            "actor_policy_loss_weight": 0.0,
            "actor_behavior_cloning_weight": 0.0,
            "residual_ev_action_scale_multiplier": 0.0,
        }
    )
    leaf_params["residual_base_policy_hyperparameters"].update(
        {"signal_price_charge_rate": 0.60, "allow_v2g": False}
    )
    config["pipeline"] = [manager, leaf]
    config["checkpointing"].update(
        {
            "fine_tune": False,
            "restore_optimizers": False,
            "restore_replay_buffer": False,
            "restore_exploration_state": False,
            "restore_reward_normalizer": False,
            "reset_replay_buffer": True,
            "checkpoint_interval": 35040,
        }
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
            "simulation_start_time_step": 0,
            "simulation_end_time_step": transitions,
            "episode_time_steps": transitions + 1,
        }
    )
    smoke["simulator"]["export"]["session_name"] += "-smoke"
    if smoke["pipeline"][0]["algorithm"] == "CCLevel2":
        smoke["simulator"]["episodes"] = 3
        manager = smoke["pipeline"][0]["hyperparameters"]
        manager.update(
            {
                "num_steps": 96,
                "mini_batch_size": 48,
                "bc_collect_steps": 96,
                "bc_train_steps": 4,
                "num_epochs": 1,
            }
        )
    else:
        smoke["simulator"]["episodes"] = 1
    smoke["checkpointing"]["checkpoint_interval"] = None
    return smoke


def generate(output_dir: Path, *, smoke: bool = False) -> list[Path]:
    configs = {
        "gate_a_original_ppo_neutral": original_neutral_recipe(),
        "gate_b_signal_path_neutral": signal_path_neutral_recipe(),
        **{name: cc_recipe(name) for name in CC_RECIPES},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, config in configs.items():
        if smoke:
            config = derive_smoke(config)
        filename = (
            f"cc_l2_ppo_{name}_seed{config['training']['seed']}.yaml"
            if name.startswith("gate_")
            else f"cc_l2_ppo_{name}.yaml"
        )
        path = output_dir / filename
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
