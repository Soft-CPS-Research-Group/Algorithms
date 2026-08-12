#!/usr/bin/env python3
"""Generate autonomous CC-L2 training over the frozen PPO seed 789.

Unlike the rejected causal-guard campaign, this protocol does not receive,
copy, or refine a Level-1 signal.  The coordinator starts at the exact neutral
PPO vector and learns seventeen contextual prices directly.  Its first episode
is an action-independent PPO-neutral control variate; later training rewards
subtract the matching neutral transition.  Deterministic validation episodes
select a policy, and the final annual replay restores only the best policy that
beat the neutral validation objective.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from scripts.build_cc_ppo_online_v6 import NEUTRAL_REFERENCES
    from scripts.generate_cc_level2_ppo_frozen_v2 import (
        NUM_BUILDINGS,
        PPO_SEED,
        REPO_ROOT,
        cc_recipe as build_v2_recipe,
        signal_path_neutral_recipe,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    from build_cc_ppo_online_v6 import NEUTRAL_REFERENCES
    from generate_cc_level2_ppo_frozen_v2 import (
        NUM_BUILDINGS,
        PPO_SEED,
        REPO_ROOT,
        cc_recipe as build_v2_recipe,
        signal_path_neutral_recipe,
    )


EXPERIMENT_NAME = "cc_level2_ppo_autonomous_v5"
OUTPUT_DIR = REPO_ROOT / "configs" / "experiments" / EXPERIMENT_NAME
ANNUAL_STEPS = 35040
ANNUAL_EPISODES = 12

VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "cost_first_seed123": {
        "seed": 123,
        "price_min": 0.60,
        "policy_deadband": 0.02,
        "initial_log_std": -1.0,
        "lr": 5.0e-5,
        "team_reward_mix": 0.10,
        "signal_price_charge_rate": 0.60,
        "w_peak": 0.02,
        "w_ramp": 0.005,
        "w_export": 0.005,
        "w_smoothness": 0.001,
    },
    "balanced_seed456": {
        "seed": 456,
        "price_min": 0.70,
        "policy_deadband": 0.03,
        "initial_log_std": -1.15,
        "lr": 3.0e-5,
        "team_reward_mix": 0.25,
        "signal_price_charge_rate": 0.50,
        "w_peak": 0.08,
        "w_ramp": 0.05,
        "w_export": 0.01,
        "w_smoothness": 0.004,
    },
    "scorecard_seed789": {
        "seed": 789,
        "price_min": 0.78,
        "policy_deadband": 0.05,
        "initial_log_std": -1.25,
        "lr": 2.0e-5,
        "team_reward_mix": 0.35,
        "signal_price_charge_rate": 0.45,
        "w_peak": 0.15,
        "w_ramp": 0.10,
        "w_export": 0.02,
        "w_smoothness": 0.008,
    },
}


def _common_tracking(config: dict[str, Any], *, recipe: str) -> None:
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": recipe,
            "controller": "autonomous_cc_level2_over_frozen_ppo",
            "cc_level": "2",
            "settlement": "enabled",
            "ppo_seed": str(PPO_SEED),
            "leaf_frozen": "True",
            "leaf_community_blind": "True",
            "joint_training": "False",
            "uses_cc_level1_signal": "False",
            "v2g_enabled": "False",
            "ppo_actor_price_conditioning": "current_only",
            "ppo_price_forecast_conditioning": "real_unmodified",
            "cc_price_scope": "ppo_current_and_local_residual_base",
            "neutral_control_variate": "True",
            "repeat_episode_scenario": "True",
            "best_policy_validation": "True",
            "promotion_requires_paired_neutral_replay": "True",
            "promotion_eligible": "False",
        }
    )
    config["tracking"].update(
        {
            "progress_update_interval": 512,
            "progress_phase_updates_enabled": True,
            "stall_watchdog_timeout_seconds": 900.0,
            "stall_watchdog_context_interval_steps": 64,
        }
    )


def build_paired_neutral_config(
    *,
    episodes: int = ANNUAL_EPISODES,
    pilot_steps: int | None = None,
) -> dict[str, Any]:
    """Exact neutral signal path at the candidate evaluation episode."""

    config = copy.deepcopy(signal_path_neutral_recipe())
    horizon = ANNUAL_STEPS if pilot_steps is None else int(pilot_steps)
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"PPO neutral matched to autonomous CC-L2 episode {episodes}",
            "description": (
                "Frozen PPO signal-path reference with the same leaf contract, "
                "episode realization, settlement and horizon as autonomous CC-L2."
            ),
        }
    )
    _common_tracking(config, recipe="paired_neutral")
    config["tracking"]["tags"].update(
        {
            "controller": "ppo_neutral_signal_path",
            "neutral_control_variate": "False",
            "best_policy_validation": "False",
            "evaluation_episode_index": str(episodes),
            "training_episodes": "0",
            "validation_episodes": "0",
            "evidence": (
                "full_year" if pilot_steps is None else "functional_smoke"
            ),
        }
    )
    config["simulator"].update(
        {
            "episodes": int(episodes),
            "deterministic_finish": True,
            "repeat_episode_scenario": True,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": horizon - 1,
            "episode_time_steps": horizon,
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-paired-neutral-ep{episodes}"
        + ("-annual" if pilot_steps is None else f"-pilot{horizon}")
    )
    leaf_params = config["pipeline"][1]["exploration"]["params"]
    leaf_params.update(
        {
            "local_price_conditioning_enabled": True,
            "local_price_forecast_mode": "real_unmodified",
            "residual_base_price_conditioning_enabled": True,
            "residual_ev_action_scale_multiplier": 0.0,
        }
    )
    leaf_params["residual_base_policy_hyperparameters"].update(
        {
            "allow_v2g": False,
            "signal_price_charge_rate": 0.60,
            "signal_price_response_mode": "linear_discount",
            "signal_price_charge_reference_multiplier": 0.90,
            "signal_price_charge_gain_max": 1.5,
        }
    )
    config["checkpointing"]["checkpoint_interval"] = None
    return config


def build_config(
    name: str,
    *,
    pilot_steps: int | None = None,
) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"Unknown autonomous CC-L2 recipe: {name}")
    variant = VARIANTS[name]
    horizon = ANNUAL_STEPS if pilot_steps is None else int(pilot_steps)
    episodes = ANNUAL_EPISODES if pilot_steps is None else 6

    # Reuse only the audited frozen-PPO/settlement contract. The V2 manager is
    # fully replaced below; no L1 policy or L1 action enters this pipeline.
    config = copy.deepcopy(build_v2_recipe("cost_seed123"))
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"Autonomous CC-L2 PPO {name}",
            "description": (
                "Seventeen-price coordinator learned directly from neutral PPO. "
                "No Level-1 signal is observed, copied or used as an incumbent."
            ),
        }
    )
    _common_tracking(config, recipe=name)
    config["tracking"]["tags"].update(
        {
            "cc_seed": str(variant["seed"]),
            "price_range": f"{variant['price_min']:.2f}_1.00",
            "cc_action_interval": "4",
            "total_episodes": str(episodes),
            "evaluation_episode_index": str(episodes),
            "training_episodes": "2" if pilot_steps is not None else "6",
            "validation_episodes": "1" if pilot_steps is not None else "3",
            "evidence": (
                "full_year" if pilot_steps is None else "functional_smoke"
            ),
        }
    )
    config["training"]["seed"] = int(variant["seed"])
    config["simulator"].update(
        {
            "episodes": episodes,
            "deterministic_finish": True,
            "repeat_episode_scenario": True,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": horizon - 1,
            "episode_time_steps": horizon,
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
                # Member decomposition divides this term by 17 so the sum is
                # the community-average service penalty. 8.5 gives an urgent
                # member a local coefficient of 0.5, unlike V4's ineffective
                # 0.5 / 17 coefficient.
                "w_ev": 8.5,
                "urgency_horizon": 4.0,
                "target_import": NEUTRAL_REFERENCES["target_import"],
                "reference_cost": NEUTRAL_REFERENCES["reference_cost"],
                "reference_peak": NEUTRAL_REFERENCES["reference_peak"],
                "reference_ramping": NEUTRAL_REFERENCES["reference_ramping"],
                "reference_export": NEUTRAL_REFERENCES["reference_export"],
                "reference_violation": 1.0,
                "credit_assignment": "member_decomposed",
                "ramp_credit_allocation": "causal_net",
            },
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{name}"
        + ("-annual" if pilot_steps is None else f"-pilot{horizon}")
    )

    manager = config["pipeline"][0]
    manager["algorithm"] = "CCLevel2"
    manager["count"] = 1
    manager["frozen"] = False
    manager["hyperparameters"] = {
        "c_dim": 121,
        "num_buildings": NUM_BUILDINGS,
        "hidden_dims": [192, 192],
        "price_min": float(variant["price_min"]),
        "price_max": 1.0,
        "reference_multipliers": [1.0] * NUM_BUILDINGS,
        "policy_residual_scale": 1.0,
        "policy_parameterization": "sparse_centered_residual",
        "policy_deadband": float(variant["policy_deadband"]),
        "include_community_headroom": True,
        "include_community_history": True,
        "separate_value_encoder": True,
        "cc_action_interval": 4,
        "num_steps": 336 if pilot_steps is None else 96,
        "lr": float(variant["lr"]),
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "num_epochs": 4 if pilot_steps is None else 1,
        "mini_batch_size": 84 if pilot_steps is None else 48,
        "clip_coef": 0.10,
        "vf_coef": 0.25,
        "ent_coef": 0.0,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
        "initial_log_std": float(variant["initial_log_std"]),
        "train_log_std": False,
        "credit_assignment": "member_decomposed",
        "team_reward_mix": float(variant["team_reward_mix"]),
        "reward_normalization": "none",
        "w_factor": 0.0,
        "w_smoothness": float(variant["w_smoothness"]),
        "bc_pretrain_enabled": False,
        "neutral_baseline_enabled": True,
        "neutral_warmup_episodes": 1,
        "counterfactual_baseline_weight": 1.0,
        "training_episodes_per_validation": 2,
        "rollback_rejected_validation": True,
        "restore_best_policy_for_deterministic": True,
        "best_policy_min_improvement": 0.0,
    }

    leaf = config["pipeline"][1]
    leaf["frozen"] = True
    leaf_params = leaf["exploration"]["params"]
    leaf_params.update(
        {
            "local_price_conditioning_enabled": True,
            "local_price_forecast_mode": "real_unmodified",
            "residual_base_price_conditioning_enabled": True,
            "actor_policy_loss_weight": 0.0,
            "actor_behavior_cloning_weight": 0.0,
            "residual_ev_action_scale_multiplier": 0.0,
        }
    )
    leaf_params["residual_base_policy_hyperparameters"].update(
        {
            "allow_v2g": False,
            "signal_price_charge_rate": float(
                variant["signal_price_charge_rate"]
            ),
            "signal_price_response_mode": "linear_discount",
            "signal_price_charge_reference_multiplier": 0.90,
            "signal_price_charge_gain_max": 1.5,
        }
    )
    config["checkpointing"].update(
        {
            "checkpoint_interval": None,
            "fine_tune": False,
            "restore_optimizers": False,
            "restore_replay_buffer": False,
            "restore_exploration_state": False,
            "restore_reward_normalizer": False,
            "reset_replay_buffer": True,
        }
    )
    return config


def generate(
    output_dir: Path = OUTPUT_DIR,
    *,
    pilot_steps: int | None = None,
) -> list[Path]:
    if pilot_steps is not None and (
        int(pilot_steps) < 384 or int(pilot_steps) % 4 != 0
    ):
        raise ValueError("pilot_steps must be a multiple of 4 and at least 384")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if pilot_steps is None else f"_pilot{pilot_steps}"
    episodes = ANNUAL_EPISODES if pilot_steps is None else 6
    configs = {
        "paired_neutral": build_paired_neutral_config(
            episodes=episodes,
            pilot_steps=pilot_steps,
        ),
        **{
            name: build_config(name, pilot_steps=pilot_steps)
            for name in VARIANTS
        },
    }
    paths: list[Path] = []
    for name, config in configs.items():
        path = output_dir / f"cc_l2_autonomous_{name}{suffix}.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pilot-steps", type=int)
    args = parser.parse_args()
    for path in generate(args.output_dir, pilot_steps=args.pilot_steps):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
