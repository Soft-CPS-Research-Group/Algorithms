#!/usr/bin/env python3
"""Generate the controlled MATD3 V3 improvement campaign.

The four candidates share one annual evaluation surface and differ by one
progressive hypothesis: SMART teacher, cooperative credit, wider storage
control, and a more aggressive cost-first policy.  This keeps a failed run
informative instead of changing every training lever at once.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    REPO_ROOT
    / "configs/experiments/matd3_settlement_annual_recovery_v2"
    / "matd3_settled_recoverable_seed789.yaml"
)
SMART_CONFIG = (
    REPO_ROOT
    / "configs/experiments/ppo_cc_settlement_annual_v1"
    / "smart_settlement_annual.yaml"
)
OUTPUT_DIR = REPO_ROOT / "configs/experiments/matd3_settlement_annual_v3"
EXPERIMENT_NAME = "matd3_settlement_annual_v3"


VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "smart_anchor": {
        "description": "Exact current SMART teacher; individual critic rewards retained.",
        "critic_team_reward_mix": 0.0,
    },
    "cooperative_team70": {
        "description": "SMART teacher plus 70% community / 30% local critic credit.",
        "critic_team_reward_mix": 0.70,
    },
    "cooperative_storage_open": {
        "description": "Cooperative critic with wider, lightly regularized battery residuals.",
        "critic_team_reward_mix": 0.70,
        "reward": {"battery_throughput_penalty": 0.002},
        "exploration": {
            "storage_exploration_noise_multiplier": 0.50,
            "actor_policy_loss_weight": 0.10,
            "actor_policy_loss_warmup_weight": 0.012,
            "actor_storage_action_l2_penalty": 0.002,
            "actor_storage_behavior_cloning_multiplier": 0.12,
            "actor_residual_delta_l2_penalty": 0.025,
            "actor_storage_smoothness_l2_penalty": 0.001,
            "residual_action_final_scale": 0.32,
            "residual_action_growth_steps": 35040,
            "residual_storage_action_scale_multiplier": 0.80,
            "residual_ev_action_scale_multiplier": 0.25,
        },
    },
    "cooperative_scorecard": {
        "description": "Cooperative battery policy with stronger peak and action-smoothness guards.",
        "critic_team_reward_mix": 0.70,
        "reward": {
            "battery_throughput_penalty": 0.001,
            "community_peak_import_penalty": 0.0024,
        },
        "exploration": {
            "storage_exploration_noise_multiplier": 0.50,
            "actor_policy_loss_weight": 0.10,
            "actor_policy_loss_warmup_weight": 0.012,
            "actor_storage_action_l2_penalty": 0.0015,
            "actor_storage_behavior_cloning_multiplier": 0.12,
            "actor_residual_delta_l2_penalty": 0.022,
            "actor_storage_smoothness_l2_penalty": 0.004,
            "residual_action_final_scale": 0.32,
            "residual_action_growth_steps": 35040,
            "residual_storage_action_scale_multiplier": 0.80,
            "residual_ev_action_scale_multiplier": 0.25,
        },
    },
    "cooperative_cost_first": {
        "description": "High-cooperation cost-first scout with broad battery authority.",
        "critic_team_reward_mix": 0.85,
        "reward": {"battery_throughput_penalty": 0.0005},
        "exploration": {
            "sigma": 0.10,
            "storage_exploration_noise_multiplier": 0.70,
            "actor_policy_loss_weight": 0.14,
            "actor_policy_loss_warmup_weight": 0.016,
            "actor_storage_action_l2_penalty": 0.0005,
            "actor_storage_behavior_cloning_multiplier": 0.08,
            "actor_behavior_cloning_min_weight": 0.18,
            "actor_residual_delta_l2_penalty": 0.012,
            "actor_storage_smoothness_l2_penalty": 0.0005,
            "residual_action_final_scale": 0.40,
            "residual_action_growth_steps": 35040,
            "residual_storage_action_scale_multiplier": 0.90,
            "residual_ev_action_scale_multiplier": 0.22,
        },
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _smart_teacher_hyperparameters() -> dict[str, Any]:
    smart = _load_yaml(SMART_CONFIG)
    stages = smart.get("pipeline") or []
    leaf = next(stage for stage in stages if stage.get("algorithm") == "SignalAwareRBC")
    return copy.deepcopy(leaf.get("hyperparameters") or {})


def build_config(
    *,
    variant_name: str,
    seed: int = 789,
    smoke_steps: int | None = None,
) -> dict[str, Any]:
    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown MATD3 V3 variant: {variant_name}")

    variant = VARIANTS[variant_name]
    config = copy.deepcopy(_load_yaml(BASE_CONFIG))
    params = config["pipeline"][0]["exploration"]["params"]
    reward = config["simulator"]["reward_function_kwargs"]

    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"MATD3 V3 {variant_name} seed {seed}",
            "description": str(variant["description"]),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": variant_name,
            "seed": str(seed),
            "teacher_policy": "RBCSmartPolicy",
            "forecast_encoding": "dataset_scaled_v1",
            "critic_team_reward_mix": str(variant["critic_team_reward_mix"]),
            "training_years": "2",
            "deterministic_evaluation_years": "1",
        }
    )
    config["training"]["seed"] = seed
    config["simulator"].update(
        {
            "episodes": 3,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": 35039,
            "episode_time_steps": 35040,
            "deterministic_finish": True,
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{variant_name}-seed{seed}"
    )
    config["simulator"]["entity_encoding"]["profile"] = "maddpg_v4_operational"
    config["checkpointing"].update(
        {
            "checkpoint_mode": "inference",
            "checkpoint_interval": 35040,
            "require_update_step": True,
            "require_initial_exploration_done": True,
        }
    )

    params["warm_start_policy"] = "RBCSmartPolicy"
    params["warm_start_policy_hyperparameters"] = _smart_teacher_hyperparameters()
    params["critic_team_reward_mix"] = float(variant["critic_team_reward_mix"])
    params.update(copy.deepcopy(variant.get("exploration") or {}))
    reward.update(copy.deepcopy(variant.get("reward") or {}))

    if smoke_steps is not None:
        if smoke_steps < 512:
            raise ValueError("MATD3 V3 smoke_steps must be at least 512.")
        config["metadata"]["run_name"] += f" smoke {smoke_steps}"
        config["tracking"]["tags"].update(
            {
                "evidence": "functional_smoke",
                "training_years": "0",
                "deterministic_evaluation_years": "0",
            }
        )
        config["simulator"].update(
            {
                "episodes": 1,
                "simulation_end_time_step": smoke_steps - 1,
                "episode_time_steps": smoke_steps,
            }
        )
        config["simulator"]["export"]["session_name"] += f"-smoke{smoke_steps}"
        config["checkpointing"]["checkpoint_interval"] = None

    return config


def generate(
    *,
    output_dir: Path = OUTPUT_DIR,
    seed: int = 789,
    smoke_steps: int | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    smoke_token = "" if smoke_steps is None else f"_smoke{smoke_steps}"
    for variant_name in VARIANTS:
        output = output_dir / f"matd3_v3_{variant_name}_seed{seed}{smoke_token}.yaml"
        output.write_text(
            yaml.safe_dump(
                build_config(
                    variant_name=variant_name,
                    seed=seed,
                    smoke_steps=smoke_steps,
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    for output in generate(
        output_dir=args.output_dir,
        seed=args.seed,
        smoke_steps=args.smoke_steps,
    ):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
