#!/usr/bin/env python3
"""Generate bidirectional, teacher-distilled CC-L2 over frozen PPO.

The evaluated coordinator is causal and receives no oracle schedule.  During
training only, one exact-neutral PPO year is labelled either by a causal
cheap/export teacher or by a fixed-service perfect-foresight battery schedule.
Those labels warm-start the CC price actor and can remain as a decaying BC
anchor while PPO optimises the settled community scorecard.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from scripts.generate_cc_level2_ppo_autonomous_v5 import (
        ANNUAL_EPISODES,
        ANNUAL_STEPS,
        NUM_BUILDINGS,
        REPO_ROOT,
        build_config as build_v5_config,
        build_paired_neutral_config as build_v5_neutral,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    from generate_cc_level2_ppo_autonomous_v5 import (
        ANNUAL_EPISODES,
        ANNUAL_STEPS,
        NUM_BUILDINGS,
        REPO_ROOT,
        build_config as build_v5_config,
        build_paired_neutral_config as build_v5_neutral,
    )


EXPERIMENT_NAME = "cc_level2_ppo_distilled_v6"
OUTPUT_DIR = REPO_ROOT / "configs" / "experiments" / EXPERIMENT_NAME
COST_TEACHER = (
    REPO_ROOT
    / "configs"
    / "demonstrations"
    / "community_fixed_service_battery_oracle_annual_v1.json.gz"
)
SCORECARD_TEACHER = (
    REPO_ROOT
    / "configs"
    / "demonstrations"
    / "community_fixed_service_battery_global_scorecard_teacher_annual_v5.json.gz"
)

PRICE_MIN = 0.70
PRICE_MAX = 1.30

VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "causal_teacher_cost_seed123": {
        "base": "cost_first_seed123",
        "teacher_mode": "cheap_and_export",
        "teacher_path": None,
        "bc_anchor_weight": 0.12,
        "bc_anchor_min_weight": 0.015,
        "signal_rate": 0.60,
    },
    "milp_cost_seed456": {
        "base": "balanced_seed456",
        "teacher_mode": "oracle_storage_schedule",
        "teacher_path": COST_TEACHER,
        "bc_anchor_weight": 0.18,
        "bc_anchor_min_weight": 0.025,
        "signal_rate": 0.60,
    },
    "milp_scorecard_seed789": {
        "base": "scorecard_seed789",
        "teacher_mode": "oracle_storage_schedule",
        "teacher_path": SCORECARD_TEACHER,
        "bc_anchor_weight": 0.22,
        "bc_anchor_min_weight": 0.030,
        "signal_rate": 0.55,
    },
}


def _apply_window(
    config: dict[str, Any],
    *,
    start_step: int,
    horizon: int,
) -> None:
    if start_step < 0:
        raise ValueError("CC-L2 V6 start_step must be non-negative")
    if horizon < 384 or horizon % 4 != 0:
        raise ValueError("CC-L2 V6 horizon must be a multiple of 4 and at least 384")
    config["simulator"].update(
        {
            "simulation_start_time_step": int(start_step),
            "simulation_end_time_step": int(start_step + horizon - 1),
            "episode_time_steps": int(horizon),
        }
    )


def _configure_bidirectional_leaf(config: dict[str, Any], *, rate: float) -> None:
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
            "signal_price_charge_rate": float(rate),
            "signal_price_discharge_rate": float(rate),
            "signal_price_response_mode": "linear_bidirectional",
            "signal_price_charge_reference_multiplier": 0.85,
            "signal_price_discharge_reference_multiplier": 1.15,
            "signal_price_charge_gain_max": 1.5,
            "signal_price_discharge_gain_max": 1.5,
        }
    )


def build_paired_neutral_config(
    *,
    episodes: int = ANNUAL_EPISODES,
    start_step: int = 0,
    horizon: int = ANNUAL_STEPS,
) -> dict[str, Any]:
    config = copy.deepcopy(
        build_v5_neutral(
            episodes=episodes,
            pilot_steps=None if horizon == ANNUAL_STEPS else horizon,
        )
    )
    _apply_window(config, start_step=start_step, horizon=horizon)
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"PPO neutral paired CC-L2 V6 episode {episodes}",
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": "paired_neutral",
            "price_range": f"{PRICE_MIN:.2f}_{PRICE_MAX:.2f}",
            "leaf_price_response": "linear_bidirectional",
            "training_teacher_access": "False",
            "evaluation_teacher_access": "False",
            "window_start": str(start_step),
            "window_steps": str(horizon),
        }
    )
    _configure_bidirectional_leaf(config, rate=0.60)
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-paired-neutral-start{start_step}-steps{horizon}"
        f"-ep{episodes}"
    )
    return config


def build_config(
    name: str,
    *,
    start_step: int = 0,
    horizon: int = ANNUAL_STEPS,
) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"Unknown CC-L2 V6 recipe: {name}")
    variant = VARIANTS[name]
    pilot = horizon != ANNUAL_STEPS
    config = copy.deepcopy(
        build_v5_config(
            str(variant["base"]),
            pilot_steps=horizon if pilot else None,
        )
    )
    _apply_window(config, start_step=start_step, horizon=horizon)
    episodes = 6 if pilot else ANNUAL_EPISODES
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"Bidirectional distilled CC-L2 {name}",
            "description": (
                "Frozen local PPO with a causal bidirectional price response. "
                "A training-only teacher labels an exact-neutral rollout; the "
                "final deterministic year has no teacher access."
            ),
        }
    )
    teacher_path = variant["teacher_path"]
    teacher_label = (
        "causal_cheap_export"
        if teacher_path is None
        else Path(teacher_path).stem.replace(".json", "")
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": name,
            "price_range": f"{PRICE_MIN:.2f}_{PRICE_MAX:.2f}",
            "leaf_price_response": "linear_bidirectional",
            "teacher": teacher_label,
            "teacher_collection": "neutral_label_only",
            "training_teacher_access": "True",
            "evaluation_teacher_access": "False",
            "teacher_perfect_foresight": str(teacher_path is not None),
            "window_start": str(start_step),
            "window_steps": str(horizon),
            "total_episodes": str(episodes),
            "evaluation_episode_index": str(episodes),
            "promotion_eligible": "False",
        }
    )
    manager = config["pipeline"][0]
    params = manager["hyperparameters"]
    params.update(
        {
            "price_min": PRICE_MIN,
            "price_max": PRICE_MAX,
            "reference_multipliers": [1.0] * NUM_BUILDINGS,
            "policy_parameterization": "sparse_centered_residual",
            "policy_deadband": 0.02,
            "initial_log_std": -1.35,
            "train_log_std": False,
            "bc_pretrain_enabled": True,
            "bc_collection_policy": "neutral_label_only",
            "bc_teacher_mode": str(variant["teacher_mode"]),
            "bc_collect_steps": horizon // 4,
            "bc_train_steps": 2000 if not pilot else 96,
            "bc_train_chunk_steps": 64 if not pilot else 16,
            "bc_progress_interval": 250 if not pilot else 24,
            "bc_lr": 5.0e-4,
            "bc_discount_multiplier": PRICE_MIN,
            "bc_oracle_schedule_path": (
                str(Path(teacher_path).relative_to(REPO_ROOT))
                if teacher_path is not None
                else None
            ),
            "bc_oracle_schedule_step_offset": int(start_step),
            "bc_oracle_deadband_kw": 0.02,
            "bc_oracle_power_scale_kw": 1.0,
            "bc_anchor_weight": float(variant["bc_anchor_weight"]),
            "bc_anchor_min_weight": float(variant["bc_anchor_min_weight"]),
            "bc_anchor_decay_updates": 130 if not pilot else 8,
            "bc_anchor_batch_size": 96 if not pilot else 48,
            "neutral_baseline_enabled": True,
            "neutral_warmup_episodes": 1,
            "training_episodes_per_validation": 2,
            "rollback_rejected_validation": True,
            "restore_best_policy_for_deterministic": True,
        }
    )
    # Only the causal heuristic needs raw physical import/export for labels.
    params["bc_use_physical_teacher_context"] = teacher_path is None
    _configure_bidirectional_leaf(config, rate=float(variant["signal_rate"]))
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{name}-start{start_step}-steps{horizon}"
    )
    config["checkpointing"].update(
        {
            "checkpoint_interval": None,
            "checkpoint_on_episode_end": False,
            "keep_episode_checkpoints": False,
        }
    )
    return config


def generate(
    output_dir: Path = OUTPUT_DIR,
    *,
    start_step: int = 0,
    horizon: int = ANNUAL_STEPS,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "annual" if horizon == ANNUAL_STEPS else f"start{start_step}_steps{horizon}"
    episodes = ANNUAL_EPISODES if horizon == ANNUAL_STEPS else 6
    configs = {
        "paired_neutral": build_paired_neutral_config(
            episodes=episodes,
            start_step=start_step,
            horizon=horizon,
        ),
        **{
            name: build_config(name, start_step=start_step, horizon=horizon)
            for name in VARIANTS
        },
    }
    outputs: list[Path] = []
    for name, config in configs.items():
        path = output_dir / f"cc_l2_v6_{name}_{suffix}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        outputs.append(path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=ANNUAL_STEPS)
    args = parser.parse_args()
    for path in generate(
        args.output_dir,
        start_step=args.start_step,
        horizon=args.horizon,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
