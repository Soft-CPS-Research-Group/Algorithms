#!/usr/bin/env python3
"""Generate CC-L2 V3 campaigns with causal per-building credit.

V2 proved that the frozen PPO signal path is neutral at 1.0, but its learned
coordinator collapsed to an almost uniform 0.99 signal. V3 keeps the accepted
frozen leaves and changes the learning problem itself: exact member settlement
and service rewards, one critic value/advantage per price factor, a corrected
battery-SoC teacher, and enough price authority to reproduce the successful
cheap/export response observed in the scalar V5.2 probe.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from scripts.generate_cc_level2_ppo_frozen_v2 import (
        REPO_ROOT,
        cc_recipe as v2_cc_recipe,
        derive_smoke as derive_v2_smoke,
        signal_path_neutral_recipe,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    from generate_cc_level2_ppo_frozen_v2 import (
        REPO_ROOT,
        cc_recipe as v2_cc_recipe,
        derive_smoke as derive_v2_smoke,
        signal_path_neutral_recipe,
    )


EXPERIMENT_NAME = "cc_level2_ppo_member_credit_v3"
OUTPUT_DIR = REPO_ROOT / "configs/experiments" / EXPERIMENT_NAME

VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "member_cost_hourly": {
        "seed": 123,
        "episodes": 10,
        "cc_action_interval": 4,
        "team_reward_mix": 0.20,
        "w_peak": 0.03,
        "w_ramp": 0.02,
        "w_export": 0.01,
        "bc_mult_scale": 0.55,
    },
    "member_cost_30min": {
        "seed": 456,
        "episodes": 8,
        "cc_action_interval": 2,
        "team_reward_mix": 0.20,
        "w_peak": 0.03,
        "w_ramp": 0.02,
        "w_export": 0.01,
        "bc_mult_scale": 0.45,
    },
    "member_scorecard_hourly": {
        "seed": 789,
        "episodes": 10,
        "cc_action_interval": 4,
        "team_reward_mix": 0.35,
        "w_peak": 0.12,
        "w_ramp": 0.08,
        "w_export": 0.02,
        "bc_mult_scale": 0.45,
    },
    "member_schedule_teacher_hourly": {
        "seed": 2024,
        "episodes": 10,
        "cc_action_interval": 4,
        "team_reward_mix": 0.20,
        "w_peak": 0.03,
        "w_ramp": 0.02,
        "w_export": 0.01,
        "bc_mult_scale": 0.45,
        "bc_teacher_mode": "cheap_and_export",
    },
}


def _apply_pilot_horizon(config: dict[str, Any], pilot_steps: int) -> None:
    if pilot_steps < 4096 or pilot_steps % 4 != 0:
        raise ValueError("pilot_steps must be a multiple of 4 and at least 4096")
    decisions = pilot_steps // int(
        config["pipeline"][0]["hyperparameters"]["cc_action_interval"]
    )
    config["metadata"]["run_name"] += f" pilot {pilot_steps}"
    config["tracking"]["tags"].update(
        {
            "evidence": "matched_slice_pilot",
            "pilot_steps": str(pilot_steps),
            "promotion_eligible": "False",
        }
    )
    # One episode collects a causal teacher dataset, three learn, and the
    # fifth is a deterministic replay. This yields several PPO updates on a
    # meaningful multi-week slice without pretending to be annual evidence.
    config["simulator"].update(
        {
            "episodes": 5,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": pilot_steps - 1,
            "episode_time_steps": pilot_steps,
        }
    )
    config["simulator"]["export"]["session_name"] += f"-pilot{pilot_steps}"
    config["pipeline"][0]["hyperparameters"].update(
        {
            "bc_collect_steps": decisions,
            "bc_train_steps": 1000,
            "bc_train_chunk_steps": 125,
            "bc_progress_interval": 125,
            "num_steps": 256,
            "mini_batch_size": 64,
        }
    )
    config["checkpointing"]["checkpoint_interval"] = None


def build_paired_neutral_config(*, pilot_steps: int) -> dict[str, Any]:
    """Exact frozen PPO signal-path reference for a V3 matched slice."""
    if pilot_steps < 4096 or pilot_steps % 4 != 0:
        raise ValueError("pilot_steps must be a multiple of 4 and at least 4096")
    config = copy.deepcopy(signal_path_neutral_recipe())
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"PPO paired neutral pilot {pilot_steps}",
            "description": (
                "Exact frozen PPO signal-path reference matched to the CC-L2 "
                "V3 pilot horizon."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": "ppo_paired_neutral",
            "evidence": "matched_slice_pilot",
            "pilot_steps": str(pilot_steps),
            "promotion_eligible": "False",
        }
    )
    config["simulator"].update(
        {
            "episodes": 1,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": pilot_steps - 1,
            "episode_time_steps": pilot_steps,
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-ppo-paired-neutral-pilot{pilot_steps}"
    )
    config["checkpointing"]["checkpoint_interval"] = None
    return config


def build_config(
    name: str,
    *,
    smoke: bool = False,
    pilot_steps: int | None = None,
) -> dict[str, Any]:
    if smoke and pilot_steps is not None:
        raise ValueError("smoke and pilot_steps are mutually exclusive")
    if name not in VARIANTS:
        raise ValueError(f"Unknown CC-L2 V3 variant: {name}")
    variant = VARIANTS[name]
    source = "scorecard_seed456" if "scorecard" in name else "cost_seed123"
    config = copy.deepcopy(v2_cc_recipe(source))
    manager = config["pipeline"][0]["hyperparameters"]
    reward = config["simulator"]["reward_function_kwargs"]

    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-L2 V3 {name}",
            "description": (
                "Frozen PPO seed 789 under a trainable per-building coordinator "
                "with exact member settlement credit and member-specific PPO advantages."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": name,
            "cc_seed": str(variant["seed"]),
            "credit_assignment": "member_decomposed",
            "team_reward_mix": str(variant["team_reward_mix"]),
            "price_range": "0.55_1.15",
            "cc_action_interval": str(variant["cc_action_interval"]),
            "training_episodes": str(variant["episodes"]),
            "battery_soc_teacher_sign": "price_semantics_corrected",
            "promotion_eligible": "False",
        }
    )
    config["training"]["seed"] = int(variant["seed"])
    config["simulator"].update(
        {
            "episodes": int(variant["episodes"]),
            "deterministic_finish": True,
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{name}"
    )
    reward.update(
        {
            "credit_assignment": "member_decomposed",
            "w_peak": float(variant["w_peak"]),
            "w_ramp": float(variant["w_ramp"]),
            "w_export": float(variant["w_export"]),
        }
    )
    manager.update(
        {
            "credit_assignment": "member_decomposed",
            "team_reward_mix": float(variant["team_reward_mix"]),
            "price_min": 0.55,
            "price_max": 1.15,
            "cc_action_interval": int(variant["cc_action_interval"]),
            "num_steps": 336,
            "mini_batch_size": 84,
            "lr": 5.0e-5,
            "vf_coef": 0.5,
            "ent_coef": 0.002,
            "target_kl": 0.02,
            "initial_log_std": -2.3,
            "w_factor": 0.0005,
            "w_smoothness": 0.001,
            "bc_mult_scale": float(variant["bc_mult_scale"]),
            "bc_teacher_mode": str(
                variant.get("bc_teacher_mode", "continuous_score")
            ),
            "bc_discount_multiplier": 0.90,
            "bc_export_activation_kw": 1.0e-9,
            "bc_w_soc": 0.15,
            "bc_w_net": 0.20,
        }
    )

    if smoke:
        config = derive_v2_smoke(config)
        # BC takes two bounded optimizer chunks after collection; a 96-step
        # rollout would then end two decisions short of a real PPO update.
        # Use 64 so the smoke exercises member GAE and the optimizer as well.
        config["pipeline"][0]["hyperparameters"].update(
            {"num_steps": 64, "mini_batch_size": 32}
        )
        config["tracking"]["tags"]["protocol"] = EXPERIMENT_NAME
        config["simulator"]["export"]["session_name"] = (
            f"{EXPERIMENT_NAME}-{name}-smoke"
        )
    elif pilot_steps is not None:
        _apply_pilot_horizon(config, pilot_steps)
    return config


def generate(
    output_dir: Path = OUTPUT_DIR,
    *,
    smoke: bool = False,
    pilot_steps: int | None = None,
) -> list[Path]:
    if smoke and pilot_steps is not None:
        raise ValueError("smoke and pilot_steps are mutually exclusive")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    suffix = (
        "_smoke"
        if smoke
        else (f"_pilot{pilot_steps}" if pilot_steps is not None else "")
    )
    if pilot_steps is not None:
        neutral_output = output_dir / f"ppo_paired_neutral{suffix}.yaml"
        neutral_output.write_text(
            yaml.safe_dump(
                build_paired_neutral_config(pilot_steps=pilot_steps),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(neutral_output)
    for name in VARIANTS:
        output = output_dir / f"cc_l2_v3_{name}{suffix}.yaml"
        output.write_text(
            yaml.safe_dump(
                build_config(name, smoke=smoke, pilot_steps=pilot_steps),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot-steps", type=int)
    args = parser.parse_args()
    for path in generate(
        args.output_dir,
        smoke=args.smoke,
        pilot_steps=args.pilot_steps,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
