#!/usr/bin/env python3
"""Generate incumbent-preserving CC-L2 campaigns over frozen local PPOs.

The V3 member-credit pilot proved that per-building credit works, but also
showed that an unconstrained learned price can destroy the strong neutral PPO
leaf. V4 makes the successful deployable Level-1 causal rule the incumbent:
prices are exactly neutral outside cheap-and-export intervals, while PPO learns
only the per-building discount strength inside those intervals.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from scripts.generate_cc_level2_ppo_member_credit_v3 import (
        REPO_ROOT,
        build_config as build_v3_config,
        build_paired_neutral_config as build_v3_neutral,
    )
    from scripts.generate_cc_ppo_causal_online_v5p3b import (
        recipe as build_v5p3b_recipe,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    from generate_cc_level2_ppo_member_credit_v3 import (
        REPO_ROOT,
        build_config as build_v3_config,
        build_paired_neutral_config as build_v3_neutral,
    )
    from generate_cc_ppo_causal_online_v5p3b import (
        recipe as build_v5p3b_recipe,
    )


EXPERIMENT_NAME = "cc_level2_ppo_causal_guard_v4"
OUTPUT_DIR = REPO_ROOT / "configs" / "experiments" / EXPERIMENT_NAME
ANNUAL_STEPS = 35040

# Matched 4096-step coordinate-search incumbent. B2--B14 and B16 benefit from
# the stronger causal discount; B15 is deliberately neutral because its
# discount caused most of the ramping regression. The combined vector passed
# every network/service gate and reduced cost by 1.14% against the paired PPO.
CAUSAL_VECTOR_INCUMBENT = [
    0.90,
    0.85,
    0.85,
    0.85,
    0.85,
    0.85,
    0.85,
    0.85,
    0.85,
    0.85,
    0.85,
    0.85,
    0.85,
    0.85,
    0.90,
    0.85,
    0.90,
]

VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "causal_member_cost_hourly": {
        "seed": 123,
        "episodes": 8,
        "cc_action_interval": 4,
        "price_min": 0.82,
        "team_reward_mix": 0.15,
        "w_peak": 0.03,
        "w_ramp": 0.02,
        "w_export": 0.01,
    },
    "causal_member_cost_30min": {
        "seed": 456,
        "episodes": 8,
        "cc_action_interval": 2,
        "price_min": 0.84,
        "team_reward_mix": 0.15,
        "w_peak": 0.03,
        "w_ramp": 0.02,
        "w_export": 0.01,
    },
    "causal_member_scorecard_hourly": {
        "seed": 789,
        "episodes": 8,
        "cc_action_interval": 4,
        "price_min": 0.86,
        "team_reward_mix": 0.35,
        "w_peak": 0.12,
        "w_ramp": 0.08,
        "w_export": 0.02,
    },
    "causal_member_cost_deep_hourly": {
        "seed": 2024,
        "episodes": 8,
        "cc_action_interval": 4,
        "price_min": 0.78,
        "team_reward_mix": 0.20,
        "w_peak": 0.03,
        "w_ramp": 0.02,
        "w_export": 0.01,
    },
    "causal_member_continuous_conservative": {
        "seed": 31415,
        "episodes": 12,
        "cc_action_interval": 4,
        "price_min": 0.84,
        "team_reward_mix": 0.25,
        "w_peak": 0.06,
        "w_ramp": 0.04,
        "w_export": 0.01,
        "reward_normalization": "none",
        "lr": 1.0e-5,
        "ent_coef": 0.0,
        "target_kl": 0.006,
        "initial_log_std": -3.2,
    },
    "causal_member_vector_incumbent": {
        "seed": 27182,
        "episodes": 12,
        "cc_action_interval": 4,
        "price_min": 0.78,
        "team_reward_mix": 0.25,
        "w_peak": 0.06,
        "w_ramp": 0.04,
        "w_export": 0.01,
        "reward_normalization": "none",
        "lr": 1.0e-5,
        "ent_coef": 0.0,
        "target_kl": 0.006,
        "initial_log_std": -3.2,
        "causal_initial_multipliers": CAUSAL_VECTOR_INCUMBENT,
    },
    "causal_member_vector_guarded": {
        "seed": 16180,
        "episodes": 12,
        "cc_action_interval": 4,
        "price_min": 0.78,
        "team_reward_mix": 0.25,
        "w_peak": 0.06,
        "w_ramp": 0.04,
        "w_export": 0.01,
        "reward_normalization": "none",
        "lr": 1.0e-5,
        "ent_coef": 0.0,
        "target_kl": 0.006,
        "initial_log_std": -3.2,
        "causal_initial_multipliers": CAUSAL_VECTOR_INCUMBENT,
        # Learn only a contextual correction around the hard-gate-safe vector,
        # rather than remapping the whole [price_min, 1.0] interval.
        "causal_residual_scale": 0.20,
    },
    "causal_member_vector_cost_explore_hourly": {
        "seed": 4242,
        "episodes": 12,
        "cc_action_interval": 4,
        "price_min": 0.75,
        "team_reward_mix": 0.10,
        "w_peak": 0.02,
        "w_ramp": 0.005,
        "w_export": 0.005,
        "reward_normalization": "running_zscore",
        "lr": 3.0e-5,
        "ent_coef": 0.0,
        "target_kl": 0.012,
        # The previous -3.2 exploration mapped to physical price changes of
        # only ~0.003, inside the measured leaf deadband. This deliberately
        # explores changes large enough to alter the frozen leaf trajectory.
        "initial_log_std": -1.25,
        "causal_initial_multipliers": CAUSAL_VECTOR_INCUMBENT,
        "causal_residual_scale": 1.0,
    },
    "causal_member_vector_cost_explore_30min": {
        "seed": 4243,
        "episodes": 12,
        "cc_action_interval": 2,
        "price_min": 0.75,
        "team_reward_mix": 0.10,
        "w_peak": 0.02,
        "w_ramp": 0.005,
        "w_export": 0.005,
        "reward_normalization": "running_zscore",
        "lr": 3.0e-5,
        "ent_coef": 0.0,
        "target_kl": 0.012,
        "initial_log_std": -1.25,
        "causal_initial_multipliers": CAUSAL_VECTOR_INCUMBENT,
        "causal_residual_scale": 1.0,
    },
    "causal_member_vector_scorecard_explore_hourly": {
        "seed": 4244,
        "episodes": 12,
        "cc_action_interval": 4,
        "price_min": 0.75,
        "team_reward_mix": 0.25,
        # Same physically identifiable exploration as the cost-first winner,
        # but restore material community peak/ramp pressure. This isolates the
        # objective trade-off without changing the action interval or leaf.
        "w_peak": 0.06,
        "w_ramp": 0.04,
        "w_export": 0.01,
        "reward_normalization": "running_zscore",
        "lr": 3.0e-5,
        "ent_coef": 0.0,
        "target_kl": 0.012,
        "initial_log_std": -1.25,
        "causal_initial_multipliers": CAUSAL_VECTOR_INCUMBENT,
        "causal_residual_scale": 1.0,
    },
}


def _apply_pilot_horizon(config: dict[str, Any], pilot_steps: int) -> None:
    if pilot_steps < 4096 or pilot_steps % 4 != 0:
        raise ValueError("pilot_steps must be a multiple of 4 and at least 4096")
    config["metadata"]["run_name"] += f" pilot {pilot_steps}"
    config["tracking"]["tags"].update(
        {
            "evidence": "matched_slice_pilot",
            "pilot_steps": str(pilot_steps),
            "promotion_eligible": "False",
        }
    )
    config["tracking"].update(
        {
            "progress_phase_updates_enabled": True,
            "stall_watchdog_context_interval_steps": 64,
        }
    )
    # Four learning episodes followed by one deterministic replay. There is no
    # BC phase: the policy is initialized exactly at the causal incumbent.
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
        {"num_steps": 256, "mini_batch_size": 64}
    )
    config["checkpointing"]["checkpoint_interval"] = None


def build_paired_neutral_config(
    *,
    pilot_steps: int | None = None,
    episodes: int = 1,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    horizon_steps = ANNUAL_STEPS if pilot_steps is None else int(pilot_steps)
    config = copy.deepcopy(build_v3_neutral(pilot_steps=horizon_steps))
    horizon_label = "annual" if pilot_steps is None else f"pilot {pilot_steps}"
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"PPO paired neutral V4 {horizon_label}",
        }
    )
    config["tracking"]["tags"]["protocol"] = EXPERIMENT_NAME
    if pilot_steps is None:
        config["tracking"]["tags"].pop("pilot_steps", None)
        config["tracking"]["tags"].update(
            {
                "evidence": "annual_episode_matched_reference",
                "promotion_eligible": "True",
            }
        )
    config["tracking"]["tags"].update(
        {
            "evaluation_episode_index": str(episodes),
            "episode_realization_matched": str(episodes > 1),
            "ppo_actor_price_conditioning": "current_only",
            "ppo_price_forecast_conditioning": "real_unmodified",
            "cc_price_scope": "ppo_current_and_local_residual_base",
            "v2g_enabled": "True",
        }
    )
    config["simulator"]["episodes"] = int(episodes)
    # V5's measured causal incumbent uses the PPO leaf with V2G enabled in
    # its SMART residual policy. Keep the neutral control physically
    # identical; otherwise the comparison changes both the CC and the leaf.
    leaf_params = config["pipeline"][1]["exploration"]["params"]
    leaf_params.update(
        {
            "local_price_conditioning_enabled": True,
            "local_price_forecast_mode": "real_unmodified",
            "residual_base_price_conditioning_enabled": True,
        }
    )
    leaf_params["residual_base_policy_hyperparameters"].update(
        {
            "allow_v2g": True,
            "signal_price_response_mode": "linear_discount",
            "signal_price_charge_reference_multiplier": 0.90,
            "signal_price_charge_gain_max": 1.5,
        }
    )
    episode_suffix = "" if episodes == 1 else f"-ep{episodes}"
    horizon_suffix = (
        "-annual"
        if pilot_steps is None
        else f"-pilot{pilot_steps}"
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-ppo-paired-neutral{episode_suffix}{horizon_suffix}"
    )
    return config


def build_causal_incumbent_config(*, pilot_steps: int) -> dict[str, Any]:
    """Exact global causal controller that V4 uses as its safe incumbent."""
    if pilot_steps < 4096 or pilot_steps % 4 != 0:
        raise ValueError("pilot_steps must be a multiple of 4 and at least 4096")
    config = copy.deepcopy(build_v5p3b_recipe("hourly_cost"))
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-L1 causal incumbent V4 pilot {pilot_steps}",
            "description": (
                "Exact global cheap-and-export 0.90 incumbent matched to the "
                "CC-L2 V4 pilot horizon and frozen PPO leaf."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": "causal_global_incumbent",
            "evidence": "matched_slice_pilot",
            "pilot_steps": str(pilot_steps),
            "promotion_eligible": "False",
        }
    )
    config["tracking"].update(
        {
            "progress_phase_updates_enabled": True,
            "stall_watchdog_context_interval_steps": 64,
        }
    )
    config["pipeline"][1]["exploration"]["params"][
        "residual_base_policy_hyperparameters"
    ]["allow_v2g"] = True
    config["simulator"].update(
        {
            "episodes": 1,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": pilot_steps - 1,
            "episode_time_steps": pilot_steps,
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-causal-global-incumbent-pilot{pilot_steps}"
    )
    config["checkpointing"]["checkpoint_interval"] = None
    return config


def build_config(
    name: str,
    *,
    pilot_steps: int | None = None,
) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"Unknown CC-L2 V4 variant: {name}")
    variant = VARIANTS[name]
    config = copy.deepcopy(build_v3_config("member_cost_hourly"))
    manager = config["pipeline"][0]["hyperparameters"]
    leaf_params = config["pipeline"][1]["exploration"]["params"]
    residual_params = leaf_params["residual_base_policy_hyperparameters"]
    reward = config["simulator"]["reward_function_kwargs"]

    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-L2 V4 {name}",
            "description": (
                "Frozen PPO seed 789 under member-credit CC-L2. The actor is "
                "causally gated: neutral outside cheap-and-export intervals "
                "and per-building discount learning inside them."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": name,
            "cc_seed": str(variant["seed"]),
            "training_episodes": str(int(variant["episodes"]) - 1),
            "total_episodes": str(variant["episodes"]),
            "evaluation_episode_index": str(variant["episodes"]),
            "episode_realization_matched": "True",
            "cc_action_interval": str(variant["cc_action_interval"]),
            "credit_assignment": "member_decomposed",
            "team_reward_mix": str(variant["team_reward_mix"]),
            "policy_parameterization": "causal_active_only",
            "causal_incumbent": "cheap_and_export_0.90",
            "inactive_actor_gradient": "masked",
            "ramp_credit_allocation": "causal_net",
            "price_range": f"{variant['price_min']}_1.0_active_only",
            "leaf_price_response": "linear_discount",
            "ppo_actor_price_conditioning": "current_only",
            "ppo_price_forecast_conditioning": "real_unmodified",
            "cc_price_scope": "ppo_current_and_local_residual_base",
            "v2g_enabled": "True",
            "reward_normalization": str(
                variant.get("reward_normalization", "running_zscore")
            ),
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
    config["simulator"]["export"]["session_name"] = f"{EXPERIMENT_NAME}-{name}"
    reward.update(
        {
            "credit_assignment": "member_decomposed",
            "ramp_credit_allocation": "causal_net",
            "w_peak": float(variant["w_peak"]),
            "w_ramp": float(variant["w_ramp"]),
            "w_export": float(variant["w_export"]),
        }
    )
    manager.update(
        {
            "credit_assignment": "member_decomposed",
            "team_reward_mix": float(variant["team_reward_mix"]),
            "reward_normalization": str(
                variant.get("reward_normalization", "running_zscore")
            ),
            "price_min": float(variant["price_min"]),
            "price_max": 1.0,
            "reference_multipliers": [1.0] * 17,
            "policy_parameterization": "causal_active_only",
            "causal_initial_multiplier": 0.90,
            "causal_initial_multipliers": list(
                variant.get("causal_initial_multipliers", [0.90] * 17)
            ),
            "causal_residual_scale": variant.get("causal_residual_scale"),
            # Match the deterministic incumbent in physical units. Current
            # and forecast prices have independent encoder ranges, so their
            # normalized values do not preserve the native cheap comparison.
            "causal_use_physical_context": True,
            "include_community_history": True,
            # 20 district/causal features (16 base + headroom + two causal
            # history values + causal-active), followed by six features for
            # each of the 17 buildings.
            "c_dim": 122,
            # Value fitting continues on inactive causal timesteps. Keep its
            # gradients out of the actor representation instead of allowing
            # the critic to move an otherwise masked price policy.
            "separate_value_encoder": True,
            "policy_residual_scale": 1.0,
            "cc_action_interval": int(variant["cc_action_interval"]),
            "bc_pretrain_enabled": False,
            "num_steps": 336,
            "mini_batch_size": 84,
            "lr": float(variant.get("lr", 3.0e-5)),
            "vf_coef": 0.5,
            "ent_coef": float(variant.get("ent_coef", 0.001)),
            "target_kl": float(variant.get("target_kl", 0.015)),
            "initial_log_std": float(variant.get("initial_log_std", -2.2)),
            "w_factor": 0.0,
            "w_smoothness": 0.001,
        }
    )
    residual_params.update(
        {
            # This is part of the frozen low-level contract and must match the
            # paired neutral PPO and deterministic incumbent.
            "allow_v2g": True,
            "signal_price_response_mode": "linear_discount",
            "signal_price_charge_reference_multiplier": 0.90,
            "signal_price_charge_gain_max": 1.5,
        }
    )
    leaf_params.update(
        {
            "local_price_conditioning_enabled": True,
            "local_price_forecast_mode": "real_unmodified",
            "residual_base_price_conditioning_enabled": True,
        }
    )
    if pilot_steps is not None:
        _apply_pilot_horizon(config, pilot_steps)
    return config


def generate(
    output_dir: Path = OUTPUT_DIR,
    *,
    pilot_steps: int | None = None,
    matched_neutral_episodes: list[int] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    suffix = "" if pilot_steps is None else f"_pilot{pilot_steps}"
    if pilot_steps is not None:
        neutral_path = output_dir / f"ppo_paired_neutral{suffix}.yaml"
        neutral_path.write_text(
            yaml.safe_dump(
                build_paired_neutral_config(pilot_steps=pilot_steps),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(neutral_path)
        for episodes in matched_neutral_episodes or []:
            if episodes == 1:
                continue
            matched_path = output_dir / (
                f"ppo_paired_neutral_ep{episodes}{suffix}.yaml"
            )
            matched_path.write_text(
                yaml.safe_dump(
                    build_paired_neutral_config(
                        pilot_steps=pilot_steps,
                        episodes=episodes,
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            outputs.append(matched_path)
        incumbent_path = output_dir / f"causal_global_incumbent{suffix}.yaml"
        incumbent_path.write_text(
            yaml.safe_dump(
                build_causal_incumbent_config(pilot_steps=pilot_steps),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(incumbent_path)
    else:
        # Emit one exact frozen-PPO control for every annual evaluation index
        # used by the variants, so a remote campaign cannot silently fall back
        # to episode 1 (currently the campaign contains episode 8 and 12).
        annual_episode_counts = sorted(
            {int(variant["episodes"]) for variant in VARIANTS.values()}
        )
        for episodes in annual_episode_counts:
            annual_neutral_path = output_dir / (
                f"ppo_paired_neutral_ep{episodes}.yaml"
            )
            annual_neutral_path.write_text(
                yaml.safe_dump(
                    build_paired_neutral_config(episodes=episodes),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            outputs.append(annual_neutral_path)
    for name in VARIANTS:
        path = output_dir / f"cc_l2_v4_{name}{suffix}.yaml"
        path.write_text(
            yaml.safe_dump(
                build_config(name, pilot_steps=pilot_steps),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pilot-steps", type=int)
    parser.add_argument(
        "--matched-neutral-episodes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Also emit frozen-PPO controls whose exported episode index "
            "matches a multi-episode CC candidate."
        ),
    )
    args = parser.parse_args()
    for path in generate(
        args.output_dir,
        pilot_steps=args.pilot_steps,
        matched_neutral_episodes=args.matched_neutral_episodes,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
