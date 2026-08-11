#!/usr/bin/env python3
"""Generate MATD3 V4 safe storage-residual campaigns over SMART.

V3 came within EUR 71 of SMART but allowed actor/exploration authority over
EVs and deferrables and trained its critic on large service penalties that a
storage correction could not control. V4 copies all service actions exactly
from the SMART teacher, learns stationary-battery corrections only, masks both
environment and target-policy noise, and compares training horizons rather
than assuming the last of two years is best.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from scripts.generate_matd3_v3_campaign import (
        REPO_ROOT,
        SMART_CONFIG,
        build_config as build_v3_config,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    from generate_matd3_v3_campaign import (
        REPO_ROOT,
        SMART_CONFIG,
        build_config as build_v3_config,
    )


EXPERIMENT_NAME = "matd3_storage_safe_v4"
OUTPUT_DIR = REPO_ROOT / "configs/experiments" / EXPERIMENT_NAME

VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "smart_zero_residual_gate": {
        "training_years": 0,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.0,
    },
    "storage_cost_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.06,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
    },
    "storage_cost_medium_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.10,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
    },
    "storage_cost_smooth_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.06,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "actor_storage_smoothness_l2_penalty": 0.01,
        "actor_storage_smoothness_deadband": 0.03,
    },
    "storage_net_smooth_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.06,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "actor_storage_smoothness_l2_penalty": 0.01,
        "actor_storage_smoothness_deadband": 0.03,
        # Align the learned peak/import pressure with the scorecard boundary:
        # simultaneous local exports offset local imports before the penalty.
        "community_penalty_use_net_exchange": True,
    },
    "storage_net_context_smooth_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.06,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "actor_storage_smoothness_l2_penalty": 0.01,
        "actor_storage_smoothness_deadband": 0.03,
        "community_penalty_use_net_exchange": True,
        # Decentralized local state cannot reveal whether another member is
        # currently exporting or creating the aggregate peak.  These current,
        # causal community features let each storage actor condition its
        # correction on the same physical boundary used by the scorecard.
        "actor_community_context_enabled": True,
    },
    "storage_context_old_recipe_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        # Reproduce the effective 4096-step authority of the strongest old
        # pilot (0.03 -> 0.24 over 35040 steps gives about 0.055 here), while
        # retaining runtime electrical projection.
        "residual_action_final_scale": 0.055,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "residual_building_gain_multipliers": {"Building_15": 1.0},
        "actor_storage_smoothness_l2_penalty": 0.0,
        "actor_storage_smoothness_deadband": 0.10,
        "community_penalty_use_net_exchange": False,
        "actor_community_context_enabled": True,
    },
    "storage_context_old_recipe_unprojected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        # Exact short-horizon authority of the strongest historical pilot,
        # without the runtime projector that changed its battery trajectory.
        # The paired seed-789 pilot had zero electrical violations on this
        # path; other seeds remain subject to the projection-tolerant gate.
        "residual_action_final_scale": 0.055,
        "local_action_safety_enabled": False,
        "residual_building_gain_multipliers": {},
        "actor_storage_smoothness_l2_penalty": 0.0,
        "actor_storage_smoothness_deadband": 0.10,
        "community_penalty_use_net_exchange": False,
        "actor_community_context_enabled": True,
    },
    "storage_context_exact_replay_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        # This is the exact contract of the reproducible seed-789 champion.
        # Its 4096-step replay is bit-identical only when the annual authority
        # ramp is preserved; replacing 0.24/35040 by an apparently equivalent
        # short-horizon final scale changes the whole learning trajectory.
        "residual_action_final_scale": 0.24,
        "preserve_annual_residual_growth": True,
        "local_action_safety_enabled": False,
        "residual_building_gain_multipliers": {},
        "actor_storage_smoothness_l2_penalty": 0.0,
        "actor_storage_smoothness_deadband": 0.10,
        "community_penalty_use_net_exchange": False,
        "actor_community_context_enabled": True,
    },
    "storage_context_exact_extension_h2": {
        "training_years": 2,
        "critic_team_reward_mix": 0.0,
        # Controlled horizon ablation of the exact H1 champion: change only
        # the number of learning episodes.  In particular, retain its annual
        # authority ramp and leave the runtime projector disabled so any H2
        # difference can be attributed to the additional training episode.
        "residual_action_final_scale": 0.24,
        "preserve_annual_residual_growth": True,
        "local_action_safety_enabled": False,
        "residual_building_gain_multipliers": {},
        "actor_storage_smoothness_l2_penalty": 0.0,
        "actor_storage_smoothness_deadband": 0.10,
        "community_penalty_use_net_exchange": False,
        "actor_community_context_enabled": True,
    },
    "storage_context_old_recipe_winners_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.055,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        # Diagnostic projection of the measured seed-789 pilot: retain
        # residual authority only for members whose settled local cost beat
        # the paired SMART trajectory. This is deliberately tagged as a
        # pilot-selection ablation and must be revalidated on other seeds and
        # the annual horizon before it can be promoted.
        "residual_building_gain_multipliers": {
            "Building_1": 1.0,
            "Building_2": 1.0,
            "Building_3": 0.0,
            "Building_4": 1.0,
            "Building_5": 1.0,
            "Building_6": 0.0,
            "Building_7": 1.0,
            "Building_8": 0.0,
            "Building_9": 1.0,
            "Building_10": 1.0,
            "Building_11": 0.0,
            "Building_12": 0.0,
            "Building_13": 1.0,
            "Building_14": 0.0,
            "Building_15": 1.0,
            "Building_16": 1.0,
            "Building_17": 1.0,
        },
        "actor_storage_smoothness_l2_penalty": 0.0,
        "actor_storage_smoothness_deadband": 0.10,
        "community_penalty_use_net_exchange": False,
        "actor_community_context_enabled": True,
    },
    "storage_context_old_recipe_strong_winners_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.055,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        # More conservative companion ablation: only the seven members that
        # saved at least EUR 0.50 on the paired pilot retain authority.
        "residual_building_gain_multipliers": {
            "Building_1": 1.0,
            "Building_2": 0.0,
            "Building_3": 0.0,
            "Building_4": 1.0,
            "Building_5": 1.0,
            "Building_6": 0.0,
            "Building_7": 1.0,
            "Building_8": 0.0,
            "Building_9": 0.0,
            "Building_10": 1.0,
            "Building_11": 0.0,
            "Building_12": 0.0,
            "Building_13": 0.0,
            "Building_14": 0.0,
            "Building_15": 1.0,
            "Building_16": 1.0,
            "Building_17": 0.0,
        },
        "actor_storage_smoothness_l2_penalty": 0.0,
        "actor_storage_smoothness_deadband": 0.10,
        "community_penalty_use_net_exchange": False,
        "actor_community_context_enabled": True,
    },
    "storage_context_old_recipe_projected_h2": {
        "training_years": 2,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.055,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "residual_building_gain_multipliers": {"Building_15": 1.0},
        "actor_storage_smoothness_l2_penalty": 0.0,
        "actor_storage_smoothness_deadband": 0.10,
        "community_penalty_use_net_exchange": False,
        "actor_community_context_enabled": True,
    },
    "storage_net_context_ramp_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.06,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "actor_storage_smoothness_l2_penalty": 0.01,
        "actor_storage_smoothness_deadband": 0.03,
        "community_penalty_use_net_exchange": True,
        "community_ramping_penalty": 0.004,
        "actor_community_context_enabled": True,
    },
    "storage_net_context_team25_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.25,
        "residual_action_final_scale": 0.06,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "actor_storage_smoothness_l2_penalty": 0.01,
        "actor_storage_smoothness_deadband": 0.03,
        "community_penalty_use_net_exchange": True,
        "actor_community_context_enabled": True,
    },
    "storage_net_context_replay_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.25,
        "residual_action_final_scale": 0.06,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "actor_storage_smoothness_l2_penalty": 0.01,
        "actor_storage_smoothness_deadband": 0.03,
        "community_penalty_use_net_exchange": True,
        "actor_community_context_enabled": True,
        # EV and deferrable actions are immutable in this campaign. Do not
        # spend prioritized replay capacity on their teacher activity.
        "storage_replay_alignment": True,
    },
    "storage_net_context_accelerated_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.25,
        "residual_action_final_scale": 0.08,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "actor_storage_smoothness_l2_penalty": 0.008,
        "actor_storage_smoothness_deadband": 0.03,
        "community_penalty_use_net_exchange": True,
        "actor_community_context_enabled": True,
        "storage_replay_alignment": True,
        "random_exploration_steps": 1024,
        "warm_start_policy_phaseout_steps": 4096,
        "train_during_initial_exploration": True,
        "initial_exploration_training_start_step": 256,
        "actor_policy_loss_weight": 0.20,
        "actor_policy_loss_warmup_weight": 0.04,
        "actor_policy_loss_warmup_steps": 4096,
        "actor_storage_behavior_cloning_multiplier": 0.04,
        "actor_behavior_cloning_min_weight": 0.04,
        "actor_residual_delta_l2_penalty": 0.008,
        "actor_lr": 1.0e-4,
    },
    "storage_net_context_accelerated_wide_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.20,
        "residual_action_final_scale": 0.16,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "actor_storage_smoothness_l2_penalty": 0.006,
        "actor_storage_smoothness_deadband": 0.03,
        "community_penalty_use_net_exchange": True,
        "actor_community_context_enabled": True,
        "storage_replay_alignment": True,
        "random_exploration_steps": 1024,
        "warm_start_policy_phaseout_steps": 4096,
        "train_during_initial_exploration": True,
        "initial_exploration_training_start_step": 256,
        "actor_policy_loss_weight": 0.20,
        "actor_policy_loss_warmup_weight": 0.04,
        "actor_policy_loss_warmup_steps": 4096,
        "actor_storage_behavior_cloning_multiplier": 0.03,
        "actor_behavior_cloning_min_weight": 0.03,
        "actor_residual_delta_l2_penalty": 0.004,
        "actor_lr": 1.0e-4,
    },
    "storage_net_context_accelerated_cost_first_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.15,
        "residual_action_final_scale": 0.12,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "actor_storage_smoothness_l2_penalty": 0.004,
        "actor_storage_smoothness_deadband": 0.03,
        "community_settlement_cost_weight": 1.50,
        "community_peak_import_penalty": 0.0004,
        "community_export_penalty": 0.00005,
        "battery_throughput_penalty": 0.0001,
        "community_penalty_use_net_exchange": True,
        "actor_community_context_enabled": True,
        "storage_replay_alignment": True,
        "random_exploration_steps": 1024,
        "warm_start_policy_phaseout_steps": 4096,
        "train_during_initial_exploration": True,
        "initial_exploration_training_start_step": 256,
        "actor_policy_loss_weight": 0.24,
        "actor_policy_loss_warmup_weight": 0.05,
        "actor_policy_loss_warmup_steps": 4096,
        "actor_storage_behavior_cloning_multiplier": 0.025,
        "actor_behavior_cloning_min_weight": 0.025,
        "actor_residual_delta_l2_penalty": 0.003,
        "actor_lr": 1.0e-4,
    },
    "storage_net_context_temporal_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.25,
        "residual_action_final_scale": 0.08,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
        "actor_storage_smoothness_l2_penalty": 0.008,
        "actor_storage_smoothness_deadband": 0.03,
        "community_penalty_use_net_exchange": True,
        "community_ramping_penalty": 0.002,
        "actor_community_context_enabled": True,
        "actor_frame_stack_steps": 4,
        "storage_replay_alignment": True,
        "random_exploration_steps": 1024,
        "warm_start_policy_phaseout_steps": 4096,
        "train_during_initial_exploration": True,
        "initial_exploration_training_start_step": 256,
        "actor_policy_loss_weight": 0.20,
        "actor_policy_loss_warmup_weight": 0.04,
        "actor_policy_loss_warmup_steps": 4096,
        "actor_storage_behavior_cloning_multiplier": 0.04,
        "actor_behavior_cloning_min_weight": 0.04,
        "actor_residual_delta_l2_penalty": 0.008,
        "actor_lr": 1.0e-4,
    },
    "storage_net_context_b15_guarded_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.25,
        "residual_action_final_scale": 0.06,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.75,
        "residual_building_gain_multipliers": {"Building_15": 0.25},
        "actor_storage_smoothness_l2_penalty": 0.01,
        "actor_storage_smoothness_deadband": 0.03,
        "community_penalty_use_net_exchange": True,
        "actor_community_context_enabled": True,
        "storage_replay_alignment": True,
    },
    "storage_cost_wide_projected_h1": {
        "training_years": 1,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.16,
        "local_action_safety_enabled": True,
        "local_action_safety_headroom_reserve_kw": 0.10,
    },
    "storage_cost_h2": {
        "training_years": 2,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.08,
    },
    "storage_team25_h2": {
        "training_years": 2,
        "critic_team_reward_mix": 0.25,
        "residual_action_final_scale": 0.08,
    },
    "storage_cost_h4": {
        "training_years": 4,
        "critic_team_reward_mix": 0.0,
        "residual_action_final_scale": 0.08,
    },
}


def build_smart_reference_config(
    *,
    seed: int = 789,
    smoke_steps: int | None = None,
    episodes: int = 1,
) -> dict[str, Any]:
    """Build the exact paired SMART control for the residual-zero gate.

    The reward is deliberately left unchanged: a frozen SMART policy cannot
    react to it.  The evidence is the exported physical trajectory and KPIs,
    which must match the zero-residual MATD3 gate on the same horizon.
    """
    if int(episodes) < 1:
        raise ValueError("MATD3 V4 SMART reference episodes must be at least 1")
    config = yaml.safe_load(SMART_CONFIG.read_text(encoding="utf-8"))
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"SMART paired reference seed {seed}",
            "description": (
                "Exact frozen SMART reference paired with the MATD3 V4 "
                "zero-residual implementation gate."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": "smart_paired_reference",
            "seed": str(seed),
            "evaluation_episode_index": str(int(episodes)),
            "episode_realization_matched": str(int(episodes) > 1),
            "promotion_eligible": "False",
        }
    )
    config["training"]["seed"] = seed
    config["simulator"]["episodes"] = int(episodes)
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-smart_paired_reference-seed{seed}"
        + ("" if int(episodes) == 1 else f"-ep{int(episodes)}")
    )
    if smoke_steps is not None:
        if smoke_steps < 4096:
            raise ValueError("MATD3 V4 smoke_steps must be at least 4096")
        config["metadata"]["run_name"] += f" smoke {smoke_steps}"
        config["tracking"]["tags"]["evidence"] = "functional_smoke"
        config["simulator"].update(
            {
                "simulation_end_time_step": smoke_steps - 1,
                "episode_time_steps": smoke_steps,
            }
        )
        config["simulator"]["export"]["session_name"] += (
            f"-smoke{smoke_steps}"
        )
    return config


def build_config(
    name: str,
    *,
    seed: int = 789,
    smoke_steps: int | None = None,
) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"Unknown MATD3 V4 variant: {name}")
    variant = VARIANTS[name]
    config = copy.deepcopy(build_v3_config(variant_name="smart_anchor", seed=seed))
    params = config["pipeline"][0]["exploration"]["params"]
    replay = config["pipeline"][0]["replay_buffer"]
    training_years = int(variant["training_years"])
    is_gate = training_years == 0

    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"MATD3 V4 {name} seed {seed}",
            "description": (
                "Exact SMART service teacher with MATD3 authority restricted "
                "to stationary batteries and controllable storage-only reward."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": name,
            "seed": str(seed),
            "training_years": str(training_years),
            "deterministic_evaluation_years": "1",
            "service_action_contract": "exact_smart_teacher",
            "residual_authority": "stationary_storage_only",
            "critic_team_reward_mix": str(variant["critic_team_reward_mix"]),
            "residual_noise_masked": "True",
            "electrical_service_building_residual": str(
                variant.get(
                    "residual_building_gain_multipliers",
                    {"Building_15": 0.0},
                )
            ),
            "promotion_eligible": "False",
        }
    )
    config["simulator"].update(
        {
            "reward_function": "CostCommunityStorageResidualRewardV55",
            "reward_function_kwargs": {
                "community_settlement_cost_weight": float(
                    variant.get("community_settlement_cost_weight", 1.25)
                ),
                "community_peak_import_penalty": float(
                    variant.get("community_peak_import_penalty", 0.0008)
                ),
                "community_ramping_penalty": float(
                    variant.get("community_ramping_penalty", 0.0)
                ),
                "community_export_penalty": float(
                    variant.get("community_export_penalty", 0.00015)
                ),
                "community_penalty_use_net_exchange": bool(
                    variant.get("community_penalty_use_net_exchange", False)
                ),
                "battery_throughput_penalty": float(
                    variant.get("battery_throughput_penalty", 0.0005)
                ),
                "grid_violation_penalty": 120.0,
            },
            "episodes": training_years + 1,
            "deterministic_finish": True,
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{name}-seed{seed}"
    )
    config["checkpointing"]["checkpoint_interval"] = (
        None if is_gate else 35040
    )

    if bool(variant.get("storage_replay_alignment", False)):
        replay.update(
            {
                "behavior_action_priority_weight": 0.0,
                "behavior_action_priority_scope": "all",
                "observation_event_priority_mode": "ev_pv_price_peak",
            }
        )
    if "actor_lr" in variant:
        config["pipeline"][0]["networks"]["actor"]["lr"] = float(
            variant["actor_lr"]
        )

    params.update(
        {
            "critic_team_reward_mix": float(variant["critic_team_reward_mix"]),
            "actor_community_context_enabled": bool(
                variant.get("actor_community_context_enabled", False)
            ),
            "actor_community_context_features": [
                "community_net_power",
                "community_import_power",
                "community_export_power",
                "community_pv_power",
                "storage_soc_mean",
            ],
            "sigma": 0.12,
            "min_sigma": 0.015,
            "storage_exploration_noise_multiplier": 1.0,
            "random_exploration_steps": int(
                variant.get("random_exploration_steps", 2048)
            ),
            "end_initial_exploration_time_step": int(
                variant.get("random_exploration_steps", 2048)
            ),
            "warm_start_policy_phaseout_steps": int(
                variant.get("warm_start_policy_phaseout_steps", 24576)
            ),
            "train_during_initial_exploration": bool(
                variant.get("train_during_initial_exploration", False)
            ),
            "initial_exploration_training_start_step": int(
                variant.get("initial_exploration_training_start_step", 0)
            ),
            "actor_policy_loss_weight": float(
                variant.get("actor_policy_loss_weight", 0.10)
            ),
            "actor_policy_loss_warmup_weight": float(
                variant.get("actor_policy_loss_warmup_weight", 0.01)
            ),
            "actor_policy_loss_warmup_steps": int(
                variant.get("actor_policy_loss_warmup_steps", 16384)
            ),
            "actor_storage_action_l2_penalty": 0.001,
            "actor_storage_smoothness_l2_penalty": float(
                variant.get("actor_storage_smoothness_l2_penalty", 0.0)
            ),
            "actor_storage_smoothness_deadband": float(
                variant.get("actor_storage_smoothness_deadband", 0.10)
            ),
            "actor_storage_behavior_cloning_multiplier": float(
                variant.get(
                    "actor_storage_behavior_cloning_multiplier",
                    0.08,
                )
            ),
            "actor_behavior_cloning_min_weight": float(
                variant.get("actor_behavior_cloning_min_weight", 0.12)
            ),
            "actor_residual_delta_l2_penalty": float(
                variant.get("actor_residual_delta_l2_penalty", 0.015)
            ),
            "actor_frame_stack_steps": int(
                variant.get("actor_frame_stack_steps", 1)
            ),
            "actor_ev_v2g_action_l2_penalty": 0.0,
            "actor_ev_v2g_action_mass_penalty": 0.0,
            "residual_action_scale": 0.03,
            "residual_action_final_scale": float(
                variant.get("residual_action_final_scale", 0.24)
            ),
            "residual_action_growth_steps": 35040,
            "residual_storage_action_scale_multiplier": float(
                variant.get(
                    "residual_storage_action_scale_multiplier",
                    0.60,
                )
            ),
            "residual_ev_action_scale_multiplier": 0.0,
            "residual_deferrable_action_scale_multiplier": 0.0,
            # Building_15 is the only member with configured three-phase
            # electrical-service limits in this dataset.  The learned storage
            # residual is frozen there at the exact audited SMART teacher;
            # the remaining sixteen actors retain storage authority.
            "residual_building_gain_multipliers": dict(
                variant.get(
                    "residual_building_gain_multipliers",
                    {"Building_15": 0.0},
                )
            ),
            # The critic still sees the action that actually reached the
            # environment (the wrapper returns the projected action to
            # update()).  This runtime layer prevents Gaussian exploration or
            # an imperfect residual actor from violating local electrical/SoC
            # constraints before the penalty can teach it about the mistake.
            # Keep the exact zero-residual SMART gate projector-free so its
            # trajectory remains an implementation identity test.
            "local_action_safety_enabled": bool(
                variant.get("local_action_safety_enabled", False)
            ),
            "local_action_safety_fail_on_infeasible": False,
            "local_action_safety_protect_ev_minimum": False,
            "local_action_safety_ev_minimum_mode": "deadline_feasible",
            "local_action_safety_protect_ev_service_target": False,
            "local_action_safety_protect_deferrable_must_start": False,
            "local_action_safety_allow_discretionary_deferrable_start": False,
            "local_action_safety_headroom_reserve_kw": float(
                variant.get("local_action_safety_headroom_reserve_kw", 0.0)
            ),
            "local_action_safety_runtime_only_export": True,
        }
    )

    if smoke_steps is not None:
        if smoke_steps < 4096:
            raise ValueError("MATD3 V4 smoke_steps must be at least 4096")
        config["metadata"]["run_name"] += f" smoke {smoke_steps}"
        config["tracking"]["tags"].update(
            {
                "evidence": "functional_smoke",
                "training_years": str(training_years),
            }
        )
        config["simulator"].update(
            {
                "episodes": 1 if is_gate else training_years + 1,
                "simulation_end_time_step": smoke_steps - 1,
                "episode_time_steps": smoke_steps,
            }
        )
        config["simulator"]["export"]["session_name"] += (
            f"-smoke{smoke_steps}"
        )
        # Most short pilots exercise their configured final authority.  The
        # exact replay is deliberately exempt: its validated learning path is
        # defined by the annual 35040-step ramp, even on the 4096-step slice.
        if not bool(variant.get("preserve_annual_residual_growth", False)):
            params["residual_action_growth_steps"] = smoke_steps
        config["checkpointing"]["checkpoint_interval"] = None
    return config


def generate(
    output_dir: Path = OUTPUT_DIR,
    *,
    seed: int = 789,
    smoke_steps: int | None = None,
    matched_reference_episodes: list[int] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    suffix = "" if smoke_steps is None else f"_smoke{smoke_steps}"
    reference_output = output_dir / f"smart_paired_reference_seed{seed}{suffix}.yaml"
    reference_output.write_text(
        yaml.safe_dump(
            build_smart_reference_config(seed=seed, smoke_steps=smoke_steps),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    outputs.append(reference_output)
    for episodes in matched_reference_episodes or []:
        if int(episodes) == 1:
            continue
        matched_reference_output = output_dir / (
            f"smart_paired_reference_ep{int(episodes)}_seed{seed}{suffix}.yaml"
        )
        matched_reference_output.write_text(
            yaml.safe_dump(
                build_smart_reference_config(
                    seed=seed,
                    smoke_steps=smoke_steps,
                    episodes=int(episodes),
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(matched_reference_output)
    for name in VARIANTS:
        output = output_dir / f"matd3_v4_{name}_seed{seed}{suffix}.yaml"
        output.write_text(
            yaml.safe_dump(
                build_config(name, seed=seed, smoke_steps=smoke_steps),
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
    parser.add_argument(
        "--matched-reference-episodes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Also emit SMART controls whose final episode realization matches "
            "the H1/H2 candidate evaluation episode."
        ),
    )
    args = parser.parse_args()
    for path in generate(
        args.output_dir,
        seed=args.seed,
        smoke_steps=args.smoke_steps,
        matched_reference_episodes=args.matched_reference_episodes,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
