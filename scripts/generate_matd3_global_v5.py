#!/usr/bin/env python3
"""Generate corrected MATD3 V5 campaigns over an immutable SMART base.

V5 fixes two protocol defects exposed by the annual V4 replay:

* the residual actor reaches its final authority during the first training
  year and receives a complete second year at that authority;
* the residual base and behavior-cloning target are independent, allowing a
  fixed-service MILP to teach battery actions without replacing SMART EV and
  deferrable service.

The default campaign is causal and does not require an oracle artifact.  An
optional perfect-foresight schedule can be supplied as a training-only teacher
ablation; the evaluated MATD3 actor never reads that schedule.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from scripts.generate_matd3_storage_safe_v4 import (
        REPO_ROOT,
        build_config as build_v4_config,
        build_smart_reference_config as build_v4_smart_reference,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    from generate_matd3_storage_safe_v4 import (
        REPO_ROOT,
        build_config as build_v4_config,
        build_smart_reference_config as build_v4_smart_reference,
    )


EXPERIMENT_NAME = "matd3_global_v5"
OUTPUT_DIR = REPO_ROOT / "configs" / "experiments" / EXPERIMENT_NAME
SEASONAL_START_STEPS = (0, 8760, 17520, 26280)

VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "cost_first_h2": {
        "critic_team_reward_mix": 0.25,
        "settlement_weight": 1.50,
        "peak_weight": 0.00045,
        "ramp_weight": 0.00015,
        "export_weight": 0.00005,
        "emissions_weight": 0.0,
        "throughput_weight": 0.00010,
        "residual_final_scale": 0.60,
        "storage_authority_multiplier": 0.95,
        "residual_penalty": 0.0005,
        "smoothness_weight": 0.0010,
    },
    "balanced_h2": {
        "critic_team_reward_mix": 0.40,
        "settlement_weight": 1.40,
        "peak_weight": 0.00090,
        "ramp_weight": 0.00100,
        "export_weight": 0.00010,
        "emissions_weight": 0.0,
        "throughput_weight": 0.00010,
        "residual_final_scale": 0.55,
        "storage_authority_multiplier": 0.95,
        "residual_penalty": 0.0010,
        "smoothness_weight": 0.0020,
    },
    "ramp_guard_h2": {
        "critic_team_reward_mix": 0.50,
        "settlement_weight": 1.30,
        "peak_weight": 0.00120,
        "ramp_weight": 0.00300,
        "export_weight": 0.00010,
        "emissions_weight": 0.0,
        "throughput_weight": 0.00010,
        "residual_final_scale": 0.50,
        "storage_authority_multiplier": 0.90,
        "residual_penalty": 0.0020,
        "smoothness_weight": 0.0040,
    },
    "global_scorecard_h2": {
        # A fully cooperative ablation: every centralized critic learns from
        # the same community objective. This tests whether mixed local rewards
        # obstruct joint cost/peak/ramp improvements even though every critic
        # already observes the joint action.
        "critic_team_reward_mix": 1.00,
        "settlement_weight": 1.45,
        "peak_weight": 0.00100,
        "ramp_weight": 0.00150,
        "export_weight": 0.00010,
        "emissions_weight": 0.20,
        "throughput_weight": 0.00005,
        "residual_final_scale": 0.60,
        "storage_authority_multiplier": 0.95,
        "residual_penalty": 0.0005,
        "smoothness_weight": 0.0020,
    },
    "global_distilled_h2": {
        # Strong-supervision ablation.  The actor still has no teacher access
        # at evaluation, but keeps a small BC anchor throughout both training
        # years so critic noise cannot erase the physically replayed schedule.
        "critic_team_reward_mix": 1.00,
        "settlement_weight": 1.45,
        "peak_weight": 0.00100,
        "ramp_weight": 0.00150,
        "export_weight": 0.00010,
        "emissions_weight": 0.20,
        "throughput_weight": 0.00005,
        "residual_final_scale": 0.65,
        "storage_authority_multiplier": 0.95,
        "residual_penalty": 0.00025,
        "smoothness_weight": 0.0015,
        "actor_policy_weight": 0.18,
        "teacher_bc_weight": 0.45,
        "teacher_bc_min_weight": 0.06,
        "teacher_bc_decay_steps": 69054,
        "teacher_bc_extra_update_end_step": 35039,
        "teacher_offline_pretrain_steps": 96,
    },
}


def _apply_window(config: dict[str, Any], *, start_step: int, steps: int | None) -> None:
    if start_step < 0:
        raise ValueError("MATD3 V5 start_step must be non-negative")
    simulator = config["simulator"]
    if steps is None:
        if start_step != 0:
            raise ValueError("A non-zero MATD3 V5 start_step requires an explicit window")
        simulator.pop("simulation_start_time_step", None)
        return
    if steps < 128:
        raise ValueError("MATD3 V5 functional windows must contain at least 128 steps")
    simulator.update(
        {
            "simulation_start_time_step": int(start_step),
            "simulation_end_time_step": int(start_step + steps - 1),
            "episode_time_steps": int(steps),
        }
    )


def build_smart_reference_config(
    *,
    seed: int = 789,
    start_step: int = 0,
    steps: int | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(build_v4_smart_reference(seed=seed, episodes=3))
    _apply_window(config, start_step=start_step, steps=steps)
    suffix = "annual" if steps is None else f"start{start_step}_steps{steps}"
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"SMART paired V5 {suffix} seed {seed}",
            "description": (
                "Exact SMART control matched to the third-episode realization "
                "of the two-train-one-evaluation MATD3 V5 protocol."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": "smart_paired_reference",
            "window": suffix,
            "evaluation_episode_index": "3",
            "episode_realization_matched": "True",
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-smart-{suffix}-seed{seed}"
    )
    return config


def build_config(
    name: str,
    *,
    seed: int = 789,
    start_step: int = 0,
    steps: int | None = None,
    teacher_schedule: Path | None = None,
    teacher_label: str = "cost",
) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"Unknown MATD3 V5 variant: {name}")
    variant = VARIANTS[name]
    config = copy.deepcopy(
        build_v4_config("storage_context_exact_extension_h2", seed=seed)
    )
    _apply_window(config, start_step=start_step, steps=steps)
    suffix = "annual" if steps is None else f"start{start_step}_steps{steps}"
    if steps is None:
        random_exploration_steps = 1024
        initial_training_start_step = 256
        policy_phaseout_steps = 8192
        authority_growth_steps = 12288
        actor_policy_warmup_steps = 8192
        cloning_decay_start_step = 1024
        training_horizon_steps = 2 * 35039
    else:
        # A pilot must expose the policy to its final deployment authority
        # during training. Reusing annual schedules in a 4,096-step window
        # left phase-out/growth running until evaluation and made the filter
        # systematically under-represent the annual policy.
        window_steps = int(steps)
        random_exploration_steps = min(512, max(64, window_steps // 8))
        initial_training_start_step = min(
            256,
            max(32, window_steps // 8),
        )
        policy_phaseout_steps = max(128, window_steps // 2)
        authority_growth_steps = max(128, window_steps // 2)
        actor_policy_warmup_steps = max(128, window_steps // 2)
        cloning_decay_start_step = min(
            512,
            max(64, window_steps // 8),
        )
        training_horizon_steps = 2 * window_steps
    teacher_label = str(teacher_label).strip().lower().replace("-", "_")
    if teacher_schedule is not None and teacher_label not in {"cost", "scorecard"}:
        raise ValueError("MATD3 V5 teacher_label must be 'cost' or 'scorecard'")
    recipe = (
        name
        if teacher_schedule is None
        else f"{name}_milp_{teacher_label}_teacher"
    )

    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"MATD3 V5 {recipe} {suffix} seed {seed}",
            "description": (
                "Corrected two-year SMART-residual MATD3 with aligned exposure, "
                "causal community context and episode-boundary model selection."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": recipe,
            "window": suffix,
            "training_years": "2",
            "deterministic_evaluation_years": "1",
            "residual_base": "RBCSmartPolicy",
            "residual_base_separate_from_bc_target": "True",
            "final_authority_first_reached_step": str(authority_growth_steps),
            "effective_storage_authority": str(
                float(variant["residual_final_scale"])
                * float(variant["storage_authority_multiplier"])
            ),
            "critic_team_reward_mix": str(
                float(variant["critic_team_reward_mix"])
            ),
            "episode_checkpoint_selection": str(steps is None),
            "promotion_eligible": "False",
        }
    )
    config["simulator"].update(
        {
            "episodes": 3,
            "deterministic_finish": True,
            "reward_function": "CostCommunityStorageResidualRewardV55",
            "reward_function_kwargs": {
                "community_settlement_cost_weight": float(variant["settlement_weight"]),
                "community_peak_import_penalty": float(variant["peak_weight"]),
                "community_ramping_penalty": float(variant["ramp_weight"]),
                "community_export_penalty": float(variant["export_weight"]),
                "community_emissions_penalty": float(
                    variant["emissions_weight"]
                ),
                "community_emissions_use_net_exchange": False,
                "community_penalty_use_net_exchange": True,
                "battery_throughput_penalty": float(variant["throughput_weight"]),
                "grid_violation_penalty": 120.0,
            },
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{recipe}-{suffix}-seed{seed}"
    )
    config["checkpointing"].update(
        {
            "checkpoint_interval": None,
            "checkpoint_on_episode_end": True,
            # Seasonal pilots are selection filters rather than artifacts to
            # deploy. Keep only their latest actor snapshot to avoid copying
            # hundreds of MB per candidate onto shared workers. Annual runs
            # retain both training-year boundaries for model selection.
            "keep_episode_checkpoints": steps is None,
        }
    )

    stage = config["pipeline"][0]
    stage["networks"]["actor"]["lr"] = 1.0e-4
    replay = stage["replay_buffer"]
    replay.update(
        {
            "behavior_action_priority_weight": 0.0,
            "behavior_action_priority_scope": "all",
            "observation_event_priority_mode": "ev_pv_price_peak",
        }
    )
    params = stage["exploration"]["params"]
    params.update(
        {
            "critic_team_reward_mix": float(variant["critic_team_reward_mix"]),
            "actor_community_context_enabled": True,
            "actor_community_context_features": [
                "community_net_power",
                "community_import_power",
                "community_export_power",
                "community_pv_power",
                "community_headroom",
                "community_export_headroom",
                "storage_soc_mean",
            ],
            "actor_frame_stack_steps": 4,
            "random_exploration_steps": random_exploration_steps,
            "end_initial_exploration_time_step": random_exploration_steps,
            "warm_start_policy_phaseout_steps": policy_phaseout_steps,
            "warm_start_policy_phaseout_mode": "blend",
            "train_during_initial_exploration": True,
            "initial_exploration_training_start_step": (
                initial_training_start_step
            ),
            "residual_action_start_step": 0,
            "residual_action_scale": 0.02,
            "residual_action_final_scale": float(variant["residual_final_scale"]),
            "residual_action_growth_steps": authority_growth_steps,
            "residual_storage_action_scale_multiplier": float(
                variant["storage_authority_multiplier"]
            ),
            "residual_ev_action_scale_multiplier": 0.0,
            "residual_deferrable_action_scale_multiplier": 0.0,
            "residual_building_gain_multipliers": {},
            "sigma": 0.08,
            "min_sigma": 0.01,
            "actor_policy_loss_weight": float(
                variant.get("actor_policy_weight", 0.30)
            ),
            "actor_policy_loss_warmup_weight": 0.05,
            "actor_policy_loss_warmup_start_step": 1024,
            "actor_policy_loss_warmup_steps": actor_policy_warmup_steps,
            "actor_behavior_cloning_source": "replay_action",
            "actor_behavior_cloning_weight": 0.0,
            "actor_behavior_cloning_min_weight": 0.0,
            "actor_behavior_cloning_extra_updates": 0,
            "actor_offline_bc_pretrain_steps": 0,
            "actor_storage_behavior_cloning_multiplier": 0.0,
            "actor_residual_delta_l2_penalty": float(
                variant["residual_penalty"]
            ),
            "actor_storage_smoothness_l2_penalty": float(
                variant["smoothness_weight"]
            ),
            "actor_storage_smoothness_deadband": 0.02,
            "local_action_safety_enabled": False,
        }
    )

    if teacher_schedule is not None:
        schedule_path = Path(teacher_schedule)
        params.update(
            {
                "actor_behavior_cloning_source": "teacher_policy",
                "actor_behavior_cloning_teacher_policy": (
                    "FixedServiceOracleReplayPolicy"
                ),
                "actor_behavior_cloning_teacher_action_scope": (
                    "residual_authority"
                ),
                "actor_behavior_cloning_clip_target_to_residual_authority": True,
                "actor_behavior_cloning_teacher_hyperparameters": {
                    "schedule_path": str(schedule_path),
                    "schedule_step_offset": int(start_step),
                    "service_policy": "RBCSmartPolicy",
                    "local_action_safety_enabled": False,
                },
                "actor_behavior_cloning_weight": float(
                    variant.get("teacher_bc_weight", 0.24)
                ),
                "actor_behavior_cloning_min_weight": float(
                    variant.get("teacher_bc_min_weight", 0.0)
                ),
                "actor_behavior_cloning_decay_start_step": (
                    cloning_decay_start_step
                ),
                "actor_behavior_cloning_decay_steps": min(
                    int(variant.get("teacher_bc_decay_steps", 33975)),
                    max(
                        1,
                        training_horizon_steps - cloning_decay_start_step,
                    ),
                ),
                "actor_behavior_cloning_extra_updates": 1,
                "actor_behavior_cloning_extra_update_start_step": (
                    initial_training_start_step
                ),
                "actor_behavior_cloning_extra_update_end_step": min(
                    int(
                        variant.get(
                            "teacher_bc_extra_update_end_step",
                            12288,
                        )
                    ),
                    training_horizon_steps - 1,
                ),
                "actor_offline_bc_pretrain_steps": int(
                    variant.get("teacher_offline_pretrain_steps", 64)
                ),
                "actor_offline_bc_pretrain_min_replay": 256,
                "actor_offline_bc_pretrain_weight": 0.50,
                "actor_ev_behavior_cloning_multiplier": 0.0,
                "actor_storage_behavior_cloning_multiplier": 1.0,
                "actor_deferrable_behavior_cloning_multiplier": 0.0,
            }
        )
        config["tracking"]["tags"].update(
            {
                "training_teacher": f"fixed_service_milp_{teacher_label}",
                "supervision_teacher": "FixedServiceOracleReplayPolicy",
                "teacher_perfect_foresight": "True",
                "evaluation_teacher_access": "False",
            }
        )
    else:
        config["tracking"]["tags"].update(
            {
                "training_teacher": "none",
                "supervision_teacher": "none",
                "teacher_perfect_foresight": "False",
            }
        )
    return config


def generate(
    output_dir: Path = OUTPUT_DIR,
    *,
    seed: int = 789,
    seasonal_pilots: bool = False,
    teacher_schedule: Path | None = None,
    scorecard_teacher_schedule: Path | None = None,
    functional_smoke_steps: int | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if functional_smoke_steps is not None:
        if seasonal_pilots:
            raise ValueError(
                "functional_smoke_steps cannot be combined with seasonal_pilots"
            )
        windows: list[tuple[int, int | None]] = [
            (0, int(functional_smoke_steps))
        ]
    else:
        windows = [(0, None)]
    if seasonal_pilots:
        windows.extend((start, 4096) for start in SEASONAL_START_STEPS)

    outputs: list[Path] = []
    for start_step, steps in windows:
        suffix = "annual" if steps is None else f"start{start_step}_steps{steps}"
        reference_path = output_dir / f"smart_v5_{suffix}_seed{seed}.yaml"
        reference_path.write_text(
            yaml.safe_dump(
                build_smart_reference_config(
                    seed=seed,
                    start_step=start_step,
                    steps=steps,
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(reference_path)
        for name in VARIANTS:
            path = output_dir / f"matd3_v5_{name}_{suffix}_seed{seed}.yaml"
            path.write_text(
                yaml.safe_dump(
                    build_config(
                        name,
                        seed=seed,
                        start_step=start_step,
                        steps=steps,
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            outputs.append(path)
        if teacher_schedule is not None:
            for name in ("cost_first_h2",):
                path = output_dir / (
                    f"matd3_v5_{name}_milp_cost_teacher_{suffix}_seed{seed}.yaml"
                )
                path.write_text(
                    yaml.safe_dump(
                        build_config(
                            name,
                            seed=seed,
                            start_step=start_step,
                            steps=steps,
                            teacher_schedule=teacher_schedule,
                            teacher_label="cost",
                        ),
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                outputs.append(path)
        if scorecard_teacher_schedule is not None:
            for name in (
                "balanced_h2",
                "ramp_guard_h2",
                "global_scorecard_h2",
                "global_distilled_h2",
            ):
                path = output_dir / (
                    f"matd3_v5_{name}_milp_scorecard_teacher_{suffix}_seed{seed}.yaml"
                )
                path.write_text(
                    yaml.safe_dump(
                        build_config(
                            name,
                            seed=seed,
                            start_step=start_step,
                            steps=steps,
                            teacher_schedule=scorecard_teacher_schedule,
                            teacher_label="scorecard",
                        ),
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                outputs.append(path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--seasonal-pilots", action="store_true")
    parser.add_argument("--teacher-schedule", type=Path)
    parser.add_argument("--scorecard-teacher-schedule", type=Path)
    parser.add_argument("--functional-smoke-steps", type=int)
    args = parser.parse_args()
    for path in generate(
        args.output_dir,
        seed=args.seed,
        seasonal_pilots=args.seasonal_pilots,
        teacher_schedule=args.teacher_schedule,
        scorecard_teacher_schedule=args.scorecard_teacher_schedule,
        functional_smoke_steps=args.functional_smoke_steps,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
