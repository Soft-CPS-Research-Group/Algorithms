#!/usr/bin/env python3
"""Build Level-1 coordinator experiments over the frozen MATD3 V5 incumbent."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_BASE_CONFIG = Path(
    "configs/experiments/matd3_global_v5/"
    "matd3_v5_global_distilled_h2_milp_scorecard_teacher_annual_seed789.yaml"
)
DEFAULT_OUTPUT_DIR = Path("configs/experiments/cc_matd3_level1_v1")
PACKAGED_DATASET_NAME = (
    "citylearn_three_phase_electrical_service_demo_15min_parquet_matd3_v5"
)
CHECKPOINT_PATH = (
    f"/data/datasets/{PACKAGED_DATASET_NAME}/"
    "_artifacts/matd3_v5_global_distilled_seed789/latest_checkpoint.pth"
)

# P75 import and P90 cost/peak-ramp/export references measured from the exact
# annual seed-789 MATD3 V5 replay (job 6e282973-1fcb-4d89-8d27-d2e9a77f2367).
MATD3_REFERENCES = {
    "target_import": 5.728705388028175,
    "reference_cost": 1.526936944946647,
    "reference_peak": 11.614026761269214,
    "reference_ramping": 1.7581765375565737,
    "reference_export": 4.545535570383072,
    "reference_price": 0.16398,
}

PROFILES: Mapping[str, Mapping[str, float]] = {
    "cost_first": {
        "w_peak": 0.02,
        "w_ramp": 0.02,
        "w_export": 0.005,
        "w_smoothness": 0.001,
    },
    "balanced": {
        "w_peak": 0.15,
        "w_ramp": 0.15,
        "w_export": 0.02,
        "w_smoothness": 0.004,
    },
    "ramp_guarded": {
        "w_peak": 0.15,
        "w_ramp": 0.35,
        "w_export": 0.03,
        "w_smoothness": 0.008,
    },
}


def _manager_stage(profile: Mapping[str, float]) -> dict[str, Any]:
    return {
        "algorithm": "CCLevel1",
        "count": 1,
        "frozen": False,
        "hyperparameters": {
            "num_steps": 336,
            "lr": 1.0e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_epochs": 4,
            "mini_batch_size": 96,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.002,
            "max_grad_norm": 0.5,
            "target_kl": 0.05,
            "hidden_dims": [128, 128],
            "c_dim": 17,
            "cc_action_interval": 4,
            "price_min": 0.5,
            "price_max": 1.3,
            "initial_log_std": -2.0,
            "reference_multiplier": 1.0,
            "policy_residual_scale": 1.0,
            "w_factor": 0.0,
            "w_smoothness": profile["w_smoothness"],
            "bc_pretrain_enabled": True,
            "bc_collect_steps": 8760,
            "bc_train_steps": 2000,
            "bc_train_chunk_steps": 256,
            "bc_lr": 1.0e-3,
            "bc_w_cost": 1.0,
            "bc_w_peak": profile["w_peak"],
            "bc_w_ramp": profile["w_ramp"],
            "bc_w_export": profile["w_export"],
            "bc_w_violation": 2.0,
            "bc_w_headroom": 0.3,
            "bc_reference_headroom": 2.0,
            "bc_target_import": MATD3_REFERENCES["target_import"],
            "bc_reference_peak": MATD3_REFERENCES["reference_peak"],
            "bc_reference_ramping": MATD3_REFERENCES["reference_ramping"],
            "bc_reference_export": MATD3_REFERENCES["reference_export"],
            "bc_reference_price": MATD3_REFERENCES["reference_price"],
            "bc_dt_hours": 0.25,
            "bc_mult_scale": 1.0,
        },
    }


def _common_config(base: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    config = deepcopy(dict(base))
    config["metadata"].update(
        {
            "experiment_name": "cc_matd3_level1_v1",
            "run_name": f"CC-L1 over frozen MATD3 V5 {label} seed 789",
            "description": (
                "Level-1 scalar price coordinator over the frozen annual MATD3 "
                "V5 global-distilled incumbent. The leaf checkpoint is immutable; "
                "only the current price fields observed by its actors are conditioned."
            ),
        }
    )
    config["tracking"].update(
        {
            "mlflow_enabled": False,
            "log_frequency": 512,
            "mlflow_step_sample_interval": 512,
            "progress_updates_enabled": True,
            "progress_update_interval": 128,
            "system_metrics_enabled": False,
            "action_diagnostics_enabled": True,
            "action_diagnostics_detail": "summary",
            "training_diagnostics_enabled": True,
            "training_diagnostics_detail": "summary",
            "reward_diagnostics_enabled": True,
            "reward_diagnostics_detail": "summary",
            "runtime_profiling_enabled": True,
            "runtime_profiling_interval": 512,
            "runtime_profiling_detail": "summary",
            "stall_watchdog_enabled": True,
            "stall_watchdog_timeout_seconds": 3600.0,
            "stall_watchdog_exit_on_timeout": True,
            "stall_watchdog_repeat": False,
            "stall_watchdog_context_interval_steps": 64,
            "resource_guard_enabled": True,
            "max_process_rss_mb": 88000.0,
            "min_available_ram_mb": 2048.0,
            # A full CC policy update at an annual episode boundary has taken
            # roughly 6 minutes on Union. Keep the genuine stall watchdog at
            # one hour, but do not misclassify that bounded update as a failed
            # simulator step.
            "max_step_seconds": 900.0,
        }
    )
    config["tracking"]["tags"] = {
        "protocol": "cc_matd3_level1_v1",
        "controller": "cc_level1_over_frozen_matd3",
        "leaf_recipe": "global_distilled_h2_milp_scorecard_teacher",
        "leaf_checkpoint_job": "6e282973-1fcb-4d89-8d27-d2e9a77f2367",
        "leaf_frozen": "True",
        "canonical_dataset": "citylearn_three_phase_electrical_service_demo_15min_parquet",
        "dataset_packaging": "canonical_dataset_plus_frozen_matd3_checkpoint",
        "settlement": "enabled",
        "settlement_reward": "exact_member_settlement",
        "cc_level": "1",
        "cc_action_interval": "4",
        "price_range": "0.5_1.3",
        "price_forecasts": "real_unmodified",
        "evidence_horizon": "full_year",
        "recipe": label,
    }

    checkpointing = config["checkpointing"]
    checkpointing.update(
        {
            "resume_training": True,
            "checkpoint_run_id": None,
            "checkpoint_local_path": None,
            "stage_checkpoint_local_paths": {1: CHECKPOINT_PATH},
            "checkpoint_artifact": "latest_checkpoint.pth",
            "checkpoint_mode": "inference",
            "use_best_checkpoint_artifact": False,
            "reset_replay_buffer": True,
            "freeze_pretrained_layers": False,
            "fine_tune": False,
            "restore_optimizers": False,
            "restore_replay_buffer": False,
            "restore_exploration_state": False,
            "restore_reward_normalizer": False,
            "checkpoint_interval": None,
            "require_update_step": True,
            "require_initial_exploration_done": False,
            "checkpoint_on_episode_end": True,
            "keep_episode_checkpoints": False,
        }
    )

    simulator = config["simulator"]
    simulator.update(
        {
            "dataset_name": PACKAGED_DATASET_NAME,
            "dataset_path": f"./datasets/{PACKAGED_DATASET_NAME}/schema.json",
            "simulation_start_time_step": 0,
            "simulation_end_time_step": 35039,
            "episode_time_steps": 35040,
            "deterministic_finish": True,
            "repeat_episode_scenario": True,
        }
    )
    simulator["community_market"].update(
        {
            "enabled": True,
            "local_price_ratio_to_grid_import": 0.8,
            "intra_community_sell_ratio": 0.8,
            "grid_export_price": 0.0,
        }
    )
    simulator["export"].update(
        {
            "mode": "end",
            "export_kpis_on_episode_end": True,
            "final_episode_only": True,
            "kpis_final_episode_only": True,
            "timeseries_final_episode_only": True,
            "include_business_as_usual": True,
            "export_business_as_usual_timeseries": False,
            "session_name": f"cc-matd3-l1-v1-{label.replace('_', '-')}-seed789",
        }
    )

    leaf = deepcopy(config["pipeline"][0])
    leaf["frozen"] = True
    leaf["hyperparameters"]["require_cuda"] = False
    leaf_params = leaf["exploration"]["params"]
    leaf_params["local_price_conditioning_enabled"] = True
    leaf_params["local_price_forecast_mode"] = "real_unmodified"
    config["pipeline"] = [leaf]
    return config


def _fixed_config(base: Mapping[str, Any], *, multiplier: float) -> dict[str, Any]:
    label = f"fixed_{str(multiplier).replace('.', 'p')}"
    config = _common_config(base, label=label)
    config["metadata"]["description"] = (
        "Frozen MATD3 V5 causal price-response replay. This is a diagnostic "
        "control, not a learned coordinator."
    )
    config["tracking"]["tags"]["controller"] = "fixed_price_over_frozen_matd3"
    config["tracking"]["tags"]["fixed_multiplier"] = str(multiplier)
    config["simulator"]["episodes"] = 1
    leaf = config["pipeline"][0]
    config["pipeline"] = [
        {
            "algorithm": "FixedPriceSignal",
            "count": 1,
            "frozen": True,
            "hyperparameters": {"multiplier": float(multiplier)},
        },
        leaf,
    ]
    return config


def _smoke_config(base: Mapping[str, Any]) -> dict[str, Any]:
    """Build a short replay that exercises checkpoint and CC-to-leaf routing."""
    config = _fixed_config(base, multiplier=1.0)
    config["metadata"].update(
        {
            "run_name": "CC-L1 over frozen MATD3 V5 checkpoint smoke seed 789",
            "description": (
                "Operational 96-step smoke for the frozen MATD3 checkpoint and "
                "the Level-1 price-conditioning route. Not scorecard evidence."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "evidence_horizon": "smoke_96_steps",
            "recipe": "fixed_1p0_smoke96",
        }
    )
    config["simulator"].update(
        {
            "simulation_end_time_step": 95,
            "episode_time_steps": 96,
            "repeat_episode_scenario": False,
        }
    )
    config["simulator"]["export"].update(
        {
            "include_business_as_usual": False,
            "session_name": "cc-matd3-l1-v1-checkpoint-smoke-seed789",
        }
    )
    config["checkpointing"]["checkpoint_on_episode_end"] = False
    return config


def _learned_config(
    base: Mapping[str, Any], *, profile_name: str, profile: Mapping[str, float]
) -> dict[str, Any]:
    config = _common_config(base, label=profile_name)
    config["metadata"]["description"] = (
        "Trainable Level-1 scalar price coordinator over the frozen annual "
        "MATD3 V5 global-distilled incumbent, optimized against exact community "
        "settlement and the declared physical profile."
    )
    config["tracking"]["tags"]["objective_profile"] = profile_name
    simulator = config["simulator"]
    simulator["episodes"] = 6
    simulator["reward_function"] = "CCRewardLevel1"
    simulator["reward_function_kwargs"] = {
        "cost_aggregation": "community_settled",
        "community_local_price_ratio": 0.8,
        "community_grid_export_price": 0.0,
        "w_cost": 1.0,
        "w_peak": profile["w_peak"],
        "w_ramp": profile["w_ramp"],
        "w_export": profile["w_export"],
        "w_violation": 2.0,
        "target_import": MATD3_REFERENCES["target_import"],
        "reference_cost": MATD3_REFERENCES["reference_cost"],
        "reference_peak": MATD3_REFERENCES["reference_peak"],
        "reference_ramping": MATD3_REFERENCES["reference_ramping"],
        "reference_export": MATD3_REFERENCES["reference_export"],
        "reference_violation": 1.0,
    }
    leaf = config["pipeline"][0]
    config["pipeline"] = [_manager_stage(profile), leaf]
    return config


def build_configs(base_config: Path, output_dir: Path) -> list[Path]:
    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    built: list[tuple[str, dict[str, Any]]] = []
    built.append(
        (
            "cc_matd3_l1_fixed_1p0_smoke96_seed789.yaml",
            _smoke_config(base),
        )
    )
    for multiplier in (1.0, 0.85, 1.15):
        name = f"cc_matd3_l1_fixed_{str(multiplier).replace('.', 'p')}_seed789.yaml"
        built.append((name, _fixed_config(base, multiplier=multiplier)))
    for profile_name, profile in PROFILES.items():
        name = f"cc_matd3_l1_{profile_name}_seed789.yaml"
        built.append(
            (
                name,
                _learned_config(
                    base,
                    profile_name=profile_name,
                    profile=profile,
                ),
            )
        )

    paths = []
    for name, config in built:
        path = output_dir / name
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in build_configs(args.base_config, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
