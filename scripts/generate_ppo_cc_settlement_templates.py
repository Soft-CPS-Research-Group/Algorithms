#!/usr/bin/env python3
"""Generate the four canonical annual settlement-on PPO/CC experiment templates."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "citylearn_three_phase_electrical_service_demo_15min_parquet"
DATASET_PATH = f"./datasets/{DATASET}/schema.json"
PPO_SEED = 789
SMART_SEED = 123
PPO_CHECKPOINT_ROOT = "./artifacts/frozen_ppo/annual_v1/seed789"
CHECKPOINT_NAME = "latest_checkpoint.pth"


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return payload


def _market() -> dict[str, Any]:
    return {
        "enabled": True,
        "local_price_ratio_to_grid_import": 0.8,
        "intra_community_sell_ratio": 0.8,
        "grid_export_price": 0.0,
        "import_member_weights": {},
        "kpis": {
            "community_local_traded_enabled": True,
            "community_self_consumption_enabled": True,
        },
    }


def _export(session_name: str) -> dict[str, Any]:
    return {
        "mode": "end",
        "export_kpis_on_episode_end": True,
        "final_episode_only": True,
        "kpis_final_episode_only": True,
        "timeseries_final_episode_only": True,
        "include_business_as_usual": True,
        "export_business_as_usual_timeseries": False,
        "kpi_round_decimals": None,
        "session_name": session_name,
    }


def _cc_reward() -> dict[str, Any]:
    return {
        "cost_aggregation": "community_net",
        "w_cost": 1.0,
        "w_peak": 0.6,
        "w_ramp": 0.4,
        "w_export": 0.05,
        "w_violation": 2.0,
    }


def _cc_stage() -> dict[str, Any]:
    return {
        "algorithm": "CCLevel1",
        "count": 1,
        "frozen": False,
        "hyperparameters": {
            "num_steps": 96,
            "lr": 0.0001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_epochs": 10,
            "mini_batch_size": 64,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.005,
            "max_grad_norm": 0.5,
            "target_kl": 0.1,
            "hidden_dims": [128, 128],
            "c_dim": 17,
            "cc_action_interval": 4,
            "price_min": 0.5,
            "price_max": 1.3,
            "reference_multiplier": 1.0,
            "policy_residual_scale": 1.0,
            "w_factor": 0.3,
            "w_smoothness": 0.1,
            "bc_pretrain_enabled": True,
            "bc_collect_steps": 8760,
            "bc_train_steps": 2000,
            "bc_lr": 0.001,
            "bc_price_p20": 0.1,
            "bc_price_p80": 0.21,
            "bc_w_cost": 1.0,
            "bc_w_peak": 0.6,
            "bc_w_ramp": 0.4,
            "bc_w_export": 0.05,
            "bc_w_violation": 2.0,
            "bc_w_headroom": 1.0,
            "bc_reference_headroom": 2.0,
            "bc_reference_ramping": 1.878,
            "bc_dt_hours": 0.25,
            "bc_mult_scale": 1.0,
        },
    }


def _fixed_stage() -> dict[str, Any]:
    return {
        "algorithm": "FixedPriceSignal",
        "count": 1,
        "frozen": True,
        "hyperparameters": {"multiplier": 1.0},
    }


def _signal_aware_stage() -> dict[str, Any]:
    return {
        "algorithm": "SignalAwareRBC",
        "count": 17,
        "frozen": True,
        "hyperparameters": {
            "seed": None,
            "pv_charge_threshold": 0.0,
            "flexibility_hours": 3.0,
            "emergency_hours": 1.0,
            "pv_preferred_charge_rate": 0.6,
            "flex_trickle_charge": 0.0,
            "min_charge_rate": 0.0,
            "emergency_charge_rate": 1.0,
            "energy_epsilon": 0.001,
            "default_capacity_kwh": 60.0,
            "non_flexible_chargers": [],
            "control_storage": True,
            "control_evs": True,
            "control_deferrables": True,
            "allow_v2g": True,
            "deferrable_start_action": 1.0,
            "deferrable_urgency_threshold": 0.75,
            "deferrable_slack_threshold": 0.25,
            "deferrable_priority_threshold": 0.5,
            "deferrable_safety_margin_steps": 1.0,
            "storage_min_soc": 0.2,
            "storage_max_soc": 0.9,
            "storage_target_soc": 0.5,
            "storage_charge_rate": 0.35,
            "storage_discharge_rate": 0.35,
            "price_charge_rate": 0.6,
            "price_discharge_rate": 0.45,
            "pv_charge_rate": 0.75,
            "peak_discharge_rate": 0.65,
            "storage_price_charge_soc_ceiling": 0.9,
            "storage_price_discharge_soc_floor": 0.2,
            "storage_peak_discharge_soc_floor": 0.2,
            "normal_storage_discharge_import_threshold_kw": 0.25,
            "storage_discharge_import_threshold_kw": 0.25,
            "ev_normal_charge_rate": 1.0,
            "ev_normal_target_soc": 1.0,
            "ev_price_charge_rate": 0.7,
            "ev_pv_charge_rate": 0.85,
            "ev_v2g_discharge_rate": 0.18,
            "ev_community_charge_rate": 0.85,
            "community_v2g_discharge_rate": 0.3,
            "community_storage_charge_rate": 0.75,
            "community_storage_discharge_rate": 0.65,
            "community_surplus_charge_soc_ceiling": 0.9,
            "community_surplus_threshold_kw": 0.25,
            "community_import_threshold_kw": 7.0,
            "community_local_price_ratio": 0.8,
            "community_grid_export_price": 0.0,
            "pv_surplus_threshold_kw": 0.25,
            "import_peak_threshold_kw": 7.0,
            "low_headroom_threshold_kw": 2.0,
            "ev_v2g_reserve_soc": 0.0,
            "ev_service_margin_rate": 0.05,
            "ev_service_floor_rate": 0.25,
            "ev_service_lookahead_hours": 4.0,
            "ev_service_target_soc": 0.0,
            "ev_deadline_buffer_hours": 0.25,
            "ev_v2g_min_departure_hours": 2.0,
            "ev_v2g_service_margin_soc": 0.02,
        },
        "networks": None,
        "replay_buffer": None,
        "exploration": None,
    }


def _tracking(tags: dict[str, Any]) -> dict[str, Any]:
    return {
        "mlflow_enabled": False,
        "tags": {key: str(value) for key, value in tags.items()},
        "log_level": "INFO",
        "log_frequency": 512,
        "mlflow_step_sample_interval": 512,
        "mlflow_artifacts_profile": "minimal",
        "progress_updates_enabled": True,
        "progress_update_interval": 128,
        "system_metrics_enabled": False,
        "system_metrics_interval": 32,
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
    }


def _configure_simulator(
    config: dict[str, Any],
    *,
    episodes: int,
    session_name: str,
    encoding_profile: str,
) -> None:
    simulator = config.setdefault("simulator", {})
    simulator.update(
        {
            "dataset_name": DATASET,
            "dataset_path": DATASET_PATH,
            "central_agent": False,
            "interface": "entity",
            "topology_mode": "static",
            "reward_function": "CCRewardLevel1",
            "reward_function_kwargs": _cc_reward(),
            "episodes": episodes,
            "deterministic_finish": True,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": 35039,
            "episode_time_steps": 35040,
            "export": _export(session_name),
            "entity_encoding": {
                "enabled": True,
                "normalization": "minmax_space",
                "profile": encoding_profile,
                "clip": True,
            },
            "community_market": _market(),
        }
    )


def _smart_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load(REPO_ROOT / "configs/templates/cc_local.yaml")
    cc_smart = copy.deepcopy(source)
    cc_smart["metadata"] = {
        "experiment_name": "ppo_cc_settlement_annual_v1",
        "run_name": "CC-SMART settlement annual seed 123",
        "community_name": "citylearn_static_15min",
        "description": "CCLevel1 over a frozen SignalAwareRBC leaf with community settlement enabled.",
    }
    cc_smart["tracking"] = _tracking(
        {
            "protocol": "ppo_cc_settlement_annual_v1",
            "controller": "cc_smart",
            "settlement": "enabled",
            "cc_seed": SMART_SEED,
            "leaf_frozen": True,
        }
    )
    cc_smart["checkpointing"] = {
        "resume_training": False,
        "checkpoint_run_id": None,
        "checkpoint_local_path": None,
        "stage_checkpoint_local_paths": {},
        "checkpoint_artifact": CHECKPOINT_NAME,
        "use_best_checkpoint_artifact": False,
        "reset_replay_buffer": True,
        "freeze_pretrained_layers": False,
        "fine_tune": False,
        "restore_optimizers": False,
        "restore_replay_buffer": False,
        "restore_exploration_state": False,
        "restore_reward_normalizer": False,
        "checkpoint_interval": 35040,
        "require_update_step": True,
        "require_initial_exploration_done": False,
    }
    _configure_simulator(
        cc_smart,
        episodes=4,
        session_name="cc-smart-settlement-annual-seed123",
        encoding_profile="cc_level1",
    )
    cc_smart["training"] = {
        "seed": SMART_SEED,
        "steps_between_training_updates": 1,
        "target_update_interval": 2,
    }
    cc_smart["pipeline"] = [_cc_stage(), _signal_aware_stage()]

    smart = copy.deepcopy(cc_smart)
    smart["metadata"].update(
        {
            "run_name": "SMART neutral settlement annual",
            "description": "Neutral fixed-price control over the exact frozen SignalAwareRBC leaf used by CC-SMART.",
        }
    )
    smart["tracking"] = _tracking(
        {
            "protocol": "ppo_cc_settlement_annual_v1",
            "controller": "smart_neutral",
            "settlement": "enabled",
            "leaf_frozen": True,
        }
    )
    smart["checkpointing"]["checkpoint_interval"] = None
    _configure_simulator(
        smart,
        episodes=1,
        session_name="smart-neutral-settlement-annual",
        encoding_profile="cc_level1",
    )
    smart["pipeline"] = [_fixed_stage(), _signal_aware_stage()]
    return smart, cc_smart


def _ppo_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load(REPO_ROOT / "configs/experiments/ppo_cc_scalar_safe_annual_seed789.yaml")
    cc_ppo = copy.deepcopy(source)
    cc_ppo["metadata"].update(
        {
            "experiment_name": "ppo_cc_settlement_annual_v1",
            "run_name": "CC-PPO settlement annual seed 789",
            "description": "CCLevel1 over seventeen deterministic frozen PPO residual-battery leaves with settlement enabled.",
        }
    )
    cc_ppo["tracking"] = _tracking(
        {
            "protocol": "ppo_cc_settlement_annual_v1",
            "controller": "cc_ppo",
            "settlement": "enabled",
            "cc_seed": PPO_SEED,
            "ppo_seed": PPO_SEED,
            "leaf_frozen": True,
            "leaf_community_blind": True,
        }
    )
    cc_ppo["checkpointing"].update(
        {
            "resume_training": True,
            "checkpoint_run_id": None,
            "checkpoint_local_path": None,
            "stage_checkpoint_local_paths": {1: PPO_CHECKPOINT_ROOT},
            "checkpoint_artifact": CHECKPOINT_NAME,
            "use_best_checkpoint_artifact": False,
            "reset_replay_buffer": True,
            "freeze_pretrained_layers": False,
            "fine_tune": False,
            "restore_optimizers": False,
            "restore_replay_buffer": False,
            "restore_exploration_state": False,
            "restore_reward_normalizer": False,
            "checkpoint_interval": 35040,
            "require_update_step": True,
            "require_initial_exploration_done": False,
        }
    )
    _configure_simulator(
        cc_ppo,
        episodes=4,
        session_name="cc-ppo-settlement-annual-seed789",
        encoding_profile="building_local_v1",
    )
    cc_ppo["training"] = {
        "seed": PPO_SEED,
        "steps_between_training_updates": 1,
        "target_update_interval": 0,
    }
    ppo_stage = copy.deepcopy(cc_ppo["pipeline"][1])
    ppo_stage["frozen"] = True
    cc_ppo["pipeline"] = [_cc_stage(), ppo_stage]

    ppo = copy.deepcopy(cc_ppo)
    ppo["metadata"].update(
        {
            "run_name": "PPO neutral settlement annual seed 789",
            "description": "Neutral fixed-price replay of the deterministic frozen PPO seed 789 leaf with settlement enabled.",
        }
    )
    ppo["tracking"] = _tracking(
        {
            "protocol": "ppo_cc_settlement_annual_v1",
            "controller": "ppo_neutral",
            "settlement": "enabled",
            "ppo_seed": PPO_SEED,
            "leaf_frozen": True,
            "leaf_community_blind": True,
        }
    )
    ppo["checkpointing"]["checkpoint_interval"] = None
    _configure_simulator(
        ppo,
        episodes=1,
        session_name="ppo-neutral-settlement-annual-seed789",
        encoding_profile="building_local_v1",
    )
    ppo["pipeline"] = [_fixed_stage(), copy.deepcopy(ppo_stage)]
    return ppo, cc_ppo


def generate(output_dir: Path) -> list[Path]:
    smart, cc_smart = _smart_configs()
    ppo, cc_ppo = _ppo_configs()
    payloads = {
        "smart_settlement_annual.yaml": smart,
        "cc_smart_settlement_annual_seed123.yaml": cc_smart,
        "ppo_settlement_annual_seed789.yaml": ppo,
        "cc_ppo_settlement_annual_seed789.yaml": cc_ppo,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, payload in payloads.items():
        path = output_dir / filename
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "configs/experiments/ppo_cc_settlement_annual_v1",
    )
    args = parser.parse_args()
    for path in generate(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
