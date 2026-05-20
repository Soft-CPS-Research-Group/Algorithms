"""Run a short comparable Phase 6A benchmark matrix.

The goal is not to produce final KPI claims. It is to make baseline/MADDPG
comparisons repeatable with one generated config per run and one aggregate table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_experiment import run_experiment
from scripts.bechmark_agents import DEFAULT_KPIS, _load_job_record
from utils.config_schema import validate_config


DATASET_CONFIGS: dict[str, dict[str, str]] = {
    "15s": {
        "random": "configs/templates/baselines/random_local.yaml",
        "normal_no_battery": "configs/templates/baselines/normal_no_battery_local.yaml",
        "normal": "configs/templates/baselines/normal_local.yaml",
        "rbc_basic": "configs/templates/baselines/rbc_basic_local.yaml",
        "rbc_smart": "configs/templates/baselines/rbc_smart_local.yaml",
        "maddpg": "configs/templates/maddpg/maddpg_local.yaml",
    },
    "2022": {
        "random": "configs/templates/baselines/random_2022_all_plus_evs_local.yaml",
        "normal_no_battery": "configs/templates/baselines/normal_no_battery_2022_all_plus_evs_local.yaml",
        "normal": "configs/templates/baselines/normal_2022_all_plus_evs_local.yaml",
        "rbc_basic": "configs/templates/baselines/rbc_basic_2022_all_plus_evs_local.yaml",
        "rbc_smart": "configs/templates/baselines/rbc_smart_2022_all_plus_evs_local.yaml",
        "maddpg": "configs/templates/maddpg/maddpg_2022_all_plus_evs_local.yaml",
    },
}

DEFAULT_DATASETS = ("15s", "2022")
DEFAULT_AGENTS = ("random", "normal_no_battery", "normal", "rbc_basic", "rbc_smart", "maddpg")
DEFAULT_MADDPG_VARIANTS = ("current",)

SELECTED_METRICS = (
    "RewardComponent/reward_total_mean",
    "RewardComponent/hard_constraint_penalty_mean",
    "RewardComponent/local_import_cost_mean",
    "RewardComponent/local_import_energy_mean",
    "RewardComponent/local_export_energy_mean",
    "RewardComponent/ev_service_penalty_mean",
    "RewardComponent/ev_schedule_deficit_penalty_mean",
    "RewardComponent/ev_v2g_service_abuse_penalty_mean",
    "RewardComponent/ev_v2g_discharge_kwh_sum_mean",
    "RewardComponent/ev_soc_over_service_sum_mean",
    "RewardComponent/ev_over_service_penalty_amount_mean",
    "RewardComponent/ev_departure_window_penalty_mean",
    "RewardComponent/ev_departure_missed_penalty_amount_mean",
    "RewardComponent/deferrable_service_penalty_mean",
    "RewardComponent/battery_safety_penalty_mean",
    "RewardComponent/battery_throughput_penalty_mean",
    "RewardComponent/community_settlement_cost_mean",
    "RewardComponent/community_settlement_reward_mean",
    "RewardComponent/community_local_import_energy_mean",
    "RewardComponent/community_local_export_energy_mean",
    "RewardComponent/community_import_penalty_mean",
    "RewardComponent/community_peak_import_penalty_mean",
    "Action/all_mean",
    "Action/all_std",
    "Action/near_low_fraction",
    "Action/near_high_fraction",
    "Action/storage_positive_fraction",
    "Action/storage_negative_fraction",
    "Action/storage_idle_fraction",
    "Action/ev_positive_fraction",
    "Action/ev_negative_fraction",
    "Action/ev_idle_fraction",
    "Action/deferrable_on_fraction",
    "Action/deferrable_off_fraction",
    "Deferrable/start_delay_steps_mean",
    "MADDPG/average_critic_loss",
    "MADDPG/average_actor_loss",
    "MADDPG/actor_update_performed",
    "MADDPG/actor_policy_loss_mean",
    "MADDPG/actor_policy_loss_weighted_mean",
    "MADDPG/actor_policy_loss_effective_weight",
    "MADDPG/actor_regularization_loss_mean",
    "MADDPG/actor_action_l2_mean",
    "MADDPG/actor_action_saturation_excess_mean",
    "MADDPG/actor_storage_action_l2_mean",
    "MADDPG/actor_ev_v2g_action_l2_mean",
    "MADDPG/actor_behavior_cloning_loss_mean",
    "MADDPG/actor_behavior_cloning_ev_loss_mean",
    "MADDPG/actor_behavior_cloning_storage_loss_mean",
    "MADDPG/actor_behavior_cloning_regularization_mean",
    "MADDPG/actor_behavior_cloning_effective_weight",
    "MADDPG/actor_behavior_cloning_source_warm_start_policy",
    "MADDPG/actor_ev_behavior_cloning_multiplier",
    "MADDPG/actor_ev_behavior_cloning_zero_target_weight",
    "MADDPG/actor_storage_behavior_cloning_multiplier",
    "MADDPG/reward_raw_std",
    "MADDPG/reward_train_std",
    "MADDPG/critic_loss_huber",
    "MADDPG/critic_target_clip_abs",
    "MADDPG/replay_buffer_size",
    "MADDPG/replay_priority_fraction",
    "MADDPG/replay_priority_alpha",
    "MADDPG/replay_priority_max",
    "MADDPG/replay_behavior_action_priority_weight",
    "MADDPG/replay_behavior_action_priority_scope_ev",
    "MADDPG/replay_observation_event_priority_last",
    "MADDPG/exploration_sigma",
    "MADDPG/storage_exploration_noise_multiplier",
    "MADDPG/ev_negative_exploration_noise_multiplier",
    "MADDPG/warm_start_policy_phaseout_probability",
    "MADDPG/warm_start_policy_phaseout_used",
    "MADDPG/warm_start_policy_phaseout_mode_blend",
)


def _rbc_smart_learning_teacher_hyperparameters() -> dict[str, Any]:
    """Softer RBCSmart teacher used for behavior cloning, not as a baseline claim.

    The benchmark baseline remains the dataset-specific RBCSmart template. This
    profile is intentionally less aggressive on EV service windows so the actor
    can learn when to stop charging instead of cloning full-rate service too
    often.
    """

    return {
        "allow_v2g": False,
        "flexibility_hours": 3.0,
        "emergency_hours": 1.0,
        "flex_trickle_charge": 0.0,
        "emergency_charge_rate": 1.0,
        "ev_price_charge_rate": 0.70,
        "ev_pv_charge_rate": 0.85,
        "ev_v2g_discharge_rate": 0.30,
        "ev_v2g_reserve_soc": 0.15,
        "ev_service_margin_rate": 0.05,
        "ev_service_floor_rate": 0.25,
        "ev_service_lookahead_hours": 4.0,
        "ev_service_target_soc": 0.0,
        "ev_deadline_buffer_hours": 0.25,
        "ev_v2g_min_departure_hours": 2.0,
        "ev_v2g_service_margin_soc": 0.05,
        "price_charge_rate": 0.60,
        "price_discharge_rate": 0.45,
        "pv_charge_rate": 0.75,
        "peak_discharge_rate": 0.65,
        "storage_min_soc": 0.20,
        "storage_max_soc": 0.90,
        "storage_target_soc": 0.50,
        "storage_price_charge_soc_ceiling": 0.90,
        "storage_price_discharge_soc_floor": 0.20,
        "storage_peak_discharge_soc_floor": 0.20,
        "pv_surplus_threshold_kw": 0.25,
        "import_peak_threshold_kw": 7.0,
        "low_headroom_threshold_kw": 2.0,
        "deferrable_start_action": 1.0,
        "deferrable_urgency_threshold": 0.60,
        "deferrable_slack_threshold": 0.40,
        "deferrable_safety_margin_steps": 1.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally run a short Phase 6A comparison matrix "
            "for baselines and MADDPG."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to runs/benchmarks/phase6a_<timestamp>.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_CONFIGS),
        default=[],
        help="Dataset key to include. Can be repeated. Defaults to 15s and 2022.",
    )
    parser.add_argument(
        "--agent",
        action="append",
        choices=sorted(next(iter(DATASET_CONFIGS.values())).keys()),
        default=[],
        help="Agent/baseline key to include. Can be repeated. Defaults to the full Phase 6A set.",
    )
    parser.add_argument(
        "--maddpg-variant",
        action="append",
        choices=(
            "current",
            "v1",
            "noop_centered",
            "noop_actor",
            "warm_rbc_basic",
            "warm_rbc_smart",
            "per_agent_critic",
            "anti_saturation",
            "anti_saturation_warm_rbc_basic",
            "anti_saturation_warm_rbc_smart",
            "ev_priority_bc_warm_rbc_basic",
            "ev_service_v2g_guard_warm_rbc_basic",
            "ev_service_v2g_guard_prioritized_warm_rbc_basic",
            "service_guard_v2_warm_rbc_basic",
            "service_guard_v2_prioritized_low_warm_rbc_basic",
            "service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic",
            "cost_balanced_v3_warm_rbc_basic",
            "cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic",
            "community_band_v4_warm_rbc_basic",
            "community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic",
            "community_storage_band_v41_regularized_warm_rbc_basic",
            "community_storage_band_v41_prioritized_regularized_warm_rbc_basic",
            "community_service_band_v42_regularized_warm_rbc_basic",
            "community_service_band_v42_prioritized_regularized_warm_rbc_basic",
            "community_service_band_v42_prioritized_regularized_warm_rbc_smart",
            "community_service_band_v42_prioritized_warmtrain_rbc_smart",
            "community_service_band_v42_prioritized_warmtrain_phaseout_rbc_smart",
            "community_battery_value_v43_prioritized_warmtrain_phaseout_rbc_smart",
            "community_battery_value_v43_teacher_bc_stable_rbc_smart",
            "community_smooth_service_v44_stable_teacher_bc_rbc_smart",
            "community_feasible_service_v45_stable_teacher_bc_rbc_smart",
            "community_feasible_service_v45_teacher_clone_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_event_rbc_smart",
            "community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_event_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_guarded_band_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_band_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_balanced_rbc_smart",
            "community_feasible_service_v45_actor_pretrain_rbc_smart",
            "community_feasible_service_v45_actor_pretrain_slow_rbc_smart",
        ),
        default=[],
        help="MADDPG variant to include when --agent maddpg is selected. Can be repeated.",
    )
    parser.add_argument("--seed", action="append", type=int, default=[], help="Seed. Can be repeated.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per generated run.")
    parser.add_argument(
        "--deterministic-finish",
        action="store_true",
        help="Run the final episode deterministically without learning updates.",
    )
    parser.add_argument("--steps", type=int, default=128, help="Default steps per episode for short windows.")
    parser.add_argument("--steps-15s", type=int, default=None, help="Override short-window steps for 15s dataset.")
    parser.add_argument("--steps-2022", type=int, default=None, help="Override short-window steps for 2022 dataset.")
    parser.add_argument("--start", type=int, default=0, help="Simulation start time step for short windows.")
    parser.add_argument(
        "--full-window",
        action="store_true",
        help="Use template simulation windows instead of overriding start/end/steps.",
    )
    parser.add_argument(
        "--no-kpi-export",
        action="store_true",
        help="Disable simulator KPI export. Useful for very fast debug runs.",
    )
    parser.add_argument("--metric-interval", type=int, default=16, help="Step metric sampling interval.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Optional checkpoint interval for generated runs. Uses template default when omitted.",
    )
    parser.add_argument(
        "--reward-diagnostics-detail",
        choices=("summary", "per_agent"),
        default="summary",
        help="Reward component logging detail. Use per_agent for targeted reward audits.",
    )
    parser.add_argument(
        "--reward-function-override",
        default=None,
        help="Optional reward function name to force in every generated run for diagnostics.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="MADDPG replay batch size for generated runs.")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="MADDPG replay buffer capacity.")
    parser.add_argument("--actor-layers", default="64,32", help="Comma-separated MADDPG actor layers.")
    parser.add_argument("--critic-layers", default="128,64", help="Comma-separated MADDPG critic layers.")
    parser.add_argument("--random-exploration-steps", type=int, default=32, help="MADDPG warm-up steps.")
    parser.add_argument(
        "--warm-start-phaseout-steps",
        type=int,
        default=None,
        help=(
            "Optional MADDPG warm-start phase-out length after warm-up. "
            "Only applies to policy warm-start variants."
        ),
    )
    parser.add_argument("--sigma", type=float, default=0.15, help="MADDPG Gaussian exploration sigma.")
    parser.add_argument(
        "--hyperparameter-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override algorithm.hyperparameters entries in generated configs. "
            "Can be repeated; values are parsed as bool/int/float when possible."
        ),
    )
    parser.add_argument(
        "--exploration-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override algorithm.exploration.params entries in generated MADDPG configs. "
            "Can be repeated; values are parsed as bool/int/float when possible."
        ),
    )
    parser.add_argument(
        "--warm-start-hyperparameter-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override exploration.params.warm_start_policy_hyperparameters entries. "
            "Useful for testing a learning teacher without changing the baseline template."
        ),
    )
    parser.add_argument(
        "--replay-buffer-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override algorithm.replay_buffer entries in generated MADDPG configs. "
            "Can be repeated; values are parsed as bool/int/float when possible."
        ),
    )
    parser.add_argument(
        "--reward-kwarg-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override simulator.reward_function_kwargs entries in generated configs. "
            "Can be repeated; values are parsed as bool/int/float when possible."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate configs and benchmark matrix; do not run experiments.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failed experiment instead of recording failure and continuing.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "item"


def _parse_layers(raw: str) -> list[int]:
    layers = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not layers:
        raise ValueError("Network layer list cannot be empty.")
    if any(layer <= 0 for layer in layers):
        raise ValueError(f"Network layers must be positive integers: {layers}")
    return layers


def _parse_scalar_override(raw: str) -> Any:
    value = str(raw).strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if re.fullmatch(r"[+-]?\d+", value):
            return int(value)
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", value):
            return float(value)
    except ValueError:
        return value
    return value


def _parse_key_value_overrides(raw_values: Iterable[str], *, label: str = "override") -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for raw in raw_values or []:
        if "=" not in str(raw):
            raise ValueError(f"Invalid {label} {raw!r}; expected KEY=VALUE.")
        key, value = str(raw).split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid {label} {raw!r}; key cannot be empty.")
        overrides[key] = _parse_scalar_override(value)
    return overrides


def _parse_hyperparameter_overrides(raw_values: Iterable[str]) -> dict[str, Any]:
    return _parse_key_value_overrides(raw_values, label="hyperparameter override")


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _baseline_agent_key_for_policy(policy_name: str) -> str | None:
    return {
        "RandomPolicy": "random",
        "NormalNoBatteryPolicy": "normal_no_battery",
        "NormalPolicy": "normal",
        "RBCBasicPolicy": "rbc_basic",
        "RBCSmartPolicy": "rbc_smart",
    }.get(str(policy_name))


def _load_baseline_hyperparameters(dataset_key: str, policy_name: str) -> dict[str, Any]:
    agent_key = _baseline_agent_key_for_policy(policy_name)
    if not agent_key:
        return {}
    template = DATASET_CONFIGS.get(dataset_key, {}).get(agent_key)
    if not template:
        return {}
    baseline_config = validate_config(_load_yaml(Path(template))).to_dict()
    return dict((baseline_config.get("algorithm") or {}).get("hyperparameters") or {})


def _selected_steps(dataset_key: str, args: argparse.Namespace) -> int:
    if dataset_key == "15s" and args.steps_15s is not None:
        return max(int(args.steps_15s), 1)
    if dataset_key == "2022" and args.steps_2022 is not None:
        return max(int(args.steps_2022), 1)
    return max(int(args.steps), 1)


def _apply_maddpg_variant(config: dict[str, Any], variant: str) -> None:
    simulator = config.setdefault("simulator", {})
    encoding = simulator.setdefault("entity_encoding", {})
    algorithm = config.setdefault("algorithm", {})
    exploration = algorithm.setdefault("exploration", {}).setdefault("params", {})

    if variant == "current":
        return

    if variant == "v1":
        encoding["enabled"] = True
        encoding["normalization"] = "minmax_space"
        encoding["profile"] = "maddpg_v1"
        encoding["clip"] = True
        return

    if variant == "noop_centered":
        exploration["initial_exploration_strategy"] = "noop_centered"
        exploration["noop_noise_scale"] = 0.12
        exploration["deferrable_on_probability"] = 0.2
        exploration["deferrable_trigger_threshold"] = 0.5
        return

    if variant == "noop_actor":
        exploration["noop_actor_initialization"] = True
        exploration["noop_actor_initialization_epsilon"] = 0.05
        return

    if variant == "warm_rbc_basic":
        exploration["initial_exploration_strategy"] = "policy"
        exploration["warm_start_policy"] = "RBCBasicPolicy"
        exploration["warm_start_policy_deterministic"] = True
        exploration["warm_start_policy_noise_scale"] = 0.0
        return

    if variant == "warm_rbc_smart":
        exploration["initial_exploration_strategy"] = "policy"
        exploration["warm_start_policy"] = "RBCSmartPolicy"
        exploration["warm_start_policy_deterministic"] = True
        exploration["warm_start_policy_noise_scale"] = 0.0
        return

    if variant == "per_agent_critic":
        exploration["critic_update_mode"] = "per_agent"
        return

    if variant in {
        "anti_saturation",
        "anti_saturation_warm_rbc_basic",
        "anti_saturation_warm_rbc_smart",
        "ev_priority_bc_warm_rbc_basic",
        "ev_service_v2g_guard_warm_rbc_basic",
        "ev_service_v2g_guard_prioritized_warm_rbc_basic",
        "service_guard_v2_warm_rbc_basic",
        "service_guard_v2_prioritized_low_warm_rbc_basic",
        "service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic",
        "cost_balanced_v3_warm_rbc_basic",
        "cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic",
        "community_band_v4_warm_rbc_basic",
        "community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic",
        "community_storage_band_v41_regularized_warm_rbc_basic",
        "community_storage_band_v41_prioritized_regularized_warm_rbc_basic",
        "community_service_band_v42_regularized_warm_rbc_basic",
        "community_service_band_v42_prioritized_regularized_warm_rbc_basic",
        "community_service_band_v42_prioritized_regularized_warm_rbc_smart",
        "community_service_band_v42_prioritized_warmtrain_rbc_smart",
        "community_service_band_v42_prioritized_warmtrain_phaseout_rbc_smart",
        "community_battery_value_v43_prioritized_warmtrain_phaseout_rbc_smart",
        "community_battery_value_v43_teacher_bc_stable_rbc_smart",
        "community_smooth_service_v44_stable_teacher_bc_rbc_smart",
        "community_feasible_service_v45_stable_teacher_bc_rbc_smart",
        "community_feasible_service_v45_teacher_clone_rbc_smart",
        "community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart",
        "community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart",
        "community_feasible_service_v45_teacher_clone_ev_learning_teacher_event_rbc_smart",
        "community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart",
        "community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart",
        "community_feasible_service_v45_teacher_clone_ev_focus_event_rbc_smart",
        "community_feasible_service_v45_teacher_clone_ev_guarded_band_rbc_smart",
        "community_feasible_service_v45_teacher_clone_ev_band_rbc_smart",
        "community_feasible_service_v45_teacher_clone_ev_balanced_rbc_smart",
        "community_feasible_service_v45_actor_pretrain_rbc_smart",
        "community_feasible_service_v45_actor_pretrain_slow_rbc_smart",
    }:
        exploration["initial_exploration_strategy"] = "noop_centered"
        exploration["noop_noise_scale"] = 0.10
        exploration["deferrable_on_probability"] = 0.2
        exploration["deferrable_trigger_threshold"] = 0.5
        exploration["noop_actor_initialization"] = True
        exploration["noop_actor_initialization_epsilon"] = 0.05
        exploration["critic_update_mode"] = "per_agent"
        exploration["actor_update_interval"] = 2
        exploration["target_policy_smoothing"] = True
        exploration["target_policy_noise"] = 0.05
        exploration["target_policy_noise_clip"] = 0.10
        exploration["actor_action_l2_penalty"] = 0.002
        exploration["actor_action_saturation_penalty"] = 0.020
        exploration["actor_action_saturation_threshold"] = 0.85
        if variant in {
            "anti_saturation_warm_rbc_basic",
            "ev_priority_bc_warm_rbc_basic",
            "ev_service_v2g_guard_warm_rbc_basic",
            "ev_service_v2g_guard_prioritized_warm_rbc_basic",
            "service_guard_v2_warm_rbc_basic",
            "service_guard_v2_prioritized_low_warm_rbc_basic",
            "service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic",
            "cost_balanced_v3_warm_rbc_basic",
            "cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic",
            "community_band_v4_warm_rbc_basic",
            "community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic",
            "community_storage_band_v41_regularized_warm_rbc_basic",
            "community_storage_band_v41_prioritized_regularized_warm_rbc_basic",
            "community_service_band_v42_regularized_warm_rbc_basic",
            "community_service_band_v42_prioritized_regularized_warm_rbc_basic",
        }:
            exploration["initial_exploration_strategy"] = "policy"
            exploration["warm_start_policy"] = "RBCBasicPolicy"
            exploration["warm_start_policy_deterministic"] = True
            exploration["warm_start_policy_noise_scale"] = 0.0
        elif variant in {
            "anti_saturation_warm_rbc_smart",
            "community_service_band_v42_prioritized_regularized_warm_rbc_smart",
            "community_service_band_v42_prioritized_warmtrain_rbc_smart",
            "community_service_band_v42_prioritized_warmtrain_phaseout_rbc_smart",
            "community_battery_value_v43_prioritized_warmtrain_phaseout_rbc_smart",
            "community_battery_value_v43_teacher_bc_stable_rbc_smart",
            "community_smooth_service_v44_stable_teacher_bc_rbc_smart",
            "community_feasible_service_v45_stable_teacher_bc_rbc_smart",
            "community_feasible_service_v45_teacher_clone_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_event_rbc_smart",
            "community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_event_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_guarded_band_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_band_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_balanced_rbc_smart",
            "community_feasible_service_v45_actor_pretrain_rbc_smart",
            "community_feasible_service_v45_actor_pretrain_slow_rbc_smart",
        }:
            exploration["initial_exploration_strategy"] = "policy"
            exploration["warm_start_policy"] = "RBCSmartPolicy"
            exploration["warm_start_policy_deterministic"] = True
            exploration["warm_start_policy_noise_scale"] = 0.0
        if variant in {
            "ev_priority_bc_warm_rbc_basic",
            "ev_service_v2g_guard_warm_rbc_basic",
            "ev_service_v2g_guard_prioritized_warm_rbc_basic",
            "service_guard_v2_warm_rbc_basic",
            "service_guard_v2_prioritized_low_warm_rbc_basic",
            "service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic",
            "cost_balanced_v3_warm_rbc_basic",
            "cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic",
            "community_band_v4_warm_rbc_basic",
            "community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic",
            "community_storage_band_v41_regularized_warm_rbc_basic",
            "community_storage_band_v41_prioritized_regularized_warm_rbc_basic",
            "community_service_band_v42_regularized_warm_rbc_basic",
            "community_service_band_v42_prioritized_regularized_warm_rbc_basic",
            "community_service_band_v42_prioritized_regularized_warm_rbc_smart",
            "community_service_band_v42_prioritized_warmtrain_rbc_smart",
            "community_service_band_v42_prioritized_warmtrain_phaseout_rbc_smart",
            "community_battery_value_v43_prioritized_warmtrain_phaseout_rbc_smart",
            "community_battery_value_v43_teacher_bc_stable_rbc_smart",
            "community_smooth_service_v44_stable_teacher_bc_rbc_smart",
            "community_feasible_service_v45_stable_teacher_bc_rbc_smart",
            "community_feasible_service_v45_teacher_clone_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_event_rbc_smart",
            "community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_event_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_guarded_band_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_band_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_balanced_rbc_smart",
            "community_feasible_service_v45_actor_pretrain_rbc_smart",
            "community_feasible_service_v45_actor_pretrain_slow_rbc_smart",
        }:
            exploration["actor_behavior_cloning_weight"] = 0.05
            exploration["actor_action_saturation_penalty"] = 0.050
            reward_kwargs = simulator.setdefault("reward_function_kwargs", {})
            reward_kwargs["ev_departure_window_hours"] = 4.0
            reward_kwargs["ev_connected_deficit_penalty"] = 120.0
            reward_kwargs["ev_schedule_deficit_penalty"] = 480.0
            reward_kwargs["ev_departure_deficit_penalty"] = 480.0
            reward_kwargs["ev_departure_missed_penalty"] = 1000.0
            if variant in {
                "ev_service_v2g_guard_warm_rbc_basic",
                "ev_service_v2g_guard_prioritized_warm_rbc_basic",
            }:
                reward_kwargs["ev_v2g_service_penalty"] = 200.0
            if variant.startswith("service_guard_v2"):
                simulator["reward_function"] = "CostServiceGuardRewardV2"
                simulator["reward_function_kwargs"] = {}
            if variant.startswith("cost_balanced_v3"):
                simulator["reward_function"] = "CostServiceCostBalancedRewardV3"
                simulator["reward_function_kwargs"] = {}
                exploration["actor_behavior_cloning_weight"] = 0.04
                exploration["actor_action_saturation_penalty"] = 0.035
            if variant.startswith("community_band_v4"):
                simulator["reward_function"] = "CostServiceCommunityBandRewardV4"
                simulator["reward_function_kwargs"] = {}
                exploration["actor_behavior_cloning_weight"] = 0.03
                exploration["actor_action_saturation_penalty"] = 0.030
            if variant.startswith("community_storage_band_v41"):
                simulator["reward_function"] = "CostServiceCommunityStorageBandRewardV41"
                simulator["reward_function_kwargs"] = {}
                exploration["actor_behavior_cloning_weight"] = 0.025
                exploration["actor_action_saturation_penalty"] = 0.030
                exploration["actor_storage_action_l2_penalty"] = 4.0
                exploration["actor_ev_v2g_action_l2_penalty"] = 0.020
            if variant.startswith("community_service_band_v42"):
                simulator["reward_function"] = "CostServiceCommunityServiceBandRewardV42"
                simulator["reward_function_kwargs"] = {}
                exploration["actor_behavior_cloning_weight"] = (
                    0.080
                    if "phaseout" in variant
                    else (0.040 if "warmtrain" in variant else (0.020 if variant.endswith("_warm_rbc_smart") else 0.025))
                )
                exploration["actor_action_saturation_penalty"] = 0.030
                exploration["actor_storage_action_l2_penalty"] = 4.0
                exploration["actor_ev_v2g_action_l2_penalty"] = 5.0 if "phaseout" in variant else 0.040
                exploration["storage_exploration_noise_multiplier"] = 0.25
                exploration["ev_negative_exploration_noise_multiplier"] = 0.0 if "phaseout" in variant else 0.35
                if "warmtrain" in variant:
                    exploration["train_during_initial_exploration"] = True
                    exploration["initial_exploration_training_start_step"] = 0
                if "phaseout" in variant:
                    exploration["use_amp"] = False
                    exploration["actor_ev_behavior_cloning_multiplier"] = 8.0
                    exploration["actor_storage_behavior_cloning_multiplier"] = 0.25
                    warmup_steps = int(exploration.get("random_exploration_steps") or 0)
                    exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 2, 1)
                    exploration["warm_start_policy_phaseout_mode"] = "blend"
            if variant.startswith("community_battery_value_v43"):
                simulator["reward_function"] = "CostServiceCommunityBatteryValueRewardV43"
                simulator["reward_function_kwargs"] = {}
                exploration["actor_behavior_cloning_weight"] = 0.070
                exploration["actor_action_saturation_penalty"] = 0.030
                exploration["actor_storage_action_l2_penalty"] = 0.50
                exploration["actor_ev_v2g_action_l2_penalty"] = 5.0
                exploration["storage_exploration_noise_multiplier"] = 0.50
                exploration["ev_negative_exploration_noise_multiplier"] = 0.0
                exploration["use_amp"] = False
                exploration["actor_ev_behavior_cloning_multiplier"] = 8.0
                exploration["actor_storage_behavior_cloning_multiplier"] = 1.0
                exploration["train_during_initial_exploration"] = True
                exploration["initial_exploration_training_start_step"] = 0
                warmup_steps = int(exploration.get("random_exploration_steps") or 0)
                exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 2, 1)
                exploration["warm_start_policy_phaseout_mode"] = "blend"
                if "teacher_bc" in variant:
                    algorithm["networks"]["critic"]["lr"] = 2.0e-4
                    exploration["actor_behavior_cloning_source"] = "warm_start_policy"
                    exploration["actor_behavior_cloning_weight"] = 0.100
                    exploration["actor_ev_behavior_cloning_multiplier"] = 10.0
                    exploration["actor_storage_behavior_cloning_multiplier"] = 0.30
                    exploration["actor_storage_action_l2_penalty"] = 0.25
                    exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 4, 1)
            if variant.startswith("community_smooth_service_v44"):
                simulator["reward_function"] = "CostServiceCommunitySmoothServiceRewardV44"
                simulator["reward_function_kwargs"] = {}
                algorithm["networks"]["critic"]["lr"] = 1.0e-4
                exploration["critic_loss"] = "huber"
                exploration["critic_huber_beta"] = 1.0
                exploration["critic_target_clip_abs"] = 35.0
                exploration["actor_behavior_cloning_source"] = "warm_start_policy"
                exploration["actor_behavior_cloning_weight"] = 0.120
                exploration["actor_ev_behavior_cloning_multiplier"] = 12.0
                exploration["actor_storage_behavior_cloning_multiplier"] = 0.25
                exploration["actor_storage_action_l2_penalty"] = 0.20
                exploration["actor_ev_v2g_action_l2_penalty"] = 8.0
                exploration["actor_action_saturation_penalty"] = 0.020
                exploration["storage_exploration_noise_multiplier"] = 0.35
                exploration["ev_negative_exploration_noise_multiplier"] = 0.0
                exploration["use_amp"] = False
                exploration["train_during_initial_exploration"] = True
                exploration["initial_exploration_training_start_step"] = 0
                warmup_steps = int(exploration.get("random_exploration_steps") or 0)
                exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 6, 1)
                exploration["warm_start_policy_phaseout_mode"] = "blend"
            if (variant.startswith("community_feasible_service_v45") or variant.startswith("community_feasible_precision_v46")):
                simulator["reward_function"] = (
                    "CostServiceCommunityFeasiblePrecisionRewardV46"
                    if variant.startswith("community_feasible_precision_v46")
                    else "CostServiceCommunityFeasibleServiceRewardV45"
                )
                simulator["reward_function_kwargs"] = {}
                algorithm["networks"]["critic"]["lr"] = 1.0e-4
                exploration["critic_loss"] = "huber"
                exploration["critic_huber_beta"] = 1.0
                exploration["critic_target_clip_abs"] = (
                    25.0 if variant.startswith("community_feasible_precision_v46") else 35.0
                )
                exploration["actor_behavior_cloning_source"] = "warm_start_policy"
                exploration["actor_behavior_cloning_weight"] = 0.120
                exploration["actor_ev_behavior_cloning_multiplier"] = 12.0
                exploration["actor_storage_behavior_cloning_multiplier"] = 0.25
                exploration["actor_storage_action_l2_penalty"] = 0.20
                exploration["actor_ev_v2g_action_l2_penalty"] = 8.0
                exploration["actor_action_saturation_penalty"] = 0.020
                exploration["storage_exploration_noise_multiplier"] = 0.35
                exploration["ev_negative_exploration_noise_multiplier"] = 0.0
                exploration["use_amp"] = False
                exploration["train_during_initial_exploration"] = True
                exploration["initial_exploration_training_start_step"] = 0
                warmup_steps = int(exploration.get("random_exploration_steps") or 0)
                exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 6, 1)
                exploration["warm_start_policy_phaseout_mode"] = "blend"
                if "teacher_clone" in variant:
                    algorithm["networks"]["actor"]["lr"] = 2.0e-4
                    algorithm["networks"]["critic"]["lr"] = 1.0e-4
                    exploration["actor_policy_loss_weight"] = 0.0
                    exploration["actor_policy_loss_warmup_weight"] = 0.0
                    exploration["actor_policy_loss_warmup_steps"] = 0
                    exploration["actor_behavior_cloning_weight"] = 0.500
                    exploration["actor_behavior_cloning_min_weight"] = 0.500
                    exploration["actor_behavior_cloning_decay_steps"] = 0
                    exploration["actor_ev_behavior_cloning_multiplier"] = 24.0
                    exploration["actor_ev_behavior_cloning_positive_target_weight"] = 8.0
                    exploration["actor_ev_behavior_cloning_positive_target_power"] = 1.0
                    exploration["actor_storage_behavior_cloning_multiplier"] = 1.00
                    exploration["actor_storage_action_l2_penalty"] = 0.10
                    exploration["actor_ev_v2g_action_l2_penalty"] = 20.0
                    exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 32, 1)
                    if "ev_learning_teacher" in variant:
                        exploration["warm_start_policy_hyperparameters"] = (
                            _rbc_smart_learning_teacher_hyperparameters()
                        )
                        exploration["actor_behavior_cloning_weight"] = 0.650
                        exploration["actor_behavior_cloning_min_weight"] = 0.650
                        exploration["actor_ev_behavior_cloning_multiplier"] = 36.0
                        exploration["actor_ev_behavior_cloning_positive_target_weight"] = 16.0
                        exploration["actor_ev_behavior_cloning_zero_target_weight"] = 4.0
                        exploration["actor_ev_behavior_cloning_zero_target_threshold"] = 0.04
                        exploration["actor_storage_behavior_cloning_multiplier"] = 0.25
                        exploration["actor_storage_action_l2_penalty"] = 0.30
                        exploration["actor_ev_v2g_action_l2_penalty"] = 24.0
                    if "ev_learning_teacher_event" in variant:
                        teacher_hyperparameters = dict(
                            exploration.get("warm_start_policy_hyperparameters") or {}
                        )
                        teacher_hyperparameters.update(
                            {
                                "storage_min_soc": 0.25,
                                "storage_price_discharge_soc_floor": 0.30,
                                "storage_peak_discharge_soc_floor": 0.30,
                                "price_discharge_rate": 0.35,
                                "peak_discharge_rate": 0.50,
                            }
                        )
                        exploration["warm_start_policy_hyperparameters"] = teacher_hyperparameters
                        exploration["actor_behavior_cloning_weight"] = 0.700
                        exploration["actor_behavior_cloning_min_weight"] = 0.700
                        exploration["actor_ev_behavior_cloning_multiplier"] = 44.0
                        exploration["actor_ev_behavior_cloning_positive_target_weight"] = 24.0
                        exploration["actor_ev_behavior_cloning_positive_target_power"] = 1.15
                        exploration["actor_ev_behavior_cloning_zero_target_weight"] = 4.0
                        exploration["actor_ev_behavior_cloning_zero_target_threshold"] = 0.04
                        exploration["actor_storage_behavior_cloning_multiplier"] = 0.15
                        exploration["actor_storage_action_l2_penalty"] = 0.45
                        exploration["actor_ev_v2g_action_l2_penalty"] = 30.0
                        exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 64, 1)
                    if "ev_focus" in variant:
                        exploration["actor_behavior_cloning_weight"] = 0.650
                        exploration["actor_behavior_cloning_min_weight"] = 0.650
                        exploration["actor_ev_behavior_cloning_multiplier"] = 36.0
                        exploration["actor_ev_behavior_cloning_positive_target_weight"] = 16.0
                        exploration["actor_storage_behavior_cloning_multiplier"] = 0.25
                        exploration["actor_storage_action_l2_penalty"] = 0.30
                        exploration["actor_ev_v2g_action_l2_penalty"] = 24.0
                    if "slow_finetune" in variant:
                        exploration["actor_policy_loss_weight"] = 0.12
                        exploration["actor_policy_loss_warmup_weight"] = 0.01
                        exploration["actor_policy_loss_warmup_steps"] = max(warmup_steps * 96, 1)
                        exploration["actor_policy_loss_warmup_start_step"] = max(warmup_steps * 2, 1)
                        exploration["actor_behavior_cloning_weight"] = 0.650
                        exploration["actor_behavior_cloning_min_weight"] = 0.450
                        exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps * 96, 1)
                        exploration["actor_ev_behavior_cloning_multiplier"] = 40.0
                        exploration["actor_ev_behavior_cloning_positive_target_weight"] = 18.0
                        exploration["actor_storage_behavior_cloning_multiplier"] = 0.35
                        exploration["actor_storage_action_l2_penalty"] = 0.25
                        exploration["actor_ev_v2g_action_l2_penalty"] = 28.0
                        exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 64, 1)
                    if "ev_focus_event" in variant:
                        exploration["actor_policy_loss_weight"] = 0.0
                        exploration["actor_policy_loss_warmup_weight"] = 0.0
                        exploration["actor_policy_loss_warmup_steps"] = 0
                        exploration["actor_behavior_cloning_weight"] = 0.700
                        exploration["actor_behavior_cloning_min_weight"] = 0.700
                        exploration["actor_behavior_cloning_decay_steps"] = 0
                        exploration["actor_ev_behavior_cloning_multiplier"] = 44.0
                        exploration["actor_ev_behavior_cloning_positive_target_weight"] = 24.0
                        exploration["actor_ev_behavior_cloning_positive_target_power"] = 1.25
                        exploration["actor_storage_behavior_cloning_multiplier"] = 0.20
                        exploration["actor_storage_action_l2_penalty"] = 0.35
                        exploration["actor_ev_v2g_action_l2_penalty"] = 30.0
                        exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 48, 1)
                    if "ev_guarded_band" in variant:
                        exploration["actor_behavior_cloning_weight"] = 0.650
                        exploration["actor_behavior_cloning_min_weight"] = 0.650
                        exploration["actor_ev_behavior_cloning_multiplier"] = 36.0
                        exploration["actor_ev_behavior_cloning_positive_target_weight"] = 16.0
                        exploration["actor_ev_behavior_cloning_zero_target_weight"] = 4.0
                        exploration["actor_ev_behavior_cloning_zero_target_threshold"] = 0.04
                        exploration["actor_storage_behavior_cloning_multiplier"] = 0.25
                        exploration["actor_storage_action_l2_penalty"] = 0.30
                        exploration["actor_ev_v2g_action_l2_penalty"] = 24.0
                    if "ev_band" in variant:
                        exploration["actor_behavior_cloning_weight"] = 0.600
                        exploration["actor_behavior_cloning_min_weight"] = 0.600
                        exploration["actor_ev_behavior_cloning_multiplier"] = 32.0
                        exploration["actor_ev_behavior_cloning_positive_target_weight"] = 12.0
                        exploration["actor_ev_behavior_cloning_zero_target_weight"] = 12.0
                        exploration["actor_ev_behavior_cloning_zero_target_threshold"] = 0.04
                        exploration["actor_storage_behavior_cloning_multiplier"] = 0.25
                        exploration["actor_storage_action_l2_penalty"] = 0.30
                        exploration["actor_ev_v2g_action_l2_penalty"] = 24.0
                    if "ev_balanced" in variant:
                        exploration["actor_behavior_cloning_weight"] = 0.625
                        exploration["actor_behavior_cloning_min_weight"] = 0.625
                        exploration["actor_ev_behavior_cloning_multiplier"] = 34.0
                        exploration["actor_ev_behavior_cloning_positive_target_weight"] = 14.0
                        exploration["actor_ev_behavior_cloning_zero_target_weight"] = 8.0
                        exploration["actor_ev_behavior_cloning_zero_target_threshold"] = 0.04
                        exploration["actor_storage_behavior_cloning_multiplier"] = 0.25
                        exploration["actor_storage_action_l2_penalty"] = 0.30
                        exploration["actor_ev_v2g_action_l2_penalty"] = 24.0
                if "actor_pretrain_slow" in variant:
                    exploration["actor_policy_loss_weight"] = 0.60
                    exploration["actor_policy_loss_warmup_weight"] = 0.02
                    exploration["actor_policy_loss_warmup_steps"] = max(warmup_steps * 64, 1)
                    exploration["actor_policy_loss_warmup_start_step"] = 0
                    exploration["actor_behavior_cloning_weight"] = 0.250
                    exploration["actor_behavior_cloning_min_weight"] = 0.200
                    exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps * 64, 1)
                    exploration["actor_ev_behavior_cloning_multiplier"] = 24.0
                    exploration["actor_ev_behavior_cloning_positive_target_weight"] = 6.0
                    exploration["actor_ev_behavior_cloning_positive_target_power"] = 1.0
                    exploration["actor_storage_behavior_cloning_multiplier"] = 1.00
                    exploration["actor_storage_action_l2_penalty"] = 0.18
                    exploration["actor_ev_v2g_action_l2_penalty"] = 16.0
                    exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 24, 1)
                elif "actor_pretrain" in variant:
                    exploration["actor_policy_loss_warmup_weight"] = 0.05
                    exploration["actor_policy_loss_warmup_steps"] = max(warmup_steps * 16, 1)
                    exploration["actor_policy_loss_warmup_start_step"] = 0
                    exploration["actor_behavior_cloning_weight"] = 0.180
                    exploration["actor_behavior_cloning_min_weight"] = 0.120
                    exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps * 16, 1)
                    exploration["actor_ev_behavior_cloning_multiplier"] = 16.0
                    exploration["actor_ev_behavior_cloning_positive_target_weight"] = 4.0
                    exploration["actor_ev_behavior_cloning_positive_target_power"] = 1.0
                    exploration["actor_storage_behavior_cloning_multiplier"] = 0.50
                    exploration["actor_storage_action_l2_penalty"] = 0.12
                    exploration["warm_start_policy_phaseout_steps"] = max(warmup_steps * 10, 1)
        if variant in {
            "ev_service_v2g_guard_prioritized_warm_rbc_basic",
            "service_guard_v2_prioritized_low_warm_rbc_basic",
            "service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic",
            "cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic",
            "community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic",
            "community_storage_band_v41_prioritized_regularized_warm_rbc_basic",
            "community_service_band_v42_prioritized_regularized_warm_rbc_basic",
            "community_service_band_v42_prioritized_regularized_warm_rbc_smart",
            "community_service_band_v42_prioritized_warmtrain_rbc_smart",
            "community_service_band_v42_prioritized_warmtrain_phaseout_rbc_smart",
            "community_battery_value_v43_prioritized_warmtrain_phaseout_rbc_smart",
            "community_battery_value_v43_teacher_bc_stable_rbc_smart",
            "community_smooth_service_v44_stable_teacher_bc_rbc_smart",
            "community_feasible_service_v45_stable_teacher_bc_rbc_smart",
            "community_feasible_service_v45_teacher_clone_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_event_rbc_smart",
            "community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_event_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_guarded_band_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_band_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_balanced_rbc_smart",
            "community_feasible_service_v45_actor_pretrain_rbc_smart",
            "community_feasible_service_v45_actor_pretrain_slow_rbc_smart",
        }:
            replay_buffer = algorithm.setdefault("replay_buffer", {})
            replay_buffer["class"] = "RewardWeightedMultiAgentReplayBuffer"
            if variant == "ev_service_v2g_guard_prioritized_warm_rbc_basic":
                replay_buffer["priority_fraction"] = 0.5
            elif "ev_learning_teacher_event" in variant:
                replay_buffer["priority_fraction"] = 0.30
            elif "ev_focus_event" in variant:
                replay_buffer["priority_fraction"] = 0.35
            elif variant.startswith("community_feasible_precision_v46"):
                replay_buffer["priority_fraction"] = 0.20
            elif variant.startswith("community_smooth_service_v44") or variant.startswith("community_feasible_service_v45") or variant.startswith("community_feasible_precision_v46"):
                replay_buffer["priority_fraction"] = 0.25
            elif "teacher_bc" in variant:
                replay_buffer["priority_fraction"] = 0.35
            elif variant.startswith("community_service_band_v42") or variant.startswith("community_battery_value_v43"):
                replay_buffer["priority_fraction"] = 0.20
            elif variant.startswith("community_band_v4") or variant.startswith("community_storage_band_v41"):
                replay_buffer["priority_fraction"] = 0.15
            else:
                replay_buffer["priority_fraction"] = 0.25
            replay_buffer["priority_alpha"] = 0.7
            replay_buffer["priority_epsilon"] = 1.0e-3
            replay_buffer["priority_mode"] = "negative_reward"
            if (variant.startswith("community_feasible_service_v45") or variant.startswith("community_feasible_precision_v46")):
                replay_buffer["behavior_action_priority_mode"] = "positive"
                replay_buffer["behavior_action_priority_weight"] = (
                    4.0
                    if "ev_learning_teacher_event" in variant
                    else 4.0
                    if "ev_focus_event" in variant
                    else 3.0
                    if "ev_learning_teacher" in variant
                    else 3.0
                    if "ev_focus" in variant
                    else 3.0
                    if "ev_guarded_band" in variant
                    else 2.75
                    if "ev_balanced" in variant
                    else 2.5
                    if "ev_band" in variant
                    else (2.0 if "teacher_clone" in variant else (1.0 if "actor_pretrain" in variant else 0.5))
                )
                replay_buffer["behavior_action_priority_scope"] = (
                    "ev"
                    if (
                        "ev_focus_event" in variant
                        or "ev_learning_teacher_event" in variant
                        or "ev_learning_teacher" in variant
                        or "ev_focus" in variant
                        or "ev_guarded_band" in variant
                        or "ev_band" in variant
                        or "ev_balanced" in variant
                    )
                    else "all"
                )
                if "ev_learning_teacher_event" in variant:
                    replay_buffer["observation_event_priority_weight"] = 6.0
                    replay_buffer["observation_event_priority_mode"] = "ev_departure_service"
                elif "ev_focus_event" in variant:
                    replay_buffer["observation_event_priority_weight"] = 8.0
                    replay_buffer["observation_event_priority_mode"] = "ev_departure_service"
            replay_buffer["priority_max"] = (
                60.0
                if "ev_learning_teacher_event" in variant or "ev_focus_event" in variant
                else 30.0
                if variant.startswith("community_feasible_precision_v46")
                else 40.0
                if variant.startswith("community_smooth_service_v44") or variant.startswith("community_feasible_service_v45") or variant.startswith("community_feasible_precision_v46")
                else (50.0 if "teacher_bc" in variant else 100.0)
            )
        if variant in {
            "service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic",
            "cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic",
            "community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic",
            "community_storage_band_v41_prioritized_regularized_warm_rbc_basic",
            "community_service_band_v42_prioritized_regularized_warm_rbc_basic",
            "community_service_band_v42_prioritized_regularized_warm_rbc_smart",
            "community_service_band_v42_prioritized_warmtrain_rbc_smart",
            "community_service_band_v42_prioritized_warmtrain_phaseout_rbc_smart",
            "community_battery_value_v43_prioritized_warmtrain_phaseout_rbc_smart",
            "community_battery_value_v43_teacher_bc_stable_rbc_smart",
            "community_smooth_service_v44_stable_teacher_bc_rbc_smart",
            "community_feasible_service_v45_stable_teacher_bc_rbc_smart",
            "community_feasible_service_v45_teacher_clone_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_learning_teacher_event_rbc_smart",
            "community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_focus_event_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_guarded_band_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_band_rbc_smart",
            "community_feasible_service_v45_teacher_clone_ev_balanced_rbc_smart",
            "community_feasible_service_v45_actor_pretrain_rbc_smart",
            "community_feasible_service_v45_actor_pretrain_slow_rbc_smart",
        }:
            warmup_steps = int(exploration.get("random_exploration_steps") or 0)
            is_community_variant = (
                variant.startswith("community_band_v4")
                or variant.startswith("community_storage_band_v41")
                or variant.startswith("community_service_band_v42")
                or variant.startswith("community_battery_value_v43")
            )
            if "teacher_clone" in variant:
                if "slow_finetune" in variant:
                    exploration["actor_behavior_cloning_min_weight"] = 0.450
                    exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps * 96, 1)
                elif (
                    "ev_learning_teacher" in variant
                    or "ev_focus" in variant
                    or "ev_guarded_band" in variant
                    or "ev_band" in variant
                    or "ev_balanced" in variant
                ):
                    exploration["actor_behavior_cloning_min_weight"] = float(
                        exploration.get("actor_behavior_cloning_weight", 0.650)
                    )
                else:
                    exploration["actor_behavior_cloning_min_weight"] = 0.500
                    exploration["actor_behavior_cloning_decay_steps"] = 0
            elif "actor_pretrain_slow" in variant:
                exploration["actor_behavior_cloning_min_weight"] = 0.200
                exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps * 64, 1)
            elif "actor_pretrain" in variant:
                exploration["actor_behavior_cloning_min_weight"] = 0.120
                exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps * 16, 1)
            elif variant.startswith("community_smooth_service_v44") or variant.startswith("community_feasible_service_v45") or variant.startswith("community_feasible_precision_v46"):
                exploration["actor_behavior_cloning_min_weight"] = 0.100
                exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps * 12, 1)
            elif "teacher_bc" in variant:
                exploration["actor_behavior_cloning_min_weight"] = 0.080
                exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps * 8, 1)
            elif "phaseout" in variant:
                exploration["actor_behavior_cloning_min_weight"] = 0.060
                exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps * 6, 1)
            elif is_community_variant:
                exploration["actor_behavior_cloning_min_weight"] = 0.005
                exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps, 1)
            else:
                exploration["actor_behavior_cloning_min_weight"] = 0.010
                exploration["actor_behavior_cloning_decay_steps"] = max(warmup_steps * 2, 1)
            exploration["actor_behavior_cloning_decay_start_step"] = warmup_steps
        return

    raise ValueError(f"Unknown MADDPG variant: {variant}")


def _build_run_config(
    *,
    template_path: Path,
    dataset_key: str,
    agent_key: str,
    maddpg_variant: str | None,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = validate_config(_load_yaml(template_path)).to_dict()
    variant_label = maddpg_variant if agent_key == "maddpg" else agent_key
    run_label = _slug(f"{dataset_key}_{agent_key}_{variant_label}_seed{seed}")

    metadata = config.setdefault("metadata", {})
    metadata["experiment_name"] = f"phase6a_{run_label}"
    metadata["run_name"] = f"Phase 6A {dataset_key} {agent_key} {variant_label} seed {seed}"
    metadata["description"] = "Phase 6A short comparable benchmark run."

    tracking = config.setdefault("tracking", {})
    tracking["mlflow_enabled"] = False
    tracking["log_level"] = "INFO"
    tracking["log_frequency"] = max(int(args.metric_interval), 1)
    tracking["mlflow_step_sample_interval"] = max(int(args.metric_interval), 1)
    tracking["progress_updates_enabled"] = True
    tracking["progress_update_interval"] = max(int(args.metric_interval), 1)
    tracking["system_metrics_enabled"] = False
    tracking["action_diagnostics_enabled"] = True
    tracking["action_diagnostics_detail"] = "summary"
    tracking["training_diagnostics_enabled"] = True
    tracking["training_diagnostics_detail"] = "summary"
    tracking["reward_diagnostics_enabled"] = True
    tracking["reward_diagnostics_detail"] = getattr(args, "reward_diagnostics_detail", "summary")

    simulator = config.setdefault("simulator", {})
    simulator["episodes"] = max(int(args.episodes), 1)
    simulator["deterministic_finish"] = bool(getattr(args, "deterministic_finish", False))
    if not args.full_window:
        steps = _selected_steps(dataset_key, args)
        start = max(int(args.start), 0)
        simulator["simulation_start_time_step"] = start
        simulator["simulation_end_time_step"] = start + steps - 1
        simulator["episode_time_steps"] = steps
    export_cfg = simulator.setdefault("export", {})
    export_cfg["mode"] = "none" if args.no_kpi_export else "end"
    export_cfg["export_kpis_on_episode_end"] = not bool(args.no_kpi_export)
    export_cfg["session_name"] = run_label

    training = config.setdefault("training", {})
    training["seed"] = int(seed)
    training["steps_between_training_updates"] = 1
    training["target_update_interval"] = 1 if agent_key == "maddpg" else 0

    checkpointing = config.setdefault("checkpointing", {})
    checkpoint_interval = getattr(args, "checkpoint_interval", None)
    if checkpoint_interval is not None:
        checkpointing["checkpoint_interval"] = max(int(checkpoint_interval), 1)

    algorithm = config.setdefault("algorithm", {})
    hyperparameters = algorithm.setdefault("hyperparameters", {})
    hyperparameters["seed"] = int(seed)
    hyperparameters.update(
        _parse_hyperparameter_overrides(getattr(args, "hyperparameter_override", []))
    )

    if agent_key == "maddpg":
        algorithm.setdefault("networks", {})
        algorithm["networks"]["actor"] = {
            "class": "Actor",
            "layers": _parse_layers(args.actor_layers),
            "lr": float((algorithm.get("networks") or {}).get("actor", {}).get("lr", 1.0e-4)),
        }
        algorithm["networks"]["critic"] = {
            "class": "Critic",
            "layers": _parse_layers(args.critic_layers),
            "lr": float((algorithm.get("networks") or {}).get("critic", {}).get("lr", 1.0e-3)),
        }
        algorithm["replay_buffer"] = {
            "class": "MultiAgentReplayBuffer",
            "capacity": int(args.buffer_capacity),
            "batch_size": int(args.batch_size),
        }
        exploration = algorithm.setdefault("exploration", {}).setdefault("params", {})
        exploration["sigma"] = float(args.sigma)
        exploration["decay"] = float(exploration.get("decay", 0.9995))
        exploration["min_sigma"] = float(exploration.get("min_sigma", 0.03))
        exploration["noise_clip"] = float(exploration.get("noise_clip", 0.3))
        exploration["end_initial_exploration_time_step"] = int(args.random_exploration_steps)
        exploration["random_exploration_steps"] = int(args.random_exploration_steps)
        exploration["reward_normalization"] = bool(exploration.get("reward_normalization", True))
        exploration["reward_normalization_clip"] = float(exploration.get("reward_normalization_clip", 10.0))
        exploration["reward_normalization_epsilon"] = float(exploration.get("reward_normalization_epsilon", 1.0e-8))
        _apply_maddpg_variant(config, maddpg_variant or "current")
        exploration.update(
            _parse_key_value_overrides(
                getattr(args, "exploration_override", []),
                label="exploration override",
            )
        )
        algorithm.setdefault("replay_buffer", {}).update(
            _parse_key_value_overrides(
                getattr(args, "replay_buffer_override", []),
                label="replay buffer override",
            )
        )
        warm_start_policy = exploration.get("warm_start_policy")
        if warm_start_policy and "warm_start_policy_hyperparameters" not in exploration:
            warm_start_hyperparameters = _load_baseline_hyperparameters(
                dataset_key,
                str(warm_start_policy),
            )
            if warm_start_hyperparameters:
                exploration["warm_start_policy_hyperparameters"] = warm_start_hyperparameters
        warm_start_overrides = _parse_key_value_overrides(
            getattr(args, "warm_start_hyperparameter_override", []),
            label="warm-start hyperparameter override",
        )
        if warm_start_overrides:
            warm_start_hyperparameters = exploration.setdefault("warm_start_policy_hyperparameters", {})
            if not isinstance(warm_start_hyperparameters, dict):
                warm_start_hyperparameters = {}
                exploration["warm_start_policy_hyperparameters"] = warm_start_hyperparameters
            warm_start_hyperparameters.update(warm_start_overrides)
        warm_start_phaseout_steps = getattr(args, "warm_start_phaseout_steps", None)
        if warm_start_phaseout_steps is not None:
            exploration["warm_start_policy_phaseout_steps"] = max(
                int(warm_start_phaseout_steps),
                0,
            )

    reward_function_override = getattr(args, "reward_function_override", None)
    if reward_function_override:
        simulator["reward_function"] = str(reward_function_override)
        simulator["reward_function_kwargs"] = {}
    simulator.setdefault("reward_function_kwargs", {}).update(
        _parse_key_value_overrides(
            getattr(args, "reward_kwarg_override", []),
            label="reward kwarg override",
        )
    )

    config.setdefault("topology", {})
    _set_nested(config, ("topology", "num_agents"), None)
    _set_nested(config, ("topology", "observation_dimensions"), None)
    _set_nested(config, ("topology", "action_dimensions"), None)
    _set_nested(config, ("topology", "action_space"), None)

    return validate_config(config).to_dict()


def _planned_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = tuple(args.dataset or DEFAULT_DATASETS)
    agents = tuple(args.agent or DEFAULT_AGENTS)
    seeds = tuple(args.seed or (123,))
    maddpg_variants = tuple(args.maddpg_variant or DEFAULT_MADDPG_VARIANTS)

    runs: list[dict[str, Any]] = []
    for dataset_key in datasets:
        for agent_key in agents:
            variants = maddpg_variants if agent_key == "maddpg" else (None,)
            for variant in variants:
                for seed in seeds:
                    template_path = Path(DATASET_CONFIGS[dataset_key][agent_key])
                    variant_label = variant if agent_key == "maddpg" else agent_key
                    job_id = _slug(f"phase6a_{dataset_key}_{agent_key}_{variant_label}_seed{seed}")
                    runs.append(
                        {
                            "dataset_key": dataset_key,
                            "agent_key": agent_key,
                            "maddpg_variant": variant,
                            "variant_label": variant_label,
                            "seed": int(seed),
                            "template_path": template_path,
                            "job_id": job_id,
                        }
                    )
    return runs


def _read_metrics_jsonl(job_dir: Path) -> dict[str, Any]:
    metrics_path = job_dir / "logs" / "metrics.jsonl"
    if not metrics_path.exists():
        return {}

    values_by_metric: dict[str, list[float]] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = record.get("metrics") or {}
            if not isinstance(metrics, Mapping):
                continue
            for key, value in metrics.items():
                parsed = _safe_float(value)
                if parsed is None:
                    continue
                values_by_metric.setdefault(str(key), []).append(parsed)

    row: dict[str, Any] = {}
    for suffix in ("Sum", "Mean", "Min", "Max"):
        prefix = f"Agent_"
        metric_suffix = f"_Overall_Reward_{suffix}"
        values = [
            vals[-1]
            for key, vals in values_by_metric.items()
            if key.startswith(prefix) and key.endswith(metric_suffix) and vals
        ]
        if values:
            array = np.asarray(values, dtype=np.float64)
            row[f"reward_overall_{suffix.lower()}_agent_mean"] = float(np.mean(array))
            row[f"reward_overall_{suffix.lower()}_agent_sum"] = float(np.sum(array))

    for metric_name in SELECTED_METRICS:
        values = values_by_metric.get(metric_name)
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        safe_name = _slug(metric_name).replace(".", "_")
        row[f"metric__{safe_name}__mean"] = float(np.mean(array))
        row[f"metric__{safe_name}__last"] = float(array[-1])
        row[f"metric__{safe_name}__records"] = int(array.shape[0])

    return row


def _collect_job_row(plan: Mapping[str, Any], *, output_dir: Path, status: str, error: str | None = None) -> dict[str, Any]:
    job_dir = output_dir / "jobs" / str(plan["job_id"])
    row: dict[str, Any] = {
        "status": status,
        "dataset_key": plan["dataset_key"],
        "agent_key": plan["agent_key"],
        "variant_label": plan["variant_label"],
        "seed": plan["seed"],
        "job_id": plan["job_id"],
        "job_dir": str(job_dir),
        "generated_config": str(plan.get("generated_config", "")),
        "template_path": str(plan["template_path"]),
        "error": error or "",
    }

    config_path = job_dir / "config.resolved.yaml"
    if not config_path.exists() and plan.get("generated_config"):
        config_path = Path(str(plan["generated_config"]))
    if config_path.exists():
        resolved = _load_yaml(config_path)
        simulator = resolved.get("simulator") if isinstance(resolved.get("simulator"), Mapping) else {}
        algorithm = resolved.get("algorithm") if isinstance(resolved.get("algorithm"), Mapping) else {}
        encoding = simulator.get("entity_encoding") if isinstance(simulator.get("entity_encoding"), Mapping) else {}
        row.update(
            {
                "dataset_name": simulator.get("dataset_name"),
                "algorithm_name": algorithm.get("name"),
                "entity_profile": encoding.get("profile"),
                "episodes": simulator.get("episodes"),
                "deterministic_finish": simulator.get("deterministic_finish"),
                "simulation_start_time_step": simulator.get("simulation_start_time_step"),
                "simulation_end_time_step": simulator.get("simulation_end_time_step"),
                "episode_time_steps": simulator.get("episode_time_steps"),
            }
        )

    record = _load_job_record(job_dir, list(DEFAULT_KPIS))
    if record:
        row["kpi_source"] = record.get("kpi_source")
        row["kpi_source_path"] = record.get("kpi_source_path")
        for kpi_name in DEFAULT_KPIS:
            row[f"kpi__{kpi_name}"] = (record.get("kpis") or {}).get(kpi_name)

    row.update(_read_metrics_jsonl(job_dir))
    return row


def _fieldnames(rows: list[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "status",
        "dataset_key",
        "dataset_name",
        "agent_key",
        "algorithm_name",
        "variant_label",
        "entity_profile",
        "seed",
        "episodes",
        "deterministic_finish",
        "episode_time_steps",
        "simulation_start_time_step",
        "simulation_end_time_step",
        "job_id",
        "kpi_source",
        *[f"kpi__{name}" for name in DEFAULT_KPIS],
        "reward_overall_sum_agent_mean",
        "reward_overall_mean_agent_mean",
        "job_dir",
        "generated_config",
        "template_path",
        "error",
    ]
    seen = set(preferred)
    dynamic = sorted({key for row in rows for key in row.keys() if key not in seen})
    return preferred + dynamic


def _write_readme(path: Path, *, rows: list[Mapping[str, Any]], args: argparse.Namespace) -> None:
    completed = sum(1 for row in rows if row.get("status") == "completed")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    planned = sum(1 for row in rows if row.get("status") == "planned")
    lines = [
        "# Phase 6A Benchmark",
        "",
        "Short comparable matrix for baselines and MADDPG. This is not the final KPI benchmark.",
        "",
        "## Status",
        "",
        f"- Rows: `{len(rows)}`",
        f"- Completed: `{completed}`",
        f"- Failed: `{failed}`",
        f"- Planned only: `{planned}`",
        f"- KPI export enabled: `{not bool(args.no_kpi_export)}`",
        f"- Deterministic finish: `{bool(getattr(args, 'deterministic_finish', False))}`",
        f"- Dry run: `{bool(args.dry_run)}`",
        "",
        "## Files",
        "",
        "- `benchmark_summary.csv`: one row per generated run.",
        "- `benchmark_summary.json`: machine-readable rows and settings.",
        "- `generated_configs/`: exact configs sent to `run_experiment.py`.",
        "- `jobs/<job_id>/`: standard run outputs from `run_experiment.py`.",
        "",
        "## Interpretation",
        "",
        "- Use this to confirm comparability, available KPIs, reward components and action diagnostics.",
        "- Do not use short-window results as final performance claims.",
        "- For final claims, rerun with longer windows, multiple seeds and the selected MADDPG variants.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase6a(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / "benchmarks" / f"phase6a_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_config_dir = output_dir / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    plans = _planned_runs(args)
    for plan in plans:
        config = _build_run_config(
            template_path=Path(plan["template_path"]),
            dataset_key=str(plan["dataset_key"]),
            agent_key=str(plan["agent_key"]),
            maddpg_variant=plan.get("maddpg_variant"),
            seed=int(plan["seed"]),
            args=args,
        )
        config_path = generated_config_dir / f"{plan['job_id']}.yaml"
        _write_yaml(config_path, config)
        plan["generated_config"] = str(config_path)

        if args.dry_run:
            rows.append(_collect_job_row(plan, output_dir=output_dir, status="planned"))
            continue

        try:
            run_experiment(str(config_path), str(plan["job_id"]), output_dir)
            rows.append(_collect_job_row(plan, output_dir=output_dir, status="completed"))
        except Exception as exc:  # pragma: no cover - exercised by real benchmark failures.
            error = f"{type(exc).__name__}: {exc}"
            (output_dir / f"{plan['job_id']}.error.log").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            rows.append(_collect_job_row(plan, output_dir=output_dir, status="failed", error=error))
            if args.fail_fast:
                raise

        _write_csv(output_dir / "benchmark_summary.csv", rows, _fieldnames(rows))
        (output_dir / "benchmark_summary.json").write_text(
            json.dumps({"rows": rows}, indent=2),
            encoding="utf-8",
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "output_dir": str(output_dir),
        "settings": {
            "datasets": args.dataset or list(DEFAULT_DATASETS),
            "agents": args.agent or list(DEFAULT_AGENTS),
            "maddpg_variants": args.maddpg_variant or list(DEFAULT_MADDPG_VARIANTS),
            "seeds": args.seed or [123],
            "episodes": args.episodes,
            "deterministic_finish": bool(getattr(args, "deterministic_finish", False)),
            "steps": args.steps,
            "steps_15s": args.steps_15s,
            "steps_2022": args.steps_2022,
            "full_window": args.full_window,
            "kpi_export": not bool(args.no_kpi_export),
            "checkpoint_interval": getattr(args, "checkpoint_interval", None),
            "reward_diagnostics_detail": getattr(args, "reward_diagnostics_detail", "summary"),
            "reward_function_override": getattr(args, "reward_function_override", None),
            "hyperparameter_overrides": list(getattr(args, "hyperparameter_override", []) or []),
            "exploration_overrides": list(getattr(args, "exploration_override", []) or []),
            "warm_start_hyperparameter_overrides": list(
                getattr(args, "warm_start_hyperparameter_override", []) or []
            ),
            "replay_buffer_overrides": list(getattr(args, "replay_buffer_override", []) or []),
            "reward_kwarg_overrides": list(getattr(args, "reward_kwarg_override", []) or []),
            "dry_run": bool(args.dry_run),
        },
        "rows": rows,
    }
    _write_csv(output_dir / "benchmark_summary.csv", rows, _fieldnames(rows))
    (output_dir / "benchmark_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_readme(output_dir / "README.md", rows=rows, args=args)
    return payload


def main() -> None:
    args = _parse_args()
    payload = run_phase6a(args)
    rows = payload["rows"]
    print(
        json.dumps(
            {
                "output_dir": payload["output_dir"],
                "rows": len(rows),
                "completed": sum(1 for row in rows if row.get("status") == "completed"),
                "failed": sum(1 for row in rows if row.get("status") == "failed"),
                "planned": sum(1 for row in rows if row.get("status") == "planned"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
