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
    "RewardComponent/ev_departure_window_penalty_mean",
    "RewardComponent/ev_departure_missed_penalty_amount_mean",
    "RewardComponent/deferrable_service_penalty_mean",
    "RewardComponent/battery_safety_penalty_mean",
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
    "MADDPG/actor_regularization_loss_mean",
    "MADDPG/actor_action_l2_mean",
    "MADDPG/actor_action_saturation_excess_mean",
    "MADDPG/reward_raw_std",
    "MADDPG/reward_train_std",
    "MADDPG/replay_buffer_size",
    "MADDPG/exploration_sigma",
)


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
        ),
        default=[],
        help="MADDPG variant to include when --agent maddpg is selected. Can be repeated.",
    )
    parser.add_argument("--seed", action="append", type=int, default=[], help="Seed. Can be repeated.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per generated run.")
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
    parser.add_argument("--batch-size", type=int, default=32, help="MADDPG replay batch size for generated runs.")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="MADDPG replay buffer capacity.")
    parser.add_argument("--actor-layers", default="64,32", help="Comma-separated MADDPG actor layers.")
    parser.add_argument("--critic-layers", default="128,64", help="Comma-separated MADDPG critic layers.")
    parser.add_argument("--random-exploration-steps", type=int, default=32, help="MADDPG warm-up steps.")
    parser.add_argument("--sigma", type=float, default=0.15, help="MADDPG Gaussian exploration sigma.")
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

    if variant in {"anti_saturation", "anti_saturation_warm_rbc_basic", "anti_saturation_warm_rbc_smart"}:
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
        if variant == "anti_saturation_warm_rbc_basic":
            exploration["initial_exploration_strategy"] = "policy"
            exploration["warm_start_policy"] = "RBCBasicPolicy"
            exploration["warm_start_policy_deterministic"] = True
            exploration["warm_start_policy_noise_scale"] = 0.0
        elif variant == "anti_saturation_warm_rbc_smart":
            exploration["initial_exploration_strategy"] = "policy"
            exploration["warm_start_policy"] = "RBCSmartPolicy"
            exploration["warm_start_policy_deterministic"] = True
            exploration["warm_start_policy_noise_scale"] = 0.0
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
    tracking["reward_diagnostics_detail"] = "summary"

    simulator = config.setdefault("simulator", {})
    simulator["episodes"] = max(int(args.episodes), 1)
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

    algorithm = config.setdefault("algorithm", {})
    hyperparameters = algorithm.setdefault("hyperparameters", {})
    hyperparameters["seed"] = int(seed)

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
            "steps": args.steps,
            "steps_15s": args.steps_15s,
            "steps_2022": args.steps_2022,
            "full_window": args.full_window,
            "kpi_export": not bool(args.no_kpi_export),
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
