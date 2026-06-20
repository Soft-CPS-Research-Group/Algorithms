#!/usr/bin/env python3
"""Build an operational report for collected remote jobs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pipeline_utils import summarise_pipeline_algorithms


OUTPUT_COLUMNS = [
    "job_id",
    "status",
    "exit_code",
    "job_name",
    "target_host",
    "image_tag",
    "algorithm",
    "seed",
    "dataset_name",
    "episodes",
    "episode_time_steps",
    "simulation_start_time_step",
    "simulation_end_time_step",
    "configured_env_steps",
    "full_year_check",
    "steps_between_training_updates",
    "target_update_interval",
    "batch_size",
    "replay_capacity",
    "use_amp",
    "device_selected",
    "run_duration_seconds",
    "seconds_per_env_step",
    "estimated_env_steps_per_hour",
    "queue_wait_seconds",
    "total_duration_seconds",
    "deucalion_partition",
    "deucalion_time",
    "deucalion_cpus_per_task",
    "deucalion_mem_gb",
    "deucalion_gpus",
    "mlflow_enabled",
    "checkpoint_interval",
    "checkpoint_effective",
    "export_kpis_on_episode_end",
    "export_final_episode_only",
    "export_kpis_final_episode_only",
    "export_timeseries_final_episode_only",
    "export_include_business_as_usual",
    "export_business_as_usual_timeseries",
    "tail_step_duration_count",
    "tail_step_duration_mean_seconds",
    "tail_step_duration_median_seconds",
    "tail_step_duration_p95_seconds",
    "risk_flags",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    return None if parsed is None else int(parsed)


def _fmt(value: Any, digits: int = 4) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "" if value is None else str(value)
    return f"{parsed:.{digits}f}"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _job_dir(results_dir: Path, job_id: str) -> Path:
    return results_dir / "jobs" / job_id


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _extract_step_durations(log_text: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(r"Step Duration:\s*([0-9.eE+-]+)", log_text):
        parsed = _safe_float(match)
        if parsed is not None:
            values.append(parsed)
    return values


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _full_year_check(dataset_name: str, episode_time_steps: int | None) -> str:
    lowered = dataset_name.lower()
    if "2022" in lowered:
        return "ok" if episode_time_steps == 8760 else "not_full_year_2022"
    if "15s" in lowered or "15_s" in lowered or "15_seconds" in lowered:
        return "ok" if episode_time_steps == 2_102_400 else "not_full_year_15s"
    return "unknown_dataset"


def _bool_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value).lower()


def _first_pipeline_stage(config: dict[str, Any]) -> dict[str, Any]:
    pipeline = config.get("pipeline")
    if not isinstance(pipeline, list) or not pipeline:
        return {}
    stage = pipeline[0]
    return stage if isinstance(stage, dict) else {}


def _build_row(results_dir: Path, summary_row: dict[str, Any]) -> dict[str, Any]:
    job_id = str(summary_row.get("job_id") or "").strip()
    directory = _job_dir(results_dir, job_id)
    job_info = _load_json(directory / "job_info.json")
    status = _load_json(directory / "status.json")
    result = _load_json(directory / "result.json")
    config = _load_yaml(directory / "resolved_config.yaml")
    log_text = (directory / "logs_tail.txt").read_text(encoding="utf-8") if (directory / "logs_tail.txt").exists() else ""

    simulator = config.get("simulator", {}) if isinstance(config.get("simulator"), dict) else {}
    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    tracking = config.get("tracking", {}) if isinstance(config.get("tracking"), dict) else {}
    checkpointing = config.get("checkpointing", {}) if isinstance(config.get("checkpointing"), dict) else {}
    algorithm = _first_pipeline_stage(config)
    replay_buffer = algorithm.get("replay_buffer", {}) if isinstance(algorithm.get("replay_buffer"), dict) else {}
    exploration = algorithm.get("exploration", {}) if isinstance(algorithm.get("exploration"), dict) else {}
    exploration_params = exploration.get("params", {}) if isinstance(exploration.get("params"), dict) else {}
    export = simulator.get("export", {}) if isinstance(simulator.get("export"), dict) else {}
    deucalion_options = job_info.get("deucalion_options", {}) if isinstance(job_info.get("deucalion_options"), dict) else {}

    episodes = _safe_int(simulator.get("episodes"))
    episode_steps = _safe_int(simulator.get("episode_time_steps"))
    configured_steps = episodes * episode_steps if episodes is not None and episode_steps is not None else None
    run_duration = _safe_float(job_info.get("run_duration_seconds") or summary_row.get("run_duration_seconds"))
    seconds_per_step = (
        None
        if run_duration is None or configured_steps is None or configured_steps <= 0
        else run_duration / configured_steps
    )
    env_steps_per_hour = None if seconds_per_step in (None, 0) else 3600.0 / seconds_per_step

    checkpoint_interval = checkpointing.get("checkpoint_interval")
    checkpoint_effective = False
    parsed_checkpoint_interval = _safe_int(checkpoint_interval)
    if parsed_checkpoint_interval is not None:
        checkpoint_effective = configured_steps is None or parsed_checkpoint_interval <= configured_steps

    durations = _extract_step_durations(log_text)
    dataset_name = str(simulator.get("dataset_name") or "")
    full_year = _full_year_check(dataset_name, episode_steps)
    flags: list[str] = []
    if full_year.startswith("not_full_year"):
        flags.append(full_year)
    if bool(tracking.get("mlflow_enabled", False)):
        flags.append("mlflow_enabled")
    if checkpoint_effective:
        flags.append("checkpoint_effective")
    export_kpis = bool(export.get("export_kpis_on_episode_end", False))
    export_final_only = bool(export.get("final_episode_only", False))
    export_kpis_final_only = bool(export.get("kpis_final_episode_only", export_final_only))
    export_timeseries_final_only = bool(export.get("timeseries_final_episode_only", export_final_only))
    export_include_bau = bool(export.get("include_business_as_usual", True))
    export_bau_timeseries = bool(export.get("export_business_as_usual_timeseries", True))

    if export_include_bau:
        flags.append("bau_kpis_enabled")
    if export_bau_timeseries:
        flags.append("bau_timeseries_enabled")
    if not export_kpis:
        flags.append("kpis_disabled")

    row: dict[str, Any] = OrderedDict()
    row["job_id"] = job_id
    row["status"] = summary_row.get("status") or status.get("status") or ""
    row["exit_code"] = summary_row.get("exit_code") or status.get("exit_code") or job_info.get("exit_code") or ""
    row["job_name"] = summary_row.get("job_name") or job_info.get("job_name") or ""
    row["target_host"] = summary_row.get("target_host") or job_info.get("target_host") or ""
    row["image_tag"] = summary_row.get("image_tag") or job_info.get("image_tag") or ""
    row["algorithm"] = summarise_pipeline_algorithms(config, default="") or ""
    row["seed"] = training.get("seed") or _nested(algorithm, "hyperparameters", "seed", default="")
    row["dataset_name"] = dataset_name
    row["episodes"] = episodes
    row["episode_time_steps"] = episode_steps
    row["simulation_start_time_step"] = simulator.get("simulation_start_time_step")
    row["simulation_end_time_step"] = simulator.get("simulation_end_time_step")
    row["configured_env_steps"] = configured_steps
    row["full_year_check"] = full_year
    row["steps_between_training_updates"] = training.get("steps_between_training_updates")
    row["target_update_interval"] = training.get("target_update_interval")
    row["batch_size"] = replay_buffer.get("batch_size")
    row["replay_capacity"] = replay_buffer.get("capacity")
    row["use_amp"] = _bool_str(exploration_params.get("use_amp"))
    row["device_selected"] = summary_row.get("device_selected") or ""
    row["run_duration_seconds"] = run_duration
    row["seconds_per_env_step"] = seconds_per_step
    row["estimated_env_steps_per_hour"] = env_steps_per_hour
    row["queue_wait_seconds"] = summary_row.get("queue_wait_seconds") or job_info.get("queue_wait_seconds") or ""
    row["total_duration_seconds"] = summary_row.get("total_duration_seconds") or job_info.get("total_duration_seconds") or ""
    row["deucalion_partition"] = summary_row.get("deucalion_partition") or deucalion_options.get("partition") or ""
    row["deucalion_time"] = summary_row.get("deucalion_time") or deucalion_options.get("time") or ""
    row["deucalion_cpus_per_task"] = summary_row.get("deucalion_cpus_per_task") or deucalion_options.get("cpus_per_task") or ""
    row["deucalion_mem_gb"] = summary_row.get("deucalion_mem_gb") or deucalion_options.get("mem_gb") or ""
    row["deucalion_gpus"] = summary_row.get("deucalion_gpus") or deucalion_options.get("gpus") or ""
    row["mlflow_enabled"] = _bool_str(tracking.get("mlflow_enabled"))
    row["checkpoint_interval"] = checkpoint_interval
    row["checkpoint_effective"] = _bool_str(checkpoint_effective)
    row["export_kpis_on_episode_end"] = _bool_str(export_kpis)
    row["export_final_episode_only"] = _bool_str(export_final_only)
    row["export_kpis_final_episode_only"] = _bool_str(export_kpis_final_only)
    row["export_timeseries_final_episode_only"] = _bool_str(export_timeseries_final_only)
    row["export_include_business_as_usual"] = _bool_str(export_include_bau)
    row["export_business_as_usual_timeseries"] = _bool_str(export_bau_timeseries)
    row["tail_step_duration_count"] = len(durations)
    row["tail_step_duration_mean_seconds"] = statistics.fmean(durations) if durations else None
    row["tail_step_duration_median_seconds"] = statistics.median(durations) if durations else None
    row["tail_step_duration_p95_seconds"] = _p95(durations)
    row["risk_flags"] = ";".join(flags)

    return row


def build_report(results_dir: Path, summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_build_row(results_dir, row) for row in summary_rows]


def write_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(OUTPUT_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Remote Job Operational Report",
        "",
        f"Generated at Unix time `{time.time():.0f}`.",
        "",
        "## Runtime",
        "",
        "| Job | Status | Host | Algorithm | Dataset | Steps | Device | Runtime | s/step | Queue | Flags |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('job_id') or ''}`",
                    str(row.get("status") or ""),
                    str(row.get("target_host") or ""),
                    str(row.get("algorithm") or ""),
                    str(row.get("dataset_name") or ""),
                    str(row.get("configured_env_steps") or ""),
                    str(row.get("device_selected") or ""),
                    _fmt(row.get("run_duration_seconds"), 2),
                    _fmt(row.get("seconds_per_env_step"), 5),
                    _fmt(row.get("queue_wait_seconds"), 2),
                    f"`{row.get('risk_flags') or ''}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Config Checks",
            "",
            "| Job | MLflow | Checkpoint interval | Checkpoint effective | BAU KPIs | BAU timeseries | KPI export | Legacy final only | KPI final only | Timeseries final only | Update every | Batch | AMP |",
            "|---|---|---:|---|---|---|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('job_id') or ''}`",
                    str(row.get("mlflow_enabled") or ""),
                    str(row.get("checkpoint_interval") or ""),
                    str(row.get("checkpoint_effective") or ""),
                    str(row.get("export_include_business_as_usual") or ""),
                    str(row.get("export_business_as_usual_timeseries") or ""),
                    str(row.get("export_kpis_on_episode_end") or ""),
                    str(row.get("export_final_episode_only") or ""),
                    str(row.get("export_kpis_final_episode_only") or ""),
                    str(row.get("export_timeseries_final_episode_only") or ""),
                    str(row.get("steps_between_training_updates") or ""),
                    str(row.get("batch_size") or ""),
                    str(row.get("use_amp") or ""),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `s/step` is computed from remote job runtime divided by configured environment steps.",
            "- `tail_step_duration_*` in `job_report.csv` is based only on the collected log tail, when step-duration logs are present.",
            "- `full_year_check` is strict: 2022 expects 8760 steps; 15s expects 2102400 steps.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    summary_csv = args.summary_csv or (args.results_dir / "summary.csv")
    rows = _read_summary(summary_csv)
    if not rows:
        raise SystemExit(f"No summary rows found in {summary_csv}")
    report = build_report(args.results_dir, rows)
    write_report_csv(args.results_dir / "job_report.csv", report)
    write_report_markdown(args.results_dir / "job_report.md", report)
    print(args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
