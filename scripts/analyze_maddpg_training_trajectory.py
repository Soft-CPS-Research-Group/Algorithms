"""Summarise MADDPG training trajectories from a run directory.

The script is intentionally file-based and independent from MLflow. It reads the
local JSONL metrics emitted by ``run_experiment.py`` and, when available, the
job log lines that include action/reward vectors.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_METRICS = (
    "RewardComponent/reward_total_mean",
    "RewardComponent/community_settlement_cost_mean",
    "RewardComponent/ev_service_penalty_mean",
    "RewardComponent/ev_schedule_deficit_penalty_mean",
    "RewardComponent/ev_departure_window_penalty_mean",
    "RewardComponent/ev_departure_missed_penalty_amount_mean",
    "RewardComponent/ev_v2g_service_abuse_penalty_mean",
    "RewardComponent/ev_v2g_discharge_kwh_sum_mean",
    "RewardComponent/ev_over_service_penalty_amount_mean",
    "RewardComponent/battery_safety_penalty_mean",
    "RewardComponent/battery_soc_violation_penalty_amount_mean",
    "RewardComponent/battery_throughput_penalty_mean",
    "Action/ev_negative_fraction",
    "Action/ev_positive_fraction",
    "Action/storage_negative_fraction",
    "Action/storage_positive_fraction",
    "Action/storage_idle_fraction",
    "MADDPG/average_critic_loss",
    "MADDPG/average_actor_loss",
    "MADDPG/critic_td_abs_mean",
    "MADDPG/critic_grad_norm_mean",
    "MADDPG/q_expected_std",
    "MADDPG/q_expected_min",
    "MADDPG/q_expected_max",
    "MADDPG/actor_policy_loss_effective_weight",
    "MADDPG/actor_policy_loss_weighted_mean",
    "MADDPG/actor_behavior_cloning_loss_mean",
    "MADDPG/actor_behavior_cloning_ev_loss_mean",
    "MADDPG/actor_behavior_cloning_storage_loss_mean",
    "MADDPG/actor_behavior_cloning_effective_weight",
    "MADDPG/actor_ev_behavior_cloning_zero_target_weight",
    "MADDPG/replay_behavior_action_priority_weight",
    "MADDPG/replay_behavior_action_priority_scope_ev",
)


LOG_PATTERN = re.compile(
    r"Time step: (?P<step>\d+)/(?P<step_total>\d+), "
    r"Episode: (?P<episode>\d+)/(?P<episode_total>\d+), "
    r"Actions: (?P<actions>.*), Rewards: (?P<rewards>\[.*?\]), CPU:",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, help="Run job directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON summaries.")
    parser.add_argument("--agent-index", type=int, default=14, help="Agent to inspect in log critical events.")
    parser.add_argument("--critical-threshold", type=float, default=-0.5, help="Reward threshold for log events.")
    parser.add_argument("--contract-csv", default=None, help="Optional action contract CSV for building/action names.")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric to include in trajectory. Can be repeated; defaults to the standard MADDPG set.",
    )
    return parser.parse_args()


def _read_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not metrics_path.exists():
        return rows
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            metrics = payload.get("metrics") or {}
            rows.append({"step": int(payload.get("step") or 0), "metrics": metrics})
    return rows


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metric_summary(values: list[tuple[int, float]]) -> dict[str, Any]:
    if not values:
        return {}
    numbers = [v for _, v in values]
    return {
        "count": len(numbers),
        "first_step": values[0][0],
        "last_step": values[-1][0],
        "first": numbers[0],
        "last": numbers[-1],
        "delta": numbers[-1] - numbers[0],
        "mean": mean(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _build_metric_outputs(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trajectory_rows: list[dict[str, Any]] = []
    values_by_metric: dict[str, list[tuple[int, float]]] = defaultdict(list)
    agent_values: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for payload in rows:
        step = int(payload["step"])
        metrics_payload = payload["metrics"]
        row: dict[str, Any] = {"step": step}
        for key in metrics:
            value = _finite_float(metrics_payload.get(key))
            if value is not None:
                row[key] = value
                values_by_metric[key].append((step, value))
        for key, raw_value in metrics_payload.items():
            if not key.startswith("Agent_") or not key.endswith("_Reward"):
                continue
            try:
                agent_index = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue
            value = _finite_float(raw_value)
            if value is not None:
                row[key] = value
                agent_values[agent_index].append((step, value))
        trajectory_rows.append(row)

    agent_rows = []
    for agent_index, values in sorted(agent_values.items()):
        numbers = [value for _, value in values]
        agent_rows.append(
            {
                "agent_index": agent_index,
                "count": len(numbers),
                "mean_reward": mean(numbers),
                "min_reward": min(numbers),
                "max_reward": max(numbers),
                "last_reward": numbers[-1],
                "worst_step": values[numbers.index(min(numbers))][0],
            }
        )

    summary = {
        "metric_summary": {key: _metric_summary(values) for key, values in sorted(values_by_metric.items())},
        "agent_reward_summary": agent_rows,
    }
    return trajectory_rows, agent_rows, summary


def _load_contract(contract_csv: Path | None) -> dict[int, dict[str, Any]]:
    if contract_csv is None or not contract_csv.exists():
        return {}
    by_agent: dict[int, dict[str, Any]] = defaultdict(lambda: {"actions": []})
    with contract_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                agent_index = int(row["agent_index"])
            except (KeyError, TypeError, ValueError):
                continue
            by_agent[agent_index]["building_name"] = row.get("building_name") or ""
            by_agent[agent_index]["actions"].append(row.get("action_name") or "")
    return dict(by_agent)


def _parse_log_events(
    log_path: Path,
    *,
    agent_index: int,
    critical_threshold: float,
    contract: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    events: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_PATTERN.search(line)
            if not match:
                continue
            try:
                actions = ast.literal_eval(match.group("actions"))
                rewards = ast.literal_eval(match.group("rewards"))
            except (SyntaxError, ValueError):
                continue
            if agent_index >= len(rewards):
                continue
            reward = _finite_float(rewards[agent_index])
            if reward is None or reward > critical_threshold:
                continue
            agent_actions = actions[agent_index] if agent_index < len(actions) else []
            action_names = contract.get(agent_index, {}).get("actions", [])
            action_pairs = []
            for idx, value in enumerate(agent_actions):
                name = action_names[idx] if idx < len(action_names) else f"action_{idx}"
                action_pairs.append(f"{name}={value}")
            events.append(
                {
                    "episode": int(match.group("episode")),
                    "episode_total": int(match.group("episode_total")),
                    "time_step": int(match.group("step")),
                    "step_total": int(match.group("step_total")),
                    "agent_index": agent_index,
                    "building_name": contract.get(agent_index, {}).get("building_name", ""),
                    "agent_reward": reward,
                    "agent_actions": "; ".join(action_pairs),
                    "all_agent_rewards": json.dumps(rewards),
                }
            )
    return events


def _find_log(job_dir: Path) -> Path | None:
    logs_dir = job_dir / "logs"
    candidates = sorted(logs_dir.glob("*.log"))
    return candidates[0] if candidates else None


def main() -> None:
    args = _parse_args()
    job_dir = Path(args.job_dir)
    output_dir = Path(args.output_dir)
    metrics = tuple(args.metric) if args.metric else DEFAULT_METRICS

    metric_rows = _read_metrics(job_dir / "logs" / "metrics.jsonl")
    trajectory_rows, agent_rows, summary = _build_metric_outputs(metric_rows, metrics)

    trajectory_fields = ["step", *metrics]
    for row in trajectory_rows:
        for key in row:
            if key.startswith("Agent_") and key.endswith("_Reward") and key not in trajectory_fields:
                trajectory_fields.append(key)
    _write_csv(output_dir / "metric_trajectory.csv", trajectory_rows, trajectory_fields)
    _write_csv(
        output_dir / "agent_reward_summary.csv",
        agent_rows,
        ["agent_index", "count", "mean_reward", "min_reward", "max_reward", "last_reward", "worst_step"],
    )

    contract = _load_contract(Path(args.contract_csv) if args.contract_csv else None)
    log_path = _find_log(job_dir)
    critical_events = (
        _parse_log_events(
            log_path,
            agent_index=args.agent_index,
            critical_threshold=float(args.critical_threshold),
            contract=contract,
        )
        if log_path is not None
        else []
    )
    _write_csv(
        output_dir / "critical_log_events.csv",
        critical_events,
        [
            "episode",
            "episode_total",
            "time_step",
            "step_total",
            "agent_index",
            "building_name",
            "agent_reward",
            "agent_actions",
            "all_agent_rewards",
        ],
    )

    summary.update(
        {
            "job_dir": str(job_dir),
            "metrics_path": str(job_dir / "logs" / "metrics.jsonl"),
            "log_path": str(log_path) if log_path else None,
            "metric_rows": len(metric_rows),
            "trajectory_csv": str(output_dir / "metric_trajectory.csv"),
            "agent_reward_summary_csv": str(output_dir / "agent_reward_summary.csv"),
            "critical_log_events_csv": str(output_dir / "critical_log_events.csv"),
            "critical_event_count": len(critical_events),
            "critical_agent_index": args.agent_index,
            "critical_threshold": args.critical_threshold,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
