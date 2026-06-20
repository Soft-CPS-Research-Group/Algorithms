"""Run a short policy rollout and summarize action distributions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.registry import ENCODED_OBSERVATION_ALGORITHMS, build_execution_unit
from run_experiment import _resolve_agent_observation_dimensions
from scripts.audit_entity_observations import _agent_building_name, _build_environment, _load_config
from utils.pipeline_utils import pipeline_algorithm_names
from utils.wrapper_citylearn import Wrapper_CityLearn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit action bounds and rollout histograms.")
    parser.add_argument("--config", required=True, help="Experiment config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Audit output directory. Defaults to runs/action_audits/<timestamp>.",
    )
    parser.add_argument("--steps", type=int, default=256, help="Maximum rollout steps to collect.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions instead of the configured exploration path.",
    )
    parser.add_argument("--bins", type=int, default=21, help="Histogram bins per action.")
    parser.add_argument("--job-id", default="action-rollout-audit", help="Wrapper job id for audit metadata.")
    return parser.parse_args()


def _load_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(config.get("simulator", {}).get("dataset_path", "")))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _charger_schema_by_action(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    schema = _load_schema(config)
    buildings = schema.get("buildings", {}) if isinstance(schema, Mapping) else {}
    result: dict[str, dict[str, Any]] = {}

    for building_name, building_cfg in buildings.items():
        if not isinstance(building_cfg, Mapping):
            continue
        chargers = building_cfg.get("chargers", {})
        if not isinstance(chargers, Mapping):
            continue
        for charger_id, charger_cfg in chargers.items():
            if not isinstance(charger_cfg, Mapping):
                continue
            attrs = charger_cfg.get("attributes", {})
            attrs = attrs if isinstance(attrs, Mapping) else {}
            action_name = f"electric_vehicle_storage_{charger_id}"
            result[action_name] = {
                "schema_building": building_name,
                "schema_charger_id": charger_id,
                "charger_type": attrs.get("charger_type"),
                "max_charging_power_kw": attrs.get("max_charging_power"),
                "min_charging_power_kw": attrs.get("min_charging_power"),
                "max_discharging_power_kw": attrs.get("max_discharging_power"),
                "min_discharging_power_kw": attrs.get("min_discharging_power"),
                "phase_connection": attrs.get("phase_connection"),
            }

    return result


def _set_topology_from_wrapper(config: dict[str, Any], wrapper: Wrapper_CityLearn) -> None:
    algorithm_names = pipeline_algorithm_names(config)
    algorithm_name = next(
        (name for name in algorithm_names if name in ENCODED_OBSERVATION_ALGORITHMS),
        algorithm_names[0] if algorithm_names else None,
    )
    topology = config.setdefault("topology", {})
    topology["observation_dimensions"] = _resolve_agent_observation_dimensions(wrapper, algorithm_name)
    topology["action_dimensions"] = list(wrapper.action_dimension)
    topology["num_agents"] = len(wrapper.action_space)


def _build_wrapper_and_agent(config_path: str, output_dir: Path, job_id: str) -> tuple[dict[str, Any], Wrapper_CityLearn]:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    config = _load_config(config_path)
    config["runtime"]["log_dir"] = str(output_dir / "logs")
    config["runtime"]["job_dir"] = str(output_dir)
    export_cfg = config.setdefault("simulator", {}).setdefault("export", {})
    export_cfg["mode"] = "none"
    export_cfg["export_kpis_on_episode_end"] = False

    env = _build_environment(config, output_dir)
    wrapper = Wrapper_CityLearn(
        env=env,
        config=config,
        job_id=job_id,
        progress_path=str(output_dir / "progress.json"),
    )
    _set_topology_from_wrapper(config, wrapper)
    agent = build_execution_unit(config=config)
    wrapper.set_model(agent)
    return config, wrapper


def _describe_actions(config: Mapping[str, Any], wrapper: Wrapper_CityLearn) -> list[dict[str, Any]]:
    env_info = wrapper.describe_environment()
    building_names = env_info.get("building_names") or []
    action_names_by_agent = env_info.get("action_names_by_agent") or {}
    charger_schema = _charger_schema_by_action(config)

    rows: list[dict[str, Any]] = []
    for agent_index, action_space in enumerate(wrapper.action_space):
        building_name = _agent_building_name(building_names, agent_index)
        lows = np.asarray(getattr(action_space, "low", []), dtype=np.float64).reshape(-1)
        highs = np.asarray(getattr(action_space, "high", []), dtype=np.float64).reshape(-1)
        action_names = action_names_by_agent.get(str(agent_index), [])

        for action_index, action_name in enumerate(action_names):
            low = float(lows[action_index]) if action_index < lows.shape[0] else None
            high = float(highs[action_index]) if action_index < highs.shape[0] else None
            schema_row = charger_schema.get(str(action_name), {})
            max_discharge = _to_float(schema_row.get("max_discharging_power_kw"))
            max_charge = _to_float(schema_row.get("max_charging_power_kw"))
            allows_negative = low is not None and low < 0.0
            schema_v2g = max_discharge is not None and max_discharge > 0.0

            issue = ""
            if "electric_vehicle_storage" in str(action_name):
                if schema_v2g and not allows_negative:
                    issue = "schema_v2g_but_action_low_not_negative"
                elif allows_negative and not schema_v2g:
                    issue = "negative_action_but_schema_no_v2g"
                elif max_charge is not None and high is not None and high <= 0.0:
                    issue = "charger_cannot_charge_by_bounds"

            rows.append(
                {
                    "agent_index": agent_index,
                    "building_name": building_name,
                    "action_index": action_index,
                    "action_name": str(action_name),
                    "low": low,
                    "high": high,
                    "allows_negative": allows_negative,
                    "schema_v2g_enabled": schema_v2g if schema_row else "",
                    "schema_building": schema_row.get("schema_building", ""),
                    "schema_charger_id": schema_row.get("schema_charger_id", ""),
                    "charger_type": schema_row.get("charger_type", ""),
                    "max_charging_power_kw": schema_row.get("max_charging_power_kw", ""),
                    "min_charging_power_kw": schema_row.get("min_charging_power_kw", ""),
                    "max_discharging_power_kw": schema_row.get("max_discharging_power_kw", ""),
                    "min_discharging_power_kw": schema_row.get("min_discharging_power_kw", ""),
                    "phase_connection": schema_row.get("phase_connection", ""),
                    "issue": issue,
                }
            )

    return rows


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed) or np.isinf(parsed):
        return None
    return parsed


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _record_action_trace(
    *,
    wrapper: Wrapper_CityLearn,
    max_steps: int,
    deterministic: bool,
) -> list[dict[str, Any]]:
    raw_payload, _ = wrapper.env.reset()
    wrapper.reset()
    if getattr(wrapper, "_entity_interface_mode", False):
        observations = wrapper._apply_entity_layout(raw_payload, force_attach=True)
    else:
        observations = [np.asarray(obs, dtype=np.float64) for obs in raw_payload]
    env_info = wrapper.describe_environment()
    building_names = env_info.get("building_names") or []
    action_names_by_agent = env_info.get("action_names_by_agent") or {}

    rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    step = 0

    while step < max_steps and not (terminated or truncated):
        model = getattr(wrapper, "model", None)
        random_steps = int(getattr(model, "random_exploration_steps", 0) or 0)
        if deterministic:
            phase = "deterministic"
        elif random_steps > 0 and int(getattr(model, "exploration_step", 0) or 0) < random_steps:
            phase = "random_exploration"
        elif hasattr(model, "random_exploration_steps"):
            phase = "policy_noise"
        else:
            phase = "policy"

        raw_actions = wrapper.predict(observations, deterministic=deterministic)
        clipped_actions = wrapper._clip_actions(raw_actions)

        for agent_index, agent_actions in enumerate(clipped_actions):
            building_name = _agent_building_name(building_names, agent_index)
            action_names = action_names_by_agent.get(str(agent_index), [])
            raw_agent_actions = raw_actions[agent_index] if agent_index < len(raw_actions) else []
            for action_index, action_value in enumerate(agent_actions):
                action_name = action_names[action_index] if action_index < len(action_names) else f"action_{action_index}"
                raw_value = raw_agent_actions[action_index] if action_index < len(raw_agent_actions) else None
                rows.append(
                    {
                        "step": step,
                        "phase": phase,
                        "agent_index": agent_index,
                        "building_name": building_name,
                        "action_index": action_index,
                        "action_name": str(action_name),
                        "raw_action": _to_float(raw_value),
                        "clipped_action": float(action_value),
                    }
                )

        env_actions = wrapper._to_env_actions(clipped_actions)
        next_payload, rewards, terminated, truncated, _ = wrapper.env.step(env_actions)
        if getattr(wrapper, "_entity_interface_mode", False):
            observations = wrapper._apply_entity_layout(next_payload, force_attach=False)
        else:
            observations = [np.asarray(obs, dtype=np.float64) for obs in next_payload]
        step += 1

    return rows


def _summarize_actions(trace_rows: list[Mapping[str, Any]], bounds_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values_by_key: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    bounds_by_key = {
        (int(row["agent_index"]), int(row["action_index"]), str(row["action_name"])): row
        for row in bounds_rows
    }

    for row in trace_rows:
        key = (
            str(row.get("phase") or "unknown"),
            int(row["agent_index"]),
            int(row["action_index"]),
            str(row["action_name"]),
        )
        value = _to_float(row.get("clipped_action"))
        if value is not None:
            values_by_key[key].append(value)

    summary_rows: list[dict[str, Any]] = []
    for key, values in sorted(values_by_key.items()):
        phase, agent_index, action_index, action_name = key
        arr = np.asarray(values, dtype=np.float64)
        bounds = bounds_by_key.get((agent_index, action_index, action_name), {})
        low = _to_float(bounds.get("low"))
        high = _to_float(bounds.get("high"))
        eps = 1.0e-6
        summary_rows.append(
            {
                **bounds,
                "phase": phase,
                "agent_index": agent_index,
                "action_index": action_index,
                "action_name": action_name,
                "count": int(arr.shape[0]),
                "min": float(np.min(arr)),
                "p05": float(np.quantile(arr, 0.05)),
                "p25": float(np.quantile(arr, 0.25)),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p75": float(np.quantile(arr, 0.75)),
                "p95": float(np.quantile(arr, 0.95)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)),
                "frac_negative": float(np.mean(arr < -eps)),
                "frac_zero": float(np.mean(np.abs(arr) <= eps)),
                "frac_positive": float(np.mean(arr > eps)),
                "frac_at_low": float(np.mean(arr <= low + eps)) if low is not None else "",
                "frac_at_high": float(np.mean(arr >= high - eps)) if high is not None else "",
            }
        )

    return summary_rows


def _histogram_rows(
    trace_rows: list[Mapping[str, Any]],
    bounds_rows: list[Mapping[str, Any]],
    *,
    bins: int,
) -> list[dict[str, Any]]:
    values_by_key: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    bounds_by_key = {
        (int(row["agent_index"]), int(row["action_index"]), str(row["action_name"])): row
        for row in bounds_rows
    }
    for row in trace_rows:
        key = (
            str(row.get("phase") or "unknown"),
            int(row["agent_index"]),
            int(row["action_index"]),
            str(row["action_name"]),
        )
        value = _to_float(row.get("clipped_action"))
        if value is not None:
            values_by_key[key].append(value)

    rows: list[dict[str, Any]] = []
    for key, values in sorted(values_by_key.items()):
        phase, agent_index, action_index, action_name = key
        bounds = bounds_by_key.get((agent_index, action_index, action_name), {})
        low = _to_float(bounds.get("low"))
        high = _to_float(bounds.get("high"))
        arr = np.asarray(values, dtype=np.float64)
        if low is None or high is None or high <= low:
            low = float(np.min(arr))
            high = float(np.max(arr))
        if high <= low:
            high = low + 1.0
        counts, edges = np.histogram(arr, bins=bins, range=(low, high))
        for bin_index, count in enumerate(counts.tolist()):
            rows.append(
                {
                    "agent_index": agent_index,
                    "phase": phase,
                    "building_name": bounds.get("building_name", ""),
                    "action_index": action_index,
                    "action_name": action_name,
                    "bin_index": bin_index,
                    "bin_left": float(edges[bin_index]),
                    "bin_right": float(edges[bin_index + 1]),
                    "count": int(count),
                }
            )
    return rows


def _write_markdown(
    path: Path,
    *,
    config_path: str,
    steps: int,
    deterministic: bool,
    bounds_rows: list[Mapping[str, Any]],
    summary_rows: list[Mapping[str, Any]],
) -> None:
    ev_rows = [row for row in bounds_rows if "electric_vehicle_storage" in str(row.get("action_name", ""))]
    issues = [row for row in bounds_rows if row.get("issue")]

    lines = [
        "# Action Rollout Audit",
        "",
        f"- Config: `{config_path}`",
        f"- Steps requested: `{steps}`",
        f"- Deterministic: `{deterministic}`",
        f"- Total actions: `{len(bounds_rows)}`",
        f"- EV charger actions: `{len(ev_rows)}`",
        f"- Bounds/schema issues: `{len(issues)}`",
        "",
        "## EV V2G Bounds",
        "",
        "| agent | building | action | bounds | schema V2G | max discharge kW | issue |",
        "|---:|---|---|---|---|---:|---|",
    ]
    for row in ev_rows:
        lines.append(
            f"| {row['agent_index']} | `{row['building_name']}` | `{row['action_name']}` | "
            f"`[{row['low']}, {row['high']}]` | `{row['schema_v2g_enabled']}` | "
            f"{row['max_discharging_power_kw']} | `{row['issue']}` |"
        )

    lines.extend(["", "## Action Summary", ""])
    lines.append("| phase | agent | action | min | p05 | mean | p95 | max | neg | zero | pos | at low | at high |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| `{row['phase']}` | {row['agent_index']} | `{row['action_name']}` | "
            f"{row['min']:.4f} | {row['p05']:.4f} | {row['mean']:.4f} | "
            f"{row['p95']:.4f} | {row['max']:.4f} | {row['frac_negative']:.3f} | "
            f"{row['frac_zero']:.3f} | {row['frac_positive']:.3f} | "
            f"{row['frac_at_low']:.3f} | {row['frac_at_high']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `action_bounds.csv`: env bounds crossed with charger schema.",
            "- `action_trace.csv`: per-step raw/clipped actions.",
            "- `action_summary.csv`: quantiles and saturation fractions.",
            "- `action_histogram_bins.csv`: histogram bins per action.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_action_audit(
    *,
    config_path: str,
    output_dir: Path,
    steps: int,
    deterministic: bool,
    bins: int,
    job_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config, wrapper = _build_wrapper_and_agent(config_path, output_dir, job_id)

    bounds_rows = _describe_actions(config, wrapper)
    trace_rows = _record_action_trace(wrapper=wrapper, max_steps=steps, deterministic=deterministic)
    summary_rows = _summarize_actions(trace_rows, bounds_rows)
    histogram_rows = _histogram_rows(trace_rows, bounds_rows, bins=bins)

    bounds_fields = [
        "agent_index",
        "building_name",
        "action_index",
        "action_name",
        "low",
        "high",
        "allows_negative",
        "schema_v2g_enabled",
        "schema_building",
        "schema_charger_id",
        "charger_type",
        "max_charging_power_kw",
        "min_charging_power_kw",
        "max_discharging_power_kw",
        "min_discharging_power_kw",
        "phase_connection",
        "issue",
    ]
    trace_fields = [
        "step",
        "phase",
        "agent_index",
        "building_name",
        "action_index",
        "action_name",
        "raw_action",
        "clipped_action",
    ]
    summary_fields = ["phase"] + bounds_fields + [
        "count",
        "min",
        "p05",
        "p25",
        "mean",
        "median",
        "p75",
        "p95",
        "max",
        "std",
        "frac_negative",
        "frac_zero",
        "frac_positive",
        "frac_at_low",
        "frac_at_high",
    ]
    histogram_fields = [
        "agent_index",
        "phase",
        "building_name",
        "action_index",
        "action_name",
        "bin_index",
        "bin_left",
        "bin_right",
        "count",
    ]

    _write_csv(output_dir / "action_bounds.csv", bounds_rows, bounds_fields)
    _write_csv(output_dir / "action_trace.csv", trace_rows, trace_fields)
    _write_csv(output_dir / "action_summary.csv", summary_rows, summary_fields)
    _write_csv(output_dir / "action_histogram_bins.csv", histogram_rows, histogram_fields)
    _write_markdown(
        output_dir / "README.md",
        config_path=config_path,
        steps=steps,
        deterministic=deterministic,
        bounds_rows=bounds_rows,
        summary_rows=summary_rows,
    )

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config_path": config_path,
        "steps_requested": steps,
        "steps_recorded": max((int(row["step"]) for row in trace_rows), default=-1) + 1,
        "deterministic": deterministic,
        "total_actions": len(bounds_rows),
        "ev_actions": sum("electric_vehicle_storage" in str(row["action_name"]) for row in bounds_rows),
        "bounds_issues": [row for row in bounds_rows if row.get("issue")],
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = _parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs") / "action_audits" / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    )
    result = run_action_audit(
        config_path=args.config,
        output_dir=output_dir,
        steps=max(args.steps, 1),
        deterministic=bool(args.deterministic),
        bins=max(args.bins, 1),
        job_id=args.job_id,
    )
    print(f"Wrote action audit to {output_dir}")
    print(json.dumps({key: result[key] for key in ("steps_recorded", "total_actions", "ev_actions")}, indent=2))


if __name__ == "__main__":
    main()
