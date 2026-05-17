"""Audit entity observations/actions exposed to per-building agents."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from citylearn.citylearn import CityLearnEnv
from reward_function.registry import REWARD_FUNCTION_MAP
from run_experiment import _resolve_citylearn_schema_input, _validate_dynamic_entity_schema_input
from utils.config_schema import validate_config
from utils.wrapper_citylearn import Wrapper_CityLearn


BUNDLE_NAMES = (
    "entity_core_electrical",
    "entity_community_operational",
    "entity_forecasts_existing",
    "entity_temporal_derived",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit entity observation/action vectors by agent.")
    parser.add_argument("--config", required=True, help="Training config used to build the environment.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for audit files. Defaults to runs/observation_audits/<timestamp>.",
    )
    parser.add_argument(
        "--job-id",
        default="observation-audit",
        help="Wrapper job id used only for local audit metadata.",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    config = validate_config(raw_config).to_dict()
    runtime = config.setdefault("runtime", {})
    tracking = config.setdefault("tracking", {})
    tracking["mlflow_enabled"] = False
    tracking["log_level"] = str(tracking.get("log_level") or "WARNING")
    return config


def _build_environment(config: Mapping[str, Any], output_dir: Path) -> CityLearnEnv:
    simulator_cfg = dict(config["simulator"])
    export_cfg = dict(simulator_cfg.get("export", {}) or {})
    schema_input = _resolve_citylearn_schema_input(simulator_cfg["dataset_path"])
    interface_mode = str(simulator_cfg.get("interface", "flat")).strip().lower() or "flat"
    topology_mode = str(simulator_cfg.get("topology_mode", "static")).strip().lower() or "static"
    _validate_dynamic_entity_schema_input(
        schema_input,
        interface=interface_mode,
        topology_mode=topology_mode,
    )

    reward_key = simulator_cfg["reward_function"]
    reward_cls = REWARD_FUNCTION_MAP.get(reward_key)
    if reward_cls is None:
        raise ValueError(f"Unsupported reward function '{reward_key}'.")

    env_kwargs: dict[str, Any] = {
        "schema": schema_input,
        "central_agent": simulator_cfg["central_agent"],
        "interface": interface_mode,
        "topology_mode": topology_mode,
        "reward_function": reward_cls,
        "offline": True,
        "render_mode": "none",
        "export_kpis_on_episode_end": False,
        "render_directory": str(output_dir / "simulation_data"),
    }
    reward_function_kwargs = simulator_cfg.get("reward_function_kwargs")
    if isinstance(reward_function_kwargs, dict) and reward_function_kwargs:
        env_kwargs["reward_function_kwargs"] = reward_function_kwargs
    if export_cfg.get("session_name"):
        env_kwargs["render_session_name"] = export_cfg["session_name"]

    for key in ("simulation_start_time_step", "simulation_end_time_step", "episode_time_steps"):
        value = simulator_cfg.get(key)
        if value is not None:
            env_kwargs[key] = value

    return CityLearnEnv(**env_kwargs)


def _agent_building_name(building_names: list[str], agent_index: int) -> str:
    if agent_index < len(building_names):
        return str(building_names[agent_index])
    return f"agent_{agent_index}"


def _extract_building_number(name: str) -> str | None:
    match = re.search(r"Building_(\d+)", name)
    if match:
        return match.group(1)

    match = re.search(r"charger_(\d+)_", name)
    if match:
        return match.group(1)

    return None


def _extract_asset_id(name: str) -> str:
    if "::" not in name:
        return ""

    parts = name.split("::")
    if len(parts) >= 3:
        return parts[1]
    return ""


def _feature_tail(name: str) -> str:
    if "::" in name:
        return name.split("::")[-1]
    if name.startswith("district__"):
        return name.split("__", 1)[1]
    return name


def _is_temporal_feature(name: str) -> bool:
    tail = _feature_tail(name)
    return tail in {"month", "day_type", "hour", "minutes", "seconds"} or tail.endswith("_time_step")


def _is_static_context(name: str) -> bool:
    static_tokens = (
        "capacity_kwh",
        "nominal_power_kw",
        "max_charging_power_kw",
        "max_discharging_power_kw",
        "max_charge_power_kw",
        "max_discharge_power_kw",
        "installed_power_kw",
        "phase_connection_",
    )
    return any(token in name for token in static_tokens)


def _classify_feature(name: str, building_name: str, *, vector: str) -> dict[str, str]:
    notes: list[str] = []
    flags: list[str] = []
    category = "unknown"
    scope = "unknown"
    decision = "local_useful"
    asset_id = _extract_asset_id(name)

    building_number = _extract_building_number(building_name)
    referenced_building_number = _extract_building_number(name)
    if referenced_building_number and building_number and referenced_building_number != building_number:
        flags.append("cross_building_reference")
        notes.append(f"references Building_{referenced_building_number} from {building_name}")

    if name.startswith("district__"):
        scope = "global"
        feature = name.split("__", 1)[1]
        if "phase_headroom" in feature or "building_headroom" in feature:
            category = "electrical_service_global"
        elif feature.startswith("community_"):
            category = "district_global"
        elif feature.startswith("community_net_prev_"):
            category = "district_global"
        else:
            category = "district_global"
        decision = "global_useful"
        if "charging_phase_one_hot" in feature:
            flags.append("asset_specific_global")
            decision = "suspicious"
            notes.append("charger-specific phase feature is global")

    elif name.startswith("charger::"):
        scope = "local"
        feature = _feature_tail(name)
        if "::connected_ev::" in name or "::incoming_ev::" in name:
            category = "ev_local"
        elif "connected_ev_" in feature or "incoming_ev_" in feature:
            category = "ev_local"
            flags.append("possible_ev_denormalized_duplicate")
            notes.append("charger-level EV context may duplicate EV table context")
        elif feature.startswith("phase_connection_"):
            category = "electrical_service_local"
            decision = "local_static_context"
        else:
            category = "charger_local"

    elif name.startswith("deferrable_appliance::"):
        category = "deferrable_local"
        scope = "local"

    elif name.startswith("storage::"):
        category = "storage_local"
        scope = "local"

    elif name.startswith("pv::"):
        category = "pv_local"
        scope = "local"

    elif name.startswith("electric_vehicle_"):
        category = "legacy_alias"
        scope = "local"
        decision = "duplicated"
        flags.append("legacy_alias_duplicate")
        notes.append("raw alias kept for RBC compatibility and dropped by MADDPG encoding profiles")

    elif name in {
        "active_chargers_count",
        "active_storages_count",
        "active_pvs_count",
        "active_deferrable_appliances_count",
        "electric_vehicle_is_flexible",
    }:
        category = "local_topology_meta"
        scope = "local"
        decision = "local_static_context"

    elif "phase_headroom" in name or "building_headroom" in name or "charging_phase_one_hot" in name:
        category = "electrical_service_local"
        scope = "local"
        if "charging_phase_one_hot" in name:
            flags.append("legacy_phase_feature")
            decision = "suspicious"
            notes.append("legacy charger phase feature should live on the local charger table")

    else:
        category = "building_local"
        scope = "local"

    if asset_id and not asset_id.startswith(building_name):
        flags.append("asset_not_owned_by_agent")
        notes.append(f"asset_id={asset_id}")
        decision = "suspicious"

    if _is_static_context(name) and decision == "local_useful":
        decision = "local_static_context"

    if _is_temporal_feature(name):
        flags.append("temporal")

    if vector == "encoded" and ("_sin" in name or "_cos" in name):
        flags.append("cyclic_encoding")

    if flags and "cross_building_reference" in flags:
        decision = "suspicious"

    return {
        "category": category,
        "scope": scope,
        "decision": decision,
        "asset_id": asset_id,
        "flags": ";".join(dict.fromkeys(flags)),
        "notes": "; ".join(dict.fromkeys(notes)),
    }


def _classify_action(name: str, building_name: str) -> dict[str, str]:
    asset_id = _extract_asset_id(name)
    flags: list[str] = []
    notes: list[str] = []
    building_number = _extract_building_number(building_name)
    referenced_building_number = _extract_building_number(name)
    if referenced_building_number and building_number and referenced_building_number != building_number:
        flags.append("cross_building_reference")
        notes.append(f"references Building_{referenced_building_number} from {building_name}")

    base = {
        "category": "building_action",
        "scope": "local",
        "decision": "controlled_action",
        "asset_id": asset_id,
        "flags": ";".join(dict.fromkeys(flags)),
        "notes": "; ".join(dict.fromkeys(notes)),
    }
    if "deferrable_appliance" in name:
        base["category"] = "deferrable_action"
    elif "electric_vehicle" in name or "charger" in name:
        base["category"] = "charger_action"
    elif "electrical_storage" in name or "storage" in name:
        base["category"] = "storage_action"
    else:
        base["category"] = "building_action"

    if flags:
        base["decision"] = "suspicious"

    return base


def _scale_issue(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return ""
    if not math.isfinite(low) or not math.isfinite(high):
        return "non_finite_bounds"
    span = high - low
    if span <= 0.0:
        return "zero_or_negative_span"
    if max(abs(low), abs(high)) >= 1.0e5:
        return "huge_bounds"
    return ""


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _row_base(
    *,
    agent_index: int,
    building_name: str,
    vector: str,
    position: int,
    name: str,
    value: Any,
    low: Any = None,
    high: Any = None,
) -> dict[str, Any]:
    low_value = _float_or_none(low)
    high_value = _float_or_none(high)
    classification = (
        _classify_action(name, building_name)
        if vector == "action"
        else _classify_feature(name, building_name, vector=vector)
    )
    return {
        "agent_index": agent_index,
        "building_name": building_name,
        "vector": vector,
        "position": position,
        "name": name,
        "feature_tail": _feature_tail(name),
        "sample_value": _float_or_none(value),
        "low": low_value,
        "high": high_value,
        "scale_issue": _scale_issue(low_value, high_value),
        **classification,
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = [
        "agent_index",
        "building_name",
        "vector",
        "position",
        "name",
        "feature_tail",
        "category",
        "scope",
        "decision",
        "asset_id",
        "sample_value",
        "low",
        "high",
        "scale_issue",
        "flags",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _count_rows(rows: list[Mapping[str, Any]], *keys: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        label = " / ".join(str(row.get(key, "")) for key in keys)
        counter[label] += 1
    return dict(sorted(counter.items()))


def _write_markdown(path: Path, summary: Mapping[str, Any], flagged_rows: list[Mapping[str, Any]]) -> None:
    lines = [
        "# Entity Observation Audit",
        "",
        f"- Dataset: `{summary['dataset_name']}`",
        f"- Interface: `{summary['interface']}`",
        f"- Topology: `{summary['topology_mode']}`",
        f"- Seconds per step: `{summary['seconds_per_time_step']}`",
        f"- Agents: `{summary['num_agents']}`",
        "",
        "## Bundles",
        "",
    ]
    for name, active in summary["observation_bundles"].items():
        lines.append(f"- `{name}`: `{active}`")

    lines.extend(["", "## Agent Summary", ""])
    lines.append("| agent | building | raw | encoded | actions | suspicious | duplicated | scale issues |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for agent in summary["agents"]:
        lines.append(
            "| {agent_index} | `{building_name}` | {raw_count} | {encoded_count} | {action_count} | "
            "{suspicious_count} | {duplicated_count} | {scale_issue_count} |".format(**agent)
        )

    lines.extend(["", "## Flagged Samples", ""])
    if flagged_rows:
        lines.append("| agent | building | vector | feature | decision | scale | flags | notes |")
        lines.append("|---:|---|---|---|---|---|---|---|")
        for row in flagged_rows[:80]:
            lines.append(
                f"| {row['agent_index']} | `{row['building_name']}` | `{row['vector']}` | "
                f"`{row['name']}` | `{row['decision']}` | `{row['scale_issue']}` | "
                f"`{row['flags']}` | {row['notes']} |"
            )
    else:
        lines.append("No suspicious, duplicated, or scale-issue rows detected by the current audit rules.")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `observation_features.csv`: raw and encoded observation rows.",
            "- `action_features.csv`: action rows.",
            "- `summary.json`: machine-readable counts and metadata.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bundle_status(env: CityLearnEnv, config: Mapping[str, Any]) -> dict[str, bool | None]:
    specs = getattr(env, "entity_specs", None)
    bundles = None
    if isinstance(specs, Mapping):
        meta = specs.get("meta", {})
        if isinstance(meta, Mapping):
            bundles = meta.get("observation_bundles")

    if isinstance(bundles, Mapping):
        return {name: bool(bundles.get(name, False)) for name in BUNDLE_NAMES}

    dataset_path = Path(str(config.get("simulator", {}).get("dataset_path", "")))
    if dataset_path.exists():
        try:
            schema = json.loads(dataset_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            schema = {}
        raw_bundles = schema.get("observation_bundles", {}) if isinstance(schema, Mapping) else {}
        if isinstance(raw_bundles, Mapping):
            return {
                name: bool((raw_bundles.get(name, {}) or {}).get("active", False))
                for name in BUNDLE_NAMES
            }

    return {name: None for name in BUNDLE_NAMES}


def run_audit(config_path: str, output_dir: Path, job_id: str) -> dict[str, Any]:
    config = _load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config["runtime"]["log_dir"] = str(output_dir / "logs")
    config["runtime"]["job_dir"] = str(output_dir)

    env = _build_environment(config, output_dir)
    wrapper = Wrapper_CityLearn(
        env=env,
        config=config,
        job_id=job_id,
        progress_path=str(output_dir / "progress.json"),
    )

    raw_payload, _ = wrapper.env.reset()
    raw_observations = wrapper._apply_entity_layout(raw_payload, force_attach=True)
    raw_names = [[str(name) for name in group] for group in wrapper.observation_names]
    encoded_names = (
        wrapper._entity_adapter.encoded_observation_names(raw_names)
        if getattr(wrapper, "_entity_adapter", None) is not None
        else raw_names
    )
    encoded_observations = wrapper.get_all_encoded_observations(raw_observations)
    building_names = wrapper.describe_environment().get("building_names") or []
    action_names_by_agent = wrapper.describe_environment().get("action_names_by_agent") or {}

    observation_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []

    for agent_index, names in enumerate(raw_names):
        building_name = _agent_building_name(building_names, agent_index)
        raw_values = np.asarray(raw_observations[agent_index], dtype=np.float64)
        raw_space = wrapper.observation_space[agent_index]
        lows = np.asarray(getattr(raw_space, "low", []), dtype=np.float64)
        highs = np.asarray(getattr(raw_space, "high", []), dtype=np.float64)

        for position, name in enumerate(names):
            observation_rows.append(
                _row_base(
                    agent_index=agent_index,
                    building_name=building_name,
                    vector="raw",
                    position=position,
                    name=name,
                    value=raw_values[position] if position < raw_values.shape[0] else None,
                    low=lows[position] if position < lows.shape[0] else None,
                    high=highs[position] if position < highs.shape[0] else None,
                )
            )

        encoded_values = np.asarray(encoded_observations[agent_index], dtype=np.float64)
        for position, name in enumerate(encoded_names[agent_index]):
            observation_rows.append(
                _row_base(
                    agent_index=agent_index,
                    building_name=building_name,
                    vector="encoded",
                    position=position,
                    name=name,
                    value=encoded_values[position] if position < encoded_values.shape[0] else None,
                )
            )

        action_names = action_names_by_agent.get(str(agent_index), [])
        action_space = wrapper.action_space[agent_index]
        action_lows = np.asarray(getattr(action_space, "low", []), dtype=np.float64)
        action_highs = np.asarray(getattr(action_space, "high", []), dtype=np.float64)
        for position, name in enumerate(action_names):
            action_rows.append(
                _row_base(
                    agent_index=agent_index,
                    building_name=building_name,
                    vector="action",
                    position=position,
                    name=str(name),
                    value=0.0,
                    low=action_lows[position] if position < action_lows.shape[0] else None,
                    high=action_highs[position] if position < action_highs.shape[0] else None,
                )
            )

        agent_observation_rows = [row for row in observation_rows if row["agent_index"] == agent_index]
        agent_action_rows = [row for row in action_rows if row["agent_index"] == agent_index]
        agents.append(
            {
                "agent_index": agent_index,
                "building_name": building_name,
                "raw_count": len(names),
                "encoded_count": len(encoded_names[agent_index]),
                "action_count": len(action_names),
                "suspicious_count": sum(row["decision"] == "suspicious" for row in agent_observation_rows),
                "duplicated_count": sum(row["decision"] == "duplicated" for row in agent_observation_rows),
                "scale_issue_count": sum(bool(row["scale_issue"]) for row in agent_observation_rows),
                "categories": _count_rows(agent_observation_rows, "vector", "category"),
                "decisions": _count_rows(agent_observation_rows, "vector", "decision"),
            }
        )

    suspicious_rows = [
        row
        for row in observation_rows
        if row["decision"] in {"suspicious", "duplicated"} or row["scale_issue"]
    ]

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config_path": str(config_path),
        "dataset_name": config["simulator"]["dataset_name"],
        "interface": getattr(env, "interface", None),
        "topology_mode": getattr(env, "topology_mode", None),
        "seconds_per_time_step": getattr(env, "seconds_per_time_step", None),
        "num_agents": len(raw_names),
        "observation_bundles": _bundle_status(env, config),
        "total_observation_rows": len(observation_rows),
        "total_action_rows": len(action_rows),
        "observation_categories": _count_rows(observation_rows, "vector", "category"),
        "observation_decisions": _count_rows(observation_rows, "vector", "decision"),
        "action_categories": _count_rows(action_rows, "category"),
        "agents": agents,
    }

    _write_csv(output_dir / "observation_features.csv", observation_rows)
    _write_csv(output_dir / "action_features.csv", action_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "README.md", summary, suspicious_rows)

    return summary


def main() -> None:
    args = _parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs") / "observation_audits" / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    )
    summary = run_audit(args.config, output_dir, args.job_id)
    print(f"Wrote audit to {output_dir}")
    print(json.dumps({key: summary[key] for key in ("num_agents", "observation_bundles")}, indent=2))


if __name__ == "__main__":
    main()
