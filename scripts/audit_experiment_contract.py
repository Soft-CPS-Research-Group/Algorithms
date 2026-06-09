"""Audit the end-to-end experiment contract for training/inference bundles."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_experiment import _resolve_agent_observation_dimensions
from algorithms.registry import ENCODED_OBSERVATION_ALGORITHMS
from scripts.audit_action_rollout import _build_wrapper_and_agent, _describe_actions
from utils.artifact_manifest import build_manifest, write_manifest
from utils.bundle_validator import validate_bundle_contract
from utils.pipeline_utils import pipeline_algorithm_names, summarise_pipeline_algorithms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit observations/actions/export manifest for an experiment config.")
    parser.add_argument("--config", required=True, help="Experiment config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Contract output directory. Defaults to runs/pipeline_contracts/<timestamp>.",
    )
    parser.add_argument("--job-id", default="pipeline-contract-audit", help="Wrapper job id for local audit metadata.")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip artifact export/manifest validation and only write observation/action contracts.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return str(value)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reset_agent_observations(wrapper: Any) -> list[np.ndarray]:
    raw_payload, _ = wrapper.env.reset()
    if getattr(wrapper, "_entity_interface_mode", False):
        return wrapper._apply_entity_layout(raw_payload, force_attach=True)
    return [np.asarray(obs, dtype=np.float64) for obs in raw_payload]


def _building_names(env_info: Mapping[str, Any], count: int) -> list[str]:
    names = env_info.get("building_names")
    if isinstance(names, list) and len(names) >= count:
        return [str(name) for name in names[:count]]
    return [f"agent_{idx}" for idx in range(count)]


def _encoder_width(encoder: Mapping[str, Any], sample_value: float) -> int:
    encoder_type = str(encoder.get("type", ""))
    params = encoder.get("params") if isinstance(encoder.get("params"), Mapping) else {}
    if encoder_type == "OnehotEncoding":
        classes = params.get("classes")
        return len(classes) if isinstance(classes, list) else 1
    if encoder_type == "PeriodicNormalization":
        return 2
    if encoder_type == "RemoveFeature":
        return 0
    _ = sample_value
    return 1


def _observation_rows(
    *,
    wrapper: Any,
    env_info: Mapping[str, Any],
    raw_observations: list[np.ndarray],
    encoded_observations: list[np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_names = env_info.get("raw_observation_names") or env_info.get("observation_names") or []
    serving_names = env_info.get("observation_names") or []
    encoders = env_info.get("encoders") or []
    building_names = _building_names(env_info, len(raw_names))

    raw_rows: list[dict[str, Any]] = []
    serving_rows: list[dict[str, Any]] = []

    for agent_index, names in enumerate(raw_names):
        building_name = building_names[agent_index] if agent_index < len(building_names) else f"agent_{agent_index}"
        values = np.asarray(raw_observations[agent_index], dtype=np.float64)
        space = wrapper.observation_space[agent_index] if agent_index < len(wrapper.observation_space) else None
        lows = np.asarray(getattr(space, "low", []), dtype=np.float64).reshape(-1)
        highs = np.asarray(getattr(space, "high", []), dtype=np.float64).reshape(-1)

        for position, name in enumerate(names):
            raw_rows.append(
                {
                    "agent_index": agent_index,
                    "building_name": building_name,
                    "position": position,
                    "name": str(name),
                    "sample_value": float(values[position]) if position < values.shape[0] else "",
                    "low": float(lows[position]) if position < lows.shape[0] else "",
                    "high": float(highs[position]) if position < highs.shape[0] else "",
                }
            )

    for agent_index, names in enumerate(serving_names):
        building_name = building_names[agent_index] if agent_index < len(building_names) else f"agent_{agent_index}"
        values = np.asarray(raw_observations[agent_index], dtype=np.float64)
        encoded_values = np.asarray(encoded_observations[agent_index], dtype=np.float64)
        agent_encoders = encoders[agent_index] if agent_index < len(encoders) else []
        encoded_cursor = 0

        for position, name in enumerate(names):
            encoder = agent_encoders[position] if position < len(agent_encoders) else {}
            sample_value = float(values[position]) if position < values.shape[0] else 0.0
            encoded_width = _encoder_width(encoder, sample_value)
            if len(names) == encoded_values.shape[0]:
                encoded_sample = float(encoded_values[position])
            elif encoded_width == 1 and encoded_cursor < encoded_values.shape[0]:
                encoded_sample = float(encoded_values[encoded_cursor])
            else:
                encoded_sample = ""

            serving_rows.append(
                {
                    "agent_index": agent_index,
                    "building_name": building_name,
                    "position": position,
                    "name": str(name),
                    "encoder_type": str(encoder.get("type", "")) if isinstance(encoder, Mapping) else "",
                    "encoder_params": json.dumps(encoder.get("params", {}), sort_keys=True, default=_json_default)
                    if isinstance(encoder, Mapping)
                    else "{}",
                    "encoded_width": encoded_width,
                    "encoded_sample": encoded_sample,
                }
            )
            encoded_cursor += encoded_width

    return raw_rows, serving_rows


def _export_manifest(
    *,
    output_dir: Path,
    config: dict[str, Any],
    wrapper: Any,
    env_info: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts_root = output_dir / "bundle"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    agent = wrapper.model
    agent_metadata = agent.export_artifacts(
        output_dir=str(artifacts_root),
        context={
            "topology": config.get("topology", {}),
            "environment": dict(env_info),
            "config": config,
        },
    )
    manifest = build_manifest(config, dict(env_info), agent_metadata)
    validate_bundle_contract(manifest, artifacts_root)
    manifest_path = write_manifest(manifest, str(artifacts_root))
    artifacts = manifest.get("agent", {}).get("artifacts", [])
    return {
        "manifest_path": str(manifest_path),
        "manifest_valid": True,
        "agent_artifact_count": len(artifacts) if isinstance(artifacts, list) else 0,
        "agent_format": manifest.get("agent", {}).get("format"),
    }


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Experiment Contract Audit",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Dataset: `{summary['dataset_name']}`",
        f"- Algorithm: `{summary['algorithm_name']}`",
        f"- Interface: `{summary['interface']}`",
        f"- Topology mode: `{summary['topology_mode']}`",
        f"- Seconds per step: `{summary['seconds_per_time_step']}`",
        f"- Agents: `{summary['num_agents']}`",
        f"- Total actions: `{summary['total_actions']}`",
        f"- Manifest valid: `{summary['manifest_valid']}`",
        "",
        "## Dimensions",
        "",
        "| agent | building | raw obs | serving obs | encoded dim | actions |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for agent in summary["agents"]:
        lines.append(
            "| {agent_index} | `{building_name}` | {raw_observation_count} | "
            "{serving_observation_count} | {encoded_observation_dimension} | {action_count} |".format(**agent)
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `raw_observations.csv`: simulator/wrapper raw observation names.",
            "- `serving_observations.csv`: names and encoders exported for inference.",
            "- `action_contract.csv`: action names and bounds.",
            "- `summary.json`: machine-readable contract summary.",
            "- `bundle/artifact_manifest.json`: exported bundle manifest, if export was enabled.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_contract_audit(
    *,
    config_path: str,
    output_dir: Path,
    job_id: str,
    skip_export: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config, wrapper = _build_wrapper_and_agent(config_path, output_dir, job_id)

    raw_observations = _reset_agent_observations(wrapper)
    encoded_observations = wrapper.get_all_encoded_observations(raw_observations)
    env_info = wrapper.describe_environment()
    building_names = _building_names(env_info, len(raw_observations))
    action_rows = _describe_actions(config, wrapper)

    raw_rows, serving_rows = _observation_rows(
        wrapper=wrapper,
        env_info=env_info,
        raw_observations=raw_observations,
        encoded_observations=encoded_observations,
    )
    _write_csv(
        output_dir / "raw_observations.csv",
        raw_rows,
        ["agent_index", "building_name", "position", "name", "sample_value", "low", "high"],
    )
    _write_csv(
        output_dir / "serving_observations.csv",
        serving_rows,
        ["agent_index", "building_name", "position", "name", "encoder_type", "encoder_params", "encoded_width", "encoded_sample"],
    )
    _write_csv(
        output_dir / "action_contract.csv",
        action_rows,
        [
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
        ],
    )

    algorithm_names = pipeline_algorithm_names(config)
    dimension_algorithm_name = next(
        (name for name in algorithm_names if name in ENCODED_OBSERVATION_ALGORITHMS),
        algorithm_names[0] if algorithm_names else None,
    )
    algorithm_name = summarise_pipeline_algorithms(config, default="") or ""
    config.setdefault("topology", {})["observation_dimensions"] = _resolve_agent_observation_dimensions(
        wrapper,
        dimension_algorithm_name,
    )
    config["topology"]["action_dimensions"] = list(wrapper.action_dimension)
    config["topology"]["num_agents"] = len(wrapper.action_space)

    manifest_result = {
        "manifest_path": None,
        "manifest_valid": None,
        "agent_artifact_count": 0,
        "agent_format": None,
    }
    if not skip_export:
        manifest_result = _export_manifest(
            output_dir=output_dir,
            config=config,
            wrapper=wrapper,
            env_info=env_info,
        )

    raw_names = env_info.get("raw_observation_names") or env_info.get("observation_names") or []
    serving_names = env_info.get("observation_names") or []
    action_names_by_agent = env_info.get("action_names_by_agent") or {}

    agents = []
    for agent_index in range(len(raw_observations)):
        action_count = len(action_names_by_agent.get(str(agent_index), [])) if isinstance(action_names_by_agent, Mapping) else 0
        agents.append(
            {
                "agent_index": agent_index,
                "building_name": building_names[agent_index] if agent_index < len(building_names) else f"agent_{agent_index}",
                "raw_observation_count": len(raw_names[agent_index]) if agent_index < len(raw_names) else 0,
                "serving_observation_count": len(serving_names[agent_index]) if agent_index < len(serving_names) else 0,
                "encoded_observation_dimension": int(np.asarray(encoded_observations[agent_index]).reshape(-1).shape[0]),
                "action_count": action_count,
            }
        )

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config_path": config_path,
        "dataset_name": config.get("simulator", {}).get("dataset_name"),
        "algorithm_name": algorithm_name,
        "interface": env_info.get("interface"),
        "topology_mode": env_info.get("topology_mode"),
        "seconds_per_time_step": env_info.get("seconds_per_time_step"),
        "num_agents": len(raw_observations),
        "total_actions": len(action_rows),
        "ev_actions": sum("electric_vehicle_storage" in str(row.get("action_name", "")) for row in action_rows),
        "bounds_issues": [row for row in action_rows if row.get("issue")],
        "topology": config.get("topology", {}),
        "entity_encoding": env_info.get("entity_encoding"),
        "agents": agents,
        **manifest_result,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    _write_markdown(output_dir / "README.md", summary)
    return summary


def main() -> None:
    args = _parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs") / "pipeline_contracts" / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    )
    summary = run_contract_audit(
        config_path=args.config,
        output_dir=output_dir,
        job_id=args.job_id,
        skip_export=bool(args.skip_export),
    )
    print(f"Wrote contract audit to {output_dir}")
    print(
        json.dumps(
            {
                "num_agents": summary["num_agents"],
                "total_actions": summary["total_actions"],
                "manifest_valid": summary["manifest_valid"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
