"""Generate a matrix of training/inference contract audits."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_experiment_contract import run_contract_audit
from scripts.bechmark_agents import DEFAULT_KPIS, KPI_ALIASES
from utils.config_schema import validate_config


DEFAULT_CONFIGS = (
    "configs/templates/maddpg/maddpg_local.yaml",
    "configs/templates/maddpg/maddpg_2022_all_plus_evs_local.yaml",
)
DEFAULT_PROFILES = ("maddpg_v1", "maddpg_v2_compact")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run contract audits for multiple configs/entity encoding profiles and "
            "write one aggregate training-contract table."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Base config to audit. Can be repeated. Defaults to the local MADDPG "
            "15s and 2022 all-plus-EVs templates."
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help=(
            "Entity encoding profile variant to generate. Can be repeated. "
            "Defaults to maddpg_v1 and maddpg_v2_compact."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to runs/training_contracts/<timestamp>.",
    )
    parser.add_argument(
        "--matrix-name",
        default=None,
        help="Optional readable matrix name used in metadata and generated config names.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip artifact export/manifest validation inside each contract audit.",
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "item"


def _profile_label(config: Mapping[str, Any], requested_profile: str | None) -> str:
    if requested_profile:
        return requested_profile
    encoding = ((config.get("simulator") or {}).get("entity_encoding") or {})
    return str(encoding.get("profile") or encoding.get("normalization") or "raw")


def _build_variant_config(
    *,
    base_config_path: Path,
    profile: str | None,
    matrix_name: str,
) -> tuple[dict[str, Any], str]:
    raw = _load_yaml(base_config_path)
    config = validate_config(raw).to_dict()
    profile_name = _profile_label(config, profile)

    simulator = config.setdefault("simulator", {})
    if profile is not None:
        entity_encoding = simulator.setdefault("entity_encoding", {})
        entity_encoding["enabled"] = True
        entity_encoding["normalization"] = "minmax_space"
        entity_encoding["profile"] = profile
        entity_encoding["clip"] = bool(entity_encoding.get("clip", True))

    metadata = config.setdefault("metadata", {})
    dataset_name = str(simulator.get("dataset_name", "dataset"))
    algorithm_name = str((config.get("algorithm") or {}).get("name", "algorithm"))
    variant_id = _slug(f"{dataset_name}__{algorithm_name}__{profile_name}")
    metadata["experiment_name"] = _slug(f"{matrix_name}_{variant_id}")
    metadata["run_name"] = f"{algorithm_name} {dataset_name} {profile_name}"
    metadata["description"] = (
        f"Training contract audit variant generated from {base_config_path} "
        f"with entity profile {profile_name}."
    )
    return config, variant_id


def _summarize_agent_dims(agents: list[Mapping[str, Any]], key: str) -> tuple[int | None, int | None]:
    values = [int(agent.get(key, 0) or 0) for agent in agents]
    if not values:
        return None, None
    return min(values), max(values)


def _matrix_row(
    *,
    summary: Mapping[str, Any],
    variant_id: str,
    profile: str,
    config_path: Path,
    contract_dir: Path,
) -> dict[str, Any]:
    agents = list(summary.get("agents") or [])
    raw_min, raw_max = _summarize_agent_dims(agents, "raw_observation_count")
    serving_min, serving_max = _summarize_agent_dims(agents, "serving_observation_count")
    encoded_min, encoded_max = _summarize_agent_dims(agents, "encoded_observation_dimension")
    action_min, action_max = _summarize_agent_dims(agents, "action_count")
    entity_encoding = summary.get("entity_encoding") if isinstance(summary.get("entity_encoding"), Mapping) else {}
    topology = summary.get("topology") if isinstance(summary.get("topology"), Mapping) else {}
    bounds_issues = list(summary.get("bounds_issues") or [])

    return {
        "variant_id": variant_id,
        "dataset_name": summary.get("dataset_name"),
        "algorithm_name": summary.get("algorithm_name"),
        "profile": profile,
        "serving_observation_names": entity_encoding.get("serving_observation_names", ""),
        "seconds_per_time_step": summary.get("seconds_per_time_step"),
        "interface": summary.get("interface"),
        "topology_mode": summary.get("topology_mode"),
        "num_agents": summary.get("num_agents"),
        "total_actions": summary.get("total_actions"),
        "ev_actions": summary.get("ev_actions"),
        "raw_obs_min": raw_min,
        "raw_obs_max": raw_max,
        "serving_obs_min": serving_min,
        "serving_obs_max": serving_max,
        "encoded_dim_min": encoded_min,
        "encoded_dim_max": encoded_max,
        "action_count_min": action_min,
        "action_count_max": action_max,
        "topology_observation_dimensions": json.dumps(topology.get("observation_dimensions", [])),
        "topology_action_dimensions": json.dumps(topology.get("action_dimensions", [])),
        "bounds_issue_count": len(bounds_issues),
        "manifest_valid": summary.get("manifest_valid"),
        "agent_artifact_count": summary.get("agent_artifact_count"),
        "agent_format": summary.get("agent_format"),
        "generated_config": str(config_path),
        "contract_dir": str(contract_dir),
    }


def _agent_rows(
    *,
    summary: Mapping[str, Any],
    variant_id: str,
    profile: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent in list(summary.get("agents") or []):
        rows.append(
            {
                "variant_id": variant_id,
                "dataset_name": summary.get("dataset_name"),
                "algorithm_name": summary.get("algorithm_name"),
                "profile": profile,
                "agent_index": agent.get("agent_index"),
                "building_name": agent.get("building_name"),
                "raw_observation_count": agent.get("raw_observation_count"),
                "serving_observation_count": agent.get("serving_observation_count"),
                "encoded_observation_dimension": agent.get("encoded_observation_dimension"),
                "action_count": agent.get("action_count"),
            }
        )
    return rows


def _write_kpi_contract(path: Path) -> None:
    payload = {
        "purpose": "Expected KPI names for later benchmark comparison; actual KPI values require a real experiment run.",
        "tracked_kpis": list(DEFAULT_KPIS),
        "aliases": KPI_ALIASES,
        "notes": [
            "Contract audits validate pipeline shape and export compatibility; they do not execute full KPI-producing runs.",
            "Use these KPI names when aggregating result.json or simulator exported_kpis.csv files in the benchmark phase.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_readme(path: Path, *, matrix_name: str, matrix_rows: list[Mapping[str, Any]]) -> None:
    lines = [
        "# Training Contract Matrix",
        "",
        f"- Matrix: `{matrix_name}`",
        f"- Variants: `{len(matrix_rows)}`",
        "",
        "## Variants",
        "",
        "| variant | dataset | algorithm | profile | agents | actions | encoded dim range | manifest |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in matrix_rows:
        lines.append(
            "| `{variant_id}` | `{dataset_name}` | `{algorithm_name}` | `{profile}` | "
            "{num_agents} | {total_actions} | {encoded_dim_min}-{encoded_dim_max} | {manifest_valid} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `matrix_summary.csv`: one row per config/profile variant.",
            "- `agent_contract_summary.csv`: one row per agent per variant.",
            "- `matrix_summary.json`: machine-readable summary including KPI contract metadata.",
            "- `kpi_contract.json`: expected KPI names for the benchmark phase.",
            "- `generated_configs/`: concrete configs generated from base templates.",
            "- `<variant_id>/`: full contract audit with raw/serving observations, actions and manifest.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training_contract_matrix(
    *,
    config_paths: list[Path],
    profiles: list[str],
    output_dir: Path,
    matrix_name: str,
    skip_export: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_config_dir = output_dir / "generated_configs"
    matrix_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []

    for base_config_path in config_paths:
        profile_variants: list[str | None] = profiles or [None]
        for profile in profile_variants:
            config, variant_id = _build_variant_config(
                base_config_path=base_config_path,
                profile=profile,
                matrix_name=matrix_name,
            )
            profile_name = _profile_label(config, profile)
            generated_config_path = generated_config_dir / f"{variant_id}.yaml"
            _write_yaml(generated_config_path, config)

            contract_dir = output_dir / variant_id
            summary = run_contract_audit(
                config_path=str(generated_config_path),
                output_dir=contract_dir,
                job_id=f"training-contract-{variant_id}",
                skip_export=skip_export,
            )
            matrix_rows.append(
                _matrix_row(
                    summary=summary,
                    variant_id=variant_id,
                    profile=profile_name,
                    config_path=generated_config_path,
                    contract_dir=contract_dir,
                )
            )
            agent_rows.extend(_agent_rows(summary=summary, variant_id=variant_id, profile=profile_name))

    matrix_fieldnames = [
        "variant_id",
        "dataset_name",
        "algorithm_name",
        "profile",
        "serving_observation_names",
        "seconds_per_time_step",
        "interface",
        "topology_mode",
        "num_agents",
        "total_actions",
        "ev_actions",
        "raw_obs_min",
        "raw_obs_max",
        "serving_obs_min",
        "serving_obs_max",
        "encoded_dim_min",
        "encoded_dim_max",
        "action_count_min",
        "action_count_max",
        "topology_observation_dimensions",
        "topology_action_dimensions",
        "bounds_issue_count",
        "manifest_valid",
        "agent_artifact_count",
        "agent_format",
        "generated_config",
        "contract_dir",
    ]
    agent_fieldnames = [
        "variant_id",
        "dataset_name",
        "algorithm_name",
        "profile",
        "agent_index",
        "building_name",
        "raw_observation_count",
        "serving_observation_count",
        "encoded_observation_dimension",
        "action_count",
    ]

    _write_csv(output_dir / "matrix_summary.csv", matrix_rows, matrix_fieldnames)
    _write_csv(output_dir / "agent_contract_summary.csv", agent_rows, agent_fieldnames)
    _write_kpi_contract(output_dir / "kpi_contract.json")

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "matrix_name": matrix_name,
        "config_paths": [str(path) for path in config_paths],
        "profiles": profiles,
        "skip_export": skip_export,
        "tracked_kpis": list(DEFAULT_KPIS),
        "kpi_aliases": KPI_ALIASES,
        "variants": matrix_rows,
    }
    (output_dir / "matrix_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_readme(output_dir / "README.md", matrix_name=matrix_name, matrix_rows=matrix_rows)
    return payload


def main() -> None:
    args = _parse_args()
    config_paths = [Path(path) for path in (args.config or DEFAULT_CONFIGS)]
    profiles = list(args.profile or DEFAULT_PROFILES)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs") / "training_contracts" / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    )
    matrix_name = args.matrix_name or output_dir.name
    payload = run_training_contract_matrix(
        config_paths=config_paths,
        profiles=profiles,
        output_dir=output_dir,
        matrix_name=matrix_name,
        skip_export=bool(args.skip_export),
    )
    print(f"Wrote training contract matrix to {output_dir}")
    print(
        json.dumps(
            {
                "variants": len(payload["variants"]),
                "profiles": payload["profiles"],
                "tracked_kpis": payload["tracked_kpis"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
