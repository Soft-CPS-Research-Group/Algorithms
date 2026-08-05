#!/usr/bin/env python3
"""Package a replay-validated total-energy schedule as a portable demonstration."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_paths(value: Any, *, dataset_name: str) -> Any:
    if isinstance(value, list):
        return [_portable_paths(item, dataset_name=dataset_name) for item in value]
    if not isinstance(value, dict):
        return value

    converted: dict[str, Any] = {}
    for key, item in value.items():
        if key == "dataset_root":
            converted[key] = f"datasets/{dataset_name}"
        elif key == "schema_path":
            converted[key] = f"datasets/{dataset_name}/schema.json"
        else:
            converted[key] = _portable_paths(item, dataset_name=dataset_name)
    return converted


def _absolute_strings(value: Any, *, location: str = "$") -> list[str]:
    if isinstance(value, Mapping):
        findings: list[str] = []
        for key, item in value.items():
            findings.extend(_absolute_strings(item, location=f"{location}.{key}"))
        return findings
    if isinstance(value, list):
        findings = []
        for index, item in enumerate(value):
            findings.extend(_absolute_strings(item, location=f"{location}[{index}]"))
        return findings
    if isinstance(value, str) and Path(value).is_absolute():
        return [f"{location}={value}"]
    return []


def _audit_evidence(path: Path, *, run_name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if item.get("run") == run_name), None)
    if row is None:
        raise ValueError(f"Audit summary has no run named {run_name!r}.")
    if row.get("all_buildings_pass_local_gates") != "1":
        raise ValueError(f"Audit run {run_name!r} did not pass all local gates.")
    return {
        "run": run_name,
        "building_count": int(row["building_count"]),
        "local_gate_pass_count": int(row["local_gate_pass_count"]),
        "local_gate_reject_count": int(row["local_gate_reject_count"]),
        "local_cost_eur_sum": float(row["local_cost_eur_sum"]),
        "buildings_beating_baseline_count": int(
            row["buildings_beating_baseline_count"]
        ),
        "all_buildings_pass_local_gates": True,
    }


def package_demonstration(
    *,
    schedule_path: Path,
    solve_summary_path: Path,
    audit_summary_path: Path,
    audit_run: str,
    output_dir: Path,
    dataset_name: str,
    demonstration_name: str,
    source_commit: str,
    source_replay_job_id: str,
    allow_inexact_boundary: bool = False,
) -> tuple[Path, Path]:
    schedule_path = schedule_path.resolve()
    solve_summary_path = solve_summary_path.resolve()
    audit_summary_path = audit_summary_path.resolve()
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    solve_summary = json.loads(solve_summary_path.read_text(encoding="utf-8"))
    if not isinstance(schedule, dict) or not isinstance(schedule.get("series"), list):
        raise ValueError("Schedule must be an object containing a series list.")
    if solve_summary.get("status") != "optimal" or not solve_summary.get(
        "has_solution"
    ):
        raise ValueError("Only an optimal schedule with a solution may be packaged.")

    portable = _portable_paths(deepcopy(schedule), dataset_name=dataset_name)
    metadata = portable.get("metadata") or {}
    boundary_exact = bool(metadata.get("boundary_service_exact"))
    if not boundary_exact and not allow_inexact_boundary:
        raise ValueError(
            "Boundary service is not exact; pass --allow-inexact-boundary only for "
            "an explicitly diagnostic demonstration."
        )
    absolute_values = _absolute_strings(portable)
    if absolute_values:
        raise ValueError(
            "Portable schedule still contains absolute paths: "
            + "; ".join(absolute_values[:5])
        )

    building_ids = sorted(
        {str(item["building_id"]) for item in portable["series"]},
        key=lambda item: int(item.rsplit("_", 1)[-1]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_schedule = output_dir / "replay_schedule.json"
    output_schedule.write_text(
        json.dumps(portable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence = _audit_evidence(audit_summary_path, run_name=audit_run)
    manifest = {
        "schema_version": 1,
        "demonstration_name": demonstration_name,
        "dataset_name": dataset_name,
        "portable": True,
        "diagnostic_only": not boundary_exact,
        "boundary_service_exact": boundary_exact,
        "settlement": metadata.get("settlement"),
        "source_window": {
            "start_time_step": metadata.get("source_start_time_step"),
            "end_time_step_exclusive": metadata.get(
                "source_end_time_step_exclusive"
            ),
            "horizon": portable.get("horizon"),
        },
        "building_ids": building_ids,
        "source_commit": source_commit,
        "source_problem_id": metadata.get("source_problem_id"),
        "source_replay_job_id": source_replay_job_id,
        "claims": {
            "optimal_for_supplied_linear_model": True,
            "citylearn_replay_validated": True,
            "citylearn_global_optimum": False,
            "community_optimum": False,
        },
        "audit": evidence,
        "artifacts": {
            "replay_schedule": {
                "path": "replay_schedule.json",
                "sha256": _sha256(output_schedule),
            },
            "source_schedule_sha256": _sha256(schedule_path),
            "source_solve_summary_sha256": _sha256(solve_summary_path),
            "source_audit_summary_sha256": _sha256(audit_summary_path),
        },
    }
    output_manifest = output_dir / "manifest.json"
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_schedule, output_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--solve-summary", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--audit-run", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--demonstration-name", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-replay-job-id", required=True)
    parser.add_argument("--allow-inexact-boundary", action="store_true")
    args = parser.parse_args()
    schedule, manifest = package_demonstration(
        schedule_path=args.schedule,
        solve_summary_path=args.solve_summary,
        audit_summary_path=args.audit_summary,
        audit_run=args.audit_run,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        demonstration_name=args.demonstration_name,
        source_commit=args.source_commit,
        source_replay_job_id=args.source_replay_job_id,
        allow_inexact_boundary=args.allow_inexact_boundary,
    )
    print(json.dumps({"schedule": str(schedule), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
