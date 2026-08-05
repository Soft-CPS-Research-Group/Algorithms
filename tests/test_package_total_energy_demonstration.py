import csv
import json
from pathlib import Path

import pytest

from scripts.package_total_energy_demonstration import package_demonstration


def _write_inputs(tmp_path: Path, *, boundary_exact: bool) -> tuple[Path, Path, Path]:
    schedule = tmp_path / "source_schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "horizon": 2,
                "metadata": {
                    "boundary_service_exact": boundary_exact,
                    "dataset_root": "/private/workspace/dataset",
                    "schema_path": "/private/workspace/dataset/schema.json",
                    "settlement": "individual",
                    "source_problem_id": "problem-1",
                    "source_start_time_step": 0,
                    "source_end_time_step_exclusive": 2,
                    "boundary_diagnostics": {
                        "dataset_root": "/private/workspace/dataset",
                        "schema_path": "/private/workspace/dataset/schema.json",
                    },
                },
                "problem_id": "problem-1",
                "series": [
                    {
                        "building_id": "Building_1",
                        "action_name": "electrical_storage",
                        "values": [0.0, 0.1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    solve = tmp_path / "solve_summary.json"
    solve.write_text(
        json.dumps({"status": "optimal", "has_solution": True}), encoding="utf-8"
    )
    audit = tmp_path / "audit_summary.csv"
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "building_count",
                "local_gate_pass_count",
                "local_gate_reject_count",
                "local_cost_eur_sum",
                "buildings_beating_baseline_count",
                "all_buildings_pass_local_gates",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run": "milp",
                "building_count": 1,
                "local_gate_pass_count": 1,
                "local_gate_reject_count": 0,
                "local_cost_eur_sum": 1.25,
                "buildings_beating_baseline_count": 1,
                "all_buildings_pass_local_gates": 1,
            }
        )
    return schedule, solve, audit


def test_packages_portable_replay_validated_schedule(tmp_path: Path) -> None:
    schedule, solve, audit = _write_inputs(tmp_path, boundary_exact=True)
    output_schedule, output_manifest = package_demonstration(
        schedule_path=schedule,
        solve_summary_path=solve,
        audit_summary_path=audit,
        audit_run="milp",
        output_dir=tmp_path / "package",
        dataset_name="fixture_dataset",
        demonstration_name="fixture",
        source_commit="abc123",
        source_replay_job_id="replay-1",
    )

    packaged = json.loads(output_schedule.read_text(encoding="utf-8"))
    assert packaged["metadata"]["schema_path"] == (
        "datasets/fixture_dataset/schema.json"
    )
    assert packaged["metadata"]["boundary_diagnostics"]["dataset_root"] == (
        "datasets/fixture_dataset"
    )
    assert "/private/workspace" not in output_schedule.read_text(encoding="utf-8")
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest["portable"] is True
    assert manifest["diagnostic_only"] is False
    assert manifest["audit"]["all_buildings_pass_local_gates"] is True
    assert len(manifest["artifacts"]["replay_schedule"]["sha256"]) == 64


def test_inexact_boundary_requires_explicit_diagnostic_flag(tmp_path: Path) -> None:
    schedule, solve, audit = _write_inputs(tmp_path, boundary_exact=False)
    kwargs = {
        "schedule_path": schedule,
        "solve_summary_path": solve,
        "audit_summary_path": audit,
        "audit_run": "milp",
        "output_dir": tmp_path / "package",
        "dataset_name": "fixture_dataset",
        "demonstration_name": "fixture",
        "source_commit": "abc123",
        "source_replay_job_id": "replay-1",
    }
    with pytest.raises(ValueError, match="Boundary service is not exact"):
        package_demonstration(**kwargs)

    _, manifest_path = package_demonstration(
        **kwargs, allow_inexact_boundary=True
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["diagnostic_only"] is True
