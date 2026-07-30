import csv
import json
from pathlib import Path

import pandas as pd

from scripts.audit_building_local_behavior import (
    audit_run,
    compare_to_baseline,
    compare_to_oracle,
    summarize,
)
from scripts.electrical_safety_evidence import executed_safety_evidence


def _write_export(
    directory: Path,
    *,
    b1_cost: float,
    b2_cost: float,
    b2_ev: float,
    b2_violation_kwh: float = 0.0,
    b2_violation_events: float = 0.0,
) -> None:
    directory.mkdir(parents=True)
    metrics = {
        "building_cost_total_control_eur": (b1_cost, b2_cost),
        "building_cost_community_market_settled_total_eur": (b1_cost, b2_cost),
        "building_cost_total_business_as_usual_eur": (20.0, 40.0),
        "building_ev_events_departure_count": (0.0, 2.0),
        "building_ev_performance_departure_min_acceptable_feasible_ratio": (None, b2_ev),
        "building_ev_performance_departure_within_tolerance_feasible_ratio": (None, 0.5),
        "building_electrical_service_phase_violations_energy_total_kwh": (0.0, b2_violation_kwh),
        "building_electrical_service_phase_violations_event_count": (0.0, b2_violation_events),
        "building_deferrable_appliance_service_completed_cycles_count": (0.0, 1.0),
        "building_deferrable_appliance_service_missed_cycles_count": (0.0, 0.0),
        "building_deferrable_appliance_service_unserved_energy_total_kwh": (0.0, 0.0),
        "building_comfort_resilience_resilience_unserved_energy_outage_normalized_ratio": (0.0, 0.0),
    }
    with (directory / "exported_kpis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["KPI", "Building_1", "Building_2", "District"])
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"KPI": name, "Building_1": values[0], "Building_2": values[1]})
    for building in (1, 2):
        pd.DataFrame({"Battery Soc-%": [0.2, 0.8]}).to_csv(
            directory / f"exported_data_building_{building}_battery_ep1.csv",
            index=False,
        )


def test_building_audit_applies_gates_and_baseline_per_building(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_export(baseline_dir, b1_cost=10.0, b2_cost=30.0, b2_ev=1.0)
    _write_export(candidate_dir, b1_cost=9.0, b2_cost=28.0, b2_ev=0.5)

    rows = audit_run("rbcsmart", baseline_dir) + audit_run("ppo", candidate_dir)
    compare_to_baseline(rows, "rbcsmart")
    by_key = {(row["run"], row["building"]): row for row in rows}

    assert by_key[("ppo", "Building_1")]["local_gate_decision"] == "PASS_LOCAL_GATES"
    assert by_key[("ppo", "Building_1")]["beats_baseline_local_cost"] == 1
    assert by_key[("ppo", "Building_2")]["local_gate_decision"] == "REJECT_LOCAL_GATES"
    assert by_key[("ppo", "Building_2")]["beats_baseline_local_cost"] == 0
    assert by_key[("ppo", "Building_2")]["local_cost_delta_to_baseline_eur"] == -2.0

    summary = {row["run"]: row for row in summarize(rows)}
    assert summary["ppo"]["local_gate_pass_count"] == 1
    assert summary["ppo"]["local_gate_reject_count"] == 1
    assert summary["ppo"]["local_cost_eur_sum"] == 37.0
    assert summary["ppo"]["buildings_beating_baseline_count"] == 1
    assert summary["ppo"]["local_cost_delta_to_baseline_eur_median"] == -1.5
    assert summary["ppo"]["local_cost_delta_to_baseline_eur_worst"] == -1.0
    assert summary["ppo"]["all_buildings_pass_local_gates"] == 0


def test_building_audit_reports_oracle_regret_and_gap_closure(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    oracle_dir = tmp_path / "oracle"
    _write_export(baseline_dir, b1_cost=10.0, b2_cost=30.0, b2_ev=1.0)
    _write_export(candidate_dir, b1_cost=9.0, b2_cost=27.0, b2_ev=1.0)
    _write_export(oracle_dir, b1_cost=8.0, b2_cost=24.0, b2_ev=1.0)

    rows = (
        audit_run("rbcsmart", baseline_dir)
        + audit_run("ppo", candidate_dir)
        + audit_run("milp_replay", oracle_dir)
    )
    compare_to_baseline(rows, "rbcsmart")
    compare_to_oracle(rows, "milp_replay")
    by_key = {(row["run"], row["building"]): row for row in rows}
    assert by_key[("ppo", "Building_1")]["local_cost_regret_to_oracle_reference_eur"] == 1.0
    assert by_key[("ppo", "Building_2")]["oracle_reference_gap_closure_ratio"] == 0.5

    summary = {row["run"]: row for row in summarize(rows)}
    assert summary["ppo"]["oracle_reference_regret_eur_sum"] == 4.0
    assert summary["ppo"]["oracle_reference_gap_closure_ratio"] == 0.5
    assert summary["ppo"]["all_buildings_no_worse_than_baseline"] == 1


def test_projection_tolerant_profile_requires_and_accepts_executed_peak_evidence(tmp_path):
    job_dir = tmp_path / "job"
    data_dir = job_dir / "results" / "simulation_data" / "run"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    schema_path = dataset_dir / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "seconds_per_time_step": 900,
                "buildings": {
                    "Building_1": {"include": True},
                    "Building_2": {
                        "include": True,
                        "electrical_service": {
                            "limits": {
                                "total": {"import_kw": 10.0, "export_kw": 10.0},
                                "per_phase": {
                                    "L1": {"import_kw": 6.0, "export_kw": 6.0}
                                },
                            }
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    job_dir.mkdir(parents=True)
    (job_dir / "config.resolved.yaml").write_text(
        f"simulator:\n  dataset_path: {schema_path}\n",
        encoding="utf-8",
    )
    _write_export(
        data_dir,
        b1_cost=10.0,
        b2_cost=30.0,
        b2_ev=1.0,
        b2_violation_kwh=1.0e-6,
        b2_violation_events=45.0,
    )
    with (data_dir / "exported_kpis.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["KPI", "Building_1", "Building_2", "District"])
        writer.writerow(
            {
                "KPI": "building_electrical_service_phase_phase_peaks_import_peak_l1_kw",
                "Building_2": 6.0,
            }
        )
        writer.writerow(
            {
                "KPI": "building_electrical_service_phase_phase_peaks_export_peak_l1_kw",
                "Building_2": 1.0,
            }
        )
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4000, freq="15min"),
            "Net Electricity Consumption-kWh": [2.5, -1.0] + [0.0] * 3998,
        }
    ).to_csv(data_dir / "exported_data_building_2_ep1.csv", index=False)

    row = {r["building"]: r for r in audit_run("candidate", data_dir)}["Building_2"]

    assert row["local_gate_decision"] == "REJECT_LOCAL_GATES"
    assert row["executed_electrical_safety_certified"] == 1
    assert row["projection_request_numerical_residue"] == 1
    assert row["projection_request_within_tolerance"] == 1
    assert row["projection_tolerant_local_gate_decision"] == "PASS_WITH_SAFETY_PROJECTION"
    assert row["executed_safety_limit_check_count"] == 4

    material_high_frequency = executed_safety_evidence(
        data_dir,
        requested_violation_kwh=0.02,
        requested_violation_events=45.0,
        building_names=["Building_2"],
        time_steps=4000,
    )
    assert material_high_frequency["executed_electrical_safety_certified"] == 1
    assert material_high_frequency["projection_request_numerical_residue"] == 0
    assert material_high_frequency["projection_request_within_tolerance"] == 0


def test_projection_tolerant_profile_rejects_executed_limit_failure(tmp_path):
    data_dir = tmp_path / "run"
    _write_export(
        data_dir,
        b1_cost=10.0,
        b2_cost=30.0,
        b2_ev=1.0,
        b2_violation_kwh=0.02,
        b2_violation_events=3.0,
    )
    row = {r["building"]: r for r in audit_run("candidate", data_dir)}["Building_2"]
    assert row["executed_electrical_safety_certified"] == 0
    assert row["projection_tolerant_local_gate_decision"] == "REJECT_LOCAL_GATES"
