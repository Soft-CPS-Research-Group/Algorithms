from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.bechmark_agents import compare_export_roots


def _write_job(root: Path, job_id: str, *, algorithm: str, seed: int, kpis: dict) -> None:
    job_dir = root / "jobs" / job_id
    simulation_data_dir = job_dir / "results" / "simulation_data" / "session-1"
    simulation_data_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = ["KPI,District"]
    for kpi_name, value in kpis.items():
        csv_rows.append(f"{kpi_name},{value}")
    (simulation_data_dir / "exported_kpis.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    (job_dir / "results" / "result.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    config_payload = {
        "algorithm": {"name": algorithm},
        "training": {"seed": seed},
    }
    (job_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config_payload), encoding="utf-8")


def test_compare_export_roots_reports_pass_when_all_gates_hold(tmp_path):
    maddpg_root = tmp_path / "maddpg_exports"
    rbc_root = tmp_path / "rbc_exports"

    _write_job(
        maddpg_root,
        "maddpg-1",
        algorithm="MADDPG",
        seed=11,
        kpis={
            "community_settled_cost_total_eur": 98.0,
            "electrical_service_violation_total_kwh": 4.8,
            "ev_departure_min_acceptable_feasible_rate": 0.96,
        },
    )
    _write_job(
        maddpg_root,
        "maddpg-2",
        algorithm="MADDPG",
        seed=22,
        kpis={
            "community_settled_cost_total_eur": 100.0,
            "electrical_service_violation_total_kwh": 5.2,
            "ev_departure_min_acceptable_feasible_rate": 0.97,
        },
    )

    _write_job(
        rbc_root,
        "rbc-1",
        algorithm="RuleBasedPolicy",
        seed=11,
        kpis={
            "community_settled_cost_total_eur": 100.0,
            "electrical_service_violation_total_kwh": 5.0,
            "ev_departure_min_acceptable_feasible_rate": 0.93,
        },
    )
    _write_job(
        rbc_root,
        "rbc-2",
        algorithm="RuleBasedPolicy",
        seed=22,
        kpis={
            "community_settled_cost_total_eur": 101.0,
            "electrical_service_violation_total_kwh": 5.3,
            "ev_departure_min_acceptable_feasible_rate": 0.94,
        },
    )

    report = compare_export_roots(maddpg_root=maddpg_root, rbc_root=rbc_root)

    assert report["checks"]["cost_parity_pass"] is True
    assert report["checks"]["grid_gate_pass"] is True
    assert report["checks"]["ev_gate_pass"] is True
    assert report["overall_pass"] is True
    assert report["aggregates"]["MADDPG"]["runs"][0]["kpi_source"] == "exported_kpis_csv"


def test_compare_export_roots_reports_fail_when_cost_or_ev_breaks_threshold(tmp_path):
    maddpg_root = tmp_path / "maddpg_exports"
    rbc_root = tmp_path / "rbc_exports"

    _write_job(
        maddpg_root,
        "maddpg-1",
        algorithm="MADDPG",
        seed=33,
        kpis={
            "community_settled_cost_total_eur": 120.0,
            "electrical_service_violation_total_kwh": 5.0,
            "ev_departure_min_acceptable_feasible_rate": 0.90,
        },
    )

    _write_job(
        rbc_root,
        "rbc-1",
        algorithm="RuleBasedPolicy",
        seed=33,
        kpis={
            "community_settled_cost_total_eur": 100.0,
            "electrical_service_violation_total_kwh": 5.0,
            "ev_departure_min_acceptable_feasible_rate": 0.95,
        },
    )

    report = compare_export_roots(maddpg_root=maddpg_root, rbc_root=rbc_root)

    assert report["checks"]["cost_parity_pass"] is False
    assert report["checks"]["ev_gate_pass"] is False
    assert report["overall_pass"] is False


def test_compare_export_roots_falls_back_to_result_json_evaluation_payload(tmp_path):
    maddpg_root = tmp_path / "maddpg_exports"
    rbc_root = tmp_path / "rbc_exports"

    maddpg_job = maddpg_root / "jobs" / "maddpg-legacy"
    rbc_job = rbc_root / "jobs" / "rbc-legacy"
    (maddpg_job / "results").mkdir(parents=True, exist_ok=True)
    (rbc_job / "results").mkdir(parents=True, exist_ok=True)

    maddpg_result = {
        "status": "completed",
        "evaluation": {
            "kpis": {
                "community_settled_cost_total_eur": 100.0,
                "electrical_service_violation_total_kwh": 5.0,
                "ev_departure_min_acceptable_feasible_rate": 0.96,
            }
        },
    }
    rbc_result = {
        "status": "completed",
        "evaluation": {
            "kpis": {
                "community_settled_cost_total_eur": 100.0,
                "electrical_service_violation_total_kwh": 5.0,
                "ev_departure_min_acceptable_feasible_rate": 0.90,
            }
        },
    }
    (maddpg_job / "results" / "result.json").write_text(json.dumps(maddpg_result), encoding="utf-8")
    (rbc_job / "results" / "result.json").write_text(json.dumps(rbc_result), encoding="utf-8")

    maddpg_config = {"algorithm": {"name": "MADDPG"}, "training": {"seed": 1}}
    rbc_config = {"algorithm": {"name": "RuleBasedPolicy"}, "training": {"seed": 1}}
    (maddpg_job / "config.resolved.yaml").write_text(yaml.safe_dump(maddpg_config), encoding="utf-8")
    (rbc_job / "config.resolved.yaml").write_text(yaml.safe_dump(rbc_config), encoding="utf-8")

    report = compare_export_roots(maddpg_root=maddpg_root, rbc_root=rbc_root)

    assert report["aggregates"]["MADDPG"]["runs"][0]["kpi_source"] == "result_json_evaluation"
    assert report["checks"]["ev_gate_pass"] is True


def test_compare_export_roots_uses_legacy_ev_success_when_feasible_kpis_are_missing(tmp_path):
    maddpg_root = tmp_path / "maddpg_exports"
    rbc_root = tmp_path / "rbc_exports"

    _write_job(
        maddpg_root,
        "maddpg-legacy",
        algorithm="MADDPG",
        seed=1,
        kpis={
            "community_settled_cost_total_eur": 100.0,
            "electrical_service_violation_total_kwh": 0.0,
            "ev_departure_success_rate": 0.96,
        },
    )
    _write_job(
        rbc_root,
        "rbc-legacy",
        algorithm="RuleBasedPolicy",
        seed=1,
        kpis={
            "community_settled_cost_total_eur": 100.0,
            "electrical_service_violation_total_kwh": 0.0,
            "ev_departure_success_rate": 0.90,
        },
    )

    report = compare_export_roots(maddpg_root=maddpg_root, rbc_root=rbc_root)

    assert report["diagnostics"]["ev_gate_maddpg_kpi"] == "ev_departure_success_rate"
    assert report["checks"]["ev_gate_pass"] is True


def test_compare_export_roots_maps_citylearn_exported_kpi_names(tmp_path):
    maddpg_root = tmp_path / "maddpg_exports"
    rbc_root = tmp_path / "rbc_exports"

    _write_job(
        maddpg_root,
        "maddpg-citylearn",
        algorithm="MADDPG",
        seed=1,
        kpis={
            "district_cost_total_control_eur": -96.0,
            "district_electrical_service_phase_violations_energy_total_kwh": 0.0,
            "district_ev_performance_departure_min_acceptable_feasible_ratio": 1.0,
            "district_ev_performance_departure_min_acceptable_ratio": 0.8,
            "district_ev_performance_departure_success_feasible_ratio": 0.4,
            "district_ev_performance_departure_success_ratio": 0.2,
            "district_ev_performance_departure_within_tolerance_feasible_ratio": 0.9,
            "district_ev_performance_departure_within_tolerance_ratio": 0.6,
            "district_ev_events_departure_count": 10,
            "district_ev_events_departure_target_infeasible_count": 2,
            "district_ev_events_departure_min_acceptable_infeasible_count": 1,
            "district_ev_events_departure_within_tolerance_infeasible_count": 3,
            "district_ev_performance_departure_soc_deficit_mean_ratio": 0.1,
            "district_ev_performance_departure_shortfall_beyond_tolerance_mean_ratio": 0.03,
        },
    )
    _write_job(
        rbc_root,
        "rbc-citylearn",
        algorithm="RuleBasedPolicy",
        seed=1,
        kpis={
            "district_cost_total_control_eur": -100.0,
            "district_electrical_service_phase_violations_energy_total_kwh": 0.0,
            "district_ev_performance_departure_min_acceptable_feasible_ratio": 0.9,
            "district_ev_performance_departure_min_acceptable_ratio": 0.7,
            "district_ev_performance_departure_success_feasible_ratio": 0.3,
            "district_ev_performance_departure_success_ratio": 0.1,
            "district_ev_performance_departure_within_tolerance_feasible_ratio": 0.8,
            "district_ev_performance_departure_within_tolerance_ratio": 0.5,
            "district_ev_events_departure_count": 10,
            "district_ev_events_departure_target_infeasible_count": 2,
            "district_ev_events_departure_min_acceptable_infeasible_count": 1,
            "district_ev_events_departure_within_tolerance_infeasible_count": 3,
            "district_ev_performance_departure_soc_deficit_mean_ratio": 0.2,
            "district_ev_performance_departure_shortfall_beyond_tolerance_mean_ratio": 0.07,
        },
    )

    report = compare_export_roots(maddpg_root=maddpg_root, rbc_root=rbc_root)

    assert report["aggregates"]["MADDPG"]["kpi_means"]["community_settled_cost_total_eur"] == -96.0
    assert report["aggregates"]["MADDPG"]["kpi_means"]["ev_departure_min_acceptable_feasible_rate"] == 1.0
    assert report["aggregates"]["MADDPG"]["kpi_means"]["ev_departure_min_acceptable_rate"] == 0.8
    assert report["aggregates"]["MADDPG"]["kpi_means"]["ev_departure_success_feasible_rate"] == 0.4
    assert report["aggregates"]["MADDPG"]["kpi_means"]["ev_departure_success_rate"] == 0.2
    assert report["aggregates"]["MADDPG"]["kpi_means"]["ev_departure_within_tolerance_feasible_rate"] == 0.9
    assert report["aggregates"]["MADDPG"]["kpi_means"]["ev_departure_within_tolerance_rate"] == 0.6
    assert report["aggregates"]["MADDPG"]["kpi_means"]["ev_departure_count"] == 10
    assert report["aggregates"]["MADDPG"]["kpi_means"]["ev_departure_min_acceptable_infeasible_count"] == 1
    assert report["diagnostics"]["ev_gate_maddpg_kpi"] == "ev_departure_min_acceptable_feasible_rate"
    assert report["checks"]["cost_parity_pass"] is True
    assert report["checks"]["grid_gate_pass"] is True
    assert report["checks"]["ev_gate_pass"] is True
