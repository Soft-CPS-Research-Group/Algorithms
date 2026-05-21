from pathlib import Path

from scripts.build_phase6_building_scorecard import build_building_scorecard
from scripts.run_phase6_remote_analysis import main as run_remote_analysis


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_phase6_building_scorecard_extracts_building_kpis_and_flags(tmp_path):
    results_dir = tmp_path / "remote_results"
    _write(
        results_dir / "summary.csv",
        "\n".join(
            [
                "job_id,status,simulation_data_session,target_host,image_tag",
                "job-1,finished,remote_20260520_baseline_15s_rbc_smart,deucalion,sha-test",
            ]
        ),
    )
    _write(
        results_dir / "jobs" / "job-1" / "exported_kpis.csv",
        "\n".join(
            [
                "KPI,Building_1,Building_15,District",
                "building_cost_total_control_eur,10.0,20.0,30.0",
                "building_cost_total_business_as_usual_eur,12.0,15.0,27.0",
                "building_cost_total_delta_to_business_as_usual_eur,-2.0,5.0,3.0",
                "building_cost_ratio_to_business_as_usual_total_ratio,0.8,1.2,1.1",
                "building_ev_events_departure_count,1.0,2.0,3.0",
                "building_ev_events_departure_min_acceptable_feasible_count,1.0,1.0,2.0",
                "building_ev_events_departure_min_acceptable_infeasible_count,0.0,1.0,1.0",
                "building_ev_performance_departure_min_acceptable_feasible_ratio,1.0,0.5,0.75",
                "building_ev_performance_departure_within_tolerance_feasible_ratio,1.0,0.25,0.5",
                "building_electrical_service_phase_violations_energy_total_kwh,0.0,0.2,0.2",
                "building_electrical_service_phase_violations_event_count,0.0,1.0,1.0",
                "building_battery_ratio_to_business_as_usual_throughput_ratio,1.0,4.5,3.0",
                "building_solar_self_consumption_total_generation_kwh,5.0,5.0,10.0",
                "building_solar_self_consumption_ratio_self_consumption_ratio,0.9,0.2,0.55",
                "building_ev_total_v2g_export_kwh,0.0,2.0,2.0",
                "building_deferrable_appliance_service_missed_cycles_count,0.0,1.0,1.0",
                "building_deferrable_appliance_service_unserved_energy_total_kwh,0.0,0.5,0.5",
            ]
        ),
    )

    rows = build_building_scorecard(results_dir)
    building_1 = next(row for row in rows if row["building"] == "Building_1")
    building_15 = next(row for row in rows if row["building"] == "Building_15")

    assert building_1["dataset"] == "15s"
    assert building_1["policy"] == "RBCSmart"
    assert building_1["cost_eur"] == 10.0
    assert "building_15" not in building_1["flags"]
    assert "grid_violation" not in building_1["flags"]

    assert building_15["building_index"] == "15"
    assert "building_15" in building_15["flags"]
    assert "ev_service_below_gate" in building_15["flags"]
    assert "grid_violation" in building_15["flags"]
    assert "battery_throughput_high" in building_15["flags"]
    assert "solar_self_consumption_low" in building_15["flags"]
    assert "v2g_used" in building_15["flags"]
    assert "deferrable_service_gap" in building_15["flags"]


def test_phase6_remote_analysis_skip_collect_builds_both_scorecards(tmp_path):
    results_dir = tmp_path / "remote_results"
    _write(
        results_dir / "summary.csv",
        "\n".join(
            [
                "job_id,status,simulation_data_session,community_cost_eur,ev_min_acceptable_feasible_rate,ev_within_tolerance_feasible_rate",
                "job-1,finished,remote_20260520_baseline_15s_random,10.0,0.0,0.0",
            ]
        ),
    )
    _write(
        results_dir / "jobs" / "job-1" / "exported_kpis.csv",
        "\n".join(
            [
                "KPI,Building_1,District",
                "building_cost_total_control_eur,10.0,10.0",
                "building_ev_events_departure_count,1.0,1.0",
                "building_ev_events_departure_min_acceptable_infeasible_count,1.0,1.0",
            ]
        ),
    )

    assert run_remote_analysis(["--skip-collect", "--output-dir", str(results_dir)]) == 0

    assert (results_dir / "scorecard.csv").exists()
    assert (results_dir / "scorecard.md").exists()
    assert (results_dir / "building_scorecard.csv").exists()
    assert (results_dir / "building_scorecard.md").exists()
