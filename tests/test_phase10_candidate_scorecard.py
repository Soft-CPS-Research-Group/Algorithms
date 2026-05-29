import csv

from scripts.build_phase10_candidate_scorecard import build_scorecard


def _write_summary(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_candidate_scorecard_marks_pass_and_fail_reasons(tmp_path):
    summary = tmp_path / "summary.csv"
    _write_summary(
        summary,
        [
            {
                "job_id": "pass-1",
                "simulation_data_session": "phase10-w6-ev-only-bc-primary-maddpg-seed123-8760steps",
                "status": "finished",
                "exit_code": "0",
                "community_cost_eur": "17000.0",
                "ev_min_acceptable_feasible_rate": "0.995",
                "ev_within_tolerance_rate": "0.45",
                "electrical_violation_kwh": "0.0",
                "battery_throughput_kwh": "40000.0",
                "v2g_export_kwh": "2.0",
                "community_import_kwh": "150000.0",
                "community_export_kwh": "50000.0",
                "community_solar_self_consumption_rate": "0.49",
            },
            {
                "job_id": "fail-1",
                "simulation_data_session": "phase10-w5a-v52-newenc-bc-2022-maddpg-seed123-4096steps",
                "status": "finished",
                "exit_code": "0",
                "community_cost_eur": "8792.2",
                "ev_min_acceptable_feasible_rate": "0.989",
                "ev_within_tolerance_rate": "0.25",
                "electrical_violation_kwh": "0.0",
                "battery_throughput_kwh": "34593.0",
                "v2g_export_kwh": "1507.0",
            },
        ],
    )

    rows = build_scorecard([summary])
    by_id = {row["job_id"]: row for row in rows}

    assert by_id["pass-1"]["verdict"] == "PASS"
    assert by_id["pass-1"]["algorithm"] == "MADDPG"
    assert by_id["pass-1"]["recipe"] == "w6_ev_only_bc_primary"
    assert by_id["pass-1"]["seed"] == 123
    assert by_id["pass-1"]["steps"] == 8760
    assert by_id["pass-1"]["cost_delta_vs_rbc_smart"] < 0
    assert by_id["pass-1"]["battery_preferred_gate"] is True

    assert by_id["fail-1"]["verdict"] == "FAIL_EV_MIN"
    assert "ev_min" in by_id["fail-1"]["verdict_reason"]
    assert by_id["fail-1"]["cost_ratio_to_bau_status"] == "unavailable_bau_disabled"
