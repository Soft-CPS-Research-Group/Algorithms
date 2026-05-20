from scripts.build_phase6_remote_scorecard import (
    build_scorecard,
    infer_dataset,
    infer_policy,
    infer_track,
    infer_variant,
)


def test_phase6_scorecard_infers_remote_config_metadata():
    name = "remote_20260520_full_2022_no_v2g_maddpg_v48_seed123.yaml"

    assert infer_dataset(name) == "2022"
    assert infer_variant(name) == "no_v2g"
    assert infer_policy(name) == "MADDPG_v48"
    assert infer_track(name) == "full"


def test_phase6_scorecard_marks_strong_candidate_against_rbcsmart():
    rows = [
        {
            "job_id": "rbc",
            "status": "finished",
            "config_path": "configs/remote_20260520_baseline_15s_rbc_smart.yaml",
            "community_cost_eur": "100.0",
            "ev_min_acceptable_feasible_rate": "1.0",
            "ev_within_tolerance_feasible_rate": "1.0",
            "electrical_violation_kwh": "0.0",
        },
        {
            "job_id": "maddpg",
            "status": "finished",
            "config_path": "configs/remote_20260520_full_15s_maddpg_v48_seed123.yaml",
            "community_cost_eur": "95.0",
            "ev_min_acceptable_feasible_rate": "1.0",
            "ev_within_tolerance_feasible_rate": "0.9",
            "electrical_violation_kwh": "0.0",
        },
    ]

    scorecard = build_scorecard(rows)
    maddpg = next(row for row in scorecard if row["job_id"] == "maddpg")

    assert maddpg["decision"] == "candidate_strong"
    assert maddpg["cost_delta_to_rbcsmart_eur"] == -5.0
    assert maddpg["cost_delta_to_rbcsmart_pct"] == -5.0


def test_phase6_scorecard_rejects_maddpg_when_ev_gate_fails():
    rows = [
        {
            "job_id": "rbc",
            "status": "finished",
            "config_path": "configs/remote_20260520_baseline_2022_no_v2g_rbc_smart.yaml",
            "community_cost_eur": "100.0",
            "ev_min_acceptable_feasible_rate": "1.0",
            "ev_within_tolerance_feasible_rate": "1.0",
        },
        {
            "job_id": "maddpg",
            "status": "finished",
            "config_path": "configs/remote_20260520_full_2022_no_v2g_maddpg_v48_seed123.yaml",
            "community_cost_eur": "90.0",
            "ev_min_acceptable_feasible_rate": "0.8",
            "ev_within_tolerance_feasible_rate": "0.8",
            "electrical_violation_kwh": "0.0",
        },
    ]

    scorecard = build_scorecard(rows)
    maddpg = next(row for row in scorecard if row["job_id"] == "maddpg")

    assert maddpg["decision"] == "reject_ev_service"
