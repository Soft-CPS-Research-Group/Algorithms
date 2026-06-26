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


def test_phase6_scorecard_infers_learning_comparator_metadata():
    assert infer_policy("remote_20260520_short_15s_original_matd3_seed123.yaml") == "MATD3"
    assert infer_policy("remote_20260520_short_2022_no_v2g_masac_seed123.yaml") == "MASAC"
    assert infer_policy("remote_20260520_short_15s_multi_charger_ippo_seed123.yaml") == "IPPO"
    assert infer_policy("remote_20260520_short_2022_original_mappo_seed123.yaml") == "MAPPO"
    assert infer_policy("remote_20260520_short_15s_original_happo_seed123.yaml") == "HAPPO"
    assert infer_track("remote_20260520_short_15s_original_happo_seed123.yaml") == "short"


def test_phase6_scorecard_infers_metadata_from_simulation_session_when_config_missing():
    scorecard = build_scorecard(
        [
            {
                "job_id": "random",
                "status": "finished",
                "simulation_data_session": "remote_20260520_baseline_15s_random",
                "community_cost_eur": "60.0",
                "ev_min_acceptable_feasible_rate": "0.0",
                "ev_within_tolerance_feasible_rate": "0.0",
            }
        ]
    )

    row = scorecard[0]

    assert row["dataset"] == "15s"
    assert row["track"] == "baseline"
    assert row["policy"] == "Random"
    assert row["decision"] == "reference"


def test_phase6_scorecard_marks_learning_candidate_against_rbcsmart():
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
            "job_id": "matd3",
            "status": "finished",
            "config_path": "configs/remote_20260520_short_15s_matd3_seed123.yaml",
            "community_cost_eur": "95.0",
            "ev_min_acceptable_feasible_rate": "1.0",
            "ev_within_tolerance_feasible_rate": "0.9",
            "electrical_violation_kwh": "0.0",
        },
    ]

    scorecard = build_scorecard(rows)
    candidate = next(row for row in scorecard if row["job_id"] == "matd3")

    assert candidate["decision"] == "candidate_strong"
    assert candidate["decision_bucket"] == "promote"
    assert candidate["next_action"] == "promote_to_multiseed_full"
    assert candidate["cost_delta_to_rbcsmart_eur"] == -5.0
    assert candidate["cost_delta_to_rbcsmart_pct"] == -5.0


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
    assert maddpg["decision_bucket"] == "reject"
    assert "ev_service_below_gate" in maddpg["risk_flags"]
