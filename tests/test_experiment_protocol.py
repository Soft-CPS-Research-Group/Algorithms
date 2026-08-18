from __future__ import annotations

import csv
from pathlib import Path

import pytest

from utils.experiment_protocol import (
    build_evaluation_record,
    build_pairing_fingerprint,
    canonical_sha256,
    extract_scorecard,
    file_sha256,
    select_checkpoint,
    verify_selected_checkpoint,
)


def _config(tmp_path: Path, *, simulator_seed: int = 17, neural_seed: int = 23):
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    return {
        "simulator": {
            "dataset_name": "annual",
            "dataset_path": str(schema),
            "central_agent": False,
            "interface": "entity",
            "topology_mode": "static",
            "simulation_start_time_step": 100,
            "simulation_end_time_step": 195,
            "episode_time_steps": 96,
            "random_seed": simulator_seed,
            "reward_function": "CostHardConstraintReward",
            "reward_function_kwargs": {"community_settlement_cost_weight": 1.0},
            "export": {"include_business_as_usual": True},
        },
        "training": {"seed": neural_seed},
        "experiment_protocol": {
            "version": "ti_marl_experiment_protocol_v1",
            "protocol_id": "ti-marl-dev-v1",
            "phase": "development",
            "role": "candidate",
            "data_split": "development",
            "window_id": "winter",
            "candidate_id": "checkpoint-1",
            "selection_rules_sha256": "a" * 64,
        },
    }


def _write_kpis(path: Path, **overrides: float) -> None:
    values = {
        "district_cost_community_market_settled_total_eur": 100.0,
        "district_energy_grid_shape_quality_peak_daily_average_to_business_as_usual_ratio": 1.0,
        "district_energy_grid_shape_quality_peak_all_time_average_to_business_as_usual_ratio": 1.0,
        "district_energy_grid_shape_quality_ramping_average_to_business_as_usual_ratio": 1.0,
        "district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_business_as_usual_ratio": 1.0,
        "district_solar_self_consumption_ratio_self_consumption_ratio": 0.7,
        "district_emissions_ratio_to_business_as_usual_total_ratio": 0.9,
        "district_electrical_service_phase_violations_energy_total_kwh": 0.0,
        "district_ev_performance_departure_min_acceptable_feasible_ratio": 1.0,
        "district_ev_performance_departure_within_tolerance_feasible_ratio": 0.99,
        "district_battery_total_throughput_kwh": 12.0,
        "district_ev_total_v2g_export_kwh": 3.0,
        "district_energy_grid_total_import_control_kwh": 50.0,
        "district_energy_grid_total_export_control_kwh": 8.0,
        "district_equity_distribution_gini_benefit_ratio": 0.2,
    }
    values.update(overrides)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["KPI", "District"])
        writer.writeheader()
        for name, value in values.items():
            writer.writerow({"KPI": name, "District": value})


def _record(
    candidate_id: str,
    pairing: str,
    checkpoint_sha: str | None,
    *,
    role: str,
    cost: float,
    peak: float = 1.0,
    ramp: float = 1.0,
    solar: float = 0.7,
    violations: float = 0.0,
    ev_min: float = 1.0,
):
    payload = {
        "format": "ti_marl_evaluation_record_v1",
        "protocol_id": "dev-v1",
        "phase": "development",
        "selection_rules_sha256": canonical_sha256(_rules()),
        "candidate_id": candidate_id,
        "role": role,
        "pairing": {"sha256": pairing},
        "checkpoint": (
            None
            if checkpoint_sha is None
            else {"path": f"/{candidate_id}.pth", "sha256": checkpoint_sha}
        ),
        "metrics": {
            "cost_eur": cost,
            "peak_daily_ratio_to_bau": peak,
            "ramping_ratio_to_bau": ramp,
            "solar_self_consumption_rate": solar,
            "electrical_violation_kwh": violations,
            "ev_min_acceptable_feasible_rate": ev_min,
        },
    }
    payload["record_sha256"] = canonical_sha256(payload)
    return payload


def _rules():
    return {
        "version": "ti_marl_selection_rules_v1",
        "aggregation": {
            "cost_eur": "sum",
            "electrical_violation_kwh": "sum",
            "ev_min_acceptable_feasible_rate": "min",
        },
        "hard_gates": {
            "electrical_violation_kwh": {"max": 0.5},
            "ev_min_acceptable_feasible_rate": {"min": 0.99},
        },
        "reference_guardrails": {
            "peak_daily_ratio_to_bau": {"max_relative_increase": 0.05},
            "ramping_ratio_to_bau": {"max_relative_increase": 0.05},
            "solar_self_consumption_rate": {"max_absolute_decrease": 0.02},
        },
        "promotion": {
            "metric": "cost_eur",
            "direction": "minimize",
            "minimum_improvement": 0.0,
        },
        "tie_breakers": [
            {"metric": "peak_daily_ratio_to_bau", "direction": "minimize"},
        ],
    }


def test_pairing_fingerprint_separates_simulator_and_neural_seeds(tmp_path):
    first = build_pairing_fingerprint(_config(tmp_path, neural_seed=1))
    same_surface = build_pairing_fingerprint(_config(tmp_path, neural_seed=999))
    changed_surface = build_pairing_fingerprint(
        _config(tmp_path, simulator_seed=18, neural_seed=1)
    )

    assert first["sha256"] == same_surface["sha256"]
    assert first["sha256"] != changed_surface["sha256"]


def test_scorecard_and_evaluation_record_are_content_addressed(tmp_path):
    kpis = tmp_path / "exported_kpis.csv"
    checkpoint = tmp_path / "checkpoint.pth"
    _write_kpis(kpis)
    checkpoint.write_bytes(b"weights")

    scorecard = extract_scorecard(kpis)
    record = build_evaluation_record(
        candidate_id="checkpoint-1",
        role="candidate",
        config=_config(tmp_path),
        exported_kpis_path=kpis,
        checkpoint_path=checkpoint,
        simulator_version="1.7.0",
    )

    assert scorecard["cost_eur"] == pytest.approx(100.0)
    assert scorecard["ev_min_acceptable_feasible_rate"] == pytest.approx(1.0)
    assert record["checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert len(record["record_sha256"]) == 64


def test_selection_requires_every_paired_surface_and_rejects_bad_scorecard():
    references = [
        _record("smart", "winter", None, role="reference", cost=100.0),
        _record("smart", "summer", None, role="reference", cost=120.0),
    ]
    candidates = [
        _record("good", "winter", "a" * 64, role="candidate", cost=95.0),
        _record("good", "summer", "a" * 64, role="candidate", cost=110.0),
        _record("unsafe", "winter", "b" * 64, role="candidate", cost=80.0, violations=0.4),
        _record("unsafe", "summer", "b" * 64, role="candidate", cost=80.0, violations=0.4),
        _record("bad-peak", "winter", "c" * 64, role="candidate", cost=70.0, peak=1.2),
        _record("bad-peak", "summer", "c" * 64, role="candidate", cost=70.0, peak=1.2),
    ]

    selection = select_checkpoint(
        references=references,
        candidates=candidates,
        rules=_rules(),
    )

    assert selection["status"] == "selected"
    assert selection["selected_candidate_id"] == "good"
    rejected = {
        row["candidate_id"]: row["rejection_reasons"]
        for row in selection["evaluated_candidates"]
    }
    assert any("electrical_violation_kwh" in reason for reason in rejected["unsafe"])
    assert any("peak_daily_ratio_to_bau" in reason for reason in rejected["bad-peak"])


def test_selection_refuses_confirmation_records():
    reference = _record("smart", "winter", None, role="reference", cost=100.0)
    candidate = _record("candidate", "winter", "a" * 64, role="candidate", cost=90.0)
    candidate["phase"] = "confirmation"

    with pytest.raises(ValueError, match="development records only"):
        select_checkpoint(references=[reference], candidates=[candidate], rules=_rules())


def test_selection_rejects_a_tampered_evaluation_record():
    reference = _record("smart", "winter", None, role="reference", cost=100.0)
    candidate = _record("candidate", "winter", "a" * 64, role="candidate", cost=90.0)
    candidate["metrics"]["cost_eur"] = 1.0

    with pytest.raises(ValueError, match="payload/hash mismatch"):
        select_checkpoint(references=[reference], candidates=[candidate], rules=_rules())


def test_selected_checkpoint_verification(tmp_path):
    checkpoint = tmp_path / "selected.pth"
    checkpoint.write_bytes(b"selected")
    selection = {"selected_checkpoint": {"sha256": file_sha256(checkpoint)}}
    assert verify_selected_checkpoint(selection, checkpoint)
    checkpoint.write_bytes(b"changed")
    assert not verify_selected_checkpoint(selection, checkpoint)
