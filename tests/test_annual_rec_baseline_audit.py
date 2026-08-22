from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.run_annual_rec_baseline_audit import (
    _attest_simulator_runtime,
    _build_config,
    _dataset_integrity_sha256,
    _expected_deferrable_service_audit,
    _expected_ev_departure_audit,
    _extract_scorecard,
    _validate_schema_contract,
)
from utils.experiment_protocol import build_pairing_fingerprint
from utils.config_schema import validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_ROOT = Path(
    os.environ.get("SIMULATOR_REPO", REPO_ROOT.parent / "Simulator")
).resolve()
SIMULATOR_VERSION = "1.8.0"


def _template():
    return yaml.safe_load(
        (
            REPO_ROOT
            / "configs/templates/baselines/rbc_smart_15min_local.yaml"
        ).read_text(encoding="utf-8")
    )


def test_bau_and_rbcsmart_configs_have_identical_pairing_surface() -> None:
    schema = SIMULATOR_ROOT / "data/datasets/rec_2023_micro_4_q/schema.json"
    common = {
        "template": _template(),
        "variant": "micro",
        "schema_path": schema,
        "topology_mode": "static",
        "start": 0,
        "end": 671,
    }
    bau = _build_config(policy="bau", job_id="bau", **common)
    smart = _build_config(policy="rbc_smart", job_id="smart", **common)

    assert bau["pipeline"] != smart["pipeline"]
    assert bau["pipeline"][0]["hyperparameters"]["ev_service_soc_tolerance"] == 0.04
    assert smart["pipeline"][0]["hyperparameters"]["ev_service_soc_tolerance"] == 0.04
    assert (
        build_pairing_fingerprint(bau, simulator_version=SIMULATOR_VERSION)["sha256"]
        == build_pairing_fingerprint(smart, simulator_version=SIMULATOR_VERSION)["sha256"]
    )


def test_full_year_config_has_35040_transitions_and_terminal_state() -> None:
    schema = SIMULATOR_ROOT / "data/datasets/rec_2023_micro_4_q/schema.json"
    config = _build_config(
        template=_template(),
        policy="bau",
        variant="micro",
        schema_path=schema,
        topology_mode="static",
        start=0,
        end=35_039,
        job_id="full-year",
    )

    simulator = config["simulator"]
    assert simulator["simulation_start_time_step"] == 0
    assert simulator["simulation_end_time_step"] == 35_040
    assert simulator["episode_time_steps"] == 35_041
    assert simulator["terminal_observation_padding"] is True
    assert validate_config(config).simulator.terminal_observation_padding is True


def test_ev_departure_obligation_audit_matches_static_catalogue() -> None:
    schema = SIMULATOR_ROOT / "data/datasets/rec_2023_micro_4_q/schema.json"

    audit = _expected_ev_departure_audit(schema)

    assert audit == {
        "catalogue_sessions": 554,
        "expected_departure_events": 554,
        "censored_member_inactive": 0,
        "censored_charger_inactive": 0,
        "back_to_back_session_boundaries": 0,
        "back_to_back_same_ev_boundaries": 0,
    }


def test_deferrable_service_audit_matches_static_catalogue() -> None:
    schema = SIMULATOR_ROOT / "data/datasets/rec_2023_micro_4_q/schema.json"

    audit = _expected_deferrable_service_audit(schema)

    assert audit == {
        "catalogue_cycles": 214,
        "fully_observable_cycles": 214,
        "topology_interrupted_cycles": 0,
        "topology_censored_no_start_opportunity": 0,
        "expected_runtime_service_count_min": 214,
        "expected_runtime_service_count_max": 214,
    }


def test_deferrable_service_audit_exposes_dynamic_censoring() -> None:
    core_schema = (
        SIMULATOR_ROOT
        / "data/datasets/rec_2023_core_30/schemas/core_30_dynamic.json"
    )
    premium_schema = (
        SIMULATOR_ROOT
        / "data/datasets/rec_2023_premium_100/schemas/premium_100_clean.json"
    )

    assert _expected_deferrable_service_audit(core_schema) == {
        "catalogue_cycles": 1892,
        "fully_observable_cycles": 1657,
        "topology_interrupted_cycles": 1,
        "topology_censored_no_start_opportunity": 234,
        "expected_runtime_service_count_min": 1657,
        "expected_runtime_service_count_max": 1658,
    }
    assert _expected_deferrable_service_audit(premium_schema) == {
        "catalogue_cycles": 7360,
        "fully_observable_cycles": 5950,
        "topology_interrupted_cycles": 1,
        "topology_censored_no_start_opportunity": 1409,
        "expected_runtime_service_count_min": 5950,
        "expected_runtime_service_count_max": 5951,
    }


def test_extract_scorecard_supports_simulator_parquet_export(tmp_path) -> None:
    path = tmp_path / "exported_kpis.parquet"
    pd.DataFrame(
        {
            "KPI": [
                "district_cost_community_market_settled_total_eur",
                "district_electrical_service_phase_violations_energy_total_kwh",
                "district_electrical_service_phase_requested_pressure_energy_total_kwh",
                "district_ev_performance_departure_min_acceptable_feasible_ratio",
            ],
            "District": [123.5, 0.0, 4.25, 1.0],
        }
    ).to_parquet(path, index=False)

    scorecard = _extract_scorecard(path)

    assert scorecard == {
        "cost_eur": 123.5,
        "electrical_violation_kwh": 0.0,
        "electrical_requested_pressure_kwh": 4.25,
        "ev_min_acceptable_feasible_rate": 1.0,
    }


def test_annual_rec_schema_contract_rejects_oracle_load_pv_forecasts() -> None:
    schema_path = SIMULATOR_ROOT / "data/datasets/rec_2023_micro_4_q/schema.json"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    _validate_schema_contract(schema, schema_path)
    assert len(_dataset_integrity_sha256(schema_path)) == 64

    schema["derived_forecasts"]["load_pv_method"] = "actual_future"
    try:
        _validate_schema_contract(schema, schema_path)
    except ValueError as exc:
        assert "causal daily-persistence" in str(exc)
    else:
        raise AssertionError("Oracle load/PV forecasts must be rejected.")


def test_annual_rec_schema_contract_rejects_unpublished_price_oracle() -> None:
    schema_path = SIMULATOR_ROOT / "data/datasets/rec_2023_micro_4_q/schema.json"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    schema["derived_forecasts"]["price_source"] = "known_day_ahead_market_input"

    try:
        _validate_schema_contract(schema, schema_path)
    except ValueError as exc:
        assert "publication-aware causal OMIE" in str(exc)
    else:
        raise AssertionError("Unpublished realised price horizons must be rejected.")


def test_dataset_integrity_rehashes_every_declared_file(tmp_path) -> None:
    schema_path = tmp_path / "schema.json"
    payload_path = tmp_path / "payload.parquet"
    schema_path.write_text("{}\n", encoding="utf-8")
    payload_path.write_bytes(b"frozen payload")
    expected = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    (tmp_path / "file_checksums.sha256").write_text(
        f"{expected}  payload.parquet\n", encoding="utf-8"
    )

    assert len(_dataset_integrity_sha256(schema_path)) == 64

    payload_path.write_bytes(b"silently changed payload")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _dataset_integrity_sha256(schema_path)


def test_annual_campaign_attests_exact_simulator_source_tree() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SIMULATOR_ROOT)

    attestation = _attest_simulator_runtime(SIMULATOR_ROOT, env)

    assert Path(attestation["citylearn_source_root"]) == SIMULATOR_ROOT / "citylearn"
    assert attestation["citylearn_source_sha256"] == attestation[
        "expected_source_sha256"
    ]
    assert attestation["dynamic_historical_charger_kpis"] is True
    assert len(attestation["attestation_sha256"]) == 64
