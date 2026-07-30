from pathlib import Path
import hashlib

import numpy as np
import pytest

from algorithms.oracles import build_citylearn_total_home_problem


SCHEMA = Path(
    "datasets/citylearn_three_phase_electrical_service_demo_15min_parquet/schema.json"
)


def test_real_building_one_closed_window_contains_full_local_services():
    built = build_citylearn_total_home_problem(
        schema_path=SCHEMA,
        building_id="Building_1",
        start_time_step=0,
        end_time_step=320,
    )

    problem = built.problem
    assert problem.horizon == 320
    assert built.ev_session_count == 4
    assert built.deferrable_cycle_count == 4
    assert problem.stationary_storage is not None
    assert problem.stationary_storage.charge_efficiency == pytest.approx(
        np.sqrt(0.9 * 0.85)
    )
    assert problem.electrical_service is None
    assert {session.action_name for session in problem.ev_sessions} == {
        "electric_vehicle_storage_charger_1_1"
    }
    assert problem.metadata["community_observations_used"] is False


def test_real_building_fifteen_loads_three_phase_service_and_two_chargers():
    built = build_citylearn_total_home_problem(
        schema_path=SCHEMA,
        building_id="Building_15",
        start_time_step=0,
        end_time_step=424,
    )

    assert built.ev_session_count == 10
    assert built.problem.electrical_service.mode == "three_phase"
    assert built.problem.electrical_service.per_phase_import_limit_kw == {
        "L1": 7.0,
        "L2": 5.0,
        "L3": 4.0,
    }
    assert {session.phase_connection for session in built.problem.ev_sessions} == {"L1", "L2"}
    assert all(
        session.charge_efficiency == pytest.approx(0.95 * np.sqrt(0.9 * 0.85))
        for session in built.problem.ev_sessions
    )

    ev5_sessions = [
        session
        for session in built.problem.ev_sessions
        if session.electric_vehicle_id == "Electric_Vehicle_5"
    ]
    seed_source = "2022:Electric_Vehicle_5:initial_soc"
    seed = int(hashlib.md5(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    expected_initial_soc = float(np.random.RandomState(seed).uniform(0.0, 1.0))
    assert ev5_sessions
    assert all(
        session.metadata["configured_or_deterministic_initial_soc"]
        == pytest.approx(expected_initial_soc)
        for session in ev5_sessions
    )
    assert {
        session.metadata["initial_soc_source"] for session in ev5_sessions
    } == {"citylearn_deterministic_schema_seed_fallback"}


def test_adapter_rejects_window_that_cuts_connected_ev_session():
    with pytest.raises(ValueError, match="cuts EV session"):
        build_citylearn_total_home_problem(
            schema_path=SCHEMA,
            building_id="Building_1",
            start_time_step=0,
            end_time_step=100,
        )
