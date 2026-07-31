from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from algorithms.oracles.citylearn_total_energy import (
    build_citylearn_total_energy_problem,
)
from algorithms.oracles.total_energy_milp import solve_total_energy_schedule


_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "citylearn_three_phase_electrical_service_demo_15min_parquet"
    / "schema.json"
)


def test_extracts_full_real_dataset_assets_and_reserved_service() -> None:
    built = build_citylearn_total_energy_problem(
        schema_path=_SCHEMA,
        start_time_step=0,
        end_time_step=35_040,
        settlement="community",
        problem_id="real-full-year-extraction",
    )

    problem = built.problem
    diagnostics = built.diagnostics
    assert problem.building_ids == tuple(f"Building_{index}" for index in range(1, 18))
    assert problem.horizon == 35_040
    assert problem.timestep_hours == pytest.approx(0.25)
    assert problem.base_net_load_kwh.shape == (17, 35_040)
    assert problem.base_net_load_kwh[0, :5] == pytest.approx(
        np.asarray([0.56895, 0.56895, 0.56895, 0.56895, 0.2127916667])
    )
    assert problem.price_eur_per_kwh[:5] == pytest.approx(
        np.asarray([0.16464, 0.16464, 0.16464, 0.16464, 0.15748])
    )

    assert len(problem.stationary_storage) == 17
    assert len(problem.ev_sessions) == 2_795
    assert len(problem.deferrable_cycles) == 365
    assert len(problem.electrical_services) == 1
    assert diagnostics.charger_count == 8
    assert diagnostics.left_truncated_ev_session_ids == ()
    assert diagnostics.assumed_initial_soc_ev_session_ids == ()
    # Twenty-five later connections have no current/arrival SOC in the
    # packaged dataset.  CityLearn drifts those EVs while disconnected, so a
    # static schema fallback is visible and cannot be called replay-exact.
    assert len(diagnostics.runtime_drift_initial_soc_ev_session_ids) == 25
    # Each charger is still state=1 in the last dataset row, so all eight
    # terminal sessions are explicitly right-censored rather than treated as
    # observed departures.
    assert len(diagnostics.right_truncated_ev_session_ids) == 8
    assert diagnostics.boundary_service_exact is False

    storage_15 = next(
        item for item in problem.stationary_storage if item.building_id == "Building_15"
    )
    assert storage_15.phase_connection == "all_phases"
    assert storage_15.capacity_kwh == pytest.approx(6.4 * 0.99)
    assert storage_15.max_charge_kw == pytest.approx(1.0)

    service = problem.electrical_services[0]
    assert service.building_id == "Building_15"
    assert service.total_import_kw == pytest.approx(11.9)
    assert service.total_export_kw == pytest.approx(11.9)
    assert service.phase_import_kw == pytest.approx(
        {"L1": 6.9, "L2": 4.9, "L3": 3.9}
    )
    assert service.phase_export_kw == pytest.approx(
        {"L1": 6.9, "L2": 4.9, "L3": 3.9}
    )
    building_15_ev = [
        item for item in problem.ev_sessions if item.building_id == "Building_15"
    ]
    assert {item.action_name for item in building_15_ev} == {
        "electric_vehicle_storage_charger_15_1",
        "electric_vehicle_storage_charger_15_2",
    }
    assert {item.phase_connection for item in building_15_ev} == {"L1", "L2"}


def test_later_window_marks_left_soc_assumption_and_restricts_cycle() -> None:
    built = build_citylearn_total_energy_problem(
        schema_path=_SCHEMA,
        start_time_step=10,
        end_time_step=20,
        building_ids=("Building_1",),
        problem_id="real-boundary-diagnostic",
    )

    diagnostics = built.diagnostics
    assert len(built.problem.ev_sessions) == 1
    assert len(diagnostics.left_truncated_ev_session_ids) == 1
    assert len(diagnostics.right_truncated_ev_session_ids) == 1
    assert diagnostics.assumed_initial_soc_ev_session_ids == (
        diagnostics.left_truncated_ev_session_ids[0],
    )
    assert len(diagnostics.restricted_deferrable_cycle_ids) == 1
    session = built.problem.ev_sessions[0]
    assert (session.start_time_step, session.end_time_step) == (0, 9)
    assert session.required_final_energy_kwh == pytest.approx(
        session.minimum_energy_kwh
    )
    cycle = built.problem.deferrable_cycles[0]
    assert (cycle.earliest_start_time_step, cycle.latest_start_time_step) == (0, 5)
    assert built.problem.metadata["boundary_service_exact"] is False
    assert (
        built.problem.metadata["ev_initial_soc_source"][session.session_id]
        == "schema_episode_reset"
    )
    assert session.initial_energy_kwh == pytest.approx(60.0 * 0.25)


def test_nonzero_episode_boundary_uses_citylearn_reset_soc_not_prior_dataset_row() -> None:
    built = build_citylearn_total_energy_problem(
        schema_path=_SCHEMA,
        start_time_step=1316,
        end_time_step=1404,
        building_ids=("Building_1",),
        problem_id="nonzero-boundary-reset-soc",
    )

    session = built.problem.ev_sessions[0]
    assert session.session_id.endswith("session_0015::1316_1396")
    # Row 1315 advertises an incoming SOC of 0.46, but it is outside the
    # episode.  CityLearn resets EV7 to its schema initial SOC of 0.25.
    assert session.initial_energy_kwh == pytest.approx(60.0 * 0.25)
    assert (
        built.problem.metadata["ev_initial_soc_source"][session.session_id]
        == "schema_episode_reset"
    )


def test_later_connection_without_arrival_soc_is_explicitly_not_replay_exact() -> None:
    built = build_citylearn_total_energy_problem(
        schema_path=_SCHEMA,
        start_time_step=1316,
        end_time_step=1500,
        building_ids=("Building_10",),
        problem_id="runtime-drift-arrival-diagnostic",
    )

    session_id = "Building_10::charger_10_1::session_0013::1440_1496"
    assert built.diagnostics.runtime_drift_initial_soc_ev_session_ids == (
        session_id,
    )
    assert built.diagnostics.boundary_service_exact is False
    assert built.problem.metadata["boundary_service_exact"] is False
    assert (
        built.problem.metadata["ev_initial_soc_source"][session_id]
        == (
            "citylearn_deterministic_schema_seed_fallback_"
            "runtime_drift_unreproducible"
        )
    )
    assert "runtime_drift_initial_soc_reason" in built.problem.metadata[
        "boundary_diagnostics"
    ]


def test_initial_sessions_match_citylearn_deterministic_soc_for_b10_and_b12() -> None:
    built = build_citylearn_total_energy_problem(
        schema_path=_SCHEMA,
        start_time_step=0,
        end_time_step=60,
        building_ids=("Building_10", "Building_12"),
        problem_id="real-deterministic-initial-ev-soc",
    )

    initial_sessions = {
        session.building_id: session
        for session in built.problem.ev_sessions
        if session.start_time_step == 0
    }
    expected_soc = {
        "Building_10": 0.71451492183283,
        "Building_12": 0.7071841267913679,
    }
    assert set(initial_sessions) == set(expected_soc)
    for building_id, expected in expected_soc.items():
        session = initial_sessions[building_id]
        assert session.initial_energy_kwh == pytest.approx(
            session.capacity_kwh * expected
        )
        assert (
            built.problem.metadata["ev_initial_soc_source"][session.session_id]
            == "citylearn_deterministic_schema_seed_fallback"
        )


def test_explicit_ev_initial_soc_overrides_deterministic_fallback(
    tmp_path: Path,
) -> None:
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    schema["electric_vehicles_def"]["Electric_Vehicle_3"]["battery"][
        "attributes"
    ]["initial_soc"] = 0.42
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    built = build_citylearn_total_energy_problem(
        schema_path=schema_path,
        start_time_step=0,
        end_time_step=224,
        building_ids=("Building_10",),
        problem_id="real-explicit-initial-ev-soc",
    )

    session = built.problem.ev_sessions[0]
    assert session.initial_energy_kwh == pytest.approx(75.0 * 0.42)
    assert (
        built.problem.metadata["ev_initial_soc_source"][session.session_id]
        == "schema"
    )


def test_small_real_building_15_window_solves_with_phase_service() -> None:
    built = build_citylearn_total_energy_problem(
        schema_path=_SCHEMA,
        start_time_step=0,
        end_time_step=20,
        building_ids=("Building_15",),
        settlement="individual",
        problem_id="real-building-15-smoke",
    )

    assert len(built.problem.ev_sessions) == 2
    assert len(built.problem.electrical_services) == 1
    result = solve_total_energy_schedule(built.problem)
    assert result.solver.optimal is True
    assert result.solver.has_solution is True
    assert result.cost_eur is not None
    assert result.schedule is not None
    assert {series.action_name for series in result.schedule.series} == {
        "electrical_storage",
        "electric_vehicle_storage_charger_15_1",
        "electric_vehicle_storage_charger_15_2",
    }
    # Independently verify the optimized trace respects the reserved aggregate
    # service envelope; per-phase feasibility is enforced inside the MILP and
    # covered above by the exact phase mappings/limits.
    net = np.asarray(result.building_net_load_kwh, dtype=np.float64)[0]
    assert np.max(net) <= 11.9 + 1.0e-7
    assert np.min(net) >= -11.9 - 1.0e-7
