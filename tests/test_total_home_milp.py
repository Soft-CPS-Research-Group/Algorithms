import numpy as np
import pytest

from algorithms.oracles import (
    DeferrableCycleSpec,
    EVSessionSpec,
    ElectricalServiceSpec,
    LinearStorageSpec,
    TotalHomeProblem,
    solve_total_home_milp,
)


def _ev(**overrides):
    values = {
        "session_id": "session-1",
        "action_name": "electric_vehicle_storage_charger_1",
        "electric_vehicle_id": "EV_1",
        "start_time_step": 0,
        "end_time_step": 4,
        "capacity_kwh": 2.0,
        "initial_energy_kwh": 0.0,
        "required_departure_energy_kwh": 1.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kw": 1.0,
        "max_discharge_kw": 1.0,
    }
    values.update(overrides)
    return EVSessionSpec(**values)


def test_total_home_jointly_optimizes_storage_ev_and_deferrable():
    problem = TotalHomeProblem(
        problem_id="home",
        building_id="Building_1",
        timestep_hours=1.0,
        price_eur_per_kwh=np.array([1.0, 1.0, 10.0, 10.0]),
        base_net_load_kwh=np.ones(4),
        stationary_storage=LinearStorageSpec(
            capacity_kwh=2.0,
            initial_energy_kwh=0.0,
            final_energy_min_kwh=0.0,
            minimum_energy_kwh=0.0,
            max_charge_kw=1.0,
            max_discharge_kw=1.0,
        ),
        ev_sessions=(_ev(),),
        deferrable_cycles=(
            DeferrableCycleSpec(
                cycle_id="wash-1",
                action_name="deferrable_appliance_washer",
                earliest_start_time_step=0,
                latest_start_time_step=3,
                load_profile_kwh=(1.0,),
            ),
        ),
    )

    result = solve_total_home_milp(problem)

    assert result.status == "optimal"
    assert result.objective_eur == pytest.approx(6.0)
    assert result.ev_departure_energy_kwh["session-1"] >= 1.0 - 1.0e-8
    assert result.deferrable_start_time_step["wash-1"] in {0, 1}
    assert max(result.grid_import_kw[2:]) <= 1.0e-8
    by_action = {series.action_name: series for series in result.schedule.series}
    assert "electrical_storage" in by_action
    assert "electric_vehicle_storage_charger_1" in by_action
    assert by_action["deferrable_appliance_washer"].unit == "binary_start"
    assert result.schedule.metadata["global_community_optimum_claim"] is False


def test_total_home_enforces_total_and_per_phase_service_limits():
    service = ElectricalServiceSpec(
        mode="three_phase",
        total_import_limit_kw=2.0,
        total_export_limit_kw=2.0,
        per_phase_import_limit_kw={"L1": 0.5, "L2": 2.0, "L3": 2.0},
        per_phase_export_limit_kw={"L1": 2.0, "L2": 2.0, "L3": 2.0},
    )
    problem = TotalHomeProblem(
        problem_id="phase-infeasible",
        building_id="Building_15",
        timestep_hours=1.0,
        price_eur_per_kwh=np.ones(1),
        base_net_load_kwh=np.zeros(1),
        ev_sessions=(
            _ev(
                end_time_step=1,
                capacity_kwh=1.0,
                required_departure_energy_kwh=1.0,
                max_discharge_kw=0.0,
                phase_connection="L1",
            ),
        ),
        electrical_service=service,
    )

    result = solve_total_home_milp(problem)

    assert result.status == "infeasible"
    assert result.has_solution is False


def test_total_home_rejects_overlapping_sessions_on_same_charger():
    with pytest.raises(ValueError, match="Overlapping EV sessions"):
        TotalHomeProblem(
            problem_id="overlap",
            building_id="Building_1",
            timestep_hours=1.0,
            price_eur_per_kwh=np.ones(4),
            base_net_load_kwh=np.zeros(4),
            ev_sessions=(
                _ev(session_id="a", start_time_step=0, end_time_step=3),
                _ev(session_id="b", start_time_step=2, end_time_step=4),
            ),
        )
