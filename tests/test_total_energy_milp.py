from __future__ import annotations

import numpy as np
import pytest

from algorithms.oracles.total_energy_milp import (
    DeferrableCycle,
    ElectricalService,
    EVSession,
    StorageAsset,
    TotalEnergyProblem,
    solve_bounded_total_energy_oracle,
    solve_total_energy_schedule,
)
from algorithms.oracles.perfect_foresight_milp import SolveOptions


def _problem(**overrides):
    data = {
        "problem_id": "tiny-total-energy",
        "timestep_hours": 1.0,
        "building_ids": ("Building_1",),
        "price_eur_per_kwh": np.array([1.0, 10.0]),
        "base_net_load_kwh": np.array([[1.0, 1.0]]),
    }
    data.update(overrides)
    return TotalEnergyProblem(**data)


def _series(result, building_id, action_name):
    assert result.schedule is not None
    return next(
        item
        for item in result.schedule.series
        if (item.building_id, item.action_name) == (building_id, action_name)
    )


def test_stationary_storage_arbitrages_and_bounded_certificate_is_valid():
    storage = StorageAsset(
        building_id="Building_1",
        action_name="electrical_storage",
        capacity_kwh=1.0,
        initial_energy_kwh=0.0,
        final_energy_min_kwh=0.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
    )
    result = solve_bounded_total_energy_oracle(
        _problem(stationary_storage=(storage,))
    )

    assert result.certificate_valid
    assert result.model_feasible_upper_bound_eur == pytest.approx(2.0)
    assert result.certified_lower_bound_eur == pytest.approx(2.0)
    assert result.relative_gap == pytest.approx(0.0)
    series = _series(result.conservative, "Building_1", "electrical_storage")
    assert series.values == pytest.approx((1.0, -1.0))
    assert result.conservative.schedule.metadata["action_power_limits_kw"] == {
        "Building_1": {
            "electrical_storage": {
                "nominal_power_kw": 1.0,
                "max_charging_power_kw": 1.0,
                "max_discharging_power_kw": 1.0,
            }
        }
    }


def test_individual_and_community_settlement_have_distinct_netting_semantics():
    base = np.array([[1.0, -1.0], [-1.0, 1.0]])
    common = {
        "problem_id": "netting",
        "timestep_hours": 1.0,
        "building_ids": ("Building_1", "Building_2"),
        "price_eur_per_kwh": np.ones(2),
        "base_net_load_kwh": base,
    }

    individual = solve_total_energy_schedule(
        TotalEnergyProblem(**common, settlement="individual")
    )
    community = solve_total_energy_schedule(
        TotalEnergyProblem(**common, settlement="community")
    )

    assert individual.cost_eur == pytest.approx(2.0)
    assert community.cost_eur == pytest.approx(0.0)
    assert np.asarray(individual.grid_import_kwh).shape == (2, 2)
    assert np.asarray(community.grid_import_kwh).shape == (1, 2)


def test_ev_charges_in_cheapest_connected_step_and_meets_departure_target():
    session = EVSession(
        session_id="charger_1_1:0",
        building_id="Building_1",
        action_name="electric_vehicle_storage_charger_1_1",
        start_time_step=0,
        end_time_step=1,
        capacity_kwh=2.0,
        initial_energy_kwh=0.0,
        required_final_energy_kwh=1.0,
        minimum_energy_kwh=0.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
    )
    result = solve_total_energy_schedule(
        _problem(base_net_load_kwh=np.zeros((1, 2)), ev_sessions=(session,))
    )

    assert result.solver.optimal
    assert result.cost_eur == pytest.approx(1.0)
    assert result.ev_final_energy_kwh[session.session_id] == pytest.approx(1.0)
    series = _series(result, "Building_1", session.action_name)
    assert series.values == pytest.approx((1.0, 0.0))
    assert result.schedule.metadata["action_power_limits_kw"]["Building_1"][
        session.action_name
    ] == {
        "max_charging_power_kw": 1.0,
        "max_discharging_power_kw": 1.0,
        "min_charging_power_kw": 0.0,
        "min_discharging_power_kw": 0.0,
    }


def test_ev_v2g_discharge_is_available_but_preserves_departure_energy():
    session = EVSession(
        session_id="v2g",
        building_id="Building_1",
        action_name="electric_vehicle_storage_charger_1_1",
        start_time_step=0,
        end_time_step=1,
        capacity_kwh=2.0,
        initial_energy_kwh=2.0,
        required_final_energy_kwh=1.0,
        minimum_energy_kwh=0.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
    )
    result = solve_total_energy_schedule(_problem(ev_sessions=(session,)))

    assert result.cost_eur == pytest.approx(1.0)
    assert result.ev_final_energy_kwh[session.session_id] == pytest.approx(1.0)
    series = _series(result, "Building_1", session.action_name)
    assert series.values == pytest.approx((0.0, -1.0))


def test_ev_deadband_is_enforced_by_binary_on_state():
    session = EVSession(
        session_id="deadband",
        building_id="Building_1",
        action_name="electric_vehicle_storage_charger_1_1",
        start_time_step=0,
        end_time_step=1,
        capacity_kwh=5.0,
        initial_energy_kwh=0.0,
        required_final_energy_kwh=1.5,
        minimum_energy_kwh=0.0,
        max_charge_kw=2.0,
        max_discharge_kw=0.0,
        min_charge_kw=1.4,
    )
    result = solve_total_energy_schedule(
        _problem(base_net_load_kwh=np.zeros((1, 2)), ev_sessions=(session,))
    )
    values = np.asarray(_series(result, "Building_1", session.action_name).values)

    assert result.solver.optimal
    assert np.count_nonzero(values) == 1
    assert values.max() >= 1.4 - 1.0e-8


def test_deferrable_cycle_selects_cheapest_feasible_start_and_emits_trigger():
    cycle = DeferrableCycle(
        cycle_id="wash-1",
        building_id="Building_1",
        action_name="deferrable_appliance_deferrable_appliance_1",
        earliest_start_time_step=0,
        latest_start_time_step=1,
        load_profile_kwh=(1.0,),
    )
    result = solve_total_energy_schedule(
        _problem(base_net_load_kwh=np.zeros((1, 2)), deferrable_cycles=(cycle,))
    )

    assert result.cost_eur == pytest.approx(1.0)
    assert result.selected_deferrable_starts[cycle.cycle_id] == 0
    series = _series(result, "Building_1", cycle.action_name)
    assert series.unit == "normalized_action"
    assert series.values == pytest.approx((1.0, 0.0))


def test_building_15_all_phase_storage_respects_tightest_phase_limit():
    storage = StorageAsset(
        building_id="Building_15",
        action_name="electrical_storage",
        capacity_kwh=9.0,
        initial_energy_kwh=0.0,
        final_energy_min_kwh=3.0,
        max_charge_kw=9.0,
        max_discharge_kw=0.0,
        phase_connection="all_phases",
    )
    service = ElectricalService(
        building_id="Building_15",
        total_import_kw=10.0,
        total_export_kw=10.0,
        phase_import_kw={"L1": 4.0, "L2": 2.0, "L3": 1.0},
        phase_export_kw={"L1": 4.0, "L2": 2.0, "L3": 1.0},
    )
    problem = TotalEnergyProblem(
        problem_id="phase-cap",
        timestep_hours=1.0,
        building_ids=("Building_15",),
        price_eur_per_kwh=np.ones(2),
        base_net_load_kwh=np.zeros((1, 2)),
        stationary_storage=(storage,),
        electrical_services=(service,),
    )
    result = solve_total_energy_schedule(problem)
    values = np.asarray(_series(result, "Building_15", "electrical_storage").values)

    assert result.solver.optimal
    assert values.max() <= 3.0 + 1.0e-8
    assert values.sum() == pytest.approx(3.0)


def test_phase_specific_ev_and_balanced_storage_share_correct_phase_envelope():
    storage = StorageAsset(
        building_id="Building_15",
        action_name="electrical_storage",
        capacity_kwh=3.0,
        initial_energy_kwh=0.0,
        final_energy_min_kwh=3.0,
        max_charge_kw=3.0,
        max_discharge_kw=0.0,
        phase_connection="all_phases",
    )
    ev = EVSession(
        session_id="l2",
        building_id="Building_15",
        action_name="electric_vehicle_storage_charger_15_2",
        start_time_step=0,
        end_time_step=0,
        capacity_kwh=1.0,
        initial_energy_kwh=0.0,
        required_final_energy_kwh=1.0,
        minimum_energy_kwh=0.0,
        max_charge_kw=1.0,
        max_discharge_kw=0.0,
        phase_connection="L2",
    )
    service = ElectricalService(
        building_id="Building_15",
        total_import_kw=4.0,
        phase_import_kw={"L1": 2.0, "L2": 2.0, "L3": 2.0},
    )
    problem = TotalEnergyProblem(
        problem_id="phase-sharing",
        timestep_hours=1.0,
        building_ids=("Building_15",),
        price_eur_per_kwh=np.ones(1),
        base_net_load_kwh=np.zeros((1, 1)),
        stationary_storage=(storage,),
        ev_sessions=(ev,),
        electrical_services=(service,),
    )
    result = solve_total_energy_schedule(problem)

    assert result.solver.optimal
    # L2 sees 3/3 kW from storage plus 1 kW from its charger.
    assert result.cost_eur == pytest.approx(4.0)


def test_infeasible_fixed_base_service_is_reported_without_schedule():
    service = ElectricalService(building_id="Building_1", total_import_kw=1.0)
    result = solve_total_energy_schedule(
        _problem(
            base_net_load_kwh=np.array([[2.0, 2.0]]),
            electrical_services=(service,),
        )
    )

    assert result.solver.status == "infeasible"
    assert not result.solver.has_solution
    assert result.schedule is None


def test_storage_standby_loss_must_be_replenished_to_meet_terminal_energy():
    storage = StorageAsset(
        building_id="Building_1",
        action_name="electrical_storage",
        capacity_kwh=2.0,
        initial_energy_kwh=1.0,
        final_energy_min_kwh=1.0,
        max_charge_kw=1.0,
        max_discharge_kw=0.0,
        loss_coefficient=0.1,
    )
    result = solve_total_energy_schedule(
        _problem(
            price_eur_per_kwh=np.ones(2),
            base_net_load_kwh=np.zeros((1, 2)),
            stationary_storage=(storage,),
        )
    )

    assert result.solver.optimal
    assert result.cost_eur == pytest.approx(0.1)


def test_problem_json_round_trip_preserves_all_asset_contracts():
    storage = StorageAsset(
        building_id="Building_1",
        action_name="electrical_storage",
        capacity_kwh=2.0,
        initial_energy_kwh=0.5,
        final_energy_min_kwh=0.5,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
        loss_coefficient=0.01,
    )
    cycle = DeferrableCycle(
        cycle_id="cycle",
        building_id="Building_1",
        action_name="deferrable_appliance_x",
        earliest_start_time_step=0,
        latest_start_time_step=1,
        load_profile_kwh=(0.2,),
    )
    original = _problem(stationary_storage=(storage,), deferrable_cycles=(cycle,))
    restored = TotalEnergyProblem.from_json(original.to_json())

    assert restored.to_dict() == original.to_dict()


def test_soft_ev_departure_target_is_lexicographic_before_cost():
    session = EVSession(
        session_id="unreachable",
        building_id="Building_1",
        action_name="electric_vehicle_storage_charger_1_1",
        start_time_step=0,
        end_time_step=0,
        capacity_kwh=10.0,
        initial_energy_kwh=0.0,
        required_final_energy_kwh=5.0,
        minimum_energy_kwh=0.0,
        max_charge_kw=2.0,
        max_discharge_kw=0.0,
        allow_departure_shortfall=True,
    )
    result = solve_total_energy_schedule(
        TotalEnergyProblem(
            problem_id="soft-target",
            timestep_hours=1.0,
            building_ids=("Building_1",),
            price_eur_per_kwh=np.array([100.0]),
            base_net_load_kwh=np.zeros((1, 1)),
            ev_sessions=(session,),
        )
    )

    assert result.solver.optimal
    assert result.ev_final_energy_kwh[session.session_id] == pytest.approx(
        1.999, abs=1.0e-8
    )
    assert result.ev_departure_shortfall_kwh[session.session_id] == pytest.approx(
        3.001, abs=1.0e-8
    )
    assert result.minimum_total_ev_shortfall_kwh == pytest.approx(3.0)
    assert result.realized_total_ev_shortfall_kwh == pytest.approx(3.001)
    assert result.lexicographic_shortfall_tolerance_kwh == pytest.approx(1.0e-3)
    assert result.cost_eur == pytest.approx(199.9)


def test_lexicographic_shortfall_tolerance_is_configurable_and_validated():
    session = EVSession(
        session_id="unreachable-configurable",
        building_id="Building_1",
        action_name="electric_vehicle_storage_charger_1_1",
        start_time_step=0,
        end_time_step=0,
        capacity_kwh=10.0,
        initial_energy_kwh=0.0,
        required_final_energy_kwh=5.0,
        minimum_energy_kwh=0.0,
        max_charge_kw=2.0,
        max_discharge_kw=0.0,
        allow_departure_shortfall=True,
    )
    result = solve_total_energy_schedule(
        TotalEnergyProblem(
            problem_id="soft-target-configurable",
            timestep_hours=1.0,
            building_ids=("Building_1",),
            price_eur_per_kwh=np.array([100.0]),
            base_net_load_kwh=np.zeros((1, 1)),
            ev_sessions=(session,),
        ),
        SolveOptions(lexicographic_shortfall_tolerance_kwh=0.01),
    )

    assert result.minimum_total_ev_shortfall_kwh == pytest.approx(3.0)
    assert result.realized_total_ev_shortfall_kwh == pytest.approx(3.01)
    assert result.lexicographic_shortfall_tolerance_kwh == pytest.approx(0.01)
    with pytest.raises(ValueError, match="lexicographic_shortfall_tolerance_kwh"):
        SolveOptions(lexicographic_shortfall_tolerance_kwh=-1.0)


def test_community_lexicographic_caps_preserve_fully_serviceable_buildings():
    sessions = (
        EVSession(
            session_id="unreachable-b1",
            building_id="Building_1",
            action_name="electric_vehicle_storage_charger_1_1",
            start_time_step=0,
            end_time_step=0,
            capacity_kwh=10.0,
            initial_energy_kwh=0.0,
            required_final_energy_kwh=5.0,
            minimum_energy_kwh=0.0,
            max_charge_kw=2.0,
            max_discharge_kw=0.0,
            allow_departure_shortfall=True,
        ),
        EVSession(
            session_id="reachable-b2",
            building_id="Building_2",
            action_name="electric_vehicle_storage_charger_2_1",
            start_time_step=0,
            end_time_step=0,
            capacity_kwh=10.0,
            initial_energy_kwh=0.0,
            required_final_energy_kwh=1.0,
            minimum_energy_kwh=0.0,
            max_charge_kw=1.0,
            max_discharge_kw=0.0,
            allow_departure_shortfall=True,
        ),
    )
    result = solve_total_energy_schedule(
        TotalEnergyProblem(
            problem_id="community-building-service-caps",
            timestep_hours=1.0,
            building_ids=("Building_1", "Building_2"),
            price_eur_per_kwh=np.array([100.0]),
            base_net_load_kwh=np.zeros((2, 1)),
            settlement="community",
            ev_sessions=sessions,
        )
    )

    assert result.solver.optimal
    assert result.service_phase_optimal is True
    assert result.service_phase_status == "optimal"
    assert result.minimum_ev_shortfall_by_building_kwh == pytest.approx(
        {"Building_1": 3.0, "Building_2": 0.0}
    )
    assert result.lexicographic_shortfall_cap_by_building_kwh == pytest.approx(
        {"Building_1": 3.001, "Building_2": 0.0}
    )
    assert result.realized_ev_shortfall_by_building_kwh == pytest.approx(
        {"Building_1": 3.001, "Building_2": 0.0}
    )
    assert result.ev_departure_shortfall_kwh["reachable-b2"] == pytest.approx(0.0)
    assert result.cost_eur == pytest.approx(299.9)
