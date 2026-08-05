from __future__ import annotations

import numpy as np
import pytest

from algorithms.oracles import (
    DeferrableCycle,
    EVSession,
    SolveOptions,
    StorageAsset,
    TotalEnergyProblem,
    solve_decomposed_individual_total_energy,
    solve_total_energy_schedule,
    split_individual_total_energy_problems,
)


def _source(*, settlement: str = "individual") -> TotalEnergyProblem:
    return TotalEnergyProblem(
        problem_id="two-building-total-energy",
        timestep_hours=1.0,
        building_ids=("Home_A", "Home_B"),
        price_eur_per_kwh=np.asarray([1.0, 10.0]),
        base_net_load_kwh=np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        settlement=settlement,
        # Deliberately use an asset order different from building order.
        stationary_storage=(
            StorageAsset(
                building_id="Home_B",
                action_name="battery_b",
                capacity_kwh=1.0,
                initial_energy_kwh=0.0,
                final_energy_min_kwh=0.0,
                max_charge_kw=1.0,
                max_discharge_kw=1.0,
            ),
        ),
        deferrable_cycles=(
            DeferrableCycle(
                cycle_id="dishwasher-a",
                building_id="Home_A",
                action_name="dishwasher_a",
                earliest_start_time_step=0,
                latest_start_time_step=1,
                load_profile_kwh=(1.0,),
            ),
        ),
        metadata={"dataset": "fixture"},
    )


def test_split_preserves_rows_assets_ids_and_rejects_community_coupling():
    home_a, home_b = split_individual_total_energy_problems(_source())

    assert home_a.problem_id.endswith("::individual::Home_A")
    assert home_a.building_ids == ("Home_A",)
    assert home_a.base_net_load_kwh.tolist() == [[1.0, 1.0]]
    assert [item.cycle_id for item in home_a.deferrable_cycles] == [
        "dishwasher-a"
    ]
    assert not home_a.stationary_storage
    assert home_b.building_ids == ("Home_B",)
    assert [item.action_name for item in home_b.stationary_storage] == [
        "battery_b"
    ]
    assert not home_b.deferrable_cycles
    assert home_a.metadata["source_problem_id"] == _source().problem_id
    assert home_a.metadata["community_optimum_claim"] is False

    with pytest.raises(ValueError, match="community netting is non-separable"):
        split_individual_total_energy_problems(_source(settlement="community"))


def test_decomposed_schedule_matches_joint_individual_model_and_keeps_semantics():
    source = _source()
    direct = solve_total_energy_schedule(source)
    decomposed = solve_decomposed_individual_total_energy(
        source, mode="schedule"
    )
    combined = decomposed.combined

    assert combined.cost_eur == pytest.approx(direct.cost_eur)
    assert np.asarray(combined.grid_import_kwh) == pytest.approx(
        np.asarray(direct.grid_import_kwh)
    )
    assert np.asarray(combined.building_net_load_kwh) == pytest.approx(
        np.asarray(direct.building_net_load_kwh)
    )
    assert combined.selected_deferrable_starts == {"dishwasher-a": 0}
    assert combined.schedule is not None
    assert [(item.building_id, item.action_name) for item in combined.schedule.series] == [
        ("Home_A", "dishwasher_a"),
        ("Home_B", "battery_b"),
    ]
    assert combined.schedule.metadata["action_power_limits_kw"]["Home_B"][
        "battery_b"
    ]["nominal_power_kw"] == pytest.approx(1.0)
    assert combined.schedule.metadata["community_optimum_claim"] is False
    assert combined.schedule.metadata["decomposition"] == (
        "exact_individual_settlement_by_building"
    )
    assert [item.building_id for item in decomposed.buildings] == [
        "Home_A",
        "Home_B",
    ]


def test_bounded_decomposition_sums_bounds_and_retains_building_evidence():
    result = solve_decomposed_individual_total_energy(_source(), mode="bounded")
    combined = result.combined

    assert combined.certificate_valid
    assert combined.certified_lower_bound_eur == pytest.approx(14.0)
    assert combined.model_feasible_upper_bound_eur == pytest.approx(14.0)
    assert combined.relative_gap == pytest.approx(0.0)
    assert combined.conservative.schedule is not None
    assert len(result.buildings) == 2
    assert all(item.result.certificate_valid for item in result.buildings)
    assert "not a community-settlement optimum" in result.guarantee


def test_soft_service_totals_preserve_exact_lexicographic_optimum():
    sessions = tuple(
        EVSession(
            session_id=f"ev-{building}",
            building_id=building,
            action_name=f"charger-{building}",
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
        for building in ("Home_A", "Home_B")
    )
    problem = TotalEnergyProblem(
        problem_id="soft-service-by-building",
        timestep_hours=1.0,
        building_ids=("Home_A", "Home_B"),
        price_eur_per_kwh=np.asarray([100.0]),
        base_net_load_kwh=np.zeros((2, 1)),
        settlement="individual",
        ev_sessions=sessions,
    )
    result = solve_decomposed_individual_total_energy(
        problem,
        mode="schedule",
    )
    combined = result.combined

    assert combined.minimum_total_ev_shortfall_kwh == pytest.approx(6.0)
    assert combined.realized_total_ev_shortfall_kwh == pytest.approx(6.0)
    assert combined.lexicographic_shortfall_tolerance_kwh == pytest.approx(0.0)
    assert combined.ev_departure_shortfall_kwh == pytest.approx(
        {"ev-Home_A": 3.0, "ev-Home_B": 3.0}
    )
    assert combined.schedule is not None
    assert combined.schedule.metadata[
        "per_building_lexicographic_shortfall_tolerance"
    ] is True

    with pytest.raises(ValueError, match="Exact individual decomposition requires"):
        solve_decomposed_individual_total_energy(
            problem,
            mode="schedule",
            options=SolveOptions(lexicographic_shortfall_tolerance_kwh=1.0e-3),
        )
