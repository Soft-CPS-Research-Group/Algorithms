from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from algorithms.configuration.milp import MILPStageConfig
from algorithms.oracles.rolling_energy import (
    BoundedDatasetForecast,
    update_measured_problem,
)
from algorithms.oracles.total_energy_milp import (
    DeferrableCycle,
    EVSession,
    StorageAsset,
    TotalEnergyProblem,
    solve_total_energy_schedule,
)

SCHEMA = (
    Path(__file__).parents[1]
    / "datasets/citylearn_three_phase_electrical_service_demo_15min_parquet/schema.json"
)


def problem():
    return TotalEnergyProblem(
        problem_id="measured",
        timestep_hours=0.25,
        building_ids=("home",),
        price_eur_per_kwh=np.array([0.1, 0.2, 0.3, 0.4]),
        base_net_load_kwh=np.zeros((1, 4)),
        settlement="community",
        stationary_storage=(StorageAsset("home", "electrical_storage", 10, 5, 5, 5, 5),),
        ev_sessions=(EVSession("ev", "home", "charger", 0, 3, 40, 20, 24, 0, 20, 0),),
    )


def test_mpc_is_registered_with_strict_bounded_settings():
    from algorithms.registry import agent_spec

    assert agent_spec("MILP").capabilities.observations == "raw"
    assert MILPStageConfig(algorithm="MILP").hyperparameters.horizon_hours == 24
    for hyper in ({"horizon_hours": 168}, {"horizon_hours": 0}, {"schedule_path": "weekly.json"}):
        with pytest.raises(ValidationError):
            MILPStageConfig(algorithm="MILP", hyperparameters=hyper)


def test_forecast_cannot_see_changed_values_after_its_window():
    forecast = BoundedDatasetForecast(SCHEMA, ["Building_1"])
    before = forecast.problem(10, 50)
    for path, frame in forecast._cache.items():
        if "electricity_pricing" in frame:
            frame.loc[50:, "electricity_pricing"] = -999.0
        if "non_shiftable_load" in frame:
            frame.loc[50:, "non_shiftable_load"] = 1e9
        if "electric_vehicle_charger_state" in frame:
            frame.loc[50:, "electric_vehicle_charger_state"] = 0
            frame.loc[50:, "electric_vehicle_required_soc_departure"] = np.nan
        if "earliest_start_time_step" in frame:
            future = frame.earliest_start_time_step >= 50
            frame.loc[future, "profile_id"] = "invalid_future_profile"
    after = forecast.problem(10, 50)
    assert before.to_dict() == after.to_dict()
    assert after.horizon == 40
    assert after.metadata["source_start_time_step"] == 10
    with pytest.raises(ValueError, match="24 hours"):
        forecast.problem(10, 107)


def test_new_measurements_replace_static_initial_energy_and_keep_fixed_reserve():
    updated, running = update_measured_problem(
        problem(),
        storage_energy={("home", "electrical_storage"): (7, 10)},
        ev_energy={("home", "charger"): (23, 40)},
        started_cycles={},
        now=0,
        terminal_storage_soc=0.5,
    )
    assert updated.stationary_storage[0].initial_energy_kwh == 7
    assert updated.stationary_storage[0].final_energy_min_kwh == 5
    assert updated.ev_sessions[0].initial_energy_kwh == 23
    assert not running
    next_problem, _ = update_measured_problem(
        problem(),
        storage_energy={("home", "electrical_storage"): (6, 10)},
        ev_energy={("home", "charger"): (24, 40)},
        started_cycles={},
        now=1,
        terminal_storage_soc=0.5,
    )
    assert next_problem.stationary_storage[0].initial_energy_kwh == 6
    assert next_problem.ev_sessions[0].initial_energy_kwh == 24


def test_committed_cycle_preserves_phase_zero_power_occupancy_and_completed_ids():
    first = DeferrableCycle("first", "home", "washer", 0, 1, (1, 0, 2), phase_connection="L1")
    second = DeferrableCycle("second", "home", "washer", 0, 3, (1,), phase_connection="L1")
    source = replace(
        problem(), stationary_storage=(), ev_sessions=(), deferrable_cycles=(first, second)
    )
    updated, running = update_measured_problem(
        source,
        storage_energy={},
        ev_energy={},
        started_cycles={"first": (10, first)},
        now=11,
        terminal_storage_soc=0.5,
    )
    tail = next(c for c in updated.deferrable_cycles if c.cycle_id == "first")
    assert tail.load_profile_kwh == (0, 2)
    assert tail.phase_connection == "L1"
    assert running == {"first"}
    result = solve_total_energy_schedule(updated)
    assert result.selected_deferrable_starts["second"] == 2
    completed, running = update_measured_problem(
        source,
        storage_energy={},
        ev_energy={},
        started_cycles={"first": (10, first)},
        now=13,
        terminal_storage_soc=0.5,
    )
    assert [c.cycle_id for c in completed.deferrable_cycles] == ["second"]
    assert not running


def test_changed_connected_ev_capacity_fails_instead_of_using_wrong_asset_state():
    with pytest.raises(ValueError, match="capacity differs"):
        update_measured_problem(
            problem(),
            storage_energy={("home", "electrical_storage"): (5, 10)},
            ev_energy={("home", "charger"): (23, 50)},
            started_cycles={},
            now=0,
            terminal_storage_soc=0.5,
        )


def test_explicit_solver_threads_can_change_without_corrupting_native_scheduler():
    from algorithms.oracles.perfect_foresight_milp import SolveOptions

    costs = []
    for threads in (None, 1, 2, 1, None):
        result = solve_total_energy_schedule(problem(), options=SolveOptions(threads=threads))
        assert result.solver.optimal
        costs.append(result.cost_eur)
    assert costs == pytest.approx([costs[0]] * len(costs))
    with pytest.raises(ValueError, match="positive integer"):
        SolveOptions(threads=0)


@pytest.mark.parametrize("asset_kind", ["storage", "ev"])
def test_measured_energy_below_reserve_is_restored_by_real_charge(asset_kind):
    source = problem()
    if asset_kind == "storage":
        source = replace(
            source,
            stationary_storage=(replace(source.stationary_storage[0], minimum_energy_kwh=1),),
            ev_sessions=(),
        )
    else:
        source = replace(
            source,
            stationary_storage=(),
            ev_sessions=(
                replace(source.ev_sessions[0], minimum_energy_kwh=1, required_final_energy_kwh=4),
            ),
        )
    updated, _ = update_measured_problem(
        source,
        storage_energy={("home", "electrical_storage"): (0.8, 10)},
        ev_energy={("home", "charger"): (0.8, 40)},
        started_cycles={},
        now=0,
        terminal_storage_soc=0.5,
    )
    asset = (updated.stationary_storage or updated.ev_sessions)[0]
    assert asset.initial_energy_kwh == 0.8
    assert asset.minimum_energy_kwh == 1
    result = solve_total_energy_schedule(updated)
    assert result.schedule is not None
    action = next(s for s in result.schedule.series if s.action_name == asset.action_name)
    assert action.values[0] * updated.timestep_hours >= 0.2 - 1e-6


def test_entity_soc_uses_explicit_ratio_units_at_full_charge():
    from algorithms.agents.milp_policy import MILPPolicy

    assert MILPPolicy._soc(1.00000001) == 1
    assert MILPPolicy._soc(0.01) == 0.01
    with pytest.raises(ValueError, match="outside"):
        MILPPolicy._soc(50)
