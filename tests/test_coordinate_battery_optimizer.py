from __future__ import annotations

import numpy as np

from algorithms.oracles import (
    BatteryAsset,
    BatteryModel,
    PerfectForesightProblem,
    ScorecardShapingOptions,
    SemanticActionSeries,
    SemanticSchedule,
    SolveOptions,
    optimize_physical_battery_schedule_coordinate_descent,
)
from algorithms.oracles.coordinate_battery_optimizer import (
    _project_signed_power_to_battery,
)


def test_coordinate_dispatch_reduces_emissions_and_preserves_global_limits() -> None:
    model = BatteryModel(
        capacity_kwh=2.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    buildings = ("Building_1", "Building_2")
    problem = PerfectForesightProblem(
        problem_id="coordinate-test",
        timestep_hours=1.0,
        building_ids=buildings,
        price_eur_per_kwh=np.ones(2),
        base_net_load_kwh=np.asarray([[0.0, 2.0], [0.0, 2.0]]),
        batteries=tuple(
            BatteryAsset(
                building_id=building,
                action_name="electrical_storage",
                initial_energy_kwh=1.0,
                final_energy_min_kwh=1.0,
                optimistic=model,
                conservative=model,
            )
            for building in buildings
        ),
    )
    initial = SemanticSchedule(
        problem_id="coordinate-initial",
        horizon=2,
        timestep_hours=1.0,
        series=tuple(
            SemanticActionSeries(
                building_id=building,
                action_name="electrical_storage",
                values=(0.0, 0.0),
            )
            for building in buildings
        ),
    )
    result = optimize_physical_battery_schedule_coordinate_descent(
        problem,
        initial,
        [0.0, 1.0],
        ScorecardShapingOptions(
            community_cost_limit_eur=4.01,
            ramping_weight=0.0,
            daily_peak_weight=0.0,
            all_time_peak_weight=0.0,
            emissions_weight=1.0,
            emissions_accounting="gross_member_import",
            enforce_exclusive_battery_direction=True,
            mean_absolute_ramp_limit_kwh=4.0,
            mean_daily_peak_import_limit_kwh=4.0,
            all_time_peak_import_limit_kwh=4.0,
        ),
        SolveOptions(time_limit_seconds=10.0, solver="simplex"),
        max_sweeps=2,
    )

    assert result.final_metrics.community_emissions_kgco2 < (
        result.initial_metrics.community_emissions_kgco2
    )
    assert result.final_metrics.community_cost_eur <= 4.01
    assert result.final_metrics.mean_absolute_ramp_kwh <= 4.0
    assert result.accepted_updates > 0
    assert result.schedule.metadata["global_optimum_claim"] is False


def test_coordinate_projection_removes_unreplayable_energy_overflow() -> None:
    model = BatteryModel(
        capacity_kwh=1.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
        charge_efficiency=0.9,
        discharge_efficiency=0.9,
    )
    problem = PerfectForesightProblem(
        problem_id="coordinate-projection-test",
        timestep_hours=1.0,
        building_ids=("Building_1",),
        price_eur_per_kwh=np.ones(3),
        base_net_load_kwh=np.zeros((1, 3)),
        batteries=(
            BatteryAsset(
                building_id="Building_1",
                action_name="electrical_storage",
                initial_energy_kwh=0.0,
                final_energy_min_kwh=0.0,
                optimistic=model,
                conservative=model,
            ),
        ),
    )

    projected, correction = _project_signed_power_to_battery(
        problem,
        0,
        np.asarray([1.0, 1.0, -1.0]),
    )
    energy = projected * problem.timestep_hours
    soc = 0.0
    states = [soc]
    for value in energy:
        soc += 0.9 * value if value >= 0.0 else value / 0.9
        states.append(soc)

    assert correction > 0.0
    assert min(states) >= -1.0e-10
    assert max(states) <= 1.0 + 1.0e-10
