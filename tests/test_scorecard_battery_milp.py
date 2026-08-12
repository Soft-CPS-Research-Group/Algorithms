from __future__ import annotations

import numpy as np
import pytest

from algorithms.oracles import (
    BatteryAsset,
    BatteryModel,
    PerfectForesightProblem,
    ScorecardShapingOptions,
    SemanticActionSeries,
    SemanticSchedule,
    SolveOptions,
    solve_scorecard_battery_schedule,
)


def _problem() -> PerfectForesightProblem:
    model = BatteryModel(
        capacity_kwh=4.0,
        max_charge_kw=4.0,
        max_discharge_kw=4.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    return PerfectForesightProblem(
        problem_id="scorecard-shaping-test",
        timestep_hours=1.0,
        building_ids=("Building_1",),
        price_eur_per_kwh=np.ones(4, dtype=np.float64),
        base_net_load_kwh=np.asarray([[0.0, 4.0, 0.0, 4.0]], dtype=np.float64),
        batteries=(
            BatteryAsset(
                building_id="Building_1",
                action_name="electrical_storage",
                initial_energy_kwh=2.0,
                final_energy_min_kwh=2.0,
                optimistic=model,
                conservative=model,
            ),
        ),
    )


def test_scorecard_shaping_respects_cost_ceiling_and_smooths_exchange() -> None:
    problem = _problem()
    result = solve_scorecard_battery_schedule(
        problem,
        ScorecardShapingOptions(
            community_cost_limit_eur=8.01,
            ramping_weight=1.0,
            daily_peak_weight=1.0,
            all_time_peak_weight=0.25,
        ),
        SolveOptions(time_limit_seconds=10.0, mip_relative_gap=0.0),
    )

    assert result.solver.has_solution
    assert result.community_cost_eur is not None
    assert result.community_cost_eur <= 8.01 + 1.0e-6
    assert result.mean_absolute_ramp_kwh is not None
    assert result.mean_absolute_ramp_kwh < np.mean(np.abs(np.diff([0.0, 4.0, 0.0, 4.0])))
    assert result.all_time_peak_import_kwh is not None
    assert result.all_time_peak_import_kwh < 4.0
    assert result.schedule is not None
    assert result.schedule.metadata["requires_citylearn_replay"] is True


def test_scorecard_shaping_rejects_empty_physical_objective() -> None:
    with pytest.raises(ValueError, match="physical scorecard weight"):
        ScorecardShapingOptions(
            community_cost_limit_eur=10.0,
            ramping_weight=0.0,
            daily_peak_weight=0.0,
            all_time_peak_weight=0.0,
        )


def test_scorecard_shaping_can_minimize_emissions_under_physical_limits() -> None:
    problem = _problem()
    result = solve_scorecard_battery_schedule(
        problem,
        ScorecardShapingOptions(
            community_cost_limit_eur=8.01,
            ramping_weight=0.0,
            daily_peak_weight=0.0,
            all_time_peak_weight=0.0,
            emissions_weight=1.0,
            mean_absolute_ramp_limit_kwh=4.0,
            mean_daily_peak_import_limit_kwh=4.0,
            all_time_peak_import_limit_kwh=4.0,
            enforce_exclusive_battery_direction=False,
        ),
        SolveOptions(time_limit_seconds=10.0, mip_relative_gap=0.0),
        carbon_intensity_kgco2_per_kwh=[0.0, 1.0, 0.0, 1.0],
    )

    assert result.solver.has_solution
    assert result.community_emissions_kgco2 is not None
    assert result.community_emissions_kgco2 < 8.0
    assert result.mean_absolute_ramp_kwh is not None
    assert result.mean_absolute_ramp_kwh <= 4.0 + 1.0e-6
    assert result.all_time_peak_import_kwh is not None
    assert result.all_time_peak_import_kwh <= 4.0 + 1.0e-6
    assert result.simultaneous_charge_discharge_kwh == pytest.approx(0.0)


def test_scorecard_shaping_can_account_for_gross_member_emissions() -> None:
    model = BatteryModel(
        capacity_kwh=2.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    problem = PerfectForesightProblem(
        problem_id="gross-member-carbon-test",
        timestep_hours=1.0,
        building_ids=("Building_1", "Building_2"),
        price_eur_per_kwh=np.ones(2),
        base_net_load_kwh=np.asarray([[-1.0, 1.0], [1.0, -1.0]]),
        batteries=tuple(
            BatteryAsset(
                building_id=f"Building_{index}",
                action_name="electrical_storage",
                initial_energy_kwh=1.0,
                final_energy_min_kwh=1.0,
                optimistic=model,
                conservative=model,
            )
            for index in (1, 2)
        ),
    )
    result = solve_scorecard_battery_schedule(
        problem,
        ScorecardShapingOptions(
            community_cost_limit_eur=0.01,
            ramping_weight=0.0,
            daily_peak_weight=0.0,
            all_time_peak_weight=0.0,
            emissions_weight=1.0,
            emissions_accounting="gross_member_import",
            enforce_exclusive_battery_direction=False,
        ),
        SolveOptions(time_limit_seconds=10.0, mip_relative_gap=0.0),
        carbon_intensity_kgco2_per_kwh=[1.0, 1.0],
    )

    assert result.solver.has_solution
    assert result.gross_member_import_kwh == pytest.approx(0.0, abs=1.0e-6)
    assert result.community_emissions_kgco2 == pytest.approx(0.0, abs=1.0e-6)


def test_scorecard_shaping_accepts_a_valid_physical_warm_start() -> None:
    problem = _problem()
    initial = SemanticSchedule(
        problem_id="scorecard-warm-start",
        horizon=4,
        timestep_hours=1.0,
        series=(
            SemanticActionSeries(
                building_id="Building_1",
                action_name="electrical_storage",
                values=(0.0, 0.0, 0.0, 0.0),
            ),
        ),
    )

    result = solve_scorecard_battery_schedule(
        problem,
        ScorecardShapingOptions(
            community_cost_limit_eur=8.01,
            ramping_weight=1.0,
            daily_peak_weight=1.0,
            all_time_peak_weight=0.25,
        ),
        SolveOptions(time_limit_seconds=10.0, mip_relative_gap=0.0),
        initial_schedule=initial,
    )

    assert result.solver.has_solution
    assert result.schedule is not None
