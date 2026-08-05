from __future__ import annotations

import json

import numpy as np
import pytest

from algorithms.oracles import (
    BatteryAsset,
    BatteryModel,
    PerfectForesightProblem,
    SemanticSchedule,
    solve_bounded_oracle,
    solve_conservative_schedule,
)


def _battery(
    *,
    optimistic: BatteryModel | None = None,
    conservative: BatteryModel | None = None,
    initial_energy_kwh: float = 0.0,
    final_energy_min_kwh: float = 0.0,
) -> BatteryAsset:
    unit = BatteryModel(
        capacity_kwh=1.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    return BatteryAsset(
        building_id="Building_1",
        action_name="electrical_storage",
        initial_energy_kwh=initial_energy_kwh,
        final_energy_min_kwh=final_energy_min_kwh,
        optimistic=optimistic or unit,
        conservative=conservative or unit,
    )


def _problem(battery: BatteryAsset, *, prices=(1.0, 10.0), base=((1.0, 1.0),)):
    return PerfectForesightProblem(
        problem_id="toy",
        timestep_hours=1.0,
        building_ids=("Building_1",),
        price_eur_per_kwh=np.asarray(prices),
        base_net_load_kwh=np.asarray(base),
        batteries=(battery,),
        metadata={"dataset": "analytic-fixture"},
    )


def test_two_step_problem_matches_analytic_optimum_and_semantic_schedule():
    result = solve_bounded_oracle(_problem(_battery()))

    assert result.certificate_valid is True
    assert result.certified_lower_bound_eur == pytest.approx(2.0)
    assert result.model_feasible_upper_bound_eur == pytest.approx(2.0)
    assert result.absolute_gap_eur == pytest.approx(0.0)
    assert result.relative_gap == pytest.approx(0.0)

    schedule = result.conservative.schedule
    assert schedule is not None
    assert schedule.series[0].building_id == "Building_1"
    assert schedule.series[0].action_name == "electrical_storage"
    assert schedule.series[0].unit == "kW"
    assert schedule.series[0].values == pytest.approx((1.0, -1.0))


def test_optimistic_relaxation_is_below_conservative_feasible_cost():
    optimistic = BatteryModel(1.0, 2.0, 1.0, 1.0, 1.0)
    conservative = BatteryModel(1.0, 2.0, 1.0, 0.8, 0.8)
    problem = _problem(
        _battery(optimistic=optimistic, conservative=conservative),
        prices=(1.0, 10.0),
        base=((0.0, 1.0),),
    )

    result = solve_bounded_oracle(problem)

    assert result.certificate_valid is True
    assert result.certified_lower_bound_eur == pytest.approx(1.0)
    assert result.model_feasible_upper_bound_eur == pytest.approx(3.25)
    assert result.certified_lower_bound_eur <= result.model_feasible_upper_bound_eur
    assert result.absolute_gap_eur == pytest.approx(2.25)


def test_conservative_formulation_reports_infeasibility_without_schedule():
    optimistic = BatteryModel(1.0, 1.0, 0.0, 1.0, 1.0)
    conservative = BatteryModel(1.0, 0.0, 0.0, 1.0, 1.0)
    battery = _battery(
        optimistic=optimistic,
        conservative=conservative,
        final_energy_min_kwh=1.0,
    )
    problem = _problem(battery, prices=(1.0,), base=((0.0,),))

    result = solve_bounded_oracle(problem)

    assert result.lower.solver.optimal is True
    assert result.conservative.solver.status == "infeasible"
    assert result.conservative.solver.has_solution is False
    assert result.conservative.schedule is None
    assert result.model_feasible_upper_bound_eur is None
    assert result.certificate_valid is False


def test_problem_rejects_invalid_dimensions_and_non_dominating_relaxation():
    with pytest.raises(ValueError, match="must have shape"):
        PerfectForesightProblem(
            problem_id="bad-shape",
            timestep_hours=0.25,
            building_ids=("B1", "B2"),
            price_eur_per_kwh=np.ones(3),
            base_net_load_kwh=np.ones((1, 3)),
        )

    with pytest.raises(ValueError, match="optimistic capacity_kwh"):
        _battery(
            optimistic=BatteryModel(0.5, 1.0, 1.0),
            conservative=BatteryModel(1.0, 1.0, 1.0),
        )


def test_problem_and_semantic_schedule_json_round_trip():
    problem = _problem(_battery())
    restored_problem = PerfectForesightProblem.from_dict(
        json.loads(json.dumps(problem.to_dict()))
    )
    result = solve_conservative_schedule(restored_problem)

    assert result.solver.optimal is True
    assert result.schedule is not None
    restored_schedule = SemanticSchedule.from_json(result.schedule.to_json())
    assert restored_schedule == result.schedule
    assert restored_schedule.metadata["requires_citylearn_replay"] is True


def test_no_battery_problem_reduces_to_positive_district_import_cost():
    problem = PerfectForesightProblem(
        problem_id="district-balance",
        timestep_hours=0.25,
        building_ids=("Importer", "Exporter"),
        price_eur_per_kwh=np.asarray([2.0, 4.0]),
        base_net_load_kwh=np.asarray([[3.0, 1.0], [-1.0, -2.0]]),
    )

    result = solve_bounded_oracle(problem)

    assert result.certificate_valid is True
    assert result.model_feasible_upper_bound_eur == pytest.approx(4.0)
    assert result.conservative.grid_import_kwh == pytest.approx((2.0, 0.0))
    assert result.conservative.schedule is not None
    assert result.conservative.schedule.series == ()
