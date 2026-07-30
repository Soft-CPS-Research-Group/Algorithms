from __future__ import annotations

import numpy as np
import pytest

from algorithms.oracles import (
    BatteryAsset,
    BatteryModel,
    PerfectForesightProblem,
    SemanticActionSeries,
    SemanticSchedule,
    combine_individual_building_schedules,
    solve_individual_building_oracles,
    split_individual_building_problems,
)


def _battery(building_id: str, action_name: str = "electrical_storage") -> BatteryAsset:
    model = BatteryModel(
        capacity_kwh=1.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
    )
    return BatteryAsset(
        building_id=building_id,
        action_name=action_name,
        initial_energy_kwh=0.0,
        final_energy_min_kwh=0.0,
        optimistic=model,
        conservative=model,
    )


def _problem(*, second_base: tuple[float, float] = (1.0, 1.0)) -> PerfectForesightProblem:
    # Deliberately order batteries differently from buildings to verify semantic
    # rather than positional schedule recombination.
    return PerfectForesightProblem(
        problem_id="two-homes",
        timestep_hours=1.0,
        building_ids=("Home_A", "Home_B"),
        price_eur_per_kwh=np.asarray([1.0, 10.0]),
        base_net_load_kwh=np.asarray([[1.0, 1.0], second_base]),
        batteries=(_battery("Home_B", "battery_b"), _battery("Home_A", "battery_a")),
        metadata={"scope": "conditional_fixed_service_stationary_battery"},
    )


def test_split_preserves_each_home_row_assets_and_common_inputs():
    source = _problem(second_base=(3.0, -2.0))

    home_a, home_b = split_individual_building_problems(source)

    assert home_a.building_ids == ("Home_A",)
    assert home_b.building_ids == ("Home_B",)
    assert home_a.base_net_load_kwh.tolist() == [[1.0, 1.0]]
    assert home_b.base_net_load_kwh.tolist() == [[3.0, -2.0]]
    assert [item.semantic_key for item in home_a.batteries] == [("Home_A", "battery_a")]
    assert [item.semantic_key for item in home_b.batteries] == [("Home_B", "battery_b")]
    assert home_a.price_eur_per_kwh.tolist() == source.price_eur_per_kwh.tolist()
    assert home_b.timestep_hours == source.timestep_hours
    assert home_a.metadata["scope"] == "individual_building_fixed_service_battery_only"
    assert home_a.metadata["global_optimum_claim"] is False


def test_individual_solves_report_per_home_bounds_and_combine_semantic_schedule():
    source = _problem()

    result = solve_individual_building_oracles(source)

    assert result.certificate_valid is True
    assert [item.building_id for item in result.buildings] == ["Home_A", "Home_B"]
    assert [item.result.model_feasible_upper_bound_eur for item in result.buildings] == pytest.approx(
        [2.0, 2.0]
    )
    assert result.certified_lower_bound_eur == pytest.approx(4.0)
    assert result.model_feasible_upper_bound_eur == pytest.approx(4.0)
    assert result.absolute_gap_eur == pytest.approx(0.0)
    assert result.combined_grid_import_kwh == pytest.approx((4.0, 0.0))

    schedule = result.combined_schedule
    assert schedule is not None
    assert [(item.building_id, item.action_name) for item in schedule.series] == [
        ("Home_B", "battery_b"),
        ("Home_A", "battery_a"),
    ]
    assert schedule.series[0].values == pytest.approx((1.0, -1.0))
    assert schedule.series[1].values == pytest.approx((1.0, -1.0))
    assert schedule.metadata["global_optimum_claim"] is False
    assert "not full-home or global community optima" in result.guarantee


def test_changing_one_home_cannot_change_the_other_home_solution():
    first = solve_individual_building_oracles(_problem(second_base=(1.0, 1.0)))
    changed = solve_individual_building_oracles(_problem(second_base=(0.0, 100.0)))

    first_home_a = first.buildings[0].result
    changed_home_a = changed.buildings[0].result
    assert changed_home_a.model_feasible_upper_bound_eur == pytest.approx(
        first_home_a.model_feasible_upper_bound_eur
    )
    assert changed_home_a.conservative.schedule == first_home_a.conservative.schedule


def test_individual_objective_does_not_net_exports_between_homes():
    source = PerfectForesightProblem(
        problem_id="opposite-meters",
        timestep_hours=1.0,
        building_ids=("Home_A", "Home_B"),
        price_eur_per_kwh=np.asarray([1.0, 1.0]),
        base_net_load_kwh=np.asarray([[2.0, -2.0], [-2.0, 2.0]]),
    )

    result = solve_individual_building_oracles(source)

    assert [item.result.model_feasible_upper_bound_eur for item in result.buildings] == pytest.approx(
        [2.0, 2.0]
    )
    assert result.model_feasible_upper_bound_eur == pytest.approx(4.0)
    assert result.combined_grid_import_kwh == pytest.approx((2.0, 2.0))
    # A district meter would see [0, 0].  The individual oracle deliberately
    # does not use that community netting and therefore keeps the four-euro cost.


def test_combiner_rejects_missing_or_renamed_semantic_actions():
    source = _problem()
    wrong = SemanticSchedule(
        problem_id="wrong",
        horizon=2,
        timestep_hours=1.0,
        series=(
            SemanticActionSeries(
                building_id="Home_A",
                action_name="renamed_battery",
                values=(0.0, 0.0),
            ),
        ),
    )

    with pytest.raises(ValueError, match="Unexpected semantic action"):
        combine_individual_building_schedules(source, (wrong,))


def test_split_rejects_fixed_service_aggregate_assets():
    source = _problem()
    aggregated_metadata = {
        **source.metadata,
        "physical_battery_count": 2,
        "oracle_battery_group_count": 1,
        "battery_groups": [
            {
                "oracle_building_id": "Home_A",
                "oracle_action_name": "oracle_group_01_electrical_storage",
                "members": [
                    {"building_id": "Home_A", "action_name": "battery_a"},
                    {"building_id": "Home_B", "action_name": "battery_b"},
                ],
            }
        ],
    }
    aggregated = PerfectForesightProblem(
        problem_id=source.problem_id,
        timestep_hours=source.timestep_hours,
        building_ids=source.building_ids,
        price_eur_per_kwh=source.price_eur_per_kwh,
        base_net_load_kwh=source.base_net_load_kwh,
        batteries=(_battery("Home_A", "oracle_group_01_electrical_storage"),),
        metadata=aggregated_metadata,
    )

    with pytest.raises(ValueError, match="non-aggregated problem"):
        split_individual_building_problems(aggregated)
