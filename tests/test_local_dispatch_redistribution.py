from __future__ import annotations

import numpy as np
import pytest

from algorithms.oracles.local_dispatch_redistribution import (
    redistribute_equivalent_battery_schedule,
)
from algorithms.oracles.perfect_foresight_milp import (
    BatteryAsset,
    BatteryModel,
    PerfectForesightProblem,
    SemanticActionSeries,
    SemanticSchedule,
)


def test_local_dispatch_preserves_district_power_and_reduces_gross_import() -> None:
    model = BatteryModel(
        capacity_kwh=2.0,
        max_charge_kw=2.0,
        max_discharge_kw=2.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    batteries = tuple(
        BatteryAsset(
            building_id=f"Building_{index}",
            action_name="electrical_storage",
            initial_energy_kwh=0.0,
            final_energy_min_kwh=0.0,
            optimistic=model,
            conservative=model,
        )
        for index in (1, 2)
    )
    problem = PerfectForesightProblem(
        problem_id="redistribution-test",
        timestep_hours=1.0,
        building_ids=("Building_1", "Building_2"),
        base_net_load_kwh=np.asarray([[-1.0, 1.0], [1.0, -1.0]]),
        price_eur_per_kwh=np.ones(2),
        batteries=batteries,
        metadata={},
    )
    source = SemanticSchedule(
        problem_id="aggregate-expanded",
        horizon=2,
        timestep_hours=1.0,
        series=(
            SemanticActionSeries(
                building_id="Building_1",
                action_name="electrical_storage",
                values=(0.5, -0.5),
            ),
            SemanticActionSeries(
                building_id="Building_2",
                action_name="electrical_storage",
                values=(0.5, -0.5),
            ),
        ),
        metadata={},
    )

    result = redistribute_equivalent_battery_schedule(
        problem,
        source,
        [1.0, 1.0],
        window_steps=2,
    )

    source_power = np.stack([series.values for series in source.series])
    result_power = np.stack([series.values for series in result.schedule.series])
    np.testing.assert_allclose(result_power.sum(axis=0), source_power.sum(axis=0))
    assert result.redistributed_gross_import_kwh < result.source_gross_import_kwh
    assert result.redistributed_emissions_kgco2 < result.source_emissions_kgco2


def test_local_dispatch_accepts_tiny_empty_soc_solver_residual() -> None:
    model = BatteryModel(
        capacity_kwh=1.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    batteries = tuple(
        BatteryAsset(
            building_id=f"Building_{index}",
            action_name="electrical_storage",
            initial_energy_kwh=0.0,
            final_energy_min_kwh=0.0,
            optimistic=model,
            conservative=model,
        )
        for index in (1, 2)
    )
    problem = PerfectForesightProblem(
        problem_id="redistribution-roundoff-test",
        timestep_hours=1.0,
        building_ids=("Building_1", "Building_2"),
        base_net_load_kwh=np.ones((2, 2)),
        price_eur_per_kwh=np.ones(2),
        batteries=batteries,
        metadata={},
    )
    residual = 5.0e-8
    source = SemanticSchedule(
        problem_id="solver-residual",
        horizon=2,
        timestep_hours=1.0,
        series=tuple(
            SemanticActionSeries(
                building_id=f"Building_{index}",
                action_name="electrical_storage",
                values=(1.0, -1.0 - residual),
            )
            for index in (1, 2)
        ),
        metadata={},
    )

    result = redistribute_equivalent_battery_schedule(
        problem,
        source,
        [1.0, 1.0],
        window_steps=1,
    )

    result_power = np.stack([series.values for series in result.schedule.series])
    source_power = np.stack([series.values for series in source.series])
    np.testing.assert_allclose(result_power.sum(axis=0), source_power.sum(axis=0))


def test_local_dispatch_uses_physical_counterflow_without_changing_district_power() -> None:
    model = BatteryModel(
        capacity_kwh=2.0,
        max_charge_kw=1.0,
        max_discharge_kw=1.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
    )
    batteries = tuple(
        BatteryAsset(
            building_id=f"Building_{index}",
            action_name="electrical_storage",
            initial_energy_kwh=1.0,
            final_energy_min_kwh=1.0,
            optimistic=model,
            conservative=model,
        )
        for index in (1, 2)
    )
    problem = PerfectForesightProblem(
        problem_id="counterflow-test",
        timestep_hours=1.0,
        building_ids=("Building_1", "Building_2"),
        base_net_load_kwh=np.asarray([[-1.0, 1.0], [1.0, -1.0]]),
        price_eur_per_kwh=np.ones(2),
        batteries=batteries,
        metadata={},
    )
    source = SemanticSchedule(
        problem_id="zero-district-battery-power",
        horizon=2,
        timestep_hours=1.0,
        series=tuple(
            SemanticActionSeries(
                building_id=f"Building_{index}",
                action_name="electrical_storage",
                values=(0.0, 0.0),
            )
            for index in (1, 2)
        ),
        metadata={},
    )

    result = redistribute_equivalent_battery_schedule(
        problem,
        source,
        [1.0, 1.0],
        window_steps=2,
    )

    actions = np.stack([series.values for series in result.schedule.series])
    np.testing.assert_allclose(actions.sum(axis=0), 0.0, atol=1.0e-7)
    assert result.redistributed_gross_import_kwh == pytest.approx(0.0, abs=1.0e-6)
    assert np.max(np.abs(actions)) > 0.9
    assert result.source_battery_throughput_kwh == pytest.approx(0.0)
    assert result.redistributed_battery_throughput_kwh > 3.9
