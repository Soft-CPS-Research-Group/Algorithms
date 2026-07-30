from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algorithms.oracles import solve_conservative_schedule
from algorithms.oracles.citylearn_fixed_service import (
    build_fixed_service_battery_problem,
    expand_aggregated_battery_schedule,
)


def _write_csv(path: Path, **columns) -> None:
    pd.DataFrame(columns).to_csv(path, index=False)


def test_builds_conditional_problem_and_reconstructs_source_cost(tmp_path: Path) -> None:
    schema = {
        "seconds_per_time_step": 900,
        "buildings": {
            "Building_1": {
                "include": True,
                "electrical_storage": {
                    "autosize": False,
                    "attributes": {
                        "capacity": 6.4,
                        "nominal_power": 5.0,
                        "efficiency": 0.9,
                        "capacity_loss_coefficient": 1.0e-5,
                    },
                },
            },
            "Building_2": {
                "include": True,
                "electrical_storage": {
                    "autosize": False,
                    "attributes": {
                        "capacity": 4.0,
                        "nominal_power": 2.0,
                        "efficiency": 0.81,
                        "initial_soc": 0.25,
                    },
                },
            },
        },
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    timestamps = ["2024-01-01T00:00:00", "2024-01-01T00:15:00"]
    _write_csv(
        tmp_path / "exported_data_pricing_ep1.csv",
        timestamp=timestamps,
        **{"electricity_pricing-$/kWh": [1.0, 2.0]},
    )
    _write_csv(
        tmp_path / "exported_data_building_1_ep1.csv",
        timestamp=timestamps,
        **{"Net Electricity Consumption-kWh": [2.0, 0.0]},
    )
    _write_csv(
        tmp_path / "exported_data_building_1_battery_ep1.csv",
        timestamp=timestamps,
        **{"Battery (Dis)Charge-kWh": [0.5, -0.25]},
    )
    _write_csv(
        tmp_path / "exported_data_building_2_ep1.csv",
        timestamp=timestamps,
        **{"Net Electricity Consumption-kWh": [1.0, 1.0]},
    )
    _write_csv(
        tmp_path / "exported_data_building_2_battery_ep1.csv",
        timestamp=timestamps,
        **{"Battery (Dis)Charge-kWh": [0.0, 0.0]},
    )

    built = build_fixed_service_battery_problem(
        schema_path=schema_path,
        simulation_data_directory=tmp_path,
        problem_id="fixture",
    )

    assert built.problem.building_ids == ("Building_1", "Building_2")
    assert built.problem.timestep_hours == pytest.approx(0.25)
    assert built.problem.base_net_load_kwh == pytest.approx(
        np.asarray([[1.5, 0.25], [1.0, 1.0]])
    )
    assert built.diagnostics.source_policy_cost_reconstructed_eur == pytest.approx(5.0)
    assert built.diagnostics.fixed_service_without_stationary_battery_cost_eur == pytest.approx(5.0)
    assert built.diagnostics.source_stationary_battery_throughput_kwh == pytest.approx(0.75)
    assert built.problem.metadata["global_optimum_claim"] is False
    assert built.problem.metadata["requires_citylearn_replay"] is True
    # Models differ by capacity/power/initial SOC, so this fixture forms two
    # exact groups rather than silently merging non-equivalent devices.
    assert len(built.problem.batteries) == 2
    assert built.problem.batteries[0].conservative.max_charge_kw == pytest.approx(1.0)
    assert built.problem.batteries[1].initial_energy_kwh == pytest.approx(1.0)
    assert built.problem.batteries[1].final_energy_min_kwh == pytest.approx(1.0)


def test_identical_batteries_aggregate_and_expand_losslessly(tmp_path: Path) -> None:
    schema = {
        "seconds_per_time_step": 3600,
        "buildings": {
            name: {
                "include": True,
                "electrical_storage": {
                    "autosize": False,
                    "attributes": {
                        "capacity": 1.0,
                        "nominal_power": 1.0,
                        "efficiency": 0.9,
                    },
                },
            }
            for name in ("Building_1", "Building_2")
        },
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    timestamps = ["t0", "t1"]
    _write_csv(
        tmp_path / "exported_data_pricing_ep1.csv",
        timestamp=timestamps,
        **{"electricity_pricing-$/kWh": [1.0, 10.0]},
    )
    for index in (1, 2):
        _write_csv(
            tmp_path / f"exported_data_building_{index}_ep1.csv",
            timestamp=timestamps,
            **{"Net Electricity Consumption-kWh": [0.0, 1.0]},
        )
        _write_csv(
            tmp_path / f"exported_data_building_{index}_battery_ep1.csv",
            timestamp=timestamps,
            **{"Battery (Dis)Charge-kWh": [0.0, 0.0]},
        )

    built = build_fixed_service_battery_problem(
        schema_path=schema_path,
        simulation_data_directory=tmp_path,
        problem_id="aggregate",
    )

    assert len(built.problem.batteries) == 1
    assert built.problem.batteries[0].conservative.capacity_kwh == pytest.approx(1.98)
    result = solve_conservative_schedule(built.problem)
    assert result.schedule is not None
    expanded = expand_aggregated_battery_schedule(result.schedule, built.problem.metadata)
    assert [(item.building_id, item.action_name) for item in expanded.series] == [
        ("Building_1", "electrical_storage"),
        ("Building_2", "electrical_storage"),
    ]
    assert np.asarray(expanded.series[0].values) == pytest.approx(
        np.asarray(result.schedule.series[0].values) * 0.5
    )


def test_rejects_timestamp_mismatch(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "seconds_per_time_step": 900,
                "buildings": {"Building_1": {"include": True}},
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        tmp_path / "exported_data_pricing_ep1.csv",
        timestamp=["t0"],
        **{"electricity_pricing-$/kWh": [1.0]},
    )
    _write_csv(
        tmp_path / "exported_data_building_1_ep1.csv",
        timestamp=["different"],
        **{"Net Electricity Consumption-kWh": [1.0]},
    )

    with pytest.raises(ValueError, match="Timestamp mismatch"):
        build_fixed_service_battery_problem(
            schema_path=schema_path,
            simulation_data_directory=tmp_path,
            problem_id="bad",
        )
