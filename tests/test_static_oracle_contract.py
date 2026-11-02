"""Static MILPs must reject mechanisms absent from their constraints."""

import pytest

from algorithms.oracles.citylearn_contract import validate_static_oracle_schema
from algorithms.oracles.citylearn_total_energy import _phase, _reserved_limit
from algorithms.oracles.total_energy_milp import ElectricalService


def test_unassigned_appliance_inherits_single_phase_connection():
    service = ElectricalService(building_id="A", default_split="L2")
    assert service.fractions(_phase(None, default_split="L2")) == {"L1": 0.0, "L2": 1.0, "L3": 0.0}
    assert service.fractions(_phase("all_phases")) == pytest.approx(
        {"L1": 1 / 3, "L2": 1 / 3, "L3": 1 / 3}
    )


def test_reserve_keeps_disconnected_phase_and_forbidden_export_closed():
    assert _reserved_limit(0, reserve_kw=0.1, name="unused phase") == 0
    assert _reserved_limit(6.9, reserve_kw=0.1, name="connected phase") == pytest.approx(6.8)
    assert _reserved_limit(None, reserve_kw=0.1, name="unbounded") is None
    with pytest.raises(ValueError, match="exceeds"):
        _reserved_limit(0.05, reserve_kw=0.1, name="invalid reserve")


@pytest.mark.parametrize(
    "schema",
    [
        {"topology_mode": "dynamic"},
        {"topology_events": [{"time_step": 4}]},
        {"robustness": {"enabled": True}},
        {"buildings": {"A": {"power_outage": {"simulate_power_outage": True}}}},
    ],
)
def test_unmodeled_runtime_mechanisms_are_not_silently_ignored(schema):
    with pytest.raises(ValueError, match="MILP"):
        validate_static_oracle_schema(schema)


def test_static_schema_with_disabled_outage_is_supported():
    validate_static_oracle_schema(
        {
            "buildings": {
                "A": {"power_outage": {"simulate_power_outage": False}},
            }
        }
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"actions": {"electrical_storage": {"active": False}}},
        {
            "buildings": {
                "A": {
                    "electrical_storage": {"attributes": {}},
                    "inactive_actions": ["electrical_storage"],
                }
            }
        },
    ],
)
def test_optimizer_rejects_equipment_it_cannot_control(extra):
    schema = {"buildings": {"A": {"electrical_storage": {"attributes": {}}}}, **extra}
    with pytest.raises(ValueError, match="requires enabled"):
        validate_static_oracle_schema(schema)


@pytest.mark.parametrize("family", ["home", "total"])
def test_solved_schedule_keeps_idle_equipment_for_replay(family):
    import numpy as np

    from algorithms.oracles.citylearn_contract import action_inventory
    from algorithms.oracles.total_energy_milp import TotalEnergyProblem, solve_total_energy_schedule
    from algorithms.oracles.total_home_milp import TotalHomeProblem, solve_total_home_milp

    inventory = action_inventory(
        {
            "buildings": {
                "A": {
                    "chargers": {"idle": {}},
                    "deferrable_appliances": {"washer": {}},
                }
            }
        },
        ["A"],
    )
    common = dict(
        problem_id="idle-devices",
        timestep_hours=0.25,
        price_eur_per_kwh=np.ones(2),
        metadata={"action_inventory": inventory},
    )
    if family == "home":
        result = solve_total_home_milp(
            TotalHomeProblem(**common, building_id="A", base_net_load_kwh=np.ones(2))
        )
    else:
        result = solve_total_energy_schedule(
            TotalEnergyProblem(**common, building_ids=("A",), base_net_load_kwh=np.ones((1, 2)))
        )
    assert result.schedule is not None
    assert {item.action_name: item.values for item in result.schedule.series} == {
        "electric_vehicle_storage_idle": (0.0, 0.0),
        "deferrable_appliance_washer": (0.0, 0.0),
    }
