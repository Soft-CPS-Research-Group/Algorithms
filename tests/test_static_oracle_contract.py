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


@pytest.mark.parametrize("schema", [
    {"topology_mode": "dynamic"},
    {"topology_events": [{"time_step": 4}]},
    {"robustness": {"enabled": True}},
    {"buildings": {"A": {"power_outage": {"simulate_power_outage": True}}}},
])
def test_unmodeled_runtime_mechanisms_are_not_silently_ignored(schema):
    with pytest.raises(ValueError, match="MILP"):
        validate_static_oracle_schema(schema)


def test_static_schema_with_disabled_outage_is_supported():
    validate_static_oracle_schema({"buildings": {
        "A": {"power_outage": {"simulate_power_outage": False}},
    }})
