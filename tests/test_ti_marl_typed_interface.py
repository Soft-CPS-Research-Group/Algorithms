from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from algorithms.ti_marl.compiler import TypedInterfaceCompiler
from algorithms.ti_marl.contracts.interface_definition import (
    InterfaceRegistry,
    TypedAgentInterface,
)
from scripts.generate_typed_interfaces import generate, write_generated
from tests.ti_marl_fixtures import (
    entity_payload,
    entity_specs,
    typed_interface_payload,
    write_typed_interfaces,
)


def _write(path: Path, payload) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_per_agent_interface_is_human_readable_and_compiles(tmp_path):
    directory = write_typed_interfaces(tmp_path / "interfaces", ("Building_1", "Building_2"))
    definition = TypedAgentInterface.load(directory / "Building_1.yaml")
    assert definition.agent_id == "Building_1"
    assert definition.role == "prosumer"
    assert any(sensor.sensor_id == "community" for sensor in definition.sensors)
    charger = next(sensor for sensor in definition.sensors if sensor.sensor_id == "charger_1")
    assert {item.channel_id for item in charger.observations} >= {
        "connection",
        "ev_state",
        "capability",
    }
    compiler = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interfaces_dir=directory,
    )
    compiler.attach_entity_specs(entity_specs(), seconds_per_time_step=900)
    snapshot = compiler.compile(entity_payload())
    assert snapshot.agent_ids == ("Building_1", "Building_2")
    assert {group.group_type for group in snapshot.action_groups} == {
        "stationary_storage",
        "ev_session",
        "deferrable",
    }
    assert all(part.sensor_id and part.channel_id for part in snapshot.observation_parts)


def test_retired_global_document_is_rejected_without_runtime_compatibility(tmp_path):
    old = _write(
        tmp_path / "old.yaml",
        {"version": "typed_interface_v1", "contract_version": "ti_marl_v1"},
    )
    with pytest.raises(ValueError, match="retired"):
        TypedAgentInterface.load(old)


def test_dependencies_are_exact_and_must_cover_every_non_nominal_health_state(tmp_path):
    payload = typed_interface_payload("Building_1")
    dependency = payload["actuators"]["charger_1"]["actions"]["charge"]["dependencies"]
    dependency["charger_1.connection.connected_state"].pop("UNKNOWN")
    path = _write(tmp_path / "Building_1.yaml", payload)
    with pytest.raises(ValueError, match="must declare outcomes"):
        TypedAgentInterface.load(path)

    payload = typed_interface_payload("Building_1")
    payload["actuators"]["charger_1"]["actions"]["charge"]["dependencies"] = {
        "charger_1.ev_state.does_not_exist": {
            state: "safe_idle"
            for state in ("DEGRADED", "STALE", "MISSING", "FAILED", "UNKNOWN")
        }
    }
    path = _write(tmp_path / "Building_1.yaml", payload)
    with pytest.raises(ValueError, match="unknown observation"):
        TypedAgentInterface.load(path)


def test_excluded_observation_requires_a_reason(tmp_path):
    payload = typed_interface_payload("Building_1")
    observation = payload["sensors"]["self"]["channels"]["energy"]["observations"]["net_power_kw"]
    observation.update({"use": "excluded", "policy_input": False})
    path = _write(tmp_path / "Building_1.yaml", payload)
    with pytest.raises(ValueError, match="requires a justification"):
        TypedAgentInterface.load(path)


def test_registry_reload_is_atomic_and_reports_join_leave(tmp_path):
    directory = write_typed_interfaces(tmp_path / "interfaces", ("Building_1", "Building_2"))
    registry = InterfaceRegistry(directory)
    original_hash = registry.registry_hash
    invalid = typed_interface_payload("Building_2")
    invalid["agent"]["role"] = "invalid-role"
    _write(directory / "Building_2.yaml", invalid)
    with pytest.raises(ValueError, match="agent.role"):
        registry.reload_interfaces()
    assert registry.registry_hash == original_hash
    assert registry.agent_ids == ("Building_1", "Building_2")

    _write(directory / "Building_2.yaml", typed_interface_payload("Building_2"))
    _write(directory / "Building_3.yaml", typed_interface_payload("Building_3"))
    (directory / "Building_1.yaml").unlink()
    delta = registry.reload_interfaces()
    assert delta.added_agent_ids == ("Building_3",)
    assert delta.removed_agent_ids == ("Building_1",)
    assert registry.agent_ids == ("Building_2", "Building_3")


def test_registry_rejects_a_mixed_generation_during_concurrent_edit(
    tmp_path,
    monkeypatch,
):
    directory = write_typed_interfaces(
        tmp_path / "interfaces",
        ("Building_1", "Building_2"),
    )
    registry = InterfaceRegistry(directory)
    original_hash = registry.registry_hash
    original_fingerprint = registry._directory_fingerprint
    calls = 0

    def racing_fingerprint(files):
        nonlocal calls
        calls += 1
        if calls == 2:
            changed = typed_interface_payload("Building_1")
            changed["description"] = "concurrent replacement"
            _write(directory / "Building_1.yaml", changed)
        return original_fingerprint(files)

    monkeypatch.setattr(registry, "_directory_fingerprint", racing_fingerprint)
    with pytest.raises(RuntimeError, match="changed during atomic reload"):
        registry.reload_interfaces()
    assert registry.registry_hash == original_hash


def test_compatibility_shape_does_not_depend_on_concrete_agent_id(tmp_path):
    first_payload = typed_interface_payload("Building_1")
    second_payload = deepcopy(first_payload)
    second_payload["agent"]["id"] = "Building_99"
    first = TypedAgentInterface.load(_write(tmp_path / "Building_1.yaml", first_payload))
    second = TypedAgentInterface.load(_write(tmp_path / "Building_99.yaml", second_payload))
    assert first.compatibility_shape == second.compatibility_shape


def test_generator_classifies_every_supported_simulator_field(tmp_path):
    specs = entity_specs()
    interfaces, coverage = generate(specs, entity_payload())
    expected = sum(
        len(table.get("features", []))
        for entity_type, table in specs["tables"].items()
        if entity_type in {
            "district",
            "building",
            "storage",
            "charger",
            "ev",
            "deferrable_appliance",
            "pv",
        }
    )
    assert len(coverage) == expected
    assert all(row["classification"] for row in coverage)
    output = tmp_path / "generated"
    write_generated(output, interfaces, coverage, source="unit fixture")
    registry = InterfaceRegistry(output)
    assert registry.agent_ids == ("Building_1", "Building_2")
    assert (output / "observation_coverage.csv").is_file()
    assert (output / "interface_manifest.json").is_file()


def test_generator_preserves_charger_minimum_and_maximum_power_bounds():
    schema = {
        "buildings": {
            "Building_1": {
                "chargers": {
                    "charger_1": {
                        "attributes": {
                            "nominal_power": 11.0,
                            "min_charging_power": 1.4,
                            "max_charging_power": 11.0,
                            "min_discharging_power": 0.8,
                            "max_discharging_power": 7.2,
                        }
                    }
                }
            }
        }
    }

    interfaces, _coverage = generate(entity_specs(), entity_payload(), schema)
    actions = interfaces["Building_1"]["actuators"]["charger_1"]["actions"]

    assert actions["charge"]["parameter"]["bounds"] == [1.4, 11.0]
    assert actions["discharge"]["parameter"]["bounds"] == [0.8, 7.2]


def test_public_yaml_contains_no_simulator_contract_section(tmp_path):
    path = _write(tmp_path / "Building_1.yaml", typed_interface_payload("Building_1"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "simulator" not in raw
    assert "entity_v1" not in path.read_text(encoding="utf-8")


def test_unknown_unit_is_rejected_before_control(tmp_path):
    payload = typed_interface_payload("Building_1")
    payload["sensors"]["self"]["channels"]["energy"]["observations"]["net_power_kw"]["unit"] = "mystery"
    path = _write(tmp_path / "Building_1.yaml", payload)
    with pytest.raises(ValueError, match="unknown unit"):
        TypedAgentInterface.load(path)


def test_unknown_profile_and_constraint_unit_are_rejected(tmp_path):
    payload = typed_interface_payload("Building_1")
    payload["sensors"]["self"]["profile"] = "unregistered_meter_v9"
    path = _write(tmp_path / "Building_1.yaml", payload)
    with pytest.raises(ValueError, match="Unknown TI-MARL profile"):
        TypedAgentInterface.load(path)

    payload = typed_interface_payload("Building_1")
    payload["constraints"]["grid_import"]["unit"] = "mystery"
    path = _write(tmp_path / "Building_1.yaml", payload)
    with pytest.raises(ValueError, match="constraint.*unknown unit"):
        TypedAgentInterface.load(path)


def test_yaml_is_not_parsed_in_the_decision_hot_path(tmp_path, monkeypatch):
    directory = write_typed_interfaces(tmp_path / "interfaces", ("Building_1", "Building_2"))
    compiler = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interfaces_dir=directory,
        interface_polling=False,
    )
    compiler.attach_entity_specs(entity_specs(), seconds_per_time_step=900)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("YAML parser entered the decision hot path")

    monkeypatch.setattr(yaml, "safe_load", fail_if_called)
    compiler.compile(entity_payload())
    compiler.compile(entity_payload(time_step=1))
    assert compiler.structure_recompilations == 1
