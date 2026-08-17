from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from algorithms.ti_marl.compiler import TypedInterfaceCompiler
from algorithms.ti_marl.contracts.interface_definition import TypedInterfaceDefinition
from scripts.generate_typed_interface import generate
from tests.ti_marl_fixtures import entity_payload, entity_specs


INTERFACE = Path("configs/ti_marl/typed_interface_v1.yaml")


def _write_yaml(path: Path, payload) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def test_single_file_definition_compiles_the_vertical_slice():
    definition = TypedInterfaceDefinition.load(INTERFACE)
    definition.validate_entity_specs(entity_specs())
    compiler = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interface_path=str(INTERFACE),
    )
    compiler.attach_entity_specs(entity_specs())
    snapshot = compiler.compile(entity_payload())
    assert snapshot.agent_ids == ("Building_1", "Building_2")
    assert {group.group_type for group in snapshot.action_groups} == {
        "stationary_storage",
        "ev_session",
        "deferrable",
    }


def test_manually_editing_observations_changes_the_compiled_view(tmp_path):
    raw = yaml.safe_load(INTERFACE.read_text(encoding="utf-8"))
    building = raw["observations"]["entities"]["building"]
    building["features"] = ["net_power_kw"]
    building["required_features"] = ["net_power_kw"]
    manual = _write_yaml(tmp_path / "manual_interface.yaml", raw)

    compiler = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interface_path=str(manual),
    )
    compiler.attach_entity_specs(entity_specs())
    snapshot = compiler.compile(entity_payload())
    part = next(
        item
        for item in snapshot.parts_for("Building_1")
        if item.source_entity_id == "Building_1"
    )
    assert part.feature_names == ("net_power_kw",)
    assert part.values == (2.0,)


def test_required_manual_observation_must_exist_in_simulator(tmp_path):
    raw = yaml.safe_load(INTERFACE.read_text(encoding="utf-8"))
    raw["observations"]["entities"]["building"]["features"].append("manual_required")
    raw["observations"]["entities"]["building"]["required_features"].append(
        "manual_required"
    )
    manual = _write_yaml(tmp_path / "invalid_interface.yaml", raw)
    definition = TypedInterfaceDefinition.load(manual)
    with pytest.raises(ValueError, match="missing required 'building' observations"):
        definition.validate_entity_specs(entity_specs())


def test_optional_inactive_module_does_not_require_an_action_table():
    specs = deepcopy(entity_specs())
    specs["tables"]["deferrable_appliance"]["ids"] = []
    specs["tables"]["deferrable_appliance"]["features"] = []
    specs["actions"].pop("deferrable_appliance")
    TypedInterfaceDefinition.load(INTERFACE).validate_entity_specs(specs)


def test_active_module_requires_its_declared_action():
    specs = deepcopy(entity_specs())
    specs["actions"]["deferrable_appliance"]["features"] = []
    with pytest.raises(ValueError, match="requires Simulator action"):
        TypedInterfaceDefinition.load(INTERFACE).validate_entity_specs(specs)


def test_generated_file_contains_catalog_and_is_still_editable(tmp_path):
    output = generate(
        base_path=INTERFACE,
        entity_specs=entity_specs(),
        output_path=tmp_path / "generated.yaml",
        source="unit-test entity_specs",
    )
    generated = TypedInterfaceDefinition.load(output)
    assert generated.catalog["generated_from"] == "unit-test entity_specs"
    assert "net_power_kw" in generated.catalog["observations"]["building"]
    assert generated.catalog["actions"]["building"] == ["electrical_storage"]
    generated.validate_entity_specs(entity_specs())

    changed_specs = deepcopy(entity_specs())
    changed_specs["tables"]["building"]["features"].remove("net_power_kw")
    with pytest.raises(ValueError, match="missing required 'building' observations"):
        generated.validate_entity_specs(changed_specs)


def test_single_file_and_legacy_split_have_same_semantic_signature(tmp_path):
    definition = TypedInterfaceDefinition.load(INTERFACE)
    schema = _write_yaml(tmp_path / "schema.yaml", definition.agent_schema)
    registry = _write_yaml(tmp_path / "registry.yaml", definition.type_registry)
    health = _write_yaml(tmp_path / "health.yaml", definition.health_rules)
    unified = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interface_path=str(INTERFACE),
    )
    legacy = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        agent_schema_path=str(schema),
        type_registry_path=str(registry),
        health_rules_path=str(health),
    )
    assert unified.compatibility_signature == legacy.compatibility_signature
