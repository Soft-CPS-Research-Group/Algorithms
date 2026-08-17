from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from algorithms.ti_marl.compiler import TypedInterfaceCompiler
from algorithms.ti_marl.contracts.models import canonical_value
from algorithms.ti_marl.policy.networks import CentralSetCritic, TypedActor, parameter_count
from algorithms.ti_marl.runtime import MappingTelemetryAdapter
from tests.ti_marl_fixtures import (
    entity_payload,
    entity_specs,
    stuck_sensor_status,
    write_typed_interfaces,
)


def _compiler(tmp_path, buildings=("Building_1", "Building_2")):
    interfaces = write_typed_interfaces(tmp_path / "interfaces", buildings)
    compiler = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interfaces_dir=interfaces,
    )
    compiler.attach_entity_specs(entity_specs(buildings), seconds_per_time_step=900)
    return compiler


def _status(collection: str, row):
    payload = stuck_sensor_status(duration=0, age=0)
    payload["active_events"] = []
    for name in (
        "asset_connections",
        "asset_availability",
        "sensor_channels",
        "actuator_channels",
        "communication_links",
        "value_quality",
    ):
        payload[name] = []
    payload[collection] = [row]
    return payload


def test_simulator_and_mapping_gateway_frames_compile_identically(tmp_path):
    compiler = _compiler(tmp_path)
    simulator_frame = compiler.adapter.to_frame(entity_payload())
    gateway = MappingTelemetryAdapter(
        compiler.interface_registry,
        provenance="simulated_modbus_gateway",
    )
    mapped = gateway.to_frame(
        {
            "frame_id": "gateway-frame",
            "timestamp_seconds": simulator_frame.timestamp_seconds,
            "sequence": simulator_frame.sequence,
            "topology_version": simulator_frame.topology_version,
            "active_agent_ids": simulator_frame.active_agent_ids,
            "samples": [canonical_value(item) for item in simulator_frame.samples],
            "entities": [canonical_value(item) for item in simulator_frame.entities],
            "health_evidence": [
                canonical_value(item) for item in simulator_frame.health_evidence
            ],
        }
    )
    from_simulator = compiler.compile_frame(simulator_frame)
    from_gateway = compiler.compile_frame(mapped)
    assert from_simulator.snapshot_hash == from_gateway.snapshot_hash


@pytest.mark.parametrize("agent_count", [1, 17, 50, 100])
def test_population_scale_does_not_resize_network_parameters(tmp_path, agent_count):
    buildings = tuple(f"Building_{index}" for index in range(1, agent_count + 1))
    compiler = _compiler(tmp_path, buildings)
    actor = TypedActor(compiler.type_registry, d_model=32, attention_heads=4, relation_layers=1)
    critic = CentralSetCritic(compiler.type_registry, d_model=32, relation_layers=1)
    before = parameter_count(actor) + parameter_count(critic)
    snapshot = compiler.compile(entity_payload(buildings))
    assert len(snapshot.agent_ids) == agent_count
    assert parameter_count(actor) + parameter_count(critic) == before


def test_missing_grid_meter_forces_isolated_safe_mode(tmp_path):
    compiler = _compiler(tmp_path)
    status = _status(
        "sensor_channels",
        {
            "target_type": "building",
            "target_id": "Building_1",
            "target_feature": "charging_building_headroom_kw",
            "availability": "UNAVAILABLE",
            "quality": "INVALID",
            "fault_mode": "missing",
            "event_ids": ["grid-meter-loss"],
        },
    )
    snapshot = compiler.compile(entity_payload(runtime_status=status))
    local = snapshot.groups_for("Building_1")
    assert local
    assert all(not group.enabled and group.forced_mode == "IDLE" for group in local)


def test_uncertain_ev_soc_prohibits_v2g_and_forces_max_safe_charge(tmp_path):
    compiler = _compiler(tmp_path)
    status = _status(
        "sensor_channels",
        {
            "target_type": "charger",
            "target_id": "Building_1/charger_1",
            "target_feature": "connected_ev_soc",
            "availability": "AVAILABLE",
            "quality": "IMPAIRED",
            "fault_mode": "noise",
            "active_duration_steps": 1,
            "age_steps": 1,
            "event_ids": ["soc-impaired"],
        },
    )
    payload = entity_payload(runtime_status=status)
    frame = compiler.adapter.to_frame(payload)
    soc_sample = next(
        item for item in frame.samples if item.observation_id == "connected_ev_soc"
    )
    assert soc_sample.quality.value == "IMPAIRED"
    assert soc_sample.fault_mode == "noise"
    assert soc_sample.age_seconds == pytest.approx(900.0)
    snapshot = compiler.compile(payload)
    charger = next(group for group in snapshot.action_groups if group.group_type == "ev_session")
    assert charger.enabled
    assert charger.forced_mode == "CHARGE_EV"
    assert charger.forced_fraction == pytest.approx(1.0)
    discharge = next(port for port in charger.ports if port.mode == "DISCHARGE_EV")
    assert not discharge.valid


def test_community_loss_removes_coordination_without_stopping_local_control(tmp_path):
    compiler = _compiler(tmp_path)
    status = _status(
        "communication_links",
        {
            "target_type": "district",
            "target_id": "district_0",
            "target_feature": "*",
            "availability": "UNAVAILABLE",
            "quality": "INVALID",
            "fault_mode": "missing",
            "event_ids": ["cloud-loss"],
        },
    )
    snapshot = compiler.compile(entity_payload(runtime_status=status))
    assert all(not part.valid for part in snapshot.parts_for("Building_1") if part.scope == "community")
    assert any(group.enabled for group in snapshot.groups_for("Building_1"))


def test_power_outage_blocks_grid_charge_ev_v2g_and_deferrable_start(tmp_path):
    compiler = _compiler(tmp_path)
    status = _status(
        "value_quality",
        {
            "target_type": "district",
            "target_id": "district_0",
            "target_feature": "*",
            "availability": "AVAILABLE",
            "quality": "IMPAIRED",
            "fault_mode": "power_outage",
            "event_ids": ["grid-outage"],
        },
    )
    status["value_quality"] = []
    status["active_events"] = [
        {
            "event_id": "grid-outage",
            "event_domain": "VALUE_QUALITY",
            "target_type": "district",
            "target_id": "district_0",
            "target_feature": "*",
            "fault_mode": "power_outage",
            "start_time_step": 0,
            "active_duration_steps": 1,
        }
    ]
    snapshot = compiler.compile(entity_payload(runtime_status=status))
    storage = next(
        group
        for group in snapshot.groups_for("Building_1")
        if group.group_type == "stationary_storage"
    )
    assert not next(
        port for port in storage.ports if port.mode == "CHARGE_STATIONARY"
    ).valid
    assert next(
        port for port in storage.ports if port.mode == "DISCHARGE_STATIONARY"
    ).valid
    charger = next(
        group
        for group in snapshot.groups_for("Building_1")
        if group.group_type == "ev_session"
    )
    assert all(
        not port.valid
        for port in charger.ports
        if port.mode in {"CHARGE_EV", "DISCHARGE_EV"}
    )
    deferrable = next(
        group
        for group in snapshot.groups_for("Building_1")
        if group.group_type == "deferrable"
    )
    assert not next(port for port in deferrable.ports if port.mode == "START").valid


def test_missing_storage_soc_and_deferrable_state_fail_safe_locally(tmp_path):
    compiler = _compiler(tmp_path)
    status = _status(
        "sensor_channels",
        {
            "target_type": "storage",
            "target_id": "Building_1/electrical_storage",
            "target_feature": "soc",
            "availability": "UNAVAILABLE",
            "quality": "INVALID",
            "fault_mode": "missing",
            "event_ids": ["storage-soc-loss"],
        },
    )
    snapshot = compiler.compile(entity_payload(runtime_status=status))
    storage = next(group for group in snapshot.action_groups if group.group_type == "stationary_storage" and group.owner_agent_id == "Building_1")
    assert not storage.enabled

    compiler = _compiler(tmp_path / "deferrable")
    status = _status(
        "sensor_channels",
        {
            "target_type": "deferrable_appliance",
            "target_id": "Building_1/washer",
            "target_feature": "can_start",
            "availability": "UNAVAILABLE",
            "quality": "INVALID",
            "fault_mode": "missing",
            "event_ids": ["can-start-loss"],
        },
    )
    snapshot = compiler.compile(entity_payload(runtime_status=status))
    group = next(item for item in snapshot.action_groups if item.group_type == "deferrable")
    assert not next(port for port in group.ports if port.mode == "START").valid


def test_repeated_requested_applied_mismatch_isolates_only_target_group(tmp_path):
    compiler = _compiler(tmp_path)
    snapshot = None
    for time_step in range(3):
        payload = entity_payload(time_step=time_step)
        payload["meta"]["entity_action_execution"] = {
            "version": "entity_action_execution_v1",
            "entries": [
                {
                    "agent_id": "Building_1",
                    "owner_module_id": "Building_1",
                    "target_entity_id": "Building_1/electrical_storage",
                    "action_name": "electrical_storage",
                    "requested_value": 1.0,
                    "post_channel_value": 1.0,
                    "limited_value": 1.0,
                    "applied_value": 0.0,
                    "applied_power_kw": 0.0,
                    "limitation_reasons": [],
                }
            ],
        }
        snapshot = compiler.compile(payload)
    assert snapshot is not None
    storage = next(group for group in snapshot.action_groups if group.owner_agent_id == "Building_1" and group.group_type == "stationary_storage")
    charger = next(group for group in snapshot.action_groups if group.group_type == "ev_session")
    assert not storage.enabled
    assert storage.fallback_reason == "repeated_requested_applied_mismatch"
    assert charger.enabled
    execution = dict(snapshot.execution_feedback[0])
    assert execution["agent_id"] == "Building_1"
    assert execution["actuator_id"] == "battery_1"
    assert execution["target_entity_id"] == "battery_1"


def test_simulator_binding_translates_physical_names_outside_public_interface(tmp_path):
    interfaces = write_typed_interfaces(
        tmp_path / "interfaces",
        ("Building_1",),
    )
    specs = entity_specs(("Building_1",))
    feature_index = specs["tables"]["building"]["features"].index("net_power_kw")
    specs["tables"]["building"]["features"][feature_index] = "native_meter_power"
    payload = entity_payload(("Building_1",))
    payload["tables"]["building"] = payload["tables"]["building"].copy()
    bindings = {
        "version": "ti_marl_simulator_bindings_v1",
        "agents": {
            "Building_1": {
                "sensors": {
                    "self": {
                        "entity_type": "building",
                        "entity_id": "Building_1",
                        "observations": {"net_power_kw": "native_meter_power"},
                    }
                },
                "actuators": {},
            }
        },
    }
    bindings_path = Path(tmp_path) / "simulator-bindings.yaml"
    bindings_path.write_text(yaml.safe_dump(bindings), encoding="utf-8")
    compiler = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interfaces_dir=interfaces,
        simulator_bindings_path=bindings_path,
    )
    compiler.attach_entity_specs(specs, seconds_per_time_step=900)
    frame = compiler.adapter.to_frame(payload)
    sample = next(item for item in frame.samples if item.observation_id == "net_power_kw")
    assert sample.source_feature == "native_meter_power"
    assert "native_meter_power" not in (interfaces / "Building_1.yaml").read_text(
        encoding="utf-8"
    )


def test_runtime_unit_shape_and_duplicate_samples_fail_safely(tmp_path):
    compiler = _compiler(tmp_path)
    frame = compiler.adapter.to_frame(entity_payload())
    target = next(
        item
        for item in frame.samples
        if item.agent_id == "Building_1"
        and item.observation_id == "charging_building_headroom_kw"
    )
    bad_unit = replace(target, unit="A", shape=(2,))
    bad_frame = replace(
        frame,
        samples=tuple(
            bad_unit if item.sample_id == target.sample_id else item
            for item in frame.samples
        ),
    )
    snapshot = compiler.compile_frame(bad_frame)
    part = next(
        item
        for item in snapshot.observation_parts
        if item.owner_agent_id == target.agent_id
        and item.sensor_id == target.sensor_id
        and item.channel_id == target.channel_id
        and item.observation_id == target.observation_id
    )
    assert not part.valid
    assert set(part.validity_reasons) == {"shape_mismatch", "unit_mismatch"}
    assert all(
        not group.enabled
        for group in snapshot.groups_for("Building_1")
    )

    duplicate = replace(frame, samples=frame.samples + (frame.samples[0],))
    with pytest.raises(ValueError, match="duplicate observation samples"):
        compiler.compile_frame(duplicate)
