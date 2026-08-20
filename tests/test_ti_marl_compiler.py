from __future__ import annotations

from copy import deepcopy

import pytest

from algorithms.ti_marl.compiler import TypedInterfaceCompiler
from algorithms.ti_marl.compiler.closure import validate_dependency_graph
from algorithms.ti_marl.contracts.enums import EventDomain, HealthState
from tests.ti_marl_fixtures import (
    entity_payload,
    entity_specs,
    stuck_sensor_status,
    write_typed_interfaces,
)


def compiler(tmp_path, specs=None):
    interfaces = write_typed_interfaces(
        tmp_path / "interfaces",
        ("Building_1", "Building_2", "Building_3"),
    )
    instance = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interfaces_dir=interfaces,
    )
    instance.attach_entity_specs(specs or entity_specs(), seconds_per_time_step=900)
    return instance


def test_stuck_cause_is_preserved_and_crosses_degraded_to_stale(tmp_path):
    tic = compiler(tmp_path)
    early = tic.compile(
        entity_payload(time_step=1, runtime_status=stuck_sensor_status(duration=1, age=1))
    )
    evidence = next(item for item in early.fault_evidence if item.fault_mode == "stuck")
    assert evidence.event_domain == EventDomain.SENSOR_CHANNEL
    assert evidence.fault_mode == "stuck"
    early_health = next(
        item for item in early.health if item.subject_id.startswith("SENSOR_CHANNEL:building:Building_1:net_power_kw")
    )
    assert early_health.state == HealthState.DEGRADED

    late = tic.compile(
        entity_payload(time_step=8, runtime_status=stuck_sensor_status(duration=8, age=8))
    )
    late_health = next(item for item in late.health if item.subject_id == early_health.subject_id)
    assert late_health.state == HealthState.STALE


def test_connection_availability_actuator_and_community_loss_have_distinct_closure(tmp_path):
    status = stuck_sensor_status(duration=1, age=1)
    status["sensor_channels"] = []
    status["active_events"] = []
    status["asset_connections"] = [
        {
            "source_type": "charger",
            "source_id": "Building_1/charger_1",
            "target_type": "ev",
            "target_id": None,
            "connection": "DISCONNECTED",
            "availability": "AVAILABLE",
            "quality": "NOMINAL",
            "event_ids": [],
        }
    ]
    status["asset_availability"] = [
        {
            "event_id": "storage-down",
            "event_ids": ["storage-down"],
            "target_type": "storage",
            "target_id": "Building_2/electrical_storage",
            "target_feature": "both",
            "availability": "UNAVAILABLE",
            "connection": "NOT_APPLICABLE",
            "quality": "INVALID",
            "fault_mode": "unavailable",
        }
    ]
    status["actuator_channels"] = [
        {
            "event_id": "storage-actuator",
            "event_ids": ["storage-actuator"],
            "target_type": "storage",
            "target_id": "Building_1/electrical_storage",
            "target_feature": "electrical_storage",
            "availability": "UNAVAILABLE",
            "connection": "NOT_APPLICABLE",
            "quality": "INVALID",
            "fault_mode": "dropout",
        }
    ]
    status["communication_links"] = [
        {
            "event_id": "cloud-loss",
            "event_ids": ["cloud-loss"],
            "target_type": "district",
            "target_id": "district_0",
            "target_feature": "electricity_pricing",
            "availability": "UNAVAILABLE",
            "connection": "NOT_APPLICABLE",
            "quality": "INVALID",
            "fault_mode": "missing",
        }
    ]
    snapshot = compiler(tmp_path).compile(entity_payload(runtime_status=status))
    domains = {item.event_domain for item in snapshot.fault_evidence}
    assert {
        EventDomain.ASSET_CONNECTION,
        EventDomain.ASSET_AVAILABILITY,
        EventDomain.ACTUATOR_CHANNEL,
        EventDomain.COMMUNICATION_LINK,
    } <= domains

    charger = next(group for group in snapshot.action_groups if group.group_type == "ev_session")
    assert charger.enabled
    assert all(not port.valid for port in charger.ports if port.mode != "IDLE")
    storage_1 = next(group for group in snapshot.action_groups if group.group_id == "Building_1:battery_1")
    assert not storage_1.enabled
    assert all(not port.valid for port in storage_1.ports if port.mode != "IDLE")
    storage_2 = next(group for group in snapshot.action_groups if group.group_id == "Building_2:battery_1")
    assert not storage_2.enabled
    assert any(part.valid for part in snapshot.parts_for("Building_1") if part.semantic_type != "community_signal")
    assert all(not part.valid for part in snapshot.parts_for("Building_1") if part.semantic_type == "community_signal")


def test_recovery_hysteresis_prevents_instant_healthy_transition(tmp_path):
    tic = compiler(tmp_path)
    tic.compile(entity_payload(time_step=8, runtime_status=stuck_sensor_status(duration=8, age=8)))
    nominal = entity_payload(time_step=9)
    first = tic.compile(nominal)
    recovered_subject = next(
        item for item in first.health if item.subject_id.startswith("SENSOR_CHANNEL:building:Building_1")
    )
    assert recovered_subject.state == HealthState.DEGRADED
    second = tic.compile(entity_payload(time_step=10))
    assert next(item for item in second.health if item.subject_id == recovered_subject.subject_id).state == HealthState.HEALTHY


def test_recovery_preserves_the_failed_channels_safety_criticality(tmp_path):
    status = stuck_sensor_status(duration=1, age=1)
    status["active_events"] = []
    status["sensor_channels"] = []
    status["actuator_channels"] = [
        {
            "event_id": "actuator-loss",
            "event_ids": ["actuator-loss"],
            "fault_mode": "dropout",
            "target_type": "storage",
            "target_id": "Building_1/electrical_storage",
            "target_feature": "electrical_storage",
            "availability": "UNAVAILABLE",
            "connection": "NOT_APPLICABLE",
            "quality": "INVALID",
        }
    ]
    tic = compiler(tmp_path)
    failed = tic.compile(entity_payload(time_step=0, runtime_status=status))
    subject = next(
        item
        for item in failed.health
        if item.subject_id.startswith("ACTUATOR_CHANNEL:storage:Building_1")
    )
    assert subject.criticality == "safety"
    assert subject.state == HealthState.MISSING

    recovering = tic.compile(entity_payload(time_step=0))
    assessment = next(item for item in recovering.health if item.subject_id == subject.subject_id)
    assert assessment.criticality == "safety"
    assert assessment.state == HealthState.DEGRADED
    recovered = tic.compile(entity_payload(time_step=1))
    assessment = next(item for item in recovered.health if item.subject_id == subject.subject_id)
    assert assessment.criticality == "safety"
    assert assessment.state == HealthState.HEALTHY


def test_identical_facts_produce_identical_snapshot_hash(tmp_path):
    payload = entity_payload()
    first = compiler(tmp_path / "first").compile(deepcopy(payload))
    second = compiler(tmp_path / "second").compile(deepcopy(payload))
    assert first.snapshot_hash == second.snapshot_hash


def test_snapshot_hash_is_cached_after_first_canonicalisation(tmp_path, monkeypatch):
    tic = compiler(tmp_path)
    snapshot = tic.compile(entity_payload())

    from algorithms.ti_marl.contracts import models

    original = models.content_hash
    calls = 0

    def counting_content_hash(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(models, "content_hash", counting_content_hash)
    first = snapshot.snapshot_hash
    second = snapshot.snapshot_hash

    assert first == second
    assert calls == 1


def test_unknown_active_entity_type_is_not_controlled_automatically(tmp_path):
    specs = entity_specs()
    specs["tables"]["mystery_asset"] = {"ids": ["x"], "features": ["value"]}
    payload = entity_payload()
    payload["tables"]["mystery_asset"] = [[1.0]]
    snapshot = compiler(tmp_path, specs).compile(payload)
    assert all(group.module_id != "x" for group in snapshot.action_groups)


def test_ambiguous_asset_binding_fails_safely(tmp_path):
    payload = entity_payload()
    payload["edges"]["building_to_storage"] = payload["edges"]["building_to_storage"].copy()
    payload["edges"]["building_to_storage"][1, 1] = 0
    with pytest.raises(ValueError, match="ambiguous binding"):
        compiler(tmp_path).compile(payload)


def test_session_replacement_rebinds_stable_charger_without_stale_ev_identity(tmp_path):
    tic = compiler(tmp_path)
    first = tic.compile(entity_payload())
    assert any(entity.entity_id == "EV_1" for entity in first.entities)

    next_specs = entity_specs()
    next_specs["tables"]["ev"]["ids"] = ["EV_2"]
    next_payload = entity_payload(time_step=1, topology_version=1)
    next_payload["meta"]["runtime_status"]["asset_connections"][0]["target_id"] = "EV_2"
    tic.attach_entity_specs(next_specs, seconds_per_time_step=900)
    following = tic.compile(next_payload)
    ev_ids = {entity.entity_id for entity in following.entities if entity.entity_type == "ev"}
    assert ev_ids == {"EV_2"}
    assert any(
        group.module_id == "charger_1"
        and group.adapter_target_entity_id == "Building_1/charger_1"
        for group in following.action_groups
    )


def test_dependency_cycles_and_conflicts_are_rejected():
    with pytest.raises(ValueError, match="cycle"):
        validate_dependency_graph(
            [
                {"source_kind": "a", "target_group_type": "b", "consequence": "disable_group"},
                {"source_kind": "b", "target_group_type": "a", "consequence": "disable_group"},
            ]
        )
    with pytest.raises(ValueError, match="conflict"):
        validate_dependency_graph(
            [
                {"source_kind": "a", "target_group_type": "b", "consequence": "disable_group"},
                {"source_kind": "a", "target_group_type": "b", "consequence": "invalidate_non_idle_ports"},
            ]
        )
