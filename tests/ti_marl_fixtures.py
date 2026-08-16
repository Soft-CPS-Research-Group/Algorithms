"""Small versioned entity/runtime fixtures for TI-MARL unit tests."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def entity_specs(buildings=("Building_1", "Building_2")):
    buildings = list(buildings)
    storage_ids = [f"{name}/electrical_storage" for name in buildings]
    charger_ids = [f"{buildings[0]}/charger_1"] if buildings else []
    deferrable_ids = [f"{buildings[0]}/washer"] if buildings else []
    return {
        "version": "entity_v1",
        "runtime_status_contract": {
            "version": "runtime_status_v1",
            "emits_health_state": False,
        },
        "action_execution_contract": {"version": "entity_action_execution_v1"},
        "tables": {
            "district": {
                "ids": ["district_0"],
                "features": ["hour", "electricity_pricing", "community_net_power_kw"],
            },
            "building": {
                "ids": buildings,
                "features": [
                    "non_shiftable_load",
                    "solar_generation",
                    "net_power_kw",
                    "charging_building_headroom_kw",
                    "charging_building_export_headroom_kw",
                ],
            },
            "storage": {
                "ids": storage_ids,
                "features": [
                    "soc",
                    "max_charge_power_kw",
                    "max_discharge_power_kw",
                    "available_charge_action_normalized",
                    "available_discharge_action_normalized",
                ],
            },
            "charger": {
                "ids": charger_ids,
                "features": [
                    "connected_state",
                    "connected_ev_soc",
                    "max_charging_power_kw",
                    "max_discharging_power_kw",
                    "available_charge_action_normalized",
                    "available_discharge_action_normalized",
                ],
            },
            "ev": {
                "ids": ["EV_1"] if charger_ids else [],
                "features": ["soc", "soc_ratio", "energy_to_full_kwh"],
            },
            "deferrable_appliance": {
                "ids": deferrable_ids,
                "features": ["pending", "running", "can_start", "slack_steps"],
            },
            "pv": {"ids": [], "features": ["generation_power_kw"]},
        },
        "actions": {
            "building": {"ids": buildings, "features": ["electrical_storage"]},
            "charger": {"ids": charger_ids, "features": ["electric_vehicle_storage"]},
            "deferrable_appliance": {"ids": deferrable_ids, "features": ["start"]},
        },
    }


def entity_payload(
    buildings=("Building_1", "Building_2"),
    *,
    time_step=0,
    topology_version=0,
    runtime_status=None,
):
    specs = entity_specs(buildings)
    n = len(buildings)
    charger_count = len(specs["tables"]["charger"]["ids"])
    deferrable_count = len(specs["tables"]["deferrable_appliance"]["ids"])
    status = runtime_status or {
        "version": "runtime_status_v1",
        "emits_health_state": False,
        "active_events": [],
        "asset_connections": [
            {
                "relation": "charger_to_ev_connected",
                "source_type": "charger",
                "source_id": specs["tables"]["charger"]["ids"][0],
                "target_type": "ev",
                "target_id": "EV_1",
                "connection": "CONNECTED",
                "availability": "AVAILABLE",
                "quality": "NOMINAL",
                "event_ids": [],
            }
        ] if charger_count else [],
        "asset_availability": [],
        "sensor_channels": [],
        "actuator_channels": [],
        "communication_links": [],
        "value_quality": [],
    }
    return {
        "tables": {
            "district": np.asarray([[12.0, 0.2, 3.0]], dtype=np.float32),
            "building": np.asarray(
                [[3.0 + index, 1.0, 2.0, 4.0, 3.0] for index in range(n)],
                dtype=np.float32,
            ),
            "storage": np.asarray(
                [[0.5, 5.0, 5.0, 1.0, 1.0] for _ in range(n)],
                dtype=np.float32,
            ),
            "charger": np.asarray(
                [[1.0, 0.4, 7.0, 7.0, 1.0, 0.5] for _ in range(charger_count)],
                dtype=np.float32,
            ),
            "ev": np.asarray([[20.0, 0.4, 30.0]], dtype=np.float32) if charger_count else np.empty((0, 3)),
            "deferrable_appliance": np.asarray(
                [[1.0, 0.0, 1.0, 0.0] for _ in range(deferrable_count)],
                dtype=np.float32,
            ),
            "pv": np.empty((0, 1), dtype=np.float32),
        },
        "edges": {
            "district_to_building": np.asarray([[0, index] for index in range(n)], dtype=np.int64),
            "building_to_storage": np.asarray([[index, index] for index in range(n)], dtype=np.int64),
            "building_to_charger": np.asarray([[0, 0]], dtype=np.int64) if charger_count else np.empty((0, 2), dtype=np.int64),
            "building_to_deferrable_appliance": np.asarray([[0, 0]], dtype=np.int64) if deferrable_count else np.empty((0, 2), dtype=np.int64),
            "building_to_pv": np.empty((0, 2), dtype=np.int64),
            "charger_to_ev_connected": np.asarray([[0, 0]], dtype=np.int64) if charger_count else np.empty((0, 2), dtype=np.int64),
            "charger_to_ev_connected_mask": np.asarray([1.0], dtype=np.float32) if charger_count else np.empty((0,), dtype=np.float32),
            "charger_to_ev_incoming": np.empty((0, 2), dtype=np.int64),
            "charger_to_ev_incoming_mask": np.empty((0,), dtype=np.float32),
        },
        "meta": {
            "spec_version": "entity_v1",
            "time_step": int(time_step),
            "topology_version": int(topology_version),
            "runtime_status": deepcopy(status),
        },
    }


def stuck_sensor_status(*, duration: int, age: int):
    return {
        "version": "runtime_status_v1",
        "emits_health_state": False,
        "active_events": [
            {
                "event_id": "sensor-stuck",
                "event_domain": "SENSOR_CHANNEL",
                "fault_mode": "stuck",
                "target_type": "building",
                "target_id": "Building_1",
                "target_feature": "net_power_kw",
                "start_time_step": 0,
                "active_duration_steps": duration,
            }
        ],
        "asset_connections": [],
        "asset_availability": [],
        "sensor_channels": [
            {
                "event_id": "sensor-stuck",
                "event_ids": ["sensor-stuck"],
                "fault_mode": "stuck",
                "target_type": "building",
                "target_id": "Building_1",
                "target_feature": "net_power_kw",
                "availability": "AVAILABLE",
                "connection": "NOT_APPLICABLE",
                "quality": "IMPAIRED",
                "last_update_time_step": age,
                "last_fresh_time_step": 0,
                "age_steps": age,
            }
        ],
        "actuator_channels": [],
        "communication_links": [],
        "value_quality": [],
    }
