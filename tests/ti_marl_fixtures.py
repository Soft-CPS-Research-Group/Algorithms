"""Small versioned entity/runtime fixtures for TI-MARL unit tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


def _all_health_outcomes(effect: str):
    return {
        state: effect
        for state in ("DEGRADED", "STALE", "MISSING", "FAILED", "UNKNOWN")
    }


def typed_interface_payload(agent_id: str, *, controllable: bool = True):
    sensors = {
        "self": {
            "type": "building_meter",
            "scope": "local",
            "channels": {
                "energy": {
                    "observations": {
                        "non_shiftable_load": {"unit": "kW", "policy_input": True},
                        "solar_generation": {"unit": "kW", "policy_input": True},
                        "net_power_kw": {"unit": "kW", "policy_input": True},
                    }
                },
                "grid": {
                    "observations": {
                        "charging_building_headroom_kw": {
                            "unit": "kW",
                            "use": "runtime_bound",
                            "policy_input": True,
                            "criticality": "safety",
                        },
                        "charging_building_export_headroom_kw": {
                            "unit": "kW",
                            "use": "runtime_bound",
                            "policy_input": True,
                            "criticality": "safety",
                        },
                    }
                },
            },
        },
        "community": {
            "type": "community_aggregate_service",
            "scope": "community",
            "channels": {
                "time": {"observations": {"hour": {"unit": "h", "policy_input": True}}},
                "market": {
                    "observations": {
                        "electricity_pricing": {"unit": "EUR/kWh", "policy_input": True}
                    }
                },
                "energy": {
                    "observations": {
                        "community_net_power_kw": {"unit": "kW", "policy_input": True},
                        "topology_version": {
                            "unit": "index",
                            "use": "trace_only",
                            "policy_input": False,
                            "reason": "structural metadata handled by the TIC",
                        },
                    }
                },
            },
        },
        "battery_1": {
            "type": "stationary_battery",
            "scope": "local",
            "channels": {
                "storage_state": {
                    "observations": {
                        "soc": {"unit": "fraction", "use": "safety_dependency", "policy_input": True},
                        "max_charge_power_kw": {"unit": "kW", "policy_input": True},
                        "max_discharge_power_kw": {"unit": "kW", "policy_input": True},
                        "available_charge_action_normalized": {"unit": "fraction", "use": "runtime_bound", "policy_input": True},
                        "available_discharge_action_normalized": {"unit": "fraction", "use": "runtime_bound", "policy_input": True},
                    }
                }
            },
        },
    }
    actuators = {}
    if controllable:
        battery_deps = {
            "battery_1.storage_state.soc": _all_health_outcomes("invalidate_port")
        }
        actuators["battery_1"] = {
            "type": "stationary_battery",
            "actions": {
                "charge": {
                    "parameter": {"unit": "kW", "bounds": [0.0, 5.0]},
                    "dependencies": battery_deps,
                },
                "discharge": {
                    "parameter": {"unit": "kW", "bounds": [0.0, 5.0]},
                    "dependencies": battery_deps,
                },
            },
        }
    if agent_id == "Building_1":
        sensors["charger_1"] = {
            "type": "bidirectional_ev_charger",
            "scope": "local",
            "channels": {
                "connection": {
                    "observations": {
                        "connected_state": {"unit": "boolean", "use": "safety_dependency", "policy_input": True}
                    }
                },
                "ev_state": {
                    "observations": {
                        "connected_ev_soc": {"unit": "fraction", "use": "safety_dependency", "policy_input": True}
                    }
                },
                "capability": {
                    "observations": {
                        "max_charging_power_kw": {"unit": "kW", "policy_input": True},
                        "max_discharging_power_kw": {"unit": "kW", "policy_input": True},
                        "available_charge_action_normalized": {"unit": "fraction", "use": "runtime_bound", "policy_input": True},
                        "available_discharge_action_normalized": {"unit": "fraction", "use": "runtime_bound", "policy_input": True},
                    }
                },
            },
        }
        sensors["ev_session_1"] = {
            "type": "ev_session",
            "scope": "local",
            "channels": {
                "ev_session": {
                    "observations": {
                        "soc": {"unit": "kWh", "policy_input": True},
                        "soc_ratio": {"unit": "fraction", "policy_input": True},
                        "energy_to_full_kwh": {"unit": "kWh", "policy_input": True},
                    }
                }
            },
        }
        sensors["deferrable_1"] = {
            "type": "deferrable_appliance",
            "scope": "local",
            "channels": {
                "schedule": {
                    "observations": {
                        "pending": {"unit": "boolean", "policy_input": True},
                        "running": {"unit": "boolean", "policy_input": True},
                        "can_start": {"unit": "boolean", "use": "runtime_bound", "policy_input": True},
                        "slack_steps": {"unit": "count", "policy_input": True},
                        "deadline_time_step": {"unit": "timestamp", "use": "safety_dependency", "policy_input": True},
                        "must_run": {"unit": "boolean", "policy_input": True},
                    }
                }
            },
        }
        if controllable:
            connection = _all_health_outcomes("invalidate_port")
            soc_charge = _all_health_outcomes("max_safe_charge")
            soc_discharge = _all_health_outcomes("no_v2g")
            actuators["charger_1"] = {
                "type": "bidirectional_ev_charger",
                "actions": {
                    "charge": {
                        "parameter": {"unit": "kW", "bounds": [0.0, 7.0]},
                        "dependencies": {
                            "charger_1.connection.connected_state": connection,
                            "charger_1.ev_state.connected_ev_soc": soc_charge,
                        },
                    },
                    "discharge": {
                        "parameter": {"unit": "kW", "bounds": [0.0, 7.0]},
                        "dependencies": {
                            "charger_1.connection.connected_state": connection,
                            "charger_1.ev_state.connected_ev_soc": soc_discharge,
                        },
                    },
                },
            }
            actuators["deferrable_1"] = {
                "type": "deferrable_appliance",
                "actions": {
                    "start": {
                        "parameter": {"unit": "boolean", "bounds": [0.0, 1.0]},
                        "dependencies": {
                            "deferrable_1.schedule.can_start": _all_health_outcomes("invalidate_port")
                        },
                    }
                },
            }
    return {
        "version": "typed_agent_interface_v1",
        "contract": "ti_marl_v1",
        "description": f"Test interface for {agent_id}",
        "agent": {"id": agent_id, "role": "prosumer", "type": "residential"},
        "sensors": sensors,
        "actuators": actuators,
        "constraints": {"grid_import": {"unit": "kW", "max": 13.8}},
        "fallback": {"mode": "ISOLATED_SAFE"},
    }


def write_typed_interfaces(path: Path, buildings=("Building_1", "Building_2", "Building_3")) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for building in buildings:
        payload = typed_interface_payload(str(building))
        (path / f"{building}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
    return path


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
                "features": [
                    "hour",
                    "electricity_pricing",
                    "community_net_power_kw",
                    "topology_version",
                ],
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
                "features": [
                    "pending",
                    "running",
                    "can_start",
                    "slack_steps",
                    "deadline_time_step",
                    "must_run",
                ],
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
            "district": np.asarray([[12.0, 0.2, 3.0, float(topology_version)]], dtype=np.float32),
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
                [[1.0, 0.0, 1.0, 0.0, 20.0, 1.0] for _ in range(deferrable_count)],
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
