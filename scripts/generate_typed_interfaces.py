"""Generate complete per-agent TI-MARL interfaces from an entity dataset.

The output is an editable deployment contract, not a Simulator contract.  The
Simulator is used only as one discovery adapter; technological bindings remain
outside the generated YAMLs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ti_marl.contracts.interface_definition import TypedAgentInterface
from algorithms.ti_marl.contracts.profile_registry import (
    CapabilityProfileRegistry,
    SENSOR_ENTITY_TYPES,
    channel_for,
)
from run_experiment import _resolve_citylearn_schema_input
from scripts.dump_entity_obs_sample import _build_env
from utils.config_schema import validate_config


SENSOR_TYPES = {
    "building": "building_meter",
    "district": "community_aggregate_service",
    "storage": "stationary_battery",
    "charger": "bidirectional_ev_charger",
    "ev": "ev_session",
    "deferrable_appliance": "deferrable_appliance",
    "pv": "pv_monitor",
}


def _pairs(raw: Any) -> list[tuple[int, int]]:
    values = np.asarray([] if raw is None else raw, dtype=np.int64)
    if values.size == 0:
        return []
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return [(int(row[0]), int(row[1])) for row in values if len(row) >= 2]


def _owners(specs: Mapping[str, Any], payload: Mapping[str, Any]) -> Mapping[tuple[str, int], str]:
    buildings = [str(item) for item in specs["tables"]["building"].get("ids", [])]
    result = {("building", index): agent_id for index, agent_id in enumerate(buildings)}
    relations = {
        "building_to_storage": "storage",
        "building_to_charger": "charger",
        "building_to_deferrable_appliance": "deferrable_appliance",
        "building_to_pv": "pv",
    }
    edges = payload.get("edges", {})
    for relation, entity_type in relations.items():
        for source, target in _pairs(edges.get(relation)):
            if 0 <= source < len(buildings):
                result[(entity_type, target)] = buildings[source]
    charger_owners = {
        row: owner for (kind, row), owner in result.items() if kind == "charger"
    }
    for relation in ("charger_to_ev_connected", "charger_to_ev_incoming"):
        pairs = _pairs(edges.get(relation))
        mask = np.asarray(edges.get(f"{relation}_mask", []), dtype=np.float64).reshape(-1)
        for index, (charger_row, ev_row) in enumerate(pairs):
            if index < len(mask) and mask[index] <= 0.5:
                continue
            if charger_row in charger_owners:
                result[("ev", ev_row)] = charger_owners[charger_row]
    # Dynamic schemas commonly encode stable ownership in entity IDs even if
    # an asset starts inactive and therefore has no current edge.
    for entity_type, table in specs.get("tables", {}).items():
        for row, entity_id in enumerate(table.get("ids", [])):
            if (str(entity_type), row) in result:
                continue
            matches = [agent_id for agent_id in buildings if str(entity_id).startswith(f"{agent_id}/")]
            if len(matches) == 1:
                result[(str(entity_type), row)] = matches[0]
    return result


def _observation_payload(
    profiles: CapabilityProfileRegistry,
    sensor_type: str,
    scope: str,
    entity_type: str,
    feature: str,
    unit: str | None,
) -> Mapping[str, Any]:
    channel = channel_for(entity_type, feature)
    defaults = profiles.observation_defaults(
        sensor_type=sensor_type,
        channel=channel,
        observation=feature,
        scope=scope,
        unit=unit,
    )
    # Public files stay compact; the resolved registry expands semantic type,
    # use, criticality, required, normalisation and any exclusion reason.
    return {"unit": defaults["unit"]}


def _sensor(
    profiles: CapabilityProfileRegistry,
    specs: Mapping[str, Any],
    entity_type: str,
    sensor_type: str,
    scope: str,
) -> Mapping[str, Any]:
    table = specs["tables"][entity_type]
    units = list(table.get("units", []))
    channels: Dict[str, Any] = {}
    for index, raw_feature in enumerate(table.get("features", [])):
        feature = str(raw_feature)
        channel = channel_for(entity_type, feature)
        unit = str(units[index]) if index < len(units) and units[index] else None
        channels.setdefault(channel, {"observations": {}})["observations"][feature] = (
            _observation_payload(
                profiles,
                sensor_type,
                scope,
                entity_type,
                feature,
                unit,
            )
        )
    return {
        "type": sensor_type,
        "scope": scope,
        "profile": f"{sensor_type}_v1",
        "channels": channels,
    }


def _outcomes(*, degraded: str, stale: str, missing: str, failed: str, unknown: str) -> Mapping[str, str]:
    return {
        "DEGRADED": degraded,
        "STALE": stale,
        "MISSING": missing,
        "FAILED": failed,
        "UNKNOWN": unknown,
    }


def _dependency(sensor_id: str, specs: Mapping[str, Any], entity_type: str, tokens: Sequence[str]) -> str | None:
    features = [str(item) for item in specs["tables"][entity_type].get("features", [])]
    feature = next((item for item in features if all(token in item.lower() for token in tokens)), None)
    if feature is None:
        return None
    return f"{sensor_id}.{channel_for(entity_type, feature)}.{feature}"


def _actuator(
    actuator_id: str,
    actuator_type: str,
    specs: Mapping[str, Any],
    *,
    charge_min_kw: float = 0.0,
    charge_max_kw: float = 0.0,
    discharge_min_kw: float = 0.0,
    discharge_max_kw: float = 0.0,
) -> Mapping[str, Any]:
    if actuator_type in {"bidirectional_ev_charger", "ev_charger"}:
        entity_type = "charger"
        bound_features = specs["tables"][entity_type].get("features", [])
        charge_min = max(float(charge_min_kw), 0.0)
        charge_max = max(float(charge_max_kw), 0.0)
        discharge_min = max(float(discharge_min_kw), 0.0)
        discharge_max = max(float(discharge_max_kw), 0.0)
        actions = ("charge", "discharge") if actuator_type == "bidirectional_ev_charger" else ("charge",)
        connected = _dependency(actuator_id, specs, entity_type, ("connected", "state"))
        soc = _dependency(actuator_id, specs, entity_type, ("connected", "soc"))
        departure = _dependency(actuator_id, specs, entity_type, ("departure",))
        charge_bound = _dependency(actuator_id, specs, entity_type, ("available", "charge", "normalized"))
        discharge_bound = _dependency(actuator_id, specs, entity_type, ("available", "discharge", "normalized"))
        result: Dict[str, Any] = {"type": actuator_type, "actions": {}}
        for action in actions:
            dependencies: Dict[str, Any] = {}
            for path in (connected, charge_bound if action == "charge" else discharge_bound):
                if path:
                    dependencies[path] = _outcomes(
                        degraded="invalidate_port",
                        stale="invalidate_port",
                        missing="invalidate_port",
                        failed="invalidate_port",
                        unknown="invalidate_port",
                    )
            for path in (soc, departure):
                if path:
                    dependencies[path] = _outcomes(
                        degraded=("max_safe_charge" if action == "charge" else "no_v2g"),
                        stale=("max_safe_charge" if action == "charge" else "no_v2g"),
                        missing=("max_safe_charge" if action == "charge" else "no_v2g"),
                        failed=("max_safe_charge" if action == "charge" else "no_v2g"),
                        unknown=("max_safe_charge" if action == "charge" else "no_v2g"),
                    )
            result["actions"][action] = {
                "parameter": {
                    "unit": "kW",
                    "bounds": [
                        charge_min if action == "charge" else discharge_min,
                        charge_max if action == "charge" else discharge_max,
                    ],
                },
                "dependencies": dependencies,
            }
        return result
    if actuator_type == "stationary_battery":
        entity_type = "storage"
        soc = _dependency(actuator_id, specs, entity_type, ("soc",))
        result = {"type": actuator_type, "actions": {}}
        for action in ("charge", "discharge"):
            bound = _dependency(actuator_id, specs, entity_type, ("available", action, "normalized"))
            dependencies = {}
            for path in (soc, bound):
                if path:
                    dependencies[path] = _outcomes(
                        degraded="invalidate_port",
                        stale="invalidate_port",
                        missing="invalidate_port",
                        failed="invalidate_port",
                        unknown="invalidate_port",
                    )
            result["actions"][action] = {
                "parameter": {
                    "unit": "kW",
                    "bounds": [
                        0.0,
                        max(
                            float(
                                charge_max_kw
                                if action == "charge"
                                else discharge_max_kw
                            ),
                            0.0,
                        ),
                    ],
                },
                "dependencies": dependencies,
            }
        return result
    if actuator_type == "deferrable_appliance":
        can_start = _dependency(actuator_id, specs, "deferrable_appliance", ("can", "start"))
        dependencies = {}
        if can_start:
            dependencies[can_start] = _outcomes(
                degraded="invalidate_port",
                stale="invalidate_port",
                missing="invalidate_port",
                failed="invalidate_port",
                unknown="invalidate_port",
            )
        return {
            "type": actuator_type,
            "actions": {
                "start": {
                    "parameter": {"unit": "boolean", "bounds": [0.0, 1.0]},
                    "dependencies": dependencies,
                }
            },
        }
    raise ValueError(f"Unsupported actuator type {actuator_type!r}")


def _runtime_capability_kw(
    specs: Mapping[str, Any],
    payload: Mapping[str, Any],
    entity_type: str,
    row: int,
    *,
    direction: str,
) -> float:
    aliases = {
        ("charger", "charge"): ("max_charging_power_kw", "nominal_power_kw"),
        ("charger", "discharge"): ("max_discharging_power_kw", "nominal_power_kw"),
        ("storage", "charge"): ("max_charge_power_kw", "nominal_power_kw"),
        ("storage", "discharge"): ("max_discharge_power_kw", "nominal_power_kw"),
    }.get((entity_type, direction), ())
    features = [str(item) for item in specs["tables"][entity_type].get("features", [])]
    matrix = np.asarray(payload.get("tables", {}).get(entity_type, []), dtype=np.float64)
    if matrix.ndim == 1 and matrix.size:
        matrix = matrix.reshape(1, -1)
    for feature in aliases:
        if feature not in features or matrix.ndim != 2 or row >= matrix.shape[0]:
            continue
        value = float(matrix[row, features.index(feature)])
        if np.isfinite(value) and value > 0.0:
            return value
    return 0.0


def _schema_asset_limits(
    schema: Mapping[str, Any] | None,
    agent_id: str,
    entity_type: str,
    entity_id: str,
) -> tuple[float, float, float, float]:
    if not schema:
        return 0.0, 0.0, 0.0, 0.0
    building = dict(dict(schema.get("buildings", {}) or {}).get(agent_id, {}) or {})
    if entity_type == "storage":
        attributes = dict(dict(building.get("electrical_storage", {}) or {}).get("attributes", {}) or {})
        rated = float(attributes.get("nominal_power", 0.0) or 0.0)
        return 0.0, rated, 0.0, rated
    if entity_type == "charger":
        asset_id = str(entity_id).split("/", 1)[-1]
        charger = dict(dict(building.get("chargers", {}) or {}).get(asset_id, {}) or {})
        attributes = dict(charger.get("attributes", {}) or {})
        nominal = float(attributes.get("nominal_power", 0.0) or 0.0)
        return (
            float(attributes.get("min_charging_power", 0.0) or 0.0),
            float(attributes.get("max_charging_power", nominal) or 0.0),
            float(attributes.get("min_discharging_power", 0.0) or 0.0),
            float(attributes.get("max_discharging_power", nominal) or 0.0),
        )
    return 0.0, 0.0, 0.0, 0.0


def _building_constraints(
    schema: Mapping[str, Any] | None,
    agent_id: str,
) -> Mapping[str, Any]:
    """Expose only limits stated by the dataset; never invent amperage/limits."""

    if not schema:
        return {}
    building = dict(dict(schema.get("buildings", {}) or {}).get(agent_id, {}) or {})
    limits = dict(dict(building.get("electrical_service", {}) or {}).get("limits", {}) or {})
    total = dict(limits.get("total", {}) or {})
    phases = dict(limits.get("per_phase", {}) or {})
    result: Dict[str, Any] = {}
    if total.get("import_kw") is not None:
        result["grid_import"] = {"unit": "kW", "max": float(total["import_kw"])}
    if total.get("export_kw") is not None:
        result["grid_export"] = {"unit": "kW", "max": float(total["export_kw"])}
    for direction in ("import", "export"):
        values = {
            str(phase): float(dict(config or {}).get(f"{direction}_kw"))
            for phase, config in phases.items()
            if dict(config or {}).get(f"{direction}_kw") is not None
        }
        if values:
            result[f"phase_{direction}"] = {"unit": "kW", "max": values}
    return result


def generate(
    specs: Mapping[str, Any],
    payload: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Mapping[str, Any]], list[Mapping[str, Any]]]:
    profiles = CapabilityProfileRegistry()
    unsupported_tables = sorted(
        str(entity_type)
        for entity_type, table in specs.get("tables", {}).items()
        if entity_type not in SENSOR_TYPES and table.get("features")
    )
    if unsupported_tables:
        raise ValueError(
            "No TI-MARL sensor profile exists for Simulator entity tables: "
            f"{unsupported_tables}"
        )
    owners = _owners(specs, payload)
    buildings = [str(item) for item in specs["tables"]["building"].get("ids", [])]
    interfaces: Dict[str, Mapping[str, Any]] = {}
    coverage = []
    for agent_id in buildings:
        sensors: Dict[str, Any] = {
            "self": _sensor(profiles, specs, "building", "building_meter", "local"),
            "community": _sensor(
                profiles,
                specs,
                "district",
                "community_aggregate_service",
                "community",
            ),
        }
        actuators: Dict[str, Any] = {}
        counts: Dict[str, int] = {}
        for entity_type in ("storage", "charger", "ev", "deferrable_appliance", "pv"):
            rows = [
                row
                for (kind, row), owner in owners.items()
                if kind == entity_type and owner == agent_id
            ]
            for row in sorted(rows):
                counts[entity_type] = counts.get(entity_type, 0) + 1
                prefix = {
                    "storage": "battery",
                    "charger": "charger",
                    "ev": "ev_session",
                    "deferrable_appliance": "deferrable",
                    "pv": "pv",
                }[entity_type]
                sensor_id = f"{prefix}_{counts[entity_type]}"
                sensor_type = SENSOR_TYPES[entity_type]
                sensors[sensor_id] = _sensor(
                    profiles,
                    specs,
                    entity_type,
                    sensor_type,
                    "local",
                )
                if entity_type in {"storage", "charger", "deferrable_appliance"}:
                    entity_ids = specs["tables"][entity_type].get("ids", [])
                    entity_id = str(entity_ids[row]) if row < len(entity_ids) else sensor_id
                    (
                        schema_charge_min,
                        schema_charge_max,
                        schema_discharge_min,
                        schema_discharge_max,
                    ) = _schema_asset_limits(
                        schema,
                        agent_id,
                        entity_type,
                        entity_id,
                    )
                    runtime_charge = _runtime_capability_kw(
                        specs,
                        payload,
                        entity_type,
                        row,
                        direction="charge",
                    )
                    runtime_discharge = _runtime_capability_kw(
                        specs,
                        payload,
                        entity_type,
                        row,
                        direction="discharge",
                    )
                    actuators[sensor_id] = _actuator(
                        sensor_id,
                        sensor_type,
                        specs,
                        charge_min_kw=schema_charge_min,
                        charge_max_kw=schema_charge_max or runtime_charge,
                        discharge_min_kw=schema_discharge_min,
                        discharge_max_kw=schema_discharge_max or runtime_discharge,
                    )
        interface = {
            "version": "typed_agent_interface_v1",
            "contract": "ti_marl_v1",
            "description": f"Deployment-neutral typed interface for {agent_id}",
            "agent": {
                "id": agent_id,
                "role": (
                    "prosumer"
                    if any(sensor["type"] == "pv_monitor" for sensor in sensors.values())
                    else "consumer"
                ),
                # The discovery adapter must not invent a real-world building
                # classification that the dataset does not expose.
                "type": "other",
            },
            "sensors": sensors,
            "actuators": actuators,
            "constraints": _building_constraints(schema, agent_id),
            "fallback": {"mode": "ISOLATED_SAFE"},
        }
        interfaces[agent_id] = interface

    for entity_type, table in specs.get("tables", {}).items():
        sensor_type = SENSOR_TYPES.get(str(entity_type))
        if sensor_type is None:
            continue
        scope = "community" if entity_type == "district" else "local"
        units = list(table.get("units", []))
        for index, feature in enumerate(table.get("features", [])):
            unit = str(units[index]) if index < len(units) and units[index] else None
            defaults = profiles.observation_defaults(
                sensor_type=sensor_type,
                channel=channel_for(str(entity_type), str(feature)),
                observation=str(feature),
                scope=scope,
                unit=unit,
            )
            coverage.append(
                {
                    "entity_type": str(entity_type),
                    "feature": str(feature),
                    "classification": defaults["use"],
                    "policy_input": defaults["policy_input"],
                    "reason": defaults.get("reason", ""),
                }
            )
    return interfaces, coverage


def generate_simulator_bindings(
    specs: Mapping[str, Any],
    payload: Mapping[str, Any],
    interfaces: Mapping[str, Mapping[str, Any]],
    future_bindings: Mapping[tuple[str, str], str] | None = None,
) -> Mapping[str, Any]:
    owners = _owners(specs, payload)
    district_ids = [str(item) for item in specs["tables"]["district"].get("ids", [])]
    result: Dict[str, Any] = {
        "version": "ti_marl_simulator_bindings_v1",
        "agents": {},
    }
    for agent_id, interface in interfaces.items():
        sensor_bindings: Dict[str, Any] = {}
        actuator_bindings: Dict[str, Any] = {}
        counters: Dict[str, int] = {}
        for sensor_id, sensor in interface["sensors"].items():
            entity_type = SENSOR_ENTITY_TYPES[str(sensor["type"])]
            if entity_type == "building":
                entity_id = agent_id
            elif entity_type == "district":
                entity_id = district_ids[0] if district_ids else None
            else:
                candidates = [
                    str(specs["tables"][entity_type]["ids"][row])
                    for (kind, row), owner in owners.items()
                    if kind == entity_type
                    and owner == agent_id
                    and row < len(specs["tables"][entity_type].get("ids", []))
                ]
                index = counters.get(entity_type, 0)
                counters[entity_type] = index + 1
                ordered = sorted(candidates)
                entity_id = ordered[index] if index < len(ordered) else None
            sensor_bindings[str(sensor_id)] = {
                "entity_type": entity_type,
                "entity_id": dict(future_bindings or {}).get(
                    (agent_id, str(sensor_id)),
                    entity_id,
                ),
                # Identity field names need no verbose per-observation map.
                # Real adapters add entries only when technology names differ.
                "observations": {},
            }
            if sensor_id in interface.get("actuators", {}):
                actuator_bindings[str(sensor_id)] = {
                    "entity_type": entity_type,
                    "entity_id": dict(future_bindings or {}).get(
                        (agent_id, str(sensor_id)),
                        entity_id,
                    ),
                }
        result["agents"][agent_id] = {
            "sensors": sensor_bindings,
            "actuators": actuator_bindings,
        }
    return result


def augment_dynamic_assets(
    interfaces: Dict[str, Mapping[str, Any]],
    schema: Mapping[str, Any],
    specs: Mapping[str, Any],
) -> Mapping[tuple[str, str], str]:
    profiles = CapabilityProfileRegistry()
    future_bindings: Dict[tuple[str, str], str] = {}
    type_map = {
        "charger": ("charger", "bidirectional_ev_charger", "charger"),
        "pv": ("pv", "pv_monitor", "pv"),
        "electrical_storage": ("storage", "stationary_battery", "battery"),
        "storage": ("storage", "stationary_battery", "battery"),
        "deferrable_appliance": (
            "deferrable_appliance",
            "deferrable_appliance",
            "deferrable",
        ),
    }
    for event in schema.get("topology_events", []) or []:
        if str(event.get("operation")) != "add_asset":
            continue
        agent_id = str(event.get("target_member_id"))
        if agent_id not in interfaces:
            continue
        raw_type = str(event.get("target_asset_type"))
        if raw_type not in type_map:
            continue
        entity_type, sensor_type, prefix = type_map[raw_type]
        interface = interfaces[agent_id]
        existing = [
            sensor_id
            for sensor_id, sensor in interface["sensors"].items()
            if str(sensor.get("type")) == sensor_type
        ]
        # An add event introduces a new registered instance even if another
        # instance of the same type is already active.
        sensor_id = f"{prefix}_{len(existing) + 1}"
        if sensor_id not in interface["sensors"]:
            interface["sensors"][sensor_id] = _sensor(
                profiles,
                specs,
                entity_type,
                sensor_type,
                "local",
            )
            if entity_type in {"storage", "charger", "deferrable_appliance"}:
                source_agent = str(event.get("source_member_id") or agent_id)
                source_asset = str(event.get("source_asset_id") or event.get("target_asset_id") or sensor_id)
                (
                    charge_min,
                    charge_max,
                    discharge_min,
                    discharge_max,
                ) = _schema_asset_limits(
                    schema,
                    source_agent,
                    entity_type,
                    source_asset,
                )
                interface["actuators"][sensor_id] = _actuator(
                    sensor_id,
                    sensor_type,
                    specs,
                    charge_min_kw=charge_min,
                    charge_max_kw=charge_max,
                    discharge_min_kw=discharge_min,
                    discharge_max_kw=discharge_max,
                )
        if entity_type == "pv":
            interface["agent"]["role"] = "prosumer"
        asset_id = str(event.get("target_asset_id") or sensor_id)
        future_bindings[(agent_id, sensor_id)] = f"{agent_id}/{asset_id}"
    return future_bindings


def write_generated(
    output: Path,
    interfaces: Mapping[str, Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    *,
    source: str,
    simulator_bindings: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    unclassified = [
        row for row in coverage if not str(row.get("classification", "")).strip()
    ]
    if unclassified:
        raise ValueError(
            f"TI-MARL observation coverage has {len(unclassified)} unclassified fields"
        )
    output.mkdir(parents=True, exist_ok=True)
    existing = tuple(output.glob("*.yaml"))
    if existing and not overwrite:
        raise FileExistsError(
            "TI-MARL generator will not overwrite editable interfaces without "
            "--overwrite"
        )
    if overwrite:
        for old in existing:
            old.unlink()
    for agent_id, payload in interfaces.items():
        path = output / f"{agent_id}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        TypedAgentInterface.load(path)
    coverage_path = output / "observation_coverage.csv"
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("entity_type", "feature", "classification", "policy_input", "reason"),
        )
        writer.writeheader()
        writer.writerows(coverage)
    manifest = {
        "version": "typed_interface_manifest_v1",
        "source": source,
        "agent_count": len(interfaces),
        "observation_catalog_count": len(coverage),
        "unclassified_fields": len(unclassified),
        "interfaces": sorted(f"{agent_id}.yaml" for agent_id in interfaces),
    }
    if simulator_bindings is not None:
        bindings_dir = output / "technology_bindings"
        bindings_dir.mkdir(parents=True, exist_ok=True)
        binding_path = bindings_dir / "simulator.yaml"
        with binding_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                dict(simulator_bindings),
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        manifest["simulator_bindings"] = str(binding_path.relative_to(output))
    (output / "interface_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated agent YAMLs in the output directory.",
    )
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = validate_config(raw).to_dict()
    schema = _resolve_citylearn_schema_input(
        config["simulator"]["dataset_path"],
        config["simulator"],
    )
    if not isinstance(schema, Mapping):
        schema = {}
    env = _build_env(config)
    payload, _ = env.reset()
    interfaces, coverage = generate(env.entity_specs, payload, schema)
    future_bindings: Mapping[tuple[str, str], str] = {}
    if schema:
        registered = dict(schema.get("buildings", {}) or {})
        for agent_id, building in registered.items():
            if agent_id in interfaces:
                continue
            donor_id = next(
                (
                    candidate
                    for candidate, candidate_building in registered.items()
                    if candidate in interfaces
                    and bool(candidate_building.get("electrical_storage"))
                    == bool(building.get("electrical_storage"))
                    and bool(candidate_building.get("pv")) == bool(building.get("pv"))
                    and len(candidate_building.get("chargers", {}))
                    == len(building.get("chargers", {}))
                    and len(candidate_building.get("deferrable_appliances", {}))
                    == len(building.get("deferrable_appliances", {}))
                ),
                next(iter(interfaces)),
            )
            cloned = deepcopy(interfaces[donor_id])
            cloned["agent"]["id"] = str(agent_id)
            cloned["description"] = f"Deployment-neutral typed interface for {agent_id}"
            interfaces[str(agent_id)] = cloned
        future_bindings = augment_dynamic_assets(
            interfaces,
            schema,
            env.entity_specs,
        )
    write_generated(
        Path(args.output).expanduser().resolve(),
        interfaces,
        coverage,
        source=str(config["simulator"]["dataset_path"]),
        simulator_bindings=generate_simulator_bindings(
            env.entity_specs,
            payload,
            interfaces,
            future_bindings,
        ),
        overwrite=args.overwrite,
    )
    print(
        f"generated {len(interfaces)} typed agent interfaces and "
        f"classified {len(coverage)} fields in {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
