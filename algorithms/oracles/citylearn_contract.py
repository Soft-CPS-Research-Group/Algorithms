"""Reject simulator mechanisms that the static dataset MILPs do not model."""

from typing import Any, Mapping


def action_inventory(schema: Mapping[str, Any], building_ids) -> list[dict[str, str]]:
    """Keep idle devices explicit even when a window contains no service event."""
    inventory = []
    for building_id in building_ids:
        building = schema["buildings"][building_id]
        names = []
        if building.get("electrical_storage"):
            names.append(("electrical_storage", "kW", "charge"))
        names.extend(
            (f"electric_vehicle_storage_{key}", "kW", "charge")
            for key in (building.get("chargers") or {})
        )
        names.extend(
            (f"deferrable_appliance_{key}", "normalized_action", "start")
            for key in (building.get("deferrable_appliances") or {})
        )
        for name, unit, direction in names:
            inventory.append(
                dict(
                    building_id=building_id,
                    action_name=name,
                    unit=unit,
                    positive_direction=direction,
                )
            )
    return inventory


def complete_idle_series(series, inventory, horizon):
    from algorithms.oracles.perfect_foresight_milp import SemanticActionSeries

    result = list(series)
    present = {(item.building_id, item.action_name) for item in result}
    for item in inventory:
        key = item["building_id"], item["action_name"]
        if key not in present:
            result.append(SemanticActionSeries(**item, values=(0.0,) * horizon))
            present.add(key)
    return tuple(result)


def validate_static_oracle_schema(schema: Mapping[str, Any]) -> None:
    if schema.get("topology_mode", "static") != "static" or schema.get("topology_events"):
        raise ValueError("Dataset MILP requires static topology; topology events are not modeled.")
    robustness = schema.get("robustness") or {}
    if robustness.get("enabled", False):
        raise ValueError("Dataset MILP does not model runtime robustness disturbances.")
    for name, building in (schema.get("buildings") or {}).items():
        if not building.get("include", True):
            continue
        if (building.get("power_outage") or {}).get("simulate_power_outage", False):
            raise ValueError(f"Dataset MILP does not model power outages for {name}.")
        # The optimizer schedules every configured flexible device. Optimizing
        # one whose action is disabled would certify a plan the replay cannot
        # enact, even if the device happens to have no event in this window.
        controls = (
            ("electrical_storage", "actions", "electrical_storage"),
            ("chargers", "chargers_actions_helper", "electric_vehicle_storage"),
            (
                "deferrable_appliances",
                "deferrable_appliance_actions_helper",
                "deferrable_appliance",
            ),
        )
        inactive = set(building.get("inactive_actions") or [])
        for asset_key, table, control in controls:
            if not building.get(asset_key):
                continue
            if (schema.get(table, {}).get(control, {}) or {}).get("active") is False or any(
                action == control or action.startswith(control + "_") for action in inactive
            ):
                raise ValueError(f"Dataset MILP requires enabled {control} actions for {name}.")
