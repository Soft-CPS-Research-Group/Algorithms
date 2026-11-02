"""Reject simulator mechanisms that the static dataset MILPs do not model."""

from typing import Any, Mapping


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
