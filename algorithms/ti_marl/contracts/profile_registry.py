"""Deployment-neutral capability profiles for typed agent interfaces."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Dict, Mapping


PROFILE_VERSION = "ti_marl_capability_profiles_v1"

SUPPORTED_ROLES = ("consumer", "producer", "prosumer")
SUPPORTED_AGENT_TYPES = (
    "residential",
    "office",
    "commercial",
    "industrial",
    "other",
)
SUPPORTED_UNITS = {
    "A",
    "EUR/kWh",
    "boolean",
    "count",
    "fraction",
    "h",
    "index",
    "kW",
    "kWh",
    "scalar",
    "timestamp",
}

# Public channel names are semantic types, not deployment bindings. New
# instances may use these channels without changing the policy shape; adding a
# genuinely new channel kind is an explicit contract/model migration.
SUPPORTED_CHANNEL_TYPES = (
    "capability",
    "connection",
    "energy",
    "ev_session",
    "ev_state",
    "execution_feedback",
    "forecast",
    "generation",
    "grid",
    "market",
    "schedule",
    "service",
    "state",
    "storage_state",
    "time",
)

# These are deliberately instance-free. ``charger_1`` and ``charger_2`` use
# the same vocabulary and parameters, while load, PV and price remain
# distinguishable to the learned encoder.
OBSERVATION_SEMANTIC_TYPES = (
    "carbon_intensity",
    "community_energy",
    "community_ev_energy",
    "community_export",
    "community_flexibility",
    "community_grid_constraint",
    "community_import",
    "community_load_forecast",
    "community_net_exchange",
    "community_net_forecast",
    "community_pv_forecast",
    "community_pv_generation",
    "community_signal",
    "community_storage_energy",
    "community_topology",
    "deferrable_capability",
    "deferrable_schedule",
    "deferrable_state",
    "ev_capability",
    "ev_connection",
    "ev_execution",
    "ev_schedule",
    "ev_service",
    "ev_soc",
    "execution_feedback",
    "grid_headroom",
    "local_energy",
    "local_ev_energy",
    "local_export",
    "local_flexibility",
    "local_grid_constraint",
    "local_import",
    "local_load",
    "local_load_forecast",
    "local_net_exchange",
    "local_net_forecast",
    "local_pv_forecast",
    "local_pv_generation",
    "local_storage_energy",
    "market_price",
    "market_price_forecast",
    "phase_connection",
    "storage_capability",
    "storage_execution",
    "storage_soc",
    "storage_state",
    "time_context",
    "weather",
    "weather_forecast",
)

SENSOR_ENTITY_TYPES: Mapping[str, str] = {
    "building_meter": "building",
    "community_aggregate_service": "district",
    "stationary_battery": "storage",
    "bidirectional_ev_charger": "charger",
    "ev_charger": "charger",
    "ev_session": "ev",
    "deferrable_appliance": "deferrable_appliance",
    "pv_monitor": "pv",
}

ACTION_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "stationary_battery": {
        "group_type": "stationary_storage",
        "entity_type": "storage",
        "modes": ("IDLE", "CHARGE_STATIONARY", "DISCHARGE_STATIONARY"),
    },
    "bidirectional_ev_charger": {
        "group_type": "ev_session",
        "entity_type": "charger",
        "modes": ("IDLE", "CHARGE_EV", "DISCHARGE_EV"),
    },
    "ev_charger": {
        "group_type": "ev_session",
        "entity_type": "charger",
        "modes": ("IDLE", "CHARGE_EV"),
    },
    "deferrable_appliance": {
        "group_type": "deferrable",
        "entity_type": "deferrable_appliance",
        "modes": ("IDLE", "START"),
    },
}


def canonical_unit(raw: str | None, name: str) -> str:
    """Return a public unit vocabulary, preferring adapter metadata."""

    aliases = {
        "kw": "kW",
        "kwh": "kWh",
        "kwh_step": "kWh",
        "h": "h",
        "ratio": "fraction",
        "flag": "boolean",
        "eur_per_kwh": "EUR/kWh",
        "time_step": "timestamp",
        "count": "count",
        "index": "index",
        "scalar": "scalar",
    }
    if raw:
        return aliases.get(str(raw).strip().lower(), str(raw))
    key = str(name).lower()
    if key.endswith("_power_kw") or key.endswith("_kw") or "headroom_kw" in key:
        return "kW"
    if key.endswith("_kwh") or key.endswith("_kwh_step") or "energy" in key:
        return "kWh"
    if key.endswith("_hours") or key.startswith("hours_until_"):
        return "h"
    if key.endswith("_ratio") or key.endswith("_normalized") or key.endswith("_soc"):
        return "fraction"
    if key.endswith("_time_step"):
        return "timestamp"
    if key.endswith("_count"):
        return "count"
    if key.startswith("clip_reason_") or key.startswith("can_") or key.endswith("_state"):
        return "boolean"
    return "scalar"


def channel_for(entity_type: str, feature: str) -> str:
    key = str(feature).lower()
    if entity_type == "district":
        if "price" in key or "pricing" in key:
            return "market"
        if "headroom" in key or "phase" in key or "loading" in key:
            return "grid"
        if "forecast" in key or "predicted" in key:
            return "forecast"
        if key in {"month", "day_type", "hour", "minutes", "seconds", "is_weekend"} or key.endswith(("_sin", "_cos")):
            return "time"
        return "energy"
    if entity_type == "building":
        if "headroom" in key or "phase" in key or "violation" in key:
            return "grid"
        if "forecast" in key:
            return "forecast"
        if key in {"month", "day_type", "hour", "minutes", "seconds", "is_weekend"} or key.endswith(("_sin", "_cos")):
            return "time"
        return "energy"
    if entity_type == "charger":
        if key in {"connected_state", "incoming_state"} or "phase_connection" in key:
            return "connection"
        if "arrival" in key or "departure" in key or "slack" in key or "priority" in key:
            return "schedule"
        if "soc" in key or "battery_capacity" in key or "energy_to_" in key or "energy_available" in key:
            return "ev_state"
        if key.startswith("last_") or key.startswith("commanded_") or key.startswith("applied_") or "projection_error" in key or key.startswith("clip_reason_"):
            return "execution_feedback"
        return "capability"
    if entity_type == "storage":
        if key.startswith("last_") or key.startswith("applied_") or key.startswith("commanded_"):
            return "execution_feedback"
        return "storage_state"
    if entity_type == "ev":
        return "ev_session"
    if entity_type == "deferrable_appliance":
        return "schedule" if any(token in key for token in ("start", "deadline", "slack", "urgent", "pending", "running")) else "capability"
    if entity_type == "pv":
        return "generation"
    return "state"


def semantic_type_for(
    entity_type: str,
    channel: str,
    scope: str,
    observation: str,
) -> str:
    """Classify an observation into an explicit, instance-free meaning."""

    key = str(observation).lower()
    is_forecast = channel == "forecast" or any(
        token in key for token in ("forecast", "predicted", "next_")
    )

    if channel == "time":
        return "time_context"
    if "carbon_intensity" in key:
        return "carbon_intensity"
    if any(
        token in key
        for token in ("temperature", "humidity", "irradiance", "weather")
    ):
        return "weather_forecast" if is_forecast else "weather"
    if channel == "market" or "price" in key or "pricing" in key:
        return "market_price_forecast" if is_forecast else "market_price"

    if entity_type == "district" or scope == "community":
        if channel == "grid" or "headroom" in key or "phase" in key:
            return "community_grid_constraint"
        if is_forecast:
            if "pv" in key or "solar" in key:
                return "community_pv_forecast"
            if "load" in key or "demand" in key:
                return "community_load_forecast"
            return "community_net_forecast"
        if "active_" in key or key == "topology_version":
            return "community_topology"
        if "import" in key:
            return "community_import"
        if "export" in key:
            return "community_export"
        if "pv" in key or "solar" in key:
            return "community_pv_generation"
        if "bess" in key or "storage" in key:
            return "community_storage_energy"
        if "ev_" in key:
            return "community_ev_energy"
        if "flexible" in key or "flexibility" in key:
            return "community_flexibility"
        if "net" in key:
            return "community_net_exchange"
        return "community_energy"

    if entity_type == "building":
        if channel == "grid" or "headroom" in key or "violation" in key:
            return "local_grid_constraint"
        if is_forecast:
            if "pv" in key or "solar" in key:
                return "local_pv_forecast"
            if "load" in key or "demand" in key:
                return "local_load_forecast"
            return "local_net_forecast"
        if "import" in key:
            return "local_import"
        if "export" in key:
            return "local_export"
        if "pv" in key or "solar" in key:
            return "local_pv_generation"
        if "non_shiftable" in key or "load_" in key or key == "load":
            return "local_load"
        if "net" in key:
            return "local_net_exchange"
        if "bess" in key or "storage" in key:
            return "local_storage_energy"
        if "ev_" in key or "charging_total" in key:
            return "local_ev_energy"
        if "flexible" in key:
            return "local_flexibility"
        return "local_energy"

    if entity_type == "storage":
        if channel == "execution_feedback":
            return "storage_execution"
        if "soc" in key and not any(
            token in key for token in ("limit", "min_", "max_", "available")
        ):
            return "storage_soc"
        if any(
            token in key
            for token in (
                "available",
                "can_",
                "capacity",
                "efficiency",
                "headroom",
                "max_",
                "min_",
                "phase_connection",
            )
        ):
            return "storage_capability"
        return "storage_state"

    if entity_type in {"charger", "ev"}:
        if channel == "connection":
            return "ev_connection"
        if channel == "execution_feedback":
            return "ev_execution"
        if channel in {"schedule", "service"}:
            return "ev_schedule"
        if channel in {"ev_state", "ev_session"} and (
            "soc" in key or "energy_" in key or "battery_capacity" in key
        ):
            return "ev_soc"
        if channel == "capability":
            return "ev_capability"
        return "ev_service"

    if entity_type == "deferrable_appliance":
        if channel == "schedule":
            if any(
                token in key
                for token in ("deadline", "slack", "start", "urgent", "must_run")
            ):
                return "deferrable_schedule"
            return "deferrable_state"
        return "deferrable_capability"
    if entity_type == "pv":
        return "local_pv_generation"
    return "local_energy"


def default_use(entity_type: str, feature: str) -> tuple[str, bool, str | None]:
    """Classify a field without hiding any adapter-emitted observation."""

    key = str(feature).lower()
    trace_patterns = (
        r"^clip_reason_",
        r"^last_requested_",
        r"^last_limited_",
        r"^last_projection_error_",
    )
    if any(re.search(pattern, key) for pattern in trace_patterns):
        return "trace_only", False, "execution diagnostic, retained for audit"
    if key == "topology_version":
        return "trace_only", False, "structural metadata handled by the TIC"
    if any(token in key for token in ("available_charge", "available_discharge", "headroom", "can_start", "can_charge", "can_discharge")):
        return "runtime_bound", True, None
    if any(token in key for token in ("violation", "connected_state", "soc", "deadline", "departure", "running", "pending")):
        return "safety_dependency", True, None
    return "policy_input", True, None


def criticality_for(entity_type: str, feature: str, use: str) -> str:
    key = str(feature).lower()
    if use in {"runtime_bound", "safety_dependency"} or any(
        token in key for token in ("headroom", "phase", "soc", "connected", "deadline")
    ):
        return "safety"
    if entity_type == "district" or "forecast" in key or "price" in key:
        return "advisory"
    return "operational"


@dataclass(frozen=True)
class CapabilityProfileRegistry:
    """Known semantic families; it never contains deployment bindings."""

    version: str = PROFILE_VERSION

    @property
    def supported_sensor_types(self) -> tuple[str, ...]:
        return tuple(sorted(SENSOR_ENTITY_TYPES))

    @property
    def supported_action_types(self) -> tuple[str, ...]:
        return tuple(sorted(ACTION_PROFILES))

    @property
    def supported_channel_types(self) -> tuple[str, ...]:
        return tuple(sorted(SUPPORTED_CHANNEL_TYPES))

    @property
    def supported_semantic_types(self) -> tuple[str, ...]:
        return tuple(sorted(OBSERVATION_SEMANTIC_TYPES))

    def entity_type(self, sensor_type: str) -> str:
        try:
            return SENSOR_ENTITY_TYPES[str(sensor_type)]
        except KeyError as exc:
            raise ValueError(f"Unknown TI-MARL sensor type: {sensor_type!r}") from exc

    def action_profile(self, actuator_type: str) -> Mapping[str, Any]:
        try:
            return deepcopy(dict(ACTION_PROFILES[str(actuator_type)]))
        except KeyError as exc:
            raise ValueError(f"Unknown TI-MARL actuator type: {actuator_type!r}") from exc

    def observation_defaults(
        self,
        *,
        sensor_type: str,
        channel: str,
        observation: str,
        scope: str,
        unit: str | None = None,
    ) -> Dict[str, Any]:
        entity_type = self.entity_type(sensor_type)
        if channel not in SUPPORTED_CHANNEL_TYPES:
            raise ValueError(f"Unknown TI-MARL channel type: {channel!r}")
        use, policy_input, reason = default_use(entity_type, observation)
        result: Dict[str, Any] = {
            "unit": canonical_unit(unit, observation),
            "semantic_type": semantic_type_for(
                entity_type,
                channel,
                scope,
                observation,
            ),
            "use": use,
            "policy_input": policy_input,
            "criticality": criticality_for(entity_type, observation, use),
            "required": use in {"runtime_bound", "safety_dependency"},
            "normalisation": "signed_log1p",
        }
        if reason is not None:
            result["reason"] = reason
        return result

    def health_rules(self) -> Mapping[str, Any]:
        return {
            "version": "ti_marl_health_rules_v2",
            "defaults": {
                "degraded_after_seconds": 0.0,
                "stale_after_seconds": 7200.0,
                "missing_after_seconds": 14400.0,
                "recovery_hysteresis_seconds": 1800.0,
                "cache_allowed": True,
            },
            "criticality": {
                "safety": {
                    "stale_after_seconds": 900.0,
                    "missing_after_seconds": 1800.0,
                    "recovery_hysteresis_seconds": 900.0,
                },
                "operational": {},
                "advisory": {
                    "stale_after_seconds": 7200.0,
                    "missing_after_seconds": 14400.0,
                    "recovery_hysteresis_seconds": 900.0,
                },
            },
            "fault_modes": {
                "stuck": {"initial_state": "DEGRADED"},
                "noise": {"initial_state": "DEGRADED"},
                "bias": {"initial_state": "DEGRADED"},
                "clip": {"initial_state": "DEGRADED"},
                "delay": {"initial_state": "DEGRADED"},
                "missing": {"initial_state": "MISSING"},
                "dropout": {"initial_state": "MISSING"},
                "unavailable": {"initial_state": "FAILED"},
            },
        }

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "version": self.version,
            "roles": list(SUPPORTED_ROLES),
            "agent_types": list(SUPPORTED_AGENT_TYPES),
            "sensor_types": dict(SENSOR_ENTITY_TYPES),
            "channel_types": list(self.supported_channel_types),
            "semantic_types": list(self.supported_semantic_types),
            "action_profiles": deepcopy(dict(ACTION_PROFILES)),
            "health": deepcopy(dict(self.health_rules())),
        }
