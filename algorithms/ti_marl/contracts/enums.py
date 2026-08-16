"""Versioned enumerations used by the Typed Interface Compiler."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """``str`` enum compatible with Python versions before stdlib StrEnum."""

    def __str__(self) -> str:
        return self.value


class EventDomain(StrEnum):
    ASSET_CONNECTION = "ASSET_CONNECTION"
    ASSET_AVAILABILITY = "ASSET_AVAILABILITY"
    SENSOR_CHANNEL = "SENSOR_CHANNEL"
    ACTUATOR_CHANNEL = "ACTUATOR_CHANNEL"
    COMMUNICATION_LINK = "COMMUNICATION_LINK"
    VALUE_QUALITY = "VALUE_QUALITY"


class AvailabilityState(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class ConnectionState(StrEnum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


class QualityState(StrEnum):
    NOMINAL = "NOMINAL"
    IMPAIRED = "IMPAIRED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class HealthState(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    MISSING = "MISSING"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


HEALTH_SEVERITY = {
    HealthState.HEALTHY: 0,
    HealthState.DEGRADED: 1,
    HealthState.STALE: 2,
    HealthState.MISSING: 3,
    HealthState.FAILED: 4,
    HealthState.UNKNOWN: 5,
}
