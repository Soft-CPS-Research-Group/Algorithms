"""Single source of truth for the offline-RL dataset schema.

Every component (collector, loader, trainer, reward calibrator) imports
column names from this module. **No string literals for column names elsewhere.**

The schema is *fixed across behaviour-policy swaps*: when we replace the RBC
with BC / IQL / etc., only the **values** in ``ACTION_COLUMNS``,
``REWARD_COLUMNS`` and the ``next_obs_*`` mirror change. Column names, order,
dtypes are invariant.
"""

from __future__ import annotations

from typing import List

import pyarrow as pa

# ---------------------------------------------------------------------------
# Building under study
# ---------------------------------------------------------------------------

TARGET_BUILDING_INDEX: int = 4
TARGET_BUILDING_NAME: str = "Building_5"
CHARGER_ID: str = "charger_5_1"

# ---------------------------------------------------------------------------
# Observation columns (Building 5 raw obs, 35 fields)
# ---------------------------------------------------------------------------
# Sourced from the env's ``observation_names[TARGET_BUILDING_INDEX]`` under
# ``interface='flat'``. Order matters — the collector validates this.

OBSERVATION_NAMES: List[str] = [
    "month",
    "day_type",
    "hour",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "outdoor_relative_humidity",
    "outdoor_relative_humidity_predicted_1",
    "outdoor_relative_humidity_predicted_2",
    "outdoor_relative_humidity_predicted_3",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_1",
    "diffuse_solar_irradiance_predicted_2",
    "diffuse_solar_irradiance_predicted_3",
    "direct_solar_irradiance",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "carbon_intensity",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "electricity_pricing_predicted_3",
    f"electric_vehicle_charger_{CHARGER_ID}_connected_state",
    f"connected_electric_vehicle_at_charger_{CHARGER_ID}_departure_time",
    f"connected_electric_vehicle_at_charger_{CHARGER_ID}_required_soc_departure",
    f"connected_electric_vehicle_at_charger_{CHARGER_ID}_soc",
    f"connected_electric_vehicle_at_charger_{CHARGER_ID}_battery_capacity",
    f"electric_vehicle_charger_{CHARGER_ID}_incoming_state",
    f"incoming_electric_vehicle_at_charger_{CHARGER_ID}_estimated_arrival_time",
]

# ---------------------------------------------------------------------------
# Action columns (Building 5: stationary battery + EV charger)
# ---------------------------------------------------------------------------

ACTION_NAMES: List[str] = [
    "electrical_storage",
    f"electric_vehicle_storage_{CHARGER_ID}",
]

# ---------------------------------------------------------------------------
# Bookkeeping columns (prefix-free)
# ---------------------------------------------------------------------------

BOOKKEEPING_COLUMNS: List[str] = ["episode", "timestep", "seed", "policy_mode"]

# ---------------------------------------------------------------------------
# Reward columns
# ---------------------------------------------------------------------------
# - reward_env: env's V2GPenaltyReward output for this building (traceability).
# - reward is computed by the reward module on top of this dataset and
#   written into the *derived* dataset, not by the collector. Including the
#   column in the schema keeps loaders schema-stable.

REWARD_COLUMNS: List[str] = ["reward_env", "reward"]

# ---------------------------------------------------------------------------
# Termination columns
# ---------------------------------------------------------------------------

TERMINATION_COLUMNS: List[str] = ["terminated", "truncated"]


# ---------------------------------------------------------------------------
# Full ordered column list
# ---------------------------------------------------------------------------


def obs_column(name: str) -> str:
    return f"obs_{name}"


def next_obs_column(name: str) -> str:
    return f"next_obs_{name}"


def action_column(name: str) -> str:
    return f"action_{name}"


OBS_COLUMNS: List[str] = [obs_column(n) for n in OBSERVATION_NAMES]
NEXT_OBS_COLUMNS: List[str] = [next_obs_column(n) for n in OBSERVATION_NAMES]
ACTION_COLUMNS: List[str] = [action_column(n) for n in ACTION_NAMES]


def all_columns() -> List[str]:
    return (
        list(BOOKKEEPING_COLUMNS)
        + list(OBS_COLUMNS)
        + list(ACTION_COLUMNS)
        + list(REWARD_COLUMNS)
        + list(NEXT_OBS_COLUMNS)
        + list(TERMINATION_COLUMNS)
    )


# ---------------------------------------------------------------------------
# PyArrow schema
# ---------------------------------------------------------------------------
# Float32 for all continuous values (fine for ML; halves disk vs float64).
# Booleans stored as uint8 (0/1) for trivial Parquet round-trip.

_FLOAT = pa.float32()
_INT8 = pa.int8()
_INT32 = pa.int32()
_UINT8 = pa.uint8()
_STRING = pa.string()


def _obs_field(name: str) -> pa.Field:
    if name in {"month", "day_type", "hour"}:
        return pa.field(obs_column(name), _INT8)
    if name.endswith("_connected_state") or name.endswith("_incoming_state"):
        return pa.field(obs_column(name), _UINT8)
    return pa.field(obs_column(name), _FLOAT)


def _next_obs_field(name: str) -> pa.Field:
    return pa.field(
        next_obs_column(name),
        _obs_field(name).type,
    )


def build_arrow_schema() -> pa.Schema:
    fields: List[pa.Field] = [
        pa.field("episode", _INT32),
        pa.field("timestep", _INT32),
        pa.field("seed", _INT32),
        pa.field("policy_mode", _STRING),
    ]
    fields.extend(_obs_field(n) for n in OBSERVATION_NAMES)
    fields.extend(pa.field(action_column(n), _FLOAT) for n in ACTION_NAMES)
    fields.append(pa.field("reward_env", _FLOAT))
    fields.append(pa.field("reward", _FLOAT))
    fields.extend(_next_obs_field(n) for n in OBSERVATION_NAMES)
    fields.append(pa.field("terminated", _UINT8))
    fields.append(pa.field("truncated", _UINT8))
    return pa.schema(fields)


def schema_hash() -> str:
    """Stable hash of the schema (fields + dtypes)."""
    import hashlib

    spec = "|".join(f"{f.name}:{f.type}" for f in build_arrow_schema())
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class SchemaError(RuntimeError):
    """Raised when collected data doesn't match the schema."""


def validate_observation_names(env_observation_names: List[str]) -> None:
    """Assert the env's per-building obs names match what we expect."""
    expected = list(OBSERVATION_NAMES)
    actual = list(env_observation_names)
    if actual != expected:
        diff_added = [n for n in actual if n not in expected]
        diff_removed = [n for n in expected if n not in actual]
        raise SchemaError(
            "Building 5 observation names do not match the schema.\n"
            f"  env-only fields:    {diff_added}\n"
            f"  schema-only fields: {diff_removed}\n"
            "Either the env changed or the schema is stale."
        )


def validate_action_names(env_action_names: List[str]) -> None:
    expected = list(ACTION_NAMES)
    actual = list(env_action_names)
    if actual != expected:
        raise SchemaError(
            f"Building 5 action names mismatch. expected={expected} got={actual}"
        )
