"""Validate and identify local simulator inputs before constructing learners.

Absolute dates/units cannot be recovered from anonymous numeric columns. The
report records the CityLearn positional/unit convention and marks absent calendar
metadata explicitly; an optional data contract makes those expectations strict.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.runtime.atomic import atomic_json, file_hash

TIME_SERIES = {"energy_simulation", "weather", "pricing", "carbon_intensity", "charger_simulation"}
FILE_KEYS = TIME_SERIES | {
    "filename",
    "parameters_filename",
    "cycle_profiles_file",
    "flexibility_schedule_file",
    "events_file",
    "schedule_file",
}
REQUIRED_FINITE = {
    "month",
    "hour",
    "day_type",
    "non_shiftable_load",
    "solar_generation",
    "dhw_demand",
    "cooling_demand",
    "heating_demand",
    "outdoor_dry_bulb_temperature",
    "outdoor_relative_humidity",
    "diffuse_solar_irradiance",
    "direct_solar_irradiance",
    "electricity_pricing",
    "carbon_intensity",
    "electric_vehicle_charger_state",
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate dataset schema key/identity: {key}")
        result[key] = value
    return result


def references(value, prefix=""):
    if isinstance(value, dict):
        if value.get("include") is False or value.get("enabled") is False:
            return
        for key, child in value.items():
            label = f"{prefix}.{key}" if prefix else key
            if (
                isinstance(child, str)
                and child
                and (key in FILE_KEYS or key.endswith(("_filename", "_filepath", "_file")))
            ):
                yield label, key, child
            elif isinstance(child, (dict, list)):
                yield from references(child, label)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from references(child, f"{prefix}[{index}]")


def validate_table(path, *, required_end=None, seconds=3600, contract=None):
    contract = contract or {}
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            columns = next(csv.reader(handle))
        if len(set(columns)) != len(columns):
            raise ValueError(f"Duplicate columns in {path}")
        frame = pd.read_csv(path, nrows=None if required_end is None else required_end + 1)
    else:
        frame = pd.read_parquet(path)
        if required_end is not None:
            frame = frame.iloc[: required_end + 1]
    if required_end is not None and len(frame) <= required_end:
        raise ValueError(
            f"Dataset has a gap/short horizon: {path} has {len(frame)} rows, needs {required_end + 1}"
        )
    columns = list(frame.columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"Empty/duplicate columns in {path}")
    if contract.get("columns") is not None and columns != contract["columns"]:
        raise ValueError(f"Dataset columns/order mismatch: {path}")
    required_columns = (
        set(contract.get("required_finite", []))
        | set(contract.get("bounds", {}))
        | set(contract.get("units", {}) if isinstance(contract.get("units"), dict) else [])
    )
    if not required_columns.issubset(columns):
        raise ValueError(
            f"Missing columns required by data contract: {sorted(required_columns - set(columns))} in {path}"
        )
    null_counts = {}
    for name in columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError(f"Infinite dataset values in {path}:{name}")
            missing = int(np.isnan(values).sum())
            if missing:
                null_counts[name] = missing
                if name in REQUIRED_FINITE or name in contract.get("required_finite", []):
                    raise ValueError(f"Missing/non-finite dataset values in {path}:{name}")
            bounds = contract.get("bounds", {}).get(name)
            if bounds is None:
                bounds = {"month": [1, 12], "hour": [0, 24], "day_type": [1, 8]}.get(name)
            if bounds is not None:
                finite = values[np.isfinite(values)]
                if ((finite < bounds[0]) | (finite > bounds[1])).any():
                    raise ValueError(f"Dataset bounds violated in {path}:{name}")
        elif name in REQUIRED_FINITE or name in contract.get("required_finite", []):
            raise ValueError(f"Dataset column must be numeric: {path}:{name}")
    calendar = {"kind": "indexed", "timezone": None, "absolute_dates": "not_declared"}
    column = contract.get("timestamp_column")
    if column is None:
        column = next(
            (name for name in ("timestamp", "datetime", "date_time") if name in columns), None
        )
    if column:
        if column not in columns:
            raise ValueError(f"Missing timestamp column {column} in {path}")
        dates = pd.to_datetime(frame[column], errors="raise")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f"Missing/duplicate/unordered timestamps in {path}")
        timezone = str(dates.dt.tz) if dates.dt.tz is not None else None
        if contract.get("timezone") is not None and timezone != contract["timezone"]:
            raise ValueError(f"Timezone mismatch in {path}: {timezone}")
        if len(dates) > 1 and not (dates.diff().iloc[1:].dt.total_seconds() == seconds).all():
            raise ValueError(f"Timestamp gaps or resolution mismatch in {path}")
        calendar = {
            "kind": "timestamp",
            "timezone": timezone,
            "start": str(dates.iloc[0]),
            "end": str(dates.iloc[-1]),
        }
    elif contract.get("timezone") is not None:
        raise ValueError(f"Cannot validate timezone without timestamps: {path}")
    return {
        "columns": columns,
        "rows_checked": len(frame),
        "nullable_columns": null_counts,
        "calendar": calendar,
        "units": contract.get(
            "units", "CityLearn schema conventions; no per-column unit metadata supplied"
        ),
    }


def validate_inputs(config, schema):
    if not isinstance(schema, dict):
        raise ValueError(
            "Jobs require an existing local JSON dataset schema; remote dataset names are not resolved implicitly"
        )
    simulator = config["simulator"]
    root = Path(schema.get("root_directory", ".")).resolve()
    seconds = float(schema.get("seconds_per_time_step", 3600))
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Dataset seconds_per_time_step must be finite and positive")
    start = simulator.get("simulation_start_time_step")
    end = simulator.get("simulation_end_time_step")
    start = schema.get("simulation_start_time_step", 0) if start is None else start
    end = schema.get("simulation_end_time_step") if end is None else end
    contract_path = simulator.get("data_contract_path")
    contracts = json.loads(Path(contract_path).read_text()) if contract_path else {}
    records, checked = [], {}
    declared_references = list(references(schema))
    future_building_ids = []
    if simulator.get("topology_mode", "static") == "dynamic":
        # Initially excluded members can be loaded by later events or supply
        # equipment to active members. Their declared assets are also inputs.
        referenced_members = {
            event[key]
            for event in schema.get("topology_events", [])
            for key in ("target_member_id", "source_member_id")
            if isinstance(event, dict) and event.get(key)
        }
        for member_id, member in schema.get("buildings", {}).items():
            if member_id in referenced_members and member.get("include") is False:
                future_building_ids.append(member_id)
                declared_references.extend(
                    references({**member, "include": True}, f"buildings.{member_id}")
                )
    for label, kind, filename in declared_references:
        path = (root / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset/auxiliary input {label}: {path}")
        if path not in checked:
            record = {
                "path": str(path),
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
                "references": [],
            }
            if path.suffix.lower() in {".csv", ".parquet", ".pq"}:
                # Dynamic member windows can be shorter than the district horizon;
                # alignment/identity rules remain owned by the Simulator loader.
                required_end = (
                    end
                    if kind in TIME_SERIES and simulator.get("topology_mode", "static") == "static"
                    else None
                )
                if simulator.get("terminal_observation_padding") and required_end is not None:
                    required_end -= 1
                record["table"] = validate_table(
                    path,
                    required_end=required_end,
                    seconds=seconds,
                    contract=contracts.get("files", {}).get(filename),
                )
            checked[path] = record
            records.append(record)
        checked[path]["references"].append(label)
    for key in ("dataset_path", "electrical_service_overrides_path", "data_contract_path"):
        value = simulator.get(key)
        if value and Path(value).is_file():
            path = Path(value).resolve()
            records.append(
                {
                    "path": str(path),
                    "sha256": file_hash(path),
                    "bytes": path.stat().st_size,
                    "references": [f"simulator.{key}"],
                }
            )
    # Agent-specific input assets share the same integrity record. Output paths
    # (e.g. traces) are deliberately not treated as inputs.
    asset_keys = {
        "tokenizer_config_path",
        "typed_interfaces_dir",
        "simulator_bindings_path",
        "model_dir",
        "dataset_path",
        "schedule_path",
        "bc_oracle_schedule_path",
    }

    def agent_assets(value, label):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in asset_keys and isinstance(child, str) and child:
                    yield f"{label}.{key}", Path(child).expanduser().resolve()
                elif isinstance(child, (dict, list)):
                    yield from agent_assets(child, f"{label}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from agent_assets(child, f"{label}[{index}]")

    for label, asset in agent_assets(config.get("pipeline", []), "pipeline"):
        if not asset.exists():
            raise FileNotFoundError(f"Missing agent input {label}: {asset}")
        files = (
            sorted(path for path in asset.rglob("*") if path.is_file())
            if asset.is_dir()
            else [asset]
        )
        for path in files:
            records.append(
                {
                    "path": str(path),
                    "sha256": file_hash(path),
                    "bytes": path.stat().st_size,
                    "references": [label],
                }
            )
    report = {
        "version": "algorithms.inputs.v1",
        "resolved_schema": schema,
        "seconds_per_time_step": seconds,
        "window": [start, end],
        "building_order": [
            name for name, item in schema.get("buildings", {}).items() if item.get("include", True)
        ],
        "files": records,
    }
    if future_building_ids:
        report["future_building_ids"] = future_building_ids
    report["identity_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    config.setdefault("runtime", {})["input_identity"] = report["identity_sha256"]
    directory = config.get("runtime", {}).get("job_dir")
    if directory:
        atomic_json(Path(directory) / "inputs.json", report)
    return report
