"""Build a total-energy MILP problem directly from a CityLearn dataset.

The adapter reads exogenous demand, absolute PV generation, prices and flexible
asset service windows from the dataset files rather than from a controller
replay.  Source windows use the half-open convention ``[start, end)`` while
the :class:`~algorithms.oracles.total_energy_milp.EVSession` contract uses an
inclusive final controllable step.

Boundary service is deliberately visible.  A connected EV interval cut by the
right edge is retained without an artificial departure target.  An interval
cut by the left edge is retained, but its initial SOC is marked as assumed
unless the charger file provides ``electric_vehicle_current_soc``.  Feasible
deferrable start windows are intersected with the requested horizon and every
restriction or omission is reported in the build diagnostics.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from algorithms.oracles.citylearn_ev import deterministic_ev_initial_soc
from algorithms.oracles.total_energy_milp import (
    DeferrableCycle,
    ElectricalService,
    EVSession,
    StorageAsset,
    TotalEnergyProblem,
)


_PHASES = ("L1", "L2", "L3")
_RUNTIME_DRIFT_INITIAL_SOC_REASON = (
    "Connection occurs after episode reset without electric_vehicle_current_soc "
    "or an explicit arrival SOC. CityLearn evolves unconnected EV SOC at runtime, "
    "so the static schema initial SOC fallback cannot reproduce that state."
)


@dataclass(frozen=True)
class CityLearnTotalEnergyDiagnostics:
    """Extraction counts and exactness qualifications for a source window."""

    schema_path: str
    dataset_root: str
    source_start_time_step: int
    source_end_time_step_exclusive: int
    horizon: int
    building_count: int
    stationary_storage_count: int
    charger_count: int
    ev_session_count: int
    deferrable_cycle_count: int
    electrical_service_count: int
    left_truncated_ev_session_ids: tuple[str, ...]
    right_truncated_ev_session_ids: tuple[str, ...]
    assumed_initial_soc_ev_session_ids: tuple[str, ...]
    runtime_drift_initial_soc_ev_session_ids: tuple[str, ...]
    restricted_deferrable_cycle_ids: tuple[str, ...]
    omitted_boundary_deferrable_cycle_ids: tuple[str, ...]
    electrical_service_reserve_kw: float

    @property
    def boundary_service_exact(self) -> bool:
        """Whether no flexible-service decision was approximated at an edge."""

        return not any(
            (
                self.left_truncated_ev_session_ids,
                self.right_truncated_ev_session_ids,
                self.assumed_initial_soc_ev_session_ids,
                self.runtime_drift_initial_soc_ev_session_ids,
                self.restricted_deferrable_cycle_ids,
                self.omitted_boundary_deferrable_cycle_ids,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_path": self.schema_path,
            "dataset_root": self.dataset_root,
            "source_start_time_step": self.source_start_time_step,
            "source_end_time_step_exclusive": self.source_end_time_step_exclusive,
            "horizon": self.horizon,
            "building_count": self.building_count,
            "stationary_storage_count": self.stationary_storage_count,
            "charger_count": self.charger_count,
            "ev_session_count": self.ev_session_count,
            "deferrable_cycle_count": self.deferrable_cycle_count,
            "electrical_service_count": self.electrical_service_count,
            "left_truncated_ev_session_ids": list(
                self.left_truncated_ev_session_ids
            ),
            "right_truncated_ev_session_ids": list(
                self.right_truncated_ev_session_ids
            ),
            "assumed_initial_soc_ev_session_ids": list(
                self.assumed_initial_soc_ev_session_ids
            ),
            "runtime_drift_initial_soc_ev_session_ids": list(
                self.runtime_drift_initial_soc_ev_session_ids
            ),
            "runtime_drift_initial_soc_reason": _RUNTIME_DRIFT_INITIAL_SOC_REASON,
            "restricted_deferrable_cycle_ids": list(
                self.restricted_deferrable_cycle_ids
            ),
            "omitted_boundary_deferrable_cycle_ids": list(
                self.omitted_boundary_deferrable_cycle_ids
            ),
            "electrical_service_reserve_kw": self.electrical_service_reserve_kw,
            "boundary_service_exact": self.boundary_service_exact,
        }


@dataclass(frozen=True)
class CityLearnTotalEnergyBuild:
    problem: TotalEnergyProblem
    diagnostics: CityLearnTotalEnergyDiagnostics


def _non_negative(name: str, value: Any, default: float | None = None) -> float:
    if value is None and default is not None:
        value = default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric; got {value!r}.") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")
    return parsed


def _efficiency(name: str, value: Any, default: float) -> float:
    parsed = _non_negative(name, value, default)
    if parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in (0, 1]; got {value!r}.")
    return parsed


def _soc(value: Any, *, default: float | None = None) -> float:
    if value is None or pd.isna(value):
        if default is None:
            raise ValueError("Missing SOC value.")
        return float(default)
    parsed = float(value)
    if parsed > 1.0:
        parsed /= 100.0
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"SOC must be in [0, 1] or [0, 100]; got {value!r}.")
    return parsed


def _dataset_root(schema_path: Path, schema: Mapping[str, Any]) -> Path:
    root_value = Path(str(schema.get("root_directory") or schema_path.parent))
    candidates = [root_value] if root_value.is_absolute() else [
        Path.cwd() / root_value,
        schema_path.parent / root_value,
        schema_path.parent,
    ]
    buildings = schema.get("buildings") or {}
    sample_file = next(
        (
            str(building["energy_simulation"])
            for building in buildings.values()
            if building.get("energy_simulation")
        ),
        None,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and (
            sample_file is None or (resolved / sample_file).is_file()
        ):
            return resolved
    raise FileNotFoundError(
        f"Could not resolve root_directory for CityLearn schema {schema_path}."
    )


def _read_parquet(root: Path, filename: Any) -> pd.DataFrame:
    path = (root / str(filename)).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _numeric_series(frame: pd.DataFrame, column: str, *, source: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"{source} is missing required column {column!r}.")
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{source}.{column} contains non-finite values.")
    return values


def _parse_profile(raw: Any) -> tuple[float, ...]:
    if isinstance(raw, np.ndarray):
        values = raw.astype(np.float64).reshape(-1)
    elif isinstance(raw, (list, tuple)):
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
    else:
        parsed = ast.literal_eval(str(raw))
        values = np.asarray(
            parsed if isinstance(parsed, (list, tuple)) else [parsed],
            dtype=np.float64,
        ).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(
            "Deferrable load profiles must contain finite non-negative kWh values."
        )
    return tuple(float(value) for value in values)


def _connected_intervals(frame: pd.DataFrame) -> list[tuple[int, int, str]]:
    """Return maximal ``state == 1`` intervals as ``[start, end)``."""

    required = {"electric_vehicle_charger_state", "electric_vehicle_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Charger simulation is missing columns: {sorted(missing)}.")
    states = pd.to_numeric(
        frame["electric_vehicle_charger_state"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    identifiers = frame["electric_vehicle_id"].fillna("").astype(str).to_numpy()
    connected = states == 1.0
    intervals: list[tuple[int, int, str]] = []
    cursor = 0
    while cursor < len(frame):
        if not connected[cursor]:
            cursor += 1
            continue
        identifier = identifiers[cursor].strip()
        if not identifier or identifier.lower() == "nan":
            raise ValueError(f"Connected charger row {cursor} has no EV id.")
        end = cursor + 1
        while (
            end < len(frame)
            and connected[end]
            and identifiers[end].strip() == identifier
        ):
            end += 1
        intervals.append((cursor, end, identifier))
        cursor = end
    return intervals


def _curve_minimum(name: str, raw: Any, *, default: float) -> float:
    if raw is None:
        return float(default)
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 array.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return max(float(np.min(array[:, 1])), 0.0)


def _minimum_directional_battery_efficiency(
    name: str, attributes: Mapping[str, Any]
) -> float:
    technical = _efficiency(f"{name}.efficiency", attributes.get("efficiency"), 0.9)
    minimum_technical = _curve_minimum(
        f"{name}.power_efficiency_curve",
        attributes.get("power_efficiency_curve"),
        default=technical * 0.85,
    )
    if not 0.0 < minimum_technical <= 1.0:
        raise ValueError(f"{name} minimum technical efficiency must be in (0, 1].")
    return 0.99 * math.sqrt(minimum_technical)


def _phase(value: Any) -> str:
    if value is None:
        return "all_phases"
    normalized = str(value).strip()
    lookup = {"l1": "L1", "l2": "L2", "l3": "L3", "all_phases": "all_phases"}
    try:
        return lookup[normalized.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported phase_connection {value!r}.") from error


def _stationary_asset(
    building_id: str,
    building: Mapping[str, Any],
    *,
    conservative_capacity_ratio: float,
) -> StorageAsset | None:
    storage = building.get("electrical_storage")
    if not storage:
        return None
    if bool(storage.get("autosize", False)):
        raise ValueError("Autosized stationary batteries are not supported.")
    attributes = dict(storage.get("attributes") or {})
    capacity = _non_negative(
        f"{building_id}.electrical_storage.capacity", attributes.get("capacity")
    )
    nominal = _non_negative(
        f"{building_id}.electrical_storage.nominal_power",
        attributes.get("nominal_power"),
    )
    depth = _non_negative(
        f"{building_id}.electrical_storage.depth_of_discharge",
        attributes.get("depth_of_discharge"),
        1.0,
    )
    if depth > 1.0:
        raise ValueError("Stationary battery depth_of_discharge must be <= 1.")
    minimum = capacity * (1.0 - depth)
    initial_soc = _soc(
        attributes.get("initial_soc"), default=(1.0 - depth)
    )
    initial = max(capacity * initial_soc, minimum)
    conservative_capacity = max(
        capacity * conservative_capacity_ratio, initial, minimum
    )
    minimum_power_ratio = _curve_minimum(
        f"{building_id}.electrical_storage.capacity_power_curve",
        attributes.get("capacity_power_curve"),
        default=0.20,
    )
    return StorageAsset(
        building_id=building_id,
        action_name="electrical_storage",
        capacity_kwh=conservative_capacity,
        initial_energy_kwh=initial,
        final_energy_min_kwh=initial,
        minimum_energy_kwh=minimum,
        max_charge_kw=nominal * minimum_power_ratio,
        max_discharge_kw=nominal * minimum_power_ratio,
        charge_efficiency=_minimum_directional_battery_efficiency(
            f"{building_id}.electrical_storage", attributes
        ),
        discharge_efficiency=_minimum_directional_battery_efficiency(
            f"{building_id}.electrical_storage", attributes
        ),
        phase_connection=_phase(attributes.get("phase_connection")),
    )


def _reserved_limit(value: Any, *, reserve_kw: float, name: str) -> float | None:
    if value is None:
        return None
    parsed = _non_negative(name, value)
    if reserve_kw > parsed:
        raise ValueError(f"reserve_kw={reserve_kw} exceeds {name}={parsed}.")
    return parsed - reserve_kw


def _electrical_service(
    building_id: str,
    building: Mapping[str, Any],
    *,
    reserve_kw: float,
) -> ElectricalService | None:
    config = building.get("electrical_service")
    if not config:
        return None
    limits = config.get("limits") or {}
    total = limits.get("total") or {}
    per_phase = limits.get("per_phase") or {}
    phase_import: dict[str, float | None] = {}
    phase_export: dict[str, float | None] = {}
    for phase in _PHASES:
        phase_limits = per_phase.get(phase) or per_phase.get(phase.lower()) or {}
        phase_import[phase] = _reserved_limit(
            phase_limits.get("import_kw"),
            reserve_kw=reserve_kw,
            name=f"{building_id}.electrical_service.{phase}.import_kw",
        )
        phase_export[phase] = _reserved_limit(
            phase_limits.get("export_kw"),
            reserve_kw=reserve_kw,
            name=f"{building_id}.electrical_service.{phase}.export_kw",
        )
    return ElectricalService(
        building_id=building_id,
        total_import_kw=_reserved_limit(
            total.get("import_kw"),
            reserve_kw=reserve_kw,
            name=f"{building_id}.electrical_service.total.import_kw",
        ),
        total_export_kw=_reserved_limit(
            total.get("export_kw"),
            reserve_kw=reserve_kw,
            name=f"{building_id}.electrical_service.total.export_kw",
        ),
        phase_import_kw=phase_import,
        phase_export_kw=phase_export,
        default_split=str(config.get("default_split", "balanced")),
    )


def _arrival_soc(
    frame: pd.DataFrame,
    session_start: int,
    *,
    ev_id: str,
    default_soc: float,
    default_source: str,
    episode_start: int,
) -> tuple[float, str]:
    if "electric_vehicle_current_soc" in frame:
        candidate = frame.iloc[session_start]["electric_vehicle_current_soc"]
        if pd.notna(candidate):
            return _soc(candidate, default=default_soc), "current_soc"
    arrival_column = "electric_vehicle_estimated_soc_arrival"
    # A non-zero CityLearn episode is reset at its local time step zero.  The
    # original dataset row immediately before the window is not loaded into
    # that episode and therefore cannot seed its EV battery.  Mirror the
    # runtime contract: use an explicit value on the current row when
    # available, otherwise retain the effective schema initial SOC.
    if session_start == episode_start and episode_start > 0:
        if arrival_column in frame:
            candidate = frame.iloc[session_start][arrival_column]
            if pd.notna(candidate):
                return (
                    _soc(candidate, default=default_soc),
                    "arrival_at_episode_start",
                )
        return float(default_soc), f"{default_source}_episode_reset"
    if arrival_column in frame:
        if session_start > 0:
            previous = frame.iloc[session_start - 1]
            previous_id = str(previous.get("electric_vehicle_id", "")).strip()
            if previous_id == ev_id and pd.notna(previous[arrival_column]):
                return _soc(previous[arrival_column], default=default_soc), "incoming_previous"
        candidate = frame.iloc[session_start][arrival_column]
        if pd.notna(candidate):
            return _soc(candidate, default=default_soc), "arrival_at_start"
    return float(default_soc), str(default_source)


def build_citylearn_total_energy_problem(
    *,
    schema_path: Path | str,
    start_time_step: int,
    end_time_step: int,
    problem_id: str | None = None,
    settlement: Literal["individual", "community"] = "individual",
    building_ids: Sequence[str] | None = None,
    electrical_service_reserve_kw: float = 0.1,
    conservative_capacity_ratio: float = 0.99,
) -> CityLearnTotalEnergyBuild:
    """Extract a total-energy problem for source transitions ``[start, end)``.

    ``start_time_step=0`` is the only origin for which a session already
    connected at the left edge has an exact static initial condition.  Later
    left-edge cuts need replay state (or an optional current-SOC dataset
    column); the diagnostics never label the fallback as exact.
    """

    schema_path = Path(schema_path).resolve()
    raw_schema = schema_path.read_bytes()
    schema = json.loads(raw_schema)
    dataset_root = _dataset_root(schema_path, schema)
    start, end = int(start_time_step), int(end_time_step)
    if start < 0 or end <= start:
        raise ValueError("Source window must satisfy 0 <= start_time_step < end_time_step.")
    timestep_hours = _non_negative(
        "seconds_per_time_step", schema.get("seconds_per_time_step")
    ) / 3600.0
    if timestep_hours <= 0.0:
        raise ValueError("seconds_per_time_step must be > 0.")
    reserve_kw = _non_negative(
        "electrical_service_reserve_kw", electrical_service_reserve_kw
    )
    capacity_ratio = _non_negative(
        "conservative_capacity_ratio", conservative_capacity_ratio
    )
    if not 0.0 < capacity_ratio <= 1.0:
        raise ValueError("conservative_capacity_ratio must be in (0, 1].")

    included = {
        str(name): data
        for name, data in (schema.get("buildings") or {}).items()
        if bool(data.get("include", True))
    }
    if building_ids is None:
        selected = tuple(included)
    else:
        selected = tuple(str(value) for value in building_ids)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("building_ids must be non-empty and unique when supplied.")
        unknown = [value for value in selected if value not in included]
        if unknown:
            raise ValueError(f"Unknown or excluded building_ids: {unknown}.")
    if not selected:
        raise ValueError("Schema contains no included buildings.")

    prices: np.ndarray | None = None
    base_rows: list[np.ndarray] = []
    stationary: list[StorageAsset] = []
    sessions: list[EVSession] = []
    cycles: list[DeferrableCycle] = []
    services: list[ElectricalService] = []
    charger_count = 0
    left_ev: list[str] = []
    right_ev: list[str] = []
    assumed_ev: list[str] = []
    runtime_drift_ev: list[str] = []
    restricted_cycles: list[str] = []
    omitted_cycles: list[str] = []
    ev_initial_sources: dict[str, str] = {}
    ev_definitions = schema.get("electric_vehicles_def") or {}

    for building_id in selected:
        building = included[building_id]
        energy = _read_parquet(dataset_root, building.get("energy_simulation"))
        pricing = _read_parquet(dataset_root, building.get("pricing"))
        if end > len(energy) or end > len(pricing):
            raise ValueError(
                f"Requested source window exceeds data for {building_id}."
            )
        non_shiftable = _numeric_series(
            energy, "non_shiftable_load", source=f"{building_id}.energy_simulation"
        )
        solar = _numeric_series(
            energy, "solar_generation", source=f"{building_id}.energy_simulation"
        )
        pv = building.get("pv") or {}
        generation_mode = str(
            (pv.get("attributes") or {}).get("generation_mode", "absolute")
        ).lower()
        if generation_mode != "absolute":
            raise ValueError(
                f"{building_id} PV generation_mode={generation_mode!r}; this adapter "
                "requires absolute solar_generation kWh."
            )
        base_rows.append((non_shiftable - solar)[start:end])
        building_prices = _numeric_series(
            pricing, "electricity_pricing", source=f"{building_id}.pricing"
        )[start:end]
        if np.any(building_prices < 0.0):
            raise ValueError("Electricity prices must be non-negative.")
        if prices is None:
            prices = building_prices
        elif not np.array_equal(prices, building_prices):
            raise ValueError("Selected buildings do not share an identical price series.")

        storage = _stationary_asset(
            building_id,
            building,
            conservative_capacity_ratio=capacity_ratio,
        )
        if storage is not None:
            stationary.append(storage)
        service = _electrical_service(
            building_id, building, reserve_kw=reserve_kw
        )
        if service is not None:
            services.append(service)

        for charger_id, charger in (building.get("chargers") or {}).items():
            charger_count += 1
            if bool(charger.get("autosize", False)):
                raise ValueError("Autosized EV chargers are not supported.")
            frame = _read_parquet(dataset_root, charger.get("charger_simulation"))
            if end > len(frame):
                raise ValueError(
                    f"Requested source window exceeds charger {charger_id!r}."
                )
            attributes = dict(charger.get("attributes") or {})
            charger_efficiency = _efficiency(
                f"{charger_id}.efficiency", attributes.get("efficiency"), 1.0
            )
            charge_efficiency = _curve_minimum(
                f"{charger_id}.charge_efficiency_curve",
                attributes.get("charge_efficiency_curve"),
                default=charger_efficiency,
            )
            discharge_efficiency = _curve_minimum(
                f"{charger_id}.discharge_efficiency_curve",
                attributes.get("discharge_efficiency_curve"),
                default=charger_efficiency,
            )
            if not 0.0 < charge_efficiency <= 1.0:
                raise ValueError("Charger charge efficiency must be in (0, 1].")
            if not 0.0 < discharge_efficiency <= 1.0:
                raise ValueError("Charger discharge efficiency must be in (0, 1].")

            for ordinal, (global_start, global_end, ev_id) in enumerate(
                _connected_intervals(frame), start=1
            ):
                if global_start >= end or global_end <= start:
                    continue
                definition = ev_definitions.get(ev_id)
                if definition is None or not bool(definition.get("include", True)):
                    raise ValueError(
                        f"Unknown or excluded EV id {ev_id!r} in charger {charger_id!r}."
                    )
                battery = definition.get("battery") or {}
                if bool(battery.get("autosize", False)):
                    raise ValueError("Autosized EV batteries are not supported.")
                battery_attributes = dict(battery.get("attributes") or {})
                capacity = _non_negative(
                    f"{ev_id}.battery.capacity", battery_attributes.get("capacity")
                )
                depth = _non_negative(
                    f"{ev_id}.battery.depth_of_discharge",
                    battery_attributes.get("depth_of_discharge"),
                    1.0,
                )
                if depth > 1.0:
                    raise ValueError("EV depth_of_discharge must be <= 1.")
                minimum = capacity * (1.0 - depth)
                configured_initial_soc = battery_attributes.get("initial_soc")
                if configured_initial_soc is None or pd.isna(configured_initial_soc):
                    schema_initial_soc = deterministic_ev_initial_soc(
                        schema_random_seed=int(schema["random_seed"]),
                        electric_vehicle_id=ev_id,
                    )
                    schema_initial_source = (
                        "citylearn_deterministic_schema_seed_fallback"
                    )
                else:
                    schema_initial_soc = _soc(configured_initial_soc)
                    schema_initial_source = "schema"
                clipped_start = max(global_start, start)
                clipped_end = min(global_end, end)
                left_truncated = global_start < start
                arrival_soc, initial_source = _arrival_soc(
                    frame,
                    clipped_start,
                    ev_id=ev_id,
                    default_soc=schema_initial_soc,
                    default_source=schema_initial_source,
                    episode_start=start,
                )
                # A run that remains connected in the final dataset row is
                # right-censored: ``global_end == len(frame)`` reflects EOF,
                # not an observed departure.  Do not manufacture a terminal
                # departure constraint from that row.
                dataset_right_censored = global_end == len(frame)
                right_truncated = global_end > end or (
                    global_end == end and dataset_right_censored
                )
                session_id = (
                    f"{building_id}::{charger_id}::session_{ordinal:04d}::"
                    f"{global_start}_{global_end}"
                )
                if left_truncated:
                    left_ev.append(session_id)
                    if initial_source != "current_soc":
                        assumed_ev.append(session_id)
                if (
                    clipped_start > start
                    and initial_source
                    in {"schema", "citylearn_deterministic_schema_seed_fallback"}
                ):
                    runtime_drift_ev.append(session_id)
                    initial_source = f"{initial_source}_runtime_drift_unreproducible"
                if right_truncated:
                    right_ev.append(session_id)
                ev_initial_sources[session_id] = initial_source

                required_soc = _soc(
                    frame.iloc[global_end - 1].get(
                        "electric_vehicle_required_soc_departure"
                    ),
                    default=arrival_soc,
                )
                required_final = (
                    minimum
                    if right_truncated
                    else max(capacity * required_soc, minimum)
                )
                battery_directional = _minimum_directional_battery_efficiency(
                    f"{ev_id}.battery", battery_attributes
                )
                ev_nominal = _non_negative(
                    f"{ev_id}.battery.nominal_power",
                    battery_attributes.get("nominal_power"),
                )
                sessions.append(
                    EVSession(
                        session_id=session_id,
                        building_id=building_id,
                        action_name=f"electric_vehicle_storage_{charger_id}",
                        start_time_step=clipped_start - start,
                        end_time_step=clipped_end - start - 1,
                        capacity_kwh=capacity,
                        initial_energy_kwh=max(capacity * arrival_soc, minimum),
                        required_final_energy_kwh=required_final,
                        minimum_energy_kwh=minimum,
                        max_charge_kw=min(
                            _non_negative(
                                f"{charger_id}.max_charging_power",
                                attributes.get("max_charging_power"),
                                50.0,
                            ),
                            ev_nominal,
                        ),
                        max_discharge_kw=min(
                            _non_negative(
                                f"{charger_id}.max_discharging_power",
                                attributes.get("max_discharging_power"),
                                50.0,
                            ),
                            ev_nominal,
                        ),
                        min_charge_kw=_non_negative(
                            f"{charger_id}.min_charging_power",
                            attributes.get("min_charging_power"),
                            0.0,
                        ),
                        min_discharge_kw=_non_negative(
                            f"{charger_id}.min_discharging_power",
                            attributes.get("min_discharging_power"),
                            0.0,
                        ),
                        charge_efficiency=charge_efficiency * battery_directional,
                        discharge_efficiency=(
                            discharge_efficiency * battery_directional
                        ),
                        phase_connection=_phase(attributes.get("phase_connection")),
                        allow_departure_shortfall=True,
                    )
                )

        for appliance_id, appliance in (
            building.get("deferrable_appliances") or {}
        ).items():
            profiles = _read_parquet(
                dataset_root, appliance.get("cycle_profiles_file")
            )
            schedule = _read_parquet(
                dataset_root, appliance.get("flexibility_schedule_file")
            )
            required_profile_columns = {"profile_id", "load_profile"}
            required_schedule_columns = {
                "cycle_id",
                "profile_id",
                "earliest_start_time_step",
                "latest_start_time_step",
                "must_run",
            }
            if missing := required_profile_columns - set(profiles.columns):
                raise ValueError(f"Cycle profiles missing columns: {sorted(missing)}.")
            if missing := required_schedule_columns - set(schedule.columns):
                raise ValueError(
                    f"Flexibility schedule missing columns: {sorted(missing)}."
                )
            profile_map = {
                str(row.profile_id): _parse_profile(row.load_profile)
                for row in profiles.itertuples(index=False)
            }
            appliance_phase = _phase(
                (appliance.get("attributes") or {}).get("phase_connection")
            )
            for row in schedule.itertuples(index=False):
                cycle_id = f"{building_id}::{appliance_id}::{row.cycle_id}"
                try:
                    profile = profile_map[str(row.profile_id)]
                except KeyError as error:
                    raise ValueError(
                        f"Unknown profile_id {row.profile_id!r} for {cycle_id}."
                    ) from error
                earliest = int(row.earliest_start_time_step)
                latest = int(row.latest_start_time_step)
                duration = len(profile)
                if earliest >= end or latest + duration <= start:
                    continue
                clipped_earliest = max(earliest, start)
                clipped_latest = min(latest, end - duration)
                restricted = clipped_earliest != earliest or clipped_latest != latest
                if clipped_earliest > clipped_latest:
                    omitted_cycles.append(cycle_id)
                    continue
                if restricted:
                    restricted_cycles.append(cycle_id)
                cycles.append(
                    DeferrableCycle(
                        cycle_id=cycle_id,
                        building_id=building_id,
                        action_name=f"deferrable_appliance_{appliance_id}",
                        earliest_start_time_step=clipped_earliest - start,
                        latest_start_time_step=clipped_latest - start,
                        load_profile_kwh=profile,
                        must_run=bool(row.must_run),
                        phase_connection=appliance_phase,
                    )
                )

    assert prices is not None
    diagnostics = CityLearnTotalEnergyDiagnostics(
        schema_path=str(schema_path),
        dataset_root=str(dataset_root),
        source_start_time_step=start,
        source_end_time_step_exclusive=end,
        horizon=end - start,
        building_count=len(selected),
        stationary_storage_count=len(stationary),
        charger_count=charger_count,
        ev_session_count=len(sessions),
        deferrable_cycle_count=len(cycles),
        electrical_service_count=len(services),
        left_truncated_ev_session_ids=tuple(left_ev),
        right_truncated_ev_session_ids=tuple(right_ev),
        assumed_initial_soc_ev_session_ids=tuple(assumed_ev),
        runtime_drift_initial_soc_ev_session_ids=tuple(runtime_drift_ev),
        restricted_deferrable_cycle_ids=tuple(restricted_cycles),
        omitted_boundary_deferrable_cycle_ids=tuple(omitted_cycles),
        electrical_service_reserve_kw=reserve_kw,
    )
    metadata = {
        "scope": "total_energy_linear_milp_from_citylearn_dataset",
        "schema_path": str(schema_path),
        "schema_sha256": hashlib.sha256(raw_schema).hexdigest(),
        "dataset_root": str(dataset_root),
        "source_start_time_step": start,
        "source_end_time_step_exclusive": end,
        "source_window_convention": "[start, end) transitions",
        "base_load_definition": "non_shiftable_load_kwh - absolute_solar_generation_kwh",
        "stationary_model": (
            "0.99 capacity by default, minimum capacity-power rate, 0.99 times "
            "minimum directional efficiency, terminal energy not below initial"
        ),
        "ev_model": (
            "contiguous state=1 sessions; charger deadbands and phase; minimum "
            "charger/battery efficiency linearization"
        ),
        "electrical_service_reserve_kw": reserve_kw,
        "boundary_service_exact": diagnostics.boundary_service_exact,
        "boundary_diagnostics": diagnostics.to_dict(),
        "ev_initial_soc_source": ev_initial_sources,
        "global_optimum_claim": False,
        "requires_citylearn_replay": True,
        "community_observations_used": False,
    }
    problem = TotalEnergyProblem(
        problem_id=(
            problem_id
            or f"citylearn-total-energy::{settlement}::{start}_{end}"
        ),
        timestep_hours=timestep_hours,
        building_ids=selected,
        price_eur_per_kwh=prices,
        base_net_load_kwh=np.stack(base_rows),
        settlement=settlement,
        stationary_storage=tuple(stationary),
        ev_sessions=tuple(sessions),
        deferrable_cycles=tuple(cycles),
        electrical_services=tuple(services),
        metadata=metadata,
    )
    return CityLearnTotalEnergyBuild(problem=problem, diagnostics=diagnostics)


__all__ = [
    "CityLearnTotalEnergyBuild",
    "CityLearnTotalEnergyDiagnostics",
    "build_citylearn_total_energy_problem",
]
