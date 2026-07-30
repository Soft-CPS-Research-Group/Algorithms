"""Build a total-home MILP problem directly from a CityLearn dataset schema."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from algorithms.oracles.citylearn_ev import deterministic_ev_initial_soc
from algorithms.oracles.total_home_milp import (
    DeferrableCycleSpec,
    EVSessionSpec,
    ElectricalServiceSpec,
    LinearStorageSpec,
    TotalHomeProblem,
)


@dataclass(frozen=True)
class CityLearnTotalHomeBuild:
    problem: TotalHomeProblem
    schema_path: Path
    source_start_time_step: int
    source_end_time_step: int
    ev_session_count: int
    deferrable_cycle_count: int


def _non_negative(name: str, value: Any, default: float | None = None) -> float:
    if value is None and default is not None:
        value = default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed


def _soc(value: Any, *, default: float | None = None) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if default is None:
            raise ValueError("Missing SOC value.")
        return float(default)
    parsed = float(value)
    if parsed > 1.0:
        parsed /= 100.0
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"SOC value must be in [0, 1] or [0, 100]; got {value!r}.")
    return parsed


def _dataset_root(schema_path: Path, schema: Mapping[str, Any]) -> Path:
    raw = Path(str(schema.get("root_directory") or schema_path.parent))
    candidates = [raw, schema_path.parent]
    if not raw.is_absolute():
        candidates.insert(1, schema_path.parent / raw)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and (resolved / Path(str(next(iter(schema["buildings"].values()))["energy_simulation"]))).is_file():
            return resolved
    raise FileNotFoundError(f"Could not resolve CityLearn root_directory for {schema_path}.")


def _read_parquet(root: Path, filename: str) -> pd.DataFrame:
    path = (root / str(filename)).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _connected_intervals(states: np.ndarray) -> list[tuple[int, int]]:
    connected = np.asarray(states) == 1
    padded = np.r_[False, connected, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _arrival_soc(
    frame: pd.DataFrame,
    start: int,
    *,
    default_soc: float,
) -> float:
    current_column = "electric_vehicle_current_soc"
    if current_column in frame and pd.notna(frame.iloc[start][current_column]):
        return _soc(frame.iloc[start][current_column], default=default_soc)
    arrival_column = "electric_vehicle_estimated_soc_arrival"
    if start > 0 and pd.notna(frame.iloc[start - 1][arrival_column]):
        return _soc(frame.iloc[start - 1][arrival_column], default=default_soc)
    if pd.notna(frame.iloc[start][arrival_column]):
        return _soc(frame.iloc[start][arrival_column], default=default_soc)
    return float(default_soc)


def _parse_profile(raw: Any) -> tuple[float, ...]:
    if isinstance(raw, np.ndarray):
        values = raw.astype(np.float64).reshape(-1)
    elif isinstance(raw, (list, tuple)):
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
    else:
        parsed = ast.literal_eval(str(raw))
        values = np.asarray(parsed if isinstance(parsed, (list, tuple)) else [parsed], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Deferrable load profile must contain finite non-negative energy values.")
    return tuple(float(value) for value in values)


def _conservative_directional_storage_efficiency(
    *,
    attributes: Mapping[str, Any],
    technical_efficiency: float,
    external_efficiency: float = 1.0,
) -> float:
    """Lower envelope of CityLearn's power-dependent battery efficiency.

    CityLearn interprets ``power_efficiency_curve`` as technical (round-trip)
    efficiency and applies its square root in each direction.  When the curve
    is omitted, its generated first point is in ``[0.85, 0.90]`` times the
    configured technical efficiency, so the 0.85 endpoint is replay-safe for
    every deterministic draw.
    """

    raw_curve = attributes.get("power_efficiency_curve")
    if raw_curve is None:
        minimum_technical_efficiency = technical_efficiency * 0.85
    else:
        curve = np.asarray(raw_curve, dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 2 or curve.shape[0] < 2:
            raise ValueError("power_efficiency_curve must contain [power, efficiency] rows.")
        if not np.all(np.isfinite(curve)) or np.any(curve[:, 1] <= 0.0):
            raise ValueError("power_efficiency_curve efficiencies must be finite and > 0.")
        minimum_technical_efficiency = float(np.min(curve[:, 1]))
    return external_efficiency * math.sqrt(minimum_technical_efficiency)


def _storage_spec(building: Mapping[str, Any]) -> LinearStorageSpec | None:
    config = building.get("electrical_storage")
    if not config:
        return None
    if bool(config.get("autosize", False)):
        raise ValueError("Autosized stationary storage is not supported by the total-home adapter.")
    attributes = config.get("attributes") or {}
    capacity = _non_negative("electrical_storage.capacity", attributes.get("capacity"))
    nominal = _non_negative("electrical_storage.nominal_power", attributes.get("nominal_power"))
    depth = _non_negative("electrical_storage.depth_of_discharge", attributes.get("depth_of_discharge"), 1.0)
    if depth > 1.0:
        raise ValueError("electrical_storage.depth_of_discharge must be <= 1.")
    initial_soc = _soc(attributes.get("initial_soc"), default=0.0)
    technical_efficiency = _non_negative("electrical_storage.efficiency", attributes.get("efficiency"), 0.9)
    directional_efficiency = _conservative_directional_storage_efficiency(
        attributes=attributes,
        technical_efficiency=technical_efficiency,
    )
    return LinearStorageSpec(
        capacity_kwh=capacity,
        initial_energy_kwh=capacity * initial_soc,
        final_energy_min_kwh=capacity * initial_soc,
        minimum_energy_kwh=capacity * (1.0 - depth),
        max_charge_kw=nominal,
        max_discharge_kw=nominal,
        charge_efficiency=directional_efficiency,
        discharge_efficiency=directional_efficiency,
        phase_connection=building.get("electrical_storage_phase_connection"),
    )


def _service_spec(building: Mapping[str, Any]) -> ElectricalServiceSpec | None:
    config = building.get("electrical_service")
    if not config:
        return None
    limits = config.get("limits") or {}
    total = limits.get("total") or {}
    per_phase = limits.get("per_phase") or {}
    return ElectricalServiceSpec(
        mode=config.get("mode", "single_phase"),
        default_split=config.get("default_split", "balanced"),
        total_import_limit_kw=total.get("import_kw"),
        total_export_limit_kw=total.get("export_kw"),
        per_phase_import_limit_kw={phase.upper(): (values or {}).get("import_kw") for phase, values in per_phase.items()},
        per_phase_export_limit_kw={phase.upper(): (values or {}).get("export_kw") for phase, values in per_phase.items()},
    )


def build_citylearn_total_home_problem(
    *,
    schema_path: Path | str,
    building_id: str,
    start_time_step: int,
    end_time_step: int,
    problem_id: str | None = None,
    require_closed_service_windows: bool = True,
    allow_physically_infeasible_ev_shortfall: bool = True,
    ev_departure_soc_margin: float = 0.0,
) -> CityLearnTotalHomeBuild:
    """Build one local perfect-foresight problem over ``[start, end)``.

    With the default closed-window guard, every connected EV interval that
    touches the requested window must be wholly represented.  Deferrable
    cycles whose feasible start window touches the horizon must likewise fit
    completely.  This prevents optimistic boundary truncation.
    """

    schema_path = Path(schema_path).resolve()
    raw_schema = schema_path.read_bytes()
    schema = json.loads(raw_schema)
    root = _dataset_root(schema_path, schema)
    building_id = str(building_id)
    building = (schema.get("buildings") or {}).get(building_id)
    if building is None or not bool(building.get("include", True)):
        raise ValueError(f"Building {building_id!r} is not included in the schema.")
    start, end = int(start_time_step), int(end_time_step)
    if start < 0 or end <= start:
        raise ValueError("The source window must satisfy 0 <= start < end.")
    ev_departure_soc_margin = _non_negative(
        "ev_departure_soc_margin", ev_departure_soc_margin
    )
    if ev_departure_soc_margin > 1.0:
        raise ValueError("ev_departure_soc_margin must be <= 1.")
    step_hours = _non_negative("seconds_per_time_step", schema.get("seconds_per_time_step")) / 3600.0
    if step_hours <= 0.0:
        raise ValueError("seconds_per_time_step must be > 0.")

    energy = _read_parquet(root, building["energy_simulation"])
    pricing = _read_parquet(root, building["pricing"])
    if end > len(energy) or end > len(pricing):
        raise ValueError("Requested source window exceeds dataset length.")
    for column in ("non_shiftable_load", "solar_generation"):
        if column not in energy:
            raise ValueError(f"Energy simulation is missing {column!r}.")
    if "electricity_pricing" not in pricing:
        raise ValueError("Pricing simulation is missing 'electricity_pricing'.")
    base = (
        pd.to_numeric(energy["non_shiftable_load"], errors="raise").to_numpy(dtype=np.float64)
        - pd.to_numeric(energy["solar_generation"], errors="raise").to_numpy(dtype=np.float64)
    )[start:end]
    prices = pd.to_numeric(pricing["electricity_pricing"], errors="raise").to_numpy(dtype=np.float64)[start:end]

    ev_definitions = schema.get("electric_vehicles_def") or {}
    ev_sessions: list[EVSessionSpec] = []
    for charger_id, charger in (building.get("chargers") or {}).items():
        if bool(charger.get("autosize", False)):
            raise ValueError("Autosized chargers are not supported by the total-home adapter.")
        frame = _read_parquet(root, charger["charger_simulation"])
        if end > len(frame):
            raise ValueError(f"Charger {charger_id!r} is shorter than the requested window.")
        attributes = charger.get("attributes") or {}
        charger_efficiency = _non_negative(f"{charger_id}.efficiency", attributes.get("efficiency"), 1.0)
        if charger_efficiency > 1.0:
            raise ValueError("Charger efficiency must be <= 1.")
        for session_index, (global_session_start, global_session_end) in enumerate(
            _connected_intervals(frame["electric_vehicle_charger_state"].to_numpy())
        ):
            intersects = global_session_start < end and global_session_end > start
            contained = global_session_start >= start and global_session_end <= end
            if intersects and not contained and require_closed_service_windows:
                raise ValueError(
                    f"Requested window cuts EV session {charger_id}[{global_session_start}:{global_session_end}]."
                )
            if not contained:
                continue
            raw_ev_id = str(frame.iloc[global_session_start]["electric_vehicle_id"]).strip()
            definition = ev_definitions.get(raw_ev_id)
            if definition is None:
                raise ValueError(f"Unknown EV id {raw_ev_id!r} in charger {charger_id!r}.")
            battery = definition.get("battery") or {}
            battery_attributes = battery.get("attributes") or {}
            if bool(battery.get("autosize", False)):
                raise ValueError("Autosized EV batteries are not supported by the total-home adapter.")
            capacity = _non_negative(f"{raw_ev_id}.capacity", battery_attributes.get("capacity"))
            depth = _non_negative(f"{raw_ev_id}.depth_of_discharge", battery_attributes.get("depth_of_discharge"), 1.0)
            if depth > 1.0:
                raise ValueError("EV depth_of_discharge must be <= 1.")
            configured_initial_soc = battery_attributes.get("initial_soc")
            if configured_initial_soc is None or (
                isinstance(configured_initial_soc, float)
                and np.isnan(configured_initial_soc)
            ):
                default_soc = deterministic_ev_initial_soc(
                    schema_random_seed=int(schema["random_seed"]),
                    electric_vehicle_id=raw_ev_id,
                )
                initial_soc_source = "citylearn_deterministic_schema_seed_fallback"
            else:
                default_soc = _soc(configured_initial_soc)
                initial_soc_source = "schema"
            arrival_soc = _arrival_soc(frame, global_session_start, default_soc=default_soc)
            required_soc = _soc(
                frame.iloc[global_session_end - 1]["electric_vehicle_required_soc_departure"],
                default=arrival_soc,
            )
            # EV batteries omit efficiency in this dataset.  CityLearn samples
            # it in [0.9, 0.98]; the lower endpoint plus the lower envelope of
            # its power curve gives a replay-conservative linear coefficient.
            battery_efficiency = _non_negative(
                f"{raw_ev_id}.efficiency", battery_attributes.get("efficiency"), 0.9
            )
            directional = _conservative_directional_storage_efficiency(
                attributes=battery_attributes,
                technical_efficiency=battery_efficiency,
                external_efficiency=charger_efficiency,
            )
            phase_connection = attributes.get("phase_connection")
            minimum_energy = capacity * (1.0 - depth)
            ev_sessions.append(
                EVSessionSpec(
                    session_id=f"{charger_id}::session_{session_index}::{global_session_start}_{global_session_end}",
                    action_name=f"electric_vehicle_storage_{charger_id}",
                    electric_vehicle_id=raw_ev_id,
                    start_time_step=global_session_start - start,
                    end_time_step=global_session_end - start,
                    capacity_kwh=capacity,
                    # CityLearn's Battery.energy_init applies the depth-of-
                    # discharge floor even when an arrival SOC is lower.
                    initial_energy_kwh=max(capacity * arrival_soc, minimum_energy),
                    required_departure_energy_kwh=capacity * min(
                        required_soc + ev_departure_soc_margin,
                        1.0,
                    ),
                    minimum_energy_kwh=minimum_energy,
                    max_charge_kw=min(
                        _non_negative(f"{charger_id}.max_charging_power", attributes.get("max_charging_power")),
                        _non_negative(f"{raw_ev_id}.nominal_power", battery_attributes.get("nominal_power")),
                    ),
                    max_discharge_kw=min(
                        _non_negative(f"{charger_id}.max_discharging_power", attributes.get("max_discharging_power"), 0.0),
                        _non_negative(f"{raw_ev_id}.nominal_power", battery_attributes.get("nominal_power")),
                    ),
                    min_charge_kw=_non_negative(f"{charger_id}.min_charging_power", attributes.get("min_charging_power"), 0.0),
                    min_discharge_kw=_non_negative(f"{charger_id}.min_discharging_power", attributes.get("min_discharging_power"), 0.0),
                    charge_efficiency=directional,
                    discharge_efficiency=directional,
                    phase_connection=phase_connection,
                    allow_departure_shortfall=bool(allow_physically_infeasible_ev_shortfall),
                    metadata={
                        "configured_or_deterministic_initial_soc": default_soc,
                        "initial_soc_source": initial_soc_source,
                    },
                )
            )

    cycles: list[DeferrableCycleSpec] = []
    for appliance_id, appliance in (building.get("deferrable_appliances") or {}).items():
        profiles = _read_parquet(root, appliance["cycle_profiles_file"])
        schedule = _read_parquet(root, appliance["flexibility_schedule_file"])
        profile_map = {
            str(row.profile_id): _parse_profile(row.load_profile)
            for row in profiles.itertuples(index=False)
        }
        for row in schedule.itertuples(index=False):
            profile = profile_map[str(row.profile_id)]
            earliest, latest = int(row.earliest_start_time_step), int(row.latest_start_time_step)
            touches = earliest < end and latest + len(profile) > start
            contained = earliest >= start and latest + len(profile) <= end
            if touches and not contained and require_closed_service_windows:
                raise ValueError(f"Requested window cuts deferrable cycle {row.cycle_id!r}.")
            if not contained:
                continue
            cycles.append(
                DeferrableCycleSpec(
                    cycle_id=str(row.cycle_id),
                    action_name=f"deferrable_appliance_{appliance_id}",
                    earliest_start_time_step=earliest - start,
                    latest_start_time_step=latest - start,
                    load_profile_kwh=profile,
                    must_run=bool(row.must_run),
                )
            )

    problem = TotalHomeProblem(
        problem_id=problem_id or f"citylearn-total-home::{building_id}::{start}_{end}",
        building_id=building_id,
        timestep_hours=step_hours,
        price_eur_per_kwh=prices,
        base_net_load_kwh=base,
        stationary_storage=_storage_spec(building),
        ev_sessions=tuple(ev_sessions),
        deferrable_cycles=tuple(cycles),
        electrical_service=_service_spec(building),
        metadata={
            "scope": "individual_total_home_linear_milp",
            "schema_path": str(schema_path),
            "schema_sha256": hashlib.sha256(raw_schema).hexdigest(),
            "dataset_root": str(root),
            "source_start_time_step": start,
            "source_end_time_step_exclusive": end,
            "closed_service_windows": bool(require_closed_service_windows),
            "price_conditioning_scope": "local_building_only",
            "community_observations_used": False,
            "physics_note": (
                "linear device approximation with conservative battery efficiency "
                "envelopes; capacity-power curves and degradation still require CityLearn replay"
            ),
            "battery_efficiency_linearization": "citylearn_power_curve_lower_envelope",
            "ev_service_objective": (
                "lexicographic_minimum_total_departure_shortfall_then_cost"
                if allow_physically_infeasible_ev_shortfall
                else "hard_departure_targets"
            ),
            "ev_departure_soc_margin": ev_departure_soc_margin,
        },
    )
    return CityLearnTotalHomeBuild(
        problem=problem,
        schema_path=schema_path,
        source_start_time_step=start,
        source_end_time_step=end,
        ev_session_count=len(ev_sessions),
        deferrable_cycle_count=len(cycles),
    )


__all__ = ["CityLearnTotalHomeBuild", "build_citylearn_total_home_problem"]
