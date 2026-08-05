"""Build a stationary-battery oracle from a CityLearn replay export.

The adapter freezes every non-stationary-battery service decision from an
existing simulator run.  It subtracts the exported stationary-battery energy
from each building's net electricity consumption and exposes only the
stationary batteries to :mod:`perfect_foresight_milp`.

This is intentionally a *conditional* oracle.  Its optimistic result is a
lower bound for the fixed-service battery subproblem, not yet for the full
joint EV/deferrable/network control problem.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from algorithms.oracles.perfect_foresight_milp import (
    BatteryAsset,
    BatteryModel,
    PerfectForesightProblem,
    SemanticActionSeries,
    SemanticSchedule,
)


_BUILDING_NAME = re.compile(r"^Building_(\d+)$")
_NET_COLUMN = "Net Electricity Consumption-kWh"
_BATTERY_COLUMN = "Battery (Dis)Charge-kWh"


@dataclass(frozen=True)
class FixedServiceDiagnostics:
    """Independent reconstruction checks for the exported trajectory."""

    source_session_directory: str
    episode: int
    horizon: int
    first_timestamp: str
    last_timestamp: str
    source_policy_cost_reconstructed_eur: float
    fixed_service_without_stationary_battery_cost_eur: float
    source_stationary_battery_throughput_kwh: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_session_directory": self.source_session_directory,
            "episode": self.episode,
            "horizon": self.horizon,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "source_policy_cost_reconstructed_eur": self.source_policy_cost_reconstructed_eur,
            "fixed_service_without_stationary_battery_cost_eur": (
                self.fixed_service_without_stationary_battery_cost_eur
            ),
            "source_stationary_battery_throughput_kwh": (
                self.source_stationary_battery_throughput_kwh
            ),
        }


@dataclass(frozen=True)
class FixedServiceProblem:
    problem: PerfectForesightProblem
    diagnostics: FixedServiceDiagnostics


def _battery_group_key(battery: BatteryAsset) -> tuple[Any, ...]:
    """Exact-equivalence key used for a lossless district aggregation."""

    return (
        battery.action_name,
        battery.initial_energy_kwh,
        battery.final_energy_min_kwh,
        *battery.optimistic.to_dict().values(),
        *battery.conservative.to_dict().values(),
    )


def _aggregate_equivalent_batteries(
    batteries: list[BatteryAsset],
) -> tuple[list[BatteryAsset], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[BatteryAsset]] = {}
    for battery in batteries:
        groups.setdefault(_battery_group_key(battery), []).append(battery)

    aggregated: list[BatteryAsset] = []
    group_metadata: list[dict[str, Any]] = []
    for group_index, members in enumerate(groups.values(), start=1):
        first = members[0]
        count = len(members)
        action_name = (
            first.action_name
            if count == 1
            else f"oracle_group_{group_index:02d}_{first.action_name}"
        )
        optimistic = BatteryModel(
            capacity_kwh=sum(item.optimistic.capacity_kwh for item in members),
            max_charge_kw=sum(item.optimistic.max_charge_kw for item in members),
            max_discharge_kw=sum(item.optimistic.max_discharge_kw for item in members),
            charge_efficiency=first.optimistic.charge_efficiency,
            discharge_efficiency=first.optimistic.discharge_efficiency,
        )
        conservative = BatteryModel(
            capacity_kwh=sum(item.conservative.capacity_kwh for item in members),
            max_charge_kw=sum(item.conservative.max_charge_kw for item in members),
            max_discharge_kw=sum(item.conservative.max_discharge_kw for item in members),
            charge_efficiency=first.conservative.charge_efficiency,
            discharge_efficiency=first.conservative.discharge_efficiency,
        )
        aggregate = BatteryAsset(
            building_id=first.building_id,
            action_name=action_name,
            initial_energy_kwh=sum(item.initial_energy_kwh for item in members),
            final_energy_min_kwh=sum(item.final_energy_min_kwh for item in members),
            optimistic=optimistic,
            conservative=conservative,
        )
        aggregated.append(aggregate)
        total_power = sum(item.conservative.max_charge_kw for item in members)
        denominator = total_power if total_power > 0.0 else float(count)
        group_metadata.append(
            {
                "oracle_building_id": aggregate.building_id,
                "oracle_action_name": aggregate.action_name,
                "aggregation_is_exact_for_district_linear_model": True,
                "members": [
                    {
                        "building_id": item.building_id,
                        "action_name": item.action_name,
                        "schedule_fraction": (
                            item.conservative.max_charge_kw / denominator
                            if total_power > 0.0
                            else 1.0 / count
                        ),
                    }
                    for item in members
                ],
            }
        )
    return aggregated, group_metadata


def expand_aggregated_battery_schedule(
    schedule: SemanticSchedule,
    problem_metadata: Mapping[str, Any],
) -> SemanticSchedule:
    """Expand an exact aggregate schedule into physical semantic actions."""

    groups = {
        (str(item["oracle_building_id"]), str(item["oracle_action_name"])): item
        for item in problem_metadata.get("battery_groups", ())
    }
    expanded: list[SemanticActionSeries] = []
    for series in schedule.series:
        group = groups.get((series.building_id, series.action_name))
        if group is None:
            raise ValueError(
                f"No battery-group metadata for {(series.building_id, series.action_name)!r}."
            )
        for member in group["members"]:
            fraction = float(member["schedule_fraction"])
            expanded.append(
                SemanticActionSeries(
                    building_id=str(member["building_id"]),
                    action_name=str(member["action_name"]),
                    values=tuple(float(value) * fraction for value in series.values),
                    unit=series.unit,
                    positive_direction=series.positive_direction,
                )
            )
    return SemanticSchedule(
        problem_id=schedule.problem_id,
        horizon=schedule.horizon,
        timestep_hours=schedule.timestep_hours,
        series=tuple(expanded),
        metadata={
            **dict(schedule.metadata),
            "aggregate_schedule_expanded": True,
            "physical_series_count": len(expanded),
            "requires_citylearn_replay": True,
        },
    )


def _finite_non_negative(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric; got {value!r}.") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}.")
    return parsed


def _resolve_session_directory(path: Path, episode: int) -> Path:
    path = path.resolve()
    direct = path / f"exported_data_pricing_ep{episode}.csv"
    if direct.is_file():
        return path
    candidates = sorted(
        item.parent
        for item in path.glob(f"*/exported_data_pricing_ep{episode}.csv")
        if item.is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one simulation session below {path}, found {len(candidates)}."
        )
    return candidates[0]


def _read_series(path: Path, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"timestamp", value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}.")
    timestamps = frame["timestamp"].astype(str).to_numpy()
    values = pd.to_numeric(frame[value_column], errors="raise").to_numpy(dtype=np.float64)
    if timestamps.size == 0 or values.size != timestamps.size:
        raise ValueError(f"{path} must contain a non-empty aligned series.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite {value_column!r} values.")
    if len(set(timestamps.tolist())) != timestamps.size:
        raise ValueError(f"{path} contains duplicate timestamps.")
    return timestamps, values


def _assert_timestamps(reference: np.ndarray, candidate: np.ndarray, path: Path) -> None:
    if not np.array_equal(reference, candidate):
        raise ValueError(f"Timestamp mismatch in {path}; oracle inputs must align exactly.")


def _building_export_stem(building_id: str) -> str:
    match = _BUILDING_NAME.fullmatch(building_id)
    if match is None:
        raise ValueError(
            f"Unsupported building id {building_id!r}; expected CityLearn Building_<n>."
        )
    return f"exported_data_building_{int(match.group(1))}"


def _minimum_capacity_power_ratio(attributes: Mapping[str, Any]) -> float:
    curve = attributes.get("capacity_power_curve")
    if curve is None:
        # CityLearn's generated default terminates in U(0.20, 0.30).
        return 0.20
    array = np.asarray(curve, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        raise ValueError("capacity_power_curve must be an Nx2 array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("capacity_power_curve contains non-finite values.")
    return max(float(np.min(array[:, 1])), 0.0)


def _minimum_directional_efficiency(attributes: Mapping[str, Any]) -> float:
    curve = attributes.get("power_efficiency_curve")
    if curve is None:
        technical_efficiency = _finite_non_negative(
            "electrical_storage.efficiency", attributes.get("efficiency", 0.9)
        )
        minimum_technical_efficiency = technical_efficiency * 0.85
    else:
        array = np.asarray(curve, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
            raise ValueError("power_efficiency_curve must be an Nx2 array.")
        if not np.all(np.isfinite(array)):
            raise ValueError("power_efficiency_curve contains non-finite values.")
        minimum_technical_efficiency = float(np.min(array[:, 1]))
    if not 0.0 < minimum_technical_efficiency <= 1.0:
        raise ValueError("Minimum technical battery efficiency must be in (0, 1].")
    # CityLearn applies sqrt(technical efficiency) in each direction.  A small
    # extra margin absorbs interpolation and floating-point differences.
    return 0.99 * math.sqrt(minimum_technical_efficiency)


def build_fixed_service_battery_problem(
    *,
    schema_path: Path | str,
    simulation_data_directory: Path | str,
    problem_id: str,
    episode: int = 1,
    conservative_capacity_ratio: float = 0.99,
    aggregate_equivalent_batteries: bool = True,
) -> FixedServiceProblem:
    """Create a conditional fixed-service stationary-battery problem.

    The conservative battery model uses the minimum CityLearn capacity-power
    curve rate, a lower directional efficiency than the generated curve, and
    a one-percent capacity margin.  It remains a replay candidate rather than
    a simulator-feasibility certificate.
    """

    schema_path = Path(schema_path).resolve()
    session_directory = _resolve_session_directory(Path(simulation_data_directory), episode)
    if not 0.0 < float(conservative_capacity_ratio) <= 1.0:
        raise ValueError("conservative_capacity_ratio must be in (0, 1].")

    raw_schema = schema_path.read_bytes()
    schema = json.loads(raw_schema)
    timestep_hours = _finite_non_negative(
        "seconds_per_time_step", schema.get("seconds_per_time_step")
    ) / 3600.0
    if timestep_hours <= 0.0:
        raise ValueError("seconds_per_time_step must be > 0.")

    price_path = session_directory / f"exported_data_pricing_ep{episode}.csv"
    price_frame = pd.read_csv(price_path)
    price_columns = [
        column
        for column in price_frame.columns
        if column.startswith("electricity_pricing-") and "predicted" not in column
    ]
    if len(price_columns) != 1:
        raise ValueError(f"Expected one current electricity price column in {price_path}.")
    if "timestamp" not in price_frame:
        raise ValueError(f"{price_path} is missing the timestamp column.")
    timestamps = price_frame["timestamp"].astype(str).to_numpy()
    prices = pd.to_numeric(price_frame[price_columns[0]], errors="raise").to_numpy(
        dtype=np.float64
    )
    if timestamps.size == 0 or prices.size != timestamps.size:
        raise ValueError("The price export must contain a non-empty aligned series.")
    if not np.all(np.isfinite(prices)) or np.any(prices < 0.0):
        raise ValueError("Electricity prices must be finite and non-negative.")

    building_ids: list[str] = []
    source_net: list[np.ndarray] = []
    fixed_base: list[np.ndarray] = []
    batteries: list[BatteryAsset] = []
    source_throughput = 0.0
    for building_id, building in schema.get("buildings", {}).items():
        if not bool(building.get("include", True)):
            continue
        building_id = str(building_id)
        stem = _building_export_stem(building_id)
        building_path = session_directory / f"{stem}_ep{episode}.csv"
        building_timestamps, net = _read_series(building_path, _NET_COLUMN)
        _assert_timestamps(timestamps, building_timestamps, building_path)

        storage = building.get("electrical_storage")
        if storage is None:
            battery_energy = np.zeros_like(net)
        else:
            battery_path = session_directory / f"{stem}_battery_ep{episode}.csv"
            battery_timestamps, battery_energy = _read_series(battery_path, _BATTERY_COLUMN)
            _assert_timestamps(timestamps, battery_timestamps, battery_path)
            attributes = dict(storage.get("attributes", {}))
            if bool(storage.get("autosize", False)):
                raise ValueError("Autosized stationary batteries are not supported by this adapter.")
            capacity = _finite_non_negative(
                f"{building_id}.electrical_storage.capacity", attributes.get("capacity")
            )
            nominal_power = _finite_non_negative(
                f"{building_id}.electrical_storage.nominal_power",
                attributes.get("nominal_power"),
            )
            depth_of_discharge = _finite_non_negative(
                f"{building_id}.electrical_storage.depth_of_discharge",
                attributes.get("depth_of_discharge", 1.0),
            )
            if depth_of_discharge > 1.0:
                raise ValueError("electrical_storage.depth_of_discharge must be <= 1.")
            initial_soc = attributes.get("initial_soc")
            initial_soc = (
                1.0 - depth_of_discharge
                if initial_soc is None
                else _finite_non_negative(
                    f"{building_id}.electrical_storage.initial_soc", initial_soc
                )
            )
            if initial_soc > 1.0:
                raise ValueError("electrical_storage.initial_soc must be <= 1.")
            initial_energy = capacity * initial_soc
            conservative_capacity = max(
                capacity * float(conservative_capacity_ratio), initial_energy
            )
            conservative_power = nominal_power * _minimum_capacity_power_ratio(attributes)
            conservative_efficiency = _minimum_directional_efficiency(attributes)
            batteries.append(
                BatteryAsset(
                    building_id=building_id,
                    action_name="electrical_storage",
                    initial_energy_kwh=initial_energy,
                    final_energy_min_kwh=initial_energy,
                    optimistic=BatteryModel(
                        capacity_kwh=capacity,
                        max_charge_kw=nominal_power,
                        max_discharge_kw=nominal_power,
                        charge_efficiency=1.0,
                        discharge_efficiency=1.0,
                    ),
                    conservative=BatteryModel(
                        capacity_kwh=conservative_capacity,
                        max_charge_kw=conservative_power,
                        max_discharge_kw=conservative_power,
                        charge_efficiency=conservative_efficiency,
                        discharge_efficiency=conservative_efficiency,
                    ),
                )
            )
            source_throughput += float(np.sum(np.abs(battery_energy)))

        building_ids.append(building_id)
        source_net.append(net)
        fixed_base.append(net - battery_energy)

    if not building_ids:
        raise ValueError("Schema does not include any buildings.")
    source_net_array = np.stack(source_net)
    fixed_base_array = np.stack(fixed_base)
    source_grid = np.maximum(np.sum(source_net_array, axis=0), 0.0)
    fixed_grid = np.maximum(np.sum(fixed_base_array, axis=0), 0.0)
    source_cost = float(np.dot(prices, source_grid))
    fixed_cost = float(np.dot(prices, fixed_grid))
    diagnostics = FixedServiceDiagnostics(
        source_session_directory=str(session_directory),
        episode=int(episode),
        horizon=int(timestamps.size),
        first_timestamp=str(timestamps[0]),
        last_timestamp=str(timestamps[-1]),
        source_policy_cost_reconstructed_eur=source_cost,
        fixed_service_without_stationary_battery_cost_eur=fixed_cost,
        source_stationary_battery_throughput_kwh=source_throughput,
    )
    oracle_batteries, battery_groups = (
        _aggregate_equivalent_batteries(batteries)
        if aggregate_equivalent_batteries
        else (
            batteries,
            [
                {
                    "oracle_building_id": item.building_id,
                    "oracle_action_name": item.action_name,
                    "aggregation_is_exact_for_district_linear_model": True,
                    "members": [
                        {
                            "building_id": item.building_id,
                            "action_name": item.action_name,
                            "schedule_fraction": 1.0,
                        }
                    ],
                }
                for item in batteries
            ],
        )
    )
    metadata = {
        "scope": "conditional_fixed_service_stationary_battery",
        "global_optimum_claim": False,
        "fixed_service_source": str(session_directory),
        "schema_path": str(schema_path),
        "schema_sha256": hashlib.sha256(raw_schema).hexdigest(),
        "price_column": price_columns[0],
        "price_numeric_unit": "dataset tariff unit; reported as EUR for scorecard consistency",
        "first_timestamp": str(timestamps[0]),
        "last_timestamp": str(timestamps[-1]),
        "conservative_capacity_ratio": float(conservative_capacity_ratio),
        "conservative_model": (
            "minimum capacity-power curve rate, 0.99 times minimum directional "
            "efficiency, terminal energy not below initial"
        ),
        "requires_citylearn_replay": True,
        "physical_battery_count": len(batteries),
        "oracle_battery_group_count": len(oracle_batteries),
        "battery_groups": battery_groups,
        "diagnostics": diagnostics.to_dict(),
    }
    problem = PerfectForesightProblem(
        problem_id=problem_id,
        timestep_hours=timestep_hours,
        building_ids=tuple(building_ids),
        price_eur_per_kwh=prices,
        base_net_load_kwh=fixed_base_array,
        batteries=tuple(oracle_batteries),
        metadata=metadata,
    )
    return FixedServiceProblem(problem=problem, diagnostics=diagnostics)
