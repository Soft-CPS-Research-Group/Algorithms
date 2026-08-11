"""Audit RBC baseline behavior from exported CityLearn timeseries and KPIs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

try:
    from scripts.electrical_safety_evidence import (
        DEFAULT_PROJECTION_EVENT_RATE,
        DEFAULT_PROJECTION_TOLERANCE_KWH,
        executed_safety_evidence,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from electrical_safety_evidence import (  # type: ignore[no-redef]
        DEFAULT_PROJECTION_EVENT_RATE,
        DEFAULT_PROJECTION_TOLERANCE_KWH,
        executed_safety_evidence,
    )


COST_KPI_CANDIDATES = (
    "district_cost_community_market_settled_total_eur",
    "district_community_settled_cost_total_eur",
    "district_cost_total_control_eur",
)

KPI_CANDIDATES: Mapping[str, Sequence[str]] = {
    "community_cost_eur": COST_KPI_CANDIDATES,
    "cost_bau_eur": ("district_cost_total_business_as_usual_eur",),
    "cost_delta_to_bau_eur": ("district_cost_total_delta_to_business_as_usual_eur",),
    "cost_ratio_to_bau": ("district_cost_ratio_to_business_as_usual_total_ratio",),
    "community_import_kwh": ("district_energy_grid_total_import_control_kwh",),
    "community_export_kwh": ("district_energy_grid_total_export_control_kwh",),
    "community_net_exchange_kwh": ("district_energy_grid_total_net_exchange_control_kwh",),
    "community_local_import_kwh": ("district_energy_grid_community_market_local_import_total_kwh",),
    "community_local_export_kwh": ("district_energy_grid_community_market_local_export_total_kwh",),
    "community_market_savings_eur": ("district_cost_community_market_savings_total_eur",),
    "community_market_counterfactual_eur": ("district_cost_community_market_counterfactual_total_eur",),
    "community_solar_self_consumption_rate": ("district_solar_self_consumption_ratio_self_consumption_ratio",),
    "community_market_import_share_rate": (
        "district_solar_self_consumption_community_market_import_share_ratio",
    ),
    "battery_throughput_kwh": ("district_battery_total_throughput_kwh",),
    "battery_throughput_ratio_to_bau": ("district_battery_ratio_to_business_as_usual_throughput_ratio",),
    "v2g_export_kwh": ("district_ev_total_v2g_export_kwh",),
    "ev_min_acceptable_feasible_rate": (
        "district_ev_performance_departure_min_acceptable_feasible_ratio",
    ),
    "ev_within_tolerance_feasible_rate": (
        "district_ev_performance_departure_within_tolerance_feasible_ratio",
    ),
    "ev_soc_deficit_mean_ratio": ("district_ev_performance_departure_soc_deficit_mean_ratio",),
    "ev_soc_surplus_mean_ratio": ("district_ev_performance_departure_soc_surplus_mean_ratio",),
    "ev_soc_absolute_error_mean_ratio": (
        "district_ev_performance_departure_soc_absolute_error_mean_ratio",
    ),
    "electrical_violation_kwh": ("district_electrical_service_phase_violations_energy_total_kwh",),
    "electrical_violation_events": ("district_electrical_service_phase_violations_event_count",),
    "deferrable_completed_cycles_count": (
        "district_deferrable_appliance_service_completed_cycles_count",
    ),
    "deferrable_missed_cycles_count": (
        "district_deferrable_appliance_service_missed_cycles_count",
    ),
    "deferrable_unserved_energy_kwh": (
        "district_deferrable_appliance_service_unserved_energy_total_kwh",
    ),
    "deferrable_service_level_rate": (
        "district_deferrable_appliance_service_service_level_ratio",
    ),
    "peak_daily_ratio_to_bau": (
        "district_energy_grid_shape_quality_peak_daily_average_to_business_as_usual_ratio",
    ),
    "peak_all_time_ratio_to_bau": (
        "district_energy_grid_shape_quality_peak_all_time_average_to_business_as_usual_ratio",
    ),
    "ramping_ratio_to_bau": (
        "district_energy_grid_shape_quality_ramping_average_to_business_as_usual_ratio",
    ),
    "load_factor_penalty_daily_ratio_to_bau": (
        "district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_business_as_usual_ratio",
    ),
    "emissions_kgco2": ("district_emissions_total_control_kgco2",),
    "emissions_ratio_to_bau": ("district_emissions_ratio_to_business_as_usual_total_ratio",),
    "outage_unserved_energy_normalized_rate": (
        "district_comfort_resilience_resilience_unserved_energy_outage_normalized_ratio",
    ),
}

DISTRICT_COLUMN = "District"
EPS = 1.0e-9
EV_MIN_GATE = 0.99
EV_PRECISION_GATE = 0.40
GATE_PROFILE = "phase10_w6_adapted_local_v1"
PROJECTION_GATE_PROFILE = "phase10_w6_executed_safety_projection_v1"


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _building_id(path: Path) -> int | None:
    match = re.search(r"building_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _export_episode_index(data_dir: Path) -> int | None:
    """Return the final exported simulator episode encoded in CSV names."""

    episodes: set[int] = set()
    for path in data_dir.glob("exported_data_*_ep*.csv"):
        match = re.search(r"_ep(\d+)\.csv$", path.name)
        if match:
            episodes.add(int(match.group(1)))
    return max(episodes) if episodes else None


def _weighted_share(mask: pd.Series, energy: pd.Series) -> float | None:
    total = float(energy.sum())
    if total <= EPS:
        return None
    return float(energy[mask.fillna(False)].sum() / total)


def _read_kpis(data_dir: Path) -> dict[str, float | None]:
    candidates = sorted(data_dir.glob("exported_kpis.csv"))
    if not candidates:
        candidates = sorted(data_dir.glob("exported_kpis_ep*.csv"))
    if not candidates:
        return {key: None for key in KPI_CANDIDATES}

    with candidates[-1].open("r", encoding="utf-8", newline="") as handle:
        rows = {row["KPI"]: row for row in csv.DictReader(handle)}

    output: dict[str, float | None] = {}
    for output_key, kpi_names in KPI_CANDIDATES.items():
        fallback: float | None = None
        value: float | None = None
        for index, kpi_name in enumerate(kpi_names):
            candidate = _to_float(rows.get(kpi_name, {}).get(DISTRICT_COLUMN))
            if candidate is None:
                continue
            if output_key == "community_cost_eur" and index < len(kpi_names) - 1:
                if abs(candidate) > EPS:
                    value = candidate
                    break
            else:
                fallback = candidate
                if output_key != "community_cost_eur":
                    value = candidate
                    break
        output[output_key] = value if value is not None else fallback

    settled = _to_float(rows.get("district_cost_community_market_settled_total_eur", {}).get(DISTRICT_COLUMN))
    output["community_market_cost_present"] = 1.0 if settled is not None and abs(settled) > EPS else 0.0
    return output


def _load_pricing(data_dir: Path) -> pd.DataFrame:
    candidates = sorted(data_dir.glob("exported_data_pricing_ep*.csv"))
    if not candidates:
        return pd.DataFrame(columns=["timestamp", "price_rate"])
    frame = pd.read_csv(candidates[-1])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    price_cols = [column for column in frame.columns if column.startswith("electricity_pricing-")]
    if not price_cols:
        frame["price_rate"] = 0.0
    else:
        frame["price_rate"] = pd.to_numeric(frame[price_cols[0]], errors="coerce").fillna(0.0)
    return frame[["timestamp", "price_rate"]]


def _load_community(data_dir: Path) -> pd.DataFrame:
    candidates = sorted(data_dir.glob("exported_data_community_ep*.csv"))
    if not candidates:
        return pd.DataFrame(columns=["timestamp", "community_net_kwh"])
    frame = pd.read_csv(candidates[-1])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["community_net_kwh"] = pd.to_numeric(
        frame.get("Net Electricity Consumption-kWh", 0.0),
        errors="coerce",
    ).fillna(0.0)
    if "Price-$" in frame.columns:
        frame["simulator_price_cost_sum"] = pd.to_numeric(frame["Price-$"], errors="coerce").fillna(0.0)
    else:
        frame["simulator_price_cost_sum"] = 0.0
    pricing = _load_pricing(data_dir)
    return frame[["timestamp", "community_net_kwh", "simulator_price_cost_sum"]].merge(
        pricing,
        on="timestamp",
        how="left",
    )


def _load_buildings(data_dir: Path) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    buildings: dict[int, pd.DataFrame] = {}
    totals: list[pd.DataFrame] = []

    for path in sorted(data_dir.glob("exported_data_building_*_ep*.csv")):
        if "_battery_" in path.name or "_charger_" in path.name or "business_as_usual" in path.name:
            continue
        building_id = _building_id(path)
        if building_id is None:
            continue
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        load = pd.to_numeric(
            frame.get("Non-shiftable Load Electricity Consumption-kWh", 0.0),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)
        pv = -pd.to_numeric(
            frame.get("Energy Production from PV-kWh", 0.0),
            errors="coerce",
        ).fillna(0.0)
        pv = pv.clip(lower=0.0)
        net = pd.to_numeric(frame.get("Net Electricity Consumption-kWh", 0.0), errors="coerce").fillna(0.0)
        building = pd.DataFrame(
            {
                "timestamp": frame["timestamp"],
                "passive_load_kwh": load,
                "passive_pv_kwh": pv,
                "passive_local_surplus_kwh": (pv - load).clip(lower=0.0),
                "passive_local_import_kwh": (load - pv).clip(lower=0.0),
                "building_net_kwh": net,
            }
        )
        buildings[building_id] = building
        totals.append(building[["timestamp", "passive_load_kwh", "passive_pv_kwh"]])

    if not totals:
        return buildings, pd.DataFrame(
            columns=[
                "timestamp",
                "passive_community_surplus_kwh",
                "passive_community_import_kwh",
            ]
        )

    total = totals[0].copy()
    for frame in totals[1:]:
        total = total.merge(frame, on="timestamp", how="outer", suffixes=("", "_next"))
        load_cols = [column for column in total.columns if column.startswith("passive_load_kwh")]
        pv_cols = [column for column in total.columns if column.startswith("passive_pv_kwh")]
        total["passive_load_kwh"] = total[load_cols].sum(axis=1)
        total["passive_pv_kwh"] = total[pv_cols].sum(axis=1)
        total = total[["timestamp", "passive_load_kwh", "passive_pv_kwh"]]

    total["passive_community_surplus_kwh"] = (
        total["passive_pv_kwh"] - total["passive_load_kwh"]
    ).clip(lower=0.0)
    total["passive_community_import_kwh"] = (
        total["passive_load_kwh"] - total["passive_pv_kwh"]
    ).clip(lower=0.0)
    return buildings, total


def _asset_frame(
    path: Path,
    buildings: Mapping[int, pd.DataFrame],
    community: pd.DataFrame,
) -> pd.DataFrame | None:
    building_id = _building_id(path)
    if building_id is None or building_id not in buildings:
        return None
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.merge(buildings[building_id], on="timestamp", how="left").merge(
        community[
            [
                "timestamp",
                "passive_community_surplus_kwh",
                "passive_community_import_kwh",
                "community_net_kwh",
            ]
        ],
        on="timestamp",
        how="left",
    )


def _summarize_storage(
    data_dir: Path,
    buildings: Mapping[int, pd.DataFrame],
    community: pd.DataFrame,
) -> dict[str, float | None]:
    charge_rows: list[tuple[float, dict[str, float | None]]] = []
    discharge_rows: list[tuple[float, dict[str, float | None]]] = []
    soc_values: list[pd.Series] = []

    for path in sorted(data_dir.glob("exported_data_building_*_battery_ep*.csv")):
        frame = _asset_frame(path, buildings, community)
        if frame is None or "Battery (Dis)Charge-kWh" not in frame.columns:
            continue
        if "Battery Soc-%" in frame.columns:
            soc_values.append(pd.to_numeric(frame["Battery Soc-%"], errors="coerce").dropna())
        energy = pd.to_numeric(frame["Battery (Dis)Charge-kWh"], errors="coerce").fillna(0.0)
        charge = energy.clip(lower=0.0)
        discharge = (-energy).clip(lower=0.0)
        if float(charge.sum()) > EPS:
            charge_rows.append(
                (
                    float(charge.sum()),
                    {
                        "battery_charge_local_surplus_share": _weighted_share(
                            frame["passive_local_surplus_kwh"] > 0.1,
                            charge,
                        ),
                        "battery_charge_community_surplus_share": _weighted_share(
                            frame["passive_community_surplus_kwh"] > 0.5,
                            charge,
                        ),
                        "battery_charge_net_export_share": _weighted_share(
                            frame["community_net_kwh"] < -0.5,
                            charge,
                        ),
                    },
                )
            )
        if float(discharge.sum()) > EPS:
            discharge_rows.append(
                (
                    float(discharge.sum()),
                    {
                        "battery_discharge_local_import_share": _weighted_share(
                            frame["passive_local_import_kwh"] > 0.1,
                            discharge,
                        ),
                        "battery_discharge_community_import_share": _weighted_share(
                            frame["passive_community_import_kwh"] > 0.5,
                            discharge,
                        ),
                        "battery_discharge_during_local_surplus_share": _weighted_share(
                            frame["passive_local_surplus_kwh"] > 0.1,
                            discharge,
                        ),
                        "battery_discharge_during_community_surplus_share": _weighted_share(
                            frame["passive_community_surplus_kwh"] > 0.5,
                            discharge,
                        ),
                        "battery_discharge_net_export_share": _weighted_share(
                            frame["community_net_kwh"] < -0.5,
                            discharge,
                        ),
                    },
                )
            )

    output = _weighted_metric_summary(charge_rows, "battery_charge_kwh") | _weighted_metric_summary(
        discharge_rows,
        "battery_discharge_kwh",
    )
    if soc_values:
        soc = pd.concat(soc_values, ignore_index=True)
        output.update(
            {
                "storage_soc_min": float(soc.min()),
                "storage_soc_max": float(soc.max()),
                "storage_soc_violation_count": int(((soc < -1.0e-6) | (soc > 1.0 + 1.0e-6)).sum()),
            }
        )
    else:
        output.update(
            {
                "storage_soc_min": None,
                "storage_soc_max": None,
                "storage_soc_violation_count": None,
            }
        )
    return output


def _summarize_ev(
    data_dir: Path,
    buildings: Mapping[int, pd.DataFrame],
    community: pd.DataFrame,
) -> dict[str, float | None]:
    charge_rows: list[tuple[float, dict[str, float | None]]] = []
    v2g_rows: list[tuple[float, dict[str, float | None]]] = []
    connected_actions: list[pd.Series] = []

    for path in sorted(data_dir.glob("exported_data_building_*_charger_*_ep*.csv")):
        frame = _asset_frame(path, buildings, community)
        if frame is None or "Charging Action-kWh" not in frame.columns:
            continue
        action = pd.to_numeric(frame["Charging Action-kWh"], errors="coerce").fillna(0.0)
        if "Is EV Connected" in frame.columns:
            connected = frame["Is EV Connected"].astype(str).str.strip().str.lower().eq("true")
            connected_actions.append(action[connected])
        charge = action.clip(lower=0.0)
        v2g = (-action).clip(lower=0.0)
        departure_hours = pd.to_numeric(frame.get("EV Departure Time", 999.0), errors="coerce").fillna(999.0)
        current_soc = pd.to_numeric(frame.get("EV SOC-%", 0.0), errors="coerce").fillna(0.0)
        required_soc = pd.to_numeric(frame.get("EV Required SOC Departure-%", 0.0), errors="coerce").fillna(0.0)
        urgent = (departure_hours <= 4.0) | ((required_soc - current_soc) > 0.10)
        no_community_surplus = frame["passive_community_surplus_kwh"] <= 0.5

        if float(charge.sum()) > EPS:
            charge_rows.append(
                (
                    float(charge.sum()),
                    {
                        "ev_charge_local_surplus_share": _weighted_share(
                            frame["passive_local_surplus_kwh"] > 0.1,
                            charge,
                        ),
                        "ev_charge_community_surplus_share": _weighted_share(
                            frame["passive_community_surplus_kwh"] > 0.5,
                            charge,
                        ),
                        "ev_charge_net_export_share": _weighted_share(
                            frame["community_net_kwh"] < -0.5,
                            charge,
                        ),
                        "ev_charge_no_surplus_urgent_share": _weighted_share(
                            no_community_surplus & urgent,
                            charge,
                        ),
                        "ev_charge_no_surplus_nonurgent_share": _weighted_share(
                            no_community_surplus & ~urgent,
                            charge,
                        ),
                    },
                )
            )
        if float(v2g.sum()) > EPS:
            v2g_rows.append(
                (
                    float(v2g.sum()),
                    {
                        "ev_v2g_local_import_share": _weighted_share(
                            frame["passive_local_import_kwh"] > 0.1,
                            v2g,
                        ),
                        "ev_v2g_community_import_share": _weighted_share(
                            frame["passive_community_import_kwh"] > 0.5,
                            v2g,
                        ),
                        "ev_v2g_during_local_surplus_share": _weighted_share(
                            frame["passive_local_surplus_kwh"] > 0.1,
                            v2g,
                        ),
                        "ev_v2g_during_community_surplus_share": _weighted_share(
                            frame["passive_community_surplus_kwh"] > 0.5,
                            v2g,
                        ),
                        "ev_v2g_net_export_share": _weighted_share(frame["community_net_kwh"] < -0.5, v2g),
                    },
                )
            )

    output = _weighted_metric_summary(charge_rows, "ev_charge_kwh") | _weighted_metric_summary(
        v2g_rows,
        "ev_v2g_kwh",
    )
    if connected_actions:
        actions = pd.concat(connected_actions, ignore_index=True)
        output.update(
            {
                "ev_connected_action_count": int(len(actions)),
                "ev_connected_charge_action_rate": float((actions > 0.01).mean()),
                "ev_connected_idle_action_rate": float((actions.abs() <= 0.01).mean()),
                "ev_connected_v2g_action_rate": float((actions < -0.01).mean()),
            }
        )
    else:
        output.update(
            {
                "ev_connected_action_count": 0,
                "ev_connected_charge_action_rate": None,
                "ev_connected_idle_action_rate": None,
                "ev_connected_v2g_action_rate": None,
            }
        )
    return output


def _weighted_metric_summary(
    rows: Sequence[tuple[float, Mapping[str, float | None]]],
    total_key: str,
) -> dict[str, float | None]:
    total = sum(energy for energy, _metrics in rows)
    output: dict[str, float | None] = {total_key: float(total)}
    if total <= EPS or not rows:
        return output
    keys = sorted({key for _energy, metrics in rows for key in metrics})
    for key in keys:
        numerator = 0.0
        denominator = 0.0
        for energy, metrics in rows:
            value = metrics.get(key)
            if value is None:
                continue
            numerator += energy * value
            denominator += energy
        output[key] = float(numerator / denominator) if denominator > EPS else None
    return output


def _status_flag(value: float | None, *, minimum: float | None = None, maximum: float | None = None) -> int:
    if value is None:
        return 0
    if minimum is not None and value < minimum:
        return 0
    if maximum is not None and value > maximum:
        return 0
    return 1


def _behavior_flags(row: dict[str, Any]) -> dict[str, int]:
    return {
        "pass_market_cost": _status_flag(row.get("community_market_cost_present"), minimum=0.5),
        "pass_battery_charge_surplus": _status_flag(
            row.get("battery_charge_community_surplus_share"),
            minimum=0.75,
        ),
        "pass_battery_discharge_import": _status_flag(
            row.get("battery_discharge_community_import_share"),
            minimum=0.75,
        ),
        "pass_battery_discharge_not_surplus": _status_flag(
            row.get("battery_discharge_during_community_surplus_share"),
            maximum=0.20,
        ),
        "pass_ev_v2g_import": _status_flag(row.get("ev_v2g_community_import_share"), minimum=0.75)
        if (row.get("ev_v2g_kwh") or 0.0) > EPS
        else 1,
        "pass_ev_charge_not_nonurgent_grid": _status_flag(
            row.get("ev_charge_no_surplus_nonurgent_share"),
            maximum=0.05,
        ),
        "pass_ev_service": _status_flag(
            row.get("ev_min_acceptable_feasible_rate"),
            minimum=EV_MIN_GATE,
        ),
        "pass_ev_precision": _status_flag(
            row.get("ev_within_tolerance_feasible_rate"),
            minimum=EV_PRECISION_GATE,
        ),
        "pass_electrical": _status_flag(row.get("electrical_violation_kwh"), maximum=1.0e-6),
        "pass_electrical_events": _status_flag(row.get("electrical_violation_events"), maximum=0.0),
        "pass_deferrable_service": int(
            _status_flag(row.get("deferrable_missed_cycles_count"), maximum=0.0)
            and _status_flag(row.get("deferrable_unserved_energy_kwh"), maximum=1.0e-6)
            and _status_flag(row.get("deferrable_service_level_rate"), minimum=0.99)
        ),
        "pass_storage_soc": _status_flag(row.get("storage_soc_violation_count"), maximum=0.0),
        "pass_outage": _status_flag(
            row.get("outage_unserved_energy_normalized_rate"),
            maximum=1.0e-9,
        ),
    }


def _hard_gate_decision(row: Mapping[str, Any]) -> str:
    failures = []
    for flag, label in (
        ("pass_ev_service", "ev_service"),
        ("pass_electrical", "electrical_energy"),
        ("pass_electrical_events", "electrical_events"),
        ("pass_deferrable_service", "deferrable_service"),
        ("pass_storage_soc", "storage_soc"),
        ("pass_outage", "outage"),
    ):
        if int(row.get(flag, 0) or 0) != 1:
            failures.append(label)
    return "PASS_HARD_GATES" if not failures else "REJECT_" + "+".join(failures)


def _learning_gate_decision(row: Mapping[str, Any]) -> str:
    hard_gate = _hard_gate_decision(row)
    if hard_gate != "PASS_HARD_GATES":
        return hard_gate
    if int(row.get("pass_ev_precision", 0) or 0) != 1:
        return "REJECT_ev_precision"
    return "PASS_LEARNING_GATES"


def _projection_tolerant_hard_gate_decision(row: Mapping[str, Any]) -> str:
    """Apply the separately named projection-tolerant safety profile.

    The historical strict decision is intentionally untouched.  A tolerant
    pass is possible only when electrical clipping is the sole hard-gate
    failure and exported post-projection peaks certify every configured limit.
    """

    strict = _hard_gate_decision(row)
    if strict == "PASS_HARD_GATES":
        return strict
    failures = []
    for flag, label in (
        ("pass_ev_service", "ev_service"),
        ("pass_deferrable_service", "deferrable_service"),
        ("pass_storage_soc", "storage_soc"),
        ("pass_outage", "outage"),
    ):
        if int(row.get(flag, 0) or 0) != 1:
            failures.append(label)
    if failures:
        return "REJECT_" + "+".join(failures)
    if (
        int(row.get("executed_electrical_safety_certified", 0) or 0) == 1
        and int(row.get("projection_request_within_tolerance", 0) or 0) == 1
    ):
        return "PASS_WITH_SAFETY_PROJECTION"
    return strict


def _projection_tolerant_learning_gate_decision(row: Mapping[str, Any]) -> str:
    hard_gate = _projection_tolerant_hard_gate_decision(row)
    if not hard_gate.startswith("PASS_"):
        return hard_gate
    if int(row.get("pass_ev_precision", 0) or 0) != 1:
        return "REJECT_ev_precision"
    return hard_gate


def _audit_run(
    run_name: str,
    data_dir: Path,
    *,
    projection_tolerance_kwh: float = DEFAULT_PROJECTION_TOLERANCE_KWH,
    projection_max_event_rate: float = DEFAULT_PROJECTION_EVENT_RATE,
) -> dict[str, Any]:
    community = _load_community(data_dir)
    buildings, passive_community = _load_buildings(data_dir)
    community = community.merge(passive_community, on="timestamp", how="left")

    row: dict[str, Any] = {
        "run": run_name,
        "data_dir": str(data_dir),
        "gate_profile": GATE_PROFILE,
        "projection_gate_profile": PROJECTION_GATE_PROFILE,
        "ev_min_gate": EV_MIN_GATE,
        "ev_precision_gate": EV_PRECISION_GATE,
        "time_steps": int(len(community)),
        "export_episode_index": _export_episode_index(data_dir),
        "passive_community_surplus_hours": int((community["passive_community_surplus_kwh"] > 0.5).sum())
        if len(community)
        else 0,
        "net_export_hours": int((community["community_net_kwh"] < -0.5).sum()) if len(community) else 0,
    }
    row.update(_read_kpis(data_dir))
    row.update(_summarize_storage(data_dir, buildings, community))
    row.update(_summarize_ev(data_dir, buildings, community))
    row.update(_behavior_flags(row))
    row.update(
        executed_safety_evidence(
            data_dir,
            requested_violation_kwh=row.get("electrical_violation_kwh"),
            requested_violation_events=row.get("electrical_violation_events"),
            tolerance_kwh=projection_tolerance_kwh,
            max_event_rate=projection_max_event_rate,
            time_steps=int(len(community)),
        )
    )
    row["hard_gate_decision"] = _hard_gate_decision(row)
    row["learning_gate_decision"] = _learning_gate_decision(row)
    row["projection_tolerant_hard_gate_decision"] = _projection_tolerant_hard_gate_decision(row)
    row["projection_tolerant_learning_gate_decision"] = _projection_tolerant_learning_gate_decision(row)
    row["behavior_pass_count"] = int(sum(value for key, value in row.items() if key.startswith("pass_")))
    return row


def _parse_run_argument(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    name, path = raw.split("=", 1)
    return name, Path(path)


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with (output_dir / "baseline_behavior_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Keep a conventional scorecard filename for experiment-history tooling
    # while retaining the historical behavior-audit outputs.
    with (output_dir / "scorecard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "baseline_behavior_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "scorecard.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    ranked = sorted(rows, key=lambda row: (row.get("community_cost_eur") is None, row.get("community_cost_eur") or 0.0))
    with (output_dir / "baseline_behavior_ranked.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ranked)

    readme = [
        "# RBC Baseline Behavior Audit",
        "",
        "Generated files:",
        "",
        "- `baseline_behavior_summary.csv`",
        "- `baseline_behavior_summary.json`",
        "- `baseline_behavior_ranked.csv`",
        "- `scorecard.csv` / `scorecard.json` (history-tooling aliases)",
        "",
        "Main pass flags:",
        "",
        "- `pass_market_cost`: settled community-market cost is present.",
        "- `pass_battery_charge_surplus`: most stationary battery charging happened during passive community surplus.",
        "- `pass_battery_discharge_import`: most stationary battery discharge happened during passive community import.",
        "- `pass_battery_discharge_not_surplus`: stationary battery rarely discharged during passive community surplus.",
        "- `pass_ev_v2g_import`: V2G mostly happened during passive community import.",
        "- `pass_ev_charge_not_nonurgent_grid`: EV charging without community surplus was mostly urgent/service-driven.",
        "- `pass_ev_service`: feasible EV minimum service is at least 0.99.",
        "- `pass_ev_precision`: feasible EV within-tolerance rate is at least 0.40.",
        "- `pass_electrical`: electrical violation energy is zero.",
        "- `pass_electrical_events`: no electrical violation events occurred.",
        "- `pass_deferrable_service`: no missed cycles or unserved deferrable energy.",
        "- `pass_storage_soc`: exported stationary-storage SOC stayed in [0, 1].",
        "- `pass_outage`: no normalized outage unserved energy.",
        "- `hard_gate_decision`: rejects a run if any mandatory safety/service gate fails.",
        "- `learning_gate_decision`: also applies the Phase 10 W6 EV precision gate (>= 0.40).",
        "- `projection_tolerant_hard_gate_decision`: separate profile; permits only small pre-projection clipping when exported executed peaks certify every configured limit.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="name=simulation_data_dir")
    parser.add_argument(
        "--projection-tolerance-kwh",
        type=float,
        default=DEFAULT_PROJECTION_TOLERANCE_KWH,
    )
    parser.add_argument(
        "--projection-max-event-rate",
        type=float,
        default=DEFAULT_PROJECTION_EVENT_RATE,
    )
    args = parser.parse_args()

    rows = [
        _audit_run(
            name,
            path,
            projection_tolerance_kwh=args.projection_tolerance_kwh,
            projection_max_event_rate=args.projection_max_event_rate,
        )
        for name, path in (_parse_run_argument(raw) for raw in args.run)
    ]
    _write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
