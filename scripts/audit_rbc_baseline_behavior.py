"""Audit RBC baseline behavior from exported CityLearn timeseries and KPIs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


COST_KPI_CANDIDATES = (
    "district_cost_community_market_settled_total_eur",
    "district_community_settled_cost_total_eur",
    "district_cost_total_control_eur",
)

KPI_CANDIDATES: Mapping[str, Sequence[str]] = {
    "community_cost_eur": COST_KPI_CANDIDATES,
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
    "v2g_export_kwh": ("district_ev_total_v2g_export_kwh",),
    "ev_min_acceptable_feasible_rate": (
        "district_ev_performance_departure_min_acceptable_feasible_ratio",
    ),
    "ev_within_tolerance_feasible_rate": (
        "district_ev_performance_departure_within_tolerance_feasible_ratio",
    ),
    "electrical_violation_kwh": ("district_electrical_service_phase_violations_energy_total_kwh",),
}

DISTRICT_COLUMN = "District"
EPS = 1.0e-9


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

    for path in sorted(data_dir.glob("exported_data_building_*_battery_ep*.csv")):
        frame = _asset_frame(path, buildings, community)
        if frame is None or "Battery (Dis)Charge-kWh" not in frame.columns:
            continue
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

    return _weighted_metric_summary(charge_rows, "battery_charge_kwh") | _weighted_metric_summary(
        discharge_rows,
        "battery_discharge_kwh",
    )


def _summarize_ev(
    data_dir: Path,
    buildings: Mapping[int, pd.DataFrame],
    community: pd.DataFrame,
) -> dict[str, float | None]:
    charge_rows: list[tuple[float, dict[str, float | None]]] = []
    v2g_rows: list[tuple[float, dict[str, float | None]]] = []

    for path in sorted(data_dir.glob("exported_data_building_*_charger_*_ep*.csv")):
        frame = _asset_frame(path, buildings, community)
        if frame is None or "Charging Action-kWh" not in frame.columns:
            continue
        action = pd.to_numeric(frame["Charging Action-kWh"], errors="coerce").fillna(0.0)
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

    return _weighted_metric_summary(charge_rows, "ev_charge_kwh") | _weighted_metric_summary(
        v2g_rows,
        "ev_v2g_kwh",
    )


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
        "pass_ev_service": _status_flag(row.get("ev_min_acceptable_feasible_rate"), minimum=0.99),
        "pass_electrical": _status_flag(row.get("electrical_violation_kwh"), maximum=1.0e-6),
    }


def _audit_run(run_name: str, data_dir: Path) -> dict[str, Any]:
    community = _load_community(data_dir)
    buildings, passive_community = _load_buildings(data_dir)
    community = community.merge(passive_community, on="timestamp", how="left")

    row: dict[str, Any] = {
        "run": run_name,
        "data_dir": str(data_dir),
        "time_steps": int(len(community)),
        "passive_community_surplus_hours": int((community["passive_community_surplus_kwh"] > 0.5).sum())
        if len(community)
        else 0,
        "net_export_hours": int((community["community_net_kwh"] < -0.5).sum()) if len(community) else 0,
    }
    row.update(_read_kpis(data_dir))
    row.update(_summarize_storage(data_dir, buildings, community))
    row.update(_summarize_ev(data_dir, buildings, community))
    row.update(_behavior_flags(row))
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

    (output_dir / "baseline_behavior_summary.json").write_text(
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
        "- `pass_electrical`: electrical violation energy is zero.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="name=simulation_data_dir")
    args = parser.parse_args()

    rows = [_audit_run(name, path) for name, path in (_parse_run_argument(raw) for raw in args.run)]
    _write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
