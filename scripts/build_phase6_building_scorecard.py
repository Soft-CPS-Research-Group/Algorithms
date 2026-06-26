#!/usr/bin/env python3
"""Build a per-building Phase 6J/6K scorecard from collected remote job KPIs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.build_phase6_remote_scorecard import enrich_row, _fmt, _safe_float
from scripts.collect_remote_results import _safe_name


BUILDING_METRICS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("cost_eur", "building_cost_total_control_eur"),
        ("cost_bau_eur", "building_cost_total_business_as_usual_eur"),
        ("cost_delta_to_bau_eur", "building_cost_total_delta_to_business_as_usual_eur"),
        ("cost_ratio_to_bau", "building_cost_ratio_to_business_as_usual_total_ratio"),
        ("community_market_settled_cost_eur", "building_cost_community_market_settled_total_eur"),
        ("grid_import_kwh", "building_energy_grid_total_import_control_kwh"),
        ("grid_export_kwh", "building_energy_grid_total_export_control_kwh"),
        ("grid_net_exchange_kwh", "building_energy_grid_total_net_exchange_control_kwh"),
        ("grid_net_exchange_delta_to_bau_kwh", "building_energy_grid_total_net_exchange_delta_to_business_as_usual_kwh"),
        ("ev_departure_count", "building_ev_events_departure_count"),
        ("ev_min_acceptable_feasible_count", "building_ev_events_departure_min_acceptable_feasible_count"),
        ("ev_min_acceptable_infeasible_count", "building_ev_events_departure_min_acceptable_infeasible_count"),
        ("ev_min_acceptable_feasible_rate", "building_ev_performance_departure_min_acceptable_feasible_ratio"),
        ("ev_within_tolerance_feasible_rate", "building_ev_performance_departure_within_tolerance_feasible_ratio"),
        ("ev_soc_deficit_mean_ratio", "building_ev_performance_departure_soc_deficit_mean_ratio"),
        ("ev_soc_surplus_mean_ratio", "building_ev_performance_departure_soc_surplus_mean_ratio"),
        ("ev_soc_absolute_error_mean_ratio", "building_ev_performance_departure_soc_absolute_error_mean_ratio"),
        ("ev_charge_kwh", "building_ev_total_charge_kwh"),
        ("ev_v2g_export_kwh", "building_ev_total_v2g_export_kwh"),
        ("electrical_violation_kwh", "building_electrical_service_phase_violations_energy_total_kwh"),
        ("electrical_violation_events", "building_electrical_service_phase_violations_event_count"),
        ("phase_imbalance_ratio", "building_electrical_service_phase_imbalance_phase_average_ratio"),
        ("battery_charge_kwh", "building_battery_total_charge_kwh"),
        ("battery_discharge_kwh", "building_battery_total_discharge_kwh"),
        ("battery_throughput_kwh", "building_battery_total_throughput_kwh"),
        ("battery_throughput_ratio_to_bau", "building_battery_ratio_to_business_as_usual_throughput_ratio"),
        ("solar_generation_kwh", "building_solar_self_consumption_total_generation_kwh"),
        ("solar_export_kwh", "building_solar_self_consumption_total_export_kwh"),
        ("solar_self_consumption_rate", "building_solar_self_consumption_ratio_self_consumption_ratio"),
        (
            "community_local_share_of_demand_rate",
            "building_solar_self_consumption_community_market_local_share_of_demand_ratio",
        ),
        (
            "community_local_share_of_export_rate",
            "building_solar_self_consumption_community_market_local_share_of_export_ratio",
        ),
        ("deferrable_service_level_rate", "building_deferrable_appliance_service_service_level_ratio"),
        ("deferrable_missed_cycles_count", "building_deferrable_appliance_service_missed_cycles_count"),
        ("deferrable_unserved_energy_kwh", "building_deferrable_appliance_service_unserved_energy_total_kwh"),
    ]
)

BASE_COLUMNS = [
    "dataset",
    "variant",
    "track",
    "policy",
    "seed",
    "status",
    "target_host",
    "job_id",
    "job_name",
    "config_path",
    "image_tag",
    "building",
    "building_index",
    "flags",
]


def _parse_kpi_matrix(path: Path) -> tuple[dict[str, dict[str, float | None]], list[str]]:
    matrix: dict[str, dict[str, float | None]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return matrix, []
        key_field = reader.fieldnames[0]
        entities = list(reader.fieldnames[1:])
        for row in reader:
            key = str(row.get(key_field) or "").strip()
            if not key:
                continue
            matrix[key] = {entity: _safe_float(row.get(entity)) for entity in entities}
    return matrix, entities


def _building_sort_key(name: str) -> tuple[int, str]:
    prefix = "Building_"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            return int(suffix), name
    return math.inf, name


def _building_entities(entities: list[str]) -> list[str]:
    return sorted([entity for entity in entities if entity.startswith("Building_")], key=_building_sort_key)


def _building_index(building: str) -> str:
    if building.startswith("Building_") and building.split("_", 1)[1].isdigit():
        return building.split("_", 1)[1]
    return ""


def _metric(matrix: dict[str, dict[str, float | None]], output_key: str, building: str) -> float | None:
    kpi_name = BUILDING_METRICS[output_key]
    return matrix.get(kpi_name, {}).get(building)


def _flags_for_row(
    row: dict[str, Any],
    *,
    ev_feasible_min: float,
    ev_within_min: float,
    max_grid_violation_kwh: float,
    battery_throughput_ratio_warn: float,
    solar_self_consumption_warn: float,
) -> str:
    flags: list[str] = []
    building = str(row.get("building") or "")
    if building == "Building_15":
        flags.append("building_15")
    if str(row.get("status") or "").lower() not in {"finished", "completed"}:
        flags.append("not_finished")

    cost_delta = _safe_float(row.get("cost_delta_to_bau_eur"))
    cost_ratio = _safe_float(row.get("cost_ratio_to_bau"))
    if (cost_delta is not None and cost_delta > 1e-9) or (
        cost_ratio is not None and cost_ratio > 1.0 + 1e-9
    ):
        flags.append("cost_worse_than_bau")

    departures = _safe_float(row.get("ev_departure_count")) or 0.0
    ev_feasible = _safe_float(row.get("ev_min_acceptable_feasible_rate"))
    ev_within = _safe_float(row.get("ev_within_tolerance_feasible_rate"))
    ev_infeasible = _safe_float(row.get("ev_min_acceptable_infeasible_count")) or 0.0
    if departures > 0 and ev_feasible is not None and ev_feasible < ev_feasible_min:
        flags.append("ev_service_below_gate")
    if departures > 0 and ev_within is not None and ev_within < ev_within_min:
        flags.append("ev_precision_below_gate")
    if ev_infeasible > 0:
        flags.append("ev_infeasible_departures")

    violation_kwh = _safe_float(row.get("electrical_violation_kwh")) or 0.0
    violation_events = _safe_float(row.get("electrical_violation_events")) or 0.0
    if violation_kwh > max_grid_violation_kwh or violation_events > 0:
        flags.append("grid_violation")

    battery_ratio = _safe_float(row.get("battery_throughput_ratio_to_bau"))
    if battery_ratio is not None and battery_ratio > battery_throughput_ratio_warn:
        flags.append("battery_throughput_high")

    solar_generation = _safe_float(row.get("solar_generation_kwh")) or 0.0
    solar_self_consumption = _safe_float(row.get("solar_self_consumption_rate"))
    if (
        solar_generation > 1e-9
        and solar_self_consumption is not None
        and solar_self_consumption < solar_self_consumption_warn
    ):
        flags.append("solar_self_consumption_low")

    v2g_export = _safe_float(row.get("ev_v2g_export_kwh")) or 0.0
    if v2g_export > 1e-9:
        flags.append("v2g_used")

    missed_cycles = _safe_float(row.get("deferrable_missed_cycles_count")) or 0.0
    unserved_energy = _safe_float(row.get("deferrable_unserved_energy_kwh")) or 0.0
    if missed_cycles > 0 or unserved_energy > 1e-9:
        flags.append("deferrable_service_gap")

    return ";".join(dict.fromkeys(flags))


def _read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [enrich_row(row) for row in csv.DictReader(handle)]


def build_building_scorecard(
    results_dir: Path,
    *,
    summary_csv: Path | None = None,
    ev_feasible_min: float = 0.999,
    ev_within_min: float = 0.80,
    max_grid_violation_kwh: float = 1e-6,
    battery_throughput_ratio_warn: float = 3.0,
    solar_self_consumption_warn: float = 0.50,
) -> list[dict[str, Any]]:
    summary_path = summary_csv or results_dir / "summary.csv"
    summary_rows = _read_summary(summary_path)
    rows: list[dict[str, Any]] = []

    for job in summary_rows:
        job_id = str(job.get("job_id") or "").strip()
        if not job_id:
            continue
        kpi_path = results_dir / "jobs" / _safe_name(job_id) / "exported_kpis.csv"
        if not kpi_path.exists():
            continue
        matrix, entities = _parse_kpi_matrix(kpi_path)
        for building in _building_entities(entities):
            row: dict[str, Any] = OrderedDict()
            for key in BASE_COLUMNS:
                if key == "building":
                    row[key] = building
                elif key == "building_index":
                    row[key] = _building_index(building)
                elif key == "flags":
                    continue
                else:
                    row[key] = job.get(key, "")
            for output_key in BUILDING_METRICS:
                row[output_key] = _metric(matrix, output_key, building)
            row["flags"] = _flags_for_row(
                row,
                ev_feasible_min=ev_feasible_min,
                ev_within_min=ev_within_min,
                max_grid_violation_kwh=max_grid_violation_kwh,
                battery_throughput_ratio_warn=battery_throughput_ratio_warn,
                solar_self_consumption_warn=solar_self_consumption_warn,
            )
            rows.append(row)

    return rows


def write_building_scorecard_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(BASE_COLUMNS) + list(BUILDING_METRICS.keys())
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _flagged_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if str(row.get("flags") or "").strip()],
        key=lambda row: (
            0 if row.get("building") == "Building_15" else 1,
            str(row.get("dataset") or ""),
            str(row.get("variant") or ""),
            str(row.get("policy") or ""),
            _building_sort_key(str(row.get("building") or "")),
        ),
    )


def _markdown_row(row: dict[str, Any]) -> str:
    cells = [
        str(row.get("policy") or ""),
        str(row.get("dataset") or ""),
        str(row.get("variant") or ""),
        str(row.get("building") or ""),
        f"`{row.get('flags') or ''}`",
        _fmt(row.get("cost_delta_to_bau_eur")),
        _fmt(row.get("ev_min_acceptable_feasible_rate")),
        _fmt(row.get("ev_within_tolerance_feasible_rate")),
        _fmt(row.get("electrical_violation_kwh")),
        _fmt(row.get("battery_throughput_ratio_to_bau")),
        _fmt(row.get("solar_self_consumption_rate")),
        _fmt(row.get("ev_v2g_export_kwh")),
        f"`{row.get('job_id') or ''}`",
    ]
    return "| " + " | ".join(cells) + " |"


def write_building_scorecard_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flag_counts: OrderedDict[str, int] = OrderedDict()
    for row in rows:
        for flag in str(row.get("flags") or "").split(";"):
            if not flag:
                continue
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    lines = [
        "# Phase 6J Building Scorecard",
        "",
        f"Generated at Unix time `{time.time():.0f}`.",
        "",
        "## Flag Counts",
        "",
        "| Flag | Count |",
        "|---|---:|",
    ]
    if not flag_counts:
        lines.append("| - | 0 |")
    else:
        for flag, count in flag_counts.items():
            lines.append(f"| `{flag}` | {count} |")

    lines.extend(
        [
            "",
            "## Flagged Building Rows",
            "",
            "| Policy | Dataset | Variant | Building | Flags | Cost delta BAU | EV feasible | EV within tol | Grid kWh | Battery ratio | Solar self use | V2G export | Job ID |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    flagged = _flagged_rows(rows)
    if not flagged:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - |")
    else:
        for row in flagged[:40]:
            lines.append(_markdown_row(row))

    building_15 = [row for row in rows if row.get("building") == "Building_15"]
    lines.extend(
        [
            "",
            "## Building 15 Focus",
            "",
            "| Policy | Dataset | Variant | Building | Flags | Cost delta BAU | EV feasible | EV within tol | Grid kWh | Battery ratio | Solar self use | V2G export | Job ID |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if not building_15:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - | - |")
    else:
        for row in building_15[:40]:
            lines.append(_markdown_row(row))

    lines.extend(
        [
            "",
            "## How To Use",
            "",
            "- Use this together with `scorecard.md`: the aggregate scorecard says whether a run is good; this file says where it is failing.",
            "- `building_15` is deliberately highlighted because it has been the main suspect for phase/headroom and EV service edge cases.",
            "- `cost_worse_than_bau` at building level can be acceptable if community cost improves, but it deserves inspection when it repeats.",
            "- `battery_throughput_high` and `v2g_used` are not failures by themselves; they show whether storage/V2G is being used enough to explain cost changes.",
            "- `solar_self_consumption_low` points at community-renewable opportunities, not necessarily a broken controller.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ev-feasible-min", type=float, default=0.999)
    parser.add_argument("--ev-within-min", type=float, default=0.80)
    parser.add_argument("--max-grid-violation-kwh", type=float, default=1e-6)
    parser.add_argument("--battery-throughput-ratio-warn", type=float, default=3.0)
    parser.add_argument("--solar-self-consumption-warn", type=float, default=0.50)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or args.results_dir
    rows = build_building_scorecard(
        args.results_dir,
        summary_csv=args.summary_csv,
        ev_feasible_min=args.ev_feasible_min,
        ev_within_min=args.ev_within_min,
        max_grid_violation_kwh=args.max_grid_violation_kwh,
        battery_throughput_ratio_warn=args.battery_throughput_ratio_warn,
        solar_self_consumption_warn=args.solar_self_consumption_warn,
    )
    write_building_scorecard_csv(output_dir / "building_scorecard.csv", rows)
    write_building_scorecard_markdown(output_dir / "building_scorecard.md", rows)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
