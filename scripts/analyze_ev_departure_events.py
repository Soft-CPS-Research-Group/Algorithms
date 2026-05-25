"""Summarise EV departure events from exported simulator CSV data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd


CHARGER_FILE_RE = re.compile(
    r"exported_data_building_(?P<building>[^_]+)_charger_(?P<charger>.+)_ep(?P<episode>\d+)\.csv$"
)
SOC_EPSILON = 1.0e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, help="Run job directory containing results/simulation_data.")
    parser.add_argument("--output-dir", required=True, help="Directory where event CSV/JSON files are written.")
    parser.add_argument("--episode", type=int, default=None, help="Episode number to inspect. Defaults to last episode.")
    parser.add_argument("--tolerance", type=float, default=0.05, help="SOC tolerance around the requested target.")
    return parser.parse_args()


def _simulation_data_dir(job_dir: Path) -> Path:
    root = job_dir / "results" / "simulation_data"
    if not root.exists():
        raise FileNotFoundError(f"simulation_data directory not found under {job_dir}")
    children = [path for path in root.iterdir() if path.is_dir()]
    if len(children) == 1:
        return children[0]
    return root


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _charger_files(simulation_dir: Path, episode: int | None) -> list[Path]:
    matches: list[tuple[int, Path]] = []
    for path in simulation_dir.glob("exported_data_building_*_charger_*_ep*.csv"):
        match = CHARGER_FILE_RE.match(path.name)
        if match is None:
            continue
        file_episode = int(match.group("episode"))
        matches.append((file_episode, path))
    if not matches:
        return []
    selected_episode = episode if episode is not None else max(ep for ep, _ in matches)
    return sorted(path for ep, path in matches if ep == selected_episode)


def _event_rows(path: Path, tolerance: float) -> list[dict[str, Any]]:
    match = CHARGER_FILE_RE.match(path.name)
    if match is None:
        return []
    building_id = match.group("building")
    charger_id = match.group("charger")
    episode = int(match.group("episode"))
    frame = pd.read_csv(path)
    required_columns = {
        "timestamp",
        "EV SOC-%",
        "EV Required SOC Departure-%",
        "EV Departure Time",
        "Is EV Connected",
    }
    if not required_columns.issubset(frame.columns):
        return []

    connected = frame["Is EV Connected"].astype(bool)
    departure_time = pd.to_numeric(frame["EV Departure Time"], errors="coerce")
    departures = frame[connected & (departure_time == 0)].copy()
    if "EV Name" in departures.columns:
        departures = departures.drop_duplicates(subset=["EV Name"], keep="first")
    else:
        departures = departures.head(1)
    rows: list[dict[str, Any]] = []
    for _, event in departures.iterrows():
        soc = _finite_float(event.get("EV SOC-%"))
        required_soc = _finite_float(event.get("EV Required SOC Departure-%"))
        if soc is None or required_soc is None:
            continue
        min_acceptable_soc = max(required_soc - tolerance, 0.0)
        deficit = max(required_soc - soc, 0.0)
        surplus = max(soc - required_soc, 0.0)
        shortfall_beyond_tolerance = max(min_acceptable_soc - soc, 0.0)
        success_strict = soc + SOC_EPSILON >= required_soc
        success_min_acceptable = soc + SOC_EPSILON >= min_acceptable_soc
        within_tolerance = abs(soc - required_soc) <= tolerance + SOC_EPSILON
        rows.append(
            {
                "episode": episode,
                "building_id": building_id,
                "charger_id": charger_id,
                "timestamp": event.get("timestamp"),
                "ev_name": event.get("EV Name") or event.get("Incoming EV Name") or "",
                "soc": soc,
                "required_soc": required_soc,
                "min_acceptable_soc": min_acceptable_soc,
                "tolerance": tolerance,
                "success_strict": success_strict,
                "success_min_acceptable": success_min_acceptable,
                "within_tolerance": within_tolerance,
                "deficit": deficit,
                "shortfall_beyond_tolerance": shortfall_beyond_tolerance,
                "surplus": surplus,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "building_id",
        "charger_id",
        "timestamp",
        "ev_name",
        "soc",
        "required_soc",
        "min_acceptable_soc",
        "tolerance",
        "success_strict",
        "success_min_acceptable",
        "within_tolerance",
        "deficit",
        "shortfall_beyond_tolerance",
        "surplus",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if bool(row[key])) / len(rows)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event_count": len(rows),
        "success_strict_rate": _rate(rows, "success_strict"),
        "success_min_acceptable_rate": _rate(rows, "success_min_acceptable"),
        "within_tolerance_rate": _rate(rows, "within_tolerance"),
        "deficit_mean": mean([float(row["deficit"]) for row in rows]) if rows else None,
        "shortfall_beyond_tolerance_mean": (
            mean([float(row["shortfall_beyond_tolerance"]) for row in rows]) if rows else None
        ),
        "surplus_mean": mean([float(row["surplus"]) for row in rows]) if rows else None,
        "failed_min_acceptable": [
            row
            for row in rows
            if not bool(row["success_min_acceptable"])
        ],
        "outside_tolerance": [
            row
            for row in rows
            if not bool(row["within_tolerance"])
        ],
    }


def _exported_kpi_summary(simulation_dir: Path) -> dict[str, float]:
    """Return official district EV departure KPIs when exported by the simulator."""

    path = simulation_dir / "exported_kpis.csv"
    if not path.exists():
        return {}

    frame = pd.read_csv(path)
    if "KPI" not in frame.columns or "District" not in frame.columns:
        return {}

    wanted = {
        "district_ev_events_departure_count": "official_departure_count",
        "district_ev_events_departure_target_feasible_count": "official_target_feasible_count",
        "district_ev_events_departure_target_infeasible_count": "official_target_infeasible_count",
        "district_ev_events_departure_min_acceptable_count": "official_min_acceptable_count",
        "district_ev_events_departure_min_acceptable_feasible_count": (
            "official_min_acceptable_feasible_count"
        ),
        "district_ev_events_departure_min_acceptable_infeasible_count": (
            "official_min_acceptable_infeasible_count"
        ),
        "district_ev_events_departure_within_tolerance_count": "official_within_tolerance_count",
        "district_ev_events_departure_within_tolerance_feasible_count": (
            "official_within_tolerance_feasible_count"
        ),
        "district_ev_events_departure_within_tolerance_infeasible_count": (
            "official_within_tolerance_infeasible_count"
        ),
        "district_ev_performance_departure_min_acceptable_ratio": (
            "official_min_acceptable_rate"
        ),
        "district_ev_performance_departure_min_acceptable_feasible_ratio": (
            "official_min_acceptable_feasible_rate"
        ),
        "district_ev_performance_departure_within_tolerance_ratio": (
            "official_within_tolerance_rate"
        ),
        "district_ev_performance_departure_within_tolerance_feasible_ratio": (
            "official_within_tolerance_feasible_rate"
        ),
        "district_ev_performance_departure_soc_deficit_mean_ratio": (
            "official_soc_deficit_mean"
        ),
        "district_ev_performance_departure_shortfall_beyond_tolerance_mean_ratio": (
            "official_shortfall_beyond_tolerance_mean"
        ),
        "district_ev_performance_departure_soc_surplus_mean_ratio": (
            "official_soc_surplus_mean"
        ),
        "district_ev_performance_departure_soc_absolute_error_mean_ratio": (
            "official_soc_absolute_error_mean"
        ),
    }
    values: dict[str, float] = {}
    indexed = frame.set_index("KPI")
    for source, target in wanted.items():
        if source not in indexed.index:
            continue
        value = _finite_float(indexed.at[source, "District"])
        if value is not None:
            values[target] = value
    return values


def main() -> int:
    args = _parse_args()
    job_dir = Path(args.job_dir)
    output_dir = Path(args.output_dir)
    simulation_dir = _simulation_data_dir(job_dir)
    files = _charger_files(simulation_dir, args.episode)
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(_event_rows(path, tolerance=float(args.tolerance)))
    rows.sort(key=lambda row: (int(row["episode"]), str(row["building_id"]), str(row["charger_id"])))

    events_csv = output_dir / "ev_departure_events.csv"
    summary_json = output_dir / "summary.json"
    _write_csv(events_csv, rows)
    payload = {
        **_summary(rows),
        "official_exported_kpis": _exported_kpi_summary(simulation_dir),
        "job_dir": str(job_dir),
        "simulation_data_dir": str(simulation_dir),
        "events_csv": str(events_csv),
        "episode": args.episode,
        "tolerance": float(args.tolerance),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
