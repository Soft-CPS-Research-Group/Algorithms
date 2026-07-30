#!/usr/bin/env python3
"""Build a strict per-building scorecard from CityLearn simulator exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
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


EV_MIN_GATE = 0.99
EV_PRECISION_GATE = 0.40
EPS = 1.0e-9
GATE_PROFILE = "building_local_phase10_w6_v1"
PROJECTION_GATE_PROFILE = "building_local_phase10_w6_safety_projection_v1"

KPI_NAMES: Mapping[str, str] = {
    "local_cost_eur": "building_cost_total_control_eur",
    "settled_cost_eur": "building_cost_community_market_settled_total_eur",
    "cost_bau_eur": "building_cost_total_business_as_usual_eur",
    "ev_departure_count": "building_ev_events_departure_count",
    "ev_min_acceptable_feasible_rate": (
        "building_ev_performance_departure_min_acceptable_feasible_ratio"
    ),
    "ev_within_tolerance_feasible_rate": (
        "building_ev_performance_departure_within_tolerance_feasible_ratio"
    ),
    "electrical_violation_kwh": (
        "building_electrical_service_phase_violations_energy_total_kwh"
    ),
    "electrical_violation_events": (
        "building_electrical_service_phase_violations_event_count"
    ),
    "deferrable_completed_cycles_count": (
        "building_deferrable_appliance_service_completed_cycles_count"
    ),
    "deferrable_missed_cycles_count": (
        "building_deferrable_appliance_service_missed_cycles_count"
    ),
    "deferrable_unserved_energy_kwh": (
        "building_deferrable_appliance_service_unserved_energy_total_kwh"
    ),
    "outage_unserved_energy_normalized_rate": (
        "building_comfort_resilience_resilience_unserved_energy_outage_normalized_ratio"
    ),
}


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _building_sort_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"Building_(\d+)", str(value))
    return (int(match.group(1)), str(value)) if match else (10**9, str(value))


def _kpi_path(data_dir: Path) -> Path:
    direct = data_dir / "exported_kpis.csv"
    if direct.is_file():
        return direct
    candidates = sorted(data_dir.glob("exported_kpis_ep*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No exported KPI file under {data_dir}.")
    return candidates[-1]


def _read_kpis(data_dir: Path) -> tuple[dict[str, dict[str, float | None]], list[str]]:
    with _kpi_path(data_dir).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "KPI" not in reader.fieldnames:
            raise ValueError("Exported KPI CSV must contain a KPI column.")
        buildings = sorted(
            (name for name in reader.fieldnames if name.startswith("Building_")),
            key=_building_sort_key,
        )
        matrix: dict[str, dict[str, float | None]] = {}
        for raw in reader:
            kpi = str(raw.get("KPI") or "").strip()
            if kpi:
                matrix[kpi] = {building: _to_float(raw.get(building)) for building in buildings}
    if not buildings:
        raise ValueError("Exported KPI CSV contains no Building_<n> columns.")
    return matrix, buildings


def _storage_soc_bounds(data_dir: Path, building: str) -> tuple[float | None, float | None, int]:
    match = re.fullmatch(r"Building_(\d+)", building)
    if match is None:
        return None, None, 0
    paths = sorted(data_dir.glob(f"exported_data_building_{int(match.group(1))}_battery_ep*.csv"))
    if not paths:
        return None, None, 0
    values = pd.to_numeric(
        pd.read_csv(paths[-1]).get("Battery Soc-%", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if values.empty:
        return None, None, 1
    return float(values.min()), float(values.max()), 1


def _metric(
    matrix: Mapping[str, Mapping[str, float | None]],
    output_name: str,
    building: str,
) -> float | None:
    return matrix.get(KPI_NAMES[output_name], {}).get(building)


def audit_run(
    name: str,
    data_dir: Path,
    *,
    projection_tolerance_kwh: float = DEFAULT_PROJECTION_TOLERANCE_KWH,
    projection_max_event_rate: float = DEFAULT_PROJECTION_EVENT_RATE,
) -> list[dict[str, Any]]:
    matrix, buildings = _read_kpis(data_dir)
    rows: list[dict[str, Any]] = []
    for building in buildings:
        row: dict[str, Any] = {
            "run": name,
            "data_dir": str(data_dir),
            "building": building,
            "gate_profile": GATE_PROFILE,
            "projection_gate_profile": PROJECTION_GATE_PROFILE,
            "ev_min_gate": EV_MIN_GATE,
            "ev_precision_gate": EV_PRECISION_GATE,
        }
        for output_name in KPI_NAMES:
            row[output_name] = _metric(matrix, output_name, building)
        soc_min, soc_max, storage_present = _storage_soc_bounds(data_dir, building)
        row.update(
            {
                "storage_present": storage_present,
                "storage_soc_min": soc_min,
                "storage_soc_max": soc_max,
            }
        )

        departures = row["ev_departure_count"] or 0.0
        row["pass_ev_service"] = int(
            departures <= EPS
            or (
                row["ev_min_acceptable_feasible_rate"] is not None
                and row["ev_min_acceptable_feasible_rate"] + EPS >= EV_MIN_GATE
            )
        )
        row["pass_ev_precision"] = int(
            departures <= EPS
            or (
                row["ev_within_tolerance_feasible_rate"] is not None
                and row["ev_within_tolerance_feasible_rate"] + EPS >= EV_PRECISION_GATE
            )
        )
        row["pass_electrical"] = int(
            row["electrical_violation_kwh"] is not None
            and row["electrical_violation_events"] is not None
            and row["electrical_violation_kwh"] <= EPS
            and row["electrical_violation_events"] <= EPS
        )
        row["pass_deferrable_service"] = int(
            row["deferrable_missed_cycles_count"] is not None
            and row["deferrable_unserved_energy_kwh"] is not None
            and row["deferrable_missed_cycles_count"] <= EPS
            and row["deferrable_unserved_energy_kwh"] <= EPS
        )
        row["pass_storage_soc"] = int(
            not storage_present
            or (
                soc_min is not None
                and soc_max is not None
                and soc_min >= -1.0e-6
                and soc_max <= 1.0 + 1.0e-6
            )
        )
        outage = row["outage_unserved_energy_normalized_rate"]
        row["pass_outage"] = int(outage is not None and outage <= EPS)
        row.update(
            executed_safety_evidence(
                data_dir,
                requested_violation_kwh=row["electrical_violation_kwh"],
                requested_violation_events=row["electrical_violation_events"],
                building_names=[building],
                tolerance_kwh=projection_tolerance_kwh,
                max_event_rate=projection_max_event_rate,
            )
        )
        mandatory = (
            "pass_ev_service",
            "pass_ev_precision",
            "pass_electrical",
            "pass_deferrable_service",
            "pass_storage_soc",
            "pass_outage",
        )
        row["local_gate_decision"] = (
            "PASS_LOCAL_GATES" if all(row[key] for key in mandatory) else "REJECT_LOCAL_GATES"
        )
        non_electrical_mandatory = tuple(key for key in mandatory if key != "pass_electrical")
        if row["local_gate_decision"] == "PASS_LOCAL_GATES":
            row["projection_tolerant_local_gate_decision"] = "PASS_LOCAL_GATES"
        elif (
            all(row[key] for key in non_electrical_mandatory)
            and int(row.get("executed_electrical_safety_certified", 0) or 0) == 1
            and int(row.get("projection_request_within_tolerance", 0) or 0) == 1
        ):
            row["projection_tolerant_local_gate_decision"] = "PASS_WITH_SAFETY_PROJECTION"
        else:
            row["projection_tolerant_local_gate_decision"] = "REJECT_LOCAL_GATES"
        rows.append(row)
    return rows


def compare_to_baseline(rows: list[dict[str, Any]], baseline_name: str) -> None:
    baseline = {
        row["building"]: row
        for row in rows
        if row["run"] == baseline_name
    }
    if not baseline:
        raise ValueError(f"Baseline run {baseline_name!r} is not present.")
    for row in rows:
        reference = baseline.get(row["building"])
        cost = _to_float(row.get("local_cost_eur"))
        baseline_cost = _to_float(reference.get("local_cost_eur")) if reference else None
        row["baseline_run"] = baseline_name
        row["baseline_local_cost_eur"] = baseline_cost
        row["local_cost_delta_to_baseline_eur"] = (
            None if cost is None or baseline_cost is None else cost - baseline_cost
        )
        row["local_cost_ratio_to_baseline"] = (
            None
            if cost is None or baseline_cost is None or abs(baseline_cost) <= EPS
            else cost / baseline_cost
        )
        row["beats_baseline_local_cost"] = int(
            row["run"] != baseline_name
            and row["local_gate_decision"] == "PASS_LOCAL_GATES"
            and row["local_cost_delta_to_baseline_eur"] is not None
            and row["local_cost_delta_to_baseline_eur"] < -EPS
        )
        row["beats_baseline_projection_tolerant_local_cost"] = int(
            row["run"] != baseline_name
            and row["projection_tolerant_local_gate_decision"]
            in {"PASS_LOCAL_GATES", "PASS_WITH_SAFETY_PROJECTION"}
            and row["local_cost_delta_to_baseline_eur"] is not None
            and row["local_cost_delta_to_baseline_eur"] < -EPS
        )


def compare_to_oracle(rows: list[dict[str, Any]], oracle_name: str) -> None:
    """Attach per-building regret and baseline-to-oracle gap closure.

    The oracle may be a replay-feasible upper bound rather than a globally exact
    optimum.  The column names deliberately retain ``reference`` so reports do
    not overstate the strength of the supplied benchmark.
    """

    oracle = {row["building"]: row for row in rows if row["run"] == oracle_name}
    if not oracle:
        raise ValueError(f"Oracle reference run {oracle_name!r} is not present.")
    for row in rows:
        reference = oracle.get(row["building"])
        cost = _to_float(row.get("local_cost_eur"))
        oracle_cost = _to_float(reference.get("local_cost_eur")) if reference else None
        baseline_cost = _to_float(row.get("baseline_local_cost_eur"))
        regret = None if cost is None or oracle_cost is None else cost - oracle_cost
        opportunity = (
            None
            if baseline_cost is None or oracle_cost is None
            else baseline_cost - oracle_cost
        )
        savings = None if baseline_cost is None or cost is None else baseline_cost - cost
        row["oracle_reference_run"] = oracle_name
        row["oracle_reference_local_cost_eur"] = oracle_cost
        row["local_cost_regret_to_oracle_reference_eur"] = regret
        row["baseline_to_oracle_reference_opportunity_eur"] = opportunity
        row["baseline_to_candidate_savings_eur"] = savings
        row["oracle_reference_gap_closure_ratio"] = (
            None
            if opportunity is None or savings is None or opportunity <= EPS
            else savings / opportunity
        )


def _jain_index(values: Sequence[float]) -> float | None:
    if not values:
        return None
    denominator = len(values) * sum(value * value for value in values)
    if denominator <= EPS:
        return None
    return sum(values) ** 2 / denominator


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run in dict.fromkeys(str(row["run"]) for row in rows):
        selected = [row for row in rows if row["run"] == run]
        deltas = [_to_float(row.get("local_cost_delta_to_baseline_eur")) for row in selected]
        valid_deltas = [value for value in deltas if value is not None]
        regrets = [
            value
            for value in (
                _to_float(row.get("local_cost_regret_to_oracle_reference_eur"))
                for row in selected
            )
            if value is not None
        ]
        opportunities = [
            value
            for value in (
                _to_float(row.get("baseline_to_oracle_reference_opportunity_eur"))
                for row in selected
            )
            if value is not None
        ]
        savings = [
            value
            for value in (
                _to_float(row.get("baseline_to_candidate_savings_eur"))
                for row in selected
            )
            if value is not None
        ]
        cost_sum = sum(
            value
            for value in (_to_float(row.get("local_cost_eur")) for row in selected)
            if value is not None
        )
        baseline_cost_sum = sum(
            value
            for value in (
                _to_float(row.get("baseline_local_cost_eur")) for row in selected
            )
            if value is not None
        )
        opportunity_sum = sum(opportunities)
        savings_sum = sum(savings)
        summaries.append(
            {
                "run": run,
                "building_count": len(selected),
                "local_gate_pass_count": sum(
                    row["local_gate_decision"] == "PASS_LOCAL_GATES" for row in selected
                ),
                "local_gate_reject_count": sum(
                    row["local_gate_decision"] != "PASS_LOCAL_GATES" for row in selected
                ),
                "projection_tolerant_gate_pass_count": sum(
                    row.get("projection_tolerant_local_gate_decision")
                    in {"PASS_LOCAL_GATES", "PASS_WITH_SAFETY_PROJECTION"}
                    for row in selected
                ),
                "safety_projection_pass_count": sum(
                    row.get("projection_tolerant_local_gate_decision")
                    == "PASS_WITH_SAFETY_PROJECTION"
                    for row in selected
                ),
                "local_cost_eur_sum": cost_sum,
                "settled_cost_eur_sum": sum(
                    value for value in (_to_float(row.get("settled_cost_eur")) for row in selected)
                    if value is not None
                ),
                "local_cost_delta_to_baseline_eur_sum": sum(valid_deltas),
                "local_cost_delta_to_baseline_eur_median": (
                    statistics.median(valid_deltas) if valid_deltas else None
                ),
                "local_cost_delta_to_baseline_eur_worst": (
                    max(valid_deltas) if valid_deltas else None
                ),
                "local_cost_improvement_to_baseline_ratio": (
                    None
                    if baseline_cost_sum <= EPS
                    else (baseline_cost_sum - cost_sum) / baseline_cost_sum
                ),
                "buildings_beating_baseline_count": sum(
                    int(row.get("beats_baseline_local_cost") or 0) for row in selected
                ),
                "buildings_beating_baseline_projection_tolerant_count": sum(
                    int(row.get("beats_baseline_projection_tolerant_local_cost") or 0)
                    for row in selected
                ),
                "all_buildings_pass_local_gates": int(
                    all(row["local_gate_decision"] == "PASS_LOCAL_GATES" for row in selected)
                ),
                "all_buildings_pass_projection_tolerant_gates": int(
                    all(
                        row.get("projection_tolerant_local_gate_decision")
                        in {"PASS_LOCAL_GATES", "PASS_WITH_SAFETY_PROJECTION"}
                        for row in selected
                    )
                ),
                "all_buildings_no_worse_than_baseline": int(
                    bool(valid_deltas) and all(value <= EPS for value in valid_deltas)
                ),
                "oracle_reference_regret_eur_sum": sum(regrets) if regrets else None,
                "oracle_reference_gap_closure_ratio": (
                    None if opportunity_sum <= EPS else savings_sum / opportunity_sum
                ),
                "jain_nonnegative_building_savings_index": _jain_index(
                    [max(value, 0.0) for value in savings]
                ),
            }
        )
    return summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_run(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run must have the form name=simulation_data_dir")
    return name.strip(), Path(raw_path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--oracle",
        help="Optional replay-feasible or exact reference run used for regret/gap closure.",
    )
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument(
        "--projection-tolerance-kwh",
        type=float,
        default=DEFAULT_PROJECTION_TOLERANCE_KWH,
        help="Maximum aggregate pre-projection violation energy for the separate tolerant profile.",
    )
    parser.add_argument(
        "--projection-max-event-rate",
        type=float,
        default=DEFAULT_PROJECTION_EVENT_RATE,
        help="Maximum fraction of time steps with pre-projection clipping for the tolerant profile.",
    )
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for name, data_dir in args.run:
        rows.extend(
            audit_run(
                name,
                data_dir,
                projection_tolerance_kwh=args.projection_tolerance_kwh,
                projection_max_event_rate=args.projection_max_event_rate,
            )
        )
    compare_to_baseline(rows, args.baseline)
    if args.oracle:
        compare_to_oracle(rows, args.oracle)
    summary_rows = summarize(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "building_scorecard.csv", rows)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    (args.output_dir / "building_scorecard.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
