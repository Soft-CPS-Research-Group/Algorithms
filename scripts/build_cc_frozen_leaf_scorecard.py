#!/usr/bin/env python3
"""Build a CC scorecard against a matching frozen-PPO neutral control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.audit_building_local_behavior import (
        audit_run as audit_buildings,
        compare_to_baseline,
        summarize as summarize_buildings,
    )
    from scripts.audit_rbc_baseline_behavior import _audit_run as audit_aggregate
    from scripts.electrical_safety_evidence import (
        DEFAULT_PROJECTION_EVENT_RATE,
        DEFAULT_PROJECTION_TOLERANCE_KWH,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from audit_building_local_behavior import (  # type: ignore[no-redef]
        audit_run as audit_buildings,
        compare_to_baseline,
        summarize as summarize_buildings,
    )
    from audit_rbc_baseline_behavior import _audit_run as audit_aggregate  # type: ignore[no-redef]
    from electrical_safety_evidence import (  # type: ignore[no-redef]
        DEFAULT_PROJECTION_EVENT_RATE,
        DEFAULT_PROJECTION_TOLERANCE_KWH,
    )


SCORECARD_PROFILE = "cc_frozen_leaf_scorecard_v1"
PASS_GATE_DECISIONS = {"PASS_HARD_GATES", "PASS_WITH_SAFETY_PROJECTION"}

GATE_METRICS = (
    "ev_min_acceptable_feasible_rate",
    "ev_within_tolerance_feasible_rate",
    "electrical_violation_kwh",
    "electrical_violation_events",
    "deferrable_completed_cycles_count",
    "deferrable_missed_cycles_count",
    "deferrable_unserved_energy_kwh",
    "deferrable_service_level_rate",
    "storage_soc_min",
    "storage_soc_max",
    "storage_soc_violation_count",
    "outage_unserved_energy_normalized_rate",
    "executed_electrical_safety_certified",
    "projection_request_within_tolerance",
)

# Secondary metrics are explicit rather than collapsed into one opaque reward.
# Cost remains the primary objective; these metrics identify material trade-offs.
SECONDARY_METRICS: tuple[tuple[str, str], ...] = (
    ("community_import_kwh", "lower"),
    ("peak_daily_ratio_to_bau", "lower"),
    ("peak_all_time_ratio_to_bau", "lower"),
    ("ramping_ratio_to_bau", "lower"),
    ("load_factor_penalty_daily_ratio_to_bau", "lower"),
    ("emissions_kgco2", "lower"),
    ("community_solar_self_consumption_rate", "higher"),
)

MONITORING_METRICS = (
    "community_export_kwh",
    "community_net_exchange_kwh",
    "battery_throughput_kwh",
    "battery_throughput_ratio_to_bau",
    "v2g_export_kwh",
)


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else None


def _delta(candidate: Any, baseline: Any) -> tuple[float | None, float | None]:
    candidate_value = _number(candidate)
    baseline_value = _number(baseline)
    if candidate_value is None or baseline_value is None:
        return None, None
    absolute = candidate_value - baseline_value
    relative = None if abs(baseline_value) <= 1.0e-12 else absolute / abs(baseline_value)
    return absolute, relative


def build_scorecard(
    aggregate_rows: Sequence[Mapping[str, Any]],
    building_summaries: Mapping[str, Mapping[str, Any]],
    *,
    baseline_name: str,
    relative_regression_tolerance: float = 0.01,
    self_consumption_abs_tolerance: float = 0.005,
) -> list[dict[str, Any]]:
    indexed = {str(row["run"]): row for row in aggregate_rows}
    if baseline_name not in indexed:
        raise ValueError(f"Missing baseline run {baseline_name!r}")
    baseline = indexed[baseline_name]
    baseline_cost = _number(baseline.get("community_cost_eur"))
    if baseline_cost is None:
        raise ValueError("Matching baseline has no official community cost")

    output: list[dict[str, Any]] = []
    for aggregate in aggregate_rows:
        run = str(aggregate["run"])
        building = dict(building_summaries.get(run, {}))
        cost = _number(aggregate.get("community_cost_eur"))
        cost_delta, cost_delta_ratio = _delta(cost, baseline_cost)
        regressions: list[str] = []
        improvements: list[str] = []

        row: dict[str, Any] = {
            "run": run,
            "scorecard_profile": SCORECARD_PROFILE,
            "baseline_run": baseline_name,
            "time_steps": aggregate.get("time_steps"),
            "gate_profile": aggregate.get("gate_profile"),
            "projection_gate_profile": aggregate.get("projection_gate_profile"),
            "hard_gate_decision": aggregate.get("projection_tolerant_learning_gate_decision"),
            "community_cost_eur": cost,
            "cost_delta_to_baseline_eur": cost_delta,
            "cost_delta_to_baseline_ratio": cost_delta_ratio,
        }

        for metric in GATE_METRICS:
            row[metric] = aggregate.get(metric)

        for metric, direction in SECONDARY_METRICS:
            candidate_value = _number(aggregate.get(metric))
            baseline_value = _number(baseline.get(metric))
            absolute, relative = _delta(candidate_value, baseline_value)
            row[metric] = candidate_value
            row[f"{metric}_delta"] = absolute
            row[f"{metric}_delta_ratio"] = relative
            if absolute is None:
                continue

            if metric == "community_solar_self_consumption_rate":
                regression = absolute < -self_consumption_abs_tolerance
                improvement = absolute > 0.0
            elif relative is not None:
                signed = relative if direction == "lower" else -relative
                regression = signed > relative_regression_tolerance
                improvement = signed < 0.0
            else:
                regression = False
                improvement = (absolute < 0.0) if direction == "lower" else (absolute > 0.0)
            if regression:
                regressions.append(metric)
            elif improvement:
                improvements.append(metric)

        for metric in MONITORING_METRICS:
            candidate_value = _number(aggregate.get(metric))
            absolute, relative = _delta(candidate_value, baseline.get(metric))
            row[metric] = candidate_value
            row[f"{metric}_delta"] = absolute
            row[f"{metric}_delta_ratio"] = relative

        for key in (
            "building_count",
            "projection_tolerant_gate_pass_count",
            "buildings_beating_baseline_count",
            "buildings_beating_baseline_projection_tolerant_count",
            "local_cost_delta_to_baseline_eur_worst",
            "all_buildings_pass_projection_tolerant_gates",
            "all_buildings_no_worse_than_baseline",
            "jain_nonnegative_building_savings_index",
        ):
            row[key] = building.get(key)

        row["secondary_improvements"] = "|".join(improvements)
        row["secondary_regressions"] = "|".join(regressions)
        row["secondary_improvement_count"] = len(improvements)
        row["secondary_regression_count"] = len(regressions)

        if run == baseline_name:
            decision = "REFERENCE"
        elif row["hard_gate_decision"] not in PASS_GATE_DECISIONS:
            decision = "REJECT_HARD_GATES"
        elif cost_delta is None or cost_delta >= 0.0:
            decision = "REJECT_COST"
        elif regressions:
            decision = "PASS_COST_WITH_TRADEOFFS"
        else:
            decision = "PASS_CC_SCORECARD"
        row["decision"] = decision
        output.append(row)

    return output


def _parse_run(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("run must have the form name=simulation_data_dir")
    return name.strip(), Path(raw_path).resolve()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Frozen-PPO CC scorecard",
        "",
        f"Profile: `{SCORECARD_PROFILE}`.",
        "",
        "| Run | Decision | Hard gates | Cost EUR | Delta EUR | Buildings better | Secondary regressions |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        cost = _number(row.get("community_cost_eur"))
        delta = _number(row.get("cost_delta_to_baseline_eur"))
        lines.append(
            "| {run} | {decision} | {gates} | {cost} | {delta} | {better} | {regressions} |".format(
                run=row.get("run", ""),
                decision=row.get("decision", ""),
                gates=row.get("hard_gate_decision", ""),
                cost="" if cost is None else f"{cost:.4f}",
                delta="" if delta is None else f"{delta:+.4f}",
                better=row.get("buildings_beating_baseline_projection_tolerant_count", ""),
                regressions=row.get("secondary_regressions", ""),
            )
        )
    lines.extend(
        [
            "",
            "Cost is primary but never overrides hard gates. Peak, ramping, load factor, imports, emissions and solar self-consumption are explicit secondary checks. Battery throughput, exports and V2G are reported as monitoring trade-offs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=_parse_run, required=True)
    parser.add_argument("--run", action="append", type=_parse_run, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relative-regression-tolerance", type=float, default=0.01)
    parser.add_argument("--self-consumption-abs-tolerance", type=float, default=0.005)
    parser.add_argument("--projection-tolerance-kwh", type=float, default=DEFAULT_PROJECTION_TOLERANCE_KWH)
    parser.add_argument("--projection-max-event-rate", type=float, default=DEFAULT_PROJECTION_EVENT_RATE)
    args = parser.parse_args(argv)

    run_specs = [args.baseline, *args.run]
    aggregate_rows = [
        audit_aggregate(
            name,
            data_dir,
            projection_tolerance_kwh=args.projection_tolerance_kwh,
            projection_max_event_rate=args.projection_max_event_rate,
        )
        for name, data_dir in run_specs
    ]
    building_rows: list[dict[str, Any]] = []
    for name, data_dir in run_specs:
        building_rows.extend(
            audit_buildings(
                name,
                data_dir,
                projection_tolerance_kwh=args.projection_tolerance_kwh,
                projection_max_event_rate=args.projection_max_event_rate,
            )
        )
    compare_to_baseline(building_rows, args.baseline[0])
    building_summary_rows = summarize_buildings(building_rows)
    building_summaries = {str(row["run"]): row for row in building_summary_rows}
    scorecard = build_scorecard(
        aggregate_rows,
        building_summaries,
        baseline_name=args.baseline[0],
        relative_regression_tolerance=args.relative_regression_tolerance,
        self_consumption_abs_tolerance=args.self_consumption_abs_tolerance,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "scorecard.csv", scorecard)
    _write_csv(args.output_dir / "building_scorecard.csv", building_rows)
    (args.output_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "scorecard.md", scorecard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
