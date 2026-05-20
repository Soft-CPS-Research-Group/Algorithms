#!/usr/bin/env python3
"""Build a Phase 6J/6K scorecard from collected remote job summaries."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


KEY_METRICS = [
    "community_cost_eur",
    "cost_bau_eur",
    "cost_delta_to_bau_eur",
    "cost_ratio_to_bau",
    "ev_min_acceptable_feasible_rate",
    "ev_within_tolerance_feasible_rate",
    "ev_departure_count",
    "ev_departure_infeasible_count",
    "ev_departure_min_acceptable_rate",
    "electrical_violation_kwh",
    "electrical_violation_events",
    "battery_throughput_kwh",
    "battery_throughput_ratio_to_bau",
    "v2g_export_kwh",
    "net_exchange_kwh",
    "peak_daily_ratio_to_bau",
    "peak_all_time_ratio_to_bau",
]

OUTPUT_COLUMNS = [
    "decision",
    "dataset",
    "variant",
    "track",
    "policy",
    "seed",
    "target_host",
    "status",
    "job_id",
    "job_name",
    "config_path",
    "image_tag",
    "device_selected",
    "slurm_state",
    "slurm_queue_position",
    "community_cost_eur",
    "cost_delta_to_bau_eur",
    "cost_ratio_to_bau",
    "rbcsmart_cost_eur",
    "cost_delta_to_rbcsmart_eur",
    "cost_delta_to_rbcsmart_pct",
    "ev_min_acceptable_feasible_rate",
    "ev_within_tolerance_feasible_rate",
    "rbcsmart_ev_within_tolerance_feasible_rate",
    "ev_within_tolerance_delta_to_rbcsmart",
    "ev_departure_count",
    "ev_departure_infeasible_count",
    "electrical_violation_kwh",
    "electrical_violation_events",
    "battery_throughput_kwh",
    "battery_throughput_ratio_to_bau",
    "v2g_export_kwh",
    "net_exchange_kwh",
    "peak_daily_ratio_to_bau",
    "peak_all_time_ratio_to_bau",
    "errors",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _fmt(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "" if value is None else str(value)
    if abs(parsed) >= 100:
        return f"{parsed:.3f}"
    return f"{parsed:.4f}"


def _config_basename(row: dict[str, Any]) -> str:
    config_path = str(row.get("config_path") or "").strip()
    if config_path:
        return Path(config_path).name
    job_name = str(row.get("job_name") or "").strip()
    return job_name


def infer_dataset(config_or_name: str) -> str:
    lowered = config_or_name.lower()
    if "2022" in lowered:
        return "2022"
    if "15s" in lowered:
        return "15s"
    return "unknown"


def infer_variant(config_or_name: str) -> str:
    lowered = config_or_name.lower()
    if "no_v2g" in lowered or "no-v2g" in lowered:
        return "no_v2g"
    if "multi_charger" in lowered or "multi-charger" in lowered:
        return "multi_charger"
    return "original"


def infer_policy(config_or_name: str) -> str:
    lowered = config_or_name.lower()
    if "maddpg" in lowered:
        if "v48" in lowered:
            return "MADDPG_v48"
        return "MADDPG"
    if "rbc_smart" in lowered or "rbc-smart" in lowered:
        return "RBCSmart"
    if "rbc_basic" in lowered or "rbc-basic" in lowered:
        return "RBCBasic"
    if "normal_no_battery" in lowered or "normal-no-battery" in lowered:
        return "NormalNoBattery"
    if "random" in lowered:
        return "Random"
    if "normal" in lowered:
        return "Normal"
    return "unknown"


def infer_track(config_or_name: str) -> str:
    lowered = config_or_name.lower()
    if "smoke" in lowered:
        return "smoke"
    if "baseline" in lowered:
        return "baseline"
    if "full" in lowered:
        return "full"
    return "unknown"


def infer_seed(config_or_name: str) -> str:
    match = re.search(r"seed[_-]?(\d+)", config_or_name.lower())
    return match.group(1) if match else ""


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    config_or_name = " ".join(
        value
        for value in (
            _config_basename(row),
            str(row.get("job_name") or ""),
        )
        if value
    )
    enriched = dict(row)
    enriched["dataset"] = infer_dataset(config_or_name)
    enriched["variant"] = infer_variant(config_or_name)
    enriched["policy"] = infer_policy(config_or_name)
    enriched["track"] = infer_track(config_or_name)
    enriched["seed"] = infer_seed(config_or_name)
    return enriched


def _baseline_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("dataset") or ""), str(row.get("variant") or ""))


def _build_rbcsmart_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("policy") != "RBCSmart":
            continue
        if row.get("track") not in {"baseline", "full"}:
            continue
        if str(row.get("status") or "").lower() != "finished":
            continue
        if _safe_float(row.get("community_cost_eur")) is None:
            continue
        key = _baseline_key(row)
        current = index.get(key)
        if current is None or row.get("track") == "baseline":
            index[key] = row
    return index


def _decision_for_row(
    row: dict[str, Any],
    rbc_row: dict[str, Any] | None,
    *,
    ev_feasible_min: float,
    ev_within_min: float,
    cost_near_pct: float,
    max_grid_violation_kwh: float,
) -> str:
    status = str(row.get("status") or "").lower()
    policy = str(row.get("policy") or "")
    track = str(row.get("track") or "")

    if status not in {"finished", "completed"}:
        return "pending" if status in {"queued", "dispatched", "running"} else f"not_finished:{status or 'unknown'}"
    if track == "smoke":
        return "smoke_ok"
    if policy != "MADDPG_v48":
        return "reference"
    if rbc_row is None:
        return "awaiting_rbcsmart_baseline"

    cost = _safe_float(row.get("community_cost_eur"))
    rbc_cost = _safe_float(rbc_row.get("community_cost_eur"))
    if cost is None or rbc_cost is None:
        return "awaiting_kpis"

    ev_feasible = _safe_float(row.get("ev_min_acceptable_feasible_rate"))
    ev_within = _safe_float(row.get("ev_within_tolerance_feasible_rate"))
    violation_kwh = _safe_float(row.get("electrical_violation_kwh")) or 0.0

    if ev_feasible is None or ev_feasible < ev_feasible_min:
        return "reject_ev_service"
    if violation_kwh > max_grid_violation_kwh:
        return "reject_grid_violation"

    denominator = abs(rbc_cost) if abs(rbc_cost) > 1e-9 else 1.0
    cost_delta_pct = ((cost - rbc_cost) / denominator) * 100.0
    precision_ok = ev_within is not None and ev_within >= ev_within_min
    if cost <= rbc_cost and precision_ok:
        return "candidate_strong"
    if cost <= rbc_cost:
        return "candidate_cost_ok_precision_watch"
    if cost_delta_pct <= cost_near_pct and precision_ok:
        return "candidate_near_cost"
    return "reject_cost"


def build_scorecard(
    rows: list[dict[str, Any]],
    *,
    ev_feasible_min: float = 0.999,
    ev_within_min: float = 0.80,
    cost_near_pct: float = 5.0,
    max_grid_violation_kwh: float = 1e-6,
) -> list[dict[str, Any]]:
    enriched_rows = [enrich_row(row) for row in rows]
    rbc_index = _build_rbcsmart_index(enriched_rows)
    output: list[dict[str, Any]] = []

    for row in enriched_rows:
        scored: dict[str, Any] = OrderedDict()
        rbc_row = rbc_index.get(_baseline_key(row))
        rbc_cost = _safe_float(rbc_row.get("community_cost_eur")) if rbc_row else None
        cost = _safe_float(row.get("community_cost_eur"))
        rbc_ev_within = (
            _safe_float(rbc_row.get("ev_within_tolerance_feasible_rate")) if rbc_row else None
        )
        ev_within = _safe_float(row.get("ev_within_tolerance_feasible_rate"))

        for key in ("dataset", "variant", "track", "policy", "seed"):
            scored[key] = row.get(key, "")
        for key in (
            "target_host",
            "status",
            "job_id",
            "job_name",
            "config_path",
            "image_tag",
            "device_selected",
            "slurm_state",
            "slurm_queue_position",
        ):
            scored[key] = row.get(key, "")

        for key in KEY_METRICS:
            scored[key] = row.get(key, "")

        scored["rbcsmart_cost_eur"] = rbc_cost
        scored["cost_delta_to_rbcsmart_eur"] = (
            None if cost is None or rbc_cost is None else cost - rbc_cost
        )
        scored["cost_delta_to_rbcsmart_pct"] = (
            None
            if cost is None or rbc_cost is None or abs(rbc_cost) <= 1e-9
            else ((cost - rbc_cost) / abs(rbc_cost)) * 100.0
        )
        scored["rbcsmart_ev_within_tolerance_feasible_rate"] = rbc_ev_within
        scored["ev_within_tolerance_delta_to_rbcsmart"] = (
            None if ev_within is None or rbc_ev_within is None else ev_within - rbc_ev_within
        )
        scored["errors"] = row.get("errors", "")
        scored["decision"] = _decision_for_row(
            row,
            rbc_row,
            ev_feasible_min=ev_feasible_min,
            ev_within_min=ev_within_min,
            cost_near_pct=cost_near_pct,
            max_grid_violation_kwh=max_grid_violation_kwh,
        )
        output.append(scored)

    return output


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_scorecard_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(OUTPUT_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("status") or "").lower() == "finished"
        and row.get("policy") == "MADDPG_v48"
        and row.get("track") == "full"
        and _safe_float(row.get("community_cost_eur")) is not None
    ]
    return sorted(candidates, key=lambda row: _safe_float(row.get("community_cost_eur")) or float("inf"))


def write_scorecard_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: OrderedDict[str, int] = OrderedDict()
    for row in rows:
        decision = str(row.get("decision") or "unknown")
        counts[decision] = counts.get(decision, 0) + 1

    lines = [
        "# Phase 6J Remote Scorecard",
        "",
        f"Generated at Unix time `{time.time():.0f}`.",
        "",
        "## Decision Counts",
        "",
        "| Decision | Count |",
        "|---|---:|",
    ]
    for decision, count in counts.items():
        lines.append(f"| `{decision}` | {count} |")

    lines.extend(
        [
            "",
            "## Best Finished MADDPG Runs",
            "",
            "| Rank | Dataset | Variant | Decision | Cost | Delta vs RBCSmart | EV feasible | EV within tol | Battery throughput | V2G export | Job ID |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    best = _best_rows(rows)
    if not best:
        lines.append("| - | - | - | pending | - | - | - | - | - | - | - |")
    else:
        for rank, row in enumerate(best[:12], start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        str(row.get("dataset") or ""),
                        str(row.get("variant") or ""),
                        f"`{row.get('decision') or ''}`",
                        _fmt(row.get("community_cost_eur")),
                        _fmt(row.get("cost_delta_to_rbcsmart_eur")),
                        _fmt(row.get("ev_min_acceptable_feasible_rate")),
                        _fmt(row.get("ev_within_tolerance_feasible_rate")),
                        _fmt(row.get("battery_throughput_kwh")),
                        _fmt(row.get("v2g_export_kwh")),
                        f"`{row.get('job_id') or ''}`",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## How To Read",
            "",
            "- `candidate_strong`: MADDPG finished, kept EV feasible gate, did not violate grid, beat RBCSmart cost and kept EV precision above the configured threshold.",
            "- `candidate_cost_ok_precision_watch`: cost beat RBCSmart but EV precision still needs attention.",
            "- `candidate_near_cost`: close enough to RBCSmart to justify tuning, but not yet a winner.",
            "- `reject_ev_service`, `reject_grid_violation`, `reject_cost`: do not promote this recipe without a targeted fix.",
            "- `awaiting_rbcsmart_baseline`/`pending`: rerun the scorecard after the missing jobs finish.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ev-feasible-min", type=float, default=0.999)
    parser.add_argument("--ev-within-min", type=float, default=0.80)
    parser.add_argument("--cost-near-pct", type=float, default=5.0)
    parser.add_argument("--max-grid-violation-kwh", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    rows = read_summary_csv(args.summary_csv)
    scorecard = build_scorecard(
        rows,
        ev_feasible_min=args.ev_feasible_min,
        ev_within_min=args.ev_within_min,
        cost_near_pct=args.cost_near_pct,
        max_grid_violation_kwh=args.max_grid_violation_kwh,
    )
    write_scorecard_csv(args.output_dir / "scorecard.csv", scorecard)
    write_scorecard_markdown(args.output_dir / "scorecard.md", scorecard)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
