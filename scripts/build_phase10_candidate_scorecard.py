"""Build a Phase 10 candidate scorecard against the RBCSmart target."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RBCSMART_TARGET = {
    "cost_eur": 17884.270294900056,
    "ev_min_acceptable_feasible_rate": 1.0,
    "ev_within_tolerance_rate": 0.437,
    "electrical_violation_kwh": 0.0,
    "community_import_kwh": 159157.97769125472,
    "community_export_kwh": 52349.811849715574,
    "community_solar_self_consumption_rate": 0.4830259027196283,
    "battery_throughput_kwh": 24510.0,
    "v2g_export_kwh": 1.0,
}

SUCCESS_COST_EUR = 17884.3
SUCCESS_EV_MIN = 0.99
SUCCESS_EV_TOLERANCE = 0.40
PREFERRED_BATTERY_THROUGHPUT_KWH = 49000.0
COMMUNITY_TOLERANCE = 0.03


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _to_int(row: Mapping[str, Any], *keys: str) -> int | None:
    value = _to_float(row, *keys)
    return int(value) if value is not None else None


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _identity(row: Mapping[str, Any]) -> str:
    fields = [
        str(row.get("job_name") or ""),
        str(row.get("simulation_data_session") or ""),
        str(row.get("config_path") or ""),
        str(row.get("job_id") or ""),
    ]
    return " ".join(fields).strip()


def _parse_algorithm(text: str) -> str:
    lowered = text.lower()
    for algorithm in ("maddpg", "matd3", "masac", "mappo", "ippo", "rbcsmart", "rbccommunity", "rbcbasic"):
        if algorithm in lowered:
            return algorithm.upper().replace("RBC", "RBC")
    return ""


def _parse_seed(text: str) -> int | None:
    match = re.search(r"seed[-_]?(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_steps(text: str) -> int | None:
    match = re.search(r"(\d+)\s*steps", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_recipe(text: str) -> str:
    lowered = text.lower().replace("_", "-")
    for recipe in (
        "w6-ev-only-bc-primary",
        "w6-balanced-bc-storage-light",
        "w6-fast-decay-less-teacher",
        "w6-clone-diagnostic",
    ):
        if recipe in lowered:
            return recipe.replace("-", "_")
    if "bc-light" in lowered:
        return "w5_bc_light"
    if "newenc-bc" in lowered or "bc-2022" in lowered:
        return "w5_bc"
    if "newenc" in lowered:
        return "w5_plain"
    return ""


def _community_delta_bad(row: Mapping[str, Any], cost: float | None) -> bool:
    if cost is not None and cost <= RBCSMART_TARGET["cost_eur"] * (1.0 - COMMUNITY_TOLERANCE):
        return False

    import_kwh = _to_float(row, "community_import_kwh")
    export_kwh = _to_float(row, "community_export_kwh")
    self_consumption = _to_float(row, "community_solar_self_consumption_rate", "self_consumption_rate")

    if import_kwh is not None and import_kwh > RBCSMART_TARGET["community_import_kwh"] * (1.0 + COMMUNITY_TOLERANCE):
        return True
    if export_kwh is not None and export_kwh > RBCSMART_TARGET["community_export_kwh"] * (1.0 + COMMUNITY_TOLERANCE):
        return True
    if (
        self_consumption is not None
        and self_consumption < RBCSMART_TARGET["community_solar_self_consumption_rate"] * (1.0 - COMMUNITY_TOLERANCE)
    ):
        return True
    return False


def _verdict(row: Mapping[str, Any], cost: float | None) -> tuple[str, str]:
    status = str(row.get("status") or "").strip().lower()
    exit_code = str(row.get("exit_code") or "").strip()
    if status and status not in {"finished", "success", "completed"}:
        return "FAIL_RUNTIME", f"status={status}"
    if exit_code not in {"", "0", "0.0"}:
        return "FAIL_RUNTIME", f"exit_code={exit_code}"

    ev_min = _to_float(row, "ev_min_acceptable_feasible_rate")
    ev_tol = _to_float(row, "ev_within_tolerance_rate", "ev_within_tolerance_feasible_rate")
    violation = _to_float(row, "electrical_violation_kwh")

    if ev_min is None or ev_min < SUCCESS_EV_MIN:
        return "FAIL_EV_MIN", f"ev_min={_fmt_float(ev_min)} < {SUCCESS_EV_MIN}"
    if violation is None or violation > 1.0e-9:
        return "FAIL_ELECTRICAL", f"electrical_violation_kwh={_fmt_float(violation)}"
    if ev_tol is None or ev_tol < SUCCESS_EV_TOLERANCE:
        return "FAIL_EV_TOL", f"ev_tol={_fmt_float(ev_tol)} < {SUCCESS_EV_TOLERANCE}"
    if cost is None or cost > SUCCESS_COST_EUR:
        return "FAIL_COST", f"cost={_fmt_float(cost, 1)} > {SUCCESS_COST_EUR}"
    if _community_delta_bad(row, cost):
        return "FAIL_COMMUNITY", "community metric worse than RBCSmart tolerance"
    return "PASS", "passes W6 RBCSmart gates"


def _normalise_row(row: Mapping[str, Any], source: Path) -> dict[str, Any]:
    label = _identity(row)
    cost = _to_float(row, "community_cost_eur", "total_cost", "cost")
    ev_min = _to_float(row, "ev_min_acceptable_feasible_rate")
    ev_tol = _to_float(row, "ev_within_tolerance_rate", "ev_within_tolerance_feasible_rate")
    violation = _to_float(row, "electrical_violation_kwh")
    battery = _to_float(row, "battery_throughput_kwh", "battery_total_throughput_kwh")
    v2g = _to_float(row, "v2g_export_kwh", "ev_v2g_discharge_kwh")
    community_import = _to_float(row, "community_import_kwh")
    community_export = _to_float(row, "community_export_kwh")
    self_consumption = _to_float(row, "community_solar_self_consumption_rate", "self_consumption_rate")
    runtime = _to_float(row, "run_duration_seconds", "runtime_seconds")
    verdict, reason = _verdict(row, cost)

    cost_delta = None if cost is None else cost - RBCSMART_TARGET["cost_eur"]
    battery_ratio = None if battery is None else battery / RBCSMART_TARGET["battery_throughput_kwh"]
    v2g_delta = None if v2g is None else v2g - RBCSMART_TARGET["v2g_export_kwh"]
    cost_ratio_bau = str(row.get("cost_ratio_to_bau") or "").strip()

    return {
        "source_file": str(source),
        "job_id": row.get("job_id", ""),
        "label": label,
        "algorithm": _parse_algorithm(label),
        "recipe": _parse_recipe(label),
        "seed": _parse_seed(label) or "",
        "steps": _parse_steps(label) or "",
        "status": row.get("status", ""),
        "exit_code": row.get("exit_code", ""),
        "verdict": verdict,
        "verdict_reason": reason,
        "cost_eur": cost,
        "cost_delta_vs_rbc_smart": cost_delta,
        "cost_ratio_vs_rbc_smart": None if cost is None else cost / RBCSMART_TARGET["cost_eur"],
        "cost_ratio_to_bau_status": cost_ratio_bau if cost_ratio_bau else "unavailable_bau_disabled",
        "ev_min_acceptable_feasible_rate": ev_min,
        "ev_within_tolerance_rate": ev_tol,
        "electrical_violation_kwh": violation,
        "battery_throughput_kwh": battery,
        "battery_ratio_vs_rbc_smart": battery_ratio,
        "battery_preferred_gate": battery is not None and battery <= PREFERRED_BATTERY_THROUGHPUT_KWH,
        "v2g_export_kwh": v2g,
        "v2g_delta_vs_rbc_smart": v2g_delta,
        "community_import_kwh": community_import,
        "community_import_delta_vs_rbc_smart": (
            None if community_import is None else community_import - RBCSMART_TARGET["community_import_kwh"]
        ),
        "community_export_kwh": community_export,
        "community_export_delta_vs_rbc_smart": (
            None if community_export is None else community_export - RBCSMART_TARGET["community_export_kwh"]
        ),
        "community_solar_self_consumption_rate": self_consumption,
        "self_consumption_delta_vs_rbc_smart": (
            None if self_consumption is None else self_consumption - RBCSMART_TARGET["community_solar_self_consumption_rate"]
        ),
        "peak_daily_ratio_to_bau": _to_float(row, "peak_daily_ratio_to_bau"),
        "peak_all_time_ratio_to_bau": _to_float(row, "peak_all_time_ratio_to_bau"),
        "runtime_seconds": runtime,
        "deucalion_time": row.get("deucalion_time", ""),
        "deucalion_mem_gb": row.get("deucalion_mem_gb", ""),
        "deucalion_gpus": row.get("deucalion_gpus", ""),
        "image_tag": row.get("image_tag", ""),
        "slurm_job_id": row.get("slurm_job_id", ""),
    }


def build_scorecard(summary_csvs: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summary_csvs:
        for row in _read_csv(path):
            rows.append(_normalise_row(row, path))

    rows.sort(
        key=lambda row: (
            str(row.get("verdict") or ""),
            int(row.get("steps") or 0),
            float(row.get("cost_eur") or 1.0e18),
            str(row.get("job_id") or ""),
        )
    )
    return rows


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = [
        "algorithm",
        "recipe",
        "seed",
        "steps",
        "verdict",
        "cost_eur",
        "ev_min_acceptable_feasible_rate",
        "ev_within_tolerance_rate",
        "battery_throughput_kwh",
        "v2g_export_kwh",
        "runtime_seconds",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                value = _fmt_float(value, 3)
            values.append(str(value if value is not None else ""))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _markdown_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict") or "")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    counts = "\n".join(f"- {verdict}: {count}" for verdict, count in sorted(verdict_counts.items()))
    return f"""# Phase 10 Candidate Scorecard

Target: RBCSmart (`cost <= {SUCCESS_COST_EUR}`, `ev_min >= {SUCCESS_EV_MIN}`,
`ev_within_tolerance >= {SUCCESS_EV_TOLERANCE}`, no electrical violations).

BAU ratios are marked unavailable when BAU export was intentionally disabled.

Verdict counts:

{counts}

{_markdown_table(rows)}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 10 scorecard rows against RBCSmart gates.")
    parser.add_argument("--summary-csv", action="append", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = build_scorecard(args.summary_csv)
    fieldnames = [
        "source_file",
        "job_id",
        "label",
        "algorithm",
        "recipe",
        "seed",
        "steps",
        "status",
        "exit_code",
        "verdict",
        "verdict_reason",
        "cost_eur",
        "cost_delta_vs_rbc_smart",
        "cost_ratio_vs_rbc_smart",
        "cost_ratio_to_bau_status",
        "ev_min_acceptable_feasible_rate",
        "ev_within_tolerance_rate",
        "electrical_violation_kwh",
        "battery_throughput_kwh",
        "battery_ratio_vs_rbc_smart",
        "battery_preferred_gate",
        "v2g_export_kwh",
        "v2g_delta_vs_rbc_smart",
        "community_import_kwh",
        "community_import_delta_vs_rbc_smart",
        "community_export_kwh",
        "community_export_delta_vs_rbc_smart",
        "community_solar_self_consumption_rate",
        "self_consumption_delta_vs_rbc_smart",
        "peak_daily_ratio_to_bau",
        "peak_all_time_ratio_to_bau",
        "runtime_seconds",
        "deucalion_time",
        "deucalion_mem_gb",
        "deucalion_gpus",
        "image_tag",
        "slurm_job_id",
    ]
    if args.output_csv:
        _write_csv(args.output_csv, rows, fieldnames)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_markdown_summary(rows), encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"Built scorecard with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
