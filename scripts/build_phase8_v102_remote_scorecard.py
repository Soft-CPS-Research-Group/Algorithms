#!/usr/bin/env python3
"""Build a compact Phase 8 v1.0.2 remote-wave scorecard."""

from __future__ import annotations

import argparse
import csv
import math
import re
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "wave",
    "recipe",
    "policy",
    "seed",
    "attempt",
    "decision",
    "status",
    "slurm_state",
    "job_id",
    "job_name",
    "config_path",
    "target_host",
    "partition",
    "gpus",
    "image_tag",
    "runtime_hours",
    "queue_hours",
    "total_hours",
    "community_cost_eur",
    "ev_min_acceptable_feasible_rate",
    "ev_within_tolerance_feasible_rate",
    "electrical_violation_kwh",
    "community_import_kwh",
    "community_export_kwh",
    "battery_throughput_kwh",
    "v2g_export_kwh",
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
    return parsed if math.isfinite(parsed) else None


def _fmt(value: Any, digits: int = 4) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "" if value is None else str(value)
    if abs(parsed) >= 100:
        return f"{parsed:.2f}"
    return f"{parsed:.{digits}f}"


def _joined_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in (
            "job_name",
            "config_path",
            "simulation_data_session",
            "kpi_file",
            "job_id",
        )
    ).lower()


def infer_wave(row: dict[str, Any]) -> str:
    text = _joined_text(row)
    if "wave1" in text:
        return "wave1_profile"
    if "retry8h" in text or "wave2-retry" in text:
        return "wave2_retry8h"
    if "wave2" in text:
        return "wave2_baselines"
    if "wave3" in text:
        return "wave3_maddpg_direct"
    if "wave4" in text:
        return "wave4_maddpg_v48_teacher"
    if "smoke" in text:
        return "gate0_smoke"
    return "unknown"


def infer_policy(row: dict[str, Any]) -> str:
    text = _joined_text(row)
    if "normal-no-battery" in text or "normal_no_battery" in text:
        return "NormalNoBatteryPolicy"
    if "rbc-community" in text or "rbc_community" in text:
        return "RBCCommunityPolicy"
    if "rbc-smart" in text or "rbc_smart" in text:
        return "RBCSmartPolicy"
    if "rbc-basic" in text or "rbc_basic" in text:
        return "RBCBasicPolicy"
    if "random" in text:
        return "RandomPolicy"
    if "normal" in text:
        return "NormalPolicy"
    if "maddpg-v48" in text or "maddpg_v48" in text or "v48-teacher" in text:
        return "MADDPG_v48_teacher"
    if "maddpg-v3-direct" in text or "maddpg_v3_direct" in text:
        return "MADDPG_v3_direct"
    if "maddpg" in text:
        return "MADDPG"
    if "matd3" in text:
        return "MATD3"
    return "unknown"


def infer_recipe(row: dict[str, Any]) -> str:
    policy = infer_policy(row)
    text = _joined_text(row)
    if policy == "MADDPG_v3_direct":
        match = re.search(r"u(\d+)[_-]?b(\d+)", text)
        if match:
            return f"maddpg_v3_direct_u{match.group(1)}_b{match.group(2)}"
        return "maddpg_v3_direct"
    if policy == "MADDPG_v48_teacher":
        return "maddpg_v48_teacher"
    return policy.replace("Policy", "").lower()


def infer_seed(row: dict[str, Any]) -> str:
    text = _joined_text(row)
    match = re.search(r"seed[_-]?(\d+)", text)
    return match.group(1) if match else ""


def infer_attempt(row: dict[str, Any]) -> str:
    text = _joined_text(row)
    if "retry8h" in text or "wave2-retry" in text:
        return "retry8h"
    return "initial"


def _hours(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return ""
    return f"{parsed / 3600.0:.3f}"


def decide(row: dict[str, Any], policy: str) -> str:
    status = str(row.get("status") or "").lower()
    slurm_state = str(row.get("slurm_state") or "").upper()
    errors = str(row.get("errors") or "").lower()
    if status == "finished":
        ev_min = _safe_float(row.get("ev_min_acceptable_feasible_rate"))
        violations = _safe_float(row.get("electrical_violation_kwh"))
        if policy == "RandomPolicy":
            return "finished_sanity_only_ev_fail" if ev_min is not None and ev_min < 0.95 else "finished_sanity_only"
        if ev_min is not None and ev_min < 0.95:
            return "finished_ev_service_fail"
        if violations is not None and violations > 1e-6:
            return "finished_grid_violation"
        return "finished_valid_scorecard_row"
    if status == "failed":
        if slurm_state == "TIMEOUT" or "timeout" in errors:
            return "failed_timeout_retry_needed"
        return "failed_investigate"
    if status == "running":
        return "wait_running"
    if status in {"queued", "dispatched"}:
        return "wait_queue"
    return "wait_unknown"


def read_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_submitted_jobs(paths: list[Path]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            job_id = str(entry.get("job_id") or "").strip()
            if not job_id:
                response = entry.get("response")
                if isinstance(response, dict):
                    job_id = str(response.get("job_id") or "").strip()
            if job_id:
                metadata[job_id] = entry
    return metadata


def _metadata_value(entry: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = entry.get(key)
        if value not in (None, ""):
            return str(value)
    request = entry.get("request")
    if isinstance(request, dict):
        for key in keys:
            value = request.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def _merge_submitted_metadata(row: dict[str, Any], entry: dict[str, Any] | None) -> dict[str, Any]:
    if entry is None:
        return row
    merged = dict(row)
    if not str(merged.get("job_name") or "").strip():
        merged["job_name"] = _metadata_value(entry, "job_name")
    if not str(merged.get("config_path") or "").strip():
        config_path = _metadata_value(entry, "config_path")
        if not config_path:
            file_name = _metadata_value(entry, "file_name", "config")
            config_path = f"configs/{file_name}" if file_name and not file_name.startswith("configs/") else file_name
        merged["config_path"] = config_path
    if not str(merged.get("target_host") or "").strip():
        merged["target_host"] = _metadata_value(entry, "target_host")
    if not str(merged.get("image_tag") or "").strip():
        merged["image_tag"] = _metadata_value(entry, "image_tag")
    for key in ("wave", "recipe", "policy_key", "algorithm", "seed"):
        value = entry.get(key)
        if value not in (None, "") and f"_submitted_{key}" not in merged:
            merged[f"_submitted_{key}"] = str(value)
    return merged


def load_rows(
    summary_paths: list[Path],
    submitted_jobs_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    submitted_metadata = _load_submitted_jobs(submitted_jobs_paths or [])
    by_job: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for summary_path in summary_paths:
        for row in read_summary(summary_path):
            job_id = str(row.get("job_id") or "").strip()
            if not job_id:
                continue
            enriched = _merge_submitted_metadata(dict(row), submitted_metadata.get(job_id))
            enriched["_source_summary"] = str(summary_path)
            # Later summaries win. This lets callers pass older snapshots first and
            # newer snapshots last without doing timestamp parsing here.
            by_job[job_id] = enriched
    return list(by_job.values())


def build_scorecard(
    summary_paths: list[Path],
    submitted_jobs_paths: list[Path] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in load_rows(summary_paths, submitted_jobs_paths):
        wave = infer_wave(raw)
        if wave == "unknown" and raw.get("_submitted_wave"):
            wave = str(raw["_submitted_wave"])
        policy = infer_policy(raw)
        if policy == "unknown" and raw.get("_submitted_algorithm"):
            policy = str(raw["_submitted_algorithm"])
        recipe = infer_recipe(raw)
        if recipe == "unknown" and raw.get("_submitted_recipe"):
            recipe = str(raw["_submitted_recipe"])
        output: dict[str, str] = OrderedDict()
        output["wave"] = wave
        output["recipe"] = recipe
        output["policy"] = policy
        output["seed"] = infer_seed(raw) or str(raw.get("_submitted_seed") or "")
        output["attempt"] = infer_attempt(raw)
        output["decision"] = decide(raw, policy)
        output["status"] = str(raw.get("status") or "")
        output["slurm_state"] = str(raw.get("slurm_state") or "")
        output["job_id"] = str(raw.get("job_id") or "")
        output["job_name"] = str(raw.get("job_name") or "")
        output["config_path"] = str(raw.get("config_path") or "")
        output["target_host"] = str(raw.get("target_host") or "")
        output["partition"] = str(raw.get("deucalion_partition") or "")
        output["gpus"] = str(raw.get("deucalion_gpus") or "")
        output["image_tag"] = str(raw.get("image_tag") or "")
        output["runtime_hours"] = _hours(raw.get("run_duration_seconds"))
        output["queue_hours"] = _hours(raw.get("queue_wait_seconds"))
        output["total_hours"] = _hours(raw.get("total_duration_seconds"))
        for metric in (
            "community_cost_eur",
            "ev_min_acceptable_feasible_rate",
            "ev_within_tolerance_feasible_rate",
            "electrical_violation_kwh",
            "community_import_kwh",
            "community_export_kwh",
            "battery_throughput_kwh",
            "v2g_export_kwh",
        ):
            output[metric] = _fmt(raw.get(metric))
        output["errors"] = str(raw.get("errors") or "")
        rows.append(output)
    return sorted(rows, key=lambda row: (row["wave"], row["recipe"], row["seed"], row["attempt"], row["job_id"]))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(row["status"] or "unknown" for row in rows)
    wave_counts = Counter(row["wave"] for row in rows)
    finished = [row for row in rows if row["status"] == "finished"]
    failed = [row for row in rows if row["status"] == "failed"]
    active = [row for row in rows if row["status"] in {"running", "queued", "dispatched"}]

    lines: list[str] = [
        "# Phase 8 v1.0.2 Remote Wave Scorecard",
        "",
        "This scorecard aggregates the current Wave 1/2/3/4 remote snapshots and retry submissions.",
        "",
        "## Counts",
        "",
        "By status:",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "By wave:"])
    for wave, count in sorted(wave_counts.items()):
        lines.append(f"- {wave}: {count}")

    lines.extend(
        [
            "",
            "## Finished Rows",
            "",
            *_markdown_table(
                finished,
                [
                    "wave",
                    "policy",
                    "recipe",
                    "seed",
                    "community_cost_eur",
                    "ev_min_acceptable_feasible_rate",
                    "ev_within_tolerance_feasible_rate",
                    "electrical_violation_kwh",
                    "runtime_hours",
                    "decision",
                ],
            ),
            "",
            "## Failed Rows",
            "",
            *_markdown_table(
                failed,
                [
                    "wave",
                    "policy",
                    "recipe",
                    "attempt",
                    "slurm_state",
                    "runtime_hours",
                    "decision",
                    "job_id",
                ],
            ),
            "",
            "## Active Rows",
            "",
            *_markdown_table(
                active,
                [
                    "wave",
                    "policy",
                    "recipe",
                    "seed",
                    "status",
                    "slurm_state",
                    "runtime_hours",
                    "queue_hours",
                    "job_id",
                ],
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        action="append",
        required=True,
        type=Path,
        help="Collected remote summary CSV. May be supplied multiple times; later files override earlier rows with the same job_id.",
    )
    parser.add_argument(
        "--submitted-jobs",
        action="append",
        default=[],
        type=Path,
        help="Optional submitted_jobs.json with job_name/config metadata for summaries that omit it.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_scorecard(args.summary_csv, args.submitted_jobs)
    write_csv(args.output_dir / "phase8_v102_scorecard.csv", rows)
    write_markdown(args.output_dir / "phase8_v102_scorecard.md", rows)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
