"""Summarize exported EV charger actions for Phase 10 local experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _load_config(job_dir: Path) -> dict[str, Any]:
    config_path = job_dir / "config.resolved.yaml"
    if not config_path.exists():
        return {}
    with config_path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _recipe_from_config(config: dict[str, Any], job_id: str) -> str:
    tracking = config.get("tracking") if isinstance(config.get("tracking"), dict) else {}
    tags = tracking.get("tags") if isinstance(tracking.get("tags"), dict) else {}
    recipe = tags.get("recipe")
    if recipe:
        return str(recipe)

    lowered = job_id.lower()
    for marker in ("_maddpg_", "_matd3_", "_mappo_", "_ippo_", "_masac_"):
        if marker not in lowered:
            continue
        head = job_id[: lowered.index(marker)]
        recipe_start = head.find("w6_")
        if recipe_start >= 0:
            return head[recipe_start:]

    algorithm = config.get("algorithm") if isinstance(config.get("algorithm"), dict) else {}
    name = algorithm.get("name")
    if name:
        return str(name).lower()
    return job_id


def _window_from_config(config: dict[str, Any], job_id: str) -> str:
    tracking = config.get("tracking") if isinstance(config.get("tracking"), dict) else {}
    tags = tracking.get("tags") if isinstance(tracking.get("tags"), dict) else {}
    window = tags.get("window")
    if window:
        return str(window)
    parts = job_id.split("_")
    for idx, part in enumerate(parts):
        if part.startswith("win") and idx + 2 < len(parts):
            return "_".join(parts[idx : idx + 3])
    return ""


def _seed_from_config(config: dict[str, Any], job_id: str) -> str:
    tracking = config.get("tracking") if isinstance(config.get("tracking"), dict) else {}
    tags = tracking.get("tags") if isinstance(tracking.get("tags"), dict) else {}
    seed = tags.get("seed")
    if seed not in (None, ""):
        return str(seed)
    match = re.search(r"_seed(\d+)(?:$|_)", job_id)
    return match.group(1) if match else ""


def _result_status(job_dir: Path) -> str:
    result_path = job_dir / "results" / "result.json"
    if not result_path.exists():
        return ""
    try:
        with result_path.open() as fh:
            payload = json.load(fh) or {}
    except (OSError, json.JSONDecodeError):
        return ""
    status = payload.get("status") if isinstance(payload, dict) else ""
    return str(status or "")


def _simulation_root(job_dir: Path) -> Path | None:
    data_dir = job_dir / "results" / "simulation_data"
    if not data_dir.exists():
        return None
    children = [path for path in data_dir.iterdir() if path.is_dir()]
    if children:
        return children[0]
    return data_dir


def _summarize_job(job_dir: Path) -> dict[str, Any] | None:
    root = _simulation_root(job_dir)
    if root is None:
        return None

    config = _load_config(job_dir)
    row: dict[str, Any] = {
        "job_id": job_dir.name,
        "status": _result_status(job_dir),
        "recipe": _recipe_from_config(config, job_dir.name),
        "seed": _seed_from_config(config, job_dir.name),
        "window": _window_from_config(config, job_dir.name),
        "charger_rows": 0,
        "connected_steps": 0,
        "charge_kwh": 0.0,
        "v2g_discharge_kwh": 0.0,
        "v2g_negative_steps": 0,
        "near_departure_connected_steps": 0,
        "near_departure_v2g_kwh": 0.0,
        "near_departure_v2g_steps": 0,
        "unsafe_v2g_kwh": 0.0,
        "unsafe_v2g_steps": 0,
        "worst_v2g_kwh": 0.0,
        "worst_v2g_timestamp": "",
        "worst_v2g_file": "",
        "worst_v2g_departure_time": "",
        "worst_v2g_soc": "",
        "worst_v2g_required_soc": "",
    }

    for path in root.glob("exported_data_building_*_charger_*_ep*.csv"):
        with path.open() as fh:
            for item in csv.DictReader(fh):
                row["charger_rows"] += 1
                connected = _bool(item.get("Is EV Connected"))
                if connected:
                    row["connected_steps"] += 1

                departure_time = _float(item.get("EV Departure Time"))
                near_departure = connected and math.isfinite(departure_time) and 0.0 <= departure_time <= 4.0
                if near_departure:
                    row["near_departure_connected_steps"] += 1

                action = _float(item.get("Charging Action-kWh"))
                if not math.isfinite(action):
                    continue
                if action >= 0.0:
                    row["charge_kwh"] += action
                    continue

                discharge = abs(action)
                row["v2g_discharge_kwh"] += discharge
                row["v2g_negative_steps"] += 1
                if near_departure:
                    row["near_departure_v2g_kwh"] += discharge
                    row["near_departure_v2g_steps"] += 1

                soc = _float(item.get("EV SOC-%"))
                required_soc = _float(item.get("EV Required SOC Departure-%"))
                unsafe = connected and math.isfinite(soc) and math.isfinite(required_soc) and soc < required_soc
                if unsafe:
                    row["unsafe_v2g_kwh"] += discharge
                    row["unsafe_v2g_steps"] += 1

                if discharge > float(row["worst_v2g_kwh"]):
                    row["worst_v2g_kwh"] = discharge
                    row["worst_v2g_timestamp"] = item.get("timestamp", "")
                    row["worst_v2g_file"] = path.name
                    row["worst_v2g_departure_time"] = "" if not math.isfinite(departure_time) else departure_time
                    row["worst_v2g_soc"] = "" if not math.isfinite(soc) else soc
                    row["worst_v2g_required_soc"] = "" if not math.isfinite(required_soc) else required_soc

    row["v2g_kwh_per_connected_step"] = _safe_ratio(
        float(row["v2g_discharge_kwh"]),
        float(row["connected_steps"]),
    )
    row["near_departure_v2g_share"] = _safe_ratio(
        float(row["near_departure_v2g_kwh"]),
        float(row["v2g_discharge_kwh"]),
    )
    row["unsafe_v2g_share"] = _safe_ratio(
        float(row["unsafe_v2g_kwh"]),
        float(row["v2g_discharge_kwh"]),
    )
    return row


def collect_rows(base_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_dir in base_dirs:
        jobs_dir = base_dir / "jobs"
        if not jobs_dir.exists():
            continue
        for job_dir in sorted(path for path in jobs_dir.iterdir() if path.is_dir()):
            result_path = job_dir / "results" / "result.json"
            if not result_path.exists():
                continue
            if _result_status(job_dir) != "completed":
                continue
            row = _summarize_job(job_dir)
            if row is not None and int(row["charger_rows"]) > 0:
                rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "job_id",
        "status",
        "recipe",
        "seed",
        "window",
        "charger_rows",
        "connected_steps",
        "charge_kwh",
        "v2g_discharge_kwh",
        "v2g_negative_steps",
        "v2g_kwh_per_connected_step",
        "near_departure_connected_steps",
        "near_departure_v2g_kwh",
        "near_departure_v2g_steps",
        "near_departure_v2g_share",
        "unsafe_v2g_kwh",
        "unsafe_v2g_steps",
        "unsafe_v2g_share",
        "worst_v2g_kwh",
        "worst_v2g_timestamp",
        "worst_v2g_file",
        "worst_v2g_departure_time",
        "worst_v2g_soc",
        "worst_v2g_required_soc",
    ]
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        action="append",
        required=True,
        type=Path,
        help="Experiment base directory containing a jobs/ subdirectory. Repeatable.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.base_dir)
    _write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
