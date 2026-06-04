"""Summarize local Phase 10 W6 jobs from exported simulator KPIs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

COMMUNITY_COST_KPI_CANDIDATES = (
    "district_cost_community_market_settled_total_eur",
    "district_community_settled_cost_total_eur",
    "district_cost_total_control_eur",
)

KPI_KEYS = {
    "community_cost_eur": COMMUNITY_COST_KPI_CANDIDATES,
    "community_import_kwh": "district_energy_grid_total_import_control_kwh",
    "community_export_kwh": "district_energy_grid_total_export_control_kwh",
    "community_net_exchange_kwh": "district_energy_grid_total_net_exchange_control_kwh",
    "peak_daily_ratio_to_baseline": "district_energy_grid_shape_quality_peak_daily_average_to_baseline_ratio",
    "peak_all_time_ratio_to_baseline": "district_energy_grid_shape_quality_peak_all_time_average_to_baseline_ratio",
    "load_factor_penalty_ratio": "district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_baseline_ratio",
    "battery_throughput_kwh": "district_battery_total_throughput_kwh",
    "v2g_export_kwh": "district_ev_total_v2g_export_kwh",
    "ev_min_acceptable_feasible_rate": "district_ev_performance_departure_min_acceptable_feasible_ratio",
    "ev_within_tolerance_rate": "district_ev_performance_departure_within_tolerance_feasible_ratio",
    "electrical_violation_kwh": "district_electrical_service_phase_violations_energy_total_kwh",
    "community_solar_self_consumption_rate": "district_solar_self_consumption_ratio_self_consumption_ratio",
}

FIELDS = [
    "run_group",
    "job_id",
    "algorithm",
    "recipe",
    "seed",
    "window",
    "status",
    "reward_function",
    *KPI_KEYS,
    "cost_delta_vs_rbcsmart_eur",
    "cost_saving_vs_rbcsmart_eur",
    "cost_ratio_vs_rbcsmart",
    "battery_delta_vs_rbcsmart_kwh",
    "battery_ratio_vs_rbcsmart",
    "cost_saving_per_battery_kwh",
    "cost_saving_per_incremental_battery_kwh",
    "v2g_delta_vs_rbcsmart",
    "self_consumption_delta_vs_rbcsmart",
    "pass_ev_min",
    "pass_ev_tol",
    "pass_cost",
    "pass_electrical",
    "runtime_last_episode_s",
    "steps_per_second_last_episode",
]


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_config(job_dir: Path) -> Mapping[str, Any]:
    config_path = job_dir / "config.resolved.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _district_kpi_value(rows: Mapping[str, Mapping[str, Any]], kpi_name: str) -> float | None:
    return _float(rows.get(kpi_name, {}).get("District"))


def _district_kpi_candidate_value(
    rows: Mapping[str, Mapping[str, Any]],
    kpi_names: str | Sequence[str],
) -> float | None:
    if isinstance(kpi_names, str):
        return _district_kpi_value(rows, kpi_names)

    fallback: float | None = None
    for index, kpi_name in enumerate(kpi_names):
        value = _district_kpi_value(rows, kpi_name)
        if value is None:
            continue
        if index == len(kpi_names) - 1:
            fallback = value
        elif abs(value) > 1e-12:
            return value

    return fallback


def _read_district_kpis(job_dir: Path) -> dict[str, float | None]:
    candidates = sorted((job_dir / "results" / "simulation_data").glob("**/exported_kpis.csv"))
    if not candidates:
        return {name: None for name in KPI_KEYS}

    with candidates[-1].open("r", encoding="utf-8", newline="") as handle:
        rows = {row["KPI"]: row for row in csv.DictReader(handle)}

    values: dict[str, float | None] = {}
    for output_name, kpi_names in KPI_KEYS.items():
        values[output_name] = _district_kpi_candidate_value(rows, kpi_names)

    return values


def _last_episode_duration(job_dir: Path) -> tuple[float | None, float | None]:
    logs = sorted((job_dir / "logs").glob("*.log"))
    if not logs:
        return None, None
    text = logs[-1].read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Completed episode\s+\d+/\d+.*?duration:\s+([0-9.]+)s", text, flags=re.S)
    if not matches:
        return None, None
    duration = float(matches[-1])
    progress = _read_json(job_dir / "progress" / "progress.json")
    step_total = _float(progress.get("step_total"))
    return duration, (step_total / duration if step_total and duration > 0.0 else None)


def _job_identity(job_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    job_id = job_dir.name
    algorithm = str(config.get("algorithm", {}).get("name") or "")
    if not algorithm:
        if "rbcsmartpolicy" in job_id:
            algorithm = "RBCSmartPolicy"
        elif "rbccommunitypolicy" in job_id:
            algorithm = "RBCCommunityPolicy"
        else:
            algorithm = "unknown"

    recipe = algorithm
    match = re.search(r"phase10_[^_]+_(.+?)_(?:maddpg|matd3)_", job_id)
    if match:
        recipe = match.group(1)
    elif "rbcsmartpolicy" in job_id:
        recipe = "RBCSmartPolicy"
    elif "rbccommunitypolicy" in job_id:
        recipe = "RBCCommunityPolicy"

    window_match = re.search(r"(win\d+_\d+_\d+)", job_id)
    seed_match = re.search(r"seed(\d+)", job_id)
    return {
        "job_id": job_id,
        "algorithm": algorithm,
        "recipe": recipe,
        "window": window_match.group(1) if window_match else "",
        "seed": int(seed_match.group(1)) if seed_match else "",
        "reward_function": str(config.get("simulator", {}).get("reward_function") or ""),
    }


def summarize(base_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_dir in base_dirs:
        jobs_dir = base_dir / "jobs"
        if not jobs_dir.exists():
            continue
        for job_dir in sorted(path for path in jobs_dir.iterdir() if path.is_dir()):
            result = _read_json(job_dir / "results" / "result.json")
            status = str(result.get("status") or "missing")
            config = _read_config(job_dir)
            identity = _job_identity(job_dir, config)
            runtime_s, steps_per_s = _last_episode_duration(job_dir)
            row = {
                "run_group": base_dir.name,
                **identity,
                "status": status,
                **_read_district_kpis(job_dir),
                "runtime_last_episode_s": runtime_s,
                "steps_per_second_last_episode": steps_per_s,
            }
            rows.append(row)

    references = {
        row["window"]: row
        for row in rows
        if row.get("recipe") == "RBCSmartPolicy" and row.get("status") == "completed"
    }
    for row in rows:
        ref = references.get(row.get("window"))
        cost = _float(row.get("community_cost_eur"))
        ref_cost = _float(ref.get("community_cost_eur")) if ref else None
        battery = _float(row.get("battery_throughput_kwh"))
        ref_battery = _float(ref.get("battery_throughput_kwh")) if ref else None
        v2g = _float(row.get("v2g_export_kwh"))
        ref_v2g = _float(ref.get("v2g_export_kwh")) if ref else None
        self_consumption = _float(row.get("community_solar_self_consumption_rate"))
        ref_self_consumption = _float(ref.get("community_solar_self_consumption_rate")) if ref else None

        row["cost_delta_vs_rbcsmart_eur"] = cost - ref_cost if cost is not None and ref_cost is not None else None
        row["cost_saving_vs_rbcsmart_eur"] = (
            ref_cost - cost if cost is not None and ref_cost is not None else None
        )
        row["cost_ratio_vs_rbcsmart"] = cost / ref_cost if cost is not None and ref_cost not in (None, 0.0) else None
        row["battery_delta_vs_rbcsmart_kwh"] = (
            battery - ref_battery if battery is not None and ref_battery is not None else None
        )
        row["battery_ratio_vs_rbcsmart"] = (
            battery / ref_battery if battery is not None and ref_battery not in (None, 0.0) else None
        )
        row["cost_saving_per_battery_kwh"] = (
            row["cost_saving_vs_rbcsmart_eur"] / battery
            if row["cost_saving_vs_rbcsmart_eur"] is not None and battery not in (None, 0.0)
            else None
        )
        row["cost_saving_per_incremental_battery_kwh"] = (
            row["cost_saving_vs_rbcsmart_eur"] / row["battery_delta_vs_rbcsmart_kwh"]
            if row["cost_saving_vs_rbcsmart_eur"] is not None
            and row["battery_delta_vs_rbcsmart_kwh"] not in (None, 0.0)
            and row["battery_delta_vs_rbcsmart_kwh"] > 0.0
            else None
        )
        row["v2g_delta_vs_rbcsmart"] = v2g - ref_v2g if v2g is not None and ref_v2g is not None else None
        row["self_consumption_delta_vs_rbcsmart"] = (
            self_consumption - ref_self_consumption
            if self_consumption is not None and ref_self_consumption is not None
            else None
        )
        row["pass_ev_min"] = int((_float(row.get("ev_min_acceptable_feasible_rate")) or 0.0) >= 0.99)
        row["pass_ev_tol"] = int((_float(row.get("ev_within_tolerance_rate")) or 0.0) >= 0.40)
        row["pass_cost"] = int(
            row["cost_delta_vs_rbcsmart_eur"] is not None and row["cost_delta_vs_rbcsmart_eur"] <= 0.0
        )
        row["pass_electrical"] = int((_float(row.get("electrical_violation_kwh")) or 0.0) == 0.0)

    rows.sort(
        key=lambda row: (
            str(row.get("window")),
            0 if row.get("recipe") == "RBCSmartPolicy" else 1,
            row.get("cost_delta_vs_rbcsmart_eur") if row.get("cost_delta_vs_rbcsmart_eur") is not None else 1e18,
        )
    )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    rows = summarize(args.base_dir)
    if args.output:
        write_csv(args.output, rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
