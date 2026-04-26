from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_KPIS = [
    "community_settled_cost_total_eur",
    "electrical_service_violation_total_kwh",
    "ev_departure_success_rate",
]


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _discover_job_dirs(root: Path) -> List[Path]:
    root = root.resolve()
    candidates: set[Path] = set()

    if (root / "results").is_dir():
        candidates.add(root)

    jobs_dir = root / "jobs"
    if jobs_dir.is_dir():
        for entry in jobs_dir.iterdir():
            if not entry.is_dir():
                continue
            if (entry / "results").is_dir():
                candidates.add(entry)

    for results_dir in root.rglob("results"):
        if results_dir.is_dir():
            candidates.add(results_dir.parent)

    return sorted(candidates)


def _load_yaml_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle) or {}


def _discover_exported_kpi_csvs(job_dir: Path) -> List[Path]:
    simulation_data_dir = job_dir / "results" / "simulation_data"
    if not simulation_data_dir.is_dir():
        return []
    return sorted(
        simulation_data_dir.rglob("exported_kpis.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _extract_kpis_from_exported_csv(csv_path: Path, kpis: List[str]) -> Dict[str, float]:
    tracked = set(kpis)
    selected_kpis: Dict[str, float] = {}

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        normalized_fieldnames = {name.strip().lower(): name for name in fieldnames if isinstance(name, str)}
        kpi_key = normalized_fieldnames.get("kpi")
        cost_function_key = normalized_fieldnames.get("cost_function")
        value_key = normalized_fieldnames.get("value")
        district_key = normalized_fieldnames.get("district")
        level_key = normalized_fieldnames.get("level")
        name_key = normalized_fieldnames.get("name")

        for row in reader:
            if not isinstance(row, dict):
                continue

            if kpi_key is not None:
                metric_name = str(row.get(kpi_key, "")).strip()
                if metric_name not in tracked:
                    continue

                metric_value = _safe_float(row.get(district_key)) if district_key is not None else None
                if metric_value is None:
                    for field in fieldnames:
                        if field == kpi_key:
                            continue
                        metric_value = _safe_float(row.get(field))
                        if metric_value is not None:
                            break

                if metric_value is not None:
                    selected_kpis[metric_name] = metric_value
                continue

            if cost_function_key is not None and value_key is not None:
                metric_name = str(row.get(cost_function_key, "")).strip()
                if metric_name not in tracked:
                    continue

                level = str(row.get(level_key, "")).strip().lower() if level_key else ""
                name = str(row.get(name_key, "")).strip().lower() if name_key else ""
                if level and level != "district":
                    continue
                if name and name != "district":
                    continue

                metric_value = _safe_float(row.get(value_key))
                if metric_value is not None:
                    selected_kpis[metric_name] = metric_value

    return selected_kpis


def _extract_kpis_from_result_payload(result_payload: Dict[str, Any], kpis: List[str]) -> Dict[str, float]:
    evaluation_payload = result_payload.get("evaluation", {}) if isinstance(result_payload, dict) else {}
    evaluation_kpis = evaluation_payload.get("kpis", {}) if isinstance(evaluation_payload, dict) else {}

    selected_kpis: Dict[str, float] = {}
    for kpi_name in kpis:
        value = _safe_float(evaluation_kpis.get(kpi_name)) if isinstance(evaluation_kpis, dict) else None
        if value is not None:
            selected_kpis[kpi_name] = value

    return selected_kpis


def _load_job_record(job_dir: Path, kpis: List[str]) -> Optional[Dict[str, Any]]:
    result_payload = _load_json_payload(job_dir / "results" / "result.json")
    config_payload = _load_yaml_payload(job_dir / "config.resolved.yaml")

    selected_kpis: Dict[str, float] = {}
    kpi_source = "none"
    kpi_source_path: Optional[str] = None

    for csv_path in _discover_exported_kpi_csvs(job_dir):
        selected_kpis = _extract_kpis_from_exported_csv(csv_path, kpis)
        if selected_kpis:
            kpi_source = "exported_kpis_csv"
            kpi_source_path = str(csv_path)
            break

    if not selected_kpis and result_payload:
        selected_kpis = _extract_kpis_from_result_payload(result_payload, kpis)
        if selected_kpis:
            kpi_source = "result_json_evaluation"
            kpi_source_path = str(job_dir / "results" / "result.json")

    if not result_payload and not config_payload and not selected_kpis:
        return None

    return {
        "job_dir": str(job_dir),
        "job_id": job_dir.name,
        "seed": config_payload.get("training", {}).get("seed"),
        "algorithm": config_payload.get("algorithm", {}).get("name"),
        "kpi_source": kpi_source,
        "kpi_source_path": kpi_source_path,
        "kpis": selected_kpis,
    }


def _aggregate(records: List[Dict[str, Any]], kpis: List[str]) -> Dict[str, Any]:
    means: Dict[str, Optional[float]] = {}
    counts: Dict[str, int] = {}
    for kpi_name in kpis:
        values = [record["kpis"][kpi_name] for record in records if kpi_name in record.get("kpis", {})]
        counts[kpi_name] = len(values)
        means[kpi_name] = mean(values) if values else None

    return {
        "run_count": len(records),
        "kpi_means": means,
        "kpi_counts": counts,
        "runs": records,
    }


def compare_export_roots(
    maddpg_root: Path,
    rbc_root: Path,
    *,
    kpis: Optional[List[str]] = None,
    cost_ratio_threshold: float = 1.05,
    grid_ratio_threshold: float = 1.05,
    ev_min_threshold: float = 0.95,
) -> Dict[str, Any]:
    tracked_kpis = kpis or list(DEFAULT_KPIS)

    maddpg_dirs = _discover_job_dirs(Path(maddpg_root))
    rbc_dirs = _discover_job_dirs(Path(rbc_root))

    maddpg_records = [
        record
        for record in (_load_job_record(job_dir, tracked_kpis) for job_dir in maddpg_dirs)
        if record is not None
    ]
    rbc_records = [
        record
        for record in (_load_job_record(job_dir, tracked_kpis) for job_dir in rbc_dirs)
        if record is not None
    ]

    maddpg_agg = _aggregate(maddpg_records, tracked_kpis)
    rbc_agg = _aggregate(rbc_records, tracked_kpis)

    maddpg_cost = maddpg_agg["kpi_means"].get("community_settled_cost_total_eur")
    rbc_cost = rbc_agg["kpi_means"].get("community_settled_cost_total_eur")
    maddpg_grid = maddpg_agg["kpi_means"].get("electrical_service_violation_total_kwh")
    rbc_grid = rbc_agg["kpi_means"].get("electrical_service_violation_total_kwh")
    maddpg_ev = maddpg_agg["kpi_means"].get("ev_departure_success_rate")

    cost_pass = (
        maddpg_cost is not None
        and rbc_cost is not None
        and maddpg_cost <= (cost_ratio_threshold * rbc_cost)
    )
    grid_pass = (
        maddpg_grid is not None
        and rbc_grid is not None
        and maddpg_grid <= (grid_ratio_threshold * rbc_grid)
    )
    ev_pass = maddpg_ev is not None and maddpg_ev >= ev_min_threshold

    report: Dict[str, Any] = {
        "inputs": {
            "maddpg_root": str(Path(maddpg_root).resolve()),
            "rbc_root": str(Path(rbc_root).resolve()),
        },
        "kpis": tracked_kpis,
        "thresholds": {
            "cost_ratio_threshold": cost_ratio_threshold,
            "grid_ratio_threshold": grid_ratio_threshold,
            "ev_min_threshold": ev_min_threshold,
        },
        "aggregates": {
            "MADDPG": maddpg_agg,
            "RBC": rbc_agg,
        },
        "checks": {
            "cost_parity_pass": cost_pass,
            "grid_gate_pass": grid_pass,
            "ev_gate_pass": ev_pass,
        },
    }
    report["overall_pass"] = all(report["checks"].values())
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MADDPG and RBC job exports.")
    parser.add_argument("--maddpg-root", required=True, type=Path, help="Root directory with MADDPG job exports")
    parser.add_argument("--rbc-root", required=True, type=Path, help="Root directory with RBC job exports")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comparison_report.json"),
        help="Path where the comparison report JSON will be written",
    )
    parser.add_argument("--cost-threshold", type=float, default=1.05)
    parser.add_argument("--grid-threshold", type=float, default=1.05)
    parser.add_argument("--ev-min", type=float, default=0.95)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    report = compare_export_roots(
        maddpg_root=args.maddpg_root,
        rbc_root=args.rbc_root,
        cost_ratio_threshold=args.cost_threshold,
        grid_ratio_threshold=args.grid_threshold,
        ev_min_threshold=args.ev_min,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps({"overall_pass": report["overall_pass"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
