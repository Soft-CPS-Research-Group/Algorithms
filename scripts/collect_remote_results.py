#!/usr/bin/env python3
"""Collect remote orchestrator job results into a compact comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_SERVER = "http://193.136.62.78:8011"

DISTRICT_ENTITY_CANDIDATES = ("District", "district", "Community", "community", "Overall", "overall")

IMPORTANT_KPIS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "community_cost_eur",
            (
                "district_community_settled_cost_total_eur",
                "district_cost_total_control_eur",
            ),
        ),
        ("cost_bau_eur", ("district_cost_total_business_as_usual_eur",)),
        ("cost_delta_to_bau_eur", ("district_cost_total_delta_to_business_as_usual_eur",)),
        ("cost_ratio_to_bau", ("district_cost_ratio_to_business_as_usual_total_ratio",)),
        (
            "ev_min_acceptable_feasible_rate",
            ("district_ev_performance_departure_min_acceptable_feasible_ratio",),
        ),
        ("ev_min_acceptable_rate", ("district_ev_performance_departure_min_acceptable_ratio",)),
        (
            "ev_within_tolerance_feasible_rate",
            ("district_ev_performance_departure_within_tolerance_feasible_ratio",),
        ),
        ("ev_within_tolerance_rate", ("district_ev_performance_departure_within_tolerance_ratio",)),
        ("ev_departure_count", ("district_ev_events_departure_count",)),
        (
            "ev_departure_infeasible_count",
            (
                "district_ev_events_departure_min_acceptable_infeasible_count",
                "district_ev_events_departure_within_tolerance_infeasible_count",
            ),
        ),
        (
            "electrical_violation_kwh",
            ("district_electrical_service_phase_violations_energy_total_kwh",),
        ),
        (
            "electrical_violation_events",
            ("district_electrical_service_phase_violations_event_count",),
        ),
        (
            "peak_daily_ratio_to_bau",
            ("district_energy_grid_shape_quality_peak_daily_average_to_business_as_usual_ratio",),
        ),
        (
            "peak_all_time_ratio_to_bau",
            ("district_energy_grid_shape_quality_peak_all_time_average_to_business_as_usual_ratio",),
        ),
        (
            "battery_throughput_kwh",
            ("district_battery_total_throughput_kwh",),
        ),
        (
            "battery_throughput_ratio_to_bau",
            ("district_battery_ratio_to_business_as_usual_throughput_ratio",),
        ),
        (
            "net_exchange_kwh",
            ("district_energy_grid_total_net_exchange_control_kwh",),
        ),
        (
            "net_exchange_delta_to_bau_kwh",
            ("district_energy_grid_total_net_exchange_delta_to_business_as_usual_kwh",),
        ),
        (
            "v2g_export_kwh",
            ("district_ev_total_v2g_export_kwh",),
        ),
    ]
)


def _json_default(value: Any) -> str:
    return str(value)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> bytes:
    url = f"{base_url.rstrip('/')}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-provided orchestrator URL.
        return response.read()


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any] | list[Any]:
    raw = _request(base_url, path, method=method, payload=payload, timeout=timeout)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _request_text(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> str:
    raw = _request(base_url, path, method=method, payload=payload, timeout=timeout)
    return raw.decode("utf-8", errors="replace")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_text(path: Path, data: str) -> None:
    path.write_text(data, encoding="utf-8")


def _as_dict(payload: Any) -> dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def _extract_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if parsed == parsed else None


def _parse_kpi_matrix(content: str) -> dict[str, dict[str, float | None]]:
    rows: dict[str, dict[str, float | None]] = {}
    reader = csv.DictReader(content.splitlines())
    if not reader.fieldnames:
        return rows
    key_field = reader.fieldnames[0]
    for row in reader:
        key = str(row.get(key_field, "")).strip()
        if not key:
            continue
        rows[key] = {entity: _parse_float(row.get(entity)) for entity in reader.fieldnames[1:]}
    return rows


def _district_value(matrix: dict[str, dict[str, float | None]], key: str) -> float | None:
    values = matrix.get(key)
    if not values:
        return None
    for entity in DISTRICT_ENTITY_CANDIDATES:
        value = values.get(entity)
        if value is not None:
            return value
    for value in values.values():
        if value is not None:
            return value
    return None


def _first_kpi_value(matrix: dict[str, dict[str, float | None]], candidates: tuple[str, ...]) -> float | None:
    for key in candidates:
        value = _district_value(matrix, key)
        if value is not None:
            return value
    return None


def _extract_device(log_text: str) -> str:
    matches = re.findall(r"Device selected:\s*([A-Za-z0-9_:.-]+)", log_text)
    if matches:
        return matches[-1].lower()
    if "cuda_available=True" in log_text:
        return "cuda_available"
    if "cuda_available=False" in log_text:
        return "cpu"
    return ""


def _extract_simulation_data_defaults(result: dict[str, Any]) -> tuple[str | None, str | None]:
    simulation_dir = result.get("simulation_data_dir") or result.get("simulation_data_path")
    session = result.get("simulation_data_session_default")
    if not isinstance(simulation_dir, str):
        simulation_dir = None
    if not isinstance(session, str):
        session = None
    return simulation_dir, session


def _read_jobs_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict):
            values = payload.get("job_ids") or payload.get("jobs") or []
            return [str(item) for item in values]
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames and "job_id" in reader.fieldnames:
            return [str(row["job_id"]).strip() for row in reader if str(row.get("job_id", "")).strip()]
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def _collect_one(base_url: str, job_id: str, output_dir: Path, tail_lines: int, timeout: float) -> dict[str, Any]:
    job_dir = output_dir / "jobs" / _safe_name(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = OrderedDict()
    row["job_id"] = job_id
    errors: list[str] = []

    status: dict[str, Any] = {}
    info: dict[str, Any] = {}
    result: dict[str, Any] = {}
    logs = ""
    kpi_matrix: dict[str, dict[str, float | None]] = {}

    endpoints = [
        ("status", f"/status/{quote(job_id)}", "json"),
        ("job_info", f"/job-info/{quote(job_id)}", "json"),
        ("result", f"/result/{quote(job_id)}", "json"),
        (
            "logs_chunk",
            f"/logs-chunk/{quote(job_id)}?{urlencode({'tail_lines': tail_lines, 'max_bytes': 200000})}",
            "json",
        ),
        ("resolved_config", f"/job-resolved-config/{quote(job_id)}", "text"),
    ]

    for name, path, kind in endpoints:
        try:
            if kind == "text":
                payload_text = _request_text(base_url, path, timeout=timeout)
                _write_text(job_dir / f"{name}.yaml", payload_text)
                continue
            payload_json = _request_json(base_url, path, timeout=timeout)
            _write_json(job_dir / f"{name}.json", payload_json)
            if name == "status":
                status = _as_dict(payload_json)
            elif name == "job_info":
                info = _as_dict(payload_json)
            elif name == "result":
                result = _as_dict(payload_json)
            elif name == "logs_chunk":
                payload_dict = _as_dict(payload_json)
                logs = str(payload_dict.get("text") or "")
                _write_text(job_dir / "logs_tail.txt", logs)
        except HTTPError as exc:
            errors.append(f"{name}: HTTP {exc.code}")
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            errors.append(f"{name}: {exc}")

    try:
        _, session_default = _extract_simulation_data_defaults(result)
        index = _request_json(
            base_url,
            "/simulation-data/index",
            method="POST",
            payload={"job_id": job_id, "session": session_default or "latest"},
            timeout=timeout,
        )
        index_dict = _as_dict(index)
        _write_json(job_dir / "simulation_data_index.json", index_dict)
        files = [str(item) for item in index_dict.get("files", []) if isinstance(item, str)]
        kpi_file = next((item for item in files if item.endswith("exported_kpis.csv")), None)
        if kpi_file:
            kpi_content = _request_text(
                base_url,
                "/simulation-data/file",
                method="POST",
                payload={
                    "job_id": job_id,
                    "session": index_dict.get("session") or session_default or "latest",
                    "relative_path": kpi_file,
                },
                timeout=timeout,
            )
            _write_text(job_dir / "exported_kpis.csv", kpi_content)
            kpi_matrix = _parse_kpi_matrix(kpi_content)
            row["simulation_data_session"] = index_dict.get("session") or ""
            row["kpi_file"] = kpi_file
    except HTTPError as exc:
        errors.append(f"simulation_data: HTTP {exc.code}")
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        errors.append(f"simulation_data: {exc}")

    details = _as_dict(status.get("details"))
    row.update(
        {
            "status": status.get("status") or "",
            "worker_id": status.get("worker_id") or "",
            "worker_version": status.get("worker_version") or "",
            "job_name": info.get("job_name") or "",
            "config_path": info.get("config_path") or "",
            "target_host": info.get("target_host") or "",
            "image_tag": info.get("image_tag") or "",
            "image": info.get("image") or details.get("image") or "",
            "slurm_job_id": details.get("slurm_job_id") or "",
            "slurm_state": details.get("slurm_state") or "",
            "slurm_reason": details.get("slurm_reason") or "",
            "slurm_start_time": details.get("slurm_start_time") or "",
            "slurm_queue_position": details.get("slurm_queue_position") or "",
            "device_selected": _extract_device(logs),
            "simulation_data_available": result.get("simulation_data_available"),
            "kpi_source": result.get("kpi_source") or "",
            "errors": "; ".join(errors),
        }
    )

    for output_key, candidates in IMPORTANT_KPIS.items():
        row[output_key] = _first_kpi_value(kpi_matrix, candidates)

    return row


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    _write_json(summary_json, rows)

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Remote Results Collection",
                "",
                f"Collected at Unix time `{time.time():.0f}`.",
                "",
                "- `summary.csv`: compact KPI comparison table.",
                "- `summary.json`: same data in JSON.",
                "- `jobs/<job_id>/`: raw status, job info, result, logs tail, resolved config and KPI CSV when available.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.environ.get("OPEVA_SERVER", DEFAULT_SERVER))
    parser.add_argument("--job-id", action="append", default=[], help="Remote orchestrator job id. Can be repeated.")
    parser.add_argument(
        "--jobs-file",
        type=Path,
        action="append",
        default=[],
        help="Text/CSV/JSON file with job ids. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/remote_results") / time.strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--tail-lines", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    job_ids = [str(item).strip() for item in args.job_id if str(item).strip()]
    for jobs_file in args.jobs_file:
        job_ids.extend(_read_jobs_file(jobs_file))

    seen: set[str] = set()
    unique_job_ids = [job_id for job_id in job_ids if not (job_id in seen or seen.add(job_id))]
    if not unique_job_ids:
        print("No job ids provided. Use --job-id or --jobs-file.", file=sys.stderr)
        return 2

    rows = []
    for job_id in unique_job_ids:
        print(f"Collecting {job_id}...", file=sys.stderr)
        rows.append(_collect_one(args.server, job_id, args.output_dir, args.tail_lines, args.timeout))

    _write_summary(args.output_dir, rows)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
