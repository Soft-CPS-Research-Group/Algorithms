#!/usr/bin/env python3
"""Prepare, submit, monitor, and archive OPEVA remote experiments safely."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ALLOWED_PAYLOAD_FIELDS = {
    "config",
    "config_path",
    "target_host",
    "target_worker_profile",
    "save_as",
    "job_name",
    "submitted_by",
    "image_tag",
    "deucalion_options",
}
TERMINAL_STATUSES = {"finished", "failed", "stopped", "canceled", "completed"}
HISTORY_FIELDS = [
    "archived_at",
    "campaign",
    "source_commit",
    "image_tags",
    "datasets",
    "windows",
    "algorithms",
    "job_count",
    "finished_count",
    "failed_count",
    "scorecard_decisions",
    "gates_profile",
    "baseline",
    "evidence_horizon",
    "submission_manifest",
    "summary_csv",
    "scorecard_csv",
    "history_document",
]


class ExperimentManagerError(RuntimeError):
    """Expected operator or input error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return cleaned.lower() or "campaign"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExperimentManagerError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExperimentManagerError(f"Invalid JSON in {path}: {exc}") from exc


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ExperimentManagerError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    if not path.exists():
        raise ExperimentManagerError(f"Missing CSV file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_server(explicit: str | None) -> str:
    server = (explicit or os.environ.get("OPEVA_SERVER") or "").strip().rstrip("/")
    if not server:
        raise ExperimentManagerError("Set OPEVA_SERVER or pass --server; no historical default is assumed.")
    if not server.startswith(("http://", "https://")):
        raise ExperimentManagerError("The orchestrator server must start with http:// or https://")
    return server


def _request_json(
    server: str,
    path: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(dict(payload)).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{server}{path}", data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-provided orchestrator URL.
        raw = response.read()
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {"text": raw.decode("utf-8", errors="replace")}


def _load_and_validate_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ExperimentManagerError(f"Missing config: {path}") from exc
    except yaml.YAMLError as exc:
        raise ExperimentManagerError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ExperimentManagerError(f"Config root must be an object: {path}")

    from utils.config_schema import validate_config

    validate_config(raw)
    return raw


def _validate_submission_payload(payload: Mapping[str, Any], *, validate_inline_config: bool) -> None:
    unknown = sorted(set(payload) - ALLOWED_PAYLOAD_FIELDS)
    if unknown:
        raise ExperimentManagerError(f"Unsupported submission payload fields: {', '.join(unknown)}")

    has_config = payload.get("config") is not None
    has_config_path = bool(str(payload.get("config_path") or "").strip())
    if has_config == has_config_path:
        raise ExperimentManagerError("Submission payload must define exactly one of config or config_path.")
    if has_config and not isinstance(payload.get("config"), dict):
        raise ExperimentManagerError("Submission payload config must be an object.")

    target_host = str(payload.get("target_host") or "").strip()
    target_profile = str(payload.get("target_worker_profile") or "").strip()
    if target_host and target_profile:
        raise ExperimentManagerError("target_host and target_worker_profile are mutually exclusive.")
    if target_profile and target_profile not in {"cpu", "gpu"}:
        raise ExperimentManagerError("target_worker_profile must be 'cpu' or 'gpu'.")

    deucalion_options = payload.get("deucalion_options")
    if deucalion_options is not None:
        if target_host != "deucalion":
            raise ExperimentManagerError("deucalion_options requires target_host='deucalion'.")
        if not isinstance(deucalion_options, dict):
            raise ExperimentManagerError("deucalion_options must be an object.")

    for field in ("job_name", "submitted_by", "image_tag"):
        if not str(payload.get(field) or "").strip():
            raise ExperimentManagerError(f"Submission payload requires {field} for traceability.")
    if has_config:
        save_as = str(payload.get("save_as") or "").strip()
        if not save_as:
            raise ExperimentManagerError("Inline config submissions require save_as.")
        if "/" in save_as or "\\" in save_as or save_as in {".", ".."}:
            raise ExperimentManagerError(
                "Inline config save_as must be a filename only, without directory components."
            )

    if validate_inline_config and has_config:
        from utils.config_schema import validate_config

        validate_config(dict(payload["config"]))


def _contains_identifier(payload: Any, expected: str, *, image_tag: bool = False) -> bool:
    if isinstance(payload, dict):
        return any(_contains_identifier(value, expected, image_tag=image_tag) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_identifier(value, expected, image_tag=image_tag) for value in payload)
    if not isinstance(payload, str):
        return False
    if payload == expected:
        return True
    return image_tag and (payload.endswith(f":{expected}") or payload.endswith(f"/{expected}"))


def _host_online(payload: Any, host: str) -> bool | None:
    entry = _host_entry(payload, host)
    if entry is not None and "online" in entry:
        return bool(entry.get("online"))
    return None


def _host_entry(payload: Any, host: str) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    hosts = payload.get("hosts")
    if not isinstance(hosts, Mapping):
        return None
    entry = hosts.get(host)
    return entry if isinstance(entry, Mapping) else None


def _image_entry(payload: Any, image_tag: str) -> Mapping[str, Any] | None:
    if not image_tag:
        return None
    if isinstance(payload, Mapping):
        name = str(payload.get("name") or payload.get("tag") or "").strip()
        if name == image_tag:
            return payload
        for value in payload.values():
            found = _image_entry(value, image_tag)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _image_entry(value, image_tag)
            if found is not None:
                return found
    return None


def _append_json_record(path: Path, record: dict[str, Any]) -> None:
    if not path.exists():
        payload: Any = []
    else:
        payload = _load_json(path)

    if isinstance(payload, list):
        if record.get("job_id") and any(
            isinstance(item, dict) and item.get("job_id") == record["job_id"] for item in payload
        ):
            raise ExperimentManagerError(f"Job {record['job_id']} already exists in {path}")
        payload.append(record)
    elif isinstance(payload, dict) and isinstance(payload.get("submissions"), list):
        payload["submissions"].append(record)
    else:
        raise ExperimentManagerError(f"Cannot append to unsupported manifest format: {path}")
    _write_json(path, payload)


def _append_submission_csv(path: Path, record: Mapping[str, Any]) -> None:
    fields = [
        "submitted_at",
        "job_id",
        "job_name",
        "status",
        "target_host",
        "target_worker_profile",
        "image_tag",
        "payload_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in fields})


def _submission_records(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("submissions", "jobs"):
            values = payload.get(key)
            if isinstance(values, list):
                return [item for item in values if isinstance(item, dict)]
    raise ExperimentManagerError(f"Unsupported submissions manifest: {path}")


def _record_job_id(record: Mapping[str, Any]) -> str:
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    return str(record.get("job_id") or response.get("job_id") or "").strip()


def _job_ids(job_ids: Iterable[str], jobs_files: Iterable[Path]) -> list[str]:
    from scripts.collect_remote_results import _read_jobs_file

    values = [str(value).strip() for value in job_ids if str(value).strip()]
    for path in jobs_files:
        values.extend(_read_jobs_file(path))
    seen: set[str] = set()
    return [value for value in values if value and not (value in seen or seen.add(value))]


def cmd_prepare(args: argparse.Namespace) -> int:
    config = _load_and_validate_config(args.config)
    payload: dict[str, Any] = {
        "config": config,
        "save_as": args.save_as or args.config.name,
        "job_name": args.job_name,
        "submitted_by": args.submitted_by,
        "image_tag": args.image_tag,
    }
    if args.target_host:
        payload["target_host"] = args.target_host
    else:
        payload["target_worker_profile"] = args.target_worker_profile
    if args.deucalion_options:
        payload["deucalion_options"] = _load_json_object(args.deucalion_options)

    _validate_submission_payload(payload, validate_inline_config=False)
    _write_json(args.output, payload)
    print(args.output)
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    server = _resolve_server(args.server)
    endpoints = [
        ("health", "/health"),
        ("hosts", "/hosts"),
        ("queue", "/queue"),
        ("images", "/job-images/versions"),
    ]
    if args.deucalion:
        endpoints.append(("deucalion_partitions", "/deucalion/partitions"))

    snapshot: dict[str, Any] = {
        "checked_at": _utc_now(),
        "server": server,
        "requested_target_host": args.target_host or "",
        "requested_image_tag": args.image_tag or "",
        "responses": {},
        "errors": {},
    }
    for name, endpoint in endpoints:
        try:
            snapshot["responses"][name] = _request_json(server, endpoint, timeout=args.timeout)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            snapshot["errors"][name] = str(exc)

    hosts_payload = snapshot["responses"].get("hosts", {})
    images_payload = snapshot["responses"].get("images", {})
    snapshot["target_host_found"] = (
        None
        if not args.target_host
        else _contains_identifier(hosts_payload, args.target_host)
    )
    snapshot["target_host_online"] = (
        None if not args.target_host else _host_online(hosts_payload, args.target_host)
    )
    snapshot["image_tag_found"] = (
        None
        if not args.image_tag
        else _contains_identifier(images_payload, args.image_tag, image_tag=True)
    )

    if args.target_host == "union-inesctec":
        host_entry = _host_entry(hosts_payload, args.target_host) or {}
        host_info = host_entry.get("info") if isinstance(host_entry.get("info"), Mapping) else {}
        auth = host_info.get("union_auth") if isinstance(host_info.get("union_auth"), Mapping) else {}
        auth_status = str(auth.get("status") or "").strip() or None
        auth_updated_at = auth.get("updated_at")
        auth_age_seconds: float | None = None
        if isinstance(auth_updated_at, (int, float)):
            auth_age_seconds = max(0.0, time.time() - float(auth_updated_at))
        snapshot["union_auth_status"] = auth_status
        snapshot["union_auth_updated_at"] = auth_updated_at
        snapshot["union_auth_age_seconds"] = auth_age_seconds
        snapshot["union_auth_fresh"] = bool(
            auth_status == "authenticated"
            and auth_age_seconds is not None
            and auth_age_seconds <= args.max_union_auth_age_seconds
        )
        image_entry = _image_entry(images_payload, args.image_tag or "")
        snapshot["union_image_ready"] = (
            None if image_entry is None else bool(image_entry.get("union_ready"))
        )

    if args.output:
        _write_json(args.output, snapshot)
    print(json.dumps(snapshot, indent=2, sort_keys=True, default=str))

    strict_fail = bool(snapshot["errors"])
    strict_fail = strict_fail or snapshot["target_host_found"] is False
    strict_fail = strict_fail or snapshot["target_host_online"] is False
    strict_fail = strict_fail or snapshot["image_tag_found"] is False
    if args.target_host == "union-inesctec":
        strict_fail = strict_fail or snapshot["union_auth_fresh"] is False
        strict_fail = strict_fail or snapshot["union_image_ready"] is not True
    return 3 if args.strict and strict_fail else 0


def cmd_submit(args: argparse.Namespace) -> int:
    if not args.confirm_submit:
        raise ExperimentManagerError("Submission requires --confirm-submit after explicit user authorization.")
    payload = _load_json_object(args.payload)
    _validate_submission_payload(payload, validate_inline_config=True)
    server = _resolve_server(args.server)
    response = _request_json(server, "/run-simulation", method="POST", payload=payload, timeout=args.timeout)
    if not isinstance(response, dict):
        raise ExperimentManagerError("Orchestrator returned a non-object submission response.")
    job_id = str(response.get("job_id") or "").strip()
    if not job_id:
        raise ExperimentManagerError(f"Submission response did not contain job_id: {response}")

    record = {
        "submitted_at": _utc_now(),
        "payload_path": str(args.payload),
        "job_id": job_id,
        "job_name": payload.get("job_name", ""),
        "status": response.get("status", ""),
        "target_host": payload.get("target_host", response.get("host", "")),
        "target_worker_profile": payload.get("target_worker_profile", ""),
        "image_tag": payload.get("image_tag", response.get("image_tag", "")),
        "payload": payload,
        "response": response,
    }
    args.campaign_dir.mkdir(parents=True, exist_ok=True)
    _append_json_record(args.campaign_dir / "submitted_jobs.json", record)
    _append_json_record(args.campaign_dir / "payloads.json", {"job_id": job_id, "payload": payload})
    _append_submission_csv(args.campaign_dir / "submitted_jobs.csv", record)
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    return 0


def _watch_snapshot(server: str, job_id: str, args: argparse.Namespace) -> dict[str, Any]:
    status = _request_json(server, f"/status/{quote(job_id)}", timeout=args.timeout)
    snapshot: dict[str, Any] = {
        "observed_at": _utc_now(),
        "job_id": job_id,
        "status_payload": status,
        "status": status.get("status", "unknown") if isinstance(status, dict) else "unknown",
    }
    if args.details:
        snapshot["job_info"] = _request_json(server, f"/job-info/{quote(job_id)}", timeout=args.timeout)
        snapshot["progress"] = _request_json(server, f"/progress/{quote(job_id)}", timeout=args.timeout)
        query = urlencode({"tail_lines": args.tail_lines, "max_bytes": args.max_bytes})
        snapshot["logs_chunk"] = _request_json(
            server,
            f"/logs-chunk/{quote(job_id)}?{query}",
            timeout=args.timeout,
        )
    return snapshot


def _write_watch_outputs(output_dir: Path, snapshots: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "latest_status.json", snapshots)
    with (output_dir / "status_history.jsonl").open("a", encoding="utf-8") as handle:
        for snapshot in snapshots:
            handle.write(json.dumps(snapshot, sort_keys=True, default=str) + "\n")


def cmd_watch(args: argparse.Namespace) -> int:
    ids = _job_ids(args.job_id, args.jobs_file)
    if not ids:
        raise ExperimentManagerError("Provide at least one --job-id or --jobs-file.")
    server = _resolve_server(args.server)
    started = time.monotonic()

    while True:
        snapshots = [_watch_snapshot(server, job_id, args) for job_id in ids]
        if args.output_dir:
            _write_watch_outputs(args.output_dir, snapshots)
        summary = {snapshot["job_id"]: snapshot["status"] for snapshot in snapshots}
        print(json.dumps({"observed_at": _utc_now(), "jobs": summary}, sort_keys=True), flush=True)

        all_terminal = all(str(snapshot["status"]).lower() in TERMINAL_STATUSES for snapshot in snapshots)
        if all_terminal or not args.until_terminal:
            return 0
        if args.max_wait_seconds > 0 and time.monotonic() - started >= args.max_wait_seconds:
            return 4
        time.sleep(args.interval)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _record_config(record: Mapping[str, Any]) -> dict[str, Any]:
    config = _record_payload(record).get("config")
    return dict(config) if isinstance(config, Mapping) else {}


def _record_algorithms(record: Mapping[str, Any]) -> list[str]:
    pipeline = _record_config(record).get("pipeline")
    if not isinstance(pipeline, list):
        return []
    return [str(stage.get("algorithm")) for stage in pipeline if isinstance(stage, Mapping) and stage.get("algorithm")]


def _repo_relative(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _markdown_cell(value: Any) -> str:
    return str(value if value not in (None, "") else "-").replace("|", "\\|").replace("\n", " ")


def _scorecard_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("job_id") or "").strip(): row for row in rows if str(row.get("job_id") or "").strip()}


def _summary_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("job_id") or "").strip(): row for row in rows if str(row.get("job_id") or "").strip()}


def _campaign_markdown(
    *,
    campaign: str,
    archived_at: str,
    source_commit: str,
    image_tags: list[str],
    datasets: list[str],
    windows: list[str],
    algorithms: list[str],
    gates_profile: str,
    baseline: str,
    evidence_horizon: str,
    jobs_file: Path,
    summary_csv: Path,
    scorecard_csv: Path | None,
    job_rows: list[dict[str, Any]],
    notes: list[str],
) -> str:
    lines = [
        f"# Experiment Campaign: {campaign}",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Archived at | `{archived_at}` |",
        f"| Source commit | `{source_commit}` |",
        f"| Image tags | {_markdown_cell(', '.join(image_tags))} |",
        f"| Datasets | {_markdown_cell(', '.join(datasets))} |",
        f"| Windows | {_markdown_cell(', '.join(windows))} |",
        f"| Algorithms | {_markdown_cell(', '.join(algorithms))} |",
        f"| Evidence horizon | `{evidence_horizon}` |",
        f"| Gate profile | {_markdown_cell(gates_profile)} |",
        f"| Baseline | {_markdown_cell(baseline)} |",
        f"| Submission manifest | `{_repo_relative(jobs_file)}` |",
        f"| Results summary | `{_repo_relative(summary_csv)}` |",
        f"| Scorecard | `{_repo_relative(scorecard_csv) if scorecard_csv else '-'}` |",
        "",
        "## Jobs and decisions",
        "",
        "| Job | Algorithm | Seed | Status | Cost EUR | EV min | EV tolerance | Grid kWh | Decision |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in job_rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row.get(key))
                for key in (
                    "job",
                    "algorithm",
                    "seed",
                    "status",
                    "community_cost_eur",
                    "ev_min_acceptable_feasible_rate",
                    "ev_within_tolerance_feasible_rate",
                    "electrical_violation_kwh",
                    "decision",
                )
            )
            + " |"
        )
    lines.extend(["", "## Notes and limitations", ""])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No additional notes were recorded.")
    lines.append("")
    return "\n".join(lines)


def _upsert_history_index(path: Path, row: dict[str, Any], *, replace: bool) -> None:
    existing = _read_csv(path) if path.exists() else []
    same = [item for item in existing if item.get("campaign") == row["campaign"]]
    if same and not replace:
        raise ExperimentManagerError(
            f"Campaign {row['campaign']!r} already exists in {path}; use --replace only with explicit authorization."
        )
    filtered = [item for item in existing if item.get("campaign") != row["campaign"]]
    filtered.append({field: row.get(field, "") for field in HISTORY_FIELDS})
    filtered.sort(key=lambda item: (item.get("archived_at", ""), item.get("campaign", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(filtered)


def cmd_archive(args: argparse.Namespace) -> int:
    records = _submission_records(args.jobs_file)
    summaries = _read_csv(args.summary_csv)
    scorecards = _read_csv(args.scorecard_csv)
    summary_by_job = _summary_index(summaries)
    scorecard_by_job = _scorecard_index(scorecards)

    image_tags: set[str] = set()
    datasets: set[str] = set()
    windows: set[str] = set()
    algorithms: set[str] = set()
    job_rows: list[dict[str, Any]] = []
    job_ids: list[str] = []

    for record in records:
        job_id = _record_job_id(record)
        if job_id:
            job_ids.append(job_id)
        payload = _record_payload(record)
        config = _record_config(record)
        simulator = config.get("simulator") if isinstance(config.get("simulator"), Mapping) else {}
        training = config.get("training") if isinstance(config.get("training"), Mapping) else {}
        image_tag = str(payload.get("image_tag") or record.get("image_tag") or "").strip()
        dataset = str(simulator.get("dataset_name") or "").strip()
        start = simulator.get("simulation_start_time_step")
        end = simulator.get("simulation_end_time_step")
        window = "" if start is None and end is None else f"{start if start is not None else '?'}..{end if end is not None else '?'}"
        image_tags.update([image_tag] if image_tag else [])
        datasets.update([dataset] if dataset else [])
        windows.update([window] if window else [])
        algorithms.update(_record_algorithms(record))

        summary = summary_by_job.get(job_id, {})
        scorecard = scorecard_by_job.get(job_id, {})
        job_rows.append(
            {
                "job": summary.get("job_name") or payload.get("job_name") or record.get("job_name") or job_id,
                "job_id": job_id,
                "algorithm": summary.get("algorithm") or "+".join(_record_algorithms(record)),
                "seed": summary.get("seed") or training.get("seed") or "",
                "status": summary.get("status") or record.get("status") or "",
                "community_cost_eur": summary.get("community_cost_eur") or scorecard.get("cost_eur") or "",
                "ev_min_acceptable_feasible_rate": summary.get("ev_min_acceptable_feasible_rate") or scorecard.get("ev_min_acceptable_feasible_rate") or "",
                "ev_within_tolerance_feasible_rate": summary.get("ev_within_tolerance_feasible_rate") or scorecard.get("ev_within_tolerance_rate") or "",
                "electrical_violation_kwh": summary.get("electrical_violation_kwh") or scorecard.get("electrical_violation_kwh") or "",
                "decision": scorecard.get("decision") or scorecard.get("verdict") or "",
            }
        )

    for summary in summaries:
        job_id = str(summary.get("job_id") or "").strip()
        if job_id and job_id not in job_ids:
            job_rows.append(
                {
                    "job": summary.get("job_name") or job_id,
                    "job_id": job_id,
                    "algorithm": summary.get("algorithm") or "",
                    "seed": summary.get("seed") or "",
                    "status": summary.get("status") or "",
                    "community_cost_eur": summary.get("community_cost_eur") or "",
                    "ev_min_acceptable_feasible_rate": summary.get("ev_min_acceptable_feasible_rate") or "",
                    "ev_within_tolerance_feasible_rate": summary.get("ev_within_tolerance_feasible_rate") or "",
                    "electrical_violation_kwh": summary.get("electrical_violation_kwh") or "",
                    "decision": scorecard_by_job.get(job_id, {}).get("decision")
                    or scorecard_by_job.get(job_id, {}).get("verdict")
                    or "",
                }
            )

    archived_at = _utc_now()
    source_commit = args.source_commit or _git_head()
    date_prefix = archived_at[:10].replace("-", "")
    history_dir = args.history_dir
    index_path = history_dir / "index.csv"
    existing_rows = _read_csv(index_path) if index_path.exists() else []
    existing_campaign = next(
        (row for row in existing_rows if row.get("campaign") == args.campaign),
        None,
    )
    if existing_campaign and not args.replace:
        raise ExperimentManagerError(
            f"Campaign {args.campaign!r} already exists in {index_path}; "
            "use --replace only with explicit authorization."
        )

    existing_document = str((existing_campaign or {}).get("history_document") or "").strip()
    if existing_document and args.replace:
        candidate = Path(existing_document)
        document = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    else:
        document = history_dir / f"{date_prefix}_{_safe_slug(args.campaign)}.md"
    history_root = history_dir.resolve()
    document = document.resolve()
    try:
        document.relative_to(history_root)
    except ValueError as exc:
        raise ExperimentManagerError(
            f"History document must stay inside {history_root}: {document}"
        ) from exc
    if document.exists() and not args.replace:
        raise ExperimentManagerError(f"History document already exists: {document}")

    markdown = _campaign_markdown(
        campaign=args.campaign,
        archived_at=archived_at,
        source_commit=source_commit,
        image_tags=sorted(image_tags),
        datasets=sorted(datasets),
        windows=sorted(windows),
        algorithms=sorted(algorithms),
        gates_profile=args.gates_profile,
        baseline=args.baseline,
        evidence_horizon=args.evidence_horizon,
        jobs_file=args.jobs_file,
        summary_csv=args.summary_csv,
        scorecard_csv=args.scorecard_csv,
        job_rows=job_rows,
        notes=args.note,
    )
    history_dir.mkdir(parents=True, exist_ok=True)
    document.write_text(markdown, encoding="utf-8")

    status_counts = Counter(str(row.get("status") or "").lower() for row in job_rows)
    decision_counts = Counter(str(row.get("decision") or "unscored") for row in job_rows)
    ledger_row = {
        "archived_at": archived_at,
        "campaign": args.campaign,
        "source_commit": source_commit,
        "image_tags": ";".join(sorted(image_tags)),
        "datasets": ";".join(sorted(datasets)),
        "windows": ";".join(sorted(windows)),
        "algorithms": ";".join(sorted(algorithms)),
        "job_count": len(job_rows),
        "finished_count": status_counts["finished"] + status_counts["completed"],
        "failed_count": status_counts["failed"],
        "scorecard_decisions": ";".join(f"{key}:{value}" for key, value in sorted(decision_counts.items())),
        "gates_profile": args.gates_profile,
        "baseline": args.baseline,
        "evidence_horizon": args.evidence_horizon,
        "submission_manifest": _repo_relative(args.jobs_file),
        "summary_csv": _repo_relative(args.summary_csv),
        "scorecard_csv": _repo_relative(args.scorecard_csv),
        "history_document": _repo_relative(document),
    }
    _upsert_history_index(index_path, ledger_row, replace=args.replace)
    print(document)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Validate YAML and build an inline submission payload.")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--job-name", required=True)
    prepare.add_argument("--submitted-by", required=True)
    prepare.add_argument("--image-tag", required=True)
    target = prepare.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-host")
    target.add_argument("--target-worker-profile", choices=("cpu", "gpu"))
    prepare.add_argument("--save-as")
    prepare.add_argument("--deucalion-options", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=cmd_prepare)

    preflight = subparsers.add_parser("preflight", help="Read orchestrator readiness without mutation.")
    preflight.add_argument("--server")
    preflight.add_argument("--target-host")
    preflight.add_argument("--image-tag")
    preflight.add_argument("--deucalion", action="store_true")
    preflight.add_argument("--strict", action="store_true")
    preflight.add_argument("--timeout", type=float, default=30.0)
    preflight.add_argument(
        "--max-union-auth-age-seconds",
        type=float,
        default=86400.0,
        help="Maximum age of Union authentication evidence in strict mode (default: 24 hours).",
    )
    preflight.add_argument("--output", type=Path)
    preflight.set_defaults(func=cmd_preflight)

    submit = subparsers.add_parser("submit", help="Submit a validated payload after explicit authorization.")
    submit.add_argument("--server")
    submit.add_argument("--payload", type=Path, required=True)
    submit.add_argument("--campaign-dir", type=Path, required=True)
    submit.add_argument("--timeout", type=float, default=30.0)
    submit.add_argument("--confirm-submit", action="store_true")
    submit.set_defaults(func=cmd_submit)

    watch = subparsers.add_parser("watch", help="Read job status once or until all jobs reach terminal state.")
    watch.add_argument("--server")
    watch.add_argument("--job-id", action="append", default=[])
    watch.add_argument("--jobs-file", type=Path, action="append", default=[])
    watch.add_argument("--until-terminal", action="store_true")
    watch.add_argument("--details", action="store_true")
    watch.add_argument("--interval", type=float, default=30.0)
    watch.add_argument("--max-wait-seconds", type=float, default=0.0)
    watch.add_argument("--timeout", type=float, default=30.0)
    watch.add_argument("--tail-lines", type=int, default=200)
    watch.add_argument("--max-bytes", type=int, default=262144)
    watch.add_argument("--output-dir", type=Path)
    watch.set_defaults(func=cmd_watch)

    archive = subparsers.add_parser("archive", help="Write a concise versioned experiment history record.")
    archive.add_argument("--campaign", required=True)
    archive.add_argument("--jobs-file", type=Path, required=True)
    archive.add_argument("--summary-csv", type=Path, required=True)
    archive.add_argument("--scorecard-csv", type=Path)
    archive.add_argument("--gates-profile", required=True)
    archive.add_argument("--baseline", required=True)
    archive.add_argument(
        "--evidence-horizon",
        choices=("smoke", "partial-window", "full-episode", "full-year", "unknown"),
        default="unknown",
    )
    archive.add_argument("--source-commit")
    archive.add_argument("--history-dir", type=Path, default=REPO_ROOT / "docs" / "experiment_history")
    archive.add_argument("--note", action="append", default=[])
    archive.add_argument("--replace", action="store_true")
    archive.set_defaults(func=cmd_archive)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "interval", 1.0) <= 0:
        raise SystemExit("--interval must be > 0")
    if getattr(args, "max_union_auth_age_seconds", 1.0) <= 0:
        raise SystemExit("--max-union-auth-age-seconds must be > 0")
    try:
        return int(args.func(args))
    except (ExperimentManagerError, HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
