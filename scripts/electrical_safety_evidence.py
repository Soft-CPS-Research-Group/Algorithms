"""Derive executed electrical-service safety evidence from simulator exports.

CityLearn's legacy violation KPI records the larger of the requested
pre-projection violation and the residual post-projection violation.  That is
the right conservative signal for the strict gate, but it cannot by itself
answer whether the action actually executed inside the configured service
limits.  This module reconstructs that second fact from exported physical
power peaks and the resolved dataset schema.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml


DEFAULT_PROJECTION_TOLERANCE_KWH = 1.0
DEFAULT_PROJECTION_EVENT_RATE = 1.0e-3
NUMERICAL_RESIDUE_TOLERANCE_KWH = 1.0e-5
POWER_EPS_KW = 1.0e-5


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _latest_kpi_path(data_dir: Path) -> Path | None:
    direct = data_dir / "exported_kpis.csv"
    if direct.is_file():
        return direct
    candidates = sorted(data_dir.glob("exported_kpis_ep*.csv"))
    return candidates[-1] if candidates else None


def _find_resolved_config(data_dir: Path) -> Path | None:
    current = data_dir.resolve()
    for directory in (current, *current.parents):
        candidate = directory / "config.resolved.yaml"
        if candidate.is_file():
            return candidate
    return None


def _resolve_dataset_schema(data_dir: Path) -> tuple[Path | None, Mapping[str, Any] | None]:
    config_path = _find_resolved_config(data_dir)
    if config_path is None:
        return None, None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_path = (config.get("simulator") or {}).get("dataset_path")
    if not raw_path:
        return None, None

    raw = Path(str(raw_path))
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(parent / raw for parent in (config_path.parent, *config_path.parents))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            import json

            return resolved, json.loads(resolved.read_text(encoding="utf-8"))
    return None, None


def _building_number(building_name: str) -> int | None:
    match = re.fullmatch(r"Building_(\d+)", str(building_name))
    return int(match.group(1)) if match else None


def _building_net_peaks_kw(
    data_dir: Path,
    building_name: str,
    seconds_per_time_step: float,
) -> tuple[float | None, float | None]:
    number = _building_number(building_name)
    if number is None or seconds_per_time_step <= 0.0:
        return None, None
    candidates = sorted(data_dir.glob(f"exported_data_building_{number}_ep*.csv"))
    if not candidates:
        return None, None
    frame = pd.read_csv(candidates[-1])
    column = "Net Electricity Consumption-kWh"
    if column not in frame:
        return None, None
    net_kwh = pd.to_numeric(frame[column], errors="coerce").dropna()
    if net_kwh.empty:
        return None, None
    multiplier = 3600.0 / float(seconds_per_time_step)
    return (
        float(net_kwh.clip(lower=0.0).max()) * multiplier,
        float((-net_kwh).clip(lower=0.0).max()) * multiplier,
    )


def _infer_time_steps(data_dir: Path) -> int:
    candidates = sorted(data_dir.glob("exported_data_community_ep*.csv"))
    if not candidates:
        candidates = sorted(data_dir.glob("exported_data_building_*_ep*.csv"))
    if not candidates:
        return 0
    return int(len(pd.read_csv(candidates[-1], usecols=[0])))


def _phase_peaks_by_building(data_dir: Path) -> Mapping[str, Mapping[str, float]]:
    path = _latest_kpi_path(data_dir)
    if path is None:
        return {}
    pattern = re.compile(
        r"^building_electrical_service_phase_phase_peaks_"
        r"(import|export)_peak_([^_]+)_kw$"
    )
    output: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        buildings = [name for name in (reader.fieldnames or []) if name.startswith("Building_")]
        for row in reader:
            match = pattern.fullmatch(str(row.get("KPI") or ""))
            if match is None:
                continue
            direction, phase = match.groups()
            for building in buildings:
                value = _finite_float(row.get(building))
                if value is not None:
                    output.setdefault(building, {})[f"{direction}_{phase.upper()}"] = value
    return output


def executed_safety_evidence(
    data_dir: Path,
    *,
    requested_violation_kwh: float | None,
    requested_violation_events: float | None,
    building_names: Sequence[str] | None = None,
    tolerance_kwh: float = DEFAULT_PROJECTION_TOLERANCE_KWH,
    max_event_rate: float = DEFAULT_PROJECTION_EVENT_RATE,
    time_steps: int | None = None,
) -> dict[str, Any]:
    """Return a conservative post-projection safety certificate.

    Certification requires all configured total and per-phase limits to be
    checkable from the exports.  Missing configuration or missing peak data is
    an explicit failure, never an assumed pass.
    """

    data_dir = Path(data_dir)
    schema_path, schema = _resolve_dataset_schema(data_dir)
    violation_kwh = _finite_float(requested_violation_kwh)
    violation_events = _finite_float(requested_violation_events)
    steps = max(int(time_steps or 0), 0) or _infer_time_steps(data_dir)
    allowed_events = max(1, int(math.ceil(steps * float(max_event_rate))))
    numerical_residue = int(
        violation_kwh is not None
        and violation_kwh <= NUMERICAL_RESIDUE_TOLERANCE_KWH + 1.0e-12
    )
    within_request_tolerance = int(
        violation_kwh is not None
        and violation_events is not None
        and violation_kwh <= float(tolerance_kwh) + 1.0e-12
        and (
            violation_events <= float(allowed_events) + 1.0e-12
            or numerical_residue == 1
        )
    )

    base: dict[str, Any] = {
        "executed_safety_evidence_profile": "executed_safety_projection_v1",
        "projection_tolerance_kwh": float(tolerance_kwh),
        "projection_max_event_rate": float(max_event_rate),
        "projection_max_event_count": allowed_events,
        "projection_request_within_tolerance": within_request_tolerance,
        "projection_request_numerical_residue": numerical_residue,
        "executed_electrical_safety_certified": 0,
        "executed_safety_schema_path": str(schema_path) if schema_path else None,
        "executed_safety_building_count": 0,
        "executed_safety_limit_check_count": 0,
        "executed_safety_missing_evidence_count": 0,
        "executed_safety_limit_failure_count": 0,
        "executed_safety_evidence": [],
    }
    if schema is None:
        base["executed_safety_missing_evidence_count"] = 1
        return base

    buildings = schema.get("buildings") or {}
    selected = list(building_names) if building_names is not None else list(buildings)
    selected = [name for name in selected if (buildings.get(name) or {}).get("electrical_service")]
    base["executed_safety_building_count"] = len(selected)
    if not selected:
        base["executed_safety_missing_evidence_count"] = 1
        return base

    seconds = _finite_float(schema.get("seconds_per_time_step")) or 0.0
    phase_peaks = _phase_peaks_by_building(data_dir)
    evidence_rows: list[dict[str, Any]] = []
    checks = 0
    missing = 0
    failures = 0

    for building_name in selected:
        service = (buildings[building_name].get("electrical_service") or {})
        limits = service.get("limits") or {}
        total_limits = limits.get("total") or {}
        per_phase_limits = limits.get("per_phase") or {}
        total_import_peak, total_export_peak = _building_net_peaks_kw(
            data_dir, building_name, seconds
        )
        row: dict[str, Any] = {
            "building": building_name,
            "total_import_peak_kw": total_import_peak,
            "total_export_peak_kw": total_export_peak,
            "checks": [],
        }

        def record(label: str, peak: float | None, limit: Any) -> None:
            nonlocal checks, missing, failures
            parsed_limit = _finite_float(limit)
            if parsed_limit is None:
                return
            checks += 1
            passed = peak is not None and peak <= parsed_limit + POWER_EPS_KW
            if peak is None:
                missing += 1
            elif not passed:
                failures += 1
            row["checks"].append(
                {"scope": label, "peak_kw": peak, "limit_kw": parsed_limit, "pass": int(passed)}
            )

        record("total_import", total_import_peak, total_limits.get("import_kw"))
        record("total_export", total_export_peak, total_limits.get("export_kw"))
        peaks = phase_peaks.get(building_name, {})
        for raw_phase, phase_limit in per_phase_limits.items():
            phase = str(raw_phase).upper()
            record(f"{phase}_import", peaks.get(f"import_{phase}"), (phase_limit or {}).get("import_kw"))
            record(f"{phase}_export", peaks.get(f"export_{phase}"), (phase_limit or {}).get("export_kw"))
        evidence_rows.append(row)

    base.update(
        {
            "executed_safety_limit_check_count": checks,
            "executed_safety_missing_evidence_count": missing,
            "executed_safety_limit_failure_count": failures,
            "executed_safety_evidence": evidence_rows,
            "executed_electrical_safety_certified": int(
                checks > 0 and missing == 0 and failures == 0
            ),
        }
    )
    return base
