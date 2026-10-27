"""Reproducible paired-evaluation and checkpoint-selection helpers.

The protocol deliberately lives outside an agent implementation: baselines and
learning agents must be compared on the same simulator surface.  It records the
surface separately from the candidate policy, extracts a canonical scorecard
from Simulator 1.7 KPI exports, and promotes checkpoints using rules frozen
before confirmation runs are opened.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "ti_marl_experiment_protocol_v1"
EVALUATION_RECORD_VERSION = "ti_marl_evaluation_record_v1"
SELECTION_RECORD_VERSION = "ti_marl_checkpoint_selection_v1"
CONFIRMATION_REPORT_VERSION = "ti_marl_confirmation_report_v1"


KPI_ROWS: Mapping[str, tuple[str, ...]] = {
    "cost_eur": (
        "district_cost_community_market_settled_total_eur",
        "district_cost_total_control_eur",
    ),
    "peak_daily_ratio_to_bau": (
        "district_energy_grid_shape_quality_peak_daily_average_to_business_as_usual_ratio",
    ),
    "peak_all_time_ratio_to_bau": (
        "district_energy_grid_shape_quality_peak_all_time_average_to_business_as_usual_ratio",
    ),
    "ramping_ratio_to_bau": (
        "district_energy_grid_shape_quality_ramping_average_to_business_as_usual_ratio",
    ),
    "load_factor_penalty_ratio_to_bau": (
        "district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_business_as_usual_ratio",
    ),
    "solar_self_consumption_rate": (
        "district_solar_self_consumption_ratio_self_consumption_ratio",
    ),
    "emissions_ratio_to_bau": (
        "district_emissions_ratio_to_business_as_usual_total_ratio",
    ),
    "electrical_violation_kwh": (
        "district_electrical_service_phase_violations_energy_total_kwh",
    ),
    "electrical_requested_pressure_kwh": (
        "district_electrical_service_phase_requested_pressure_energy_total_kwh",
    ),
    "ev_min_acceptable_feasible_rate": (
        "district_ev_performance_departure_min_acceptable_feasible_ratio",
    ),
    "ev_within_tolerance_feasible_rate": (
        "district_ev_performance_departure_within_tolerance_feasible_ratio",
    ),
    "deferrable_service_level_rate": (
        "district_deferrable_appliance_service_service_level_ratio",
    ),
    "deferrable_completed_cycles_count": (
        "district_deferrable_appliance_service_completed_cycles_count",
    ),
    "deferrable_missed_cycles_count": (
        "district_deferrable_appliance_service_missed_cycles_count",
    ),
    "deferrable_unserved_energy_kwh": (
        "district_deferrable_appliance_service_unserved_energy_total_kwh",
    ),
    "battery_throughput_kwh": ("district_battery_total_throughput_kwh",),
    "v2g_export_kwh": ("district_ev_total_v2g_export_kwh",),
    "community_import_kwh": ("district_energy_grid_total_import_control_kwh",),
    "community_export_kwh": ("district_energy_grid_total_export_control_kwh",),
    "gini_benefit_ratio": ("district_equity_distribution_gini_benefit_ratio",),
}


DEFAULT_AGGREGATION: Mapping[str, str] = {
    "cost_eur": "sum",
    "electrical_violation_kwh": "sum",
    "electrical_requested_pressure_kwh": "sum",
    "battery_throughput_kwh": "sum",
    "v2g_export_kwh": "sum",
    "community_import_kwh": "sum",
    "community_export_kwh": "sum",
    "ev_min_acceptable_feasible_rate": "min",
    "ev_within_tolerance_feasible_rate": "min",
    "deferrable_service_level_rate": "min",
    "deferrable_completed_cycles_count": "sum",
    "deferrable_missed_cycles_count": "sum",
    "deferrable_unserved_energy_kwh": "sum",
}

# Settlement is a commercial estimate. Causal response/rebound are imported
# separately from matched counterfactual traces, never relabelled as these KPIs.
for _suffix in (
    "requested_total_kwh",
    "valid_requested_total_kwh",
    "delivered_total_kwh",
    "credited_total_kwh",
    "shortfall_total_kwh",
    "revenue_total_eur",
    "penalty_total_eur",
    "net_revenue_total_eur",
    "invalid_baseline_time_step_count",
):
    KPI_ROWS["dr_" + _suffix] = ("district_demand_response_" + _suffix,)
    DEFAULT_AGGREGATION["dr_" + _suffix] = "sum"
KPI_ROWS["dr_settlement_compliance_ratio"] = (
    "district_demand_response_compliance_ratio",
)
DEFAULT_AGGREGATION["dr_settlement_compliance_ratio"] = "min"

DIAGNOSTIC_METRICS = {
    "soc_bound_violations",
    "energy_balance_residual_kwh",
    "external_trip_unserved_kwh",
    "outside_charge_kwh",
    "native_unserved_kwh",
    "ev_departure_deficit_kwh",
    "health_exposed_asset_hours",
    "health_changed_dispatch_kwh",
    "recovery_hours",
    "recovery_censored",
    "terminal_state_l1_kwh",
    "dr_additional_signed_kwh",
    "dr_additional_credited_kwh",
    "dr_quarter_hour_compliance_ratio",
    "dr_baseline_mae_kw",
    "dr_reference_valid_fraction",
    "dr_reference_member_coverage",
    "dr_reference_bess_power_coverage",
    "dr_reference_charger_coverage",
    "dr_reference_uncertainty_kw",
    "dr_reference_interval_coverage",
    "dr_credit_without_response_kwh",
    "dr_rebound_2h_kwh",
    "dr_rebound_6h_kwh",
    "dr_rebound_24h_kwh",
    "dr_rebound_168h_kwh",
}
DR_REFERENCE_FRACTIONS = {
    "dr_reference_valid_fraction",
    "dr_reference_member_coverage",
    "dr_reference_bess_power_coverage",
    "dr_reference_charger_coverage",
    "dr_reference_interval_coverage",
}
for _metric in DIAGNOSTIC_METRICS:
    DEFAULT_AGGREGATION[_metric] = (
        "max"
        if _metric
        in {
            "energy_balance_residual_kwh",
            "recovery_hours",
            "recovery_censored",
            "terminal_state_l1_kwh",
            "dr_baseline_mae_kw",
            "dr_reference_uncertainty_kw",
        }
        else (
            "min"
            if _metric
            in {
                "dr_quarter_hour_compliance_ratio",
                "dr_reference_valid_fraction",
                "dr_reference_member_coverage",
                "dr_reference_bess_power_coverage",
                "dr_reference_charger_coverage",
                "dr_reference_interval_coverage",
            }
            else "sum"
        )
    )


def _load_diagnostics(path: str | Path, pairing_hash: str) -> tuple[dict, dict]:
    source = Path(path).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if (
        data.get("format") != "portugal_paired_diagnostics_v1"
        or data.get("pairing_sha256") != pairing_hash
    ):
        raise ValueError("Diagnostics must match the exact paired simulator surface")
    if not data.get("sources"):
        raise ValueError("Diagnostics require content-addressed source traces")
    for item in data["sources"]:
        trace = Path(item["path"])
        if not trace.is_absolute():
            trace = source.parent / trace
        if file_sha256(trace) != item["sha256"]:
            raise ValueError("Diagnostics source trace hash mismatch")
    metrics = {}
    for name, value in data.get("metrics", {}).items():
        if name not in DIAGNOSTIC_METRICS or not math.isfinite(float(value)):
            raise ValueError(f"Unsupported or non-finite diagnostic metric: {name}")
        if name in DR_REFERENCE_FRACTIONS and not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"DR reference coverage must be a fraction in [0, 1]: {name}"
            )
        metrics[name] = float(value)
    return metrics, {
        "path": str(source),
        "sha256": file_sha256(source),
        "sources": data["sources"],
    }


def _audited_dataset_inputs(schema_path: Any) -> Mapping[str, Any] | None:
    """Hash actual referenced inputs for current Portugal overlays only.

    Preserve historical fingerprint semantics for previously archived records.
    """
    if not schema_path or not Path(str(schema_path)).is_file():
        return None
    path = Path(str(schema_path)).resolve()
    schema = json.loads(path.read_text(encoding="utf-8"))
    design = schema.get("experimental_design", {})
    if design.get("version") != "portugal-evaluation-v2":
        return None
    root = Path(schema.get("root_directory") or ".")
    if not root.is_absolute():
        root = path.parent / root
    inputs = {}

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value.endswith((".parquet", ".csv")):
            source = Path(value)
            if not source.is_absolute():
                source = root / source
            if not source.is_file():
                raise FileNotFoundError(f"Audited dataset input is missing: {source}")
            inputs[value] = file_sha256(source)

    visit(schema)
    return {
        "profile": design["version"],
        "phase": design["phase"],
        "factors": design["factors"],
        "inputs_sha256": canonical_sha256(inputs),
        "input_file_count": len(inputs),
    }


def canonical_sha256(payload: Any) -> str:
    """Hash a JSON-compatible payload with stable ordering and encoding."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_identity(raw_path: Any) -> Mapping[str, Any] | None:
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path)).expanduser()
    return {
        "name": path.name,
        "sha256": file_sha256(path) if path.is_file() else None,
    }


def build_pairing_fingerprint(
    config: Mapping[str, Any],
    *,
    simulator_version: str | None = None,
) -> Mapping[str, Any]:
    """Describe only the simulator surface that must match pairwise.

    Algorithm, checkpoint, runtime paths and neural seed are intentionally
    absent.  The Simulator seed is included because it controls stochastic
    exogenous details such as EV drift.
    """

    simulator = dict(config.get("simulator") or {})
    export = dict(simulator.get("export") or {})
    payload = {
        "dataset_name": simulator.get("dataset_name"),
        "dataset_schema": _optional_file_identity(simulator.get("dataset_path")),
        "electrical_service_overrides": _optional_file_identity(
            simulator.get("electrical_service_overrides_path")
        ),
        "building_ids": simulator.get("building_ids"),
        "central_agent": simulator.get("central_agent", False),
        "interface": simulator.get("interface", "flat"),
        "topology_mode": simulator.get("topology_mode", "static"),
        "simulation_start_time_step": simulator.get("simulation_start_time_step"),
        "simulation_end_time_step": simulator.get("simulation_end_time_step"),
        "episode_time_steps": simulator.get("episode_time_steps"),
        "terminal_observation_padding": simulator.get(
            "terminal_observation_padding", False
        ),
        "repeat_episode_scenario": simulator.get("repeat_episode_scenario", False),
        "simulator_random_seed": simulator.get("random_seed"),
        "reward_function": simulator.get("reward_function"),
        "reward_function_kwargs": simulator.get("reward_function_kwargs") or {},
        "community_market": simulator.get("community_market"),
        "include_business_as_usual": export.get("include_business_as_usual", True),
        "simulator_version": simulator_version,
    }
    audited = _audited_dataset_inputs(simulator.get("dataset_path"))
    if audited is not None:
        payload["audited_dataset_inputs"] = audited
        spec = importlib.util.find_spec("citylearn")
        if spec is None or not spec.origin:
            raise RuntimeError(
                "Portugal evaluation requires an inspectable Simulator runtime"
            )
        runtime_root = Path(spec.origin).parent
        runtime_sources = {
            str(p.relative_to(runtime_root)): file_sha256(p)
            for p in sorted(runtime_root.rglob("*.py"))
        }
        payload["simulator_source_sha256"] = canonical_sha256(runtime_sources)
    return {
        "format": "paired_simulator_surface_v1",
        "sha256": canonical_sha256(payload),
        "payload": payload,
    }


def extract_scorecard(exported_kpis_path: str | Path) -> Mapping[str, float]:
    """Extract the canonical district scorecard from a Simulator KPI CSV."""

    rows: dict[str, Mapping[str, str]] = {}
    with Path(exported_kpis_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("KPI") or "").strip()
            if name:
                rows[name] = row

    metrics: dict[str, float] = {}
    for metric, candidate_rows in KPI_ROWS.items():
        for row_name in candidate_rows:
            row = rows.get(row_name)
            if row is None:
                continue
            value = row.get("District")
            if value in (None, ""):
                continue
            try:
                number = float(value)
                if not math.isfinite(number):
                    continue
                metrics[metric] = number
            except (TypeError, ValueError):
                continue
            break
    return metrics


def build_evaluation_record(
    *,
    candidate_id: str,
    role: str,
    config: Mapping[str, Any],
    exported_kpis_path: str | Path,
    checkpoint_path: str | Path | None = None,
    simulator_version: str | None = None,
    pairing_fingerprint: Mapping[str, Any] | None = None,
    diagnostics_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Build one immutable local evidence record for a deterministic replay."""

    protocol = dict(config.get("experiment_protocol") or {})
    if protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"experiment_protocol.version must be {PROTOCOL_VERSION!r}")
    if protocol.get("phase") not in {"development", "confirmation"}:
        raise ValueError("Evaluation records require development or confirmation phase")
    if role not in {"candidate", "reference"}:
        raise ValueError("role must be 'candidate' or 'reference'")
    if str(protocol.get("candidate_id") or "") != str(candidate_id):
        raise ValueError("candidate_id must match experiment_protocol.candidate_id")
    if str(protocol.get("role") or "") != role:
        raise ValueError("role must match experiment_protocol.role")

    checkpoint: Mapping[str, Any] | None = None
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint is not a file: {resolved}")
        checkpoint = {
            "path": str(resolved),
            "name": resolved.name,
            "sha256": file_sha256(resolved),
        }
    if role == "candidate" and checkpoint is None:
        raise ValueError("Candidate evaluation records require a checkpoint")

    kpis = Path(exported_kpis_path).expanduser().resolve()
    if not kpis.is_file():
        raise FileNotFoundError(f"KPI export is not a file: {kpis}")
    pairing = (
        dict(pairing_fingerprint)
        if pairing_fingerprint is not None
        else dict(
            build_pairing_fingerprint(
                config,
                simulator_version=simulator_version,
            )
        )
    )
    if pairing.get("format") != "paired_simulator_surface_v1":
        raise ValueError("Unsupported pairing fingerprint format")
    if canonical_sha256(pairing.get("payload")) != pairing.get("sha256"):
        raise ValueError("Pairing fingerprint payload/hash mismatch")

    payload: dict[str, Any] = {
        "format": EVALUATION_RECORD_VERSION,
        "protocol_id": protocol.get("protocol_id"),
        "phase": protocol.get("phase"),
        "data_split": protocol.get("data_split"),
        "window_id": protocol.get("window_id"),
        "selection_rules_sha256": protocol.get("selection_rules_sha256"),
        "selection_record_sha256": protocol.get("selection_record_sha256"),
        "selected_checkpoint_sha256": protocol.get("selected_checkpoint_sha256"),
        "candidate_id": str(candidate_id),
        "paired_reference_id": protocol.get("paired_reference_id"),
        "role": role,
        "neural_seed": (config.get("training") or {}).get("seed"),
        "pairing": pairing,
        "checkpoint": checkpoint,
        "kpis": {
            "path": str(kpis),
            "sha256": file_sha256(kpis),
        },
        "metrics": dict(extract_scorecard(kpis)),
    }
    if diagnostics_path is not None:
        diagnostic_metrics, provenance = _load_diagnostics(
            diagnostics_path, pairing["sha256"]
        )
        payload["metrics"].update(diagnostic_metrics)
        payload["diagnostics"] = provenance
    payload["record_sha256"] = canonical_sha256(payload)
    return payload


def _aggregate(values: Sequence[float], operation: str) -> float:
    if operation == "sum":
        return float(sum(values))
    if operation == "min":
        return float(min(values))
    if operation == "max":
        return float(max(values))
    if operation == "mean":
        return float(sum(values) / len(values))
    raise ValueError(f"Unsupported aggregation operation: {operation!r}")


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    aggregation: Mapping[str, str] | None = None,
) -> Mapping[str, float]:
    if not records:
        raise ValueError("Cannot aggregate an empty evaluation record set")
    configured = {**DEFAULT_AGGREGATION, **dict(aggregation or {})}
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for metric, value in (record.get("metrics") or {}).items():
            values[str(metric)].append(float(value))
    return {
        metric: _aggregate(metric_values, configured.get(metric, "mean"))
        for metric, metric_values in sorted(values.items())
        if len(metric_values) == len(records)
    }


def _gate_reasons(
    metrics: Mapping[str, float],
    rules: Mapping[str, Any],
    reference: Mapping[str, float],
) -> list[str]:
    reasons: list[str] = []
    for metric in rules.get("required_metrics", []):
        if metric not in metrics or not math.isfinite(float(metrics[metric])):
            reasons.append(f"missing_required:{metric}")
        if metric not in reference or not math.isfinite(float(reference[metric])):
            reasons.append(f"missing_reference:{metric}")
    for metric, bounds in (rules.get("hard_gates") or {}).items():
        if metric not in metrics:
            # A service gate is not applicable when the paired reference also
            # has no such asset/event metric. Missing only on the candidate is
            # still a hard evidence failure.
            if metric in reference:
                reasons.append(f"missing:{metric}")
            continue
        value = float(metrics[metric])
        if "min" in bounds and value < float(bounds["min"]):
            reasons.append(f"{metric}<{bounds['min']}")
        if "max" in bounds and value > float(bounds["max"]):
            reasons.append(f"{metric}>{bounds['max']}")
    return reasons


def _guardrail_reasons(
    metrics: Mapping[str, float],
    reference: Mapping[str, float],
    rules: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for metric, guardrail in (rules.get("reference_guardrails") or {}).items():
        if metric not in metrics or metric not in reference:
            reasons.append(f"missing_guardrail:{metric}")
            continue
        value = float(metrics[metric])
        baseline = float(reference[metric])
        if "max_relative_increase" in guardrail:
            limit = baseline + abs(baseline) * float(guardrail["max_relative_increase"])
            if value > limit:
                reasons.append(f"{metric}>{limit}")
        if "max_absolute_increase" in guardrail:
            limit = baseline + float(guardrail["max_absolute_increase"])
            if value > limit:
                reasons.append(f"{metric}>{limit}")
        if "max_absolute_decrease" in guardrail:
            limit = baseline - float(guardrail["max_absolute_decrease"])
            if value < limit:
                reasons.append(f"{metric}<{limit}")
    return reasons


def _selection_key(
    metrics: Mapping[str, float], rules: Mapping[str, Any]
) -> tuple[float, ...]:
    promotion = dict(rules.get("promotion") or {})
    primary = str(promotion.get("metric", "cost_eur"))
    direction = str(promotion.get("direction", "minimize"))
    if primary not in metrics:
        raise ValueError(f"Candidate is missing primary metric {primary!r}")
    key = [float(metrics[primary]) * (1.0 if direction == "minimize" else -1.0)]
    for item in rules.get("tie_breakers") or []:
        metric = str(item["metric"])
        if metric not in metrics:
            raise ValueError(f"Candidate is missing tie-break metric {metric!r}")
        sign = 1.0 if item.get("direction", "minimize") == "minimize" else -1.0
        key.append(float(metrics[metric]) * sign)
    return tuple(key)


def select_checkpoint(
    *,
    references: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Select one checkpoint without consulting confirmation results."""

    if rules.get("version") != "ti_marl_selection_rules_v1":
        raise ValueError("Unsupported checkpoint-selection rules version")
    if not references or not candidates:
        raise ValueError("Selection requires reference and candidate records")
    if any(
        record.get("phase") != "development" for record in [*references, *candidates]
    ):
        raise ValueError("Checkpoint selection may consume development records only")
    for record in [*references, *candidates]:
        if record.get("format") != EVALUATION_RECORD_VERSION:
            raise ValueError("Unsupported evaluation record format")
        recorded_hash = record.get("record_sha256")
        unhashed = dict(record)
        unhashed.pop("record_sha256", None)
        if not recorded_hash or canonical_sha256(unhashed) != recorded_hash:
            raise ValueError("Evaluation record payload/hash mismatch")
    if any(record.get("role") != "reference" for record in references):
        raise ValueError("Reference inputs must have role='reference'")
    if any(record.get("role") != "candidate" for record in candidates):
        raise ValueError("Candidate inputs must have role='candidate'")
    rules_hash = canonical_sha256(rules)
    configured_rule_hashes = {
        record.get("selection_rules_sha256") for record in [*references, *candidates]
    }
    if configured_rule_hashes != {rules_hash}:
        raise ValueError("Evaluation records do not match the supplied frozen rules")

    protocol_ids = {record.get("protocol_id") for record in [*references, *candidates]}
    if len(protocol_ids) != 1:
        raise ValueError("All records must use the same protocol_id")

    reference_by_pairing: dict[str, Mapping[str, Any]] = {}
    for record in references:
        pairing_hash = str((record.get("pairing") or {}).get("sha256") or "")
        if not pairing_hash or pairing_hash in reference_by_pairing:
            raise ValueError(
                "References must contain one unique record per paired surface"
            )
        reference_by_pairing[pairing_hash] = record

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in candidates:
        candidate_id = str(record.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("Candidate record is missing candidate_id")
        grouped[candidate_id].append(record)
    expected_surfaces = set(reference_by_pairing)
    aggregation = dict(rules.get("aggregation") or {})
    reference_metrics = aggregate_records(
        list(reference_by_pairing.values()), aggregation=aggregation
    )

    evaluated: list[dict[str, Any]] = []
    for candidate_id, records in sorted(grouped.items()):
        surfaces = {
            str((record.get("pairing") or {}).get("sha256") or "") for record in records
        }
        if surfaces != expected_surfaces or len(records) != len(expected_surfaces):
            raise ValueError(
                f"Candidate {candidate_id!r} does not cover the exact paired development surfaces"
            )
        checkpoints = {
            str((record.get("checkpoint") or {}).get("sha256") or "")
            for record in records
        }
        if len(checkpoints) != 1 or "" in checkpoints:
            raise ValueError(
                f"Candidate {candidate_id!r} must replay one identical checkpoint on every surface"
            )
        metrics = aggregate_records(records, aggregation=aggregation)
        reasons = _gate_reasons(metrics, rules, reference_metrics)
        reasons.extend(_guardrail_reasons(metrics, reference_metrics, rules))

        promotion = dict(rules.get("promotion") or {})
        primary = str(promotion.get("metric", "cost_eur"))
        if primary not in metrics or primary not in reference_metrics:
            reasons.append(f"missing_promotion:{primary}")
        else:
            improvement = max(
                float(promotion.get("minimum_improvement", 0.0)),
                abs(float(reference_metrics[primary]))
                * float(promotion.get("minimum_relative_improvement", 0.0)),
            )
            direction = str(promotion.get("direction", "minimize"))
            if (
                direction == "minimize"
                and metrics[primary] > reference_metrics[primary] - improvement
            ):
                reasons.append(f"no_required_improvement:{primary}")
            if (
                direction == "maximize"
                and metrics[primary] < reference_metrics[primary] + improvement
            ):
                reasons.append(f"no_required_improvement:{primary}")

        evaluated.append(
            {
                "candidate_id": candidate_id,
                "checkpoint": records[0]["checkpoint"],
                "metrics": metrics,
                "record_sha256s": sorted(
                    str(record["record_sha256"]) for record in records
                ),
                "accepted": not reasons,
                "rejection_reasons": reasons,
            }
        )

    accepted = [item for item in evaluated if item["accepted"]]
    accepted.sort(
        key=lambda item: (_selection_key(item["metrics"], rules), item["candidate_id"])
    )
    selected = accepted[0] if accepted else None
    payload: dict[str, Any] = {
        "format": SELECTION_RECORD_VERSION,
        "protocol_id": next(iter(protocol_ids)),
        "status": "selected" if selected is not None else "no_promotion",
        "rules_sha256": rules_hash,
        "paired_surface_sha256s": sorted(expected_surfaces),
        "reference_metrics": reference_metrics,
        "evaluated_candidates": evaluated,
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
    }
    payload["selection_sha256"] = canonical_sha256(payload)
    return payload


def build_confirmation_report(
    *,
    references: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Aggregate one frozen candidate over exact paired confirmation surfaces.

    This deliberately does not select a checkpoint. It verifies that every
    record points back to the same pre-existing selection and selected
    checkpoint, then reports whether the frozen development gates replicate on
    the reserved confirmation windows.
    """

    records = [*references, *candidates]
    if rules.get("version") != "ti_marl_selection_rules_v1":
        raise ValueError("Unsupported checkpoint-selection rules version")
    if not references or not candidates:
        raise ValueError("Confirmation requires reference and candidate records")
    if any(record.get("phase") != "confirmation" for record in records):
        raise ValueError("Confirmation reports may consume confirmation records only")
    for record in records:
        if record.get("format") != EVALUATION_RECORD_VERSION:
            raise ValueError("Unsupported evaluation record format")
        recorded_hash = record.get("record_sha256")
        unhashed = dict(record)
        unhashed.pop("record_sha256", None)
        if not recorded_hash or canonical_sha256(unhashed) != recorded_hash:
            raise ValueError("Evaluation record payload/hash mismatch")
    if any(record.get("role") != "reference" for record in references):
        raise ValueError("Reference inputs must have role='reference'")
    if any(record.get("role") != "candidate" for record in candidates):
        raise ValueError("Candidate inputs must have role='candidate'")

    rules_hash = canonical_sha256(rules)
    if {record.get("selection_rules_sha256") for record in records} != {rules_hash}:
        raise ValueError("Confirmation records do not match the frozen rules")
    protocol_ids = {record.get("protocol_id") for record in records}
    if len(protocol_ids) != 1:
        raise ValueError("All records must use the same protocol_id")
    selection_hashes = {
        str(record.get("selection_record_sha256") or "") for record in records
    }
    if len(selection_hashes) != 1 or "" in selection_hashes:
        raise ValueError("Confirmation records must share one selection record hash")

    if selection.get("format") != SELECTION_RECORD_VERSION:
        raise ValueError("Unsupported checkpoint-selection record format")
    selection_hash = str(selection.get("selection_sha256") or "")
    unhashed_selection = dict(selection)
    unhashed_selection.pop("selection_sha256", None)
    if not selection_hash or canonical_sha256(unhashed_selection) != selection_hash:
        raise ValueError("Checkpoint-selection record payload/hash mismatch")
    if selection_hashes != {selection_hash}:
        raise ValueError(
            "Confirmation does not reference the supplied selection record"
        )
    if selection.get("status") != "selected":
        raise ValueError("Confirmation requires a successful checkpoint selection")
    if selection.get("rules_sha256") != rules_hash:
        raise ValueError("Checkpoint selection does not match the frozen rules")

    candidate_ids = {str(record.get("candidate_id") or "") for record in candidates}
    if len(candidate_ids) != 1 or "" in candidate_ids:
        raise ValueError("Confirmation must report exactly one frozen candidate")
    if selection.get("selected_candidate_id") != next(iter(candidate_ids)):
        raise ValueError("Confirmation candidate differs from the selected candidate")
    configured_checkpoint_hashes = {
        str(record.get("selected_checkpoint_sha256") or "") for record in candidates
    }
    replayed_checkpoint_hashes = {
        str((record.get("checkpoint") or {}).get("sha256") or "")
        for record in candidates
    }
    if (
        len(configured_checkpoint_hashes) != 1
        or "" in configured_checkpoint_hashes
        or configured_checkpoint_hashes != replayed_checkpoint_hashes
    ):
        raise ValueError(
            "Confirmation candidates must replay the one selected checkpoint"
        )
    selected_checkpoint_hash = str(
        (selection.get("selected_checkpoint") or {}).get("sha256") or ""
    )
    if configured_checkpoint_hashes != {selected_checkpoint_hash}:
        raise ValueError("Confirmation checkpoint differs from the selected checkpoint")

    reference_by_pairing: dict[str, Mapping[str, Any]] = {}
    for record in references:
        pairing_hash = str((record.get("pairing") or {}).get("sha256") or "")
        if not pairing_hash or pairing_hash in reference_by_pairing:
            raise ValueError(
                "References must contain one unique record per paired surface"
            )
        reference_by_pairing[pairing_hash] = record
    candidate_by_pairing: dict[str, Mapping[str, Any]] = {}
    for record in candidates:
        pairing_hash = str((record.get("pairing") or {}).get("sha256") or "")
        if not pairing_hash or pairing_hash in candidate_by_pairing:
            raise ValueError(
                "Candidates must contain one unique record per paired surface"
            )
        candidate_by_pairing[pairing_hash] = record
    if set(candidate_by_pairing) != set(reference_by_pairing):
        raise ValueError(
            "Candidate does not cover the exact paired confirmation surfaces"
        )
    for pairing_hash, candidate in candidate_by_pairing.items():
        reference = reference_by_pairing[pairing_hash]
        if candidate.get("paired_reference_id") != reference.get("candidate_id"):
            raise ValueError(
                "Candidate paired_reference_id does not match its reference"
            )

    aggregation = dict(rules.get("aggregation") or {})
    reference_metrics = aggregate_records(references, aggregation=aggregation)
    candidate_metrics = aggregate_records(candidates, aggregation=aggregation)
    reasons = _gate_reasons(candidate_metrics, rules, reference_metrics)
    reasons.extend(_guardrail_reasons(candidate_metrics, reference_metrics, rules))

    promotion = dict(rules.get("promotion") or {})
    primary = str(promotion.get("metric", "cost_eur"))
    if primary not in candidate_metrics or primary not in reference_metrics:
        reasons.append(f"missing_promotion:{primary}")
    else:
        improvement = max(
            float(promotion.get("minimum_improvement", 0.0)),
            abs(float(reference_metrics[primary]))
            * float(promotion.get("minimum_relative_improvement", 0.0)),
        )
        direction = str(promotion.get("direction", "minimize"))
        if (
            direction == "minimize"
            and candidate_metrics[primary] > reference_metrics[primary] - improvement
        ):
            reasons.append(f"no_required_improvement:{primary}")
        if (
            direction == "maximize"
            and candidate_metrics[primary] < reference_metrics[primary] + improvement
        ):
            reasons.append(f"no_required_improvement:{primary}")

    common_metrics = sorted(set(reference_metrics) & set(candidate_metrics))
    deltas = {
        metric: float(candidate_metrics[metric]) - float(reference_metrics[metric])
        for metric in common_metrics
    }
    relative_deltas = {
        metric: deltas[metric] / abs(float(reference_metrics[metric]))
        for metric in common_metrics
        if float(reference_metrics[metric]) != 0.0
    }
    paired_windows = []
    for pairing_hash in sorted(reference_by_pairing):
        reference = reference_by_pairing[pairing_hash]
        candidate = candidate_by_pairing[pairing_hash]
        shared = sorted(
            set(reference.get("metrics") or {}) & set(candidate.get("metrics") or {})
        )
        paired_windows.append(
            {
                "window_id": candidate.get("window_id"),
                "pairing_sha256": pairing_hash,
                "reference_record_sha256": reference.get("record_sha256"),
                "candidate_record_sha256": candidate.get("record_sha256"),
                "metric_deltas": {
                    metric: float(candidate["metrics"][metric])
                    - float(reference["metrics"][metric])
                    for metric in shared
                },
                "metric_relative_deltas": {
                    metric: (
                        float(candidate["metrics"][metric])
                        - float(reference["metrics"][metric])
                    )
                    / abs(float(reference["metrics"][metric]))
                    for metric in shared
                    if float(reference["metrics"][metric]) != 0.0
                },
            }
        )

    payload: dict[str, Any] = {
        "format": CONFIRMATION_REPORT_VERSION,
        "protocol_id": next(iter(protocol_ids)),
        "status": "confirmed" if not reasons else "not_confirmed",
        "candidate_id": next(iter(candidate_ids)),
        "selection_record_sha256": next(iter(selection_hashes)),
        "selected_checkpoint_sha256": next(iter(configured_checkpoint_hashes)),
        "rules_sha256": rules_hash,
        "paired_surface_sha256s": sorted(reference_by_pairing),
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": deltas,
        "metric_relative_deltas": relative_deltas,
        "paired_windows": paired_windows,
        "rejection_reasons": reasons,
    }
    payload["report_sha256"] = canonical_sha256(payload)
    return payload


def verify_selected_checkpoint(
    selection: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> bool:
    selected = selection.get("selected_checkpoint")
    if not isinstance(selected, Mapping) or not selected.get("sha256"):
        return False
    return file_sha256(checkpoint_path) == str(selected["sha256"])


def load_json_records(paths: Iterable[str | Path]) -> list[Mapping[str, Any]]:
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
