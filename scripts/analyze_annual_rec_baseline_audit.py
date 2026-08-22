#!/usr/bin/env python3
"""Build an interpretation report for the validated annual BAU/RBC campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_CONTRASTS = (
    ("core30_safety_vs_nominal", "core30_nominal", "core30_safety", "electrical limits"),
    ("core30_health_vs_nominal", "core30_nominal", "core30_health", "health faults"),
    (
        "core30_combined_vs_dynamic",
        "core30_dynamic",
        "core30_combined",
        "limits, faults and demand response on the dynamic topology",
    ),
    (
        "premium_allin_vs_clean",
        "premium_clean",
        "premium_allin",
        "faults and grid outages on the matched premium topology",
    ),
)


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio_delta_percent(candidate: Any, reference: Any) -> float | None:
    if candidate is None or reference in (None, 0, 0.0):
        return None
    return 100.0 * (float(candidate) / float(reference) - 1.0)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/annual_rec_baseline_audit_v1_9_asset_count_attested_full_summary"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary_dir / "campaign_summary.json"
    validation_path = args.summary_dir / "aggregate_validation.json"
    summary = _read_json(summary_path)
    validation = _read_json(validation_path)
    if validation.get("status") != "pass":
        raise RuntimeError("Aggregate campaign validation must pass before analysis.")

    runs = list(summary.get("runs", []))
    by_variant: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in runs:
        by_variant.setdefault(str(row["variant"]), {})[str(row["policy"])] = row

    comparisons = []
    integrity_findings = []
    controller_watches = []
    for variant, policies in sorted(by_variant.items()):
        bau = policies["bau"]
        smart = policies["rbc_smart"]
        b = bau.get("scorecard", {}) or {}
        s = smart.get("scorecard", {}) or {}
        ev_audit = smart.get("ev_departure_obligation_audit", {}) or {}
        def_audit = smart.get("deferrable_service_obligation_audit", {}) or {}
        comparison = {
            "variant": variant,
            "cost": {
                "bau_eur": b.get("cost_eur"),
                "rbc_smart_eur": s.get("cost_eur"),
                "rbc_delta_eur": (
                    None
                    if b.get("cost_eur") is None or s.get("cost_eur") is None
                    else float(s["cost_eur"]) - float(b["cost_eur"])
                ),
                "rbc_delta_percent": _ratio_delta_percent(
                    s.get("cost_eur"), b.get("cost_eur")
                ),
            },
            "grid_shape_rbc_delta_percent": {
                "daily_peak": _ratio_delta_percent(
                    s.get("peak_daily_ratio_to_passive_baseline"),
                    b.get("peak_daily_ratio_to_passive_baseline"),
                ),
                "all_time_peak": _ratio_delta_percent(
                    s.get("peak_all_time_ratio_to_passive_baseline"),
                    b.get("peak_all_time_ratio_to_passive_baseline"),
                ),
                "ramping": _ratio_delta_percent(
                    s.get("ramping_ratio_to_passive_baseline"),
                    b.get("ramping_ratio_to_passive_baseline"),
                ),
            },
            "settlement": {
                "bau_savings_eur": b.get("community_settlement_savings_eur"),
                "rbc_smart_savings_eur": s.get("community_settlement_savings_eur"),
                "bau_local_trade_kwh": b.get("community_local_traded_kwh"),
                "rbc_smart_local_trade_kwh": s.get("community_local_traded_kwh"),
            },
            "ev": {
                "expected_departures": ev_audit.get("expected_departure_events"),
                "topology_censored": (
                    int(ev_audit.get("censored_member_inactive", 0))
                    + int(ev_audit.get("censored_charger_inactive", 0))
                ),
                "bau_departures": b.get("ev_departure_events"),
                "rbc_smart_departures": s.get("ev_departure_events"),
                "bau_exact_target_rate": b.get("ev_exact_target_feasible_rate"),
                "rbc_smart_exact_target_rate": s.get("ev_exact_target_feasible_rate"),
                "bau_minimum_rate": b.get("ev_min_acceptable_feasible_rate"),
                "rbc_smart_minimum_rate": s.get("ev_min_acceptable_feasible_rate"),
                "bau_within_tolerance_rate": b.get("ev_within_tolerance_feasible_rate"),
                "rbc_smart_within_tolerance_rate": s.get("ev_within_tolerance_feasible_rate"),
                "bau_soc_deficit_mean": b.get("ev_departure_soc_deficit_mean"),
                "rbc_smart_soc_deficit_mean": s.get("ev_departure_soc_deficit_mean"),
                "bau_energy_accounting_shortfall_kwh": b.get(
                    "ev_energy_accounting_shortfall_kwh"
                ),
                "rbc_smart_energy_accounting_shortfall_kwh": s.get(
                    "ev_energy_accounting_shortfall_kwh"
                ),
            },
            "deferrable": {
                "runtime_count_min": def_audit.get(
                    "expected_runtime_service_count_min"
                ),
                "runtime_count_max": def_audit.get(
                    "expected_runtime_service_count_max"
                ),
                "topology_censored": def_audit.get(
                    "topology_censored_no_start_opportunity"
                ),
                "topology_interrupted": def_audit.get(
                    "topology_interrupted_cycles"
                ),
                "bau_completed": b.get("deferrable_completed_cycles_count"),
                "bau_missed": b.get("deferrable_missed_cycles_count"),
                "rbc_smart_completed": s.get("deferrable_completed_cycles_count"),
                "rbc_smart_missed": s.get("deferrable_missed_cycles_count"),
                "bau_service_rate": b.get("deferrable_service_level_rate"),
                "rbc_smart_service_rate": s.get("deferrable_service_level_rate"),
                "bau_start_delay_hours": b.get(
                    "deferrable_average_start_delay_hours"
                ),
                "rbc_smart_start_delay_hours": s.get(
                    "deferrable_average_start_delay_hours"
                ),
            },
            "safety": {
                "bau_requested_pressure_kwh": b.get(
                    "electrical_requested_pressure_kwh"
                ),
                "rbc_smart_requested_pressure_kwh": s.get(
                    "electrical_requested_pressure_kwh"
                ),
                "bau_residual_violation_kwh": b.get("electrical_violation_kwh"),
                "rbc_smart_residual_violation_kwh": s.get(
                    "electrical_violation_kwh"
                ),
                "bau_residual_violation_events": b.get("electrical_violation_events"),
                "rbc_smart_residual_violation_events": s.get(
                    "electrical_violation_events"
                ),
                "bau_requested_pressure_events": b.get(
                    "electrical_requested_pressure_events"
                ),
                "rbc_smart_requested_pressure_events": s.get(
                    "electrical_requested_pressure_events"
                ),
            },
            "battery": {
                "bau_throughput_kwh": b.get("battery_throughput_kwh"),
                "rbc_smart_throughput_kwh": s.get("battery_throughput_kwh"),
                "bau_capacity_fade_ratio": b.get("battery_capacity_fade"),
                "rbc_smart_capacity_fade_ratio": s.get("battery_capacity_fade"),
            },
            "demand_response": {
                "bau_events": b.get("demand_response_events"),
                "rbc_smart_events": s.get("demand_response_events"),
                "bau_compliance_rate": b.get("demand_response_compliance_rate"),
                "rbc_smart_compliance_rate": s.get(
                    "demand_response_compliance_rate"
                ),
                "bau_shortfall_kwh": b.get("demand_response_shortfall_kwh"),
                "rbc_smart_shortfall_kwh": s.get(
                    "demand_response_shortfall_kwh"
                ),
            },
            "robustness": {
                key: s.get(key)
                for key in (
                    "robustness_events",
                    "robustness_observation_corruptions",
                    "robustness_forecast_corruptions",
                    "robustness_action_corruptions",
                    "robustness_asset_unavailable_steps",
                    "outage_unserved_normalized",
                )
            },
            "pairing_fingerprint_match": (
                bau.get("pairing_fingerprint_sha256")
                == smart.get("pairing_fingerprint_sha256")
            ),
            "dataset_integrity_match": (
                bau.get("dataset_integrity_sha256")
                == smart.get("dataset_integrity_sha256")
                == bau.get("current_dataset_integrity_sha256")
                == smart.get("current_dataset_integrity_sha256")
            ),
        }
        comparisons.append(comparison)

        if not comparison["pairing_fingerprint_match"]:
            integrity_findings.append(f"{variant}: pairing fingerprint mismatch")
        if not comparison["dataset_integrity_match"]:
            integrity_findings.append(f"{variant}: dataset-integrity mismatch")
        for policy_name, scorecard in (("BAU", b), ("RBC Smart", s)):
            if float(scorecard.get("ev_energy_accounting_shortfall_kwh", 0.0) or 0.0) > 1.0e-5:
                integrity_findings.append(
                    f"{variant}/{policy_name}: EV energy-accounting shortfall"
                )
            if float(scorecard.get("electrical_violation_kwh", 0.0) or 0.0) > 1.0e-6:
                integrity_findings.append(
                    f"{variant}/{policy_name}: post-projection electrical "
                    "service-limit violation"
                )
            if float(scorecard.get("electrical_requested_pressure_kwh", 0.0) or 0.0) > 1.0e-6:
                controller_watches.append(
                    f"{variant}/{policy_name}: requested action activated "
                    "electrical constraint projection"
                )
        cost_delta = comparison["cost"]["rbc_delta_percent"]
        if cost_delta is not None and cost_delta > 0.0:
            controller_watches.append(
                f"{variant}: RBC Smart cost exceeds the matched BAU by "
                f"{cost_delta:.2f}%"
            )
        ramp_delta = comparison["grid_shape_rbc_delta_percent"]["ramping"]
        if ramp_delta is not None and ramp_delta > 0.0:
            controller_watches.append(
                f"{variant}: RBC Smart ramping exceeds the matched BAU by "
                f"{ramp_delta:.2f}%"
            )

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_contract_version": "2023-q15-v1.9",
        "validated_runs": len(runs),
        "integrity_status": "pass" if not integrity_findings else "fail",
        "integrity_findings": integrity_findings,
        "controller_outcome_watches": controller_watches,
        "comparisons": comparisons,
        "scenario_contrasts": [],
        "interpretation": (
            "Integrity findings concern dataset/runtime evidence. Controller "
            "watches are measured baseline outcomes and do not invalidate the "
            "dataset; they identify where BAU or RBC Smart is not sufficient."
        ),
    }
    for contrast_id, reference_variant, candidate_variant, isolated_dimension in SCENARIO_CONTRASTS:
        for policy in ("bau", "rbc_smart"):
            reference = by_variant[reference_variant][policy].get("scorecard", {}) or {}
            candidate = by_variant[candidate_variant][policy].get("scorecard", {}) or {}
            output["scenario_contrasts"].append(
                {
                    "contrast_id": contrast_id,
                    "policy": policy,
                    "reference_variant": reference_variant,
                    "candidate_variant": candidate_variant,
                    "isolated_dimension": isolated_dimension,
                    "cost_delta_percent": _ratio_delta_percent(
                        candidate.get("cost_eur"), reference.get("cost_eur")
                    ),
                    "daily_peak_delta_percent": _ratio_delta_percent(
                        candidate.get("peak_daily_ratio_to_passive_baseline"),
                        reference.get("peak_daily_ratio_to_passive_baseline"),
                    ),
                    "ramping_delta_percent": _ratio_delta_percent(
                        candidate.get("ramping_ratio_to_passive_baseline"),
                        reference.get("ramping_ratio_to_passive_baseline"),
                    ),
                    "ev_minimum_service_delta": (
                        None
                        if candidate.get("ev_min_acceptable_feasible_rate") is None
                        or reference.get("ev_min_acceptable_feasible_rate") is None
                        else float(candidate["ev_min_acceptable_feasible_rate"])
                        - float(reference["ev_min_acceptable_feasible_rate"])
                    ),
                    "deferrable_service_delta": (
                        None
                        if candidate.get("deferrable_service_level_rate") is None
                        or reference.get("deferrable_service_level_rate") is None
                        else float(candidate["deferrable_service_level_rate"])
                        - float(reference["deferrable_service_level_rate"])
                    ),
                    "electrical_pressure_delta_kwh": (
                        float(candidate.get("electrical_requested_pressure_kwh", 0.0) or 0.0)
                        - float(reference.get("electrical_requested_pressure_kwh", 0.0) or 0.0)
                    ),
                    "electrical_residual_violation_delta_kwh": (
                        float(candidate.get("electrical_violation_kwh", 0.0) or 0.0)
                        - float(reference.get("electrical_violation_kwh", 0.0) or 0.0)
                    ),
                    "robustness_events": candidate.get("robustness_events"),
                    "demand_response_compliance_rate": candidate.get(
                        "demand_response_compliance_rate"
                    ),
                    "outage_unserved_normalized": candidate.get(
                        "outage_unserved_normalized"
                    ),
                }
            )
    (args.summary_dir / "annual_analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Annual REC BAU/RBC Smart analysis",
        "",
        f"Integrity status: **{output['integrity_status']}**; validated runs: **{len(runs)}**.",
        "",
        "| Variant | RBC cost vs BAU | Daily peak | All-time peak | Ramping | EV min/tol | Deferrable service | BAU/RBC requested pressure (kWh) | BAU/RBC residual violation (kWh) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        ev = item["ev"]
        de = item["deferrable"]
        safety = item["safety"]
        shape = item["grid_shape_rbc_delta_percent"]
        lines.append(
            "| {variant} | {cost}% | {daily}% | {peak}% | {ramp}% | {ev_min}/{ev_tol} | {bau_def}/{rbc_def} | {bau_p}/{rbc_p} | {bau_v}/{rbc_v} |".format(
                variant=item["variant"],
                cost=_fmt(item["cost"]["rbc_delta_percent"], 2),
                daily=_fmt(shape["daily_peak"], 2),
                peak=_fmt(shape["all_time_peak"], 2),
                ramp=_fmt(shape["ramping"], 2),
                ev_min=_fmt(ev["rbc_smart_minimum_rate"], 4),
                ev_tol=_fmt(ev["rbc_smart_within_tolerance_rate"], 4),
                bau_def=_fmt(de["bau_service_rate"], 4),
                rbc_def=_fmt(de["rbc_smart_service_rate"], 4),
                bau_p=_fmt(safety["bau_requested_pressure_kwh"], 3),
                rbc_p=_fmt(safety["rbc_smart_requested_pressure_kwh"], 3),
                bau_v=_fmt(safety["bau_residual_violation_kwh"], 6),
                rbc_v=_fmt(safety["rbc_smart_residual_violation_kwh"], 6),
            )
        )
    lines.extend(
        [
            "",
            "Negative percentages indicate a reduction by RBC Smart relative to the matched BAU.",
            "Requested pressure is measured before constraint projection; residual violation is measured from post-projection total and per-phase active-power histories.",
            "",
            "## Integrity findings",
            "",
            *(integrity_findings or ["No integrity finding."]),
            "",
            "## Controller outcome watches",
            "",
            *(controller_watches or ["No controller outcome watch."]),
            "",
            "## Matched scenario contrasts",
            "",
            "| Contrast | Policy | Cost | Daily peak | Ramping | EV minimum delta | Deferrable delta | Requested pressure delta (kWh) | Residual violation delta (kWh) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *(
                "| {contrast} | {policy} | {cost}% | {peak}% | {ramp}% | {ev} | {defer} | {pressure} | {residual} |".format(
                    contrast=item["contrast_id"],
                    policy=item["policy"],
                    cost=_fmt(item["cost_delta_percent"], 2),
                    peak=_fmt(item["daily_peak_delta_percent"], 2),
                    ramp=_fmt(item["ramping_delta_percent"], 2),
                    ev=_fmt(item["ev_minimum_service_delta"], 4),
                    defer=_fmt(item["deferrable_service_delta"], 4),
                    pressure=_fmt(item["electrical_pressure_delta_kwh"], 3),
                    residual=_fmt(item["electrical_residual_violation_delta_kwh"], 6),
                )
                for item in output["scenario_contrasts"]
            ),
            "",
        ]
    )
    (args.summary_dir / "annual_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "integrity_status": output["integrity_status"],
                "comparisons": len(comparisons),
                "controller_watches": len(controller_watches),
            },
            indent=2,
        )
    )
    if integrity_findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
