#!/usr/bin/env python3
"""Aggregate and validate the matched full-year annual REC baseline campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_annual_rec_baseline_audit import (
    _dataset_integrity_sha256,
    _expected_deferrable_service_audit,
    _expected_ev_departure_audit,
    _extract_scorecard,
    _gate_status,
    _source_tree_sha256,
    _write_summary,
)


EXPECTED_VARIANTS = (
    "micro",
    "core15",
    "core30_nominal",
    "core30_safety",
    "core30_health",
    "core30_dynamic",
    "core30_combined",
    "premium_clean",
    "premium_allin",
)
EXPECTED_ROBUSTNESS_EVENTS = {
    "core30_health": 4,
    "core30_combined": 4,
    "premium_allin": 8,
}
EXPECTED_ASSET_UNAVAILABLE_STEPS = {
    "core30_health": 48,
    "core30_combined": 48,
    "premium_allin": 120,
}
EXPECTED_DEMAND_RESPONSE_EVENTS = {
    "core30_combined": 4,
    "premium_clean": 4,
    "premium_allin": 4,
}
DIRECTORY_TEMPLATE = "annual_rec_baseline_audit_v1_9_asset_count_attested_{variant}_full"
DATASET_CONTRACT_VERSION = "2023-q15-v1.9"
RUNTIME_CONTRACT = (
    "publication_aware_declared_horizon_with_causal_historical_issue_alignment"
)
EV_RUNTIME_CONTRACT = "boundary_reference_applied_once_per_runtime_connection"
TERMINAL_RUNTIME_CONTRACT = (
    "one_observation_only_boundary_after_declared_control_intervals"
)
ELECTRICAL_KPI_RUNTIME_CONTRACT = (
    "requested_pressure_separate_from_post_projection_residual_v1"
)


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/annual_rec_baseline_audit_v1_9_asset_count_attested_full_summary"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    source_summaries = []
    findings = []
    expected_runtime_source_sha256 = _source_tree_sha256(
        REPO_ROOT.parent / "Simulator/citylearn"
    )

    for variant in EXPECTED_VARIANTS:
        source_dir = args.runs_root / DIRECTORY_TEMPLATE.format(variant=variant)
        if (source_dir / "INVALIDATED.md").exists():
            findings.append(f"{variant}: source directory is explicitly invalidated")
            continue
        summary_path = source_dir / "campaign_summary.json"
        if not summary_path.is_file():
            findings.append(f"{variant}: campaign_summary.json is missing")
            continue

        summary = _read_json(summary_path)
        variant_rows = list(summary.get("runs", []))
        source_summaries.append(str(summary_path.resolve()))
        if len(variant_rows) != 2:
            findings.append(f"{variant}: expected two policy runs, found {len(variant_rows)}")
        for row in variant_rows:
            # Re-read canonical exports so a stricter aggregation audit can
            # expose additional metrics without repeating a valid annual run.
            kpi_path = row.get("kpi_path")
            if kpi_path and Path(kpi_path).is_file():
                row["scorecard"] = dict(_extract_scorecard(Path(kpi_path)))
                row["gates"] = dict(_gate_status(row["scorecard"]))
            config_path = row.get("config_path")
            if config_path and Path(config_path).is_file():
                import yaml

                config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
                schema_path = Path(config["simulator"]["dataset_path"])
                row["ev_departure_obligation_audit"] = dict(
                    _expected_ev_departure_audit(schema_path)
                )
                row["deferrable_service_obligation_audit"] = dict(
                    _expected_deferrable_service_audit(schema_path)
                )
                row["current_dataset_integrity_sha256"] = (
                    _dataset_integrity_sha256(schema_path)
                )
        rows.extend(variant_rows)

    expected_pairs = {(variant, policy) for variant in EXPECTED_VARIANTS for policy in ("bau", "rbc_smart")}
    observed_pairs = {(str(row.get("variant")), str(row.get("policy"))) for row in rows}
    if observed_pairs != expected_pairs:
        findings.append(
            "run matrix mismatch: missing="
            f"{sorted(expected_pairs - observed_pairs)}, extra={sorted(observed_pairs - expected_pairs)}"
        )

    for row in rows:
        label = f"{row.get('variant')}/{row.get('policy')}"
        if row.get("status") != "completed" or int(row.get("return_code", 1)) != 0:
            findings.append(f"{label}: run did not complete successfully")
        if int(row.get("steps", 0)) != 35_040:
            findings.append(f"{label}: run is not a 35,040-step full-year rollout")
        if int(row.get("state_points", 0)) != 35_041:
            findings.append(f"{label}: run does not expose the required terminal state point")
        if int(row.get("control_transitions", 0)) != 35_040:
            findings.append(f"{label}: run did not execute 35,040 control transitions")
        if row.get("dataset_contract_version") != DATASET_CONTRACT_VERSION:
            findings.append(f"{label}: wrong dataset contract")
        if (
            not row.get("dataset_integrity_sha256")
            or row.get("dataset_integrity_sha256")
            != row.get("current_dataset_integrity_sha256")
        ):
            findings.append(f"{label}: dataset checksum manifest changed after the run")
        if row.get("derived_price_forecast_runtime") != RUNTIME_CONTRACT:
            findings.append(f"{label}: causal derived-price runtime contract is absent")
        if row.get("ev_current_soc_runtime") != EV_RUNTIME_CONTRACT:
            findings.append(f"{label}: EV current-SOC boundary runtime contract is absent")
        if row.get("terminal_observation_runtime") != TERMINAL_RUNTIME_CONTRACT:
            findings.append(f"{label}: terminal-observation runtime contract is absent")
        if row.get("electrical_kpi_runtime") != ELECTRICAL_KPI_RUNTIME_CONTRACT:
            findings.append(f"{label}: separated electrical-KPI runtime contract is absent")
        if row.get("resolved_terminal_observation_padding") is not True:
            findings.append(
                f"{label}: resolved worker config dropped terminal observation padding"
            )
        if not row.get("simulator_runtime_attestation_sha256"):
            findings.append(f"{label}: Simulator runtime attestation is absent")
        if row.get("simulator_runtime_source_sha256") != expected_runtime_source_sha256:
            findings.append(
                f"{label}: Simulator runtime source digest differs from the audited tree"
            )
        scorecard = row.get("scorecard", {}) or {}
        for metric, value in scorecard.items():
            if value is not None and not math.isfinite(float(value)):
                findings.append(f"{label}: non-finite scorecard metric {metric}")
        if scorecard.get("cost_eur") is None:
            findings.append(f"{label}: annual cost KPI is absent")
        if scorecard.get("electrical_violation_kwh") is None:
            findings.append(f"{label}: post-projection residual-violation KPI is absent")
        if scorecard.get("electrical_requested_pressure_kwh") is None:
            findings.append(f"{label}: requested electrical-pressure KPI is absent")
        if (row.get("gates", {}) or {}).get("electrical") is not True:
            findings.append(f"{label}: post-projection electrical safety gate did not pass")
        if scorecard.get("ev_energy_accounting_shortfall_kwh") is None:
            findings.append(f"{label}: EV energy-accounting KPI is absent")
        elif float(scorecard["ev_energy_accounting_shortfall_kwh"]) > 1.0e-5:
            findings.append(f"{label}: EV energy-accounting shortfall is non-zero")
        if (row.get("gates", {}) or {}).get("ev_energy_accounting") is not True:
            findings.append(f"{label}: EV energy-accounting gate did not pass")
        if not row.get("pairing_fingerprint_sha256"):
            findings.append(f"{label}: pairing fingerprint is absent")
        obligation_audit = row.get("ev_departure_obligation_audit", {}) or {}
        expected_departures = obligation_audit.get("expected_departure_events")
        observed_departures = scorecard.get("ev_departure_events")
        if expected_departures is None or observed_departures is None:
            findings.append(f"{label}: EV departure-obligation audit is incomplete")
        elif abs(float(observed_departures) - float(expected_departures)) > 1.0e-9:
            findings.append(
                f"{label}: observed {observed_departures} EV departures but topology "
                f"declares {expected_departures} observable obligations"
            )
        deferrable_audit = row.get("deferrable_service_obligation_audit", {}) or {}
        completed_cycles = scorecard.get("deferrable_completed_cycles_count")
        missed_cycles = scorecard.get("deferrable_missed_cycles_count")
        expected_min = deferrable_audit.get("expected_runtime_service_count_min")
        expected_max = deferrable_audit.get("expected_runtime_service_count_max")
        if None in (completed_cycles, missed_cycles, expected_min, expected_max):
            findings.append(f"{label}: deferrable service-obligation audit is incomplete")
        else:
            observed_service = float(completed_cycles) + float(missed_cycles)
            if not float(expected_min) <= observed_service <= float(expected_max):
                findings.append(
                    f"{label}: observed {observed_service} completed-plus-missed "
                    f"deferrable cycles outside topology-valid interval "
                    f"[{expected_min}, {expected_max}]"
                )
        expected_robustness = EXPECTED_ROBUSTNESS_EVENTS.get(
            str(row.get("variant")), 0
        )
        observed_robustness = scorecard.get("robustness_events")
        if expected_robustness:
            if observed_robustness is None or abs(
                float(observed_robustness) - expected_robustness
            ) > 1.0e-9:
                findings.append(
                    f"{label}: expected {expected_robustness} applied robustness "
                    f"events, observed {observed_robustness}"
                )
            for metric in (
                "robustness_observation_corruptions",
                "robustness_forecast_corruptions",
                "robustness_action_corruptions",
                "robustness_asset_unavailable_steps",
            ):
                if float(scorecard.get(metric, 0.0) or 0.0) <= 0.0:
                    findings.append(
                        f"{label}: declared robustness dimension {metric} was not exercised"
                    )
            expected_asset_steps = EXPECTED_ASSET_UNAVAILABLE_STEPS[str(row.get("variant"))]
            observed_asset_steps = scorecard.get(
                "robustness_asset_unavailable_steps"
            )
            if observed_asset_steps is None or abs(
                float(observed_asset_steps) - float(expected_asset_steps)
            ) > 1.0e-9:
                findings.append(
                    f"{label}: expected {expected_asset_steps} unique "
                    "asset-unavailable time steps, observed "
                    f"{observed_asset_steps}"
                )
        elif observed_robustness not in (None, 0, 0.0):
            findings.append(
                f"{label}: unexpected robustness events in a non-health variant"
            )

        expected_dr = EXPECTED_DEMAND_RESPONSE_EVENTS.get(
            str(row.get("variant")), 0
        )
        observed_dr = scorecard.get("demand_response_events")
        if expected_dr:
            if observed_dr is None or abs(float(observed_dr) - expected_dr) > 1.0e-9:
                findings.append(
                    f"{label}: expected {expected_dr} demand-response requests, "
                    f"observed {observed_dr}"
                )
            invalid_baseline_steps = scorecard.get(
                "demand_response_invalid_baseline_steps"
            )
            if invalid_baseline_steps is None or float(invalid_baseline_steps) != 0.0:
                findings.append(
                    f"{label}: demand-response baseline is absent or invalid"
                )
        elif observed_dr not in (None, 0, 0.0):
            findings.append(
                f"{label}: unexpected demand-response requests in this variant"
            )

    by_variant: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant")), {})[str(row.get("policy"))] = row
    for variant, policies in by_variant.items():
        bau = policies.get("bau")
        smart = policies.get("rbc_smart")
        if bau and smart and bau.get("pairing_fingerprint_sha256") != smart.get("pairing_fingerprint_sha256"):
            findings.append(f"{variant}: BAU and RBC Smart pairing fingerprints differ")
        if bau and smart and bau.get("dataset_integrity_sha256") != smart.get("dataset_integrity_sha256"):
            findings.append(f"{variant}: BAU and RBC Smart dataset-integrity hashes differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(args.output_dir, rows)
    validation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not findings else "fail",
        "expected_variants": list(EXPECTED_VARIANTS),
        "expected_runs": 18,
        "observed_runs": len(rows),
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "derived_price_forecast_runtime": RUNTIME_CONTRACT,
        "ev_current_soc_runtime": EV_RUNTIME_CONTRACT,
        "terminal_observation_runtime": TERMINAL_RUNTIME_CONTRACT,
        "electrical_kpi_runtime": ELECTRICAL_KPI_RUNTIME_CONTRACT,
        "source_summaries": source_summaries,
        "findings": findings,
    }
    (args.output_dir / "aggregate_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "aggregate_validation.md").write_text(
        "\n".join(
            [
                "# Full-year annual REC campaign validation",
                "",
                f"Status: **{validation['status']}**.",
                "",
                f"Observed {len(rows)} of 18 required matched runs.",
                "",
                "## Findings",
                "",
                *(findings or ["No structural campaign finding."]),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": validation["status"], "runs": len(rows), "findings": len(findings)}, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
