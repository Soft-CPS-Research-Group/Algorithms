#!/usr/bin/env python3
"""Run comparable BAU and RBCSmart audits on the canonical annual REC suite.

The campaign deliberately uses the same dataset schema, time window, entity
interface, reward, market, export policy and seed for both controllers.  It
writes the exact generated configs beside the run artefacts and extracts the
canonical Simulator KPI scorecard without recomputing costs in Algorithms.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_protocol import KPI_ROWS
from utils.config_schema import validate_config


VARIANT_PATHS = {
    "micro": "rec_2023_micro_4_q/schema.json",
    "core15": "rec_2023_core_15_stripped/schema.json",
    "core30_nominal": "rec_2023_core_30/schemas/core_30_nominal.json",
    "core30_safety": "rec_2023_core_30/schemas/core_30_safety.json",
    "core30_health": "rec_2023_core_30/schemas/core_30_health.json",
    "core30_dynamic": "rec_2023_core_30/schemas/core_30_dynamic.json",
    "core30_combined": "rec_2023_core_30/schemas/core_30_combined.json",
    "premium_clean": "rec_2023_premium_100/schemas/premium_100_clean.json",
    "premium_allin": "rec_2023_premium_100/schemas/premium_100_allin.json",
}

POLICIES = ("bau", "rbc_smart")
DATASET_CONTRACT_VERSION = "2023-q15-v1.9"
PHASE6_GRID_LIMIT_KWH = 1.0e-6
PHASE6_EV_MIN_RATE = 0.999
PHASE6_EV_PRECISION_RATE = 0.80
PHASE6_EV_ENERGY_ACCOUNTING_KWH = 1.0e-5
BASELINE_EV_SERVICE_SOC_TOLERANCE = 0.04
ELECTRICAL_KPI_RUNTIME_CONTRACT = (
    "requested_pressure_separate_from_post_projection_residual_v1"
)
EXTRA_KPI_ROWS = {
    "peak_daily_ratio_to_passive_baseline": (
        "district_energy_grid_shape_quality_peak_daily_average_to_baseline_ratio",
    ),
    "peak_all_time_ratio_to_passive_baseline": (
        "district_energy_grid_shape_quality_peak_all_time_average_to_baseline_ratio",
    ),
    "ramping_ratio_to_passive_baseline": (
        "district_energy_grid_shape_quality_ramping_average_to_baseline_ratio",
    ),
    "electrical_violation_events": (
        "district_electrical_service_phase_violations_event_count",
    ),
    "electrical_requested_pressure_kwh": (
        "district_electrical_service_phase_requested_pressure_energy_total_kwh",
    ),
    "electrical_requested_pressure_events": (
        "district_electrical_service_phase_requested_pressure_event_count",
    ),
    "community_counterfactual_cost_eur": (
        "district_cost_community_market_counterfactual_total_eur",
    ),
    "community_settlement_savings_eur": (
        "district_cost_community_market_savings_total_eur",
    ),
    "community_local_traded_kwh": (
        "district_energy_grid_community_market_local_traded_total_kwh",
    ),
    "ev_departure_soc_surplus_mean": (
        "district_ev_performance_departure_soc_surplus_mean_ratio",
    ),
    "ev_exact_target_feasible_rate": (
        "district_ev_performance_departure_success_feasible_ratio",
    ),
    "ev_exact_target_rate": (
        "district_ev_performance_departure_success_ratio",
    ),
    "ev_min_acceptable_rate": (
        "district_ev_performance_departure_min_acceptable_ratio",
    ),
    "ev_within_tolerance_rate": (
        "district_ev_performance_departure_within_tolerance_ratio",
    ),
    "ev_departure_soc_deficit_mean": (
        "district_ev_performance_departure_soc_deficit_mean_ratio",
    ),
    "ev_departure_shortfall_beyond_tolerance_mean": (
        "district_ev_performance_departure_shortfall_beyond_tolerance_mean_ratio",
    ),
    "ev_departure_events": (
        "district_ev_events_departure_count",
    ),
    "ev_departure_target_feasible_events": (
        "district_ev_events_departure_target_feasible_count",
    ),
    "ev_departure_target_infeasible_events": (
        "district_ev_events_departure_target_infeasible_count",
    ),
    "ev_departure_within_tolerance_infeasible_events": (
        "district_ev_events_departure_within_tolerance_infeasible_count",
    ),
    "ev_charge_total_kwh": (
        "district_ev_total_charge_kwh",
    ),
    "ev_connected_soc_gain_total_kwh": (
        "district_ev_energy_accounting_connected_soc_gain_kwh",
    ),
    "ev_energy_accounting_shortfall_kwh": (
        "district_ev_energy_accounting_shortfall_kwh",
    ),
    "deferrable_average_start_delay_hours": (
        "district_deferrable_appliance_service_average_start_delay_hours",
    ),
    "outage_unserved_normalized": (
        "district_comfort_resilience_resilience_unserved_energy_outage_normalized_ratio",
    ),
    "battery_capacity_fade": (
        "district_battery_health_capacity_fade_ratio",
    ),
    "demand_response_events": (
        "district_demand_response_events_count",
    ),
    "demand_response_compliance_rate": (
        "district_demand_response_compliance_ratio",
    ),
    "demand_response_shortfall_kwh": (
        "district_demand_response_shortfall_total_kwh",
    ),
    "demand_response_net_revenue_eur": (
        "district_demand_response_net_revenue_total_eur",
    ),
    "demand_response_invalid_baseline_steps": (
        "district_demand_response_invalid_baseline_time_step_count",
    ),
    "robustness_events": (
        "district_robustness_events_count",
    ),
    "robustness_observation_corruptions": (
        "district_robustness_observation_corruption_count",
    ),
    "robustness_forecast_corruptions": (
        "district_robustness_forecast_corruption_count",
    ),
    "robustness_action_corruptions": (
        "district_robustness_action_corruption_count",
    ),
    "robustness_asset_unavailable_steps": (
        "district_robustness_asset_unavailable_time_step_count",
    ),
}


def _source_tree_sha256(source_root: Path) -> str:
    """Hash an importable Simulator Python source tree deterministically."""

    source_root = source_root.resolve()
    digest = hashlib.sha256()
    paths = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    if not paths:
        raise ValueError(f"No Python sources found below {source_root}")
    for path in paths:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _attest_simulator_runtime(
    simulator_repo: Path,
    env: Mapping[str, str],
) -> Mapping[str, Any]:
    """Fail unless a fresh worker imports the exact audited source tree."""

    simulator_repo = simulator_repo.resolve()
    expected_source_root = (simulator_repo / "citylearn").resolve()
    expected_sha256 = _source_tree_sha256(expected_source_root)
    probe = r"""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import citylearn
from citylearn.internal.topology import CityLearnTopologyService

root = Path(citylearn.__file__).resolve().parent
digest = hashlib.sha256()
for path in sorted(path for path in root.rglob("*.py") if path.is_file()):
    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    payload = path.read_bytes()
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
try:
    distribution_version = version("softcpsrecsimulator")
except PackageNotFoundError:
    distribution_version = None
print(json.dumps({
    "citylearn_init_file": str(Path(citylearn.__file__).resolve()),
    "citylearn_source_root": str(root),
    "citylearn_source_sha256": digest.hexdigest(),
    "distribution_version": distribution_version,
    "dynamic_historical_charger_kpis": hasattr(
        CityLearnTopologyService, "historical_chargers"
    ),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=dict(env),
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout.strip())
    observed_root = Path(observed["citylearn_source_root"]).resolve()
    if observed_root != expected_source_root:
        raise RuntimeError(
            "Simulator runtime source mismatch: expected "
            f"{expected_source_root}, imported {observed_root}."
        )
    if observed.get("citylearn_source_sha256") != expected_sha256:
        raise RuntimeError(
            "Simulator runtime digest mismatch: the worker did not import the "
            "audited source bytes."
        )
    if observed.get("dynamic_historical_charger_kpis") is not True:
        raise RuntimeError(
            "Simulator runtime lacks dynamic historical-charger KPI support."
        )
    observed["expected_source_root"] = str(expected_source_root)
    observed["expected_source_sha256"] = expected_sha256
    canonical = json.dumps(observed, sort_keys=True, separators=(",", ":"))
    observed["attestation_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return observed


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_schema_contract(schema: Mapping[str, Any], schema_path: Path) -> None:
    if schema.get("dataset_contract_version") != DATASET_CONTRACT_VERSION:
        raise ValueError(
            f"{schema_path} must declare dataset_contract_version "
            f"{DATASET_CONTRACT_VERSION!r}."
        )
    forecast = schema.get("derived_forecasts", {}) or {}
    if (
        forecast.get("load_pv_method") != "daily_persistence"
        or float(forecast.get("persistence_period_seconds", 0.0)) != 86_400.0
        or forecast.get("cold_start") != "current_step"
    ):
        raise ValueError(
            f"{schema_path} must use the causal daily-persistence load/PV "
            "forecast contract."
        )
    if (
        forecast.get("price_source") != "publication_aware_day_ahead_market_input"
        or forecast.get("price_publication_time_local") != "13:00"
        or forecast.get("price_unpublished_fallback") != "daily_persistence"
        or forecast.get("price_horizon_steps") != [4, 24, 96]
    ):
        raise ValueError(
            f"{schema_path} must use the publication-aware causal OMIE price "
            "forecast contract."
        )


def _dataset_integrity_sha256(schema_path: Path) -> str:
    family_root = schema_path.parent.parent if schema_path.parent.name == "schemas" else schema_path.parent
    checksum_manifest = family_root / "file_checksums.sha256"
    if not checksum_manifest.is_file():
        raise FileNotFoundError(checksum_manifest)
    for line_number, raw_line in enumerate(
        checksum_manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            expected, relative = raw_line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"Malformed dataset checksum line {line_number}: {raw_line!r}"
            ) from exc
        path = family_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(
                f"Dataset file checksum mismatch for {path}: "
                f"expected {expected}, observed {digest.hexdigest()}"
            )
    return hashlib.sha256(checksum_manifest.read_bytes()).hexdigest()


def _expected_ev_departure_audit(schema_path: Path) -> Mapping[str, int]:
    """Count service departures observable under the declared topology.

    A catalogue session is an evaluated service obligation when its member and
    physical charger are active during the final controllable interval before
    departure. Sessions hidden by pre-entry history, permanent member exit, or
    an explicit charger removal are reported as censored rather than silently
    disappearing from the KPI denominator.
    """

    import pandas as pd

    schema = _read_json(schema_path)
    family_root = (
        schema_path.parent.parent
        if schema_path.parent.name == "schemas"
        else schema_path.parent
    )
    sessions = pd.read_parquet(
        family_root / "catalogs/charging_sessions.parquet",
        columns=[
            "member_id",
            "charger_id",
            "ev_id",
            "arrival_time_step",
            "departure_time_step",
        ],
    ).sort_values(["departure_time_step", "member_id", "charger_id"])
    adjacency = sessions.sort_values(["charger_id", "arrival_time_step"]).copy()
    adjacency["previous_departure_time_step"] = adjacency.groupby("charger_id")[
        "departure_time_step"
    ].shift()
    adjacency["previous_ev_id"] = adjacency.groupby("charger_id")["ev_id"].shift() if "ev_id" in adjacency.columns else None
    back_to_back = adjacency[
        adjacency["arrival_time_step"] == adjacency["previous_departure_time_step"]
    ]
    buildings = schema.get("buildings", {}) or {}
    active_members = {
        str(member_id)
        for member_id, building in buildings.items()
        if bool((building or {}).get("include", True))
    }
    active_chargers = {
        (str(member_id), str(charger_id))
        for member_id, building in buildings.items()
        for charger_id in ((building or {}).get("chargers", {}) or {})
    }
    events = sorted(
        list(schema.get("topology_events", []) or []),
        key=lambda event: (int(event.get("time_step", 0)), str(event.get("id", ""))),
    )
    event_index = 0
    eligible = 0
    member_inactive = 0
    charger_inactive = 0

    for session in sessions.itertuples(index=False):
        departure_action_step = min(int(session.departure_time_step) - 1, 35_039)
        while (
            event_index < len(events)
            and int(events[event_index].get("time_step", 0)) <= departure_action_step
        ):
            event = events[event_index]
            member_id = str(event.get("target_member_id"))
            operation = event.get("operation")
            asset_type = event.get("target_asset_type")
            asset_id = event.get("target_asset_id")
            if operation == "add_member":
                active_members.add(member_id)
            elif operation == "remove_member":
                active_members.discard(member_id)
            elif operation == "add_asset" and asset_type == "charger":
                active_chargers.add((member_id, str(asset_id)))
            elif operation == "remove_asset" and asset_type == "charger":
                active_chargers.discard((member_id, str(asset_id)))
            event_index += 1

        member_id = str(session.member_id)
        charger_id = str(session.charger_id)
        if member_id not in active_members:
            member_inactive += 1
        elif (member_id, charger_id) not in active_chargers:
            charger_inactive += 1
        else:
            eligible += 1

    return {
        "catalogue_sessions": int(len(sessions)),
        "expected_departure_events": int(eligible),
        "censored_member_inactive": int(member_inactive),
        "censored_charger_inactive": int(charger_inactive),
        "back_to_back_session_boundaries": int(len(back_to_back)),
        "back_to_back_same_ev_boundaries": int(
            (back_to_back["ev_id"] == back_to_back["previous_ev_id"]).sum()
        ) if "ev_id" in back_to_back.columns else 0,
    }


def _expected_deferrable_service_audit(schema_path: Path) -> Mapping[str, int]:
    """Classify deferrable requests by topology-visible service opportunity.

    A request is topology-censored when the member and appliance are absent
    throughout its admissible start window. It is fully observable when both
    remain active from earliest start through deadline. The remaining requests
    are interrupted: a valid start opportunity exists, but a topology change
    can remove the service before its deadline. Completion before that change
    is policy-dependent, so the audit returns a valid interval for the runtime
    completed-plus-missed denominator.
    """

    import numpy as np
    import pandas as pd

    schema = _read_json(schema_path)
    family_root = (
        schema_path.parent.parent
        if schema_path.parent.name == "schemas"
        else schema_path.parent
    )
    catalogue = pd.read_parquet(
        family_root / "catalogs/deferrables.parquet",
        columns=["member_id", "deferrable_id"],
    )
    buildings = schema.get("buildings", {}) or {}
    events = sorted(
        list(schema.get("topology_events", []) or []),
        key=lambda event: (int(event.get("time_step", 0)), str(event.get("id", ""))),
    )
    horizon = 35_040
    catalogue_cycles = 0
    fully_observable = 0
    interrupted = 0
    censored = 0

    for item in catalogue.itertuples(index=False):
        member_id = str(item.member_id)
        asset_id = str(item.deferrable_id)
        building = buildings.get(member_id, {}) or {}
        member_active = np.full(horizon, bool(building.get("include", True)), dtype=bool)
        asset_active = np.full(
            horizon,
            asset_id in ((building.get("deferrable_appliances", {}) or {})),
            dtype=bool,
        )
        for event in events:
            if str(event.get("target_member_id")) != member_id:
                continue
            time_step = int(event.get("time_step", 0))
            operation = event.get("operation")
            asset_type = event.get("target_asset_type")
            target_asset_id = str(event.get("target_asset_id"))
            if operation == "add_member":
                member_active[time_step:] = True
            elif operation == "remove_member":
                member_active[time_step:] = False
            elif asset_type == "deferrable_appliance" and target_asset_id == asset_id:
                if operation == "add_asset":
                    asset_active[time_step:] = True
                elif operation == "remove_asset":
                    asset_active[time_step:] = False

        availability = member_active & asset_active
        schedule = pd.read_parquet(
            family_root / "deferrables" / f"{asset_id}_schedule.parquet",
            columns=[
                "earliest_start_time_step",
                "latest_start_time_step",
                "deadline_time_step",
            ],
        )
        catalogue_cycles += len(schedule)
        for cycle in schedule.itertuples(index=False):
            earliest = int(cycle.earliest_start_time_step)
            latest = int(cycle.latest_start_time_step)
            deadline = int(cycle.deadline_time_step)
            start_window = availability[earliest : latest + 1]
            service_window = availability[earliest : deadline + 1]
            if not bool(start_window.any()):
                censored += 1
            elif bool(service_window.all()):
                fully_observable += 1
            else:
                interrupted += 1

    return {
        "catalogue_cycles": int(catalogue_cycles),
        "fully_observable_cycles": int(fully_observable),
        "topology_interrupted_cycles": int(interrupted),
        "topology_censored_no_start_opportunity": int(censored),
        "expected_runtime_service_count_min": int(fully_observable),
        "expected_runtime_service_count_max": int(fully_observable + interrupted),
    }


def _build_config(
    *,
    template: Mapping[str, Any],
    policy: str,
    variant: str,
    schema_path: Path,
    topology_mode: str,
    start: int,
    end: int,
    job_id: str,
) -> dict[str, Any]:
    config = json.loads(json.dumps(template))
    config["metadata"] = {
        "experiment_name": "annual_rec_suite_baseline_audit_v1_9_asset_count_attested",
        "run_name": f"{policy} | {variant} | {start}-{end}",
        "community_name": variant,
        "description": (
            "Matched BAU/RBCSmart audit on the canonical 2023-q15-v1.9 "
            "annual REC benchmark suite after the causal-price, EV-boundary, "
            "dynamic-PV, energy-accounting, source-attested runtime and "
            "validated terminal-observation audits, with requested electrical "
            "pressure separated from post-projection residual violations and "
            "asset unavailability counted by physical identity and time step."
        ),
    }
    config["tracking"] = {
        **dict(config.get("tracking") or {}),
        "mlflow_enabled": False,
        "log_level": "INFO",
        "log_frequency": 2048,
        "progress_update_interval": 2048,
        "runtime_profiling_enabled": True,
        "runtime_profiling_interval": 2048,
    }
    simulator = config["simulator"]
    simulator.update(
        {
            "dataset_name": variant,
            "dataset_path": str(schema_path.resolve()),
            "central_agent": False,
            "interface": "entity",
            "topology_mode": topology_mode,
            "episodes": 1,
            "simulation_start_time_step": start,
            # CityLearn represents T controlled intervals with T+1 state
            # points. The canonical files contain exactly the 35,040 physical
            # intervals; the loader may append one observation-only terminal
            # boundary when the requested next point lies beyond the file.
            "simulation_end_time_step": end + 1,
            "episode_time_steps": end - start + 2,
            "terminal_observation_padding": True,
            "random_seed": 2023,
        }
    )
    simulator["export"] = {
        "mode": "none",
        "export_kpis_on_episode_end": True,
        "final_episode_only": True,
        "kpis_final_episode_only": True,
        "timeseries_final_episode_only": True,
        "include_business_as_usual": False,
        "export_business_as_usual_timeseries": False,
        "kpi_round_decimals": None,
        "session_name": job_id,
    }
    config["training"] = {
        **dict(config.get("training") or {}),
        "seed": 123,
        "steps_between_training_updates": 1,
        "target_update_interval": 0,
    }
    if policy == "bau":
        config["pipeline"] = [
            {
                "algorithm": "NormalNoBatteryPolicy",
                "count": 1,
                "hyperparameters": {
                    "control_storage": False,
                    "ev_normal_charge_rate": 1.0,
                    # A service-equivalent BAU charges immediately and stops
                    # within the declared service tolerance of the departure
                    # requirement. Charging every EV to 100% would confound
                    # scheduling benefit with extra energy.
                    "ev_normal_target_soc": 0.0,
                    "ev_service_soc_tolerance": BASELINE_EV_SERVICE_SOC_TOLERANCE,
                    "deferrable_start_action": 1.0,
                },
            }
        ]
    elif policy == "rbc_smart":
        # Pin what would otherwise be a policy implementation default so that
        # the matched baseline contract remains explicit and reproducible.
        config["pipeline"][0]["hyperparameters"]["ev_service_soc_tolerance"] = (
            BASELINE_EV_SERVICE_SOC_TOLERANCE
        )
    else:
        raise ValueError(policy)
    config["experiment_protocol"] = {
        "version": "ti_marl_experiment_protocol_v1",
        "protocol_id": "annual-rec-suite-baseline-audit-v1-9-asset-count-attested",
        "phase": "train",
        "role": "candidate",
        "data_split": "audit",
        "window_id": f"{start}-{end}",
        "candidate_id": f"{policy}-{variant}",
        "paired_reference_id": None,
        "selection_rules_sha256": None,
        "selection_record_sha256": None,
        "selected_checkpoint_sha256": None,
    }
    return config


def _find_kpi_file(job_dir: Path) -> Path | None:
    candidates = sorted(
        [
            *job_dir.glob("results/simulation_data/**/exported_kpis.csv"),
            *job_dir.glob("results/simulation_data/**/exported_kpis.parquet"),
        ]
    )
    return candidates[-1] if candidates else None


def _extract_scorecard(kpi_path: Path) -> Mapping[str, float]:
    import pandas as pd

    frame = (
        pd.read_csv(kpi_path)
        if kpi_path.suffix.lower() == ".csv"
        else pd.read_parquet(kpi_path)
    )
    rows = frame.set_index("KPI")
    metrics = {}
    for metric, candidate_rows in {**KPI_ROWS, **EXTRA_KPI_ROWS}.items():
        for row_name in candidate_rows:
            if row_name not in rows.index or "District" not in rows.columns:
                continue
            value = rows.at[row_name, "District"]
            if pd.isna(value):
                continue
            metrics[metric] = float(value)
            break
    return metrics


def _gate_status(scorecard: Mapping[str, float]) -> Mapping[str, Any]:
    grid = scorecard.get("electrical_violation_kwh")
    ev_min = scorecard.get("ev_min_acceptable_feasible_rate")
    ev_precision = scorecard.get("ev_within_tolerance_feasible_rate")
    ev_energy_accounting = scorecard.get("ev_energy_accounting_shortfall_kwh")
    gates = {
        "profile": "phase6_aggregate",
        "electrical": None if grid is None else grid <= PHASE6_GRID_LIMIT_KWH,
        "ev_minimum_service": None if ev_min is None else ev_min >= PHASE6_EV_MIN_RATE,
        "ev_target_precision": (
            None if ev_precision is None else ev_precision >= PHASE6_EV_PRECISION_RATE
        ),
        "ev_energy_accounting": (
            None
            if ev_energy_accounting is None
            else ev_energy_accounting <= PHASE6_EV_ENERGY_ACCOUNTING_KWH
        ),
    }
    known = [value for key, value in gates.items() if key != "profile" and value is not None]
    gates["hard_gate_pass"] = bool(known) and all(known)
    return gates


def _write_summary(base_dir: Path, rows: list[dict[str, Any]]) -> None:
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "full-year" if rows and rows[0]["steps"] == 35_040 else "partial-window",
        "gate_profile": {
            "name": "phase6_aggregate",
            "ev_minimum_service_rate": PHASE6_EV_MIN_RATE,
            "ev_target_precision_rate": PHASE6_EV_PRECISION_RATE,
            "ev_energy_accounting_shortfall_kwh": PHASE6_EV_ENERGY_ACCOUNTING_KWH,
            "electrical_violation_kwh": PHASE6_GRID_LIMIT_KWH,
        },
        "bau_definition": (
            "NormalNoBatteryPolicy: no stationary-storage action, immediate EV "
            "charging until the state of charge is within 0.04 of the declared "
            "departure requirement, and earliest feasible deferrable start."
        ),
        "runs": rows,
        "comparisons": [],
    }
    by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], {})[row["policy"]] = row
    for variant, policies in sorted(by_variant.items()):
        bau = policies.get("bau")
        smart = policies.get("rbc_smart")
        if not bau or not smart:
            continue
        bau_cost = bau["scorecard"].get("cost_eur")
        smart_cost = smart["scorecard"].get("cost_eur")
        output["comparisons"].append(
            {
                "variant": variant,
                "pairing_fingerprint_match": (
                    bau.get("pairing_fingerprint_sha256")
                    == smart.get("pairing_fingerprint_sha256")
                    and bau.get("pairing_fingerprint_sha256") is not None
                ),
                "bau_cost_eur": bau_cost,
                "rbc_smart_cost_eur": smart_cost,
                "rbc_smart_cost_delta_to_bau_eur": (
                    None if bau_cost is None or smart_cost is None else smart_cost - bau_cost
                ),
                "rbc_smart_cost_ratio_to_bau": (
                    None
                    if bau_cost in (None, 0.0) or smart_cost is None
                    else smart_cost / bau_cost
                ),
                **{
                    f"rbc_smart_{metric.replace('_to_passive_baseline', '')}_ratio_to_bau": (
                        None
                        if bau["scorecard"].get(metric) in (None, 0.0)
                        or smart["scorecard"].get(metric) is None
                        else smart["scorecard"][metric] / bau["scorecard"][metric]
                    )
                    for metric in (
                        "peak_daily_ratio_to_passive_baseline",
                        "peak_all_time_ratio_to_passive_baseline",
                        "ramping_ratio_to_passive_baseline",
                    )
                },
                "bau_gates": bau["gates"],
                "rbc_smart_gates": smart["gates"],
            }
        )

    (base_dir / "campaign_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "variant": row["variant"],
                "policy": row["policy"],
                "status": row["status"],
                "steps": row["steps"],
                "pairing_fingerprint_sha256": row.get("pairing_fingerprint_sha256"),
                **row.get("scorecard", {}),
                "hard_gate_pass": row.get("gates", {}).get("hard_gate_pass"),
            }
        )
    columns = sorted({key for row in flat_rows for key in row})
    with (base_dir / "campaign_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_rows)

    lines = [
        "# Annual REC baseline audit",
        "",
        f"Evidence: **{output['evidence_label']}**; gate profile: **phase6_aggregate**.",
        "",
        "| Variant | BAU cost | RBCSmart cost | Ratio | Pair matched | RBC hard gates |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for item in output["comparisons"]:
        ratio = item["rbc_smart_cost_ratio_to_bau"]
        lines.append(
            "| {variant} | {bau} | {smart} | {ratio} | {pair} | {gate} |".format(
                variant=item["variant"],
                bau="n/a" if item["bau_cost_eur"] is None else f"{item['bau_cost_eur']:.3f}",
                smart="n/a" if item["rbc_smart_cost_eur"] is None else f"{item['rbc_smart_cost_eur']:.3f}",
                ratio="n/a" if ratio is None else f"{ratio:.4f}",
                pair="yes" if item["pairing_fingerprint_match"] else "no",
                gate="pass" if item["rbc_smart_gates"]["hard_gate_pass"] else "fail/watch",
            )
        )
    (base_dir / "campaign_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulator-repo",
        type=Path,
        default=REPO_ROOT.parent / "Simulator",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT / "runs/annual_rec_baseline_audit_v1_9_asset_count_attested",
    )
    parser.add_argument("--variant", action="append", choices=sorted(VARIANT_PATHS))
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=671)
    parser.add_argument("--full-year", action="store_true")
    parser.add_argument("--job-prefix", default="annual-rec-v19-asset-count-attested")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Reuse already completed, source-attested rows in the same base directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = int(args.start)
    end = 35_039 if args.full_year else int(args.end)
    if start < 0 or end < start or end > 35_039:
        raise ValueError(f"Invalid window {start}-{end}")
    variants = args.variant or ["micro", "core15"]
    policies = args.policy or list(POLICIES)
    dataset_root = args.simulator_repo.resolve() / "data/datasets"
    base_dir = args.base_dir.resolve()
    config_dir = base_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(
        (REPO_ROOT / "configs/templates/baselines/rbc_smart_15min_local.yaml").read_text(
            encoding="utf-8"
        )
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(args.simulator_repo.resolve()) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    env["OPEVA_STARTUP_TRACE"] = "0"
    runtime_attestation = _attest_simulator_runtime(args.simulator_repo, env)
    (base_dir / "runtime_attestation.json").write_text(
        json.dumps(runtime_attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = []
    summary_path = base_dir / "campaign_summary.json"
    if args.resume_completed and summary_path.is_file():
        existing_summary = _read_json(summary_path)
        rows = list(existing_summary.get("runs", []))
    for variant in variants:
        schema_path = dataset_root / VARIANT_PATHS[variant]
        schema = _read_json(schema_path)
        _validate_schema_contract(schema, schema_path)
        topology_mode = str(schema.get("topology_mode", "static"))
        dataset_integrity_sha256 = _dataset_integrity_sha256(schema_path)
        expected_ev_departures = _expected_ev_departure_audit(schema_path)
        expected_deferrable_service = _expected_deferrable_service_audit(schema_path)
        for policy in policies:
            job_id = f"{args.job_prefix}-{variant}-{policy}-{start}-{end}"
            reusable = next(
                (
                    row
                    for row in rows
                    if row.get("variant") == variant
                    and row.get("policy") == policy
                    and row.get("job_id") == job_id
                    and row.get("status") == "completed"
                    and int(row.get("return_code", 1)) == 0
                    and int(row.get("control_transitions", 0)) == end - start + 1
                    and row.get("dataset_integrity_sha256") == dataset_integrity_sha256
                    and row.get("simulator_runtime_attestation_sha256")
                    == runtime_attestation["attestation_sha256"]
                    and row.get("resolved_terminal_observation_padding") is True
                    and row.get("electrical_kpi_runtime")
                    == ELECTRICAL_KPI_RUNTIME_CONTRACT
                ),
                None,
            )
            if args.resume_completed and reusable is not None:
                print(f"reuse completed: {variant}/{policy}", flush=True)
                continue
            config_path = config_dir / f"{job_id}.yaml"
            config = _build_config(
                template=template,
                policy=policy,
                variant=variant,
                schema_path=schema_path,
                topology_mode=topology_mode,
                start=start,
                end=end,
                job_id=job_id,
            )
            validated_config = validate_config(config)
            if validated_config.simulator.terminal_observation_padding is not True:
                raise RuntimeError(
                    "Validated campaign config dropped terminal_observation_padding."
                )
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_experiment.py"),
                    "--config",
                    str(config_path),
                    "--job_id",
                    job_id,
                    "--base-dir",
                    str(base_dir),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=False,
            )
            job_dir = base_dir / "jobs" / job_id
            resolved_config_path = job_dir / "config.resolved.yaml"
            resolved_config = (
                yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
                if resolved_config_path.is_file()
                else {}
            )
            resolved_terminal_padding = bool(
                (resolved_config.get("simulator", {}) or {}).get(
                    "terminal_observation_padding", False
                )
            )
            if completed.returncode == 0 and not resolved_terminal_padding:
                raise RuntimeError(
                    "Completed worker did not preserve terminal_observation_padding "
                    "in config.resolved.yaml."
                )
            result_path = job_dir / "results/result.json"
            result = _read_json(result_path) if result_path.is_file() else {}
            progress_path = job_dir / "progress/progress.json"
            progress = _read_json(progress_path) if progress_path.is_file() else {}
            kpi_path = _find_kpi_file(job_dir)
            scorecard = _extract_scorecard(kpi_path) if kpi_path else {}
            if completed.returncode == 0 and (
                scorecard.get("electrical_violation_kwh") is None
                or scorecard.get("electrical_requested_pressure_kwh") is None
            ):
                raise RuntimeError(
                    "Completed worker did not export both post-projection residual "
                    "violation and requested electrical pressure KPIs."
                )
            fingerprint = result.get("pairing_fingerprint") or {}
            row = {
                "variant": variant,
                "policy": policy,
                "job_id": job_id,
                "steps": end - start + 1,
                "state_points": end - start + 2,
                "control_transitions": int(progress.get("global_step", 0) or 0),
                "status": result.get(
                    "status", "completed" if completed.returncode == 0 else "failed"
                ),
                "return_code": completed.returncode,
                "config_path": str(config_path),
                "kpi_path": None if kpi_path is None else str(kpi_path),
                "pairing_fingerprint_sha256": fingerprint.get("sha256"),
                "dataset_contract_version": DATASET_CONTRACT_VERSION,
                "dataset_integrity_sha256": dataset_integrity_sha256,
                "simulator_runtime_attestation_sha256": runtime_attestation[
                    "attestation_sha256"
                ],
                "simulator_runtime_source_sha256": runtime_attestation[
                    "citylearn_source_sha256"
                ],
                "simulator_runtime_source_root": runtime_attestation[
                    "citylearn_source_root"
                ],
                "ev_departure_obligation_audit": dict(expected_ev_departures),
                "deferrable_service_obligation_audit": dict(expected_deferrable_service),
                "load_pv_forecast_method": "daily_persistence",
                "derived_price_forecast_runtime": (
                    "publication_aware_declared_horizon_with_causal_historical_issue_alignment"
                ),
                "ev_current_soc_runtime": (
                    "boundary_reference_applied_once_per_runtime_connection"
                ),
                "terminal_observation_runtime": (
                    "one_observation_only_boundary_after_declared_control_intervals"
                ),
                "resolved_terminal_observation_padding": resolved_terminal_padding,
                "electrical_kpi_runtime": ELECTRICAL_KPI_RUNTIME_CONTRACT,
                "scorecard": scorecard,
                "gates": _gate_status(scorecard),
            }
            rows = [
                existing
                for existing in rows
                if not (
                    existing.get("variant") == variant
                    and existing.get("policy") == policy
                    and existing.get("job_id") == job_id
                )
            ]
            rows.append(row)
            _write_summary(base_dir, rows)
            if completed.returncode and not args.continue_on_error:
                raise SystemExit(completed.returncode)

    _write_summary(base_dir, rows)
    print(base_dir / "campaign_summary.json")


if __name__ == "__main__":
    main()
