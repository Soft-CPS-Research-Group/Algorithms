#!/usr/bin/env python3
"""Audit a saved MILP trajectory against independent Simulator job exports.

Bounds are for the declared real-valued physical model, within solver numerical
tolerances. Replay residuals are reported separately; an empirical replay error
is never presented as a universal mathematical bound on float32 arithmetic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles.total_energy_milp import TotalEnergyProblem


def integrate(initial, values, efficiency, discharge_efficiency, loss):
    energy = float(initial)
    result = []
    for index, flow in enumerate(values):
        energy *= 1 if index == 0 else 1 - loss
        energy += max(flow, 0) * efficiency + min(flow, 0) / discharge_efficiency
        result.append(energy)
    return np.array(result)


def audit(reference_dir: Path, job_dir: Path):
    problem = TotalEnergyProblem.from_json((reference_dir / "problem.json").read_text())
    solved = json.loads((reference_dir / "result.json").read_text())
    kpis = json.loads((job_dir / "results/kpis.json").read_text())
    if not kpis["eligible_for_scorecard"]:
        raise ValueError("The simulator job is not eligible for a scorecard.")
    manifests = list((job_dir / "results/simulation_data").rglob("exported_kpis.manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Exactly one final deterministic episode is required.")
    manifest = json.loads(manifests[0].read_text())
    if manifest["completed_intervals"] != problem.horizon:
        raise ValueError("Replay and optimization horizons differ.")
    export = manifests[0].parent
    metrics = {
        m["name"]: m["value"] for m in kpis["metrics"] if m["level"] == "community" and m["valid"]
    }

    def read(stem):
        files = sorted(export.glob(f"exported_data_{stem}_ep0_part*.parquet"))
        if not files:
            raise FileNotFoundError(stem)
        frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        if len(frame) != problem.horizon:
            raise ValueError(f"Incomplete trajectory: {stem}")
        return frame

    actions = {
        (s["building_id"], s["action_name"]): np.array(s["values"]) * problem.timestep_hours
        for s in solved["schedule"]["series"]
        if s["unit"] == "kW"
    }
    storage = []
    for asset in problem.stationary_storage:
        frame = read(f"{asset.building_id.lower()}_battery")
        prediction = integrate(
            asset.initial_energy_kwh,
            actions[(asset.building_id, asset.action_name)],
            asset.charge_efficiency,
            asset.discharge_efficiency,
            asset.loss_coefficient,
        )
        actual = frame["Battery Soc-%"].to_numpy(dtype=float) * asset.capacity_kwh
        flow = frame["Battery (Dis)Charge-kWh"].to_numpy(dtype=float)
        storage.append(
            {
                "building": asset.building_id,
                "max_energy_residual_kwh": float(np.max(abs(prediction - actual))),
                "max_flow_residual_kwh": float(
                    np.max(abs(flow - actions[(asset.building_id, asset.action_name)]))
                ),
                "final_energy_kwh": float(actual[-1]),
                "minimum_final_energy_kwh": asset.final_energy_min_kwh,
                "physical_bounds_pass": bool(
                    np.all(actual >= -1e-5) and np.all(actual <= asset.capacity_kwh + 1e-5)
                ),
            }
        )
    ev_frames = {}
    charger_frames = {}
    ev_rows = []
    for session in problem.ev_sessions:
        identity = problem.metadata["ev_identity_by_session"][session.session_id]
        if identity not in ev_frames:
            ev_frames[identity] = read(identity.lower())
        key = session.building_id, session.action_name
        if key not in charger_frames:
            charger_frames[key] = read(
                f"{session.building_id.lower()}_{session.action_name.removeprefix('electric_vehicle_storage_')}"
            )
        sl = slice(session.start_time_step, session.end_time_step + 1)
        flows = actions[key][sl]
        prediction = integrate(
            session.initial_energy_kwh,
            flows,
            session.charge_efficiency,
            session.discharge_efficiency,
            session.loss_coefficient,
        )
        actual = (
            ev_frames[identity]["electric_vehicle_soc"].to_numpy(dtype=float)[sl]
            * session.capacity_kwh
        )
        charger = charger_frames[key]
        delivered = (
            charger["Charger Consumption-kWh"].clip(lower=0)
            - charger["Charger Production-kWh"].clip(lower=0)
        ).to_numpy(dtype=float)[sl]
        ev_rows.append(
            {
                "session": session.session_id,
                "max_energy_residual_kwh": float(np.max(abs(prediction - actual))),
                "max_flow_residual_kwh": float(np.max(abs(flows - delivered))),
                "final_energy_kwh": float(actual[-1]),
                "required_energy_kwh": session.required_final_energy_kwh,
                "shortfall_kwh": max(0, session.required_final_energy_kwh - float(actual[-1])),
                "physical_bounds_pass": bool(
                    np.all(actual >= -1e-5) and np.all(actual <= session.capacity_kwh + 1e-5)
                ),
            }
        )
    net_residual = 0.0
    for index, building in enumerate(problem.building_ids):
        actual = read(building.lower())["Net Electricity Consumption-kWh"].to_numpy(dtype=float)
        net_residual = max(
            net_residual,
            float(np.max(abs(actual - np.array(solved["building_net_load_kwh"][index])))),
        )
    community = read("community")
    power = community["Net Electricity Consumption-kWh"].clip(lower=0) / problem.timestep_hours
    cost = metrics["district_cost_community_market_settled_total_eur"]
    lower = solved["solver"]["dual_bound_eur"]
    gates = {
        "coverage": metrics["district_ev_departure_audit_min_acceptable_coverage_ratio"] == 1
        and metrics["district_ev_departure_audit_min_acceptable_unknown_count"] == 0
        and metrics["district_deferrable_appliance_audit_resolved_coverage_ratio"] == 1,
        "minimum_service": metrics[
            "district_ev_departure_audit_min_acceptable_service_feasible_ratio"
        ]
        >= 0.999,
        "precision": metrics["district_ev_departure_audit_within_tolerance_service_feasible_ratio"]
        >= 0.8,
        "electrical": metrics["district_electrical_service_phase_violations_energy_total_kwh"]
        <= 1e-6
        and metrics["district_electrical_service_phase_violations_event_count"] == 0,
        "appliances": metrics["district_deferrable_appliance_service_missed_cycles_count"] == 0,
        "physical_bounds": all(r["physical_bounds_pass"] for r in storage + ev_rows),
        "terminal_storage": all(
            r["final_energy_kwh"] >= r["minimum_final_energy_kwh"] - 1e-4 for r in storage
        ),
        "ev_targets": all(r["shortfall_kwh"] <= 1e-4 for r in ev_rows),
        "state_fidelity": all(r["max_energy_residual_kwh"] <= 1e-4 for r in storage + ev_rows),
        "flow_fidelity": all(r["max_flow_residual_kwh"] <= 1e-4 for r in storage + ev_rows)
        and net_residual <= 1e-4,
        "service_phase": solved["service_phase_optimal"] is True
        and solved["minimum_total_ev_shortfall_kwh"] <= 1e-6,
    }
    bound_usable = (
        problem.metadata.get("physics") == "simulator_linear"
        and lower is not None
        and gates["service_phase"]
    )
    return {
        "reference_dir": str(reference_dir),
        "job_dir": str(job_dir),
        "scope": "declared constant-efficiency real-valued model, independent float32 simulator replay; numerical tolerances apply",
        "gate_profile": {
            "minimum_service": 0.999,
            "precision": 0.8,
            "electrical_kwh": 1e-6,
            "state_flow_terminal_tolerance_kwh": 1e-4,
        },
        "gates": gates,
        "gates_pass": all(gates.values()),
        "model_lower_bound_eur": lower if bound_usable else None,
        "model_incumbent_eur": solved["cost_eur"],
        "replay_cost_eur": cost,
        "replay_minus_model_cost_eur": cost - solved["cost_eur"],
        "replay_gap_to_model_bound_eur": max(0, cost - lower) if bound_usable else None,
        "replay_gap_to_model_bound_percent": 100 * max(0, cost - lower) / abs(lower)
        if bound_usable and lower
        else None,
        "model_mip_gap": solved["solver"]["mip_gap"],
        "model_solver_status": solved["solver"]["status"],
        "solve_wall_seconds": solved.get("solve_wall_seconds", solved.get("wall_seconds")),
        "peak_kw": float(power.max()),
        "mean_daily_peak_kw": float(
            power.groupby(pd.to_datetime(community.timestamp, utc=True).dt.date).max().mean()
        ),
        "max_building_net_residual_kwh": net_residual,
        "storage": storage,
        "ev_sessions": ev_rows,
        "community_service_metrics": {
            k: v for k, v in metrics.items() if "ev_departure_audit" in k
        },
        "manifest": {
            k: manifest[k]
            for k in [
                "data_files_sha256",
                "community_market",
                "simulator_version",
                "seconds_per_time_step",
                "episode_start_time_step",
                "episode_end_time_step",
                "completed_intervals",
                "random_seed",
            ]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.reference_dir.resolve(), args.job_dir.resolve())
    path = args.reference_dir / "replay_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"storage", "ev_sessions", "manifest", "community_service_metrics"}
            },
            indent=2,
        )
    )
    return 0 if report["gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
