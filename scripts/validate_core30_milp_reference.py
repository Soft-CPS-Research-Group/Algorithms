#!/usr/bin/env python3
"""Solve and replay a simulator-aligned community reference for one job window.

The input is a normal evaluation config. Terminal requirements are fixed before
solving: 50% stationary SOC (configurable), and the announced user target for EV
sessions still connected at the right edge. No MPC result is used as foresight.
Raw solver evidence and ordinary runtime exports are retained under output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles.citylearn_total_energy import build_citylearn_total_energy_problem
from algorithms.oracles.perfect_foresight_milp import SolveOptions
from algorithms.oracles.total_energy_milp import solve_total_energy_schedule


def build_reference(config, *, terminal_storage_soc=0.5, service_margin_kwh=0.0):
    if (
        not 0 <= terminal_storage_soc <= 1
        or not np.isfinite(service_margin_kwh)
        or service_margin_kwh < 0
    ):
        raise ValueError("Expected a terminal SOC in [0, 1] and a finite, nonnegative EV margin.")
    simulator = config["simulator"]
    schema_path = Path(simulator["dataset_path"]).resolve()
    schema = json.loads(schema_path.read_text())
    start, end = simulator["simulation_start_time_step"], simulator["simulation_end_time_step"]
    build = build_citylearn_total_energy_problem(
        schema_path=schema_path,
        start_time_step=start,
        end_time_step=end,
        settlement="community",
        building_ids=simulator.get("building_ids"),
        physics="simulator_linear",
    )
    diagnostics = build.diagnostics
    if any(
        (
            diagnostics.assumed_initial_soc_ev_session_ids,
            diagnostics.runtime_drift_initial_soc_ev_session_ids,
            diagnostics.restricted_deferrable_cycle_ids,
            diagnostics.omitted_boundary_deferrable_cycle_ids,
        )
    ):
        raise ValueError(
            "Reference requires exact arrival states and complete appliance service windows."
        )
    problem = build.problem
    root = Path(problem.metadata["dataset_root"])
    tails = set(diagnostics.right_truncated_ev_session_ids)
    sessions = []
    for session in problem.ev_sessions:
        target = session.required_final_energy_kwh
        if session.session_id in tails:
            charger = schema["buildings"][session.building_id]["chargers"][
                session.action_name.removeprefix("electric_vehicle_storage_")
            ]
            row = pd.read_parquet(root / charger["charger_simulation"]).iloc[end - 1]
            soc = np.float32(row["electric_vehicle_required_soc_departure"])
            soc = soc / np.float32(100) if soc > 1 else soc
            target = session.capacity_kwh * float(soc)
        sessions.append(
            replace(
                session,
                required_final_energy_kwh=min(session.capacity_kwh, target + service_margin_kwh),
            )
        )
    return replace(
        problem,
        stationary_storage=tuple(
            replace(
                a,
                final_energy_min_kwh=max(
                    a.minimum_energy_kwh, a.capacity_kwh * terminal_storage_soc
                ),
            )
            for a in problem.stationary_storage
        ),
        ev_sessions=tuple(sessions),
        metadata={
            **problem.metadata,
            "terminal_storage_soc": terminal_storage_soc,
            "terminal_ev_policy": "announced_user_target",
            "numerical_service_margin_kwh": service_margin_kwh,
            "reference_objective": "minimum EV deficit, then community import cost; no peak or wear penalty",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=600)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-6)
    parser.add_argument("--service-margin-kwh", type=float, default=0)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--solve-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text())
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__).resolve(),
        source_root / "algorithms/oracles/total_energy_milp.py",
        source_root / "algorithms/oracles/citylearn_total_energy.py",
        source_root / "algorithms/oracles/simulator_linear_physics.py",
        source_root / "algorithms/oracles/perfect_foresight_milp.py",
    ]
    provenance = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=source_root
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=source_root)
        ),
        "source_files_sha256": {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
    }
    problem = build_reference(config, service_margin_kwh=args.service_margin_kwh)
    (out / "problem.json").write_text(problem.to_json() + "\n")
    options = SolveOptions(
        time_limit_seconds=args.time_limit_seconds,
        mip_relative_gap=args.mip_relative_gap,
        threads=args.solver_threads,
        throughput_tiebreaker_eur_per_kwh=0,
        lexicographic_shortfall_tolerance_kwh=0,
    )
    begin = time.perf_counter()
    result = solve_total_energy_schedule(problem, options)
    seconds = time.perf_counter() - begin
    evidence = {
        **result.to_dict(),
        "solve_wall_seconds": seconds,
        **provenance,
        "options": vars(options),
        "config_path": str(args.config.resolve()),
    }
    (out / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(out),
                "seconds": seconds,
                "cost": result.cost_eur,
                "solver": result.solver.to_dict(),
                "service_shortfall_kwh": result.minimum_total_ev_shortfall_kwh,
            }
        ),
        flush=True,
    )
    if result.schedule is None:
        return 1
    schedule = out / "replay_schedule.json"
    schedule.write_text(result.schedule.to_json() + "\n")
    config["pipeline"] = [
        {
            "algorithm": "TotalOracleReplayPolicy",
            "count": 1,
            "hyperparameters": {"schedule_path": str(schedule)},
        }
    ]
    config["metadata"]["run_name"] = out.name
    config["bundle"]["require_deployable"] = False
    config_path = out / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if args.solve_only:
        return 0
    with (out / "replay.log").open("w") as log:
        process = subprocess.run(
            [
                sys.executable,
                "run_experiment.py",
                "--config",
                str(config_path),
                "--job_id",
                out.name,
                "--base-dir",
                str(out / "replay"),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
