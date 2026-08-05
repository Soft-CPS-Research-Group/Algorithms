from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from algorithms.oracles import TotalEnergyProblem


def test_cli_solves_serialized_problem_and_writes_schedule(tmp_path):
    problem = TotalEnergyProblem(
        problem_id="cli-smoke",
        timestep_hours=1.0,
        building_ids=("Building_1",),
        price_eur_per_kwh=np.ones(2),
        base_net_load_kwh=np.ones((1, 2)),
    )
    problem_path = tmp_path / "input.json"
    problem_path.write_text(problem.to_json(), encoding="utf-8")
    output_directory = tmp_path / "output"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/solve_total_energy_problem.py",
            "--problem",
            str(problem_path),
            "--output-dir",
            str(output_directory),
            "--mode",
            "schedule",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_directory / "summary.json").read_text())
    assert summary["status"] == "optimal"
    assert summary["cost_eur"] == 2.0
    assert (output_directory / "replay_schedule.json").is_file()


def test_cli_explicit_individual_decomposition_retains_building_results(tmp_path):
    problem = TotalEnergyProblem(
        problem_id="cli-decomposition-smoke",
        timestep_hours=1.0,
        building_ids=("Building_1", "Building_2"),
        price_eur_per_kwh=np.ones(2),
        base_net_load_kwh=np.ones((2, 2)),
        settlement="individual",
    )
    problem_path = tmp_path / "input-decomposed.json"
    problem_path.write_text(problem.to_json(), encoding="utf-8")
    output_directory = tmp_path / "output-decomposed"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/solve_total_energy_problem.py",
            "--problem",
            str(problem_path),
            "--output-dir",
            str(output_directory),
            "--mode",
            "bounded",
            "--decompose-individual",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output_directory / "summary.json").read_text())
    payload = json.loads((output_directory / "result.json").read_text())
    assert summary["decomposition"] == "exact_individual_settlement_by_building"
    assert summary["community_optimum_claim"] is False
    assert summary["building_count"] == 2
    assert summary["minimum_total_ev_shortfall_kwh"] is None
    assert summary["realized_total_ev_shortfall_kwh"] is None
    assert [item["building_id"] for item in summary["buildings"]] == [
        "Building_1",
        "Building_2",
    ]
    assert payload["combined"]["model_feasible_upper_bound_eur"] == 4.0
    assert len(payload["buildings"]) == 2
    assert (output_directory / "replay_schedule.json").is_file()
