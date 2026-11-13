"""A real week-boundary model that crashed the old bundled HiGHS 1.8."""

import gzip
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint

from algorithms.oracles import perfect_foresight_milp as backend
from algorithms.oracles.total_energy_milp import (
    TotalEnergyProblem,
    solve_total_energy_schedule,
)


def test_week_boundary_service_and_cost_solve_without_native_crash():
    # Frozen physical model from Core30 Normal at step 672, with measured EV
    # and BESS state. No schedules, result tables or machine paths are stored.
    fixture = Path(__file__).parent / "fixtures/oracles/weekly_boundary_service.json.gz"
    with gzip.open(fixture, "rt") as stream:
        problem = TotalEnergyProblem.from_json(stream.read())
    result = solve_total_energy_schedule(
        problem,
        options=backend.SolveOptions(
            threads=1,
            time_limit_seconds=10,
            mip_relative_gap=1e-6,
            lexicographic_shortfall_tolerance_kwh=0,
            throughput_tiebreaker_eur_per_kwh=0,
        ),
    )
    assert result.service_phase_optimal
    assert result.minimum_total_ev_shortfall_kwh == pytest.approx(0, abs=1e-7)
    assert result.solver.has_solution
    assert result.schedule is not None
    assert result.schedule.horizon == 672
    assert result.solver.dual_bound_eur <= result.cost_eur + 1e-6


def test_missing_standalone_solver_never_falls_back_to_affected_build(monkeypatch):
    monkeypatch.setattr(backend, "_highs_core", None)
    with pytest.raises(RuntimeError, match="requires highspy"):
        backend._milp_primal_only(
            c=np.array([1.0]),
            integrality=np.array([1]),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint([[1.0]], 0, 1),
            options={},
        )
