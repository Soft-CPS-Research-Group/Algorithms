"""A mutable, single-threaded HiGHS feasibility model for one site decision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic

import numpy as np
from scipy.sparse import csr_matrix

from algorithms.utils.highs_scheduler import highs_scheduler


class ReservationUnavailable(RuntimeError):
    """No independently verified decision was obtained, including on timeout."""


class ReservationInfeasible(ReservationUnavailable):
    """HiGHS proved the current prefix infeasible."""


@dataclass(frozen=True)
class VerificationProblem:
    bounds: np.ndarray
    rows: csr_matrix
    row_bounds: np.ndarray
    integer_columns: tuple[int, ...]

    def verify(self, x):
        if len(x) != len(self.bounds) or not np.isfinite(x).all():
            return False
        tol = 1e-6
        if self.bounds.size:
            lower, upper = self.bounds.T
            if np.any(x < lower - tol) or np.any(x > upper + tol):
                return False
        if self.row_bounds.size:
            lower, upper = self.row_bounds.T
            activity = self.rows @ x
            if np.any(activity < lower - tol) or np.any(activity > upper + tol):
                return False
        binary = x[list(self.integer_columns)]
        return bool(np.all(np.abs(binary - np.rint(binary)) <= tol))


class LinearModel:
    def __init__(self, budget_seconds):
        import highspy

        if not math.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError("Reservation time budget must be finite and positive")
        self.finish = monotonic() + budget_seconds
        self.solver = highspy.Highs()
        self.solver.setOptionValue("output_flag", False)
        self.solver.setOptionValue("threads", 1)
        self.solver.setOptionValue("mip_rel_gap", 0.0)
        self.solver.setOptionValue("mip_abs_gap", 0.0)
        self.solver.setOptionValue("mip_feasibility_tolerance", 1e-8)
        self.bounds = []
        self.integer = []
        self.rows = []
        self.limits = []
        self.matrix = None
        self.cost = {}
        self.loaded = False
        self.solves = 0

    def remaining_seconds(self):
        remaining = self.finish - monotonic()
        if remaining <= 0:
            raise ReservationUnavailable("Site reservation time budget expired")
        return remaining

    def var(self, lo=0.0, hi=1.0, *, binary=False):
        if self.loaded:
            raise RuntimeError("All reservation variables must precede the first solve")
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ReservationUnavailable(f"Invalid reservation bounds {lo}, {hi}")
        col = len(self.bounds)
        self.bounds.append((lo, hi))
        if binary:
            self.integer.append(col)
        return col

    def row(self, terms, lo=-np.inf, hi=np.inf):
        terms = {k: v for k, v in terms.items() if v}
        if any(not math.isfinite(v) for v in terms.values()) or math.isnan(lo) or math.isnan(hi):
            raise ReservationUnavailable("Invalid reservation equation")
        self.rows.append(terms)
        self.limits.append((lo, hi))
        self.matrix = None
        if self.loaded:
            self.solver.addRow(lo, hi, len(terms), list(terms), list(terms.values()))

    def bound(self, col, lo, hi):
        self.bounds[col] = (lo, hi)
        if self.loaded:
            self.solver.changeColBounds(col, lo, hi)

    def _csr(self):
        if self.matrix is None:
            ptr, indices, values = [0], [], []
            for row in self.rows:
                indices.extend(row)
                values.extend(row.values())
                ptr.append(len(values))
            self.matrix = csr_matrix(
                (values, indices, ptr), shape=(len(self.rows), len(self.bounds))
            )
        return self.matrix

    def solve(self, cost=None, *, known_feasible=None):
        """Return an optimal, independently checked primal; never use a timed-out incumbent."""
        import highspy

        if not self.bounds:
            x = np.empty(0)
            if self.verify(x):
                return x
            raise ReservationInfeasible("Unavoidable load exceeds the bound")
        if not self.loaded:
            lower, upper = np.array(self.bounds, dtype=float).T
            self.solver.addVars(len(lower), lower, upper)
            if self.integer:
                self.solver.changeColsIntegrality(
                    len(self.integer), self.integer, np.ones(len(self.integer), dtype=np.uint8)
                )
            matrix = self._csr()
            if self.rows:
                lo, hi = np.asarray(self.limits).T
                self.solver.addRows(
                    len(lo), lo, hi, matrix.nnz, matrix.indptr, matrix.indices, matrix.data
                )
            self.loaded = True
        cost = cost or {}
        if self.cost != cost:
            cols = sorted(self.cost.keys() | cost.keys())
            self.solver.changeColsCost(len(cols), cols, [cost.get(col, 0.0) for col in cols])
            self.cost = dict(cost)

        def optimise():
            self.solver.setOptionValue("time_limit", self.remaining_seconds())
            self.solves += 1
            if self.solver.run() == highspy.HighsStatus.kError:
                raise ReservationUnavailable("HiGHS failed to run the reservation model")
            return self.solver.getModelStatus()

        try:
            with highs_scheduler(highspy, 1, timeout=self.remaining_seconds()):
                status = optimise()
                if (
                    status == highspy.HighsModelStatus.kInfeasible
                    and known_feasible is not None
                    and self.verify(known_feasible)
                ):
                    # A previous optimum satisfies the new priority floor. Resolve the
                    # identical problem without reductions before accepting infeasibility.
                    _, presolve = self.solver.getOptionValue("presolve")
                    self.solver.clearSolver()
                    self.solver.setOptionValue("presolve", "off")
                    try:
                        self.solver.setSolution(
                            len(self.bounds),
                            np.arange(len(self.bounds), dtype=np.int32),
                            np.asarray(known_feasible, dtype=float),
                        )
                        status = optimise()
                    finally:
                        self.solver.setOptionValue("presolve", presolve)
        except TimeoutError as exc:
            raise ReservationUnavailable("Site time budget expired waiting for HiGHS") from exc
        if status == highspy.HighsModelStatus.kInfeasible:
            raise ReservationInfeasible("No continuation for these conditions")
        if status != highspy.HighsModelStatus.kOptimal:
            raise ReservationUnavailable(
                f"Unresolved reservation ({self.solver.modelStatusToString(status)})"
            )
        x = np.asarray(self.solver.getSolution().col_value)
        if not self.verify(x):
            raise ReservationUnavailable("Returned witness failed the independent check")
        self.remaining_seconds()
        return x

    def relax_fixed_integers(self):
        # All integer modes (and programme delivery fractions) must be fixed
        # before dropping integrality. Keep the indices for independent checks.
        fixed = [j for j in self.integer if self.bounds[j][0] == self.bounds[j][1]]
        if fixed:
            self.solver.changeColsIntegrality(
                len(fixed), fixed, np.zeros(len(fixed), dtype=np.uint8)
            )
        if len(fixed) != len(self.integer):
            raise ReservationUnavailable("A discrete choice remained free in the power pass")

    def verification_problem(self):
        # Updating the model creates a new CSR matrix when rows are added;
        # the old matrix remains an immutable copy of the equations.
        return VerificationProblem(
            np.asarray(self.bounds), self._csr(), np.asarray(self.limits), tuple(self.integer)
        )

    def verify(self, x):
        return self.verification_problem().verify(x)
