"""Check training labels against an independent mixed-integer formulation."""

import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .demonstrations import best
from .protocol import HERE, fit_nights, split


def optimum(s, t, delivered, lo, hi, wait, forced=None):
    n = s.quarters - t
    terms = 3 * n + 1
    discount = 0.99 ** (0.25 * np.arange(n))
    price, pv, _ = np.asarray([s.window(i) for i in range(t, s.quarters)]).T
    c = np.r_[np.zeros(2 * n), price * discount, discount[-1]]
    lower = np.zeros(terms)
    upper = np.r_[np.ones(2 * n), np.full(n, np.inf), 1.0]
    if not wait:
        lower[n] = 1.0
    if forced is not None:
        lower[0] = upper[0] = forced
    coef = []
    lb = []
    ub = []

    def row(terms, low=-np.inf, high=np.inf):
        vals = np.zeros(len(c))
        for j, x in terms:
            vals[j] = x
        coef.append(vals)
        lb.append(low)
        ub.append(high)

    for j in range(n):
        row([(j, 1.0), (n + j, -(lo if j == 0 else 0.35))], low=0.0)
        row([(j, 1.0), (n + j, -(hi if j == 0 else 1.0))], high=0.0)
        row([(j, 1.0), (2 * n + j, -1.0)], high=pv[j] * 0.25)
    row([(j, 1.0) for j in range(n)] + [(3 * n, 1.0)], low=s.request_kwh - delivered)
    result = milp(
        c,
        integrality=np.r_[np.zeros(n), np.ones(n), np.zeros(n + 1)],
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.asarray(coef), lb, ub),
        options={"mip_rel_gap": 0.0},
    )
    assert result.success
    return result.fun


def main():
    train, _, _ = split()
    rng = np.random.default_rng(543)
    diff = []
    for i in range(160):
        s = train[rng.integers(len(train))]
        t = int(rng.integers(s.quarters))
        n = s.quarters - t
        d = float(np.clip(rng.uniform(0, t + 0.01), 0, t))
        if d + n < s.request_kwh - 1.0 + 1e-7:
            continue
        lo = 0.35
        hi = 1.0
        wait = d + n - 1 >= s.request_kwh - 1.0 - 1e-8
        q = best(s, t, d, wait, True, lo, hi)
        diff.append(abs(optimum(s, t, d, lo, hi, wait, q) - optimum(s, t, d, lo, hi, wait)))
    n_short = len(diff)
    nights = fit_nights(train, 397713, 32)
    for s in nights:
        t = int(rng.integers(0, max(1, s.quarters - 8)))
        delivered = min(float(t), max(0.0, s.request_kwh + rng.uniform(-5, 1.0)))
        q = best(s, t, delivered, True, True, 0.35, 1.0)
        diff.append(
            abs(
                optimum(s, t, delivered, 0.35, 1.0, True, q)
                - optimum(s, t, delivered, 0.35, 1.0, True)
            )
        )
    assert max(diff) < 2e-7, (len(diff), max(diff))
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "teacher-check.json").write_text(
        json.dumps(
            {
                "short_training_states": n_short,
                "overnight_training_states": len(diff) - n_short,
                "max_objective_disagreement": max(diff),
            }
        )
    )


if __name__ == "__main__":
    main()
