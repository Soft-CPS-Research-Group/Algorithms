"""Generate labelled training states. This module is never imported by RL or evaluation."""

import argparse
from dataclasses import replace
from functools import lru_cache
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from algorithms.ti_marl.experiments.ev_pretraining import fields
from algorithms.ti_marl.policy.joint import JointActor
from scripts.train_ti_marl_ev_sessions import SessionProbe

from .protocol import TRACE, fit_nights, load, split


@lru_cache(None)
def patterns(n):
    return np.asarray(list(product((0.0, 1.0), repeat=n)))


def best(s, t, delivered, wait, can_charge, lo, hi):
    n = s.quarters - t
    if n > 8:
        # With no PV, identical future limits and no export, at a fixed number
        # of activated future slots an optimal set consists of the cheapest ones.
        assert s.pv_kw == 0
        order = sorted(range(1, n), key=lambda j: (s.window(t + j)[0] * 0.99 ** (0.25 * j), -j))
        levels = np.zeros((n, n))
        for k in range(1, n):
            levels[k] = levels[k - 1]
            levels[k, order[k - 1]] = 1
        on = levels.copy()
        on[:, 0] = 1.0
        grid = np.concatenate((levels, on))
    else:
        grid = patterns(n)
    price = np.array([s.window(j)[0] for j in range(t, s.quarters)])
    pv = np.array([s.window(j)[1] * 0.25 for j in range(t, s.quarters)])
    discount = 0.99 ** (0.25 * np.arange(n))
    # For these nonnegative tariffs an attainable requested kWh is worth more than
    # its price, even after discounting. This simplifies the exact small-horizon LP.
    assert price.min() >= 0 and max(price * discount) < discount[-1]
    need = s.request_kwh - delivered
    minimum = grid * np.r_[lo, [0.35] * (n - 1)]
    maximum = grid * np.r_[hi, [1.0] * (n - 1)]
    req = np.clip(need, minimum.sum(1), maximum.sum(1))
    valid = (
        (need - req <= 1.0 + 1e-7)
        & ((grid[:, 0] == 0) if not can_charge else True)
        & ((grid[:, 0] == 1) if not wait else True)
    )
    solar = np.clip(pv - minimum, 0.0, maximum - minimum)
    paid = maximum - minimum - solar
    caps = np.column_stack((solar, paid))
    slopes = np.r_[np.zeros(n), price * discount]
    time = np.tile(np.arange(n), 2)
    # For equal price, fill later slots first; for equal return prefer less energy.
    order = np.lexsort((-time, slopes))
    sorted_caps = caps[:, order]
    taken = np.minimum(
        sorted_caps,
        np.maximum(
            0.0, (req - minimum.sum(1))[:, None] - (np.cumsum(sorted_caps, axis=1) - sorted_caps)
        ),
    )
    costs = (
        (np.maximum(0.0, minimum - pv) * price * discount).sum(1)
        + (taken * slopes[order]).sum(1)
        + np.maximum(0.0, need - req) * discount[-1]
    )
    first = minimum[:, 0] + taken[:, time[order] == 0].sum(1)
    idx = np.flatnonzero(valid)
    if not len(idx):
        raise ValueError("No feasible demonstration")
    winner = idx[
        np.lexsort((np.round(first[idx], 8), np.round(req[idx], 8), np.round(costs[idx], 10)))[0]
    ]
    return float(first[winner])


@torch.no_grad()
def generate(width, out, count, wide=False):
    torch.set_num_threads(1)
    train, _, _ = split()
    rng = np.random.default_rng(2009)
    if wide:
        train = fit_nights(train, 40198, 1200)
    a = load(width, 43).actor
    a.eval()
    probe = SessionProbe(TRACE)
    records = []
    for _ in range(count):
        s = train[rng.integers(len(train))]
        t = int(rng.integers(s.quarters))
        delivered = rng.integers(0, 2 * t + 1) / 2
        if wide:
            delivered = min(t, max(0.0, s.request_kwh + rng.uniform(-5.5, 2.0)))
        if rng.random() < 0.48:
            delivered = float(np.clip(delivered + rng.uniform(-0.32, 0.32), 0.0, t))
        if rng.random() < 0.24:
            delivered = float(
                np.clip(
                    s.request_kwh
                    + rng.choice([0.0, -0.001, 0.001, -0.02, 0.02, -0.06, 0.06, -0.35, -0.5]),
                    0.0,
                    t,
                )
            )
        if delivered + s.quarters - t < s.request_kwh - 1.0 - 1e-8:
            continue
        snap = probe.snapshot(
            s,
            False,
            t,
            s.arrival_kwh + delivered,
            (s.arrival_kwh + s.request_kwh) / 20.0,
            start_step=s.start,
            quarters=s.quarters,
        )
        domain = {}
        choose = a._choose

        def collect(c, context, state, deterministic):
            if c.stage == "mode":
                modes = a.group_modes["ev_session"]
                ch = modes.index("CHARGE_EV")
                idle = modes.index("IDLE")
                domain.update(wait=idle in c.allowed, can=ch in c.allowed, mode_logit=0.0)
                if c.sampled:
                    d = JointActor._distribution(a, context, state, c)
                    domain["mode_logit"] = float(
                        d.logits[c.allowed.index(ch)] - d.logits[c.allowed.index(idle)]
                    )
                return replace(c, mode=ch if ch in c.allowed else idle)
            domain.update(lo=c.lower, hi=c.upper, power_logit=0.0)
            if c.sampled:
                d = JointActor._distribution(a, context, state, c)
                domain["power_logit"] = float(d.concentration1.log() - d.concentration0.log())
            return choose(c, context, state, True)

        with patch.object(a, "_choose", side_effect=collect):
            ev = a(snap, deterministic=True)
        assert not ev.traces[probe.site].supervised
        lo = domain.get("lo", 0.0)
        hi = domain.get("hi", 0.0)
        q = best(s, t, delivered, domain["wait"], domain["can"], lo, hi)
        assert (q < 1e-8 and domain["wait"]) or (domain["can"] and lo - 1e-6 <= q <= hi + 1e-6)
        records.append(
            [
                *fields(snap),
                lo,
                hi,
                domain["mode_logit"],
                domain.get("power_logit", 0.0),
                float(domain["wait"] and domain["can"]),
                q,
            ]
        )
    np.save(out, np.asarray(records, dtype=np.float32))
    print(width, len(records), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("width", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--wide", action="store_true")
    p.add_argument("--count", type=int, default=24000)
    args = p.parse_args()
    generate(args.width, args.out, args.count, args.wide)
