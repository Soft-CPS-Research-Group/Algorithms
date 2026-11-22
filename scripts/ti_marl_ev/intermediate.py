"""Fractional EV requests. Train on January–August and select using development data."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from scripts.train_ti_marl_ev_sessions import SessionProbe, fingerprint, make_cases, play

from .from_scratch import fresh, matched_return_update
from .protocol import RESERVED_DATES, TRACE, schema_path, split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/ti-marl-2023-continuous-dose"
PRICE = 0.05
STEPS = (3072, 6144, 12288)


def training_case(base, rng):
    u = rng.random()
    if u < 0.56:
        request = max(0.35, base.request_kwh + rng.uniform(-0.48, 0.48))
    elif u < 0.70:
        request = base.request_kwh
    elif u < 0.83:
        return replace(
            base,
            start=base.start + base.quarters - 1,
            quarters=1,
            request_kwh=rng.uniform(0.35, 1.0),
        )
    elif u < 0.94:
        request = rng.uniform(0.35, 1.75)
    else:
        request = rng.choice([0.0, -0.25])
    return replace(base, request_kwh=float(request))


def fractional_cases(cases, seed):
    rng = np.random.default_rng(seed)
    return [
        replace(
            s,
            early=True,
            request_kwh=float(s.request_kwh + rng.choice([-1, 1]) * rng.uniform(0.13, 0.36)),
        )
        for s in cases
    ]


def sealed_cases():
    _, _, prior = split()
    _, pool = make_cases(schema_path(), 0, 100)
    other = [s for s in pool if s.day not in RESERVED_DATES]
    days = sorted(
        {s.day for s in other},
        key=lambda d: hashlib.sha256(("ev-final-fraction-v1:" + d).encode()).hexdigest(),
    )[:32]
    cases = [s for s in other if s.day in days]
    assert len(cases) == 64 and len(days) == 32
    return {
        "previous_round": prior["new"],
        "previous_fractional": prior["off_grid"],
        "fresh": fractional_cases(cases, 801294),
    }


def save_model(actor, width, seed, method, stage):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{method}-{width}-{seed}-{stage}.pth"
    torch.save(
        {
            "actor": actor.state_dict(),
            "width": width,
            "seed": seed,
            "stage": stage,
            "method": method,
            "power": "continuous",
            "excess_price_eur_per_kwh": PRICE,
        },
        path,
    )


def native(width, seed, prefix=False):
    torch.set_num_threads(1)
    a = fresh(width, seed)
    probe = SessionProbe(TRACE)
    fit, _, _ = split()
    stream = np.random.default_rng(456703)
    checksum = hashlib.sha256()
    for batch in range(1, STEPS[-1] // 4 + 1):
        s = training_case(fit[stream.integers(len(fit))], stream)
        checksum.update(json.dumps(asdict(s), sort_keys=True).encode())
        eps = []
        for _ in range(4):
            start = len(a.learner.rollout.steps)
            play(s, a, None, probe, "local", excess_price=PRICE)
            eps.append(a.learner.rollout.steps[start:])
        if prefix:
            a.actor.train()
            a.critic.train()
            statistics = a.learner.update()
            assert all(np.isfinite(v) for v in statistics.values())
        else:
            matched_return_update(a.learner, eps)
        n = batch * 4
        if n in STEPS:
            save_model(a.actor, width, seed, "prefix" if prefix else "native", n)
            print(n, checksum.hexdigest(), fingerprint(a.actor), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--prefix", action="store_true", help="Use the original V2 prefix PPO update")
    a = p.parse_args()
    native(a.width, a.seed, a.prefix)
