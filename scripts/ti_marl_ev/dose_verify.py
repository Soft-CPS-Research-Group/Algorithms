"""Final blind check of the learned low-energy response; use new dates and fractional requests."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch

from algorithms.ti_marl.experiments.ev_pretraining import load_ev_policy
from scripts.train_ti_marl_ev_sessions import SessionProbe, make_cases

from .dose_calibrate import STEPS
from .dose_evaluate import bands, development, fast_sessions, paired, single
from .dose_select import real_dev
from .dose_service_study import additional_cases
from .dose_window import assess, scenarios
from .from_scratch import fresh
from .intermediate import OUT, fractional_cases, sealed_cases
from .native_curves import original_curve_sessions
from .protocol import TRACE, schema_path, split


def blind():
    earlier = {s.day for g in sealed_cases().values() for s in g} | {
        s.day for s in additional_cases()
    }
    _, pool = make_cases(schema_path(), 0, 100)
    days = sorted({s.day for s in pool if s.day not in earlier and s.day < "2023-12-31"})
    assert len(days) >= 14
    return fractional_cases([s for s in pool if s.day in days], 591403)


def challenge(base, key):
    out = []
    for s in scenarios(base, varied=True):
        digest = hashlib.sha256(f"{key}-{s.day}-{s.variable}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        interval = [(0.36, 0.98), (1.01, 1.34), (1.01, 1.34), (1.36, 2.0), (2.05, 4.3)][
            int(rng.integers(5))
        ]
        out.append(replace(s, request_kwh=float(rng.uniform(*interval))))
    return out


@torch.no_grad()
def window_residuals(actor, cases):
    delivered = fast_sessions(actor, cases)
    out = {}
    for variable in ("price", "solar"):
        errors = np.array(
            [
                sum(r["kwh"]) - s.request_kwh
                for r, s in zip(delivered, cases)
                if s.variable == variable
            ]
        )
        out[variable] = dict(
            short_over_01=int((errors < -0.10000001).sum()),
            excess_over_01=int((errors > 0.10000001).sum()),
            max_short=float(max(0.0, -min(errors))),
            max_excess=float(max(0.0, max(errors))),
        )
    return out


@torch.no_grad()
def selection(joint=False):
    torch.set_num_threads(1)
    _, dev, _ = split()
    choose = {}
    for w in (32, 64, 128):
        rec = {}
        for seed in (43, 73):
            a = fresh(w, seed)
            for step in (0, 512, 2048, 4096) if joint else STEPS:
                name = (
                    f"switch-{w}-{seed}-{step}.pth"
                    if joint and step
                    else f"v2-{w}-{seed}-{2048 if joint else step}.pth"
                )
                a.actor = load_ev_policy(OUT / "calibration" / name)
                m = development(dev, a, None, True)
                r = real_dev(a.actor, dev)
                windows = assess(a, challenge(dev, 23871))["summary"]
                v = [
                    m["metrics"]["near_target"] / m["metrics"]["sessions"],
                    r["within"] / r["n"],
                    m["last"]["within"] / m["last"]["n"],
                    m["stops"]["within"] / m["stops"]["n"],
                ]
                for typ in ("price", "solar"):
                    v += [
                        m["metrics"][typ]["timing_and_target"] / m["metrics"][typ]["pairs"],
                        windows[typ]["done_by_end"] / windows[typ]["n"],
                        windows[typ]["within"] / windows[typ]["n"],
                    ]
                rec.setdefault(step, []).append(
                    dict(seed=seed, score=0.65 * min(v) + 0.35 * np.mean(v), components=v)
                )
        best = max(
            rec,
            key=lambda s: (min(z["score"] for z in rec[s]), np.mean([z["score"] for z in rec[s]])),
        )
        choose[w] = dict(step=best, dev=rec)
    (OUT / ("final-choice.json" if joint else "calibration-choice.json")).write_text(
        json.dumps(choose)
    )


@torch.no_grad()
def test(w, seed, arm, dest):
    torch.set_num_threads(1)
    a = fresh(w, seed)
    step = json.loads((OUT / "calibration-choice.json").read_text())[str(w)]["step"]
    if arm in ("calibrated", "joint"):
        if arm == "joint":
            step = json.loads((OUT / "final-choice.json").read_text())[str(w)]["step"]
            source = (
                OUT
                / "calibration"
                / (f"switch-{w}-{seed}-{step}.pth" if step else f"v2-{w}-{seed}-2048.pth")
            )
        else:
            source = OUT / "calibration" / f"v2-{w}-{seed}-{step}.pth"
        model = (
            OUT
            / "models"
            / ("validated" if arm == "joint" else "calibrated")
            / f"v2-{w}-{seed}.pth"
        )
        model.parent.mkdir(exist_ok=True)
        copyfile(source, model)
    else:
        model = OUT / "models" / "final" / f"v2-{w}-{seed}.pth"
    a.actor = load_ev_policy(model)
    cases = blind()
    probe = SessionProbe(TRACE)
    windows = challenge(cases, 610970)
    response = dict(
        days=sorted({s.day for s in cases}),
        short=paired(cases, a, probe, False),
        long=assess(a, windows, verify=True),
        dose=single(bands(cases), a, probe, False),
        stop=single(
            [replace(s, request_kwh=0.0, early=b) for s in cases for b in (True, False)],
            a,
            probe,
            False,
        ),
    )
    original = original_curve_sessions(
        a.actor,
        include_cost_reference=False,
        conditions=cases,
        variable_need=True,
        need_seed=285016,
    )
    response["original"] = {}
    for h in (8, 16):
        ls = [r for r in original if r["hour"] == h]
        errors = np.array([r["error"] for r in ls])
        response["original"][h] = dict(
            n=len(ls),
            within=int((abs(errors) <= 0.10000001).sum()),
            worst_shortfall=float(max(0.0, -min(errors))),
            worst_excess=float(max(0.0, max(errors))),
            **{
                k: sum(r[k] for r in ls)
                for k in (
                    "applied",
                    "requested",
                    "cost",
                    "equal_energy_immediate_cost",
                    "solar_kwh",
                    "equal_energy_immediate_solar",
                )
            },
        )
    if arm in ("calibrated", "joint"):
        # Independent metrics above use only the third date set. The following are explicit regressions.
        previous = additional_cases()
        response["previous_short"] = paired(previous, a, probe, True)
        conditions = scenarios(previous, varied=True)
        response["previous_long"] = assess(a, conditions, verify=True)
        diagnostic = [s for s in conditions if abs(s.request_kwh - 1.17) < 1e-8]
        energy = fast_sessions(a.actor, diagnostic)
        delivered = [sum(r["kwh"]) for r in energy]
        response["repeat_117"] = dict(
            n=len(energy),
            within=sum(abs(q - 1.17) <= 0.10000001 for q in delivered),
            min=min(delivered),
            max=max(delivered),
            first_pulses=[q for q in energy[0]["kwh"] if q > 1e-6],
        )
    response["sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    dest.write_text(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("select", "select-switch", "test"))
    p.add_argument("--width", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--arm", choices=("calibrated", "candidate", "joint"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.mode in ("select", "select-switch"):
        selection(joint=a.mode == "select-switch")
    else:
        test(a.width, a.seed, a.arm, a.output)
