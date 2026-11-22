"""Frozen native V2 replays for service, shifted timing and fractional delivery."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from scripts.train_ti_marl_ev_sessions import SessionProbe, evaluate, play

from .dose_training import rollout
from .from_scratch import fresh
from .intermediate import fractional_cases, sealed_cases
from .metrics import measures
from .native_curves import original_curve_sessions
from .protocol import TRACE, split


def bands(cases):
    # Include minimum and maximum, with interior requests independent of the train stream.
    quantities = [0.35, 0.41, 0.48, 0.56, 0.64, 0.70, 0.75, 0.83, 0.91, 0.98, 1.0]
    out = []
    for i, q in enumerate(quantities):
        s = cases[i % len(cases)]
        for tariff, solar in ((0.08, 0.0), (0.24, 0.0), (0.24, 4.0)):
            out.append(
                replace(
                    s,
                    quarters=1,
                    request_kwh=q,
                    early=False,
                    variable="both",
                    low=tariff,
                    high=tariff,
                    pv_kw=solar,
                )
            )
    return out


def fast_sessions(actor, conditions):
    """Batch equal-length sessions, while preserving native evaluation order."""
    response = [None] * len(conditions)
    for n in sorted({s.quarters for s in conditions}):
        ids = [i for i, s in enumerate(conditions) if s.quarters == n]
        cases = [conditions[i] for i in ids]
        w = np.array([[s.window(t) for t in range(n)] for s in cases])
        q = np.array([s.request_kwh for s in cases])
        a = np.array([s.arrival_kwh for s in cases])
        price = w[:, :, 0]
        net = w[:, :, 1] / 4
        if hasattr(actor, "entry_router"):
            from types import SimpleNamespace

            from algorithms.ti_marl.experiments.ev_pretraining import entry_features

            select = (
                (actor.entry_router(torch.from_numpy(entry_features(price, net, q))).flatten() > 0)
                .cpu()
                .numpy()
            )
            delivery = torch.empty((len(q), n), dtype=torch.float64)
            for choose in (False, True):
                ids_selected = np.flatnonzero(select == choose)
                if not len(ids_selected):
                    continue
                heads = (
                    SimpleNamespace(ev_gate=actor.other_gate, ev_watts=actor.other_watts)
                    if choose
                    else actor
                )
                _, values = rollout(
                    heads,
                    price[ids_selected],
                    net[ids_selected],
                    q[ids_selected],
                    a[ids_selected],
                    explore=False,
                )
                delivery[ids_selected] = values
        else:
            _, delivery = rollout(actor, price, net, q, a, explore=False)
        energy = delivery.detach().numpy()
        for row, i in enumerate(ids):
            miss = float(q[row] - energy[row].sum())
            response[i] = dict(
                kwh=energy[row].tolist(),
                miss=max(0.0, miss),
                extra=max(0.0, -miss),
                minimum_miss=max(0.0, miss - 1.0),
                bill=float((w[row, :, 0] * (energy[row] - w[row, :, 1] / 4).clip(min=0.0)).sum()),
            )
    return response


def paired(cases, agent, probe, fast):
    if fast:
        charges = fast_sessions(
            agent.actor, [replace(s, early=b) for s in cases for b in (True, False)]
        )
        from dataclasses import asdict

        rows = []
        for j, s in enumerate(cases):
            first, second = charges[2 * j : 2 * j + 2]
            mid = s.quarters // 2
            rows.append(
                dict(
                    session=asdict(s),
                    early=first,
                    late=second,
                    shift=sum(first["kwh"][:mid]) - sum(second["kwh"][:mid]),
                )
            )
    else:
        rows = evaluate(cases, agent, None, probe)
    result = measures(rows)
    outcomes = [r[k] for r in rows for k in ("early", "late")]
    result["short_by_over_0_1"] = sum(r["miss"] > 0.10000001 for r in outcomes)
    result["over_by_over_0_1"] = sum(r["extra"] > 0.10000001 for r in outcomes)
    result["worst_shortfall"] = max(r["miss"] for r in outcomes)
    result["worst_excess"] = max(r["extra"] for r in outcomes)
    for variable in ("price", "solar"):
        active = total = solar_used = 0.0
        for row in rows:
            s = row["session"]
            if s["variable"] != variable:
                continue
            mid = s["quarters"] // 2
            for label in ("early", "late"):
                q = row[label]["kwh"]
                favorable = q[:mid] if label == "early" else q[mid:]
                active += sum(favorable)
                solar_used += sum(min(v, s["pv_kw"] / 4) for v in favorable)
                total += sum(q)
        result[variable]["in_window_share"] = active / total if total else None
        if variable == "solar":
            result[variable]["from_solar_share"] = solar_used / total if total else None
    return result


def single(cases, agent, probe, fast):
    deliveries = (
        fast_sessions(agent.actor, cases)
        if fast
        else [play(s, agent, None, probe, "eval") for s in cases]
    )
    q = np.array([sum(a["kwh"]) for a in deliveries])
    target = np.array([s.request_kwh for s in cases])
    err = q - target
    return dict(
        n=len(q),
        within=int((abs(err) <= 0.1 + 1e-8).sum()),
        mae=float(np.mean(abs(err))),
        max_error=float(max(abs(err))),
        mean_error=float(err.mean()),
        example_075=q[np.isclose(target, 0.75)].tolist(),
        max_delivered=float(max(q)),
    )


def development(cases, agent, probe, fast):
    mixed = fractional_cases(cases, 66911)
    metrics = paired(mixed, agent, probe, fast)
    last = single(bands(cases), agent, probe, fast)
    stops = single([replace(s, request_kwh=0.0) for s in cases], agent, probe, fast)
    # Equal weight to service, two timing interventions, dose at departure and stopping.
    score = np.mean(
        [
            metrics["near_target"] / metrics["sessions"],
            *[metrics[k]["timing_and_target"] / metrics[k]["pairs"] for k in ("price", "solar")],
            last["within"] / last["n"],
            stops["within"] / stops["n"],
        ]
    )
    return dict(score=float(score), metrics=metrics, last=last, stops=stops)


@torch.no_grad()
def run(path, width, seed, stage, dest, fast=False):
    torch.set_num_threads(1)
    agent = fresh(width, seed)
    weights = torch.load(path, map_location="cpu", weights_only=True)["actor"]
    agent.actor.load_state_dict(weights)
    agent.actor.cpu().eval()
    probe = SessionProbe(TRACE)
    fit, dev, _ = split()
    if stage == "dev":
        answer = development(dev, agent, probe, fast)
    else:
        assert not fast
        sets = sealed_cases()
        fresh_cases = sets["fresh"]
        assert not (
            {s.day for s in fresh_cases} & {s.day for s in fit + dev + sets["previous_round"]}
        )
        answer = {k: paired(v, agent, probe, False) for k, v in sets.items()}
        answer["dose"] = single(bands(fresh_cases), agent, probe, False)
        answer["stop"] = single(
            [replace(s, early=b, request_kwh=0.0) for s in fresh_cases for b in (True, False)],
            agent,
            probe,
            False,
        )
        for group, conditions in (("original", None), ("fresh_curves", fresh_cases)):
            original = original_curve_sessions(
                agent.actor,
                include_cost_reference=False,
                conditions=conditions,
                variable_need=conditions is not None,
            )
            answer[group] = {}
            for hour in (8, 16):
                rs = [r for r in original if r["hour"] == hour]
                errs = np.array([r["error"] for r in rs])
                matched = [r for r in rs if abs(r["error"]) <= 0.10000001]
                answer[group][hour] = dict(
                    n=len(rs),
                    within=len(matched),
                    mae=float(abs(errs).mean()),
                    mean_error=float(errs.mean()),
                    bill=sum(r["cost"] for r in rs),
                    immediate_bill=sum(r["immediate"] for r in rs),
                    matched_bill=sum(r["cost"] for r in matched),
                    matched_immediate=sum(r["immediate"] for r in matched),
                    solar=sum(r["solar_kwh"] for r in rs),
                    immediate_solar=sum(r["immediate_solar_kwh"] for r in rs),
                    delivered=sum(r["applied"] for r in rs),
                    requested=sum(r["requested"] for r in rs),
                    worst_shortfall=float(max(0.0, -min(errs))),
                    worst_excess=float(max(0.0, max(errs))),
                )
    dest.write_text(json.dumps(answer, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("stage", choices=["dev", "test"])
    p.add_argument("dest", type=Path)
    p.add_argument("--fast", action="store_true")
    a = p.parse_args()
    run(a.path, a.width, a.seed, a.stage, a.dest, a.fast)
