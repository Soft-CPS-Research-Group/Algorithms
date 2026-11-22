"""Select the EV readout using development only, then compare on a second untouched set."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from algorithms.ti_marl.experiments.ev_pretraining import load_ev_policy
from scripts.train_ti_marl_ev_sessions import SessionProbe, make_cases

from .dose_evaluate import bands, development, paired, single
from .dose_select import real_dev
from .dose_service import STAGES
from .dose_window import assess, scenarios
from .from_scratch import fresh
from .intermediate import OUT, fractional_cases, sealed_cases
from .native_curves import original_curve_sessions
from .protocol import TRACE, schema_path, split


def additional_cases():
    used = {s.day for span in sealed_cases().values() for s in span}
    _, pool = make_cases(schema_path(), 0, 100)
    candidates = {s.day for s in pool if s.day not in used and "2023-09-02" <= s.day < "2023-12-31"}
    dates = sorted(
        candidates,
        key=lambda day: hashlib.sha256(("second-blind-8072:" + day).encode()).hexdigest(),
    )[:32]
    assert len(dates) == 32 and not (set(dates) & used)
    return fractional_cases([s for s in pool if s.day in dates], 749231)


def load(agent, width, seed, stage, method="service"):
    if stage == 0:
        state = torch.load(
            OUT / "models" / f"v2-{width}-{seed}.pth", map_location="cpu", weights_only=True
        )["weights"]
    else:
        state = torch.load(
            OUT / f"{method}-{width}-{seed}-{stage}.pth", map_location="cpu", weights_only=True
        )["actor"]
    agent.actor.load_state_dict(state)
    agent.actor.cpu().eval()


@torch.no_grad()
def choose():
    torch.set_num_threads(1)
    _, dev, _ = split()
    chosen = {}
    for width in (32, 64, 128):
        records = {}
        for seed in (43, 73):
            agent = fresh(width, seed)
            for method, stage in [("prior", 0)] + [
                (m, k) for m in ("service", "context", "mixed") for k in STAGES
            ]:
                load(agent, width, seed, stage, method)
                d = development(dev, agent, None, True)
                actual = real_dev(agent.actor, dev)
                windows = assess(agent, scenarios(dev))["summary"]
                metrics = d["metrics"]
                components = [
                    metrics["near_target"] / metrics["sessions"],
                    actual["within"] / actual["n"],
                    d["last"]["within"] / d["last"]["n"],
                    d["stops"]["within"] / d["stops"]["n"],
                ]
                for variable in ("price", "solar"):
                    components.extend(
                        (
                            metrics[variable]["timing_and_target"] / metrics[variable]["pairs"],
                            windows[variable]["within"] / windows[variable]["n"],
                            windows[variable]["done_by_end"] / windows[variable]["n"],
                        )
                    )
                score = 0.65 * min(components) + 0.35 * float(np.mean(components))
                records.setdefault(f"{method}-{stage}", []).append(
                    dict(
                        seed=seed,
                        score=score,
                        controlled=d["score"],
                        actual=actual,
                        windows=windows,
                    )
                )
        selected = max(
            records,
            key=lambda k: (
                min(v["score"] for v in records[k]),
                np.mean([v["score"] for v in records[k]]),
            ),
        )
        method, stage = selected.split("-")
        chosen[width] = dict(method=method, selected=int(stage), development=records)
    (OUT / "service-selection.json").write_text(json.dumps(chosen))


@torch.no_grad()
def evaluate(width, seed, phase, path):
    torch.set_num_threads(1)
    agent = fresh(width, seed)
    chosen = json.loads((OUT / "service-selection.json").read_text())[str(width)]
    stage = chosen["selected"] if phase == "candidate" else 0
    method = chosen["method"] if phase == "candidate" else "prior"
    load(agent, width, seed, stage, method)
    # Use the existing standalone policy loader for every final evaluation, including the safety layer.
    if phase == "candidate":
        source = OUT / "models" / f"v2-{width}-{seed}.pth"
        payload = torch.load(source, map_location="cpu", weights_only=True)
        payload.update(weights=agent.actor.state_dict())
        payload["training"].update(
            refinement=method,
            refinement_updates=stage,
            local_power_warmup_updates=700 if method in ("service", "context") else 0,
            intermediate_service_updates=2048 if method == "context" else 0,
        )
        model = source.parent / "final" / source.name
        model.parent.mkdir(exist_ok=True)
        torch.save(payload, model)
        agent.actor = load_ev_policy(model)
    if phase == "router":
        method = "entry_bandit"
        agent.actor = load_ev_policy(OUT / "models" / "session" / f"v2-{width}-{seed}.pth")
    probe = SessionProbe(TRACE)
    extra = additional_cases()
    old = sealed_cases()["fresh"]
    ans = {"method": method, "stage": stage, "dates": sorted({s.day for s in extra})}
    for k, cases in [("new", extra), ("regression", old)]:
        if phase == "prior" and k == "regression":
            continue
        ans[k] = paired(cases, agent, probe, False)
        ans[k + "_window"] = assess(agent, scenarios(cases, varied=(k == "new")), verify=True)
    ans["dose"] = single(bands(extra), agent, probe, False)
    from dataclasses import replace

    ans["stop"] = single(
        [replace(s, early=b, request_kwh=0.0) for s in extra for b in (True, False)],
        agent,
        probe,
        False,
    )
    actual = original_curve_sessions(
        agent.actor,
        include_cost_reference=False,
        conditions=extra,
        variable_need=True,
        need_seed=802551,
    )
    ans["original"] = {}
    for hour in (8, 16):
        rows = [r for r in actual if r["hour"] == hour]
        errors = np.array([r["error"] for r in rows])
        ans["original"][hour] = dict(
            n=len(rows),
            within=int((abs(errors) <= 0.10000001).sum()),
            mae=float(abs(errors).mean()),
            worst_shortfall=max(0.0, float(-min(errors))),
            worst_excess=max(0.0, float(max(errors))),
            **{
                k: sum(r[k] for r in rows)
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
    path.write_text(json.dumps(ans, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("choose", "evaluate"))
    p.add_argument("--width", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--phase", choices=("prior", "candidate", "router"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.mode == "choose":
        choose()
    else:
        evaluate(a.width, a.seed, a.phase, a.output)
