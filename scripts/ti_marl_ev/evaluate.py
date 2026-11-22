"""Select on development only, then evaluate frozen checkpoints on the reserved dates."""

import argparse
import hashlib
import json
import statistics as stats
import sys
from dataclasses import dataclass, replace

import torch

from scripts.train_ti_marl_ev_sessions import Session, SessionProbe, evaluate, play

from .metrics import measures
from .protocol import HERE, TRACE, load, split


@dataclass(frozen=True)
class Night(Session):
    def window(self, index):
        start = 2 if self.early else 30
        return (self.low if start <= index < start + 4 else self.high), 0.0, 0.0


def dev_nights(width, seed, variant, rate):
    """Compare actual cheap-window energy with both placements on development dates."""
    from .protocol import wide_sessions

    torch.set_num_threads(1)
    _, dev, _ = split()
    sessions = wide_sessions(dev, 73520, 12)
    agent = load(width, seed, variant)
    tag = f"v2-{width}-{variant}-s{seed}-bc-lr{rate:g}"
    state = torch.load(HERE / (tag + "-1536.pth"), map_location="cpu", weights_only=True)
    agent.actor.load_state_dict(state["actor"])
    probe = SessionProbe(TRACE)
    successes = 0
    errors = []
    with torch.no_grad():
        for s in sessions:
            left = play(replace(s, early=True), agent, None, probe, "eval")
            right = play(replace(s, early=False), agent, None, probe, "eval")
            i = s.start_cheap
            j = s.quarters - i - s.cheap_length

            def window(run, t):
                return sum(run["kwh"][t : t + s.cheap_length])

            delta = min(window(left, i) - window(right, i), window(right, j) - window(left, j))
            err = max(left["miss"], left["extra"], right["miss"], right["extra"])
            errors.append(err)
            successes += int(delta > 0.1 and err <= 0.1 + 1e-8)
    (HERE / (tag + "-nightdev.json")).write_text(
        json.dumps({"passed": successes, "pairs": len(sessions), "errors": errors})
    )


def selection():
    selected = {}
    for width in (32, 64):
        candidates = []
        for variant in ("continuous", "boundary", "wide"):
            for rate in (1.0, 0.1, 0.01) if variant == "wide" else (1.0, 0.1):
                trials = []
                errors = []
                for seed in (43, 73):
                    tag = f"v2-{width}-{variant}-s{seed}-bc-lr{rate:g}"
                    log = json.loads((HERE / (tag + ".json")).read_text())
                    m = measures(log["milestones"]["1536"]["dev"])
                    n = json.loads((HERE / (tag + "-nightdev.json")).read_text())
                    trials.extend(
                        m[k]["timing_and_target"] / m[k]["pairs"] for k in ("price", "solar")
                    )
                    trials.append(n["passed"] / n["pairs"])
                    errors.extend((m["mae_kwh"], stats.mean(n["errors"])))
                key = [
                    min(trials),
                    stats.mean(trials),
                    -stats.mean(errors),
                    {"continuous": 2, "boundary": 1, "wide": 0}[variant],
                    int(rate == 1.0),
                ]
                candidates.append(
                    {"variant": variant, "rate": rate, "score": key, "individual_rates": trials}
                )
        winner = max(candidates, key=lambda c: c["score"])
        selected[str(width)] = {"winner": winner, "compared": candidates}
    (HERE / "selection.json").write_text(json.dumps(selected, indent=2, allow_nan=False) + "\n")


def recap(rows):
    m = measures(rows)
    m["excess_gt_0.1"] = sum(
        run["extra"] > 0.1 for row in rows for run in (row["early"], row["late"])
    )
    m["short_gt_0.1"] = sum(
        run["miss"] > 0.1 for row in rows for run in (row["early"], row["late"])
    )
    actual = sum(q for row in rows for side in ("early", "late") for q in row[side]["kwh"])
    assert m["sessions"] == len(rows) * 2 and actual >= 0
    return m


@torch.no_grad()
def evaluate_one(width, seed):
    torch.set_num_threads(1)
    probe = SessionProbe(TRACE)
    selected = json.loads((HERE / "selection.json").read_text())[str(width)]["winner"]
    variant, rate = selected["variant"], selected["rate"]
    name = f"v2-{width}-{variant}-s{seed}"
    _, _, sets = split()
    out = {}
    states = {}
    stages = {
        "initial": None,
        "pretrained": name + "-bc.pth",
        "rl384": name + f"-bc-lr{rate:g}-384.pth",
        "rl1536": name + f"-bc-lr{rate:g}-1536.pth",
        "rl_only": name + f"-cold-lr{rate:g}-1536.pth",
    }
    for stage, file in stages.items():
        a = load(width, seed, variant)
        if file:
            w = torch.load(HERE / file, map_location="cpu", weights_only=True)
            a.actor.load_state_dict(w["actor"])
            a.critic.load_state_dict(w["critic"])
            states[stage] = hashlib.sha256((HERE / file).read_bytes()).hexdigest()
        a.actor.eval()
        all_rows = {k: evaluate(cases, a, None, probe) for k, cases in sets.items()}
        # Driver asks for no additional energy, or is already above request.
        stops = {}
        for request in (0.0, -0.35):
            trajectories = []
            for c in sets["new"]:
                for early in (True, False):
                    r = play(replace(c, request_kwh=request, early=early), a, None, probe, "eval")
                    trajectories.append(sum(r["kwh"]))
            stops[str(request)] = {
                "sessions": len(trajectories),
                "no_charge": sum(q < 1e-8 for q in trajectories),
                "max_kwh": max(trajectories),
            }
        night = []
        if stage in ("pretrained", "rl1536"):
            for c in [s for s in sets["new"] if s.variable == "price" and s.day > "2023-09-01"][:8]:
                from datetime import datetime
                from zoneinfo import ZoneInfo

                tz = ZoneInfo("Europe/Lisbon")
                departure = datetime.fromisoformat(c.day + "T08:00:00").replace(tzinfo=tz)
                origin = datetime(2023, 1, 1, tzinfo=tz)
                step = round((departure.timestamp() - origin.timestamp()) / 900) - 40
                # In a 10 h overnight service, move a single 1 h cheap window.
                s = Night(c.day, "08:00", step, 40, 10.0, 3.0, c.low, c.high, 0.0, "price", True)
                left = play(s, a, None, probe, "eval")
                right = play(replace(s, early=False), a, None, probe, "eval")
                shift = min(
                    sum(left["kwh"][2:6]) - sum(right["kwh"][2:6]),
                    sum(right["kwh"][30:34]) - sum(left["kwh"][30:34]),
                )
                night.append({"day": s.day, "shift": shift, "early": left, "late": right})
        out[stage] = {
            "measures": {k: recap(v) for k, v in all_rows.items()},
            "rows": all_rows,
            "at_target": stops,
            "overnight": night,
        }
    assert not any(x in sys.modules for x in ("control", "demonstrations"))
    (HERE / f"v2-{width}-s{seed}-sealed.json").write_text(
        json.dumps(
            {"width": width, "seed": seed, "choice": selected, "hashes": states, "stages": out},
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--select", action="store_true")
    p.add_argument("--width", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--development", action="store_true")
    p.add_argument("--variant", choices=("wide", "boundary", "continuous"))
    p.add_argument("--rate", type=float)
    a = p.parse_args()
    if a.select:
        selection()
    elif a.development:
        dev_nights(a.width, a.seed, a.variant, a.rate)
    else:
        evaluate_one(a.width, a.seed)
