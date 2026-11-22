"""Check whether the EV completes early when its entire request fits a cheap/solar block."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_ti_marl_ev_sessions import Session, SessionProbe, play

from .dose_evaluate import fast_sessions
from .from_scratch import fresh
from .intermediate import OUT, sealed_cases
from .protocol import TRACE


@dataclass(frozen=True)
class Block(Session):
    begin: int = 8
    length: int = 8

    def window(self, index):
        inside = self.begin <= index < self.begin + self.length
        return (
            self.low if inside and self.variable == "price" else self.high,
            self.pv_kw if inside and self.variable == "solar" else 0.0,
            0.0,
        )


def scenarios(base=None, varied=False):
    origin = pd.Timestamp("2023-01-01", tz="Europe/Lisbon")
    cases = []
    if base is None:
        base = sealed_cases()["fresh"]
    for i, s in enumerate(base):
        # Each date contributes one night and one afternoon; move the two-hour block independently.
        hour, n = (8, 40) if s.variable == "price" else (16, 32)
        if varied:
            hour += (0, -1, 1, 0)[(i // 2) % 4]
        day = pd.Timestamp(s.day, tz="Europe/Lisbon")
        end = round((day + pd.Timedelta(hours=hour) - origin).total_seconds() / 900)
        request = (0.75, 1.43, 2.37, 3.65)[(i // 2) % 4]
        length = (6, 8, 10)[(i // 2) % 3] if varied else 8
        if varied:
            request = (0.63, 1.17, 2.71, 3.83)[(i // 2) % 4]
        for begin in (8, n - length - 4):
            cases.append(
                Block(
                    s.day,
                    f"{hour:02d}:00",
                    end - n,
                    n,
                    s.arrival_kwh,
                    request,
                    s.low,
                    s.high,
                    s.pv_kw,
                    s.variable,
                    True,
                    begin,
                    length,
                )
            )
    return cases


@torch.no_grad()
def run(width, seed, path):
    torch.set_num_threads(1)
    state = torch.load(
        OUT / "models" / f"v2-{width}-{seed}.pth", map_location="cpu", weights_only=True
    )
    agent = fresh(width, seed)
    agent.actor.cpu().eval()
    agent.actor.load_state_dict(state["weights"])
    cases = scenarios()
    path.write_text(json.dumps(assess(agent, cases, verify=True)))


def assess(agent, cases, verify=False):
    trials = fast_sessions(agent.actor, cases)
    probe = SessionProbe(TRACE) if verify else None
    error = 0.0
    count = 0
    records = {"price": [], "solar": []}
    for i, (s, result) in enumerate(zip(cases, trials)):
        q = np.array(result["kwh"])
        if verify and i % 13 == 0:
            actual = play(s, agent, None, probe, "eval")
            error = max(error, float(max(abs(np.array(actual["kwh"]) - q))))
            count += len(q)
        stop = s.begin + s.length
        total = q.sum()
        residual = abs(total - s.request_kwh)
        records[s.variable].append(
            dict(
                within=residual <= 0.10000001,
                in_block=q[s.begin : stop].sum(),
                total=total,
                after=q[stop:].sum(),
                done_by_end=abs(q[:stop].sum() - s.request_kwh) <= 0.10000001
                and q[stop:].sum() < 1e-6,
                example=(
                    q.round(5).tolist() if 2.3 < s.request_kwh < 2.8 and s.begin == 8 else None
                ),
            )
        )
    assert error < 1e-5
    summary = {}
    for v, rows in records.items():
        summary[v] = dict(
            n=len(rows),
            within=int(sum(x["within"] for x in rows)),
            done_by_end=int(sum(x["done_by_end"] for x in rows)),
            fraction_in_block=float(
                sum(x["in_block"] for x in rows) / sum(x["total"] for x in rows)
            ),
            with_later_charge=sum(int(x["after"] > 1e-6) for x in rows),
            example=next((x["example"] for x in rows if x["example"] is not None), None),
        )
    return dict(summary=summary, native_steps=count, max_error=error)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    run(a.width, a.seed, a.dest)
