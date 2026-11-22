"""Isolate the appliance's learned response, then confirm on six unused dates.

All candidates execute actual SoftCPS episodes with the same V2 parameters,
including EV and battery policies, except for the weight on forecast solar surplus at the instant of starting.
Selection uses realised reward; no target start or preferred action is supplied.
"""

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from scripts.ti_marl_household import experiment
from scripts.ti_marl_household.audit import metrics, run_control, timing


def summary(rows):
    return dict(
        overall=metrics(rows),
        price=timing(rows, "price"),
        solar=timing(rows, "solar"),
        untouched_curves=metrics([r for r in rows if r["stimulus"] == "native"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("previous", type=Path)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--processes", type=int, default=7)
    args = p.parse_args()
    path = args.directory
    cases = json.loads((path / "cases.json").read_text())
    base = np.asarray(json.loads(args.previous.read_text())["weights"])
    # The previous machine had no SOC. Its SOC input was the constant -0.5.
    # Move that constant into the bias to preserve its previous decisions exactly.
    base[2, 0] -= 0.5 * base[2, 5]
    base[2, 5] = 0.0
    search = []
    for weight in (0.0, -16.0, -8.0, -4.0, -2.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        w = base.copy()
        w[2, 5] = weight
        search.append(w)
    ids = [
        i
        for i, c in enumerate(cases)
        if c["day"] in (40, 152) and c["first"] == int(c["day"] == 152)
    ]
    dev = [i for i, c in enumerate(cases) if c["split"] == "dev"]
    held = [i for i, c in enumerate(cases) if c["split"] == "replication"]
    assert len(held) == 60 and {cases[i]["day"] for i in held}.isdisjoint(
        cases[i]["day"] for i in ids + dev
    )
    with get_context("fork").Pool(
        args.processes, experiment.worker_init, (path, args.seed)
    ) as pool:

        def score(params, indices):
            rows = pool.map(
                run_control, [(i, w, "strict") for w in params for i in indices], chunksize=1
            )
            return np.array([r["reward"] for r in rows]).reshape(len(params), len(indices)).mean(1)

        # First screen one date. Resolve ties on the second date and validation.
        first = [i for i in ids if cases[i]["day"] == 40]
        second = [i for i in ids if cases[i]["day"] == 152]
        train = score(search, first)
        top = np.unique(np.r_[0, np.argsort(-train, kind="stable")[:8]])
        train2 = score([search[i] for i in top], second)
        short = np.unique(np.r_[0, top[np.argsort(-(train[top] + train2), kind="stable")[:4]]])
        values = score([search[i] for i in short], dev)
        winner = int(short[np.argmax(values)])
        selected = search[winner]
        (path / f"refined-{args.seed}.json").write_text(
            json.dumps(
                dict(
                    weights=selected.tolist(),
                    validation=dict(
                        before=float(values[np.where(short == 0)[0][0]]), after=float(values.max())
                    ),
                    increment=(selected - base)[2].tolist(),
                )
            )
        )
        print("selection", winner, values.tolist(), flush=True)
        arms = {}
        for name, w, mode in (
            ("transfer", np.zeros((5, 8)), "strict"),
            ("before_programme", base, "strict"),
            ("selected", selected, "strict"),
            ("blind", selected, "blind_strict"),
        ):
            rows = pool.map(run_control, [(i, w, mode) for i in held], chunksize=1)
            arms[name] = summary(rows)
            if name == "selected":
                example = [r for r in rows if r["day"] == cases[held[0]]["day"] and r["first"] == 0]
                (path / "examples.json").write_text(json.dumps(example))
            print("completed", name, flush=True)
        (path / "confirmation.json").write_text(json.dumps(arms, indent=2))


if __name__ == "__main__":
    main()
