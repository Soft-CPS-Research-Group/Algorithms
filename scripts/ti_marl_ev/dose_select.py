"""Select one recipe and training duration per width using both seeds and development dates."""

import argparse
import json
from itertools import product

import numpy as np
import torch

from .dose_evaluate import development
from .dose_fork import CHECKS
from .dose_training import BATCH, MILESTONES, TrainingCurves, rollout
from .from_scratch import fresh
from .intermediate import OUT
from .protocol import split


@torch.no_grad()
def real_dev(actor, dev):
    data = TrainingCurves(dev)
    days = data.days[:16]
    rng = np.random.default_rng(7029)
    errors = []
    for hour, n in ((8, 24), (16, 16)):
        # Both arrival and departure are on the same, held-out development date.
        idx = days[:, None] + hour * 4 - n + np.arange(n)[None, :]
        need = rng.uniform(1.4, 4.6, len(days))
        _, q = rollout(
            actor, data.price[idx], data.surplus[idx], need, np.full(len(days), 10.0), explore=False
        )
        errors.extend((q.sum(1).numpy() - need).tolist())
    return dict(
        n=len(errors),
        within=int((np.abs(errors) <= 0.10000001).sum()),
        mae=float(np.mean(np.abs(errors))),
    )


@torch.no_grad()
def select():
    torch.set_num_threads(1)
    _, dev, _ = split()
    recipes = [("gradient", s * BATCH) for s in MILESTONES] + list(
        product(("fork", "polish"), CHECKS)
    )
    by_width = {}
    for width in (32, 64, 128):
        scores = {}
        for seed in (43, 73):
            agent = fresh(width, seed)
            agent.actor.cpu().eval()
            for method, step in recipes:
                ck = OUT / f"{method}-{width}-{seed}-{step}.pth"
                agent.actor.load_state_dict(
                    torch.load(ck, map_location="cpu", weights_only=True)["actor"]
                )
                d = development(dev, agent, None, True)
                real = real_dev(agent.actor, dev)
                score = 0.75 * d["score"] + 0.25 * real["within"] / real["n"]
                scores.setdefault((method, step), []).append(
                    dict(
                        seed=seed,
                        score=score,
                        dose=d["last"]["within"],
                        stops=d["stops"]["within"],
                        within=d["metrics"]["near_target"],
                        price=d["metrics"]["price"]["timing_and_target"],
                        solar=d["metrics"]["solar"]["timing_and_target"],
                        original=real,
                    )
                )
        # Maximise the score of the weaker initialisation. Resolve exact ties by mean, then shorter recipe.
        best = max(
            scores,
            key=lambda k: (
                min(s["score"] for s in scores[k]),
                np.mean([s["score"] for s in scores[k]]),
            ),
        )
        by_width[width] = dict(
            selected=dict(method=best[0], step=best[1]),
            validation={f"{k[0]}-{k[1]}": v for k, v in scores.items()},
        )
    (OUT / "selection.json").write_text(json.dumps(by_width, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    select()
