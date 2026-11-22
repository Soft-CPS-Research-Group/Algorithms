"""Evaluate one frozen EV policy on non-grid targets and original Core-30 curves."""

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from scripts.train_ti_marl_ev_sessions import SessionProbe, fingerprint

from .from_scratch import evaluation, fresh
from .native_curves import original_curve_sessions
from .protocol import TRACE, split


@torch.no_grad()
def run(checkpoint, levels=None):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    declared = saved.get("power")
    if levels is None:
        levels = declared == "levels"
    elif declared is not None and levels != (declared == "levels"):
        raise ValueError("Requested power representation differs from the training checkpoint")
    agent = fresh(saved["width"], saved["seed"], levels=levels)
    agent.actor.load_state_dict(saved["actor"])
    agent.actor.eval()
    cases = split()[2]
    native = original_curve_sessions(agent.actor, include_cost_reference=False)

    def service(rows):
        errors = [r["error"] for r in rows]
        if levels:
            assert all(
                min(abs(q - x) for x in (0.0, 0.5, 1.0)) < 1e-8
                for r in rows
                for q in r["realised_kwh"]
            )
        return {
            "sessions": len(rows),
            "near": sum(abs(e) <= 0.1 + 1e-8 for e in errors),
            "extra": mean(max(0.0, e) for e in errors),
            "missing": mean(max(0.0, -e) for e in errors),
        }

    return {
        "actor": fingerprint(agent.actor),
        "off_grid": evaluation(agent, SessionProbe(TRACE), cases["off_grid"]),
        "native": {
            str(hour): service([r for r in native if r["hour"] == hour]) for hour in (8, 16)
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--levels",
        action="store_true",
        default=None,
        help="For older discrete checkpoints without power metadata",
    )
    args = p.parse_args()
    args.out.write_text(json.dumps(run(args.checkpoint, args.levels), allow_nan=False))
