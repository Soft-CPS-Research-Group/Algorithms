"""Train a local continuous EV dose response using only physical rollout returns."""

import argparse

import numpy as np
import torch

from algorithms.ti_marl.experiments.ev_pretraining import CalibratedEVActor

from .dose_service import random_window
from .dose_training import TrainingCurves, rollout
from .intermediate import OUT
from .protocol import split

STEPS = (0, 256, 1024, 2048)


def train(width, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed + 514)
    source = OUT / "models" / "final" / f"v2-{width}-{seed}.pth"
    base = torch.load(source, map_location="cpu", weights_only=True)
    actor = CalibratedEVActor(base["type_registry"], **base["actor_parameters"])
    transformed = {
        k.replace("ev_watts.", "ev_watts.original.", 1) if k.startswith("ev_watts.") else k: v
        for k, v in base["weights"].items()
    }
    missing, extra = actor.load_state_dict(transformed, strict=False)
    assert set(missing) == {"ev_watts.curve", "ev_watts.terminal"} and not extra
    for k, v in base["operating_parameters"].items():
        setattr(actor, k, v)
    actor.eval().requires_grad_(False)
    curve = actor.ev_watts.curve.requires_grad_(True)
    terminal = actor.ev_watts.terminal.requires_grad_(True)
    rng = np.random.default_rng(18146 + seed)
    data = TrainingCurves(split()[0])
    optimiser = torch.optim.Adam([curve, terminal], lr=0.004)
    for t in range(1100):
        p, net, _, stored = data.sample(rng, 1, 192)
        need = rng.uniform(-0.05, 1.99, 192)
        # The conditional power readout experiences its own metered charge at departure.
        # No target power, optimal action or target threshold is supplied as a label.
        cost, _ = rollout(actor, p, net, need, stored, explore=False, forced=1, straight=False)
        optimiser.zero_grad(set_to_none=True)
        cost.mean().backward()
        optimiser.param_groups[0]["lr"] = 0.004 if t < 650 else 0.0012
        optimiser.step()
    # Preserve the one-quarter response already learned from reward. Only the
    # earlier-service knots above one full quarter remain trainable in multi-step RL.
    eligible = torch.arange(len(curve)) > 20
    curve.register_hook(lambda grad: grad * eligible)
    terminal.requires_grad_(False)
    opt = torch.optim.Adam([curve], lr=0.001)
    root = OUT / "calibration"
    root.mkdir(exist_ok=True)
    for step in range(max(STEPS) + 1):
        if step in STEPS:
            payload = dict(
                base,
                adapter="calibrated",
                weights=actor.state_dict(),
                training=dict(
                    base["training"],
                    one_quarter_calibration_episodes=1100 * 192,
                    continuous_response_updates=step,
                    knots_spacing_in_input_kwh=0.05,
                ),
            )
            torch.save(payload, root / f"v2-{width}-{seed}-{step}.pth")
            print(step, flush=True)
        if step == max(STEPS):
            break
        n = int(rng.choice([2, 3, 4, 4, 6, 8, 12, 16, 32, 40]))
        p, net, need, stored, _ = random_window(data, rng, n, 96)
        # Probe the entire two-quarter range of residual service without action labels.
        if rng.random() < 0.72:
            need = rng.uniform(0.35, 2.1, len(need))
        actual, _ = rollout(actor, p, net, need, stored, temperature=0.025, straight=False)
        opt.zero_grad(set_to_none=True)
        actual.mean().backward()
        opt.param_groups[0]["lr"] = 0.001 if step < 1000 else 0.00035
        opt.step()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    a = p.parse_args()
    train(a.width, a.seed)
