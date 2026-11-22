"""Adapt the EV switch to the learned, frozen continuous dose using its own returns."""

import argparse

import numpy as np
import torch

from algorithms.ti_marl.experiments.ev_pretraining import load_ev_policy

from .dose_fork import update_gradients
from .dose_service import random_window
from .dose_training import TrainingCurves, rollout
from .intermediate import OUT
from .protocol import split

STAGES = (512, 2048, 4096)


def train(width, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed + 959)
    source = OUT / "calibration" / f"v2-{width}-{seed}-2048.pth"
    actor = load_ev_policy(source).requires_grad_(False)
    actor.ev_gate.requires_grad_(True)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    data = TrainingCurves(split()[0])
    rng = np.random.default_rng(482991)
    gate = torch.optim.Adam(actor.ev_gate.parameters(), lr=0.00045)
    for step in range(1, max(STAGES) + 1):
        n = int(rng.choice([2, 4, 4, 6, 8, 8, 16, 16, 24, 32, 40]))
        p, net, need, stored, t = random_window(data, rng, n, 96)
        fraction = rng.random()
        if fraction < 0.33:
            p, net, need, stored = data.sample(rng, n, 96)
        elif fraction < 0.65:
            ends = (
                rng.choice(data.nights, 96) + rng.integers(26, 38, 96)
                if n > 24
                else rng.choice(data.days, 96) + rng.integers(n, 95, 96)
            )
            ix = ends[:, None] - n + np.arange(n)[None, :]
            p, net = data.price[ix], data.surplus[ix]
            need = rng.uniform(0.35, min(5.0, float(n)), 96)
        if fraction < 0.65:
            t = int(rng.integers(n))
        if rng.random() < 0.3:
            t = 0
        temp = max(0.015, 0.16 * (1 - step / 2500))
        if t:
            with torch.no_grad():
                _, q = rollout(actor, p, net, need, stored, temperature=temp, straight=False)
            v = q[:, :t].sum(1).numpy()
            need -= v
            stored += v
        gate.param_groups[0]["lr"] = (
            0.00055 if step < 1300 else (0.00027 if step < 2600 else 0.00012)
        )
        # Only the switch is trainable; the helper clears both readout gradients.
        update_gradients(actor, gate, gate, p[:, t:], net[:, t:], need, stored, temp)
        gate.step()
        if step in STAGES:
            payload["weights"] = actor.state_dict()
            payload["training"]["post_calibration_switch_updates"] = step
            torch.save(payload, OUT / "calibration" / f"switch-{width}-{seed}-{step}.pth")
            print(step, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    a = p.parse_args()
    train(a.width, a.seed)
