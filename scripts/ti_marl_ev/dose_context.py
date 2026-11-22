"""Refine power using the reward of two nearby continuous actions and their own continuations."""

import argparse

import numpy as np
import torch

from .dose_fork import update_gradients
from .dose_service import random_window
from .dose_training import BATCH, TrainingCurves, configure, rollout
from .from_scratch import fresh
from .intermediate import OUT, save_model
from .protocol import split

STAGES = (512, 2048, 4096, 8192)


def power_gradient(actor, p, net, need, stored, epsilon):
    args = (actor, p, net, need, stored)
    # Use the actual switching decisions downstream: changing today's charge may avoid a
    # full minimum-power quarter later. Both continuations come from this same actor.
    with torch.no_grad():
        plus, q_plus = rollout(
            *args, forced=1, explore=False, straight=False, first_perturbation=epsilon
        )
        minus, q_minus = rollout(
            *args, forced=1, explore=False, straight=False, first_perturbation=-epsilon
        )
        span = q_plus[:, 0] - q_minus[:, 0]
        slope = torch.where(span > 1e-5, (plus - minus) / span.clamp_min(1e-5), 0.0).clamp(
            -2.0, 2.0
        )
    from .dose_training import curve_features, decision

    _, _, _, power = decision(
        actor,
        curve_features(p, net)[0],
        torch.as_tensor(need),
        torch.as_tensor(stored),
        p.shape[1],
        explore=False,
        temperature=0.0,
        forced=1,
        straight=False,
    )
    active = ((slope > 0) & (power > -6.0)) | ((slope < 0) & (power < 6.0))
    # Natural logit-space step keeps exploration useful even close to either power endpoint.
    return (active * slope * power).mean()


def train(width, seed):
    torch.set_num_threads(1)
    agent = fresh(width, seed)
    actor = agent.actor.cpu().eval()
    configure(actor)
    state = torch.load(
        OUT / f"service-{width}-{seed}-2048.pth", map_location="cpu", weights_only=True
    )
    actor.load_state_dict(state["actor"])
    data = TrainingCurves(split()[0])
    rng = np.random.default_rng(985231)
    power = torch.optim.Adam(actor.ev_watts.parameters(), lr=0.0006)
    gate = torch.optim.Adam(actor.ev_gate.parameters(), lr=0.0006)
    # Restore context to the power head after its local service calibration.
    for step in range(1, max(STAGES) + 1):
        n = int(rng.choice([1, 1, 2, 2, 3, 4, 4, 6, 6, 8, 8, 12, 16, 24, 32, 40]))
        if rng.random() < 0.40:
            p, net, need, stored = data.sample(rng, n, BATCH)
            t = int(rng.integers(n))
        else:
            p, net, need, stored, t = random_window(data, rng, n, BATCH)
        if n >= 2 and rng.random() < 0.28:
            need = rng.uniform(1.0, min(n, 2.45), BATCH)
        if rng.random() < 0.3:
            t = 0
        temp = max(0.012, 0.19 * (1 - step / 5000))
        if t:
            with torch.no_grad():
                _, q = rollout(actor, p, net, need, stored, temperature=temp, straight=False)
            value = q[:, :t].sum(1).numpy()
            need -= value
            stored += value
        price = p[:, t:]
        surplus = net[:, t:]
        rate = 0.0007 if step < 1800 else (0.00036 if step < 5500 else 0.00014)
        power.param_groups[0]["lr"] = rate
        gate.param_groups[0]["lr"] = rate
        update_gradients(actor, gate, power, price, surplus, need, stored, temp)
        if step % 2 == 0:
            epsilon = rng.uniform(0.045, 0.29) if step < 4500 else rng.uniform(0.025, 0.16)
            (0.35 * power_gradient(actor, price, surplus, need, stored, epsilon)).backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(actor.ev_watts.parameters(), 2.0))
        power.step()
        gate.step()
        if step in STAGES:
            save_model(actor, width, seed, "context", step)
            print(step, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    a = p.parse_args()
    train(a.width, a.seed)
