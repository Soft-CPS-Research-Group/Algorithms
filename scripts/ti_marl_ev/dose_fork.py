"""Improve the switch using two continuations of the same reward-only EV policy.

The switch is trained by a Bernoulli natural policy gradient: its direction
comes from the cost difference of charge and idle with the current continuation.
Power is trained by differentiating the operating cost through metered energy.
"""

import argparse

import numpy as np
import torch

from .dose_training import BATCH, TrainingCurves, configure, rollout
from .from_scratch import fresh
from .intermediate import OUT, save_model
from .protocol import split

START = 49152
CHECKS = (256, 1024, 2048, 4096)


def update_gradients(actor, opt, watts, p, net, need, stored, temp):
    args = (actor, p, net, need, stored)
    # Common random numbers cancel unrelated future exploration when comparing the switch.
    random_state = torch.get_rng_state()
    on, _, logit, free = rollout(
        *args, temperature=temp, forced=1, straight=False, return_switch=True
    )
    torch.set_rng_state(random_state)
    off, _ = rollout(*args, temperature=temp, forced=0, straight=False)
    advantage = (on - off).detach()
    prob = logit.sigmoid().detach()
    active = free & (((advantage > 0) & (logit > -5.0)) | ((advantage < 0) & (logit < 5.0)))
    # The natural gradient cancels the Bernoulli p*(1-p) factor. Cap further
    # reinforcement of already-confident decisions, retaining the reverse gradient.
    mode_loss = (active * logit * advantage).mean()
    power_loss = (prob * on + (1 - prob) * off).mean()
    opt.zero_grad(set_to_none=True)
    watts.zero_grad(set_to_none=True)
    (mode_loss + power_loss).backward()
    for head in (actor.ev_gate, actor.ev_watts):
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
        assert torch.isfinite(norm)


def train(width, seed, polish=False, *, max_updates=None):
    torch.set_num_threads(1)
    agent = fresh(width, seed)
    actor = agent.actor.cpu().eval()
    configure(actor)
    prior = f"fork-{width}-{seed}-4096" if polish else f"gradient-{width}-{seed}-{START}"
    actor.load_state_dict(
        torch.load(OUT / f"{prior}.pth", map_location="cpu", weights_only=True)["actor"]
    )
    data = TrainingCurves(split()[0])
    rng = np.random.default_rng(702146 if polish else 702145)
    opt = torch.optim.Adam(actor.ev_gate.parameters(), lr=0.0015)
    watts = torch.optim.Adam(actor.ev_watts.parameters(), lr=0.0015)
    if max_updates is None:
        max_updates = max(CHECKS)
    if max_updates not in CHECKS:
        raise ValueError("Select a recorded training duration")
    for step in range(1, max_updates + 1):
        horizons = (
            [1, 1, 1, 1, 2, 4, 4, 6, 8, 12, 16, 32, 40]
            if polish
            else [1, 1, 2, 4, 4, 6, 6, 8, 8, 12, 16, 32, 40]
        )
        n = int(rng.choice(horizons))
        p, net, need, stored = data.sample(rng, n, BATCH)
        # Visit both arrival and on-policy intermediate states, including late residual demand.
        t = 0 if rng.random() < 0.45 else int(rng.integers(n))
        temp = max(0.10, 0.7 * (1.0 - step / 2200))
        if polish:
            temp = max(0.012, 0.12 * (1.0 - step / 3200))
        if t:
            with torch.no_grad():
                _, q = rollout(actor, p, net, need, stored, temperature=temp, straight=False)
            added = q[:, :t].sum(1).numpy()
            need = need - added
            stored = stored + added
        update_gradients(actor, opt, watts, p[:, t:], net[:, t:], need, stored, temp)
        rate = 0.0015 if step < 1000 else (0.0007 if step < 2600 else 0.00028)
        if polish:
            rate = 0.0004 if step < 1100 else (0.00018 if step < 2700 else 0.00007)
        opt.param_groups[0]["lr"] = rate
        watts.param_groups[0]["lr"] = rate
        opt.step()
        watts.step()
        if step in CHECKS:
            save_model(actor, width, seed, "polish" if polish else "fork", step)
            print(step, "saved", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--polish", action="store_true")
    p.add_argument("--max-updates", type=int, choices=CHECKS)
    a = p.parse_args()
    train(a.width, a.seed, a.polish, max_updates=a.max_updates)
