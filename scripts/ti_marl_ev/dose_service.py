"""Reward-only EV power restricted to remaining service, current solar and legal bounds.

Train the switch with the original outlook, including short low-cost blocks inside
long connections. No target-action schedule is provided to either readout.
"""

import argparse

import numpy as np
import torch

from .dose_fork import update_gradients
from .dose_training import BATCH, TrainingCurves, configure, rollout
from .from_scratch import fresh
from .intermediate import OUT, save_model
from .protocol import split

STAGES = (512, 2048, 4096, 8192)


def random_window(data, rng, n, b):
    p, net, need, stored = data.sample(rng, n, b)
    if n < 4 or rng.random() < 0.28:
        return p, net, need, stored, int(rng.integers(n))
    low, high, pv = data.params[rng.integers(len(data.params), size=b)].T
    length = int(rng.integers(2, min(n - 1, 12) + 1))
    start = int(rng.integers(n - length + 1))
    end = start + length
    window = (np.arange(n) >= start) & (np.arange(n) < end)
    kind = rng.integers(3, size=b)
    p = np.where(window & (kind[:, None] != 1), low[:, None], high[:, None])
    net = np.where(window & (kind[:, None] != 0), pv[:, None], 0.0)
    need = rng.uniform(0.35, min(5.8, n - 0.05), b)
    small = rng.random(b) < 0.4
    need[small] = rng.uniform(0.35, 1.5, int(small.sum()))
    need[rng.random(b) < 0.08] = 0.0
    visit = int(
        rng.choice([0, 0, start, end - 1, max(start, end - 2), min(end, n - 1), rng.integers(n)])
    )
    return p, net, need, stored, visit


def train(width, seed, mixed=False):
    torch.set_num_threads(1)
    agent = fresh(width, seed)
    actor = agent.actor.cpu().eval()
    configure(actor)
    prior = OUT / "models" / f"v2-{width}-{seed}.pth"
    actor.load_state_dict(torch.load(prior, map_location="cpu", weights_only=True)["weights"])
    if not mixed:
        layer = actor.ev_watts[0]
        keep = torch.zeros_like(layer.weight)
        keep[:, [0, 5, -2, -1]] = 1.0
        with torch.no_grad():
            layer.weight.mul_(keep)
        layer.weight.register_hook(lambda grad: grad * keep)
    rng = np.random.default_rng(840074 if mixed else 840073)
    data = TrainingCurves(split()[0])
    power = torch.optim.Adam(actor.ev_watts.parameters(), lr=0.001)
    gate = torch.optim.Adam(actor.ev_gate.parameters(), lr=0.001)
    for k in range(0 if mixed else 700):
        # A self-chosen continuous power is billed and settled at a real terminal event.
        p, net, need, stored = data.sample(rng, 1, BATCH)
        need = rng.uniform(0.01, 1.8, BATCH)
        cost, _ = rollout(
            actor,
            p,
            net,
            need,
            stored,
            forced=1,
            straight=False,
            temperature=max(0.01, 0.15 - k / 4500),
        )
        power.zero_grad(set_to_none=True)
        cost.mean().backward()
        torch.nn.utils.clip_grad_norm_(actor.ev_watts.parameters(), 2.0)
        power.step()
    for step in range(1, max(STAGES) + 1):
        n = int(rng.choice([1, 1, 2, 4, 6, 8, 8, 12, 16, 24, 32, 40, 40]))
        if mixed and step % 2 == 0:
            n = int(rng.choice([1, 1, 2, 4, 4, 6, 8, 8, 16, 32, 40]))
            p, net, need, stored = data.sample(rng, n, BATCH)
            t = int(rng.integers(n)) if rng.random() < 0.5 else 0
        else:
            p, net, need, stored, t = random_window(data, rng, n, BATCH)
        temp = (
            max(0.012, 0.28 * (1.0 - step / 4300))
            if not mixed
            else max(0.012, 0.12 * (1.0 - step / 3400))
        )
        if rng.random() < 0.22:
            t = 0
        if t:
            with torch.no_grad():
                _, qs = rollout(actor, p, net, need, stored, temperature=temp, straight=False)
            delivered = qs[:, :t].sum(1).numpy()
            need -= delivered
            stored += delivered
        rate = 0.0008 if step < 2200 else (0.00036 if step < 5300 else 0.00013)
        if mixed:
            rate = 0.00045 if step < 2200 else (0.00024 if step < 5200 else 0.00009)
        power.param_groups[0]["lr"] = rate
        gate.param_groups[0]["lr"] = rate
        update_gradients(actor, gate, power, p[:, t:], net[:, t:], need, stored, temp)
        power.step()
        gate.step()
        if step in STAGES:
            if not mixed:
                assert torch.count_nonzero(layer.weight * (1.0 - keep)) == 0
            save_model(actor, width, seed, "mixed" if mixed else "service", step)
            print(step, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--mixed", action="store_true")
    a = p.parse_args()
    train(a.width, a.seed, a.mixed)
