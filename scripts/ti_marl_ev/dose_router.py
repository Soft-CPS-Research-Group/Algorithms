"""Optional session-level contextual bandit over two reward-trained EV readouts."""

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from algorithms.ti_marl.experiments.ev_pretraining import SessionEVActor, entry_features

from .dose_service import random_window
from .dose_training import TrainingCurves, rollout
from .intermediate import OUT
from .protocol import split


def train(width, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed + 436)
    primary = torch.load(
        OUT / "models" / f"v2-{width}-{seed}.pth", map_location="cpu", weights_only=True
    )
    spec = torch.load(
        OUT / f"service-{width}-{seed}-8192.pth", map_location="cpu", weights_only=True
    )["actor"]
    actor = SessionEVActor(primary["type_registry"], **primary["actor_parameters"])
    missing, extra = actor.load_state_dict(primary["weights"], strict=False)
    assert not extra and all(k.startswith(("other_", "entry_router.")) for k in missing)
    for name, head in [("ev_gate.", actor.other_gate), ("ev_watts.", actor.other_watts)]:
        head.load_state_dict({k[len(name) :]: v for k, v in spec.items() if k.startswith(name)})
    for key, val in primary["weights"].items():
        if not key.startswith(("ev_gate.", "ev_watts.")):
            assert torch.equal(val, spec[key]), key
    for key, value in primary["operating_parameters"].items():
        setattr(actor, key, value)
    actor.eval().requires_grad_(False)
    actor.entry_router.requires_grad_(True)
    opt = torch.optim.Adam(actor.entry_router.parameters(), lr=0.003)
    data = TrainingCurves(split()[0])
    rng = np.random.default_rng(387251)
    other = SimpleNamespace(ev_gate=actor.other_gate, ev_watts=actor.other_watts)
    # Both actions are already learned policies. Roll both under their own switches and power.
    # Optimise the expected operating cost directly; no assignment or action labels are generated.
    for step in range(2048):
        n = int(rng.choice([1, 2, 4, 4, 6, 8, 8, 12, 16, 24, 32, 40]))
        if rng.random() < 0.5:
            p, net, need, store = data.sample(rng, n, 96)
        else:
            p, net, need, store, _ = random_window(data, rng, n, 96)
        with torch.no_grad():
            base, _ = rollout(actor, p, net, need, store, explore=False)
            alt, _ = rollout(other, p, net, need, store, explore=False)
        logits = actor.entry_router(torch.from_numpy(entry_features(p, net, need))).flatten()
        diff = alt - base
        active = ((diff > 0) & (logits > -6.0)) | ((diff < 0) & (logits < 6.0))
        loss = (active * diff * logits).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dest = OUT / "models" / "session"
    dest.mkdir(exist_ok=True)
    primary.update(adapter="session", weights=actor.state_dict())
    primary["training"].update(
        secondary="service-8192",
        entry_bandit_episodes=2048 * 96,
        entry_bandit_continuations_per_episode=2,
    )
    torch.save(primary, dest / f"v2-{width}-{seed}.pth")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    a = p.parse_args()
    train(a.width, a.seed)
