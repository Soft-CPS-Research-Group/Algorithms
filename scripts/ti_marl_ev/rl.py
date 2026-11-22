"""Teacher-free V2 PPO continuation. Only realised native rewards reach PPO."""

import argparse
import hashlib
import json
import sys

import numpy as np
import torch

from scripts.train_ti_marl_ev_sessions import SessionProbe, evaluate, fingerprint, make_cases, play

from .protocol import HERE, TRACE, load, schema_path, split


def run(width, seed, variant, lr_factor, start):
    HERE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    a = load(width, seed, variant)
    probe = SessionProbe(TRACE)
    name = f"v2-{width}-{variant}-s{seed}"
    if start == "bc":
        ckpt = torch.load(HERE / (name + "-bc.pth"), map_location="cpu", weights_only=True)
        a.actor.load_state_dict(ckpt["actor"])
        a.critic.load_state_dict(ckpt["critic"])
    pre_hash = fingerprint(a.actor)
    hashes = [fingerprint(a.actor.ev_gate), fingerprint(a.actor.ev_watts)]
    for group in a.learner.actor_optimizer.param_groups:
        group["lr"] *= lr_factor
    train, dev, sets = split()
    days = {s.day for s in train}
    extra, _ = make_cases(schema_path(), 4500, 48)
    # Use new sessions from the fit dates. Never train on development or test dates.
    sessions = [s for s in extra[1536:] if s.day in days][:1536]
    assert len(sessions) == 1536 and not days & {s.day for s in sets["new"]}
    order = np.random.default_rng(seed).permutation(len(sessions))
    name += f"-{start}-lr{lr_factor:g}"
    counts = {"steps": 0, "free": 0, "free_charge": 0, "updates": 0}
    log = {
        "before_hash": pre_hash,
        "gates_before": hashes,
        "reward": a.reward_profile,
        "penalties": a.reward_penalties.model_dump(),
        "reference_sites": a.reference_sites,
        "order_sha256": hashlib.sha256(
            json.dumps([str(sessions[j]) for j in order]).encode()
        ).hexdigest(),
        "local_actor_lr": a.learner.actor_optimizer.param_groups[0]["lr"],
        "milestones": {},
    }
    for i, idx in enumerate(order, 1):
        row = play(sessions[idx], a, None, probe, "local")
        counts["steps"] += len(row["kwh"])
        counts["free"] += row["free"]
        counts["free_charge"] += row["free_charge"]
        if i % 4 == 0:
            assert a.learner.rollout.steps[-1].terminated
            a.actor.train()
            a.critic.train()
            info = a.learner.update()
            assert all(np.isfinite(v) for v in info.values()), (i, info)
            counts["updates"] += 1
        if i in (384, 1536):
            rows = evaluate(dev, a, None, probe)
            post = [fingerprint(a.actor.ev_gate), fingerprint(a.actor.ev_watts)]
            assert all(x != y for x, y in zip(hashes, post))
            log["milestones"][i] = {
                "counts": dict(counts),
                "dev": rows,
                "actor_hash": fingerprint(a.actor),
                "head_hash": post,
            }
            torch.save(
                {
                    "actor": a.actor.state_dict(),
                    "critic": a.critic.state_dict(),
                    "variant": variant,
                    "width": width,
                    "seed": seed,
                },
                HERE / (name + f"-{i}.pth"),
            )
            print(name, i, counts, flush=True)
    assert not any(x.split(".")[-1] in ("control", "demonstrations") for x in sys.modules)
    (HERE / (name + ".json")).write_text(json.dumps(log, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("variant")
    p.add_argument("lr", type=float)
    p.add_argument("start", choices=("bc", "cold"))
    v = p.parse_args()
    run(v.width, v.seed, v.variant, v.lr, v.start)
