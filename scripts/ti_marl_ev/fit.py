"""Fit on training demonstrations. Choose epochs on development sessions only."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from algorithms.ti_marl.experiments.ev_pretraining import FEATURES
from scripts.train_ti_marl_ev_sessions import SessionProbe, evaluate

from .metrics import measures
from .protocol import HERE, TRACE, load, next_price_features, split, wide_sessions


def score(rows):
    m = measures(rows)
    return (
        sum(m[k]["timing_and_target"] for k in ("price", "solar")),
        m["near_target"],
        -m["mae_kwh"],
    )


def fit(width, seed, variant, labels, epochs=160):
    if epochs < 1:
        raise ValueError("At least one supervised epoch is required")
    torch.set_num_threads(1)
    a = load(width, seed, variant)
    if variant == "wide":
        origin = torch.load(
            HERE / f"v2-{width}-boundary-s{seed}-bc.pth", map_location="cpu", weights_only=True
        )["actor"]
        state = a.actor.state_dict()
        for key, v in origin.items():
            if v.shape == state[key].shape:
                state[key] = v
            else:
                assert key in ("ev_gate.0.weight", "ev_watts.0.weight", "ev_band.0.weight")
                state[key].zero_()
                state[key][:, :FEATURES] = v[:, :FEATURES]
                if key != "ev_gate.0.weight":
                    state[key][:, -2:] = v[:, -2:]
        a.actor.load_state_dict(state)
    heads = [a.actor.ev_gate, a.actor.ev_watts]
    if variant in ("boundary", "wide"):
        heads.append(a.actor.ev_band)
    opt = torch.optim.AdamW(
        [p for h in heads for p in h.parameters()], lr=0.0018, weight_decay=0.0001
    )
    HERE.mkdir(parents=True, exist_ok=True)
    records = torch.from_numpy(np.concatenate([np.load(path) for path in labels]))
    if records.ndim != 2 or not len(records) or records.shape[1] != FEATURES + 6:
        raise ValueError("Expected non-empty EV feature and demonstration rows")
    probe = SessionProbe(TRACE)
    _, dev, _ = split()
    x = records[:, :FEATURES]
    lo, hi, base_mode, base_power, free, q = records[:, FEATURES:].T
    if variant == "wide":
        x = torch.cat(
            (x, torch.from_numpy(next_price_features(x.numpy()).astype(np.float32))), dim=1
        )
    extra_dev = (
        [replace(s, early=True) for s in wide_sessions(dev, 39801, 8)] if variant == "wide" else []
    )
    px = torch.cat([x, lo[:, None], hi[:, None]], dim=1)
    learn_power = (q > 1e-6) & (hi - lo > 1e-6)
    u = ((q - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    band = torch.where(u <= 1e-6, 0, torch.where(u >= 1.0 - 1e-6, 2, 1))
    rng = torch.Generator().manual_seed(seed)
    name = f"v2-{width}-{variant}-s{seed}"
    best = None
    log = {}
    for epoch in range(1, epochs + 1):
        for ids in torch.randperm(len(x), generator=rng).split(256):
            g = ids[free[ids] > 0.5]
            gate = a.actor.ev_gate(x[g]).squeeze(-1) + base_mode[g]
            loss = (
                F.binary_cross_entropy_with_logits(gate, (q[g] > 1e-6).float())
                if len(g)
                else gate.sum()
            )
            p = ids[learn_power[ids]]
            if len(p):
                if variant in ("boundary", "wide"):
                    logits = a.actor.ev_band(px[p])
                    loss = loss + F.cross_entropy(logits, band[p])
                    p = p[band[p] == 1]
                if len(p):
                    z = a.actor.ev_watts(px[p]).squeeze(-1) + base_power[p]
                    desired = torch.logit(u[p].clamp(0.0001, 0.9999))
                    loss = (
                        loss
                        + 0.3 * F.smooth_l1_loss(z, desired)
                        + 5 * F.mse_loss(z.sigmoid(), u[p])
                    )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for h in heads for p in h.parameters()], 5.0)
            opt.step()
        if epoch in (25, 60, 100, epochs):
            with torch.no_grad():
                rows = evaluate(dev, a, None, probe)
            key = score(rows)
            if extra_dev:
                m = measures(evaluate(extra_dev, a, None, probe))
                key = (
                    min(key[0] / len(dev), m["price"]["timing_and_target"] / len(extra_dev)),
                    key[0] + m["price"]["timing_and_target"],
                    key[1],
                    key[2] - m["mae_kwh"],
                )
            log[epoch] = {"score": key, "metrics": measures(rows)}
            print(name, epoch, key, flush=True)
            if best is None or key > best:
                best = key
                torch.save(
                    {
                        "width": width,
                        "seed": seed,
                        "variant": variant,
                        "epoch": epoch,
                        "actor": a.actor.state_dict(),
                        "critic": a.critic.state_dict(),
                    },
                    HERE / (name + "-bc.pth"),
                )
    (HERE / (name + "-fit.json")).write_text(json.dumps(log, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("variant", choices=("continuous", "boundary", "wide"))
    p.add_argument(
        "labels",
        type=Path,
        nargs="+",
        help="One or more compatible demonstration files for this width",
    )
    p.add_argument("--epochs", type=int, default=160)
    o = p.parse_args()
    fit(o.width, o.seed, o.variant, o.labels, o.epochs)
