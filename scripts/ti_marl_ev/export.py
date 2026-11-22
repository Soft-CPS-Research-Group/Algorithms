"""Reload a standalone actor and replay held-out rollout decisions and their densities."""

import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch

from algorithms.ti_marl.experiments.ev_pretraining import FORMAT, load_ev_policy
from scripts.train_ti_marl_ev_sessions import Session, SessionProbe

from .evaluate import Night
from .protocol import HERE, PRE, TRACE, load


@torch.no_grad()
def check(width, seed):
    torch.set_num_threads(1)
    report = json.loads((HERE / f"v2-{width}-s{seed}-sealed.json").read_text())
    winner = report["choice"]
    variant = winner["variant"]
    tag = f"v2-{width}-{variant}-s{seed}-bc-lr{winner['rate']:g}"
    study_agent = load(width, seed, variant)
    base = torch.load(PRE / "models" / f"v2-{width}.pth", map_location="cpu", weights_only=False)
    flat = dict(base["configuration"]["algorithm"]["hyperparameters"]["actor"])
    flat.update(variant="v2", group_context_kind="action_conditioned")
    weights = torch.load(HERE / (tag + "-1536.pth"), map_location="cpu", weights_only=True)["actor"]
    root = HERE / "models"
    root.mkdir(exist_ok=True)
    dest = root / f"v2-{width}-s{seed}.pth"
    torch.save(
        dict(
            format=FORMAT,
            type_registry=study_agent.compiler.type_registry,
            actor_parameters=flat,
            adapter=variant,
            weights=weights,
            operating_parameters={
                k: getattr(study_agent.actor, k)
                for k in ("step_seconds", "departure_margin_seconds", "tolerance_soc", "order_seed")
            },
        ),
        dest,
    )
    policy = load_ev_policy(dest)
    ref = report["stages"]["rl1536"]
    probe = SessionProbe(TRACE)
    count = 0
    max_error = 0.0
    max_logp = 0.0
    draws = 0
    stages = []
    for rows in ref["rows"].values():
        for row in rows:
            for early in (True, False):
                params = dict(row["session"])
                params["early"] = early
                stages.append((Session(**params), row["early" if early else "late"]["kwh"]))
    for row in ref["overnight"]:
        tz = ZoneInfo("Europe/Lisbon")
        end = datetime.fromisoformat(row["day"] + "T08:00:00").replace(tzinfo=tz)
        start = round((end.timestamp() - datetime(2023, 1, 1, tzinfo=tz).timestamp()) / 900) - 40
        p = next(
            r["session"]
            for r in ref["rows"]["new"]
            if r["session"]["day"] == row["day"] and r["session"]["variable"] == "price"
        )
        for early in (True, False):
            stages.append(
                (
                    Night(
                        row["day"],
                        "08:00",
                        start,
                        40,
                        10.0,
                        3.0,
                        p["low"],
                        p["high"],
                        0.0,
                        "price",
                        early,
                    ),
                    row["early" if early else "late"]["kwh"],
                )
            )
    torch.manual_seed(521)
    for case, expected in stages:
        stored = case.arrival_kwh
        target = (stored + case.request_kwh) / 20.0
        for t, x in enumerate(expected):
            obs = probe.snapshot(
                case, False, t, stored, target, start_step=case.start, quarters=case.quarters
            )
            result = policy(obs, deterministic=True)
            assert not result.traces[probe.site].supervised
            decision = result.executed[0].decisions[0]
            q = decision.fraction if decision.mode == "CHARGE_EV" else 0.0
            error = abs(q - x)
            max_error = max(max_error, error)
            count += 1
            assert error < 2e-5, (width, seed, case, t, q, x)
            stored += q
            if t == 0 and draws < 80:
                drawn = policy(obs, deterministic=False)
                scored = policy.replay(obs, drawn.traces)[probe.site]
                for i, new, entropy in scored:
                    d = abs(float(new) - drawn.traces[probe.site].choices[i].old_log_prob)
                    max_logp = max(max_logp, d)
                    assert np.isfinite(float(entropy)) and d < 1e-5
                draws += 1
    (HERE / f"v2-{width}-s{seed}-reload.json").write_text(
        json.dumps(
            {
                "decisions": count,
                "max_kwh_error": max_error,
                "stochastic_replays": draws,
                "max_logp_error": max_logp,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    o = p.parse_args()
    check(o.width, o.seed)
