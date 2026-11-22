"""Out-of-distribution EV replay on original tariff, non-shiftable load and PV series.

Perfect future curves and artificial service requests are deliberate. Evaluate a
frozen exported policy with no updates. Optimise the same requested energy for
the cost comparator; shorter delivery cannot be credited as a cost saving.
"""

import json
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

from algorithms.ti_marl.experiments.ev_pretraining import load_ev_policy
from scripts.train_ti_marl_ev_sessions import SessionProbe

from .protocol import HERE, TRACE, load, schema_path, split


def cost(q, tariff, load, pv):
    return float(tariff @ (np.maximum(0.0, load + q - pv) - np.maximum(0.0, load - pv)))


def immediate_schedule(energy, quarters):
    """Earliest legal plug-in delivery of the same energy, including the 1.4 kW minimum."""
    delivered = min(float(energy), float(quarters))
    if delivered < -1e-8 or energy > quarters + 1e-8 or 1e-8 < delivered < 0.35 - 1e-8:
        raise ValueError("This amount cannot be delivered by the isolated port")
    q = np.zeros(quarters)
    whole = int(max(0.0, delivered))
    q[:whole] = 1.0
    tail = delivered - whole
    if tail > 1e-8:
        q[whole] = max(0.35, tail)
        if tail < 0.35 and whole:
            q[whole - 1] -= 0.35 - tail
    assert abs(q.sum() - energy) < 1e-7
    return q


def least_cost(tariff, load, solar, need):
    n = len(tariff)
    eye = np.eye(n)
    z = np.zeros_like(eye)
    pr = np.r_[np.zeros(2 * n), tariff]
    # Variables: EV energy, activation, grid imports. Retain the physical 1.4 kW minimum.
    A = np.vstack(
        [
            np.c_[eye, -0.35 * eye, z],
            np.c_[-eye, eye, z],
            np.c_[-eye, z, eye],
            np.r_[np.ones(n), np.zeros(2 * n)],
        ]
    )
    lb = np.r_[np.zeros(2 * n), load - solar, need]
    result = milp(
        pr,
        integrality=np.r_[np.zeros(n), np.ones(n), np.zeros(n)],
        bounds=Bounds(np.zeros(3 * n), np.r_[np.ones(2 * n), np.full(n, np.inf)]),
        constraints=LinearConstraint(A, lb, np.r_[np.full(3 * n, np.inf), need]),
        options={"mip_rel_gap": 0},
    )
    assert result.success
    return cost(result.x[:n], tariff, load, solar)


@torch.no_grad()
def run(width, seed, phase="final"):
    torch.set_num_threads(1)
    if phase == "final":
        policy = load_ev_policy(HERE / "models" / f"v2-{width}-s{seed}.pth")
    else:
        winner = json.loads((HERE / "selection.json").read_text())[str(width)]["winner"]
        name = f"v2-{width}-{winner['variant']}-s{seed}"
        suffix = {"pretrained": "-bc", "rl_only": f"-cold-lr{winner['rate']:g}-1536"}[phase]
        policy = load(width, seed, winner["variant"]).actor
        ckpt = torch.load(HERE / (name + suffix + ".pth"), map_location="cpu", weights_only=True)
        policy.load_state_dict(ckpt["actor"])
        policy.eval()
    sessions = original_curve_sessions(policy)
    (HERE / f"v2-{width}-s{seed}-{phase}-native.json").write_text(
        json.dumps(sessions, allow_nan=False)
    )


@torch.no_grad()
def original_curve_sessions(
    policy, *, include_cost_reference=True, conditions=None, variable_need=False, need_seed=951815
):
    """Replay a frozen research policy on measured curves and controlled EV requests."""
    schema = schema_path()
    cfg = json.loads(schema.read_text())
    h = cfg["buildings"]["Member_001"]
    root = schema.parent / cfg["root_directory"]
    p = pd.read_parquet(root / h["pricing"], columns=["electricity_pricing"]).iloc[:, 0].to_numpy()
    raw = pd.read_parquet(
        root / h["energy_simulation"], columns=["non_shiftable_load", "solar_generation"]
    )
    d = raw["non_shiftable_load"].to_numpy()
    pv = raw["solar_generation"].to_numpy()
    probe = SessionProbe(TRACE)
    tz = ZoneInfo(cfg["timezone"])
    origin = datetime(2023, 1, 1, tzinfo=tz)
    sessions = []
    _, _, sets = split()
    if conditions is None:
        conditions = sets["new"]
    rng = np.random.default_rng(need_seed)
    for c in [s for s in conditions if s.variable == "price" and s.day > "2023-09-01"]:
        for hour, n, need in ((8, 40, 3.0), (16, 16, 3.23)):
            if variable_need:
                need = float(rng.uniform(1.4, 4.6))
            end = datetime.fromisoformat(c.day + f"T{hour:02d}:00:00").replace(tzinfo=tz)
            offset = round((end.timestamp() - origin.timestamp()) / 900) - n
            sl = slice(offset, offset + n)
            assert p[sl].min() >= 0.0 and pv[sl].min() >= 0
            session = replace(c, start=offset, quarters=n, arrival_kwh=10.0, request_kwh=need)
            q = []
            if hasattr(policy, "start_session"):
                policy.start_session()
            stored = 10.0
            for t in range(n):
                i = offset + t
                obs = probe.snapshot(
                    session, False, t, stored, (10.0 + need) / 20.0, start_step=offset, quarters=n
                )
                (curve,) = obs.outlooks
                f = len(curve.values[0])
                sub = slice(i, i + f)
                curves = tuple(tuple(map(float, x)) for x in (p[sub], d[sub] * 4, pv[sub] * 4))
                assert all(len(v) == f for v in curves)
                current = {
                    "electricity_pricing": p[i],
                    "pv_power_kw": pv[i] * 4,
                    "non_shiftable_load": d[i] * 4,
                }
                parts = tuple(
                    replace(z, values=(float(current[z.observation_id]),))
                    if z.observation_id in current
                    else z
                    for z in obs.observation_parts
                )
                obs = replace(
                    obs, outlooks=(replace(curve, values=curves),), observation_parts=parts
                )
                result = policy(obs, deterministic=True)
                assert not result.traces[probe.site].supervised
                action = result.executed[0].decisions[0]
                applied = action.fraction if action.mode == "CHARGE_EV" else 0.0
                q.append(applied)
                stored += applied
            immediate = immediate_schedule(need, n)
            matched_energy = immediate_schedule(sum(q), n)
            actual = cost(np.array(q), p[sl], d[sl], pv[sl])
            instant = cost(immediate, p[sl], d[sl], pv[sl])
            opt = least_cost(p[sl], d[sl], pv[sl], need) if include_cost_reference else None
            if opt is not None:
                assert opt <= instant + 1e-7
            sessions.append(
                dict(
                    day=c.day,
                    hour=hour,
                    requested=need,
                    applied=sum(q),
                    error=sum(q) - need,
                    cost=actual,
                    immediate=instant,
                    optimum=opt,
                    equal_energy_immediate_cost=cost(matched_energy, p[sl], d[sl], pv[sl]),
                    equal_energy_immediate_solar=float(
                        np.minimum(matched_energy, np.maximum(0.0, pv[sl] - d[sl])).sum()
                    ),
                    solar_kwh=float(np.minimum(q, np.maximum(0.0, pv[sl] - d[sl])).sum()),
                    immediate_solar_kwh=float(
                        np.minimum(immediate, np.maximum(0.0, pv[sl] - d[sl])).sum()
                    ),
                    realised_kwh=q,
                )
            )
    return sessions


if __name__ == "__main__":
    import sys

    run(int(sys.argv[1]), int(sys.argv[2]), *sys.argv[3:])
