"""Reward-only, differentiable single-EV training of the V2 continuous readouts.

Forward actions use the native single-port domain. A straight-through relaxation
of the binary switch is used only for its derivative. Device-level energy,
cost and departure penalties provide the whole learning signal.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from .from_scratch import fresh
from .intermediate import PRICE, save_model
from .protocol import schema_path, split

BATCH = 96
MILESTONES = (32, 128, 512, 2048, 4096)
GAMMA = 0.99**0.25


class TrainingCurves:
    def __init__(self, cases):
        self.cases = cases
        path = schema_path()
        cfg = json.loads(path.read_text())
        member = cfg["buildings"]["Member_001"]
        root = path.parent / cfg["root_directory"]
        p = pd.read_parquet(root / member["pricing"], columns=["electricity_pricing"])
        s = pd.read_parquet(
            root / member["energy_simulation"], columns=["non_shiftable_load", "solar_generation"]
        )
        self.price = p.iloc[:, 0].to_numpy(dtype=float)
        self.surplus = (s["solar_generation"] - s["non_shiftable_load"]).to_numpy(dtype=float)
        zone = cfg["timezone"]
        start = pd.Timestamp("2023-01-01", tz=zone)
        midnights = [pd.Timestamp(day, tz=zone) for day in sorted({c.day for c in cases})]
        self.days = np.array([round((t - start).total_seconds() / 900) for t in midnights])
        days = set(self.days)
        self.nights = np.array([t for t in self.days if t - 96 in days])
        self.params = np.array([(c.low, c.high, c.pv_kw / 4) for c in cases])

    def sample(self, rng, n, b):
        low, high, peak = self.params[rng.integers(len(self.params), size=b)].T
        early = rng.random(b) < 0.5
        kind = rng.integers(0, 3, size=b)
        clock = np.arange(n)[None, :]
        window = np.where(early[:, None], clock < n // 2, clock >= n // 2)
        price = np.where(window & (kind != 1)[:, None], low[:, None], high[:, None])
        surplus = np.where(window & (kind != 0)[:, None], peak[:, None], 0.0)
        original = rng.random(b) < 0.28
        if n > 16:
            ends = rng.choice(self.nights, b) + rng.choice([24, 28, 32, 36], b)
        else:
            ends = rng.choice(self.days, b) + rng.integers(n, 96, size=b)
        idx = ends[:, None] - n + clock
        price[original] = self.price[idx[original]]
        surplus[original] = self.surplus[idx[original]]
        if n == 1:
            target = rng.uniform(0.35, 1.0, b)
        else:
            target = rng.uniform(0.35, min(6.0, n - 0.05), b)
            focus = rng.random(b) < 0.45
            target[focus] = np.clip(
                min(4.0, n / 2) + rng.uniform(-0.48, 1.48, focus.sum()), 0.35, n
            )
        stop = rng.random(b) < 0.08
        target[stop] = rng.choice([0.0, -0.25], stop.sum())
        return price, surplus, target, rng.uniform(7.0, 11.0, b)


def curve_features(price, net):
    """Exogenous parts of LongEVActor fields. net is net solar in kWh per quarter."""
    clipped = np.clip(net, 0.0, 1.0)
    entries = []
    for t in range(price.shape[1]):
        p = price[:, t:]
        s = clipped[:, t:]
        now = p[:, 0]
        others = p[:, 1:]
        entries.append(
            torch.from_numpy(
                np.column_stack(
                    (
                        now / 0.1,
                        (now - p.mean(1)) / 0.1,
                        s[:, 0],
                        s[:, 0] - s.mean(1),
                        (now - p.min(1)) / 0.1,
                        (p.max(1) - now) / 0.1,
                        s.max(1) - s[:, 0],
                        (others < now[:, None]).sum(1),
                        (others <= now[:, None]).sum(1),
                        s[:, 1:].sum(1),
                    )
                )
            ).float()
        )
    return entries


def legal_interval(need, stored, n):
    """Exact single-port charge interval after V2's earliest-coverage reservation."""
    protected = (need - 1.0).clamp_min(0.0)
    idle_allowed = protected <= n - 1 + 1e-8
    charge_allowed = stored <= 20.0 - 0.35 + 1e-8
    slots = torch.ceil(protected - 1.0 - 1e-8).clamp_min(0.0)
    low = (protected - slots).clamp_min(0.35)
    high = (20.0 - stored).clamp(max=1.0)
    return idle_allowed, charge_allowed, low, high


def decision(
    actor, row, need, stored, n, *, explore, temperature, forced=None, straight=True, perturb=0.0
):
    (
        current_price,
        price_vs_mean,
        available_solar,
        solar_vs_mean,
        price_above_min,
        price_below_max,
        solar_below_max,
        cheaper_quarters,
        no_dearer_quarters,
        later_solar,
    ) = row.unbind(1)
    remaining_kwh = need.float()
    fields = torch.stack(
        (
            remaining_kwh,
            torch.full_like(remaining_kwh, n / 4.0),
            (n - remaining_kwh) / 4.0,
            current_price,
            price_vs_mean,
            available_solar,
            solar_vs_mean,
            (remaining_kwh <= 1e-7).float(),
            price_above_min,
            price_below_max,
            solar_below_max,
            (remaining_kwh - cheaper_quarters).clamp(-6, 6),
            (remaining_kwh - no_dearer_quarters).clamp(-6, 6),
            later_solar.clamp(max=6.0),
            (remaining_kwh - later_solar).clamp(-6, 6),
        ),
        dim=1,
    )
    idle, charge, lo, hi = legal_interval(need, stored, n)
    logit = actor.ev_gate(fields).squeeze(1)
    base_logit = logit
    if explore:
        z = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
        logit = logit + temperature * torch.logit(z)
    prob = torch.sigmoid(logit / 0.7)
    switch = (logit > 0).float() + (prob - prob.detach())
    if not straight:
        switch = switch.detach()
    if forced is not None:
        switch = torch.full_like(switch, float(forced))
    switch = torch.where(idle, switch, 1.0)
    switch = torch.where(charge, switch, 0.0)
    pred = actor.ev_watts(torch.cat((fields, lo.float()[:, None], hi.float()[:, None]), 1)).squeeze(
        1
    )
    if explore:
        pred = pred + temperature * 0.4 * torch.randn_like(pred)
    unit = pred.clamp(-14.0, 14.0).double().sigmoid()
    if perturb:
        unit = (unit + perturb).clamp(0.0, 1.0)
    return switch.double() * (lo + (hi - lo) * unit), base_logit, idle & charge, pred


def rollout(
    actor,
    price,
    net,
    target,
    arrival,
    *,
    explore=True,
    temperature=1.0,
    forced=None,
    straight=True,
    return_switch=False,
    first_perturbation=0.0,
):
    exog = curve_features(price, net)
    need = torch.as_tensor(target, dtype=torch.float64)
    stored = torch.as_tensor(arrival, dtype=torch.float64)
    p = torch.from_numpy(price)
    pv = torch.from_numpy(net)
    total = 0.0
    previous = torch.zeros_like(need)
    delivery = []
    for t, row in enumerate(exog):
        n = len(exog) - t
        q, score, free, _ = decision(
            actor,
            row,
            need,
            stored,
            n,
            explore=explore,
            temperature=temperature,
            forced=forced if t == 0 else None,
            perturb=first_perturbation if t == 0 else 0.0,
            straight=straight,
        )
        if t == 0:
            first = (score, free)
        stored = stored + q
        need = need - q
        deficit = need.clamp_min(0.0)
        unavoid = (deficit - 1.0 - (n - 1)).clamp_min(0.0)
        rise = (unavoid - previous).clamp_min(0.0)
        previous = torch.maximum(previous, unavoid)
        cost = ((q - pv[:, t]).clamp_min(0.0) - (-pv[:, t]).clamp_min(0.0)) * p[:, t]
        extra = torch.minimum(q, (-need).clamp_min(0.0))
        total = (
            total
            + (cost + 120.0 * rise + PRICE * extra + (deficit.clamp(max=1.0) if n == 1 else 0.0))
            * GAMMA**t
        )
        delivery.append(q)
    result = (total, torch.stack(delivery, dim=1))
    return (*result, *first) if return_switch else result


def configure(actor):
    # Neutral backbone readouts leave the native EV logits under the two trainable heads.
    with torch.no_grad():
        for layer in (
            actor.mode_heads["ev_session"],
            actor.beta_heads["ev_session"],
            actor.ev_band[-1],
        ):
            layer.weight.zero_()
            layer.bias.zero_()
        actor.ev_band[-1].bias[1] = 15.0
    for p in actor.parameters():
        p.requires_grad_(False)
    for head in (actor.ev_gate, actor.ev_watts):
        head.requires_grad_(True)


def train(width, seed, *, max_updates=None):
    torch.set_num_threads(1)
    agent = fresh(width, seed)
    actor = agent.actor.cpu().eval()
    configure(actor)
    assert (
        agent.reward_penalties.ev_tolerance_kwh == 1.0
        and agent.reward_penalties.ev_deficit_kwh == 120.0
    )
    assert agent.reference_sites == 30 and abs(agent.learner.gamma - GAMMA) < 1e-10
    data = TrainingCurves(split()[0])
    rng = np.random.default_rng(33217)
    optim = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=0.002)
    if max_updates is None:
        max_updates = max(MILESTONES)
    if max_updates not in MILESTONES:
        raise ValueError("Select a recorded training duration")
    for step in range(1, max_updates + 1):
        horizons = [1, 1, 1, 4, 6, 8] if step < 150 else [1, 1, 1, 2, 4, 4, 6, 8, 8, 16, 32, 40]
        n = int(rng.choice(horizons))
        batch = data.sample(rng, n, BATCH)
        temp = max(0.17, 1.0 - step / 2000)
        optim.param_groups[0]["lr"] = 0.002 if step < 1200 else (0.0008 if step < 2800 else 0.00032)
        total, _ = rollout(actor, *batch, temperature=temp)
        optim.zero_grad(set_to_none=True)
        total.mean().backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in actor.parameters() if p.requires_grad], 2.0
        )
        assert torch.isfinite(norm)
        optim.step()
        if step in MILESTONES:
            save_model(actor, width, seed, "gradient", step * BATCH)
            print(step * BATCH, float(total.detach().mean()), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("width", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("--max-updates", type=int, choices=MILESTONES)
    args = p.parse_args()
    train(args.width, args.seed, max_updates=args.max_updates)
