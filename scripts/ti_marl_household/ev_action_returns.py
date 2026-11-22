"""Estimate EV charge/wait advantages from paired operating returns in training windows."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch

from scripts.ti_marl_household import community
from scripts.ti_marl_household import ev_power as experiment
from scripts.ti_marl_household import paired_choices as paired

WEIGHT = "ev_gate.4.weight"


@torch.no_grad()
def trajectory(case, *, intervention=None, capture=False):
    from scripts.ti_marl_household import start_timing as timing

    env = timing.environment(case)
    actor = community.policy.actor
    native = actor._choose
    records = []
    hits = []

    def choose(c, context, state, deterministic):
        selected = native(c, context, state, deterministic)
        if c.group_type != "ev_session" or c.stage != "mode" or len(c.allowed) != 2:
            return selected
        modes = actor.group_modes[c.group_type]
        active = modes.index("CHARGE_EV")
        idle = modes.index("IDLE")
        if set(c.allowed) != {active, idle}:
            return selected
        slot = env.time_step
        if intervention == (slot, c.group_id):
            hits.append((slot, c.group_id))
            return replace(selected, mode=idle if selected.mode == active else active)
        if capture:
            x = context.new_tensor(actor.household_inputs[c.group_id][0])
            logits = actor._distribution(context, state, c).logits
            odds = logits[c.allowed.index(active)] - logits[c.allowed.index(idle)]
            records.append(
                dict(
                    slot=slot,
                    group=c.group_id,
                    charge=selected.mode == active,
                    need=float(x[0]),
                    logit=float(odds),
                    hidden=actor.ev_gate[:4](x).tolist(),
                )
            )
        return selected

    with patch.object(actor, "_choose", choose):
        row = experiment.rollout(case, env=env)
    if intervention is not None and hits != [intervention]:
        raise ValueError(f"Training intervention did not match the natural trajectory: {hits}")
    return row["discounted_reward"], records


def collect(case):
    value, decisions = trajectory(case, capture=True)
    picks = []
    for group in sorted({r["group"] for r in decisions}):
        for charge in (False, True):
            options = sorted(
                (
                    r
                    for r in decisions
                    if r["group"] == group and r["charge"] == charge and r["need"] > 0
                ),
                key=lambda r: r["need"],
            )
            if not options:
                continue
            for i in np.unique(np.linspace(0, len(options) - 1, min(3, len(options))).astype(int)):
                picks.append(options[i])
    for r in picks:
        alternative, _ = trajectory(case, intervention=(r["slot"], r["group"]))
        r["charge_advantage"] = (value - alternative) * (1 if r["charge"] else -1)
    return dict(case=case["start"], paired=picks, observations=decisions)


def fit(d, workers):
    experiment.setup(d)
    groups = {}
    for c in community.read_json(d / "cases.json"):
        if c["split"] == "train":
            groups.setdefault((c["stimulus"], c["early"], c["first"]), []).append(c)
    rng = np.random.default_rng(7141)
    cases = [g[i] for g in groups.values() for i in rng.choice(len(g), 2, replace=False)]
    result = paired.parallel(collect, cases, workers)
    paired_states = [r for g in result for r in g["paired"]]
    x = torch.tensor([r["hidden"] for r in paired_states])
    logits = torch.tensor([r["logit"] for r in paired_states])
    adv = torch.tensor([r["charge_advantage"] for r in paired_states])
    adv = adv / adv.abs().mean().clamp_min(1e-8)
    all_states = [r for g in result for r in g["observations"]]
    h = torch.tensor([r["hidden"] for r in all_states])
    prior = torch.tensor([r["logit"] for r in all_states])
    summary = []
    for i, trust in enumerate((0.10, 0.015, 0.002)):
        delta = torch.zeros(x.shape[1], requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=0.012)
        for _ in range(450):
            new = logits + x @ delta
            # Positive/negative advantages come exclusively from the two
            # simulator returns under the same state and continuation policy.
            improve = (adv.abs() * torch.nn.functional.softplus(-adv.sign() * new)).mean()
            changed = h @ delta
            kl = (
                torch.nn.functional.softplus(prior + changed)
                - torch.nn.functional.softplus(prior)
                - prior.sigmoid() * changed
            ).mean()
            objective = improve + trust * kl + 0.002 * delta.square().sum()
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
        obj = torch.load(experiment.SOURCE, map_location="cpu", weights_only=False)
        obj["actor"][WEIGHT] += delta.detach()[None, :]
        opt = obj["learner"]["actor_optimizer"]
        names = list(dict(community.policy.actor.named_parameters()))
        opt["state"].pop(opt["param_groups"][0]["params"][names.index(WEIGHT)], None)
        torch.save(obj, d / f"paired-ev-{i}.pth")
        summary.append(
            dict(
                trust=trust,
                paired_decisions=len(logits),
                agreement=float(((new.detach() > 0) == (adv > 0))[adv.abs() > 1e-5].float().mean()),
            )
        )
    community.write_json(
        d / "ev-advantages.json",
        dict(
            windows=[g["case"] for g in result],
            initial_agreement=float(((logits > 0) == (adv > 0))[adv.abs() > 1e-5].float().mean()),
            candidates=summary,
        ),
    )
