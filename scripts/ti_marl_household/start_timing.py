"""Improve the seed-71 dishwasher actor using returns from all legal start times."""

import argparse
import hashlib
import shutil
from datetime import date, timedelta
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from scripts.ti_marl_household import community
from scripts.ti_marl_household import dishwasher_critic as probe
from scripts.ti_marl_household import paired_choices as paired

PARENT = paired.RUNS / "paired-71/latest_checkpoint.pth"
DAYS = (
    community.DAYS[0],
    ("validation", [228, 257, 289, 312]),
    ("test", [242, 254, 268, 283, 296, 311, 329, 348]),
)
SLOTS = tuple(range(2, 13))
TRUST = (0.12, 0.04, 0.012, 0.012)


def price_start(split, day, early):
    windows = (
        [(2, 10), (3, 8), (2, 11), (3, 10)]
        if split == "train"
        else [(2, 10), (4, 9), (3, 12), (2, 10)]
    )
    return windows[dict(DAYS)[split].index(day) % len(windows)][0 if early else 1]


def environment(case):
    cfg = community.config
    cfg["simulator"].update(
        simulation_start_time_step=case["start"],
        simulation_end_time_step=case["start"] + community.DECISIONS,
    )
    return community._build_env(cfg)


def observation(agent, snapshot, index, traces):
    score = probe.score(agent, snapshot, index, traces)
    key = traces["START"][probe.SITE].choices[index].group_id
    x = agent.actor.household_inputs[key][1]
    return dict(**score, features=list(map(float, x)))


@torch.no_grad()
def explore_case(case):
    env, agent = environment(case), community.policy
    rows = []
    prefix = None
    discounts = agent.learner.gamma ** np.arange(community.DECISIONS)
    for t in SLOTS:
        row = probe.simulate(
            env,
            agent,
            case,
            arm="timing",
            branch="explore_start",
            anchor=t,
            observe_choice=observation,
        )
        if "unavailable" in row:
            return dict(case=case, incomplete=t)
        assert row["start"] == t and row["completed"]
        if prefix is not None:
            assert row["actions"][: t - 1] == prefix["actions"][: t - 1]
            np.testing.assert_allclose(
                row["rewards"][: t - 1], prefix["rewards"][: t - 1], atol=1e-7
            )
        prefix = row
        if t < SLOTS[-1] and row["at_start"] is None:
            return dict(case=case, incomplete=t)
        rows.append(
            dict(
                slot=t,
                choice=row["at_start"],
                value=float(discounts @ row["rewards"]),
                excess=float(np.maximum(0, row["ev_errors"]).mean()),
                over=sum(x > 0.1 for x in row["ev_errors"]),
            )
        )
    return dict(case=case, outcomes=rows)


def explore(d, workers):
    community.init_eval(d, PARENT)
    cases = [
        c
        for c in community.read_json(d / "cases.json")
        if c["split"] == "train" and (c["stimulus"] == "price" or c["day"] in DAYS[0][1][::3])
    ]
    out = paired.parallel(explore_case, cases, workers)
    community.write_json(d / "branches.json", out)


@torch.no_grad()
def diagnosis(case):
    agent, env = community.policy, environment(case)
    a = probe.simulate(env, agent, case, arm="timing", branch="unforced")
    r = dict(start=a["start"], case=case["start"], tariff=case["price_start"], score=a["at_start"])
    if a["at_start"] is not None:
        b = probe.simulate(env, agent, case, arm="timing", branch="wait_once", anchor=a["start"])
        t = a["start"]
        assert a["actions"][:t] == b["actions"][:t]
        np.testing.assert_allclose(a["rewards"][:t], b["rewards"][:t], atol=1e-7)
        r["gain"] = float(
            (agent.learner.gamma ** np.arange(community.DECISIONS - t))
            @ np.subtract(b["rewards"][t:], a["rewards"][t:])
        )
        r["bill_saved_eur"] = a["bill"] - b["bill"]
    return r


def meter(env, case):
    m = paired.meter(env, case)
    if case["stimulus"] == "price":
        t = case["price_start"]
        e = env.buildings[1].deferrable_appliances_electricity_consumption[: community.DECISIONS]
        prices = np.asarray(
            [
                r[0]["grid_import_price"]
                for r in env._community_market_settlement_history[: community.DECISIONS]
            ]
        )
        np.testing.assert_array_equal(np.flatnonzero(np.isclose(prices, 0.07)), np.arange(t, t + 7))
        m["cheap"] = float(e[t : t + 7].sum())
    return m


def evaluate(d, path, label, split, workers, *, diagnose=False):
    community.init_eval(d, path)
    cases = [c for c in community.read_json(d / "cases.json") if c["split"] == split]
    rows = paired.parallel(partial(community.evaluate_case, inspect_episode=meter), cases, workers)
    for r, c in zip(rows, cases, strict=True):
        assert r["start"] == c["start"]
        r.update(early=c["early"], price_start=c.get("price_start"))
    community.write_json(d / f"{label}-{split}.json", rows)
    if diagnose:
        probes = paired.parallel(
            diagnosis, [c for c in cases if c["stimulus"] == "price" and not c["early"]], workers
        )
        community.write_json(d / f"{label}-diagnosis.json", probes)


def start_probabilities(logits):
    """Probability of the first START. The remaining mass starts at the last legal slot."""
    wait = (-logits).sigmoid().cumprod(dim=1)
    survival = torch.cat((torch.ones_like(wait[:, :1]), wait), dim=1)
    p = torch.cat((logits.sigmoid(), torch.ones_like(wait[:, :1])), dim=1) * survival
    return p


def audit_price_input(branches, coefficients):
    """Isolate the cycle-mean price term while keeping all other local logits fixed."""
    examples = []
    for b in branches:
        c = b["case"]
        if c["stimulus"] != "price" or c["early"] or "outcomes" not in b:
            continue
        for r in b["outcomes"]:
            s = r["choice"]
            if s is None or s["p_start"] < 0.5:
                continue
            if r["slot"] < c["price_start"]:
                # The discounted tariff is inside the local outlook in these cases.
                # Input 2 equals (cycle mean - minimum outlook price) / 0.1.
                mean = 0.07 + 0.1 * s["features"][2]
                shift = (0.25 - mean) / 0.1
                weight = float(coefficients[1] + coefficients[2] - coefficients[3])
                examples.append(
                    dict(
                        slot=r["slot"],
                        tariff=c["price_start"],
                        mean_price=mean,
                        observed_log_odds=s["log_odds"],
                        spot_only_readout_log_odds=s["log_odds"] + weight * shift,
                    )
                )
            break
    return dict(
        cases=len(examples),
        switches_to_wait=sum(e["spot_only_readout_log_odds"] < 0 for e in examples),
        cases_by_window=examples,
    )


def train(d):
    branches = community.read_json(d / "branches.json")
    rows = [b["outcomes"] for b in branches if "outcomes" in b]
    if not rows:
        raise ValueError("No complete legal start sequences")

    def array(v):
        return torch.as_tensor(v, dtype=torch.float64)

    features = array([[s["choice"]["features"] for s in row[:-1]] for row in rows])
    logits = array([[s["choice"]["log_odds"] for s in row[:-1]] for row in rows])
    reward = array([[s["value"] for s in row] for row in rows])
    # Only a state-independent baseline and positive scaling are applied to the
    # already-discounted operating returns. The simulator reward stays intact.
    returns = reward - reward.mean(dim=1, keepdim=True)
    returns = returns / returns.std().clamp_min(1e-8)
    prior = start_probabilities(logits)
    torch.testing.assert_close(prior.sum(dim=1), torch.ones_like(prior[:, 0]))
    original = torch.load(PARENT, map_location="cpu", weights_only=False)
    prior_row = original["actor"]["coordination"].clone()
    community.init_eval(d, PARENT)
    old_h = original["configuration"]["algorithm"]["hyperparameters"]
    assert community.policy.reward_penalties.model_dump() == old_h["reward_penalties"]
    assert community.policy.reward_profile == old_h["operating_reward"]
    # The old critic and EV actor tensors stay bit-for-bit identical.
    opt = original["learner"]["actor_optimizer"]
    keys = list(dict(community.policy.actor.named_parameters()))
    opt["state"].pop(opt["param_groups"][0]["params"][keys.index("coordination")], None)
    deltas = {}
    expected = {}
    for idx, trust in enumerate(TRUST, 1):
        active = features.clone()
        if idx == 4:
            # A one-parameter control. This tariff spread vanishes on a flat
            # tariff, so this candidate reproduces the original solar-only policy.
            active[:, :, :2] = 0
            active[:, :, 3:] = 0
        delta = torch.zeros(features.shape[-1], dtype=features.dtype, requires_grad=True)
        optimiser = torch.optim.Adam([delta], lr=0.045)
        for _ in range(550):
            prob = start_probabilities(logits + active @ delta)
            kl = (prob * (prob.clamp_min(1e-12).log() - prior.clamp_min(1e-12).log())).sum(dim=1)
            objective = (
                (prob * returns).sum(dim=1).mean()
                - trust * kl.mean()
                - 0.0002 * delta.square().sum()
            )
            optimiser.zero_grad()
            (-objective).backward()
            optimiser.step()
        delta = delta.detach()
        original["actor"]["coordination"] = prior_row.clone()
        original["actor"]["coordination"][2] += delta.to(prior_row)
        torch.save(original, d / f"actor-{idx}.pth")
        with torch.no_grad():
            prob = start_probabilities(logits + features @ delta)
            expected[idx] = float(((prob - prior) * reward).sum(dim=1).mean())
        deltas[idx] = delta.tolist()
    return dict(
        total_training_windows=len(branches),
        complete_training_windows=len(rows),
        valid_start_trajectories=len(rows) * len(SLOTS),
        start_slots=list(SLOTS),
        trust_strengths=list(TRUST),
        price_spread_only_candidate=4,
        updates=deltas,
        expected_training_return_improvement=expected,
        original_price_input_audit=audit_price_input(branches, prior_row[2]),
        open_choice_at_final_slot=sum(r[-1]["choice"] is not None for r in rows),
        penalties=old_h["reward_penalties"],
        reward=old_h["operating_reward"],
    )


def summary(rows):
    result = paired.metrics(rows)
    for label, early in (("early", True), ("late", False)):
        g = [r for r in rows if r["stimulus"] == "price" and r["early"] == early]
        offsets = np.asarray([r["inspection"]["start"] - r["price_start"] for r in g])
        part = result[label]
        part.update(
            starts_in_cheap=int(np.count_nonzero((offsets >= 0) & (offsets < 7))),
            mean_minutes_before=float(np.maximum(-offsets, 0).mean() * 15),
            minutes_relative_to_discount=(offsets * 15).tolist(),
            full_cycles_in_cheap=sum(
                np.isclose(r["inspection"]["cheap"], r["inspection"]["requested"], atol=1e-5)
                for r in g
            ).item(),
        )
    return result


def judge(ref, new):
    a, b = summary(ref), summary(new)
    assert [(r["day"], r["start"]) for r in ref] == [(r["day"], r["start"]) for r in new]
    excess = lambda rows: np.maximum(0, np.asarray([r["ev_errors_kwh"] for r in rows]))
    increase = excess(new) - excess(ref)
    # An improvement in average underdelivery must not mask a new missed target.
    guard = (
        paired.passes_guard(a, b)
        and b["largest_shortfall_kwh"] <= a["largest_shortfall_kwh"] + 1e-6
    )
    return dict(
        ev_and_early_guard=bool(guard),
        progressed=bool(
            b["late"]["cheap_share"] > a["late"]["cheap_share"] + 1e-4
            and b["late"]["mean_minutes_before"] < a["late"]["mean_minutes_before"]
        ),
        individual_excess_increases_over_01=int((increase > 0.1).sum()),
        individual_excess_increases_over_1e6th_kwh=int((increase > 1e-6).sum()),
        largest_individual_excess_increase_kwh=max(0, float(increase.max())),
    )


def ready(guard):
    return (
        guard["ev_and_early_guard"]
        and guard["progressed"]
        and not guard["individual_excess_increases_over_01"]
    )


def select(d, workers):
    info = train(d)
    ref = community.read_json(d / "reference-validation.json")
    scores = {}
    checks = {}
    for i in range(1, len(TRUST) + 1):
        key = f"actor-{i}"
        evaluate(d, d / f"{key}.pth", key, "validation", workers)
        rows = community.read_json(d / f"{key}-validation.json")
        scores[key] = summary(rows)
        checks[key] = judge(ref, rows)
    supported = [key for key in scores if ready(checks[key])]
    # If all candidates fail, test the most informative trained model once as a diagnostic.
    chosen = max(
        supported or list(scores),
        key=lambda k: (
            round(scores[k]["late"]["cheap_share"], 7),
            -scores[k]["late"]["mean_minutes_before"],
            -scores[k]["mean_excess_kwh"],
            -np.count_nonzero(info["updates"][int(k.split("-")[-1])]),
        ),
    )
    community.write_json(
        d / "selection.json",
        dict(
            chosen=chosen,
            passed=bool(supported),
            reference=summary(ref),
            scores=scores,
            guards=checks,
            training=info,
        ),
    )
    evaluate(d, d / f"{chosen}.pth", "chosen", "test", workers, diagnose=True)


def diagnostic_summary(rows):
    compared = [r for r in rows if r["score"] is not None]

    def means(g):
        if not g:
            return dict(decisions=0)
        m = lambda key: float(np.mean([r["score"][key] for r in g]))
        return dict(
            decisions=len(g),
            p_start=m("p_start"),
            base_log_odds=m("base_log_odds"),
            price_log_odds=m("price_log_odds"),
            other_log_odds=m("other_readout_log_odds"),
            total_log_odds=m("log_odds"),
            critic_prefers_wait=sum(r["score"]["wait"] > r["score"]["start"] for r in g),
            measured_wait_better=sum(r["gain"] > 1e-5 for r in g),
            measured_start_better=sum(r["gain"] < -1e-5 for r in g),
            mean_measured_wait_gain=float(np.mean([r["gain"] for r in g])),
            mean_bill_saved_waiting=float(np.mean([r["bill_saved_eur"] for r in g])),
        )

    return dict(
        windows=len(rows),
        all=means(compared),
        before_discount=means([r for r in compared if r["start"] < r["tariff"]]),
        at_or_after_discount=means([r for r in compared if r["start"] >= r["tariff"]]),
    )


def report(d):
    selection = community.read_json(d / "selection.json")
    selected = d / (selection["chosen"] + ".pth")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    trained = torch.load(selected, map_location="cpu", weights_only=False)
    assert parent["configuration"] == trained["configuration"]
    for k in parent["critic"]:
        assert torch.equal(parent["critic"][k], trained["critic"][k])
    for k in parent["actor"]:
        if k != "coordination":
            assert torch.equal(parent["actor"][k], trained["actor"][k])
        else:
            mask = torch.any(parent["actor"][k] != trained["actor"][k], dim=1).tolist()
            assert mask == [False, False, True, False, False]
    ref = community.read_json(d / "reference-test.json")
    model = community.read_json(d / "chosen-test.json")
    acceptance = judge(ref, model)
    result = dict(
        parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        selected_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
        simulator=version("softcpsrecsimulator"),
        dates={s: [str(date(2023, 1, 1) + timedelta(days=i)) for i in days] for s, days in DAYS},
        protocol="Three houses from Core-30 Normal 2023. Deterministic simulations with controlled EVs and programme. Price windows last seven quarters. Prior policy and selected policy see identical cases. Exact expected return for a stochastic first start on the training grid; all other actions deterministic. The critic is used only for diagnosis. Final inference uses the local actor.",
        selection=selection,
        final=dict(
            reference=summary(ref),
            trained=summary(model),
            guard=acceptance,
            confirmed=bool(selection["passed"] and ready(acceptance)),
        ),
        diagnosis={
            label: diagnostic_summary(community.read_json(d / f"{label}-diagnosis.json"))
            for label in ["reference", "chosen"]
        },
        public_job=paired.verify_checkpoint(d, selected, "timing-71"),
    )
    dest = paired.RUNS / "timing-71/latest_checkpoint.pth"
    dest.parent.mkdir(exist_ok=True)
    shutil.copyfile(selected, dest)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == result["selected_sha256"]
    result["model"] = str(dest)
    community.write_json(paired.RUNS / "start-timing-71.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "explore", "reference", "select", "report"))
    p.add_argument("directory", type=Path)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    d = a.directory
    if a.action == "prepare":
        community.make_fixture(d, community.schema_path(), case_days=DAYS, price_start=price_start)
    elif a.action == "explore":
        explore(d, a.workers)
    elif a.action == "reference":
        for split in ("validation", "test"):
            evaluate(d, PARENT, "reference", split, a.workers, diagnose=split == "test")
    elif a.action == "select":
        select(d, a.workers)
    else:
        report(d)


if __name__ == "__main__":
    main()
