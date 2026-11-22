"""Learn a shared EV power readout with whole-episode SoftCPS operating returns."""

import argparse
import hashlib
from datetime import date, timedelta
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import torch

from scripts.ti_marl_household import community
from scripts.ti_marl_household import conservative_timing as previous
from scripts.ti_marl_household import paired_choices as paired
from scripts.ti_marl_household import start_timing as timing

SOURCE = paired.RUNS / "conservative-timing-71/latest_checkpoint.pth"
REFERENCE = timing.PARENT
HISTORY = paired.RUNS / "conservative-timing-71.json"
DAYS = (
    ("train", [8, 33, 57, 84, 107, 141, 182, 209]),
    ("validation", [222, 250, 280, 310]),
    ("test", [241, 255, 271, 287, 303, 323, 340, 358]),
)
WINDOWS = ((2, 10), (4, 9), (3, 12), (5, 11))


def price_start(split, day, early):
    return WINDOWS[dict(DAYS)[split].index(day) % len(WINDOWS)][0 if early else 1]


def prepare(d):
    community.make_fixture(
        d, community.schema_path(), case_days=DAYS, price_start=price_start, session_seed=934107
    )
    community.make_fixture(
        d / "regression",
        community.schema_path(),
        case_days=previous.DAYS,
        price_start=previous.price_start,
    )


def failed_case(d):
    saved = community.read_json(HISTORY)["excess_replays"][0]["case"]
    return next(c for c in community.read_json(d / "cases.json") if c == saved)


@torch.no_grad()
def rollout(case, *, trace=False, env=None):
    agent = community.policy
    env = timing.environment(case) if env is None else env
    history = []

    def observe(t, agent, env):
        snapshot = agent._pending["snapshot"]
        for j, site in enumerate(community.SITES[:2]):
            event = agent._pending["prefix_traces"][site]
            for c in event.choices:
                if c.group_type != "ev_session" or c.stage != "mode":
                    continue
                x = agent.actor.household_inputs[c.group_id][0]
                if x is None:
                    continue
                power = next(
                    (v for v in event.choices if v.group_id == c.group_id and v.stage == "power"),
                    None,
                )
                group = next(g for g in snapshot.groups_for(site) if g.group_id == c.group_id)
                history.append(
                    dict(
                        slot=t,
                        port=j,
                        group=c.group_id,
                        need_units=float(x[0]),
                        hours_left=float(x[1]),
                        rated_kw=group.max_charge_power_kw,
                        mode=agent.actor.group_modes["ev_session"][c.mode],
                        allowed=[agent.actor.group_modes["ev_session"][i] for i in c.allowed],
                        interval_kw=None
                        if power is None
                        else [power.lower * power.scale_kw, power.upper * power.scale_kw],
                        selected_kw=None if power is None else power.fraction * power.scale_kw,
                    )
                )

    r = timing.probe.simulate(
        env, agent, case, arm="power", branch="meter", observe_step=observe if trace else None
    )
    row = dict(
        start=case["start"],
        day=case["day"],
        early=case["early"],
        stimulus=case["stimulus"],
        price_start=case.get("price_start"),
        ev_errors_kwh=np.asarray(r["ev_errors"])[
            np.argsort(np.argsort(case["departures"], kind="stable"))
        ].tolist(),
        programme_complete=r["completed"],
        bill_eur=r["bill"],
        inspection=timing.meter(env, case),
        undiscounted_reward=float(np.sum(r["rewards"])),
        discounted_reward=float(
            np.dot(agent.learner.gamma ** np.arange(community.DECISIONS), r["rewards"])
        ),
    )
    if trace:
        row["decisions"] = history
        row["ac_kwh_by_quarter"] = [
            b.electric_vehicle_chargers[0].electricity_consumption[: community.DECISIONS].tolist()
            for b in env.buildings[:2]
        ]
    return row


def diagnostic(d, path):
    community.init_eval(d, path)
    return rollout(failed_case(d), trace=True)


def reference(d, workers):
    old = d / "regression"
    history = community.read_json(HISTORY)["excess_replays"][0]
    for key, path, expected in (
        ("reference", REFERENCE, "parent"),
        ("source", SOURCE, "candidate"),
    ):
        row = diagnostic(old, path)
        np.testing.assert_allclose(
            row["ev_errors_kwh"][history["port"]],
            history["traces"][expected]["departure_error_kwh"],
            atol=1e-6,
        )
        community.write_json(d / f"{key}-regression.json", row)
        for split in ("validation", "test"):
            timing.evaluate(d, path, key, split, workers)


# The search directions span existing shared EV weights. Training uses only
# total discounted operating return. No action targets or test cases enter the fit.
KEYS = ("coordination", "ev_watts.original.4.bias", "ev_watts.curve")
starting = {}


def setup(d):
    community.init_eval(d, SOURCE)
    state = community.policy.actor.state_dict()
    starting.clear()
    starting.update({k: state[k].clone() for k in KEYS})


@torch.no_grad()
def change(v):
    actor = community.policy.actor
    v = torch.as_tensor(v, dtype=actor.coordination.dtype)
    actor.coordination[0].copy_(starting[KEYS[0]][0])
    actor.coordination[0, :4].add_(v[:4])
    actor.ev_watts.original[4].bias.copy_(starting[KEYS[1]] + v[4])
    knots = torch.linspace(0, 2.1, len(actor.ev_watts.curve))
    basis = (1 - (knots[:, None] - knots.new_tensor([0.6, 1.15, 1.7])).abs() / 0.55).clamp_min(0)
    actor.ev_watts.curve.copy_(starting[KEYS[2]] + basis @ v[5:])


def window_return(task):
    v, case = task
    change(v)
    return rollout(case)["discounted_reward"]


def save_candidate(d, name, v):
    change(v)
    obj = torch.load(SOURCE, map_location="cpu", weights_only=False)
    updated = community.policy.actor.state_dict()
    opt = obj["learner"]["actor_optimizer"]
    names = list(dict(community.policy.actor.named_parameters()))
    for key in KEYS:
        if not torch.equal(obj["actor"][key], updated[key]):
            obj["actor"][key] = updated[key].clone()
            opt["state"].pop(opt["param_groups"][0]["params"][names.index(key)], None)
    torch.save(obj, d / f"{name}.pth")


def train(d, workers):
    setup(d)
    rng = np.random.default_rng(71)
    groups = {}
    cases = [c for c in community.read_json(d / "cases.json") if c["split"] == "train"]
    for c in cases:
        groups.setdefault((c["stimulus"], c["early"], c["first"]), []).append(c)
    for g in groups.values():
        rng.shuffle(g)
    axes = np.array([4.0, 2.0, 2.0, 2.0, 3.0, 0.25, 0.3, 0.3])
    prior = np.zeros(len(axes))
    report = []
    for generation in range(3):
        # Every candidate sees the same weather, price, departures and port order.
        batch = [g[generation] for g in groups.values()]
        if generation == 2:
            batch += [g[3] for g in groups.values()]
        noises = rng.normal(size=(5, 8)) * axes * (0.75**generation)
        if generation == 0:
            # Pure continuous-power exploration and joint mode/power exploration.
            noises[:2, :4] = 0.0
            noises[2, 1:] = 0.0
        vectors = np.vstack((prior, np.zeros(8), prior + noises, prior - noises))
        vectors = np.unique(vectors, axis=0)
        tasks = [(v.tolist(), c) for v in vectors for c in batch]
        values = np.array(paired.parallel(window_return, tasks, workers)).reshape(len(vectors), -1)
        ref = values[np.where(np.all(vectors == 0, axis=1))[0][0]]
        gains = (values - ref).mean(axis=1)
        order = np.argsort(-gains, kind="stable")
        report.append(
            dict(
                generation=generation,
                windows=[c["start"] for c in batch],
                candidates=[
                    dict(weights=v.tolist(), mean_return_gain=float(x))
                    for v, x in zip(vectors, gains, strict=True)
                ],
            )
        )
        top = vectors[order[0]]
        print("generation", generation, "gain", gains[order[0]], "top", top.tolist(), flush=True)
        if generation == 0:
            pure = [i for i in order if np.all(vectors[i, :4] == 0) and np.any(vectors[i, 4:])]
            save_candidate(d, "power-only", vectors[pure[0]])
        save_candidate(d, f"generation-{generation}", top)
        prior = np.mean(vectors[order[:3]], axis=0)
    community.write_json(d / "training.json", report)


def compare(a, b):
    g = timing.judge(a, b)
    old = timing.summary(a)
    new = timing.summary(b)
    excess = lambda s: np.maximum(0, np.asarray([r["ev_errors_kwh"] for r in s]))
    delta = excess(b) - excess(a)
    g.update(
        within_service=new["cycles_completed"] == len(b) and new["shortfalls_over_01"] == 0,
        no_extra_excess=new["excesses_over_01"] <= old["excesses_over_01"]
        and new["mean_excess_kwh"] <= old["mean_excess_kwh"] + 1e-6,
        discounted_cycles_preserved=all(
            new[k]["starts_in_cheap"] >= old[k]["starts_in_cheap"]
            and new[k]["cheap_share"] >= old[k]["cheap_share"] - 1e-6
            for k in ("early", "late")
        ),
        individual_excess_decreases=int((delta < -1e-6).sum()),
        individual_excess_decreases_over_01=int((delta < -0.1).sum()),
        new_exits_above_01=int(((excess(b) > 0.1) & (excess(a) <= 0.1)).sum()),
        recovered_exits_above_01=int(((excess(a) > 0.1) & (excess(b) <= 0.1)).sum()),
    )
    return g


def passes(a, b):
    return (
        a["within_service"]
        and a["ev_and_early_guard"]
        and a["no_extra_excess"]
        and a["largest_individual_excess_increase_kwh"] <= 1e-6
        and b["discounted_cycles_preserved"]
        and b["within_service"]
        and b["ev_and_early_guard"]
        and b["no_extra_excess"]
        and b["largest_individual_excess_increase_kwh"] <= 1e-6
    )


def probe_intervention(task):
    """One diagnostic choice in the old case, followed by the unmodified actor."""
    from dataclasses import replace
    from unittest.mock import patch

    case, group, slot, stage, position = task
    env = timing.environment(case)
    actor = community.policy.actor
    natural = actor._choose
    hits = []

    def choose(c, context, state, deterministic):
        result = natural(c, context, state, deterministic)
        if (
            env.time_step != slot
            or c.group_id != group
            or (stage != "charge_and_power" and c.stage != stage)
            or not c.sampled
        ):
            return result
        if stage == "charge_and_power" and c.stage == "mode":
            active = actor.group_modes[c.group_type].index("CHARGE_EV")
            assert active in c.allowed and result.mode != active
            hits.append("CHARGE_EV")
            return replace(result, mode=active)
        if stage == "mode":
            alternative = [m for m in c.allowed if m != result.mode]
            if len(alternative) != 1:
                return result
            hits.append(actor.group_modes[c.group_type][alternative[0]])
            return replace(result, mode=alternative[0])
        fraction = c.lower + (c.upper - c.lower) * position
        hits.append(fraction * c.scale_kw)
        return replace(result, unit_sample=position, fraction=fraction)

    with patch.object(actor, "_choose", choose):
        r = rollout(case, env=env)
    assert (stage == "charge_and_power" and hits and hits[0] == "CHARGE_EV") or len(hits) == 1
    return dict(
        slot=slot,
        stage=stage,
        action=hits[-1],
        both_evs=r["ev_errors_kwh"],
        dishwasher_start=r["inspection"]["start"],
        complete=r["programme_complete"],
        bill=r["bill_eur"],
        discounted_return=r["discounted_reward"],
    )


def alternatives(d, workers):
    old = d / "regression"
    case = failed_case(old)
    r = diagnostic(old, SOURCE)
    port = community.read_json(HISTORY)["excess_replays"][0]["port"]
    tasks = []
    for row in r["decisions"]:
        if row["port"] != port or row["need_units"] <= 0:
            continue
        if len(row["allowed"]) == 2:
            tasks.append((case, row["group"], row["slot"], "mode", None))
            if row["mode"] == "IDLE":
                for unit in (0.01, 0.25, 0.5, 0.75, 0.99):
                    tasks.append((case, row["group"], row["slot"], "charge_and_power", unit))
        bounds = row["interval_kw"]
        if bounds and bounds[1] - bounds[0] > 1e-6:
            for unit in (0.01, 0.5, 0.99):
                tasks.append((case, row["group"], row["slot"], "power", unit))
    out = paired.parallel(probe_intervention, tasks, workers)
    community.write_json(d / "regression-alternatives.json", out)


def combine(d):
    # The last search generation supplies EV power and price adjustments. The
    # middle trust setting supplies the separately learned EV charge/wait weights.
    neural_key = "ev_gate.4.weight"
    joint = torch.load(d / "generation-2.pth", map_location="cpu", weights_only=False)
    paired_mode = torch.load(d / "paired-ev-1.pth", map_location="cpu", weights_only=False)
    joint["actor"][neural_key] = paired_mode["actor"][neural_key]
    names = list(dict(community.policy.actor.named_parameters()))
    opt = joint["learner"]["actor_optimizer"]
    opt["state"].pop(opt["param_groups"][0]["params"][names.index(neural_key)], None)
    torch.save(joint, d / "combined.pth")


def select(d, workers):
    refs = {k: community.read_json(d / f"{k}-validation.json") for k in ("reference", "source")}
    entries = {}
    source_actor = torch.load(SOURCE, map_location="cpu", weights_only=False)["actor"]
    for key in (
        ["power-only"]
        + [f"generation-{i}" for i in range(3)]
        + [f"paired-ev-{i}" for i in range(3)]
        + ["combined"]
    ):
        if key == "combined":
            combine(d)
        weights = torch.load(d / f"{key}.pth", map_location="cpu", weights_only=False)["actor"]
        changed = any(not torch.equal(weights[k], v) for k, v in source_actor.items())
        if changed:
            timing.evaluate(d, d / f"{key}.pth", key, "validation", workers)
            rows = community.read_json(d / f"{key}-validation.json")
        else:
            rows = refs["source"]
        guards = {k: compare(v, rows) for k, v in refs.items()}
        entries[key] = dict(
            metrics=timing.summary(rows),
            learned=changed,
            checks=guards,
            passed=passes(guards["reference"], guards["source"]),
        )

    def rank(key):
        e = entries[key]
        guards = e["checks"]
        return (
            e["passed"],
            guards["source"]["discounted_cycles_preserved"],
            guards["source"]["within_service"],
            -max(guards[k]["new_exits_above_01"] for k in refs),
            -max(guards[k]["largest_individual_excess_increase_kwh"] for k in refs),
            -e["metrics"]["mean_excess_kwh"],
        )

    chosen = max((key for key in entries if entries[key]["learned"]), key=rank)
    community.write_json(d / "selection.json", dict(chosen=chosen, candidates=entries))
    print("locked", chosen, flush=True)


def report(d, workers):
    selection = community.read_json(d / "selection.json")
    chosen = selection["chosen"]
    path = d / f"{chosen}.pth"
    timing.evaluate(d, path, "chosen", "test", workers)
    rows = {k: community.read_json(d / f"{k}-test.json") for k in ("reference", "source", "chosen")}
    checks = {k: compare(rows[k], rows["chosen"]) for k in ("reference", "source")}
    old = d / "regression"
    case = failed_case(old)
    port = community.read_json(HISTORY)["excess_replays"][0]["port"]
    traces = {k: community.read_json(d / f"{k}-regression.json") for k in ("reference", "source")}
    traces["chosen"] = diagnostic(old, path)
    failed_guard = abs(traces["chosen"]["ev_errors_kwh"][port]) <= 0.1
    original = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model = torch.load(path, map_location="cpu", weights_only=False)
    assert model["configuration"] == original["configuration"]
    assert all(torch.equal(v, model["critic"][k]) for k, v in original["critic"].items())
    changed = [k for k, v in original["actor"].items() if not torch.equal(model["actor"][k], v)]
    assert set(changed) <= set(KEYS) | {"ev_gate.4.weight"}
    assert torch.equal(original["actor"]["coordination"][1:], model["actor"]["coordination"][1:])
    public = paired.verify_checkpoint(d, path, "ev-power-71")
    keys = ["start", "day", "ev_errors_kwh", "inspection", "programme_complete", "bill_eur"]
    info = dict(
        simulator=distribution("softcpsrecsimulator").version,
        step_seconds=900,
        dates={
            s: [(date(2023, 1, 1) + timedelta(days=v)).isoformat() for v in days]
            for s, days in DAYS
        },
        session_seed=934107,
        actor_seed=71,
        training=community.read_json(d / "training.json"),
        selection=selection,
        paired_ev_training=community.read_json(d / "ev-advantages.json"),
        test={k: timing.summary(v) for k, v in rows.items()},
        checks=checks,
        regression=dict(
            case=case,
            port=port,
            traces=traces,
            within_01=failed_guard,
            counterfactuals=community.read_json(d / "regression-alternatives.json"),
        ),
        passed=bool(
            selection["candidates"][chosen]["passed"]
            and passes(checks["reference"], checks["source"])
            and failed_guard
        ),
        changed_actor_tensors=changed,
        hyperparameters={
            k: model["configuration"]["algorithm"]["hyperparameters"][k]
            for k in ("operating_reward", "reward_penalties")
        },
        public_job=public,
        test_windows={k: [{key: r[key] for key in keys} for r in v] for k, v in rows.items()},
    )
    # Every unchanged tensor already exists in the source checkpoint. Persist
    # just the trained readouts in the result so a rejected model is replayable.
    info["source_checkpoint"] = str(SOURCE)
    info["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    info["trained_readouts"] = {k: model["actor"][k].tolist() for k in changed}
    for k in changed:
        torch.testing.assert_close(
            torch.tensor(info["trained_readouts"][k]).to(original["actor"][k]),
            model["actor"][k],
            rtol=0,
            atol=0,
        )
    community.write_json(paired.RUNS / "ev-power-71.json", info)
    print({k: v for k, v in info.items() if k in ("passed", "checks", "test")}, flush=True)


def study(action, d, workers):
    if action == "counterfactual":
        alternatives(d, workers)
    elif action == "paired":
        from scripts.ti_marl_household import ev_action_returns

        ev_action_returns.fit(d, workers)
    elif action == "train":
        train(d, workers)
    elif action == "select":
        select(d, workers)
    elif action == "report":
        report(d, workers)
    else:
        raise ValueError(action)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action")
    p.add_argument("directory", type=Path)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    d = a.directory
    if a.action == "prepare":
        prepare(d)
    elif a.action == "reference":
        reference(d, a.workers)
    else:
        study(a.action, d, a.workers)


if __name__ == "__main__":
    main()
