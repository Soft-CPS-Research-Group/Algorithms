"""Controlled 2023 EV timing experiment, fine-tuning a saved TI-MARL policy.

Use 2023 Core-30 daily tariff and PV amplitudes to construct announced windows.
The single EV, intervals and service ledger are controlled. This is a timing
diagnostic, not a replay of measured trips or a multi-home settlement run.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from algorithms.ti_marl.agent import TIMARL
from algorithms.ti_marl.learning.operating_reward import TERM_WEIGHTS, operating_reward
from algorithms.ti_marl.learning.prefix_ppo import PrefixStep
from scripts.probe_ti_marl_flexibility import CAPACITY, DT, POWER, EVProbe


@dataclass(frozen=True)
class Session:
    day: str
    departure: str
    start: int
    quarters: int
    arrival_kwh: float
    request_kwh: float
    low: float
    high: float
    pv_kw: float
    variable: str
    early: bool

    def window(self, index):
        middle = self.quarters // 2
        favourable = 0 <= index < middle if self.early else middle <= index < self.quarters
        price = self.low if favourable and self.variable in ("price", "both") else self.high
        solar = self.pv_kw if favourable and self.variable in ("solar", "both") else 0.0
        return price, solar, 0.0


class SessionProbe(EVProbe):
    def window(self, session, first, t):
        return session.window(t)


def days_from_core(schema_path):
    schema = json.loads(schema_path.read_text())
    assert schema["start_date"] == "2023-01-01" and schema["seconds_per_time_step"] == 900
    root = (schema_path.parent / schema["root_directory"]).resolve()
    house = schema["buildings"]["Member_001"]
    last = schema["simulation_end_time_step"] + 1
    assert last == 365 * 96
    price = pd.read_parquet(root / house["pricing"], columns=["electricity_pricing"]).iloc[:last, 0]
    solar = (
        pd.read_parquet(root / house["energy_simulation"], columns=["solar_generation"]).iloc[
            :last, 0
        ]
        / DT
    )
    origin = pd.Timestamp("2023-01-01", tz=schema["timezone"])
    times = pd.date_range(origin, periods=last, freq="15min")
    profiles = pd.DataFrame({"price": price.to_numpy(), "pv": solar.to_numpy()}, index=times)
    pools = {"train": [], "test": []}
    for day, rows in profiles.groupby(times.date):
        if len(rows) != 96 or (day.month in (2, 5, 8, 11) and day.day <= 14):
            continue
        low, high = rows.price.quantile([0.1, 0.9])
        solar_kw = min(POWER, float(rows.pv.max()))
        if high - low < 0.02 or solar_kw < 1.0:
            continue
        pool = "train" if day < date(2023, 9, 1) else "test"
        pools[pool].append((rows.index[0], float(low), float(high), solar_kw))
    return origin, pools


def make_cases(schema_path, n_train, n_pairs):
    origin, pools = days_from_core(schema_path)

    def case(day, i, variable, rng):
        midnight, low, high, pv = day
        quarters = int(rng.choice([4, 6, 8]))
        # Daylight exits for a solar intervention. Price cases also include night.
        hours = [8, 6.5, 8, 11, 15.5, 18, 20.5, 22] if variable == "price" else [11, 12.5, 14, 15.5]
        hour = hours[i % len(hours)]
        departure = midnight + pd.Timedelta(hours=hour)
        end = int((departure - origin).total_seconds() / 900)
        stored = float(rng.choice([7.0, 9.0, 10.0, 11.0]))
        # At least one full quarter can be postponed even with the requested target.
        request = quarters / 2 + float(rng.choice([0.0, 0.5, 1.0]))
        return Session(
            str(midnight.date()),
            departure.strftime("%H:%M"),
            end - quarters,
            quarters,
            stored,
            request,
            low,
            high,
            pv,
            variable,
            bool(rng.integers(2)),
        )

    train_rng = np.random.default_rng(202301)
    test_rng = np.random.default_rng(202309)
    train = []
    for i in range(n_train):
        day = pools["train"][train_rng.integers(len(pools["train"]))]
        train.append(case(day, i // 3, ("price", "solar", "both")[i % 3], train_rng))
    test = []
    selected = test_rng.choice(len(pools["test"]), n_pairs, replace=False)
    for i, index in enumerate(selected):
        for variable in ("price", "solar"):
            c = case(pools["test"][index], i, variable, test_rng)
            test.append(replace(c, early=True))
    assert not {s.day for s in train} & {s.day for s in test}
    return train, test


def excess_delivered_kwh(stored_after, target_kwh, delivered):
    """Charge above the request added by this action, including a partial crossing."""
    return min(delivered, max(0.0, stored_after - target_kwh))


def play(session, agent, manager, probe, phase, *, excess_price=0.0):
    training_leaf = phase == "local"
    training_manager = phase == "coordinator"
    agent.actor.eval()
    agent.critic.eval()
    if manager:
        manager.reset()
    if hasattr(agent.actor, "start_session"):
        agent.actor.start_session()
    stored = session.arrival_kwh
    target = (stored + session.request_kwh) / CAPACITY
    delivery, bills, free = [], [], []
    unavoidable_before = 0.0
    adjustments = {}
    for t in range(session.quarters):
        frame = probe.snapshot(
            session, False, t, stored, target, start_step=session.start, quarters=session.quarters
        )
        if manager:
            frame = replace(frame, measured_peak_kw=manager.peak.peak)
            frame = replace(frame, incentive=manager.curve_for(frame, training=phase != "eval"))
        with torch.no_grad():
            action = agent.actor(frame, deterministic=not training_leaf)
        trace = action.traces[probe.site]
        assert not trace.supervised
        decision = action.executed[0].decisions[0]
        charge = decision.fraction * POWER * DT if decision.mode == "CHARGE_EV" else 0.0
        price, solar, _ = session.window(t)
        balance = charge - solar * DT
        imported = max(0.0, balance)
        bill = imported * price
        stored += charge
        remaining = session.quarters - t - 1
        final = remaining == 0
        deficit = max(0.0, target * CAPACITY - stored)
        protected = max(0.0, deficit - CAPACITY * agent.actor.tolerance_soc)
        unavoidable = max(0.0, protected - remaining * DT * POWER)
        chargeable_deficit = max(0.0, unavoidable - unavoidable_before)
        unavoidable_before = max(unavoidable_before, unavoidable)
        ledger = dict.fromkeys(TERM_WEIGHTS, 0.0)
        ledger.update(
            energy_cost_eur=bill,
            ev_deficit_kwh=chargeable_deficit,
            ev_tolerance_kwh=deficit - protected if final else 0.0,
        )
        reward = operating_reward(
            {"thesis_reward_terms": ledger},
            reference_sites=agent.reference_sites,
            profile=agent.reward_profile,
            penalties=agent.reward_penalties,
        ).value
        if excess_price:
            added_excess = excess_delivered_kwh(stored, target * CAPACITY, charge)
            reward -= excess_price * added_excess / agent.reference_sites
        if manager:
            rise = manager.peak.observe(frame.clock_seconds, 900.0, imported, max(0.0, -balance))
            if training_manager:
                manager.observe_reward(
                    reward - manager.peak_weight * rise / agent.reference_sites, 900.0
                )
            elif training_leaf:
                adjustments = {probe.site: -frame.incentive[0] * imported / agent.reference_sites}
        if training_leaf:
            # Every rollout contains whole terminated sessions; there is no bootstrap at a cut.
            agent.learner.rollout.add(
                PrefixStep(frame, action.traces, reward, adjustments, {}, final, False, 900.0)
            )
        delivery.append(charge)
        bills.append(bill)
        mode = next(c for c in trace.choices if c.stage == "mode")
        if mode.sampled:
            free.append(decision.mode == "CHARGE_EV")
    if training_manager:
        manager.finish()
    return {
        "kwh": delivery,
        "bill": sum(bills),
        "miss": deficit,
        "minimum_miss": protected,
        "extra": max(0.0, stored - target * CAPACITY),
        "free": len(free),
        "free_charge": sum(free),
    }


def evaluate(cases, agent, manager, probe):
    rows = []
    for s in cases:
        early = play(s, agent, manager, probe, "eval")
        late = play(replace(s, early=False), agent, manager, probe, "eval")
        half = s.quarters // 2
        shift = sum(early["kwh"][:half]) - sum(late["kwh"][:half])
        rows.append({"session": asdict(s), "early": early, "late": late, "shift": shift})
    return rows


def fingerprint(module):
    h = hashlib.sha256()
    for tensor in module.state_dict().values():
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(args):
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent = TIMARL(saved["configuration"])
    agent.actor.load_state_dict(saved["actor"])
    agent.critic.load_state_dict(saved["critic"])
    agent.actor.step_seconds = 900.0
    manager = agent.manager
    if manager:
        manager.network.load_state_dict(saved["extra_policy_state"]["manager"])
        manager.rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    train, test = make_cases(args.schema, args.sessions, args.pairs)
    probe = SessionProbe(args.trace)
    before = evaluate(test, agent, manager, probe)
    order = np.random.default_rng(args.seed).permutation(len(train))
    counts = Counter()
    audit = {"initial_actor": fingerprint(agent.actor), "initial_critic": fingerprint(agent.critic)}
    if manager:
        audit["initial_manager"] = fingerprint(manager.network)
    for phase in ("local", "coordinator") if manager else ("local",):
        if manager:
            manager.phase = phase
        for i, idx in enumerate(order, start=1):
            s = train[idx]
            row = play(s, agent, manager, probe, phase)
            counts[phase + "_decisions"] += s.quarters
            counts[phase + "_sessions"] += 1
            counts[phase + "_free"] += row["free"]
            counts[phase + "_free_charge"] += row["free_charge"]
            if phase == "local" and (i % 4 == 0 or i == len(train)):
                agent.actor.train()
                agent.critic.train()
                metrics = agent.learner.update()
                assert all(np.isfinite(value) for value in metrics.values())
            if i % 96 == 0:
                print(phase, i, flush=True)
        if phase == "local":
            audit["trained_actor"] = fingerprint(agent.actor)
            audit["trained_critic"] = fingerprint(agent.critic)
            assert audit["trained_actor"] != audit["initial_actor"]
    if manager:
        assert audit["trained_actor"] == fingerprint(agent.actor)
        assert audit["trained_critic"] == fingerprint(agent.critic)
        audit["trained_manager"] = fingerprint(manager.network)
        assert audit["trained_manager"] != audit["initial_manager"]
    after = evaluate(test, agent, manager, probe)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / ("%s-s%d" % (args.checkpoint.stem, args.seed))
    weights = {"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}
    if manager:
        weights["manager"] = manager.network.state_dict()
    torch.save(weights, path.with_suffix(".pth"))
    result = dict(
        seed=args.seed,
        source=str(args.checkpoint),
        counts=dict(counts),
        training=[asdict(train[idx]) for idx in order],
        before=before,
        after=after,
        hashes=audit,
        reference_sites=agent.reference_sites,
        reward=agent.reward_profile,
        penalties=agent.reward_penalties.model_dump(),
    )
    path.with_suffix(".json").write_text(json.dumps(result, allow_nan=False))
    print("finished", path.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("schema", "trace", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sessions", type=int, default=384)
    parser.add_argument("--pairs", type=int, default=48)
    run(parser.parse_args())
