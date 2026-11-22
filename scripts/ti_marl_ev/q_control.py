"""Model-free Q-learning control for the half-kWh isolated-EV timing experiment.

The table starts at zero. It learns from sampled transitions, never from a
schedule or an action label. The reduced action set is 0, 2 and 4 kW.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from algorithms.ti_marl.learning.operating_reward import TERM_WEIGHTS, operating_reward
from scripts.train_ti_marl_ev_sessions import SessionProbe, evaluate, play

from .from_scratch import OUTPUT, fresh
from .metrics import measures
from .protocol import RESERVED_DATES, TRACE, schema_path, split

ACTIONS = np.array([0.0, 0.5, 1.0])
GAMMA = 0.99**0.25


def outlook_id(price, pv):
    # Describe the announced profile in the short one-window experiment.
    # The observations encode the direction and length of a block, not an action.
    pv = np.clip(np.asarray(pv), 0.0, 1.0)
    price = np.asarray(price)
    change = np.flatnonzero((abs(price - price[0]) > 1e-8) | (abs(pv - pv[0]) > 1e-8))
    first_change = int(change[0]) if len(change) else 0
    tariff_direction = int(np.sign(price[-1] - price[0])) + 1
    solar_direction = int(np.sign(pv[-1] - pv[0])) + 1
    bins = lambda x: min(3, int(x * 4))
    return np.ravel_multi_index(
        (
            len(price) - 1,
            first_change,
            tariff_direction,
            solar_direction,
            bins(pv[0]),
            bins(pv[-1]),
        ),
        (8, 8, 3, 3, 4, 4),
    )


def physical_domains(example):
    """Cache legal actions for this single-port grid using V2's own reservation."""
    from algorithms.ti_marl.runtime.reservation import SiteReservation

    probe = SessionProbe(TRACE)
    domains = np.zeros((9, 18, 3), dtype=bool)
    for n in range(1, 9):
        for half_units in range(-1, min(16, 2 * (n + 1)) + 1):
            need = half_units / 2
            snap = probe.snapshot(
                example, False, 0, 10.0, (10.0 + need) / 20.0, start_step=example.start, quarters=n
            )
            (site,) = snap.agent_ids
            key = snap.groups_for(site)[0].group_id
            domain = SiteReservation.from_snapshot(snap, site)
            modes = domain.admissible_modes(key)
            mask = domains[n, state_id(need)]
            mask[0] = "IDLE" in modes
            if "CHARGE_EV" in modes:
                domain.select_mode(key, "CHARGE_EV")
                lo, hi = domain.fraction_bounds(key)
                mask[1:] = (lo - 1e-7 <= ACTIONS[1:]) & (ACTIONS[1:] <= hi + 1e-7)
            assert mask.any(), (n, need)
    return domains


def state_id(need):
    return np.clip(np.rint(2 * need).astype(int), -1, 16) + 1


def curves(case, t):
    pairs = [case.window(j) for j in range(t, case.quarters)]
    return np.array([a for a, _, _ in pairs]), np.array([b * 0.25 for _, b, _ in pairs])


def test_policy(table, cases, seed, domains):
    agent = fresh(32, seed)
    probe = SessionProbe(TRACE)
    actor = agent.actor
    original = actor._contexts
    chosen = [0.0]
    current_mask = np.zeros(3, dtype=bool)

    def context(obs):
        from algorithms.ti_marl.runtime.forecasts import samples_for

        z = samples_for(obs, probe.site)
        n = round(4 * z["hours_until_departure"])
        need = (z["connected_ev_required_soc_departure"] - z["connected_ev_soc"]) * z[
            "connected_ev_battery_capacity_kwh"
        ]
        (outlook,) = obs.outlooks
        prices = np.array(outlook.values[0][:n])
        pv = np.array(outlook.values[2][:n]) * 0.25
        key = outlook_id(prices, pv)
        allowed = domains[n, state_id(need)]
        current_mask[:] = allowed
        chosen[0] = ACTIONS[np.argmax(np.where(allowed, table[key, state_id(need)], -np.inf))]
        return original(obs)

    def choice(c, ctx, memory, deterministic):
        q = chosen[0]
        if c.stage == "mode":
            mode = actor.group_modes[c.group_type].index("CHARGE_EV" if q else "IDLE")
            assert mode in c.allowed
            return replace(c, mode=mode)
        assert np.array_equal(
            current_mask[1:], (ACTIONS[1:] >= c.lower - 1e-7) & (ACTIONS[1:] <= c.upper + 1e-7)
        )
        assert c.lower - 1e-7 <= q <= c.upper + 1e-7, (q, c)
        return replace(c, fraction=float(q))

    with patch.object(actor, "_contexts", context), patch.object(actor, "_choose", choice):
        rows = evaluate(cases, agent, None, probe)
        stops = [
            sum(play(replace(s, request_kwh=0.0, early=b), agent, None, probe, "eval")["kwh"])
            for s in cases
            for b in (True, False)
        ]
    return measures(rows), sum(q < 1e-8 for q in stops)


def train(seed, episodes=262144, include_arrivals_at_target=False):
    pool, dev, tests = split()
    domains = physical_domains(pool[0])
    rng = np.random.default_rng(seed)
    shapes = (len(pool), 8)
    codes = np.zeros(shapes, int)
    price = np.zeros(shapes)
    pv = np.zeros(shapes)
    for i, s in enumerate(pool):
        for t in range(s.quarters):
            pp, ss = curves(s, t)
            codes[i, t] = outlook_id(pp, ss)
            price[i, t] = pp[0]
            pv[i, t] = ss[0]
    duration = np.array([c.quarters for c in pool])
    target = np.array([c.request_kwh for c in pool])
    table = np.zeros((8 * 8 * 3 * 3 * 4 * 4, 18, 3))
    # Random tiny tie-breaks prevent a preference for the lowest power at equal return.
    table += rng.uniform(-1e-10, 1e-10, size=table.shape)
    ids = np.arange(256)
    assert episodes % len(ids) == 0
    # Match the native learner's four independent attempts per starting condition.
    scenario_indices = np.random.default_rng(seed).integers(len(pool), size=episodes // 4)
    schedule = np.repeat(scenario_indices, 4).reshape(-1, len(ids))
    visited = np.zeros(table.size, dtype=int)
    audited = 0
    for batch in range(episodes // len(ids)):
        i = schedule[batch]
        n = duration[i].copy()
        need = target[i].copy()
        if include_arrivals_at_target:
            vary = rng.random(len(i)) < 0.35
            need[vary] = rng.choice([-0.5, 0.0, 0.0, 0.5, 1.0], size=int(vary.sum()))
        for t in range(8):
            active = n > 0
            k = codes[i, t]
            j = state_id(need)
            free = domains[np.maximum(1, n), j]
            random = rng.random((len(ids), 3))
            exploration = rng.random(len(ids)) < max(0.15, 1 - batch / (episodes / len(ids)))
            action = np.argmax(
                np.where(free, np.where(exploration[:, None], random, table[k, j]), -np.inf), axis=1
            )
            delivered = ACTIONS[action]
            rest = need - delivered
            penalty = np.where(n == 1, np.clip(rest, 0.0, 1.0), 0.0)
            bill = price[i, t] * np.maximum(0.0, delivered - pv[i, t])
            assert np.all(rest[active] <= n[active] + 1e-7)
            reward = -(bill + penalty) / 30.0
            if batch < 2:
                for pos in ids[active][:4]:
                    ledger = dict.fromkeys(TERM_WEIGHTS, 0.0)
                    ledger.update(energy_cost_eur=bill[pos], ev_tolerance_kwh=penalty[pos])
                    actual = operating_reward(
                        {"thesis_reward_terms": ledger},
                        reference_sites=30,
                        profile="thesis_common_reward_v1",
                    ).value
                    assert abs(actual - reward[pos]) < 1e-10
                    audited += 1
            next_q = table[codes[i, min(7, t + 1)], state_id(rest)]
            best = np.max(
                np.where(domains[np.maximum(1, n - 1), state_id(rest)], next_q, -np.inf), axis=1
            )
            returns = reward + np.where(n > 1, GAMMA * best, 0.0)
            flat = np.ravel_multi_index((k[active], j[active], action[active]), table.shape)
            keys, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
            avg = np.bincount(inv, weights=returns[active]) / counts
            lr = np.maximum(0.035, 1 / (1 + visited[keys] / 24) ** 0.60)
            table.flat[keys] += lr * (avg - table.flat[keys])
            visited[keys] += counts
            need = rest
            n -= 1
    out, stops = test_policy(table, tests["new"], seed, domains)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    condition = "mixed" if include_arrivals_at_target else "standard"
    extra = {}
    if include_arrivals_at_target:
        from scripts.train_ti_marl_ev_sessions import make_cases

        _, pool = make_cases(schema_path(), 0, 100)
        extra_cases = [s for s in pool if s.day not in RESERVED_DATES][:40]
        assert len({s.day for s in extra_cases}) == 20
        extra["test"], extra["at_target_idle_out_of_80"] = test_policy(
            table, extra_cases, seed, domains
        )
        extra["dates"] = sorted({s.day for s in extra_cases})
    (OUTPUT / f"q-control-{condition}-{episodes}-{seed}.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "training_order": hashlib.sha256(scenario_indices.tobytes()).hexdigest(),
                "seed": seed,
                "test": out,
                "training_starts": condition,
                "additional_test": extra,
                "at_target_idle_out_of_80": stops,
                "audited_rewards": audited,
                "updates_from_sampled_transitions": int(visited.sum()),
                "actions_kw": [0, 2, 4],
            },
            allow_nan=False,
        )
    )
    print(
        seed,
        out["price"]["timing_and_target"],
        out["solar"]["timing_and_target"],
        out["near_target"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", type=int)
    p.add_argument("--episodes", type=int, default=262144)
    p.add_argument("--mixed-starts", action="store_true")
    args = p.parse_args()
    train(args.seed, args.episodes, args.mixed_starts)
