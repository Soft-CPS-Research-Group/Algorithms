"""Replay sequences for the recorded EV departure using the stored seed-71 policy.

Run python -m scripts.ti_marl_household.ev_sequences DIRECTORY in Algorithms.
The directory holds the reconstructed Core-30 fixture and the sequence results.
"""

import argparse
import itertools
from dataclasses import replace
from functools import partial
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.ti_marl_household import community
from scripts.ti_marl_household import ev_power as power
from scripts.ti_marl_household import paired_choices as paired
from scripts.ti_marl_household import start_timing as timing


class InfeasibleSequence(Exception):
    """A requested charge/wait mode is outside the site's native action mask."""


def measure(env, case, port):
    building = env.buildings[port]
    charger = building.electric_vehicle_chargers[0]
    departure = case["departures"][port] - 1
    ev = charger.past_connected_evs[departure]
    capacity = float(ev.battery.capacity)
    target = float(charger.charger_simulation.electric_vehicle_required_soc_departure[departure])
    soc = float(ev.battery.soc[departure])
    n = community.DECISIONS
    tariff = np.array(
        [r[port]["grid_import_price"] for r in env._community_market_settlement_history[:n]]
    )
    cheap = np.isclose(tariff, tariff.min())
    end_discount = np.flatnonzero(cheap & (np.arange(n) <= departure))[-1]
    end_soc = float(ev.battery.soc[end_discount])
    ev_ac = charger.electricity_consumption[:n]
    appliance = env.buildings[1].deferrable_appliances_electricity_consumption[:n]
    return dict(
        capacity_kwh=capacity,
        target_soc=target,
        departure_soc=soc,
        soc_at_end_discount=end_soc,
        error_at_end_discount_kwh=(end_soc - target) * capacity,
        error_kwh=(soc - target) * capacity,
        ev_ac_kwh=ev_ac.tolist(),
        cheap_slots=np.flatnonzero(cheap).tolist(),
        ev_outside_kwh=float(ev_ac[~cheap].sum()),
        programme_outside_kwh=float(appliance[~cheap].sum()),
    )


def replay(plan, *, case, port, group, slots):
    """Replay from reset, using the native feasible interval for every requested power."""
    env = timing.environment(case)
    actor = community.policy.actor
    native = actor._choose
    pulse = dict(zip(slots, plan, strict=True)) if plan is not None else {}

    def choose(c, context, state, deterministic):
        choice = native(c, context, state, deterministic)
        if plan is None or c.group_id != group:
            return choice
        # This diagnostic searches for an all-discount sequence. Fixed physical
        # choices must agree too; no fallback or safety mask is overridden.
        value = pulse.get(env.time_step)
        if c.stage == "mode":
            mode = actor.group_modes[c.group_type].index("IDLE" if value is None else "CHARGE_EV")
            if mode not in c.allowed:
                raise InfeasibleSequence
            return replace(choice, mode=mode)
        if c.stage == "power" and value is not None and value != "actor" and c.sampled:
            return replace(
                choice, unit_sample=value, fraction=c.lower + (c.upper - c.lower) * value
            )
        return choice

    try:
        with patch.object(actor, "_choose", choose):
            r = power.rollout(case, env=env)
    except InfeasibleSequence:
        return dict(plan=plan, feasible=False)
    m = measure(env, case, port)
    np.testing.assert_allclose(m["error_kwh"], r["ev_errors_kwh"][port], atol=1e-7)
    return dict(
        plan=plan,
        feasible=True,
        **m,
        reward=r["discounted_reward"],
        undiscounted_reward=r["undiscounted_reward"],
        bill=r["bill_eur"],
        dishwasher_start=r["inspection"]["start"],
        complete=r["programme_complete"],
        other_ev_error_kwh=r["ev_errors_kwh"][1 - port],
    )


def applied(r):
    if not r["feasible"]:
        return False
    requested = [slot for slot, v in zip(r["cheap_slots"], r["plan"], strict=True) if v is not None]
    actual = np.flatnonzero(np.asarray(r["ev_ac_kwh"]) > 1e-6).tolist()
    return requested == actual


def qualifies(r, baseline, tolerance):
    return (
        applied(r)
        and r["complete"]
        and r["ev_outside_kwh"] < 1e-6
        and r["programme_outside_kwh"] < 1e-6
        and abs(r["other_ev_error_kwh"] - baseline["other_ev_error_kwh"]) < 1e-6
        and abs(r["error_kwh"]) <= tolerance
    )


def scan(d, workers):
    case = power.failed_case(d)
    community.init_eval(d, power.SOURCE)
    past = community.read_json(power.paired.RUNS / "ev-power-71.json")["regression"]
    port = past["port"]
    obs = power.rollout(case, trace=True)
    (group,) = {r["group"] for r in obs["decisions"] if r["port"] == port}
    native_env = timing.environment(case)
    power.rollout(case, env=native_env)
    slots = measure(native_env, case, port)["cheap_slots"]
    run = partial(replay, case=case, port=port, group=group, slots=slots)
    baseline = run(None)
    original = past["traces"]["source"]
    np.testing.assert_allclose(
        [baseline["error_kwh"], baseline["reward"], baseline["bill"]],
        [original["ev_errors_kwh"][port], original["discounted_reward"], original["bill_eur"]],
        rtol=0,
        atol=1e-7,
    )
    rng = np.random.default_rng(71)
    masks = list(itertools.product((False, True), repeat=len(slots)))
    plans = []
    for mask in masks:
        for unit in ("actor", 0.0, 0.5, 1.0):
            plans.append(tuple(unit if active else None for active in mask))
    rows = paired.parallel(run, plans, workers)

    def feasible_masks():
        return sorted({tuple(v is not None for v in r["plan"]) for r in rows if applied(r)})

    active = feasible_masks()
    print("Enumerated", len(masks), "masks;", len(active), "feasible at sampled powers", flush=True)
    # Sample mixed continuous powers for every feasible mask, independently of
    # its earlier reward. Include mixed powers for the other masks as a check.
    plans = []
    for mask in masks:
        for _ in range(12 if mask in active else 1):
            plans.append(tuple(float(rng.random()) if on else None for on in mask))
    rows.extend(paired.parallel(run, plans, workers))
    final = analyse(rows, baseline)
    final.update(
        simulator=version("softcpsrecsimulator"),
        case=case,
        port=port,
        checkpoint=str(power.SOURCE),
        baseline=baseline,
        binary_patterns=len(masks),
        patterns_with_feasible_power=len(feasible_masks()),
    )
    for label, r in final["examples"].items():
        if r is None:
            continue
        repeat = run(r["plan"])
        for field in ("reward", "bill", "departure_soc", "ev_ac_kwh", "programme_outside_kwh"):
            np.testing.assert_allclose(repeat[field], r[field], rtol=0, atol=1e-7)
        final["examples"][label] = repeat
    community.write_json(d / "ev-sequences-71.json", final)


def analyse(rows, baseline):
    good = [r for r in rows if r["feasible"]]
    same = [r for r in good if qualifies(r, baseline, float("inf"))]
    counts = {}
    examples = {}
    for tolerance in (0.5, 0.1, 0.01):
        group = [r for r in same if qualifies(r, baseline, tolerance)]
        better = [r for r in group if r["reward"] > baseline["reward"] + 1e-6]
        counts[str(tolerance)] = dict(
            achieved=len(group),
            better_reward=len(better),
            cheaper_and_better=sum(r["bill"] < baseline["bill"] - 1e-6 for r in better),
        )
        examples[str(tolerance)] = max(group, key=lambda r: r["reward"], default=None)
    examples["closest"] = min(same, key=lambda r: (abs(r["error_kwh"]), -r["reward"]), default=None)
    return dict(
        attempts=len(rows),
        completed_rollouts=len(good),
        dispatch_matches_meter=sum(applied(r) for r in good),
        same_service_and_discount=len(same),
        counts=counts,
        examples=examples,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if not (args.directory / "cases.json").is_file():
        community.make_fixture(
            args.directory,
            community.schema_path(),
            case_days=power.previous.DAYS,
            price_start=power.previous.price_start,
        )
    scan(args.directory, args.workers)
