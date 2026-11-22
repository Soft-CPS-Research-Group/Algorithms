"""Attribute settled price-window costs to the actual AC meters of the four devices."""

import argparse
from collections import Counter
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from scripts.ti_marl_household import community

LOW, HIGH = 0.07, 0.25
METERS = ("ev_a", "ev_b", "programme", "battery", "fixed", "grid")
PERIODS = ("before", "cheap", "after")
RUNS = Path("runs/ti-marl-2023-three-houses")


def inspect_episode(env, case):
    n = community.DECISIONS
    a, b, c = env.buildings
    # Read the settled import price, and the AC power actually delivered, after the last decision.
    settlement = [env._community_market_settlement_history[t] for t in range(n)]
    prices = np.array([[row["grid_import_price"] for row in rows] for rows in settlement])
    cheap = np.isclose(prices[:, 0], LOW)
    indices = np.flatnonzero(cheap)
    expected = np.arange(2, 9) if case["early"] else np.arange(10, 17)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(prices, np.tile(np.where(cheap, LOW, HIGH)[:, None], (1, 3)))
    for key in ("local_import_kwh", "grid_export_kwh"):
        np.testing.assert_allclose(
            [sum(row[key] for row in rows) for rows in settlement], 0, atol=1e-6
        )
    for member in env.buildings:
        np.testing.assert_allclose(member.solar_generation[:n], 0, atol=1e-6)
    meters = np.column_stack(
        (
            a.electric_vehicle_chargers[0].electricity_consumption[:n],
            b.electric_vehicle_chargers[0].electricity_consumption[:n],
            b.deferrable_appliances_electricity_consumption[:n],
            c.electrical_storage_electricity_consumption[:n],
            sum(h.non_shiftable_load_electricity_consumption[:n] for h in env.buildings),
            [sum(row["grid_import_kwh"] for row in rows) for rows in settlement],
        )
    ).astype(float)
    np.testing.assert_allclose(meters[:, :-1].sum(axis=1), meters[:, -1], atol=2e-6)
    np.testing.assert_allclose(
        meters[:, -1] * prices[:, 0],
        [sum(r["settled_cost_eur"] for r in rows) for rows in settlement],
        atol=2e-7,
    )
    blocks = (meters[: indices[0]], meters[cheap], meters[indices[-1] + 1 :])
    return dict(
        kwh={
            key: dict(zip(PERIODS, (float(block[:, i].sum()) for block in blocks)))
            for i, key in enumerate(METERS)
        },
        first_active_slot={
            key: int(np.flatnonzero(meters[:, i] > 1e-6)[0]) for i, key in enumerate(METERS[:3])
        },
        early_price=case["early"],
        first_departure=case["first"],
    )


def replay(directory, arm, workers):
    path = community.SOURCE_POLICY if arm == "frozen" else RUNS / arm / "latest_checkpoint.pth"
    community.init_eval(directory, path)
    cases = [
        c
        for c in community.read_json(directory / "cases.json")
        if c["split"] == "test" and c["stimulus"] == "price"
    ]
    with get_context("fork").Pool(workers) as pool:
        rows = pool.map(partial(community.evaluate_case, inspect_episode=inspect_episode), cases)
    community.write_json(directory / f"price-{arm}.json", rows)


def mean_ledger(rows):
    result = {}
    for meter in METERS:
        blocks = {
            period: float(np.mean([r["inspection"]["kwh"][meter][period] for r in rows]))
            for period in PERIODS
        }
        result[meter] = dict(
            **blocks, bill_eur=blocks["cheap"] * LOW + (blocks["before"] + blocks["after"]) * HIGH
        )
    return result


def paired(reference, current):
    if [r["start"] for r in reference] != [r["start"] for r in current]:
        raise ValueError("Expected exactly the same cases in the same order")
    a, b = mean_ledger(reference), mean_ledger(current)
    result = {}
    for meter in METERS:
        cheap_loss = a[meter]["cheap"] - b[meter]["cheap"]
        outside_gain = sum(b[meter][p] - a[meter][p] for p in ("before", "after"))
        total_change = outside_gain - cheap_loss
        bill_change = b[meter]["bill_eur"] - a[meter]["bill_eur"]
        np.testing.assert_allclose(
            cheap_loss * (HIGH - LOW) + total_change * HIGH, bill_change, atol=1e-8
        )
        result[meter] = dict(
            net_kwh_leaving_cheap=cheap_loss,
            added_expensive_kwh=outside_gain,
            total_kwh_change=total_change,
            bill_change_eur=bill_change,
        )
    shifts = {}
    for meter in METERS[:3]:
        minutes = Counter(
            15
            * (
                y["inspection"]["first_active_slot"][meter]
                - x["inspection"]["first_active_slot"][meter]
            )
            for x, y in zip(reference, current)
        )
        shifts[meter] = dict(sorted(minutes.items()))
    return dict(n=len(current), mean_per_window=result, first_active_shift_minutes=shifts)


def report(directory, destination):
    previous = community.read_json(RUNS / "comparison.json")
    arms = {
        arm: community.read_json(directory / f"price-{arm}.json")
        for arm in ("frozen", "joint-43", "joint-71")
    }
    frozen = arms["frozen"]
    result = dict(
        windows=len(frozen),
        dates=previous["evaluation_dates"],
        decisions_per_window=23,
        low_eur_per_kwh=LOW,
        high_eur_per_kwh=HIGH,
        units="Net metered AC kWh and EUR per window unless labelled otherwise. Battery discharge is negative.",
        devices={
            "ev_a": "Member_001 / CHG-M001-01",
            "ev_b": "Member_006 / CHG-M006-02",
            "programme": "Member_006 / DEF-M006-02 (dishwasher)",
            "battery": "Member_014",
        },
        ledger={},
        changes_vs_frozen={},
        by_opportunity={},
    )
    for arm, rows in arms.items():
        actual = community.summarize(rows)
        original = previous["by_stimulus"][arm]["price"]
        for key in original:
            np.testing.assert_allclose(actual[key], original[key], rtol=1e-7, atol=1e-7)
        ledger = mean_ledger(rows)
        np.testing.assert_allclose(ledger["grid"]["bill_eur"], original["bill_eur"], atol=2e-7)
        result["ledger"][arm] = ledger
        if arm == "frozen":
            continue
        result["changes_vs_frozen"][arm] = paired(frozen, rows)
        result["by_opportunity"][arm] = {
            "early" if early else "late": paired(
                [r for r in frozen if r["inspection"]["early_price"] == early],
                [r for r in rows if r["inspection"]["early_price"] == early],
            )
            for early in (True, False)
        }
    # Show the first matched late-price case so the effect of a single start can be inspected.
    example = next(i for i, row in enumerate(frozen) if not row["inspection"]["early_price"])
    result["first_late_case"] = {
        "start": frozen[example]["start"],
        "source_day_0based": frozen[example]["day"],
        "cheap_slots_0based": [10, 16],
        "by_arm": {
            arm: {
                "first_dishwasher_slot_0based": rows[example]["inspection"]["first_active_slot"][
                    "programme"
                ],
                "dishwasher_cheap_kwh": rows[example]["inspection"]["kwh"]["programme"]["cheap"],
                "bill_eur": rows[example]["bill_eur"],
            }
            for arm, rows in arms.items()
        },
    }
    community.write_json(destination, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Output of community prepare")
    parser.add_argument("arm", choices=("frozen", "joint-43", "joint-71", "report"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=RUNS / "price-windows.json")
    args = parser.parse_args()
    if args.arm == "report":
        report(args.directory, args.output)
    else:
        replay(args.directory, args.arm, args.workers)
