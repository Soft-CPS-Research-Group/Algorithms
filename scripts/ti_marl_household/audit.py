"""Pair timing interventions while preserving the physical reservation and actuators."""

import argparse
import json
from contextlib import ExitStack
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.ti_marl_household import experiment


def run_control(task):
    case, weights, condition = task
    if "transfer" in condition:
        weights = np.zeros((5, 8))
    episodes = experiment.episodes
    actor = episodes.agent.actor
    original = actor._contexts

    def without_forecast(snapshot):
        curves = []
        for c in snapshot.outlooks:
            price, load, pv = c.values
            curves.append(replace(c, values=((0.16,) * len(price), load, (0.0,) * len(pv))))
        # The reservation and simulator still receive their original, unaltered inputs.
        return original(replace(snapshot, outlooks=tuple(curves)))

    with ExitStack() as changes:
        if "blind" in condition:
            changes.enter_context(patch.object(actor, "_contexts", without_forecast))
        if "strict" in condition:
            changes.enter_context(patch.object(actor, "tolerance_soc", 0.001))
        return episodes.episode(case, weights)


def metrics(rows):
    errors = np.asarray([r["ev_error"] for r in rows])
    return dict(
        cases=len(rows),
        departures=errors.size,
        ready_within_01=int((errors >= -0.1).sum()),
        within_01=int((abs(errors) <= 0.1).sum()),
        mean_abs_error_kwh=float(abs(errors).mean()),
        worst_deficit_kwh=float(max(0, -errors.min())),
        worst_excess_kwh=float(max(0, errors.max())),
        completed_programmes=sum(r["programme_complete"] for r in rows),
        grid_violations=sum(r["max_violation"] > 1e-5 for r in rows),
        peak_kw=max(r["peak"] for r in rows),
        fallbacks=sum(r["fallback"] for r in rows),
        mean_bill_eur=float(np.mean([r["bill"] for r in rows])),
        mean_bill_with_refill_eur=float(np.mean([r["stock_adjusted_bill"] for r in rows])),
        mean_terminal_stock_change_kwh=float(np.mean([r["stock_delta"] for r in rows])),
        mean_stationary_throughput_kwh=float(np.mean([r["battery_throughput"] for r in rows])),
        mean_episode_reward=float(np.mean([r["reward"] for r in rows])),
        at_limit_steps=sum(np.count_nonzero(np.array(r["flows"])[:, 0] * 4 > 6.899) for r in rows),
        steps=sum(len(r["flows"]) for r in rows),
    )


def timing(rows, stimulus):
    pairs = {}
    for r in rows:
        if r["stimulus"] == stimulus:
            pairs.setdefault((r["day"], r["first"]), {})[r["early"]] = r
    deltas = []
    for (_, first), pair in pairs.items():
        # The later-departing vehicle can actually use both offered windows.
        j = 2 - first
        early, late = np.array(pair[True]["flows"]), np.array(pair[False]["flows"])
        deltas.append(
            [
                early[2:9, j].sum() - late[2:9, j].sum(),
                late[10:17, j].sum() - early[10:17, j].sum(),
                pair[False]["programme_start"] - pair[True]["programme_start"],
                late[10:17, 4].sum() - early[10:17, 4].sum(),
            ]
        )
    d = np.array(deltas)
    return dict(
        pairs=len(d),
        late_ev_moves_both_ways=int((np.min(d[:, :2], axis=1) > 0.1).sum()),
        mean_ev_left_early_kwh=float(d[:, 0].mean()),
        mean_ev_added_late_kwh=float(d[:, 1].mean()),
        programme_moves_later=int((d[:, 2] > 0).sum()),
        mean_programme_shift_quarters=float(d[:, 2].mean()),
        mean_programme_added_late_kwh=float(d[:, 3].mean()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--processes", type=int, default=4)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--ppo-checkpoint", type=Path)
    source.add_argument("--transfer", action="store_true")
    p.add_argument("--strict-only", action="store_true")
    p.add_argument("--departure-margin", type=int, default=0)
    p.add_argument(
        "--subset", choices=("dev", "test", "confirmation", "replication", "final"), default="test"
    )
    p.add_argument("--final-controls", action="store_true")
    args = p.parse_args()
    directory = args.directory
    cases = json.loads((directory / "cases.json").read_text())
    weights = np.zeros((5, 8))
    if not args.ppo_checkpoint and not args.transfer:
        weights = json.loads((directory / f"refined-{args.seed}.json").read_text())["weights"]
    ids = [i for i, c in enumerate(cases) if c["split"] == args.subset]
    if args.subset == "test" and not (args.ppo_checkpoint or args.transfer or args.strict_only):
        rows = json.loads((directory / f"test-refined-{args.seed}.json").read_text())
    else:
        rows = {}
    if args.ppo_checkpoint:
        import torch

        weights = torch.load(args.ppo_checkpoint, map_location="cpu", weights_only=False)["actor"][
            "coordination"
        ].numpy()
    params = (directory, args.seed, args.ppo_checkpoint, args.departure_margin)
    with get_context("fork").Pool(args.processes, experiment.worker_init, params) as pool:
        if not rows and not args.strict_only and not args.final_controls:
            rows["trained"] = pool.map(experiment.evaluate, [(i, weights) for i in ids])
        controls = (
            ("strict",)
            if args.subset == "dev" or args.strict_only
            else ("blind", "strict", "blind_strict")
        )
        if args.final_controls:
            controls = ("strict", "transfer_strict", "blind_strict")
        for condition in controls:
            rows[condition] = pool.map(
                run_control, [(i, weights, condition) for i in ids], chunksize=1
            )
    if args.final_controls:
        sample = [r for r in rows["strict"] if r["day"] == cases[ids[0]]["day"] and r["first"] == 0]
        (directory / "examples.json").write_text(json.dumps(sample))
    summary = {
        arm: dict(
            overall=metrics(data),
            **{s: timing(data, s) for s in ("price", "solar")},
            untouched_curves=metrics([r for r in data if r["stimulus"] == "native"]),
        )
        for arm, data in rows.items()
    }
    kind = "ppo-" if args.ppo_checkpoint else "transfer-" if args.transfer else ""
    if args.strict_only:
        kind += "strict-only-"
    if args.departure_margin:
        kind += f"margin{args.departure_margin}-"
    (directory / f"audit-{kind}{args.subset}-{args.seed}.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
