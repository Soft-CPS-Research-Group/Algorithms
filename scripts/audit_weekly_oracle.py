"""Verify weekly solver inputs and executed starts against one continuous Simulator export."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def audit(job):
    manifests = list((job / "results/simulation_data").rglob("exported_kpis.manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected one exported evaluation episode.")
    export = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    first = manifest["episode_start_time_step"]
    end = manifest["episode_end_time_step"]
    total = manifest["completed_intervals"]
    seconds = manifest["seconds_per_time_step"]
    week, remainder = divmod(7 * 86400, seconds)
    if remainder or end - first != total:
        raise ValueError("Incomplete episode or nonintegral week.")
    decisions = [
        json.loads(line)
        for line in (job / "offline_training/milp_decisions.jsonl").read_text().splitlines()
    ]
    if [r["source_start"] for r in decisions] != list(range(first, end, int(week))):
        raise ValueError("Missing, repeated or misaligned weekly solve.")

    @lru_cache(None)
    def column(stem, name):
        parts = sorted(export.glob(f"exported_data_{stem}_ep0_part*.parquet"))
        if not parts:
            raise ValueError(f"Missing export for {stem}.")
        data = pd.concat(
            [
                pd.read_parquet(p, columns=[name], use_threads=False, pre_buffer=False)[name]
                for p in parts
            ],
            ignore_index=True,
        )
        if len(data) != total:
            raise ValueError(f"Incomplete trajectory for {stem}.")
        return data

    times = pd.to_datetime(column("community", "timestamp"), utc=True)
    if not times.diff().iloc[1:].eq(pd.Timedelta(seconds=seconds)).all():
        raise ValueError("Simulator timestamps skip or repeat intervals.")
    ledger = json.loads((export / "exported_kpis.deferrable_service.json").read_text())
    offered = {
        "::".join((r["member_id"], r["appliance_id"], r["cycle_id"])): r
        for r in ledger["records"]
        if r["included"]
    }
    selected = {}
    profiles = {}
    boundaries = []
    max_net_error = 0.0
    for decision in decisions:
        start = decision["source_start"]
        stop = min(start + int(week), end)
        offset = start - first
        if (
            decision["source_end_exclusive"] != stop
            or decision["failed"]
            or not decision["solver"]["has_solution"]
        ):
            raise ValueError(f"Incomplete solve at {start}.")
        block = job / "offline_training" / f"episode-{decision['episode']}" / f"week-{start}"
        issue = json.loads((block / "problem.json").read_text())
        plan = json.loads((block / "result.json").read_text())
        meta = issue["metadata"]
        # Building totals cover every battery flow, EV action and appliance phase.
        for bid, model in zip(issue["building_ids"], plan["building_net_load_kwh"], strict=True):
            physical = column(bid.lower(), "Net Electricity Consumption-kWh").iloc[
                offset : stop - first
            ]
            if len(model) != len(physical):
                raise ValueError(f"Wrong trajectory length for {bid}.")
            error = float(np.max(np.abs(physical.to_numpy() - model)))
            if not np.isfinite(error) or error > 1e-4:
                raise ValueError(f"Building {bid} at {start}: {error} kWh replay error.")
            max_net_error = max(max_net_error, error)

        running = set(decision["running_cycle_ids"])
        actual_running = {
            cid for cid, began in selected.items() if began < start < began + len(profiles[cid])
        }
        if running != actual_running:
            raise ValueError("Running appliance identities changed at the weekly boundary.")
        cycles = {c["cycle_id"]: c for c in issue["deferrable_cycles"]}
        if offset:
            bess, ev, arrivals = {}, {}, {}
            for a in issue["stationary_storage"]:
                bid = a["building_id"]
                raw = (
                    float(column(bid.lower() + "_battery", "Battery Soc-%")[offset - 1])
                    * a["capacity_kwh"]
                )
                expected = raw * (1 - a["loss_coefficient"])
                bess[bid] = {
                    "stored_kwh": raw,
                    "model_initial_kwh": a["initial_energy_kwh"],
                    "error_kwh": abs(a["initial_energy_kwh"] - expected),
                }
            sessions = {
                (s["building_id"], s["action_name"]): s
                for s in issue["ev_sessions"]
                if s["start_time_step"] == 0
            }
            for entry in meta["action_inventory"]:
                action = entry["action_name"]
                if not action.startswith("electric_vehicle_storage_"):
                    continue
                bid = entry["building_id"]
                stem = bid.lower() + "_" + action.removeprefix("electric_vehicle_storage_")
                states = column(stem, "EV Charger State")
                s = sessions.get((bid, action))
                if (states[offset] == 1) != (s is not None):
                    raise ValueError(f"Wrong connected charger at {start}: {stem}.")
                if s is None:
                    continue
                identity = meta["ev_identity_by_session"][s["session_id"]]
                names = column(stem, "EV Name")
                if names[offset] != identity:
                    raise ValueError(f"Wrong connected EV at {start}: {stem}.")
                soc = column(identity.lower(), "electric_vehicle_soc")
                if states[offset - 1] == 1 and names[offset - 1] == identity:
                    raw = float(soc[offset - 1]) * s["capacity_kwh"]
                    group = ev
                else:
                    # At a fresh arrival the previous disconnected SOC is not the
                    # arrival state. Undo the first executed (charge-only) interval.
                    delivered = max(float(column(stem, "Charger Consumption-kWh")[offset]), 0)
                    raw = (
                        float(soc[offset]) * s["capacity_kwh"] - delivered * s["charge_efficiency"]
                    )
                    group = arrivals
                group[identity] = {
                    "charger": stem,
                    "stored_kwh": raw,
                    "model_initial_kwh": s["initial_energy_kwh"],
                    "error_kwh": abs(s["initial_energy_kwh"] - raw),
                }
            if any(
                not np.isfinite(x["error_kwh"]) or x["error_kwh"] > 1e-4
                for x in [*bess.values(), *ev.values(), *arrivals.values()]
            ):
                raise ValueError("Measured BESS or continuing EV energy was not transferred.")
            for cid in running:
                residual = profiles[cid][start - selected[cid] :][: stop - start]
                if cycles[cid]["load_profile_kwh"] != residual:
                    raise ValueError(f"Wrong appliance phase at {start}: {cid}.")
            pending = [
                cid
                for cid, rec in offered.items()
                if rec["earliest_start_time_step"] < start <= rec["latest_start_time_step"]
                and rec["start_time_step"] is not None
                and rec["start_time_step"] >= start
            ]
            if not set(pending).issubset(cycles):
                raise ValueError("An outstanding appliance request disappeared.")
            boundaries.append(
                {
                    "utc": times.iloc[offset].isoformat(),
                    "storage": bess,
                    "continuing_evs": ev,
                    "arriving_evs": arrivals,
                    "running_cycles": sorted(running),
                    "pending_cycles": sorted(pending),
                }
            )
        for cid, step in plan["selected_deferrable_starts"].items():
            if cid in running or step is None:
                continue
            if (
                cid in selected
                or cid not in offered
                or offered[cid]["start_time_step"] != start + step
            ):
                raise ValueError(f"Duplicate, missing or mistimed appliance start: {cid}.")
            selected[cid] = start + step
            profiles[cid] = cycles[cid]["load_profile_kwh"]

    district = ledger["district"]
    if (
        set(selected) != set(offered)
        or district["completed_count"] != district["offered_count"]
        or district["duplicate_offered_count"]
        or abs(district["physical_balance_residual_kwh"]) > 1e-4
    ):
        raise ValueError("Appliance starts, completion or executed energy do not reconcile.")
    return {
        "passed": True,
        "intervals": total,
        "solves": len(decisions),
        "completed_cycles": len(selected),
        "max_building_replay_error_kwh": max_net_error,
        "boundaries": boundaries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    print(json.dumps(audit(parser.parse_args().job_dir), indent=2, allow_nan=False))
