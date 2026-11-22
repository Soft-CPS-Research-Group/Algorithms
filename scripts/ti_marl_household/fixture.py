"""Controlled two-port Core-30 household for independent 2023 contention episodes."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.dump_entity_obs_sample import _build_env
from scripts.generate_typed_interfaces import generate, generate_simulator_bindings, write_generated
from scripts.ti_marl_ev.protocol import schema_path

SITE = "Member_001"
PORTS = ("CHG-M001-01", "CHG-M006-02")
CARS = ("EV-M001-01", "EV-M006-02")
SLOTS = 24


def evaluation_job(cfg, checkpoint, start):
    cfg["job"] = {"profile": "evaluate"}
    cfg["training"]["episodes"] = 0
    cfg["evaluation"] = {"episodes": 1}
    cfg["pipeline"][0]["hyperparameters"]["actor"].setdefault("readout", "household_start")
    cfg["checkpointing"] = {
        "resume_training": True,
        "checkpoint_local_path": str(checkpoint.resolve()),
    }
    cfg["simulator"].update(
        simulation_start_time_step=start, simulation_end_time_step=start + SLOTS - 1
    )
    return cfg


def prepare(
    out,
    source,
    checkpoint=None,
    case_index=0,
    tolerance_soc=0.05,
    cycle_mean=False,
    margin=0,
    case_days=None,
    price_start=None,
    session_seed=89273,
):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    schema = json.loads(source.read_text())
    root = (source.parent / schema["root_directory"]).resolve()
    b = schema["buildings"][SITE]
    donor = schema["buildings"]["Member_006"]
    app_id = "DEF-M006-02"
    app = deepcopy(donor["deferrable_appliances"][app_id])
    profiles = pd.read_parquet(root / app["cycle_profiles_file"])
    profile = profiles.iloc[0]
    app["cycle_profiles_file"] = str(root / app["cycle_profiles_file"])
    app["flexibility_schedule_file"] = str(out / "cycles.parquet")
    b["deferrable_appliances"] = {app_id: app}
    b["chargers"][PORTS[1]] = deepcopy(donor["chargers"]["CHG-M006-02"])
    for port in PORTS:
        b["chargers"][port]["attributes"]["phase_connection"] = "L1"
    raw = pd.read_parquet(root / b["energy_simulation"])
    tariff = pd.read_parquet(root / b["pricing"])
    schema["electric_vehicles_def"] = {name: schema["electric_vehicles_def"][name] for name in CARS}
    schema["buildings"] = {SITE: b}
    schema["root_directory"] = str(root)
    rng = np.random.default_rng(session_seed)
    cases, cycles, energy, prices = [], [], [], []
    chargers = [[], []]
    if case_days is None:
        case_days = (
            ("train", [12, 40, 65, 93, 128, 152, 187, 216]),
            ("dev", [226, 234]),
            ("test", [260, 284, 307, 326, 345, 359]),
            ("confirmation", [253, 277, 297, 319, 337, 351]),
            ("replication", [249, 270, 293, 313, 334, 357]),
            ("final", [247, 264, 288, 309, 332, 353]),
        )
    for split, days in case_days:
        for day in days:
            raw_offset = day * 96 + 40
            demand = rng.uniform(2.8, 3.9, 2).round(3)
            early_departure = int(rng.integers(9, 12))
            late_departure = int(rng.integers(18, 22))
            for stimulus in ("price", "solar", "native"):
                for early in (True,) if stimulus == "native" else (False, True):
                    for first_port in (0, 1):
                        start = len(cases) * SLOTS
                        deadlines = [early_departure, late_departure]
                        if first_port == 1:
                            deadlines.reverse()
                        # Equal requested kWh under a port swap, irrespective of EV capacity.
                        needs = demand if first_port == 0 else demand[::-1]
                        cases.append(
                            dict(
                                start=start,
                                day=day,
                                split=split,
                                stimulus=stimulus,
                                early=early,
                                first=first_port,
                                needs=needs.tolist(),
                                departures=deadlines,
                                programme_request_kwh=float(profile.total_energy_kwh),
                            )
                        )
                        e = raw.iloc[raw_offset : raw_offset + SLOTS].copy()
                        p = tariff.iloc[raw_offset : raw_offset + SLOTS].astype(float)
                        first = 2 if early else 10
                        if stimulus == "price" and price_start is not None:
                            first = price_start(split, day, early)
                            assert 0 <= first <= SLOTS - 7
                            cases[-1]["price_start"] = first
                        window = slice(first, first + 7)
                        if stimulus != "native":
                            e["solar_generation"] = 0.0
                            p.loc[:, :] = 0.25 if stimulus == "price" else 0.16
                            if stimulus == "price":
                                p.iloc[window] = 0.07
                            else:
                                e.iloc[window, e.columns.get_loc("solar_generation")] = 1.25
                        energy.append(e)
                        prices.append(p)
                        cycles.append(
                            dict(
                                cycle_id=f"cycle-{start}",
                                profile_id=profile.profile_id,
                                earliest_start_time_step=start + 2,
                                latest_start_time_step=start + 12,
                                deadline_time_step=start + 18,
                                priority=0.7,
                                must_run=True,
                                day_kind="working_day",
                                year=2023,
                            )
                        )
                        for i, car in enumerate(CARS):
                            cap = schema["electric_vehicles_def"][car]["battery"]["attributes"][
                                "capacity"
                            ]
                            for t in range(SLOTS):
                                connected = 1 <= t < deadlines[i]
                                chargers[i].append(
                                    dict(
                                        electric_vehicle_charger_state=1 if connected else 3,
                                        electric_vehicle_id=car if connected else "",
                                        electric_vehicle_session_id=f"{car}-{start}"
                                        if connected
                                        else "",
                                        electric_vehicle_departure_time=deadlines[i] - t
                                        if connected
                                        else np.nan,
                                        electric_vehicle_required_soc_departure=0.50
                                        + needs[i] / cap
                                        if connected
                                        else np.nan,
                                        electric_vehicle_estimated_arrival_time=0.0
                                        if connected
                                        else np.nan,
                                        electric_vehicle_estimated_soc_arrival=0.50
                                        if connected
                                        else np.nan,
                                        electric_vehicle_current_soc=0.50 if connected else np.nan,
                                    )
                                )
    pd.concat(energy, ignore_index=True).to_parquet(out / "site.parquet", index=False)
    p = pd.concat(prices, ignore_index=True)
    p.to_parquet(out / "tariff.parquet", index=False)
    pd.DataFrame(cycles).to_parquet(out / "cycles.parquet", index=False)
    for port, rows in zip(PORTS, chargers):
        path = out / (port + ".parquet")
        pd.DataFrame(rows).to_parquet(path, index=False)
        b["chargers"][port]["charger_simulation"] = str(path)
    for name in ("weather", "carbon_intensity"):
        shared = pd.read_parquet(root / b[name])
        paired = [shared.iloc[c["day"] * 96 + 40 : c["day"] * 96 + 40 + SLOTS] for c in cases]
        path = out / f"{name}.parquet"
        pd.concat(paired, ignore_index=True).to_parquet(path, index=False)
        b[name] = str(path)
    b["energy_simulation"], b["pricing"] = str(out / "site.parquet"), str(out / "tariff.parquet")
    end = len(cases) * SLOTS - 1
    schema.update(
        simulation_end_time_step=end,
        episode_time_steps=None,
        random_episode_split=False,
        rolling_episode_split=False,
    )
    (out / "schema.json").write_text(json.dumps(schema))
    (out / "cases.json").write_text(json.dumps(cases))
    signals = np.stack(
        [
            p.electricity_pricing,
            pd.concat(energy).non_shiftable_load * 4,
            pd.concat(energy).solar_generation * 4,
        ],
        axis=1,
    )
    with (out / "outlooks.jsonl").open("w") as f:
        for case in cases:
            start = case["start"]
            # Each case is a separate experiment. Do not issue curves from the following case.
            curve = np.pad(signals[start : start + SLOTS], ((0, 96), (0, 0)), mode="edge")
            for t in range(SLOTS):
                now = (start + t) * 900
                f.write(
                    json.dumps(
                        dict(
                            site=SITE,
                            issued_at_seconds=now,
                            received_at_seconds=now,
                            valid_from_seconds=now,
                            interval_seconds=900,
                            **dict(
                                zip(
                                    ("price_eur_per_kwh", "non_shiftable_kw", "solar_kw"),
                                    curve[t : t + 96].T.tolist(),
                                )
                            ),
                        )
                    )
                    + "\n"
                )
    cfg = yaml.safe_load(Path("configs/templates/rl/ti_marl_core30.yaml").read_text())
    cfg["simulator"].update(
        dataset_path=str(out / "schema.json"),
        simulation_start_time_step=0,
        simulation_end_time_step=SLOTS - 1,
        episode_time_steps=SLOTS,
    )
    h = cfg["pipeline"][0]["hyperparameters"]
    h.update(
        typed_interfaces_dir=str(out / "interfaces"),
        simulator_bindings_path=str(out / "interfaces/technology_bindings/simulator.yaml"),
        issued_forecasts_path=str(out / "outlooks.jsonl"),
        reference_sites=1,
        reservation_error_kw=0.0,
        reward_penalties={
            "ev_excess_kwh": 0.05,
            "stationary_terminal_deficit_kwh": float(0.16 / np.sqrt(0.94)),
        },
    )
    h["actor"].update(d_model=128, readout="household" if cycle_mean else "household_start")
    h["departure_margin_seconds"] = margin
    h.setdefault("feasibility", {})["ev_service_tolerance_ratio"] = tolerance_soc
    env = _build_env(cfg)
    payload, _ = env.reset()
    interfaces, coverage = generate(env.entity_specs, payload, schema)
    write_generated(
        out / "interfaces",
        interfaces,
        coverage,
        source=str(source),
        simulator_bindings=generate_simulator_bindings(env.entity_specs, payload, interfaces, {}),
        overwrite=True,
    )
    if checkpoint is not None:
        evaluation_job(cfg, checkpoint, cases[case_index]["start"])
    (out / "job.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Run a frozen V2 household checkpoint")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument(
        "--service-tolerance",
        type=float,
        default=0.05,
        help="Reservation tolerance as a fraction of EV capacity",
    )
    parser.add_argument(
        "--cycle-mean",
        action="store_true",
        help="Use the earlier cycle-average-only appliance encoder",
    )
    parser.add_argument(
        "--departure-margin",
        type=int,
        default=0,
        help="Service reserve before departure in seconds",
    )
    args = parser.parse_args()
    prepare(
        args.output,
        args.schema or schema_path(),
        args.checkpoint,
        args.case,
        args.service_tolerance,
        args.cycle_mean,
        args.departure_margin,
    )
