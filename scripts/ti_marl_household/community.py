"""Paired three-house V2 transfer and joint MAPPO on controlled Core-30 profiles.

Prepare the working directory with `prepare DIR`; run `evaluate DIR` for the frozen
checkpoint first. Use `prepare-single DIR/single` for the equal-episode control.
For each seed, `train DIR` and `train DIR/single` create native job checkpoints under
each directory's public/jobs/joint-SEED/checkpoints/latest_checkpoint.pth.
Evaluate both with `evaluate DIR --checkpoint PATH --output FILE`, naming the files
DIR/test-SEED.json and DIR/single-test-SEED.json respectively. Finally `report DIR`
verifies the executed windows and updates before comparing the held-out results.
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from copy import deepcopy
from datetime import date, timedelta
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from algorithms.ti_marl.agent import TIMARL
from scripts.dump_entity_obs_sample import _build_env
from scripts.generate_typed_interfaces import generate, generate_simulator_bindings, write_generated
from scripts.ti_marl_ev.protocol import schema_path
from scripts.ti_marl_household.fixture import PORTS, SLOTS, prepare
from utils.entity_adapter import EntityContractAdapter

SITES = ("Member_001", "Member_006", "Member_014")
DAYS = (("train", [12, 40, 65, 93, 128, 152, 187, 216]), ("test", [244, 263, 282, 302, 325, 349]))
SOURCE_POLICY = Path("runs/ti-marl-2023-household/v2-large/latest_checkpoint.pth")
DECISIONS = SLOTS - 1


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_fixture(directory, source, case_days=DAYS, price_start=None, session_seed=89273):
    """Two EVs at different addresses, solar at A, a programme at B and a battery at C."""
    prepare(
        directory, source, tolerance_soc=0.001, margin=900, case_days=case_days,
        price_start=price_start,
        session_seed=session_seed,
    )
    base = read_json(source)
    root = (source.parent / base["root_directory"]).resolve()
    cases = read_json(directory / "cases.json")
    schema = read_json(directory / "schema.json")
    template = schema["buildings"][SITES[0]]
    buildings = {site: deepcopy(base["buildings"][site]) for site in SITES}
    prices = pd.read_parquet(template["pricing"]).electricity_pricing.to_numpy()
    for site, building in buildings.items():
        native_path = root / building["energy_simulation"]
        for key in ("chargers", "pv", "electrical_storage", "deferrable_appliances"):
            building.pop(key, None)
        for key in ("pricing", "weather", "carbon_intensity"):
            building[key] = template[key]
        if site == SITES[0]:
            building["pv"] = template["pv"]
            building["chargers"] = {PORTS[0]: template["chargers"][PORTS[0]]}
            building["energy_simulation"] = template["energy_simulation"]
        else:
            demand = pd.read_parquet(native_path)
            starts = [case["day"] * 96 + 40 for case in cases]
            profile = pd.concat(
                [demand.iloc[start : start + SLOTS] for start in starts], ignore_index=True
            )
            profile["solar_generation"] = 0.0
            path = directory / f"{site}.parquet"
            profile.to_parquet(path, index=False)
            building["energy_simulation"] = str(path)
        if site == SITES[1]:
            port = deepcopy(base["buildings"][site]["chargers"][PORTS[1]])
            port["charger_simulation"] = template["chargers"][PORTS[1]]["charger_simulation"]
            building["chargers"] = {PORTS[1]: port}
            building["deferrable_appliances"] = template["deferrable_appliances"]
        if site == SITES[2]:
            building["electrical_storage"] = base["buildings"][site]["electrical_storage"]
        for asset, action in (
            ("electrical_storage", "electrical_storage"),
            ("chargers", "electric_vehicle_storage"),
            ("deferrable_appliances", "deferrable_appliance"),
        ):
            if asset not in building and action not in building["inactive_actions"]:
                building["inactive_actions"].append(action)
    schema["buildings"] = buildings
    write_json(directory / "schema.json", schema)
    with (directory / "outlooks.jsonl").open("w") as file:
        for site, building in buildings.items():
            profile = pd.read_parquet(building["energy_simulation"])
            signals = np.stack(
                (prices, profile.non_shiftable_load * 4, profile.solar_generation * 4), axis=1
            )
            for case in cases:
                start = case["start"]
                curve = np.pad(signals[start : start + SLOTS], ((0, 96), (0, 0)), mode="edge")
                for t in range(SLOTS):
                    now = (start + t) * 900
                    price, load, solar = curve[t : t + 96].T.tolist()
                    row = dict(
                        site=site,
                        issued_at_seconds=now,
                        received_at_seconds=now,
                        valid_from_seconds=now,
                        interval_seconds=900,
                        price_eur_per_kwh=price,
                        non_shiftable_kw=load,
                        solar_kw=solar,
                    )
                    file.write(json.dumps(row, separators=(",", ":")) + "\n")
    cfg = yaml.safe_load((directory / "job.yaml").read_text())
    cfg["pipeline"][0]["hyperparameters"]["reference_sites"] = len(SITES)
    env = _build_env(cfg)
    payload, _ = env.reset()
    interfaces, coverage = generate(env.entity_specs, payload, schema)
    write_generated(
        directory / "interfaces",
        interfaces,
        coverage,
        source=str(source),
        simulator_bindings=generate_simulator_bindings(env.entity_specs, payload, interfaces, {}),
        overwrite=True,
    )
    (directory / "job.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def init_eval(directory, checkpoint):
    global policy, config
    torch.set_num_threads(1)
    config = yaml.safe_load((directory / "job.yaml").read_text())
    policy = TIMARL({**config, "algorithm": config["pipeline"][0]})
    policy.load_checkpoint(str(checkpoint))
    policy.actor.eval()
    policy.critic.eval()


@torch.no_grad()
def evaluate_case(case, *, inspect_episode=None):
    start = case["start"]
    config["simulator"].update(
        simulation_start_time_step=start, simulation_end_time_step=start + DECISIONS
    )
    env = _build_env(config)
    adapter = EntityContractAdapter(env, normalization_enabled=False, clip=False)
    payload, info = env.reset()
    adapter.rebuild_layout(payload)
    policy.attach_environment(
        observation_names=[],
        action_names=env.action_names,
        action_space=env.flat_action_space,
        observation_space=[],
        metadata={
            "interface": "entity",
            "seconds_per_time_step": 900,
            "entity_specs": env.entity_specs,
        },
    )
    policy.on_episode_start(episode=0, training=False)
    assert tuple(b.name for b in env.buildings) == SITES
    a, b, c = env.buildings
    battery = c.electrical_storage
    initial = battery.initial_soc * battery.capacity
    errors = [None, None]
    bill = peak = local = ev_kwh = throughput = 0.0
    for t in range(DECISIONS):
        policy.set_entity_observation_context(
            observation_payload=payload, info={**info, "episode_start_time_step": start}
        )
        commands = policy.predict([], deterministic=True)
        if set(policy._pending["prefix_traces"]) != set(SITES):
            raise AssertionError("Missing a local agent in the V2 decision")
        payload, _, done, truncated, info = env.step(
            adapter.to_entity_actions(commands, env.action_names)
        )
        if (done or truncated) != (t == DECISIONS - 1):
            raise AssertionError("Unexpected simulated horizon")
        for j, member in enumerate((a, b)):
            charger = member.electric_vehicle_chargers[0]
            ev_kwh += float(charger.electricity_consumption[t])
            if t == case["departures"][j] - 1:
                ev = charger.past_connected_evs[t]
                target = charger.charger_simulation.electric_vehicle_required_soc_departure[t]
                errors[j] = float((ev.battery.soc[t] - target) * ev.battery.capacity)
        settlement = env._community_market_settlement_history[t]
        if len(settlement) != len(SITES):
            raise AssertionError("Incomplete settlement")
        bill += sum(float(row["settled_cost_eur"]) for row in settlement)
        local += sum(float(row["local_import_kwh"]) for row in settlement)
        net = sum(float(member.net_electricity_consumption[t]) for member in env.buildings)
        grid_import = sum(float(row["grid_import_kwh"]) for row in settlement)
        if not np.isclose(max(net, 0), grid_import, atol=1e-4):
            raise AssertionError("Grid meter does not reconcile with settlement")
        peak = max(peak, grid_import * 4)
        throughput += abs(float(battery.energy_balance[t]))
    stock = float(battery.soc[DECISIONS - 1]) * battery.capacity - initial
    app = b.deferrable_appliances[0]
    row = dict(
        start=start,
        day=case["day"],
        stimulus=case["stimulus"],
        bill_eur=bill,
        adjusted_eur=bill + max(-stock, 0) * 0.16 / np.sqrt(0.94),
        peak_kw=peak,
        local_kwh=local,
        ev_ac_kwh=ev_kwh,
        ev_errors_kwh=errors,
        programme_complete=app.cycle_state.get(f"cycle-{start}") == "completed",
        battery_throughput_kwh=throughput,
        battery_delta_kwh=stock,
    )
    if inspect_episode is not None:
        row["inspection"] = inspect_episode(env, case)
    return row


def evaluate(directory, checkpoint, destination, workers):
    cases = [case for case in read_json(directory / "cases.json") if case["split"] == "test"]
    # Fail visibly in the parent on an incompatible checkpoint instead of respawning workers.
    init_eval(directory, checkpoint)
    with get_context("fork").Pool(workers) as pool:
        rows = pool.map(evaluate_case, cases, chunksize=1)
    write_json(destination, rows)


def train(directory, checkpoint, seed, episodes):
    cfg = yaml.safe_load((directory / "job.yaml").read_text())
    cases = [c for c in read_json(directory / "cases.json") if c["split"] == "train"]
    schedule = np.random.default_rng(1128).permutation(len(cases))[:episodes]
    if len(schedule) != episodes:
        raise ValueError("Training requests more windows than available")
    cfg["training"].update(episodes=episodes, seed=seed)
    cfg["simulator"].update(
        simulation_start_time_step=0,
        simulation_end_time_step=cases[-1]["start"] + DECISIONS,
        repeat_episode_scenario=False,
        episode_time_steps=[[cases[i]["start"], cases[i]["start"] + DECISIONS] for i in schedule],
    )
    cfg["pipeline"][0]["hyperparameters"].update(
        learning_rate=3e-5, rollout_steps=DECISIONS, ppo_epochs=2, trace={"enabled": False}
    )
    # Native checkpoint loading restores RNG state. Re-seed the starting RNG for each replica
    # without touching the frozen weights, optimizers or test data.
    init_eval(directory, checkpoint)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    initial = Path(policy.save_checkpoint(str(directory / f"initial-{seed}"), step=0))
    cfg["checkpointing"] = dict(
        resume_training=True, checkpoint_local_path=str(initial), restore_optimizers=False
    )
    job = f"joint-{seed}"
    config_path = directory / f"{job}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run(
        [
            str(Path(sys.executable).with_name("algorithms-job")),
            "--config",
            str(config_path),
            "--job_id",
            job,
            "--base-dir",
            str(directory / "public"),
        ],
        check=True,
    )


def summarize(rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    errors = np.array([r["ev_errors_kwh"] for r in rows])
    return dict(
        windows=len(rows),
        bill_eur=mean("bill_eur"),
        adjusted_eur=mean("adjusted_eur"),
        mean_peak_kw=mean("peak_kw"),
        max_peak_kw=max(r["peak_kw"] for r in rows),
        ev_within_01=int((abs(errors) <= 0.1).sum()),
        ev_under_by_more_than_01=int((errors < -0.1).sum()),
        ev_over_by_more_than_01=int((errors > 0.1).sum()),
        max_shortfall_kwh=float(max(0, -errors.min())),
        mean_shortfall_per_ev_kwh=float(np.maximum(-errors, 0).mean()),
        mean_excess_per_ev_kwh=float(np.maximum(errors, 0).mean()),
        programme_completed=sum(r["programme_complete"] for r in rows),
        ev_ac_kwh=mean("ev_ac_kwh"),
        local_kwh=mean("local_kwh"),
        battery_throughput_kwh=mean("battery_throughput_kwh"),
    )


def verify_training(directory, seed, expected, source_policy):
    root = directory / "public/jobs" / f"joint-{seed}"
    result = read_json(root / "results/result.json")
    if result["status"] != "completed":
        raise ValueError("Training job is incomplete")
    observed = [
        (e["start_time_step"], e["end_time_step"])
        for e in result["episode_execution"]
        if e["training"]
    ]
    if observed != expected:
        raise ValueError("Executed training windows differ from the paired design")
    metrics = [json.loads(line)["metrics"] for line in (root / "logs/metrics.jsonl").open()]
    sampled = [
        m["TI_MARL/train_sampled_choices"] for m in metrics if "TI_MARL/train_sampled_choices" in m
    ]
    if len(sampled) != len(expected) or min(sampled) <= 0:
        raise ValueError("Expected a non-empty policy update after every training episode")
    checkpoint = root / "checkpoints/latest_checkpoint.pth"

    def tensors(path):
        return torch.load(path, weights_only=False, map_location="cpu")

    initial = tensors(directory / f"initial-{seed}/latest_checkpoint.pth")
    final = tensors(checkpoint)
    changes = {}
    for name in ("actor", "critic"):
        a, b, c = [state[name] for state in (source_policy, initial, final)]
        if (
            a.keys() != b.keys()
            or a.keys() != c.keys()
            or not all(torch.equal(a[k], b[k]) for k in a)
        ):
            raise ValueError(f"Initial {name} differs from the frozen policy")
        delta = (
            sum(float((b[k] - c[k]).double().square().sum()) for k in a if a[k].is_floating_point())
            ** 0.5
        )
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError(f"Expected a finite update of the {name}")
        changes[name + "_l2_change"] = delta
    return dict(
        episodes=len(observed),
        updates=len(sampled),
        environment_decisions=len(observed) * DECISIONS,
        sampled_choices=int(sum(sampled)),
        **changes,
    )


def comparison(directory, checkpoint, seeds, destination):
    frozen = read_json(directory / "frozen.json")
    cases = read_json(directory / "cases.json")
    held_out = [c["start"] for c in cases if c["split"] == "test"]
    src = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if read_json(directory / "single/cases.json") != cases:
        raise ValueError("Single-house and joint training requests differ")
    arms = {"frozen": frozen}
    training = {}
    reference_schedule = None
    for seed in seeds:
        for name, workdir, evaluation in (
            ("joint", directory, f"test-{seed}.json"),
            ("single", directory / "single", f"single-test-{seed}.json"),
        ):
            arm = f"{name}-{seed}"
            cfg = yaml.safe_load((workdir / f"joint-{seed}.yaml").read_text())
            windows = [tuple(w) for w in cfg["simulator"]["episode_time_steps"]]
            train_starts = {c["start"] for c in cases if c["split"] == "train"}
            if any(start not in train_starts or end != start + DECISIONS for start, end in windows):
                raise ValueError("Training schedule contains a window outside the training period")
            if reference_schedule is not None and windows != reference_schedule:
                raise ValueError("Unequal training budgets or windows")
            reference_schedule = windows
            training[arm] = verify_training(workdir, seed, windows, src)
            arms[arm] = read_json(directory / evaluation)
    dates = sorted({c["day"] for c in frozen})
    report = dict(
        source_checkpoint=str(checkpoint),
        frozen_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        dataset="Core-30 Normal 2023, three-house controlled fixture",
        simulator_version=version("softcpsrecsimulator"),
        torch_version=version("torch"),
        training_days=[
            (date(2023, 1, 1) + timedelta(days=i)).isoformat()
            for i in sorted({c["day"] for c in cases if c["split"] == "train"})
        ],
        training_window_starts=[s for s, _ in reference_schedule],
        method="V2 width 128, shared local actor, centralized prefix critic, 3e-5 learning rate, two PPO epochs per rollout, no teacher.",
        single_control="Same initialized weights, date profiles, requests and episode count, with the four flexible devices in a single home during training. All policies evaluated in the three-home community market.",
        test_design="Six source dates; ten matched independent windows per date (four price shifts, four solar shifts, two with native tariff and PV). The two EV departure orders are counterbalanced. The simulator calendar concatenates windows.",
        sites=list(SITES),
        evaluation_dates=[(date(2023, 1, 1) + timedelta(days=i)).isoformat() for i in dates],
        native_step_seconds=900,
        decisions_per_window=DECISIONS,
        foreknowledge="Controlled within-window outlooks, padded at the boundary",
        service="Shortfall and excess measured at last connected step in battery kWh; tolerance 0.10 kWh.",
        bill="Sum of actual settlement of all three members; peak at their common grid boundary.",
        adjusted_bill="Settlement plus hypothetical refill of lost stationary stock at 0.16/sqrt(0.94) EUR per kWh.",
        training=training,
        overall={},
        by_stimulus={},
        by_date={},
    )
    for arm, rows in arms.items():
        if [r["start"] for r in rows] != held_out:
            raise ValueError(f"Unmatched evaluation windows for {arm}")
        report["overall"][arm] = summarize(rows)
        report["by_stimulus"][arm] = {
            kind: summarize([r for r in rows if r["stimulus"] == kind])
            for kind in ("price", "solar", "native")
        }
        report["by_date"][arm] = {
            (date(2023, 1, 1) + timedelta(days=i)).isoformat(): summarize(
                [r for r in rows if r["day"] == i]
            )
            for i in dates
        }
    destination.parent.mkdir(exist_ok=True, parents=True)
    write_json(destination, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "prepare-single", "evaluate", "train", "report"])
    p.add_argument("directory", type=Path)
    p.add_argument("--schema", type=Path)
    p.add_argument("--checkpoint", type=Path, default=SOURCE_POLICY)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[43, 71])
    args = p.parse_args()
    directory = args.directory.resolve()
    if args.action == "prepare":
        make_fixture(directory, args.schema or schema_path())
    elif args.action == "prepare-single":
        prepare(
            directory, args.schema or schema_path(), tolerance_soc=0.001, margin=900, case_days=DAYS
        )
    elif args.action == "report":
        comparison(
            directory, args.checkpoint, args.seeds, args.output or directory / "comparison.json"
        )
    elif args.action == "train":
        train(directory, args.checkpoint, args.seed, args.episodes)
    else:
        destination = args.output or directory / "frozen.json"
        evaluate(directory, args.checkpoint, destination, args.workers)
        print(json.dumps(summarize(read_json(destination)), indent=2))
