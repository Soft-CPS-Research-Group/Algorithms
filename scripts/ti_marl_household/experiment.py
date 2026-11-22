"""Fit and evaluate a V2 household on executed SoftCPS decisions, without a teacher."""

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.ti_marl.agent import TIMARL
from algorithms.ti_marl.experiments.ev_pretraining import load_ev_policy
from scripts.dump_entity_obs_sample import _build_env
from scripts.ti_marl_household.fixture import SITE, SLOTS
from utils.entity_adapter import EntityContractAdapter


class Episodes:
    def __init__(self, directory, seed, readout=None):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.directory = directory
        self.cases = json.loads((directory / "cases.json").read_text())
        self.cfg = yaml.safe_load((directory / "job.yaml").read_text())
        stage = self.cfg["pipeline"][0]
        actor = stage["hyperparameters"]["actor"]
        actor["readout"] = readout or actor.get("readout", "household")
        self.agent = TIMARL({**self.cfg, "algorithm": stage})
        ev = load_ev_policy(
            Path("runs/ti-marl-2023-continuous-dose/models/validated") / f"v2-128-{seed}.pth"
        )
        self.agent.actor.initialise_from_ev(ev)
        self.agent.actor.eval()
        self.agent.actor.step_seconds = 900
        self.cache = {}

    def environment(self, i):
        if i not in self.cache:
            case = self.cases[i]
            self.cfg["simulator"].update(
                simulation_start_time_step=case["start"],
                simulation_end_time_step=case["start"] + SLOTS - 1,
            )
            env = _build_env(self.cfg)
            self.cache[i] = (
                env,
                EntityContractAdapter(env, normalization_enabled=False, clip=False),
            )
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    @torch.no_grad()
    def episode(self, i, weights):
        agent = self.agent
        parameters = torch.as_tensor(weights).reshape(-1, 8)
        if len(parameters) not in (3, 5):
            raise ValueError("Expected three mode readouts or all five household readouts")
        agent.actor.coordination.zero_()
        agent.actor.coordination[: len(parameters)].copy_(parameters)
        env, adapter = self.environment(i)
        payload, info = env.reset()
        adapter.rebuild_layout(payload)
        agent.attach_environment(
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
        agent.on_episode_start(episode=0, training=False)
        case = self.cases[i]
        b = env.buildings[0]
        flows = []
        ev_errors = [None, None]
        forced = 0
        fallbacks = 0
        cost = 0.0
        app = b.deferrable_appliances[0]
        for t in range(SLOTS - 1):
            agent.set_entity_observation_context(
                observation_payload=payload, info={**info, "episode_start_time_step": case["start"]}
            )
            commands = agent.predict([], deterministic=True)
            trace = agent._pending["prefix_traces"][SITE]
            forced += sum(c.stage == "mode" and len(c.allowed) == 1 for c in trace.choices)
            fallbacks += int(trace.supervised)
            payload, _, done, truncated, info = env.step(
                adapter.to_entity_actions(commands, env.action_names)
            )
            c = b.electric_vehicle_chargers
            for j, dep in enumerate(case["departures"]):
                if t == dep - 1:
                    ev = c[j].past_connected_evs[t]
                    cap = ev.battery.capacity
                    target = c[j].charger_simulation.electric_vehicle_required_soc_departure[t]
                    ev_errors[j] = float((ev.battery.soc[t] - target) * cap)
            cost += sum(
                float(r["settled_cost_eur"]) for r in env._community_market_settlement_history[t]
            )
            flows.append(
                [
                    float(b.net_electricity_consumption[t]),
                    *[float(port.electricity_consumption[t]) for port in c],
                    float(b.electrical_storage.energy_balance[t]),
                    float(b.deferrable_appliances_electricity_consumption[t]),
                    float(-b.solar_generation[t]),
                ]
            )
            if done or truncated:
                if t != SLOTS - 2:
                    raise AssertionError("Unexpected episode truncation")
        a = np.array(flows)
        battery_delta = (
            float(b.electrical_storage.soc[SLOTS - 2]) - 0.5
        ) * b.electrical_storage.capacity
        # Refill missing initial stock at a common tariff; unsold surplus earns no credit.
        inventory_cost = max(0.0, -battery_delta) * 0.16 / np.sqrt(0.94)
        # Report actual settlement and this hypothetical refill separately.
        err = np.array(ev_errors)
        programme_kwh = float(a[:, 4].sum())
        starts = np.flatnonzero(a[:, 4] > 1e-5)
        served = app.cycle_state.get(f"cycle-{case['start']}") == "completed"
        deficit = np.maximum(-err, 0.0)
        tol = np.array(
            [
                port.past_connected_evs[dep - 1].battery.capacity * 0.05
                for port, dep in zip(b.electric_vehicle_chargers, case["departures"])
            ]
        )
        reward = -(
            cost
            + inventory_cost
            + 0.01 * abs(a[:, 3]).sum()
            + 0.05 * np.maximum(err, 0).sum()
            + np.minimum(deficit, tol).sum()
            + 120 * np.maximum(deficit - tol, 0.0).sum()
            + 120 * max(0.0, case["programme_request_kwh"] - programme_kwh)
        )
        window = slice(2, 9) if case["early"] else slice(10, 17)
        return dict(
            i=i,
            **case,
            ev_error=err.tolist(),
            peak=float(a[:, 0].max() * 4),
            bill=cost,
            stock_adjusted_bill=cost + inventory_cost,
            stock_delta=battery_delta,
            programme_complete=served,
            programme_kwh=programme_kwh,
            programme_start=int(starts[0]) if len(starts) else None,
            battery_active_steps=int((abs(a[:, 3]) > 1e-5).sum()),
            battery_throughput=float(abs(a[:, 3]).sum()),
            grid_export=float(np.maximum(-a[:, 0], 0.0).sum()),
            window_ev=float(a[window, 1:3].sum()),
            early_ev=a[2:9, 1:3].sum(0).tolist(),
            ev_ac=a[:, 1:3].sum(0).tolist(),
            forced=forced,
            fallback=fallbacks,
            max_violation=max(0.0, float(a[:, 0].max() * 4 - 6.9)),
            reward=float(reward),
            flows=a.round(6).tolist(),
        )


def worker_init(directory, seed, checkpoint=None, margin_seconds=None):
    global episodes
    readout = None
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        readout = saved["learning_architecture"].get("actor_readout", "standard")
        del saved
    episodes = Episodes(directory, seed, readout)
    if checkpoint is not None:
        episodes.agent.load_checkpoint(str(checkpoint))
    if margin_seconds is not None:
        episodes.agent.actor.departure_margin_seconds = margin_seconds


def evaluate(item):
    i, w = item
    return episodes.episode(i, w)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--population", type=int, default=10)
    p.add_argument("--smoke", type=int)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument(
        "--refine", action="store_true", help="Also fit storage activity and continuous power"
    )
    p.add_argument("--fit-only", action="store_true")
    args = p.parse_args()
    directory = args.directory
    if args.smoke is not None:
        x = Episodes(directory, args.seed).episode(args.smoke, np.zeros((3, 8)))
        print(json.dumps(x))
        return
    cases = json.loads((directory / "cases.json").read_text())
    output = directory / f"learn-{args.seed}.json"
    first_fit = output
    if args.refine:
        output = directory / f"refined-{args.seed}.json"
    rng = np.random.default_rng(args.seed + (991 if args.refine else 0))
    train = [i for i, c in enumerate(cases) if c["split"] == "train"]
    dev = [i for i, c in enumerate(cases) if c["split"] == "dev"]
    with get_context("fork").Pool(args.processes, worker_init, (directory, args.seed)) as pool:

        def batch(weights, ids):
            rows = pool.map(evaluate, [(i, w) for w in weights for i in ids], chunksize=1)
            return np.array([r["reward"] for r in rows]).reshape(len(weights), len(ids)).mean(1)

        if not args.evaluate:
            mean = np.zeros((3, 8))
            if args.refine:
                first = json.loads(first_fit.read_text())["weights"]
                mean = np.vstack((first, np.zeros((2, 8))))
            scale = np.full_like(mean, 2.0)
            if args.refine:
                scale[:3] = 0.45
            best = mean.copy()
            best_score = float(batch([best], dev[::3])[0])
            history = []
            for k in range(args.iterations):
                candidates = rng.normal(mean, scale, size=(args.population, *mean.shape))
                candidates[0] = mean
                ids = [
                    int(rng.choice([i for i in train if cases[i]["stimulus"] == kind]))
                    for kind in ("price", "solar", "native")
                ]
                if args.refine:
                    # Paired early/late controlled opportunities on the same training day.
                    ids += [
                        next(
                            i
                            for i in train
                            if cases[i]["day"] == cases[j]["day"]
                            and cases[i]["stimulus"] == cases[j]["stimulus"]
                            and cases[i]["first"] == cases[j]["first"]
                            and cases[i]["early"] != cases[j]["early"]
                        )
                        for j in ids[:2]
                    ]
                scores = batch(candidates, ids)
                elite = candidates[np.argsort(scores)[-max(2, len(candidates) // 4) :]]
                mean = 0.25 * mean + 0.75 * elite.mean(0)
                scale = np.maximum(0.3, 0.5 * scale + 0.5 * elite.std(0))
                choices = [mean, candidates[scores.argmax()]]
                validation = batch(choices, dev[::3])
                if max(validation) > best_score:
                    best_score = float(max(validation))
                    best = choices[np.argmax(validation)].copy()
                history.append(dict(iteration=k, dev=best_score, train=float(max(scores))))
                tmp = output.with_suffix(".tmp")
                tmp.write_text(json.dumps(dict(weights=best.tolist(), history=history)))
                os.replace(tmp, output)
                print(history[-1], flush=True)
        if args.fit_only:
            return
        weights = json.loads(output.read_text())["weights"]
        ids = [i for i, c in enumerate(cases) if c["split"] == "test"]
        arms = {}
        for name, w in (("transferred", np.zeros((3, 8))), ("trained", weights)):
            arms[name] = pool.map(evaluate, [(i, w) for i in ids], chunksize=1)
        (
            directory
            / (f"test-refined-{args.seed}.json" if args.refine else f"test-{args.seed}.json")
        ).write_text(json.dumps(arms))


if __name__ == "__main__":
    main()
