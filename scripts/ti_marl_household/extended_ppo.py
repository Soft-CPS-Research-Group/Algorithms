"""Continue the two V2 seeds and check dishwasher decisions on disjoint 2023 dates."""

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import date, timedelta
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.runtime.randomness import derived_seed
from scripts.ti_marl_household import community
from scripts.ti_marl_household import dishwasher_critic as probe
from scripts.ti_marl_household.price_windows import RUNS

DAYS = (
    community.DAYS[0],
    ("validation", [252, 281, 317]),
    ("test", [246, 273, 294, 315, 339, 356]),
)
SEEDS = (43, 71)
INITIAL_EPISODES = 48
EXTRA_EPISODES = 192
CADENCE = 48
CHECKPOINTS = (48, 96, 144, 192, 240)
PROBE_SLOT = 8
EPS = 1e-5


def prepare_jobs(directory):
    community.make_fixture(directory, community.schema_path(), case_days=DAYS)
    cases = community.read_json(directory / "cases.json")
    train = [c for c in cases if c["split"] == "train"]
    rng = np.random.default_rng(89274)
    ids = []
    while len(ids) < EXTRA_EPISODES:
        ids.extend(rng.permutation(len(train)).tolist())
    windows = [
        [train[i]["start"], train[i]["start"] + community.DECISIONS] for i in ids[:EXTRA_EPISODES]
    ]
    for seed in SEEDS:
        cfg = yaml.safe_load((directory / "job.yaml").read_text())
        cfg["training"].update(episodes=EXTRA_EPISODES, seed=seed)
        cfg["simulator"].update(
            simulation_start_time_step=0,
            simulation_end_time_step=train[-1]["start"] + community.DECISIONS,
            repeat_episode_scenario=False,
            episode_time_steps=windows,
        )
        h = cfg["pipeline"][0]["hyperparameters"]
        h.update(
            learning_rate=3e-5,
            rollout_steps=community.DECISIONS,
            ppo_epochs=2,
            trace={"enabled": False},
        )
        cfg["checkpointing"] = dict(
            resume_training=True,
            restore_optimizers=True,
            checkpoint_local_path=str((RUNS / f"joint-{seed}/latest_checkpoint.pth").resolve()),
            checkpoint_interval=CADENCE * community.DECISIONS,
            require_initial_exploration_done=False,
        )
        (directory / f"continued-{seed}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def checkpoint(directory, seed, episodes):
    if episodes == INITIAL_EPISODES:
        return RUNS / f"joint-{seed}/latest_checkpoint.pth"
    root = directory / f"public/jobs/continued-{seed}/checkpoints"
    offset = INITIAL_EPISODES
    tail = directory / f"public/jobs/continued-{seed}-tail"
    if episodes > 144 and tail.is_dir():
        root = tail / "checkpoints"
        offset = 144
    index = root / "index.json"
    if not index.exists():
        return None
    step = (episodes - offset) * community.DECISIONS
    for entry in reversed(community.read_json(index)["entries"]):
        if entry["step"] == step:
            return root / entry["directory"] / entry["artifact"]
    return None


@torch.no_grad()
def evaluate_case(case):
    agent = community.policy
    cfg = community.config
    cfg["simulator"].update(
        simulation_start_time_step=case["start"],
        simulation_end_time_step=case["start"] + community.DECISIONS,
    )
    env = community._build_env(cfg)

    def roll(branch, anchor=None):
        return probe.simulate(env, agent, case, arm="continued", branch=branch, anchor=anchor)

    natural = roll("unforced")

    def compare(a, b, anchor):
        if a["actions"][:anchor] != b["actions"][:anchor]:
            raise ValueError("The two dishwasher choices do not share the same physical history")
        np.testing.assert_allclose(a["rewards"][:anchor], b["rewards"][:anchor], atol=1e-7)
        difference = np.subtract(b["rewards"][anchor:], a["rewards"][anchor:])
        values = a["at_start"]
        return dict(
            predicted_wait_gain=values["wait"] - values["start"],
            measured_wait_gain=float(
                np.dot(difference, agent.learner.gamma ** np.arange(len(difference)))
            ),
            actor_wait=values["p_start"] < 0.5,
            bill_saved_eur=a["bill"] - b["bill"],
            wait_next_start=b["start"],
        )

    start = roll("probe_start", PROBE_SLOT)
    fixed, unavailable = None, None
    if "unavailable" in start:
        unavailable = start
    else:
        wait = roll("probe_wait", PROBE_SLOT)
        if "unavailable" in wait:
            unavailable = wait
        else:
            assert start["start"] == PROBE_SLOT
            assert wait["start"] is None or wait["start"] > PROBE_SLOT
            fixed = compare(start, wait, PROBE_SLOT)
    own = None
    if not case["early"] and natural["at_start"] is not None:
        other = roll("wait_once", natural["start"])
        own = compare(natural, other, natural["start"])
    return dict(
        day=case["day"],
        start=case["start"],
        early=case["early"],
        actual_start=natural["start"],
        supervised_slots=natural["supervised_slots"],
        cheap_kwh=natural["cheap_kwh"],
        requested_kwh=case["programme_request_kwh"],
        complete=natural["completed"],
        ev_errors_kwh=natural["ev_errors"],
        bill_eur=natural["bill"],
        fixed=fixed,
        unavailable=unavailable,
        at_start=own,
    )


def evaluate(directory, seed, episodes, split, weights, workers):
    community.init_eval(directory, weights)
    cases = [
        c
        for c in community.read_json(directory / "cases.json")
        if c["split"] == split and c["stimulus"] == "price"
    ]
    with get_context("fork").Pool(workers) as pool:
        rows = pool.map(evaluate_case, cases, chunksize=1)
    assert all(isinstance(r, dict) and r["start"] == c["start"] for r, c in zip(rows, cases, strict=True))
    out = directory / f"{split}-{seed}-{episodes}.json"
    community.write_json(out, rows)
    if split == "test":
        other = [
            c
            for c in community.read_json(directory / "cases.json")
            if c["split"] == "test" and c["stimulus"] != "price"
        ]
        with get_context("fork").Pool(workers) as pool:
            service = pool.map(community.evaluate_case, other, chunksize=1)
        community.write_json(directory / f"service-{seed}-{episodes}.json", service)
    print(f"{out.name}: evaluated {len(rows)} windows", flush=True)


def monitor(directory, seed, workers):
    for episodes in CHECKPOINTS:
        path = checkpoint(directory, seed, episodes)
        while path is None:
            result_path = directory / f"public/jobs/continued-{seed}/results/result.json"
            tail = directory / f"public/jobs/continued-{seed}-tail"
            if episodes > 144 and tail.is_dir():
                result_path = tail / "results/result.json"
            if result_path.exists() and community.read_json(result_path).get("status") in {
                "failed",
                "cancelled",
                "completed",
            }:
                path = checkpoint(directory, seed, episodes)
                if path is None:
                    raise RuntimeError(f"Missing snapshot after {episodes} episodes")
                break
            time.sleep(12)
            path = checkpoint(directory, seed, episodes)
        if not (directory / f"validation-{seed}-{episodes}.json").exists():
            evaluate(directory, seed, episodes, "validation", path, workers)


def service_metrics(rows):
    errors = np.asarray([r["ev_errors_kwh"] for r in rows])
    return dict(
        windows=len(rows),
        departures=errors.size,
        cycles_completed=sum(r["complete"] for r in rows),
        shortfalls_over_01=int((errors < -0.1).sum()),
        within_target_01=int((abs(errors) <= 0.1).sum()),
        excesses_over_01=int((errors > 0.1).sum()),
        mean_excess_kwh=float(np.maximum(errors, 0).mean()),
        largest_shortfall_kwh=max(0.0, float(-errors.min())),
        mean_bill_eur=float(np.mean([r["bill_eur"] for r in rows])),
    )


def choice_metrics(choices):
    if not choices:
        return {"cases": 0}
    predicted = np.asarray([v["predicted_wait_gain"] for v in choices])
    measured = np.asarray([v["measured_wait_gain"] for v in choices])
    useful = abs(measured) > EPS
    wait = measured > EPS
    start = measured < -EPS
    actor = np.asarray([v["actor_wait"] for v in choices])
    critic_wait = predicted > 0
    return dict(
        cases=len(choices),
        wait_better=int(wait.sum()),
        start_better=int(start.sum()),
        ties=int((~useful).sum()),
        critic_prefers_wait=int(critic_wait.sum()),
        actor_prefers_wait=int(actor.sum()),
        correct_critic=int(((predicted * measured > 0) & useful).sum()),
        correct_actor=int(((actor == (measured > 0)) & useful).sum()),
        correct_critic_when_wait_better=int((critic_wait & wait).sum()),
        correct_critic_when_start_better=int(((predicted < 0) & start).sum()),
        mean_predicted_wait_gain=float(predicted.mean()),
        mean_observed_wait_gain=float(measured.mean()),
        mean_bill_saved_by_waiting_eur=float(np.mean([v["bill_saved_eur"] for v in choices])),
    )


def summarize(rows):
    result = {"service": service_metrics(rows)}
    for early, name in ((True, "early"), (False, "late")):
        cells = [r for r in rows if r["early"] == early]
        first = 2 if early else 10
        starts = [r["actual_start"] for r in cells if r["actual_start"] is not None]
        result[name] = dict(
            n=len(cells),
            fixed=choice_metrics([r["fixed"] for r in cells if r["fixed"] is not None]),
            unavailable_fixed=[
                dict(day=r["day"], start=r["start"], **r["unavailable"])
                for r in cells if r.get("unavailable")
            ],
            supervised_dishwasher_site_steps=[
                dict(day=r["day"], start=r["start"], slots=r["supervised_slots"])
                for r in cells if r.get("supervised_slots")
            ],
            at_own_start=choice_metrics([r["at_start"] for r in cells if r["at_start"]]),
            starts_in_cheap=sum(first <= t < first + 7 for t in starts),
            complete_cycle_in_cheap=sum(
                r["complete"] and abs(r["cheap_kwh"] - r["requested_kwh"]) < 1e-5 for r in cells
            ),
            start_minutes_from_window_open=dict(sorted(Counter(15 * t for t in starts).items())),
            mean_start_slot=float(np.mean(starts)) if starts else None,
            share_dishwasher_energy_cheap=sum(r["cheap_kwh"] for r in cells)
            / sum(r["requested_kwh"] for r in cells),
            mean_bill_eur=float(np.mean([r["bill_eur"] for r in cells])),
        )
    return result


def audit(directory, seed):
    root = directory / f"public/jobs/continued-{seed}"
    cfg = yaml.safe_load((directory / f"continued-{seed}.yaml").read_text())
    tail = root.with_name(root.name + "-tail")
    jobs = [root, tail] if tail.exists() else [root]
    assert all(
        community.read_json(j / "config.initial.json")["training"]["seed"] == seed for j in jobs
    )
    agent_seed = derived_seed(seed, "stage", 0, "member", 0)

    def sampled_updates(job):
        records = (job / "logs/metrics.jsonl").read_text().splitlines()
        metrics = (json.loads(line)["metrics"] for line in records)
        return [
            m["TI_MARL/train_sampled_choices"]
            for m in metrics
            if "TI_MARL/train_sampled_choices" in m
        ]

    training = community.read_json((tail if tail.exists() else root) / "results/result.json")
    assert training["status"] == "completed"
    records = training["episode_execution"]
    if tail.exists():
        runtime = checkpoint(directory, seed, 144).parent / "runtime_state.pt"
        prefix = torch.load(runtime, map_location="cpu", weights_only=False)["episode_history"]
        assert len(prefix) == 96
        records = prefix + records
        updates = sampled_updates(root)[: len(prefix)] + sampled_updates(tail)
    else:
        updates = sampled_updates(root)
    actual = [[r["start_time_step"], r["end_time_step"]] for r in records if r["training"]]
    assert actual == cfg["simulator"]["episode_time_steps"] and len(actual) == EXTRA_EPISODES
    train = {
        c["start"] for c in community.read_json(directory / "cases.json") if c["split"] == "train"
    }
    assert all(a in train and b - a == community.DECISIONS for a, b in actual)
    assert len(updates) == EXTRA_EPISODES and min(updates) > 0
    history = {}
    base = None
    for episodes in CHECKPOINTS:
        path = checkpoint(directory, seed, episodes)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if base is None:
            base = state
        assert base["learning_architecture"] == state["learning_architecture"]
        paths = {"typed_interfaces_dir", "simulator_bindings_path", "issued_forecasts_path"}

        def hyperparameters(checkpoint_state):
            hyper = checkpoint_state["configuration"]["algorithm"]["hyperparameters"]
            return {key: value for key, value in hyper.items() if key not in paths}

        assert hyperparameters(base) == hyperparameters(state)
        assert state["configuration"]["training"]["seed"] == agent_seed
        for opt in ("actor_optimizer", "critic_optimizer"):
            step = max(float(x["step"]) for x in state["learner"][opt]["state"].values())
            if not np.isclose(step, 2 * episodes):
                raise ValueError(
                    f"{opt}: expected {2 * episodes} Adam steps at episode {episodes}, got {step}"
                )
        history[str(episodes)] = dict(
            adam_steps=int(step), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return dict(
        verified_episodes=len(actual),
        updated_rollouts=len(updates),
        simulated_decisions=len(actual) * community.DECISIONS,
        sampled_choices=sum(updates),
        checkpoints=history,
    )


def validation_candidate(directory, seed):
    """Optional extra test point if learning to wait generalises across both tariff conditions."""
    baseline = summarize(community.read_json(directory / f"validation-{seed}-48.json"))
    candidates = []
    for episodes in CHECKPOINTS[1:]:
        score = summarize(community.read_json(directory / f"validation-{seed}-{episodes}.json"))
        service = score["service"]
        if service["cycles_completed"] != service["windows"] or service["shortfalls_over_01"]:
            continue
        if (
            score["late"]["share_dishwasher_energy_cheap"]
            <= baseline["late"]["share_dishwasher_energy_cheap"]
        ):
            continue
        if (
            score["early"]["share_dishwasher_energy_cheap"]
            < baseline["early"]["share_dishwasher_energy_cheap"] - 1e-4
        ):
            continue
        early, late = score["early"]["fixed"], score["late"]["fixed"]
        if min(early.get("start_better", 0), late.get("wait_better", 0)) == 0:
            continue
        accuracy = min(
            early["correct_critic_when_start_better"] / early["start_better"],
            late["correct_critic_when_wait_better"] / late["wait_better"],
        )
        if accuracy > 0.5:
            candidates.append((accuracy, score["late"]["share_dishwasher_energy_cheap"], episodes))
    return max(candidates)[2] if candidates else None


def final_tests(directory, seed, workers):
    candidate = validation_candidate(directory, seed)
    for episodes in sorted({INITIAL_EPISODES, CHECKPOINTS[-1], candidate} - {None}):
        if all(
            (directory / f"{kind}-{seed}-{episodes}.json").exists() for kind in ("test", "service")
        ):
            continue
        weights = checkpoint(directory, seed, episodes)
        if not weights or not weights.is_file():
            raise ValueError(f"Missing checkpoint for {episodes} episodes")
        evaluate(directory, seed, episodes, "test", weights, workers)


def report(directory, destination):
    split_days = {
        name: [(date(2023, 1, 1) + timedelta(days=i)).isoformat() for i in days]
        for name, days in DAYS
    }
    assert sum(map(len, split_days.values())) == len(set().union(*split_days.values()))
    prior_test = set(dict(community.DAYS)["test"])
    assert all(prior_test.isdisjoint(days) for part, days in DAYS if part != "train")
    result = dict(
        dataset="Core-30 Normal 2023, controlled three-house windows",
        simulator=version("softcpsrecsimulator"),
        days=split_days,
        seeds=list(SEEDS),
        initial_episodes=INITIAL_EPISODES,
        extra_episodes=EXTRA_EPISODES,
        probe="Hold the dishwasher off to quarter 8. Where both choices are legal, start vs wait one quarter, then resume the same policy. Actual gain is discounted full operating reward from the fork, using identical earlier actions. Positive means waiting was better. Physically unavailable comparisons are recorded and excluded from critic accuracy. Absolute ties <= 0.00001 are excluded from accuracy. Actor timing and service use every unforced episode.",
        tariff="EUR 0.07/kWh in quarters 2–8 or 10–16 inclusive, EUR 0.25 otherwise. Cycle lasts six quarters. Other service scenarios use solar and native tariffs.",
        training="Shared V2 width 128, native MAPPO/PPO, continuing both Adam optimizers and checkpoint RNG. Same 80-window training population as the first 48 episodes, reshuffled without replacement per pass. No teacher or forced action in training.",
        audit={},
        validation={},
        final_test={},
        optional_candidate={},
        optional_candidate_rule="Selected on validation only. All cycles complete, no EV deficit over 0.1 kWh, no reduction of early cheap-cycle energy share and an increase in late share against episode 48. The critic must correctly rank a majority when starting wins in early windows and when waiting wins in late windows. Rank by the lower of these two accuracies, then late cheap share.",
    )
    for evidence in ("rerun-equality", "retry-evidence"):
        file = directory / f"{evidence}.json"
        if file.exists():
            result[evidence] = community.read_json(file)
    expected = {
        part: [
            r["start"]
            for r in community.read_json(directory / "cases.json")
            if r["split"] == part and r["stimulus"] == "price"
        ]
        for part in ("validation", "test")
    }
    for seed in SEEDS:
        result["audit"][str(seed)] = audit(directory, seed)
        candidate = validation_candidate(directory, seed)
        result["optional_candidate"][str(seed)] = candidate
        test_points = sorted({48, 240, candidate} - {None})
        for part, key, points in (
            ("validation", "validation", CHECKPOINTS),
            ("test", "final_test", test_points),
        ):
            scores = {}
            for ep in points:
                rows = community.read_json(directory / f"{part}-{seed}-{ep}.json")
                assert [r["start"] for r in rows] == expected[part]
                scores[str(ep)] = summarize(rows)
                if part == "test":
                    others = community.read_json(directory / f"service-{seed}-{ep}.json")
                    extra_ids = [
                        c["start"]
                        for c in community.read_json(directory / "cases.json")
                        if c["split"] == "test" and c["stimulus"] != "price"
                    ]
                    assert [r["start"] for r in others] == extra_ids
                    scores[str(ep)]["all_service"] = service_metrics(
                        rows
                        + [
                            dict(
                                ev_errors_kwh=r["ev_errors_kwh"],
                                complete=r["programme_complete"],
                                bill_eur=r["bill_eur"],
                            )
                            for r in others
                        ]
                    )
            result[key][str(seed)] = scores
    tests = {
        f"{seed}-{ep}": community.read_json(directory / f"test-{seed}-{ep}.json")
        for seed in SEEDS for ep in (INITIAL_EPISODES, CHECKPOINTS[-1])
    }
    shared = set.intersection(*({r["start"] for r in rows if r["fixed"]} for rows in tests.values()))
    result["matched_final_critic"] = {
        model: {
            name: choice_metrics(
                [r["fixed"] for r in rows if r["early"] == early and r["start"] in shared]
            )
            for early, name in ((True, "early"), (False, "late"))
        }
        for model, rows in tests.items()
    }
    community.write_json(destination, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "monitor", "test", "final-tests", "report"])
    p.add_argument("directory", type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS, default=43)
    p.add_argument("--episodes", type=int, choices=CHECKPOINTS, default=240)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output", type=Path, default=RUNS / "extended-ppo.json")
    args = p.parse_args()
    if args.action == "prepare":
        prepare_jobs(args.directory)
    elif args.action == "final-tests":
        final_tests(args.directory, args.seed, args.workers)
    elif args.action == "report":
        report(args.directory, args.output)
    elif args.action == "monitor":
        monitor(args.directory, args.seed, args.workers)
    else:
        weights = checkpoint(args.directory, args.seed, args.episodes)
        if not weights or not weights.is_file():
            raise ValueError("Missing requested checkpoint")
        evaluate(args.directory, args.seed, args.episodes, "test", weights, args.workers)
