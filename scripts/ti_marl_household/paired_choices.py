"""Fit dishwasher action values from paired simulator returns in the three-house V2 study."""

import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import date, timedelta
from functools import partial
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.ti_marl_household import community, extended_ppo
from scripts.ti_marl_household import dishwasher_critic as probe
from scripts.ti_marl_household.fixture import evaluation_job
from scripts.ti_marl_household.price_windows import RUNS

DAYS = (
    community.DAYS[0],
    ("validation", [232, 266, 301]),
    ("test", [251, 274, 298, 316, 342, 360]),
)
SLOTS = (2, 4, 6, 8, 10, 12)
STRENGTHS = (0.25, 0.6, 1.0)


def base(seed):
    return RUNS / f"joint-{seed}/latest_checkpoint.pth"


def inputs(agent, snapshot, k, traces):
    """Capture the existing critic's input to its final linear head for each legal action."""
    vectors = {}
    for mode, trace in traces.items():
        last_inputs = []

        def record(_layer, args):
            last_inputs.append(args[0].detach())

        with agent.critic.value_head[-1].register_forward_pre_hook(record):
            values = agent.critic.prefix_values(snapshot, trace)[probe.SITE]
        index = tuple(snapshot.agent_ids).index(probe.SITE)
        hidden = last_inputs[index][k + 1]
        torch.testing.assert_close(agent.critic.value_head[-1](hidden).squeeze(), values[k + 1])
        vectors[mode] = hidden.cpu().numpy().copy()
    scores = probe.score(agent, snapshot, k, traces)
    choice = traces["START"][probe.SITE].choices[k]
    features = agent.actor.household_inputs[choice.group_id][1]
    return dict(
        hidden=(vectors["IDLE"] - vectors["START"]).tolist(),
        mean_hidden=((vectors["IDLE"] + vectors["START"]) / 2).tolist(),
        features=list(map(float, features)),
        odds=scores["log_odds"],
        old_gain=scores["wait"] - scores["start"],
    )


@torch.no_grad()
def collect(case):
    cfg, agent = community.config, community.policy
    cfg["simulator"].update(
        simulation_start_time_step=case["start"],
        simulation_end_time_step=case["start"] + community.DECISIONS,
    )
    env = community._build_env(cfg)
    traces = {}
    for t in SLOTS:
        r = probe.simulate(
            env, agent, case, arm="paired", branch="explore_start", anchor=t, observe_choice=inputs
        )
        if "unavailable" not in r:
            assert r["start"] == t and r["completed"]
            traces[t] = r
    samples = []
    for t, start in traces.items():
        if start["at_start"] is None:
            continue
        later = [r for at, r in traces.items() if at > t]
        if not later:
            continue
        discounts = agent.learner.gamma ** np.arange(community.DECISIONS - t)
        diffs = []
        for wait in later:
            assert wait["actions"][:t] == start["actions"][:t]
            np.testing.assert_allclose(wait["rewards"][:t], start["rewards"][:t], atol=1e-7)
            diffs.append(
                float(np.dot(discounts, np.subtract(wait["rewards"][t:], start["rewards"][t:])))
            )
        samples.append(
            dict(
                **start["at_start"],
                day=case["day"],
                slot=t,
                start=case["start"],
                gain=float(np.mean(diffs)),
                comparisons=len(diffs),
            )
        )
    return dict(rollouts=len(SLOTS), legal=len(traces), samples=samples)


def parallel(fn, cases, workers):
    with get_context("fork").Pool(workers) as pool:
        values = pool.map(fn, cases, chunksize=1)
    if any(v is None for v in values):
        raise ValueError("Incomplete simulator trajectory")
    return values


def explore(directory, seed, workers):
    community.init_eval(directory, base(seed))
    # Both tariff placements and EV port orders. Include native and solar regimes in
    # exploration so an update to the dishwasher is not fitted solely to tariff episodes.
    cases = [
        c
        for c in community.read_json(directory / "cases.json")
        if c["split"] == "train" and (c["stimulus"] == "price" or c["day"] in DAYS[0][1][::3])
    ]
    rows = parallel(collect, cases, workers)
    community.write_json(directory / f"explore-{seed}.json", rows)


def ridge(data, alpha):
    x = np.asarray([s["hidden"] for s in data])
    z = np.asarray([s["mean_hidden"] for s in data])
    # Penalise drift in the common prefix value as well as movement of its weights.
    residual = np.array([s["gain"] - s["old_gain"] for s in data])
    return np.linalg.solve(x.T @ x + 0.01 * z.T @ z + alpha * np.eye(x.shape[1]), x.T @ residual)


def estimate(data, delta):
    return np.array([s["old_gain"] + np.dot(s["hidden"], delta) for s in data])


def improve_actor(data, gains):
    # Exact expected-return gradient for the two legal actions. All targets come
    # from the fitted critic. The old log odds include the complete frozen actor.
    x = torch.tensor([s["features"] for s in data], dtype=torch.float64)
    old = torch.tensor([s["odds"] for s in data], dtype=torch.float64)
    advantage = torch.from_numpy(gains).double()
    advantage = advantage / advantage.abs().mean().clamp_min(1e-8)
    delta = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=0.04)
    for _ in range(300):
        logit = old + x @ delta
        p = logit.sigmoid()
        kl = (
            p * (logit - old)
            + torch.nn.functional.softplus(old)
            - torch.nn.functional.softplus(logit)
        )
        loss = (p * advantage + 0.06 * kl).mean() + 0.0003 * (delta**2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return delta.detach().float()


def fit(directory, seed):
    rows = community.read_json(directory / f"explore-{seed}.json")
    samples = [s for r in rows for s in r["samples"]]
    by_day = sorted({s["day"] for s in samples})
    scores = {}
    for alpha in (1e-5, 0.0001, 0.001, 0.01, 0.1):
        err = []
        for group in (by_day[::2], by_day[1::2]):
            train = [s for s in samples if s["day"] not in group]
            hold = [s for s in samples if s["day"] in group]
            d = ridge(train, alpha)
            err.extend((estimate(hold, d) - np.asarray([s["gain"] for s in hold])) ** 2)
        scores[alpha] = float(np.mean(err))
    alpha = min(scores, key=scores.get)
    delta = ridge(samples, alpha)
    gains = estimate(samples, delta)
    movement = improve_actor(samples, gains)
    source = torch.load(base(seed), map_location="cpu", weights_only=False)
    community.init_eval(directory, base(seed))
    hyper = source["configuration"]["algorithm"]["hyperparameters"]
    here = {
        **community.config["pipeline"][0]["hyperparameters"],
        "reward_penalties": community.policy.reward_penalties.model_dump(),
    }
    for key in (
        "operating_reward",
        "reference_sites",
        "reward_penalties",
        "gamma",
        "discount_timebase_seconds",
        "departure_margin_seconds",
    ):
        assert hyper[key] == here[key]
    critic_key = "value_head.2.weight"
    source["critic"][critic_key] += torch.from_numpy(delta).to(source["critic"][critic_key])
    # These two tensors are updated outside PPO. Discard stale Adam moments for
    # them so the resulting checkpoint can also initialise a future PPO run.
    for name, key in (("actor", "coordination"), ("critic", critic_key)):
        opt = source["learner"][name + "_optimizer"]
        names = dict(getattr(community.policy, name).named_parameters())
        ids = opt["param_groups"][0]["params"]
        assert len(ids) == len(names)
        opt["state"].pop(ids[list(names).index(key)], None)
    initial = source["actor"]["coordination"].clone()
    for i, strength in enumerate(STRENGTHS, 1):
        source["actor"]["coordination"] = initial.clone()
        source["actor"]["coordination"][2] += strength * movement
        path = directory / f"candidate-{seed}-{i}.pth"
        torch.save(source, path)
    info = dict(
        source_sha256=hashlib.sha256(base(seed).read_bytes()).hexdigest(),
        reward=here["operating_reward"],
        penalties=here["reward_penalties"],
        attempted_rollouts=sum(r["rollouts"] for r in rows),
        legal_rollouts=sum(r["legal"] for r in rows),
        states=len(samples),
        branch_comparisons=sum(s["comparisons"] for s in samples),
        ridge=alpha,
        held_training_day_mse=scores,
        actor_delta=movement.tolist(),
    )
    community.write_json(directory / f"fit-{seed}.json", info)


def candidate(directory, seed, i):
    return base(seed) if i == 0 else directory / f"candidate-{seed}-{i}.pth"


def meter(env, case):
    energy = env.buildings[1].deferrable_appliances_electricity_consumption[: community.DECISIONS]
    active = np.flatnonzero(energy > 1e-6)
    start = 2 if case["early"] else 10
    return dict(
        start=int(active[0]) if active.size else None,
        cheap=float(np.sum(energy[start : start + 7])),
        requested=case["programme_request_kwh"],
    )


def evaluate(directory, seed, i, split, workers):
    path = directory / f"eval-{seed}-{i}-{split}.json"
    if path.exists():
        return
    community.init_eval(directory, candidate(directory, seed, i))
    cases = [c for c in community.read_json(directory / "cases.json") if c["split"] == split]
    rows = parallel(partial(community.evaluate_case, inspect_episode=meter), cases, workers)
    for row, case in zip(rows, cases, strict=True):
        assert row["start"] == case["start"]
        row["early"] = case["early"]
    community.write_json(path, rows)


def metrics(rows):
    result = extended_ppo.service_metrics(
        [
            dict(
                ev_errors_kwh=r["ev_errors_kwh"],
                complete=r["programme_complete"],
                bill_eur=r["bill_eur"],
            )
            for r in rows
        ]
    )
    excess = np.maximum(np.asarray([r["ev_errors_kwh"] for r in rows]), 0.0)
    result["largest_excess_kwh"] = float(excess.max())
    result["p95_excess_kwh"] = float(np.quantile(excess, 0.95))
    for early, label in ((True, "early"), (False, "late")):
        price = [r["inspection"] for r in rows if r["stimulus"] == "price" and r["early"] == early]
        result[label] = dict(
            windows=len(price),
            cheap_share=sum(m["cheap"] for m in price) / sum(m["requested"] for m in price),
            starts_in_cheap=sum(
                m["start"] is not None and (2 if early else 10) <= m["start"] < (9 if early else 17)
                for m in price
            ),
            starts=[m["start"] for m in price],
        )
    return result


def passes_guard(base, x):
    return (
        x["cycles_completed"] == x["windows"]
        and not x["shortfalls_over_01"]
        and x["excesses_over_01"] <= base["excesses_over_01"]
        and x["mean_excess_kwh"] <= base["mean_excess_kwh"] + 1e-6
        and x["early"]["cheap_share"] >= base["early"]["cheap_share"] - 1e-4
    )


def select(directory, seed, workers):
    for i in range(len(STRENGTHS) + 1):
        evaluate(directory, seed, i, "validation", workers)
    scores = [
        metrics(community.read_json(directory / f"eval-{seed}-{i}-validation.json"))
        for i in range(len(STRENGTHS) + 1)
    ]
    accepted = [
        i
        for i in range(1, len(scores))
        if passes_guard(scores[0], scores[i])
        and scores[i]["late"]["cheap_share"] > scores[0]["late"]["cheap_share"] + 1e-4
    ]
    # Always evaluate a trained candidate. A validation failure is a rejection, even
    # if the preselected diagnostic candidate performs well on the final dates.
    best = max(
        accepted or list(range(1, len(scores))), key=lambda i: scores[i]["late"]["cheap_share"]
    )
    community.write_json(
        directory / f"selection-{seed}.json",
        dict(index=best, accepted=bool(accepted), scores=scores),
    )
    for i in (0, best):
        evaluate(directory, seed, i, "test", workers)


def diagnostic(directory, seed, index, workers):
    path = directory / f"probe-{seed}-{index}.json"
    if path.exists():
        return
    community.init_eval(directory, candidate(directory, seed, index))
    cases = [
        c
        for c in community.read_json(directory / "cases.json")
        if c["split"] == "test" and c["stimulus"] == "price"
    ]
    rows = parallel(extended_ppo.evaluate_case, cases, workers)
    assert [r["start"] for r in rows] == [c["start"] for c in cases]
    community.write_json(path, rows)


def public_evaluation(directory, seed):
    chosen = community.read_json(directory / f"selection-{seed}.json")["index"]
    return verify_checkpoint(directory, candidate(directory, seed, chosen), f"paired-{seed}")


def verify_checkpoint(directory, weights, name):
    cases = community.read_json(directory / "cases.json")
    case = next(c for c in cases if c["split"] == "test")
    cfg = yaml.safe_load((directory / "job.yaml").read_text())
    cfg = evaluation_job(cfg, weights, case["start"])
    path = directory / f"public-{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from algorithms.runtime.cli import cli_main; cli_main()",
            "--config",
            str(path),
            "--base-dir",
            str(directory / "public"),
            "--job_id",
            name,
        ],
        check=True,
    )
    completed = community.read_json(directory / f"public/jobs/{name}/results/result.json")
    assert completed["status"] == "completed"
    episodes = completed["episode_execution"]
    assert len(episodes) == 1 and episodes[0]["deterministic"] and not episodes[0]["training"]
    assert episodes[0]["start_time_step"] == case["start"]
    return episodes[0]


def report(directory):
    result = dict(
        simulator=version("softcpsrecsimulator"),
        dataset="Core-30 Normal, 2023, three controlled households",
        dates={
            s: [(date(2023, 1, 1) + timedelta(days=d)).isoformat() for d in days]
            for s, days in DAYS
        },
        exploration="Six candidate starts at quarters 2, 4, 6, 8, 10, 12, independently of price. Start now versus IDLE followed by a uniformly chosen later feasible candidate. Identical physical histories up to the fork. Targets use complete discounted operating rewards.",
        update="Ridge regression of measured return differences on the original critic's last hidden layer, followed by expected actor return improvement from the fitted critic. Only the dishwasher row of the actor changes.",
        selection="On validation, require all cycles completed, zero EV shortfalls above 0.1 kWh, no increase in EV excess count or mean excess, no loss of early cheap share and improvement in late share. Choose the best late share; if none pass, evaluate the highest late share for diagnosis only. Final dates used for confirmation once.",
        models={},
    )
    for seed in (43, 71):
        selection = community.read_json(directory / f"selection-{seed}.json")
        best = selection["index"]
        final = {}
        probes = {}
        departure_errors = {}
        for name, i in (("48", 0), ("paired", best)):
            natural = community.read_json(directory / f"eval-{seed}-{i}-test.json")
            final[name] = metrics(natural)
            departure_errors[name] = {r["start"]: np.asarray(r["ev_errors_kwh"]) for r in natural}
            probes[name] = community.read_json(directory / f"probe-{seed}-{i}.json")
        for data in probes.values():
            assert len(data) == final["48"]["early"]["windows"] + final["48"]["late"]["windows"]
        assert [r["start"] for r in probes["48"]] == [r["start"] for r in probes["paired"]]
        available = {
            a["start"]
            for a, b in zip(probes["48"], probes["paired"], strict=True)
            if a["fixed"] and b["fixed"]
        }
        measured = {}
        for name, rows in probes.items():
            measured[name] = {}
            for early, label in ((False, "late"), (True, "early")):
                group = [r for r in rows if r["early"] == early]
                measured[name][label] = dict(
                    fixed=extended_ppo.choice_metrics(
                        [r["fixed"] for r in group if r["start"] in available]
                    ),
                    own_start=extended_ppo.choice_metrics(
                        [r["at_start"] for r in group if r["at_start"]]
                    ),
                )
        src = candidate(directory, seed, best)
        old = torch.load(base(seed), map_location="cpu", weights_only=False)
        new = torch.load(src, map_location="cpu", weights_only=False)
        for key in ("configuration", "learning_architecture"):
            assert old[key] == new[key]
        for network, allowed in (("actor", "coordination"), ("critic", "value_head.2.weight")):
            changed = [k for k in old[network] if not torch.equal(old[network][k], new[network][k])]
            assert changed == [allowed]
        mask = [0, 1, 3, 4]
        assert torch.equal(old["actor"]["coordination"][mask], new["actor"]["coordination"][mask])
        dest = RUNS / f"paired-{seed}/latest_checkpoint.pth"
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, dest)
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
        result["models"][str(seed)] = dict(
            fit=community.read_json(directory / f"fit-{seed}.json"),
            validation=selection,
            final=final,
            critic=measured,
            individual_excess_increase_over_01=sum(
                int(
                    np.count_nonzero(
                        np.maximum(0, after) - np.maximum(0, departure_errors["48"][key]) > 0.1
                    )
                )
                for key, after in departure_errors["paired"].items()
            ),
            checkpoint=str(dest),
            sha256=digest,
            passed_validation_and_test=(
                selection["accepted"]
                and passes_guard(final["48"], final["paired"])
                and final["paired"]["late"]["cheap_share"]
                > final["48"]["late"]["cheap_share"] + 1e-4
            ),
            public_episode=public_evaluation(directory, seed),
        )
    community.write_json(RUNS / "paired-choices.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "explore",
            "fit",
            "select",
            "baseline",
            "probe-base",
            "probe-trained",
            "report",
        ],
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", type=int, choices=(43, 71), default=43)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    d = args.directory
    if args.action == "prepare":
        community.make_fixture(d, community.schema_path(), case_days=DAYS)
    elif args.action == "report":
        report(d)
    elif args.action == "explore":
        explore(d, args.seed, args.workers)
    elif args.action == "fit":
        fit(d, args.seed)
    elif args.action == "baseline":
        for split in ("validation", "test"):
            evaluate(d, args.seed, 0, split, args.workers)
    elif args.action in {"probe-base", "probe-trained"}:
        i = (
            community.read_json(d / f"selection-{args.seed}.json")["index"]
            if args.action == "probe-trained"
            else 0
        )
        diagnostic(d, args.seed, i, args.workers)
    else:
        select(d, args.seed, args.workers)
