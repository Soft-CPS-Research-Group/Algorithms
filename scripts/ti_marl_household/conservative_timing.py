"""Constrain a dishwasher policy update after an EV overdelivery in the first timing study."""

import argparse
import hashlib
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from scripts.ti_marl_household import community
from scripts.ti_marl_household import paired_choices as paired
from scripts.ti_marl_household import start_timing as timing

DAYS = (
    ("train", [17, 45, 75, 114, 159, 202]),
    ("validation", [224, 261, 291, 320]),
    ("test", [243, 258, 272, 286, 305, 324, 338, 354]),
)
WINDOWS = ((2, 10), (4, 9), (3, 12), (5, 11))


def price_start(split, day, early):
    return WINDOWS[dict(DAYS)[split].index(day) % len(WINDOWS)][0 if early else 1]


def explore(d, workers):
    community.init_eval(d, timing.PARENT)
    cases = [
        c
        for c in community.read_json(d / "cases.json")
        if c["split"] == "train" and (c["stimulus"] == "price" or c["day"] == DAYS[0][1][0])
    ]
    community.write_json(d / "branches.json", paired.parallel(timing.explore_case, cases, workers))


def fit(first_rows, second_rows):
    from scipy.optimize import LinearConstraint, minimize

    groups = [r["outcomes"] for r in first_rows + second_rows if "outcomes" in r]

    def tensor(a):
        return torch.tensor(a, dtype=torch.float64)

    x = tensor([[s["choice"]["features"] for s in g[:-1]] for g in groups])
    logits = tensor([[s["choice"]["log_odds"] for s in g[:-1]] for g in groups])
    returns = tensor([[s["value"] for s in g] for g in groups])
    returns -= returns.mean(dim=1, keepdim=True)
    returns /= returns.std()
    prior = timing.start_probabilities(logits)
    signs = torch.where(logits[:, 0] >= 0, 1.0, -1.0)
    first = logits[:, 0].abs()
    candidates = []
    for trust, margin in ((0.02, 0.35), (0.02, 1.4), (0.12, 1.4)):
        constraint = LinearConstraint(
            (x[:, 0, :] * signs[:, None]).numpy(), (first.clamp_max(margin) - first).numpy(), np.inf
        )

        def loss_and_grad(w):
            w = tensor(w).requires_grad_()
            p = timing.start_probabilities(logits + x @ w)
            kl = (p * (p.clamp_min(1e-12).log() - prior.clamp_min(1e-12).log())).sum(dim=1)
            loss = -(p * returns).sum(dim=1).mean() + trust * kl.mean() + 0.0002 * w.square().sum()
            (grad,) = torch.autograd.grad(loss, w)
            return float(loss.detach()), grad.detach().numpy()

        result = minimize(
            loss_and_grad,
            np.zeros(x.shape[2]),
            method="SLSQP",
            jac=True,
            constraints=constraint,
            options={"ftol": 1e-10, "maxiter": 350},
        )
        if not result.success:
            raise ValueError(f"Constrained actor update did not converge: {result.message}")
        revised = logits + x @ tensor(result.x)
        assert ((revised[:, 0] * signs) >= first.clamp_max(margin) - 1e-6).all()
        candidates.append(dict(delta=result.x.tolist(), trust=trust, margin=margin))
    return candidates, len(groups)


def accepted(check):
    return timing.ready(check) and check["largest_individual_excess_increase_kwh"] <= 1e-6


@torch.no_grad()
def replay_excess(d, weights, ref, rows):
    cases = {c["start"]: c for c in community.read_json(d / "cases.json")}
    replays = []
    for a, b in zip(ref, rows, strict=True):
        change = np.maximum(0, b["ev_errors_kwh"]) - np.maximum(0, a["ev_errors_kwh"])
        if change.max() <= 0.1:
            continue
        port = int(change.argmax())
        case = cases[a["start"]]
        starts = [r["inspection"]["start"] for r in (a, b)]
        traces = {}
        for label, model, forced in (
            ("parent", timing.PARENT, None),
            ("candidate", weights, None),
            ("parent_at_candidate_start", timing.PARENT, starts[1]),
            ("candidate_at_parent_start", weights, starts[0]),
        ):
            community.init_eval(d, model)
            env = timing.environment(case)
            episode = timing.probe.simulate(
                env,
                community.policy,
                case,
                arm="meter",
                branch="meter" if forced is None else "explore_start",
                anchor=forced,
                observe_choice=lambda *_: None,
            )
            charger = env.buildings[port].electric_vehicle_chargers[0]
            traces[label] = dict(
                dishwasher_start=episode["start"],
                departure_error_kwh=episode["ev_errors"][port],
                ac_kwh_by_quarter=charger.electricity_consumption[
                    : case["departures"][port]
                ].tolist(),
            )
        for label, original in (("parent", a), ("candidate", b)):
            np.testing.assert_allclose(
                traces[label]["departure_error_kwh"], original["ev_errors_kwh"][port], atol=1e-6
            )
        replays.append(dict(case=case, port=port, traces=traces))
    return replays


def finish(d, workers):
    first = d.parent
    opts, n = fit(
        community.read_json(first / "branches.json"), community.read_json(d / "branches.json")
    )
    orig = torch.load(timing.PARENT, map_location="cpu", weights_only=False)
    initial = orig["actor"]["coordination"].clone()
    community.init_eval(d, timing.PARENT)
    assert (
        community.policy.reward_penalties.model_dump()
        == orig["configuration"]["algorithm"]["hyperparameters"]["reward_penalties"]
    )
    ids = orig["learner"]["actor_optimizer"]["param_groups"][0]["params"]
    i = list(dict(community.policy.actor.named_parameters())).index("coordination")
    orig["learner"]["actor_optimizer"]["state"].pop(ids[i], None)
    score = {}
    guards = {}
    ref = community.read_json(d / "reference-validation.json")
    for i, o in enumerate(opts):
        k = f"conservative-{i}"
        orig["actor"]["coordination"] = initial.clone()
        orig["actor"]["coordination"][2] += torch.tensor(o["delta"]).to(initial)
        torch.save(orig, d / f"{k}.pth")
        timing.evaluate(d, d / f"{k}.pth", k, "validation", workers)
        rows = community.read_json(d / f"{k}-validation.json")
        score[k] = timing.summary(rows)
        guards[k] = timing.judge(ref, rows)
    allowed = [k for k in score if accepted(guards[k])]
    key = max(
        allowed or list(score),
        key=lambda k: (
            round(score[k]["late"]["cheap_share"], 7),
            -score[k]["late"]["mean_minutes_before"],
            -score[k]["mean_excess_kwh"],
        ),
    )
    info = dict(
        chosen=key,
        validation_passed=bool(allowed),
        scores=score,
        guards=guards,
        complete_training_windows=n,
        candidates=opts,
    )
    community.write_json(d / "selected.json", info)
    path = d / f"{key}.pth"
    # These previously used windows are a regression check. The subsequent
    # confirmation uses different days and EV sessions, and the weights are fixed.
    timing.evaluate(first, path, "conservative-regression", "test", workers)
    timing.evaluate(d, path, "chosen", "test", workers, diagnose=True)
    regression = timing.judge(
        community.read_json(first / "reference-test.json"),
        community.read_json(first / "conservative-regression-test.json"),
    )
    ref = community.read_json(d / "reference-test.json")
    rows = community.read_json(d / "chosen-test.json")
    guard = timing.judge(ref, rows)
    model = torch.load(path, map_location="cpu", weights_only=False)
    old = torch.load(timing.PARENT, map_location="cpu", weights_only=False)
    assert model["configuration"] == old["configuration"]
    assert all(torch.equal(model["critic"][k], v) for k, v in old["critic"].items())
    for k, v in old["actor"].items():
        if k == "coordination":
            assert (model["actor"][k] != v).any(dim=1).tolist() == [
                False,
                False,
                True,
                False,
                False,
            ]
        else:
            assert torch.equal(model["actor"][k], v)
    result = dict(
        dates={s: [str(date(2023, 1, 1) + timedelta(days=i)) for i in ds] for s, ds in DAYS},
        update="Identical operating rewards. Maximise expected return of the first-start distribution while preserving the parent actor's first available decision on training trajectories via linear constraints. No external action labels or inference-time guard.",
        selection=info,
        previous_test_regression=regression,
        previous_test=timing.summary(
            community.read_json(first / "conservative-regression-test.json")
        ),
        final=dict(
            reference=timing.summary(ref),
            trained=timing.summary(rows),
            guard=guard,
            confirmed=bool(allowed and accepted(regression) and accepted(guard)),
        ),
        diagnosis=timing.diagnostic_summary(community.read_json(d / "chosen-diagnosis.json")),
        excess_replays=replay_excess(d, path, ref, rows),
        public_job=paired.verify_checkpoint(d, path, "conservative-71"),
    )
    dest = paired.RUNS / "conservative-timing-71/latest_checkpoint.pth"
    dest.parent.mkdir(exist_ok=True)
    shutil.copyfile(path, dest)
    result["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    result["parent_sha256"] = hashlib.sha256(timing.PARENT.read_bytes()).hexdigest()
    result["model"] = str(dest)
    community.write_json(paired.RUNS / "conservative-timing-71.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "explore", "reference", "finish"))
    p.add_argument("directory", type=Path)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    d = a.directory
    if a.action == "prepare":
        earlier = {v for _, days in timing.DAYS for v in days}
        assert not earlier.intersection(v for _, days in DAYS for v in days)
        community.make_fixture(d, community.schema_path(), case_days=DAYS, price_start=price_start)
    elif a.action == "explore":
        explore(d, a.workers)
    elif a.action == "reference":
        for split in ("validation", "test"):
            timing.evaluate(d, timing.PARENT, "reference", split, a.workers)
    else:
        finish(d, a.workers)


if __name__ == "__main__":
    main()
