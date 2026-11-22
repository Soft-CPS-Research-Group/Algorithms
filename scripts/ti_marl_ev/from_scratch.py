"""Compare the V2 prefix learner with a reward-only, matched-session policy gradient.

Both use the complete isolated-EV V2 actor with newly initialised weights.
Each training batch contains four independent trajectories of one session.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from algorithms.ti_marl.agent import TIMARL
from algorithms.ti_marl.experiments.ev_levels import LevelEVActor
from algorithms.ti_marl.experiments.ev_pretraining import LongEVActor
from scripts.train_ti_marl_ev_sessions import SessionProbe, evaluate, fingerprint, play

from .metrics import measures
from .protocol import PRE, TRACE, split

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/ti-marl-2023-ev-from-scratch"


def fresh(width, seed, *, levels=False):
    recipe = yaml.safe_load((ROOT / "configs/templates/rl/ti_marl_core30.yaml").read_text())
    hyper = recipe["pipeline"][0]["hyperparameters"]
    hyper["actor"]["d_model"] = width
    inputs = PRE / "inputs"
    hyper.update(
        typed_interfaces_dir=str(inputs / "interfaces"),
        simulator_bindings_path=str(inputs / "interfaces/technology_bindings/simulator.yaml"),
        issued_forecasts_path=str(inputs / "forecasts-6116.jsonl"),
    )
    with patch(
        "algorithms.ti_marl.policy.joint.JointActor", LevelEVActor if levels else LongEVActor
    ):
        agent = TIMARL(
            {
                "training": {"seed": seed},
                "algorithm": {"hyperparameters": hyper},
                "simulator": recipe["simulator"],
            }
        )
    assert not agent.manager and agent.reward_profile == "thesis_common_reward_v1"
    assert agent.reference_sites == 30 and agent.actor.step_seconds == 900
    return agent


def matched_return_update(learner, episodes):
    # All alternatives are the actor's own stochastic decisions. Compare each
    # realised return with independent rollouts starting in the same state.
    returns = [
        sum(learner.gamma**t * st.community_reward for t, st in enumerate(ep)) for ep in episodes
    ]
    total = sum(returns)
    learner.actor_optimizer.zero_grad(set_to_none=True)
    for i, episode in enumerate(episodes):
        advantage = returns[i] - (total - returns[i]) / (len(episodes) - 1)
        for step in episode:
            for scores in learner.actor.replay(step.snapshot, step.traces).values():
                if scores:
                    logp = torch.stack([v for _, v, _ in scores]).sum()
                    # Constant multiplier only rescales the gradient in a small per-site reward.
                    (-30 * advantage * logp / len(episodes)).backward()
    norm = torch.nn.utils.clip_grad_norm_(learner.actor.parameters(), learner.max_grad_norm)
    assert torch.isfinite(norm)
    learner.actor_optimizer.step()
    learner.rollout.clear()


def evaluation(agent, probe, cases):
    with torch.no_grad():
        rows = evaluate(cases, agent, None, probe)
    return measures(rows)


def run(width, seed, learner_name, sessions, *, excess_price=0.0, levels=False, output=OUTPUT):
    if not np.isfinite(excess_price) or excess_price < 0:
        raise ValueError("The excess price must be finite and nonnegative")
    if sessions < 4 or sessions % 4:
        raise ValueError("Use a positive multiple of four sessions")
    torch.set_num_threads(1)
    agent = fresh(width, seed, levels=levels)
    train, dev, tests = split()
    rng = np.random.default_rng(seed)
    probe = SessionProbe(TRACE)
    init = fingerprint(agent.actor)
    report = {
        "width": width,
        "seed": seed,
        "excess_price_eur_per_kwh": excess_price,
        "power": "levels" if levels else "continuous",
        "learner": learner_name,
        "initial_actor": init,
        "initial_critic": fingerprint(agent.critic),
        "actor_lr": agent.learner.actor_optimizer.param_groups[0]["lr"],
        "reward": agent.reward_profile,
        "penalties": agent.reward_penalties.model_dump(),
        "reference_sites": agent.reference_sites,
        "random_initialisation": True,
        "dev_before": evaluation(agent, probe, dev),
        "training_sessions": sessions,
        "milestones": {},
    }
    # Use a deterministic stream by seed; compared learners receive identical cases.
    idxs = rng.integers(len(train), size=sessions // 4)
    report["training_order"] = hashlib.sha256(idxs.tobytes()).hexdigest()
    for group, idx in enumerate(idxs, 1):
        case = train[idx]
        episodes = []
        for _ in range(4):
            start = len(agent.learner.rollout.steps)
            play(case, agent, None, probe, "local", excess_price=excess_price)
            ep = agent.learner.rollout.steps[start:]
            assert ep[-1].terminated and not any(st.truncated for st in ep)
            episodes.append(ep)
        if learner_name == "prefix":
            agent.actor.train()
            agent.critic.train()
            assert all(np.isfinite(x) for x in agent.learner.update().values())
        else:
            matched_return_update(agent.learner, episodes)
        n = group * 4
        if n in (384, 1536, sessions):
            m = evaluation(agent, probe, dev)
            report["milestones"][n] = m
            print(
                n,
                m["near_target"],
                m["price"]["timing_and_target"],
                m["solar"]["timing_and_target"],
                flush=True,
            )
    assert init != fingerprint(agent.actor)
    report["final_actor"] = fingerprint(agent.actor)
    report["test"] = evaluation(agent, probe, tests["new"])
    report["fixed_08"] = evaluate(tests["08h"], agent, None, probe)
    at_target = [
        sum(play(replace(c, request_kwh=0.0, early=early), agent, None, probe, "eval")["kwh"])
        for c in tests["new"]
        for early in (True, False)
    ]
    report["already_at_target"] = {
        "sessions": len(at_target),
        "idle": sum(v < 1e-8 for v in at_target),
        "max_extra": max(at_target),
    }
    assert not any(
        x.split(".")[-1] in ("demonstrations", "fit", "check_teacher") for x in sys.modules
    )
    output.mkdir(exist_ok=True, parents=True)
    tag = f"{learner_name}-{width}-{seed}"
    if output != OUTPUT or levels or excess_price:
        tag += f"-{report['power']}-{excess_price:g}"
        report["off_grid"] = evaluation(agent, probe, tests["off_grid"])
    dest = output / (tag + ".json")
    dest.write_text(json.dumps(report, allow_nan=False))
    torch.save(
        {
            "width": width,
            "seed": seed,
            "power": report["power"],
            "excess_price_eur_per_kwh": excess_price,
            "learner": learner_name,
            "actor": agent.actor.state_dict(),
        },
        dest.with_suffix(".pth"),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("width", type=int, choices=(32, 64))
    p.add_argument("seed", type=int)
    p.add_argument("learner", choices=("prefix", "matched"))
    p.add_argument("--sessions", type=int, default=3072)
    p.add_argument("--excess-price", type=float, default=0.0)
    p.add_argument("--levels", action="store_true")
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    run(
        args.width,
        args.seed,
        args.learner,
        args.sessions,
        excess_price=args.excess_price,
        levels=args.levels,
        output=args.output,
    )
