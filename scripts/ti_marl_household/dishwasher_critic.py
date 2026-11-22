"""Probe the saved V2 critics with legal dishwasher choices and paired SoftCPS rollouts."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from algorithms.ti_marl.agent import TIMARL
from algorithms.ti_marl.policy.joint import JointActor
from scripts.ti_marl_household import community
from scripts.ti_marl_household.price_windows import RUNS
from utils.entity_adapter import EntityContractAdapter

SITE = community.SITES[1]
ARMS = ("frozen", "joint-43", "joint-71")
STEPS = community.DECISIONS


class InadmissibleDishwasherChoice(ValueError):
    """The site's physical action mask excludes the requested intervention."""

    def __init__(self, mode):
        self.mode = mode
        super().__init__(f"Requested an inadmissible {mode}")


def setup(directory):
    global cfg, agents, initial_critic
    torch.set_num_threads(1)
    cfg = yaml.safe_load((directory / "job.yaml").read_text())
    agents = {}
    for arm in ARMS:
        agent = TIMARL({**cfg, "algorithm": cfg["pipeline"][0]})
        path = community.SOURCE_POLICY if arm == "frozen" else RUNS / arm / "latest_checkpoint.pth"
        fresh = {k: v.detach().clone() for k, v in agent.critic.state_dict().items()}
        agent.load_checkpoint(str(path))
        if arm == "frozen":
            initial_critic = all(
                torch.equal(v, fresh[k]) for k, v in agent.critic.state_dict().items()
            )
        agent.actor.eval()
        agent.critic.eval()
        agents[arm] = agent


@contextmanager
def dishwasher_mode(actor, mode):
    if mode is None:
        yield
        return
    normal = actor._choose
    hits = []

    def choose(c, context, state, deterministic):
        c = normal(c, context, state, deterministic)
        if c.stage == "mode" and c.group_type == "deferrable":
            idx = actor.group_modes[c.group_type].index(mode)
            if idx not in c.allowed:
                raise InadmissibleDishwasherChoice(mode)
            hits.append(c.group_id)
            return replace(c, mode=idx)
        return c

    with patch.object(actor, "_choose", choose):
        yield
    if not hits:
        raise InadmissibleDishwasherChoice(mode)
    if len(hits) != 1:
        raise ValueError("Expected exactly one dishwasher command")


def decision(trace):
    return next(
        (i, c)
        for i, c in enumerate(trace.choices)
        if c.stage == "mode" and c.group_type == "deferrable"
    )


def alternatives(agent):
    snapshot = agent._pending["snapshot"]
    natural = agent._pending["prefix_traces"]
    k, chosen = decision(natural[SITE])
    modes = agent.actor.group_modes["deferrable"]
    observed = modes[chosen.mode]
    other = {"IDLE": "START", "START": "IDLE"}[observed]
    with dishwasher_mode(agent.actor, other):
        alternate = agent.actor(snapshot, deterministic=True).traces
    if alternate[SITE].supervised:
        raise InadmissibleDishwasherChoice(other)
    j, alternative = decision(alternate[SITE])
    assert j == k and modes[alternative.mode] == other
    assert natural[SITE].choices[:k] == alternate[SITE].choices[:k]
    assert set(modes[i] for i in chosen.allowed) == {"START", "IDLE"}
    return snapshot, k, {observed: natural, other: alternate}


def score(agent, snapshot, k, traces):
    site_trace = traces["START"][SITE]
    c = site_trace.choices[k]
    actor = agent.actor
    local, contexts = actor._contexts(snapshot)[SITE]
    prefix = actor.memory.start([contexts[g] for g in site_trace.order], local)
    for past in site_trace.choices[:k]:
        prefix = actor.memory.extend(prefix, contexts[past.group_id], past)
    distribution = actor._distribution(contexts[c.group_id], prefix, c)
    base = JointActor._distribution(actor, contexts[c.group_id], prefix, c)
    idx = c.allowed.index(actor.group_modes[c.group_type].index("START"))
    idle = c.allowed.index(actor.group_modes[c.group_type].index("IDLE"))
    base_odds = float(base.logits[idx] - base.logits[idle])
    features = actor.household_inputs[c.group_id][1]
    coefficients = actor.coordination[2].detach().cpu().numpy()
    contributions = np.asarray(features) * coefficients
    odds = float(distribution.logits[idx] - distribution.logits[idle])
    np.testing.assert_allclose(odds, base_odds + sum(contributions), atol=5e-6)
    values = {}
    for mode in ("START", "IDLE"):
        v = agent.critic.prefix_values(snapshot, traces[mode])[SITE]
        values[mode] = float(v[k + 1])
        if mode == "START":
            before = float(v[k])
        else:
            np.testing.assert_allclose(float(v[k]), before, atol=1e-6)
    return dict(
        before=before,
        start=values["START"],
        wait=values["IDLE"],
        p_start=float(distribution.probs[idx]),
        log_odds=odds,
        base_log_odds=base_odds,
        price_log_odds=float(sum(contributions[1:4])),
        other_readout_log_odds=float(contributions[[0, 4, 5, 6, 7]].sum()),
    )


def attach(agent, env):
    agent.attach_environment(
        observation_names=[],
        action_names=env.action_names,
        action_space=env.flat_action_space,
        observation_space=[],
        metadata=dict(interface="entity", seconds_per_time_step=900, entity_specs=env.entity_specs),
    )


def simulate(env, agent, case, *, arm, branch="natural", anchor=None, observe_choice=None, observe_step=None):
    payload, info = env.reset()
    adapter = EntityContractAdapter(env, normalization_enabled=False, clip=False)
    adapter.rebuild_layout(payload)
    attach(agent, env)
    agent.on_episode_start(episode=0, training=False)
    bills, rewards, action_history, errors = [], [], [], []
    native_start, native_scores, shared_scores = None, None, None
    supervised_slots = []
    cheap_slot = case.get("price_start", 2 if case["early"] else 10)
    for t in range(STEPS):
        info = dict(info, episode_start_time_step=case["start"])
        agent.set_entity_observation_context(observation_payload=payload, info=info)
        mode = None
        probing = branch in {"probe_start", "probe_wait", "explore_start"}
        if probing and t <= anchor:
            mode = "START" if (branch != "probe_wait" and t == anchor) else "IDLE"
        if branch == "wait_once" and t == anchor:
            mode = "IDLE"
        if branch == "wait_cheap" and anchor <= t <= cheap_slot:
            mode = "START" if t == cheap_slot else "IDLE"
        try:
            with dishwasher_mode(agent.actor, mode):
                actions = agent.predict([], deterministic=True)
        except InadmissibleDishwasherChoice as exc:
            if not probing:
                raise
            return dict(unavailable=exc.mode, slot=t)
        trace = agent._pending["prefix_traces"][SITE]
        if observe_step is not None:
            observe_step(t, agent, env)
        choice = None if trace.supervised else decision(trace)[1]
        if trace.supervised:
            supervised_slots.append(t)
            if mode is not None:
                if probing:
                    return dict(unavailable=mode, slot=t, supervised=True)
                raise InadmissibleDishwasherChoice(mode)
        start = choice is not None and agent.actor.group_modes["deferrable"][choice.mode] == "START"
        if native_start is None and start:
            native_start = t
        shared = branch in {"natural", "forecast_check"} and arm == "frozen" and t == 2
        at_start = (
            branch in {"natural", "unforced"}
            and start
            and native_start == t
            and native_scores is None
            and not case["early"]
            and len(choice.allowed) > 1
        )
        fixed_probe = branch == "probe_start" and t == anchor
        if branch == "explore_start" and t == anchor and len(choice.allowed) > 1:
            try:
                snap, index, traces = alternatives(agent)
            except InadmissibleDishwasherChoice:
                pass
            else:
                native_scores = observe_choice(agent, snap, index, traces)
        if shared or at_start or fixed_probe:
            if fixed_probe and len(choice.allowed) < 2:
                return dict(unavailable="IDLE", slot=t)
            try:
                snap, index, traces = alternatives(agent)
            except InadmissibleDishwasherChoice as exc:
                if fixed_probe:
                    return dict(unavailable=exc.mode, slot=t)
                if not at_start or shared:
                    raise
                at_start = False
        if shared or at_start or fixed_probe:
            outlook = next(o for o in snap.outlooks if o.site == SITE)
            prices = np.asarray(outlook.values)[0]
            assert min(outlook.quality[0]) > 0
            cheap_now = cheap_slot <= t < cheap_slot + 7
            np.testing.assert_allclose(prices[0], 0.07 if cheap_now else 0.25, atol=1e-6)
            if t <= cheap_slot:
                np.testing.assert_allclose(prices[cheap_slot - t], 0.07, atol=1e-6)
            if shared:
                shared_scores = {name: score(agents[name], snap, index, traces) for name in ARMS}
            if branch == "forecast_check":
                curves = []
                for curve in snap.outlooks:
                    prices_now = curve.values[0]
                    np.testing.assert_allclose(prices_now[0], 0.25, atol=1e-6)
                    curves.append(
                        replace(
                            curve, values=((prices_now[0],) * len(prices_now), *curve.values[1:])
                        )
                    )
                flat = replace(snap, outlooks=tuple(curves))
                return dict(
                    case=case["start"],
                    actual=shared_scores,
                    flat={name: score(agents[name], flat, index, traces) for name in ARMS},
                )
            if at_start or fixed_probe:
                native_scores = score(agent, snap, index, traces)
                if at_start:
                    assert native_scores["p_start"] >= 0.5
        if shared and case["early"]:
            return dict(shared=shared_scores)
        action_history.append([np.asarray(x).tolist() for x in actions])
        following, raw_reward, done, truncated, next_info = env.step(
            adapter.to_entity_actions(actions, env.action_names)
        )
        if trace.supervised and native_start is None:
            if env.buildings[1].deferrable_appliances_electricity_consumption[t] > 1e-6:
                native_start = t
        assert bool(done or truncated) == (t == STEPS - 1)
        start_id = case["start"]
        feedback = dict(next_info, episode_start_time_step=start_id)
        if "reward_components" not in feedback:
            feedback["reward_components"] = env.reward_function.get_last_components()
        feedback.update(agent.interval_reward_feedback(env, t, feedback))
        agent.set_entity_transition_context(
            observation_payload=payload, next_observation_payload=following, info=feedback
        )
        rewards.append(float(agent._real_reward(agent._pending["snapshot"], raw_reward)))
        bills.append(
            sum(
                float(row["settled_cost_eur"])
                for row in env._community_market_settlement_history[t]
            )
        )
        for building, departure in zip(env.buildings, case["departures"]):
            if t == departure - 1:
                charger = building.electric_vehicle_chargers[0]
                ev = charger.past_connected_evs[t]
                target = charger.charger_simulation.electric_vehicle_required_soc_departure[t]
                errors.append(float((ev.battery.soc[t] - target) * ev.battery.capacity))
        payload, info = following, next_info
    app = env.buildings[1].deferrable_appliances[0]
    measured = np.flatnonzero(
        env.buildings[1].deferrable_appliances_electricity_consumption[:STEPS] > 1e-6
    )
    assert (int(measured[0]) if len(measured) else None) == native_start and len(errors) == 2
    programme = env.buildings[1].deferrable_appliances_electricity_consumption[:STEPS]
    return dict(
        start=native_start,
        supervised_slots=supervised_slots,
        ev_errors=errors,
        programme_kwh=float(np.sum(programme)),
        cheap_kwh=float(np.sum(programme[cheap_slot : cheap_slot + 7])),
        shared=shared_scores,
        at_start=native_scores,
        actions=action_history,
        rewards=rewards,
        bill=sum(bills),
        completed=app.cycle_state.get(f"cycle-{case['start']}") == "completed",
        shortfall=max(0.0, -min(errors)),
    )


@torch.no_grad()
def run_case(task):
    arm, case = task
    cfg["simulator"].update(
        simulation_start_time_step=case["start"], simulation_end_time_step=case["start"] + STEPS
    )
    env = community._build_env(cfg)
    agent = agents[arm]
    nat = simulate(env, agent, case, arm=arm)
    result = dict(arm=arm, case=case["start"], early=case["early"], shared=nat["shared"])
    if case["early"]:
        return result
    anchor = nat["start"]
    assert 2 <= anchor < 10
    gamma = agent.learner.gamma

    def future_reward(run):
        return sum(gamma**i * r for i, r in enumerate(run["rewards"][anchor:]))

    result.update(
        anchor=anchor,
        at_start=nat["at_start"],
        start_bill=nat["bill"],
        gamma=gamma,
        start_return=future_reward(nat),
        start_completed=nat["completed"],
        start_shortfall=nat["shortfall"],
    )
    for branch in ("wait_once", "wait_cheap"):
        other = simulate(env, agent, case, arm=arm, branch=branch, anchor=anchor)
        assert other["actions"][:anchor] == nat["actions"][:anchor]
        np.testing.assert_allclose(other["rewards"][:anchor], nat["rewards"][:anchor], atol=1e-7)
        assert other["start"] > anchor and (branch != "wait_cheap" or other["start"] == 10)
        result[branch] = dict(
            start=other["start"],
            bill=other["bill"],
            completed=other["completed"],
            shortfall=other["shortfall"],
            future_return=future_reward(other),
        )
    return result


@torch.no_grad()
def forecast_check(case):
    cfg["simulator"].update(
        simulation_start_time_step=case["start"], simulation_end_time_step=case["start"] + STEPS
    )
    env = community._build_env(cfg)
    return simulate(env, agents["frozen"], case, arm="frozen", branch="forecast_check")


def score_means(scores):
    return {
        **{key: float(np.mean([r[key] for r in scores])) for key in scores[0]},
        "critic_prefers_start": sum(v["start"] > v["wait"] for v in scores),
        "actor_chooses_start": sum(v["p_start"] >= 0.5 for v in scores),
    }


def summarize(rows, controls, frozen_is_initial):
    ref = {r["case"]: r for r in rows if r["arm"] == "frozen" and not r["early"]}
    flat = {r["case"]: r for r in controls}
    assert len(ref) == 12 and ref.keys() == flat.keys()
    first = community.read_json(RUNS / "price-windows.json")["first_late_case"]
    result = dict(
        late_cases=len(ref),
        dates=community.read_json(RUNS / "comparison.json")["evaluation_dates"],
        frozen_critic_matches_initialization=frozen_is_initial,
        procedure="Deterministic 2023 evaluation. Scores at index k+1 of the native central prefix critic, directly after each admissible mode of the dishwasher. Higher is better. Shared decisions use identical physical snapshots, earlier local choices and witnesses for all critics. The two modes retain their respective native witnesses.",
        matched_state="First legal interval, two hours before the cheap block (same state and prefix from the frozen policy). In the flat control only the issued price curves change; their current price, load and PV are unchanged.",
        outcome="Paired at the first actual start of each policy. All earlier decisions identical within each pair. Hold one quarter then resume the policy, or hold until the first cheap slot then start. Settlement is for all three homes. Discounted per-site community return uses the unchanged thesis reward, including service and terminal stock.",
        reward_profile=cfg["pipeline"][0]["hyperparameters"]["operating_reward"],
        results={},
    )
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm and not r["early"]]
        assert len(selected) == len(ref)
        for r in selected:
            if r["case"] == first["start"]:
                expected = first["by_arm"][arm]
                assert r["anchor"] == expected["first_dishwasher_slot_0based"]
                np.testing.assert_allclose(r["start_bill"], expected["bill_eur"], atol=1e-6)
            for key, val in ref[r["case"]]["shared"][arm].items():
                np.testing.assert_allclose(val, flat[r["case"]]["actual"][arm][key], atol=1e-6)
        outcome = {}
        for key in ("wait_once", "wait_cheap"):
            reward_delta = np.array([r[key]["future_return"] - r["start_return"] for r in selected])
            savings = np.array([r["start_bill"] - r[key]["bill"] for r in selected])
            outcome[key] = dict(
                mean_bill_saved_eur=float(savings.mean()),
                mean_discounted_return_gain=float(reward_delta.mean()),
                return_improved_cases=int((reward_delta > 1e-6).sum()),
                bill_saved_over_0_001_eur=int((savings > 0.001).sum()),
                completed=sum(r[key]["completed"] for r in selected),
                worst_ev_shortfall_kwh=max(r[key]["shortfall"] for r in selected),
            )
        result["results"][arm] = dict(
            matched=score_means([flat[i]["actual"][arm] for i in ref]),
            matched_flat=score_means([flat[i]["flat"][arm] for i in ref]),
            matched_cheap_now=score_means([r["shared"][arm] for r in rows if r["early"]]),
            at_actual_start=score_means([r["at_start"] for r in selected]),
            natural_completed=sum(r["start_completed"] for r in selected),
            natural_worst_ev_shortfall_kwh=max(r["start_shortfall"] for r in selected),
            gamma=selected[0]["gamma"],
            observed=outcome,
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--output", type=Path, default=RUNS / "dishwasher-critic.json")
    args = p.parse_args()
    setup(args.directory)
    cases = [
        c
        for c in community.read_json(args.directory / "cases.json")
        if c["split"] == "test" and c["stimulus"] == "price"
    ]
    tasks = [(arm, c) for c in cases for arm in ARMS if arm == "frozen" or not c["early"]]
    with get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(run_case, tasks, chunksize=1)
        controls = pool.map(forecast_check, [c for c in cases if not c["early"]])
    community.write_json(args.output, summarize(rows, controls, initial_critic))
