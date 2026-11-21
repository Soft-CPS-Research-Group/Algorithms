"""Regression tests at the boundary between the action constructor and prefix PPO."""

from dataclasses import replace

import pytest
import torch

from algorithms.ti_marl.agent import TIMARL
from algorithms.ti_marl.contracts.models import LocalConstraint
from algorithms.ti_marl.learning.prefix_ppo import PrefixStep, credit_for_stages, return_targets
from tests.test_ti_marl_policy_runtime import agent_config, compile_snapshot


def storage_site(tmp_path, variant="v2"):
    _, snap = compile_snapshot(tmp_path, buildings=("Building_1",))
    group = next(g for g in snap.action_groups if g.group_type == "stationary_storage")
    group = replace(group, reservation_profile=(("hourly_loss", 0.0), ("battery_efficiency", 0.94)))
    copy = replace(group, group_id="Building_1:battery_2", module_id="battery_2")
    local = [p for p in snap.observation_parts if p.sensor_id == group.module_id]
    soc = next(p for p in local if p.observation_id == "soc")
    local.append(
        replace(
            soc,
            part_id="capacity",
            observation_id="capacity_kwh",
            feature_names=("capacity_kwh",),
            values=(20.0,),
        )
    )
    other = [replace(p, part_id=p.part_id + "-2", sensor_id=copy.module_id) for p in local]
    limit = LocalConstraint("connection", "Building_1", "group_import_limit_kw", upper_bound=3.0)
    snapshot = replace(
        snap,
        action_groups=(group, copy),
        observation_parts=tuple(local + other),
        constraints=(limit,),
    )
    cfg = agent_config(tmp_path / "agent")
    cfg["algorithm"]["hyperparameters"].update(method_variant=variant, ppo_epochs=1)
    agent = TIMARL(cfg)
    return agent, snapshot


@pytest.mark.parametrize("variant", ["v0", "v1", "v1c", "v2"])
def test_joint_replay_uses_recorded_order_and_physical_density(tmp_path, variant):
    agent, snapshot = storage_site(tmp_path, variant)
    with torch.no_grad():
        # Both advance a charge request: the two 5 kW devices share 3 kW.
        head = agent.actor.mode_heads["stationary_storage"]
        head.weight.zero_()
        head.bias.copy_(head.bias.new_tensor((-30.0, 30.0, -30.0)))
        evaluation = agent.actor(snapshot, deterministic=True)
    trace = evaluation.traces["Building_1"]
    bundle = evaluation.executed[0]
    total_kw = sum(5 * d.fraction for d in bundle.decisions if d.mode == "CHARGE_STATIONARY")
    assert total_kw <= 3.000001
    if variant != "v0":
        assert evaluation.bundles == evaluation.executed
        assert [s.stage for s in trace.choices] == ["mode", "mode", "power", "power"]
        assert sum(d.fraction * 5 for d in bundle.decisions) > 0.0
    scores = agent.actor.replay(snapshot, evaluation.traces)["Building_1"]
    assert len(scores) == sum(s.sampled for s in trace.choices)
    for k, score, _ in scores:
        assert score.item() == pytest.approx(trace.choices[k].old_log_prob, abs=1.0e-5)
    values = agent.critic.prefix_values(snapshot, evaluation.traces)["Building_1"]
    assert len(values) == 1 + len(trace.choices)
    assert torch.allclose(values[0], agent.critic(snapshot)["Building_1"])


def test_prefix_train_checkpoint_and_inapplicable_stage(tmp_path):
    agent, snapshot = storage_site(tmp_path)
    with torch.no_grad():
        head = agent.actor.mode_heads["stationary_storage"]
        head.weight.zero_()
        head.bias.copy_(head.bias.new_tensor((2.0, 0.0, 0.0)))
        evaluation = agent.actor(snapshot, deterministic=True)
        values = agent.critic.prefix_values(snapshot, evaluation.traces)["Building_1"]
    assert evaluation.traces["Building_1"].choices[-1].stage == "inapplicable"
    assert torch.equal(values[-3], values[-2]) and torch.equal(values[-2], values[-1])
    for reward in (1.0, -2.0):
        agent.learner.rollout.add(
            PrefixStep(snapshot, evaluation.traces, reward, {}, {}, True, False, 900.0)
        )
    before_actor = [p.detach().clone() for p in agent.actor.parameters()]
    before_critic = [p.detach().clone() for p in agent.critic.parameters()]
    metrics = agent.learner.update()
    assert metrics["sampled_choices"] == 4
    assert any(not torch.equal(x, y) for x, y in zip(before_actor, agent.actor.parameters()))
    assert any(not torch.equal(x, y) for x, y in zip(before_critic, agent.critic.parameters()))
    saved = agent.save_checkpoint(str(tmp_path / "check"), 2)
    fresh = TIMARL(agent.config)
    fresh.load_checkpoint(saved)
    with torch.no_grad():
        assert (
            agent.actor(snapshot, deterministic=True).bundles
            == fresh.actor(snapshot, deterministic=True).bundles
        )
    legacy = agent_config(tmp_path / "legacy")
    with pytest.raises(ValueError, match="architecture"):
        TIMARL(legacy).load_checkpoint(saved)


def test_physical_discount_truncation_and_credit():
    from types import SimpleNamespace

    sites = SimpleNamespace(agent_ids=("A", "B"))
    first = PrefixStep(sites, {}, 1.0, {"A": -1.0}, {}, False, False, 1800.0)
    boundary = PrefixStep(sites, {}, 2.0, {}, {"A": 3.0, "B": 5.0}, False, True, 3600.0)
    separate = PrefixStep(sites, {}, 120.0, {}, {}, True, False, 900.0)
    targets = return_targets([first, boundary, separate], 0.81, 3600.0)
    assert targets[1] == pytest.approx({"A": 2 + 0.81 * 3, "B": 2 + 0.81 * 5})
    assert targets[0]["A"] == pytest.approx(0.9 * targets[1]["A"])
    values = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)
    assert credit_for_stages("v2", values, 7.0).tolist() == [2.0, -1.0]
    assert credit_for_stages("v1c", values, 7.0).tolist() == [1.0, 1.0]
    assert credit_for_stages("v1", values, 7.0).tolist() == [6.0, 6.0]
    assert not credit_for_stages("v2", values, 7.0).requires_grad


def test_local_critic_only_observes_its_site(tmp_path):
    _, snapshot = compile_snapshot(tmp_path)
    config = agent_config(tmp_path / "local")
    hyper = config["algorithm"]["hyperparameters"]
    hyper.update(method_variant="v2", backbone={"name": "ppo"}, critic={"kind": "local"})
    local = TIMARL(config).critic
    changed = replace(
        snapshot,
        observation_parts=tuple(
            replace(p, values=tuple(17 * x + 13 for x in p.values))
            if p.owner_agent_id == "Building_2"
            else p
            for p in snapshot.observation_parts
        ),
    )
    with torch.no_grad():
        before, after = local(snapshot), local(changed)
    assert torch.allclose(before["Building_1"], after["Building_1"], atol=1e-6)
    assert not torch.allclose(before["Building_2"], after["Building_2"])


def test_verified_fallback_does_not_train_on_a_solver_decision(tmp_path, monkeypatch):
    from algorithms.ti_marl.runtime.reservation import SiteReservation
    from algorithms.ti_marl.runtime.reservation_solver import ReservationUnavailable

    agent, snapshot = storage_site(tmp_path)

    def timeout(self, key):
        raise ReservationUnavailable("simulated timeout after initial witness")

    monkeypatch.setattr(SiteReservation, "admissible_modes", timeout)
    with torch.no_grad():
        evaluation = agent.actor(snapshot)
    trace = evaluation.traces["Building_1"]
    assert trace.supervised and not trace.choices
    assert not evaluation.bundles[0].decisions
    assert evaluation.executed[0].interventions[0]["reason"] == "reservation_supervised_fallback"
    agent.learner.rollout.add(
        PrefixStep(snapshot, evaluation.traces, 1.0, {}, {}, True, False, 900.0)
    )
    before = [p.detach().clone() for p in agent.actor.parameters()]
    assert agent.learner.update()["sampled_choices"] == 0.0
    assert all(torch.equal(x, y) for x, y in zip(before, agent.actor.parameters()))


def test_replaying_a_saved_prefix_never_reads_unsampled_future_choices(tmp_path):
    agent, snapshot = storage_site(tmp_path)
    with torch.no_grad():
        evaluation = agent.actor(snapshot, deterministic=True)
        site = "Building_1"
        trace = evaluation.traces[site]
        k = len(trace.order)
        partial_trace = replace(trace, choices=trace.choices[:k])
        full = agent.critic.prefix_values(snapshot, evaluation.traces)[site]
        partial = agent.critic.prefix_values(snapshot, {site: partial_trace})[site]
        assert torch.allclose(partial, full[: k + 1])


def test_constraint_messages_follow_weighted_equipment_memberships():
    from algorithms.ti_marl.policy.joint import ConstraintMessages

    torch.manual_seed(17)
    graph = ConstraintMessages(8, 2)
    contexts = {key: torch.randn(8) for key in ("A", "B", "C")}
    limit = LocalConstraint(
        "group",
        "site",
        "group_import_limit_kw",
        upper_bound=4.0,
        member_group_ids=("A", "B"),
        member_group_coefficients=(("A", 1.0), ("B", 0.5)),
    )
    baseline = graph(contexts, (limit,))
    changed = graph({**contexts, "B": -contexts["B"]}, (limit,))
    reweighted = graph(
        contexts, (replace(limit, member_group_coefficients=(("A", 1.0), ("B", 1.0))),)
    )
    assert not torch.allclose(baseline["A"], changed["A"])
    assert not torch.allclose(baseline["A"], reweighted["A"])
    assert torch.equal(changed["C"], contexts["C"])
    assert torch.equal(baseline["C"], contexts["C"])


def test_physical_density_scales_with_the_port_rating():
    from math import log

    from algorithms.ti_marl.policy.joint import Choice, JointActor

    beta = torch.distributions.Beta(2.0, 3.0)
    choice = Choice("A", "ev", "power", 1, lower=0.2, upper=0.8, scale_kw=2.0, unit_sample=0.3)
    before, entropy = JointActor._score(beta, choice, torch.zeros(()))
    after, wider_entropy = JointActor._score(beta, replace(choice, scale_kw=4.0), torch.zeros(()))
    assert (after - before).item() == pytest.approx(-log(2))
    assert (wider_entropy - entropy).item() == pytest.approx(log(2))
