from __future__ import annotations

from dataclasses import replace
import gzip
import json

from gymnasium import spaces
import numpy as np
import pytest
import torch

from algorithms.ti_marl.agent import TIMARL
from algorithms.ti_marl.compiler import TypedInterfaceCompiler
from algorithms.ti_marl.contracts.models import (
    ActionDecision,
    LocalActionBundle,
    TypedTransition,
)
from algorithms.ti_marl.learning.rollout import RolloutStep, TypedRolloutBuffer
from algorithms.ti_marl.policy.networks import CentralSetCritic, TypedActor, parameter_count
from algorithms.ti_marl.runtime import (
    AnalyticLocalProjector,
    BufferedTraceWriter,
    CityLearnTypedActionCodec,
)
from tests.ti_marl_fixtures import entity_payload, entity_specs


SCHEMA = "configs/ti_marl/agent_schema_v1.yaml"
REGISTRY = "configs/ti_marl/type_registry_v1.yaml"
HEALTH = "configs/ti_marl/health_rules_v1.yaml"


def compile_snapshot(buildings=("Building_1", "Building_2"), *, time_step=0, topology_version=0):
    compiler = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        agent_schema_path=SCHEMA,
        type_registry_path=REGISTRY,
        health_rules_path=HEALTH,
    )
    compiler.attach_entity_specs(entity_specs(buildings))
    return compiler, compiler.compile(
        entity_payload(buildings, time_step=time_step, topology_version=topology_version)
    )


def test_actor_and_critic_are_permutation_equivariant_and_cardinality_independent():
    compiler, snapshot = compile_snapshot()
    torch.manual_seed(3)
    actor = TypedActor(compiler.type_registry, d_model=32, attention_heads=4, relation_layers=1)
    critic = CentralSetCritic(compiler.type_registry, d_model=32, relation_layers=1)
    initial_parameters = parameter_count(actor) + parameter_count(critic)

    actor.eval()
    critic.eval()
    with torch.no_grad():
        baseline = actor(snapshot, deterministic=True)
        baseline_values = critic(snapshot)
        permuted = replace(snapshot, agent_ids=tuple(reversed(snapshot.agent_ids)))
        reordered = actor(permuted, deterministic=True)
        reordered_values = critic(permuted)

    baseline_bundles = {bundle.agent_id: bundle for bundle in baseline.bundles}
    reordered_bundles = {bundle.agent_id: bundle for bundle in reordered.bundles}
    assert baseline_bundles.keys() == reordered_bundles.keys()
    for agent_id in baseline_bundles:
        assert baseline_bundles[agent_id].decisions == reordered_bundles[agent_id].decisions
        assert torch.allclose(baseline_values[agent_id], reordered_values[agent_id], atol=1e-6)

    compiler.attach_entity_specs(entity_specs(("Building_1",)))
    smaller = compiler.compile(entity_payload(("Building_1",), topology_version=1))
    with torch.no_grad():
        assert set(actor(smaller, deterministic=True).latent_by_agent) == {"Building_1"}
        assert set(critic(smaller)) == {"Building_1"}
    assert parameter_count(actor) + parameter_count(critic) == initial_parameters


def test_actor_is_local_while_set_critic_observes_other_agents():
    compiler, snapshot = compile_snapshot()
    torch.manual_seed(5)
    actor = TypedActor(compiler.type_registry, d_model=32, attention_heads=4, relation_layers=1)
    critic = CentralSetCritic(compiler.type_registry, d_model=32, relation_layers=1)
    changed_parts = tuple(
        replace(part, values=tuple(value + 100.0 for value in part.values))
        if part.owner_agent_id == "Building_2" and part.source_entity_id == "Building_2"
        else part
        for part in snapshot.observation_parts
    )
    changed = replace(snapshot, observation_parts=changed_parts)

    actor.eval()
    critic.eval()
    with torch.no_grad():
        baseline_policy = actor(snapshot, deterministic=True)
        changed_policy = actor(changed, deterministic=True)
        baseline_value = critic(snapshot)["Building_1"]
        changed_value = critic(changed)["Building_1"]

    baseline_bundle = next(
        bundle for bundle in baseline_policy.bundles if bundle.agent_id == "Building_1"
    )
    changed_bundle = next(
        bundle for bundle in changed_policy.bundles if bundle.agent_id == "Building_1"
    )
    assert baseline_bundle.decisions == changed_bundle.decisions
    assert not torch.allclose(baseline_value, changed_value)


def test_local_projection_jointly_enforces_headroom_and_deferrable_deadline():
    _compiler, snapshot = compile_snapshot()
    groups = {group.group_type: group for group in snapshot.groups_for("Building_1")}
    raw = LocalActionBundle(
        agent_id="Building_1",
        decisions=(
            ActionDecision(groups["stationary_storage"].group_id, "CHARGE_STATIONARY", 1.0, 1),
            ActionDecision(groups["ev_session"].group_id, "CHARGE_EV", 1.0, 1),
            ActionDecision(groups["deferrable"].group_id, "IDLE", 0.0, 0),
        ),
    )
    projector = AnalyticLocalProjector()
    final = projector.project(snapshot, (raw,))[0]
    projector.assert_feasible(snapshot, (final,))
    decisions = {decision.group_id: decision for decision in final.decisions}
    charge_power = 0.0
    for group_type in ("stationary_storage", "ev_session"):
        group = groups[group_type]
        decision = decisions[group.group_id]
        port = next(item for item in group.ports if item.mode == decision.mode)
        charge_power += (
            group.max_charge_power_kw * decision.fraction * port.upper_bound
        )
    assert charge_power == pytest.approx(4.0)
    assert decisions[groups["deferrable"].group_id].mode == "START"
    assert any(item["reason"] == "deferrable_must_start" for item in final.interventions)


def test_codec_applies_a_dynamic_port_bound_exactly_once():
    _compiler, snapshot = compile_snapshot()
    ev_group = next(
        group
        for group in snapshot.groups_for("Building_1")
        if group.group_type == "ev_session"
    )
    bounded_ev_group = replace(
        ev_group,
        ports=tuple(
            replace(port, upper_bound=0.5)
            if port.mode == "CHARGE_EV"
            else port
            for port in ev_group.ports
        ),
    )
    snapshot = replace(
        snapshot,
        action_groups=tuple(
            bounded_ev_group if group.group_id == ev_group.group_id else group
            for group in snapshot.action_groups
        ),
        constraints=tuple(
            replace(constraint, upper_bound=100.0)
            if constraint.owner_agent_id == "Building_1"
            and constraint.constraint_type == "charging_headroom_kw"
            else constraint
            for constraint in snapshot.constraints
        ),
    )
    ev_group = bounded_ev_group
    bundle = LocalActionBundle(
        agent_id="Building_1",
        decisions=(
            ActionDecision(ev_group.group_id, "CHARGE_EV", 1.0, 1),
        ),
    )
    projector = AnalyticLocalProjector()
    projected = projector.project(snapshot, (bundle,))[0]
    assert projected.decisions[0].fraction == pytest.approx(1.0)

    codec = CityLearnTypedActionCodec()
    codec.attach(
        building_names=("Building_1", "Building_2"),
        action_names=(
            (
                "electrical_storage",
                "electric_vehicle_storage_charger_1",
                "deferrable_appliance_washer",
            ),
            ("electrical_storage",),
        ),
        action_space=(
            spaces.Box(low=np.asarray([-1.0, -1.0, 0.0]), high=np.ones(3)),
            spaces.Box(low=np.asarray([-1.0]), high=np.ones(1)),
        ),
    )
    commands = codec.encode(snapshot, (projected,))
    # The fixture exposes 0.5 available EV charge action.  Full policy intent
    # therefore becomes 0.5, not 0.25 (double contraction) or 1.0 (unbounded).
    assert commands[0][1] == pytest.approx(0.5)


def test_rollout_gae_handles_leave_and_does_not_create_predecessor_for_join():
    _c1, first = compile_snapshot(("Building_1", "Building_2"), time_step=0)
    _c2, second = compile_snapshot(("Building_1",), time_step=1, topology_version=1)
    _c3, third = compile_snapshot(("Building_1", "Building_3"), time_step=2, topology_version=2)
    buffer = TypedRolloutBuffer()
    buffer.add(
        RolloutStep(
            snapshot=first,
            next_snapshot=second,
            bundles=(),
            old_log_probs={"Building_1": 0.0, "Building_2": 0.0},
            values={"Building_1": 1.0, "Building_2": 2.0},
            next_values={"Building_1": 1.5},
            rewards={"Building_1": 1.0, "Building_2": 3.0},
            terminated_agent_ids=("Building_2",),
            truncated=False,
        )
    )
    buffer.add(
        RolloutStep(
            snapshot=second,
            next_snapshot=third,
            bundles=(),
            old_log_probs={"Building_1": 0.0},
            values={"Building_1": 1.5},
            next_values={"Building_1": 2.0, "Building_3": 7.0},
            rewards={"Building_1": 2.0},
            terminated_agent_ids=(),
            truncated=False,
        )
    )
    samples = {(item.step_index, item.agent_id): item for item in buffer.advantages(gamma=0.9, gae_lambda=0.95)}
    assert (0, "Building_2") in samples
    assert samples[(0, "Building_2")].advantage == pytest.approx(1.0)
    assert (1, "Building_3") not in samples
    assert samples[(0, "Building_1")].advantage != pytest.approx(1.0 + 0.9 * 1.5 - 1.0)


def test_buffered_trace_contains_every_referenced_snapshot_once(tmp_path):
    _compiler, first = compile_snapshot(time_step=0)
    _compiler, second = compile_snapshot(time_step=1)
    transition = TypedTransition(
        snapshot_hash=first.snapshot_hash,
        next_snapshot_hash=second.snapshot_hash,
        agent_ids=first.agent_ids,
        next_agent_ids=second.agent_ids,
        raw_bundles=(),
        final_bundles=(),
        commands=(),
        execution={"version": "entity_action_execution_v1"},
        rewards=(),
        reward_components={},
        terminated_agent_ids=(),
        bootstrap_agent_ids=first.agent_ids,
    )
    writer = BufferedTraceWriter(tmp_path, chunk_size=8, snapshot_interval=99)
    writer.record(first, second, transition)
    writer.close()
    records = []
    with gzip.open(next(tmp_path.glob("*.jsonl.gz")), "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    hashes = {row["hash"] for row in records if row["kind"] == "snapshot"}
    assert hashes == {first.snapshot_hash, second.snapshot_hash}
    assert sum(row["kind"] == "transition" for row in records) == 1


def agent_config(tmp_path):
    return {
        "algorithm": {
            "name": "TIMARL",
            "hyperparameters": {
                "contract_version": "ti_marl_v1",
                "agent_schema_path": SCHEMA,
                "type_registry_path": REGISTRY,
                "health_rules_path": HEALTH,
                "backbone": {"name": "mappo"},
                "actor": {"d_model": 32, "attention_heads": 4, "relation_layers": 1},
                "rollout_steps": 4,
                "trace": {"enabled": False},
            },
        },
        "training": {"seed": 9},
        "runtime": {"job_dir": str(tmp_path)},
    }


def attach_agent(agent, buildings):
    specs = entity_specs(buildings)
    names = []
    spaces_ = []
    for index, _building in enumerate(buildings):
        row = ["electrical_storage"]
        if index == 0:
            row += ["electric_vehicle_storage_charger_1", "deferrable_appliance_washer"]
        names.append(row)
        spaces_.append(spaces.Box(low=np.asarray([-1.0] * len(row)), high=np.asarray([1.0] * len(row))))
    agent.attach_environment(
        observation_names=[[] for _ in buildings],
        action_names=names,
        action_space=spaces_,
        observation_space=[spaces.Box(low=np.zeros(1), high=np.ones(1)) for _ in buildings],
        metadata={
            "interface": "entity",
            "topology_mode": "dynamic",
            "entity_specs": specs,
            "building_names": list(buildings),
        },
    )


def test_checkpoint_round_trip_accepts_a_different_compatible_composition(tmp_path):
    first = TIMARL(agent_config(tmp_path / "first"))
    attach_agent(first, ("Building_1", "Building_2"))
    first.set_entity_observation_context(observation_payload=entity_payload(), info={})
    first.predict([], deterministic=True)
    checkpoint = first.save_checkpoint(str(tmp_path / "checkpoint"), step=7)

    restored = TIMARL(agent_config(tmp_path / "restored"))
    attach_agent(restored, ("Building_1",))
    restored.load_checkpoint(checkpoint)
    assert restored.codec.building_names == ("Building_1",)
    assert restored._parameter_count == first._parameter_count
    restored.set_entity_observation_context(
        observation_payload=entity_payload(("Building_1",), topology_version=3),
        info={},
    )
    actions = restored.predict([], deterministic=True)
    assert len(actions) == 1
