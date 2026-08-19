from __future__ import annotations

from collections import Counter
from dataclasses import replace
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from gymnasium import spaces
import numpy as np
import pytest
import torch
import yaml

from algorithms.ti_marl.agent import TIMARL
from algorithms.ti_marl.compiler import TypedInterfaceCompiler
from algorithms.ti_marl.contracts.models import (
    ActionDecision,
    LocalConstraint,
    LocalActionBundle,
    ObservationPart,
    TypedTransition,
)
from algorithms.ti_marl.contracts.enums import HealthState
from algorithms.ti_marl.learning.mappo import TIMAPPO
from algorithms.ti_marl.learning.behavior_cloning import (
    TypedBehaviorCloningWarmStart,
)
from algorithms.ti_marl.learning.rollout import (
    AdvantageSample,
    RolloutStep,
    TypedRolloutBuffer,
)
from algorithms.ti_marl.policy.networks import (
    CentralSetCritic,
    LocalTypedCritic,
    TypedActor,
    TypedGroupCritic,
    parameter_count,
)
from algorithms.ti_marl.runtime import (
    AnalyticLocalProjector,
    BufferedTraceWriter,
    CityLearnTypedActionCodec,
    TypedCommandBuilder,
)
from tests.ti_marl_fixtures import entity_payload, entity_specs, write_typed_interfaces
from utils.artifact_manifest import build_manifest
from utils.bundle_validator import validate_bundle_contract


def compile_snapshot(
    tmp_path,
    buildings=("Building_1", "Building_2"),
    *,
    time_step=0,
    topology_version=0,
):
    interfaces = write_typed_interfaces(
        tmp_path / "interfaces",
        ("Building_1", "Building_2", "Building_3"),
    )
    compiler = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interfaces_dir=interfaces,
    )
    compiler.attach_entity_specs(entity_specs(buildings), seconds_per_time_step=900)
    return compiler, compiler.compile(
        entity_payload(buildings, time_step=time_step, topology_version=topology_version)
    )


def test_actor_and_critic_are_permutation_equivariant_and_cardinality_independent(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
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
        baseline_decisions = baseline_bundles[agent_id].decisions
        reordered_decisions = reordered_bundles[agent_id].decisions
        assert len(baseline_decisions) == len(reordered_decisions)
        for first, second in zip(baseline_decisions, reordered_decisions):
            assert first.group_id == second.group_id
            assert first.mode == second.mode
            assert first.mode_index == second.mode_index
            assert first.fraction == pytest.approx(second.fraction, abs=1.0e-6)
            assert first.raw_log_prob == pytest.approx(
                second.raw_log_prob, abs=1.0e-6
            )
        assert torch.allclose(baseline_values[agent_id], reordered_values[agent_id], atol=1e-6)

    compiler.attach_entity_specs(entity_specs(("Building_1",)))
    smaller = compiler.compile(entity_payload(("Building_1",), topology_version=1))
    with torch.no_grad():
        assert set(actor(smaller, deterministic=True).latent_by_agent) == {"Building_1"}
        assert set(critic(smaller)) == {"Building_1"}
    assert parameter_count(actor) + parameter_count(critic) == initial_parameters


def test_expected_signed_deterministic_decoder_uses_hybrid_policy_mean(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
        deterministic_mode_strategy="expected_signed",
    )
    group = next(
        group
        for group in snapshot.groups_for("Building_1")
        if group.group_type == "stationary_storage"
    )
    modes = actor.group_modes[group.group_type]
    logits = {"IDLE": 0.0, "CHARGE_STATIONARY": 1.0, "DISCHARGE_STATIONARY": -1.0}
    with torch.no_grad():
        actor.mode_heads[group.group_type].weight.zero_()
        actor.mode_heads[group.group_type].bias.copy_(
            torch.tensor([logits[mode] for mode in modes])
        )
        actor.beta_heads[group.group_type].weight.zero_()
        actor.beta_heads[group.group_type].bias.zero_()
        decision, _log_prob, _entropy = actor._group_decision(
            group,
            torch.zeros(actor.d_model),
            deterministic=True,
            expected=None,
            materialize_decision=True,
        )

    assert decision is not None
    assert decision.mode == "CHARGE_STATIONARY"
    probabilities = torch.softmax(
        torch.tensor([logits[mode] for mode in modes]), dim=0
    )
    expected_fraction = 0.5 * (
        probabilities[modes.index("CHARGE_STATIONARY")]
        - probabilities[modes.index("DISCHARGE_STATIONARY")]
    )
    assert decision.fraction == pytest.approx(float(expected_fraction), abs=1.0e-6)


def test_typed_group_critic_is_equivariant_and_cardinality_independent(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    torch.manual_seed(31)
    critic = TypedGroupCritic(
        compiler.type_registry,
        d_model=32,
        relation_layers=1,
        centralized=True,
    )
    initial_parameters = parameter_count(critic)

    with torch.no_grad():
        baseline = critic(snapshot)
        permuted = replace(snapshot, agent_ids=tuple(reversed(snapshot.agent_ids)))
        reordered = critic(permuted)
    assert baseline.keys() == reordered.keys()
    for agent_id, values_by_group in baseline.items():
        assert values_by_group.keys() == reordered[agent_id].keys()
        for group_id, value in values_by_group.items():
            assert torch.allclose(
                value, reordered[agent_id][group_id], atol=1.0e-6
            )

    compiler.attach_entity_specs(entity_specs(("Building_1",)))
    smaller = compiler.compile(
        entity_payload(("Building_1",), topology_version=1)
    )
    with torch.no_grad():
        assert set(critic(smaller)) == {"Building_1"}
    assert parameter_count(critic) == initial_parameters


def test_typed_encoder_distinguishes_load_pv_and_exact_observation_identity(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    parts = {
        part.observation_id: part
        for part in snapshot.parts_for("Building_1")
    }
    load = replace(parts["non_shiftable_load"], values=(2.0,))
    pv = replace(parts["solar_generation"], values=(2.0,))
    same_family_other_identity = replace(
        load,
        part_id=f"{load.part_id}:alternative",
        observation_id="alternative_load_signal",
        feature_names=("alternative_load_signal",),
    )
    assert load.semantic_type == "local_load"
    assert pv.semantic_type == "local_pv_generation"
    assert load.sensor_type == pv.sensor_type == "building_meter"

    torch.manual_seed(29)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    actor.eval()
    device = next(actor.parameters()).device

    def encoded(part):
        isolated = replace(snapshot, observation_parts=(part,))
        return actor.encoder(isolated, "Building_1", device)

    with torch.no_grad():
        load_latent = encoded(load)
        pv_latent = encoded(pv)
        alternative_latent = encoded(same_family_other_identity)
    assert not torch.allclose(load_latent, pv_latent)
    assert not torch.allclose(load_latent, alternative_latent)


def test_typed_encoder_is_invariant_to_sensor_and_observation_order(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    torch.manual_seed(31)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    actor.eval()
    reordered = replace(
        snapshot,
        observation_parts=tuple(reversed(snapshot.observation_parts)),
    )
    device = next(actor.parameters()).device
    with torch.no_grad():
        baseline = actor.encoder(snapshot, "Building_1", device)
        changed_order = actor.encoder(reordered, "Building_1", device)
    assert torch.allclose(baseline, changed_order, atol=1.0e-6)


def test_typed_encoder_rejects_unknown_semantic_types_safely(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    first = snapshot.observation_parts[0]
    invalid = replace(
        snapshot,
        observation_parts=(replace(first, semantic_type="unknown_new_signal"),),
    )
    with pytest.raises(ValueError, match="Unknown TI-MARL semantic type"):
        actor.encoder(invalid, first.owner_agent_id, next(actor.parameters()).device)


def test_known_charger_instance_can_be_added_without_resizing_the_model(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    initial_parameters = parameter_count(actor)
    charger_parts = tuple(
        part
        for part in snapshot.parts_for("Building_1")
        if part.sensor_type == "bidirectional_ev_charger"
    )
    second_charger_parts = tuple(
        replace(
            part,
            part_id=part.part_id.replace("charger_1", "charger_2"),
            sensor_id="charger_2",
            source_entity_id=part.source_entity_id.replace(
                "charger_1", "charger_2"
            ),
        )
        for part in charger_parts
    )
    ev_group = next(
        group
        for group in snapshot.groups_for("Building_1")
        if group.group_type == "ev_session"
    )
    second_group = replace(
        ev_group,
        group_id=ev_group.group_id.replace("charger_1", "charger_2"),
        module_id="charger_2",
        ports=tuple(
            replace(
                port,
                port_id=port.port_id.replace("charger_1", "charger_2"),
                target_entity_id=port.target_entity_id.replace(
                    "charger_1", "charger_2"
                ),
            )
            for port in ev_group.ports
        ),
    )
    expanded = replace(
        snapshot,
        observation_parts=snapshot.observation_parts + second_charger_parts,
        action_groups=snapshot.action_groups + (second_group,),
    )
    with torch.no_grad():
        result = actor(expanded, deterministic=True)
    building = next(
        bundle for bundle in result.bundles if bundle.agent_id == "Building_1"
    )
    assert len(building.decisions) == len(snapshot.groups_for("Building_1")) + 1
    assert parameter_count(actor) == initial_parameters


def test_actor_is_local_while_set_critic_observes_other_agents(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
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


def test_actor_replay_skips_bundle_materialization_without_changing_density(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    torch.manual_seed(17)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    sampled = actor(snapshot, deterministic=True)
    decisions = {
        bundle.agent_id: {
            decision.group_id: decision for decision in bundle.decisions
        }
        for bundle in sampled.bundles
    }

    replayed = actor(
        snapshot,
        decisions=decisions,
        materialize_bundles=False,
    )
    packed = actor.evaluate_actions_many(((snapshot, decisions),))

    assert replayed.bundles == ()
    for agent_id in snapshot.agent_ids:
        assert torch.equal(
            sampled.log_prob_by_agent[agent_id],
            replayed.log_prob_by_agent[agent_id],
        )
        assert torch.equal(
            sampled.entropy_by_agent[agent_id],
            replayed.entropy_by_agent[agent_id],
        )
        assert torch.allclose(
            sampled.log_prob_by_agent[agent_id],
            packed.log_prob_by_step[0][agent_id],
            atol=1.0e-6,
        )
        assert torch.allclose(
            sampled.entropy_by_agent[agent_id],
            packed.entropy_by_step[0][agent_id],
            atol=1.0e-6,
        )
        assert torch.allclose(
            packed.log_prob_by_step[0][agent_id],
            torch.stack(
                tuple(packed.log_prob_by_group_step[0][agent_id].values())
            ).sum(),
            atol=1.0e-6,
        )


def test_action_conditioned_actor_replays_group_densities_and_relations(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    torch.manual_seed(23)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
        group_context_kind="action_conditioned",
    )
    sampled = actor(snapshot, deterministic=True)
    decisions = {
        bundle.agent_id: {
            decision.group_id: decision for decision in bundle.decisions
        }
        for bundle in sampled.bundles
    }
    packed = actor.evaluate_actions_many(((snapshot, decisions),))

    for agent_id in snapshot.agent_ids:
        assert torch.allclose(
            sampled.log_prob_by_agent[agent_id],
            packed.log_prob_by_step[0][agent_id],
            atol=1.0e-6,
        )
    group = snapshot.action_groups[0]
    own_part = next(
        part
        for part in snapshot.parts_for(group.owner_agent_id)
        if part.sensor_id == group.module_id
    )
    community_part = next(
        part
        for part in snapshot.parts_for(group.owner_agent_id)
        if part.scope == "community"
    )
    assert actor._group_part_relation(group, own_part) == 0
    assert actor._group_part_relation(group, community_part) == 3


def test_behavior_cloning_balances_rare_action_modes_per_group_type(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    warm_start = TypedBehaviorCloningWarmStart(
        demonstration_episodes=1,
        max_samples=16,
        pretraining_epochs=1,
        batch_size=4,
        learning_rate=1.0e-4,
        balance_action_modes=True,
        mode_balance_exponent=0.5,
        max_mode_weight=3.0,
        seed=7,
    )
    target_group = next(
        group
        for group in snapshot.action_groups
        if any(port.valid and port.mode != "IDLE" for port in group.ports)
    )
    target_mode = next(
        port.mode
        for port in target_group.ports
        if port.valid and port.mode != "IDLE"
    )
    for sample_index in range(10):
        bundles = []
        for agent_id in snapshot.agent_ids:
            decisions = []
            for group in snapshot.groups_for(agent_id):
                mode = (
                    target_mode
                    if sample_index == 9 and group.group_id == target_group.group_id
                    else "IDLE"
                )
                decisions.append(
                    ActionDecision(
                        group_id=group.group_id,
                        mode=mode,
                        fraction=0.5 if mode != "IDLE" else 0.0,
                        mode_index=actor.group_modes[group.group_type].index(mode),
                    )
                )
            bundles.append(LocalActionBundle(agent_id, tuple(decisions)))
        warm_start.record(snapshot, bundles)

    weights = warm_start._build_mode_weights()
    assert weights[(target_group.group_type, target_mode)] > weights[
        (target_group.group_type, "IDLE")
    ]
    assert weights[(target_group.group_type, target_mode)] <= 3.0


def test_behavior_cloning_hierarchical_loss_balances_modes_and_group_types():
    losses = [
        ("stationary_storage", "IDLE", torch.tensor(1.0)),
        ("stationary_storage", "IDLE", torch.tensor(3.0)),
        ("stationary_storage", "CHARGE_STATIONARY", torch.tensor(10.0)),
        ("deferrable", "IDLE", torch.tensor(2.0)),
        ("deferrable", "START", torch.tensor(6.0)),
    ]

    reduced = TypedBehaviorCloningWarmStart._reduce_losses(
        losses,
        loss_kind="hierarchical_mode_mean",
    )

    # storage: mean(mean(1, 3), 10) = 6; deferrable: mean(2, 6) = 4.
    assert reduced.item() == pytest.approx(5.0)


def test_action_mode_diagnostics_separate_actor_from_feasibility(tmp_path):
    _compiler, snapshot = compile_snapshot(tmp_path)
    raw_bundles = []
    final_bundles = []
    changed_group = next(
        group
        for group in snapshot.action_groups
        if any(port.valid and port.mode != "IDLE" for port in group.ports)
    )
    changed_mode = next(
        port.mode
        for port in changed_group.ports
        if port.valid and port.mode != "IDLE"
    )
    for agent_id in snapshot.agent_ids:
        raw = []
        final = []
        for group in snapshot.groups_for(agent_id):
            raw.append(ActionDecision(group.group_id, "IDLE", 0.0, 0))
            final.append(
                ActionDecision(
                    group.group_id,
                    changed_mode if group.group_id == changed_group.group_id else "IDLE",
                    1.0 if group.group_id == changed_group.group_id else 0.0,
                    1 if group.group_id == changed_group.group_id else 0,
                )
            )
        raw_bundles.append(LocalActionBundle(agent_id, tuple(raw)))
        final_bundles.append(LocalActionBundle(agent_id, tuple(final)))

    agent = TIMARL.__new__(TIMARL)
    agent._current_snapshot = snapshot
    agent._episode_raw_modes = Counter()
    agent._episode_final_modes = Counter()
    metrics = agent._action_mode_diagnostics(raw_bundles, final_bundles)
    group_type = changed_group.group_type.lower()

    assert metrics[f"TI_MARL/raw_non_idle_rate_{group_type}"] == 0.0
    assert metrics[f"TI_MARL/final_non_idle_rate_{group_type}"] > 0.0


def test_behavior_cloning_mode_counts_follow_the_retained_reservoir(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    warm_start = TypedBehaviorCloningWarmStart(
        demonstration_episodes=1,
        max_samples=1,
        pretraining_epochs=1,
        batch_size=1,
        learning_rate=1.0e-4,
        balance_action_modes=True,
        mode_balance_exponent=0.5,
        max_mode_weight=3.0,
        seed=1,
        calibration_epochs=1,
        calibration_learning_rate=5.0e-5,
    )
    target_group = next(
        group
        for group in snapshot.action_groups
        if any(port.valid and port.mode != "IDLE" for port in group.ports)
    )
    target_mode = next(
        port.mode
        for port in target_group.ports
        if port.valid and port.mode != "IDLE"
    )

    def bundles_with_target(mode):
        return tuple(
            LocalActionBundle(
                agent_id,
                tuple(
                    ActionDecision(
                        group.group_id,
                        mode if group.group_id == target_group.group_id else "IDLE",
                        0.5 if group.group_id == target_group.group_id else 0.0,
                        (
                            actor.group_modes[group.group_type].index(mode)
                            if group.group_id == target_group.group_id
                            else 0
                        ),
                    )
                    for group in snapshot.groups_for(agent_id)
                ),
            )
            for agent_id in snapshot.agent_ids
        )

    warm_start.record(snapshot, bundles_with_target("IDLE"))
    warm_start.record(snapshot, bundles_with_target(target_mode))

    assert warm_start.seen_samples == 2
    assert len(warm_start._demonstrations) == 1
    assert warm_start.mode_counts[(target_group.group_type, target_mode)] == 1
    retained_group_type_count = sum(
        group.group_type == target_group.group_type
        for group in snapshot.action_groups
    )
    assert warm_start.mode_counts[(target_group.group_type, "IDLE")] == (
        retained_group_type_count - 1
    )
    metrics = warm_start.pretrain(actor, max_grad_norm=0.5)
    assert metrics["bc_balanced_batches"] == 1.0
    assert metrics["bc_calibration_batches"] == 1.0
    assert metrics["bc_calibration_loss"] > 0.0
    target_prefix = (
        f"bc_mode_{target_group.group_type.lower()}_{target_mode.lower()}"
    )
    assert 0.0 <= metrics[f"{target_prefix}_recall"] <= 1.0
    assert 0.0 < metrics[f"{target_prefix}_target_probability"] <= 1.0
    predicted_count = sum(
        value
        for key, value in metrics.items()
        if key.startswith(f"bc_mode_{target_group.group_type.lower()}_")
        and key.endswith("_predicted_count")
    )
    assert predicted_count == float(retained_group_type_count)


@pytest.mark.parametrize("critic_class", [LocalTypedCritic, CentralSetCritic])
def test_packed_critic_matches_individual_snapshots(tmp_path, critic_class):
    compiler, first = compile_snapshot(tmp_path / "first", time_step=0)
    _compiler, second = compile_snapshot(tmp_path / "second", time_step=1)
    torch.manual_seed(19)
    critic = critic_class(
        compiler.type_registry,
        d_model=32,
        relation_layers=1,
    )

    individual = (critic(first), critic(second))
    packed = critic.forward_many((first, second))

    for individual_step, packed_step in zip(individual, packed):
        assert individual_step.keys() == packed_step.keys()
        for agent_id in individual_step:
            assert torch.allclose(
                individual_step[agent_id], packed_step[agent_id], atol=1.0e-6
            )


def test_local_critic_is_independent_of_other_agents_and_cardinality(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    torch.manual_seed(7)
    critic = LocalTypedCritic(
        compiler.type_registry,
        d_model=32,
        relation_layers=1,
    )
    initial_parameters = parameter_count(critic)
    changed = replace(
        snapshot,
        observation_parts=tuple(
            replace(part, values=tuple(value + 100.0 for value in part.values))
            if part.owner_agent_id == "Building_2"
            else part
            for part in snapshot.observation_parts
        ),
    )
    only_first = replace(
        snapshot,
        agent_ids=("Building_1",),
        observation_parts=tuple(
            part
            for part in snapshot.observation_parts
            if part.owner_agent_id == "Building_1"
        ),
        action_groups=tuple(
            group
            for group in snapshot.action_groups
            if group.owner_agent_id == "Building_1"
        ),
    )

    critic.eval()
    with torch.no_grad():
        baseline = critic(snapshot)["Building_1"]
        other_changed = critic(changed)["Building_1"]
        smaller = critic(only_first)["Building_1"]

    assert torch.equal(baseline, other_changed)
    assert torch.allclose(baseline, smaller, atol=1.0e-6)
    assert parameter_count(critic) == initial_parameters


def test_trace_only_observations_are_auditable_but_never_enter_the_actor(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    trace_part = next(
        part
        for part in snapshot.parts_for("Building_1")
        if part.observation_id == "topology_version"
    )
    assert trace_part.use == "trace_only"
    assert not trace_part.policy_input
    changed = replace(
        snapshot,
        observation_parts=tuple(
            replace(part, values=(9999.0,))
            if part.part_id == trace_part.part_id
            else part
            for part in snapshot.observation_parts
        ),
    )
    torch.manual_seed(6)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    actor.eval()
    with torch.no_grad():
        baseline = actor(snapshot, deterministic=True)
        modified = actor(changed, deterministic=True)
    assert torch.equal(
        baseline.latent_by_agent["Building_1"],
        modified.latent_by_agent["Building_1"],
    )


def test_local_projection_jointly_enforces_headroom_and_deferrable_deadline(tmp_path):
    _compiler, snapshot = compile_snapshot(tmp_path)
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


def test_deferrable_service_margin_uses_physical_time(tmp_path):
    _compiler, snapshot = compile_snapshot(tmp_path)
    group = next(
        item
        for item in snapshot.groups_for("Building_1")
        if item.group_type == "deferrable"
    )
    snapshot = replace(
        snapshot,
        observation_parts=tuple(
            replace(part, values=(3.0,))
            if part.sensor_id == group.module_id
            and part.observation_id == "slack_steps"
            else part
            for part in snapshot.observation_parts
        ),
    )
    raw = LocalActionBundle(
        agent_id="Building_1",
        decisions=(ActionDecision(group.group_id, "IDLE", 0.0, 0),),
    )

    default = AnalyticLocalProjector()
    default.set_seconds_per_time_step(900.0)
    assert default.project(snapshot, (raw,))[0].decisions[0].mode == "IDLE"

    guarded = AnalyticLocalProjector(
        deferrable_service_margin_seconds=3600.0,
    )
    guarded.set_seconds_per_time_step(900.0)
    final = guarded.project(snapshot, (raw,))[0]

    assert final.decisions[0].mode == "START"
    assert any(
        item["reason"] == "deferrable_service_margin_start"
        for item in final.interventions
    )
    assert guarded.configuration()[
        "deferrable_service_margin_seconds"
    ] == pytest.approx(3600.0)


def test_compiler_keeps_total_and_phase_headroom_separate(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    groups = {group.group_type: group for group in snapshot.groups_for("Building_1")}
    parts = []
    for observation_id, value in (
        ("charging_building_headroom_kw", 8.0),
        ("charging_phase_L1_headroom_kw", 3.0),
        ("charging_phase_L2_headroom_kw", 5.0),
        ("charging_phase_L3_headroom_kw", 4.0),
    ):
        parts.append(
            ObservationPart(
                part_id=f"Building_1:self.grid.{observation_id}",
                owner_agent_id="Building_1",
                source_entity_id="Building_1",
                semantic_type="grid_headroom",
                feature_names=(observation_id,),
                values=(value,),
                health=HealthState.HEALTHY,
                sensor_id="self",
                channel_id="grid",
                observation_id=observation_id,
                unit="kW",
                scope="local",
                use="runtime_bound",
                policy_input=True,
                criticality="safety",
            )
        )
    parts.append(
        ObservationPart(
            part_id=f"Building_1:{groups['ev_session'].module_id}.execution_feedback.last_applied_power_kw",
            owner_agent_id="Building_1",
            source_entity_id=groups["ev_session"].module_id,
            semantic_type="execution_feedback",
            feature_names=("last_applied_power_kw",),
            values=(4.0,),
            health=HealthState.HEALTHY,
            sensor_id=groups["ev_session"].module_id,
            channel_id="execution_feedback",
            observation_id="last_applied_power_kw",
            unit="kW",
            scope="local",
            use="runtime_bound",
            policy_input=True,
            criticality="safety",
        )
    )
    for module_id, active_phases in (
        (groups["ev_session"].module_id, ("L1",)),
        (groups["stationary_storage"].module_id, ("L1", "L2", "L3")),
    ):
        for phase in ("L1", "L2", "L3"):
            observation_id = f"phase_connection_{phase}"
            parts.append(
                ObservationPart(
                    part_id=f"Building_1:{module_id}.connection.{observation_id}",
                    owner_agent_id="Building_1",
                    source_entity_id=module_id,
                    semantic_type="phase_connection",
                    feature_names=(observation_id,),
                    values=(1.0 if phase in active_phases else 0.0,),
                    health=HealthState.HEALTHY,
                    sensor_id=module_id,
                    channel_id="connection",
                    observation_id=observation_id,
                    unit="scalar",
                    scope="local",
                    use="runtime_bound",
                    policy_input=True,
                    criticality="safety",
                )
            )

    assert compiler._headroom(parts, "Building_1", export=False) == pytest.approx(8.0)
    assert compiler._phase_headrooms(
        parts,
        "Building_1",
        export=False,
    ) == {"L1": 3.0, "L2": 5.0, "L3": 4.0}
    weights = compiler._group_phase_weights(
        parts,
        "Building_1",
        snapshot.action_groups,
    )
    assert weights[groups["ev_session"].group_id] == {"L1": 1.0}
    assert weights[groups["stationary_storage"].group_id] == {
        "L1": pytest.approx(1.0 / 3.0),
        "L2": pytest.approx(1.0 / 3.0),
        "L3": pytest.approx(1.0 / 3.0),
    }
    assert compiler._current_direction_power(
        parts,
        "Building_1",
        snapshot.action_groups,
        export=False,
    ) == {groups["ev_session"].group_id: 4.0}
    constraints = compiler._constraints(
        SimpleNamespace(active_agent_ids=("Building_1",)),
        {},
        snapshot.action_groups,
        parts,
    )
    by_id = {item.constraint_id: item for item in constraints}
    assert by_id["Building_1:charging_headroom_kw"].upper_bound == pytest.approx(12.0)
    assert by_id[
        "Building_1:charging_phase_L1_headroom_kw"
    ].upper_bound == pytest.approx(7.0)
    assert by_id[
        "Building_1:charging_phase_L2_headroom_kw"
    ].upper_bound == pytest.approx(5.0)
    assert by_id[
        "Building_1:charging_phase_L3_headroom_kw"
    ].upper_bound == pytest.approx(4.0)


def test_phase_projection_does_not_apply_unrelated_weakest_phase_globally(tmp_path):
    _compiler, snapshot = compile_snapshot(tmp_path)
    groups = {group.group_type: group for group in snapshot.groups_for("Building_1")}
    battery_id = groups["stationary_storage"].group_id
    ev_id = groups["ev_session"].group_id
    constraints = (
        LocalConstraint(
            constraint_id="Building_1:charging_headroom_kw",
            owner_agent_id="Building_1",
            constraint_type="charging_headroom_kw",
            upper_bound=12.0,
            member_group_ids=(battery_id, ev_id),
        ),
        LocalConstraint(
            constraint_id="Building_1:charging_phase_L1_headroom_kw",
            owner_agent_id="Building_1",
            constraint_type="charging_phase_headroom_kw",
            upper_bound=7.0,
            member_group_ids=(battery_id, ev_id),
            member_group_coefficients=((battery_id, 1.0 / 3.0), (ev_id, 1.0)),
        ),
        LocalConstraint(
            constraint_id="Building_1:charging_phase_L2_headroom_kw",
            owner_agent_id="Building_1",
            constraint_type="charging_phase_headroom_kw",
            upper_bound=5.0,
            member_group_ids=(battery_id,),
            member_group_coefficients=((battery_id, 1.0 / 3.0),),
        ),
        LocalConstraint(
            constraint_id="Building_1:charging_phase_L3_headroom_kw",
            owner_agent_id="Building_1",
            constraint_type="charging_phase_headroom_kw",
            upper_bound=4.0,
            member_group_ids=(battery_id,),
            member_group_coefficients=((battery_id, 1.0 / 3.0),),
        ),
    )
    snapshot = replace(snapshot, constraints=constraints)
    raw = LocalActionBundle(
        agent_id="Building_1",
        decisions=(
            ActionDecision(battery_id, "CHARGE_STATIONARY", 1.0, 1),
            ActionDecision(ev_id, "CHARGE_EV", 1.0, 1),
        ),
    )
    projector = AnalyticLocalProjector(enforce_ev_service=False)
    final = projector.project(snapshot, (raw,))[0]
    projector.assert_feasible(snapshot, (final,))
    decisions = {item.group_id: item for item in final.decisions}
    total_power = 0.0
    for group_id in (battery_id, ev_id):
        group = next(item for item in snapshot.action_groups if item.group_id == group_id)
        decision = decisions[group_id]
        port = next(item for item in group.ports if item.mode == decision.mode)
        total_power += group.max_charge_power_kw * decision.fraction * port.upper_bound
    assert total_power > 4.0
    assert total_power <= 12.0 + 1.0e-6


def _snapshot_with_ev_service_requirement(
    snapshot,
    ev_group,
    *,
    required_average_power_kw: float,
    hours_until_departure: float,
    efficiency: float,
    headroom_kw: float,
):
    parts = list(snapshot.observation_parts)
    for observation_id, value, unit in (
        ("hours_until_departure", hours_until_departure, "h"),
        ("required_average_power_kw", required_average_power_kw, "kW"),
        ("charge_efficiency_at_max_ratio", efficiency, "fraction"),
    ):
        parts.append(
            ObservationPart(
                part_id=f"Building_1:{ev_group.module_id}.service.{observation_id}",
                owner_agent_id="Building_1",
                source_entity_id=ev_group.adapter_target_entity_id or ev_group.module_id,
                semantic_type="ev_service",
                feature_names=(observation_id,),
                values=(float(value),),
                health=HealthState.HEALTHY,
                sensor_id=ev_group.module_id,
                channel_id="service",
                observation_id=observation_id,
                unit=unit,
                scope="local",
                use="safety_dependency",
                policy_input=True,
                criticality="service",
            )
        )
    return replace(
        snapshot,
        observation_parts=tuple(parts),
        constraints=tuple(
            replace(constraint, upper_bound=headroom_kw)
            if constraint.owner_agent_id == "Building_1"
            and constraint.constraint_type == "charging_headroom_kw"
            else constraint
            for constraint in snapshot.constraints
        ),
    )


def test_ev_service_floor_precedes_discretionary_charge(tmp_path):
    _compiler, snapshot = compile_snapshot(tmp_path)
    groups = {group.group_type: group for group in snapshot.groups_for("Building_1")}
    snapshot = _snapshot_with_ev_service_requirement(
        snapshot,
        groups["ev_session"],
        required_average_power_kw=2.0,
        hours_until_departure=2.0,
        efficiency=0.8,
        headroom_kw=3.0,
    )
    raw = LocalActionBundle(
        agent_id="Building_1",
        decisions=(
            ActionDecision(
                groups["stationary_storage"].group_id,
                "CHARGE_STATIONARY",
                1.0,
                1,
            ),
            ActionDecision(groups["ev_session"].group_id, "IDLE", 0.0, 0),
            ActionDecision(groups["deferrable"].group_id, "IDLE", 0.0, 0),
        ),
    )
    projector = AnalyticLocalProjector(ev_service_margin_ratio=0.0)
    final = projector.project(snapshot, (raw,))[0]
    projector.assert_feasible(snapshot, (final,))
    decisions = {item.group_id: item for item in final.decisions}

    def power(group_type):
        group = groups[group_type]
        decision = decisions[group.group_id]
        port = next(item for item in group.ports if item.mode == decision.mode)
        return group.max_charge_power_kw * decision.fraction * port.upper_bound

    assert decisions[groups["ev_session"].group_id].mode == "CHARGE_EV"
    assert power("ev_session") == pytest.approx(2.5)
    assert power("stationary_storage") == pytest.approx(0.5)
    assert any(
        item["reason"] == "ev_service_minimum_charge"
        for item in final.interventions
    )


def test_ev_service_floor_records_insufficient_local_headroom(tmp_path):
    _compiler, snapshot = compile_snapshot(tmp_path)
    groups = {group.group_type: group for group in snapshot.groups_for("Building_1")}
    snapshot = _snapshot_with_ev_service_requirement(
        snapshot,
        groups["ev_session"],
        required_average_power_kw=2.0,
        hours_until_departure=1.0,
        efficiency=0.8,
        headroom_kw=1.0,
    )
    raw = LocalActionBundle(
        agent_id="Building_1",
        decisions=(
            ActionDecision(
                groups["stationary_storage"].group_id,
                "CHARGE_STATIONARY",
                1.0,
                1,
            ),
            ActionDecision(groups["ev_session"].group_id, "IDLE", 0.0, 0),
            ActionDecision(groups["deferrable"].group_id, "IDLE", 0.0, 0),
        ),
    )
    projector = AnalyticLocalProjector(ev_service_margin_ratio=0.0)
    final = projector.project(snapshot, (raw,))[0]
    projector.assert_feasible(snapshot, (final,))
    decisions = {item.group_id: item for item in final.decisions}
    ev = decisions[groups["ev_session"].group_id]
    ev_port = next(item for item in groups["ev_session"].ports if item.mode == ev.mode)
    ev_power = (
        groups["ev_session"].max_charge_power_kw
        * ev.fraction
        * ev_port.upper_bound
    )
    assert ev_power == pytest.approx(1.0)
    assert decisions[groups["stationary_storage"].group_id].fraction == pytest.approx(0.0)
    assert any(
        item["reason"] == "ev_service_headroom_limited"
        for item in final.interventions
    )


def test_projector_can_reserve_headroom_for_next_step_base_load(tmp_path):
    _compiler, snapshot = compile_snapshot(tmp_path)
    groups = {group.group_type: group for group in snapshot.groups_for("Building_1")}
    snapshot = _snapshot_with_ev_service_requirement(
        snapshot,
        groups["ev_session"],
        required_average_power_kw=0.0,
        hours_until_departure=2.0,
        efficiency=1.0,
        headroom_kw=3.0,
    )
    raw = LocalActionBundle(
        agent_id="Building_1",
        decisions=(
            ActionDecision(
                groups["stationary_storage"].group_id,
                "CHARGE_STATIONARY",
                1.0,
                1,
            ),
            ActionDecision(groups["ev_session"].group_id, "IDLE", 0.0, 0),
            ActionDecision(groups["deferrable"].group_id, "IDLE", 0.0, 0),
        ),
    )
    projector = AnalyticLocalProjector(
        enforce_ev_service=False,
        headroom_reserve_kw=0.25,
    )
    final = projector.project(snapshot, (raw,))[0]
    projector.assert_feasible(snapshot, (final,))
    battery = next(
        item
        for item in final.decisions
        if item.group_id == groups["stationary_storage"].group_id
    )
    port = next(
        item
        for item in groups["stationary_storage"].ports
        if item.mode == battery.mode
    )
    power_kw = (
        groups["stationary_storage"].max_charge_power_kw
        * battery.fraction
        * port.upper_bound
    )
    assert power_kw == pytest.approx(2.75)
    assert projector.configuration()["headroom_reserve_kw"] == pytest.approx(0.25)


def test_projector_defers_binary_deferrable_start_when_headroom_is_insufficient(
    tmp_path,
):
    _compiler, snapshot = compile_snapshot(tmp_path)
    groups = {group.group_type: group for group in snapshot.groups_for("Building_1")}
    deferrable = replace(groups["deferrable"], activation_power_kw=4.0)
    snapshot = replace(
        snapshot,
        action_groups=tuple(
            deferrable if group.group_id == deferrable.group_id else group
            for group in snapshot.action_groups
        ),
        constraints=tuple(
            replace(
                constraint,
                upper_bound=3.0,
                member_group_ids=(deferrable.group_id,),
            )
            if constraint.owner_agent_id == "Building_1"
            and constraint.constraint_type == "charging_headroom_kw"
            else constraint
            for constraint in snapshot.constraints
        ),
    )
    raw = LocalActionBundle(
        agent_id="Building_1",
        decisions=(
            ActionDecision(
                groups["stationary_storage"].group_id,
                "IDLE",
                0.0,
                0,
            ),
            ActionDecision(groups["ev_session"].group_id, "IDLE", 0.0, 0),
            ActionDecision(deferrable.group_id, "START", 1.0, 1),
        ),
    )
    projector = AnalyticLocalProjector(enforce_ev_service=False)
    final = projector.project(snapshot, (raw,))[0]
    projector.assert_feasible(snapshot, (final,))
    start = next(
        item for item in final.decisions if item.group_id == deferrable.group_id
    )
    assert start.mode == "IDLE"
    assert start.fraction == pytest.approx(0.0)
    assert any(
        item["reason"] == "deferrable_headroom_limited"
        for item in final.interventions
    )


def test_just_in_time_ev_service_uses_laxity_before_forcing_charge(tmp_path):
    _compiler, snapshot = compile_snapshot(tmp_path)
    groups = {group.group_type: group for group in snapshot.groups_for("Building_1")}
    ev_group = groups["ev_session"]
    snapshot = _snapshot_with_ev_service_requirement(
        snapshot,
        ev_group,
        required_average_power_kw=2.0,
        hours_until_departure=4.0,
        efficiency=1.0,
        headroom_kw=20.0,
    )
    state_parts = []
    for observation_id, value, unit in (
        ("connected_ev_soc", 0.5, "fraction"),
        ("connected_ev_required_soc_departure", 0.8, "fraction"),
        ("connected_ev_battery_capacity_kwh", 60.0, "kWh"),
    ):
        state_parts.append(
            ObservationPart(
                part_id=f"Building_1:{ev_group.module_id}.ev_state.{observation_id}",
                owner_agent_id="Building_1",
                source_entity_id=ev_group.adapter_target_entity_id or ev_group.module_id,
                semantic_type="ev_service",
                feature_names=(observation_id,),
                values=(value,),
                health=HealthState.HEALTHY,
                sensor_id=ev_group.module_id,
                channel_id="ev_state",
                observation_id=observation_id,
                unit=unit,
                scope="local",
                use="safety_dependency",
                policy_input=True,
                criticality="service",
            )
        )
    snapshot = replace(
        snapshot,
        observation_parts=snapshot.observation_parts + tuple(state_parts),
    )
    idle = LocalActionBundle(
        agent_id="Building_1",
        decisions=(ActionDecision(ev_group.group_id, "IDLE", 0.0, 0),),
    )
    projector = AnalyticLocalProjector(
        ev_service_strategy="just_in_time",
        ev_service_margin_ratio=0.0,
        ev_service_tolerance_ratio=0.05,
    )
    projector.set_seconds_per_time_step(900.0)
    with_slack = projector.project(snapshot, (idle,))[0]
    assert with_slack.decisions[0].mode == "IDLE"

    minimum_average = AnalyticLocalProjector(
        ev_service_strategy="minimum_average",
        ev_service_margin_ratio=0.0,
        ev_service_tolerance_ratio=0.05,
    ).project(snapshot, (idle,))[0]
    minimum_decision = minimum_average.decisions[0]
    minimum_port = next(
        item for item in ev_group.ports if item.mode == minimum_decision.mode
    )
    minimum_power = (
        ev_group.max_charge_power_kw
        * minimum_decision.fraction
        * minimum_port.upper_bound
    )
    assert minimum_power == pytest.approx(3.75)

    urgent_parts = tuple(
        replace(part, values=(0.5,))
        if part.observation_id == "hours_until_departure"
        else part
        for part in snapshot.observation_parts
    )
    urgent = projector.project(
        replace(snapshot, observation_parts=urgent_parts),
        (idle,),
    )[0]
    assert urgent.decisions[0].mode == "CHARGE_EV"
    assert urgent.decisions[0].fraction == pytest.approx(1.0)
    assert any(
        item["reason"] == "ev_service_capacity_limited"
        for item in urgent.interventions
    )


def test_mappo_value_loss_normalizes_large_targets_and_keeps_raw_diagnostic():
    learner = TIMAPPO(
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1),
        normalize_value_targets=True,
        value_target_scale_floor=1.0,
        critic_loss="huber",
    )
    predictions = torch.zeros(3, requires_grad=True)
    targets = torch.tensor([-1000.0, 0.0, 1000.0])

    loss, mean, scale, raw_mse = learner._value_loss(predictions, targets)

    assert mean == pytest.approx(0.0)
    assert scale == pytest.approx(float(targets.std(unbiased=False)))
    assert loss < 1.0
    assert raw_mse == pytest.approx(2_000_000.0 / 3.0)
    loss.backward()
    assert torch.isfinite(predictions.grad).all()


def test_mappo_can_normalize_advantages_per_agent_without_scale_leakage():
    learner = TIMAPPO(
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1),
        advantage_normalization="per_agent",
    )
    samples = (
        AdvantageSample(0, "small", 1.0, 1.0),
        AdvantageSample(1, "small", 3.0, 3.0),
        AdvantageSample(0, "large", 100.0, 100.0),
        AdvantageSample(1, "large", 300.0, 300.0),
    )

    normalized = learner._normalize_advantages(samples)

    assert normalized == pytest.approx((-1.0, 1.0, -1.0, 1.0))


@pytest.mark.parametrize("critic_kind", ["local", "set"])
@pytest.mark.parametrize(
    "policy_credit_assignment", ["joint_agent", "typed_group"]
)
def test_ti_ppo_update_reports_finite_ratio_and_value_diagnostics(
    tmp_path,
    critic_kind,
    policy_credit_assignment,
):
    compiler, first = compile_snapshot(tmp_path / "first", time_step=0)
    _compiler, second = compile_snapshot(tmp_path / "second", time_step=1)
    torch.manual_seed(13)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    critic_class = LocalTypedCritic if critic_kind == "local" else CentralSetCritic
    critic = critic_class(
        compiler.type_registry,
        d_model=32,
        relation_layers=1,
    )
    group_critic = (
        TypedGroupCritic(
            compiler.type_registry,
            d_model=32,
            relation_layers=1,
            centralized=critic_kind == "set",
        )
        if policy_credit_assignment == "typed_group"
        else None
    )
    learner = TIMAPPO(
        actor,
        critic,
        group_critic=group_critic,
        rollout_steps=1,
        ppo_epochs=2,
        target_kl=None,
        advantage_normalization="per_agent",
        entropy_coeff_by_group_type={"stationary_storage": 0.05},
        policy_credit_assignment=policy_credit_assignment,
        policy_anchor_coeff=0.1,
    )
    with torch.no_grad():
        evaluation = actor(first, deterministic=True)
        values = critic(first)
        next_values = critic(second)
    learner.rollout.add(
        RolloutStep(
            snapshot=first,
            next_snapshot=second,
            bundles=evaluation.bundles,
            old_log_probs={
                key: float(value) for key, value in evaluation.log_prob_by_agent.items()
            },
            values={key: float(value) for key, value in values.items()},
            next_values={key: float(value) for key, value in next_values.items()},
            rewards={key: 1.0 for key in first.agent_ids},
            terminated_agent_ids=first.agent_ids,
            truncated=False,
            reward_components_by_agent={
                agent_id: {
                    "battery_safety_penalty": 0.0,
                    "ev_service_penalty": 2.0,
                    "deferrable_service_penalty": 0.0,
                }
                for agent_id in first.agent_ids
            },
            group_values=(
                {}
                if group_critic is None
                else {
                    agent_id: {
                        group_id: float(value)
                        for group_id, value in values_by_group.items()
                    }
                    for agent_id, values_by_group in group_critic(first).items()
                }
            ),
            next_group_values=(
                {}
                if group_critic is None
                else {
                    agent_id: {
                        group_id: float(value)
                        for group_id, value in values_by_group.items()
                    }
                    for agent_id, values_by_group in group_critic(second).items()
                }
            ),
        )
    )

    metrics = learner.update()

    for name in (
        "actor_loss",
        "critic_loss",
        "approx_kl",
        "clip_fraction",
        "ratio_error_max",
        "explained_variance",
        "actor_grad_norm",
        "policy_anchor_loss",
        "policy_anchor_coeff",
        "critic_grad_norm",
        "update_seconds",
        "evaluated_samples_per_second",
    ):
        assert np.isfinite(metrics[name]), name
    assert metrics["approx_kl"] >= -1.0e-7
    assert metrics["policy_anchor_loss"] >= 0.0
    assert metrics["policy_anchor_coeff"] == pytest.approx(0.1)
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert metrics["update_seconds"] > 0.0
    assert metrics["evaluated_samples_per_second"] > 0.0
    assert np.isfinite(metrics["entropy_bonus"])
    assert np.isfinite(metrics["entropy_stationary_storage"])
    if policy_credit_assignment == "typed_group":
        assert metrics["actor_samples"] >= metrics["samples"]
    else:
        assert metrics["actor_samples"] == metrics["samples"]


def test_ti_mappo_policy_anchor_is_frozen_and_checkpointed(tmp_path):
    compiler, _snapshot = compile_snapshot(tmp_path / "anchor", time_step=0)
    torch.manual_seed(23)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    critic = CentralSetCritic(
        compiler.type_registry,
        d_model=32,
        relation_layers=1,
    )
    learner = TIMAPPO(actor, critic, policy_anchor_coeff=0.25)
    assert learner.policy_anchor_actor is not None
    frozen = {
        key: value.detach().clone()
        for key, value in learner.policy_anchor_actor.state_dict().items()
    }

    with torch.no_grad():
        next(actor.parameters()).add_(1.0)
    assert any(
        not torch.equal(value, actor.state_dict()[key])
        for key, value in frozen.items()
    )

    restored_actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
    restored_critic = CentralSetCritic(
        compiler.type_registry,
        d_model=32,
        relation_layers=1,
    )
    restored = TIMAPPO(
        restored_actor,
        restored_critic,
        learning_rate=1.0e-4,
        policy_anchor_coeff=0.25,
    )
    restored.load_state_dict(
        learner.state_dict(),
        restore_optimizers=False,
        restore_rollout=False,
    )

    assert restored.policy_anchor_actor is not None
    assert all(
        torch.equal(value, restored.policy_anchor_actor.state_dict()[key])
        for key, value in frozen.items()
    )
    assert all(
        not parameter.requires_grad
        for parameter in restored.policy_anchor_actor.parameters()
    )
    assert restored.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)


def test_typed_group_advantage_routes_only_related_constraint_penalties(
    tmp_path,
):
    _compiler, first = compile_snapshot(tmp_path / "first", time_step=0)
    _compiler, second = compile_snapshot(tmp_path / "second", time_step=1)
    buffer = TypedRolloutBuffer()
    components = {
        agent_id: {
            "battery_safety_penalty": 1.0,
            "ev_service_penalty": 100.0,
            "deferrable_service_penalty": 3.0,
        }
        for agent_id in first.agent_ids
    }
    buffer.add(
        RolloutStep(
            snapshot=first,
            next_snapshot=second,
            bundles=(),
            old_log_probs={agent_id: 0.0 for agent_id in first.agent_ids},
            values={agent_id: 0.0 for agent_id in first.agent_ids},
            next_values={agent_id: 0.0 for agent_id in first.agent_ids},
            rewards={agent_id: -104.0 for agent_id in first.agent_ids},
            terminated_agent_ids=first.agent_ids,
            truncated=False,
            reward_components_by_agent=components,
        )
    )

    samples = buffer.typed_group_advantages(gamma=0.99, gae_lambda=0.95)
    advantages_by_type = {
        sample.group_type: sample.advantage for sample in samples
    }

    assert advantages_by_type["stationary_storage"] == pytest.approx(-1.0)
    assert advantages_by_type["ev_session"] == pytest.approx(-100.0)
    assert advantages_by_type["deferrable"] == pytest.approx(-3.0)


def test_required_deferrable_with_unknown_deadline_starts_once_when_safe(tmp_path):
    interfaces = write_typed_interfaces(
        tmp_path / "interfaces",
        ("Building_1", "Building_2"),
    )
    compiler = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interfaces_dir=interfaces,
    )
    compiler.attach_entity_specs(entity_specs(), seconds_per_time_step=900)
    payload = entity_payload()
    payload["meta"]["runtime_status"]["sensor_channels"] = [
        {
            "event_id": f"missing-{feature}",
            "event_ids": [f"missing-{feature}"],
            "fault_mode": "missing",
            "target_type": "deferrable_appliance",
            "target_id": "Building_1/washer",
            "target_feature": feature,
            "availability": "UNAVAILABLE",
            "quality": "INVALID",
        }
        for feature in ("deadline_time_step", "slack_steps")
    ]
    snapshot = compiler.compile(payload)
    group = next(
        item
        for item in snapshot.groups_for("Building_1")
        if item.group_type == "deferrable"
    )
    raw = LocalActionBundle(
        agent_id="Building_1",
        decisions=(ActionDecision(group.group_id, "IDLE", 0.0, 0),),
    )
    final = AnalyticLocalProjector().project(snapshot, (raw,))[0]
    assert final.decisions[0].mode == "START"
    assert any(
        item["reason"] == "required_deferrable_unknown_deadline"
        for item in final.interventions
    )


def test_codec_applies_a_dynamic_port_bound_exactly_once(tmp_path):
    _compiler, snapshot = compile_snapshot(tmp_path)
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


def test_discharge_uses_its_own_bound_and_keeps_a_negative_simulator_sign(tmp_path):
    _compiler, snapshot = compile_snapshot(tmp_path)
    ev_group = next(
        group
        for group in snapshot.groups_for("Building_1")
        if group.group_type == "ev_session"
    )
    ev_group = replace(
        ev_group,
        max_charge_power_kw=11.0,
        max_discharge_power_kw=7.2,
        ports=tuple(
            replace(port, upper_bound=0.5)
            if port.mode == "DISCHARGE_EV"
            else replace(port, upper_bound=1.0)
            if port.mode == "CHARGE_EV"
            else port
            for port in ev_group.ports
        ),
    )
    snapshot = replace(
        snapshot,
        action_groups=tuple(
            ev_group if group.group_id == ev_group.group_id else group
            for group in snapshot.action_groups
        ),
        constraints=tuple(
            replace(constraint, upper_bound=100.0)
            if constraint.owner_agent_id == "Building_1"
            and constraint.constraint_type == "export_headroom_kw"
            else constraint
            for constraint in snapshot.constraints
        ),
    )
    bundle = LocalActionBundle(
        agent_id="Building_1",
        decisions=(
            ActionDecision(ev_group.group_id, "DISCHARGE_EV", 1.0, 2),
        ),
    )
    projected = AnalyticLocalProjector().project(snapshot, (bundle,))[0]
    commands = TypedCommandBuilder().build(snapshot, (projected,))
    assert commands[0].action_id == "discharge"
    assert commands[0].value == pytest.approx(3.6)

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
            spaces.Box(low=-np.ones(3), high=np.ones(3)),
            spaces.Box(low=-np.ones(1), high=np.ones(1)),
        ),
    )
    encoded = codec.encode_typed(snapshot, commands)
    assert encoded[0][1] == pytest.approx(-0.5)


def test_codec_teacher_round_trip_preserves_typed_safe_action(tmp_path):
    compiler, snapshot = compile_snapshot(tmp_path)
    actor = TypedActor(
        compiler.type_registry,
        d_model=32,
        attention_heads=4,
        relation_layers=1,
    )
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
    teacher = ((0.25, -0.4, 1.0), (-0.3,))

    bundles = codec.decode_teacher_actions(
        snapshot,
        teacher,
        group_modes=actor.group_modes,
    )
    encoded = codec.encode(snapshot, bundles)

    assert encoded[0] == pytest.approx(teacher[0])
    assert encoded[1] == pytest.approx(teacher[1])


def test_ti_marl_typed_teacher_episode_never_enters_ppo_rollout(tmp_path):
    config = agent_config(tmp_path / "job")
    config["algorithm"]["hyperparameters"]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": 1,
        "max_samples": 8,
        "pretraining_epochs": 1,
        "batch_size": 2,
        "learning_rate": 1.0e-4,
        "teacher": {"policy": "RBCSmartPolicy", "hyperparameters": {}},
    }
    agent = TIMARL(config)
    attach_agent(agent, ("Building_1", "Building_2"))
    teacher_actions = [[0.25, 0.0, 0.0], [-0.25]]
    assert agent._bc_teacher is not None
    agent._bc_teacher.predict = lambda observations, deterministic: teacher_actions
    critic_before = {
        key: value.detach().clone() for key, value in agent.critic.state_dict().items()
    }
    agent.on_episode_start(episode=0, training=True)
    agent.set_observation_context(
        raw_observations=[np.zeros(1), np.zeros(1)],
        encoded_observations=None,
    )
    agent.set_entity_observation_context(
        observation_payload=entity_payload(time_step=0), info={}
    )

    actions = agent.predict([], deterministic=False)
    agent.set_entity_transition_context(
        observation_payload=entity_payload(time_step=0),
        next_observation_payload=entity_payload(time_step=1),
        info={},
    )
    agent.update(
        [],
        [np.asarray(row) for row in actions],
        [0.0, 0.0],
        [],
        False,
        False,
        update_target_step=False,
        global_learning_step=0,
        update_step=True,
        initial_exploration_done=True,
    )

    assert len(agent.learner.rollout) == 0
    assert agent.behavior_cloning is not None
    assert agent.behavior_cloning.seen_samples == 1
    agent.on_episode_end(episode=0, training=True)
    assert agent.behavior_cloning.pretraining_complete
    assert agent.behavior_cloning.training_samples == 1
    cloning_state = agent.behavior_cloning.state_dict()
    assert cloning_state["demonstrations"] == ()
    assert all(
        torch.equal(value, critic_before[key])
        for key, value in agent.critic.state_dict().items()
    )


def test_ti_marl_aligns_typed_credit_evidence_with_stable_agent_ids(tmp_path):
    config = agent_config(tmp_path / "job")
    config["algorithm"]["hyperparameters"][
        "policy_credit_assignment"
    ] = "typed_group"
    config["algorithm"]["hyperparameters"]["rollout_steps"] = 64
    agent = TIMARL(config)
    attach_agent(agent, ("Building_1", "Building_2"))
    agent.on_episode_start(episode=0, training=True)
    agent.set_entity_observation_context(
        observation_payload=entity_payload(time_step=0), info={}
    )
    actions = agent.predict([], deterministic=False)
    agent.set_entity_transition_context(
        observation_payload=entity_payload(time_step=0),
        next_observation_payload=entity_payload(time_step=1),
        info={
            "reward_components": {
                "per_agent": [
                    {
                        "battery_safety_penalty": 1.0,
                        "ev_service_penalty": 2.0,
                    },
                    {"battery_safety_penalty": 3.0},
                ]
            }
        },
    )
    agent.update(
        [],
        [np.asarray(row) for row in actions],
        [-3.0, -3.0],
        [],
        False,
        False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )

    step = agent.learner.rollout.steps[0]
    assert step.reward_components_by_agent["Building_1"][
        "ev_service_penalty"
    ] == pytest.approx(2.0)
    assert step.reward_components_by_agent["Building_2"][
        "battery_safety_penalty"
    ] == pytest.approx(3.0)
    assert step.group_values["Building_1"]
    assert step.next_group_values["Building_1"]


def test_rollout_gae_handles_leave_and_does_not_create_predecessor_for_join(tmp_path):
    _c1, first = compile_snapshot(tmp_path / "first", ("Building_1", "Building_2"), time_step=0)
    _c2, second = compile_snapshot(tmp_path / "second", ("Building_1",), time_step=1, topology_version=1)
    _c3, third = compile_snapshot(tmp_path / "third", ("Building_1", "Building_3"), time_step=2, topology_version=2)
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
    _compiler, first = compile_snapshot(tmp_path / "first", time_step=0)
    _compiler, second = compile_snapshot(tmp_path / "second", time_step=1)
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


def test_buffered_trace_keeps_intermediate_transitions_but_not_full_snapshots(
    tmp_path,
):
    _compiler, first = compile_snapshot(tmp_path / "first", time_step=0)
    _compiler, second = compile_snapshot(tmp_path / "second", time_step=1)
    _compiler, third = compile_snapshot(tmp_path / "third", time_step=2)

    def transition(current, following):
        return TypedTransition(
            snapshot_hash=current.snapshot_hash,
            next_snapshot_hash=following.snapshot_hash,
            agent_ids=current.agent_ids,
            next_agent_ids=following.agent_ids,
            raw_bundles=(),
            final_bundles=(),
            commands=(),
            execution={"version": "entity_action_execution_v1"},
            rewards=(),
            reward_components={},
            terminated_agent_ids=(),
            bootstrap_agent_ids=current.agent_ids,
        )

    writer = BufferedTraceWriter(tmp_path, chunk_size=8, snapshot_interval=99)
    writer.record(first, second, transition(first, second))
    writer.record(second, third, transition(second, third))
    writer.close()

    with gzip.open(next(tmp_path.glob("*.jsonl.gz")), "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    hashes = {row["hash"] for row in records if row["kind"] == "snapshot"}
    transitions = [row for row in records if row["kind"] == "transition"]

    assert hashes == {first.snapshot_hash, second.snapshot_hash}
    assert len(transitions) == 2
    assert transitions[-1]["payload"]["next_snapshot_hash"] == third.snapshot_hash


def agent_config(tmp_path):
    interfaces = write_typed_interfaces(
        tmp_path / "interfaces",
        ("Building_1", "Building_2", "Building_3"),
    )
    return {
        "algorithm": {
            "name": "TIMARL",
            "hyperparameters": {
                "contract_version": "ti_marl_v1",
                "typed_interfaces_dir": str(interfaces),
                "backbone": {"name": "mappo"},
                "actor": {"d_model": 32, "attention_heads": 4, "relation_layers": 1},
                "rollout_steps": 4,
                "trace": {"enabled": False},
            },
        },
        "training": {"seed": 9},
        "runtime": {"job_dir": str(tmp_path)},
    }


def ti_ppo_agent_config(tmp_path):
    config = agent_config(tmp_path)
    config["algorithm"]["hyperparameters"]["backbone"] = {"name": "ppo"}
    config["algorithm"]["hyperparameters"]["critic"] = {"kind": "local"}
    return config


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
            "seconds_per_time_step": 900,
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


def test_checkpoint_rejects_a_different_learning_architecture(tmp_path):
    source = TIMARL(agent_config(tmp_path / "source"))
    attach_agent(source, ("Building_1",))
    checkpoint = source.save_checkpoint(str(tmp_path / "checkpoint"), step=1)

    local_ppo = TIMARL(ti_ppo_agent_config(tmp_path / "local"))
    attach_agent(local_ppo, ("Building_1",))
    with pytest.raises(ValueError, match="learning architecture"):
        local_ppo.load_checkpoint(checkpoint)


def test_checkpoint_rejects_a_different_actor_group_context(tmp_path):
    source = TIMARL(agent_config(tmp_path / "source"))
    attach_agent(source, ("Building_1",))
    checkpoint = source.save_checkpoint(str(tmp_path / "checkpoint"), step=1)

    conditioned_config = agent_config(tmp_path / "conditioned")
    conditioned_config["algorithm"]["hyperparameters"]["actor"][
        "group_context_kind"
    ] = "action_conditioned"
    conditioned = TIMARL(conditioned_config)
    attach_agent(conditioned, ("Building_1",))
    with pytest.raises(ValueError, match="learning architecture"):
        conditioned.load_checkpoint(checkpoint)


def test_checkpoint_pins_typed_group_credit_and_restores_its_critic(tmp_path):
    typed_config = agent_config(tmp_path / "typed")
    typed_config["algorithm"]["hyperparameters"][
        "policy_credit_assignment"
    ] = "typed_group"
    source = TIMARL(typed_config)
    attach_agent(source, ("Building_1",))
    checkpoint = source.save_checkpoint(str(tmp_path / "checkpoint"), step=1)

    restored_config = agent_config(tmp_path / "restored")
    restored_config["algorithm"]["hyperparameters"][
        "policy_credit_assignment"
    ] = "typed_group"
    restored = TIMARL(restored_config)
    attach_agent(restored, ("Building_1",))
    restored.load_checkpoint(checkpoint)
    assert restored.group_critic is not None
    assert restored._parameter_count == source._parameter_count

    joint = TIMARL(agent_config(tmp_path / "joint"))
    attach_agent(joint, ("Building_1",))
    with pytest.raises(ValueError, match="learning architecture"):
        joint.load_checkpoint(checkpoint)


def test_checkpoint_compiler_migration_requires_explicit_opt_in(tmp_path):
    source = TIMARL(agent_config(tmp_path / "source"))
    attach_agent(source, ("Building_1",))
    checkpoint = Path(source.save_checkpoint(str(tmp_path / "checkpoint"), step=3))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["compatibility_signature"]["compiler_hash"] = "older-compiler"
    torch.save(payload, checkpoint)

    rejected = TIMARL(agent_config(tmp_path / "rejected"))
    attach_agent(rejected, ("Building_1",))
    with pytest.raises(ValueError, match="compatibility signature"):
        rejected.load_checkpoint(checkpoint)

    migration_config = agent_config(tmp_path / "migration")
    migration_config["algorithm"]["hyperparameters"][
        "allow_checkpoint_compiler_migration"
    ] = True
    migrated = TIMARL(migration_config)
    attach_agent(migrated, ("Building_1",))
    migrated.load_checkpoint(checkpoint)
    assert migrated.allow_checkpoint_compiler_migration is True


def test_export_includes_resolved_editable_interface_without_extra_model_artifact(
    tmp_path,
):
    agent = TIMARL(agent_config(tmp_path / "job"))
    attach_agent(agent, ("Building_1", "Building_2"))
    output = tmp_path / "bundle"
    metadata = agent.export_artifacts(str(output))
    assert len(metadata["artifacts"]) == 1
    assert metadata["artifacts"][0]["format"] == "ti_marl_torch"
    assert metadata["typed_interfaces_path"] == "typed_interfaces.resolved.yaml"
    resolved = output / metadata["typed_interfaces_path"]
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert payload["version"] == "typed_interface_registry_v1"
    assert set(payload["interfaces"]) >= {"Building_1", "Building_2"}
    deployment = torch.load(output / metadata["deployment_bundle_path"], weights_only=False)
    assert "actor" in deployment
    assert "critic" not in deployment
    manifest = build_manifest(
        {
            "metadata": {"experiment_name": "ti", "run_name": "export"},
            "simulator": {"central_agent": False},
            "training": {"seed": 9},
            "topology": {"num_agents": 2},
            "pipeline": [
                {"algorithm": "TIMARL", "count": 1, "hyperparameters": {}}
            ],
        },
        {
            "observation_names": [["entity"], ["entity"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_names_by_agent": {
                "0": ["electrical_storage"],
                "1": ["electrical_storage"],
            },
        },
        metadata,
    )
    validate_bundle_contract(manifest, output)
