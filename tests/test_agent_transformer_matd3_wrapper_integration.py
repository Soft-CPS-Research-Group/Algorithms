from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from algorithms.registry import ALGORITHM_REGISTRY
from tests._entity_sample_obs_names import (
    load_sample_observation_names_for_first_building,
)
from tests.test_agent_transformer_matd3 import (
    _ACTION_NAMES,
    _Box,
    _make_agent,
    _parameters,
    _transition,
)


def _expanded_topology() -> tuple[list[str], list[str], _Box]:
    names = load_sample_observation_names_for_first_building()
    charger_id = next(
        name.split("::")[1]
        for name in names
        if name.startswith("charger::")
        and "::connected_ev::" not in name
        and "::incoming_ev::" not in name
    )
    new_id = "Building_1/charger_NEW"
    expanded = list(names)
    expanded.extend(
        name.replace(f"charger::{charger_id}::", f"charger::{new_id}::", 1)
        for name in names
        if name.startswith(f"charger::{charger_id}::")
    )
    actions = list(_ACTION_NAMES) + ["electric_vehicle_storage"]
    return expanded, actions, _Box([-2.0, -0.5, -0.5], [1.0, 0.75, 0.75])


def _attach_expanded(agent) -> tuple[list[str], list[str]]:
    names, actions, space = _expanded_topology()
    agent.attach_environment(
        observation_names=[names],
        action_names=[actions],
        action_space=[space],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    return names, actions


def _attach_layout(agent, names, actions, spaces=None, building_names=None) -> None:
    if spaces is None:
        spaces = [_Box([-2.0, -0.5], [1.0, 0.75]) for _ in names]
    agent.attach_environment(
        observation_names=[list(value) for value in names],
        action_names=[list(value) for value in actions],
        action_space=spaces,
        observation_space=[None] * len(names),
        metadata={
            "building_names": building_names
            or [f"Building_{index + 1}" for index in range(len(names))]
        },
    )


def _without_asset(names, asset_id: str) -> list[str]:
    return [
        name
        for name in names
        if not name.startswith(f"charger::{asset_id}::")
    ]


def _without_controllable_assets(names) -> list[str]:
    return [
        name
        for name in names
        if not name.startswith(("storage::", "charger::"))
    ]


def test_registry_exposes_dynamic_topology_hooks() -> None:
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3

    assert AgentTransformerMATD3.supports_dynamic_topology is True
    assert ALGORITHM_REGISTRY["AgentTransformerMATD3"] is AgentTransformerMATD3


def test_compatible_topology_commit_preserves_neural_optimizer_and_history() -> None:
    agent, obs_dim = _make_agent(buildings=1, batch_size=1)
    _transition(agent, obs_dim, 0)
    state = agent._per_building[0]
    old_signature = agent._layout_signature
    actor_id = id(state.actor)
    optimizer_id = id(state.actor_optimizer)
    actor_before = _parameters(state.actor)
    replay_size = agent.replay_buffer.total_size()

    new_names, _ = _attach_expanded(agent)

    state = agent._per_building[0]
    assert id(state.actor) == actor_id
    assert id(state.actor_optimizer) == optimizer_id
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(actor_before, state.actor.parameters())
    )
    assert state.topology_version == 1
    assert state.layout.n_ca == 3
    assert agent.replay_buffer.total_size() == replay_size
    assert agent.replay_buffer.bucket_size(old_signature) == replay_size
    assert agent.replay_buffer.bucket_size(agent._layout_signature) == 0

    _transition(agent, len(new_names), 1)
    assert agent.replay_buffer.bucket_size(agent._layout_signature) == 1
    assert tuple(agent.replay_buffer.signatures()) == (
        old_signature,
        agent._layout_signature,
    )


def test_removing_one_charger_shrinks_actions_bounds_and_only_version_for_building(
    monkeypatch,
) -> None:
    from algorithms.transformer_matd3 import agent as agent_module

    agent, _ = _make_agent(buildings=2)
    base_names = load_sample_observation_names_for_first_building()
    expanded_names, expanded_actions, expanded_space = _expanded_topology()
    _attach_layout(
        agent,
        [expanded_names, base_names],
        [expanded_actions, _ACTION_NAMES],
        spaces=[expanded_space, _Box([-2.0, -0.5], [1.0, 0.75])],
    )
    assert [state.topology_version for state in agent._per_building] == [1, 0]

    messages = []
    monkeypatch.setattr(
        agent_module.logger,
        "info",
        lambda *args, **kwargs: messages.append((args, kwargs)),
    )
    charger_id = "Building_1/charger_NEW"
    removed_names = _without_asset(expanded_names, charger_id)
    _attach_layout(
        agent,
        [removed_names, base_names],
        [_ACTION_NAMES, _ACTION_NAMES],
    )

    first, second = agent._per_building
    assert first.layout.n_ca == 2
    assert first.action_names == tuple(_ACTION_NAMES)
    assert first.action_low.tolist() == pytest.approx([-2.0, -0.5])
    assert first.action_high.tolist() == pytest.approx([1.0, 0.75])
    assert [state.topology_version for state in agent._per_building] == [2, 0]
    assert second.layout.n_ca == 2
    assert len(messages) == 1
    log_args, log_kwargs = messages[0]
    assert log_kwargs == {}
    assert "operation=matd3_topology_commit" in log_args[0]
    assert log_args[1:] == (
        "Building_1",
        2,
        1,
        2,
        3,
        2,
        expanded_actions,
        list(_ACTION_NAMES),
        "removal",
    )


def test_removal_preserves_learned_modules_optimizers_and_runtime_state() -> None:
    agent, obs_dim = _make_agent(buildings=1, batch_size=1)
    _transition(agent, obs_dim, 0)
    expanded_names, expanded_actions, expanded_space = _expanded_topology()
    _attach_layout(agent, [expanded_names], [expanded_actions], [expanded_space])
    state = agent._per_building[0]
    _transition(agent, len(expanded_names), 1)

    modules = (
        "tokenizer", "backbone", "actor", "tokenizer_target", "backbone_target",
        "actor_target", "critic_1", "critic_1_target", "critic_2", "critic_2_target",
    )
    module_ids = {name: id(getattr(state, name)) for name in modules}
    module_values = {name: _parameters(getattr(state, name)) for name in modules}
    optimizer_ids = {
        name: id(getattr(state, name))
        for name in ("actor_optimizer", "critic_1_optimizer", "critic_2_optimizer")
    }
    counters = (
        agent.exploration_step,
        agent.reward_norm_count,
        agent.critic_update_count,
        agent.actor_update_count,
        agent.target_update_count,
    )

    removed_names = _without_asset(expanded_names, "Building_1/charger_NEW")
    _attach_layout(agent, [removed_names], [_ACTION_NAMES])
    state = agent._per_building[0]

    assert {name: id(getattr(state, name)) for name in modules} == module_ids
    assert {name: id(getattr(state, name)) for name in optimizer_ids} == optimizer_ids
    for name in modules:
        assert all(
            torch.equal(before, after.detach())
            for before, after in zip(module_values[name], getattr(state, name).parameters())
        )
    assert (
        agent.exploration_step,
        agent.reward_norm_count,
        agent.critic_update_count,
        agent.actor_update_count,
        agent.target_update_count,
    ) == counters


def test_removal_preserves_behavior_cloning_reservoir_and_regularizer() -> None:
    from tests.test_agent_transformer_matd3_behavior_cloning import _bc_b_agent

    agent, obs_dim = _bc_b_agent()
    assert agent._bc_b is not None
    state = agent._per_building[0]
    agent._bc_b.record_demonstration(
        0,
        np.zeros(obs_dim, dtype=np.float32),
        state.layout,
        [0.25, -0.25],
    )
    regularizer_id = id(agent._bc_b)
    demonstration_count = agent._bc_b.demonstration_count(0)
    expanded_names, expanded_actions, expanded_space = _expanded_topology()
    _attach_layout(agent, [expanded_names], [expanded_actions], [expanded_space])
    reduced_names = _without_asset(expanded_names, "Building_1/charger_NEW")
    _attach_layout(agent, [reduced_names], [_ACTION_NAMES])

    assert id(agent._bc_b) == regularizer_id
    assert agent._bc_b.demonstration_count(0) == demonstration_count
    assert agent._bc_b.teacher_policy is not None


def test_predict_after_removal_is_finite_bounded_and_has_reduced_width() -> None:
    agent, obs_dim = _make_agent(buildings=1)
    expanded_names, expanded_actions, expanded_space = _expanded_topology()
    _attach_layout(agent, [expanded_names], [expanded_actions], [expanded_space])
    removed_names = _without_asset(expanded_names, "Building_1/charger_NEW")
    _attach_layout(agent, [removed_names], [_ACTION_NAMES])

    observation = np.linspace(-1.0, 1.0, len(removed_names), dtype=np.float32)
    deterministic = agent.predict([observation], deterministic=True)[0]
    exploratory = agent.predict([observation], deterministic=False)[0]
    assert len(deterministic) == len(exploratory) == 2
    assert np.isfinite(deterministic).all() and np.isfinite(exploratory).all()
    assert np.all(np.asarray(exploratory) >= [-2.0, -0.5])
    assert np.all(np.asarray(exploratory) <= [1.0, 0.75])


def test_removing_all_controllable_assets_supports_empty_prediction_and_critics() -> None:
    agent, _ = _make_agent(buildings=1)
    names = _without_controllable_assets(load_sample_observation_names_for_first_building())
    _attach_layout(agent, [names], [[]], [_Box([], [])])
    state = agent._per_building[0]
    observation = np.zeros(len(names), dtype=np.float32)
    actions = agent.predict([observation], deterministic=True)

    assert state.layout.n_ca == 0
    assert actions == [[]]
    observations = [torch.zeros((2, len(names)), device=agent.device)]
    critic_actions = [torch.zeros((2, 0), device=agent.device)]
    value = state.critic_1(observations, [state.layout], critic_actions)
    assert value.shape == (2, 1)
    assert torch.isfinite(value).all()


def test_add_remove_readd_restores_exact_signature_and_historical_bucket() -> None:
    agent, obs_dim = _make_agent(buildings=1, buffer_capacity=64)
    base_signature = agent._layout_signature
    base_names = load_sample_observation_names_for_first_building()
    expanded_names, expanded_actions, expanded_space = _expanded_topology()
    _transition(agent, obs_dim, 0)
    _attach_layout(agent, [expanded_names], [expanded_actions], [expanded_space])
    expanded_signature = agent._layout_signature
    _transition(agent, len(expanded_names), 1)
    expanded_bucket_size = agent.replay_buffer.bucket_size(expanded_signature)

    _attach_layout(agent, [base_names], [_ACTION_NAMES])
    assert agent.replay_buffer.bucket_size(expanded_signature) == expanded_bucket_size
    _attach_layout(agent, [expanded_names], [expanded_actions], [expanded_space])

    assert agent._layout_signature == expanded_signature
    assert agent.replay_buffer.bucket_size(expanded_signature) == expanded_bucket_size
    assert tuple(agent.replay_buffer.signatures()) == (base_signature, expanded_signature)


def test_readd_rejects_bounds_that_conflict_with_historical_replay() -> None:
    agent, _ = _make_agent(buildings=1, buffer_capacity=64)
    base_names = load_sample_observation_names_for_first_building()
    expanded_names, expanded_actions, expanded_space = _expanded_topology()
    _attach_layout(agent, [expanded_names], [expanded_actions], [expanded_space])
    expanded_signature = agent._layout_signature
    _transition(agent, len(expanded_names), 1)
    _attach_layout(agent, [base_names], [_ACTION_NAMES])
    signature_before = agent._layout_signature
    replay_before = agent.replay_buffer.get_state()

    changed_bounds = _Box([-2.0, -0.5, -0.4], [1.0, 0.75, 0.75])
    with pytest.raises(ValueError, match="historical layout signature"):
        _attach_layout(
            agent,
            [expanded_names],
            [expanded_actions],
            [changed_bounds],
        )

    assert agent._layout_signature == signature_before
    assert agent.replay_buffer.bucket_size(expanded_signature) == 1
    assert agent.replay_buffer.get_state()["global_fifo"] == replay_before[
        "global_fifo"
    ]


def test_invalid_action_order_or_bounds_fails_atomically() -> None:
    agent, _ = _make_agent(buildings=1)
    signature_before = agent._layout_signature
    state_before = agent.snapshot_topology_state()
    names = load_sample_observation_names_for_first_building()
    with pytest.raises(ValueError):
        _attach_layout(agent, [names], [["electric_vehicle_storage", "electrical_storage"]])
    assert agent._layout_signature == signature_before
    assert agent.replay_buffer.get_state()["global_fifo"] == state_before.agent_state[
        "replay_buffer"
    ].get_state()["global_fifo"]

    with pytest.raises(ValueError):
        _attach_layout(
            agent,
            [names],
            [_ACTION_NAMES],
            [_Box([-2.1, -0.5], [1.0, 0.75])],
        )
    assert agent._layout_signature == signature_before


def test_unknown_asset_type_fails_atomically() -> None:
    agent, _ = _make_agent(buildings=1)
    names = load_sample_observation_names_for_first_building()
    signature_before = agent._layout_signature
    with pytest.raises(ValueError):
        _attach_layout(
            agent,
            [names + ["unknown::Building_1/asset::value"]],
            [_ACTION_NAMES],
        )
    assert agent._layout_signature == signature_before


def test_nfc_drift_fails_atomically(monkeypatch) -> None:
    agent, _ = _make_agent(buildings=1)
    original_build = agent._layout_builder.build
    names = load_sample_observation_names_for_first_building()

    def drifted_build(*args, **kwargs):
        layout = original_build(*args, **kwargs)
        segments = tuple(
            replace(
                segment,
                derived=replace(segment.derived, op="add"),
            )
            if segment.family == "nfc" and segment.derived is not None
            else segment
            for segment in layout.segments
        )
        return replace(layout, segments=segments)

    monkeypatch.setattr(agent._layout_builder, "build", drifted_build)
    signature_before = agent._layout_signature
    with pytest.raises(ValueError, match="topology schema drift"):
        _attach_layout(agent, [names], [_ACTION_NAMES])
    assert agent._layout_signature == signature_before


def test_late_bc_failure_restores_pending_queue_rng_and_has_no_commit_log(monkeypatch) -> None:
    from algorithms.transformer_matd3 import agent as agent_module

    agent, obs_dim = _make_agent(buildings=1, n_step_returns=3)
    _transition(agent, obs_dim, 0)
    snapshot = agent.snapshot_topology_state()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    messages = []
    monkeypatch.setattr(
        agent_module.logger,
        "info",
        lambda *args, **kwargs: messages.append((args, kwargs)),
    )

    def fail_bc(**kwargs):
        del kwargs
        raise RuntimeError("late BC attachment failed")

    monkeypatch.setattr(agent, "_attach_bc_b_environment", fail_bc)
    expanded_names, expanded_actions, expanded_space = _expanded_topology()
    with pytest.raises(RuntimeError, match="late BC attachment failed"):
        _attach_layout(agent, [expanded_names], [expanded_actions], [expanded_space])

    assert agent._layout_signature == snapshot.agent_state["_layout_signature"]
    assert len(agent._n_step_queue) == len(snapshot.agent_state["_n_step_queue"]) == 1
    assert agent.replay_buffer.get_state()["global_fifo"] == snapshot.agent_state[
        "replay_buffer"
    ].get_state()["global_fifo"]
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert not any("commit" in str(args).lower() for args, _ in messages)


def test_feature_width_failure_rolls_back_all_live_state_and_rng() -> None:
    agent, obs_dim = _make_agent(buildings=1, batch_size=1)
    _transition(agent, obs_dim, 0)
    signature_before = agent._layout_signature
    actor_before = _parameters(agent._per_building[0].actor)
    replay_before = agent.replay_buffer.get_state()
    layout_cache_before = dict(agent._layout_builder._cache)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    names = load_sample_observation_names_for_first_building()
    storage_id = next(
        name.split("::")[1] for name in names if name.startswith("storage::")
    )
    drifted = list(names) + [f"storage::{storage_id}::new_width_field"]

    with pytest.raises(ValueError, match="topology schema drift"):
        agent.attach_environment(
            observation_names=[drifted],
            action_names=[list(_ACTION_NAMES)],
            action_space=[_Box([-2.0, -0.5], [1.0, 0.75])],
            observation_space=[None],
            metadata={"building_names": ["Building_1"]},
        )

    assert agent._layout_signature == signature_before
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(
            actor_before, agent._per_building[0].actor.parameters()
        )
    )
    assert (
        agent.replay_buffer.get_state()["global_fifo"]
        == replay_before["global_fifo"]
    )
    assert tuple(agent.replay_buffer.signatures()) == (signature_before,)
    assert agent._layout_builder._cache == layout_cache_before
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_topology_feature_order_drift_fails_atomically() -> None:
    agent, _ = _make_agent(buildings=1)
    names = load_sample_observation_names_for_first_building()
    storage_indices = [
        index for index, name in enumerate(names) if name.startswith("storage::")
    ]
    assert len(storage_indices) >= 2
    reordered = list(names)
    first, second = storage_indices[:2]
    reordered[first], reordered[second] = reordered[second], reordered[first]
    signature_before = agent._layout_signature

    with pytest.raises(ValueError, match="topology schema drift"):
        agent.attach_environment(
            observation_names=[reordered],
            action_names=[list(_ACTION_NAMES)],
            action_space=[_Box([-2.0, -0.5], [1.0, 0.75])],
            observation_space=[None],
            metadata={"building_names": ["Building_1"]},
        )

    assert agent._layout_signature == signature_before


def test_retained_segment_reorder_fails_atomically(monkeypatch) -> None:
    agent, _ = _make_agent(buildings=1)
    original_build = agent._layout_builder.build
    names = load_sample_observation_names_for_first_building()

    def reordered_build(*args, **kwargs):
        layout = original_build(*args, **kwargs)
        segments = list(layout.segments)
        sro_indices = [
            index
            for index, segment in enumerate(segments)
            if segment.family == "sro"
        ]
        assert len(sro_indices) >= 2
        first, second = sro_indices[:2]
        segments[first], segments[second] = segments[second], segments[first]
        return replace(layout, segments=tuple(segments))

    monkeypatch.setattr(agent._layout_builder, "build", reordered_build)
    signature_before = agent._layout_signature
    with pytest.raises(ValueError, match="ordered segments changed"):
        _attach_layout(agent, [names], [_ACTION_NAMES])

    assert agent._layout_signature == signature_before


def test_topology_commit_flushes_pending_n_step_entries_as_truncated() -> None:
    agent, obs_dim = _make_agent(buildings=1, n_step_returns=3, batch_size=1)
    _transition(agent, obs_dim, 0, rewards=[2.0])
    old_signature = agent._layout_signature
    assert len(agent._n_step_queue) == 1
    assert agent.replay_buffer.total_size() == 0

    _attach_expanded(agent)

    assert len(agent._n_step_queue) == 0
    transition = agent.replay_buffer.get_state()["transitions"][0]
    assert transition.signature == old_signature
    assert transition.rewards.tolist() == pytest.approx([2.0])
    assert transition.truncated.tolist() == [True]
    assert transition.terminated.tolist() == [False]
    _attach_expanded(agent)
    assert len(agent.replay_buffer.get_state()["transitions"]) == 1


def test_wrapper_boundary_hook_records_and_truncates_old_layout_tail() -> None:
    agent, obs_dim = _make_agent(buildings=1, n_step_returns=3, batch_size=1)
    _transition(agent, obs_dim, 0, rewards=[1.0])
    observations = [np.zeros(obs_dim, dtype=np.float32)]
    actions = agent.predict(observations, deterministic=True)
    old_signature = agent._layout_signature

    agent.record_topology_transition(
        observations=observations,
        actions=actions,
        rewards=[2.0],
        terminated=False,
        truncated=False,
        global_learning_step=1,
    )
    _attach_expanded(agent)

    transitions = agent.replay_buffer.get_state()["transitions"]
    assert len(transitions) == 2
    assert all(item.signature == old_signature for item in transitions)
    assert all(item.truncated.tolist() == [True] for item in transitions)
    assert transitions[0].rewards.tolist() == pytest.approx([2.9])
    assert transitions[1].rewards.tolist() == pytest.approx([2.0])


def test_building_count_change_applies_reset_full() -> None:
    agent, obs_dim = _make_agent(buildings=1, batch_size=1)
    _transition(agent, obs_dim, 0)
    old_actor = _parameters(agent._per_building[0].actor)
    agent.reward_norm_count = 9
    agent.exploration_step = 11
    names = load_sample_observation_names_for_first_building()

    agent.attach_environment(
        observation_names=[list(names), list(names)],
        action_names=[list(_ACTION_NAMES), list(_ACTION_NAMES)],
        action_space=[
            _Box([-2.0, -0.5], [1.0, 0.75]),
            _Box([-2.0, -0.5], [1.0, 0.75]),
        ],
        observation_space=[None, None],
        metadata={"building_names": ["Building_1", "Building_2"]},
    )

    assert len(agent._per_building) == 2
    assert agent.replay_buffer.total_size() == 0
    assert len(agent._n_step_queue) == 0
    assert agent.reward_norm_count == 0
    assert agent.exploration_step == 0
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(old_actor, agent._per_building[0].actor.parameters())
    )


def test_late_topology_attach_failure_restores_committed_state(monkeypatch) -> None:
    agent, _ = _make_agent(buildings=1)
    snapshot = agent.snapshot_topology_state()
    signature_before = agent._layout_signature

    def fail_teacher(**kwargs):
        del kwargs
        raise RuntimeError("teacher reconstruction failed")

    monkeypatch.setattr(agent, "_attach_bc_b_environment", fail_teacher)

    with pytest.raises(RuntimeError, match="teacher reconstruction failed"):
        _attach_expanded(agent)

    assert agent._layout_signature == signature_before
    assert agent._per_building[0].topology_version == 0
    assert agent.replay_buffer.get_state()["global_fifo"] == snapshot.agent_state[
        "replay_buffer"
    ].get_state()["global_fifo"]
