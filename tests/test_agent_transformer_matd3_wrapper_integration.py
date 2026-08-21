from __future__ import annotations

import random

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


def test_pr6_enables_agent_hooks_without_registry_integration() -> None:
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3

    assert AgentTransformerMATD3.supports_dynamic_topology is True
    assert "AgentTransformerMATD3" not in ALGORITHM_REGISTRY


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

    with pytest.raises(ValueError, match="feature width"):
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
