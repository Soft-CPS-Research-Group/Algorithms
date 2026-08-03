"""Behavior-cloning integration tests for AgentTransformerPPO."""
from __future__ import annotations

from copy import deepcopy
from typing import List

import numpy as np
import pytest
import torch

from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
from algorithms.agents.baseline_policies import RBCCommunityPolicy
from tests.test_agent_transformer_ppo import (
    _DEFAULT_ACTIONS,
    _base_config,
)
from tests._entity_sample_obs_names import (
    load_sample_observation_names_for_first_building,
)


class _DummySpace:
    def __init__(self, size: int) -> None:
        self.low = np.full(size, -1.0, dtype=np.float64)
        self.high = np.full(size, 1.0, dtype=np.float64)


def _bc_config(
    *,
    phaseout_mode: str = "probability",
    phaseout_steps: int = 0,
    weight: float = 0.4,
) -> dict:
    cfg = _base_config()
    cfg["algorithm"]["behavior_cloning"] = {
        "enabled": True,
        "weight": weight,
        "min_weight": 0.1,
        "decay_start_step": 0,
        "decay_steps": 100,
        "ev_multiplier": 2.0,
        "storage_multiplier": 1.0,
        "warm_start": {
            "policy": "RBCCommunityPolicy",
            "deterministic": True,
            "noise_scale": 0.0,
            "phaseout_steps": phaseout_steps,
            "phaseout_mode": phaseout_mode,
            "hyperparameters": {},
        },
    }
    return cfg


def _make_agent(
    *,
    config: dict | None = None,
    n_buildings: int = 1,
) -> tuple[AgentTransformerPPO, List[List[str]], List[List[str]], int]:
    obs_names = load_sample_observation_names_for_first_building()
    obs_names_per = [list(obs_names) for _ in range(n_buildings)]
    act_names_per = [list(_DEFAULT_ACTIONS) for _ in range(n_buildings)]
    agent = AgentTransformerPPO(config or _base_config())
    action_space = [_DummySpace(len(actions)) for actions in act_names_per]
    agent.attach_environment(
        observation_names=obs_names_per,
        action_names=act_names_per,
        action_space=action_space,
        observation_space=[None] * n_buildings,
        metadata={
            "building_names": [f"Building_{b+1}" for b in range(n_buildings)],
            "seconds_per_time_step": 3600,
        },
    )
    obs_dim = max(
        max(seg.feature_indices) for seg in agent._per_building[0].layout.segments
    ) + 1
    return agent, obs_names_per, act_names_per, obs_dim


def _random_transition(agent: AgentTransformerPPO, obs_dim: int, rng: np.random.Generator):
    obs = [rng.standard_normal(obs_dim).astype(np.float64)]
    next_obs = [rng.standard_normal(obs_dim).astype(np.float64)]
    actions = [rng.uniform(-0.5, 0.5, size=(agent._per_building[0].layout.n_ca,))]
    return obs, actions, next_obs


def _random_transition_for_all_buildings(
    agent: AgentTransformerPPO,
    obs_dim: int,
    rng: np.random.Generator,
):
    obs = [
        rng.standard_normal(obs_dim).astype(np.float64)
        for _ in agent._per_building
    ]
    next_obs = [
        rng.standard_normal(obs_dim).astype(np.float64)
        for _ in agent._per_building
    ]
    actions = [
        rng.uniform(-0.5, 0.5, size=(state.layout.n_ca,))
        for state in agent._per_building
    ]
    return obs, actions, next_obs


def _set_fake_teacher(agent: AgentTransformerPPO, actions: List[List[float]]) -> None:
    assert agent._bc is not None

    def compute_teacher_actions(raw_or_encoded):
        agent._bc.set_latest_teacher_actions(actions)
        return [list(row) for row in actions]

    agent._bc.compute_teacher_actions = compute_teacher_actions


def test_bc_absent_leaves_agent_without_regularizer() -> None:
    agent, _, _, obs_dim = _make_agent(config=_base_config())

    assert agent._bc is None

    obs = [np.zeros(obs_dim, dtype=np.float64)]
    agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
    actions = agent.predict(obs, deterministic=True)

    assert len(actions) == 1
    assert len(actions[0]) == agent._per_building[0].layout.n_ca


def test_bc_absent_does_not_require_raw_observation_context() -> None:
    agent = AgentTransformerPPO(_base_config())

    assert getattr(agent, "requires_raw_observation_context", False) is False


def test_bc_present_requires_raw_observation_context() -> None:
    agent = AgentTransformerPPO(_bc_config())

    assert agent.requires_raw_observation_context is True


def test_bc_present_attaches_teacher_policy() -> None:
    agent, _, _, _ = _make_agent(config=_bc_config())

    assert agent._bc is not None
    assert isinstance(agent._bc.teacher_policy, RBCCommunityPolicy)


def test_bounds_only_reattach_rebuilds_bc_teacher_with_new_bounds() -> None:
    agent, obs_per, act_per, _ = _make_agent(config=_bc_config())
    assert agent._bc is not None
    previous_teacher = agent._bc.teacher_policy
    bounds = _DummySpace(len(act_per[0]))
    bounds.low = np.array([-0.75, -0.5], dtype=np.float64)
    bounds.high = np.array([0.5, 0.75], dtype=np.float64)

    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[bounds],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    assert agent._bc.teacher_policy is not previous_teacher
    assert [bound.flatten().tolist() for bound in agent._action_bounds[0]] == [
        [-0.75, -0.5],
        [0.5, 0.75],
    ]
    assert agent._bc.teacher_policy._action_bounds == [{
        "low": [-0.75, -0.5],
        "high": [0.5, 0.75],
    }]


def test_failed_bounds_only_bc_reattach_preserves_state_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, obs_per, act_per, _ = _make_agent(config=_bc_config())
    assert agent._bc is not None
    teacher = agent._bc.teacher_policy
    teacher.mutable_state = {"history": ["before-reattach"]}
    bounds_before = agent._action_bounds
    original_prepare = agent._prepare_bc_topology_change
    bounds = _DummySpace(len(act_per[0]))
    bounds.low = np.array([-0.75, -0.5], dtype=np.float64)
    bounds.high = np.array([0.5, 0.75], dtype=np.float64)

    def fail_bc_reattach(**_kwargs):
        raise RuntimeError("BC reattach failed")

    monkeypatch.setattr(agent, "_prepare_bc_topology_change", fail_bc_reattach)
    with pytest.raises(RuntimeError, match="BC reattach failed"):
        agent.attach_environment(
            observation_names=obs_per,
            action_names=act_per,
            action_space=[bounds],
            observation_space=[None],
            metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
        )

    assert agent._action_bounds is bounds_before
    assert agent._bc.teacher_policy is teacher
    assert teacher.mutable_state == {"history": ["before-reattach"]}

    monkeypatch.setattr(agent, "_prepare_bc_topology_change", original_prepare)
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[bounds],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    assert agent._bc.teacher_policy is not teacher
    assert [bound.flatten().tolist() for bound in agent._action_bounds[0]] == [
        [-0.75, -0.5],
        [0.5, 0.75],
    ]


def test_initial_bc_attach_failure_rolls_back_and_valid_retry_attaches_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = AgentTransformerPPO(_bc_config())
    assert agent._bc is not None
    regularizer = agent._bc
    original_build_teacher = regularizer._build_teacher_policy
    observation_names = [load_sample_observation_names_for_first_building()]
    action_names = [list(_DEFAULT_ACTIONS)]
    action_space = [_DummySpace(len(action_names[0]))]

    def fail_teacher_setup(**_kwargs):
        raise RuntimeError("teacher setup failed")

    monkeypatch.setattr(regularizer, "_build_teacher_policy", fail_teacher_setup)
    with pytest.raises(RuntimeError, match="teacher setup failed"):
        agent.attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=[None],
            metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
        )

    assert agent._first_attach_done is False
    assert agent._per_building == []
    assert agent._pending_decisions == []
    assert agent._action_bounds == []
    assert agent._bc is regularizer
    assert regularizer.teacher_policy is None
    assert regularizer.teacher_action_buffers == []

    monkeypatch.setattr(regularizer, "_build_teacher_policy", original_build_teacher)
    agent.attach_environment(
        observation_names=observation_names,
        action_names=action_names,
        action_space=action_space,
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    assert agent._first_attach_done is True
    assert isinstance(agent._bc.teacher_policy, RBCCommunityPolicy)


def test_predict_with_legacy_blend_phaseout_keeps_exact_actor_decision() -> None:
    agent, _, _, obs_dim = _make_agent(
        config=_bc_config(phaseout_mode="blend", phaseout_steps=4)
    )
    teacher_actions = [[0.25 for _ in range(agent._per_building[0].layout.n_ca)]]
    _set_fake_teacher(agent, teacher_actions)
    raw_obs = [np.ones(obs_dim, dtype=np.float64)]
    encoded_obs = [np.zeros(obs_dim, dtype=np.float64)]

    agent.set_observation_context(
        raw_observations=raw_obs,
        encoded_observations=encoded_obs,
    )
    actions = agent.predict(encoded_obs, deterministic=False)
    cached = agent._pending_decisions[0]

    assert agent._bc is not None
    assert cached is not None
    assert agent._bc.latest_teacher_actions == teacher_actions
    np.testing.assert_array_equal(np.asarray(actions[0]), cached.action.cpu().numpy().squeeze(-1))

    agent.update(
        observations=encoded_obs,
        actions=[np.asarray(actions[0])],
        rewards=[0.1],
        next_observations=encoded_obs,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )

    state = agent._per_building[0]
    assert torch.equal(state.buffer.actions[-1], cached.action.cpu())
    assert torch.equal(state.buffer.log_probs[-1], cached.log_prob.cpu())
    assert torch.equal(state.buffer.values[-1], cached.value.cpu())


def test_update_records_teacher_actions_aligned_with_buffer() -> None:
    agent, _, _, obs_dim = _make_agent(config=_bc_config())
    teacher_actions = [[0.1 for _ in range(agent._per_building[0].layout.n_ca)]]
    _set_fake_teacher(agent, teacher_actions)
    rng = np.random.default_rng(1)

    for step in range(3):
        obs, actions, next_obs = _random_transition(agent, obs_dim, rng)
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        actions = agent.predict(obs, deterministic=bool(step % 2))
        agent.update(
            observations=obs,
            actions=actions,
            rewards=[0.1],
            next_observations=next_obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )

    assert agent._bc is not None
    for b, state in enumerate(agent._per_building):
        assert len(agent._bc.teacher_action_buffers[b]) == len(state.buffer)


def test_ppo_update_records_bc_metrics_when_teacher_available() -> None:
    agent, _, _, obs_dim = _make_agent(config=_bc_config(weight=0.6))
    teacher_actions = [[0.0 for _ in range(agent._per_building[0].layout.n_ca)]]
    _set_fake_teacher(agent, teacher_actions)
    rng = np.random.default_rng(2)

    for step in range(agent._minibatch_size):
        obs, actions, next_obs = _random_transition(agent, obs_dim, rng)
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        actions = agent.predict(obs, deterministic=False)
        agent.update(
            observations=obs,
            actions=actions,
            rewards=[0.1],
            next_observations=next_obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=10,
            update_step=step == agent._minibatch_size - 1,
            initial_exploration_done=True,
        )

    metrics = agent.consume_latest_training_metrics()
    assert metrics["behavior_cloning_effective_weight"] > 0.0
    assert np.isfinite(metrics["behavior_cloning_effective_weight"])
    assert metrics["behavior_cloning_valid_samples"] >= 0.0


def test_topology_change_flushes_bc_buffers() -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(config=_bc_config())
    assert agent._bc is not None
    old_teacher = agent._bc.teacher_policy
    teacher_actions = [[0.2 for _ in range(agent._per_building[0].layout.n_ca)]]
    _set_fake_teacher(agent, teacher_actions)
    rng = np.random.default_rng(3)

    obs, actions, next_obs = _random_transition(agent, obs_dim, rng)
    agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
    actions = agent.predict(obs, deterministic=False)
    agent.update(
        observations=obs,
        actions=actions,
        rewards=[0.1],
        next_observations=next_obs,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )
    assert len(agent._bc.teacher_action_buffers[0]) == 1

    orig_id = next(
        n.split("::")[1]
        for n in obs_per[0]
        if n.startswith("charger::") and "::connected_ev::" not in n and "::incoming_ev::" not in n
    )
    new_id = "Building_1/charger_BC_NEW"
    new_obs = list(obs_per[0])
    new_obs.extend(
        n.replace(f"charger::{orig_id}::", f"charger::{new_id}::", 1)
        for n in obs_per[0]
        if n.startswith(f"charger::{orig_id}::")
    )
    new_acts = list(act_per[0]) + ["electric_vehicle_storage"]

    agent.attach_environment(
        observation_names=[new_obs],
        action_names=[new_acts],
        action_space=[_DummySpace(len(new_acts))],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )

    assert agent._bc.teacher_action_buffers == [[]]
    assert agent._bc.teacher_policy is not old_teacher
    assert isinstance(agent._bc.teacher_policy, RBCCommunityPolicy)


def test_one_sample_topology_flush_discards_rollout_and_keeps_bc_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(config=_bc_config())
    assert agent._bc is not None
    teacher = agent._bc.teacher_policy
    teacher.mutable_state = {"history": ["before-flush"]}
    teacher_actions = [[0.2 for _ in range(agent._per_building[0].layout.n_ca)]]
    _set_fake_teacher(agent, teacher_actions)
    rng = np.random.default_rng(5)

    obs, _, next_obs = _random_transition(agent, obs_dim, rng)
    agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
    actions = agent.predict(obs, deterministic=False)
    agent.update(
        observations=obs,
        actions=actions,
        rewards=[0.1],
        next_observations=next_obs,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )

    orig_id = next(
        n.split("::")[1]
        for n in obs_per[0]
        if n.startswith("charger::") and "::connected_ev::" not in n and "::incoming_ev::" not in n
    )
    new_id = "Building_1/charger_BC_FAILURE"
    new_obs = list(obs_per[0])
    new_obs.extend(
        n.replace(f"charger::{orig_id}::", f"charger::{new_id}::", 1)
        for n in obs_per[0]
        if n.startswith(f"charger::{orig_id}::")
    )
    new_acts = list(act_per[0]) + ["electric_vehicle_storage"]
    ppo_updates = 0
    warnings: list[str] = []

    def count_ppo_update(state, last_value, *, building_idx):
        nonlocal ppo_updates
        ppo_updates += 1
        return False

    monkeypatch.setattr(agent, "_run_ppo_update_with_last_value", count_ppo_update)
    monkeypatch.setattr(
        "algorithms.agents.agent_transformer_ppo.logger.warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    agent.attach_environment(
        observation_names=[new_obs],
        action_names=[new_acts],
        action_space=[_DummySpace(len(new_acts))],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )

    assert agent._bc is not None
    assert teacher.mutable_state == {"history": ["before-flush"]}
    assert ppo_updates == 0
    assert len(agent._per_building[0].buffer) == 0
    assert agent._per_building[0].action_names_tuple == tuple(new_acts)
    assert agent._bc.teacher_action_buffers == [[]]
    assert agent._bc.latest_teacher_actions is None
    assert any("Discarding invalid one-sample PPO rollout" in warning for warning in warnings)


def test_episode_end_flush_failure_restores_all_state_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, _, _, obs_dim = _make_agent(config=_bc_config(), n_buildings=2)
    assert agent._bc is not None
    teacher_actions = [
        [0.2 for _ in range(state.layout.n_ca)]
        for state in agent._per_building
    ]
    _set_fake_teacher(agent, teacher_actions)
    rng = np.random.default_rng(8)

    for step in range(agent._minibatch_size):
        observations, _, next_observations = _random_transition_for_all_buildings(
            agent, obs_dim, rng
        )
        agent.set_observation_context(
            raw_observations=observations,
            encoded_observations=observations,
        )
        agent.update(
            observations=observations,
            actions=agent.predict(observations, deterministic=False),
            rewards=[0.1, 0.2],
            next_observations=next_observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )

    # Keep decisions live so a failed lifecycle boundary must preserve retry input.
    observations, _, _ = _random_transition_for_all_buildings(agent, obs_dim, rng)
    agent.set_observation_context(
        raw_observations=observations,
        encoded_observations=observations,
    )
    agent.predict(observations, deterministic=False)

    states = list(agent._per_building)
    pending = list(agent._pending_decisions)
    model_states = [
        {
            "tokenizer": {k: v.detach().clone() for k, v in state.tokenizer.state_dict().items()},
            "backbone": {k: v.detach().clone() for k, v in state.backbone.state_dict().items()},
            "actor": {k: v.detach().clone() for k, v in state.actor.state_dict().items()},
            "critic": {k: v.detach().clone() for k, v in state.critic.state_dict().items()},
        }
        for state in states
    ]
    optimizer_states = [deepcopy(state.optimizer.state_dict()) for state in states]
    buffer_states = [deepcopy(state.buffer) for state in states]
    normalizer_states = [state.value_normalizer.state_dict() for state in states]
    rollout_bookkeeping = [
        (
            None
            if state.last_next_observation is None
            else state.last_next_observation.detach().clone(),
            state.last_transition_terminated,
        )
        for state in states
    ]
    training_metrics = dict(agent._latest_training_metrics)
    bc = agent._bc
    teacher = bc.teacher_policy
    teacher.mutable_state = {"history": ["before-episode-end"]}
    bc_buffers = [[None if row is None else list(row) for row in buffer] for buffer in bc.teacher_action_buffers]
    bc_latest_actions = [list(row) for row in bc.latest_teacher_actions or []]
    bc_diagnostics = (
        bc.phaseout_step,
        bc._latest_bc_effective_weight,
        bc._latest_bc_loss,
        bc._latest_bc_weighted_loss,
        bc._latest_bc_valid_samples,
        bc._latest_phaseout_probability,
        bc._latest_phaseout_used,
    )

    original_flush = agent._flush_rollout_boundary

    def fail_second_flush(building_idx, state, *, boundary, last_value=None):
        if building_idx == 1:
            raise RuntimeError("episode flush failed")
        return original_flush(
            building_idx,
            state,
            boundary=boundary,
            last_value=last_value,
        )

    monkeypatch.setattr(agent, "_flush_rollout_boundary", fail_second_flush)
    with pytest.raises(RuntimeError, match="episode flush failed"):
        agent.on_episode_end(episode=0, training=True)

    assert all(state is expected for state, expected in zip(agent._per_building, states))
    assert all(current is expected for current, expected in zip(agent._pending_decisions, pending))
    for state, expected_models, expected_optimizer, expected_buffer, expected_normalizer, expected_bookkeeping in zip(
        agent._per_building,
        model_states,
        optimizer_states,
        buffer_states,
        normalizer_states,
        rollout_bookkeeping,
    ):
        for name, module in (
            ("tokenizer", state.tokenizer),
            ("backbone", state.backbone),
            ("actor", state.actor),
            ("critic", state.critic),
        ):
            for key, value in module.state_dict().items():
                assert torch.equal(value, expected_models[name][key])
        torch.testing.assert_close(state.optimizer.state_dict(), expected_optimizer)
        assert state.buffer.gamma == expected_buffer.gamma
        assert state.buffer.gae_lambda == expected_buffer.gae_lambda
        assert state.buffer.rewards == expected_buffer.rewards
        assert state.buffer.terminated == expected_buffer.terminated
        assert state.buffer.truncated == expected_buffer.truncated
        for name in (
            "observations",
            "actions",
            "pre_tanh_actions",
            "log_probs",
            "values",
        ):
            actual = getattr(state.buffer, name)
            expected = getattr(expected_buffer, name)
            assert len(actual) == len(expected)
            assert all(torch.equal(value, saved) for value, saved in zip(actual, expected))
        for name in ("advantages", "returns"):
            actual = getattr(state.buffer, name)
            expected = getattr(expected_buffer, name)
            assert (actual is None) == (expected is None)
            if actual is not None:
                assert torch.equal(actual, expected)
        assert state.value_normalizer.state_dict() == expected_normalizer
        expected_observation, expected_terminated = expected_bookkeeping
        assert state.last_transition_terminated == expected_terminated
        if expected_observation is None:
            assert state.last_next_observation is None
        else:
            assert state.last_next_observation is not None
            assert torch.equal(state.last_next_observation, expected_observation)
    assert agent._latest_training_metrics == training_metrics
    assert agent._bc is bc
    assert bc.teacher_policy is teacher
    assert teacher.mutable_state == {"history": ["before-episode-end"]}
    assert bc.teacher_action_buffers == bc_buffers
    assert bc.latest_teacher_actions == bc_latest_actions
    assert (
        bc.phaseout_step,
        bc._latest_bc_effective_weight,
        bc._latest_bc_loss,
        bc._latest_bc_weighted_loss,
        bc._latest_bc_valid_samples,
        bc._latest_phaseout_probability,
        bc._latest_phaseout_used,
    ) == bc_diagnostics

    monkeypatch.setattr(agent, "_flush_rollout_boundary", original_flush)
    agent.on_episode_end(episode=0, training=True)

    assert [len(state.buffer) for state in agent._per_building] == [0, 0]
    assert agent._pending_decisions == [None, None]
    assert agent._bc is bc
    assert bc.teacher_action_buffers == [[], []]


def test_cardinality_change_flushes_old_rollouts_before_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(
        config=_bc_config(), n_buildings=2
    )
    assert agent._bc is not None
    teacher_actions = [
        [0.2 for _ in range(state.layout.n_ca)]
        for state in agent._per_building
    ]
    _set_fake_teacher(agent, teacher_actions)
    rng = np.random.default_rng(6)

    for step in range(2):
        obs, _, next_obs = _random_transition_for_all_buildings(agent, obs_dim, rng)
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        actions = agent.predict(obs, deterministic=False)
        agent.update(
            observations=obs,
            actions=actions,
            rewards=[0.1, 0.2],
            next_observations=next_obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )

    old_states = list(agent._per_building)
    assert [len(state.buffer) for state in old_states] == [2, 2]
    assert [len(buffer) for buffer in agent._bc.teacher_action_buffers] == [2, 2]
    updates: list[tuple[object, int, torch.Tensor, int, int]] = []
    original_update = agent._run_ppo_update_with_last_value

    def record_topology_flush(state, last_value, *, building_idx):
        updates.append(
            (
                state,
                building_idx,
                last_value.detach().clone(),
                len(state.buffer),
                len(agent._bc.teacher_action_buffers[building_idx]),
            )
        )
        return original_update(state, last_value, building_idx=building_idx)

    monkeypatch.setattr(
        agent, "_run_ppo_update_with_last_value", record_topology_flush
    )
    agent.attach_environment(
        observation_names=[obs_per[0]],
        action_names=[act_per[0]],
        action_space=[_DummySpace(len(act_per[0]))],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    assert [(state, building_idx, size, teacher_size) for state, building_idx, _, size, teacher_size in updates] == [
        (old_states[0], 0, 2, 2),
        (old_states[1], 1, 2, 2),
    ]
    assert all(torch.equal(last_value, torch.zeros_like(last_value)) for _, _, last_value, _, _ in updates)
    assert len(agent._per_building) == 1
    assert agent._per_building[0] is not old_states[0]
    assert agent._bc.teacher_action_buffers == [[]]
    metrics = agent.consume_latest_training_metrics()
    assert metrics["behavior_cloning_valid_samples"] == 2.0


def test_cardinality_change_discards_one_sample_rollouts_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=2)
    observations = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]
    actions = agent.predict(observations, deterministic=False)
    agent.update(
        observations=observations,
        actions=actions,
        rewards=[0.1, 0.2],
        next_observations=observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )

    old_states = list(agent._per_building)
    warnings: list[str] = []
    monkeypatch.setattr(
        "algorithms.agents.agent_transformer_ppo.logger.warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    agent.attach_environment(
        observation_names=[obs_per[0]],
        action_names=[act_per[0]],
        action_space=[None],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )

    assert [len(state.buffer) for state in old_states] == [0, 0]
    assert sum(
        "Discarding invalid one-sample PPO rollout" in warning
        and "rollout_boundary=topology_change" in warning
        for warning in warnings
    ) == 2


def test_cardinality_change_flush_failure_restores_old_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(
        config=_bc_config(), n_buildings=2
    )
    assert agent._bc is not None
    teacher_actions = [
        [0.2 for _ in range(state.layout.n_ca)]
        for state in agent._per_building
    ]
    _set_fake_teacher(agent, teacher_actions)
    rng = np.random.default_rng(7)

    for step in range(2):
        obs, _, next_obs = _random_transition_for_all_buildings(agent, obs_dim, rng)
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        actions = agent.predict(obs, deterministic=False)
        agent.update(
            observations=obs,
            actions=actions,
            rewards=[0.1, 0.2],
            next_observations=next_obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )

    old_states = list(agent._per_building)
    old_teacher = agent._bc.teacher_policy
    original_update = agent._run_ppo_update_with_last_value

    def fail_later_flush(state, last_value, *, building_idx):
        if building_idx == 1:
            raise RuntimeError("cardinality flush failed")
        return original_update(state, last_value, building_idx=building_idx)

    monkeypatch.setattr(agent, "_run_ppo_update_with_last_value", fail_later_flush)
    with pytest.raises(RuntimeError, match="cardinality flush failed"):
        agent.attach_environment(
            observation_names=[obs_per[0]],
            action_names=[act_per[0]],
            action_space=[_DummySpace(len(act_per[0]))],
            observation_space=[None],
            metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
        )

    assert agent._per_building == old_states
    assert [len(state.buffer) for state in agent._per_building] == [2, 2]
    assert agent._bc is not None
    assert agent._bc.teacher_policy is old_teacher
    assert [len(buffer) for buffer in agent._bc.teacher_action_buffers] == [2, 2]


def test_partial_topology_change_preserves_unchanged_bc_buffer_alignment() -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(config=_bc_config(), n_buildings=2)
    assert agent._bc is not None
    teacher_actions = [
        [0.1 for _ in range(state.layout.n_ca)]
        for state in agent._per_building
    ]
    _set_fake_teacher(agent, teacher_actions)
    rng = np.random.default_rng(4)

    for step in range(2):
        obs, actions, next_obs = _random_transition_for_all_buildings(agent, obs_dim, rng)
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        actions = agent.predict(obs, deterministic=bool(step % 2))
        agent.update(
            observations=obs,
            actions=actions,
            rewards=[0.1, 0.2],
            next_observations=next_obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )

    assert [len(state.buffer) for state in agent._per_building] == [2, 2]
    assert [len(buffer) for buffer in agent._bc.teacher_action_buffers] == [2, 2]

    orig_id = next(
        n.split("::")[1]
        for n in obs_per[0]
        if n.startswith("charger::") and "::connected_ev::" not in n and "::incoming_ev::" not in n
    )
    new_id = "Building_1/charger_PARTIAL_BC_NEW"
    new_obs_0 = list(obs_per[0])
    new_obs_0.extend(
        n.replace(f"charger::{orig_id}::", f"charger::{new_id}::", 1)
        for n in obs_per[0]
        if n.startswith(f"charger::{orig_id}::")
    )
    new_acts_0 = list(act_per[0]) + ["electric_vehicle_storage"]

    agent.attach_environment(
        observation_names=[new_obs_0, list(obs_per[1])],
        action_names=[new_acts_0, list(act_per[1])],
        action_space=[_DummySpace(len(new_acts_0)), _DummySpace(len(act_per[1]))],
        observation_space=[None, None],
        metadata={"building_names": ["Building_1", "Building_2"]},
    )

    assert [len(state.buffer) for state in agent._per_building] == [0, 2]
    assert [len(buffer) for buffer in agent._bc.teacher_action_buffers] == [0, 2]
