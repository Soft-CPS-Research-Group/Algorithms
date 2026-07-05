"""Behavior-cloning integration tests for AgentTransformerPPO."""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

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


def test_bc_present_attaches_teacher_policy() -> None:
    agent, _, _, _ = _make_agent(config=_bc_config())

    assert agent._bc is not None
    assert isinstance(agent._bc.teacher_policy, RBCCommunityPolicy)


def test_predict_computes_teacher_actions_and_blends_phaseout() -> None:
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

    assert agent._bc is not None
    assert agent._bc.latest_teacher_actions == teacher_actions
    metrics = agent._bc.snapshot_metrics()
    assert metrics["behavior_cloning_phaseout_probability"] > 0.0
    assert metrics["behavior_cloning_phaseout_used"] == pytest.approx(1.0)
    assert actions != teacher_actions


def test_update_records_teacher_actions_aligned_with_buffer() -> None:
    agent, _, _, obs_dim = _make_agent(config=_bc_config())
    teacher_actions = [[0.1 for _ in range(agent._per_building[0].layout.n_ca)]]
    _set_fake_teacher(agent, teacher_actions)
    rng = np.random.default_rng(1)

    for step in range(3):
        obs, actions, next_obs = _random_transition(agent, obs_dim, rng)
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        agent.predict(obs, deterministic=bool(step % 2))
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
        agent.predict(obs, deterministic=False)
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
    agent.predict(obs, deterministic=False)
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
        agent.predict(obs, deterministic=bool(step % 2))
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
