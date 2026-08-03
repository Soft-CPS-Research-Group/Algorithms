"""Separate demonstration and PPO phase tests for AgentTransformerPPO."""
from __future__ import annotations

import numpy as np
import torch

from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
from tests.test_agent_transformer_ppo import _DEFAULT_ACTIONS, _base_config
from tests._entity_sample_obs_names import load_sample_observation_names_for_first_building


class _DummySpace:
    def __init__(self, size: int) -> None:
        self.low = np.full(size, -1.0, dtype=np.float64)
        self.high = np.full(size, 1.0, dtype=np.float64)


def _config(*, demonstrations: int = 1, weight: float = 0.0) -> dict:
    config = _base_config()
    config["algorithm"]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": demonstrations,
        "max_samples_per_building": 16,
        "pretraining_epochs": 2,
        "batch_size": 1,
        "weight": weight,
        "min_weight": 0.0,
        "decay_start_step": 0,
        "decay_steps": 1,
        "ev_multiplier": 1.0,
        "storage_multiplier": 1.0,
        "teacher": {"policy": "RBCSmartPolicy", "deterministic": True, "hyperparameters": {}},
    }
    return config


def _agent(*, demonstrations: int = 1, weight: float = 0.0) -> tuple[AgentTransformerPPO, int]:
    names = load_sample_observation_names_for_first_building()
    actions = list(_DEFAULT_ACTIONS)
    agent = AgentTransformerPPO(_config(demonstrations=demonstrations, weight=weight))
    agent.attach_environment(
        observation_names=[names], action_names=[actions],
        action_space=[_DummySpace(len(actions))], observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )
    dimension = max(max(segment.feature_indices) for segment in agent._per_building[0].layout.segments) + 1
    return agent, dimension


def _teacher(agent: AgentTransformerPPO, value: float) -> list[list[float]]:
    actions = [[value] * agent._per_building[0].layout.n_ca]
    assert agent._bc is not None
    agent._bc.compute_teacher_actions = lambda _observations: [list(row) for row in actions]
    return actions


def _update(agent: AgentTransformerPPO, observation: np.ndarray, actions, step: int) -> None:
    agent.update(
        observations=[observation], actions=actions, rewards=[0.1],
        next_observations=[observation], terminated=False, truncated=False,
        update_target_step=False, global_learning_step=step, update_step=False,
        initial_exploration_done=True,
    )


def test_demo_episode_executes_teacher_only_records_immutable_demo_and_no_ppo() -> None:
    agent, dimension = _agent()
    teacher_actions = _teacher(agent, 0.25)
    observation = np.ones(dimension, dtype=np.float64)
    agent.on_episode_start(episode=0, training=True)
    agent.set_observation_context(raw_observations=[observation], encoded_observations=[observation])

    actions = agent.predict([observation], deterministic=False)
    _update(agent, observation, actions, 0)

    assert actions == teacher_actions
    assert len(agent._per_building[0].buffer) == 0
    assert agent._pending_decisions == [None]
    assert agent._bc is not None
    demo = next(iter(agent._bc.demonstrations_by_signature.values()))[0]
    observation[0] = 99.0
    assert demo.observation[0] == 1.0
    assert demo.layout is not agent._per_building[0].layout


def test_final_demo_end_pretrains_actor_then_ppo_uses_only_actor_actions() -> None:
    agent, dimension = _agent()
    teacher_actions = _teacher(agent, 0.75)
    observation = np.ones(dimension, dtype=np.float64)
    before = [parameter.detach().clone() for parameter in agent._per_building[0].actor.parameters()]
    agent.on_episode_start(episode=0, training=True)
    agent.set_observation_context(raw_observations=[observation], encoded_observations=[observation])
    _update(agent, observation, agent.predict([observation]), 0)
    agent.on_episode_end(episode=0, training=True)

    after = list(agent._per_building[0].actor.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after))
    metrics = agent.consume_latest_training_metrics()
    assert metrics["behavior_cloning_pretraining_epochs"] == 2.0
    assert metrics["behavior_cloning_demonstration_samples"] == 1.0

    agent.on_episode_start(episode=1, training=True)
    agent.set_observation_context(raw_observations=[observation], encoded_observations=[observation])
    ppo_actions = agent.predict([observation], deterministic=True)
    assert ppo_actions != teacher_actions
    _update(agent, observation, ppo_actions, 1)
    assert len(agent._per_building[0].buffer) == 1


def test_auxiliary_bc_never_changes_ppo_actions() -> None:
    agent, dimension = _agent(demonstrations=0, weight=1.0)
    teacher_actions = _teacher(agent, 0.9)
    observation = np.ones(dimension, dtype=np.float64)
    assert agent._bc is not None
    agent._bc.record_demonstration(0, observation, agent._per_building[0].layout, teacher_actions[0])
    agent.on_episode_start(episode=1, training=True)
    agent.set_observation_context(raw_observations=[observation], encoded_observations=[observation])
    actor_actions = agent.predict([observation], deterministic=True)
    assert actor_actions != teacher_actions


def test_auxiliary_bc_samples_demonstrations_during_ppo_update() -> None:
    agent, dimension = _agent(demonstrations=0, weight=1.0)
    teacher_actions = _teacher(agent, 0.9)
    observation = np.ones(dimension, dtype=np.float64)
    assert agent._bc is not None
    agent._bc.record_demonstration(0, observation, agent._per_building[0].layout, teacher_actions[0])
    sampled = []
    original_sample = agent._bc.sample_demonstrations

    def record_sample(layout, batch_size):
        sampled.append((layout, batch_size))
        return original_sample(layout, batch_size)

    agent._bc.sample_demonstrations = record_sample
    agent.on_episode_start(episode=1, training=True)
    for step in range(agent._minibatch_size):
        agent.set_observation_context(
            raw_observations=[observation], encoded_observations=[observation]
        )
        actions = agent.predict([observation], deterministic=False)
        agent.update(
            observations=[observation], actions=actions, rewards=[0.1],
            next_observations=[observation], terminated=False, truncated=False,
            update_target_step=False, global_learning_step=step,
            update_step=step == agent._minibatch_size - 1,
            initial_exploration_done=True,
        )

    assert sampled
    assert all(batch_size == agent._bc.batch_size for _, batch_size in sampled)
