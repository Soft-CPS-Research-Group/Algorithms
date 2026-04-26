from __future__ import annotations

import numpy as np

from algorithms.agents.maddpg_agent import MADDPG


def _build_agent_for_exploration() -> MADDPG:
    agent = MADDPG.__new__(MADDPG)
    agent.num_agents = 2
    agent.action_dimension = [2, 1]
    agent.exploration_step = 0
    agent.random_exploration_steps = 0
    agent.sigma = 0.2
    agent.sigma_decay = 0.5
    agent.min_sigma = 0.1
    agent.bias = 0.0
    agent.noise_clip = None
    return agent


def test_predict_with_exploration_uses_random_actions_during_warmup():
    agent = _build_agent_for_exploration()
    agent.random_exploration_steps = 2

    actions = agent._predict_with_exploration(observations=[None, None])

    assert len(actions) == 2
    assert len(actions[0]) == 2
    assert len(actions[1]) == 1
    assert all(-1.0 <= value <= 1.0 for row in actions for value in row)
    # Sigma does not decay during pure random warmup.
    assert agent.sigma == 0.2


def test_predict_with_exploration_applies_noise_clip_and_sigma_decay(monkeypatch):
    agent = _build_agent_for_exploration()
    agent.noise_clip = 0.15

    monkeypatch.setattr(agent, "_predict_deterministic", lambda _obs: [np.array([0.0, 0.0]), np.array([0.0])])
    monkeypatch.setattr(np.random, "normal", lambda loc, scale, size: np.full(size, 0.4, dtype=np.float64))

    actions = agent._predict_with_exploration(observations=[None, None])

    assert actions[0] == [0.15, 0.15]
    assert actions[1] == [0.15]
    # Sigma decays once warmup is over.
    assert agent.sigma == 0.1
