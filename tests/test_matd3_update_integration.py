"""Plan D integration tests - update loop for AgentTransformerMATD3."""
from __future__ import annotations

import numpy as np

from tests._matd3_test_helpers import _make_matd3_full


class TestContextHooks:
    def test_set_observation_context_stores_raw(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        assert agent._latest_raw_observations is not None
        assert len(agent._latest_raw_observations) == 2
        assert np.allclose(agent._latest_raw_observations[0], obs[0])

    def test_set_observation_context_stores_encoded(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        assert agent._latest_encoded_observations is not None
        assert len(agent._latest_encoded_observations) == 2

    def test_set_transition_context_stores_next(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        agent.set_transition_context(
            raw_observations=obs,
            raw_next_observations=next_obs,
            encoded_observations=obs,
            encoded_next_observations=next_obs,
        )
        assert agent._latest_raw_next_observations is not None
        assert len(agent._latest_raw_next_observations) == 2
        assert agent._latest_encoded_next_observations is not None

    def test_set_transition_context_computes_teacher_actions(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        agent.set_transition_context(
            raw_observations=obs,
            raw_next_observations=next_obs,
            encoded_observations=obs,
            encoded_next_observations=next_obs,
        )
        assert agent._latest_teacher_actions is not None
        assert agent._latest_next_teacher_actions is not None
        assert len(agent._latest_teacher_actions) == 2
        assert len(agent._latest_next_teacher_actions) == 2

    def test_context_hook_noop_when_teacher_released(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        agent._warm_start_policy = None
        agent._teacher_alive = False
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        agent.set_transition_context(
            raw_observations=obs,
            raw_next_observations=next_obs,
            encoded_observations=obs,
            encoded_next_observations=next_obs,
        )
        assert agent._latest_teacher_actions is None
        assert agent._latest_next_teacher_actions is None
