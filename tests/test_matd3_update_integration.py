"""Plan D integration tests - update loop for AgentTransformerMATD3."""
from __future__ import annotations

import numpy as np

from tests._matd3_test_helpers import (
    _make_matd3_full,
    _generate_transition,
    _run_update_step,
)


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


class TestUpdateGating:
    def test_update_skips_before_initial_exploration(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=0,
            initial_exploration_done=False,
        )
        assert agent._replay is not None
        assert agent._replay.total_size >= 1
        assert agent._critic_update_count == 0

    def test_update_skips_when_replay_too_small(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=10,
            initial_exploration_done=True,
        )
        assert agent._critic_update_count == 0

    def test_update_stores_transition_with_topology_sig(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=10,
            initial_exploration_done=True,
        )
        sig = agent._current_topology_signature()
        assert agent._replay.partition_size(sig) == 1

    def test_update_stores_teacher_actions_in_replay(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        agent.set_transition_context(
            raw_observations=obs,
            raw_next_observations=next_obs,
            encoded_observations=obs,
            encoded_next_observations=next_obs,
        )
        agent.update(
            observations=obs, actions=actions, rewards=rewards,
            next_observations=next_obs, terminated=term, truncated=trunc,
            update_target_step=False, global_learning_step=10,
            update_step=True, initial_exploration_done=True,
        )
        sig = agent._current_topology_signature()
        assert agent._replay.partition_size(sig) >= 1
        agent._replay.batch_size = 1
        agent._replay.set_active_signature(sig)
        batch = agent._replay.sample()
        assert batch is not None
        assert batch.base_actions is not None
        assert batch.next_base_actions is not None
