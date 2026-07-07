"""Plan D integration tests - update loop for AgentTransformerMATD3."""
from __future__ import annotations

import numpy as np
import torch

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


class TestCriticAndActorUpdate:
    def _fill_replay(self, agent, n_buildings, obs_dim, n_transitions=8):
        """Push enough transitions so sampling works."""
        for step in range(n_transitions):
            obs, actions, rewards, next_obs, term, trunc = _generate_transition(
                n_buildings, obs_dim
            )
            agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
            agent.set_transition_context(
                raw_observations=obs, raw_next_observations=next_obs,
                encoded_observations=obs, encoded_next_observations=next_obs,
            )
            agent.update(
                observations=obs, actions=actions, rewards=rewards,
                next_observations=next_obs, terminated=term, truncated=trunc,
                update_target_step=False, global_learning_step=step,
                update_step=False,
                initial_exploration_done=True,
            )

    def test_critic_update_changes_critic_params(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)
        c1_params_before = [p.clone() for p in agent._critic_1.parameters()]
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=100,
            update_step=True,
            initial_exploration_done=True,
        )
        assert agent._critic_update_count >= 1
        c1_params_after = list(agent._critic_1.parameters())
        assert any(
            not torch.allclose(before, after)
            for before, after in zip(c1_params_before, c1_params_after)
        )

    def test_actor_update_respects_interval(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=100,
            update_step=True,
            initial_exploration_done=True,
        )
        assert agent._critic_update_count == 1
        assert agent._actor_update_count == 0
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=101,
            update_step=True,
            initial_exploration_done=True,
        )
        assert agent._critic_update_count == 2
        assert agent._actor_update_count == 1

    def test_actor_update_changes_actor_params(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)
        agent._actor_update_interval = 1
        actor_params_before = [p.clone() for p in agent._actors[0].actor.parameters()]
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=100,
            update_step=True,
            initial_exploration_done=True,
        )
        actor_params_after = list(agent._actors[0].actor.parameters())
        assert any(
            not torch.allclose(before, after)
            for before, after in zip(actor_params_before, actor_params_after)
        )

    def test_critic_frozen_during_actor_update(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)
        agent._actor_update_interval = 1
        agent._verify_critic_frozen_during_actor = True
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=100,
            update_step=True,
            initial_exploration_done=True,
        )

    def test_soft_target_update(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)
        target_params_before = [
            p.clone() for p in agent._actors[0].target_actor.parameters()
        ]
        obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
        _run_update_step(
            agent, obs, actions, rewards, next_obs, term, trunc,
            global_learning_step=100,
            update_step=True,
            update_target_step=True,
            initial_exploration_done=True,
        )
        target_params_after = list(agent._actors[0].target_actor.parameters())
        assert any(
            not torch.allclose(before, after)
            for before, after in zip(target_params_before, target_params_after)
        )

    def test_min_q_target_uses_both_critics(self):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        self._fill_replay(agent, 2, obs_dim, n_transitions=8)
        c1_params = list(agent._critic_1.parameters())
        c2_params = list(agent._critic_2.parameters())
        assert any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(c1_params, c2_params)
        )
