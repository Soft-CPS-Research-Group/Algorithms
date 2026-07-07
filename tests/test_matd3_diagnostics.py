"""Plan D diagnostics tests for AgentTransformerMATD3."""
from __future__ import annotations

from tests._matd3_test_helpers import (
    _make_matd3_full,
    _generate_transition,
    _run_update_step,
)


class TestDiagnostics:
    def _trained_agent(self, n_transitions=10):
        agent, _, _, obs_dim = _make_matd3_full(n_buildings=2)
        for step in range(n_transitions):
            obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
            _run_update_step(
                agent, obs, actions, rewards, next_obs, term, trunc,
                global_learning_step=step + 10,
                update_step=True,
                update_target_step=(step % 2 == 0),
                initial_exploration_done=True,
            )
        return agent

    def test_diagnostics_namespace(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        for key in metrics:
            assert key.startswith("TransformerMATD3/"), f"Key {key} not in namespace"

    def test_replay_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/replay_size" in metrics
        assert "TransformerMATD3/active_partition_size" in metrics
        assert "TransformerMATD3/partition_count" in metrics

    def test_critic_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/critic_q1_loss" in metrics
        assert "TransformerMATD3/critic_q2_loss" in metrics
        assert "TransformerMATD3/critic_q_gap" in metrics
        assert "TransformerMATD3/target_q_mean" in metrics
        assert "TransformerMATD3/target_q_std" in metrics

    def test_actor_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/actor_loss" in metrics
        assert "TransformerMATD3/actor_grad_norm" in metrics

    def test_teacher_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/teacher_alive" in metrics
        assert "TransformerMATD3/residual_scale" in metrics

    def test_bc_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/bc_loss" in metrics
        assert "TransformerMATD3/bc_effective_weight" in metrics

    def test_reward_norm_diagnostics_present(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/reward_norm_mean" in metrics
        assert "TransformerMATD3/reward_norm_std" in metrics

    def test_critic_action_input_mode_reported(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        assert "TransformerMATD3/critic_action_input_mode_final" in metrics

    def test_diagnostics_are_floats(self):
        agent = self._trained_agent()
        metrics = agent._collect_diagnostics(global_learning_step=20)
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"
