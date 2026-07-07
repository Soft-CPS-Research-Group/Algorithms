"""Plan D - Dynamic topology integration smoke tests for AgentTransformerMATD3."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._matd3_test_helpers import (
    _make_matd3_full,
    _generate_transition,
    _run_update_step,
    _add_charger_to_building_obs,
)


class TestDynamicTopologySmoke:
    def _build_agent_and_train(self, n_steps=8):
        """Create agent, run n_steps of predict+update."""
        agent, obs_per, act_per, obs_dim = _make_matd3_full(n_buildings=2)
        for step in range(n_steps):
            obs, actions, rewards, next_obs, term, trunc = _generate_transition(2, obs_dim)
            _run_update_step(
                agent, obs, actions, rewards, next_obs, term, trunc,
                global_learning_step=step + 10,
                update_step=True,
                update_target_step=(step % 2 == 0),
                initial_exploration_done=True,
            )
        return agent, obs_per, act_per, obs_dim

    def test_topology_change_switches_replay_signature(self):
        agent, obs_per, act_per, _ = self._build_agent_and_train()
        sig_before = agent._current_topology_signature()
        replay_size_before = agent._replay.partition_size(sig_before)
        assert replay_size_before > 0
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )
        sig_after = agent._current_topology_signature()
        assert sig_after != sig_before
        assert agent._replay.partition_size(sig_before) == replay_size_before
        assert agent._replay.partition_size(sig_after) == 0

    def test_actor_weights_survive_topology_change(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train()
        b1_params_before = [p.clone() for p in agent._actors[1].actor.parameters()]
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )
        for before, after in zip(b1_params_before, agent._actors[1].actor.parameters()):
            assert torch.allclose(before, after)
        obs = [
            np.random.randn(len(new_obs_0)).astype(np.float64),
            np.random.randn(obs_dim).astype(np.float64),
        ]
        actions = agent.predict(obs, deterministic=True)
        assert len(actions[0]) == 3
        assert len(actions[1]) == 2

    def test_critic_weights_survive_topology_change(self):
        agent, obs_per, act_per, _ = self._build_agent_and_train()
        c1_params_before = [p.clone() for p in agent._critic_1.parameters()]
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )
        for before, after in zip(c1_params_before, agent._critic_1.parameters()):
            assert torch.allclose(before, after)

    def test_teacher_reattaches_on_topology_change(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train()
        assert agent._teacher_alive is True
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )
        assert agent._teacher_alive is True
        obs = [
            np.random.randn(len(new_obs_0)).astype(np.float64),
            np.random.randn(obs_dim).astype(np.float64),
        ]
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        assert agent._latest_teacher_actions is not None

    def test_training_continues_after_topology_change(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train(n_steps=8)
        critic_updates_before = agent._critic_update_count
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )
        for step in range(10):
            obs = [
                np.random.randn(len(new_obs_0)).astype(np.float64),
                np.random.randn(obs_dim).astype(np.float64),
            ]
            actions = [
                np.random.uniform(-1, 1, size=3).astype(np.float64),
                np.random.uniform(-1, 1, size=2).astype(np.float64),
            ]
            next_obs = [
                np.random.randn(len(new_obs_0)).astype(np.float64),
                np.random.randn(obs_dim).astype(np.float64),
            ]
            _run_update_step(
                agent, obs, actions, [float(np.random.randn()), float(np.random.randn())],
                next_obs, False, False,
                global_learning_step=100 + step,
                update_step=True,
                initial_exploration_done=True,
            )
        assert agent._critic_update_count > critic_updates_before

    def test_update_skips_until_new_partition_has_batch_size(self):
        agent, obs_per, act_per, obs_dim = self._build_agent_and_train(n_steps=8)
        critic_updates_at_change = agent._critic_update_count
        new_obs_0, new_action = _add_charger_to_building_obs(
            obs_per[0], building_id="Building_0", new_charger_id="new_charger"
        )
        new_act_0 = list(act_per[0]) + [new_action]
        agent.attach_environment(
            observation_names=[new_obs_0, obs_per[1]],
            action_names=[new_act_0, act_per[1]],
            action_space=[None, None],
            observation_space=[None, None],
        )
        obs = [np.random.randn(len(new_obs_0)), np.random.randn(obs_dim)]
        actions = [np.random.uniform(-1, 1, size=3), np.random.uniform(-1, 1, size=2)]
        _run_update_step(
            agent, obs, actions, [0.1, 0.2],
            [np.random.randn(len(new_obs_0)), np.random.randn(obs_dim)],
            False, False,
            global_learning_step=200,
            update_step=True,
            initial_exploration_done=True,
        )
        assert agent._critic_update_count == critic_updates_at_change

    def test_export_after_training_only_actors(self):
        agent, _, _, _ = self._build_agent_and_train()
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = agent.export_artifacts(tmpdir)
            assert manifest["format"] == "onnx"
            for art in manifest["artifacts"]:
                assert "critic" not in art["path"].lower()
                assert Path(tmpdir, art["path"]).exists()
            manifest_str = str(manifest).lower()
            assert "critic" not in manifest_str or "critic_action_input_mode" not in manifest_str

    def test_building_count_change_fails_fast(self):
        agent, obs_per, act_per, _ = self._build_agent_and_train()
        with pytest.raises(ValueError, match="[Bb]uilding.count"):
            agent.attach_environment(
                observation_names=[obs_per[0]],
                action_names=[act_per[0]],
                action_space=[None],
                observation_space=[None],
            )
