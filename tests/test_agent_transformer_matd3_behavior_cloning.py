from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.test_agent_transformer_matd3 import _parameters
from tests.test_agent_transformer_matd3_residual import _agent


def _same(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return all(
        torch.equal(previous, current.detach())
        for previous, current in zip(before, module.parameters())
    )


def test_bc_a_extra_update_changes_only_actor_stack() -> None:
    agent, obs_dim = _agent(
        replay_bc={
            "enabled": True,
            "teacher": "replay_action",
            "weight": 1.0,
            "extra_updates": 1,
        }
    )
    state = agent._per_building[0]
    observations = [torch.zeros((2, obs_dim), device=agent.device)]
    cloning_actions = [torch.ones((2, 2), device=agent.device) * 0.7]
    actor_before = _parameters(agent._actor_modules(state))
    critic_1_before = _parameters(state.critic_1)
    critic_2_before = _parameters(state.critic_2)
    critic_1_target_before = _parameters(state.critic_1_target)
    critic_2_target_before = _parameters(state.critic_2_target)
    normalizer_before = (
        agent.reward_norm_count,
        agent.reward_norm_mean,
        agent.reward_norm_m2,
    )
    replay_before = agent.replay_buffer.get_state()

    losses, _ = agent._run_bc_a_extra_updates(
        observations=observations,
        behavior_actions=None,
        cloning_actions=cloning_actions,
        effective_weight=1.0,
        global_learning_step=0,
    )

    assert losses
    assert not _same(actor_before, agent._actor_modules(state))
    assert _same(critic_1_before, state.critic_1)
    assert _same(critic_2_before, state.critic_2)
    assert _same(critic_1_target_before, state.critic_1_target)
    assert _same(critic_2_target_before, state.critic_2_target)
    assert normalizer_before == (
        agent.reward_norm_count,
        agent.reward_norm_mean,
        agent.reward_norm_m2,
    )
    assert replay_before["transitions"] == agent.replay_buffer.get_state()[
        "transitions"
    ]


def test_bc_a_zero_weight_short_circuits_before_actor_forward(monkeypatch) -> None:
    agent, obs_dim = _agent(
        replay_bc={
            "enabled": True,
            "teacher": "replay_action",
            "weight": 0.0,
            "extra_updates": 1,
        }
    )
    monkeypatch.setattr(
        agent,
        "_policy_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("actor forward must not run")
        ),
    )

    losses, gradients = agent._run_bc_a_extra_updates(
        observations=[torch.zeros((1, obs_dim), device=agent.device)],
        behavior_actions=None,
        cloning_actions=[torch.zeros((1, 2), device=agent.device)],
        effective_weight=0.0,
        global_learning_step=0,
    )

    assert losses == []
    assert gradients == []


def test_bc_a_disabled_allocates_no_optimizer_or_optional_replay_fields() -> None:
    agent, obs_dim = _agent()
    state = agent._per_building[0]
    observations = [np.zeros(obs_dim, dtype=np.float32)]
    actions = agent.predict(observations, deterministic=True)
    agent.update(
        observations,
        actions,
        [0.0],
        observations,
        False,
        False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=False,
    )

    transition = agent.replay_buffer.get_state()["transitions"][0]
    assert state.bc_a_optimizer is None
    assert transition.behavior_actions is None
    assert transition.next_behavior_actions is None
    assert transition.cloning_actions is None


def test_bc_a_weight_decay_reaches_configured_floor() -> None:
    agent, _ = _agent(
        replay_bc={
            "enabled": True,
            "teacher": "replay_action",
            "weight": 1.0,
            "min_weight": 0.2,
            "decay_start_step": 5,
            "decay_steps": 10,
        }
    )

    assert agent._bc_a_effective_weight(5) == 1.0
    assert agent._bc_a_effective_weight(10) == pytest.approx(0.6)
    assert agent._bc_a_effective_weight(20) == pytest.approx(0.2)


def test_bc_a_external_targets_use_lazy_cloning_replay_field() -> None:
    agent, obs_dim = _agent(
        replay_bc={
            "enabled": True,
            "teacher": "external",
            "weight": 1.0,
        }
    )
    observations = [np.zeros(obs_dim, dtype=np.float32)]
    actions = agent.predict(observations, deterministic=True)
    cloning_actions = [np.asarray([0.8, 0.6], dtype=np.float32)]
    agent.set_transition_context(cloning_actions=cloning_actions)

    agent.update(
        observations,
        actions,
        [0.0],
        observations,
        False,
        False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=False,
    )

    transition = agent.replay_buffer.get_state()["transitions"][0]
    assert transition.behavior_actions is None
    assert transition.cloning_actions is not None
    assert transition.cloning_actions[0].tolist() == pytest.approx(
        cloning_actions[0].tolist()
    )


def test_missing_warm_start_bc_a_context_does_not_block_replay() -> None:
    agent, _ = _agent(
        hyperparameters={"warm_start_policy_name": "RandomPolicy"},
        replay_bc={
            "enabled": True,
            "teacher": "warm_start",
            "weight": 1.0,
        },
    )
    actions = [[0.2, -0.1]]

    behavior_actions = agent._transition_behavior_actions(actions)
    next_behavior_actions = agent._transition_next_behavior_actions(
        behavior_actions
    )
    cloning_actions = agent._transition_cloning_actions(
        actions,
        base_actions=behavior_actions,
    )

    assert behavior_actions == actions
    assert next_behavior_actions == actions
    assert cloning_actions == actions
