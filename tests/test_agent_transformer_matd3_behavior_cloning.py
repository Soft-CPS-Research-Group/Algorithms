from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from algorithms.registry import ALGORITHM_REGISTRY
from tests.test_agent_transformer_matd3 import _parameters
from tests.test_agent_transformer_matd3_residual import _agent


def _bc_b_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "demonstration_episodes": 1,
        "max_samples_per_building": 8,
        "pretraining_epochs": 1,
        "batch_size": 2,
        "weight": 1.0,
        "min_weight": 0.0,
        "decay_start_step": 0,
        "decay_steps": 0,
        "ev_multiplier": 1.0,
        "storage_multiplier": 1.0,
        "teacher": {"policy": "RBCSmartPolicy", "hyperparameters": {}},
    }
    config.update(overrides)
    return config


def _bc_b_agent(**overrides):
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3
    from tests._entity_sample_obs_names import (
        load_sample_observation_names_for_first_building,
    )
    from tests.test_agent_transformer_matd3 import _ACTION_NAMES, _Box, _config

    config = _config()
    config["algorithm"]["behavior_cloning"] = {
        "demonstration_based": _bc_b_config(**overrides)
    }
    names = load_sample_observation_names_for_first_building()
    agent = AgentTransformerMATD3(config)
    agent.attach_environment(
        observation_names=[list(names)],
        action_names=[list(_ACTION_NAMES)],
        action_space=[_Box([-2.0, -0.5], [1.0, 0.75])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    return agent, len(names)


def _same(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return all(
        torch.equal(previous, current.detach())
        for previous, current in zip(before, module.parameters())
    )


def test_bc_b_keeps_agent_unregistered_and_is_disabled_by_default() -> None:
    agent, _ = _agent()

    assert "AgentTransformerMATD3" not in ALGORITHM_REGISTRY
    assert agent._bc_b is None
    assert agent._per_building[0].bc_b_optimizer is None


def test_bc_b_collects_immutable_normalized_demonstration_without_rl_state() -> None:
    agent, obs_dim = _bc_b_agent()
    assert agent._bc_b is not None

    class _Teacher:
        def predict(self, observations, deterministic):
            del observations, deterministic
            return [[-0.5, 0.75]]

    agent._bc_b.teacher_policy = _Teacher()
    state = agent._per_building[0]
    observation = np.zeros(obs_dim, dtype=np.float32)
    critic_before = _parameters(state.critic_1)
    normalizer_before = (
        agent.reward_norm_count,
        agent.reward_norm_mean,
        agent.reward_norm_m2,
    )
    replay_before = agent.replay_buffer.get_state()
    agent.on_episode_start(episode=0, training=True)

    actions = agent.predict([observation], deterministic=False)
    agent.update(
        [observation],
        actions,
        [10.0],
        [observation],
        False,
        False,
        update_target_step=True,
        global_learning_step=0,
        update_step=True,
        initial_exploration_done=True,
    )

    demonstration = agent._bc_b.demonstrations_for_building_by_signature(0)
    stored = next(iter(demonstration.values()))[0]
    observation[0] = 99.0
    assert actions == [[-0.5, 0.75]]
    assert stored.target.tolist() == pytest.approx([0.0, 1.0])
    assert stored.observation[0] == 0.0
    assert not stored.observation.flags.writeable
    assert not stored.target.flags.writeable
    assert _same(critic_before, state.critic_1)
    assert normalizer_before == (
        agent.reward_norm_count,
        agent.reward_norm_mean,
        agent.reward_norm_m2,
    )
    assert replay_before["transitions"] == agent.replay_buffer.get_state()[
        "transitions"
    ]


def test_bc_b_reservoir_capacity_applies_per_building() -> None:
    agent, obs_dim = _bc_b_agent(max_samples_per_building=2)
    assert agent._bc_b is not None
    layout = agent._per_building[0].layout

    for value in range(5):
        agent._bc_b.record_demonstration(
            0,
            np.full(obs_dim, value, dtype=np.float32),
            layout,
            [0.0, 0.0],
        )

    assert agent._bc_b.demonstration_count(0) == 2
    assert agent._bc_b.state_dict()["seen_per_building"] == {0: 5}


def test_bc_b_pretraining_fails_before_rl_when_building_has_no_usable_demo() -> None:
    agent, _ = _bc_b_agent()
    agent.on_episode_start(episode=0, training=True)

    with pytest.raises(RuntimeError, match="zero usable demonstrations"):
        agent.on_episode_end(episode=0, training=True)

    assert not agent._bc_b_pretraining_complete
    assert agent.replay_buffer.total_size() == 0


def test_bc_b_missing_pretraining_fails_before_first_rl_episode() -> None:
    agent, _ = _bc_b_agent()

    with pytest.raises(RuntimeError, match="zero usable demonstrations"):
        agent.on_episode_start(episode=1, training=True)

    assert not agent._bc_b_pretraining_complete
    assert agent.replay_buffer.total_size() == 0


def test_bc_b_pretraining_trains_current_and_historical_signature_groups(
    monkeypatch,
) -> None:
    agent, obs_dim = _bc_b_agent()
    assert agent._bc_b is not None
    state = agent._per_building[0]
    current_layout = state.layout
    first = current_layout.segments[0]
    historical_layout = replace(
        current_layout,
        segments=(replace(first, instance_id=f"{first.instance_id}_old"),)
        + current_layout.segments[1:],
    )
    agent._bc_b.record_demonstration(
        0, np.zeros(obs_dim, dtype=np.float32), current_layout, [0.0, 0.0]
    )
    agent._bc_b.record_demonstration(
        0, np.ones(obs_dim, dtype=np.float32), historical_layout, [0.1, -0.1]
    )
    trained_signatures = []

    def record_group(**kwargs):
        trained_signatures.append(agent._bc_b.layout_signature(kwargs["layout"]))
        return 0.0, 0.0

    monkeypatch.setattr(agent, "_apply_bc_b_gradient_step", record_group)

    agent._run_bc_b_pretraining()

    assert set(trained_signatures) == set(
        agent._bc_b.demonstrations_for_building_by_signature(0)
    )


def test_bc_b_pretraining_skips_incompatible_historical_groups(monkeypatch) -> None:
    agent, obs_dim = _bc_b_agent()
    assert agent._bc_b is not None
    state = agent._per_building[0]
    current_layout = state.layout
    first = current_layout.segments[0]
    incompatible_layout = replace(
        current_layout,
        segments=(
            replace(
                first,
                feature_indices=first.feature_indices + (obs_dim,),
                feature_names=first.feature_names + ("historical_extra",),
            ),
        )
        + current_layout.segments[1:],
    )
    agent._bc_b.record_demonstration(
        0, np.zeros(obs_dim, dtype=np.float32), current_layout, [0.0, 0.0]
    )
    agent._bc_b.record_demonstration(
        0,
        np.zeros(obs_dim + 1, dtype=np.float32),
        incompatible_layout,
        [0.0, 0.0],
    )
    trained_layouts = []
    monkeypatch.setattr(
        agent,
        "_apply_bc_b_gradient_step",
        lambda **kwargs: (trained_layouts.append(kwargs["layout"]) or (0.0, 0.0)),
    )

    agent._run_bc_b_pretraining()

    assert trained_layouts == [current_layout]
    assert agent._bc_b.snapshot_metrics()[
        "behavior_cloning_incompatible_demonstration_samples"
    ] == 1.0


def test_bc_b_pretraining_changes_only_actor_stack() -> None:
    agent, obs_dim = _bc_b_agent()
    assert agent._bc_b is not None
    state = agent._per_building[0]
    agent._bc_b.record_demonstration(
        0,
        np.zeros(obs_dim, dtype=np.float32),
        state.layout,
        [0.8, -0.8],
    )
    actor_before = _parameters(agent._actor_modules(state))
    critic_before = _parameters(state.critic_1)
    critic_target_before = _parameters(state.critic_1_target)
    replay_before = agent.replay_buffer.get_state()

    agent._run_bc_b_pretraining()

    assert not _same(actor_before, agent._actor_modules(state))
    assert _same(critic_before, state.critic_1)
    assert _same(critic_target_before, state.critic_1_target)
    assert replay_before["transitions"] == agent.replay_buffer.get_state()[
        "transitions"
    ]


def test_bc_a_and_bc_b_use_separate_optimizers() -> None:
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3
    from tests._entity_sample_obs_names import (
        load_sample_observation_names_for_first_building,
    )
    from tests.test_agent_transformer_matd3 import _ACTION_NAMES, _Box, _config

    config = _config()
    config["algorithm"]["behavior_cloning"] = {
        "replay_based": {
            "enabled": True,
            "teacher": "replay_action",
            "weight": 1.0,
        },
        "demonstration_based": _bc_b_config(),
    }
    names = load_sample_observation_names_for_first_building()
    agent = AgentTransformerMATD3(config)
    agent.attach_environment(
        observation_names=[list(names)],
        action_names=[list(_ACTION_NAMES)],
        action_space=[_Box([-2.0, -0.5], [1.0, 0.75])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    state = agent._per_building[0]

    assert state.bc_a_optimizer is not None
    assert state.bc_b_optimizer is not None
    assert state.bc_a_optimizer is not state.bc_b_optimizer


def test_bc_b_zero_weight_short_circuits_before_actor_forward(monkeypatch) -> None:
    agent, _ = _bc_b_agent(weight=0.0)
    agent._bc_b_pretraining_complete = True
    monkeypatch.setattr(
        agent,
        "_apply_bc_b_gradient_step",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("actor forward must not run")
        ),
    )

    assert agent._run_bc_b_auxiliary_updates(global_learning_step=0) == ([], [])


def test_bc_b_teacher_is_rebuilt_for_topology_attachment(monkeypatch) -> None:
    agent, _ = _bc_b_agent()
    assert agent._bc_b is not None
    original = agent._bc_b.teacher_policy
    replacement = object()
    monkeypatch.setattr(
        agent._bc_b,
        "_build_teacher_policy",
        lambda *args, **kwargs: replacement,
    )

    agent._attach_bc_b_environment(
        observation_names=[["unused"]],
        action_names=[["unused"]],
        action_space=[None],
        observation_space=[None],
        metadata=None,
        topology_change=True,
    )

    assert agent._bc_b.teacher_policy is replacement
    assert agent._bc_b.teacher_policy is not original


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
