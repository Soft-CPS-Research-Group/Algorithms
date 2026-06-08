from __future__ import annotations

import torch

from algorithms.utils.networks import (
    Critic,
    LateFusionCritic,
    MultiHeadActor,
    SemanticMultiHeadActor,
    build_actor_network,
    build_critic_network,
)


def test_critic_outputs_scalar_q_value_with_multiple_hidden_layers() -> None:
    critic = Critic(state_size=10, action_size=4, seed=1, fc_units=[32, 16])
    states = torch.randn(5, 10)
    actions = torch.randn(5, 4)

    q_values = critic(states, actions)

    assert q_values.shape == (5, 1)


def test_critic_outputs_scalar_q_value_with_single_hidden_layer() -> None:
    critic = Critic(state_size=6, action_size=2, seed=1, fc_units=[16])
    states = torch.randn(3, 6)
    actions = torch.randn(3, 2)

    q_values = critic(states, actions)

    assert q_values.shape == (3, 1)


def test_late_fusion_critic_outputs_scalar_q_value() -> None:
    critic = LateFusionCritic(
        state_size=10,
        action_size=4,
        seed=1,
        fc_units=[32, 16],
        state_fc_units=[24],
        action_fc_units=[12],
        joint_fc_units=[16],
    )
    states = torch.randn(5, 10)
    actions = torch.randn(5, 4)

    q_values = critic(states, actions)

    assert q_values.shape == (5, 1)
    assert len(critic.state_layers) == 1
    assert len(critic.action_layers) == 1
    assert len(critic.joint_layers) == 1


def test_build_critic_network_uses_configured_class() -> None:
    critic = build_critic_network(
        state_size=8,
        action_size=3,
        seed=1,
        network_config={
            "class": "LateFusionCritic",
            "layers": [32, 16],
            "state_layers": [24],
            "action_layers": [12],
            "joint_layers": [16],
        },
    )

    assert isinstance(critic, LateFusionCritic)


def test_multi_head_actor_outputs_action_vector_and_accepts_output_bias() -> None:
    actor = MultiHeadActor(state_size=6, action_size=3, seed=1, fc_units=[16], head_units=[8])
    states = torch.randn(4, 6)

    actions = actor(states)

    assert actions.shape == (4, 3)
    assert len(actor.action_heads) == 3

    actor.set_output_bias(torch.tensor([0.0, 0.5, -0.5]))
    biased_actions = actor(torch.zeros(1, 6))

    assert biased_actions.shape == (1, 3)
    assert biased_actions[0, 1].item() > 0.0
    assert biased_actions[0, 2].item() < 0.0


def test_build_actor_network_uses_configured_multi_head_class() -> None:
    actor = build_actor_network(
        state_size=8,
        action_size=3,
        seed=1,
        network_config={
            "class": "MultiHeadActor",
            "layers": [32],
            "head_layers": [8],
        },
    )

    assert isinstance(actor, MultiHeadActor)


def test_semantic_multi_head_actor_preserves_action_order_and_accepts_single_state() -> None:
    actor = SemanticMultiHeadActor(
        state_size=5,
        action_size=4,
        seed=1,
        fc_units=[16],
        head_units=[8],
        action_groups={"storage": [2], "ev": [0, 3], "deferrable": [1]},
    )

    batched_actions = actor(torch.randn(3, 5))
    single_actions = actor(torch.randn(5))

    assert batched_actions.shape == (3, 4)
    assert single_actions.shape == (4,)
    assert actor.group_indices == [[0, 3], [2], [1]]

    actor.set_output_bias(torch.tensor([-0.25, 0.0, 0.75, 0.25]))
    biased_actions = actor(torch.zeros(1, 5))

    assert biased_actions[0, 2].item() > biased_actions[0, 3].item()


def test_build_actor_network_uses_configured_semantic_multi_head_class() -> None:
    actor = build_actor_network(
        state_size=8,
        action_size=4,
        seed=1,
        network_config={
            "class": "SemanticMultiHeadActor",
            "layers": [32],
            "head_layers": [8],
            "action_groups": {"storage": [1], "ev": [0, 3]},
        },
    )

    assert isinstance(actor, SemanticMultiHeadActor)
