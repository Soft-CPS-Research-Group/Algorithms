from __future__ import annotations

import torch

from algorithms.utils.networks import Critic, LateFusionCritic, build_critic_network


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
