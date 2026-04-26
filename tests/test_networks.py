from __future__ import annotations

import torch

from algorithms.utils.networks import Critic


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
