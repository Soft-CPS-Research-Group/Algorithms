from __future__ import annotations

import numpy as np
import pytest
import torch

from algorithms.utils.replay_buffer import MultiAgentReplayBuffer


def test_multi_agent_replay_buffer_push_and_sample_without_cuda() -> None:
    buffer = MultiAgentReplayBuffer(capacity=16, num_agents=2, batch_size=2)

    for _ in range(2):
        buffer.push(
            states=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            actions=[np.array([0.1]), np.array([0.2])],
            rewards=[1.0, 2.0],
            next_states=[np.array([1.5, 2.5]), np.array([3.5, 4.5])],
            done=False,
        )

    assert len(buffer) == 2
    states, actions, rewards, next_states, terminated = buffer.sample()

    assert len(states) == 2
    assert len(actions) == 2
    assert len(rewards) == 2
    assert len(next_states) == 2
    assert terminated.shape[0] == 2


def test_multi_agent_replay_buffer_sampling_preserves_joint_transition_alignment() -> None:
    buffer = MultiAgentReplayBuffer(capacity=16, num_agents=2, batch_size=4)

    for step in range(4):
        buffer.push(
            states=[np.array([step], dtype=np.float32), np.array([100 + step], dtype=np.float32)],
            actions=[np.array([step + 0.1], dtype=np.float32), np.array([100 + step + 0.1], dtype=np.float32)],
            rewards=[float(step), float(100 + step)],
            next_states=[np.array([step + 1], dtype=np.float32), np.array([101 + step], dtype=np.float32)],
            done=(step == 3),
        )

    states, actions, rewards, next_states, terminated = buffer.sample()

    for row in range(buffer.batch_size):
        assert states[1][row, 0] - states[0][row, 0] == pytest.approx(100.0)
        assert actions[1][row, 0] - actions[0][row, 0] == pytest.approx(100.0)
        assert rewards[1][row, 0] - rewards[0][row, 0] == pytest.approx(100.0)
        assert next_states[1][row, 0] - next_states[0][row, 0] == pytest.approx(100.0)

    assert terminated.shape == (2, 4, 1)
    assert torch.equal(terminated[0], terminated[1])
