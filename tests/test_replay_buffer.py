from __future__ import annotations

import numpy as np

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
