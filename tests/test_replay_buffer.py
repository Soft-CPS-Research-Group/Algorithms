from __future__ import annotations

import numpy as np
import pytest
import torch

from algorithms.utils.replay_buffer import MultiAgentReplayBuffer, RewardWeightedMultiAgentReplayBuffer


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


def test_multi_agent_replay_buffer_can_sample_behavior_action_targets() -> None:
    buffer = MultiAgentReplayBuffer(capacity=4, num_agents=1, batch_size=1)
    buffer.push(
        states=[np.array([1.0], dtype=np.float32)],
        actions=[np.array([0.1], dtype=np.float32)],
        rewards=[1.0],
        next_states=[np.array([2.0], dtype=np.float32)],
        done=False,
        behavior_actions=[np.array([0.9], dtype=np.float32)],
    )

    sample = buffer.sample()
    sample_with_behavior = buffer.sample_with_behavior_actions()

    assert len(sample) == 5
    assert len(sample_with_behavior) == 6
    assert sample_with_behavior[-1][0][0, 0].item() == pytest.approx(0.9)
    assert sample_with_behavior[1][0][0, 0].item() == pytest.approx(0.1)


def test_reward_weighted_multi_agent_replay_buffer_biases_high_reward_transitions() -> None:
    buffer = RewardWeightedMultiAgentReplayBuffer(
        capacity=16,
        num_agents=2,
        batch_size=4,
        priority_fraction=1.0,
        priority_alpha=8.0,
        priority_epsilon=1.0e-6,
    )

    for step, reward in enumerate([0.01, 0.01, 0.01, 10.0]):
        buffer.push(
            states=[np.array([step], dtype=np.float32), np.array([100 + step], dtype=np.float32)],
            actions=[np.array([step], dtype=np.float32), np.array([100 + step], dtype=np.float32)],
            rewards=[reward, reward],
            next_states=[np.array([step + 1], dtype=np.float32), np.array([101 + step], dtype=np.float32)],
            done=False,
        )

    states, actions, rewards, next_states, terminated = buffer.sample()

    assert torch.all(states[0][:, 0] == 3)
    assert torch.all(states[1][:, 0] == 103)
    assert torch.allclose(rewards[0][:, 0], torch.full((4,), 10.0))
    assert terminated.shape == (2, 4, 1)


def test_reward_weighted_multi_agent_replay_buffer_checkpoint_roundtrip() -> None:
    buffer = RewardWeightedMultiAgentReplayBuffer(
        capacity=16,
        num_agents=1,
        batch_size=2,
        priority_fraction=0.75,
        priority_alpha=0.7,
        priority_epsilon=1.0e-4,
    )
    buffer.push(
        states=[np.array([1.0], dtype=np.float32)],
        actions=[np.array([0.0], dtype=np.float32)],
        rewards=[2.0],
        next_states=[np.array([2.0], dtype=np.float32)],
        done=False,
    )

    restored = RewardWeightedMultiAgentReplayBuffer(capacity=16, num_agents=1, batch_size=2)
    restored.set_state(buffer.get_state())

    assert len(restored) == 1
    assert list(restored.priorities) == pytest.approx(list(buffer.priorities))


def test_reward_weighted_multi_agent_replay_buffer_negative_reward_mode_and_cap() -> None:
    buffer = RewardWeightedMultiAgentReplayBuffer(
        capacity=16,
        num_agents=1,
        batch_size=2,
        priority_fraction=1.0,
        priority_alpha=1.0,
        priority_epsilon=1.0e-3,
        priority_mode="negative_reward",
        priority_max=5.0,
    )

    buffer.push(
        states=[np.array([0.0], dtype=np.float32)],
        actions=[np.array([0.0], dtype=np.float32)],
        rewards=[100.0],
        next_states=[np.array([1.0], dtype=np.float32)],
        done=False,
    )
    buffer.push(
        states=[np.array([1.0], dtype=np.float32)],
        actions=[np.array([0.0], dtype=np.float32)],
        rewards=[-100.0],
        next_states=[np.array([2.0], dtype=np.float32)],
        done=False,
    )

    assert list(buffer.priorities) == pytest.approx([1.0e-3, 5.001])


def test_reward_weighted_multi_agent_replay_buffer_can_prioritize_behavior_actions() -> None:
    buffer = RewardWeightedMultiAgentReplayBuffer(
        capacity=16,
        num_agents=1,
        batch_size=2,
        priority_fraction=1.0,
        priority_alpha=1.0,
        priority_epsilon=1.0e-3,
        priority_mode="negative_reward",
        behavior_action_priority_weight=2.0,
        behavior_action_priority_mode="positive",
    )

    buffer.push(
        states=[np.array([0.0], dtype=np.float32)],
        actions=[np.array([0.0], dtype=np.float32)],
        rewards=[0.0],
        next_states=[np.array([1.0], dtype=np.float32)],
        done=False,
        behavior_actions=[np.array([-1.0], dtype=np.float32)],
    )
    buffer.push(
        states=[np.array([1.0], dtype=np.float32)],
        actions=[np.array([0.0], dtype=np.float32)],
        rewards=[0.0],
        next_states=[np.array([2.0], dtype=np.float32)],
        done=False,
        behavior_actions=[np.array([0.8], dtype=np.float32)],
    )

    assert list(buffer.priorities) == pytest.approx([1.0e-3, 1.601])


def test_reward_weighted_multi_agent_replay_buffer_accepts_external_priority_boost() -> None:
    buffer = RewardWeightedMultiAgentReplayBuffer(
        capacity=16,
        num_agents=1,
        batch_size=2,
        priority_fraction=1.0,
        priority_alpha=1.0,
        priority_epsilon=1.0e-3,
        priority_mode="negative_reward",
    )

    buffer.push(
        states=[np.array([0.0], dtype=np.float32)],
        actions=[np.array([0.0], dtype=np.float32)],
        rewards=[0.0],
        next_states=[np.array([1.0], dtype=np.float32)],
        done=False,
        priority_boost=0.0,
    )
    buffer.push(
        states=[np.array([1.0], dtype=np.float32)],
        actions=[np.array([0.0], dtype=np.float32)],
        rewards=[0.0],
        next_states=[np.array([2.0], dtype=np.float32)],
        done=False,
        priority_boost=2.5,
    )

    assert list(buffer.priorities) == pytest.approx([1.0e-3, 2.501])


def test_reward_weighted_multi_agent_replay_buffer_can_mask_behavior_action_priority() -> None:
    buffer = RewardWeightedMultiAgentReplayBuffer(
        capacity=16,
        num_agents=1,
        batch_size=2,
        priority_fraction=1.0,
        priority_alpha=1.0,
        priority_epsilon=1.0e-3,
        priority_mode="negative_reward",
        behavior_action_priority_weight=2.0,
        behavior_action_priority_mode="positive",
        behavior_action_priority_scope="ev",
    )
    buffer.set_behavior_action_priority_masks([[False, True]])

    buffer.push(
        states=[np.array([0.0], dtype=np.float32)],
        actions=[np.array([0.0], dtype=np.float32)],
        rewards=[0.0],
        next_states=[np.array([1.0], dtype=np.float32)],
        done=False,
        behavior_actions=[np.array([0.9, 0.0], dtype=np.float32)],
    )
    buffer.push(
        states=[np.array([1.0], dtype=np.float32)],
        actions=[np.array([0.0], dtype=np.float32)],
        rewards=[0.0],
        next_states=[np.array([2.0], dtype=np.float32)],
        done=False,
        behavior_actions=[np.array([0.0, 0.8], dtype=np.float32)],
    )

    assert list(buffer.priorities) == pytest.approx([1.0e-3, 1.601])


def test_reward_weighted_multi_agent_replay_buffer_keeps_behavior_actions() -> None:
    buffer = RewardWeightedMultiAgentReplayBuffer(
        capacity=4,
        num_agents=1,
        batch_size=1,
        priority_fraction=1.0,
        priority_alpha=1.0,
    )
    buffer.push(
        states=[np.array([1.0], dtype=np.float32)],
        actions=[np.array([0.2], dtype=np.float32)],
        rewards=[-10.0],
        next_states=[np.array([2.0], dtype=np.float32)],
        done=False,
        behavior_actions=[np.array([0.7], dtype=np.float32)],
    )

    *_, behavior_actions = buffer.sample_with_behavior_actions()

    assert behavior_actions[0][0, 0].item() == pytest.approx(0.7)
