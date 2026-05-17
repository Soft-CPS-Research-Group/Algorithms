from __future__ import annotations

import numpy as np
import pytest
import torch

from algorithms.agents.maddpg_agent import MADDPG


def _build_agent_for_exploration() -> MADDPG:
    agent = MADDPG.__new__(MADDPG)
    agent.num_agents = 2
    agent.action_dimension = [2, 1]
    agent.exploration_step = 0
    agent.random_exploration_steps = 0
    agent.sigma = 0.2
    agent.sigma_decay = 0.5
    agent.min_sigma = 0.1
    agent.bias = 0.0
    agent.noise_clip = None
    agent.initial_exploration_strategy = "uniform_full_range"
    agent.noop_noise_scale = 0.15
    agent.deferrable_on_probability = 0.2
    agent.deferrable_trigger_threshold = 0.5
    agent.warm_start_policy_noise_scale = 0.0
    agent.warm_start_policy_deterministic = True
    agent._warm_start_policy = None
    agent._latest_raw_observations = None
    agent._warned_missing_raw_context = False
    return agent


def test_predict_with_exploration_uses_random_actions_during_warmup():
    agent = _build_agent_for_exploration()
    agent.random_exploration_steps = 2

    actions = agent._predict_with_exploration(observations=[None, None])

    assert len(actions) == 2
    assert len(actions[0]) == 2
    assert len(actions[1]) == 1
    assert all(-1.0 <= value <= 1.0 for row in actions for value in row)
    # Sigma does not decay during pure random warmup.
    assert agent.sigma == 0.2


def test_predict_with_exploration_applies_noise_clip_and_sigma_decay(monkeypatch):
    agent = _build_agent_for_exploration()
    agent.noise_clip = 0.15

    monkeypatch.setattr(agent, "_predict_deterministic", lambda _obs: [np.array([0.0, 0.0]), np.array([0.0])])
    monkeypatch.setattr(np.random, "normal", lambda loc, scale, size: np.full(size, 0.4, dtype=np.float64))

    actions = agent._predict_with_exploration(observations=[None, None])

    assert actions[0] == [0.15, 0.15]
    assert actions[1] == [0.15]
    # Sigma decays once warmup is over.
    assert agent.sigma == 0.1


def test_action_scaling_respects_asymmetric_environment_bounds():
    agent = _build_agent_for_exploration()
    agent.device = torch.device("cpu")
    agent.attach_environment(
        observation_names=[[], []],
        action_names=[["a", "b"], ["c"]],
        action_space=[
            type("space", (), {"low": np.array([0.0, -2.0]), "high": np.array([1.0, 2.0])})(),
            type("space", (), {"low": np.array([0.5]), "high": np.array([1.5])})(),
        ],
        observation_space=[],
        metadata={},
    )

    scaled = agent._scale_action_tensor(0, torch.tensor([-1.0, 0.5]))

    assert scaled.tolist() == pytest.approx([0.0, 1.0], abs=1e-6)
    random_action = agent._predict_random()[0]
    assert 0.0 <= random_action[0] <= 1.0
    assert -2.0 <= random_action[1] <= 2.0


def test_target_policy_smoothing_uses_fractional_action_span_and_clips(monkeypatch):
    agent = _build_agent_for_exploration()
    agent.action_low = [np.array([0.0, -2.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 2.0], dtype=np.float32)]
    agent.target_policy_smoothing = True
    agent.target_policy_noise = 1.0
    agent.target_policy_noise_clip = 0.10

    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))

    action = torch.tensor([[0.95, 1.95]], dtype=torch.float32)
    smoothed = agent._add_target_policy_smoothing(0, action)

    np.testing.assert_allclose(smoothed.numpy(), np.array([[1.0, 2.0]], dtype=np.float32), atol=1e-6)


def test_actor_action_regularization_uses_normalized_bounds():
    agent = _build_agent_for_exploration()
    agent.action_low = [np.array([0.0, -2.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 2.0], dtype=np.float32)]
    agent.actor_action_l2_penalty = 0.10
    agent.actor_action_saturation_penalty = 0.20
    agent.actor_action_saturation_threshold = 0.80

    action = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    action_l2, saturation_excess, regularization = agent._actor_action_regularization_terms(0, action)

    assert action_l2.item() == pytest.approx(0.5, abs=1e-6)
    assert saturation_excess.item() == pytest.approx(0.02, abs=1e-6)
    assert regularization.item() == pytest.approx(0.054, abs=1e-6)


def test_noop_centered_exploration_keeps_deferrable_off_when_probability_zero():
    agent = _build_agent_for_exploration()
    agent.num_agents = 1
    agent.action_dimension = [3]
    agent.action_low = [np.array([-1.0, 0.0, 0.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]
    agent.action_names = [["electrical_storage", "electric_vehicle_storage", "deferrable_appliance_1"]]
    agent.initial_exploration_strategy = "noop_centered"
    agent.random_exploration_steps = 1
    agent.noop_noise_scale = 0.0
    agent.deferrable_on_probability = 0.0

    actions = agent._predict_with_exploration(observations=[None])

    assert actions[0][0] == pytest.approx(0.0, abs=1e-6)
    assert actions[0][1] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= actions[0][2] <= 0.5


def test_noop_centered_exploration_can_sample_deferrable_on():
    agent = _build_agent_for_exploration()
    agent.num_agents = 1
    agent.action_dimension = [1]
    agent.action_low = [np.array([0.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0], dtype=np.float32)]
    agent.action_names = [["deferrable_appliance_1"]]
    agent.initial_exploration_strategy = "noop_centered"
    agent.random_exploration_steps = 1
    agent.noop_noise_scale = 0.0
    agent.deferrable_on_probability = 1.0

    actions = agent._predict_with_exploration(observations=[None])

    assert actions[0][0] > 0.5
    assert actions[0][0] <= 1.0


def test_policy_warm_start_uses_raw_observation_context_and_clips_actions():
    class _WarmStartPolicy:
        def __init__(self):
            self.received = None
            self.deterministic = None

        def predict(self, observations, deterministic=None):
            self.received = observations
            self.deterministic = deterministic
            return [[1.5, -3.0]]

    agent = _build_agent_for_exploration()
    policy = _WarmStartPolicy()
    agent.num_agents = 1
    agent.action_dimension = [2]
    agent.action_low = [np.array([0.0, -1.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 1.0], dtype=np.float32)]
    agent.initial_exploration_strategy = "policy"
    agent.random_exploration_steps = 1
    agent._warm_start_policy = policy
    agent.set_observation_context(
        raw_observations=[np.array([42.0])],
        encoded_observations=[np.array([0.42])],
    )

    actions = agent._predict_with_exploration(observations=[np.array([0.42])])

    assert policy.received[0][0] == pytest.approx(42.0)
    assert policy.deterministic is True
    assert actions == [[1.0, -1.0]]


def test_noop_actor_initialization_sets_initial_scaled_action_near_noop():
    from algorithms.utils.networks import Actor

    agent = MADDPG.__new__(MADDPG)
    agent.num_agents = 1
    agent.action_dimension = [2]
    agent.action_low = [np.array([0.0, -1.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 1.0], dtype=np.float32)]
    agent.noop_actor_initialization = True
    agent.noop_actor_initialization_epsilon = 0.05
    agent._noop_actor_initialized = False
    agent.actors = [Actor(3, 2, seed=1, fc_units=[4])]
    agent.actor_targets = [Actor(3, 2, seed=1, fc_units=[4])]

    agent._apply_noop_actor_initialization()

    raw = agent.actors[0](torch.zeros(1, 3))
    scaled = agent._scale_action_tensor(0, raw).detach().numpy()[0]
    assert scaled[0] == pytest.approx(0.05, abs=1e-5)
    assert scaled[1] == pytest.approx(0.0, abs=1e-5)


def test_maddpg_rejects_single_agent_prioritized_replay_buffer():
    agent = MADDPG.__new__(MADDPG)
    agent.num_agents = 1
    agent.config = {
        "algorithm": {
            "replay_buffer": {
                "class": "PrioritizedReplayBuffer",
                "capacity": 10,
                "batch_size": 2,
            }
        }
    }

    with pytest.raises(ValueError, match="not supported by MADDPG"):
        agent._initialize_replay_buffer()
