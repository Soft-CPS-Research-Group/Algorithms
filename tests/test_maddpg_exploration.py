from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import torch

from algorithms.agents.maddpg_agent import MADDPG, _select_torch_device


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
    agent.storage_exploration_noise_multiplier = 1.0
    agent.ev_negative_exploration_noise_multiplier = 1.0
    agent.initial_exploration_strategy = "uniform_full_range"
    agent.noop_noise_scale = 0.15
    agent.deferrable_on_probability = 0.2
    agent.deferrable_trigger_threshold = 0.5
    agent.warm_start_policy_noise_scale = 0.0
    agent.warm_start_policy_deterministic = True
    agent.warm_start_policy_phaseout_steps = 0
    agent.warm_start_policy_phaseout_mode = "probability"
    agent.actor_behavior_cloning_source = "replay_action"
    agent.actor_policy_loss_normalization = False
    agent.actor_policy_loss_normalization_epsilon = 1.0e-3
    agent.actor_policy_loss_normalization_max_scale = 100.0
    agent._warm_start_policy = None
    agent._latest_raw_observations = None
    agent._warned_missing_raw_context = False
    agent._last_warm_start_phaseout_probability = 0.0
    agent._last_warm_start_phaseout_used = False
    return agent


def test_select_torch_device_fails_when_cuda_is_required_but_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="require_cuda=true"):
        _select_torch_device(require_cuda=True)

    assert _select_torch_device(require_cuda=False).type == "cpu"


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


def test_exploration_noise_can_be_scaled_for_storage_and_negative_ev():
    agent = _build_agent_for_exploration()
    agent.num_agents = 1
    agent.action_dimension = [3]
    agent.action_names = [["electrical_storage", "electric_vehicle_storage_charger_1", "deferrable_appliance_1"]]
    agent.action_low = [np.array([-1.0, -1.0, 0.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]
    agent.storage_exploration_noise_multiplier = 0.25
    agent.ev_negative_exploration_noise_multiplier = 0.5

    scaled = agent._scale_exploration_noise(0, np.array([0.4, -0.4, 0.4], dtype=np.float64))

    assert scaled.tolist() == pytest.approx([0.1, -0.2, 0.4], abs=1e-6)


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
    action_l2, saturation_excess, storage_l2, storage_smoothness_l2, ev_v2g_l2, ev_v2g_mass, regularization = (
        agent._actor_action_regularization_terms(0, action)
    )

    assert action_l2.item() == pytest.approx(0.5, abs=1e-6)
    assert saturation_excess.item() == pytest.approx(0.02, abs=1e-6)
    assert storage_l2.item() == pytest.approx(0.0, abs=1e-6)
    assert storage_smoothness_l2.item() == pytest.approx(0.0, abs=1e-6)
    assert ev_v2g_l2.item() == pytest.approx(0.0, abs=1e-6)
    assert ev_v2g_mass.item() == pytest.approx(0.0, abs=1e-6)
    assert regularization.item() == pytest.approx(0.054, abs=1e-6)


def test_actor_action_regularization_can_penalize_residual_delta_from_teacher():
    agent = _build_agent_for_exploration()
    agent.action_low = [np.array([0.0, -2.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 2.0], dtype=np.float32)]
    agent.actor_action_l2_penalty = 0.0
    agent.actor_action_saturation_penalty = 0.0
    agent.actor_storage_action_l2_penalty = 0.0
    agent.actor_ev_v2g_action_l2_penalty = 0.0
    agent.actor_ev_v2g_action_mass_penalty = 0.0
    agent.actor_residual_delta_l2_penalty = 0.5

    action = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    teacher_action = torch.tensor([[0.5, 2.0]], dtype=torch.float32)
    *_, regularization = agent._actor_action_regularization_terms(
        0,
        action,
        base_action=teacher_action,
    )

    assert regularization.item() == pytest.approx(0.5, abs=1e-6)


def test_actor_action_regularization_can_target_storage_and_ev_v2g_actions():
    agent = _build_agent_for_exploration()
    agent.action_dimension = [3, 1]
    agent.action_names = [["electrical_storage", "electric_vehicle_storage_charger_1", "deferrable_appliance_1"]]
    agent.action_low = [np.array([-1.0, -1.0, 0.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 1.0, 1.0], dtype=np.float32)]
    agent.actor_action_l2_penalty = 0.0
    agent.actor_action_saturation_penalty = 0.0
    agent.actor_storage_action_l2_penalty = 0.10
    agent.actor_ev_v2g_action_l2_penalty = 0.20
    agent.actor_ev_v2g_action_mass_penalty = 0.30
    agent.actor_ev_behavior_cloning_multiplier = 1.0
    agent.actor_storage_behavior_cloning_multiplier = 1.0

    action = torch.tensor([[0.5, -0.5, 1.0], [-0.5, 0.0, 0.0]], dtype=torch.float32)
    _, _, storage_l2, storage_smoothness_l2, ev_v2g_l2, ev_v2g_mass, regularization = agent._actor_action_regularization_terms(
        0, action
    )

    assert storage_l2.item() == pytest.approx(0.25, abs=1e-6)
    assert storage_smoothness_l2.item() == pytest.approx(0.0, abs=1e-6)
    assert ev_v2g_l2.item() == pytest.approx(0.25, abs=1e-6)
    assert ev_v2g_mass.item() == pytest.approx(0.25, abs=1e-6)
    assert regularization.item() == pytest.approx(0.150, abs=1e-6)


def test_actor_behavior_cloning_loss_uses_normalized_bounds():
    agent = _build_agent_for_exploration()
    agent.action_low = [np.array([0.0, -2.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 2.0], dtype=np.float32)]
    agent.actor_behavior_cloning_weight = 0.10

    predicted = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    replay = torch.tensor([[0.5, 2.0]], dtype=torch.float32)

    loss = agent._actor_behavior_cloning_loss(0, predicted, replay)

    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_critic_action_features_can_include_teacher_and_normalized_delta():
    agent = _build_agent_for_exploration()
    agent.action_low = [
        np.array([0.0, -2.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ]
    agent.action_high = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([10.0], dtype=np.float32),
    ]
    agent.critic_action_input_mode = "final_base_delta_normalized"

    actions = [
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[8.0]], dtype=torch.float32),
    ]
    base_actions = [
        torch.tensor([[0.5, 2.0]], dtype=torch.float32),
        torch.tensor([[3.0]], dtype=torch.float32),
    ]

    features = agent._critic_action_features(actions, base_actions)

    assert features.shape == (1, 9)
    assert features.tolist()[0] == pytest.approx(
        [1.0, 0.0, 0.5, 2.0, 0.5, -0.5, 8.0, 3.0, 0.5],
        abs=1e-6,
    )


def test_actor_behavior_cloning_loss_can_weight_ev_actions_more_than_storage():
    agent = _build_agent_for_exploration()
    agent.action_names = [["electrical_storage", "electric_vehicle_storage_charger_1"]]
    agent.action_low = [np.array([-1.0, -1.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 1.0], dtype=np.float32)]
    agent.actor_behavior_cloning_weight = 0.10
    agent.actor_storage_behavior_cloning_multiplier = 0.5
    agent.actor_ev_behavior_cloning_multiplier = 4.0

    predicted = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
    replay = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    loss = agent._actor_behavior_cloning_loss(0, predicted, replay)

    assert loss.item() == pytest.approx(1.5 / 4.5, abs=1e-6)


def test_actor_behavior_cloning_loss_can_upweight_positive_ev_targets():
    agent = _build_agent_for_exploration()
    agent.action_names = [["electric_vehicle_storage_charger_1"]]
    agent.action_low = [np.array([-1.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0], dtype=np.float32)]
    agent.actor_behavior_cloning_weight = 0.10
    agent.actor_ev_behavior_cloning_multiplier = 1.0
    agent.actor_ev_behavior_cloning_positive_target_weight = 3.0

    predicted = torch.tensor([[0.0], [-1.0]], dtype=torch.float32)
    replay = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)

    loss = agent._actor_behavior_cloning_loss(0, predicted, replay)

    assert loss.item() == pytest.approx(4.0 / 5.0, abs=1e-6)


def test_actor_behavior_cloning_loss_can_upweight_zero_ev_targets():
    agent = _build_agent_for_exploration()
    agent.action_names = [["electric_vehicle_storage_charger_1"]]
    agent.action_low = [np.array([-1.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0], dtype=np.float32)]
    agent.actor_behavior_cloning_weight = 0.10
    agent.actor_ev_behavior_cloning_multiplier = 1.0
    agent.actor_ev_behavior_cloning_zero_target_weight = 3.0
    agent.actor_ev_behavior_cloning_zero_target_threshold = 0.05

    predicted = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    replay = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    loss = agent._actor_behavior_cloning_loss(0, predicted, replay)

    assert loss.item() == pytest.approx(4.0 / 5.0, abs=1e-6)


def test_actor_behavior_cloning_loss_is_zero_when_disabled():
    agent = _build_agent_for_exploration()
    agent.actor_behavior_cloning_weight = 0.0

    predicted = torch.tensor([[1.0]], dtype=torch.float32)
    replay = torch.tensor([[-1.0]], dtype=torch.float32)

    loss = agent._actor_behavior_cloning_loss(0, predicted, replay)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_transition_behavior_actions_can_use_deterministic_warm_start_policy():
    class _WarmStartPolicy:
        def __init__(self):
            self.deterministic = None

        def predict(self, observations, deterministic=None):
            self.deterministic = deterministic
            return [[0.8, -0.8]]

    agent = _build_agent_for_exploration()
    agent.num_agents = 1
    agent.action_dimension = [2]
    agent.action_low = [np.array([-1.0, -1.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 1.0], dtype=np.float32)]
    agent.actor_behavior_cloning_source = "warm_start_policy"
    agent.warm_start_policy_deterministic = False
    agent.warm_start_policy_noise_scale = 1.0
    agent._warm_start_policy = _WarmStartPolicy()
    agent.set_observation_context(raw_observations=[np.array([42.0])])

    behavior_actions = agent._transition_behavior_actions([[0.1, 0.2]])

    assert behavior_actions == [[0.8, -0.8]]
    assert agent._warm_start_policy.deterministic is True


def test_transition_observation_event_priority_boost_targets_ev_departure_service_windows():
    agent = _build_agent_for_exploration()
    agent.num_agents = 1
    agent.observation_names = [[
        "charger::Building_1/charger_1_1::connected_ev_departure_available",
        "charger::Building_1/charger_1_1::hours_until_departure",
        "charger::Building_1/charger_1_1::connected_ev_soc_deficit",
        "charger::Building_1/charger_1_1::energy_to_required_soc_kwh",
        "charger::Building_1/charger_1_1::required_average_power_kw",
    ]]
    agent.replay_observation_event_priority_weight = 8.0
    agent.set_observation_context(
        raw_observations=[np.array([1.0, 1.0, 0.2, 5.0, 4.0], dtype=np.float64)]
    )

    boost = agent._transition_observation_event_priority_boost()

    assert boost > 0.0
    assert boost == pytest.approx(agent._last_observation_event_priority_boost)


def test_transition_observation_event_priority_boost_ignores_ev_without_deficit():
    agent = _build_agent_for_exploration()
    agent.num_agents = 1
    agent.observation_names = [[
        "charger::Building_1/charger_1_1::connected_ev_departure_available",
        "charger::Building_1/charger_1_1::hours_until_departure",
        "charger::Building_1/charger_1_1::connected_ev_soc_deficit",
        "charger::Building_1/charger_1_1::energy_to_required_soc_kwh",
    ]]
    agent.replay_observation_event_priority_weight = 8.0
    agent.set_observation_context(
        raw_observations=[np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float64)]
    )

    assert agent._transition_observation_event_priority_boost() == pytest.approx(0.0)


def test_actor_behavior_cloning_effective_weight_decays_after_start_step():
    agent = _build_agent_for_exploration()
    agent.actor_behavior_cloning_weight = 0.05
    agent.actor_behavior_cloning_min_weight = 0.01
    agent.actor_behavior_cloning_decay_start_step = 10
    agent.actor_behavior_cloning_decay_steps = 20

    assert agent._actor_behavior_cloning_effective_weight(5) == pytest.approx(0.05)
    assert agent._actor_behavior_cloning_effective_weight(20) == pytest.approx(0.03)
    assert agent._actor_behavior_cloning_effective_weight(40) == pytest.approx(0.01)


def test_actor_behavior_cloning_extra_updates_respect_training_window():
    agent = _build_agent_for_exploration()
    agent.actor_behavior_cloning_extra_updates = 2
    agent.actor_behavior_cloning_extra_update_start_step = 10
    agent.actor_behavior_cloning_extra_update_end_step = 20

    assert agent._actor_behavior_cloning_extra_updates_for_step(9, 1.0) == 0
    assert agent._actor_behavior_cloning_extra_updates_for_step(10, 1.0) == 2
    assert agent._actor_behavior_cloning_extra_updates_for_step(20, 1.0) == 2
    assert agent._actor_behavior_cloning_extra_updates_for_step(21, 1.0) == 0
    assert agent._actor_behavior_cloning_extra_updates_for_step(10, 0.0) == 0


def test_actor_behavior_cloning_extra_updates_move_actor_toward_teacher_action():
    agent = _build_agent_for_exploration()
    agent.num_agents = 1
    agent.device = torch.device("cpu")
    agent.use_amp = False
    agent.scaler = torch.amp.GradScaler(enabled=False)
    agent.action_names = [["electric_vehicle_storage_charger_1"]]
    agent.action_low = [np.array([-1.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0], dtype=np.float32)]
    agent.actor_behavior_cloning_weight = 1.0
    agent.actor_ev_behavior_cloning_multiplier = 1.0
    agent.actor_storage_behavior_cloning_multiplier = 1.0
    actor = torch.nn.Linear(1, 1)
    torch.nn.init.zeros_(actor.weight)
    torch.nn.init.zeros_(actor.bias)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.20)
    observations = torch.ones((4, 1), dtype=torch.float32)
    behavior_actions = torch.ones((4, 1), dtype=torch.float32)

    before = agent._scale_action_tensor(0, actor(observations)).mean().item()
    losses, grad_norms = agent._run_actor_behavior_cloning_extra_updates(
        agent_num=0,
        actor=actor,
        actor_optimizer=optimizer,
        observations=observations,
        behavior_actions=behavior_actions,
        behavior_cloning_weight=1.0,
        extra_updates=2,
    )
    after = agent._scale_action_tensor(0, actor(observations)).mean().item()

    assert len(losses) == 2
    assert len(grad_norms) == 2
    assert after > before


def test_actor_policy_loss_effective_weight_can_ramp_after_start_step():
    agent = _build_agent_for_exploration()
    agent.actor_policy_loss_weight = 1.0
    agent.actor_policy_loss_warmup_weight = 0.05
    agent.actor_policy_loss_warmup_start_step = 10
    agent.actor_policy_loss_warmup_steps = 20

    assert agent._actor_policy_loss_effective_weight(5) == pytest.approx(0.05)
    assert agent._actor_policy_loss_effective_weight(20) == pytest.approx(0.525)
    assert agent._actor_policy_loss_effective_weight(40) == pytest.approx(1.0)


def test_actor_policy_loss_can_be_normalized_by_q_scale():
    class SumCritic(torch.nn.Module):
        def forward(self, global_state, global_actions):
            del global_state
            return global_actions.sum(dim=1, keepdim=True) + 4.0

    agent = _build_agent_for_exploration()
    state = torch.zeros((3, 2), dtype=torch.float32)
    actions = torch.zeros((3, 2), dtype=torch.float32)

    raw_loss, optimized_loss, scale, q_abs_mean = agent._actor_policy_loss_from_critic(
        SumCritic(),
        state,
        actions,
    )
    assert raw_loss.item() == pytest.approx(-4.0)
    assert optimized_loss.item() == pytest.approx(-4.0)
    assert scale.item() == pytest.approx(1.0)
    assert q_abs_mean.item() == pytest.approx(4.0)

    agent.actor_policy_loss_normalization = True
    raw_loss, optimized_loss, scale, q_abs_mean = agent._actor_policy_loss_from_critic(
        SumCritic(),
        state,
        actions,
    )
    assert raw_loss.item() == pytest.approx(-4.0)
    assert optimized_loss.item() == pytest.approx(-1.0)
    assert scale.item() == pytest.approx(0.25)
    assert q_abs_mean.item() == pytest.approx(4.0)


def test_n_step_replay_pushes_discounted_oldest_transition():
    class FakeReplayBuffer:
        def __init__(self):
            self.pushes = []

        def push(self, *args, **kwargs):
            self.pushes.append((args, kwargs))

    agent = MADDPG.__new__(MADDPG)
    agent.num_agents = 1
    agent.replay_buffer = FakeReplayBuffer()
    agent.n_step_returns = 3
    agent.n_step_gamma = 0.5
    agent.n_step_priority_aggregation = "max"
    agent._n_step_queue = deque()
    agent._last_n_step_queue_size = 0
    agent._replay_push_count = 0

    def transition(step: int):
        value = float(step)
        return {
            "observations": [np.array([value], dtype=np.float32)],
            "actions": [np.array([value + 0.1], dtype=np.float32)],
            "rewards": [value],
            "next_observations": [np.array([value + 10.0], dtype=np.float32)],
            "behavior_actions": [np.array([value + 0.2], dtype=np.float32)],
            "next_behavior_actions": [np.array([value + 0.3], dtype=np.float32)],
        }

    for step, boost in [(1, 1.0), (2, 3.0), (4, 2.0)]:
        payload = transition(step)
        agent._store_replay_transition(
            **payload,
            done=False,
            priority_boost=boost,
        )

    assert len(agent.replay_buffer.pushes) == 1
    args, kwargs = agent.replay_buffer.pushes[0]
    states, actions, rewards, next_states, done = args
    assert states[0].tolist() == pytest.approx([1.0])
    assert actions[0].tolist() == pytest.approx([1.1])
    assert rewards == pytest.approx([1.0 + 0.5 * 2.0 + 0.25 * 4.0])
    assert next_states[0].tolist() == pytest.approx([14.0])
    assert done is False
    assert kwargs["behavior_actions"][0].tolist() == pytest.approx([1.2])
    assert kwargs["next_behavior_actions"][0].tolist() == pytest.approx([4.3])
    assert kwargs["priority_boost"] == pytest.approx(3.0)
    assert agent._replay_push_count == 1
    assert agent._last_n_step_queue_size == 2


def test_train_during_initial_exploration_can_enable_warmup_updates():
    agent = _build_agent_for_exploration()
    agent.train_during_initial_exploration = False
    agent.initial_exploration_training_start_step = 10

    assert agent._should_train_on_step(initial_exploration_done=False, global_learning_step=20) is False
    assert agent._should_train_on_step(initial_exploration_done=True, global_learning_step=0) is True

    agent.train_during_initial_exploration = True
    assert agent._should_train_on_step(initial_exploration_done=False, global_learning_step=9) is False
    assert agent._should_train_on_step(initial_exploration_done=False, global_learning_step=10) is True


def test_critic_loss_can_use_huber():
    agent = _build_agent_for_exploration()
    agent.critic_loss_function = "huber"
    agent.critic_huber_beta = 1.0

    expected = torch.tensor([0.0, 3.0], dtype=torch.float32)
    target = torch.tensor([2.0, 0.0], dtype=torch.float32)

    loss = agent._critic_loss(expected, target)

    assert loss.item() == pytest.approx(2.0, abs=1e-6)


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


def test_policy_warm_start_phaseout_can_keep_teacher_actions(monkeypatch):
    class _WarmStartPolicy:
        def predict(self, observations, deterministic=None):
            return [[0.75]]

    agent = _build_agent_for_exploration()
    agent.num_agents = 1
    agent.action_dimension = [1]
    agent.action_low = [np.array([0.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0], dtype=np.float32)]
    agent.initial_exploration_strategy = "policy"
    agent.random_exploration_steps = 2
    agent.exploration_step = 2
    agent.warm_start_policy_phaseout_steps = 4
    agent._warm_start_policy = _WarmStartPolicy()
    agent.set_observation_context(raw_observations=[np.array([42.0])])

    monkeypatch.setattr(np.random, "random", lambda: 0.99)

    actions = agent._predict_with_exploration(observations=[np.array([0.42])])

    assert actions == [[0.75]]
    assert agent._last_warm_start_phaseout_probability == pytest.approx(1.0)
    assert agent._last_warm_start_phaseout_used is True


def test_policy_warm_start_phaseout_decays_to_actor_noise(monkeypatch):
    class _WarmStartPolicy:
        def predict(self, observations, deterministic=None):
            return [[0.75]]

    agent = _build_agent_for_exploration()
    agent.num_agents = 1
    agent.action_dimension = [1]
    agent.action_low = [np.array([0.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0], dtype=np.float32)]
    agent.initial_exploration_strategy = "policy"
    agent.random_exploration_steps = 2
    agent.exploration_step = 5
    agent.warm_start_policy_phaseout_steps = 4
    agent._warm_start_policy = _WarmStartPolicy()
    agent.set_observation_context(raw_observations=[np.array([42.0])])

    monkeypatch.setattr(np.random, "random", lambda: 0.90)
    monkeypatch.setattr(agent, "_predict_deterministic", lambda _obs: [np.array([0.5])])
    monkeypatch.setattr(np.random, "normal", lambda loc, scale, size: np.zeros(size, dtype=np.float64))

    actions = agent._predict_with_exploration(observations=[np.array([0.42])])

    assert actions == [[0.5]]
    assert agent._last_warm_start_phaseout_probability == pytest.approx(0.25)
    assert agent._last_warm_start_phaseout_used is False


def test_policy_warm_start_phaseout_can_blend_teacher_and_actor(monkeypatch):
    class _WarmStartPolicy:
        def predict(self, observations, deterministic=None):
            return [[1.0]]

    agent = _build_agent_for_exploration()
    agent.num_agents = 1
    agent.action_dimension = [1]
    agent.action_low = [np.array([0.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0], dtype=np.float32)]
    agent.initial_exploration_strategy = "policy"
    agent.random_exploration_steps = 2
    agent.exploration_step = 4
    agent.warm_start_policy_phaseout_steps = 4
    agent.warm_start_policy_phaseout_mode = "blend"
    agent._warm_start_policy = _WarmStartPolicy()
    agent.set_observation_context(raw_observations=[np.array([42.0])])

    monkeypatch.setattr(agent, "_predict_deterministic", lambda _obs: [np.array([0.0])])
    monkeypatch.setattr(np.random, "normal", lambda loc, scale, size: np.zeros(size, dtype=np.float64))

    actions = agent._predict_with_exploration(observations=[np.array([0.42])])

    assert actions == [[0.5]]
    assert agent._last_warm_start_phaseout_probability == pytest.approx(0.5)
    assert agent._last_warm_start_phaseout_used is True


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


def test_residual_policy_action_adds_bounded_delta_to_base_action():
    agent = MADDPG.__new__(MADDPG)
    agent.action_dimension = [2]
    agent.action_low = [np.array([-1.0, 0.0], dtype=np.float32)]
    agent.action_high = [np.array([1.0, 2.0], dtype=np.float32)]
    agent.action_names = [["electrical_storage::charge", "charger::charge"]]
    agent.residual_policy_enabled = True
    agent.residual_action_scale = 0.10
    agent.residual_action_final_scale = 0.20
    agent.residual_action_start_step = 0
    agent.residual_action_growth_steps = 10
    agent.residual_storage_action_scale_multiplier = 0.5
    agent.residual_ev_action_scale_multiplier = 1.0
    agent.residual_deferrable_action_scale_multiplier = 1.0
    agent.exploration_step = 10

    raw_action = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    base_action = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    action = agent._policy_action_from_actor_output(
        0,
        raw_action,
        base_action=base_action,
        global_learning_step=10,
    )

    assert action.detach().numpy()[0, 0] == pytest.approx(0.1, abs=1e-6)
    assert action.detach().numpy()[0, 1] == pytest.approx(0.8, abs=1e-6)
    assert agent._last_residual_action_scale == pytest.approx(0.2)


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
