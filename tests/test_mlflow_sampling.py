from __future__ import annotations

import numpy as np
import pytest
import torch

from algorithms.agents.maddpg_agent import MADDPG
from utils.wrapper_citylearn import Wrapper_CityLearn


def test_wrapper_step_logging_uses_sampling_interval():
    wrapper = Wrapper_CityLearn.__new__(Wrapper_CityLearn)
    wrapper.log_frequency = 1
    wrapper.mlflow_step_sample_interval = 10
    wrapper.step_metric_interval = max(wrapper.log_frequency, wrapper.mlflow_step_sample_interval)

    assert wrapper._should_log_step(0) is True
    assert wrapper._should_log_step(9) is False
    assert wrapper._should_log_step(10) is True


def test_maddpg_step_logging_uses_sampling_interval():
    agent = MADDPG.__new__(MADDPG)
    agent.mlflow_step_sample_interval = 10

    assert agent._should_log_training_step(0) is True
    assert agent._should_log_training_step(3) is False
    assert agent._should_log_training_step(10) is True


def test_maddpg_initial_exploration_control_is_algorithm_owned():
    agent = MADDPG.__new__(MADDPG)
    agent.end_initial_exploration_time_step = 5

    assert agent.is_initial_exploration_done(4) is False
    assert agent.is_initial_exploration_done(5) is True


def test_wrapper_delegates_initial_exploration_decision_to_agent():
    class DummyModel:
        def __init__(self):
            self.use_raw_observations = True
            self.received = None

        def is_initial_exploration_done(self, global_learning_step: int) -> bool:
            return global_learning_step >= 10

        def update(self, **kwargs):
            self.received = kwargs
            return None

    wrapper = Wrapper_CityLearn.__new__(Wrapper_CityLearn)
    wrapper.model = DummyModel()
    wrapper.steps_between_training_updates = 1
    wrapper.target_update_interval = 0
    wrapper.time_step = 0
    wrapper.global_step = 7
    wrapper.initial_exploration_done = False
    wrapper.update_step = False
    wrapper.update_target_step = False

    wrapper.update(
        observations=[np.array([0.0])],
        actions=[np.array([0.0])],
        reward=[0.0],
        next_observations=[np.array([0.0])],
        terminated=False,
        truncated=False,
    )

    assert wrapper.initial_exploration_done is False
    assert wrapper.model.received is not None
    assert wrapper.model.received["initial_exploration_done"] is False


def test_wrapper_passes_raw_and_encoded_observation_context_to_model():
    class DummyModel:
        use_raw_observations = False

        def __init__(self):
            self.context = None

        def set_observation_context(self, **kwargs):
            self.context = kwargs

        def predict(self, observations, deterministic=None):
            self.received_observations = observations
            self.received_deterministic = deterministic
            return [[0.0]]

    wrapper = Wrapper_CityLearn.__new__(Wrapper_CityLearn)
    wrapper.model = DummyModel()
    wrapper._entity_interface_mode = False
    wrapper.get_all_encoded_observations = lambda observations: [np.asarray([2.0])]  # type: ignore[method-assign]
    wrapper.next_time_step = lambda: None  # type: ignore[method-assign]
    wrapper._Agent__action_space = [object()]
    wrapper._Agent__actions = [[None]]
    wrapper._Environment__time_step = 0

    actions = wrapper.predict([np.asarray([1.0])], deterministic=False)

    assert actions == [[0.0]]
    assert wrapper.model.context is not None
    assert wrapper.model.context["raw_observations"][0][0] == 1.0
    assert wrapper.model.context["encoded_observations"][0][0] == 2.0


def test_wrapper_action_diagnostics_include_categories_and_deferrable_delay():
    class DummyModel:
        deferrable_trigger_threshold = 0.5

    wrapper = Wrapper_CityLearn.__new__(Wrapper_CityLearn)
    wrapper.action_diagnostics_enabled = True
    wrapper.action_diagnostics_detail = "per_action"
    wrapper.action_saturation_tolerance = 0.01
    wrapper.action_idle_tolerance = 0.02
    wrapper._deferrable_wait_steps = {}
    wrapper.model = DummyModel()
    wrapper.action_names = [[
        "electrical_storage",
        "electric_vehicle_storage_charger_1_1",
        "deferrable_appliance_washing_machine_1",
    ]]
    wrapper.observation_names = [[
        "deferrable_appliance::Building_1/washing_machine_1::pending",
        "deferrable_appliance::Building_1/washing_machine_1::can_start",
    ]]
    wrapper.action_space = [
        type(
            "space",
            (),
            {
                "low": np.array([-1.0, -1.0, 0.0], dtype=np.float64),
                "high": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            },
        )()
    ]
    observations = [np.array([1.0, 1.0], dtype=np.float64)]

    first = wrapper._build_action_diagnostic_metrics([[0.0, -0.2, 0.1]], observations)
    second = wrapper._build_action_diagnostic_metrics([[0.0, -0.2, 0.6]], observations)

    assert first["Action/ev_negative_fraction"] == 1.0
    assert first["Action/storage_idle_fraction"] == 1.0
    assert first["Action/deferrable_off_fraction"] == 1.0
    assert second["Action/deferrable_on_fraction"] == 1.0
    assert second["Deferrable/start_when_available_count"] == 1.0
    assert second["Deferrable/start_delay_steps_mean"] == 1.0
    assert "Action/agent_0/2_deferrable_appliance_washing_machine_1/value" in second


def test_maddpg_diagnostic_metrics_are_consumable_without_mlflow():
    agent = MADDPG.__new__(MADDPG)
    agent.replay_buffer = [object(), object()]
    agent.exploration_step = 7
    agent.sigma = 0.12
    agent.random_exploration_steps = 10
    agent.end_initial_exploration_time_step = 10
    agent.reward_normalization_enabled = False
    agent.reward_norm_count = 0
    agent.reward_norm_mean = 0.0
    agent.reward_norm_m2 = 0.0
    agent.reward_normalization_epsilon = 1.0e-8
    agent._latest_training_metrics = {}

    status = agent.get_diagnostic_metrics()
    agent._record_training_metrics({"MADDPG/test_metric": 3.0}, step=7)

    assert status["MADDPG/replay_buffer_size"] == 2.0
    assert status["MADDPG/exploration_sigma"] == 0.12
    assert agent.consume_latest_training_metrics() == {"MADDPG/test_metric": 3.0}
    assert agent.consume_latest_training_metrics() == {}


def test_maddpg_reward_normalization_uses_running_stats():
    agent = MADDPG.__new__(MADDPG)
    agent.reward_normalization_enabled = True
    agent.reward_normalization_clip = 10.0
    agent.reward_normalization_epsilon = 1.0e-8
    agent.reward_norm_count = 0
    agent.reward_norm_mean = 0.0
    agent.reward_norm_m2 = 0.0

    agent._update_reward_normalizer([1.0, 3.0])
    normalized = agent._normalize_reward_tensor(torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float32))

    assert agent.reward_norm_count == 2
    assert agent.reward_norm_mean == 2.0
    assert agent._reward_normalization_std() == pytest.approx(2.0**0.5)
    assert normalized[0, 0, 0].item() == pytest.approx(-1.0 / (2.0**0.5), abs=1e-6)
    assert normalized[1, 0, 0].item() == pytest.approx(1.0 / (2.0**0.5), abs=1e-6)


def test_wrapper_reward_component_diagnostics_are_summarized():
    class DummyReward:
        def get_last_components(self):
            return {
                "per_agent": [
                    {"local_cost_reward": -1.0, "ev_service_penalty": 2.0},
                    {"local_cost_reward": -3.0, "ev_service_penalty": 0.0},
                ],
                "community": {"community_import_energy": 4.0},
            }

    wrapper = Wrapper_CityLearn.__new__(Wrapper_CityLearn)
    wrapper.reward_diagnostics_enabled = True
    wrapper.reward_diagnostics_detail = "per_agent"
    wrapper.env = type("env", (), {"reward_function": DummyReward()})()

    metrics = wrapper._build_reward_component_metrics()

    assert metrics["RewardComponent/local_cost_reward_mean"] == -2.0
    assert metrics["RewardComponent/local_cost_reward_sum"] == -4.0
    assert metrics["RewardComponent/ev_service_penalty_sum"] == 2.0
    assert metrics["RewardComponent/agent_0/local_cost_reward"] == -1.0
    assert metrics["RewardComponent/community/community_import_energy"] == 4.0
