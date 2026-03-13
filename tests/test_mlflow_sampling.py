from __future__ import annotations

import numpy as np

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
