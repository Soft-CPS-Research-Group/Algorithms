from __future__ import annotations

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
