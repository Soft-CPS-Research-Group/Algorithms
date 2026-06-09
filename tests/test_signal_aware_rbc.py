from __future__ import annotations

import numpy as np

from algorithms.agents.baseline_policies import SignalAwareRBC


class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


def _agent() -> SignalAwareRBC:
    agent = SignalAwareRBC(
        {
            "algorithm": {
                "hyperparameters": {
                    "control_evs": False,
                    "control_deferrables": False,
                }
            }
        }
    )
    agent.attach_environment(
        observation_names=[["electrical_storage_soc_ratio"]],
        action_names=[["electrical_storage"]],
        action_space=[_Box(low=[-1.0], high=[1.0])],
        observation_space=[],
        metadata={"seconds_per_time_step": 3600},
    )
    return agent


def test_signal_aware_rbc_forces_storage_charge_and_discharge():
    agent = _agent()
    observations = [np.asarray([0.5], dtype=np.float32)]

    assert agent.predict(observations, context=1.0) == [[1.0]]
    assert agent.predict(observations, context=-1.0) == [[-1.0]]


def test_signal_aware_rbc_neutral_context_falls_back_to_smart_policy():
    agent = _agent()
    observations = [np.asarray([0.5], dtype=np.float32)]

    assert agent.predict(observations, context=0.0) == [[0.0]]
    assert agent.predict(observations, context=None) == [[0.0]]
