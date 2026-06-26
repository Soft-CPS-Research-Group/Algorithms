from __future__ import annotations

import numpy as np
import pytest

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
                    "price_charge_rate": 0.6,
                    "price_discharge_rate": 0.45,
                }
            }
        }
    )
    agent.attach_environment(
        observation_names=[
            [
                "electrical_storage_soc_ratio",
                "electricity_pricing",
                "electricity_pricing_predicted_1",
                "electricity_pricing_predicted_2",
                "electricity_pricing_predicted_3",
                "import_power_kw",
            ]
        ],
        action_names=[["electrical_storage"]],
        action_space=[_Box(low=[-1.0], high=[1.0])],
        observation_space=[],
        metadata={"seconds_per_time_step": 3600},
    )
    return agent


def test_signal_aware_rbc_scales_price_context():
    agent = _agent()
    observations = [np.asarray([0.5, 0.10, 0.10, 0.10, 0.10, 8.0], dtype=np.float32)]
    obs = observations[0]
    obs_map = agent._obs_index[0]

    agent.predict(observations, context=2.0)
    expensive = agent._get_price_context(obs, obs_map)

    agent.predict(observations, context=0.5)
    cheap = agent._get_price_context(obs, obs_map)

    assert expensive["price"] == pytest.approx(0.20)
    assert expensive["expensive"] is True
    assert cheap["price"] == pytest.approx(0.05)
    assert cheap["cheap"] is True


def test_signal_aware_rbc_neutral_multiplier_matches_base_policy():
    # context=1.0 is the neutral multiplier: price appears unchanged.
    agent = _agent()
    observations = [np.asarray([0.5, 0.10, 0.10, 0.10, 0.10, 8.0], dtype=np.float32)]

    assert agent.predict(observations, context=1.0) == agent.predict(observations, context=None)


def test_signal_aware_rbc_neutral_context_falls_back_to_smart_policy():
    agent = _agent()
    observations = [np.asarray([0.5, 0.10, 0.08, 0.10, 0.12, 0.0], dtype=np.float32)]

    assert agent.predict(observations, context=1.0) == [[0.0]]
    assert agent.predict(observations, context=None) == [[0.0]]


def test_signal_aware_rbc_multiplier_changes_storage_action():
    agent = _agent()
    cheap_observations = [np.asarray([0.5, 0.10, 0.08, 0.10, 0.12, 0.0], dtype=np.float32)]
    expensive_observations = [np.asarray([0.5, 0.10, 0.08, 0.10, 0.12, 8.0], dtype=np.float32)]

    assert agent.predict(cheap_observations, context=0.5) == [[0.6]]
    assert agent.predict(expensive_observations, context=2.0)[0][0] < 0.0
