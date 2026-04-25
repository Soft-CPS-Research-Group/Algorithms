from __future__ import annotations

import numpy as np
import pytest

from utils.wrapper_citylearn import Wrapper_CityLearn


class _Space:
    def __init__(self, low, high):
        self.low = np.array(low, dtype=np.float64)
        self.high = np.array(high, dtype=np.float64)


class _IdentityEncoder:
    def transform(self, value):
        return np.array([value], dtype=np.float64)


class _DummyEnv:
    def __init__(self, observation_names):
        self.observation_names = observation_names
        self.observation_space = [_Space([0.0] * len(names), [1.0] * len(names)) for names in observation_names]
        self.action_space = [_Space([-1.0], [1.0]) for _ in observation_names]
        self.action_names = [["action_0"] for _ in observation_names]
        self.reward_function = type("reward", (), {})()
        self.time_steps = 1
        self.seconds_per_time_step = 3600
        self.time_step_ratio = 1.0
        self.random_seed = 0
        self.episode_tracker = type("tracker", (), {"episode_time_steps": self.time_steps})()
        self.unwrapped = self

    def reset(self):
        return [np.zeros(len(names), dtype=np.float64) for names in self.observation_names], {}

    def get_metadata(self):
        return {"buildings": [{} for _ in self.observation_names]}


class _WrapperWithIdentityEncoders(Wrapper_CityLearn):
    def set_encoders(self):
        return [[_IdentityEncoder() for _ in names] for names in self.observation_names]


def _build_wrapper(observation_names, wrapper_reward_cfg):
    env = _DummyEnv(observation_names)
    config = {
        "training": {},
        "checkpointing": {},
        "tracking": {},
        "simulator": {"wrapper_reward": wrapper_reward_cfg},
    }
    return _WrapperWithIdentityEncoders(env=env, config=config, job_id="test")


def test_wrapper_reward_shaping_combines_profile_terms():
    wrapper = _build_wrapper(
        [
            [
                "net_electricity_consumption",
                "electricity_pricing",
                "electrical_service_violation_kwh",
                "ev_departure_success",
            ],
            ["net_electricity_consumption", "electricity_pricing"],
        ],
        {
            "enabled": True,
            "profile": "cost_limits_v1",
            "clip_enabled": True,
            "clip_min": -10.0,
            "clip_max": 10.0,
            "squash": "none",
        },
    )

    rewards = [0.0, 0.0]
    observations = [
        np.array([2.0, 0.5, 1.0, 1.0], dtype=np.float64),
        np.array([-1.0, 0.5], dtype=np.float64),
    ]

    shaped = wrapper._shape_rewards(rewards, observations)

    assert shaped[0] == pytest.approx(-1.55, rel=1e-6)
    assert shaped[1] == pytest.approx(0.35, rel=1e-6)


def test_wrapper_reward_shaping_falls_back_to_base_reward_when_signals_missing():
    wrapper = _build_wrapper(
        [["feature_a", "feature_b"]],
        {
            "enabled": True,
            "profile": "cost_limits_v1",
            "clip_enabled": True,
            "clip_min": -10.0,
            "clip_max": 10.0,
            "squash": "none",
        },
    )

    shaped = wrapper._shape_rewards([1.2], [np.array([0.0, 1.0], dtype=np.float64)])

    assert shaped == [pytest.approx(1.2)]


def test_wrapper_reward_shaping_respects_clip_and_squash():
    wrapper = _build_wrapper(
        [["net_electricity_consumption", "electricity_pricing"]],
        {
            "enabled": True,
            "profile": "cost_limits_v1",
            "clip_enabled": True,
            "clip_min": -1.0,
            "clip_max": 1.0,
            "squash": "tanh",
        },
    )

    shaped = wrapper._shape_rewards([0.0], [np.array([50.0, 10.0], dtype=np.float64)])

    assert shaped[0] == pytest.approx(np.tanh(-1.0), rel=1e-6)
