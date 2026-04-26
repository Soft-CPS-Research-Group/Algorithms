from __future__ import annotations

import numpy as np

from utils.wrapper_citylearn import Wrapper_CityLearn


class _Space:
    def __init__(self, low, high):
        self.low = np.array(low, dtype=np.float64)
        self.high = np.array(high, dtype=np.float64)


class _DummyEnv:
    def __init__(self):
        self.observation_names = [["obs_0"]]
        self.observation_space = [_Space([0.0], [1.0])]
        self.action_space = [_Space([-1.0, 0.0], [1.0, 1.0])]
        self.action_names = [["a", "b"]]
        self.reward_function = type("reward", (), {})()
        self.time_steps = 1
        self.seconds_per_time_step = 3600
        self.time_step_ratio = 1.0
        self.random_seed = 0
        self.episode_tracker = type("tracker", (), {"episode_time_steps": self.time_steps})()
        self.unwrapped = self

    def reset(self):
        return [np.array([0.0], dtype=np.float64)], {}

    def get_metadata(self):
        return {"buildings": [{} for _ in self.observation_names]}


def test_wrapper_clips_actions_to_agent_bounds():
    env = _DummyEnv()
    wrapper = Wrapper_CityLearn(
        env=env,
        config={"training": {}, "checkpointing": {}, "tracking": {}},
        job_id="test",
    )

    clipped = wrapper._clip_actions([[2.0, -0.5]])

    assert clipped == [[1.0, 0.0]]
