import numpy as np

from utils.wrapper_citylearn import Wrapper_CityLearn
from utils.preprocessing import NoNormalization, PeriodicNormalization, RemoveFeature


class MinimalEnv:
    def __init__(self):
        self.observation_names = [[
            "month",
            "outdoor_dry_bulb_temperature",
            "non_shiftable_load"
        ]]
        self.observation_space = [
            type(
                "space",
                (),
                {
                    "high": np.array([12, 40, 1000], dtype=float),
                    "low": np.array([1, -20, 0], dtype=float),
                },
            )()
        ]
        self.action_space = [
            type("space", (), {"high": np.array([1]), "low": np.array([-1])})()
        ]
        self.action_names = ["a1"]
        self.reward_function = type("reward", (), {"__dict__": {"param": 1}})()
        self.time_steps = 1
        self.seconds_per_time_step = 3600
        self.random_seed = 0
        self.episode_tracker = type("tracker", (), {"episode_time_steps": self.time_steps})()
        self.unwrapped = self

    def reset(self):
        return [np.array([1, 0, 0], dtype=float)], {}

    def get_metadata(self):
        return {"buildings": [{}]}


def test_default_encoder_rules_applied():
    env = MinimalEnv()
    wrapper = Wrapper_CityLearn(env=env, config={"training": {}, "checkpointing": {}, "tracking": {}}, job_id="test")
    encoders = wrapper.set_encoders()

    assert isinstance(encoders[0][0], PeriodicNormalization)
    assert isinstance(encoders[0][1], RemoveFeature)
    assert isinstance(encoders[0][2], NoNormalization)
