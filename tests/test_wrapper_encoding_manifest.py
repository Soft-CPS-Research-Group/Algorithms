import numpy as np

from utils.wrapper_citylearn import Wrapper_CityLearn


class DummyEncoder:
    def __init__(self):
        self.x_min = 0
        self.x_max = 1

    def __mul__(self, value):
        return value

    def __rmul__(self, value):
        return value


class DummyEnv:
    def __init__(self):
        self.observation_names = [["feat1", "feat2"], ["feat3"]]
        self.observation_space = [
            type("space", (), {"high": np.array([1, 1]), "low": np.array([0, 0])})(),
            type("space", (), {"high": np.array([1]), "low": np.array([0])})(),
        ]
        self.action_space = [
            type("space", (), {"high": np.array([1]), "low": np.array([-1])})(),
            type("space", (), {"high": np.array([1]), "low": np.array([-1])})(),
        ]
        self.action_names = ["a1", "a2"]
        self.reward_function = type("reward", (), {"__dict__": {"param": 1}})()
        self.time_steps = 1
        self.seconds_per_time_step = 3600
        self.random_seed = 0
        self.episode_tracker = type("tracker", (), {"episode_time_steps": self.time_steps})()
        self.unwrapped = self

    def reset(self):
        return [np.array([0, 0]), np.array([0])], {}

    def get_metadata(self):
        return {"buildings": [{} for _ in self.observation_names]}


class DummyWrapper(Wrapper_CityLearn):
    def set_encoders(self):
        return [[DummyEncoder(), DummyEncoder()], [DummyEncoder()]]


def test_describe_environment_contains_encoder_params():
    env = DummyEnv()
    wrapper = DummyWrapper(env=env, config={"training": {}, "checkpointing": {}, "tracking": {}}, job_id="test")
    metadata = wrapper.describe_environment()

    assert metadata["encoders"][0][0]["type"] == "DummyEncoder"
    assert metadata["encoders"][0][0]["params"]["x_min"] == 0
    assert metadata["action_names"] == ["a1", "a2"]
    assert metadata["reward_function"]["name"] == "reward"
