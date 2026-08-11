from __future__ import annotations

import numpy as np
import pytest

from algorithms.agents.maddpg_agent import MADDPG
from algorithms.agents.matd3_agent import MATD3
from algorithms.utils.price_multiplier_adapter import (
    CURRENT_PRICE_NAME,
    PREDICTED_PRICE_NAMES,
)


class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


def _config(*, num_agents: int, observation_dim: int) -> dict:
    return {
        "training": {"seed": 7},
        "tracking": {"training_diagnostics_enabled": True},
        "checkpointing": {},
        "topology": {
            "num_agents": num_agents,
            "observation_dimensions": [observation_dim] * num_agents,
            "action_dimensions": [1] * num_agents,
        },
        "algorithm": {
            "name": "MATD3",
            "hyperparameters": {"gamma": 0.99, "require_cuda": False},
            "networks": {
                "actor": {"class": "Actor", "layers": [8], "lr": 1.0e-3},
                "critic": {"class": "Critic", "layers": [8], "lr": 1.0e-3},
            },
            "replay_buffer": {
                "class": "MultiAgentReplayBuffer",
                "capacity": 16,
                "batch_size": 2,
            },
            "exploration": {
                "strategy": "GaussianNoise",
                "params": {
                    "sigma": 0.0,
                    "min_sigma": 0.0,
                    "decay": 1.0,
                    "use_amp": False,
                    "random_exploration_steps": 0,
                    "end_initial_exploration_time_step": 0,
                    "local_price_conditioning_enabled": True,
                    "local_price_forecast_mode": "real_unmodified",
                },
            },
        },
    }


def _attached_agent() -> tuple[MATD3, list[np.ndarray]]:
    names = [
        CURRENT_PRICE_NAME,
        *PREDICTED_PRICE_NAMES,
        "local_feature",
        "district__community_import_power_kw",
    ]
    agent = MATD3(_config(num_agents=2, observation_dim=len(names)))
    lows = [0.1, 0.1, 0.1, 0.1, -10.0, -100.0]
    highs = [0.5, 0.5, 0.5, 0.5, 10.0, 100.0]
    agent.attach_environment(
        observation_names=[names, names],
        action_names=[["electrical_storage"], ["electrical_storage"]],
        action_space=[_Box([-1.0], [1.0]), _Box([-1.0], [1.0])],
        observation_space=[None, None],
        metadata={
            "building_names": ["Building_1", "Building_2"],
            "raw_observation_names": [names, names],
            "encoded_observation_names": [names, names],
            "raw_observation_bounds": [
                {"low": lows, "high": highs},
                {"low": lows, "high": highs},
            ],
        },
    )
    observations = [
        np.asarray([0.5, 0.25, 0.5, 0.75, 0.3, 0.4], dtype=np.float32),
        np.asarray([0.5, 0.75, 0.5, 0.25, 0.6, 0.7], dtype=np.float32),
    ]
    return agent, observations


def test_matd3_level1_scalar_price_reaches_every_actor(monkeypatch) -> None:
    agent, observations = _attached_agent()
    captured = []

    def capture(self, values, deterministic=False, context=None):
        del self, deterministic, context
        captured.extend(np.asarray(value).copy() for value in values)
        return [[0.0], [0.0]]

    monkeypatch.setattr(MADDPG, "predict", capture)
    agent.predict(observations, deterministic=True, context=1.5)

    assert [value[0] for value in captured] == pytest.approx([0.875, 0.875])
    assert np.array_equal(captured[0][1:], observations[0][1:])
    assert np.array_equal(captured[1][1:], observations[1][1:])


def test_matd3_level2_vector_routes_one_price_per_actor(monkeypatch) -> None:
    agent, observations = _attached_agent()
    captured = []

    def capture(self, values, deterministic=False, context=None):
        del self, deterministic, context
        captured.extend(np.asarray(value).copy() for value in values)
        return [[0.0], [0.0]]

    monkeypatch.setattr(MADDPG, "predict", capture)
    agent.predict(observations, deterministic=True, context=[0.5, 1.5])

    assert [value[0] for value in captured] == pytest.approx([0.125, 0.875])
    assert np.array_equal(captured[0][1:], observations[0][1:])
    assert np.array_equal(captured[1][1:], observations[1][1:])
    diagnostics = agent.get_diagnostic_metrics()
    assert diagnostics["MATD3/local_price_context_non_neutral"] == 1.0


def test_matd3_neutral_vector_is_exact_and_bad_vector_fails(monkeypatch) -> None:
    agent, observations = _attached_agent()
    captured = []

    def capture(self, values, deterministic=False, context=None):
        del self, deterministic, context
        captured.extend(np.asarray(value).copy() for value in values)
        return [[0.0], [0.0]]

    monkeypatch.setattr(MADDPG, "predict", capture)
    agent.predict(observations, deterministic=True, context=[1.0, 1.0])

    assert np.array_equal(captured[0], observations[0])
    assert np.array_equal(captured[1], observations[1])
    with pytest.raises(ValueError, match="length must match"):
        agent.predict(observations, deterministic=True, context=[0.9])


def test_matd3_non_neutral_price_is_inference_only(monkeypatch) -> None:
    agent, observations = _attached_agent()
    monkeypatch.setattr(
        MADDPG,
        "predict",
        lambda self, values, deterministic=False, context=None: [[0.0], [0.0]],
    )
    agent.predict(observations, deterministic=True, context=[0.9, 1.1])

    with pytest.raises(RuntimeError, match="inference-only"):
        agent.update(
            [],
            [],
            [],
            [],
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=1,
            update_step=False,
            initial_exploration_done=False,
        )


def test_matd3_projects_each_building_after_joint_policy_action(monkeypatch) -> None:
    building_names = ["Building_1", "Building_2"]
    observation_names = []
    raw_observations = []
    for building in building_names:
        prefix = f"storage::{building}/electrical_storage::"
        observation_names.append(
            [
                "charging_building_headroom_kw",
                f"{prefix}soc",
                f"{prefix}soc_min_ratio",
                f"{prefix}nominal_power_kw",
                f"{prefix}available_charge_action_normalized",
                f"{prefix}available_discharge_action_normalized",
            ]
        )
        raw_observations.append(
            np.asarray([1.0, 0.5, 0.0, 5.0, 1.0, 1.0], dtype=np.float64)
        )

    config = _config(num_agents=2, observation_dim=len(observation_names[0]))
    params = config["algorithm"]["exploration"]["params"]
    params["local_price_conditioning_enabled"] = False
    params["local_action_safety_enabled"] = True
    agent = MATD3(config)
    agent.attach_environment(
        observation_names=observation_names,
        action_names=[["electrical_storage"], ["electrical_storage"]],
        action_space=[_Box([-1.0], [1.0]), _Box([-1.0], [1.0])],
        observation_space=[None, None],
        metadata={"building_names": building_names},
    )
    agent.set_observation_context(raw_observations=raw_observations)
    monkeypatch.setattr(
        MADDPG,
        "predict",
        lambda self, observations, deterministic=False, context=None: [[1.0], [1.0]],
    )

    actions = agent.predict(
        [np.zeros(len(observation_names[0])), np.zeros(len(observation_names[1]))],
        deterministic=True,
    )

    assert actions[0][0] == pytest.approx(0.2)
    assert actions[1][0] == pytest.approx(0.2)
    metrics = agent.get_diagnostic_metrics()
    assert metrics["MATD3/local_action_safety_enabled"] == 1.0
    assert metrics["MATD3/local_action_safety_interventions"] == 2.0


def test_td3_does_not_enable_joint_matd3_safety_adapter() -> None:
    from algorithms.agents.td3_agent import TD3

    config = _config(num_agents=1, observation_dim=1)
    config["algorithm"]["name"] = "TD3"
    config["algorithm"]["exploration"]["params"].update(
        {
            "local_price_conditioning_enabled": False,
            "local_action_safety_enabled": True,
        }
    )

    agent = TD3(config)

    assert agent._joint_local_action_safety_enabled is False
    assert agent.local_action_safety_enabled is True
