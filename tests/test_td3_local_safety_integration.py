from __future__ import annotations

import numpy as np
import pytest

from algorithms.agents.matd3_agent import MATD3
from algorithms.agents.td3_agent import TD3
from algorithms.utils.price_multiplier_adapter import (
    CURRENT_PRICE_NAME,
    PREDICTED_PRICE_NAMES,
)


class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


def _config(observation_dim: int) -> dict:
    return {
        "training": {"seed": 7},
        "tracking": {"training_diagnostics_enabled": True},
        "checkpointing": {},
        "topology": {
            "num_agents": 1,
            "observation_dimensions": [observation_dim],
            "action_dimensions": [3],
        },
        "algorithm": {
            "name": "TD3",
            "hyperparameters": {"gamma": 0.99, "require_cuda": False},
            "networks": {
                "actor": {"class": "Actor", "layers": [16], "lr": 1.0e-3},
                "critic": {"class": "Critic", "layers": [16], "lr": 1.0e-3},
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
                    "local_action_safety_enabled": True,
                },
            },
        },
    }


def test_td3_projects_final_action_and_exposes_diagnostics(monkeypatch) -> None:
    building = "Building_15"
    names = [
        "charging_building_headroom_kw",
        "charging_phase_L1_headroom_kw",
        "charging_phase_L2_headroom_kw",
        f"storage::{building}/electrical_storage::soc",
        f"storage::{building}/electrical_storage::nominal_power_kw",
        f"storage::{building}/electrical_storage::available_charge_action_normalized",
        f"storage::{building}/electrical_storage::available_discharge_action_normalized",
    ]
    values = [12.0, 7.0, 5.0, 0.5, 5.0, 1.0, 1.0]
    for charger, max_power, phase, minimum in (
        ("charger_15_1", 7.4, "L1", 0.5),
        ("charger_15_2", 11.0, "L2", 0.4),
    ):
        prefix = f"charger::{building}/{charger}::"
        names.extend(
            [
                f"{prefix}connected_state",
                f"{prefix}max_charging_power_kw",
                f"{prefix}max_discharging_power_kw",
                f"{prefix}available_charge_action_normalized",
                f"{prefix}available_discharge_action_normalized",
                f"{prefix}min_required_action_normalized",
                f"{prefix}phase_connection_{phase}",
            ]
        )
        values.extend([1.0, max_power, max_power, 1.0, 1.0, minimum, 1.0])

    agent = TD3(_config(len(names)))
    agent.attach_environment(
        observation_names=[names],
        action_names=[
            [
                "electrical_storage",
                "electric_vehicle_storage_charger_15_1",
                "electric_vehicle_storage_charger_15_2",
            ]
        ],
        action_space=[_Box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])],
        observation_space=[None],
        metadata={"building_names": [building]},
    )
    agent.set_observation_context(raw_observations=[np.asarray(values)])
    monkeypatch.setattr(
        MATD3,
        "predict",
        lambda self, observations, deterministic=False, context=None: [[1.0, -1.0, -1.0]],
    )

    actions = agent.predict([np.zeros(len(names))], deterministic=True)

    assert actions[0][1:] == [0.5, 0.4]
    assert actions[0][0] < 1.0
    metrics = agent.get_diagnostic_metrics()
    assert metrics["TD3/local_action_safety_enabled"] == 1.0
    assert metrics["TD3/local_action_safety_interventions"] >= 1.0


def test_td3_rejects_onnx_export_that_omits_enabled_safety(tmp_path) -> None:
    agent = TD3(_config(1))

    with pytest.raises(RuntimeError, match="not embedded in the ONNX actor"):
        agent.export_artifacts(str(tmp_path))


def test_td3_price_context_changes_only_current_local_price_and_is_inference_only(
    monkeypatch,
) -> None:
    names = [CURRENT_PRICE_NAME, *PREDICTED_PRICE_NAMES, "local_feature"]
    config = _config(len(names))
    params = config["algorithm"]["exploration"]["params"]
    params["local_action_safety_enabled"] = False
    params["local_price_conditioning_enabled"] = True
    agent = TD3(config)
    low = [0.1, 0.1, 0.1, 0.1, -10.0]
    high = [0.5, 0.5, 0.5, 0.5, 10.0]
    agent.attach_environment(
        observation_names=[names],
        action_names=[["electrical_storage", "ev", "deferrable"]],
        action_space=[_Box([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0])],
        observation_space=[None],
        metadata={
            "building_names": ["Building_1"],
            "raw_observation_names": [names],
            "raw_observation_bounds": [{"low": low, "high": high}],
        },
    )
    observation = np.asarray([0.5, 0.25, 0.5, 0.75, 0.3], dtype=np.float32)
    captured = []

    def capture(self, observations, deterministic=False, context=None):
        captured.append(np.asarray(observations[0]).copy())
        return [[0.0, 0.0, 0.0]]

    monkeypatch.setattr(MATD3, "predict", capture)
    agent.predict([observation], deterministic=True, context=1.0)
    agent.predict([observation], deterministic=True, context=1.5)

    assert np.array_equal(captured[0], observation)
    assert captured[1][0] == pytest.approx(0.875)
    assert np.array_equal(captured[1][1:], observation[1:])
    with pytest.raises(RuntimeError, match="inference-only"):
        agent.update()


def test_td3_service_teacher_can_be_disabled_for_deterministic_evaluation(
    monkeypatch,
) -> None:
    config = _config(1)
    params = config["algorithm"]["exploration"]["params"]
    params["local_action_safety_enabled"] = False
    params["local_action_safety_service_teacher_enabled"] = True
    params["local_action_safety_service_teacher_eval_enabled"] = False
    agent = TD3(config)

    class _Teacher:
        def predict_at_step(self, *_args, **_kwargs):
            raise AssertionError("deterministic evaluation queried the service teacher")

    agent._warm_start_policy = _Teacher()
    monkeypatch.setattr(
        MATD3,
        "predict",
        lambda self, observations, deterministic=False, context=None: [[0.0, 0.0, 0.0]],
    )

    agent.predict([np.zeros(1)], deterministic=True)
    assert agent._last_service_teacher_applied is False
    with pytest.raises(AssertionError, match="queried the service teacher"):
        agent.predict([np.zeros(1)], deterministic=False)
