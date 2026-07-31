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


def _single_ev_agent(*, safety_enabled: bool) -> tuple[TD3, np.ndarray]:
    building = "Building_10"
    charger = "charger_10_1"
    prefix = f"charger::{building}/{charger}::"
    names = [
        f"{prefix}connected_state",
        f"{prefix}max_charging_power_kw",
        f"{prefix}max_discharging_power_kw",
        f"{prefix}available_charge_action_normalized",
        f"{prefix}available_discharge_action_normalized",
        f"{prefix}min_required_action_normalized",
    ]
    values = np.asarray([1.0, 7.4, 7.4, 1.0, 1.0, 0.5], dtype=np.float64)
    config = _config(len(names))
    config["topology"]["action_dimensions"] = [1]
    config["algorithm"]["exploration"]["params"][
        "local_action_safety_enabled"
    ] = safety_enabled
    agent = TD3(config)
    agent.attach_environment(
        observation_names=[names],
        action_names=[[f"electric_vehicle_storage_{charger}"]],
        action_space=[_Box([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": [building]},
    )
    return agent, values


class _NegativeBehaviorTeacher:
    def predict(self, observations, deterministic=None):
        del observations, deterministic
        return [[-1.0]]


def _store_one_teacher_transition(
    agent: TD3,
    raw_observation: np.ndarray,
    *,
    executed_action: float,
) -> None:
    agent.actor_behavior_cloning_source = "warm_start_policy"
    agent._warm_start_policy = _NegativeBehaviorTeacher()
    agent.set_episode_context(episode_step=0, next_episode_step=1)
    agent.set_observation_context(raw_observations=[raw_observation])
    agent._predict_warm_start_policy(apply_noise=False, deterministic=True)
    agent.set_transition_context(
        raw_observations=[raw_observation],
        raw_next_observations=[raw_observation],
        encoded_observations=[raw_observation],
        encoded_next_observations=[raw_observation],
    )
    agent.update(
        [raw_observation],
        [np.asarray([executed_action], dtype=np.float32)],
        [-1.0],
        [raw_observation],
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=1,
        update_step=False,
        initial_exploration_done=False,
    )


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


def test_td3_replay_uses_projected_behavior_teacher_target() -> None:
    agent, raw_observation = _single_ev_agent(safety_enabled=True)

    _store_one_teacher_transition(
        agent,
        raw_observation,
        executed_action=0.5,
    )

    assert agent.replay_buffer._behavior_actions is not None
    assert agent.replay_buffer._next_behavior_actions is not None
    assert agent.replay_buffer._behavior_actions[0][0, 0] == pytest.approx(0.5)
    assert agent.replay_buffer._next_behavior_actions[0][0, 0] == pytest.approx(0.5)
    assert agent._last_raw_behavior_teacher_actions == [[-1.0]]
    assert agent._last_projected_behavior_teacher_actions == [[0.5]]
    assert agent._last_raw_next_behavior_teacher_actions == [[-1.0]]
    assert agent._last_projected_next_behavior_teacher_actions == [[0.5]]

    metrics = agent.get_diagnostic_metrics()
    assert (
        metrics["TD3/behavior_teacher_raw_projected_target_available"]
        == 1.0
    )
    assert (
        metrics["TD3/behavior_teacher_raw_projected_projection_applied"]
        == 1.0
    )
    assert (
        metrics["TD3/behavior_teacher_raw_projected_disagreement_count"]
        == 1.0
    )
    assert (
        metrics["TD3/behavior_teacher_raw_projected_disagreement_ratio"]
        == 1.0
    )
    assert metrics["TD3/behavior_teacher_raw_projected_mae"] == pytest.approx(1.5)
    assert metrics["TD3/behavior_teacher_raw_projected_max_abs"] == pytest.approx(
        1.5
    )
    assert (
        metrics["TD3/next_behavior_teacher_raw_projected_disagreement_count"]
        == 1.0
    )


def test_td3_behavior_teacher_target_is_unchanged_when_safety_is_disabled() -> None:
    agent, raw_observation = _single_ev_agent(safety_enabled=False)

    _store_one_teacher_transition(
        agent,
        raw_observation,
        executed_action=-1.0,
    )

    assert agent.replay_buffer._behavior_actions is not None
    assert agent.replay_buffer._next_behavior_actions is not None
    assert agent.replay_buffer._behavior_actions[0][0, 0] == pytest.approx(-1.0)
    assert agent.replay_buffer._next_behavior_actions[0][0, 0] == pytest.approx(
        -1.0
    )
    metrics = agent.get_diagnostic_metrics()
    assert (
        metrics["TD3/behavior_teacher_raw_projected_target_available"]
        == 1.0
    )
    assert (
        metrics["TD3/behavior_teacher_raw_projected_projection_applied"]
        == 0.0
    )
    assert (
        metrics["TD3/behavior_teacher_raw_projected_disagreement_count"]
        == 0.0
    )


def test_td3_behavior_replay_is_unchanged_without_teacher() -> None:
    agent, raw_observation = _single_ev_agent(safety_enabled=True)
    agent.set_transition_context(
        raw_observations=[raw_observation],
        raw_next_observations=[raw_observation],
        encoded_observations=[raw_observation],
        encoded_next_observations=[raw_observation],
    )

    agent.update(
        [raw_observation],
        [np.asarray([0.25], dtype=np.float32)],
        [-1.0],
        [raw_observation],
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=1,
        update_step=False,
        initial_exploration_done=False,
    )

    assert agent.replay_buffer._behavior_actions is not None
    assert agent.replay_buffer._next_behavior_actions is not None
    assert agent.replay_buffer._behavior_actions[0][0, 0] == pytest.approx(0.25)
    assert agent.replay_buffer._next_behavior_actions[0][0, 0] == pytest.approx(
        0.25
    )
    metrics = agent.get_diagnostic_metrics()
    assert (
        metrics["TD3/behavior_teacher_raw_projected_target_available"]
        == 0.0
    )
    assert (
        metrics["TD3/behavior_teacher_raw_projected_disagreement_count"]
        == 0.0
    )


def test_td3_building_15_replay_detects_and_combines_both_ev_actions() -> None:
    config = _config(observation_dim=1)
    config["algorithm"]["replay_buffer"] = {
        "class": "RewardWeightedMultiAgentReplayBuffer",
        "capacity": 16,
        "batch_size": 2,
        "priority_fraction": 1.0,
        "priority_alpha": 1.0,
        "priority_epsilon": 1.0e-3,
        "priority_mode": "negative_reward",
        "behavior_action_priority_weight": 2.0,
        "behavior_action_priority_mode": "positive",
        "behavior_action_priority_scope": "ev",
        "behavior_action_stratified_sampling": True,
        "behavior_action_positive_threshold": 0.1,
    }
    agent = TD3(config)
    agent.attach_environment(
        observation_names=[["feature"]],
        action_names=[
            [
                "electrical_storage",
                "electric_vehicle_storage_charger_15_1",
                "electric_vehicle_storage_charger_15_2",
            ]
        ],
        action_space=[_Box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_15"]},
    )

    assert [mask.tolist() for mask in agent.replay_buffer.behavior_action_priority_masks] == [
        [False, True, True]
    ]
    assert agent.replay_buffer.behavior_action_stratified_sampling is True
    assert agent.replay_buffer.behavior_action_positive_threshold == pytest.approx(0.1)

    agent.replay_buffer.push(
        states=[np.array([0.0], dtype=np.float32)],
        actions=[np.zeros(3, dtype=np.float32)],
        rewards=[0.0],
        next_states=[np.array([1.0], dtype=np.float32)],
        done=False,
        behavior_actions=[np.array([1.0, 0.8, 0.6], dtype=np.float32)],
    )

    # Storage is ignored, while the two concurrent EV targets are combined:
    # (0.8 + 0.6) * weight 2.0 + epsilon.
    assert list(agent.replay_buffer.priorities) == pytest.approx([2.801])
    metrics = agent.get_diagnostic_metrics()
    assert metrics["TD3/replay_behavior_action_stratified_sampling"] == 1.0
    assert metrics["TD3/replay_behavior_action_positive_threshold"] == pytest.approx(0.1)


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
