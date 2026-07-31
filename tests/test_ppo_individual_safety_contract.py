from __future__ import annotations

import numpy as np
import pytest
import torch

import algorithms.agents.ppo_agents as ppo_module
from algorithms.agents.ppo_agents import PPO
from algorithms.utils.price_multiplier_adapter import (
    CURRENT_PRICE_NAME,
    PREDICTED_PRICE_NAMES,
)


class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


def _ppo_config(**exploration_overrides) -> dict:
    exploration = {
        "rollout_length": 4,
        "minibatch_size": 2,
        "ppo_epochs": 1,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "entropy_coef": 0.0,
        "value_loss_coef": 0.5,
        "max_grad_norm": 0.5,
        "initial_log_std": -1.0,
        "min_log_std": -5.0,
        "max_log_std": 1.0,
        "end_initial_exploration_time_step": 0,
        "random_exploration_steps": 0,
        "initial_exploration_strategy": "uniform_full_range",
        "train_during_initial_exploration": True,
    }
    exploration.update(exploration_overrides)
    return {
        "training": {
            "seed": 17,
            "steps_between_training_updates": 1,
            "target_update_interval": 0,
        },
        "tracking": {
            "mlflow_step_sample_interval": 1,
            "training_diagnostics_enabled": True,
        },
        "checkpointing": {
            "checkpoint_artifact": "latest_checkpoint.pth",
            "reset_replay_buffer": False,
            "fine_tune": False,
        },
        "topology": {
            "num_agents": 1,
            "observation_dimensions": [3],
            "action_dimensions": [1],
        },
        "algorithm": {
            "name": "PPO",
            "hyperparameters": {"gamma": 0.99, "require_cuda": False},
            "networks": {
                "actor": {"class": "Actor", "layers": [16], "lr": 1.0e-3},
                "critic": {"class": "Critic", "layers": [16], "lr": 1.0e-3},
            },
            "replay_buffer": {
                "class": "OnPolicyRolloutBuffer",
                "capacity": 4,
                "batch_size": 2,
            },
            "exploration": {"strategy": "PPO", "params": exploration},
        },
    }


def _agent(**exploration_overrides) -> PPO:
    agent = PPO(_ppo_config(**exploration_overrides))
    agent.attach_environment(
        observation_names=[["a", "b", "c"]],
        action_names=[["electrical_storage"]],
        action_space=[_Box([-2.0], [2.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    return agent


def _transition(agent: PPO, observation, action) -> None:
    agent.update(
        observations=[observation],
        actions=action,
        rewards=[-0.25],
        next_observations=[observation + 0.1],
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=1,
        update_step=False,
        initial_exploration_done=True,
    )


class _NegativeBehaviorTeacher(torch.nn.Module):
    def predict(self, observations, deterministic=None):
        del observations, deterministic
        return [[-1.0]]


def _single_ev_bc_agent(
    *,
    safety_enabled: bool,
    with_teacher: bool,
) -> tuple[PPO, np.ndarray, np.ndarray]:
    exploration = {
        "local_action_safety_enabled": safety_enabled,
        "actor_behavior_cloning_replay_capacity": 4,
    }
    if with_teacher:
        exploration.update(
            {
                "initial_exploration_strategy": "policy",
                "warm_start_policy": "RandomPolicy",
                "warm_start_policy_deterministic": True,
                "random_exploration_steps": 2,
                "end_initial_exploration_time_step": 2,
                "actor_behavior_cloning_weight": 1.0,
            }
        )

    agent = PPO(_ppo_config(**exploration))
    building = "Building_10"
    charger = "charger_10_1"
    prefix = f"charger::{building}/{charger}::"
    raw_observation_names = [
        f"{prefix}connected_state",
        f"{prefix}max_charging_power_kw",
        f"{prefix}max_discharging_power_kw",
        f"{prefix}available_charge_action_normalized",
        f"{prefix}available_discharge_action_normalized",
        f"{prefix}min_required_action_normalized",
    ]
    raw_observation = np.asarray(
        [1.0, 7.4, 7.4, 1.0, 1.0, 0.5],
        dtype=np.float64,
    )
    actor_observation = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    agent.attach_environment(
        observation_names=[raw_observation_names],
        action_names=[[f"electric_vehicle_storage_{charger}"]],
        action_space=[_Box([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": [building]},
    )
    if with_teacher:
        agent._warm_start_policy = _NegativeBehaviorTeacher()
    return agent, raw_observation, actor_observation


def _store_single_ev_transition(
    agent: PPO,
    raw_observation: np.ndarray,
    actor_observation: np.ndarray,
) -> list[list[float]]:
    agent.set_episode_context(episode_step=0)
    agent.set_observation_context(
        raw_observations=[raw_observation],
        encoded_observations=[actor_observation],
    )
    action = agent.predict([actor_observation], deterministic=False)
    _transition(agent, actor_observation, action)
    return action


def test_policy_rollout_uses_atomic_predict_cache_without_recomputation(monkeypatch):
    agent = _agent()
    observation = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    action = agent.predict([observation], deterministic=False)
    cached = agent._last_policy_samples[0]

    assert cached["stochastic"] is True
    assert torch.allclose(torch.tanh(cached["raw_action"]), cached["normalized_action"])
    np.testing.assert_allclose(action[0], cached["scaled_action"].numpy(), atol=1.0e-7)

    def fail_recomputation(*_args, **_kwargs):
        raise AssertionError("rollout append recomputed a cached prediction")

    monkeypatch.setattr(agent.actors[0], "distribution", fail_recomputation)
    monkeypatch.setattr(agent.value_nets[0], "forward", fail_recomputation)
    _transition(agent, observation, action)

    rollout = agent.rollout[0]
    assert rollout["policy_eligible"].tolist() == [True]
    assert torch.equal(rollout["latent_actions"][0], cached["raw_action"])
    assert torch.equal(rollout["policy_actions"][0], cached["normalized_action"])
    assert torch.equal(rollout["old_log_probs"][0], cached["log_prob"])
    assert torch.equal(rollout["values"][0], cached["value"])
    assert agent._last_policy_samples is None


def test_cached_latent_gives_unit_ratio_before_any_actor_update():
    agent = _agent()
    observation = np.asarray([0.4, 0.2, -0.3], dtype=np.float32)
    action = agent.predict([observation], deterministic=False)
    _transition(agent, observation, action)

    rollout = agent.rollout[0]
    obs_batch = rollout["observations"][0].to(agent.device).view(1, -1)
    latent = rollout["latent_actions"][0].to(agent.device).view(1, -1)
    normalized = rollout["policy_actions"][0].to(agent.device).view(1, -1)
    with torch.no_grad():
        distribution = agent.actors[0].distribution(obs_batch)
        current_log_prob = agent._squashed_log_prob_from_latent(
            distribution,
            latent,
            normalized,
        )
    old_log_prob = rollout["old_log_probs"][0].to(agent.device)

    assert torch.exp(current_log_prob - old_log_prob).item() == pytest.approx(
        1.0,
        abs=1.0e-7,
    )


def test_policy_eligible_update_rejects_an_action_not_returned_by_predict():
    agent = _agent()
    observation = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    action = agent.predict([observation], deterministic=False)
    changed_action = [[float(np.clip(action[0][0] + 0.25, -2.0, 2.0))]]
    if changed_action == action:
        changed_action = [[float(np.clip(action[0][0] - 0.25, -2.0, 2.0))]]

    with pytest.raises(RuntimeError, match="action differs from the action returned"):
        _transition(agent, observation, changed_action)


def test_teacher_controlled_row_has_no_policy_log_prob_and_never_needs_actor_recompute(
    monkeypatch,
):
    agent = _agent(
        warm_start_policy="RandomPolicy",
        warm_start_policy_deterministic=True,
        initial_exploration_strategy="policy",
        end_initial_exploration_time_step=2,
        random_exploration_steps=2,
    )
    observation = np.asarray([0.3, 0.2, 0.1], dtype=np.float32)
    agent.set_observation_context(
        raw_observations=[observation],
        encoded_observations=[observation],
    )
    action = agent.predict([observation], deterministic=False)

    assert agent._last_policy_samples[0]["stochastic"] is False
    assert agent._last_policy_samples[0]["log_prob"] is None

    def fail_recomputation(*_args, **_kwargs):
        raise AssertionError("teacher row evaluated the PPO actor during append")

    monkeypatch.setattr(agent.actors[0], "distribution", fail_recomputation)
    monkeypatch.setattr(agent.value_nets[0], "forward", fail_recomputation)
    _transition(agent, observation, action)

    rollout = agent.rollout[0]
    assert rollout["policy_eligible"].tolist() == [False]
    assert rollout["old_log_probs"].tolist() == pytest.approx([0.0])


def test_actor_branch_of_teacher_phaseout_is_still_excluded_from_ratio(monkeypatch):
    agent = _agent(
        warm_start_policy="RandomPolicy",
        warm_start_policy_deterministic=True,
        warm_start_policy_phaseout_steps=10,
        warm_start_policy_phaseout_mode="probability",
        random_exploration_steps=0,
    )
    observation = np.asarray([-0.2, 0.3, 0.5], dtype=np.float32)
    agent.set_observation_context(
        raw_observations=[observation],
        encoded_observations=[observation],
    )
    # Force the actor branch while the trajectory still belongs to the mixed
    # teacher/actor behaviour policy.
    monkeypatch.setattr(ppo_module.random, "random", lambda: 1.0)
    action = agent.predict([observation], deterministic=False)
    cached_actor_action = agent._last_policy_samples[0]["scaled_action"].numpy()
    np.testing.assert_allclose(action[0], cached_actor_action, atol=1.0e-7)

    _transition(agent, observation, action)
    rollout = agent.rollout[0]
    assert rollout["policy_eligible"].tolist() == [False]
    assert rollout["old_log_probs"].tolist() == pytest.approx([0.0])


def test_projected_action_preserves_latent_ratio_contract() -> None:
    agent = _agent(local_action_safety_enabled=True)
    names = [
        "storage::Building_1/electrical_storage::soc",
        "storage::Building_1/electrical_storage::nominal_power_kw",
        "storage::Building_1/electrical_storage::available_charge_action_normalized",
        "storage::Building_1/electrical_storage::available_discharge_action_normalized",
    ]
    agent.attach_environment(
        observation_names=[names],
        action_names=[["electrical_storage"]],
        action_space=[_Box([-2.0], [2.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    raw_observation = np.asarray([1.0, 5.0, 0.0, 1.0], dtype=np.float32)
    actor_observation = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    agent.set_observation_context(raw_observations=[raw_observation])

    action = agent.predict([actor_observation], deterministic=False)
    cached = agent._last_policy_samples[0]

    assert action[0] == pytest.approx([0.0])
    assert cached["executed_action"].tolist() == pytest.approx([0.0])
    assert cached["scaled_action"].tolist() != pytest.approx([0.0])
    _transition(agent, actor_observation, action)
    rollout = agent.rollout[0]
    assert rollout["actions"][0].tolist() == pytest.approx([0.0])
    assert rollout["latent_actions"][0].tolist() == pytest.approx(
        cached["raw_action"].tolist()
    )


def test_ppo_rollout_and_bc_replay_use_projected_behavior_teacher_target() -> None:
    agent, raw_observation, actor_observation = _single_ev_bc_agent(
        safety_enabled=True,
        with_teacher=True,
    )

    action = _store_single_ev_transition(
        agent,
        raw_observation,
        actor_observation,
    )

    assert action[0] == pytest.approx([0.5])
    assert agent.rollout[0]["teacher_actions"][0].tolist() == pytest.approx([0.5])
    assert (
        agent.behavior_cloning_replay[0]["teacher_actions"][0].tolist()
        == pytest.approx([0.5])
    )
    assert agent._last_raw_behavior_teacher_actions == [[-1.0]]
    assert agent._last_projected_behavior_teacher_actions == [[0.5]]
    metrics = agent.get_diagnostic_metrics()
    assert metrics["PPO/behavior_teacher_raw_projected_target_available"] == 1.0
    assert metrics["PPO/behavior_teacher_raw_projected_projection_applied"] == 1.0
    assert metrics["PPO/behavior_teacher_raw_projected_disagreement_count"] == 1.0
    assert metrics["PPO/behavior_teacher_raw_projected_disagreement_ratio"] == 1.0
    assert metrics["PPO/behavior_teacher_raw_projected_mae"] == pytest.approx(1.5)
    assert metrics["PPO/behavior_teacher_raw_projected_max_abs"] == pytest.approx(1.5)


def test_ppo_behavior_teacher_target_is_unchanged_when_safety_is_disabled() -> None:
    agent, raw_observation, actor_observation = _single_ev_bc_agent(
        safety_enabled=False,
        with_teacher=True,
    )

    action = _store_single_ev_transition(
        agent,
        raw_observation,
        actor_observation,
    )

    assert action[0] == pytest.approx([-1.0])
    assert agent.rollout[0]["teacher_actions"][0].tolist() == pytest.approx([-1.0])
    assert (
        agent.behavior_cloning_replay[0]["teacher_actions"][0].tolist()
        == pytest.approx([-1.0])
    )
    assert agent._last_raw_behavior_teacher_actions == [[-1.0]]
    assert agent._last_projected_behavior_teacher_actions == [[-1.0]]
    metrics = agent.get_diagnostic_metrics()
    assert metrics["PPO/behavior_teacher_raw_projected_target_available"] == 1.0
    assert metrics["PPO/behavior_teacher_raw_projected_projection_applied"] == 0.0
    assert metrics["PPO/behavior_teacher_raw_projected_disagreement_count"] == 0.0


def test_ppo_behavior_teacher_diagnostics_stay_empty_without_teacher() -> None:
    agent, raw_observation, actor_observation = _single_ev_bc_agent(
        safety_enabled=True,
        with_teacher=False,
    )

    _store_single_ev_transition(
        agent,
        raw_observation,
        actor_observation,
    )

    assert agent._last_raw_behavior_teacher_actions is None
    assert agent._last_projected_behavior_teacher_actions is None
    assert len(agent.behavior_cloning_replay) == 0
    assert not torch.isfinite(agent.rollout[0]["teacher_actions"][0]).any()
    metrics = agent.get_diagnostic_metrics()
    assert metrics["PPO/behavior_teacher_raw_projected_target_available"] == 0.0
    assert metrics["PPO/behavior_teacher_raw_projected_projection_applied"] == 0.0
    assert metrics["PPO/behavior_teacher_raw_projected_disagreement_count"] == 0.0


def test_ppo_rejects_onnx_export_that_omits_enabled_safety(tmp_path) -> None:
    agent = _agent(local_action_safety_enabled=True)

    with pytest.raises(RuntimeError, match="not embedded in the ONNX actor"):
        agent.export_artifacts(str(tmp_path))


def test_service_teacher_can_be_disabled_only_for_deterministic_evaluation(
    monkeypatch,
) -> None:
    agent = _agent(
        warm_start_policy="RandomPolicy",
        warm_start_policy_deterministic=True,
        initial_exploration_strategy="policy",
        local_action_safety_service_teacher_enabled=True,
        local_action_safety_service_teacher_eval_enabled=False,
    )
    observation = np.asarray([0.2, 0.1, -0.3], dtype=np.float32)
    agent.set_observation_context(
        raw_observations=[observation],
        encoded_observations=[observation],
    )

    def fail_teacher_query():
        raise AssertionError("deterministic evaluation queried the service teacher")

    monkeypatch.setattr(agent, "_predict_warm_start_policy", fail_teacher_query)
    agent.predict([observation], deterministic=True)

    assert agent._last_service_teacher_applied is False
    with pytest.raises(AssertionError, match="queried the service teacher"):
        agent.predict([observation], deterministic=False)


def test_checkpoint_restores_rollout_storage_on_cpu(tmp_path) -> None:
    agent = _agent()
    observation = np.asarray([0.2, -0.1, 0.4], dtype=np.float32)
    action = agent.predict([observation], deterministic=False)
    _transition(agent, observation, action)
    checkpoint = agent.save_checkpoint(str(tmp_path), step=1)

    restored = _agent()
    restored.load_checkpoint(checkpoint)

    assert len(restored.rollout) == 1
    assert restored.rollout[0]["rewards"].device.type == "cpu"
    assert restored.rollout[0]["observations"][0].device.type == "cpu"


def test_ppo_price_context_changes_only_current_local_price_and_is_inference_only(
    monkeypatch,
) -> None:
    names = [CURRENT_PRICE_NAME, *PREDICTED_PRICE_NAMES, "local_feature"]
    config = _ppo_config(local_price_conditioning_enabled=True)
    config["topology"]["observation_dimensions"] = [len(names)]
    agent = PPO(config)
    low = [0.1, 0.1, 0.1, 0.1, -10.0]
    high = [0.5, 0.5, 0.5, 0.5, 10.0]
    agent.attach_environment(
        observation_names=[names],
        action_names=[["electrical_storage"]],
        action_space=[_Box([-1.0], [1.0])],
        observation_space=[None],
        metadata={
            "building_names": ["Building_1"],
            "raw_observation_names": [names],
            "raw_observation_bounds": [{"low": low, "high": high}],
        },
    )
    observation = np.asarray([0.5, 0.25, 0.5, 0.75, 0.3], dtype=np.float32)
    captured = []

    def capture(observations, *, deterministic):
        captured.append(np.asarray(observations[0]).copy())
        return [[0.0]]

    monkeypatch.setattr(agent, "_predict_actor", capture)
    agent.predict([observation], deterministic=True, context=1.0)
    agent.predict([observation], deterministic=True, context=1.5)

    assert np.array_equal(captured[0], observation)
    assert captured[1][0] == pytest.approx(0.875)
    assert np.array_equal(captured[1][1:], observation[1:])
    assert np.array_equal(
        observation,
        np.asarray([0.5, 0.25, 0.5, 0.75, 0.3], dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="inference-only"):
        _transition(agent, observation, [[0.0]])
