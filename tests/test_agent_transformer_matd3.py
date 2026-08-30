from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from algorithms.registry import ALGORITHM_REGISTRY
from tests._entity_sample_obs_names import (
    load_sample_observation_names_for_first_building,
)


_TOKENIZER_CONFIG = "configs/tokenizers/entity_default.json"
_ACTION_NAMES = ["electrical_storage", "electric_vehicle_storage"]


class _Box:
    def __init__(self, low: list[float], high: list[float]) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


def _config(**overrides) -> dict:
    hyperparameters = {
        "learning_rate": 1.0e-3,
        "gamma": 0.95,
        "tau": 0.25,
        "batch_size": 2,
        "buffer_capacity": 32,
        "max_grad_norm": 1.0,
        "n_step_returns": 1,
        "critic_team_reward_mix": 0.0,
        "reward_normalization_enabled": False,
        "target_policy_smoothing": True,
        "target_policy_noise": 0.2,
        "target_policy_noise_clip": 0.1,
        "actor_update_interval": 2,
        "sigma": 0.4,
        "sigma_decay": 0.5,
        "min_sigma": 0.1,
        "bias": 0.0,
        "noise_clip": 0.25,
        "random_exploration_steps": 0,
    }
    hyperparameters.update(overrides)
    return {
        "algorithm": {
            "name": "AgentTransformerMATD3",
            "tokenizer_config_path": _TOKENIZER_CONFIG,
            "transformer": {
                "d_model": 8,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
            },
            "hyperparameters": hyperparameters,
        }
    }


def _make_agent(*, buildings: int = 2, **overrides):
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3

    observation_names = load_sample_observation_names_for_first_building()
    names = [list(observation_names) for _ in range(buildings)]
    action_names = [list(_ACTION_NAMES) for _ in range(buildings)]
    spaces = [_Box([-2.0, -0.5], [1.0, 0.75]) for _ in range(buildings)]
    agent = AgentTransformerMATD3(_config(**overrides))
    agent.attach_environment(
        observation_names=names,
        action_names=action_names,
        action_space=spaces,
        observation_space=[None] * buildings,
        metadata={"building_names": [f"Building_{i + 1}" for i in range(buildings)]},
    )
    return agent, len(observation_names)


def _transition(agent, obs_dim: int, step: int, *, rewards=None, terminal=False):
    rng = np.random.default_rng(step)
    observations = [rng.standard_normal(obs_dim).astype(np.float32) for _ in agent._per_building]
    next_observations = [
        rng.standard_normal(obs_dim).astype(np.float32)
        for _ in agent._per_building
    ]
    actions = agent.predict(observations, deterministic=True)
    agent.update(
        observations,
        actions,
        rewards or [0.1] * len(agent._per_building),
        next_observations,
        terminal,
        False,
        update_target_step=True,
        global_learning_step=step,
        update_step=True,
        initial_exploration_done=True,
    )


def _parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, module.parameters())
    )


def test_agent_is_registered_for_runtime_construction() -> None:
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3

    assert ALGORITHM_REGISTRY["AgentTransformerMATD3"] is AgentTransformerMATD3


def test_actor_policy_loss_weight_is_configurable() -> None:
    agent, _ = _make_agent(actor_policy_loss_weight=0.085)

    assert agent.actor_policy_loss_weight == pytest.approx(0.085)


def test_critic_loss_defaults_to_mse_and_supports_huber_delta() -> None:
    mse_agent, _ = _make_agent()
    expected = torch.tensor([[3.0, 0.0]])
    target = torch.zeros_like(expected)
    assert mse_agent.critic_loss_type == "mse"
    assert mse_agent._critic_regression_loss(expected, target).item() == pytest.approx(4.5)

    huber_agent, _ = _make_agent(critic_loss_type="huber", critic_huber_delta=1.0)
    assert huber_agent._critic_regression_loss(expected, target).item() == pytest.approx(1.25)


def test_critic_huber_delta_must_be_positive() -> None:
    with pytest.raises(ValueError, match="critic_huber_delta"):
        _make_agent(critic_loss_type="huber", critic_huber_delta=0.0)


def test_policy_replay_q_gap_uses_aligned_actor_update_values() -> None:
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3

    policy = [torch.tensor([[-2.0], [-1.0]])]
    replay = [torch.tensor([[1.0], [1.0]])]
    assert AgentTransformerMATD3._aligned_q_gap(policy, replay) == pytest.approx(2.5)
    assert AgentTransformerMATD3._aligned_q_gap(policy, []) == 0.0


def test_policy_replay_q_gap_is_retained_across_critic_only_flush() -> None:
    agent, _ = _make_agent(buildings=1)
    prefix = "TransformerMATD3/"
    agent._merge_latest_training_metrics(
        {
            f"{prefix}actor_update_performed": 1.0,
            f"{prefix}actor_update_event_count": 3.0,
            f"{prefix}policy_replay_q_abs_gap": 2.5,
        }
    )
    agent._merge_latest_training_metrics(
        {
            f"{prefix}actor_update_performed": 0.0,
            f"{prefix}actor_update_event_count": 3.0,
            f"{prefix}policy_replay_q_abs_gap": 0.0,
        }
    )

    metrics = agent.consume_latest_training_metrics()
    assert metrics[f"{prefix}policy_replay_q_abs_gap"] == pytest.approx(2.5)
    assert metrics[f"{prefix}actor_update_event_count"] == pytest.approx(3.0)


def test_policy_replay_q_gap_reports_zero_for_identical_paired_tensors() -> None:
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3

    values = [torch.tensor([[1.0], [-2.0]])]
    assert AgentTransformerMATD3._aligned_q_gap(values, values) == pytest.approx(0.0)


def test_predict_is_repeatable_and_respects_per_ca_bounds() -> None:
    agent, obs_dim = _make_agent(buildings=1)
    observations = [np.linspace(-1.0, 1.0, obs_dim, dtype=np.float32)]

    first = agent.predict(observations, deterministic=True)
    second = agent.predict(observations, deterministic=True)

    assert first == second
    assert len(first[0]) == 2
    assert np.isfinite(first[0]).all()
    assert np.all(np.asarray(first[0]) >= np.asarray([-2.0, -0.5]))
    assert np.all(np.asarray(first[0]) <= np.asarray([1.0, 0.75]))


def test_exploration_decays_to_sigma_floor_and_stays_bounded() -> None:
    agent, obs_dim = _make_agent(buildings=1)
    observations = [np.zeros(obs_dim, dtype=np.float32)]
    deterministic = agent.predict(observations, deterministic=True)[0]

    noisy = agent.predict(observations, deterministic=False)[0]
    agent.predict(observations, deterministic=False)
    agent.predict(observations, deterministic=False)

    assert noisy != deterministic
    assert np.all(np.asarray(noisy) >= np.asarray([-2.0, -0.5]))
    assert np.all(np.asarray(noisy) <= np.asarray([1.0, 0.75]))
    assert agent.exploration_sigma == pytest.approx(0.1)


def test_initial_exploration_uses_configured_end_boundary() -> None:
    agent, _ = _make_agent(
        random_exploration_steps=2,
        end_initial_exploration_time_step=7,
    )

    assert not agent.is_initial_exploration_done(6)
    assert agent.is_initial_exploration_done(7)

    # The random-action boundary remains independent from the training gate.
    assert agent.random_exploration_steps == 2


def test_learning_uses_twin_target_minimum() -> None:
    agent, obs_dim = _make_agent(buildings=1, target_policy_smoothing=False)
    for critic in agent._per_building[0].critic_1_target.modules():
        if isinstance(critic, torch.nn.Linear):
            torch.nn.init.zeros_(critic.weight)
            torch.nn.init.constant_(critic.bias, 4.0)
    for critic in agent._per_building[0].critic_2_target.modules():
        if isinstance(critic, torch.nn.Linear):
            torch.nn.init.zeros_(critic.weight)
            torch.nn.init.constant_(critic.bias, 2.0)
    _transition(agent, obs_dim, 0, rewards=[1.0])
    _transition(agent, obs_dim, 1, rewards=[1.0])

    metrics = agent.consume_latest_training_metrics()

    assert metrics["TransformerMATD3/q_target_mean"] == pytest.approx(2.9)


def test_actor_and_targets_update_only_on_delayed_due_step() -> None:
    agent, obs_dim = _make_agent(buildings=1, actor_update_interval=2)
    _transition(agent, obs_dim, 0)
    actor_before = _parameters(agent._per_building[0].actor)
    target_before = _parameters(agent._per_building[0].actor_target)
    critic_1_target_before = _parameters(agent._per_building[0].critic_1_target)
    critic_2_target_before = _parameters(agent._per_building[0].critic_2_target)

    _transition(agent, obs_dim, 1)

    assert not _changed(actor_before, agent._per_building[0].actor)
    assert not _changed(target_before, agent._per_building[0].actor_target)

    _transition(agent, obs_dim, 2)

    assert _changed(actor_before, agent._per_building[0].actor)
    assert _changed(target_before, agent._per_building[0].actor_target)
    assert _changed(critic_1_target_before, agent._per_building[0].critic_1_target)
    assert _changed(critic_2_target_before, agent._per_building[0].critic_2_target)


def test_actor_delay_counts_successful_critic_updates_not_environment_steps() -> None:
    agent, obs_dim = _make_agent(
        buildings=1,
        batch_size=1,
        actor_update_interval=2,
        minimum_successful_critic_updates_before_actor=1,
    )

    _transition(agent, obs_dim, 0)
    assert agent.critic_update_count == 1
    assert agent.actor_update_count == 0
    _transition(agent, obs_dim, 1)

    assert agent.critic_update_count == 2
    assert agent.actor_update_count == 1
    assert agent.target_update_count == 1


def test_pending_actor_metrics_survive_intervening_critic_event() -> None:
    agent, _ = _make_agent(buildings=1)
    agent._merge_latest_training_metrics(
        {
            "TransformerMATD3/actor_update_performed": 1.0,
            "TransformerMATD3/actor_policy_loss_mean": 2.5,
            "TransformerMATD3/policy_action_q_mean": -3.0,
        }
    )
    agent._merge_latest_training_metrics(
        {
            "TransformerMATD3/actor_update_performed": 0.0,
            "TransformerMATD3/actor_policy_loss_mean": 0.0,
            "TransformerMATD3/policy_action_q_mean": 0.0,
            "TransformerMATD3/critic_loss_mean": 1.25,
        }
    )

    metrics = agent.consume_latest_training_metrics()

    assert metrics["TransformerMATD3/actor_update_performed"] == 1.0
    assert metrics["TransformerMATD3/actor_policy_loss_mean"] == 2.5
    assert metrics["TransformerMATD3/policy_action_q_mean"] == -3.0
    assert metrics["TransformerMATD3/critic_loss_mean"] == 1.25


def test_action_diagnostic_maxima_are_not_cumulative() -> None:
    agent, _ = _make_agent(buildings=1)
    agent._record_action_diagnostics(
        base_actions=[[0.0, 0.0]],
        proposed_actions=[[0.2, 0.1]],
        executed_actions=[[0.0, 0.0]],
        raw_proposed_actions=[[0.2, 0.1]],
        exploration=False,
    )
    agent._record_action_diagnostics(
        base_actions=[[0.0, 0.0]],
        proposed_actions=[[0.5, 0.05]],
        executed_actions=[[0.0, 0.0]],
        raw_proposed_actions=[[0.5, 0.05]],
        exploration=False,
    )

    metrics = agent.get_diagnostic_metrics()

    assert metrics["TransformerMATD3/base_proposed_abs_delta_storage_max"] == pytest.approx(0.5)
    assert metrics["TransformerMATD3/proposed_executed_abs_delta_storage_max"] == pytest.approx(0.5)


def test_skipped_replay_does_not_advance_critic_update_count() -> None:
    agent, obs_dim = _make_agent(buildings=1, batch_size=2)

    _transition(agent, obs_dim, 0)

    assert agent.critic_update_count == 0
    assert agent.actor_update_count == 0


def test_target_policy_smoothing_is_clipped_per_ca(monkeypatch) -> None:
    agent, obs_dim = _make_agent(
        buildings=1,
        target_policy_noise=2.0,
        target_policy_noise_clip=0.1,
    )
    state = agent._per_building[0]
    observations = torch.zeros((3, obs_dim), device=agent.device)
    base = agent._actor_action(state, observations, target=True)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))

    smoothed = agent._target_action(state, observations)

    expected_limit = 0.1 * (state.action_high - state.action_low)
    assert torch.all(smoothed - base <= expected_limit + 1.0e-6)
    assert torch.all(smoothed <= state.action_high)
    assert torch.all(smoothed >= state.action_low)


def test_team_reward_mix_replaces_individual_rewards_at_one() -> None:
    agent, obs_dim = _make_agent(buildings=2, critic_team_reward_mix=1.0)
    _transition(agent, obs_dim, 0, rewards=[1.0, 3.0])
    _transition(agent, obs_dim, 1, rewards=[1.0, 3.0])

    assert agent._last_train_rewards is not None
    assert torch.allclose(agent._last_train_rewards[:, 0], torch.full((2,), 2.0))
    assert torch.allclose(agent._last_train_rewards[:, 1], torch.full((2,), 2.0))


def test_reward_normalization_uses_global_welford_state() -> None:
    agent, obs_dim = _make_agent(
        buildings=1,
        reward_normalization_enabled=True,
    )
    _transition(agent, obs_dim, 0, rewards=[1.0])
    _transition(agent, obs_dim, 1, rewards=[3.0])

    assert agent.reward_norm_count == 2
    assert agent.reward_norm_mean == pytest.approx(2.0)
    assert agent.reward_norm_m2 == pytest.approx(2.0)
    normalized = agent._normalize_reward_tensor(
        torch.tensor([[1.0], [3.0]], device=agent.device)
    )
    assert normalized.cpu().reshape(-1).tolist() == pytest.approx(
        [-2.0**-0.5, 2.0**-0.5]
    )
    assert agent._last_train_rewards is not None
    assert torch.isfinite(agent._last_train_rewards).all()


def test_reward_normalization_keeps_per_building_scales_independent() -> None:
    agent, _ = _make_agent(
        buildings=2,
        reward_normalization_enabled=True,
        reward_normalization_scope="per_building",
    )

    agent._update_reward_normalizer([1.0, 100.0])
    agent._update_reward_normalizer([3.0, 300.0])
    normalized = agent._normalize_reward_tensor(
        torch.tensor([[3.0, 300.0]], device=agent.device)
    )

    assert agent.reward_norm_counts.tolist() == [2, 2]
    assert agent.reward_norm_means.tolist() == pytest.approx([2.0, 200.0])
    assert agent.reward_norm_m2s.tolist() == pytest.approx([2.0, 20000.0])
    assert normalized.cpu().tolist()[0] == pytest.approx(
        [2.0**-0.5, 2.0**-0.5]
    )


def test_extreme_building_does_not_change_other_normalized_reward() -> None:
    common = {
        "buildings": 2,
        "reward_normalization_enabled": True,
        "reward_normalization_scope": "per_building",
    }
    reference, _ = _make_agent(**common)
    extreme, _ = _make_agent(**common)
    for left, right in (([1.0, 10.0], [1.0, 10.0]), ([3.0, 30.0], [3.0, 3.0e9])):
        reference._update_reward_normalizer(left)
        extreme._update_reward_normalizer(right)

    reward = torch.tensor([[2.5, 20.0]])
    reference_value = reference._normalize_reward_tensor(reward)[0, 0]
    extreme_value = extreme._normalize_reward_tensor(reward)[0, 0]

    assert extreme_value.item() == pytest.approx(reference_value.item())


def test_per_building_reward_normalization_skips_non_finite_statistics() -> None:
    agent, _ = _make_agent(
        buildings=2,
        reward_normalization_enabled=True,
        reward_normalization_scope="per_building",
    )

    agent._update_reward_normalizer([1.0, float("nan")])
    agent._update_reward_normalizer([3.0, float("inf")])
    normalized = agent._normalize_reward_tensor(
        torch.tensor([[3.0, float("nan")]], device=agent.device)
    )

    assert agent.reward_norm_counts.tolist() == [2, 0]
    assert torch.isfinite(normalized[0, 0])
    assert torch.isnan(normalized[0, 1])


def test_n_step_returns_flush_terminal_tail() -> None:
    agent, obs_dim = _make_agent(
        buildings=1,
        batch_size=1,
        n_step_returns=3,
        n_step_gamma=0.5,
    )
    _transition(agent, obs_dim, 0, rewards=[1.0])
    _transition(agent, obs_dim, 1, rewards=[2.0])
    assert agent.replay_buffer.total_size() == 0

    _transition(agent, obs_dim, 2, rewards=[4.0], terminal=True)

    state = agent.replay_buffer.get_state()
    rewards = [item.rewards[0] for item in state["transitions"]]
    assert rewards == pytest.approx([3.0, 4.0, 4.0])
    assert all(item.terminated[0] for item in state["transitions"])


def test_static_fixture_completes_multiple_finite_learning_steps() -> None:
    agent, obs_dim = _make_agent(buildings=2)

    for step in range(5):
        _transition(agent, obs_dim, step)

    metrics = agent.consume_latest_training_metrics()
    required = {
        "TransformerMATD3/critic_1_loss_mean",
        "TransformerMATD3/critic_2_loss_mean",
        "TransformerMATD3/actor_loss_mean",
        "TransformerMATD3/training_step_time",
    }
    assert required <= metrics.keys()
    assert all(np.isfinite(metrics[name]) for name in required)


def test_runtime_profiling_reports_transformer_training_phases() -> None:
    agent, obs_dim = _make_agent(buildings=1)
    agent.runtime_profiling_enabled = True
    agent.runtime_profiling_interval = 2

    _transition(agent, obs_dim, 0)
    _transition(agent, obs_dim, 1)
    _transition(agent, obs_dim, 2)

    metrics = agent.consume_latest_training_metrics()
    required = {
        "TransformerMATD3/runtime_replay_push_seconds",
        "TransformerMATD3/runtime_replay_sample_seconds",
        "TransformerMATD3/runtime_target_compute_seconds",
        "TransformerMATD3/runtime_critic_update_seconds",
        "TransformerMATD3/runtime_actor_update_seconds",
        "TransformerMATD3/runtime_bc_a_extra_seconds",
        "TransformerMATD3/runtime_bc_b_auxiliary_seconds",
        "TransformerMATD3/runtime_training_step_seconds",
    }
    assert required <= metrics.keys()
    assert all(np.isfinite(metrics[name]) and metrics[name] >= 0.0 for name in required)


def test_learning_skips_actor_update_for_building_without_actions() -> None:
    agent, obs_dim = _make_agent(buildings=2, batch_size=2)
    state = agent._per_building[1]
    state.layout = replace(
        state.layout,
        segments=tuple(
            segment for segment in state.layout.segments if segment.family != "ca"
        ),
        n_ca=0,
        ca_action_names=(),
    )
    state.action_names = ()
    state.action_low = state.action_low[:0]
    state.action_high = state.action_high[:0]
    agent._layout_signature = agent._build_layout_signature(agent._layouts)

    _transition(agent, obs_dim, 0)
    _transition(agent, obs_dim, 2)
    _transition(agent, obs_dim, 3)

    metrics = agent.consume_latest_training_metrics()
    assert metrics["TransformerMATD3/actor_update_performed"] == 1.0
    assert np.isfinite(metrics["TransformerMATD3/actor_loss_mean"])
