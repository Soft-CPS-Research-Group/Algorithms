"""AgentTransformerPPO unit tests.

Covers:
- ``predict`` returns ``[B][N_ca]`` floats clamped to ``[-1, 1]``.
- ``update`` accumulates rollouts and runs PPO step on ``update_step``.
- Topology change rebuilds layout, preserves weights for stable types.
- Layout-drift on existing type (feature-count change) hard-fails.
- New type appearing on existing tokenizer hard-fails.
- ``save_checkpoint`` / ``load_checkpoint`` round-trip + signature mismatch
  rejection.
- ``export_artifacts`` returns a well-formed manifest with one ONNX file
  per building.
- Registered in ``algorithms.registry.ALGORITHM_REGISTRY``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from gymnasium import spaces

from algorithms.agents.agent_transformer_ppo import (
    AgentTransformerPPO,
    _synthetic_sample_from_obs_names,
)
from algorithms.agents.base_agent import BaseAgent
from algorithms.registry import ALGORITHM_REGISTRY, build_execution_unit
from tests._entity_sample_obs_names import (
    load_sample_observation_names_for_first_building,
)


_TOKENIZER_CFG = "configs/tokenizers/entity_default.json"
_DEFAULT_ACTIONS = ["electrical_storage", "electric_vehicle_storage"]


def _base_config() -> dict:
    return {
        "algorithm": {
            "name": "AgentTransformerPPO",
            "tokenizer_config_path": _TOKENIZER_CFG,
            "transformer": {
                "d_model": 16,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
            },
            "hyperparameters": {
                "learning_rate": 1.0e-3,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "ppo_epochs": 1,
                "minibatch_size": 4,
                "entropy_coeff": 0.0,
                "value_coeff": 0.5,
                "max_grad_norm": 0.5,
                "actor_hidden_dim": 32,
                "critic_hidden_dim": 32,
            },
        },
    }


def _make_agent(n_buildings: int = 1) -> tuple[AgentTransformerPPO, List[List[str]], List[List[str]], int]:
    return _make_agent_with_config(_base_config(), n_buildings=n_buildings)


def _make_agent_with_config(
    config: dict,
    n_buildings: int = 1,
) -> tuple[AgentTransformerPPO, List[List[str]], List[List[str]], int]:
    obs_names = load_sample_observation_names_for_first_building()
    obs_names_per = [list(obs_names) for _ in range(n_buildings)]
    act_names_per = [list(_DEFAULT_ACTIONS) for _ in range(n_buildings)]
    agent = AgentTransformerPPO(config)
    agent.attach_environment(
        observation_names=obs_names_per,
        action_names=act_names_per,
        action_space=[None] * n_buildings,
        observation_space=[None] * n_buildings,
        metadata={"building_names": [f"Building_{b+1}" for b in range(n_buildings)]},
    )
    obs_dim = max(
        max(seg.feature_indices) for seg in agent._per_building[0].layout.segments
    ) + 1
    return agent, obs_names_per, act_names_per, obs_dim


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered_under_canonical_name() -> None:
    assert ALGORITHM_REGISTRY.get("AgentTransformerPPO") is AgentTransformerPPO


def test_base_agent_declares_no_op_episode_lifecycle_hooks() -> None:
    assert "on_episode_start" in BaseAgent.__dict__
    assert "on_episode_end" in BaseAgent.__dict__

    BaseAgent.on_episode_start(object(), episode=0, training=True)
    BaseAgent.on_episode_end(object(), episode=0, training=False)


def test_create_agent_via_registry() -> None:
    base = _base_config()
    algo = base.pop("algorithm")
    stage = {"algorithm": algo.pop("name")}
    stage.update(algo)
    base["pipeline"] = [stage]
    agent = build_execution_unit(base)
    assert isinstance(agent, AgentTransformerPPO)


def test_supports_dynamic_topology_classvar_true() -> None:
    assert AgentTransformerPPO.supports_dynamic_topology is True


# ---------------------------------------------------------------------------
# predict / update
# ---------------------------------------------------------------------------


def test_predict_shape_and_range() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=2)
    obs = [np.random.rand(obs_dim).astype(np.float64) for _ in range(2)]
    actions = agent.predict(obs, deterministic=False)
    assert isinstance(actions, list) and len(actions) == 2
    for b, vec in enumerate(actions):
        assert isinstance(vec, list)
        assert len(vec) == agent._per_building[b].layout.n_ca
        for v in vec:
            assert -1.0 <= v <= 1.0


def test_predict_deterministic_is_repeatable() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    obs = [np.zeros(obs_dim, dtype=np.float64)]
    a1 = agent.predict(obs, deterministic=True)
    a2 = agent.predict(obs, deterministic=True)
    assert a1 == a2


def test_predict_affinely_maps_tanh_actions_to_declared_bounds_and_preserves_ppo_ratio() -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=1)
    bounds = spaces.Box(
        low=np.array([-0.75, -0.25], dtype=np.float32),
        high=np.array([-0.5, 0.5], dtype=np.float32),
        dtype=np.float32,
    )
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[bounds],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    state = agent._per_building[0]
    actor_output = state.actor.mlp[-1]
    assert isinstance(actor_output, torch.nn.Linear)
    with torch.no_grad():
        actor_output.weight.zero_()
        actor_output.bias.fill_(0.5)

    expected = bounds.low + (np.tanh(0.5) + 1.0) * (bounds.high - bounds.low) / 2.0

    for step in range(agent._minibatch_size):
        observations = [np.zeros(obs_dim, dtype=np.float64)]
        actions = agent.predict(observations, deterministic=True)
        cached = agent._pending_decisions[0]
        assert cached is not None
        np.testing.assert_array_equal(
            np.asarray(actions[0]), cached.action.cpu().numpy().squeeze(-1)
        )
        assert np.all(np.asarray(actions[0]) >= bounds.low)
        assert np.all(np.asarray(actions[0]) <= bounds.high)
        # The same tanh sample is affinely mapped per dimension. Clamping
        # would produce [-0.5, tanh(0.5)] instead.
        np.testing.assert_allclose(actions[0], expected, rtol=1.0e-6, atol=1.0e-6)
        agent.update(
            observations=observations,
            actions=[np.asarray(actions[0])],
            rewards=[0.1],
            next_observations=observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=step == agent._minibatch_size - 1,
            initial_exploration_done=True,
        )

    assert agent.consume_latest_training_metrics()["ratio_error_max"] <= 1.0e-5


def test_predict_keeps_saturated_actor_log_prob_for_ppo_ratio() -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=1)
    bounds = spaces.Box(
        low=np.array([-0.75, -0.25], dtype=np.float32),
        high=np.array([-0.5, 0.5], dtype=np.float32),
        dtype=np.float32,
    )
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[bounds],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    state = agent._per_building[0]
    for parameter in state.actor.parameters():
        parameter.data.zero_()
    state.actor.mlp[-1].bias.data.fill_(20.0)
    observations = [np.zeros(obs_dim, dtype=np.float64)]

    actions = agent.predict(observations, deterministic=True)
    pending = agent._pending_decisions[0]
    assert pending is not None
    low, high = agent._action_bounds[0]
    with torch.no_grad():
        observation = pending.observation.unsqueeze(0)
        tokenized = state.tokenizer(observation, state.layout)
        ca_embeddings, _ = state.backbone(
            tokenized.sro_tokens,
            tokenized.nfc_token,
            tokenized.ca_tokens,
        )
        _, raw_log_probs, _ = state.actor(ca_embeddings, deterministic=True)
        expected = raw_log_probs - torch.log((high - low) / 2.0).squeeze(-1)
        tanh_actions = (2.0 * (pending.action.unsqueeze(0) - low) / (high - low)) - 1.0
        legacy_pre_tanh = torch.atanh(tanh_actions.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
        std = torch.exp(state.actor.log_std.clamp(-2.0, 0.5)).expand_as(legacy_pre_tanh)
        legacy = (
            torch.distributions.Normal(torch.full_like(legacy_pre_tanh, 20.0), std)
            .log_prob(legacy_pre_tanh)
            - torch.log(1.0 - tanh_actions.pow(2) + 1.0e-6)
            - torch.log((high - low) / 2.0)
        ).squeeze(-1)

    assert torch.equal(tanh_actions, torch.ones_like(tanh_actions))
    assert not torch.allclose(legacy, expected)
    assert torch.allclose(pending.log_prob, expected.squeeze(0), rtol=1.0e-6, atol=1.0e-6)

    agent.update(
        observations=observations,
        actions=[np.asarray(actions[0])],
        rewards=[0.1],
        next_observations=observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )
    assert torch.equal(state.buffer.log_probs[-1], expected.squeeze(0).cpu())
    with torch.no_grad():
        recomputed = agent._compute_log_prob(
            state.actor,
            ca_embeddings,
            state.buffer.pre_tanh_actions[-1].unsqueeze(0),
            low,
            high,
        )
    ratio = torch.exp((recomputed - state.buffer.log_probs[-1].unsqueeze(0)).sum())
    assert abs(ratio.item() - 1.0) <= 1.0e-5


def test_bounds_only_reattach_rejects_nonempty_rollout_transactionally() -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=1)
    observations = [np.zeros(obs_dim, dtype=np.float64)]
    actions = agent.predict(observations, deterministic=True)
    agent.update(
        observations=observations,
        actions=[np.asarray(actions[0])],
        rewards=[0.1],
        next_observations=observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )
    state = agent._per_building[0]
    bounds_before = agent._action_bounds
    buffer_before = copy.deepcopy(state.buffer)
    new_bounds = spaces.Box(
        low=np.array([0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="cannot change action bounds"):
        agent.attach_environment(
            observation_names=obs_per,
            action_names=act_per,
            action_space=[new_bounds],
            observation_space=[None],
            metadata={"building_names": ["Building_1"]},
        )

    assert agent._per_building[0] is state
    assert agent._action_bounds is bounds_before
    assert state.buffer.rewards == buffer_before.rewards
    assert state.buffer.terminated == buffer_before.terminated
    assert state.buffer.truncated == buffer_before.truncated
    for field in ("observations", "actions", "log_probs", "values"):
        actual = getattr(state.buffer, field)
        expected = getattr(buffer_before, field)
        assert len(actual) == len(expected)
        assert all(torch.equal(value, expected[idx]) for idx, value in enumerate(actual))

    state.buffer.clear()
    agent.predict(observations, deterministic=True)
    pending_before_bounds_change = agent._pending_decisions[0]
    assert pending_before_bounds_change is not None
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[new_bounds],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    low, high = agent._action_bounds[0]
    assert torch.equal(low.squeeze(-1), torch.tensor([0.0, 0.0]))
    assert torch.equal(high.squeeze(-1), torch.tensor([1.0, 1.0]))
    assert agent._pending_decisions[0] is None


def test_topology_reattach_rejects_unchanged_building_bound_change_with_rollout() -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=2)
    observations = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]
    initial_bounds = [
        spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        for _ in range(2)
    ]
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=initial_bounds,
        observation_space=[None, None],
        metadata={"building_names": ["Building_1", "Building_2"]},
    )

    actions = agent.predict(observations, deterministic=True)
    agent.update(
        observations=observations,
        actions=[np.asarray(action) for action in actions],
        rewards=[0.1, 0.2],
        next_observations=observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )
    agent._per_building[0].buffer.clear()
    agent.predict(observations, deterministic=True)

    charger_id = next(
        name.split("::")[1]
        for name in obs_per[0]
        if name.startswith("charger::")
        and "::connected_ev::" not in name
        and "::incoming_ev::" not in name
    )
    replacement_obs_0 = list(obs_per[0]) + [
        name.replace(
            f"charger::{charger_id}::", "charger::Building_1/charger_NEW::", 1
        )
        for name in obs_per[0]
        if name.startswith(f"charger::{charger_id}::")
    ]
    replacement_actions_0 = list(act_per[0]) + ["electric_vehicle_storage"]
    replacement_bounds = [
        spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        ),
        spaces.Box(
            low=np.array([-0.5, -0.25], dtype=np.float32),
            high=np.array([0.5, 0.75], dtype=np.float32),
            dtype=np.float32,
        ),
    ]
    snapshots = [
        {
            "state": state,
            "names": (state.obs_names_tuple, state.action_names_tuple),
            "layout": state.layout,
            "topology_version": state.topology_version,
            "buffer": copy.deepcopy(state.buffer),
            "pending": agent._pending_decisions[index],
        }
        for index, state in enumerate(agent._per_building)
    ]
    bounds_before = agent._action_bounds

    with pytest.raises(ValueError, match="cannot change action bounds"):
        agent.attach_environment(
            observation_names=[replacement_obs_0, obs_per[1]],
            action_names=[replacement_actions_0, act_per[1]],
            action_space=replacement_bounds,
            observation_space=[None, None],
            metadata={"building_names": ["Building_1", "Building_2"]},
        )

    assert agent._action_bounds is bounds_before
    for index, snapshot in enumerate(snapshots):
        state = agent._per_building[index]
        assert state is snapshot["state"]
        assert (state.obs_names_tuple, state.action_names_tuple) == snapshot["names"]
        assert state.layout is snapshot["layout"]
        assert state.topology_version == snapshot["topology_version"]
        assert state.buffer.rewards == snapshot["buffer"].rewards
        assert state.buffer.terminated == snapshot["buffer"].terminated
        assert state.buffer.truncated == snapshot["buffer"].truncated
        assert agent._pending_decisions[index] is snapshot["pending"]
        for field in ("observations", "actions", "log_probs", "values"):
            actual = getattr(state.buffer, field)
            expected = getattr(snapshot["buffer"], field)
            assert len(actual) == len(expected)
            assert all(torch.equal(value, expected[idx]) for idx, value in enumerate(actual))

    agent._per_building[1].buffer.clear()
    agent.attach_environment(
        observation_names=[replacement_obs_0, obs_per[1]],
        action_names=[replacement_actions_0, act_per[1]],
        action_space=replacement_bounds,
        observation_space=[None, None],
        metadata={"building_names": ["Building_1", "Building_2"]},
    )

    assert agent._per_building[0].action_names_tuple == tuple(replacement_actions_0)
    assert agent._per_building[0].topology_version == 1
    assert agent._per_building[1].action_names_tuple == tuple(act_per[1])
    assert agent._per_building[1].topology_version == 0
    assert [bound.flatten().tolist() for bound in agent._action_bounds[1]] == [
        [-0.5, -0.25],
        [0.5, 0.75],
    ]
    assert agent._pending_decisions == [None, None]


def test_predict_rejects_wrong_cardinality_without_replacing_pending_decisions() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=2)
    observations = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]
    agent.predict(observations, deterministic=True)
    pending_before = list(agent._pending_decisions)

    for invalid_observations in (observations[:1], observations + [observations[0]]):
        with pytest.raises(ValueError, match="TPPO predict observations has .* expected 2"):
            agent.predict(invalid_observations, deterministic=True)
        assert agent._pending_decisions == pending_before
        assert all(
            actual is expected
            for actual, expected in zip(agent._pending_decisions, pending_before)
        )

    agent.predict(observations, deterministic=True)
    assert all(
        actual is not expected
        for actual, expected in zip(agent._pending_decisions, pending_before)
    )


def test_predict_later_malformed_row_preserves_pending_decisions_for_valid_retry() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=2)
    observations = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]
    agent.predict(observations, deterministic=True)
    pending_before = list(agent._pending_decisions)

    with pytest.raises(RuntimeError, match="out of DATA bounds"):
        agent.predict(
            [observations[0], np.zeros(obs_dim - 1, dtype=np.float64)],
            deterministic=True,
        )

    assert all(
        actual is expected
        for actual, expected in zip(agent._pending_decisions, pending_before)
    )

    agent.predict(observations, deterministic=True)
    assert all(
        actual is not expected
        for actual, expected in zip(agent._pending_decisions, pending_before)
    )


def test_predict_caches_exact_decision_for_ppo_update_with_dropout() -> None:
    config = _base_config()
    config["algorithm"]["transformer"]["dropout"] = 0.1
    agent, _, _, obs_dim = _make_agent_with_config(config)
    state = agent._per_building[0]
    rng = np.random.default_rng(17)

    for step in range(4):
        obs = [rng.standard_normal(obs_dim)]
        actions = agent.predict(obs, deterministic=False)
        cached = agent._pending_decisions[0]
        assert cached is not None
        agent.update(
            observations=obs,
            actions=[np.asarray(actions[0])],
            rewards=[0.1],
            next_observations=[rng.standard_normal(obs_dim)],
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=step == 3,
            initial_exploration_done=True,
        )
        if step < 3:
            assert torch.equal(state.buffer.actions[-1], cached.action.cpu())
            assert torch.equal(state.buffer.log_probs[-1], cached.log_prob.cpu())
            assert torch.equal(state.buffer.values[-1], cached.value.cpu())
            assert agent._pending_decisions[0] is None

    metrics = agent.consume_latest_training_metrics()
    assert metrics["ratio_error_max"] <= 1.0e-5


def test_update_rejects_action_that_differs_from_pending_decision() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    obs = [np.zeros(obs_dim, dtype=np.float64)]
    actions = agent.predict(obs, deterministic=False)
    actions[0][0] += 0.1

    with pytest.raises(ValueError, match="does not match the pending TPPO action"):
        agent.update(
            observations=obs,
            actions=[np.asarray(actions[0])],
            rewards=[0.1],
            next_observations=obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=0,
            update_step=False,
            initial_exploration_done=True,
        )


def test_update_rejects_action_within_previous_allclose_tolerance() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    obs = [np.zeros(obs_dim, dtype=np.float64)]
    actions = agent.predict(obs, deterministic=False)
    actions[0][0] += 5.0e-7

    with pytest.raises(ValueError, match="does not match the pending TPPO action"):
        agent.update(
            observations=obs,
            actions=[np.asarray(actions[0])],
            rewards=[0.1],
            next_observations=obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=0,
            update_step=False,
            initial_exploration_done=True,
        )


def test_update_rejection_preserves_global_step_until_successful_retry() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=2)
    obs = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]
    actions = agent.predict(obs, deterministic=False)
    mismatched_actions = [list(action) for action in actions]
    mismatched_actions[1][0] += 0.1
    original_step = agent._latest_global_learning_step

    with pytest.raises(ValueError, match="does not match the pending TPPO action"):
        agent.update(
            observations=obs,
            actions=[np.asarray(action) for action in mismatched_actions],
            rewards=[0.1, 0.1],
            next_observations=obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=11,
            update_step=False,
            initial_exploration_done=True,
        )

    assert len(agent._per_building[0].buffer) == 0
    assert agent._pending_decisions[0] is not None
    assert agent._latest_global_learning_step == original_step

    agent.update(
        observations=obs,
        actions=[np.asarray(action) for action in actions],
        rewards=[0.1, 0.1],
        next_observations=obs,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=12,
        update_step=False,
        initial_exploration_done=True,
    )

    assert [len(state.buffer) for state in agent._per_building] == [1, 1]
    assert agent._pending_decisions == [None, None]
    assert agent._latest_global_learning_step == 12


@pytest.mark.parametrize("malformed_field", ["reward", "next_observation"])
def test_update_later_conversion_failure_preserves_all_state_for_valid_retry(
    malformed_field: str,
) -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=2)
    observations = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]

    # Seed each rollout, then cache the decisions that the rejected update must keep.
    initial_actions = agent.predict(observations, deterministic=True)
    agent.update(
        observations=observations,
        actions=[np.asarray(action) for action in initial_actions],
        rewards=[0.1, 0.2],
        next_observations=observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=3,
        update_step=False,
        initial_exploration_done=True,
    )
    actions = agent.predict(observations, deterministic=True)
    buffers_before = [copy.deepcopy(state.buffer) for state in agent._per_building]
    pending_before = list(agent._pending_decisions)
    global_step_before = agent._latest_global_learning_step
    rewards: List[object] = [0.3, 0.4]
    next_observations: List[object] = list(observations)
    if malformed_field == "reward":
        rewards[1] = "not-a-reward"
    else:
        next_observations[1] = object()

    with pytest.raises((TypeError, ValueError)):
        agent.update(
            observations=observations,
            actions=[np.asarray(action) for action in actions],
            rewards=rewards,  # type: ignore[arg-type]
            next_observations=next_observations,  # type: ignore[arg-type]
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=4,
            update_step=False,
            initial_exploration_done=True,
        )

    for index, state in enumerate(agent._per_building):
        before = buffers_before[index]
        assert state.buffer.rewards == before.rewards
        assert state.buffer.terminated == before.terminated
        assert state.buffer.truncated == before.truncated
        for field in ("observations", "actions", "log_probs", "values"):
            actual = getattr(state.buffer, field)
            expected = getattr(before, field)
            assert len(actual) == len(expected)
            assert all(torch.equal(value, expected[i]) for i, value in enumerate(actual))
        assert agent._pending_decisions[index] is pending_before[index]
    assert agent._latest_global_learning_step == global_step_before

    agent.update(
        observations=observations,
        actions=[np.asarray(action) for action in actions],
        rewards=[0.3, 0.4],
        next_observations=observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=4,
        update_step=False,
        initial_exploration_done=True,
    )

    assert [len(state.buffer) for state in agent._per_building] == [2, 2]
    assert agent._pending_decisions == [None, None]
    assert agent._latest_global_learning_step == 4


def test_update_step_bootstrap_failure_preserves_state_for_valid_retry() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    observations = [np.zeros(obs_dim, dtype=np.float64)]

    for step in range(agent._minibatch_size - 1):
        actions = agent.predict(observations, deterministic=True)
        agent.update(
            observations=observations,
            actions=[np.asarray(actions[0])],
            rewards=[0.1],
            next_observations=observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )

    actions = agent.predict(observations, deterministic=True)
    state = agent._per_building[0]
    buffer_before = copy.deepcopy(state.buffer)
    pending_before = agent._pending_decisions[0]
    global_step_before = agent._latest_global_learning_step
    metrics_before = copy.deepcopy(agent._latest_training_metrics)

    with pytest.raises((IndexError, RuntimeError)):
        agent.update(
            observations=observations,
            actions=[np.asarray(actions[0])],
            rewards=[0.1],
            next_observations=[np.zeros(1, dtype=np.float64)],
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=agent._minibatch_size - 1,
            update_step=True,
            initial_exploration_done=True,
        )

    assert state.buffer.rewards == buffer_before.rewards
    assert state.buffer.terminated == buffer_before.terminated
    assert state.buffer.truncated == buffer_before.truncated
    for field in ("observations", "actions", "log_probs", "values"):
        actual = getattr(state.buffer, field)
        expected = getattr(buffer_before, field)
        assert len(actual) == len(expected)
        assert all(torch.equal(value, expected[i]) for i, value in enumerate(actual))
    assert agent._pending_decisions[0] is pending_before
    assert agent._latest_global_learning_step == global_step_before
    assert agent._latest_training_metrics == metrics_before

    agent.update(
        observations=observations,
        actions=[np.asarray(actions[0])],
        rewards=[0.1],
        next_observations=observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=agent._minibatch_size - 1,
        update_step=True,
        initial_exploration_done=True,
    )

    assert len(state.buffer) == 0
    assert agent._pending_decisions == [None]
    assert agent._latest_global_learning_step == agent._minibatch_size - 1


def test_update_rejects_extra_action_rows_without_mutating_state() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=2)
    observations = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]
    actions = agent.predict(observations)
    pending = list(agent._pending_decisions)
    global_step = agent._latest_global_learning_step

    with pytest.raises(ValueError, match="actions has 3 rows; expected 2"):
        agent.update(
            observations=observations,
            actions=[np.asarray(action) for action in actions] + [np.zeros(1)],
            rewards=[0.1, 0.2],
            next_observations=observations,
            terminated=[False, False],
            truncated=[False, False],
            update_target_step=False,
            global_learning_step=4,
            update_step=False,
            initial_exploration_done=True,
        )

    assert [len(state.buffer) for state in agent._per_building] == [0, 0]
    assert agent._pending_decisions == pending
    assert agent._latest_global_learning_step == global_step


def test_update_rejects_later_reward_cardinality_mismatch_without_mutating_state() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=2)
    observations = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]
    actions = agent.predict(observations)
    pending = list(agent._pending_decisions)
    global_step = agent._latest_global_learning_step

    with pytest.raises(ValueError, match="rewards has 1 rows; expected 2"):
        agent.update(
            observations=observations,
            actions=[np.asarray(action) for action in actions],
            rewards=[0.1],
            next_observations=observations,
            terminated=[False, False],
            truncated=[False, False],
            update_target_step=False,
            global_learning_step=5,
            update_step=False,
            initial_exploration_done=True,
        )

    assert [len(state.buffer) for state in agent._per_building] == [0, 0]
    assert agent._pending_decisions == pending
    assert agent._latest_global_learning_step == global_step


def test_actor_log_std_init_and_cpu_device_are_configured() -> None:
    config = _base_config()
    config["algorithm"]["hyperparameters"]["actor_log_std_init"] = -0.25
    config["algorithm"]["hyperparameters"]["require_cuda"] = False
    agent, _, _, _ = _make_agent_with_config(config)

    assert agent.device.type == "cpu"
    assert agent._per_building[0].actor.log_std.item() == pytest.approx(-0.25)


def test_require_cuda_rejects_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _base_config()
    config["algorithm"]["hyperparameters"]["require_cuda"] = True

    with pytest.raises(RuntimeError, match="AgentTransformerPPO.*require_cuda=true"):
        AgentTransformerPPO(config)


def test_transformer_ppo_schema_rejects_nonzero_dropout() -> None:
    from utils.config_schema import TransformerPPOTransformerConfig

    with pytest.raises(ValueError, match="AgentTransformerPPO requires transformer.dropout=0.0 because PPO old/new probability ratios must use the same representation."):
        TransformerPPOTransformerConfig(
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.1,
        )


def test_transformer_ppo_schema_defaults_dropout_to_zero() -> None:
    from utils.config_schema import TransformerPPOTransformerConfig

    config = TransformerPPOTransformerConfig(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
    )

    assert config.dropout == 0.0


def test_episode_boundary_discards_undersized_truncated_rollout() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    state = agent._per_building[0]

    observations = [np.zeros(obs_dim, dtype=np.float64)]
    agent.update(
        observations=observations,
        actions=[np.asarray(agent.predict(observations)[0])],
        rewards=[0.1],
        next_observations=[np.zeros(obs_dim, dtype=np.float64)],
        terminated=False,
        truncated=True,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )

    assert len(state.buffer) == 1
    agent.on_episode_end(episode=0, training=True)
    assert len(state.buffer) == 0


def test_update_step_retains_undersized_nonterminal_rollout() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    state = agent._per_building[0]
    observations = [np.zeros(obs_dim, dtype=np.float64)]
    actions = agent.predict(observations, deterministic=True)

    agent.update(
        observations=observations,
        actions=[np.asarray(actions[0])],
        rewards=[0.1],
        next_observations=observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=True,
        initial_exploration_done=True,
    )

    assert len(state.buffer) == 1


def test_terminal_episode_flushes_partial_rollout_with_two_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    state = agent._per_building[0]
    observations = [np.zeros(obs_dim, dtype=np.float64)]
    optimizer_steps = 0
    original_step = state.optimizer.step

    def count_optimizer_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(state.optimizer, "step", count_optimizer_step)

    for step in range(2):
        actions = agent.predict(observations, deterministic=True)
        agent.update(
            observations=observations,
            actions=[np.asarray(actions[0])],
            rewards=[0.1],
            next_observations=observations,
            terminated=step == 1,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )

    agent.on_episode_end(episode=0, training=True)
    assert len(state.buffer) == 0
    assert optimizer_steps == 1


def test_topology_flush_preserves_unaffected_building_weights_and_logs_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=2)
    observations = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]
    for step in range(2):
        actions = agent.predict(observations, deterministic=True)
        agent.update(
            observations=observations,
            actions=[np.asarray(action) for action in actions],
            rewards=[0.1, 0.2],
            next_observations=observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )
    unaffected_weights = {
        name: value.detach().clone()
        for name, value in agent._per_building[1].actor.state_dict().items()
    }
    logged_messages: list[str] = []
    monkeypatch.setattr(
        "algorithms.agents.agent_transformer_ppo.logger.info",
        lambda message, *args: logged_messages.append(message.format(*args)),
    )
    charger_id = next(
        name.split("::")[1]
        for name in obs_per[0]
        if name.startswith("charger::")
        and "::connected_ev::" not in name
        and "::incoming_ev::" not in name
    )
    replacement_obs = list(obs_per[0]) + [
        name.replace(
            f"charger::{charger_id}::", "charger::Building_1/charger_new::", 1
        )
        for name in obs_per[0]
        if name.startswith(f"charger::{charger_id}::")
    ]

    agent.attach_environment(
        observation_names=[replacement_obs, obs_per[1]],
        action_names=[act_per[0] + ["electric_vehicle_storage"], act_per[1]],
        action_space=[None, None],
        observation_space=[None, None],
        metadata={"building_names": ["Building_1", "Building_2"]},
    )

    assert len(agent._per_building[0].buffer) == 0
    assert len(agent._per_building[1].buffer) == 2
    for name, value in agent._per_building[1].actor.state_dict().items():
        assert torch.equal(value, unaffected_weights[name])
    assert any("rollout_boundary=topology_change" in message for message in logged_messages)


def test_update_appends_to_buffer_then_ppo_step_clears() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    state = agent._per_building[0]
    rng = np.random.default_rng(0)
    n_ca = state.layout.n_ca

    # Five non-update steps → buffer fills
    for _ in range(5):
        obs = [rng.standard_normal(obs_dim)]
        next_obs = [rng.standard_normal(obs_dim)]
        actions_arr = np.asarray(agent.predict(obs)[0])
        agent.update(
            observations=obs,
            actions=[actions_arr],
            rewards=[0.1],
            next_observations=next_obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=0,
            update_step=False,
            initial_exploration_done=True,
        )
    assert len(state.buffer) == 5

    # Snapshot a parameter to confirm gradient step actually moved weights.
    p_before = next(state.actor.parameters()).clone().detach()

    obs = [rng.standard_normal(obs_dim)]
    next_obs = [rng.standard_normal(obs_dim)]
    actions_arr = np.asarray(agent.predict(obs)[0])
    agent.update(
        observations=obs,
        actions=[actions_arr],
        rewards=[0.1],
        next_observations=next_obs,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=True,
        initial_exploration_done=True,
    )
    assert len(state.buffer) == 0  # cleared after PPO step
    p_after = next(state.actor.parameters()).clone().detach()
    assert not torch.allclose(p_before, p_after), "PPO step should update actor weights"


def test_off_cadence_truncation_flushes_before_next_episode_update() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    state = agent._per_building[0]
    rng = np.random.default_rng(1)

    for step in range(4):
        observations = [rng.standard_normal(obs_dim)]
        agent.update(
            observations=observations,
            actions=[np.asarray(agent.predict(observations)[0])],
            rewards=[0.1],
            next_observations=[rng.standard_normal(obs_dim)],
            terminated=False,
            truncated=step == 3,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )

    agent.on_episode_end(episode=0, training=True)
    assert len(state.buffer) == 0

    observations = [rng.standard_normal(obs_dim)]
    agent.update(
        observations=observations,
        actions=[np.asarray(agent.predict(observations)[0])],
        rewards=[0.1],
        next_observations=[rng.standard_normal(obs_dim)],
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=4,
        update_step=True,
        initial_exploration_done=True,
    )

    assert len(state.buffer) == 1


def test_transformer_ppo_schema_requires_valid_update_cadence() -> None:
    from utils.config_schema import ProjectConfig

    def config_with_interval(interval: int) -> dict:
        return {
            "metadata": {"experiment_name": "test", "run_name": "test"},
            "simulator": {
                "dataset_name": "test",
                "dataset_path": "test",
                "reward_function": "test",
            },
            "training": {"steps_between_training_updates": interval},
            "pipeline": [
                {
                    "algorithm": "AgentTransformerPPO",
                    "tokenizer_config_path": _TOKENIZER_CFG,
                    "transformer": {
                        "d_model": 16,
                        "nhead": 2,
                        "num_layers": 1,
                        "dim_feedforward": 32,
                    },
                    "hyperparameters": {
                        "learning_rate": 1.0e-3,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_eps": 0.2,
                        "ppo_epochs": 1,
                        "minibatch_size": 4,
                        "entropy_coeff": 0.0,
                        "value_coeff": 0.5,
                        "max_grad_norm": 0.5,
                    },
                }
            ],
        }

    ProjectConfig.model_validate(config_with_interval(4))

    with pytest.raises(
        ValueError,
        match=(
            r"AgentTransformerPPO requires training.steps_between_training_updates "
            r">= pipeline\[\]\.hyperparameters\.minibatch_size\."
        ),
    ):
        ProjectConfig.model_validate(config_with_interval(2))


def test_successful_ppo_update_publishes_training_diagnostics() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    state = agent._per_building[0]
    rng = np.random.default_rng(2)

    for step in range(4):
        observations = [rng.standard_normal(obs_dim)]
        agent.update(
            observations=observations,
            actions=[np.asarray(agent.predict(observations)[0])],
            rewards=[0.1],
            next_observations=[rng.standard_normal(obs_dim)],
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=step == 3,
            initial_exploration_done=True,
        )

    metrics = agent.consume_latest_training_metrics()
    assert {"approx_kl", "ratio_error_max", "explained_variance"} <= metrics.keys()


def test_ppo_update_normalizes_returns_but_keeps_gae_values_in_raw_scale() -> None:
    """Critic normalizer updates from returns without changing stored raw values."""
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    state = agent._per_building[0]
    state.value_normalizer.update(torch.tensor([100.0, 104.0]))
    critic_output = state.critic.mlp[-1]
    assert isinstance(critic_output, torch.nn.Linear)
    with torch.no_grad():
        critic_output.weight.zero_()
        critic_output.bias.fill_(3.0)

    observed: dict[str, torch.Tensor] = {}
    original_compute = state.buffer.compute_returns_and_advantages

    def capture_gae_inputs(last_value: torch.Tensor) -> None:
        observed["values"] = torch.stack(state.buffer.values).clone()
        observed["last_value"] = last_value.clone()
        original_compute(last_value)

    state.buffer.compute_returns_and_advantages = capture_gae_inputs  # type: ignore[method-assign]
    rng = np.random.default_rng(8)
    for step in range(4):
        observations = [rng.standard_normal(obs_dim)]
        agent.update(
            observations=observations,
            actions=[np.asarray(agent.predict(observations)[0])],
            rewards=[0.1],
            next_observations=[rng.standard_normal(obs_dim)],
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=step == 3,
            initial_exploration_done=True,
        )

    assert torch.allclose(observed["values"], torch.full((4, 1), 108.0))
    assert torch.allclose(observed["last_value"], torch.tensor([108.0]))
    assert state.value_normalizer.count == 6


# ---------------------------------------------------------------------------
# Topology change handling
# ---------------------------------------------------------------------------


def test_topology_change_no_op_when_names_unchanged() -> None:
    agent, obs_per, act_per, _ = _make_agent(n_buildings=1)
    state_before = agent._per_building[0]
    actor_id_before = id(state_before.actor)
    agent.attach_environment(
        observation_names=copy.deepcopy(obs_per),
        action_names=copy.deepcopy(act_per),
        action_space=[None],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    # Same instance — no rebuild.
    assert id(agent._per_building[0].actor) == actor_id_before


def test_topology_change_rebuilds_layout_and_preserves_weights() -> None:
    """Add a second charger (and its EV blocks). The 'charger' projection
    weights survive (per-type weight sharing) and the layout grows by one
    CA segment."""
    agent, obs_per, act_per, _ = _make_agent(n_buildings=1)
    state = agent._per_building[0]
    n_ca_before = state.layout.n_ca
    # Snapshot the storage projection weights — should survive the rebuild.
    storage_w_before = state.tokenizer.projections["storage"].weight.detach().clone()
    charger_w_before = state.tokenizer.projections["charger"].weight.detach().clone()
    actor_w_before = next(state.actor.parameters()).detach().clone()

    # Identify existing charger block and replicate it under a fresh id.
    orig_id = next(
        n.split("::")[1]
        for n in obs_per[0]
        if n.startswith("charger::") and "::connected_ev::" not in n and "::incoming_ev::" not in n
    )
    new_id = "Building_1/charger_NEW"
    new_obs: List[str] = []
    new_block: List[str] = []
    for n in obs_per[0]:
        new_obs.append(n)
        if n.startswith(f"charger::{orig_id}::"):
            # Append a parallel entry for the new charger right after each
            # existing one (order doesn't matter to the layout builder, but
            # this keeps the diff readable).
            new_block.append(n.replace(f"charger::{orig_id}::", f"charger::{new_id}::", 1))
    new_obs.extend(new_block)
    new_acts = list(act_per[0]) + ["electric_vehicle_storage"]

    agent.attach_environment(
        observation_names=[new_obs],
        action_names=[new_acts],
        action_space=[None],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )

    state_after = agent._per_building[0]
    assert state_after.layout.n_ca == n_ca_before + 1
    # Per-type weights preserved
    assert torch.allclose(
        storage_w_before,
        state_after.tokenizer.projections["storage"].weight,
    )
    assert torch.allclose(
        charger_w_before,
        state_after.tokenizer.projections["charger"].weight,
    )
    # Actor preserved (per-CA weight sharing)
    actor_w_after = next(state_after.actor.parameters())
    assert torch.allclose(actor_w_before, actor_w_after)


def test_topology_change_feature_count_drift_hard_fails() -> None:
    """Inject an extra storage feature that wasn't present at attach time —
    feature count for type 'storage' changes. Must raise."""
    agent, obs_per, act_per, _ = _make_agent(n_buildings=1)
    storage_id = next(
        n.split("::")[1] for n in obs_per[0] if n.startswith("storage::")
    )
    drifted = list(obs_per[0])
    drifted.append(f"storage::{storage_id}::brand_new_storage_feature")

    with pytest.raises(ValueError, match=r"feature count for type 'storage'"):
        agent.attach_environment(
            observation_names=[drifted],
            action_names=copy.deepcopy(act_per),
            action_space=[None],
            observation_space=[None],
            metadata={"building_names": ["Building_1"]},
        )


def test_failed_topology_mutation_preserves_state_for_valid_retry() -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=1)
    state = agent._per_building[0]
    agent.predict([np.zeros(obs_dim, dtype=np.float64)])
    pending_before = agent._pending_decisions[0]
    assert pending_before is not None
    names_before = (state.obs_names_tuple, state.action_names_tuple)
    layout_before = state.layout
    tokenizer_before = state.tokenizer
    actor_before = state.actor
    topology_version_before = state.topology_version

    storage_id = next(
        name.split("::")[1] for name in obs_per[0] if name.startswith("storage::")
    )
    invalid_obs = list(obs_per[0]) + [
        f"storage::{storage_id}::brand_new_storage_feature"
    ]

    with pytest.raises(ValueError, match=r"feature count for type 'storage'"):
        agent.attach_environment(
            observation_names=[invalid_obs],
            action_names=copy.deepcopy(act_per),
            action_space=[None],
            observation_space=[None],
            metadata={"building_names": ["Building_1"]},
        )

    assert state.obs_names_tuple == names_before[0]
    assert state.action_names_tuple == names_before[1]
    assert state.layout is layout_before
    assert state.tokenizer is tokenizer_before
    assert state.actor is actor_before
    assert state.topology_version == topology_version_before
    assert agent._pending_decisions[0] is pending_before

    charger_id = next(
        name.split("::")[1]
        for name in obs_per[0]
        if name.startswith("charger::") and "::connected_ev::" not in name
        and "::incoming_ev::" not in name
    )
    valid_obs = list(obs_per[0]) + [
        name.replace(
            f"charger::{charger_id}::", "charger::Building_1/charger_NEW::", 1
        )
        for name in obs_per[0]
        if name.startswith(f"charger::{charger_id}::")
    ]
    agent.attach_environment(
        observation_names=[valid_obs],
        action_names=[list(act_per[0]) + ["electric_vehicle_storage"]],
        action_space=[None],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )

    assert state.obs_names_tuple == tuple(valid_obs)
    assert state.layout is not layout_before
    assert state.topology_version == topology_version_before + 1
    assert agent._pending_decisions[0] is None


def test_invalid_replacement_action_bounds_preserve_state_for_valid_retry() -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=1)
    initial_bounds = spaces.Box(
        low=np.array([-0.75, -0.5], dtype=np.float32),
        high=np.array([0.5, 0.75], dtype=np.float32),
        dtype=np.float32,
    )
    agent.attach_environment(
        observation_names=copy.deepcopy(obs_per),
        action_names=copy.deepcopy(act_per),
        action_space=[initial_bounds],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    state = agent._per_building[0]
    agent.predict([np.zeros(obs_dim, dtype=np.float64)])
    snapshot = {
        "bounds": agent._action_bounds,
        "bounds_values": [
            (low.clone(), high.clone()) for low, high in agent._action_bounds
        ],
        "names": (state.obs_names_tuple, state.action_names_tuple),
        "layout": state.layout,
        "topology_version": state.topology_version,
        "pending": agent._pending_decisions[0],
    }

    charger_id = next(
        name.split("::")[1]
        for name in obs_per[0]
        if name.startswith("charger::")
        and "::connected_ev::" not in name
        and "::incoming_ev::" not in name
    )
    replacement_obs = list(obs_per[0]) + [
        name.replace(
            f"charger::{charger_id}::", "charger::Building_1/charger_NEW::", 1
        )
        for name in obs_per[0]
        if name.startswith(f"charger::{charger_id}::")
    ]
    replacement_actions = list(act_per[0]) + ["electric_vehicle_storage"]
    invalid_bounds = spaces.Box(
        low=np.array([-0.75, -0.5, 2.0], dtype=np.float32),
        high=np.array([0.5, 0.75, 3.0], dtype=np.float32),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match=r"ActorHead supported action domain"):
        agent.attach_environment(
            observation_names=[replacement_obs],
            action_names=[replacement_actions],
            action_space=[invalid_bounds],
            observation_space=[None],
            metadata={"building_names": ["Building_1"]},
        )

    assert agent._action_bounds is snapshot["bounds"]
    assert all(
        torch.equal(actual, expected)
        for pair, expected_pair in zip(agent._action_bounds, snapshot["bounds_values"])
        for actual, expected in zip(pair, expected_pair)
    )
    assert (state.obs_names_tuple, state.action_names_tuple) == snapshot["names"]
    assert state.layout is snapshot["layout"]
    assert state.topology_version == snapshot["topology_version"]
    assert agent._pending_decisions[0] is snapshot["pending"]

    valid_bounds = spaces.Box(
        low=np.array([-0.75, -0.5, 0.0], dtype=np.float32),
        high=np.array([0.5, 0.75, 1.0], dtype=np.float32),
        dtype=np.float32,
    )
    agent.attach_environment(
        observation_names=[replacement_obs],
        action_names=[replacement_actions],
        action_space=[valid_bounds],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )

    assert state.obs_names_tuple == tuple(replacement_obs)
    assert state.action_names_tuple == tuple(replacement_actions)
    assert state.layout is not snapshot["layout"]
    assert state.topology_version == snapshot["topology_version"] + 1
    assert agent._pending_decisions[0] is None
    assert [bound.flatten().tolist() for bound in agent._action_bounds[0]] == [
        [-0.75, -0.5, 0.0],
        [0.5, 0.75, 1.0],
    ]


@pytest.mark.parametrize("bound_name", ["low", "high"])
def test_partial_action_bounds_reject_transactionally_then_valid_retry(
    bound_name: str,
) -> None:
    class _PartialBounds:
        def __init__(self, bound_name: str) -> None:
            setattr(self, bound_name, np.array([-0.75, -0.5], dtype=np.float32))

    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=1)
    initial_bounds = spaces.Box(
        low=np.array([-0.75, -0.5], dtype=np.float32),
        high=np.array([0.5, 0.75], dtype=np.float32),
        dtype=np.float32,
    )
    agent.attach_environment(
        observation_names=copy.deepcopy(obs_per),
        action_names=copy.deepcopy(act_per),
        action_space=[initial_bounds],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    state = agent._per_building[0]
    agent.predict([np.zeros(obs_dim, dtype=np.float64)])
    snapshot = {
        "bounds": agent._action_bounds,
        "bounds_values": [
            (low.clone(), high.clone()) for low, high in agent._action_bounds
        ],
        "names": (state.obs_names_tuple, state.action_names_tuple),
        "layout": state.layout,
        "topology_version": state.topology_version,
        "pending": agent._pending_decisions[0],
    }

    with pytest.raises(ValueError, match=r"must expose both low and high"):
        agent.attach_environment(
            observation_names=copy.deepcopy(obs_per),
            action_names=copy.deepcopy(act_per),
            action_space=[_PartialBounds(bound_name)],
            observation_space=[None],
            metadata={"building_names": ["Building_1"]},
        )

    assert agent._action_bounds is snapshot["bounds"]
    assert all(
        torch.equal(actual, expected)
        for pair, expected_pair in zip(agent._action_bounds, snapshot["bounds_values"])
        for actual, expected in zip(pair, expected_pair)
    )
    assert (state.obs_names_tuple, state.action_names_tuple) == snapshot["names"]
    assert state.layout is snapshot["layout"]
    assert state.topology_version == snapshot["topology_version"]
    assert agent._pending_decisions[0] is snapshot["pending"]

    agent.attach_environment(
        observation_names=copy.deepcopy(obs_per),
        action_names=copy.deepcopy(act_per),
        action_space=[initial_bounds],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )

    assert agent._action_bounds is not snapshot["bounds"]
    assert [bound.flatten().tolist() for bound in agent._action_bounds[0]] == [
        [-0.75, -0.5],
        [0.5, 0.75],
    ]


@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("observation_names", "observation_names and action_names must have equal counts"),
        ("action_names", "observation_names and action_names must have equal counts"),
        ("action_space", "action_space has 1 per-building entries; expected 2"),
        ("observation_space", "observation_space has 1 per-building entries; expected 2"),
    ],
)
def test_mismatched_environment_metadata_rejects_transactionally_then_valid_retry(
    field: str,
    expected_message: str,
) -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=2)
    agent.predict([np.zeros(obs_dim, dtype=np.float64) for _ in range(2)])
    snapshot = {
        "states": agent._per_building,
        "bounds": agent._action_bounds,
        "pending": agent._pending_decisions,
    }
    malformed = {
        "observation_names": copy.deepcopy(obs_per),
        "action_names": copy.deepcopy(act_per),
        "action_space": [None, None],
        "observation_space": [None, None],
    }
    malformed[field] = malformed[field][:-1]

    with pytest.raises(ValueError, match=expected_message):
        agent.attach_environment(
            **malformed,
            metadata={"building_names": ["Building_1", "Building_2"]},
        )

    assert agent._per_building is snapshot["states"]
    assert agent._action_bounds is snapshot["bounds"]
    assert agent._pending_decisions is snapshot["pending"]
    assert all(
        pending is not None for pending in agent._pending_decisions
    )

    agent.attach_environment(
        observation_names=copy.deepcopy(obs_per),
        action_names=copy.deepcopy(act_per),
        action_space=[None, None],
        observation_space=[None, None],
        metadata={"building_names": ["Building_1", "Building_2"]},
    )

    assert agent._per_building is snapshot["states"]
    assert agent._action_bounds is not snapshot["bounds"]


@pytest.mark.parametrize(
    ("low", "high", "message"),
    [
        ([-1.0, np.nan], [1.0, 1.0], "must be finite"),
        ([-1.0, -1.0], [np.inf, 1.0], "must be finite"),
        ([0.5, -1.0], [0.25, 1.0], "low < high"),
        ([-1.0, 0.0], [1.0, 0.0], "low < high"),
    ],
)
def test_invalid_initial_action_bounds_preserve_state_for_valid_retry(
    low: List[float],
    high: List[float],
    message: str,
) -> None:
    class _Bounds:
        def __init__(self, low: List[float], high: List[float]) -> None:
            self.low = np.asarray(low, dtype=np.float64)
            self.high = np.asarray(high, dtype=np.float64)

    agent = AgentTransformerPPO(_base_config())
    observation_names = [load_sample_observation_names_for_first_building()]
    action_names = [list(_DEFAULT_ACTIONS)]

    with pytest.raises(ValueError, match=message):
        agent.attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=[_Bounds(low, high)],
            observation_space=[None],
            metadata={"building_names": ["Building_1"]},
        )

    assert agent._first_attach_done is False
    assert agent._per_building == []
    assert agent._pending_decisions == []
    assert agent._action_bounds == []

    agent.attach_environment(
        observation_names=observation_names,
        action_names=action_names,
        action_space=[_Bounds([-0.75, -0.5], [0.5, 0.75])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )

    assert agent._first_attach_done is True
    assert len(agent._per_building) == 1


def test_multi_building_topology_failure_is_atomic_until_valid_retry() -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=2)
    observations = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]

    # Keep both an in-flight decision and an old-layout rollout for each
    # building. A rejected batch must leave both kinds of state intact.
    for step in range(2):
        first_actions = agent.predict(observations, deterministic=False)
        agent.update(
            observations=observations,
            actions=[np.asarray(actions) for actions in first_actions],
            rewards=[0.1, 0.1],
            next_observations=observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )
    agent.predict(observations, deterministic=False)

    snapshots = [
        {
            "state": state,
            "names": (state.obs_names_tuple, state.action_names_tuple),
            "layout": state.layout,
            "tokenizer": state.tokenizer,
            "backbone": state.backbone,
            "actor": state.actor,
            "critic": state.critic,
            "buffer": state.buffer,
            "buffer_observation": state.buffer.observations[0].clone(),
            "topology_version": state.topology_version,
            "pending": agent._pending_decisions[index],
        }
        for index, state in enumerate(agent._per_building)
    ]

    def add_charger(building_index: int) -> tuple[List[str], List[str]]:
        charger_id = next(
            name.split("::")[1]
            for name in obs_per[building_index]
            if name.startswith("charger::")
            and "::connected_ev::" not in name
            and "::incoming_ev::" not in name
        )
        return (
            list(obs_per[building_index])
            + [
                name.replace(
                    f"charger::{charger_id}::",
                    f"charger::Building_{building_index + 1}/charger_NEW::",
                    1,
                )
                for name in obs_per[building_index]
                if name.startswith(f"charger::{charger_id}::")
            ],
            list(act_per[building_index]) + ["electric_vehicle_storage"],
        )

    valid_obs_0, valid_actions_0 = add_charger(0)
    valid_obs_1, valid_actions_1 = add_charger(1)
    storage_id = next(
        name.split("::")[1] for name in obs_per[1] if name.startswith("storage::")
    )
    invalid_obs_1 = valid_obs_1 + [
        f"storage::{storage_id}::brand_new_storage_feature"
    ]

    with pytest.raises(ValueError, match=r"feature count for type 'storage'"):
        agent.attach_environment(
            observation_names=[valid_obs_0, invalid_obs_1],
            action_names=[valid_actions_0, valid_actions_1],
            action_space=[None, None],
            observation_space=[None, None],
            metadata={"building_names": ["Building_1", "Building_2"]},
        )

    for index, snapshot in enumerate(snapshots):
        state = agent._per_building[index]
        assert state is snapshot["state"]
        assert (state.obs_names_tuple, state.action_names_tuple) == snapshot["names"]
        assert state.layout is snapshot["layout"]
        assert state.tokenizer is snapshot["tokenizer"]
        assert state.backbone is snapshot["backbone"]
        assert state.actor is snapshot["actor"]
        assert state.critic is snapshot["critic"]
        assert state.buffer is snapshot["buffer"]
        assert torch.equal(state.buffer.observations[0], snapshot["buffer_observation"])
        assert state.topology_version == snapshot["topology_version"]
        assert agent._pending_decisions[index] is snapshot["pending"]

    agent.attach_environment(
        observation_names=[valid_obs_0, valid_obs_1],
        action_names=[valid_actions_0, valid_actions_1],
        action_space=[None, None],
        observation_space=[None, None],
        metadata={"building_names": ["Building_1", "Building_2"]},
    )

    for index, snapshot in enumerate(snapshots):
        state = agent._per_building[index]
        assert state.obs_names_tuple == tuple((valid_obs_0, valid_obs_1)[index])
        assert state.action_names_tuple == tuple(
            (valid_actions_0, valid_actions_1)[index]
        )
        assert state.layout is not snapshot["layout"]
        assert len(state.buffer) == 0
        assert state.topology_version == snapshot["topology_version"] + 1
        assert agent._pending_decisions[index] is None


def test_topology_commit_rolls_back_all_buildings_when_later_flush_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=2)
    observations = [np.zeros(obs_dim, dtype=np.float64) for _ in range(2)]
    for step in range(2):
        first_actions = agent.predict(observations, deterministic=False)
        agent.update(
            observations=observations,
            actions=[np.asarray(actions) for actions in first_actions],
            rewards=[0.1, 0.1],
            next_observations=observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=False,
            initial_exploration_done=True,
        )
    agent.predict(observations, deterministic=False)

    snapshots = [
        {
            "state": state,
            "model": {
                "tokenizer": copy.deepcopy(state.tokenizer.state_dict()),
                "backbone": copy.deepcopy(state.backbone.state_dict()),
                "actor": copy.deepcopy(state.actor.state_dict()),
                "critic": copy.deepcopy(state.critic.state_dict()),
                "optimizer": copy.deepcopy(state.optimizer.state_dict()),
                "normalizer": copy.deepcopy(state.value_normalizer.state_dict()),
            },
            "buffer": copy.deepcopy(state.buffer),
            "layout": state.layout,
            "names": (state.obs_names_tuple, state.action_names_tuple),
            "topology_version": state.topology_version,
            "pending": agent._pending_decisions[index],
        }
        for index, state in enumerate(agent._per_building)
    ]
    metrics_before = copy.deepcopy(agent._latest_training_metrics)

    def add_charger(building_index: int) -> tuple[List[str], List[str]]:
        charger_id = next(
            name.split("::")[1]
            for name in obs_per[building_index]
            if name.startswith("charger::")
            and "::connected_ev::" not in name
            and "::incoming_ev::" not in name
        )
        return (
            list(obs_per[building_index])
            + [
                name.replace(
                    f"charger::{charger_id}::",
                    f"charger::Building_{building_index + 1}/charger_NEW::",
                    1,
                )
                for name in obs_per[building_index]
                if name.startswith(f"charger::{charger_id}::")
            ],
            list(act_per[building_index]) + ["electric_vehicle_storage"],
        )

    new_obs_0, new_actions_0 = add_charger(0)
    new_obs_1, new_actions_1 = add_charger(1)
    original_update = agent._run_ppo_update_with_last_value

    def fail_later_flush(state, last_value, *, building_idx):
        if building_idx == 1:
            raise RuntimeError("later building flush failed")
        return original_update(state, last_value, building_idx=building_idx)

    monkeypatch.setattr(agent, "_run_ppo_update_with_last_value", fail_later_flush)
    with pytest.raises(RuntimeError, match="later building flush failed"):
        agent.attach_environment(
            observation_names=[new_obs_0, new_obs_1],
            action_names=[new_actions_0, new_actions_1],
            action_space=[None, None],
            observation_space=[None, None],
            metadata={"building_names": ["Building_1", "Building_2"]},
        )

    assert agent._latest_training_metrics == metrics_before
    for index, snapshot in enumerate(snapshots):
        state = agent._per_building[index]
        assert state is snapshot["state"]
        assert state.layout is snapshot["layout"]
        assert (state.obs_names_tuple, state.action_names_tuple) == snapshot["names"]
        assert state.topology_version == snapshot["topology_version"]
        assert state.value_normalizer.state_dict() == snapshot["model"]["normalizer"]
        assert state.optimizer.state_dict() == snapshot["model"]["optimizer"]
        assert state.buffer.rewards == snapshot["buffer"].rewards
        assert state.buffer.terminated == snapshot["buffer"].terminated
        assert state.buffer.truncated == snapshot["buffer"].truncated
        for field in ("observations", "actions", "log_probs", "values"):
            actual = getattr(state.buffer, field)
            expected = getattr(snapshot["buffer"], field)
            assert len(actual) == len(expected)
            assert all(torch.equal(value, expected[idx]) for idx, value in enumerate(actual))
        assert agent._pending_decisions[index] is snapshot["pending"]
        for module_name in ("tokenizer", "backbone", "actor", "critic"):
            actual = getattr(state, module_name).state_dict()
            expected = snapshot["model"][module_name]
            assert actual.keys() == expected.keys()
            assert all(torch.equal(value, expected[key]) for key, value in actual.items())

    monkeypatch.setattr(agent, "_run_ppo_update_with_last_value", original_update)
    agent.attach_environment(
        observation_names=[new_obs_0, new_obs_1],
        action_names=[new_actions_0, new_actions_1],
        action_space=[None, None],
        observation_space=[None, None],
        metadata={"building_names": ["Building_1", "Building_2"]},
    )

    for index, snapshot in enumerate(snapshots):
        state = agent._per_building[index]
        assert state.obs_names_tuple == tuple((new_obs_0, new_obs_1)[index])
        assert state.action_names_tuple == tuple((new_actions_0, new_actions_1)[index])
        assert len(state.buffer) == 0
        assert state.topology_version == snapshot["topology_version"] + 1
        assert agent._pending_decisions[index] is None


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    agent, _, _, _ = _make_agent(n_buildings=1)
    # Mutate weights so round-trip is meaningful.
    with torch.no_grad():
        for p in agent._per_building[0].actor.parameters():
            p.add_(0.1)
    actor_w = (
        next(agent._per_building[0].actor.parameters()).detach().clone()
    )
    path = agent.save_checkpoint(str(tmp_path), step=42)
    assert path is not None and Path(path).exists()

    # Build a fresh agent with identical layout and load.
    obs_names = load_sample_observation_names_for_first_building()
    fresh = AgentTransformerPPO(_base_config())
    fresh.attach_environment(
        observation_names=[list(obs_names)],
        action_names=[list(_DEFAULT_ACTIONS)],
        action_space=[None],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    fresh.load_checkpoint(path)
    actor_w_loaded = next(fresh._per_building[0].actor.parameters()).detach()
    assert torch.allclose(actor_w, actor_w_loaded)


def test_checkpoint_layout_signature_mismatch_rejected(tmp_path: Path) -> None:
    """Save a 1-building checkpoint, then try to load into a 2-building agent.
    Cardinality mismatch is rejected before signature check, exercising the
    same guarantee (cross-topology resume disallowed)."""
    agent, _, _, _ = _make_agent(n_buildings=1)
    path = agent.save_checkpoint(str(tmp_path), step=1)
    assert path is not None

    fresh, _, _, _ = _make_agent(n_buildings=2)
    with pytest.raises(ValueError, match=r"Cross-cardinality resume|layout_signature"):
        fresh.load_checkpoint(path)


def test_checkpoint_signature_mismatch_same_cardinality(tmp_path: Path) -> None:
    """Save with the bundled obs_names; reload into an agent whose obs_names
    have an extra (allowed) charger appended → signature differs but
    cardinality matches → ``layout_signature mismatch`` raised."""
    agent, obs_per, act_per, _ = _make_agent(n_buildings=1)
    path = agent.save_checkpoint(str(tmp_path), step=1)
    assert path is not None

    # Build a fresh agent with one extra charger (same trick as the
    # rebuild test) — different obs_names_tuple.
    orig_id = next(
        n.split("::")[1]
        for n in obs_per[0]
        if n.startswith("charger::") and "::connected_ev::" not in n and "::incoming_ev::" not in n
    )
    new_id = "Building_1/charger_NEW"
    extended = list(obs_per[0]) + [
        n.replace(f"charger::{orig_id}::", f"charger::{new_id}::", 1)
        for n in obs_per[0]
        if n.startswith(f"charger::{orig_id}::")
    ]

    fresh = AgentTransformerPPO(_base_config())
    fresh.attach_environment(
        observation_names=[extended],
        action_names=[list(act_per[0]) + ["electric_vehicle_storage"]],
        action_space=[None],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    with pytest.raises(ValueError, match=r"layout_signature mismatch"):
        fresh.load_checkpoint(path)


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------


def test_export_artifacts_writes_files_and_returns_manifest(tmp_path: Path) -> None:
    agent, _, _, _ = _make_agent(n_buildings=2)
    manifest = agent.export_artifacts(
        str(tmp_path), context={"topology_version": 7}
    )
    assert manifest["format"] == "onnx"
    assert manifest["supports_dynamic_topology"] is True
    assert manifest["tokenizer_config_path"] == _TOKENIZER_CFG
    assert len(manifest["artifacts"]) == 2
    assert len(manifest["agent_models"]) == 2
    for entry in manifest["artifacts"]:
        p = tmp_path / entry["path"]
        assert p.exists() and p.stat().st_size > 0
        assert entry["path"].endswith(".onnx")
        assert "topology_v7" in entry["path"]
        assert entry["config"]["n_ca"] == 2
        assert set(entry["config"]["ca_types"]) <= {"storage", "charger"}


def test_onnx_wrapper_affinely_maps_actions_to_positive_bounds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent, obs_per, act_per, obs_dim = _make_agent(n_buildings=1)
    bounds = spaces.Box(
        low=np.array([0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[bounds],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    state = agent._per_building[0]
    actor_output = state.actor.mlp[-1]
    assert isinstance(actor_output, torch.nn.Linear)
    with torch.no_grad():
        actor_output.weight.zero_()
        actor_output.bias.fill_(0.5)
    exported: dict[str, torch.nn.Module] = {}

    def fake_export(wrapper: torch.nn.Module, *_args: object, **_kwargs: object) -> None:
        exported["wrapper"] = wrapper

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    agent._export_onnx(
        state,
        tmp_path / "agent.onnx",
        obs_dim,
        *agent._action_bounds[0],
    )

    output = exported["wrapper"](torch.zeros(3, obs_dim)).detach().cpu().numpy()
    expected = (np.tanh(0.5) + 1.0) / 2.0
    assert np.all(output >= 0.0)
    assert np.all(output <= 1.0)
    np.testing.assert_allclose(output, expected, rtol=1.0e-6, atol=1.0e-6)


def test_onnx_export_builds_wrapper_and_input_on_agent_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    agent, _, _, _ = _make_agent()
    state = agent._per_building[0]
    agent.device = torch.device("cuda")
    observed: dict[str, object] = {}
    torch_zeros = torch.zeros
    torch_tensor = torch.tensor

    def fake_to(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        observed["wrapper_device"] = device
        return module

    def fake_zeros(*shape: int, **kwargs: object) -> torch.Tensor:
        observed["input_device"] = kwargs["device"]
        return torch_zeros(*shape)

    def fake_tensor(data: object, **kwargs: object) -> torch.Tensor:
        assert kwargs["device"] == torch.device("cuda")
        return torch_tensor(data, dtype=kwargs.get("dtype"))

    def fake_export(
        wrapper: torch.nn.Module,
        inputs: tuple[torch.Tensor],
        path: str,
        **kwargs: object,
    ) -> None:
        observed["wrapper"] = wrapper
        observed["input"] = inputs[0]

    monkeypatch.setattr(torch.nn.Module, "to", fake_to)
    monkeypatch.setattr(torch, "zeros", fake_zeros)
    monkeypatch.setattr(torch, "tensor", fake_tensor)
    monkeypatch.setattr(torch.onnx, "export", fake_export)

    agent._export_onnx(
        state,
        tmp_path / "agent.onnx",
        3,
        *agent._action_bounds[0],
    )

    assert observed["wrapper_device"] == torch.device("cuda")
    assert observed["input_device"] == torch.device("cuda")


# ---------------------------------------------------------------------------
# Synthetic-sample reverse parser
# ---------------------------------------------------------------------------


def test_synthetic_sample_routes_features_by_table() -> None:
    obs = [
        "district__hour",
        "non_shiftable_load",
        "storage::s1::soc",
        "pv::p1::generation",
        "charger::c1::state",
        "charger::c1::connected_ev::soc",
        "charger::c1::incoming_ev::departure",
    ]
    sample = _synthetic_sample_from_obs_names(obs)
    feats = sample.feature_names_per_table
    assert "district__hour" in feats["district"]
    assert "non_shiftable_load" in feats["building"]
    assert "soc" in feats["storage"]
    assert "generation" in feats["pv"]
    assert "state" in feats["charger"]
    assert {"soc", "departure"} <= set(feats["ev"])
