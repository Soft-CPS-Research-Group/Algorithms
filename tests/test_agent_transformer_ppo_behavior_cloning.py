"""Separate demonstration and PPO phase tests for AgentTransformerPPO."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from loguru import logger

from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
from algorithms.utils.behavior_cloning import (
    BehaviorCloningRegularizer,
    Demonstration,
)
from tests.test_agent_transformer_ppo import _DEFAULT_ACTIONS, _base_config
from tests._entity_sample_obs_names import load_sample_observation_names_for_first_building


class _DummySpace:
    def __init__(self, size: int) -> None:
        self.low = np.full(size, -1.0, dtype=np.float64)
        self.high = np.full(size, 1.0, dtype=np.float64)


class _BoundedDummySpace:
    def __init__(self, low: list[float], high: list[float]) -> None:
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)


def _config(*, demonstrations: int = 1, weight: float = 0.0) -> dict:
    config = _base_config()
    config["algorithm"]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": demonstrations,
        "max_samples_per_building": 16,
        "pretraining_epochs": 2,
        "batch_size": 1,
        "weight": weight,
        "min_weight": 0.0,
        "decay_start_step": 0,
        "decay_steps": 1,
        "ev_multiplier": 1.0,
        "storage_multiplier": 1.0,
        "teacher": {"policy": "RBCSmartPolicy", "deterministic": True, "hyperparameters": {}},
    }
    return config


def _agent(
    *, demonstrations: int = 1, weight: float = 0.0, building_count: int = 1
) -> tuple[AgentTransformerPPO, int]:
    names = load_sample_observation_names_for_first_building()
    actions = list(_DEFAULT_ACTIONS)
    agent = AgentTransformerPPO(_config(demonstrations=demonstrations, weight=weight))
    agent.attach_environment(
        observation_names=[names] * building_count,
        action_names=[actions] * building_count,
        action_space=[_DummySpace(len(actions)) for _ in range(building_count)],
        observation_space=[None] * building_count,
        metadata={
            "building_names": [f"Building_{idx + 1}" for idx in range(building_count)],
            "seconds_per_time_step": 3600,
        },
    )
    dimension = BehaviorCloningRegularizer.full_representation_width(
        agent._per_building[0].layout
    )
    return agent, dimension


def _teacher(agent: AgentTransformerPPO, value: float) -> list[list[float]]:
    actions = [[value] * agent._per_building[0].layout.n_ca]
    assert agent._bc is not None
    agent._bc.compute_teacher_actions = lambda _observations: [list(row) for row in actions]
    return actions


def _update(agent: AgentTransformerPPO, observation: np.ndarray, actions, step: int) -> None:
    agent.update(
        observations=[observation], actions=actions, rewards=[0.1],
        next_observations=[observation], terminated=False, truncated=False,
        update_target_step=False, global_learning_step=step, update_step=False,
        initial_exploration_done=True,
    )


def _materialize_optimizer_state(*optimizers: torch.optim.Optimizer) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()


def _assert_structured_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_structured_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_structured_equal(actual_item, expected_item)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            _assert_structured_equal(
                getattr(actual, field.name), getattr(expected, field.name)
            )
    else:
        assert actual == expected


def _snapshot_restore_state(agent: AgentTransformerPPO) -> dict:
    state = agent._per_building[0]
    assert agent._bc is not None
    return {
        "model_states": {
            name: {
                key: value.detach().clone()
                for key, value in getattr(state, name).state_dict().items()
            }
            for name in ("tokenizer", "backbone", "actor", "critic")
        },
        "optimizer": deepcopy(state.optimizer.state_dict()),
        "bc_optimizer": deepcopy(state.bc_optimizer.state_dict()),
        "normalizer": deepcopy(state.value_normalizer.state_dict()),
        "action_bounds": [
            (low.detach().clone(), high.detach().clone())
            for low, high in agent._action_bounds
        ],
        "counters": (
            agent._latest_global_learning_step,
            agent._ppo_update_count,
            agent._current_episode,
        ),
        "topology": state.topology_version,
        "metrics": dict(agent._latest_training_metrics),
        "pending": agent._pending_decisions[0],
        "bc_state": agent._bc.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }


def _assert_restore_state_unchanged(agent: AgentTransformerPPO, snapshot: dict) -> None:
    state = agent._per_building[0]
    for name, before in snapshot["model_states"].items():
        actual = getattr(state, name).state_dict()
        assert all(torch.equal(value, before[key]) for key, value in actual.items())
    _assert_structured_equal(state.optimizer.state_dict(), snapshot["optimizer"])
    _assert_structured_equal(state.bc_optimizer.state_dict(), snapshot["bc_optimizer"])
    assert state.value_normalizer.state_dict() == snapshot["normalizer"]
    assert state.topology_version == snapshot["topology"]
    assert all(
        torch.equal(low_before, low_after)
        and torch.equal(high_before, high_after)
        for (low_before, high_before), (low_after, high_after) in zip(
            snapshot["action_bounds"], agent._action_bounds
        )
    )
    assert (
        agent._latest_global_learning_step,
        agent._ppo_update_count,
        agent._current_episode,
    ) == snapshot["counters"]
    assert agent._latest_training_metrics == snapshot["metrics"]
    assert agent._pending_decisions[0] is snapshot["pending"]
    assert agent._bc is not None
    _assert_structured_equal(agent._bc.state_dict(), snapshot["bc_state"])
    _assert_structured_equal(random.getstate(), snapshot["python_rng"])
    _assert_structured_equal(np.random.get_state(), snapshot["numpy_rng"])
    assert torch.equal(torch.get_rng_state(), snapshot["torch_rng"])


def _expand_charger_topology(
    agent: AgentTransformerPPO,
    observation_names: list[str],
) -> tuple[list[str], list[str]]:
    original_charger_id = next(
        name.split("::")[1]
        for name in observation_names
        if name.startswith("charger::")
        and "::connected_ev::" not in name
        and "::incoming_ev::" not in name
    )
    new_charger_id = "Building_1/charger_dynamic"
    expanded_names = list(observation_names)
    expanded_names.extend(
        name.replace(
            f"charger::{original_charger_id}::",
            f"charger::{new_charger_id}::",
            1,
        )
        for name in observation_names
        if name.startswith(f"charger::{original_charger_id}::")
    )
    expanded_actions = list(_DEFAULT_ACTIONS) + ["electric_vehicle_storage"]
    agent.attach_environment(
        observation_names=[expanded_names],
        action_names=[expanded_actions],
        action_space=[_DummySpace(len(expanded_actions))],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )
    return expanded_names, expanded_actions


def test_demo_episode_executes_teacher_only_records_immutable_demo_and_no_ppo() -> None:
    agent, dimension = _agent()
    teacher_actions = _teacher(agent, 0.25)
    observation = np.ones(dimension, dtype=np.float64)
    agent.on_episode_start(episode=0, training=True)
    agent.set_observation_context(raw_observations=[observation], encoded_observations=[observation])

    actions = agent.predict([observation], deterministic=False)
    _update(agent, observation, actions, 0)

    assert actions == teacher_actions
    assert len(agent._per_building[0].buffer) == 0
    assert agent._pending_decisions == [None]
    assert agent._bc is not None
    demo = next(iter(agent._bc.demonstrations_by_signature.values()))[0]
    observation[0] = 99.0
    assert demo.observation[0] == 1.0
    assert demo.layout is not agent._per_building[0].layout


def test_demo_teacher_actions_are_normalized_to_actor_tanh_space() -> None:
    names = load_sample_observation_names_for_first_building()
    actions = list(_DEFAULT_ACTIONS)
    low = np.linspace(-0.9, -0.3, len(actions), dtype=np.float64)
    high = np.linspace(0.2, 0.8, len(actions), dtype=np.float64)
    agent = AgentTransformerPPO(_config())
    agent.attach_environment(
        observation_names=[names],
        action_names=[actions],
        action_space=[_BoundedDummySpace(low.tolist(), high.tolist())],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )
    dimension = BehaviorCloningRegularizer.full_representation_width(
        agent._per_building[0].layout
    )
    teacher_action = low + 0.75 * (high - low)
    observation = np.ones(dimension, dtype=np.float64)

    agent.update(
        observations=[observation],
        actions=[teacher_action],
        rewards=[0.1],
        next_observations=[observation],
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )

    assert agent._bc is not None
    demonstration = next(
        iter(agent._bc.demonstrations_for_building_by_signature(0).values())
    )[0]
    expected_tanh_action = 2.0 * (teacher_action - low) / (high - low) - 1.0
    assert demonstration.target == pytest.approx(expected_tanh_action)
    loss = agent._bc.demonstration_loss(
        layout=agent._per_building[0].layout,
        demonstrations=[demonstration],
        predicted_means=torch.as_tensor(expected_tanh_action).view(1, -1, 1),
        global_learning_step=0,
    )
    assert loss.item() == pytest.approx(0.0)


def test_final_demo_end_pretrains_actor_then_ppo_uses_only_actor_actions() -> None:
    agent, dimension = _agent()
    teacher_actions = _teacher(agent, 0.75)
    observation = np.ones(dimension, dtype=np.float64)
    before = [parameter.detach().clone() for parameter in agent._per_building[0].actor.parameters()]
    agent.on_episode_start(episode=0, training=True)
    agent.set_observation_context(raw_observations=[observation], encoded_observations=[observation])
    _update(agent, observation, agent.predict([observation]), 0)
    agent.on_episode_end(episode=0, training=True)

    after = list(agent._per_building[0].actor.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after))
    metrics = agent.consume_latest_training_metrics()
    assert metrics["TPPO/behavior_cloning_pretraining_epochs"] == 2.0
    assert metrics["TPPO/behavior_cloning_demonstration_samples"] == 1.0
    assert metrics["TPPO/behavior_cloning_building_Building_1_usable_samples"] == 1.0
    assert metrics["TPPO/behavior_cloning_building_Building_1_trained_batches"] == 2.0
    assert metrics["TPPO/behavior_cloning_pretraining_batches"] == 2.0

    agent.on_episode_start(episode=1, training=True)
    agent.set_observation_context(raw_observations=[observation], encoded_observations=[observation])
    ppo_actions = agent.predict([observation], deterministic=True)
    assert ppo_actions != teacher_actions
    _update(agent, observation, ppo_actions, 1)
    assert len(agent._per_building[0].buffer) == 1


def test_final_demo_lifecycle_rejects_zero_usable_demonstrations() -> None:
    agent, _ = _agent()

    agent.on_episode_start(episode=0, training=True)

    with pytest.raises(RuntimeError, match=r"zero compatible demonstrations.*Building_1"):
        agent.on_episode_end(episode=0, training=True)


def test_pretraining_rejects_each_building_without_usable_demonstrations() -> None:
    agent, dimension = _agent(building_count=2)
    assert agent._bc is not None
    first_state = agent._per_building[0]
    before = [parameter.detach().clone() for parameter in first_state.actor.parameters()]
    agent._bc.record_demonstration(
        0,
        np.ones(dimension),
        first_state.layout,
        [0.25] * first_state.layout.n_ca,
    )

    with pytest.raises(RuntimeError, match=r"zero compatible demonstrations.*Building_2"):
        agent._run_bc_pretraining()

    assert all(
        torch.equal(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(before, first_state.actor.parameters())
    )


def test_pretraining_reports_positive_evidence_for_every_building() -> None:
    agent, dimension = _agent(building_count=2)
    assert agent._bc is not None
    for building_idx, state in enumerate(agent._per_building):
        agent._bc.record_demonstration(
            building_idx,
            np.ones(dimension),
            state.layout,
            [0.25] * state.layout.n_ca,
        )

    agent._run_bc_pretraining()

    metrics = agent.consume_latest_training_metrics()
    for building in ("Building_1", "Building_2"):
        assert metrics[f"TPPO/behavior_cloning_building_{building}_usable_samples"] == 1.0
        assert metrics[f"TPPO/behavior_cloning_building_{building}_trained_batches"] == 2.0
    assert metrics["TPPO/behavior_cloning_pretraining_batches"] == 4.0


def test_pretraining_logs_start_group_and_complete_events() -> None:
    agent, dimension = _agent(building_count=2)
    assert agent._bc is not None
    for building_idx, state in enumerate(agent._per_building):
        agent._bc.record_demonstration(
            building_idx,
            np.ones(dimension),
            state.layout,
            [0.25] * state.layout.n_ca,
        )

    messages = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="INFO",
    )
    try:
        agent._run_bc_pretraining()
    finally:
        logger.remove(sink_id)

    assert messages[0] == "event=bc_pretraining_start buildings=2"
    assert messages[1] == (
        "event=bc_pretraining_group building_id=Building_1 group_index=1 "
        "group_count=1 group_samples=1 usable_samples=1 trained_batches=2"
    )
    assert messages[2] == (
        "event=bc_pretraining_group building_id=Building_2 group_index=1 "
        "group_count=1 group_samples=1 usable_samples=1 trained_batches=2"
    )
    assert messages[3] == (
        "event=bc_pretraining_complete buildings=2 usable_samples=2 trained_batches=4"
    )


def test_pretraining_logs_missing_and_total_buildings_before_zero_demo_error() -> None:
    agent, dimension = _agent(building_count=2)
    assert agent._bc is not None
    first_state = agent._per_building[0]
    agent._bc.record_demonstration(
        0,
        np.ones(dimension),
        first_state.layout,
        [0.25] * first_state.layout.n_ca,
    )

    messages = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="INFO",
    )
    try:
        with pytest.raises(RuntimeError, match=r"zero compatible demonstrations.*Building_2"):
            agent._run_bc_pretraining()
    finally:
        logger.remove(sink_id)

    assert messages == [
        "event=bc_pretraining_start buildings=2",
        "event=bc_pretraining_failure reason=zero_usable_demonstrations "
        "missing_buildings=1 total_buildings=2",
    ]


def test_pretraining_logs_failure_before_unexpected_error() -> None:
    agent, dimension = _agent()
    assert agent._bc is not None
    state = agent._per_building[0]
    agent._bc.record_demonstration(
        0, np.ones(dimension), state.layout, [0.25] * state.layout.n_ca
    )
    agent._bc.demonstration_loss = lambda **_kwargs: (_ for _ in ()).throw(ValueError())

    messages = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="INFO",
    )
    try:
        with pytest.raises(ValueError):
            agent._run_bc_pretraining()
    finally:
        logger.remove(sink_id)

    assert messages == [
        "event=bc_pretraining_start buildings=1",
        "event=bc_pretraining_failure reason=pretraining_error error_type=ValueError",
    ]


def test_record_rejection_is_reported_without_incompatible_skip_metric() -> None:
    agent, dimension = _agent()
    assert agent._bc is not None
    agent._bc.record_demonstration(
        0,
        np.ones(dimension + 1),
        agent._per_building[0].layout,
        [0.25] * agent._per_building[0].layout.n_ca,
    )

    metrics = agent._bc.snapshot_metrics()

    assert metrics["behavior_cloning_rejected_at_record"] == 1.0


def test_final_demo_boundary_pretrains_every_stored_topology_group() -> None:
    agent, old_dimension = _agent()
    old_observation = np.ones(old_dimension, dtype=np.float64)
    agent.on_episode_start(episode=0, training=True)
    agent.set_observation_context(
        raw_observations=[old_observation], encoded_observations=[old_observation]
    )
    _update(agent, old_observation, agent.predict([old_observation]), 0)

    expanded_names, _ = _expand_charger_topology(
        agent, load_sample_observation_names_for_first_building()
    )
    current_dimension = BehaviorCloningRegularizer.full_representation_width(
        agent._per_building[0].layout
    )
    assert current_dimension > old_dimension
    current_observation = np.ones(current_dimension, dtype=np.float64)
    agent.set_observation_context(
        raw_observations=[current_observation],
        encoded_observations=[current_observation],
    )
    _update(agent, current_observation, agent.predict([current_observation]), 1)

    assert agent._bc is not None
    trained_signatures = []
    original_loss = agent._bc.demonstration_loss

    def record_training_group(**kwargs):
        trained_signatures.append(agent._bc.layout_signature(kwargs["layout"]))
        return original_loss(**kwargs)

    agent._bc.demonstration_loss = record_training_group
    current_signature = agent._bc.layout_signature(agent._per_building[0].layout)
    agent.on_episode_end(episode=0, training=True)

    old_signature = agent._bc.layout_signature(
        next(iter(agent._bc.demonstrations_by_signature.values()))[0].layout
    )
    assert trained_signatures == (
        [old_signature] * agent._bc.pretraining_epochs
        + [current_signature] * agent._bc.pretraining_epochs
    )
    metrics = agent.consume_latest_training_metrics()
    assert metrics["TPPO/behavior_cloning_rejected_at_record"] == 0.0
    assert metrics["TPPO/behavior_cloning_building_Building_1_usable_samples"] == 2.0
    assert metrics["TPPO/behavior_cloning_building_Building_1_trained_batches"] == 4.0
    assert metrics["TPPO/behavior_cloning_pretraining_batches"] == 4.0
    assert len(expanded_names) == current_dimension


def test_pretraining_uses_full_width_demo_with_trailing_excluded_features() -> None:
    agent, dimension = _agent()
    state = agent._per_building[0]
    assert agent._bc is not None
    layout = state.layout
    layout = type(layout)(
        building_id=layout.building_id,
        segments=layout.segments,
        n_sro=layout.n_sro,
        n_ca=layout.n_ca,
        ca_action_names=layout.ca_action_names,
        excluded_feature_names=layout.excluded_feature_names + ("trailing_excluded",),
    )
    full_width = BehaviorCloningRegularizer.full_representation_width(layout)
    assert full_width == dimension + 1
    agent._bc.record_demonstration(0, np.ones(full_width), layout, [0.25] * layout.n_ca)

    trained_layouts = []
    original_loss = agent._bc.demonstration_loss

    def record_training_layout(**kwargs):
        trained_layouts.append(kwargs["layout"])
        return original_loss(**kwargs)

    agent._bc.demonstration_loss = record_training_layout
    agent._run_bc_pretraining()

    assert trained_layouts == [layout] * agent._bc.pretraining_epochs
    assert agent._bc.snapshot_metrics()["behavior_cloning_rejected_at_record"] == 0.0


def test_pretraining_distinguishes_layouts_with_different_excluded_features() -> None:
    agent, dimension = _agent()
    state = agent._per_building[0]
    assert agent._bc is not None
    base_layout = state.layout
    extended_layout = type(base_layout)(
        building_id=base_layout.building_id,
        segments=base_layout.segments,
        n_sro=base_layout.n_sro,
        n_ca=base_layout.n_ca,
        ca_action_names=base_layout.ca_action_names,
        excluded_feature_names=base_layout.excluded_feature_names + ("trailing_excluded",),
    )
    extended_dimension = BehaviorCloningRegularizer.full_representation_width(
        extended_layout
    )
    assert extended_dimension == dimension + 1
    agent._bc.record_demonstration(
        0, np.ones(dimension), base_layout, [0.25] * base_layout.n_ca
    )
    agent._bc.record_demonstration(
        0, np.ones(extended_dimension), extended_layout,
        [0.5] * extended_layout.n_ca,
    )

    groups = agent._bc.demonstrations_for_building_by_signature(0)
    assert len(groups) == 2

    trained_signatures = []
    original_loss = agent._bc.demonstration_loss

    def record_training_group(**kwargs):
        trained_signatures.append(agent._bc.layout_signature(kwargs["layout"]))
        return original_loss(**kwargs)

    agent._bc.demonstration_loss = record_training_group
    agent._run_bc_pretraining()

    assert trained_signatures == [
        agent._bc.layout_signature(base_layout)
    ] * agent._bc.pretraining_epochs + [
        agent._bc.layout_signature(extended_layout)
    ] * agent._bc.pretraining_epochs
    assert agent._bc.snapshot_metrics()["behavior_cloning_rejected_at_record"] == 0.0


def test_auxiliary_bc_never_changes_ppo_actions() -> None:
    agent, dimension = _agent(demonstrations=0, weight=1.0)
    teacher_actions = _teacher(agent, 0.9)
    observation = np.ones(dimension, dtype=np.float64)
    assert agent._bc is not None
    agent._bc.record_demonstration(0, observation, agent._per_building[0].layout, teacher_actions[0])
    agent.on_episode_start(episode=1, training=True)
    agent.set_observation_context(raw_observations=[observation], encoded_observations=[observation])
    actor_actions = agent.predict([observation], deterministic=True)
    assert actor_actions != teacher_actions


def test_auxiliary_bc_update_changes_actor_and_tokenizer_but_not_critic() -> None:
    agent, dimension = _agent(demonstrations=0, weight=1.0)
    observation = np.ones(dimension, dtype=np.float64)
    state = agent._per_building[0]
    assert agent._bc is not None
    agent._bc.record_demonstration(
        0,
        observation,
        state.layout,
        [0.9] * state.layout.n_ca,
    )
    actor_before = [parameter.detach().clone() for parameter in state.actor.parameters()]
    tokenizer_before = [
        parameter.detach().clone() for parameter in state.tokenizer.parameters()
    ]
    critic_before = [parameter.detach().clone() for parameter in state.critic.parameters()]

    agent._run_auxiliary_bc_update(0, state)

    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, state.actor.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(tokenizer_before, state.tokenizer.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(critic_before, state.critic.parameters())
    )


def test_auxiliary_bc_uses_only_owning_building_demonstrations() -> None:
    agent, dimension = _agent(demonstrations=0, weight=1.0, building_count=2)
    assert agent._bc is not None
    agent._bc.batch_size = 2
    observation = np.ones(dimension, dtype=np.float64)
    for parameter in agent._per_building[0].actor.mlp.parameters():
        parameter.data.zero_()
    agent._bc.record_demonstration(
        0,
        observation,
        agent._per_building[0].layout,
        [0.0] * agent._per_building[0].layout.n_ca,
    )
    agent._bc.record_demonstration(
        1,
        observation,
        agent._per_building[0].layout,
        [1.0] * agent._per_building[1].layout.n_ca,
    )
    pooled = agent._bc.sample_demonstrations(
        0,
        agent._per_building[0].layout,
        batch_size=2,
    )
    assert [demo.target.tolist() for demo in pooled] == [
        [0.0] * agent._per_building[0].layout.n_ca,
    ]

    agent._run_auxiliary_bc_update(0, agent._per_building[0])

    assert agent._bc.snapshot_metrics()["behavior_cloning_loss"] == pytest.approx(0.0)


def test_auxiliary_bc_samples_demonstrations_during_ppo_update() -> None:
    agent, dimension = _agent(demonstrations=0, weight=1.0)
    teacher_actions = _teacher(agent, 0.9)
    observation = np.ones(dimension, dtype=np.float64)
    assert agent._bc is not None
    agent._bc.record_demonstration(0, observation, agent._per_building[0].layout, teacher_actions[0])
    sampled = []
    original_sample = agent._bc.sample_demonstrations

    def record_sample(building_idx, layout, batch_size):
        sampled.append((building_idx, layout, batch_size))
        return original_sample(building_idx, layout, batch_size)

    agent._bc.sample_demonstrations = record_sample
    agent.on_episode_start(episode=1, training=True)
    for step in range(agent._minibatch_size):
        agent.set_observation_context(
            raw_observations=[observation], encoded_observations=[observation]
        )
        actions = agent.predict([observation], deterministic=False)
        agent.update(
            observations=[observation], actions=actions, rewards=[0.1],
            next_observations=[observation], terminated=False, truncated=False,
            update_target_step=False, global_learning_step=step,
            update_step=step == agent._minibatch_size - 1,
            initial_exploration_done=True,
        )

    assert sampled
    assert all(building_idx == 0 for building_idx, _, _ in sampled)
    assert all(batch_size == agent._bc.batch_size for _, _, batch_size in sampled)


def test_auxiliary_bc_runs_after_all_ppo_epochs() -> None:
    agent, dimension = _agent(demonstrations=0, weight=1.0)
    agent._ppo_epochs = 2
    teacher_actions = _teacher(agent, 0.9)
    observation = np.ones(dimension, dtype=np.float64)
    assert agent._bc is not None
    agent._bc.record_demonstration(
        0, observation, agent._per_building[0].layout, teacher_actions[0]
    )
    events = []
    original_auxiliary_update = agent._run_auxiliary_bc_update
    original_optimizer_step = agent._per_building[0].optimizer.step

    def record_auxiliary_update(building_idx, state):
        events.append("auxiliary_bc")
        return original_auxiliary_update(building_idx, state)

    def record_optimizer_step():
        events.append("ppo")
        return original_optimizer_step()

    agent._run_auxiliary_bc_update = record_auxiliary_update
    agent._per_building[0].optimizer.step = record_optimizer_step
    agent.on_episode_start(episode=1, training=True)
    for step in range(agent._minibatch_size):
        agent.set_observation_context(
            raw_observations=[observation], encoded_observations=[observation]
        )
        actions = agent.predict([observation], deterministic=False)
        agent.update(
            observations=[observation], actions=actions, rewards=[0.1],
            next_observations=[observation], terminated=False, truncated=False,
            update_target_step=False, global_learning_step=step,
            update_step=step == agent._minibatch_size - 1,
            initial_exploration_done=True,
        )

    assert events == ["ppo"] * agent._ppo_epochs + ["auxiliary_bc"]


def test_checkpoint_restores_bc_demonstrations_phase_and_decay_progress(
    tmp_path: Path,
) -> None:
    agent, dimension = _agent(demonstrations=2, weight=1.0)
    observation = np.ones(dimension, dtype=np.float64)
    _teacher(agent, 0.5)
    agent.on_episode_start(episode=0, training=True)
    agent.set_observation_context(
        raw_observations=[observation], encoded_observations=[observation]
    )
    _update(agent, observation, agent.predict([observation]), step=7)
    assert agent._bc is not None
    expected_count = agent._bc.demonstration_count()
    expected_weight = agent._bc.effective_weight(agent._latest_global_learning_step)
    path = agent.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    fresh, _ = _agent(demonstrations=2, weight=1.0)
    fresh.load_checkpoint(path)

    assert fresh._bc is not None
    assert fresh._bc.demonstration_count() == expected_count == 1
    assert fresh._current_episode == 0
    assert fresh._in_demonstration_phase()
    assert fresh._latest_global_learning_step == 7
    assert fresh._bc.effective_weight(fresh._latest_global_learning_step) == expected_weight
    expected_actions = fresh._bc.compute_teacher_actions([observation])
    assert fresh.predict([observation]) == expected_actions
    assert fresh._pending_decisions == [None]


def test_checkpoint_restores_historical_layout_demonstrations_after_topology_change(
    tmp_path: Path,
) -> None:
    source, old_dimension = _agent(demonstrations=2, weight=1.0)
    assert source._bc is not None
    old_layout = source._per_building[0].layout
    source._bc.record_demonstration(
        0, np.ones(old_dimension), old_layout, [0.25] * old_layout.n_ca
    )
    expanded_names, expanded_actions = _expand_charger_topology(
        source, load_sample_observation_names_for_first_building()
    )
    current_layout = source._per_building[0].layout
    current_dimension = BehaviorCloningRegularizer.full_representation_width(
        current_layout
    )
    source._bc.record_demonstration(
        0, np.ones(current_dimension), current_layout,
        [0.5] * current_layout.n_ca,
    )
    expected_groups = source._bc.demonstrations_for_building_by_signature(0)
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    fresh, _ = _agent(demonstrations=2, weight=1.0)
    fresh.attach_environment(
        observation_names=[expanded_names], action_names=[expanded_actions],
        action_space=[_DummySpace(len(expanded_actions))], observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )
    fresh.load_checkpoint(path)

    assert fresh._bc is not None
    restored_groups = fresh._bc.demonstrations_for_building_by_signature(0)
    assert restored_groups.keys() == expected_groups.keys()
    assert [len(group) for group in restored_groups.values()] == [1, 1]


def test_checkpoint_rejects_legacy_bc_state_before_mutating_agent(
    tmp_path: Path,
) -> None:
    source, _ = _agent(demonstrations=2, weight=1.0)
    _materialize_optimizer_state(
        source._per_building[0].optimizer,
        source._per_building[0].bc_optimizer,
    )
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None
    target, dimension = _agent(demonstrations=2, weight=1.0)
    state = target._per_building[0]
    target._latest_training_metrics = {"target_metric": 1.0}
    _materialize_optimizer_state(state.optimizer, state.bc_optimizer)
    assert target._bc is not None
    observation = np.ones(dimension, dtype=np.float64)
    target._bc.record_demonstration(
        0, observation, state.layout, [0.25] * state.layout.n_ca
    )
    target._bc.record_demonstration(
        0, observation + 1.0, state.layout, [0.5] * state.layout.n_ca
    )
    sampled_demos = target._bc.sample_demonstrations(0, state.layout, batch_size=1)
    target._bc.demonstration_loss(
        layout=state.layout,
        demonstrations=sampled_demos,
        predicted_means=torch.full((1, state.layout.n_ca, 1), 0.75),
        global_learning_step=0,
    )
    target._bc.set_pretraining_epochs(3)
    target._bc.set_incompatible_demonstration_samples(2)
    target.on_episode_start(episode=2, training=True)
    target.predict([observation], deterministic=True)
    pending_before = target._pending_decisions[0]
    assert pending_before is not None
    model_states_before = {
        name: {
            key: value.detach().clone()
            for key, value in getattr(state, name).state_dict().items()
        }
        for name in ("tokenizer", "backbone", "actor", "critic")
    }
    optimizer_before = deepcopy(state.optimizer.state_dict())
    bc_optimizer_before = deepcopy(state.bc_optimizer.state_dict())
    normalizer_before = state.value_normalizer.state_dict()
    action_bounds_before = [
        (low.detach().clone(), high.detach().clone())
        for low, high in target._action_bounds
    ]
    counters_before = (
        target._latest_global_learning_step,
        target._ppo_update_count,
        target._current_episode,
    )
    topology_before = state.topology_version
    metrics_before = dict(target._latest_training_metrics)
    bc_state_before = target._bc.state_dict()

    payload = torch.load(path, weights_only=False)
    assert source._bc is not None
    layout = source._per_building[0].layout
    legacy_demo = Demonstration.__new__(Demonstration)
    object.__setattr__(legacy_demo, "observation", np.zeros(3, dtype=np.float32))
    object.__setattr__(legacy_demo, "layout", layout)
    object.__setattr__(
        legacy_demo,
        "layout_signature",
        source._bc.layout_signature(layout),
    )
    object.__setattr__(legacy_demo, "target", np.zeros(layout.n_ca, dtype=np.float32))
    payload["behavior_cloning_state"]["demonstrations"] = {0: [legacy_demo]}
    payload["behavior_cloning_state"]["seen_per_building"] = {0: 1}
    saved = payload["agents"][0]
    for name, before in model_states_before.items():
        saved_state = saved[f"{name}_state"]
        for key, value in before.items():
            saved_state[key] = value + torch.ones_like(value)
    for optimizer_name, expected in (
        ("optimizer_state", optimizer_before),
        ("bc_optimizer_state", bc_optimizer_before),
    ):
        saved_optimizer = saved[optimizer_name]
        for parameter_state in saved_optimizer["state"].values():
            for key, value in parameter_state.items():
                if isinstance(value, torch.Tensor):
                    parameter_state[key] = value + torch.ones_like(value)
        for saved_group, before_group in zip(
            saved_optimizer["param_groups"], expected["param_groups"]
        ):
            saved_group["lr"] = before_group["lr"] + 1.0
    saved["value_normalizer_state"] = {
        "mean": normalizer_before["mean"] + 1.0,
        "variance": normalizer_before["variance"] + 1.0,
        "count": normalizer_before["count"] + 1,
    }
    saved["topology_version"] = topology_before + 1
    payload["action_bounds"] = [
        (low + 1.0, high + 1.0) for low, high in action_bounds_before
    ]
    payload["global_learning_step"] = counters_before[0] + 1
    payload["ppo_update_count"] = counters_before[1] + 1
    payload["current_episode"] = counters_before[2] + 1
    payload["latest_training_metrics"] = {"checkpoint_metric": 1.0}
    torch.save(payload, path)
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state()

    with pytest.raises(RuntimeError, match="predates BC data contract"):
        target.load_checkpoint(path)

    for name, before in model_states_before.items():
        actual = getattr(state, name).state_dict()
        assert all(torch.equal(value, before[key]) for key, value in actual.items())
    _assert_structured_equal(state.optimizer.state_dict(), optimizer_before)
    _assert_structured_equal(state.bc_optimizer.state_dict(), bc_optimizer_before)
    assert state.value_normalizer.state_dict() == normalizer_before
    assert state.topology_version == topology_before
    assert all(
        torch.equal(low_before, low_after)
        and torch.equal(high_before, high_after)
        for (low_before, high_before), (low_after, high_after) in zip(
            action_bounds_before, target._action_bounds
        )
    )
    assert (
        target._latest_global_learning_step,
        target._ppo_update_count,
        target._current_episode,
    ) == counters_before
    assert target._latest_training_metrics == metrics_before
    assert target._pending_decisions[0] is pending_before
    assert target._bc is not None
    _assert_structured_equal(target._bc.state_dict(), bc_state_before)
    _assert_structured_equal(random.getstate(), python_rng_before)
    _assert_structured_equal(np.random.get_state(), numpy_rng_before)
    assert torch.equal(torch.get_rng_state(), torch_rng_before)


def test_checkpoint_rejects_corrupt_modern_demo_before_mutating_agent(
    tmp_path: Path,
) -> None:
    source, dimension = _agent(demonstrations=2, weight=1.0)
    assert source._bc is not None
    observation = np.ones(dimension, dtype=np.float64)
    source._bc.record_demonstration(
        0, observation, source._per_building[0].layout,
        [0.25] * source._per_building[0].layout.n_ca,
    )
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    target, _ = _agent(demonstrations=2, weight=1.0)
    assert target._bc is not None
    state = target._per_building[0]
    model_before = {
        name: {
            key: value.detach().clone()
            for key, value in getattr(state, name).state_dict().items()
        }
        for name in ("tokenizer", "backbone", "actor", "critic")
    }
    bc_before = target._bc.state_dict()
    payload = torch.load(path, weights_only=False)
    demo = payload["behavior_cloning_state"]["demonstrations"][0][0]
    payload["behavior_cloning_state"]["demonstrations"][0][0] = Demonstration(
        observation=demo.observation,
        encoded_length=demo.encoded_length + 1,
        layout=demo.layout,
        layout_signature=demo.layout_signature,
        target=demo.target,
    )
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="encoded_length"):
        target.load_checkpoint(path)

    for name, before in model_before.items():
        actual = getattr(state, name).state_dict()
        assert all(torch.equal(value, before[key]) for key, value in actual.items())
    _assert_structured_equal(target._bc.state_dict(), bc_before)


def test_checkpoint_rejects_bc_state_before_mutating_bc_disabled_agent(
    tmp_path: Path,
) -> None:
    source, dimension = _agent(demonstrations=2, weight=1.0)
    assert source._bc is not None
    observation = np.ones(dimension, dtype=np.float64)
    source._bc.record_demonstration(
        0, observation, source._per_building[0].layout,
        [0.25] * source._per_building[0].layout.n_ca,
    )
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    names = load_sample_observation_names_for_first_building()
    target = AgentTransformerPPO(_base_config())
    target.attach_environment(
        observation_names=[names], action_names=[list(_DEFAULT_ACTIONS)],
        action_space=[_DummySpace(len(_DEFAULT_ACTIONS))], observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )
    assert target._bc is None
    state = target._per_building[0]
    actor_before = {
        key: value.detach().clone() for key, value in state.actor.state_dict().items()
    }
    payload = torch.load(path, weights_only=False)
    demo = payload["behavior_cloning_state"]["demonstrations"][0][0]
    payload["behavior_cloning_state"]["demonstrations"][0][0] = Demonstration(
        observation=demo.observation,
        encoded_length=demo.encoded_length + 1,
        layout=demo.layout,
        layout_signature=demo.layout_signature,
        target=demo.target,
    )
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="BC-disabled target"):
        target.load_checkpoint(path)

    assert all(
        torch.equal(value, actor_before[key])
        for key, value in state.actor.state_dict().items()
    )


@pytest.mark.parametrize("with_demonstration", [False, True])
def test_checkpoint_rejects_bc_state_for_bc_disabled_target_before_mutating_agent(
    tmp_path: Path,
    with_demonstration: bool,
) -> None:
    source, dimension = _agent(demonstrations=2, weight=1.0)
    assert source._bc is not None
    if with_demonstration:
        source._bc.record_demonstration(
            0,
            np.ones(dimension),
            source._per_building[0].layout,
            [0.25] * source._per_building[0].layout.n_ca,
        )
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    names = load_sample_observation_names_for_first_building()
    target = AgentTransformerPPO(_base_config())
    target.attach_environment(
        observation_names=[names],
        action_names=[list(_DEFAULT_ACTIONS)],
        action_space=[_DummySpace(len(_DEFAULT_ACTIONS))],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )
    state = target._per_building[0]
    actor_before = {
        key: value.detach().clone() for key, value in state.actor.state_dict().items()
    }
    bounds_before = [(low.clone(), high.clone()) for low, high in target._action_bounds]
    counters_before = (
        target._latest_global_learning_step,
        target._ppo_update_count,
        target._current_episode,
    )

    with pytest.raises(RuntimeError, match="BC-disabled target"):
        target.load_checkpoint(path)

    assert all(
        torch.equal(value, actor_before[key])
        for key, value in state.actor.state_dict().items()
    )
    assert all(
        torch.equal(low, expected_low) and torch.equal(high, expected_high)
        for (low, high), (expected_low, expected_high) in zip(
            target._action_bounds, bounds_before
        )
    )
    assert (
        target._latest_global_learning_step,
        target._ppo_update_count,
        target._current_episode,
    ) == counters_before


def test_checkpoint_rejects_out_of_bounds_demo_layout_before_mutating_agent(
    tmp_path: Path,
) -> None:
    source, dimension = _agent(demonstrations=2, weight=1.0)
    assert source._bc is not None
    observation = np.ones(dimension, dtype=np.float64)
    source._bc.record_demonstration(
        0, observation, source._per_building[0].layout,
        [0.25] * source._per_building[0].layout.n_ca,
    )
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    target, _ = _agent(demonstrations=2, weight=1.0)
    state = target._per_building[0]
    actor_before = {
        key: value.detach().clone() for key, value in state.actor.state_dict().items()
    }
    payload = torch.load(path, weights_only=False)
    demo = payload["behavior_cloning_state"]["demonstrations"][0][0]
    bad_segment = replace(
        demo.layout.segments[0],
        feature_indices=(demo.encoded_length,) + demo.layout.segments[0].feature_indices[1:],
    )
    bad_layout = replace(
        demo.layout, segments=(bad_segment,) + demo.layout.segments[1:]
    )
    payload["behavior_cloning_state"]["demonstrations"][0][0] = Demonstration(
        observation=demo.observation,
        encoded_length=demo.encoded_length,
        layout=bad_layout,
        layout_signature=source._bc.layout_signature(bad_layout),
        target=demo.target,
    )
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="invalid BC layout"):
        target.load_checkpoint(path)

    assert all(
        torch.equal(value, actor_before[key])
        for key, value in state.actor.state_dict().items()
    )


@pytest.mark.parametrize("corruption", ["unknown_type", "feature_width"])
def test_checkpoint_rejects_tokenizer_incompatible_demo_layout_before_mutating_agent(
    tmp_path: Path,
    corruption: str,
) -> None:
    source, dimension = _agent(demonstrations=2, weight=1.0)
    assert source._bc is not None
    observation = np.ones(dimension, dtype=np.float64)
    source._bc.record_demonstration(
        0, observation, source._per_building[0].layout,
        [0.25] * source._per_building[0].layout.n_ca,
    )
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    target, _ = _agent(demonstrations=2, weight=1.0)
    state = target._per_building[0]
    assert target._bc is not None
    _materialize_optimizer_state(state.optimizer, state.bc_optimizer)
    target._latest_training_metrics = {"target_metric": 1.0}
    target._bc.record_demonstration(
        0, np.ones(dimension), state.layout, [0.25] * state.layout.n_ca
    )
    target._bc.record_demonstration(
        0, np.full(dimension, 2.0), state.layout, [0.5] * state.layout.n_ca
    )
    target._bc.demonstration_loss(
        layout=state.layout,
        demonstrations=target._bc.sample_demonstrations(0, state.layout, batch_size=1),
        predicted_means=torch.full((1, state.layout.n_ca, 1), 0.75),
        global_learning_step=0,
    )
    target._bc.set_pretraining_epochs(3)
    target._bc.set_incompatible_demonstration_samples(2)
    target.on_episode_start(episode=2, training=True)
    target.predict([np.ones(dimension)], deterministic=True)
    assert target._pending_decisions[0] is not None
    snapshot = _snapshot_restore_state(target)
    payload = torch.load(path, weights_only=False)
    demo = payload["behavior_cloning_state"]["demonstrations"][0][0]
    segment_idx = next(
        index for index, segment in enumerate(demo.layout.segments)
        if segment.family == "sro"
    )
    segment = demo.layout.segments[segment_idx]
    if corruption == "unknown_type":
        bad_segment = replace(segment, type_name="unknown_stored_type")
        bad_layout = replace(
            demo.layout,
            segments=(
                demo.layout.segments[:segment_idx]
                + (bad_segment,)
                + demo.layout.segments[segment_idx + 1:]
            ),
        )
    else:
        bad_segment = replace(
            segment,
            feature_indices=segment.feature_indices[:-1],
            feature_names=segment.feature_names[:-1],
        )
        bad_layout = replace(
            demo.layout,
            segments=(
                demo.layout.segments[:segment_idx]
                + (bad_segment,)
                + demo.layout.segments[segment_idx + 1:]
            ),
            excluded_feature_names=demo.layout.excluded_feature_names + ("preserved_width",),
        )
    payload["behavior_cloning_state"]["demonstrations"][0][0] = Demonstration(
        observation=demo.observation,
        encoded_length=demo.encoded_length,
        layout=bad_layout,
        layout_signature=source._bc.layout_signature(bad_layout),
        target=demo.target,
    )
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="BC layout/tokenizer compatibility"):
        target.load_checkpoint(path)

    _assert_restore_state_unchanged(target, snapshot)


@pytest.mark.parametrize(
    ("source_family", "target_family"),
    [("sro", "ca"), ("ca", "sro")],
)
def test_checkpoint_rejects_demo_segment_in_wrong_tokenizer_family_before_mutating_agent(
    tmp_path: Path,
    source_family: str,
    target_family: str,
) -> None:
    source, dimension = _agent(demonstrations=2, weight=1.0)
    assert source._bc is not None
    source._bc.record_demonstration(
        0, np.ones(dimension), source._per_building[0].layout,
        [0.25] * source._per_building[0].layout.n_ca,
    )
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    target, _ = _agent(demonstrations=2, weight=1.0)
    state = target._per_building[0]
    assert target._bc is not None
    _materialize_optimizer_state(state.optimizer, state.bc_optimizer)
    target._bc.record_demonstration(
        0, np.ones(dimension), state.layout, [0.5] * state.layout.n_ca
    )
    target.on_episode_start(episode=2, training=True)
    target.predict([np.ones(dimension)], deterministic=True)
    snapshot = _snapshot_restore_state(target)

    payload = torch.load(path, weights_only=False)
    assert payload["checkpoint_format_version"] == 2
    demo = payload["behavior_cloning_state"]["demonstrations"][0][0]
    segment_idx = next(
        index for index, segment in enumerate(demo.layout.segments)
        if segment.family == source_family
    )
    segment = demo.layout.segments[segment_idx]
    bad_segment = replace(segment, family=target_family)
    remaining_segments = (
        demo.layout.segments[:segment_idx] + demo.layout.segments[segment_idx + 1:]
    )
    nfc_idx = next(
        index for index, item in enumerate(remaining_segments)
        if item.family == "nfc"
    )
    if target_family == "sro":
        bad_segments = (
            remaining_segments[:nfc_idx]
            + (bad_segment,)
            + remaining_segments[nfc_idx:]
        )
        ca_action_names = demo.layout.ca_action_names[:-1]
        bad_target = demo.target[:-1]
    else:
        bad_segments = remaining_segments + (bad_segment,)
        ca_action_names = demo.layout.ca_action_names + ("tampered_action",)
        bad_target = np.append(demo.target, 0.0).astype(np.float32)
    bad_layout = replace(
        demo.layout,
        segments=bad_segments,
        n_sro=demo.layout.n_sro + (1 if target_family == "sro" else -1),
        n_ca=demo.layout.n_ca + (1 if target_family == "ca" else -1),
        ca_action_names=ca_action_names,
    )
    payload["behavior_cloning_state"]["demonstrations"][0][0] = Demonstration(
        observation=demo.observation,
        encoded_length=demo.encoded_length,
        layout=bad_layout,
        layout_signature=source._bc.layout_signature(bad_layout),
        target=bad_target,
    )
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="BC layout/tokenizer compatibility"):
        target.load_checkpoint(path)

    _assert_restore_state_unchanged(target, snapshot)


def test_checkpoint_rejects_incomplete_bc_state_before_mutating_agent(
    tmp_path: Path,
) -> None:
    source, dimension = _agent(demonstrations=2, weight=1.0)
    assert source._bc is not None
    source._bc.record_demonstration(
        0, np.ones(dimension), source._per_building[0].layout,
        [0.25] * source._per_building[0].layout.n_ca,
    )
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    target, _ = _agent(demonstrations=2, weight=1.0)
    state = target._per_building[0]
    assert target._bc is not None
    _materialize_optimizer_state(state.optimizer, state.bc_optimizer)
    target._bc.record_demonstration(
        0, np.ones(dimension), state.layout, [0.5] * state.layout.n_ca
    )
    target.on_episode_start(episode=2, training=True)
    target.predict([np.ones(dimension)], deterministic=True)
    snapshot = _snapshot_restore_state(target)

    payload = torch.load(path, weights_only=False)
    payload["behavior_cloning_state"].pop("seen_per_building")
    saved_actor = payload["agents"][0]["actor_state"]
    for key, value in saved_actor.items():
        saved_actor[key] = value + torch.ones_like(value)
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="missing required key.*seen_per_building"):
        target.load_checkpoint(path)

    _assert_restore_state_unchanged(target, snapshot)


def test_checkpoint_rejects_inconsistent_bc_reservoir_before_mutating_agent(
    tmp_path: Path,
) -> None:
    source, dimension = _agent(demonstrations=2, weight=1.0)
    assert source._bc is not None
    source._bc.record_demonstration(
        0, np.ones(dimension), source._per_building[0].layout,
        [0.25] * source._per_building[0].layout.n_ca,
    )
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    target, _ = _agent(demonstrations=2, weight=1.0)
    state = target._per_building[0]
    assert target._bc is not None
    _materialize_optimizer_state(state.optimizer, state.bc_optimizer)
    target._bc.record_demonstration(
        0, np.ones(dimension), state.layout, [0.5] * state.layout.n_ca
    )
    target.on_episode_start(episode=2, training=True)
    target.predict([np.ones(dimension)], deterministic=True)
    snapshot = _snapshot_restore_state(target)

    payload = torch.load(path, weights_only=False)
    payload["behavior_cloning_state"]["seen_per_building"].pop(0)
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="reservoir building keys"):
        target.load_checkpoint(path)

    _assert_restore_state_unchanged(target, snapshot)


def test_checkpoint_rejects_bc_buffer_above_receiver_capacity_before_mutating_agent(
    tmp_path: Path,
) -> None:
    source, dimension = _agent(demonstrations=2, weight=1.0)
    assert source._bc is not None
    source._bc.max_samples_per_building = 2
    source._bc.record_demonstration(
        0, np.ones(dimension), source._per_building[0].layout,
        [0.25] * source._per_building[0].layout.n_ca,
    )
    source._bc.record_demonstration(
        0, np.full(dimension, 2.0), source._per_building[0].layout,
        [0.5] * source._per_building[0].layout.n_ca,
    )
    path = source.save_checkpoint(str(tmp_path), step=7)
    assert path is not None

    target, _ = _agent(demonstrations=2, weight=1.0)
    state = target._per_building[0]
    assert target._bc is not None
    target._bc.max_samples_per_building = 1
    _materialize_optimizer_state(state.optimizer, state.bc_optimizer)
    target._bc.record_demonstration(
        0, np.ones(dimension), state.layout, [0.75] * state.layout.n_ca
    )
    target.on_episode_start(episode=2, training=True)
    target.predict([np.ones(dimension)], deterministic=True)
    snapshot = _snapshot_restore_state(target)

    with pytest.raises(RuntimeError, match="BC capacity incompatibility"):
        target.load_checkpoint(path)

    _assert_restore_state_unchanged(target, snapshot)
