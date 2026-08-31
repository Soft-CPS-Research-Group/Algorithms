from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest
import torch

from tests._entity_sample_obs_names import (
    load_sample_observation_names_for_first_building,
)
from tests.test_agent_transformer_matd3 import (
    _ACTION_NAMES,
    _Box,
    _config,
    _make_agent,
    _parameters,
    _transition,
)
from tests.test_agent_transformer_matd3_behavior_cloning import _bc_b_agent
from tests.test_agent_transformer_matd3_residual import _agent


def _attached_agent(config: dict):
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3

    names = load_sample_observation_names_for_first_building()
    agent = AgentTransformerMATD3(config)
    agent.attach_environment(
        observation_names=[list(names)],
        action_names=[list(_ACTION_NAMES)],
        action_space=[_Box([-2.0, -0.5], [1.0, 0.75])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    return agent, len(names)


def _assert_module_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    assert all(torch.equal(left_state[key], right_state[key]) for key in left_state)


def test_full_format_6_round_trip_restores_training_replay_queue_and_rng(
    tmp_path: Path,
) -> None:
    source, obs_dim = _make_agent(
        buildings=1,
        n_step_returns=3,
        batch_size=2,
        reward_normalization_enabled=True,
    )
    for step in range(4):
        _transition(source, obs_dim, step, rewards=[float(step + 1)])
    source.exploration_sigma = 0.123
    source.exploration_step = 17
    path = source.save_checkpoint(str(tmp_path), step=41)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    restored, _ = _make_agent(
        buildings=1,
        n_step_returns=3,
        batch_size=2,
        reward_normalization_enabled=True,
    )
    restored.load_checkpoint(path)

    assert Path(path).name == "transformer_matd3_step41.pt"
    assert payload["checkpoint_version"] == 6
    assert payload["algorithm"] == "AgentTransformerMATD3"
    assert payload["checkpoint_mode"] == "full"
    source_state = source._per_building[0]
    restored_state = restored._per_building[0]
    for name in (
        "tokenizer", "backbone", "actor", "tokenizer_target",
        "backbone_target", "actor_target", "critic_1", "critic_1_target",
        "critic_2", "critic_2_target",
    ):
        _assert_module_equal(getattr(source_state, name), getattr(restored_state, name))
    assert restored.replay_buffer.get_state()["global_fifo"] == (
        source.replay_buffer.get_state()["global_fifo"]
    )
    assert tuple(restored.replay_buffer.signatures()) == tuple(
        source.replay_buffer.signatures()
    )
    assert len(restored._n_step_queue) == len(source._n_step_queue) == 2
    assert restored.exploration_sigma == pytest.approx(0.123)
    assert restored.exploration_step == 17
    assert restored.reward_norm_count == source.reward_norm_count
    assert restored.reward_norm_mean == pytest.approx(source.reward_norm_mean)
    assert restored.reward_norm_m2 == pytest.approx(source.reward_norm_m2)
    assert restored.critic_update_count == source.critic_update_count
    assert restored.actor_update_count == source.actor_update_count
    assert restored.target_update_count == source.target_update_count
    assert random.getstate() == payload["rng_state"]["python"]
    assert np.array_equal(np.random.get_state()[1], payload["rng_state"]["numpy"][1])
    assert torch.equal(torch.get_rng_state(), payload["rng_state"]["torch"])
    assert len(restored_state.critic_1_optimizer.state) > 0


def test_inference_round_trip_restores_actor_stack_and_operational_step(
    tmp_path: Path,
) -> None:
    config = _config()
    config["checkpointing"] = {"checkpoint_mode": "inference"}
    source, _ = _attached_agent(config)
    with torch.no_grad():
        next(source._per_building[0].actor.parameters()).add_(0.25)
    source.exploration_step = 23
    path = source.save_checkpoint(str(tmp_path), step=7)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "replay_buffer" not in payload
    assert "critic_1_state_dict_0" not in payload

    restored, _ = _attached_agent(config)
    restored.frozen = True
    restored.load_checkpoint(path)

    _assert_module_equal(
        source._per_building[0].tokenizer, restored._per_building[0].tokenizer
    )
    _assert_module_equal(
        source._per_building[0].backbone, restored._per_building[0].backbone
    )
    _assert_module_equal(source._per_building[0].actor, restored._per_building[0].actor)
    assert restored.exploration_step == 23


def test_inference_restore_rejects_non_frozen_stage(tmp_path: Path) -> None:
    config = _config()
    config["checkpointing"] = {"checkpoint_mode": "inference"}
    source, _ = _attached_agent(config)
    path = source.save_checkpoint(str(tmp_path), step=1)
    target, _ = _attached_agent(config)

    with pytest.raises(RuntimeError, match="frozen pipeline stage"):
        target.load_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_version", 5, "checkpoint_version"),
        ("algorithm", "MATD3", "algorithm"),
        ("num_agents", 2, "num_agents"),
        ("building_names", ["Other"], "building_names"),
    ],
)
def test_strict_header_rejection_precedes_mutation(
    tmp_path: Path, field: str, value, message: str
) -> None:
    source, _ = _make_agent(buildings=1)
    path = source.save_checkpoint(str(tmp_path), step=2)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload[field] = value
    corrupt = tmp_path / f"bad_{field}.pt"
    torch.save(payload, corrupt)
    target, _ = _make_agent(buildings=1)
    actor_before = _parameters(target._per_building[0].actor)

    with pytest.raises(ValueError, match=message):
        target.load_checkpoint(str(corrupt))

    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(
            actor_before, target._per_building[0].actor.parameters()
        )
    )


@pytest.mark.parametrize(
    "field", ["layout_signature_0", "action_names_0", "action_bounds_0"]
)
def test_strict_layout_rejection_precedes_mutation(tmp_path: Path, field: str) -> None:
    source, _ = _make_agent(buildings=1)
    path = source.save_checkpoint(str(tmp_path), step=3)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if field == "layout_signature_0":
        signature = list(payload[field])
        signature[0] += 1
        payload[field] = tuple(signature)
    elif field == "action_names_0":
        payload[field] = ("wrong", "names")
    else:
        low, high = payload[field]
        payload[field] = (low - 0.1, high)
    corrupt = tmp_path / f"bad_{field}.pt"
    torch.save(payload, corrupt)
    target, _ = _make_agent(buildings=1)
    actor_before = _parameters(target._per_building[0].actor)

    with pytest.raises(ValueError, match="mismatch"):
        target.load_checkpoint(str(corrupt))

    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(
            actor_before, target._per_building[0].actor.parameters()
        )
    )


def test_apply_failure_rolls_back_partial_neural_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    source, _ = _make_agent(buildings=1)
    with torch.no_grad():
        next(source._per_building[0].actor.parameters()).add_(0.5)
    path = source.save_checkpoint(str(tmp_path), step=5)
    target, _ = _make_agent(buildings=1)
    actor_before = _parameters(target._per_building[0].actor)

    def fail_load(state_dict):
        del state_dict
        raise RuntimeError("critic apply failed")

    monkeypatch.setattr(target._per_building[0].critic_1, "load_state_dict", fail_load)

    with pytest.raises(RuntimeError, match="critic apply failed"):
        target.load_checkpoint(path)

    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(
            actor_before, target._per_building[0].actor.parameters()
        )
    )


def test_checkpoint_rejects_unstable_n_step_optional_fields_before_mutation(
    tmp_path: Path,
) -> None:
    source, obs_dim = _make_agent(buildings=1, n_step_returns=3, batch_size=1)
    _transition(source, obs_dim, 0)
    _transition(source, obs_dim, 1)
    path = source.save_checkpoint(str(tmp_path), step=6)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["n_step_queue"][1]["behavior_actions"] = [
        np.zeros(2, dtype=np.float32)
    ]
    corrupt = tmp_path / "unstable_optional_fields.pt"
    torch.save(payload, corrupt)
    target, _ = _make_agent(buildings=1, n_step_returns=3, batch_size=1)
    actor_before = _parameters(target._per_building[0].actor)

    with pytest.raises(ValueError, match="optional action presence is unstable"):
        target.load_checkpoint(str(corrupt))

    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(
            actor_before, target._per_building[0].actor.parameters()
        )
    )


def test_full_checkpoint_restores_bc_b_reservoir_without_teacher(
    tmp_path: Path,
) -> None:
    source, obs_dim = _bc_b_agent()
    source._bc_b.record_demonstration(
        0,
        np.zeros(obs_dim, dtype=np.float32),
        source._per_building[0].layout,
        [0.25, -0.25],
    )
    path = source.save_checkpoint(str(tmp_path), step=6)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "teacher_policy" not in payload["bc_state"]["bc_b_state"]["regularizer"]
    target, _ = _bc_b_agent()

    target.load_checkpoint(path)

    assert target._bc_b.demonstration_count(0) == 1
    assert target._bc_b.teacher_policy is not None


def test_full_checkpoint_restores_bc_a_clock_and_optimizer(tmp_path: Path) -> None:
    replay_bc = {
        "enabled": True,
        "teacher": "replay_action",
        "weight": 1.0,
        "offline_pretrain_steps": 4,
    }
    source, _ = _agent(replay_bc=replay_bc)
    source.bc_a_offline_pretrain_completed_steps = 3
    path = source.save_checkpoint(str(tmp_path), step=8)
    target, _ = _agent(replay_bc=replay_bc)

    target.load_checkpoint(path)

    assert target.bc_a_offline_pretrain_completed_steps == 3
    assert target._per_building[0].bc_a_optimizer is not None
    assert target._per_building[0].bc_a_optimizer is target._per_building[0].actor_optimizer
