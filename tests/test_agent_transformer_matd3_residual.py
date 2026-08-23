from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algorithms.registry import ALGORITHM_REGISTRY
from algorithms.transformer_matd3.agent import AgentTransformerMATD3
from tests.test_agent_transformer_matd3 import (
    _ACTION_NAMES,
    _Box,
    _config,
)
from tests._entity_sample_obs_names import (
    load_sample_observation_names_for_first_building,
)


def _agent(
    *,
    hyperparameters: dict | None = None,
    replay_bc: dict | None = None,
) -> tuple[AgentTransformerMATD3, int]:
    config = _config(**(hyperparameters or {}))
    if replay_bc is not None:
        config["algorithm"]["behavior_cloning"] = {
            "replay_based": replay_bc,
        }
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


def _residual_hyperparameters(**overrides) -> dict:
    values = {
        "residual_policy_enabled": True,
        "warm_start_policy_name": "RandomPolicy",
        "residual_action_scale": 0.5,
        "residual_action_final_scale": 0.5,
        "residual_storage_action_scale_multiplier": 0.2,
        "residual_ev_action_scale_multiplier": 0.4,
    }
    values.update(overrides)
    return values


def test_agent_registry_preserves_final_stage_capability() -> None:
    assert ALGORITHM_REGISTRY["AgentTransformerMATD3"].requires_final_pipeline_stage is True


def test_residual_composition_uses_ca_order_span_and_authority() -> None:
    agent, _ = _agent(hyperparameters=_residual_hyperparameters())
    unit_action = torch.tensor([[1.0, -1.0]], device=agent.device)
    base_action = torch.tensor([[0.2, 0.1]], device=agent.device)

    action = agent._compose_policy_action(0, unit_action, base_action)

    expected = base_action + 0.5 * torch.tensor(
        [3.0, 1.25], device=agent.device
    ) * 0.5 * torch.tensor([0.2, 0.4], device=agent.device) * unit_action
    assert torch.allclose(action, expected)


def test_warm_start_teacher_width_is_validated_before_composition() -> None:
    agent, obs_dim = _agent(hyperparameters=_residual_hyperparameters())

    class _WrongWidthPolicy(torch.nn.Module):
        def predict(self, observations, deterministic):
            del observations, deterministic
            return [[0.0]]

    agent._warm_start_policy = _WrongWidthPolicy()
    observations = [np.zeros(obs_dim, dtype=np.float32)]
    agent.set_observation_context(raw_observations=observations)

    with pytest.raises(ValueError, match="warm-start action width"):
        agent.predict(observations, deterministic=True)


def test_target_smoothing_uses_residual_authority(monkeypatch) -> None:
    agent, obs_dim = _agent(
        hyperparameters=_residual_hyperparameters(
            target_policy_noise=2.0,
            target_policy_noise_clip=0.1,
        )
    )
    state = agent._per_building[0]
    observations = torch.zeros((2, obs_dim), device=agent.device)
    base = torch.zeros((2, 2), device=agent.device)
    unsmoothed = agent._policy_action(
        0,
        state,
        observations,
        target=True,
        base_action=base,
    )
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))

    smoothed = agent._target_action(
        state,
        observations,
        index=0,
        base_action=base,
    )

    expected_limit = 0.1 * (state.action_high - state.action_low) * 0.5 * torch.tensor(
        [0.2, 0.4], device=agent.device
    )
    expected = torch.maximum(
        torch.minimum(unsmoothed + expected_limit, state.action_high),
        state.action_low,
    )
    assert torch.allclose(smoothed, expected)


def test_local_safety_projection_is_returned_and_stored_in_replay() -> None:
    agent, obs_dim = _agent(
        hyperparameters={
            "local_action_safety_enabled": True,
            "batch_size": 2,
        }
    )
    projected = [0.25, -0.1]
    agent._local_action_safety_adapters[0] = SimpleNamespace(
        project=lambda raw, proposed: SimpleNamespace(
            executed_actions=projected,
            interventions=(),
            infeasible_reasons=(),
        )
    )
    observations = [np.zeros(obs_dim, dtype=np.float32)]
    next_observations = [np.ones(obs_dim, dtype=np.float32)]
    agent.set_observation_context(raw_observations=observations)

    actions = agent.predict(observations, deterministic=True)
    agent.update(
        observations,
        actions,
        [0.0],
        next_observations,
        False,
        False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=False,
    )

    transition = agent.replay_buffer.get_state()["transitions"][0]
    assert actions == [projected]
    assert transition.actions[0].tolist() == pytest.approx(projected)
    assert agent.get_diagnostic_metrics()[
        "TransformerMATD3/local_action_safety_projections"
    ] == 1.0


def test_local_safety_requires_raw_observation_context() -> None:
    agent, obs_dim = _agent(
        hyperparameters={"local_action_safety_enabled": True}
    )

    with pytest.raises(RuntimeError, match="requires raw observation context"):
        agent.predict([np.zeros(obs_dim, dtype=np.float32)], deterministic=True)


def test_residual_replay_requires_next_warm_start_context() -> None:
    agent, obs_dim = _agent(hyperparameters=_residual_hyperparameters())
    observations = [np.zeros(obs_dim, dtype=np.float32)]
    agent.set_observation_context(raw_observations=observations)
    actions = agent.predict(observations, deterministic=True)

    with pytest.raises(RuntimeError, match="requires next warm-start base"):
        agent.update(
            observations,
            actions,
            [0.0],
            observations,
            False,
            False,
            update_target_step=False,
            global_learning_step=0,
            update_step=False,
            initial_exploration_done=False,
        )
