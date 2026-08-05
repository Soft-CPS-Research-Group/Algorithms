from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from algorithms.agents.cc_level2_agent import (
    CCLevel2Agent,
    CommunityMarketMakerNetV2,
    DeterministicVectorMultiplierPolicy,
    RolloutBufferV2,
    _CC_LEVEL2_BUILDING_FEATURES,
    _CC_LEVEL2_DISTRICT_FEATURES,
)
from utils.config_schema import CCLevel2Hyperparameters


def _observation_names(building: int) -> list[str]:
    names = list(_CC_LEVEL2_DISTRICT_FEATURES)
    for feature in _CC_LEVEL2_BUILDING_FEATURES:
        if "::" in feature:
            prefix, tail = feature.split("::", 1)
            names.append(f"{prefix}::Building_{building}/asset::{tail}")
        else:
            names.append(f"charger::Building_{building}/charger::{feature}")
    return names


def _attached_agent(*, interval: int = 4, num_steps: int = 8) -> CCLevel2Agent:
    count = 2
    agent = CCLevel2Agent(
        {
            "algorithm": {
                "hyperparameters": {
                    "num_buildings": count,
                    "c_dim": len(_CC_LEVEL2_DISTRICT_FEATURES)
                    + len(_CC_LEVEL2_BUILDING_FEATURES) * count,
                    "hidden_dims": [8],
                    "cc_action_interval": interval,
                    "num_steps": num_steps,
                    "price_min": 0.8,
                    "price_max": 1.1,
                    "reference_multipliers": [0.9, 1.05],
                    "initial_log_std": -3.0,
                    "bc_pretrain_enabled": False,
                }
            }
        }
    )
    names = [_observation_names(1), _observation_names(2)]
    agent.attach_environment(
        observation_names=names,
        action_names=[[], []],
        action_space=[None, None],
        observation_space=[None, None],
        metadata={},
    )
    return agent


def test_cc_level2_requests_its_own_pipeline_observation_profile() -> None:
    assert CCLevel2Agent.observation_encoding_profile == "cc_level2"


def test_vector_policy_honors_initial_log_std() -> None:
    policy = CommunityMarketMakerNetV2(
        c_dim=2,
        num_buildings=3,
        hidden_dims=[4],
        initial_log_std=-2.25,
    )

    torch.testing.assert_close(policy.log_std, torch.full((3,), -2.25))

    action, log_prob, entropy, value = policy.get_action_and_value(
        torch.zeros((5, 2), dtype=torch.float32)
    )
    assert action.shape == (5, 3)
    assert log_prob.shape == (5, 3)
    assert entropy.shape == (5, 3)
    assert value.shape == (5,)


def test_cc_level2_starts_exactly_at_reference_vector() -> None:
    agent = _attached_agent()
    observations = [
        np.zeros(len(_observation_names(1)), dtype=np.float32),
        np.zeros(len(_observation_names(2)), dtype=np.float32),
    ]

    output = agent.predict(observations, deterministic=True)

    np.testing.assert_allclose(output, [0.9, 1.05], atol=1e-6)


def test_deterministic_vector_policy_exports_price_mapping() -> None:
    agent = _attached_agent()
    inference = DeterministicVectorMultiplierPolicy(
        agent.policy,
        agent._price_min,
        agent._price_max,
        agent._reference_multipliers,
        agent._policy_residual_scale,
    )

    output = inference(torch.zeros((3, agent._c_dim), dtype=torch.float32))

    assert output.shape == (3, 2)
    torch.testing.assert_close(
        output,
        torch.tensor([[0.9, 1.05]]).expand(3, -1),
        atol=1e-6,
        rtol=0.0,
    )


def test_cc_level2_can_conservatively_scale_policy_away_from_reference() -> None:
    agent = _attached_agent()
    agent._policy_residual_scale = 0.5
    with torch.no_grad():
        agent.policy.mean_head.weight.zero_()
        agent.policy.mean_head.bias.zero_()
    observations = [
        np.zeros(len(_observation_names(1)), dtype=np.float32),
        np.zeros(len(_observation_names(2)), dtype=np.float32),
    ]

    output = agent.predict(observations, deterministic=True)

    # Full policy output is the midpoint 0.95; blend halfway from each
    # building's measured reference [0.90, 1.05].
    np.testing.assert_allclose(output, [0.925, 1.0], atol=1e-6)


def test_cc_level2_temporal_abstraction_adds_one_joint_transition() -> None:
    agent = _attached_agent(interval=4)
    observations = [
        np.zeros(len(_observation_names(1)), dtype=np.float32),
        np.zeros(len(_observation_names(2)), dtype=np.float32),
    ]

    outputs = []
    for step in range(4):
        outputs.append(agent.predict(observations, deterministic=True))
        agent.update(
            observations,
            [[], []],
            [-1.0, -1.0],
            observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=True,
            initial_exploration_done=True,
        )

    assert outputs == [outputs[0]] * 4
    assert agent.rollout_buffer._ptr == 1


def test_cc_level2_temporal_abstraction_does_not_depend_on_learning_updates() -> None:
    agent = _attached_agent(interval=4)
    observations = [
        np.zeros(len(_observation_names(1)), dtype=np.float32),
        np.zeros(len(_observation_names(2)), dtype=np.float32),
    ]

    outputs = [agent.predict(observations, deterministic=True) for _ in range(8)]

    assert len(agent._decision_trace) == 2
    assert outputs[:4] == [outputs[0]] * 4
    assert outputs[4:] == [outputs[4]] * 4


def test_cc_level2_rollout_gae_stops_at_episode_boundary() -> None:
    buffer = RolloutBufferV2(num_steps=3, c_dim=1, num_buildings=2)
    buffer.values[:] = 0.0
    buffer.rewards[:] = [1.0, 2.0, 3.0]
    buffer.dones[:] = [0.0, 1.0, 0.0]

    buffer.compute_gae(last_value=0.0, last_done=False, gamma=1.0, gae_lambda=1.0)

    np.testing.assert_allclose(buffer.returns, [3.0, 2.0, 3.0])


def test_cc_level2_export_persists_vector_contract_and_trace(tmp_path) -> None:
    agent = _attached_agent()
    agent._decision_trace = [
        {
            "timestep": 0,
            "mult_mean": 0.975,
            "mult_std": 0.075,
            "mult_min": 0.9,
            "mult_max": 1.05,
            "value_est": 0.0,
            "import_norm": 0.3,
            "pv_norm": 0.1,
            "carbon_norm": 0.2,
            "ev_harm_mean": 0.0,
            "ev_harm_max": 0.0,
            "n_ev_connected": 0.0,
            "mult_b0": 0.9,
            "mult_b1": 1.05,
        }
    ]

    metadata = agent.export_artifacts(str(tmp_path))

    assert metadata["output_contract"] == "deterministic_per_building_price_multiplier_vector"
    assert metadata["reference_multipliers"] == pytest.approx([0.9, 1.05])
    assert metadata["policy_residual_scale"] == 1.0
    assert (tmp_path / "onnx_models" / "cc2_market_maker.onnx").is_file()
    with (tmp_path / "decision_trace.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["episode"] == "1"
    assert rows[0]["mult_b1"] == "1.05"


def test_cc_level2_schema_rejects_reference_vector_mismatch() -> None:
    with pytest.raises(ValueError, match="length must equal num_buildings"):
        CCLevel2Hyperparameters.model_validate(
            {"num_buildings": 2, "reference_multipliers": [1.0]}
        )


def test_cc_level2_schema_rejects_policy_residual_scale_outside_unit_interval() -> None:
    with pytest.raises(ValueError):
        CCLevel2Hyperparameters.model_validate({"policy_residual_scale": 1.1})
