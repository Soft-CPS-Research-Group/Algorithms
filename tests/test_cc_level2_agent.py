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
    _CC_LEVEL2_HEADROOM_FEATURE,
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


def test_cc_level2_headroom_and_physical_bc_teacher_are_explicit() -> None:
    count = 2
    district_names = [
        *_CC_LEVEL2_DISTRICT_FEATURES,
        _CC_LEVEL2_HEADROOM_FEATURE,
    ]
    names = []
    for building in range(1, count + 1):
        building_names = list(district_names)
        for feature in _CC_LEVEL2_BUILDING_FEATURES:
            if "::" in feature:
                prefix, tail = feature.split("::", 1)
                building_names.append(
                    f"{prefix}::Building_{building}/asset::{tail}"
                )
            else:
                building_names.append(
                    f"charger::Building_{building}/charger::{feature}"
                )
        names.append(building_names)

    agent = CCLevel2Agent(
        {
            "algorithm": {
                "hyperparameters": {
                    "num_buildings": count,
                    "c_dim": len(district_names)
                    + len(_CC_LEVEL2_BUILDING_FEATURES) * count,
                    "hidden_dims": [8],
                    "include_community_headroom": True,
                    "bc_pretrain_enabled": True,
                    "bc_use_physical_teacher_context": True,
                    "bc_target_import": 2.0,
                    "bc_reference_peak": 4.0,
                    "bc_reference_export": 3.0,
                    "bc_w_cost": 0.0,
                    "bc_w_peak": 1.0,
                    "bc_w_export": 0.0,
                }
            }
        }
    )
    agent.attach_environment(
        observation_names=names,
        action_names=[[], []],
        action_space=[None, None],
        observation_space=[None, None],
        metadata={"raw_observation_names": names},
    )

    encoded = [np.zeros(len(member_names), dtype=np.float32) for member_names in names]
    encoded[0][names[0].index(_CC_LEVEL2_HEADROOM_FEATURE)] = 0.25
    raw = [member.copy() for member in encoded]
    raw_values = {
        "district__electricity_pricing": 0.20,
        "district__electricity_pricing_predicted_1": 0.25,
        "district__electricity_pricing_predicted_2": 0.30,
        "district__electricity_pricing_predicted_3": 0.35,
        "district__community_import_power_kw": 10.0,
        "district__community_export_power_kw": 1.5,
    }
    for name, value in raw_values.items():
        raw[0][names[0].index(name)] = value
    agent.set_observation_context(raw_observations=raw)

    policy_context = agent._build_context(encoded)
    teacher_context = agent._build_teacher_context(policy_context)

    assert len(policy_context) == 17 + 6 * count
    assert policy_context[district_names.index(_CC_LEVEL2_HEADROOM_FEATURE)] == (
        pytest.approx(0.25)
    )
    for name, value in raw_values.items():
        assert teacher_context[district_names.index(name)] == pytest.approx(value)
    # 10 kW × 0.25 h = 2.5 kWh. Against a 2.0 kWh target and 4.0
    # reference, the physical peak signal is (0.5² / 4) = 0.0625.
    assert agent._community_signal(teacher_context) == pytest.approx(0.0625)


def test_cc_level2_physical_bc_teacher_requires_raw_metadata() -> None:
    with pytest.raises(ValueError, match="requires raw district features"):
        agent = CCLevel2Agent(
            {
                "algorithm": {
                    "hyperparameters": {
                        "num_buildings": 2,
                        "c_dim": len(_CC_LEVEL2_DISTRICT_FEATURES)
                        + len(_CC_LEVEL2_BUILDING_FEATURES) * 2,
                        "bc_pretrain_enabled": True,
                        "bc_use_physical_teacher_context": True,
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
        agent._policy_parameterization,
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


def test_centered_residual_can_learn_away_from_reference_at_price_bound() -> None:
    agent = CCLevel2Agent(
        {
            "algorithm": {
                "hyperparameters": {
                    "num_buildings": 2,
                    "c_dim": len(_CC_LEVEL2_DISTRICT_FEATURES)
                    + len(_CC_LEVEL2_BUILDING_FEATURES) * 2,
                    "hidden_dims": [8],
                    "price_min": 0.5,
                    "price_max": 1.3,
                    "reference_multipliers": [1.3, 0.5],
                    "policy_residual_scale": 1.0,
                    "policy_parameterization": "centered_residual",
                    "cc_action_interval": 1,
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
    observations = [np.zeros(len(names[0]), dtype=np.float32) for _ in range(2)]

    np.testing.assert_allclose(
        agent.predict(observations, deterministic=True), [1.3, 0.5], atol=1e-6
    )
    with torch.no_grad():
        agent.policy.mean_head.bias.copy_(torch.tensor([-0.2, 0.2]))

    moved = agent.predict(observations, deterministic=True)
    assert moved[0] < 1.3
    assert moved[1] > 0.5
    assert moved[0] == pytest.approx(1.3 - 0.8 * np.tanh(0.2))
    assert moved[1] == pytest.approx(0.5 + 0.8 * np.tanh(0.2))


def test_centered_residual_export_matches_runtime_mapping() -> None:
    agent = _attached_agent()
    agent._policy_parameterization = "centered_residual"
    agent._initialize_policy_at_reference()
    with torch.no_grad():
        agent.policy.mean_head.bias.copy_(torch.tensor([-0.3, 0.4]))
    inference = DeterministicVectorMultiplierPolicy(
        agent.policy,
        agent._price_min,
        agent._price_max,
        agent._reference_multipliers,
        agent._policy_residual_scale,
        agent._policy_parameterization,
    )
    observations = [
        np.zeros(len(_observation_names(1)), dtype=np.float32),
        np.zeros(len(_observation_names(2)), dtype=np.float32),
    ]

    expected = agent.predict(observations, deterministic=True)
    exported = inference(torch.zeros((1, agent._c_dim), dtype=torch.float32))

    np.testing.assert_allclose(exported.detach().numpy()[0], expected, atol=1e-6)


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


def test_cc_level2_does_not_store_bc_boundary_as_on_policy_transition() -> None:
    agent = _attached_agent(interval=1, num_steps=2)
    agent._bc_enabled = True
    agent._bc_pretrain_done = False
    agent._bc_collect_steps = 1
    agent._bc_train_steps = 1
    agent._bc_target_import = 1.0
    agent._bc_reference_peak = 1.0
    agent._bc_reference_export = 1.0
    observations = [
        np.zeros(len(_observation_names(1)), dtype=np.float32),
        np.zeros(len(_observation_names(2)), dtype=np.float32),
    ]

    agent.predict(observations, deterministic=False)
    assert agent._bc_pretrain_done is True
    assert agent._cached_policy_sample is False
    agent.update(
        observations,
        [[], []],
        [-1.0, -1.0],
        observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=True,
        initial_exploration_done=True,
    )

    assert agent.rollout_buffer._ptr == 0

    agent.predict(observations, deterministic=False)
    assert agent._cached_policy_sample is True
    agent.update(
        observations,
        [[], []],
        [-1.0, -1.0],
        observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=1,
        update_step=True,
        initial_exploration_done=True,
    )

    assert agent.rollout_buffer._ptr == 1


def test_cc_level2_bc_pretraining_is_incremental_and_restores_threads() -> None:
    agent = _attached_agent(interval=1, num_steps=2)
    agent._bc_enabled = True
    agent._bc_pretrain_done = False
    agent._bc_collect_steps = 1
    agent._bc_train_steps = 5
    agent._bc_train_chunk_steps = 2
    agent._bc_progress_interval = 2
    agent._bc_max_torch_threads = 1
    agent._bc_target_import = 1.0
    agent._bc_reference_peak = 1.0
    agent._bc_reference_export = 1.0
    observations = [
        np.zeros(len(_observation_names(1)), dtype=np.float32),
        np.zeros(len(_observation_names(2)), dtype=np.float32),
    ]
    original_threads = torch.get_num_threads()

    agent.predict(observations, deterministic=False)

    assert agent._bc_pretrain_done is False
    assert agent._bc_train_step == 2
    assert agent._bc_train_inputs is not None
    assert agent._bc_contexts == []
    assert torch.get_num_threads() == original_threads

    agent.predict(observations, deterministic=False)
    assert agent._bc_pretrain_done is False
    assert agent._bc_train_step == 4
    assert torch.get_num_threads() == original_threads

    agent.predict(observations, deterministic=False)
    assert agent._bc_pretrain_done is True
    assert agent._bc_train_step == 5
    assert agent._bc_train_inputs is None
    assert agent._bc_train_optimizer is None
    assert torch.get_num_threads() == original_threads


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
    assert metadata["policy_parameterization"] == "absolute_blend"
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
