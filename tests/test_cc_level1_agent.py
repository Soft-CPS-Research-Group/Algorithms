from __future__ import annotations

import csv

import numpy as np
import torch

from algorithms.agents.cc_level1_agent import (
    CCLevel1Agent,
    CommunityMarketMakerNet,
    DeterministicMultiplierPolicy,
    RolloutBuffer,
    _CC_LEVEL1_FEATURES,
)
from utils.config_schema import CCLevel1Hyperparameters


def test_cc_level1_schema_width_matches_runtime_feature_contract() -> None:
    assert CCLevel1Hyperparameters().c_dim == len(_CC_LEVEL1_FEATURES)


def test_rollout_gae_stops_at_episode_boundary() -> None:
    buffer = RolloutBuffer(num_steps=3, c_dim=1)
    buffer.values[:] = 0.0
    buffer.rewards[:] = [1.0, 2.0, 3.0]
    buffer.dones[:] = [0.0, 1.0, 0.0]

    buffer.compute_gae(last_value=0.0, last_done=False, gamma=1.0, gae_lambda=1.0)

    np.testing.assert_allclose(buffer.returns, [3.0, 2.0, 3.0])


def test_deterministic_multiplier_policy_includes_price_mapping() -> None:
    policy = CommunityMarketMakerNet(c_dim=2, hidden_dims=[4])
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    inference = DeterministicMultiplierPolicy(policy, price_min=0.8, price_max=1.2)

    output = inference(torch.zeros((3, 2), dtype=torch.float32))

    torch.testing.assert_close(output, torch.ones(3))


def test_deterministic_multiplier_policy_scales_residual_from_reference() -> None:
    policy = CommunityMarketMakerNet(c_dim=2, hidden_dims=[4])
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.mean_head.bias.fill_(float(np.arctanh(0.5)))
    inference = DeterministicMultiplierPolicy(
        policy,
        price_min=0.8,
        price_max=1.2,
        reference_multiplier=0.95,
        policy_residual_scale=0.25,
    )

    output = inference(torch.zeros((1, 2), dtype=torch.float32))

    # Full policy output is 1.10; retain only 25% of its deviation from 0.95.
    torch.testing.assert_close(output, torch.tensor([0.9875]))


def test_cc_level1_zero_residual_scale_is_exact_reference() -> None:
    agent = CCLevel1Agent(
        {
            "algorithm": {
                "hyperparameters": {
                    "c_dim": len(_CC_LEVEL1_FEATURES),
                    "hidden_dims": [8],
                    "price_min": 0.8,
                    "price_max": 1.2,
                    "reference_multiplier": 0.975,
                    "policy_residual_scale": 0.0,
                    "bc_pretrain_enabled": False,
                }
            }
        }
    )
    agent.attach_environment(
        observation_names=[list(_CC_LEVEL1_FEATURES)],
        action_names=[[]],
        action_space=[None],
        observation_space=[None],
        metadata={},
    )

    output = agent.predict(
        [np.zeros(len(_CC_LEVEL1_FEATURES), dtype=np.float32)],
        deterministic=False,
    )

    assert output == 0.975


def test_cc_level1_deterministic_policy_starts_at_non_midpoint_reference() -> None:
    agent = CCLevel1Agent(
        {
            "algorithm": {
                "hyperparameters": {
                    "c_dim": len(_CC_LEVEL1_FEATURES),
                    "hidden_dims": [8],
                    "price_min": 0.85,
                    "price_max": 1.15,
                    "reference_multiplier": 1.025,
                    "policy_residual_scale": 0.05,
                    "bc_pretrain_enabled": False,
                }
            }
        }
    )
    agent.attach_environment(
        observation_names=[list(_CC_LEVEL1_FEATURES)],
        action_names=[[]],
        action_space=[None],
        observation_space=[None],
        metadata={},
    )

    output = agent.predict(
        [np.zeros(len(_CC_LEVEL1_FEATURES), dtype=np.float32)],
        deterministic=True,
    )

    assert np.isclose(output, 1.025)


def test_market_maker_honors_initial_log_standard_deviation() -> None:
    policy = CommunityMarketMakerNet(c_dim=2, hidden_dims=[4], initial_log_std=-1.5)

    torch.testing.assert_close(policy.log_std, torch.tensor([-1.5]))


def test_cc_export_persists_complete_multiplier_trace(tmp_path) -> None:
    agent = CCLevel1Agent(
        {
            "algorithm": {
                "hyperparameters": {
                    "c_dim": 17,
                    "hidden_dims": [8],
                    "price_min": 0.8,
                    "price_max": 1.2,
                    "bc_pretrain_enabled": False,
                }
            }
        }
    )
    agent._decision_trace = [
        {
            "timestep": 0,
            "cc_step": 0,
            "price": 0.2,
            "multiplier": 1.0,
            "value_est": 0.0,
            "import_norm": 0.3,
            "pv_norm": 0.1,
            "carbon_norm": 0.2,
        }
    ]

    metadata = agent.export_artifacts(str(tmp_path))

    assert metadata["output_contract"] == "deterministic_global_price_multiplier"
    assert metadata["price_min"] == 0.8
    assert metadata["price_max"] == 1.2
    assert (tmp_path / "onnx_models" / "cc_market_maker.onnx").is_file()
    trace_path = tmp_path / "decision_trace.csv"
    assert trace_path.is_file()
    with trace_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["episode"] == "1"
    assert rows[0]["multiplier"] == "1.0"


def _attached_cc(*, interval: int = 4, num_steps: int = 8) -> CCLevel1Agent:
    agent = CCLevel1Agent(
        {
            "algorithm": {
                "hyperparameters": {
                    "c_dim": len(_CC_LEVEL1_FEATURES),
                    "hidden_dims": [8],
                    "cc_action_interval": interval,
                    "num_steps": num_steps,
                    "bc_pretrain_enabled": False,
                }
            }
        }
    )
    agent.attach_environment(
        observation_names=[list(_CC_LEVEL1_FEATURES)],
        action_names=[[]],
        action_space=[None],
        observation_space=[None],
        metadata={},
    )
    return agent


def test_cc_temporal_abstraction_does_not_depend_on_learning_updates() -> None:
    agent = _attached_cc(interval=4)
    observations = [np.zeros(len(_CC_LEVEL1_FEATURES), dtype=np.float32)]

    outputs = [agent.predict(observations, deterministic=True) for _ in range(8)]

    assert len(agent._decision_trace) == 2
    assert outputs[:4] == [outputs[0]] * 4
    assert outputs[4:] == [outputs[4]] * 4


def test_cc_training_flushes_one_transition_per_decision_interval() -> None:
    agent = _attached_cc(interval=4)
    observations = [np.zeros(len(_CC_LEVEL1_FEATURES), dtype=np.float32)]

    for step in range(4):
        agent.set_episode_context(episode_step=step)
        agent.predict(observations, deterministic=False)
        agent.update(
            observations,
            [[]],
            [-1.0],
            observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=step,
            update_step=True,
            initial_exploration_done=True,
        )

    assert agent.rollout_buffer._ptr == 1
    assert len(agent._decision_trace) == 1


def test_cc_bc_teacher_uses_raw_physical_values_not_policy_encoding() -> None:
    agent = CCLevel1Agent(
        {
            "algorithm": {
                "hyperparameters": {
                    "c_dim": len(_CC_LEVEL1_FEATURES),
                    "hidden_dims": [8],
                    "cc_action_interval": 4,
                    "bc_pretrain_enabled": True,
                    "bc_collect_steps": 2,
                }
            }
        }
    )
    names = list(_CC_LEVEL1_FEATURES) + ["charging_constraint_violation_kwh"]
    agent.attach_environment(
        observation_names=[list(_CC_LEVEL1_FEATURES)],
        action_names=[[]],
        action_space=[None],
        observation_space=[None],
        metadata={"raw_observation_names": [names]},
    )
    price_index = _CC_LEVEL1_FEATURES.index("district__electricity_pricing")
    import_index = _CC_LEVEL1_FEATURES.index("district__community_import_power_kw")
    encoded = np.zeros(len(_CC_LEVEL1_FEATURES), dtype=np.float32)
    encoded[price_index] = 0.75
    encoded[import_index] = 0.25
    raw = np.zeros(len(names), dtype=np.float64)
    raw[price_index] = 0.20
    raw[import_index] = 8.0
    raw[-1] = 0.5

    agent.set_observation_context(raw_observations=[raw])
    agent.predict([encoded], deterministic=False)

    assert agent._bc_contexts[0][price_index] == np.float32(0.75)
    assert agent._bc_teacher_contexts[0][price_index] == np.float32(0.20)
    assert agent._bc_price_samples == [0.20000000298023224]
    assert agent._bc_import_samples == [2.0]
    assert agent._bc_violation_samples == [0.5]


def test_cc_does_not_mix_final_bc_teacher_action_into_ppo_rollout() -> None:
    agent = CCLevel1Agent(
        {
            "algorithm": {
                "hyperparameters": {
                    "c_dim": len(_CC_LEVEL1_FEATURES),
                    "hidden_dims": [8],
                    "cc_action_interval": 1,
                    "num_steps": 8,
                    "bc_pretrain_enabled": True,
                    "bc_collect_steps": 1,
                    "bc_train_steps": 1,
                }
            }
        }
    )
    agent.attach_environment(
        observation_names=[list(_CC_LEVEL1_FEATURES)],
        action_names=[[]],
        action_space=[None],
        observation_space=[None],
        metadata={},
    )
    observations = [np.zeros(len(_CC_LEVEL1_FEATURES), dtype=np.float32)]

    agent.predict(observations, deterministic=False)
    agent.update(
        observations,
        [[]],
        [-1.0],
        observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=True,
        initial_exploration_done=True,
    )

    assert agent._bc_pretrain_done is True
    assert agent.rollout_buffer._ptr == 0
