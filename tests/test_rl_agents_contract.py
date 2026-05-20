from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.agents.matd3_agent import MATD3
from algorithms.agents.ppo_agents import IPPO, MAPPO
from algorithms.registry import create_agent, is_algorithm_supported
from utils.config_schema import validate_config


class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


def _base_rl_config(name: str) -> dict:
    return {
        "training": {
            "seed": 7,
            "steps_between_training_updates": 1,
            "target_update_interval": 1,
        },
        "tracking": {
            "mlflow_step_sample_interval": 1,
            "training_diagnostics_enabled": True,
        },
        "checkpointing": {
            "checkpoint_artifact": "latest_checkpoint.pth",
            "reset_replay_buffer": False,
            "freeze_pretrained_layers": False,
            "fine_tune": False,
        },
        "topology": {
            "num_agents": 2,
            "observation_dimensions": [3, 2],
            "action_dimensions": [1, 2],
        },
        "algorithm": {
            "name": name,
            "hyperparameters": {
                "gamma": 0.95,
                "require_cuda": False,
            },
            "networks": {
                "actor": {
                    "class": "Actor",
                    "layers": [16],
                    "lr": 1.0e-3,
                },
                "critic": {
                    "class": "Critic",
                    "layers": [16],
                    "lr": 1.0e-3,
                },
            },
            "replay_buffer": {
                "class": "MultiAgentReplayBuffer" if name == "MATD3" else "OnPolicyRolloutBuffer",
                "capacity": 8,
                "batch_size": 2,
            },
            "exploration": {
                "strategy": "GaussianNoise" if name == "MATD3" else "PPO",
                "params": {
                    "gamma": 0.95,
                    "tau": 0.01,
                    "sigma": 0.1,
                    "decay": 1.0,
                    "min_sigma": 0.0,
                    "bias": 0.0,
                    "use_amp": False,
                    "end_initial_exploration_time_step": 0,
                    "random_exploration_steps": 0,
                    "initial_exploration_strategy": "uniform_full_range",
                    "train_during_initial_exploration": False,
                    "critic_update_mode": "per_agent",
                    "actor_update_interval": 2 if name == "MATD3" else 1,
                    "target_policy_smoothing": name == "MATD3",
                    "target_policy_noise": 0.01,
                    "target_policy_noise_clip": 0.05,
                    "reward_normalization": False,
                    "rollout_length": 2,
                    "minibatch_size": 2,
                    "ppo_epochs": 1,
                    "gae_lambda": 0.95,
                    "clip_ratio": 0.2,
                    "entropy_coef": 0.01,
                    "value_loss_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "initial_log_std": -0.5,
                },
            },
        },
    }


def _attach_bounds(agent) -> None:
    agent.attach_environment(
        observation_names=[["o1", "o2", "o3"], ["o1", "o2"]],
        action_names=[["battery"], ["charger", "deferrable_appliance_1"]],
        action_space=[
            _Box(low=[-2.0], high=[2.0]),
            _Box(low=[0.0, 0.0], high=[7.4, 1.0]),
        ],
        observation_space=[],
        metadata={"seconds_per_time_step": 3600},
    )


def _transition(step: int):
    obs = [
        np.asarray([0.1 + step, 0.2, 0.3], dtype=np.float32),
        np.asarray([0.4, 0.5 + step], dtype=np.float32),
    ]
    next_obs = [
        np.asarray([0.2 + step, 0.2, 0.3], dtype=np.float32),
        np.asarray([0.4, 0.6 + step], dtype=np.float32),
    ]
    rewards = [-1.0 + 0.1 * step, -0.5 + 0.1 * step]
    return obs, next_obs, rewards


@pytest.mark.parametrize("algorithm_name", ["MATD3", "IPPO", "MAPPO"])
def test_registry_supports_new_rl_agents(algorithm_name):
    assert is_algorithm_supported(algorithm_name)
    agent = create_agent(_base_rl_config(algorithm_name))
    assert agent.__class__.__name__ == algorithm_name


@pytest.mark.parametrize("algorithm_name", ["MATD3", "IPPO", "MAPPO"])
def test_config_schema_accepts_new_rl_templates(algorithm_name):
    config_path = {
        "MATD3": Path("configs/templates/rl/matd3_local.yaml"),
        "IPPO": Path("configs/templates/rl/ippo_local.yaml"),
        "MAPPO": Path("configs/templates/rl/mappo_local.yaml"),
    }[algorithm_name]
    with config_path.open("r", encoding="utf-8") as handle:
        validate_config(yaml.safe_load(handle))


@pytest.mark.parametrize("agent_cls", [MATD3, IPPO, MAPPO])
def test_rl_agent_predict_update_and_checkpoint_contract(agent_cls, tmp_path):
    agent = agent_cls(_base_rl_config(agent_cls.__name__))
    _attach_bounds(agent)

    for step in range(2):
        obs, next_obs, rewards = _transition(step)
        actions = agent.predict(obs, deterministic=False)
        assert len(actions) == 2
        assert -2.0 <= actions[0][0] <= 2.0
        assert 0.0 <= actions[1][0] <= 7.4
        assert 0.0 <= actions[1][1] <= 1.0

        agent.update(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            terminated=False,
            truncated=step == 1,
            update_target_step=True,
            global_learning_step=step + 1,
            update_step=True,
            initial_exploration_done=True,
        )

    checkpoint = agent.save_checkpoint(str(tmp_path / "checkpoints"), step=2)
    assert Path(checkpoint).exists()
    diagnostics = agent.get_diagnostic_metrics()
    assert diagnostics


@pytest.mark.parametrize("agent_cls", [IPPO, MAPPO])
def test_ppo_agents_export_onnx_contract(agent_cls, tmp_path):
    agent = agent_cls(_base_rl_config(agent_cls.__name__))
    _attach_bounds(agent)

    metadata = agent.export_artifacts(
        output_dir=str(tmp_path),
        context={
            "config": {
                "bundle": {
                    "require_observations_envelope": True,
                    "artifact_config": {"input_site_key": "site_a"},
                }
            }
        },
    )

    assert metadata["format"] == "onnx"
    assert len(metadata["artifacts"]) == 2
    for artifact in metadata["artifacts"]:
        assert artifact["format"] == "onnx"
        assert artifact["config"]["require_observations_envelope"] is True
        assert (tmp_path / artifact["path"]).exists()
