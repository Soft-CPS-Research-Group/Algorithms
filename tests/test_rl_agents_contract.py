from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.agents.maddpg_agent import MADDPG
from algorithms.agents.matd3_agent import MATD3
from algorithms.agents.masac_agent import MASAC
from algorithms.agents.ppo_agents import HAPPO, IPPO, MAPPO
from algorithms.utils.networks import LateFusionCritic
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
                "class": "MultiAgentReplayBuffer" if name in {"MATD3", "MASAC"} else "OnPolicyRolloutBuffer",
                "capacity": 8,
                "batch_size": 2,
            },
            "exploration": {
                "strategy": "SAC" if name == "MASAC" else ("GaussianNoise" if name == "MATD3" else "PPO"),
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
                    "entropy_alpha": 0.2,
                    "automatic_entropy_tuning": name == "MASAC",
                    "alpha_lr": 1.0e-3,
                    "agent_update_order": "random" if name == "HAPPO" else "fixed",
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


@pytest.mark.parametrize("algorithm_name", ["MATD3", "MASAC", "IPPO", "MAPPO", "HAPPO"])
def test_registry_supports_new_rl_agents(algorithm_name):
    assert is_algorithm_supported(algorithm_name)
    agent = create_agent(_base_rl_config(algorithm_name))
    assert agent.__class__.__name__ == algorithm_name


@pytest.mark.parametrize("agent_cls", [MADDPG, MATD3, MASAC])
def test_centralized_critic_respects_configured_late_fusion_class(agent_cls):
    config = _base_rl_config(agent_cls.__name__)
    config["algorithm"]["replay_buffer"]["class"] = "MultiAgentReplayBuffer"
    config["algorithm"]["networks"]["critic"].update(
        {
            "class": "LateFusionCritic",
            "state_layers": [16],
            "action_layers": [8],
            "joint_layers": [16],
        }
    )

    agent = agent_cls(config)

    assert isinstance(agent.critics[0], LateFusionCritic)


@pytest.mark.parametrize("algorithm_name", ["MATD3", "MASAC", "IPPO", "MAPPO", "HAPPO"])
def test_config_schema_accepts_new_rl_templates(algorithm_name):
    config_path = {
        "MATD3": Path("configs/templates/rl/matd3_local.yaml"),
        "MASAC": Path("configs/templates/rl/masac_local.yaml"),
        "IPPO": Path("configs/templates/rl/ippo_local.yaml"),
        "MAPPO": Path("configs/templates/rl/mappo_local.yaml"),
        "HAPPO": Path("configs/templates/rl/happo_local.yaml"),
    }[algorithm_name]
    with config_path.open("r", encoding="utf-8") as handle:
        validate_config(yaml.safe_load(handle))


@pytest.mark.parametrize("agent_cls", [MATD3, MASAC, IPPO, MAPPO, HAPPO])
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


@pytest.mark.parametrize("agent_cls", [MASAC, IPPO, MAPPO, HAPPO])
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


def test_ppo_warm_start_policy_and_behavior_cloning_contract():
    config = _base_rl_config("MAPPO")
    params = config["algorithm"]["exploration"]["params"]
    params.update(
        {
            "initial_exploration_strategy": "policy",
            "warm_start_policy": "RandomPolicy",
            "warm_start_policy_deterministic": True,
            "warm_start_policy_phaseout_steps": 100,
            "warm_start_policy_phaseout_mode": "blend",
            "actor_behavior_cloning_weight": 0.5,
            "actor_behavior_cloning_min_weight": 0.5,
            "actor_behavior_cloning_decay_steps": 0,
            "actor_behavior_cloning_extra_updates": 1,
        }
    )
    agent = MAPPO(config)
    _attach_bounds(agent)

    for step in range(2):
        obs, next_obs, rewards = _transition(step)
        agent.set_observation_context(raw_observations=obs, encoded_observations=obs)
        actions = agent.predict(obs, deterministic=False)
        assert len(actions) == 2
        assert agent.get_diagnostic_metrics()["MAPPO/warm_start_policy_enabled"] == 1.0
        assert agent.get_diagnostic_metrics()["MAPPO/warm_start_policy_phaseout_used"] == 1.0

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

    metrics = agent.consume_latest_training_metrics()
    assert metrics["MAPPO/behavior_cloning_effective_weight"] == pytest.approx(0.5)
    assert "MAPPO/behavior_cloning_loss_mean" in metrics
    assert metrics["MAPPO/behavior_cloning_extra_updates"] == pytest.approx(1.0)


def test_masac_behavior_cloning_and_action_regularization_metrics():
    config = _base_rl_config("MASAC")
    params = config["algorithm"]["exploration"]["params"]
    params.update(
        {
            "automatic_entropy_tuning": False,
            "actor_behavior_cloning_weight": 0.25,
            "actor_behavior_cloning_min_weight": 0.25,
            "actor_storage_action_l2_penalty": 0.1,
        }
    )
    agent = MASAC(config)
    _attach_bounds(agent)

    for step in range(2):
        obs, next_obs, rewards = _transition(step)
        actions = agent.predict(obs, deterministic=False)
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

    metrics = agent.consume_latest_training_metrics()
    assert metrics["MASAC/actor_behavior_cloning_effective_weight"] == pytest.approx(0.25)
    assert "MASAC/actor_behavior_cloning_loss_mean" in metrics
    assert "MASAC/actor_regularization_mean" in metrics
