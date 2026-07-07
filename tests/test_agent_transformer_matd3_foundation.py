"""Plan A tests for AgentTransformerMATD3 foundation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from utils.config_schema import TransformerMATD3StageConfig


class TestTransformerMATD3StageConfig:
    def test_valid_minimal_config(self):
        cfg = TransformerMATD3StageConfig(
            algorithm="AgentTransformerMATD3",
            tokenizer_config_path="configs/tokenizers/entity_default.json",
            transformer_actor={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1},
            transformer_critic={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1},
            hyperparameters={
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 256,
                "replay_capacity": 100000,
                "actor_lr": 1e-4,
                "critic_lr": 3e-4,
                "target_policy_noise": 0.2,
                "target_policy_noise_clip": 0.5,
                "actor_update_interval": 2,
            },
        )
        assert cfg.algorithm == "AgentTransformerMATD3"
        assert cfg.transformer_actor.d_model == 64
        assert cfg.transformer_critic.d_model == 64

    def test_rejects_wrong_algorithm_name(self):
        with pytest.raises(ValidationError):
            TransformerMATD3StageConfig(
                algorithm="MATD3",
                tokenizer_config_path="configs/tokenizers/entity_default.json",
                transformer_actor={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                transformer_critic={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                hyperparameters={"gamma": 0.99, "tau": 0.005, "batch_size": 256, "replay_capacity": 100000, "actor_lr": 1e-4, "critic_lr": 3e-4, "target_policy_noise": 0.2, "target_policy_noise_clip": 0.5, "actor_update_interval": 2},
            )

    def test_rejects_missing_tokenizer_path(self):
        with pytest.raises(ValidationError):
            TransformerMATD3StageConfig(
                algorithm="AgentTransformerMATD3",
                tokenizer_config_path="",
                transformer_actor={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                transformer_critic={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                hyperparameters={"gamma": 0.99, "tau": 0.005, "batch_size": 256, "replay_capacity": 100000, "actor_lr": 1e-4, "critic_lr": 3e-4, "target_policy_noise": 0.2, "target_policy_noise_clip": 0.5, "actor_update_interval": 2},
            )


import torch

from algorithms.utils.matd3_actor_head import DeterministicActorHead


class TestDeterministicActorHead:
    def test_output_shape(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        ca_emb = torch.randn(2, 3, 16)  # [batch=2, n_ca=3, d_model=16]
        actions = head(ca_emb)
        assert actions.shape == (2, 3, 1)

    def test_output_range_tanh(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        ca_emb = torch.randn(4, 5, 16) * 10.0  # large inputs
        actions = head(ca_emb)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_deterministic_same_output(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        head.eval()
        ca_emb = torch.randn(1, 2, 16)
        a1 = head(ca_emb)
        a2 = head(ca_emb)
        assert torch.allclose(a1, a2)

    def test_pre_tanh_accessor(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        ca_emb = torch.randn(1, 2, 16)
        actions, pre_tanh = head.forward_with_pre_tanh(ca_emb)
        assert torch.allclose(actions, torch.tanh(pre_tanh))


from algorithms.registry import ALGORITHM_REGISTRY


class TestRegistry:
    def test_agent_registered(self):
        assert "AgentTransformerMATD3" in ALGORITHM_REGISTRY

    def test_supports_dynamic_topology(self):
        cls = ALGORITHM_REGISTRY["AgentTransformerMATD3"]
        assert cls.supports_dynamic_topology is True


import numpy as np

from algorithms.agents.agent_transformer_matd3 import AgentTransformerMATD3
from tests._entity_sample_obs_names import load_sample_observation_names_for_first_building


_TOKENIZER_CFG = "configs/tokenizers/entity_default.json"
_DEFAULT_ACTIONS = ["electrical_storage", "electric_vehicle_storage"]


def _matd3_config() -> dict:
    return {
        "algorithm": {
            "name": "AgentTransformerMATD3",
            "tokenizer_config_path": _TOKENIZER_CFG,
            "transformer_actor": {
                "d_model": 16, "nhead": 2, "num_layers": 1,
                "dim_feedforward": 32, "dropout": 0.0,
            },
            "transformer_critic": {
                "d_model": 16, "nhead": 2, "num_layers": 1,
                "dim_feedforward": 32, "dropout": 0.0,
            },
            "hyperparameters": {
                "gamma": 0.99, "tau": 0.005, "batch_size": 4,
                "replay_capacity": 100, "actor_lr": 1e-3, "critic_lr": 3e-4,
                "target_policy_noise": 0.2, "target_policy_noise_clip": 0.5,
                "actor_update_interval": 2, "actor_hidden_dim": 32,
            },
        },
    }


def _make_matd3(n_buildings: int = 1):
    obs_names = load_sample_observation_names_for_first_building()
    obs_per = [list(obs_names) for _ in range(n_buildings)]
    act_per = [list(_DEFAULT_ACTIONS) for _ in range(n_buildings)]
    agent = AgentTransformerMATD3(_matd3_config())
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[None] * n_buildings,
        observation_space=[None] * n_buildings,
        metadata={"building_names": [f"Building_{b}" for b in range(n_buildings)]},
    )
    obs_dim = len(obs_names)
    return agent, obs_per, act_per, obs_dim


class TestAttachAndPredict:
    def test_attach_builds_actors(self):
        agent, _, _, _ = _make_matd3(n_buildings=2)
        assert len(agent._actors) == 2
        for s in agent._actors:
            assert s.layout.n_ca == 2

    def test_attach_noop_on_same_names(self):
        agent, obs_per, act_per, _ = _make_matd3()
        layout_before = agent._actors[0].layout
        agent.attach_environment(
            observation_names=obs_per,
            action_names=act_per,
            action_space=[None],
            observation_space=[None],
        )
        assert agent._actors[0].layout is layout_before

    def test_attach_rejects_building_count_change(self):
        agent, obs_per, act_per, _ = _make_matd3(n_buildings=1)
        with pytest.raises(ValueError, match="building count changed"):
            agent.attach_environment(
                observation_names=obs_per + [obs_per[0]],
                action_names=act_per + [act_per[0]],
                action_space=[None, None],
                observation_space=[None, None],
            )

    def test_predict_returns_correct_shape(self):
        agent, _, act_per, obs_dim = _make_matd3(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        actions = agent.predict(obs, deterministic=True)
        assert len(actions) == 2
        for a, expected_names in zip(actions, act_per):
            assert len(a) == len(expected_names)

    def test_predict_actions_in_range(self):
        agent, _, _, obs_dim = _make_matd3()
        obs = [np.random.randn(obs_dim).astype(np.float64)]
        actions = agent.predict(obs, deterministic=True)
        for val in actions[0]:
            assert -1.0 <= val <= 1.0

    def test_predict_deterministic_reproducible(self):
        agent, _, _, obs_dim = _make_matd3()
        obs = [np.random.randn(obs_dim).astype(np.float64)]
        a1 = agent.predict(obs, deterministic=True)
        a2 = agent.predict(obs, deterministic=True)
        assert a1 == a2
