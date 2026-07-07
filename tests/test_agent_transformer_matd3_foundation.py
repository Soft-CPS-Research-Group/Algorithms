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
