"""Tests for AgentTransformerPPO."""

import json
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock


class TestAgentInstantiation:
    """Tests for AgentTransformerPPO instantiation."""

    @pytest.fixture
    def sample_config(self, tmp_path: Path) -> Dict[str, Any]:
        """Create sample config with tokenizer file."""
        # Create tokenizer config file
        tokenizer_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                }
            },
            "sro_types": {
                "temporal": {"features": ["month", "hour"], "input_dim": 4},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }
        tokenizer_path = tmp_path / "tokenizer.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f)
        
        return {
            "algorithm": {
                "name": "AgentTransformerPPO",
                "tokenizer_config_path": str(tokenizer_path),
                "transformer": {
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                    "dim_feedforward": 128,
                    "dropout": 0.1,
                },
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_eps": 0.2,
                    "ppo_epochs": 4,
                    "minibatch_size": 64,
                    "entropy_coeff": 0.01,
                    "value_coeff": 0.5,
                    "max_grad_norm": 0.5,
                    "hidden_dim": 128,
                    "rollout_length": 2048,
                },
            }
        }

    def test_agent_creation(self, sample_config: Dict[str, Any]) -> None:
        """Agent should instantiate with valid config."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        assert agent is not None
        assert agent.is_transformer_agent is True

    def test_agent_has_tokenizer(self, sample_config: Dict[str, Any]) -> None:
        """Agent should have tokenizer after creation."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        assert agent.tokenizer is not None

    def test_agent_has_backbone(self, sample_config: Dict[str, Any]) -> None:
        """Agent should have Transformer backbone."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        assert agent.backbone is not None

    def test_agent_has_actor_critic(self, sample_config: Dict[str, Any]) -> None:
        """Agent should have actor and critic heads."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        assert agent.actor is not None
        assert agent.critic is not None


class TestAgentPredict:
    """Tests for AgentTransformerPPO.predict()."""

    @pytest.fixture
    def agent_with_env(self, sample_config: Dict[str, Any]) -> Any:
        """Create agent and attach mock environment."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        # Attach environment
        agent.attach_environment(
            observation_names=[["month", "hour", "electrical_storage_soc", "non_shiftable_load"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        return agent

    @pytest.fixture
    def sample_config(self, tmp_path: Path) -> Dict[str, Any]:
        """Create sample config with tokenizer file."""
        tokenizer_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                }
            },
            "sro_types": {
                "temporal": {"features": ["month", "hour"], "input_dim": 4},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }
        tokenizer_path = tmp_path / "tokenizer.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f)
        
        return {
            "algorithm": {
                "name": "AgentTransformerPPO",
                "tokenizer_config_path": str(tokenizer_path),
                "transformer": {
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                    "dim_feedforward": 128,
                    "dropout": 0.1,
                },
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_eps": 0.2,
                    "ppo_epochs": 4,
                    "minibatch_size": 64,
                    "entropy_coeff": 0.01,
                    "value_coeff": 0.5,
                    "max_grad_norm": 0.5,
                    "hidden_dim": 128,
                    "rollout_length": 2048,
                },
            }
        }

    def test_predict_returns_actions(self, agent_with_env: Any) -> None:
        """predict() should return actions for each building."""
        # Create encoded observation with markers
        # [CA_1001, soc, SRO_2001, month, hour, d1, d2, NFC_3001, load]
        encoded_obs = np.array([[
            1001.0, 0.5,  # CA: battery
            2001.0, 0.5, 0.5, 0.5, 0.5,  # SRO: temporal (4 features)
            3001.0, 100.0,  # NFC (1 feature)
        ]])
        
        actions = agent_with_env.predict([encoded_obs], deterministic=False)
        
        assert len(actions) == 1  # One building
        assert actions[0].shape[-1] == 1  # One action per CA

    def test_predict_deterministic(self, agent_with_env: Any) -> None:
        """Deterministic predict should return same actions."""
        encoded_obs = np.array([[
            1001.0, 0.5,
            2001.0, 0.5, 0.5, 0.5, 0.5,
            3001.0, 100.0,
        ]])
        
        actions1 = agent_with_env.predict([encoded_obs], deterministic=True)
        actions2 = agent_with_env.predict([encoded_obs], deterministic=True)
        
        np.testing.assert_array_almost_equal(actions1[0], actions2[0])

    def test_predict_action_range(self, agent_with_env: Any) -> None:
        """Actions should be in [-1, 1] range."""
        encoded_obs = np.array([[
            1001.0, 0.5,
            2001.0, 0.5, 0.5, 0.5, 0.5,
            3001.0, 100.0,
        ]])
        
        # Multiple predictions to check range
        for _ in range(10):
            actions = agent_with_env.predict([encoded_obs], deterministic=False)
            assert (actions[0] >= -1.0).all()
            assert (actions[0] <= 1.0).all()
