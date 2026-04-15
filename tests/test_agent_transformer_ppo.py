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
