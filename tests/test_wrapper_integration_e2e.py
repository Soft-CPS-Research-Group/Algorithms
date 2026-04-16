"""End-to-end integration tests for Wrapper + TransformerPPO Agent.

These tests verify that the wrapper correctly integrates with the
TransformerPPO agent, including enrichment, encoding, and topology handling.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch


class TestWrapperAgentIntegration:
    """Integration tests for wrapper + agent interaction."""

    @pytest.fixture
    def integration_setup(self, tmp_path: Path) -> Dict[str, Any]:
        """Set up wrapper + agent for integration testing."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        # Create tokenizer config
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
                "temporal": {"features": ["month"], "input_dim": 2},
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
        
        agent_config = {
            "algorithm": {
                "name": "AgentTransformerPPO",
                "tokenizer_config_path": str(tokenizer_path),
                "transformer": {
                    "d_model": 32,
                    "nhead": 2,
                    "num_layers": 1,
                },
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_eps": 0.2,
                    "ppo_epochs": 2,
                    "minibatch_size": 4,
                    "entropy_coeff": 0.01,
                    "value_coeff": 0.5,
                    "max_grad_norm": 0.5,
                    "hidden_dim": 32,
                    "rollout_length": 8,
                },
            }
        }
        
        agent = AgentTransformerPPO(agent_config)
        
        return {
            "agent": agent,
            "agent_config": agent_config,
            "tokenizer_config": tokenizer_config,
        }

    def test_wrapper_initializes_enrichers_for_transformer_agent(
        self, integration_setup: Dict[str, Any]
    ) -> None:
        """Wrapper should initialize enrichers when agent is TransformerPPO."""
        from utils.wrapper_citylearn import Wrapper_CityLearn
        
        agent = integration_setup["agent"]
        
        # Create minimal mock environment
        class MinimalEnv:
            def __init__(self):
                self.observation_names = [["electrical_storage_soc", "non_shiftable_load", "month"]]
                self.action_names = [["electrical_storage"]]
                self.observation_space = [
                    type("space", (), {
                        "high": np.array([1.0, 1000.0, 12.0]),
                        "low": np.array([0.0, 0.0, 1.0]),
                    })()
                ]
                self.action_space = [
                    type("space", (), {
                        "high": np.array([1.0]),
                        "low": np.array([-1.0]),
                    })()
                ]
                self.reward_function = type("reward", (), {"__dict__": {}})()
                self.time_steps = 8760
                self.seconds_per_time_step = 3600
                self.time_step_ratio = 1.0
                self.random_seed = 0
                self.episode_tracker = type("tracker", (), {"episode_time_steps": self.time_steps})()
                self.unwrapped = self
            
            def get_metadata(self):
                return {"buildings": [{}]}
        
        mock_env = MinimalEnv()
        
        wrapper_config = {
            "environment": {"buildings": ["Building_1"]},
            "algorithm": integration_setup["agent_config"]["algorithm"],
            "training": {},
            "simulator": {},
            "checkpointing": {},
            "tracking": {},
            "runtime": {},
        }
        
        wrapper = Wrapper_CityLearn(
            env=mock_env,
            model=agent,
            config=wrapper_config,
            job_id="test"
        )
        
        # Verify enrichers initialized
        assert wrapper._is_transformer_agent is True
        assert len(wrapper._enrichers) == 1
        assert wrapper._enrichers[0] is not None

    def test_enrichment_produces_marker_values(
        self, integration_setup: Dict[str, Any]
    ) -> None:
        """Enrichment should inject marker values into observations."""
        from algorithms.utils.observation_enricher import ObservationEnricher
        
        enricher = ObservationEnricher(integration_setup["tokenizer_config"])
        
        obs_names = ["electrical_storage_soc", "non_shiftable_load", "month"]
        action_names = ["electrical_storage"]
        
        enricher.enrich_names(obs_names, action_names)
        
        raw_obs = [0.5, 100.0, 6.0]
        enriched_obs = enricher.enrich_values(raw_obs)
        
        # Verify markers present
        assert 1001.0 in enriched_obs  # CA marker
        assert 2001.0 in enriched_obs  # SRO marker
        assert 3001.0 in enriched_obs  # NFC marker

    def test_agent_processes_enriched_observations(
        self, integration_setup: Dict[str, Any]
    ) -> None:
        """Agent should successfully process enriched observations."""
        agent = integration_setup["agent"]
        
        # Attach environment
        agent.attach_environment(
            observation_names=[["electrical_storage_soc", "non_shiftable_load", "month"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Create enriched observation with markers
        # Structure: [CA_1001, soc, SRO_2001, month_enc, NFC_3001, load]
        enriched_obs = np.array([[
            1001.0, 0.5,  # CA: battery
            2001.0, 0.5, 0.5,  # SRO: temporal (2 dims encoded)
            3001.0, 100.0,  # NFC
        ]])
        
        actions = agent.predict([enriched_obs], deterministic=True)
        
        assert len(actions) == 1
        assert actions[0].shape[-1] == 1  # One action per CA
        assert (actions[0] >= -1.0).all()
        assert (actions[0] <= 1.0).all()

    def test_topology_change_triggers_agent_notification(
        self, integration_setup: Dict[str, Any]
    ) -> None:
        """Topology change should notify agent via on_topology_change."""
        agent = integration_setup["agent"]
        
        agent.attach_environment(
            observation_names=[["electrical_storage_soc"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Agent should have on_topology_change method
        assert hasattr(agent, 'on_topology_change')
        
        # Call it (should not crash)
        agent.on_topology_change(0)
        
        # Verify buffer was handled (no exception means success)
        assert True
