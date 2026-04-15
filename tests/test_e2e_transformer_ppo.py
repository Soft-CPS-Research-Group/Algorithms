"""End-to-end tests for TransformerPPO agent.

These tests verify the complete training pipeline works correctly,
from environment setup through training to artifact export.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock


class TestE2ESingleBuilding:
    """E2E tests with single building configuration."""

    @pytest.fixture
    def full_config(self, tmp_path: Path) -> Dict[str, Any]:
        """Create full experiment config."""
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
                "temporal": {"features": ["month", "hour"], "input_dim": 4},
                "pricing": {"features": ["electricity_pricing"], "input_dim": 1},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": ["solar_generation"],
                "extra_features": [],
                "input_dim": 2,
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
                    "d_model": 32,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 64,
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

    def test_training_loop_runs(self, full_config: Dict[str, Any]) -> None:
        """Complete training loop should run without errors."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(full_config)
        
        # Attach environment
        agent.attach_environment(
            observation_names=[["month", "hour", "electricity_pricing",
                               "electrical_storage_soc", "non_shiftable_load",
                               "solar_generation"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Simulate training loop
        for step in range(20):
            # Create encoded observation with markers
            # Structure: [CA_1001, soc, SRO_2001, month, hour, d1, d2, SRO_2002, price, NFC_3001, load, gen]
            obs = np.array([[
                1001.0, 0.5 + np.random.randn() * 0.1,  # CA: battery
                2001.0, 0.5, 0.5, 0.5, 0.5,  # SRO: temporal (4 dims)
                2002.0, 0.7,  # SRO: pricing (1 dim)
                3001.0, 100.0, 50.0,  # NFC (2 dims)
            ]])
            
            # Predict
            actions = agent.predict([obs], deterministic=False)
            
            # Update
            reward = float(np.random.randn())
            agent.update(
                observations=[obs],
                actions=actions,
                rewards=[reward],
                next_observations=[obs],
                terminated=[False],
                truncated=[False],
                update_target_step=False,
                global_learning_step=step,
                update_step=(step % 8 == 7),  # Update every 8 steps
                initial_exploration_done=True,
            )
        
        # Should complete without error

    def test_actions_valid_range(self, full_config: Dict[str, Any]) -> None:
        """All actions should be in valid range [-1, 1]."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(full_config)
        agent.attach_environment(
            observation_names=[["month", "electrical_storage_soc", "non_shiftable_load"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Multiple predictions
        for _ in range(50):
            obs = np.array([[
                1001.0, np.random.rand(),
                2001.0, 0.5, 0.5, 0.5, 0.5,
                3001.0, 100.0, 50.0,
            ]])
            
            actions = agent.predict([obs], deterministic=False)
            
            assert (actions[0] >= -1.0).all(), "Action below -1"
            assert (actions[0] <= 1.0).all(), "Action above 1"
            assert not np.isnan(actions[0]).any(), "NaN in actions"

    def test_kpis_generated(self, full_config: Dict[str, Any], tmp_path: Path) -> None:
        """Training should produce exportable artifacts."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(full_config)
        agent.attach_environment(
            observation_names=[["month", "electrical_storage_soc", "non_shiftable_load"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Brief training
        for step in range(10):
            obs = np.array([[
                1001.0, 0.5,
                2001.0, 0.5, 0.5, 0.5, 0.5,
                3001.0, 100.0, 50.0,
            ]])
            actions = agent.predict([obs], deterministic=False)
            agent.update(
                observations=[obs],
                actions=actions,
                rewards=[1.0],
                next_observations=[obs],
                terminated=[False],
                truncated=[False],
                update_target_step=False,
                global_learning_step=step,
                update_step=(step == 9),
                initial_exploration_done=True,
            )
        
        # Export
        output_dir = tmp_path / "artifacts"
        manifest = agent.export_artifacts(output_dir)
        
        assert manifest is not None
        assert Path(manifest["model_path"]).exists()


class TestE2EVariableTopology:
    """E2E tests for variable topology support."""

    @pytest.fixture
    def config_with_ev(self, tmp_path: Path) -> Dict[str, Any]:
        """Create config with battery and EV charger."""
        tokenizer_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                },
                "ev_charger": {
                    "features": ["electric_vehicle_soc"],
                    "action_name": "electric_vehicle_storage",
                    "input_dim": 2,
                },
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
        
        return {
            "algorithm": {
                "name": "AgentTransformerPPO",
                "tokenizer_config_path": str(tokenizer_path),
                "transformer": {"d_model": 32, "nhead": 2, "num_layers": 1},
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "ppo_epochs": 1,
                    "minibatch_size": 2,
                    "hidden_dim": 32,
                    "rollout_length": 4,
                },
            }
        }

    def test_variable_ca_runtime(self, config_with_ev: Dict[str, Any]) -> None:
        """Same model should handle different CA counts."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(config_with_ev)
        agent.attach_environment(
            observation_names=[["month", "electrical_storage_soc", "non_shiftable_load"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Test with 1 CA (battery only)
        obs_1ca = np.array([[
            1001.0, 0.5,  # CA: battery (1 feature)
            2001.0, 0.5, 0.5,  # SRO: temporal (2 features)
            3001.0, 100.0,  # NFC (1 feature)
        ]])
        
        actions_1ca = agent.predict([obs_1ca], deterministic=False)
        assert actions_1ca[0].shape[-1] == 1  # 1 action for 1 CA
        
        # Test with 2 CAs (battery + EV)
        obs_2ca = np.array([[
            1001.0, 0.5,  # CA: battery (1 feature)
            1002.0, 0.8, 0.9,  # CA: ev_charger (2 features)
            2001.0, 0.5, 0.5,  # SRO: temporal
            3001.0, 100.0,  # NFC
        ]])
        
        actions_2ca = agent.predict([obs_2ca], deterministic=False)
        assert actions_2ca[0].shape[0] == 2  # 2 actions for 2 CAs
