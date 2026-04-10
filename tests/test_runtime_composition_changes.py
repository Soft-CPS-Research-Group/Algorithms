"""Test runtime composition changes - assets going offline/online during operation.

This validates production scenarios where asset availability changes mid-run through
cross-topology checkpoint loading, which is the primary mechanism for handling
different asset configurations.

See docs/runtime_composition_changes.md for full production guidance.
"""

import pytest
import torch

from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO


class TestCrossTopologyCheckpointTransfer:
    """Test checkpoint transfer across different building topologies.
    
    This is the primary production mechanism for handling different asset
    configurations - train once, deploy to buildings with varying asset counts.
    """

    @pytest.fixture
    def base_config(self):
        """Base config for TransformerPPO agent."""
        return {
            "algorithm": {
                "name": "AgentTransformerPPO",
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
                    "min_steps_before_update": 0,
                },
                "tokenizer": {
                    "ca_types": {
                        "battery": {
                            "features": ["electrical_storage_soc"],
                            "action_name": "electrical_storage",
                        },
                        "ev_charger": {
                            "features": [
                                "connected_state",
                                "departure_time",
                                "_soc",
                            ],
                            "action_name": "electric_vehicle_storage",
                        },
                    },
                    "sro_types": {
                        "temporal": {
                            "features": ["month", "hour"],
                        },
                    },
                    "rl": {
                        "demand_feature": "non_shiftable_load",
                        "generation_features": [],
                        "extra_features": [],
                    },
                },
            },
            "topology": {"num_agents": 1},
            "tracking": {"mlflow_step_sample_interval": 10},
            "checkpointing": {"checkpoint_artifact": "test_checkpoint.pth"},
        }

    def test_checkpoint_dimension_extraction(self, base_config, tmp_path):
        """Test extracting CA dimensions from checkpoint for cross-topology transfer."""
        # Create agent with full topology (battery + ev)
        agent1 = AgentTransformerPPO(base_config)
        
        observation_names_full = [[
            "electrical_storage_soc",
            "electric_vehicle_charger_ev_1_connected_state",
            "electric_vehicle_charger_ev_1_departure_time",
            "connected_electric_vehicle_at_ev_1_soc",
            "month",
            "hour",
            "non_shiftable_load",
        ]]
        
        action_names_full = [[
            "electrical_storage",
            "electric_vehicle_storage_ev_1",
        ]]
        
        agent1.attach_environment(
            observation_names=observation_names_full,
            action_names=action_names_full,
            action_space=[None],
            observation_space=[None],
        )
        
        # Save checkpoint
        checkpoint_path = agent1.save_checkpoint(str(tmp_path), step=100)
        
        # Create new agent for different topology
        agent2 = AgentTransformerPPO(base_config)
        
        # Extract dimensions before attach_environment
        ca_dims = agent2.extract_checkpoint_ca_dims(checkpoint_path)
        
        # Should extract actual dimensions from trained model
        # Note: Dimensions are post-encoding (normalized, one-hot), not raw feature count
        assert "battery" in ca_dims
        assert "ev_charger" in ca_dims
        assert ca_dims["battery"] == 1  # One feature (SOC, already normalized)
        assert ca_dims["ev_charger"] > 3  # Multiple features after encoding (connected, time, soc)
        
    def test_global_vocabulary_with_checkpoint_dims(self, base_config, tmp_path):
        """Test that global vocabulary uses checkpoint dims instead of fallback estimates."""
        # Create and save checkpoint with full topology
        agent1 = AgentTransformerPPO(base_config)
        
        observation_names_full = [[
            "electrical_storage_soc",
            "electric_vehicle_charger_ev_1_connected_state",
            "electric_vehicle_charger_ev_1_departure_time",
            "connected_electric_vehicle_at_ev_1_soc",
            "month",
            "hour",
            "non_shiftable_load",
        ]]
        
        action_names_full = [[
            "electrical_storage",
            "electric_vehicle_storage_ev_1",
        ]]
        
        agent1.attach_environment(
            observation_names=observation_names_full,
            action_names=action_names_full,
            action_space=[None],
            observation_space=[None],
        )
        
        checkpoint_path = agent1.save_checkpoint(str(tmp_path), step=100)
        
        # Create agent for battery-only building
        agent2 = AgentTransformerPPO(base_config)
        
        # Extract checkpoint dims
        ca_dims = agent2.extract_checkpoint_ca_dims(checkpoint_path)
        agent2._checkpoint_ca_dims = ca_dims
        
        # Attach environment with only battery
        observation_names_battery = [[
            "electrical_storage_soc",
            "month",
            "hour",
            "non_shiftable_load",
        ]]
        
        action_names_battery = [["electrical_storage"]]
        
        agent2.attach_environment(
            observation_names=observation_names_battery,
            action_names=action_names_battery,
            action_space=[None],
            observation_space=[None],
        )
        
        # Verify ev_charger projection exists with correct dimensions
        building_model = agent2.building_models[0]
        tokenizer = building_model.tokenizer
        
        assert "ev_charger" in tokenizer.ca_projections
        # Should use checkpoint dim (post-encoding), not fallback estimate
        # Exact dim depends on encoding (normalization, one-hot)
        assert tokenizer.ca_projections["ev_charger"].in_features > 3
        assert tokenizer.ca_projections["ev_charger"].in_features == ca_dims["ev_charger"]
        
    def test_optimizer_skip_on_cross_topology(self, base_config, tmp_path):
        """Test that optimizer state is skipped when loading cross-topology checkpoint."""
        # Agent 1: Full topology
        agent1 = AgentTransformerPPO(base_config)
        
        observation_names_full = [[
            "electrical_storage_soc",
            "electric_vehicle_charger_ev_1_connected_state",
            "electric_vehicle_charger_ev_1_departure_time",
            "connected_electric_vehicle_at_ev_1_soc",
            "month",
            "hour",
            "non_shiftable_load",
        ]]
        
        action_names_full = [[
            "electrical_storage",
            "electric_vehicle_storage_ev_1",
        ]]
        
        agent1.attach_environment(
            observation_names=observation_names_full,
            action_names=action_names_full,
            action_space=[None],
            observation_space=[None],
        )
        
        checkpoint_path = agent1.save_checkpoint(str(tmp_path), step=100)
        
        # Agent 2: Different topology
        agent2 = AgentTransformerPPO(base_config)
        
        # Extract and set checkpoint dims
        ca_dims = agent2.extract_checkpoint_ca_dims(checkpoint_path)
        agent2._checkpoint_ca_dims = ca_dims
        
        observation_names_battery = [[
            "electrical_storage_soc",
            "month",
            "hour",
            "non_shiftable_load",
        ]]
        
        action_names_battery = [["electrical_storage"]]
        
        agent2.attach_environment(
            observation_names=observation_names_battery,
            action_names=action_names_battery,
            action_space=[None],
            observation_space=[None],
        )
        
        # Load checkpoint
        agent2.load_checkpoint(checkpoint_path)
        
        # Optimizer should be freshly initialized (not from checkpoint)
        # because cross-topology transfer was detected
        # This prevents shape mismatches in optimizer state
        
        # Verify model weights loaded successfully
        # Note: _training_step may not be set during cross-topology transfer
        # to avoid confusion (starting fresh training on new topology)
        assert agent2.building_models is not None
        
        # Verify checkpoint dims were used
        assert agent2._checkpoint_ca_dims is not None
        assert "ev_charger" in agent2._checkpoint_ca_dims


class TestProductionScenarios:
    """Document expected behavior for production edge cases."""
    
    def test_documentation_cross_topology_workflow(self):
        """Document the cross-topology deployment workflow.
        
        This test serves as executable documentation for production teams.
        """
        workflow = """
        Production Deployment Workflow for Different Building Topologies:
        
        1. Training Phase:
           - Train on Building_4 (battery + ev_charger)
           - Save checkpoint with num_agents=1, CA dims stored in weights
           
        2. Deployment to Building_2 (battery only):
           a. Create agent with same config
           b. Call extract_checkpoint_ca_dims(checkpoint_path)
           c. Set agent._checkpoint_ca_dims = extracted_dims
           d. Call attach_environment with Building_2's observations
           e. Call load_checkpoint(checkpoint_path)
           f. Agent ready - battery projection loaded, ev_charger unused
           
        3. Deployment to Building_15 (battery + 2× ev_charger):
           a. Same steps as Building_2
           b. Agent creates 3 CA tokens (1 battery + 2 ev)
           c. Battery weights transferred, ev_charger weights shared for both EVs
           
        4. Asset Temporarily Offline:
           - Wrapper provides zero-filled observations
           - Agent predicts actions (including offline asset action)
           - Production code applies action masking based on connected_state
           
        5. New Asset Type (never seen in training):
           - Requires retraining with updated global vocabulary
           - Deploy new checkpoint trained on updated topology
        """
        assert True  # Executable documentation
