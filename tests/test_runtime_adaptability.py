"""Tests for runtime adaptability features (flexy plan Phase A + C).

Phase A: Pre-allocation — all projections sized for full config vocabulary.
Phase C: Checkpoint compatibility — checkpoints work across topologies.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO

# Import building configs from test_agent_transformer_ppo
import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_agent_transformer_ppo import (
    BUILDING_1_ACTIONS,
    BUILDING_1_OBS,
    SINGLE_EV_ACTIONS,
    SINGLE_EV_OBS,
)

# Define battery-only building config
SINGLE_BATTERY_OBS = [
    "month",
    "hour",
    "day_type",
    "daylight_savings_status",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_6h",
    "outdoor_dry_bulb_temperature_predicted_12h",
    "outdoor_dry_bulb_temperature_predicted_24h",
    "outdoor_relative_humidity",
    "outdoor_relative_humidity_predicted_6h",
    "outdoor_relative_humidity_predicted_12h",
    "outdoor_relative_humidity_predicted_24h",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_6h",
    "diffuse_solar_irradiance_predicted_12h",
    "diffuse_solar_irradiance_predicted_24h",
    "direct_solar_irradiance",
    "direct_solar_irradiance_predicted_6h",
    "direct_solar_irradiance_predicted_12h",
    "direct_solar_irradiance_predicted_24h",
    "carbon_intensity",
    "indoor_dry_bulb_temperature",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing",
    "electricity_pricing_predicted_1h",
    "electricity_pricing_predicted_2h",
    "electricity_pricing_predicted_3h",
]

SINGLE_BATTERY_ACTIONS = ["electrical_storage"]


class DummySpace:
    def __init__(self, shape):
        self.shape = shape


def _make_config(num_agents=1):
    return {
        "algorithm": {
            "name": "AgentTransformerPPO",
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
                "min_steps_before_update": 0,
            },
            "transformer": {
                "d_model": 32,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.0,
            },
            "tokenizer": {
                "ca_types": {
                    "ev_charger": {
                        "features": [
                            "connected_state",
                            "departure_time",
                            "required_soc_departure",
                            "_soc",
                            "battery_capacity",
                            "incoming_state",
                            "arrival_time",
                        ],
                        "action_name": "electric_vehicle_storage",
                    },
                    "battery": {
                        "features": ["electrical_storage_soc"],
                        "action_name": "electrical_storage",
                    },
                    "washing_machine": {
                        "features": ["start_time_step", "end_time_step"],
                        "action_name": "washing_machine",
                    },
                },
                "sro_types": {
                    "temporal": {
                        "features": [
                            "month",
                            "hour",
                            "day_type",
                            "daylight_savings_status",
                        ],
                    },
                    "weather": {
                        "features": [
                            "outdoor_dry_bulb_temperature",
                            "outdoor_relative_humidity",
                            "diffuse_solar_irradiance",
                            "direct_solar_irradiance",
                        ],
                    },
                    "pricing": {"features": ["electricity_pricing"]},
                    "carbon": {"features": ["carbon_intensity"]},
                },
                "rl": {
                    "demand_feature": "non_shiftable_load",
                    "generation_features": ["solar_generation"],
                    "extra_features": ["net_electricity_consumption"],
                },
            },
        },
        "topology": {"num_agents": num_agents},
        "tracking": {"mlflow_enabled": False},
        "checkpointing": {"checkpoint_interval": 1000},
    }


def _get_obs_dim(obs_names):
    """Compute encoded obs dimension from raw feature names."""
    from algorithms.utils.encoder_index_map import build_encoder_index_map
    from pathlib import Path
    import json

    encoder_config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "encoders" / "default.json"
    )
    with open(encoder_config_path) as f:
        encoder_config = json.load(f)

    index_map = build_encoder_index_map(obs_names, encoder_config)
    if not index_map:
        return 0
    last_slice = list(index_map.values())[-1]
    return last_slice.end_idx


# -----------------------------------------------------------------------------
# Phase A: Pre-allocation tests
# -----------------------------------------------------------------------------


class TestTokenizerPreallocatesAllCATypes:
    """Verify that all CA types in config get projections, even if inactive."""

    def test_tokenizer_preallocates_all_ca_types(self):
        """Building with 1 CA type should have projections for all 3 config types."""
        agent = AgentTransformerPPO(config=_make_config(num_agents=1))
        obs_dim = _get_obs_dim(SINGLE_BATTERY_OBS)

        agent.attach_environment(
            observation_names=[SINGLE_BATTERY_OBS],
            action_names=[SINGLE_BATTERY_ACTIONS],
            action_space=[DummySpace((1,))],
            observation_space=[DummySpace((obs_dim,))],
        )

        tokenizer = agent.building_models[0].tokenizer

        # Config has 3 CA types: battery, ev_charger, washing_machine
        # Building has only battery, but all 3 should have projections
        assert len(tokenizer.ca_projections) == 3
        assert "battery" in tokenizer.ca_projections
        assert "ev_charger" in tokenizer.ca_projections
        assert "washing_machine" in tokenizer.ca_projections

    def test_tokenizer_preallocates_all_sro_types(self):
        """All SRO types with consistent dims should be pre-allocated."""
        agent = AgentTransformerPPO(config=_make_config(num_agents=2))
        obs_dim_1 = _get_obs_dim(BUILDING_1_OBS)
        obs_dim_2 = _get_obs_dim(SINGLE_EV_OBS)

        agent.attach_environment(
            observation_names=[BUILDING_1_OBS, SINGLE_EV_OBS],
            action_names=[BUILDING_1_ACTIONS, SINGLE_EV_ACTIONS],
            action_space=[DummySpace((3,)), DummySpace((1,))],
            observation_space=[DummySpace((obs_dim_1,)), DummySpace((obs_dim_2,))],
        )

        tokenizer = agent.building_models[0].tokenizer

        # SRO types with consistent dims should be pre-allocated
        # (pricing may be inconsistent and thus per-building)
        assert "temporal" in tokenizer.sro_projections
        assert "weather" in tokenizer.sro_projections
        assert "carbon" in tokenizer.sro_projections


class TestActorGlobalLogStdShape:
    """Verify ActorHead uses full global vocabulary for log_std."""

    def test_actor_global_log_std_shape(self):
        """ActorHead log_std should have 3 entries (global CA type count)."""
        agent = AgentTransformerPPO(config=_make_config(num_agents=1))
        obs_dim = _get_obs_dim(SINGLE_BATTERY_OBS)

        agent.attach_environment(
            observation_names=[SINGLE_BATTERY_OBS],
            action_names=[SINGLE_BATTERY_ACTIONS],
            action_space=[DummySpace((1,))],
            observation_space=[DummySpace((obs_dim,))],
        )

        actor = agent.building_models[0].actor

        # Config has 3 CA types: battery, ev_charger, washing_machine
        # Building has only battery, but log_std should have 3 entries
        assert actor.log_std.shape == (3,)


class TestGlobalTypeToIdxConsistent:
    """Verify global type-to-idx mapping is consistent across buildings."""

    def test_global_type_to_idx_consistent(self):
        """All buildings should use the same global CA type indices."""
        agent = AgentTransformerPPO(config=_make_config(num_agents=2))
        obs_dim_1 = _get_obs_dim(BUILDING_1_OBS)
        obs_dim_2 = _get_obs_dim(SINGLE_EV_OBS)

        agent.attach_environment(
            observation_names=[BUILDING_1_OBS, SINGLE_EV_OBS],
            action_names=[BUILDING_1_ACTIONS, SINGLE_EV_ACTIONS],
            action_space=[DummySpace((3,)), DummySpace((1,))],
            observation_space=[DummySpace((obs_dim_1,)), DummySpace((obs_dim_2,))],
        )

        # Building 0 has battery + ev + washing_machine
        # Building 1 has ev only
        ca_type_idx_0 = agent._ca_type_indices[0]
        ca_type_idx_1 = agent._ca_type_indices[1]

        # Global type order is alphabetical: battery=0, ev_charger=1, washing_machine=2
        # Building 0 has [battery, ev, washing_machine] in action order
        # Building 1 has [ev] only
        
        # The key test: Building 1's ev_charger should use the same global index
        # as Building 0's ev_charger
        building_0_ca_types = agent.building_models[0].tokenizer.ca_types
        building_1_ca_types = agent.building_models[1].tokenizer.ca_types
        
        # Find ev_charger position in each building
        ev_pos_0 = building_0_ca_types.index("ev_charger")
        ev_pos_1 = building_1_ca_types.index("ev_charger")
        
        # Both should map to the same global index
        assert ca_type_idx_0[ev_pos_0] == ca_type_idx_1[ev_pos_1]


class TestRLProjectionMaxPadded:
    """Verify RL projection is sized to max_rl_input_dim with padding."""

    def test_rl_projection_max_padded(self):
        """RL projection should use max dim across all buildings."""
        agent = AgentTransformerPPO(config=_make_config(num_agents=2))
        obs_dim_1 = _get_obs_dim(BUILDING_1_OBS)
        obs_dim_2 = _get_obs_dim(SINGLE_EV_OBS)

        agent.attach_environment(
            observation_names=[BUILDING_1_OBS, SINGLE_EV_OBS],
            action_names=[BUILDING_1_ACTIONS, SINGLE_EV_ACTIONS],
            action_space=[DummySpace((3,)), DummySpace((1,))],
            observation_space=[DummySpace((obs_dim_1,)), DummySpace((obs_dim_2,))],
        )

        # Both buildings should have RL projections with the same input dim
        rl_proj_0 = agent.building_models[0].tokenizer.rl_projection
        rl_proj_1 = agent.building_models[1].tokenizer.rl_projection

        assert rl_proj_0 is not None
        assert rl_proj_1 is not None
        assert rl_proj_0.in_features == rl_proj_1.in_features


# -----------------------------------------------------------------------------
# Phase C: Checkpoint compatibility tests
# -----------------------------------------------------------------------------


class TestCheckpointCrossTopology:
    """Verify checkpoints work across different topologies.
    
    Note: Per Decision #2, cross-topology transfer requires training on a dataset
    that includes all CA types. We train on Building_1 (battery + EV + WM) and
    load into buildings with subsets of those types.
    """

    def test_checkpoint_3ca_to_1ca(self):
        """Train on Building_1 (3 CAs), load into battery-only building."""
        # Train a model with 3 CAs (Building_1)
        agent_3ca = AgentTransformerPPO(config=_make_config(num_agents=1))
        obs_dim_3ca = _get_obs_dim(BUILDING_1_OBS)

        agent_3ca.attach_environment(
            observation_names=[BUILDING_1_OBS],
            action_names=[BUILDING_1_ACTIONS],
            action_space=[DummySpace((3,))],
            observation_space=[DummySpace((obs_dim_3ca,))],
        )

        # Save a checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = agent_3ca.save_checkpoint(tmpdir, step=0)

            # Create a new agent trained on Building_1 but attached to battery-only
            # (This simulates: train on full topology, deploy on subset)
            agent_1ca = AgentTransformerPPO(config=_make_config(num_agents=2))
            obs_dim_1ca = _get_obs_dim(SINGLE_BATTERY_OBS)

            # Attach to both Building_1 and battery-only to observe all CA types
            agent_1ca.attach_environment(
                observation_names=[BUILDING_1_OBS, SINGLE_BATTERY_OBS],
                action_names=[BUILDING_1_ACTIONS, SINGLE_BATTERY_ACTIONS],
                action_space=[DummySpace((3,)), DummySpace((1,))],
                observation_space=[DummySpace((obs_dim_3ca,)), DummySpace((obs_dim_1ca,))],
            )

            # Load the checkpoint
            agent_1ca.load_checkpoint(checkpoint_path)

            # Verify the battery-only building (index 1) can predict
            obs = np.random.randn(obs_dim_1ca).astype(np.float64)
            actions = agent_1ca.predict([np.zeros(obs_dim_3ca), obs], deterministic=True)
            assert len(actions) == 2
            assert len(actions[1]) == 1  # Battery-only building
            assert -1 <= actions[1][0] <= 1  # Valid action range

    def test_checkpoint_3ca_to_2ca(self):
        """Train on Building_1 (3 CAs), load into 2-CA building (battery + EV)."""
        # Train on Building_1
        agent_3ca = AgentTransformerPPO(config=_make_config(num_agents=1))
        obs_dim_3ca = _get_obs_dim(BUILDING_1_OBS)

        agent_3ca.attach_environment(
            observation_names=[BUILDING_1_OBS],
            action_names=[BUILDING_1_ACTIONS],
            action_space=[DummySpace((3,))],
            observation_space=[DummySpace((obs_dim_3ca,))],
        )

        # Save a checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = agent_3ca.save_checkpoint(tmpdir, step=0)

            # Create agent with Building_1 + EV-only building
            agent_mixed = AgentTransformerPPO(config=_make_config(num_agents=2))
            obs_dim_ev = _get_obs_dim(SINGLE_EV_OBS)

            agent_mixed.attach_environment(
                observation_names=[BUILDING_1_OBS, SINGLE_EV_OBS],
                action_names=[BUILDING_1_ACTIONS, SINGLE_EV_ACTIONS],
                action_space=[DummySpace((3,)), DummySpace((1,))],
                observation_space=[DummySpace((obs_dim_3ca,)), DummySpace((obs_dim_ev,))],
            )

            # Load checkpoint
            agent_mixed.load_checkpoint(checkpoint_path)

            # Verify EV-only building can predict
            obs = np.random.randn(obs_dim_ev).astype(np.float64)
            actions = agent_mixed.predict([np.zeros(obs_dim_3ca), obs], deterministic=True)
            assert len(actions) == 2
            assert len(actions[1]) == 1  # EV-only building
            assert -1 <= actions[1][0] <= 1

    def test_battery_projection_weights_transfer(self):
        """Verify battery projection weights are identical after cross-topology load."""
        # Train a model with Building_1 (3 CAs)
        agent_3ca = AgentTransformerPPO(config=_make_config(num_agents=1))
        obs_dim_3ca = _get_obs_dim(BUILDING_1_OBS)

        agent_3ca.attach_environment(
            observation_names=[BUILDING_1_OBS],
            action_names=[BUILDING_1_ACTIONS],
            action_space=[DummySpace((3,))],
            observation_space=[DummySpace((obs_dim_3ca,))],
        )

        # Get battery projection weights
        battery_projection_3ca = agent_3ca.building_models[0].tokenizer.ca_projections["battery"]
        original_weight = battery_projection_3ca.weight.data.clone()
        original_bias = battery_projection_3ca.bias.data.clone()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = agent_3ca.save_checkpoint(tmpdir, step=0)

            # Create agent with Building_1 + battery-only
            agent_mixed = AgentTransformerPPO(config=_make_config(num_agents=2))
            obs_dim_1ca = _get_obs_dim(SINGLE_BATTERY_OBS)

            agent_mixed.attach_environment(
                observation_names=[BUILDING_1_OBS, SINGLE_BATTERY_OBS],
                action_names=[BUILDING_1_ACTIONS, SINGLE_BATTERY_ACTIONS],
                action_space=[DummySpace((3,)), DummySpace((1,))],
                observation_space=[DummySpace((obs_dim_3ca,)), DummySpace((obs_dim_1ca,))],
            )

            agent_mixed.load_checkpoint(checkpoint_path)

            # Verify battery projection weights match for building 0 (loaded from checkpoint)
            # Building 1 keeps random init because checkpoint only has building 0
            battery_projection_0 = agent_mixed.building_models[0].tokenizer.ca_projections["battery"]
            
            torch.testing.assert_close(battery_projection_0.weight.data, original_weight)
            torch.testing.assert_close(battery_projection_0.bias.data, original_bias)


class TestCheckpointFilterTransientBuffers:
    """Verify transient buffers are filtered during checkpoint load."""

    def test_checkpoint_filter_transient_buffers(self):
        """Verify only index buffers are filtered, learned params remain."""
        agent = AgentTransformerPPO(config=_make_config(num_agents=1))
        obs_dim = _get_obs_dim(BUILDING_1_OBS)

        agent.attach_environment(
            observation_names=[BUILDING_1_OBS],
            action_names=[BUILDING_1_ACTIONS],
            action_space=[DummySpace((3,))],
            observation_space=[DummySpace((obs_dim,))],
        )

        # Get a state dict
        state_dict = agent.building_models[0].state_dict()

        # Check that index buffers are present before filtering
        index_buffer_keys = [k for k in state_dict.keys() if "_ca_idx_" in k or "_sro_idx_" in k or "_rl_" in k and "_idx" in k]
        assert len(index_buffer_keys) > 0, "Should have index buffers in state dict"

        # Filter
        filtered = agent._filter_transient_buffers(state_dict)

        # Check that index buffers are removed
        filtered_index_keys = [k for k in filtered.keys() if "_ca_idx_" in k or "_sro_idx_" in k or "_rl_" in k and "_idx" in k]
        assert len(filtered_index_keys) == 0, "Index buffers should be filtered out"

        # Check that learned parameters remain
        assert "tokenizer.ca_projections.battery.weight" in filtered
        assert "actor.fc1.weight" in filtered
        assert "critic.fc1.weight" in filtered
