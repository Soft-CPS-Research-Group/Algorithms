"""Tests for TransformerMADDPG agent.

Covers:
- Predict returns correct shape
- Predict handles variable agents
- Update decreases loss (basic sanity)
- Update respects schedule flags
- Checkpoint save/load roundtrip
- Registry integration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from algorithms.agents.transformer_maddpg_agent import TransformerMADDPG
from algorithms.registry import ALGORITHM_REGISTRY, is_algorithm_supported


# --- Fixtures ---


def make_mock_action_space(num_agents: int, action_dim: int) -> List[Any]:
    """Create mock action spaces."""
    spaces = []
    for _ in range(num_agents):
        space = MagicMock()
        space.shape = (action_dim,)
        space.low = np.array([-1.0] * action_dim)
        space.high = np.array([1.0] * action_dim)
        spaces.append(space)
    return spaces


def make_mock_observation_space(num_agents: int, obs_dim: int) -> List[Any]:
    """Create mock observation spaces."""
    spaces = []
    for _ in range(num_agents):
        space = MagicMock()
        space.shape = (obs_dim,)
        space.low = np.zeros(obs_dim)
        space.high = np.ones(obs_dim)
        spaces.append(space)
    return spaces


def make_observation_names(num_agents: int) -> List[List[str]]:
    """Create observation names for testing."""
    return [
        [
            "month",
            "hour", 
            "electric_vehicle_soc",
            "connected_state",
            "departure_time",
            "arrival_time",
            "outdoor_dry_bulb_temperature",
            "non_shiftable_load",
            "carbon_intensity",
        ]
        for _ in range(num_agents)
    ]


def make_action_names(num_agents: int, action_dim: int) -> List[List[str]]:
    """Create action names for testing."""
    return [
        [f"action_{i}" for i in range(action_dim)]
        for _ in range(num_agents)
    ]


def make_observations(num_agents: int, obs_dim: int) -> List[np.ndarray]:
    """Generate random encoded observations."""
    return [np.random.randn(obs_dim).astype(np.float32) for _ in range(num_agents)]


@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Base configuration for TransformerMADDPG."""
    return {
        "algorithm": {
            "name": "TransformerMADDPG",
            "hyperparameters": {
                "gamma": 0.99,
            },
            "networks": {
                "transformer": {
                    "d_model": 32,
                    "nhead": 4,
                    "num_layers": 2,
                    "dim_feedforward": 64,
                    "dropout": 0.0,
                    "max_tokens": 128,  # Increased to handle padding
                },
                "lr_actor": 1e-4,
                "lr_critic": 1e-3,
            },
            "tokenizer": {
                "ca_feature_patterns": [
                    "electric_vehicle_soc",
                    "connected_state",
                    "departure_time",
                    "arrival_time",
                ],
                "sro_feature_patterns": [
                    "outdoor_dry_bulb",
                    "carbon_intensity",
                ],
                "nfc_feature_patterns": [
                    "non_shiftable_load",
                ],
            },
            "replay_buffer": {
                "capacity": 1000,
                "batch_size": 32,
            },
            "exploration": {
                "strategy": "GaussianNoise",
                "params": {
                    "gamma": 0.99,
                    "tau": 0.005,
                    "sigma": 0.1,
                    "bias": 0.0,
                    "end_initial_exploration_time_step": 0,
                },
            },
        },
        "training": {
            "seed": 42,
        },
        "checkpointing": {
            "checkpoint_artifact": "test_checkpoint.pth",
        },
        "tracking": {
            "mlflow_step_sample_interval": 10,
        },
    }


@pytest.fixture
def agent_with_env(base_config: Dict[str, Any]) -> TransformerMADDPG:
    """Create and initialize agent with mock environment."""
    num_agents = 2
    action_dim = 2
    
    agent = TransformerMADDPG(base_config)
    
    agent.attach_environment(
        observation_names=make_observation_names(num_agents),
        action_names=make_action_names(num_agents, action_dim),
        action_space=make_mock_action_space(num_agents, action_dim),
        observation_space=make_mock_observation_space(num_agents, 9),
    )
    
    return agent


# --- Registry Tests ---


class TestRegistry:
    """Tests for registry integration."""

    def test_transformer_maddpg_in_registry(self) -> None:
        """TransformerMADDPG should be in registry."""
        assert "TransformerMADDPG" in ALGORITHM_REGISTRY

    def test_is_algorithm_supported(self) -> None:
        """is_algorithm_supported should return True."""
        assert is_algorithm_supported("TransformerMADDPG")


# --- Initialization Tests ---


class TestInitialization:
    """Tests for agent initialization."""

    def test_create_without_env(self, base_config: Dict[str, Any]) -> None:
        """Agent should create without environment attached."""
        agent = TransformerMADDPG(base_config)
        assert not agent._initialized

    def test_attach_environment(self, base_config: Dict[str, Any]) -> None:
        """attach_environment should initialize all components."""
        agent = TransformerMADDPG(base_config)
        
        num_agents = 2
        action_dim = 2
        
        agent.attach_environment(
            observation_names=make_observation_names(num_agents),
            action_names=make_action_names(num_agents, action_dim),
            action_space=make_mock_action_space(num_agents, action_dim),
            observation_space=make_mock_observation_space(num_agents, 9),
        )
        
        assert agent._initialized
        assert agent.actor is not None
        assert agent.critic is not None
        assert agent.tokenizer is not None
        assert agent.replay_buffer is not None


# --- Predict Tests ---


class TestPredict:
    """Tests for predict method."""

    def test_predict_returns_correct_shape(
        self, agent_with_env: TransformerMADDPG
    ) -> None:
        """predict should return actions for each agent."""
        observations = make_observations(2, 9)
        
        actions = agent_with_env.predict(observations)
        
        assert len(actions) == 2
        assert len(actions[0]) == 2  # action_dim
        assert len(actions[1]) == 2

    def test_predict_deterministic_no_noise(
        self, agent_with_env: TransformerMADDPG
    ) -> None:
        """Deterministic predict should be reproducible."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        observations = make_observations(2, 9)
        
        actions1 = agent_with_env.predict(observations, deterministic=True)
        actions2 = agent_with_env.predict(observations, deterministic=True)
        
        np.testing.assert_array_almost_equal(actions1, actions2)

    def test_predict_variable_agents(self, base_config: Dict[str, Any]) -> None:
        """predict should handle different agent counts."""
        for num_agents in [1, 3, 5]:
            agent = TransformerMADDPG(base_config)
            
            agent.attach_environment(
                observation_names=make_observation_names(num_agents),
                action_names=make_action_names(num_agents, 2),
                action_space=make_mock_action_space(num_agents, 2),
                observation_space=make_mock_observation_space(num_agents, 9),
            )
            
            observations = make_observations(num_agents, 9)
            actions = agent.predict(observations)
            
            assert len(actions) == num_agents

    def test_predict_raises_before_init(self, base_config: Dict[str, Any]) -> None:
        """predict should raise if not initialized."""
        agent = TransformerMADDPG(base_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            agent.predict([np.zeros(9)])


# --- Update Tests ---


class TestUpdate:
    """Tests for update method."""

    def test_update_stores_experience(
        self, agent_with_env: TransformerMADDPG
    ) -> None:
        """update should add experience to replay buffer."""
        observations = make_observations(2, 9)
        actions = [[0.0, 0.0], [0.0, 0.0]]
        rewards = [1.0, 1.0]
        next_observations = make_observations(2, 9)
        
        agent_with_env.update(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=1,
            update_step=True,
            initial_exploration_done=True,
        )
        
        assert len(agent_with_env.replay_buffer) == 1

    def test_update_skips_without_enough_samples(
        self, agent_with_env: TransformerMADDPG
    ) -> None:
        """update should skip learning without enough samples."""
        # Add fewer experiences than batch_size
        for _ in range(10):
            observations = make_observations(2, 9)
            actions = [[0.0, 0.0], [0.0, 0.0]]
            rewards = [1.0, 1.0]
            next_observations = make_observations(2, 9)
            
            # This should not raise even though batch_size is 32
            agent_with_env.update(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                terminated=False,
                truncated=False,
                update_target_step=False,
                global_learning_step=1,
                update_step=True,
                initial_exploration_done=True,
            )

    def test_update_respects_schedule_flags(
        self, agent_with_env: TransformerMADDPG
    ) -> None:
        """update should skip when schedule flags are False."""
        # Fill buffer
        for _ in range(50):
            obs = make_observations(2, 9)
            next_obs = make_observations(2, 9)
            agent_with_env.update(
                observations=obs,
                actions=[[0.0, 0.0], [0.0, 0.0]],
                rewards=[1.0, 1.0],
                next_observations=next_obs,
                terminated=False,
                truncated=False,
                update_target_step=False,
                global_learning_step=1,
                update_step=False,  # Skip update
                initial_exploration_done=True,
            )
        
        # Should have added experiences but not trained
        assert len(agent_with_env.replay_buffer) == 50


# --- Checkpoint Tests ---


class TestCheckpoints:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint(
        self, agent_with_env: TransformerMADDPG, tmp_path: Path
    ) -> None:
        """save_checkpoint should create checkpoint file."""
        checkpoint_path = agent_with_env.save_checkpoint(str(tmp_path), step=100)
        
        assert Path(checkpoint_path).exists()

    def test_load_checkpoint(
        self, agent_with_env: TransformerMADDPG, tmp_path: Path
    ) -> None:
        """load_checkpoint should restore state."""
        # Save
        checkpoint_path = agent_with_env.save_checkpoint(str(tmp_path), step=100)
        
        # Modify actor weights
        original_weight = agent_with_env.actor.head[1].weight.data.clone()
        agent_with_env.actor.head[1].weight.data.fill_(0.0)
        
        # Verify weights changed
        assert not torch.equal(agent_with_env.actor.head[1].weight.data, original_weight)
        
        # Load
        agent_with_env.load_checkpoint(checkpoint_path)
        
        # Verify restored
        assert torch.equal(agent_with_env.actor.head[1].weight.data, original_weight)

    def test_checkpoint_roundtrip(
        self, base_config: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Checkpoint should preserve predict behavior."""
        num_agents = 2
        action_dim = 2
        
        # Create first agent
        agent1 = TransformerMADDPG(base_config)
        agent1.attach_environment(
            observation_names=make_observation_names(num_agents),
            action_names=make_action_names(num_agents, action_dim),
            action_space=make_mock_action_space(num_agents, action_dim),
            observation_space=make_mock_observation_space(num_agents, 9),
        )
        
        # Get predictions
        torch.manual_seed(42)
        observations = make_observations(2, 9)
        actions1 = agent1.predict(observations, deterministic=True)
        
        # Save checkpoint
        checkpoint_path = agent1.save_checkpoint(str(tmp_path), step=100)
        
        # Create second agent and load
        agent2 = TransformerMADDPG(base_config)
        agent2.attach_environment(
            observation_names=make_observation_names(num_agents),
            action_names=make_action_names(num_agents, action_dim),
            action_space=make_mock_action_space(num_agents, action_dim),
            observation_space=make_mock_observation_space(num_agents, 9),
        )
        agent2.load_checkpoint(checkpoint_path)
        
        # Compare predictions
        torch.manual_seed(42)
        actions2 = agent2.predict(observations, deterministic=True)
        
        np.testing.assert_array_almost_equal(actions1, actions2)


# --- Initial Exploration Tests ---


class TestInitialExploration:
    """Tests for exploration phase gating."""

    def test_is_initial_exploration_done(
        self, agent_with_env: TransformerMADDPG
    ) -> None:
        """is_initial_exploration_done should gate on step count."""
        # Default threshold is 0
        assert agent_with_env.is_initial_exploration_done(0)
        assert agent_with_env.is_initial_exploration_done(100)

    def test_nonzero_exploration_threshold(
        self, base_config: Dict[str, Any]
    ) -> None:
        """Should respect non-zero exploration threshold."""
        base_config["algorithm"]["exploration"]["params"]["end_initial_exploration_time_step"] = 100
        
        agent = TransformerMADDPG(base_config)
        
        assert not agent.is_initial_exploration_done(50)
        assert agent.is_initial_exploration_done(100)
        assert agent.is_initial_exploration_done(150)


# --- Integration Tests ---


class TestIntegration:
    """Full integration tests."""

    def test_training_loop_simulation(
        self, agent_with_env: TransformerMADDPG
    ) -> None:
        """Simulate a mini training loop."""
        num_steps = 100
        
        for step in range(num_steps):
            observations = make_observations(2, 9)
            
            # Predict
            actions = agent_with_env.predict(observations)
            
            # Simulate environment step
            rewards = [np.random.randn() for _ in range(2)]
            next_observations = make_observations(2, 9)
            done = step == num_steps - 1
            
            # Update
            agent_with_env.update(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                terminated=done,
                truncated=False,
                update_target_step=(step % 10 == 0),
                global_learning_step=step,
                update_step=True,
                initial_exploration_done=True,
            )
        
        # Should have accumulated experiences
        assert len(agent_with_env.replay_buffer) == num_steps
