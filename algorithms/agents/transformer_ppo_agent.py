"""TransformerPPO Agent — Transformer-based PPO agent for variable-topology buildings.

This agent uses a Transformer backbone to process variable numbers of controllable
assets (CAs), shared read-only observations (SROs), and non-flexible context (NFC)
as tokens. The architecture naturally handles topology changes (assets connecting/
disconnecting) without retraining.

Key features:
- Dynamic cardinality: Handles variable numbers of CAs at runtime
- Per-type projections: Each asset type has its own learned projection
- Marker-based tokenization: Uses marker values to identify token boundaries
- On-policy PPO: Uses GAE for advantage estimation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.agents.base_agent import BaseAgent
from algorithms.utils.observation_tokenizer import ObservationTokenizer
from algorithms.utils.transformer_backbone import TransformerBackbone
from algorithms.utils.ppo_components import (
    ActorHead,
    CriticHead,
    RolloutBuffer,
    compute_ppo_loss,
)


class AgentTransformerPPO(BaseAgent):
    """Transformer-based PPO agent for energy management.

    Satisfies the BaseAgent contract while using a Transformer architecture
    that handles variable numbers of controllable assets.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the agent from configuration.

        Args:
            config: Full experiment configuration dict.
                Expected keys:
                - algorithm.tokenizer_config_path: Path to tokenizer JSON
                - algorithm.transformer: Transformer architecture params
                - algorithm.hyperparameters: PPO hyperparameters
        """
        super().__init__()
        
        # Mark as Transformer agent for wrapper
        self.is_transformer_agent = True
        
        # Extract config sections
        algo_config = config.get("algorithm", {})
        transformer_config = algo_config.get("transformer", {})
        hyperparams = algo_config.get("hyperparameters", {})
        
        # Load tokenizer config
        tokenizer_config_path = algo_config.get("tokenizer_config_path")
        if tokenizer_config_path:
            with open(tokenizer_config_path) as f:
                self.tokenizer_config = json.load(f)
        else:
            raise ValueError("tokenizer_config_path is required")
        
        # Store hyperparameters
        self.d_model = transformer_config.get("d_model", 64)
        self.nhead = transformer_config.get("nhead", 4)
        self.num_layers = transformer_config.get("num_layers", 2)
        self.dim_feedforward = transformer_config.get("dim_feedforward", 128)
        self.dropout = transformer_config.get("dropout", 0.1)
        
        self.learning_rate = hyperparams.get("learning_rate", 3e-4)
        self.gamma = hyperparams.get("gamma", 0.99)
        self.gae_lambda = hyperparams.get("gae_lambda", 0.95)
        self.clip_eps = hyperparams.get("clip_eps", 0.2)
        self.ppo_epochs = hyperparams.get("ppo_epochs", 4)
        self.minibatch_size = hyperparams.get("minibatch_size", 64)
        self.entropy_coeff = hyperparams.get("entropy_coeff", 0.01)
        self.value_coeff = hyperparams.get("value_coeff", 0.5)
        self.max_grad_norm = hyperparams.get("max_grad_norm", 0.5)
        self.hidden_dim = hyperparams.get("hidden_dim", 128)
        self.rollout_length = hyperparams.get("rollout_length", 2048)
        
        # Create components
        self.tokenizer = ObservationTokenizer(self.tokenizer_config, self.d_model)
        self.backbone = TransformerBackbone(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self.actor = ActorHead(d_model=self.d_model, hidden_dim=self.hidden_dim)
        self.critic = CriticHead(d_model=self.d_model, hidden_dim=self.hidden_dim)
        
        # Combine all parameters for optimizer
        self.all_params = (
            list(self.tokenizer.parameters()) +
            list(self.backbone.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters())
        )
        self.optimizer = optim.Adam(self.all_params, lr=self.learning_rate)
        
        # Per-building rollout buffers (created when num_buildings is known)
        self.rollout_buffers: List[RolloutBuffer] = []
        self._num_buildings: int = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()
        
        # Training state
        self._step_count = 0
        self._last_values: List[Optional[torch.Tensor]] = []
        self._last_log_probs: List[Optional[torch.Tensor]] = []

    def _move_to_device(self) -> None:
        """Move all modules to the configured device."""
        self.tokenizer.to(self.device)
        self.backbone.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)

    def attach_environment(
        self,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List,
        observation_space: List,
        metadata: Dict[str, Any],
    ) -> None:
        """Receive environment metadata from wrapper.

        Called by the wrapper after environment setup. Creates per-building
        rollout buffers.

        Args:
            observation_names: Observation names per building.
            action_names: Action names per building.
            action_space: Action spaces per building.
            observation_space: Observation spaces per building.
            metadata: Additional environment metadata.
        """
        self._num_buildings = len(observation_names)
        self.observation_names = observation_names
        self.action_names = action_names
        
        # Create per-building rollout buffers
        self.rollout_buffers = [
            RolloutBuffer(gamma=self.gamma, gae_lambda=self.gae_lambda)
            for _ in range(self._num_buildings)
        ]
        
        # Initialize tracking lists
        self._last_values = [None] * self._num_buildings
        self._last_log_probs = [None] * self._num_buildings

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False,
    ) -> List[np.ndarray]:
        """Predict actions for all buildings.

        Args:
            observations: List of encoded observations per building.
                Each observation should already have markers injected.
            deterministic: If True, use mean actions without sampling.

        Returns:
            List of action arrays per building.
        """
        # Placeholder - will be implemented in Task 5
        raise NotImplementedError("predict() will be implemented in Task 5")

    def update(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        terminated: List[bool],
        truncated: List[bool],
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> Dict[str, float]:
        """PPO on-policy update step.

        Args:
            observations: Encoded observations per building.
            actions: Actions taken per building.
            rewards: Rewards received per building.
            next_observations: Next observations per building.
            terminated: Episode termination flags per building.
            truncated: Episode truncation flags per building.
            update_target_step: Ignored (no target network in PPO).
            global_learning_step: Current learning step.
            update_step: Whether to perform PPO update.
            initial_exploration_done: Whether initial exploration is done.

        Returns:
            Metrics dict (empty if no update performed).
        """
        # Placeholder - will be implemented in Task 6
        return {}

    def export_artifacts(
        self,
        output_dir: Path,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Export model artifacts for deployment.

        Args:
            output_dir: Directory to save artifacts.
            context: Optional context from wrapper.

        Returns:
            Manifest metadata dict.
        """
        # Placeholder - will be implemented in Task 7
        return {"algorithm": "AgentTransformerPPO"}

