"""Transformer-based Multi-Agent DDPG implementation.

This module provides a MADDPG agent using Transformer policy networks
that support variable-cardinality inputs and outputs at runtime.

Key features:
- Single TransformerActor handles all CAs jointly (not per-agent actors)
- Single centralized TransformerCritic for Q-value estimation
- Variable N_ca and N_sro at runtime without retraining
- Strict 1-to-1 CA-to-output correspondence
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import torch
from loguru import logger
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_

from algorithms.agents.base_agent import BaseAgent
from algorithms.constants import DEFAULT_ONNX_OPSET
from algorithms.utils.tokenizer import ObservationTokenizer, TokenizerConfig
from algorithms.utils.transformer_networks import (
    TransformerActor,
    TransformerConfig,
    TransformerCritic,
)
from algorithms.utils.transformer_replay_buffer import (
    TransformerReplayBuffer,
    TransformerReplayBufferConfig,
)


class TransformerMADDPG(BaseAgent):
    """Multi-Agent DDPG with Transformer policy networks.
    
    This agent uses a shared Transformer encoder architecture that:
    - Accepts variable numbers of CAs (controllable assets) at runtime
    - Produces one action per CA token (1-to-1 mapping)
    - Uses attention to condition each CA output on all inputs
    
    The architecture enables dynamic adaptation to different building
    configurations without retraining.
    """

    def __init__(self, config: dict) -> None:
        """Initialize TransformerMADDPG agent.
        
        Args:
            config: Configuration dict containing algorithm hyperparameters,
                network config, replay buffer config, exploration settings.
        """
        super().__init__()
        self.config = config
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("TransformerMADDPG device: {}", self.device)
        
        # Extract config sections
        algo_cfg = config.get("algorithm", {})
        exploration_cfg = algo_cfg.get("exploration", {}).get("params", {})
        buffer_cfg = algo_cfg.get("replay_buffer", {})
        network_cfg = algo_cfg.get("networks", {})
        transformer_cfg = network_cfg.get("transformer", {})
        tokenizer_cfg = algo_cfg.get("tokenizer", {})
        
        training_cfg = config.get("training", {})
        checkpoint_cfg = config.get("checkpointing", {})
        tracking_cfg = config.get("tracking", {})
        
        # Hyperparameters
        self.gamma = float(exploration_cfg.get("gamma", 0.99))
        self.tau = float(exploration_cfg.get("tau", 0.005))
        self.sigma = float(exploration_cfg.get("sigma", 0.1))
        self.bias = float(exploration_cfg.get("bias", 0.0))
        self.end_initial_exploration_time_step = int(
            exploration_cfg.get("end_initial_exploration_time_step", 0) or 0
        )
        
        # Learning rates
        self.lr_actor = float(network_cfg.get("lr_actor", 1e-4))
        self.lr_critic = float(network_cfg.get("lr_critic", 1e-3))
        
        # Seeds
        self.seed = training_cfg.get("seed", 42)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Checkpoint config
        self.checkpoint_artifact = checkpoint_cfg.get(
            "checkpoint_artifact", "latest_checkpoint.pth"
        )
        self.reset_replay_buffer = checkpoint_cfg.get("reset_replay_buffer", False)
        self.freeze_pretrained_layers = checkpoint_cfg.get(
            "freeze_pretrained_layers", False
        )
        self.fine_tune = checkpoint_cfg.get("fine_tune", False)
        
        # MLflow logging interval
        try:
            self.mlflow_step_sample_interval = int(
                tracking_cfg.get("mlflow_step_sample_interval", 10) or 10
            )
        except (TypeError, ValueError):
            self.mlflow_step_sample_interval = 10
        if self.mlflow_step_sample_interval < 1:
            self.mlflow_step_sample_interval = 1
        
        # Store raw configs for deferred initialization
        self._transformer_cfg = transformer_cfg
        self._tokenizer_cfg = tokenizer_cfg
        self._buffer_cfg = buffer_cfg
        
        # These will be initialized in attach_environment
        self.transformer_config: Optional[TransformerConfig] = None
        self.tokenizer_config: Optional[TokenizerConfig] = None
        self.tokenizer: Optional[ObservationTokenizer] = None
        self.actor: Optional[TransformerActor] = None
        self.critic: Optional[TransformerCritic] = None
        self.actor_target: Optional[TransformerActor] = None
        self.critic_target: Optional[TransformerCritic] = None
        self.actor_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        self.replay_buffer: Optional[TransformerReplayBuffer] = None
        
        # Topology info from environment
        self.observation_names: Optional[List[List[str]]] = None
        self.action_names: Optional[List[List[str]]] = None
        self.action_space: Optional[List[Any]] = None
        self.observation_space: Optional[List[Any]] = None
        self.num_agents: int = 0
        self.action_dimension: int = 0
        
        self._initialized = False
        
        logger.info("TransformerMADDPG created (awaiting attach_environment)")

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize networks and tokenizer with environment metadata.
        
        Args:
            observation_names: Per-agent observation feature names.
            action_names: Per-agent action names.
            action_space: Per-agent action spaces.
            observation_space: Per-agent observation spaces.
            metadata: Optional additional metadata.
        """
        self.observation_names = observation_names
        self.action_names = action_names
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_agents = len(observation_names)
        
        # Derive action dimension from first agent's action space
        if action_space and hasattr(action_space[0], "shape"):
            self.action_dimension = action_space[0].shape[0]
        else:
            self.action_dimension = 1
        
        logger.info(
            "Attaching environment: {} agents, action_dim={}",
            self.num_agents,
            self.action_dimension,
        )
        
        # Initialize components
        self._initialize_transformer_config()
        self._initialize_tokenizer()
        self._initialize_networks()
        self._initialize_replay_buffer()
        self._initialize_optimizers()
        
        self._initialized = True
        logger.info("TransformerMADDPG initialization complete")

    def _initialize_transformer_config(self) -> None:
        """Create TransformerConfig from raw config."""
        cfg = self._transformer_cfg
        self.transformer_config = TransformerConfig(
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 128),
            dropout=cfg.get("dropout", 0.0),
            max_tokens=cfg.get("max_tokens", 128),
            action_dim=self.action_dimension,
        )
        logger.debug("TransformerConfig: {}", self.transformer_config)

    def _initialize_tokenizer(self) -> None:
        """Create tokenizer with observation names."""
        cfg = self._tokenizer_cfg
        self.tokenizer_config = TokenizerConfig(
            d_model=self.transformer_config.d_model,
            ca_feature_patterns=cfg.get("ca_feature_patterns", TokenizerConfig().ca_feature_patterns),
            sro_feature_patterns=cfg.get("sro_feature_patterns", TokenizerConfig().sro_feature_patterns),
            nfc_feature_patterns=cfg.get("nfc_feature_patterns", TokenizerConfig().nfc_feature_patterns),
        )
        self.tokenizer = ObservationTokenizer(
            self.tokenizer_config,
            self.observation_names,
        ).to(self.device)
        logger.debug("Tokenizer initialized with {} agents", self.num_agents)

    def _initialize_networks(self) -> None:
        """Create actor, critic, and target networks."""
        # Actor and targets
        self.actor = TransformerActor(self.transformer_config).to(self.device)
        self.actor_target = TransformerActor(self.transformer_config).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic and targets
        self.critic = TransformerCritic(self.transformer_config).to(self.device)
        self.critic_target = TransformerCritic(self.transformer_config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Set targets to eval mode
        self.actor_target.eval()
        self.critic_target.eval()
        
        # Freeze target parameters
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        logger.debug("Networks initialized")

    def _initialize_replay_buffer(self) -> None:
        """Create replay buffer."""
        cfg = self._buffer_cfg
        
        # Calculate max_ca and max_sro to fit within max_tokens
        # Reserve space: max_ca + max_sro + 1 (NFC) <= max_tokens
        max_tokens = self.transformer_config.max_tokens
        max_ca = cfg.get("max_ca", min(self.num_agents * 2, max_tokens // 2))
        max_sro = cfg.get("max_sro", max(1, (max_tokens - max_ca - 1) // 2))
        
        self.replay_buffer = TransformerReplayBuffer(
            TransformerReplayBufferConfig(
                capacity=cfg.get("capacity", 100000),
                batch_size=cfg.get("batch_size", 256),
                max_ca=max_ca,
                max_sro=max_sro,
                d_model=self.transformer_config.d_model,
                action_dim=self.action_dimension,
            )
        )
        self.batch_size = cfg.get("batch_size", 256)
        logger.debug(
            "Replay buffer initialized with capacity {}, max_ca={}, max_sro={}", 
            cfg.get("capacity", 100000), max_ca, max_sro
        )

    def _initialize_optimizers(self) -> None:
        """Create optimizers for actor and critic."""
        # Include tokenizer parameters in actor optimizer for end-to-end training
        actor_params = list(self.actor.parameters()) + list(self.tokenizer.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_critic
        )
        logger.debug("Optimizers initialized (lr_actor={}, lr_critic={})", 
                     self.lr_actor, self.lr_critic)

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False,
    ) -> List[List[float]]:
        """Predict actions for current observations.
        
        Args:
            observations: Per-agent encoded observations.
            deterministic: If True, skip exploration noise.
            
        Returns:
            Per-agent action lists.
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call attach_environment first.")
        
        self.actor.eval()
        
        with torch.no_grad():
            # Tokenize observations
            ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask = self.tokenizer.tokenize(
                observations, device=self.device
            )
            
            # Forward through actor
            actions = self.actor(ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask)
            # actions: [1, N_ca, action_dim]
            
            actions = actions.squeeze(0).cpu().numpy()  # [N_ca, action_dim]
        
        self.actor.train()
        
        # Add exploration noise if not deterministic
        if not deterministic:
            noise = np.random.normal(scale=self.sigma, size=actions.shape) - self.bias
            actions = np.clip(actions + noise, -1, 1)
        
        # Convert to list of lists
        return [actions[i].tolist() for i in range(actions.shape[0])]

    def update(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        """Update agent with new experience.
        
        Args:
            observations: Per-agent observations.
            actions: Per-agent actions taken.
            rewards: Per-agent rewards received.
            next_observations: Per-agent next observations.
            terminated: Episode terminated flag.
            truncated: Episode truncated flag.
            update_target_step: Whether to update target networks.
            global_learning_step: Global step counter.
            update_step: Whether this is an update step.
            initial_exploration_done: Whether exploration phase is complete.
        """
        if not self._initialized:
            return
        
        done = bool(terminated or truncated)
        
        # Tokenize current and next observations
        with torch.no_grad():
            ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask = self.tokenizer.tokenize(
                observations, device=self.device
            )
            next_ca_tokens, next_sro_tokens, next_nfc_token, next_ca_mask, next_sro_mask = self.tokenizer.tokenize(
                next_observations, device=self.device
            )
            
            # Convert actions to tensor
            actions_tensor = torch.tensor(
                np.array(actions), dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # [1, N_ca, action_dim]
            
            # Mean reward across agents for single scalar
            reward_tensor = torch.tensor(
                [[np.mean(rewards)]], dtype=torch.float32, device=self.device
            )
            done_tensor = torch.tensor(
                [[float(done)]], dtype=torch.float32, device=self.device
            )
        
        # Store experience
        self.replay_buffer.push(
            ca_tokens=ca_tokens,
            sro_tokens=sro_tokens,
            nfc_token=nfc_token,
            ca_mask=ca_mask,
            sro_mask=sro_mask,
            actions=actions_tensor,
            rewards=reward_tensor,
            next_ca_tokens=next_ca_tokens,
            next_sro_tokens=next_sro_tokens,
            next_nfc_token=next_nfc_token,
            next_ca_mask=next_ca_mask,
            next_sro_mask=next_sro_mask,
            done=done_tensor,
        )
        
        # Check if we should update
        if len(self.replay_buffer) < self.batch_size:
            logger.debug("Not enough samples in replay buffer. Skipping update.")
            return
        
        if not initial_exploration_done:
            logger.debug("Initial exploration not done. Skipping update.")
            return
        
        if not update_step:
            logger.debug("Update step skipped based on schedule.")
            return
        
        # Sample batch and update networks
        update_start = time.time()
        batch = self.replay_buffer.sample(device=self.device)
        
        # Critic update
        critic_loss = self._update_critic(batch)
        
        # Actor update
        actor_loss = self._update_actor(batch)
        
        # Soft update targets
        if update_target_step:
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.critic, self.critic_target, self.tau)
        
        # Logging
        should_log = global_learning_step % self.mlflow_step_sample_interval == 0
        if should_log and mlflow.active_run():
            mlflow.log_metrics(
                {
                    "critic_loss": critic_loss.item(),
                    "actor_loss": actor_loss.item(),
                    "training_step_time": time.time() - update_start,
                },
                step=global_learning_step,
            )
        
        logger.debug(
            "Update complete. Critic loss: {:.4f}, Actor loss: {:.4f}",
            critic_loss.item(),
            actor_loss.item(),
        )

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Update critic network.
        
        Args:
            batch: Batched experience dict from replay buffer.
            
        Returns:
            Critic loss tensor.
        """
        with torch.no_grad():
            # Target actions from target actor
            next_actions = self.actor_target(
                batch["next_ca_tokens"],
                batch["next_sro_tokens"],
                batch["next_nfc_token"],
                batch["next_ca_mask"],
                batch["next_sro_mask"],
            )
            
            # Target Q-value
            q_next = self.critic_target(
                batch["next_ca_tokens"],
                batch["next_sro_tokens"],
                batch["next_nfc_token"],
                next_actions,
                batch["next_ca_mask"],
                batch["next_sro_mask"],
            )
            
            q_target = batch["rewards"] + self.gamma * q_next * (1 - batch["dones"])
        
        # Current Q-value
        q_pred = self.critic(
            batch["ca_tokens"],
            batch["sro_tokens"],
            batch["nfc_token"],
            batch["actions"],
            batch["ca_mask"],
            batch["sro_mask"],
        )
        
        # Loss
        critic_loss = mse_loss(q_pred, q_target)
        
        # Backward
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        return critic_loss

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Update actor network.
        
        Args:
            batch: Batched experience dict from replay buffer.
            
        Returns:
            Actor loss tensor.
        """
        # Predicted actions
        pred_actions = self.actor(
            batch["ca_tokens"],
            batch["sro_tokens"],
            batch["nfc_token"],
            batch["ca_mask"],
            batch["sro_mask"],
        )
        
        # Q-value of predicted actions (policy gradient)
        q_value = self.critic(
            batch["ca_tokens"].detach(),
            batch["sro_tokens"].detach(),
            batch["nfc_token"].detach(),
            pred_actions,
            batch["ca_mask"],
            batch["sro_mask"],
        )
        
        # Maximize Q-value
        actor_loss = -q_value.mean()
        
        # Backward
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        clip_grad_norm_(self.tokenizer.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        return actor_loss

    def _soft_update(
        self, local: torch.nn.Module, target: torch.nn.Module, tau: float
    ) -> None:
        """Soft update target network parameters.
        
        Args:
            local: Source network.
            target: Target network.
            tau: Interpolation factor.
        """
        with torch.no_grad():
            for target_param, local_param in zip(
                target.parameters(), local.parameters()
            ):
                target_param.data.lerp_(local_param.data, tau)

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        """Check if initial exploration phase is complete.
        
        Args:
            global_learning_step: Current global step.
            
        Returns:
            True if exploration phase is complete.
        """
        return global_learning_step >= self.end_initial_exploration_time_step

    def save_checkpoint(self, output_dir: str, step: int) -> str:
        """Save training checkpoint.
        
        Args:
            output_dir: Directory to save checkpoint.
            step: Current training step.
            
        Returns:
            Path to saved checkpoint.
        """
        checkpoint: Dict[str, Any] = {
            "step": step,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "tokenizer_state_dict": self.tokenizer.state_dict(),
            "transformer_config": {
                "d_model": self.transformer_config.d_model,
                "nhead": self.transformer_config.nhead,
                "num_layers": self.transformer_config.num_layers,
                "dim_feedforward": self.transformer_config.dim_feedforward,
                "dropout": self.transformer_config.dropout,
                "max_tokens": self.transformer_config.max_tokens,
                "action_dim": self.transformer_config.action_dim,
            },
        }
        
        if hasattr(self.replay_buffer, "get_state"):
            checkpoint["replay_buffer"] = self.replay_buffer.get_state()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = self.checkpoint_artifact or "latest_checkpoint.pth"
        checkpoint_path = output_path / checkpoint_name
        torch.save(checkpoint, checkpoint_path)
        
        logger.info("Checkpoint saved at step {} -> {}", step, checkpoint_path)
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # Load network states
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        
        # Load tokenizer
        if "tokenizer_state_dict" in checkpoint:
            self.tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
        
        # Load optimizers (unless fine-tuning)
        if not self.fine_tune:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        
        # Load replay buffer
        if "replay_buffer" in checkpoint and not self.reset_replay_buffer:
            self.replay_buffer.set_state(checkpoint["replay_buffer"])
        
        # Freeze layers if configured
        if self.freeze_pretrained_layers:
            self._freeze_layers()
        
        logger.info("Checkpoint loaded from {}", checkpoint_path)

    def _freeze_layers(self, freeze_actor: bool = True, freeze_critic: bool = False) -> None:
        """Freeze network layers.
        
        Args:
            freeze_actor: Freeze actor parameters.
            freeze_critic: Freeze critic parameters.
        """
        if freeze_actor:
            for param in self.actor.parameters():
                param.requires_grad = False
        if freeze_critic:
            for param in self.critic.parameters():
                param.requires_grad = False
        logger.info("Layers frozen - actor={}, critic={}", freeze_actor, freeze_critic)

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export inference artifacts (ONNX models).
        
        Args:
            output_dir: Directory for exported artifacts.
            context: Optional context with config info.
            
        Returns:
            Metadata dict about exported artifacts.
        """
        from algorithms.export.transformer_onnx_export import (
            export_transformer_actor_to_onnx,
        )
        
        context = context or {}
        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Exporting TransformerMADDPG to ONNX under {}", onnx_dir)
        
        # Create example inputs for tracing
        B = 1
        N_ca = self.num_agents
        N_sro = 1
        d_model = self.transformer_config.d_model
        
        example_ca = torch.randn(B, N_ca, d_model, device=self.device)
        example_sro = torch.randn(B, N_sro, d_model, device=self.device)
        example_nfc = torch.randn(B, 1, d_model, device=self.device)
        
        # Export actor
        actor_path = onnx_dir / "transformer_actor.onnx"
        export_transformer_actor_to_onnx(
            self.actor,
            str(actor_path),
            example_ca_tokens=example_ca,
            example_sro_tokens=example_sro,
            example_nfc_token=example_nfc,
        )
        
        logger.info("ONNX model exported: {}", actor_path)
        
        metadata: Dict[str, Any] = {
            "format": "onnx",
            "architecture": "TransformerMADDPG",
            "variable_cardinality": True,
            "d_model": d_model,
            "action_dim": self.action_dimension,
            "artifacts": [
                {
                    "name": "transformer_actor",
                    "path": str(actor_path.relative_to(export_root)),
                    "format": "onnx",
                    "dynamic_axes": {
                        "ca_tokens": [0, 1],
                        "sro_tokens": [0, 1],
                        "actions": [0, 1],
                    },
                }
            ],
        }
        
        if mlflow.active_run():
            mlflow.log_artifact(str(actor_path), artifact_path="onnx")
        
        return metadata
