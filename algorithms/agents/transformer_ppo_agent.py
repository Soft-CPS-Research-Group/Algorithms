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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from algorithms.agents.base_agent import BaseAgent
from algorithms.constants import DEFAULT_ONNX_OPSET
from algorithms.utils.observation_tokenizer import ObservationTokenizer
from algorithms.utils.transformer_backbone import TransformerBackbone
from algorithms.utils.ppo_components import (
    ActorHead,
    CriticHead,
    RolloutBuffer,
    compute_ppo_loss,
)


class _DeterministicActorExport(nn.Module):
    """Minimal export wrapper for deterministic actor inference."""

    def __init__(self, actor: ActorHead) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, ca_embeddings: torch.Tensor) -> torch.Tensor:
        means = self.actor.mlp(ca_embeddings)
        return torch.tanh(means)


class _DeterministicPolicyExport(nn.Module):
    """End-to-end deterministic policy export module.

    Input is an encoded observation tensor containing marker values.
    Output is deterministic actions in [-1, 1].
    """

    def __init__(
        self,
        tokenizer: ObservationTokenizer,
        backbone: TransformerBackbone,
        actor: ActorHead,
        marker_registry: Optional[Dict[float, tuple[str, str, Optional[str]]]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.actor = actor
        self.marker_registry = marker_registry or {}

    def forward(self, encoded_obs: torch.Tensor) -> torch.Tensor:
        tokenized = self.tokenizer(encoded_obs, marker_registry=self.marker_registry)
        backbone_out = self.backbone(
            tokenized.ca_tokens,
            tokenized.sro_tokens,
            tokenized.nfc_token,
        )
        means = self.actor.mlp(backbone_out.ca_embeddings)
        return torch.tanh(means)


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
        self.all_params: List[torch.nn.Parameter] = (
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
        self._last_obs: List[Optional[torch.Tensor]] = []
        self._last_actions: List[Optional[torch.Tensor]] = []
        self._marker_registry_by_building: List[Dict[float, tuple[str, str, Optional[str]]]] = []

        logger.info(
            "Initialized AgentTransformerPPO (d_model={}, nhead={}, layers={}, device={})",
            self.d_model,
            self.nhead,
            self.num_layers,
            self.device,
        )

    def _move_to_device(self) -> None:
        """Move all modules to the configured device."""
        self.tokenizer.to(self.device)
        self.backbone.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        logger.debug("Moved TransformerPPO modules to device {}", self.device)

    def _as_done_flags(
        self,
        value: bool | List[bool] | np.ndarray,
        *,
        name: str,
    ) -> List[bool]:
        """Normalize done-like values to one boolean flag per building."""
        if isinstance(value, np.ndarray):
            value = value.tolist()

        if isinstance(value, list):
            if len(value) != self._num_buildings:
                logger.warning(
                    "{} length ({}) does not match number of buildings ({}); "
                    "broadcasting best-effort.",
                    name,
                    len(value),
                    self._num_buildings,
                )
                if not value:
                    return [False] * self._num_buildings
                return [bool(value[min(i, len(value) - 1)]) for i in range(self._num_buildings)]
            return [bool(v) for v in value]

        return [bool(value)] * self._num_buildings

    def _build_export_dummy_observation(self, agent_index: int) -> torch.Tensor:
        """Build a safe dummy encoded observation for ONNX export."""
        if (
            0 <= agent_index < len(self._last_obs)
            and self._last_obs[agent_index] is not None
        ):
            cached = self._last_obs[agent_index].detach().to(self.device)
            if cached.ndim == 1:
                return cached.unsqueeze(0)
            if cached.ndim == 2:
                return cached[:1]

        marker_values = self.tokenizer_config.get("marker_values", {})
        ca_base = float(marker_values.get("ca_base", 1000))
        sro_base = float(marker_values.get("sro_base", 2000))
        nfc_marker = float(marker_values.get("nfc", 3001))

        values: List[float] = []

        # Include one CA token using the first configured CA type.
        ca_types = self.tokenizer_config.get("ca_types", {})
        if ca_types:
            first_ca_spec = next(iter(ca_types.values()))
            ca_dim = int(first_ca_spec.get("input_dim", 1))
            values.append(ca_base + 1.0)
            values.extend([0.0] * max(ca_dim, 0))

        # Include all configured SRO tokens in order.
        sro_types = self.tokenizer_config.get("sro_types", {})
        for idx, spec in enumerate(sro_types.values(), start=1):
            sro_dim = int(spec.get("input_dim", 0))
            if sro_dim <= 0:
                continue
            values.append(sro_base + float(idx))
            values.extend([0.0] * sro_dim)

        # Include NFC token if configured.
        nfc_dim = int((self.tokenizer_config.get("nfc", {}) or {}).get("input_dim", 0))
        if nfc_dim > 0:
            values.append(nfc_marker)
            values.extend([0.0] * nfc_dim)

        if not values:
            # Defensive fallback in degenerate configs.
            values = [ca_base + 1.0, 0.0]

        return torch.tensor([values], dtype=torch.float32, device=self.device)

    def _export_end_to_end_onnx(self, export_root: Path) -> List[Dict[str, Any]]:
        """Export end-to-end deterministic policy ONNX per building."""
        num_exports = max(self._num_buildings, 1)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        artifacts: List[Dict[str, Any]] = []

        for agent_index in range(num_exports):
            dummy_obs = self._build_export_dummy_observation(agent_index)
            export_path = onnx_dir / f"agent_{agent_index}.onnx"
            export_module = _DeterministicPolicyExport(
                tokenizer=self.tokenizer,
                backbone=self.backbone,
                actor=self.actor,
                marker_registry=self._marker_registry_for_building(agent_index),
            ).to(self.device)
            export_module.eval()

            try:
                previous_fastpath = torch.backends.mha.get_fastpath_enabled()
                torch.backends.mha.set_fastpath_enabled(False)
                try:
                    with torch.no_grad():
                        torch.onnx.export(
                            export_module,
                            dummy_obs,
                            str(export_path),
                            export_params=True,
                            opset_version=max(DEFAULT_ONNX_OPSET, 17),
                            do_constant_folding=True,
                            input_names=[f"observation_agent_{agent_index}"],
                            output_names=[f"action_agent_{agent_index}"],
                            dynamic_axes={
                                f"observation_agent_{agent_index}": {0: "batch_size"},
                                f"action_agent_{agent_index}": {0: "batch_size", 1: "n_ca"},
                            },
                        )
                finally:
                    torch.backends.mha.set_fastpath_enabled(previous_fastpath)

                logger.info(
                    "Exported end-to-end TransformerPPO ONNX for agent {} at {}",
                    agent_index,
                    export_path,
                )
            except Exception as exc:
                logger.warning(
                    "End-to-end ONNX export failed for agent {} ({}); "
                    "falling back to actor-only ONNX export.",
                    agent_index,
                    exc,
                )
                try:
                    actor_fallback = _DeterministicActorExport(self.actor).to(self.device).eval()
                    dummy_ca_embeddings = torch.randn(1, 1, self.d_model, device=self.device)
                    with torch.no_grad():
                        torch.onnx.export(
                            actor_fallback,
                            dummy_ca_embeddings,
                            str(export_path),
                            export_params=True,
                            opset_version=DEFAULT_ONNX_OPSET,
                            do_constant_folding=True,
                            input_names=[f"ca_embeddings_agent_{agent_index}"],
                            output_names=[f"action_agent_{agent_index}"],
                            dynamic_axes={
                                f"ca_embeddings_agent_{agent_index}": {0: "batch_size", 1: "n_ca"},
                                f"action_agent_{agent_index}": {0: "batch_size", 1: "n_ca"},
                            },
                        )
                    logger.warning(
                        "Exported actor-only fallback ONNX for agent {} at {}",
                        agent_index,
                        export_path,
                    )
                except Exception as fallback_exc:
                    logger.error(
                        "ONNX fallback export also failed for agent {} ({}).",
                        agent_index,
                        fallback_exc,
                    )
                    raise RuntimeError(
                        f"Failed to export ONNX artifact for agent {agent_index}."
                    ) from fallback_exc

            artifacts.append(
                {
                    "agent_index": agent_index,
                    "path": str(export_path.relative_to(export_root)),
                    "format": "onnx",
                    "config": {},
                }
            )

        return artifacts

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
        self._last_obs = [None] * self._num_buildings
        self._last_actions = [None] * self._num_buildings
        previous_marker_registry = list(self._marker_registry_by_building)
        self._marker_registry_by_building = [dict() for _ in range(self._num_buildings)]
        for idx in range(min(len(previous_marker_registry), self._num_buildings)):
            if previous_marker_registry[idx]:
                self._marker_registry_by_building[idx] = dict(previous_marker_registry[idx])
        logger.info(
            "Attached environment metadata to TransformerPPO agent (buildings={})",
            self._num_buildings,
        )

    def update_marker_registry(
        self,
        building_idx: int,
        marker_registry: Dict[float, tuple[str, str, Optional[str]]],
    ) -> None:
        """Update marker-value registry for a building topology."""
        if building_idx < 0:
            return

        if building_idx >= len(self._marker_registry_by_building):
            self._marker_registry_by_building.extend(
                {} for _ in range(building_idx + 1 - len(self._marker_registry_by_building))
            )

        self._marker_registry_by_building[building_idx] = dict(marker_registry)

    def _marker_registry_for_building(
        self,
        building_idx: int,
    ) -> Optional[Dict[float, tuple[str, str, Optional[str]]]]:
        if 0 <= building_idx < len(self._marker_registry_by_building):
            marker_registry = self._marker_registry_by_building[building_idx]
            if marker_registry:
                return marker_registry
        return None

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
        logger.debug(
            "Predict called for {} building(s), deterministic={}",
            len(observations),
            deterministic,
        )

        self.tokenizer.eval()
        self.backbone.eval()
        self.actor.eval()
        self.critic.eval()
        
        all_actions: List[np.ndarray] = []
        
        context = torch.no_grad() if deterministic else torch.enable_grad()
        with context:
            for b_idx, obs in enumerate(observations):
                # Convert to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dim
                
                # Tokenize
                tokenized = self.tokenizer(
                    obs_tensor,
                    marker_registry=self._marker_registry_for_building(b_idx),
                )
                
                # Transformer backbone
                backbone_out = self.backbone(
                    tokenized.ca_tokens,
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                )
                
                # Actor head
                actions, log_probs, _ = self.actor(
                    backbone_out.ca_embeddings,
                    deterministic=deterministic,
                )
                
                # Critic head (for storing value in buffer during training)
                value = self.critic(backbone_out.pooled)
                
                # Store for update step
                if not deterministic:
                    self._last_values[b_idx] = value
                    self._last_log_probs[b_idx] = log_probs
                    self._last_obs[b_idx] = obs_tensor
                    self._last_actions[b_idx] = actions
                
                # Convert to numpy
                actions_np = actions.squeeze(0).squeeze(-1).detach().cpu().numpy()  # [N_ca]
                all_actions.append(actions_np)

                logger.debug(
                    "Predicted action tensor for building {} with shape {}",
                    b_idx,
                    actions_np.shape,
                )

        return all_actions

    def update(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        terminated: bool | List[bool] | np.ndarray,
        truncated: bool | List[bool] | np.ndarray,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> Dict[str, float]:
        """PPO on-policy update step.

        Stores transitions in rollout buffer. When update_step is True and
        buffer has enough data, performs PPO update.

        Args:
            observations: Encoded observations per building.
            actions: Actions taken per building.
            rewards: Rewards received per building.
            next_observations: Next observations per building.
            terminated: Episode termination flag(s), scalar or per building.
            truncated: Episode truncation flag(s), scalar or per building.
            update_target_step: Ignored (no target network in PPO).
            global_learning_step: Current learning step.
            update_step: Whether to perform PPO update.
            initial_exploration_done: Whether initial exploration is done.

        Returns:
            Metrics dict (empty if no update performed).
        """
        metrics: Dict[str, float] = {}

        if not initial_exploration_done:
            logger.debug(
                "Skipping update at global step {} because initial exploration is not done.",
                global_learning_step,
            )
            return metrics

        terminated_flags = self._as_done_flags(terminated, name="terminated")
        truncated_flags = self._as_done_flags(truncated, name="truncated")

        # Store transitions in buffers
        for b_idx in range(self._num_buildings):
            if self._last_values[b_idx] is None:
                logger.debug(
                    "Skipping transition storage for building {} because no cached value/log-prob is available.",
                    b_idx,
                )
                continue
                
            obs_tensor = torch.tensor(
                observations[b_idx], dtype=torch.float32, device=self.device
            )
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                
            action_tensor = torch.tensor(
                actions[b_idx], dtype=torch.float32, device=self.device
            )
            
            done = terminated_flags[b_idx] or truncated_flags[b_idx]

            self.rollout_buffers[b_idx].add(
                observation=obs_tensor.squeeze(0),
                action=action_tensor.squeeze(0) if action_tensor.ndim > 1 else action_tensor,
                log_prob=self._last_log_probs[b_idx].sum(dim=-1).squeeze(),  # Sum over CAs
                reward=rewards[b_idx],
                value=self._last_values[b_idx].squeeze(),
                done=done,
            )

            logger.debug(
                "Stored transition for building {} (buffer size={})",
                b_idx,
                len(self.rollout_buffers[b_idx]),
            )

        self._step_count += 1

        # Perform PPO update if requested and buffer is ready
        if update_step:
            for b_idx in range(self._num_buildings):
                buffer = self.rollout_buffers[b_idx]
                if len(buffer) >= self.minibatch_size:
                    update_metrics = self._ppo_update(b_idx, next_observations[b_idx])
                    for k, v in update_metrics.items():
                        metrics[f"building_{b_idx}/{k}"] = v
                else:
                    logger.debug(
                        "Skipping PPO update for building {} (buffer size={} < minibatch_size={})",
                        b_idx,
                        len(buffer),
                        self.minibatch_size,
                    )
        else:
            logger.debug("Update step flag is false at global step {}; not running PPO update.", global_learning_step)

        return metrics

    def _ppo_update(
        self, building_idx: int, last_obs: np.ndarray
    ) -> Dict[str, float]:
        """Perform PPO update for a single building.

        Args:
            building_idx: Index of the building.
            last_obs: Last observation for bootstrapping value.

        Returns:
            Metrics from the update.
        """
        buffer = self.rollout_buffers[building_idx]

        logger.debug(
            "Starting PPO update for building {} with {} buffered transitions.",
            building_idx,
            len(buffer),
        )

        # Compute last value for GAE
        with torch.no_grad():
            last_obs_tensor = torch.tensor(
                last_obs, dtype=torch.float32, device=self.device
            )
            if last_obs_tensor.ndim == 1:
                last_obs_tensor = last_obs_tensor.unsqueeze(0)
            
            tokenized = self.tokenizer(
                last_obs_tensor,
                marker_registry=self._marker_registry_for_building(building_idx),
            )
            backbone_out = self.backbone(
                tokenized.ca_tokens,
                tokenized.sro_tokens,
                tokenized.nfc_token,
            )
            last_value = self.critic(backbone_out.pooled)
        
        # Compute advantages
        buffer.compute_returns_and_advantages(last_value.squeeze())
        
        # PPO epochs
        all_metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }
        
        self.tokenizer.train()
        self.backbone.train()
        self.actor.train()
        self.critic.train()
        
        for _ in range(self.ppo_epochs):
            for batch in buffer.get_batches(self.minibatch_size):
                # Forward pass
                tokenized = self.tokenizer(
                    batch.observations,
                    marker_registry=self._marker_registry_for_building(building_idx),
                )
                backbone_out = self.backbone(
                    tokenized.ca_tokens,
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                )
                
                # Get new log probs
                _, log_probs_new, _ = self.actor(
                    backbone_out.ca_embeddings,
                    deterministic=False,
                )
                # Sum over CAs (joint action log prob = sum of individual log probs)
                log_probs_new = log_probs_new.sum(dim=-1)
                
                # Get new values
                values_new = self.critic(backbone_out.pooled).squeeze(-1)
                
                # Compute loss
                loss, batch_metrics = compute_ppo_loss(
                    log_probs_new=log_probs_new,
                    log_probs_old=batch.log_probs,
                    advantages=batch.advantages,
                    values=values_new,
                    returns=batch.returns,
                    clip_eps=self.clip_eps,
                    value_coeff=self.value_coeff,
                    entropy_coeff=self.entropy_coeff,
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.all_params, self.max_grad_norm)
                self.optimizer.step()
                
                for k, v in batch_metrics.items():
                    all_metrics[k].append(v)
        
        # Clear buffer
        buffer.clear()
        
        # Average metrics
        averaged = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
        logger.info(
            "Completed PPO update for building {} with metrics {}",
            building_idx,
            averaged,
        )
        return averaged

    def save_checkpoint(self, output_dir: Path, step: int) -> None:
        """Save training checkpoint.

        Args:
            output_dir: Directory to save checkpoint.
            step: Current training step.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "step": step,
            "tokenizer_state_dict": self.tokenizer.state_dict(),
            "backbone_state_dict": self.backbone.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info("TransformerPPO checkpoint saved at step {} -> {}", step, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        self.tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
        self.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self._step_count = checkpoint.get("step", 0)
        logger.info("TransformerPPO checkpoint loaded from {}", checkpoint_path)

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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save final checkpoint
        checkpoint_path = output_dir / "final_model.pt"
        checkpoint = {
            "tokenizer_state_dict": self.tokenizer.state_dict(),
            "backbone_state_dict": self.backbone.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "tokenizer_config": self.tokenizer_config,
        }
        torch.save(checkpoint, checkpoint_path)

        artifacts = self._export_end_to_end_onnx(output_dir)

        manifest = {
            "format": "onnx",
            "artifacts": artifacts,
            "model_path": str(checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
            "algorithm": "AgentTransformerPPO",
            "d_model": self.d_model,
            "num_layers": self.num_layers,
        }

        logger.info("TransformerPPO artifacts exported to {}", output_dir)

        return manifest

    def on_topology_change(self, building_idx: int) -> None:
        """Handle topology change for a building.

        Called by wrapper when observation count changes mid-episode.
        Triggers PPO update if buffer has data, then flushes buffer.

        Args:
            building_idx: Index of the building with topology change.
        """
        buffer = self.rollout_buffers[building_idx]
        logger.info(
            "Topology change received for building {} (buffer size={})",
            building_idx,
            len(buffer),
        )
        if len(buffer) >= self.minibatch_size:
            # Trigger update with current buffer
            # Use last stored observation as bootstrap (best available approximation)
            if self._last_obs[building_idx] is not None:
                last_obs_np = self._last_obs[building_idx].cpu().numpy().squeeze()
                self._ppo_update(building_idx, last_obs_np)
            else:
                # No valid observation available, just clear buffer
                buffer.clear()
                logger.warning(
                    "Topology change for building {} had sufficient buffered data but no "
                    "cached last observation; cleared buffer without PPO update.",
                    building_idx,
                )
        else:
            # Just clear buffer
            buffer.clear()
            logger.debug(
                "Cleared buffer for building {} due to topology change (insufficient samples).",
                building_idx,
            )
