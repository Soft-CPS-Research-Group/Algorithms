"""AgentTransformerPPO — PPO-based agent with a Transformer backbone.

Each building gets its own agent instance (same class, separate weights),
mirroring the existing MADDPG pattern.  The tokenizer is built in
``attach_environment()`` once observation/action names are available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from torch.nn.utils import clip_grad_norm_

from algorithms.agents.base_agent import BaseAgent
from algorithms.constants import DEFAULT_ONNX_OPSET
from algorithms.utils.observation_tokenizer import ObservationTokenizer
from algorithms.utils.ppo_components import (
    ActorHead,
    CriticHead,
    PPORolloutBuffer,
    ppo_loss,
)
from algorithms.utils.transformer_backbone import TransformerBackbone


ENCODER_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "encoders" / "default.json"
)


class _BuildingModel(torch.nn.Module):
    """Combined tokenizer + backbone + heads for one building.

    Wrapping these together makes ONNX export and checkpointing cleaner.
    """

    def __init__(
        self,
        tokenizer: ObservationTokenizer,
        backbone: TransformerBackbone,
        actor: ActorHead,
        critic: CriticHead,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.actor = actor
        self.critic = critic


class AgentTransformerPPO(BaseAgent):
    """Transformer-PPO agent with tokenized observations.

    Configuration keys
    ------------------
    algorithm.hyperparameters:
        learning_rate, gamma, gae_lambda, clip_eps, ppo_epochs,
        minibatch_size, entropy_coeff, value_coeff, max_grad_norm,
        min_steps_before_update
    algorithm.transformer:
        d_model, nhead, num_layers, dim_feedforward, dropout
    algorithm.tokenizer:
        ca_types, sro_types, rl
    topology:
        num_agents
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("AgentTransformerPPO device: {}", self.device)

        # --- Read config sections -------------------------------------------
        algo_cfg = config["algorithm"]
        hp = algo_cfg.get("hyperparameters", {})
        transformer_cfg = algo_cfg.get("transformer", {})
        self.tokenizer_config: Dict[str, Any] = algo_cfg.get("tokenizer", {})

        # PPO hyperparameters
        self.lr: float = float(hp.get("learning_rate", 3e-4))
        self.gamma: float = float(hp.get("gamma", 0.99))
        self.gae_lambda: float = float(hp.get("gae_lambda", 0.95))
        self.clip_eps: float = float(hp.get("clip_eps", 0.2))
        self.ppo_epochs: int = int(hp.get("ppo_epochs", 4))
        self.minibatch_size: int = int(hp.get("minibatch_size", 64))
        self.entropy_coeff: float = float(hp.get("entropy_coeff", 0.01))
        self.value_coeff: float = float(hp.get("value_coeff", 0.5))
        self.max_grad_norm: float = float(hp.get("max_grad_norm", 0.5))
        self.min_steps_before_update: int = int(hp.get("min_steps_before_update", 0))

        # Transformer architecture
        self.d_model: int = int(transformer_cfg.get("d_model", 64))
        self.nhead: int = int(transformer_cfg.get("nhead", 4))
        self.num_layers: int = int(transformer_cfg.get("num_layers", 2))
        self.dim_feedforward: int = int(transformer_cfg.get("dim_feedforward", 128))
        self.dropout: float = float(transformer_cfg.get("dropout", 0.1))

        # Topology
        topology = config.get("topology", {})
        self.num_agents: int = int(topology.get("num_agents", 1))

        # Tracking config
        tracking_cfg = config.get("tracking", {})
        try:
            self.mlflow_step_sample_interval = int(
                tracking_cfg.get("mlflow_step_sample_interval", 10) or 10,
            )
        except (TypeError, ValueError):
            self.mlflow_step_sample_interval = 10

        # Checkpointing config
        checkpoint_cfg = config.get("checkpointing", {})
        self.checkpoint_artifact: str = checkpoint_cfg.get(
            "checkpoint_artifact", "latest_checkpoint.pth",
        )

        # Per-building models and buffers — created in attach_environment()
        self.building_models: Optional[torch.nn.ModuleList] = None
        self.rollout_buffers: List[PPORolloutBuffer] = []
        self.optimizers: List[torch.optim.Adam] = []

        # Per-building metadata
        self._n_ca_per_building: List[int] = []
        self._ca_type_indices: List[Optional[torch.Tensor]] = []
        self._action_ca_maps: List[List[int]] = []
        self._training_step: int = 0
        
        # Checkpoint CA dimensions (populated if loading from checkpoint)
        self._checkpoint_ca_dims: Optional[Dict[str, int]] = None

        logger.info("AgentTransformerPPO initialization complete (awaiting attach_environment).")

    # --------------------------------------------------------------------- #
    # Global vocabulary computation
    # --------------------------------------------------------------------- #

    def _compute_global_vocabulary(
        self,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        encoder_config: Dict[str, Any],
        checkpoint_ca_dims: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], int]:
        """Compute global CA/SRO type dims and type-to-index mapping.
        
        Supports both enriched (marker-based) and raw (heuristic-based) observation names.
        When observation names contain __tkn_*__ markers, uses marker parsing.
        Otherwise, falls back to heuristic classification.
        
        Parameters
        ----------
        checkpoint_ca_dims : Optional[Dict[str, int]]
            CA type -> actual dimensions from checkpoint (overrides fallback estimates).
        
        Returns
        -------
        global_ca_type_dims : Dict[str, int]
            CA type -> encoded feature dims.
        global_sro_type_dims : Dict[str, int]
            SRO type -> encoded feature dims.
        global_type_to_idx : Dict[str, int]
            CA type -> global index (for ActorHead log_std indexing).
        max_rl_input_dim : int
            Maximum RL input dimension across all buildings.
        """
        from algorithms.utils.observation_tokenizer import (
            _parse_markers,
            _build_encoded_dims_map,
            _has_markers,
        )
        
        ca_config = self.tokenizer_config.get("ca_types", {})
        sro_config = self.tokenizer_config.get("sro_types", {})
        rl_config = self.tokenizer_config.get("rl", {})
        
        # Track observed dims per type
        ca_type_dims_observed: Dict[str, int] = {}
        sro_type_dims_observed: Dict[str, int] = {}
        rl_dims_observed: List[int] = []
        
        for i, (obs_names, act_names) in enumerate(zip(observation_names, action_names)):
            # Check if observation names are enriched (have markers)
            if _has_markers(obs_names):
                # --- Marker-based classification ---
                self._compute_vocab_from_markers(
                    obs_names, encoder_config, 
                    ca_type_dims_observed, sro_type_dims_observed, rl_dims_observed,
                    rl_config,
                )
            else:
                # --- Heuristic-based classification (legacy) ---
                self._compute_vocab_from_heuristics(
                    obs_names, act_names, encoder_config, ca_config, sro_config, rl_config,
                    ca_type_dims_observed, sro_type_dims_observed, rl_dims_observed,
                )
        
        # --- Compute fallback dims for unseen CA types ---
        for ca_type_name, ca_spec in ca_config.items():
            if ca_type_name not in ca_type_dims_observed:
                if checkpoint_ca_dims and ca_type_name in checkpoint_ca_dims:
                    ca_type_dims_observed[ca_type_name] = checkpoint_ca_dims[ca_type_name]
                    logger.info(
                        "CA type '{}' not observed; using checkpoint dim: {}",
                        ca_type_name, checkpoint_ca_dims[ca_type_name],
                    )
                else:
                    ca_feature_patterns = ca_spec.get("features", [])
                    estimated_dims = len(ca_feature_patterns)
                    if estimated_dims > 0:
                        ca_type_dims_observed[ca_type_name] = estimated_dims
                        logger.info(
                            "CA type '{}' not observed; using fallback dim estimate: {}",
                            ca_type_name, estimated_dims,
                        )
        
        # --- Compute fallback dims for unseen SRO types ---
        for sro_type_name, sro_spec in sro_config.items():
            if sro_type_name not in sro_type_dims_observed:
                sro_features = sro_spec.get("features", [])
                estimated_dims = len(sro_features)
                if estimated_dims > 0:
                    sro_type_dims_observed[sro_type_name] = estimated_dims
                    logger.info(
                        "SRO type '{}' not observed; using fallback dim estimate: {}",
                        sro_type_name, estimated_dims,
                    )
        
        # --- Build global type-to-index mapping ---
        sorted_ca_types = sorted(ca_type_dims_observed.keys())
        global_type_to_idx = {t: idx for idx, t in enumerate(sorted_ca_types)}
        
        # Filter out inconsistent SRO types (marked with -1)
        consistent_sro_type_dims = {
            k: v for k, v in sro_type_dims_observed.items() if v > 0
        }
        
        max_rl_dim = max(rl_dims_observed) if rl_dims_observed else 0
        
        return (
            ca_type_dims_observed,
            consistent_sro_type_dims,
            global_type_to_idx,
            max_rl_dim,
        )

    def _compute_vocab_from_markers(
        self,
        obs_names: List[str],
        encoder_config: Dict[str, Any],
        ca_type_dims: Dict[str, int],
        sro_type_dims: Dict[str, int],
        rl_dims: List[int],
        rl_config: Dict[str, Any],
    ) -> None:
        """Compute vocabulary from enriched (marker-based) observation names."""
        from algorithms.utils.observation_tokenizer import (
            _parse_markers,
            _build_encoded_dims_map,
        )
        
        token_groups = _parse_markers(obs_names)
        dims_map = _build_encoded_dims_map(obs_names, encoder_config)
        
        for group in token_groups:
            # Compute total encoded dims for this group's features
            group_dims = sum(
                dims_map[n].n_dims
                for n in group.feature_names
                if n in dims_map
            )
            
            if group.family == "ca" and group.type_name and group_dims > 0:
                if group.type_name in ca_type_dims:
                    if ca_type_dims[group.type_name] != group_dims:
                        logger.warning(
                            "CA type '{}' has inconsistent dims: {} vs {}",
                            group.type_name, ca_type_dims[group.type_name], group_dims,
                        )
                else:
                    ca_type_dims[group.type_name] = group_dims
            
            elif group.family == "sro" and group.type_name and group_dims > 0:
                if group.type_name in sro_type_dims:
                    if sro_type_dims[group.type_name] != group_dims:
                        sro_type_dims[group.type_name] = -1  # Mark inconsistent
                else:
                    sro_type_dims[group.type_name] = group_dims
            
            elif group.family == "nfc":
                # Compute RL input dim: residual (1) + extra features
                rl_demand_feature = rl_config.get("demand_feature")
                rl_generation_features = rl_config.get("generation_features", [])
                rl_extra_features = rl_config.get("extra_features", [])
                
                has_demand_or_gen = False
                extra_dims = 0
                
                for feat_name in group.feature_names:
                    if rl_demand_feature and rl_demand_feature in feat_name:
                        has_demand_or_gen = True
                    elif any(gen in feat_name for gen in rl_generation_features):
                        has_demand_or_gen = True
                    elif any(extra in feat_name for extra in rl_extra_features):
                        if feat_name in dims_map:
                            extra_dims += dims_map[feat_name].n_dims
                
                rl_input_dim = (1 if has_demand_or_gen else 0) + extra_dims
                if rl_input_dim > 0:
                    rl_dims.append(rl_input_dim)

    def _compute_vocab_from_heuristics(
        self,
        obs_names: List[str],
        act_names: List[str],
        encoder_config: Dict[str, Any],
        ca_config: Dict[str, Dict[str, Any]],
        sro_config: Dict[str, Dict[str, Any]],
        rl_config: Dict[str, Any],
        ca_type_dims: Dict[str, int],
        sro_type_dims: Dict[str, int],
        rl_dims: List[int],
    ) -> None:
        """Compute vocabulary using heuristic classification (legacy mode)."""
        from algorithms.utils.observation_tokenizer import (
            _build_encoded_dims_map,
            _extract_device_ids,
            _contains_device_id,
            _feature_matches_ca_type,
        )
        
        index_map = _build_encoded_dims_map(obs_names, encoder_config)
        device_ids_by_type = _extract_device_ids(act_names, ca_config)
        
        for ca_type_name, ca_spec in ca_config.items():
            device_ids = device_ids_by_type.get(ca_type_name, [])
            if not device_ids:
                continue
            
            ca_feature_patterns = ca_spec.get("features", [])
            first_device_id = device_ids[0]
            instance_features: List[str] = []
            
            for raw_name in obs_names:
                if not _feature_matches_ca_type(raw_name, ca_feature_patterns):
                    continue
                
                if first_device_id is None:
                    instance_features.append(raw_name)
                elif _contains_device_id(raw_name, first_device_id):
                    instance_features.append(raw_name)
            
            dims = sum(index_map[n].n_dims for n in instance_features if n in index_map)
            if dims > 0:
                if ca_type_name in ca_type_dims:
                    if ca_type_dims[ca_type_name] != dims:
                        logger.warning(
                            "CA type '{}' has inconsistent dims: {} vs {}",
                            ca_type_name, ca_type_dims[ca_type_name], dims,
                        )
                else:
                    ca_type_dims[ca_type_name] = dims
        
        for sro_type_name, sro_spec in sro_config.items():
            sro_features = sro_spec.get("features", [])
            matched_names = []
            
            for raw_name in obs_names:
                for sro_feat in sro_features:
                    if sro_feat in raw_name:
                        matched_names.append(raw_name)
                        break
            
            dims = sum(index_map[n].n_dims for n in matched_names if n in index_map)
            if dims > 0:
                if sro_type_name in sro_type_dims:
                    if sro_type_dims[sro_type_name] != dims:
                        sro_type_dims[sro_type_name] = -1
                else:
                    sro_type_dims[sro_type_name] = dims
        
        rl_demand_feature = rl_config.get("demand_feature")
        rl_generation_features = rl_config.get("generation_features", [])
        rl_extra_features = rl_config.get("extra_features", [])
        
        rl_input_dim = 0
        has_demand_or_gen = False
        
        for raw_name in obs_names:
            if rl_demand_feature and rl_demand_feature in raw_name:
                has_demand_or_gen = True
                break
            if any(gen in raw_name for gen in rl_generation_features):
                has_demand_or_gen = True
                break
        
        if has_demand_or_gen:
            rl_input_dim += 1
        
        for raw_name in obs_names:
            if any(extra in raw_name for extra in rl_extra_features):
                if raw_name in index_map:
                    rl_input_dim += index_map[raw_name].n_dims
        
        if rl_input_dim > 0:
            rl_dims.append(rl_input_dim)

    # --------------------------------------------------------------------- #
    # attach_environment — tokenizer construction
    # --------------------------------------------------------------------- #

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Build tokenizers and per-building models from environment metadata.
        
        Uses a two-pass approach:
        Pass 1: Compute global vocabulary (all CA/SRO types across all buildings).
        Pass 2: Build per-building models with pre-allocated projections.
        """

        encoder_config = self._load_encoder_config()
        n_buildings = len(observation_names)
        self.num_agents = n_buildings

        # ------------------------------------------------------------------- #
        # Pass 1: Compute global vocabulary
        # ------------------------------------------------------------------- #
        global_ca_type_dims, global_sro_type_dims, global_type_to_idx, max_rl_input_dim = (
            self._compute_global_vocabulary(
                observation_names, action_names, encoder_config,
                checkpoint_ca_dims=self._checkpoint_ca_dims,
            )
        )

        logger.info(
            "Global vocabulary: CA types={}, SRO types={}, max RL dim={}",
            list(global_ca_type_dims.keys()),
            list(global_sro_type_dims.keys()),
            max_rl_input_dim,
        )

        # ------------------------------------------------------------------- #
        # Pass 2: Build per-building models
        # ------------------------------------------------------------------- #
        models: List[_BuildingModel] = []

        for i in range(n_buildings):
            obs_names = observation_names[i]
            act_names = action_names[i]

            # Build tokenizer with global vocabulary
            tokenizer = ObservationTokenizer(
                observation_names=obs_names,
                action_names=act_names,
                encoder_config=encoder_config,
                tokenizer_config=self.tokenizer_config,
                d_model=self.d_model,
                global_ca_type_dims=global_ca_type_dims,
                global_sro_type_dims=global_sro_type_dims,
                max_rl_input_dim=max_rl_input_dim,
            )

            n_ca = tokenizer.n_ca
            self._n_ca_per_building.append(n_ca)
            self._action_ca_maps.append(tokenizer.action_ca_map)

            # Build CA type index tensor for the actor head using global mapping
            ca_type_names = tokenizer.ca_types
            if ca_type_names:
                ca_type_idx = torch.tensor(
                    [global_type_to_idx[t] for t in ca_type_names], dtype=torch.long,
                )
            else:
                ca_type_idx = None
            self._ca_type_indices.append(ca_type_idx)

            # Build backbone
            backbone = TransformerBackbone(
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
            )

            # Build actor & critic heads with full global vocabulary
            n_ca_types = len(global_type_to_idx)
            actor = ActorHead(self.d_model, self.dim_feedforward, n_ca_types=n_ca_types)
            critic = CriticHead(self.d_model, self.dim_feedforward)

            model = _BuildingModel(tokenizer, backbone, actor, critic)
            model.to(self.device)
            models.append(model)

            # Rollout buffer & optimizer
            self.rollout_buffers.append(
                PPORolloutBuffer(gamma=self.gamma, gae_lambda=self.gae_lambda),
            )
            self.optimizers.append(
                torch.optim.Adam(model.parameters(), lr=self.lr),
            )

            logger.info(
                "Building {}: {} CA tokens, {} SRO tokens, obs_dim={}",
                i, n_ca, tokenizer.n_sro, tokenizer.total_encoded_dims,
            )

        self.building_models = torch.nn.ModuleList(models)
        logger.info("attach_environment complete: {} buildings configured.", n_buildings)

    # --------------------------------------------------------------------- #
    # predict
    # --------------------------------------------------------------------- #

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        """Return actions for the current time step."""

        if self.building_models is None:
            raise RuntimeError("attach_environment() must be called before predict()")

        is_deterministic = deterministic if deterministic is not None else False
        all_actions: List[List[float]] = []

        for i, (model, obs_np) in enumerate(zip(self.building_models, observations)):
            n_ca = self._n_ca_per_building[i]

            if n_ca == 0:
                all_actions.append([])
                continue

            obs_tensor = torch.as_tensor(
                obs_np, dtype=torch.float32, device=self.device,
            ).unsqueeze(0)  # [1, obs_dim]

            with torch.no_grad() if is_deterministic else torch.enable_grad():
                tokenized = model.tokenizer(obs_tensor)
                transformer_out = model.backbone(tokenized)

                ca_type_idx = self._ca_type_indices[i]
                if ca_type_idx is not None:
                    ca_type_idx = ca_type_idx.to(self.device)

                actions, log_probs, entropy = model.actor(
                    transformer_out.ca_embeddings,
                    ca_type_indices=ca_type_idx,
                    deterministic=is_deterministic,
                )
                value = model.critic(transformer_out.pooled)

            # actions: [1, N_ca, 1] → [N_ca]
            actions_flat = actions.squeeze(0).squeeze(-1)

            if not is_deterministic:
                # Store transition data for PPO (reward/done added in update)
                self.rollout_buffers[i].push(
                    observation=obs_tensor.squeeze(0),
                    action=actions.squeeze(0).detach(),
                    log_prob=log_probs.squeeze(0).detach(),
                    reward=0.0,  # placeholder — set in update()
                    value=value.detach(),
                    done=False,  # placeholder — set in update()
                )

            # Reorder from CA-token order to action-name order
            action_ca_map = self._action_ca_maps[i]
            action_list = [0.0] * len(action_ca_map)
            for act_idx, ca_idx in enumerate(action_ca_map):
                action_list[act_idx] = actions_flat[ca_idx].item()

            all_actions.append(action_list)

        return all_actions

    # --------------------------------------------------------------------- #
    # update
    # --------------------------------------------------------------------- #

    def update(
        self,
        observations: List[npt.NDArray[np.float64]],
        actions: List[npt.NDArray[np.float64]],
        rewards: List[float],
        next_observations: List[npt.NDArray[np.float64]],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        """PPO on-policy update."""

        if self.building_models is None:
            return

        done = terminated or truncated

        # Patch the last transition with actual reward and done flag
        for i in range(self.num_agents):
            buf = self.rollout_buffers[i]
            if len(buf) > 0:
                buf.rewards[-1] = rewards[i] if i < len(rewards) else 0.0
                buf.dones[-1] = done

        # Only run PPO update when scheduled
        if not update_step or not initial_exploration_done:
            return

        self._run_ppo_update(next_observations, done, global_learning_step)

    def _run_ppo_update(
        self,
        next_observations: List[npt.NDArray[np.float64]],
        done: bool,
        global_learning_step: int,
    ) -> None:
        """Execute the PPO update loop for all buildings."""

        for i, model in enumerate(self.building_models):
            buf = self.rollout_buffers[i]
            if len(buf) == 0:
                continue

            n_ca = self._n_ca_per_building[i]
            if n_ca == 0:
                buf.clear()
                continue

            # Bootstrap value for GAE
            next_obs = torch.as_tensor(
                next_observations[i], dtype=torch.float32, device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                tok = model.tokenizer(next_obs)
                t_out = model.backbone(tok)
                last_value = model.critic(t_out.pooled).squeeze()

            buf.compute_returns_and_advantages(last_value, done)

            ca_type_idx = self._ca_type_indices[i]
            if ca_type_idx is not None:
                ca_type_idx = ca_type_idx.to(self.device)

            # PPO epochs
            for epoch in range(self.ppo_epochs):
                for batch in buf.get_batches(self.minibatch_size):
                    obs_batch = batch["observations"].to(self.device)
                    old_actions = batch["actions"].to(self.device)
                    old_log_probs = batch["old_log_probs"].to(self.device)
                    returns = batch["returns"].to(self.device)
                    advantages = batch["advantages"].to(self.device)

                    # Forward pass with current policy
                    tokenized = model.tokenizer(obs_batch)
                    t_out = model.backbone(tokenized)
                    values = model.critic(t_out.pooled)

                    new_log_probs, entropy = model.actor.evaluate_actions(
                        t_out.ca_embeddings, old_actions,
                        ca_type_indices=ca_type_idx,
                    )

                    # Sum over CA dimensions for per-sample scalar
                    new_lp_sum = new_log_probs.sum(dim=(1, 2))
                    old_lp_sum = old_log_probs.sum(dim=(1, 2))
                    entropy_sum = entropy.sum(dim=(1, 2))

                    loss, metrics = ppo_loss(
                        new_log_probs=new_lp_sum,
                        old_log_probs=old_lp_sum,
                        advantages=advantages,
                        values=values,
                        returns=returns,
                        entropy=entropy_sum,
                        clip_eps=self.clip_eps,
                        value_coeff=self.value_coeff,
                        entropy_coeff=self.entropy_coeff,
                    )

                    self.optimizers[i].zero_grad()
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    self.optimizers[i].step()

            buf.clear()
            self._training_step += 1

            logger.debug(
                "Building {} PPO update done (step {}): policy_loss={:.4f}, value_loss={:.4f}",
                i, self._training_step,
                metrics.get("policy_loss", 0.0),
                metrics.get("value_loss", 0.0),
            )

    # --------------------------------------------------------------------- #
    # is_initial_exploration_done
    # --------------------------------------------------------------------- #

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return global_learning_step >= self.min_steps_before_update

    # --------------------------------------------------------------------- #
    # export_artifacts
    # --------------------------------------------------------------------- #

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export per-building models to ONNX."""

        if self.building_models is None:
            raise RuntimeError("attach_environment() must be called before export")

        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        metadata: Dict[str, Any] = {"format": "onnx", "artifacts": []}

        for i, model in enumerate(self.building_models):
            n_ca = self._n_ca_per_building[i]
            obs_dim = model.tokenizer.total_encoded_dims

            if n_ca == 0:
                logger.info("Building {} has no CAs — skipping ONNX export.", i)
                continue

            export_path = onnx_dir / f"agent_{i}.onnx"

            # Build a thin wrapper for clean ONNX export
            wrapper = _OnnxExportWrapper(model, self._ca_type_indices[i])
            wrapper.eval()

            dummy_input = torch.randn(1, obs_dim, device=self.device)

            torch.onnx.export(
                wrapper,
                dummy_input,
                str(export_path),
                export_params=True,
                opset_version=DEFAULT_ONNX_OPSET,
                do_constant_folding=True,
                input_names=[f"observation_agent_{i}"],
                output_names=[f"action_agent_{i}"],
                dynamic_axes={
                    f"observation_agent_{i}": {0: "batch_size"},
                    f"action_agent_{i}": {0: "batch_size"},
                },
            )
            logger.info("ONNX exported for building {}: {}", i, export_path)

            relative_path = export_path.relative_to(export_root)
            metadata["artifacts"].append({
                "agent_index": i,
                "path": str(relative_path),
                "format": "onnx",
                "observation_dimension": obs_dim,
                "action_dimension": n_ca,
            })

        return metadata

    # --------------------------------------------------------------------- #
    # Checkpointing
    # --------------------------------------------------------------------- #

    def save_checkpoint(self, output_dir: str, step: int) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "training_step": self._training_step,
            "num_agents": self.num_agents,
        }

        if self.building_models is not None:
            for i, model in enumerate(self.building_models):
                checkpoint[f"model_state_dict_{i}"] = model.state_dict()
            for i, opt in enumerate(self.optimizers):
                checkpoint[f"optimizer_state_dict_{i}"] = opt.state_dict()

        filename = self.checkpoint_artifact or "latest_checkpoint.pth"
        save_path = output_path / filename
        torch.save(checkpoint, save_path)
        logger.info("Checkpoint saved at step {} → {}", step, save_path)
        return str(save_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._training_step = checkpoint.get("training_step", 0)
        
        # Detect cross-topology transfer by checking if checkpoint CA dims were extracted
        # If we extracted CA dims, it means we're loading from a different building topology
        is_cross_topology = self._checkpoint_ca_dims is not None and len(self._checkpoint_ca_dims) > 0

        if self.building_models is not None:
            for i, model in enumerate(self.building_models):
                key = f"model_state_dict_{i}"
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    # Filter transient index buffers for cross-topology compatibility
                    filtered_state_dict = self._filter_transient_buffers(state_dict)
                    # Use strict=False to allow missing/extra keys (cross-topology)
                    model.load_state_dict(filtered_state_dict, strict=False)
                    logger.debug(
                        "Loaded checkpoint for building {} with strict=False (cross-topology compatible)",
                        i,
                    )
            
            # Only load optimizer state if NOT cross-topology transfer
            if not is_cross_topology:
                for i, opt in enumerate(self.optimizers):
                    key = f"optimizer_state_dict_{i}"
                    if key in checkpoint:
                        opt.load_state_dict(checkpoint[key])
                logger.debug("Loaded optimizer state (same topology)")
            else:
                logger.info(
                    "Skipped optimizer state loading due to cross-topology transfer "
                    "(checkpoint CA dims: {})",
                    list(self._checkpoint_ca_dims.keys()) if self._checkpoint_ca_dims else [],
                )

        logger.info("Checkpoint loaded from {}", path)

    @staticmethod
    def _filter_transient_buffers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Filter transient index buffers from checkpoint state dict.
        
        Index buffers (_ca_idx_*, _sro_idx_*, _rl_*_idx) are topology-specific
        and should not be loaded from checkpoints when the topology differs.
        They are re-registered by the tokenizer during attach_environment.
        """
        import re
        
        TRANSIENT_PATTERN = re.compile(
            r"tokenizer\.(_ca_idx_\d+|_sro_idx_\d+|_rl_demand_idx|_rl_gen_idx|_rl_extra_idx)$"
        )
        
        filtered = {
            key: value
            for key, value in state_dict.items()
            if not TRANSIENT_PATTERN.search(key)
        }
        
        n_filtered = len(state_dict) - len(filtered)
        if n_filtered > 0:
            logger.debug("Filtered {} transient buffers from checkpoint", n_filtered)
        
        return filtered

    def extract_checkpoint_ca_dims(self, checkpoint_path: str) -> Dict[str, int]:
        """Extract CA type dimensions from checkpoint before attach_environment.
        
        This allows the agent to use the actual trained dimensions for CA types
        that aren't present in the current building, enabling cross-topology transfer.
        
        Returns
        -------
        Dict[str, int]
            CA type name -> input dimension from checkpoint projection weights.
        """
        from pathlib import Path
        
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning("Checkpoint not found for dimension extraction: {}", path)
            return {}
        
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            # Extract from first building model (all buildings share same global vocab)
            state_dict = checkpoint.get("model_state_dict_0", {})
            
            ca_dims = {}
            # Pattern: tokenizer.ca_projections.<ca_type>.weight with shape [d_model, input_dim]
            for key, tensor in state_dict.items():
                if key.startswith("tokenizer.ca_projections.") and key.endswith(".weight"):
                    # Extract CA type name
                    parts = key.split(".")
                    if len(parts) == 4:  # tokenizer, ca_projections, <type>, weight
                        ca_type = parts[2]
                        input_dim = tensor.shape[1]  # Second dimension is input
                        ca_dims[ca_type] = input_dim
                        logger.debug(
                            "Extracted checkpoint dim for CA type '{}': {}",
                            ca_type, input_dim,
                        )
            
            return ca_dims
        
        except Exception as e:
            logger.warning("Failed to extract checkpoint dimensions: {}", e)
            return {}

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _load_encoder_config() -> Dict[str, Any]:
        if not ENCODER_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Encoder config not found: {ENCODER_CONFIG_PATH}")
        with ENCODER_CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)


class _OnnxExportWrapper(torch.nn.Module):
    """Thin wrapper for ONNX export: flat obs → actions."""

    def __init__(
        self,
        building_model: _BuildingModel,
        ca_type_indices: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.model = building_model
        self.ca_type_indices = ca_type_indices

    def forward(self, encoded_obs: torch.Tensor) -> torch.Tensor:
        tokenized = self.model.tokenizer(encoded_obs)
        t_out = self.model.backbone(tokenized)
        actions, _, _ = self.model.actor(
            t_out.ca_embeddings,
            ca_type_indices=self.ca_type_indices,
            deterministic=True,
        )
        return actions.squeeze(-1)  # [batch, N_ca]
