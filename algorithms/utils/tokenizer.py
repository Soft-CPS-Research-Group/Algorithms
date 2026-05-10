"""Observation tokenizer for Transformer-based MADDPG.

This module converts flat encoded observations into tokenized format
expected by the Transformer networks. It handles:
- Classifying features into CA (Controllable Asset), SRO (Shared Read-Only),
  and NFC (Non-Flexible Consumption) token types
- Projecting features to the Transformer's embedding dimension (d_model)
- Building token tensors and validity masks

The tokenizer runs AFTER the existing encoder rules in the wrapper,
receiving already-encoded observations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TokenizerConfig:
    """Configuration for feature-to-token mapping.
    
    Attributes:
        d_model: Target embedding dimension for tokens.
        ca_feature_patterns: Regex patterns identifying CA features.
        sro_feature_patterns: Regex patterns identifying SRO features.
        nfc_feature_patterns: Regex patterns identifying NFC features.
    """
    d_model: int = 64
    
    # Patterns for Controllable Asset features (per-asset, need actions)
    ca_feature_patterns: List[str] = field(default_factory=lambda: [
        r"electric_vehicle_soc",
        r"connected_state",
        r"incoming_state",
        r"departure_time",
        r"arrival_time",
        r"required_soc_departure",
        r"estimated_soc_arrival",
        r"electrical_storage",
        r"ev_charger",
    ])
    
    # Patterns for Shared Read-Only observations (context, no actions)
    sro_feature_patterns: List[str] = field(default_factory=lambda: [
        r"outdoor_dry_bulb",
        r"outdoor_relative_humidity",
        r"diffuse_solar",
        r"direct_solar",
        r"carbon_intensity",
        r"electricity_pricing",
        r"solar_generation",
        r"temperature",
        r"humidity",
        r"irradiance",
        r"pricing",
        r"weather",
        r"forecast",
    ])
    
    # Patterns for Non-Flexible Consumption (global context, exactly 1)
    nfc_feature_patterns: List[str] = field(default_factory=lambda: [
        r"non_shiftable_load",
        r"non_flexible",
        r"base_load",
    ])


class FeatureEmbedding(nn.Module):
    """Linear projection from feature space to d_model with LayerNorm.
    
    Args:
        input_dim: Dimension of input features.
        d_model: Target embedding dimension.
    """
    
    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize input features.
        
        Args:
            x: [B, N, input_dim] or [B, input_dim] input features.
            
        Returns:
            embedded: Same shape with last dim = d_model.
        """
        return self.norm(self.projection(x))


class FeatureClassifier:
    """Classifies feature names into CA, SRO, or NFC token types.
    
    Uses regex pattern matching against feature names.
    """
    
    TYPE_CA = "CA"
    TYPE_SRO = "SRO"
    TYPE_NFC = "NFC"
    TYPE_UNKNOWN = "UNKNOWN"
    
    def __init__(self, config: TokenizerConfig) -> None:
        """Initialize classifier with patterns from config.
        
        Args:
            config: TokenizerConfig with pattern lists.
        """
        self.ca_patterns = [re.compile(p, re.IGNORECASE) for p in config.ca_feature_patterns]
        self.sro_patterns = [re.compile(p, re.IGNORECASE) for p in config.sro_feature_patterns]
        self.nfc_patterns = [re.compile(p, re.IGNORECASE) for p in config.nfc_feature_patterns]
    
    def classify(self, feature_name: str) -> str:
        """Classify a single feature name.
        
        Args:
            feature_name: Name of the feature to classify.
            
        Returns:
            One of TYPE_CA, TYPE_SRO, TYPE_NFC, or TYPE_UNKNOWN.
        """
        # Check CA patterns first (most specific)
        for pattern in self.ca_patterns:
            if pattern.search(feature_name):
                return self.TYPE_CA
        
        # Check NFC patterns (before SRO to avoid false matches)
        for pattern in self.nfc_patterns:
            if pattern.search(feature_name):
                return self.TYPE_NFC
        
        # Check SRO patterns
        for pattern in self.sro_patterns:
            if pattern.search(feature_name):
                return self.TYPE_SRO
        
        # Default to SRO for unknown features (safe assumption)
        return self.TYPE_SRO
    
    def classify_all(
        self, feature_names: List[str]
    ) -> Dict[str, List[Tuple[int, str]]]:
        """Classify all feature names and group by type.
        
        Args:
            feature_names: List of feature names.
            
        Returns:
            Dict mapping type to list of (index, name) tuples.
        """
        result: Dict[str, List[Tuple[int, str]]] = {
            self.TYPE_CA: [],
            self.TYPE_SRO: [],
            self.TYPE_NFC: [],
        }
        
        for idx, name in enumerate(feature_names):
            ftype = self.classify(name)
            if ftype in result:
                result[ftype].append((idx, name))
        
        return result


class ObservationTokenizer(nn.Module):
    """Converts encoded observations into token embeddings.
    
    This tokenizer:
    1. Classifies features by type (CA, SRO, NFC)
    2. Projects each feature group to d_model dimension
    3. Produces token tensors and validity masks for Transformer input
    
    Args:
        config: TokenizerConfig with d_model and feature patterns.
        observation_names: Per-agent list of feature names from wrapper.
    """
    
    def __init__(
        self,
        config: TokenizerConfig,
        observation_names: List[List[str]],
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.observation_names = observation_names
        self.num_agents = len(observation_names)
        
        # Classify features for each agent
        self.classifier = FeatureClassifier(config)
        self.feature_maps: List[Dict[str, List[Tuple[int, str]]]] = []
        
        for agent_names in observation_names:
            self.feature_maps.append(self.classifier.classify_all(agent_names))
        
        # Calculate dimensions for each token type
        self._compute_dimensions()
        
        # Create embedding layers
        self._create_embeddings()
    
    def _compute_dimensions(self) -> None:
        """Compute feature dimensions for each token type."""
        # Each agent is one CA token - aggregate CA features per agent
        # SRO features are shared across agents - use first agent's SRO features
        # NFC is exactly one token - use first agent's NFC features
        
        self.ca_dims: List[int] = []
        for fmap in self.feature_maps:
            ca_indices = [idx for idx, _ in fmap.get(FeatureClassifier.TYPE_CA, [])]
            self.ca_dims.append(len(ca_indices))
        
        # SRO and NFC from first agent (assumed shared)
        first_fmap = self.feature_maps[0] if self.feature_maps else {}
        sro_indices = [idx for idx, _ in first_fmap.get(FeatureClassifier.TYPE_SRO, [])]
        nfc_indices = [idx for idx, _ in first_fmap.get(FeatureClassifier.TYPE_NFC, [])]
        
        self.sro_dim = len(sro_indices)
        self.nfc_dim = len(nfc_indices)
        
        # Store indices for fast extraction
        self.ca_indices: List[List[int]] = []
        for fmap in self.feature_maps:
            self.ca_indices.append([idx for idx, _ in fmap.get(FeatureClassifier.TYPE_CA, [])])
        
        self.sro_indices = sro_indices
        self.nfc_indices = nfc_indices
    
    def _create_embeddings(self) -> None:
        """Create embedding layers for each token type."""
        # Per-agent CA embeddings (may have different dims due to encoders)
        self.ca_embeddings = nn.ModuleList()
        for dim in self.ca_dims:
            embed_dim = max(dim, 1)  # Handle empty case
            self.ca_embeddings.append(FeatureEmbedding(embed_dim, self.d_model))
        
        # Shared SRO embedding
        sro_embed_dim = max(self.sro_dim, 1)
        self.sro_embedding = FeatureEmbedding(sro_embed_dim, self.d_model)
        
        # NFC embedding
        nfc_embed_dim = max(self.nfc_dim, 1)
        self.nfc_embedding = FeatureEmbedding(nfc_embed_dim, self.d_model)
    
    def get_feature_indices(self) -> Dict[str, Any]:
        """Return feature index mappings for debugging.
        
        Returns:
            Dict with CA, SRO, NFC indices.
        """
        return {
            "ca_indices": self.ca_indices,
            "sro_indices": self.sro_indices,
            "nfc_indices": self.nfc_indices,
            "ca_dims": self.ca_dims,
            "sro_dim": self.sro_dim,
            "nfc_dim": self.nfc_dim,
        }
    
    def tokenize(
        self,
        observations: List[np.ndarray],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert encoded observations into token embeddings.
        
        Args:
            observations: Per-agent encoded observations from wrapper.
            device: Target device for tensors.
            
        Returns:
            Tuple of:
            - ca_tokens: [B, N_ca, d_model] CA token embeddings
            - sro_tokens: [B, N_sro, d_model] SRO token embeddings
            - nfc_token: [B, 1, d_model] NFC token embedding
            - ca_mask: [B, N_ca] validity mask (True=valid)
            - sro_mask: [B, N_sro] validity mask (True=valid)
        """
        if device is None:
            device = next(self.parameters()).device
        
        B = 1  # Single timestep, batch dim for Transformer
        N_ca = len(observations)  # Number of agents = number of CA tokens
        N_sro = 1  # All SRO features aggregated into one token per observation
        
        # Extract and embed CA features (one token per agent)
        ca_list = []
        for agent_idx, obs in enumerate(observations):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            
            # Extract CA features for this agent
            ca_idxs = self.ca_indices[agent_idx]
            if ca_idxs:
                ca_features = obs_tensor[ca_idxs]
            else:
                ca_features = torch.zeros(1, device=device)
            
            # Embed
            ca_embedded = self.ca_embeddings[agent_idx](ca_features)  # [d_model]
            ca_list.append(ca_embedded)
        
        ca_tokens = torch.stack(ca_list, dim=0).unsqueeze(0)  # [B, N_ca, d_model]
        
        # Extract SRO features (use first agent's observation, shared)
        first_obs = torch.as_tensor(observations[0], dtype=torch.float32, device=device)
        if self.sro_indices:
            sro_features = first_obs[self.sro_indices]
        else:
            sro_features = torch.zeros(1, device=device)
        
        sro_embedded = self.sro_embedding(sro_features)  # [d_model]
        sro_tokens = sro_embedded.unsqueeze(0).unsqueeze(0)  # [B, N_sro=1, d_model]
        
        # Extract NFC features
        if self.nfc_indices:
            nfc_features = first_obs[self.nfc_indices]
        else:
            nfc_features = torch.zeros(1, device=device)
        
        nfc_embedded = self.nfc_embedding(nfc_features)  # [d_model]
        nfc_token = nfc_embedded.unsqueeze(0).unsqueeze(0)  # [B, 1, d_model]
        
        # Masks (all valid in single-step case)
        ca_mask = torch.ones(B, N_ca, dtype=torch.bool, device=device)
        sro_mask = torch.ones(B, N_sro, dtype=torch.bool, device=device)
        
        return ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask
    
    def tokenize_batch(
        self,
        observations_batch: List[List[np.ndarray]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a batch of observations into token embeddings.
        
        Args:
            observations_batch: List of per-agent observation lists.
            device: Target device for tensors.
            
        Returns:
            Same as tokenize() but with B = len(observations_batch).
        """
        if device is None:
            device = next(self.parameters()).device
        
        B = len(observations_batch)
        N_ca = len(observations_batch[0]) if observations_batch else 0
        N_sro = 1
        
        all_ca = []
        all_sro = []
        all_nfc = []
        
        for batch_idx, observations in enumerate(observations_batch):
            # CA tokens
            ca_list = []
            for agent_idx, obs in enumerate(observations):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                ca_idxs = self.ca_indices[agent_idx]
                if ca_idxs:
                    ca_features = obs_tensor[ca_idxs]
                else:
                    ca_features = torch.zeros(1, device=device)
                ca_embedded = self.ca_embeddings[agent_idx](ca_features)
                ca_list.append(ca_embedded)
            
            all_ca.append(torch.stack(ca_list, dim=0))  # [N_ca, d_model]
            
            # SRO and NFC from first agent
            first_obs = torch.as_tensor(observations[0], dtype=torch.float32, device=device)
            
            if self.sro_indices:
                sro_features = first_obs[self.sro_indices]
            else:
                sro_features = torch.zeros(1, device=device)
            all_sro.append(self.sro_embedding(sro_features))  # [d_model]
            
            if self.nfc_indices:
                nfc_features = first_obs[self.nfc_indices]
            else:
                nfc_features = torch.zeros(1, device=device)
            all_nfc.append(self.nfc_embedding(nfc_features))  # [d_model]
        
        ca_tokens = torch.stack(all_ca, dim=0)  # [B, N_ca, d_model]
        sro_tokens = torch.stack(all_sro, dim=0).unsqueeze(1)  # [B, 1, d_model]
        nfc_token = torch.stack(all_nfc, dim=0).unsqueeze(1)  # [B, 1, d_model]
        
        ca_mask = torch.ones(B, N_ca, dtype=torch.bool, device=device)
        sro_mask = torch.ones(B, N_sro, dtype=torch.bool, device=device)
        
        return ca_tokens, sro_tokens, nfc_token, ca_mask, sro_mask
