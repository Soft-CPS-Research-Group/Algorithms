"""Transformer-compatible replay buffer with variable-length token sequences.

This module provides a replay buffer optimized for storing and sampling
tokenized experiences with variable N_ca and N_sro cardinalities.

The default strategy pads sequences to max cardinality and stores attention
masks alongside, enabling efficient batched training.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch


class TransformerExperience(NamedTuple):
    """Single experience tuple for Transformer replay buffer."""
    ca_tokens: torch.Tensor        # [N_ca, d_model]
    sro_tokens: torch.Tensor       # [N_sro, d_model]
    nfc_token: torch.Tensor        # [1, d_model]
    ca_mask: torch.Tensor          # [N_ca]
    sro_mask: torch.Tensor         # [N_sro]
    actions: torch.Tensor          # [N_ca, action_dim]
    rewards: torch.Tensor          # [1]
    next_ca_tokens: torch.Tensor   # [N_ca, d_model]
    next_sro_tokens: torch.Tensor  # [N_sro, d_model]
    next_nfc_token: torch.Tensor   # [1, d_model]
    next_ca_mask: torch.Tensor     # [N_ca]
    next_sro_mask: torch.Tensor    # [N_sro]
    done: torch.Tensor             # [1]
    n_ca: int                      # Actual number of CAs
    n_sro: int                     # Actual number of SROs


@dataclass
class TransformerReplayBufferConfig:
    """Configuration for TransformerReplayBuffer.
    
    Attributes:
        capacity: Maximum number of experiences to store.
        batch_size: Number of experiences per sample batch.
        max_ca: Maximum number of CA tokens.
        max_sro: Maximum number of SRO tokens.
        d_model: Embedding dimension.
        action_dim: Action dimension per CA.
    """
    capacity: int = 100000
    batch_size: int = 256
    max_ca: int = 64
    max_sro: int = 32
    d_model: int = 64
    action_dim: int = 1


class ReplayBufferStrategy(ABC):
    """Abstract strategy for replay buffer storage and batching."""
    
    @abstractmethod
    def collate_batch(
        self,
        experiences: List[TransformerExperience],
        config: TransformerReplayBufferConfig,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Collate experiences into a batched dictionary.
        
        Args:
            experiences: List of experiences to collate.
            config: Buffer configuration.
            device: Target device for tensors.
            
        Returns:
            Dict with batched tensors for training.
        """


class PaddedStorageStrategy(ReplayBufferStrategy):
    """Strategy that pads sequences to max cardinality.
    
    Simple approach suitable for moderate max_tokens values.
    Stores masks to indicate valid vs padded positions.
    """
    
    def collate_batch(
        self,
        experiences: List[TransformerExperience],
        config: TransformerReplayBufferConfig,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Collate with padding to max cardinality.
        
        Returns dict with keys:
        - ca_tokens: [B, max_ca, d_model]
        - sro_tokens: [B, max_sro, d_model]
        - nfc_token: [B, 1, d_model]
        - ca_mask: [B, max_ca]
        - sro_mask: [B, max_sro]
        - actions: [B, max_ca, action_dim]
        - rewards: [B, 1]
        - next_* versions of above
        - dones: [B, 1]
        - n_ca: [B] actual counts
        - n_sro: [B] actual counts
        """
        B = len(experiences)
        max_ca = config.max_ca
        max_sro = config.max_sro
        d_model = config.d_model
        action_dim = config.action_dim
        
        # Initialize padded tensors
        ca_tokens = torch.zeros(B, max_ca, d_model, device=device)
        sro_tokens = torch.zeros(B, max_sro, d_model, device=device)
        nfc_token = torch.zeros(B, 1, d_model, device=device)
        ca_mask = torch.zeros(B, max_ca, dtype=torch.bool, device=device)
        sro_mask = torch.zeros(B, max_sro, dtype=torch.bool, device=device)
        actions = torch.zeros(B, max_ca, action_dim, device=device)
        rewards = torch.zeros(B, 1, device=device)
        
        next_ca_tokens = torch.zeros(B, max_ca, d_model, device=device)
        next_sro_tokens = torch.zeros(B, max_sro, d_model, device=device)
        next_nfc_token = torch.zeros(B, 1, d_model, device=device)
        next_ca_mask = torch.zeros(B, max_ca, dtype=torch.bool, device=device)
        next_sro_mask = torch.zeros(B, max_sro, dtype=torch.bool, device=device)
        dones = torch.zeros(B, 1, device=device)
        
        n_ca_list = []
        n_sro_list = []
        
        for i, exp in enumerate(experiences):
            n_ca = exp.n_ca
            n_sro = exp.n_sro
            n_ca_list.append(n_ca)
            n_sro_list.append(n_sro)
            
            # Copy to padded tensors
            ca_tokens[i, :n_ca] = exp.ca_tokens.to(device)
            sro_tokens[i, :n_sro] = exp.sro_tokens.to(device)
            nfc_token[i] = exp.nfc_token.to(device)
            ca_mask[i, :n_ca] = True
            sro_mask[i, :n_sro] = True
            actions[i, :n_ca] = exp.actions.to(device)
            rewards[i] = exp.rewards.to(device)
            
            next_ca_tokens[i, :n_ca] = exp.next_ca_tokens.to(device)
            next_sro_tokens[i, :n_sro] = exp.next_sro_tokens.to(device)
            next_nfc_token[i] = exp.next_nfc_token.to(device)
            next_ca_mask[i, :n_ca] = True
            next_sro_mask[i, :n_sro] = True
            dones[i] = exp.done.to(device)
        
        return {
            "ca_tokens": ca_tokens,
            "sro_tokens": sro_tokens,
            "nfc_token": nfc_token,
            "ca_mask": ca_mask,
            "sro_mask": sro_mask,
            "actions": actions,
            "rewards": rewards,
            "next_ca_tokens": next_ca_tokens,
            "next_sro_tokens": next_sro_tokens,
            "next_nfc_token": next_nfc_token,
            "next_ca_mask": next_ca_mask,
            "next_sro_mask": next_sro_mask,
            "dones": dones,
            "n_ca": torch.tensor(n_ca_list, dtype=torch.long, device=device),
            "n_sro": torch.tensor(n_sro_list, dtype=torch.long, device=device),
        }


class TransformerReplayBuffer:
    """Replay buffer for Transformer-based agents.
    
    Stores tokenized experiences and provides batched sampling
    with proper padding and masking for variable-length sequences.
    
    Args:
        config: Buffer configuration with capacity, batch_size, etc.
        strategy: Storage/batching strategy. Defaults to PaddedStorageStrategy.
    """
    
    def __init__(
        self,
        config: TransformerReplayBufferConfig,
        strategy: Optional[ReplayBufferStrategy] = None,
    ) -> None:
        self.config = config
        self.strategy = strategy or PaddedStorageStrategy()
        self.buffer: Deque[TransformerExperience] = deque(maxlen=config.capacity)
    
    def push(
        self,
        ca_tokens: torch.Tensor,
        sro_tokens: torch.Tensor,
        nfc_token: torch.Tensor,
        ca_mask: torch.Tensor,
        sro_mask: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_ca_tokens: torch.Tensor,
        next_sro_tokens: torch.Tensor,
        next_nfc_token: torch.Tensor,
        next_ca_mask: torch.Tensor,
        next_sro_mask: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Store a single experience.
        
        All inputs should have batch dimension of 1 (single timestep).
        Tensors are stored on CPU to save GPU memory.
        
        Args:
            ca_tokens: [1, N_ca, d_model] CA token embeddings.
            sro_tokens: [1, N_sro, d_model] SRO token embeddings.
            nfc_token: [1, 1, d_model] NFC token embedding.
            ca_mask: [1, N_ca] CA validity mask.
            sro_mask: [1, N_sro] SRO validity mask.
            actions: [1, N_ca, action_dim] Actions taken.
            rewards: [1, 1] or scalar reward.
            next_ca_tokens: [1, N_ca, d_model] Next CA tokens.
            next_sro_tokens: [1, N_sro, d_model] Next SRO tokens.
            next_nfc_token: [1, 1, d_model] Next NFC token.
            next_ca_mask: [1, N_ca] Next CA mask.
            next_sro_mask: [1, N_sro] Next SRO mask.
            done: [1, 1] or scalar done flag.
        """
        # Squeeze batch dimension and move to CPU for storage
        def squeeze_and_cpu(t: torch.Tensor) -> torch.Tensor:
            if t.dim() > 0 and t.shape[0] == 1:
                t = t.squeeze(0)
            return t.detach().cpu()
        
        ca_t = squeeze_and_cpu(ca_tokens)
        sro_t = squeeze_and_cpu(sro_tokens)
        nfc_t = squeeze_and_cpu(nfc_token)
        ca_m = squeeze_and_cpu(ca_mask)
        sro_m = squeeze_and_cpu(sro_mask)
        acts = squeeze_and_cpu(actions)
        rews = squeeze_and_cpu(rewards)
        if rews.dim() == 0:
            rews = rews.unsqueeze(0)
        
        next_ca_t = squeeze_and_cpu(next_ca_tokens)
        next_sro_t = squeeze_and_cpu(next_sro_tokens)
        next_nfc_t = squeeze_and_cpu(next_nfc_token)
        next_ca_m = squeeze_and_cpu(next_ca_mask)
        next_sro_m = squeeze_and_cpu(next_sro_mask)
        d = squeeze_and_cpu(done)
        if d.dim() == 0:
            d = d.unsqueeze(0)
        
        # Derive actual counts from masks
        n_ca = int(ca_m.sum().item())
        n_sro = int(sro_m.sum().item())
        
        exp = TransformerExperience(
            ca_tokens=ca_t,
            sro_tokens=sro_t,
            nfc_token=nfc_t,
            ca_mask=ca_m,
            sro_mask=sro_m,
            actions=acts,
            rewards=rews,
            next_ca_tokens=next_ca_t,
            next_sro_tokens=next_sro_t,
            next_nfc_token=next_nfc_t,
            next_ca_mask=next_ca_m,
            next_sro_mask=next_sro_m,
            done=d,
            n_ca=n_ca,
            n_sro=n_sro,
        )
        
        self.buffer.append(exp)
    
    def sample(
        self,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample. Defaults to config.batch_size.
            device: Target device for tensors. Defaults to CPU.
            
        Returns:
            Dict with batched tensors for training.
            
        Raises:
            ValueError: If buffer has fewer samples than batch_size.
        """
        batch_size = batch_size or self.config.batch_size
        device = device or torch.device("cpu")
        
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}"
            )
        
        experiences = random.sample(list(self.buffer), batch_size)
        return self.strategy.collate_batch(experiences, self.config, device)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def get_state(self) -> Dict[str, Any]:
        """Return serializable state for checkpointing.
        
        Returns:
            Dict with buffer contents.
        """
        return {
            "experiences": list(self.buffer),
            "config": {
                "capacity": self.config.capacity,
                "batch_size": self.config.batch_size,
                "max_ca": self.config.max_ca,
                "max_sro": self.config.max_sro,
                "d_model": self.config.d_model,
                "action_dim": self.config.action_dim,
            },
        }
    
    def set_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore buffer from checkpointed state.
        
        Args:
            state: State dict from get_state(). If None, buffer is cleared.
        """
        if state is None:
            self.buffer.clear()
            return
        
        experiences = state.get("experiences", [])
        self.buffer.clear()
        for exp in experiences:
            if isinstance(exp, TransformerExperience):
                self.buffer.append(exp)
            elif isinstance(exp, (tuple, list)):
                # Reconstruct from tuple
                self.buffer.append(TransformerExperience(*exp))
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
