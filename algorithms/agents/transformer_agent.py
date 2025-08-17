from __future__ import annotations

"""Transformer-based agent that handles a variable number of assets."""

from typing import List
import os

import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from algorithms.agents.base_agent import BaseAgent


class _TransformerWrapper(nn.Module):
    """Helper module to make ONNX export easier."""

    def __init__(self, agent: "TransformerAgent") -> None:
        super().__init__()
        self.agent = agent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        x = self.agent.embedding(obs)
        x = self.agent.encoder(x.unsqueeze(1)).squeeze(1)
        return self.agent.head(x)


class TransformerAgent(BaseAgent):
    """Skeleton agent using a Transformer encoder."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        hp = config.get("algorithm", {}).get("hyperparameters", {})
        self.max_agents = hp.get("max_agents", 16)
        self.obs_dim = hp.get("observation_dimensions", 1)
        self.action_dim = hp.get("action_dimensions", 1)
        self.action_space = hp.get("action_space", [[-1.0, 1.0]] * self.max_agents)

        d_model = hp.get("d_model", 64)
        nhead = hp.get("nhead", 4)
        num_layers = hp.get("num_layers", 2)

        self.embedding = nn.Linear(self.obs_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, self.action_dim)

        self.log_dir = config.get("experiment", {}).get("logging", {}).get("log_dir", ".")

    # ------------------------------------------------------------------
    def predict(self, observations: List[np.ndarray], deterministic: bool | None = None) -> List[List[float]]:
        """Compute actions for a variable number of agents."""
        obs = torch.tensor(observations, dtype=torch.float32)  # [num_agents, obs_dim]
        x = self.embedding(obs).unsqueeze(1)  # [num_agents, 1, d_model]
        x = self.encoder(x).squeeze(1)  # [num_agents, d_model]
        out = self.head(x).detach().cpu().numpy()  # [num_agents, action_dim]

        actions: List[List[float]] = []
        for a, (low, high) in zip(out, self.action_space):
            actions.append(np.clip(a, low, high).tolist())
        logger.debug(f"Transformer actions: {actions}")
        return actions[: len(observations)]

    # ------------------------------------------------------------------
    def update(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        """Training loop to be implemented later."""
        return None

    # ------------------------------------------------------------------
    def save_checkpoint(self, step: int) -> None:
        ckpt_path = os.path.join(self.log_dir, "transformer_checkpoint.pth")
        torch.save(
            {
                "embedding": self.embedding.state_dict(),
                "encoder": self.encoder.state_dict(),
                "head": self.head.state_dict(),
            },
            ckpt_path,
        )
        mlflow.log_artifact(ckpt_path)
        logger.info("Saved Transformer checkpoint to %s", ckpt_path)

    # ------------------------------------------------------------------
    def load_checkpoint(self) -> None:
        ckpt_path = os.path.join(self.log_dir, "transformer_checkpoint.pth")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            self.embedding.load_state_dict(state["embedding"])
            self.encoder.load_state_dict(state["encoder"])
            self.head.load_state_dict(state["head"])
            logger.info("Loaded Transformer checkpoint from %s", ckpt_path)
        else:  # pragma: no cover - defensive
            logger.warning("Transformer checkpoint %s not found", ckpt_path)

    # ------------------------------------------------------------------
    def export_to_onnx(self, log_dir: str) -> None:
        """Export the encoder and head as a single ONNX graph."""
        wrapper = _TransformerWrapper(self)
        dummy = torch.zeros((1, self.obs_dim), dtype=torch.float32)
        onnx_path = os.path.join(log_dir, "transformer_agent.onnx")
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "num_agents"}, "actions": {0: "num_agents"}},
        )
        mlflow.log_artifact(onnx_path)
        logger.info("Transformer agent exported to %s", onnx_path)
