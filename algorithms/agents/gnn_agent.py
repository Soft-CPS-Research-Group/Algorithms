from __future__ import annotations

"""Graph neural network based agent for decentralised control."""

from typing import List
import os

import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from algorithms.agents.base_agent import BaseAgent


class SimpleGNN(nn.Module):
    """Minimal message-passing network."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.matmul(adj, x)  # aggregate neighbour information
        h = torch.relu(self.fc1(h))
        return self.fc2(h)


class GNNAgent(BaseAgent):
    """Skeleton agent that uses a simple GNN to compute actions."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        hp = config.get("algorithm", {}).get("hyperparameters", {})
        self.num_agents = hp.get("num_agents", 1)
        self.obs_dim = hp.get("observation_dimensions", 1)
        self.action_dim = hp.get("action_dimensions", 1)
        self.action_space = hp.get("action_space", [[-1.0, 1.0]] * self.num_agents)
        hidden_dim = hp.get("hidden_dim", 64)

        adj = hp.get("adjacency_matrix")
        if adj is None:
            adj = np.eye(self.num_agents, dtype=np.float32)
        self.adj = torch.tensor(adj, dtype=torch.float32)

        self.model = SimpleGNN(self.obs_dim, hidden_dim, self.action_dim)
        self.log_dir = config.get("experiment", {}).get("logging", {}).get("log_dir", ".")

    # ------------------------------------------------------------------
    def predict(self, observations: List[np.ndarray], deterministic: bool | None = None) -> List[List[float]]:
        """Compute actions using the GNN model."""
        obs = torch.tensor(observations, dtype=torch.float32)
        out = self.model(obs, self.adj).detach().cpu().numpy()
        actions: List[List[float]] = []
        for a, (low, high) in zip(out, self.action_space):
            actions.append(np.clip(a, low, high).tolist())
        logger.debug(f"GNN actions: {actions}")
        return actions

    # ------------------------------------------------------------------
    def update(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        """Training loop to be implemented in future iterations."""
        return None

    # ------------------------------------------------------------------
    def save_checkpoint(self, step: int) -> None:
        ckpt_path = os.path.join(self.log_dir, "gnn_checkpoint.pth")
        torch.save(self.model.state_dict(), ckpt_path)
        mlflow.log_artifact(ckpt_path)
        logger.info("Saved GNN checkpoint to %s", ckpt_path)

    # ------------------------------------------------------------------
    def load_checkpoint(self) -> None:
        ckpt_path = os.path.join(self.log_dir, "gnn_checkpoint.pth")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(state)
            logger.info("Loaded GNN checkpoint from %s", ckpt_path)
        else:  # pragma: no cover - defensive
            logger.warning("GNN checkpoint %s not found", ckpt_path)

    # ------------------------------------------------------------------
    def export_to_onnx(self, log_dir: str) -> None:
        """Export the policy network to ONNX for serving."""
        dummy_obs = torch.zeros((self.num_agents, self.obs_dim), dtype=torch.float32)
        onnx_path = os.path.join(log_dir, "gnn_agent.onnx")
        torch.onnx.export(
            self.model,
            (dummy_obs, self.adj),
            onnx_path,
            input_names=["obs", "adj"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "num_agents"}, "actions": {0: "num_agents"}},
        )
        mlflow.log_artifact(onnx_path)
        logger.info("GNN agent exported to %s", onnx_path)
