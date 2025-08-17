from __future__ import annotations

"""Heuristic rule-based controller for energy flexibility."""

from typing import List

import json
import os

import mlflow
import numpy as np
from loguru import logger

from algorithms.agents.base_agent import BaseAgent


class RBCAgent(BaseAgent):
    """Simple heuristic agent that charges when price is low and discharges otherwise.

    The first element of each observation is assumed to be the normalized
    electricity price. If the price is below ``charge_price_threshold`` the
    agent will charge at the maximum rate; if it is above
    ``discharge_price_threshold`` the agent will discharge at the minimum rate.
    Between these thresholds the agent optionally balances state of charge with
    its neighbours. This agent is stateless and mainly suited for benchmarking
    or as a fallback when no learning is desired.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        hp = config.get("algorithm", {}).get("hyperparameters", {})
        self.num_agents = hp.get("num_agents", 1)
        self.action_space = hp.get("action_space", [[-1.0, 1.0]] * self.num_agents)

        algo_cfg = config.get("algorithm", {})
        self.charge_price_threshold = algo_cfg.get("charge_price_threshold", 0.3)
        self.discharge_price_threshold = algo_cfg.get("discharge_price_threshold", 0.7)
        self.target_soc = algo_cfg.get("target_soc", 0.8)
        self.min_soc = algo_cfg.get("min_soc", 0.2)
        self.neighbor_balance = algo_cfg.get("neighbor_balance", True)
        self.balance_buffer = algo_cfg.get("balance_buffer", 0.05)
        logger.info(
            "RBCAgent initialised with %s agents", self.num_agents
        )

    # ------------------------------------------------------------------
    def predict(
        self, observations: List[np.ndarray], deterministic: bool | None = None
    ) -> List[List[float]]:
        """Return heuristic actions for each agent."""

        socs = [float(obs[1]) if len(obs) > 1 else self.target_soc for obs in observations]
        avg_soc = float(np.mean(socs))

        actions: List[List[float]] = []
        for obs, (low, high), soc in zip(observations, self.action_space, socs):
            price = float(obs[0]) if len(obs) > 0 else 1.0
            connected = bool(obs[2]) if len(obs) > 2 else True

            if not connected:
                action = 0.0
            elif price <= self.charge_price_threshold and soc < self.target_soc:
                action = high
            elif price >= self.discharge_price_threshold and soc > self.min_soc:
                action = low
            elif self.neighbor_balance:
                if soc > avg_soc + self.balance_buffer:
                    action = low
                elif soc < avg_soc - self.balance_buffer:
                    action = high
                else:
                    action = 0.0
            else:
                action = 0.0
            actions.append([float(action)])
        logger.debug(f"RBC actions: {actions}")
        return actions

    # ------------------------------------------------------------------
    def update(self, *args, **kwargs) -> None:  # pragma: no cover - no training
        """RBC has no learning step."""
        return None

    # ------------------------------------------------------------------
    def save_checkpoint(self, step: int) -> None:  # pragma: no cover - no state
        """No state to persist but log an informational message."""
        logger.info("RBCAgent has no trainable parameters; skipping checkpoint save.")

    # ------------------------------------------------------------------
    def load_checkpoint(self) -> None:  # pragma: no cover - no state
        """Nothing to load for RBC."""
        logger.info("RBCAgent has no trainable parameters; skipping checkpoint load.")

    # ------------------------------------------------------------------
    def export_to_onnx(self, log_dir: str) -> None:
        """Export heuristic configuration as a JSON artifact for consistency."""
        export_path = os.path.join(log_dir, "rbc_config.json")
        data = {
            "charge_price_threshold": self.charge_price_threshold,
            "discharge_price_threshold": self.discharge_price_threshold,
            "target_soc": self.target_soc,
            "min_soc": self.min_soc,
            "neighbor_balance": self.neighbor_balance,
            "balance_buffer": self.balance_buffer,
            "action_space": self.action_space,
        }
        os.makedirs(log_dir, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(export_path)
        logger.info("RBCAgent configuration exported to %s", export_path)
