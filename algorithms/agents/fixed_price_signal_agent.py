"""Deterministic price signal for hierarchical integration checks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from algorithms.agents.base_agent import BaseAgent


class FixedPriceSignalAgent(BaseAgent):
    """Emit one constant effective-price multiplier to the next pipeline stage.

    This manager is deliberately observation-independent.  Its main purpose is
    the neutral ``multiplier=1.0`` control run that proves a frozen local policy
    behaves identically before an adaptive community coordinator is introduced.
    """

    _use_raw_observations: bool = True

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.use_raw_observations = True
        hyperparameters = config.get("algorithm", {}).get("hyperparameters") or {}
        self.multiplier = float(hyperparameters.get("multiplier", 1.0))

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> float:
        _ = observations, deterministic, context
        return self.multiplier

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
        _ = (
            observations,
            actions,
            rewards,
            next_observations,
            terminated,
            truncated,
            update_target_step,
            global_learning_step,
            update_step,
            initial_exploration_done,
        )

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = output_dir, context
        return {
            "format": "fixed_price_signal",
            "multiplier": self.multiplier,
            "artifacts": [],
        }
