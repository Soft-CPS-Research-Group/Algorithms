"""Deterministic price signal for hierarchical integration checks."""

from __future__ import annotations

from bisect import bisect_right
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
        configured_vector = hyperparameters.get("multipliers")
        self.multipliers = (
            None
            if configured_vector is None
            else [float(value) for value in configured_vector]
        )
        configured_schedule = hyperparameters.get("schedule") or []
        self.schedule = [
            (int(entry["start_step"]), float(entry["multiplier"]))
            for entry in configured_schedule
        ]
        self._schedule_starts = [entry[0] for entry in self.schedule]
        self._episode_step = 0

    def set_episode_context(
        self,
        *,
        episode_step: Optional[int] = None,
        next_episode_step: Optional[int] = None,
    ) -> None:
        _ = next_episode_step
        if episode_step is not None:
            self._episode_step = int(episode_step)

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> float | List[float]:
        _ = observations, deterministic, context
        if self.multipliers is not None:
            return list(self.multipliers)
        if self.schedule:
            index = bisect_right(self._schedule_starts, self._episode_step) - 1
            return self.schedule[max(index, 0)][1]
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
        manifest: Dict[str, Any] = {
            "format": "fixed_price_signal",
            "multiplier": self.multiplier,
            "artifacts": [],
        }
        if self.multipliers is not None:
            manifest["multipliers"] = list(self.multipliers)
            manifest["output_contract"] = "per_member_price_multiplier_vector"
        elif self.schedule:
            manifest["schedule"] = [
                {"start_step": start_step, "multiplier": multiplier}
                for start_step, multiplier in self.schedule
            ]
            manifest["output_contract"] = "scheduled_global_price_multiplier"
        else:
            manifest["output_contract"] = "global_price_multiplier"
        return manifest
