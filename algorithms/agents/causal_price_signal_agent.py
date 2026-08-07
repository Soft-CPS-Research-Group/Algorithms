"""Causal, deterministic community price signal for a frozen local leaf."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from algorithms.agents.base_agent import BaseAgent


class CausalPriceSignalAgent(BaseAgent):
    """Emit a discount when the community is exporting at a cheap tariff.

    The rule consumes only the current pre-action observation.  It never reads
    an annual schedule, a next observation, or an outcome from the interval it
    is about to control.  A decision is held for ``cc_action_interval``
    simulator steps, matching the hourly Level-1 CC contract at 15 minutes.
    """

    _use_raw_observations: bool = True

    _PRICE = "district__electricity_pricing"
    _FORECASTS = (
        "district__electricity_pricing_predicted_1",
        "district__electricity_pricing_predicted_2",
        "district__electricity_pricing_predicted_3",
    )
    _EXPORT = "district__community_export_power_kw"
    _NET = "district__community_net_power_kw"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.use_raw_observations = True
        hyper = config.get("algorithm", {}).get("hyperparameters") or {}
        self.neutral_multiplier = float(hyper.get("neutral_multiplier", 1.0))
        self.discount_multiplier = float(hyper.get("discount_multiplier", 0.95))
        self.cc_action_interval = int(hyper.get("cc_action_interval", 4))
        self.community_export_threshold_kw = float(
            hyper.get("community_export_threshold_kw", 1.0e-9)
        )
        self.forecast_mean_margin = float(hyper.get("forecast_mean_margin", 0.20))
        self.forecast_min_margin = float(hyper.get("forecast_min_margin", 0.10))
        self.spread_floor_ratio = float(hyper.get("spread_floor_ratio", 0.05))

        self._obs_index: Dict[str, int] = {}
        self._episode_step_context: Optional[int] = None
        self._step_in_interval = 0
        self._has_decision = False
        self._cached_multiplier = self.neutral_multiplier
        self._decision_index = 0
        self._decision_trace: List[Dict[str, Any]] = []

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        _ = action_names, action_space, observation_space, metadata
        if not observation_names or not observation_names[0]:
            raise ValueError("CausalPriceSignal requires named raw observations")
        self._obs_index = {
            str(name): index for index, name in enumerate(observation_names[0])
        }
        missing = [
            name for name in (self._PRICE, *self._FORECASTS)
            if name not in self._obs_index
        ]
        if missing:
            raise ValueError(
                "CausalPriceSignal is missing required observation features: "
                + ", ".join(missing)
            )
        if self._EXPORT not in self._obs_index and self._NET not in self._obs_index:
            raise ValueError(
                "CausalPriceSignal requires community export power or community net power"
            )

    def set_episode_context(
        self,
        *,
        episode_step: Optional[int] = None,
        next_episode_step: Optional[int] = None,
    ) -> None:
        _ = next_episode_step
        normalized = None if episode_step is None else int(episode_step)
        if normalized == 0 and self._episode_step_context != 0:
            self._step_in_interval = 0
            self._has_decision = False
            self._cached_multiplier = self.neutral_multiplier
        self._episode_step_context = normalized

    @staticmethod
    def _finite(value: float, feature: str) -> float:
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(
                f"CausalPriceSignal received non-finite {feature}: {value!r}"
            )
        return parsed

    @classmethod
    def native_cheap(
        cls,
        price: float,
        forecasts: List[float],
        *,
        forecast_mean_margin: float = 0.20,
        forecast_min_margin: float = 0.10,
        spread_floor_ratio: float = 0.05,
    ) -> bool:
        """Match the causal tariff rule used to derive the V5 schedule."""
        if len(forecasts) != 3:
            raise ValueError("CausalPriceSignal requires exactly three forecasts")
        values = [float(value) for value in forecasts]
        forecast_mean = sum(values) / len(values)
        forecast_min = min(values)
        forecast_max = max(values)
        spread = max(
            forecast_max - forecast_min,
            abs(forecast_mean) * float(spread_floor_ratio),
            1.0e-9,
        )
        return bool(
            float(price) <= forecast_mean - float(forecast_mean_margin) * spread
            or float(price) <= forecast_min + float(forecast_min_margin) * spread
        )

    def _value(self, observation: np.ndarray, name: str) -> float:
        if name not in self._obs_index:
            raise ValueError(f"CausalPriceSignal observation feature unavailable: {name}")
        index = self._obs_index[name]
        if index >= len(observation):
            raise ValueError(
                f"CausalPriceSignal observation is too short for feature {name}"
            )
        return self._finite(float(observation[index]), name)

    def _new_decision(self, observations: List[np.ndarray]) -> float:
        if not observations:
            raise ValueError("CausalPriceSignal requires at least one observation")
        observation = np.asarray(observations[0], dtype=np.float64).reshape(-1)
        price = self._value(observation, self._PRICE)
        forecasts = [self._value(observation, name) for name in self._FORECASTS]
        if self._EXPORT in self._obs_index:
            export_kw = max(0.0, self._value(observation, self._EXPORT))
        else:
            export_kw = max(0.0, -self._value(observation, self._NET))

        cheap = self.native_cheap(
            price,
            forecasts,
            forecast_mean_margin=self.forecast_mean_margin,
            forecast_min_margin=self.forecast_min_margin,
            spread_floor_ratio=self.spread_floor_ratio,
        )
        exporting = export_kw > self.community_export_threshold_kw
        multiplier = (
            self.discount_multiplier
            if cheap and exporting
            else self.neutral_multiplier
        )
        self._decision_trace.append(
            {
                "episode_step": self._episode_step_context,
                "decision_index": self._decision_index,
                "price": price,
                "price_forecast_1": forecasts[0],
                "price_forecast_2": forecasts[1],
                "price_forecast_3": forecasts[2],
                "community_export_power_kw": export_kw,
                "cheap": int(cheap),
                "exporting": int(exporting),
                "multiplier": multiplier,
            }
        )
        self._decision_index += 1
        self._cached_multiplier = multiplier
        self._has_decision = True
        return multiplier

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> float:
        _ = deterministic, context
        if self._episode_step_context is None:
            decision_due = not self._has_decision or self._step_in_interval == 0
        else:
            decision_due = (
                not self._has_decision
                or self._episode_step_context % self.cc_action_interval == 0
            )
        if decision_due:
            self._new_decision(observations)
        self._step_in_interval = (
            self._step_in_interval + 1
        ) % self.cc_action_interval
        return self._cached_multiplier

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
        _ = context
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        diagnostic_artifacts: List[Dict[str, Any]] = []
        if self._decision_trace:
            path = root / "decision_trace.csv"
            fields = list(self._decision_trace[0])
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self._decision_trace)
            diagnostic_artifacts.append(
                {"path": path.name, "format": "csv"}
            )
        return {
            "format": "causal_price_signal",
            "output_contract": "causal_global_price_multiplier",
            "rule": "current_native_cheap_and_current_community_export",
            "neutral_multiplier": self.neutral_multiplier,
            "discount_multiplier": self.discount_multiplier,
            "cc_action_interval": self.cc_action_interval,
            "community_export_threshold_kw": self.community_export_threshold_kw,
            "forecast_mean_margin": self.forecast_mean_margin,
            "forecast_min_margin": self.forecast_min_margin,
            "spread_floor_ratio": self.spread_floor_ratio,
            "artifacts": [],
            "diagnostic_artifacts": diagnostic_artifacts,
        }
