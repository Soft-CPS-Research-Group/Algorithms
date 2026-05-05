"""Community Coordinator reward function.

Objective: teach the CA to perform price arbitrage at the community level.

    reward = -net_electricity_consumption * electricity_pricing

Intuition
---------
- When price is HIGH and consumption is HIGH  → large negative reward → bad
- When price is HIGH and consumption is LOW   → small negative reward → good
  (battery discharged to avoid buying expensive grid energy)
- When price is LOW and consumption is HIGH   → small negative reward → acceptable
  (battery charging on cheap energy)
- Export (negative net consumption) at high price → positive reward → great

This is intentionally simple. The CA learns one thing first: shift consumption
away from expensive hours. Complexity (carbon, solar forecast, peak shaving)
can be layered in later.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Union

from citylearn.reward_function import RewardFunction


class CCReward(RewardFunction):
    """Price-arbitrage reward for the Community Coordinator Agent."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _building_reward(self, observation: Mapping[str, Union[int, float]]) -> float:
        net = self._safe_float(observation.get("net_electricity_consumption"))
        price = self._safe_float(observation.get("electricity_pricing"))
        return -net * price

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        return [self._building_reward(obs) for obs in observations]