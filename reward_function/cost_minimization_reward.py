"""Simple price-aware cost minimization reward."""

from __future__ import annotations

from typing import Any, List, Mapping, Union

from citylearn.reward_function import RewardFunction


class CostMinimizationReward(RewardFunction):
    """Reward that directly optimizes electricity cost with export credit."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        export_credit_ratio: float = 0.8,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.export_credit_ratio = float(export_credit_ratio)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _building_reward(self, observation: Mapping[str, Union[int, float]]) -> float:
        net_consumption = self._safe_float(observation.get("net_electricity_consumption"), default=0.0)
        electricity_price = max(self._safe_float(observation.get("electricity_pricing"), default=0.0), 0.0)

        import_cost = max(net_consumption, 0.0) * electricity_price
        export_credit = max(-net_consumption, 0.0) * electricity_price * self.export_credit_ratio

        return -(import_cost - export_credit)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = [self._building_reward(observation) for observation in observations]

        if self.central_agent:
            return [sum(reward_list)]

        return reward_list
