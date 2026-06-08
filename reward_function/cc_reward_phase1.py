"""CC Phase 1 reward — placeholder that returns zeros.

The Community Coordinator computes its own reward internally:

    reward_i = -o1_i × (price_now - ref_price)
    ref_price = mean(price_now, pred_1, pred_2, pred_3)

This reward is computed inside CommunityCoordinatorAgent.update() using
the cached decision (o1) and raw community context — fully independent
of env outcomes or Building Agent actions.

This class exists so the simulator has a valid reward_function entry and
Building Agents (Phase 2) can receive a separate reward if needed.
It returns 0.0 for every building.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Union

from citylearn.reward_function import RewardFunction


class CCRewardPhase1(RewardFunction):
    """Placeholder reward — CC uses internal decision-quality reward."""

    def calculate(
        self, observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        return [0.0 for _ in observations]
