"""Reward-function registry used by run_experiment."""

from __future__ import annotations

from typing import Dict, Type

from citylearn.reward_function import RewardFunction

from reward_function.V2G_Reward import V2GPenaltyReward
from reward_function.cost_hard_constraint_reward import CostHardConstraintReward
from reward_function.cost_minimization_reward import CostMinimizationReward

REWARD_FUNCTION_MAP: Dict[str, Type[RewardFunction]] = {
    "RewardFunction": RewardFunction,
    "V2GPenaltyReward": V2GPenaltyReward,
    "CostMinimizationReward": CostMinimizationReward,
    "CostHardConstraintReward": CostHardConstraintReward,
}


def get_available_reward_function_names() -> list[str]:
    return sorted(REWARD_FUNCTION_MAP.keys())
