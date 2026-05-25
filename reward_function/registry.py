"""Reward-function registry used by run_experiment."""

from __future__ import annotations

from typing import Dict, Type

from citylearn.reward_function import RewardFunction

from reward_function.V2G_Reward import V2GPenaltyReward
from reward_function.cost_hard_constraint_reward import (
    CostHardConstraintReward,
    CostServiceCostBalancedRewardV3,
    CostServiceCommunityBandRewardV4,
    CostServiceCommunityBatteryValueRewardV43,
    CostServiceCommunityFeasibleServiceRewardV45,
    CostServiceCommunityFeasiblePrecisionRewardV46,
    CostServiceCommunityFeasiblePrecisionRewardV47,
    CostServiceCommunityDeadlineValueRewardV50,
    CostServiceCommunityPeakDeadlineRewardV52,
    CostServiceCommunityPrecisionValueRewardV51,
    CostServiceCommunityStorageValueRewardV49,
    CostServiceCommunityServiceBandRewardV42,
    CostServiceCommunitySmoothServiceRewardV44,
    CostServiceCommunityStorageBandRewardV41,
    CostServiceGuardRewardV2,
)
from reward_function.cost_minimization_reward import CostMinimizationReward

REWARD_FUNCTION_MAP: Dict[str, Type[RewardFunction]] = {
    "RewardFunction": RewardFunction,
    "V2GPenaltyReward": V2GPenaltyReward,
    "CostMinimizationReward": CostMinimizationReward,
    "CostHardConstraintReward": CostHardConstraintReward,
    "CostServiceGuardRewardV2": CostServiceGuardRewardV2,
    "CostServiceCostBalancedRewardV3": CostServiceCostBalancedRewardV3,
    "CostServiceCommunityBandRewardV4": CostServiceCommunityBandRewardV4,
    "CostServiceCommunityStorageBandRewardV41": CostServiceCommunityStorageBandRewardV41,
    "CostServiceCommunityServiceBandRewardV42": CostServiceCommunityServiceBandRewardV42,
    "CostServiceCommunityBatteryValueRewardV43": CostServiceCommunityBatteryValueRewardV43,
    "CostServiceCommunitySmoothServiceRewardV44": CostServiceCommunitySmoothServiceRewardV44,
    "CostServiceCommunityFeasibleServiceRewardV45": CostServiceCommunityFeasibleServiceRewardV45,
    "CostServiceCommunityFeasiblePrecisionRewardV46": CostServiceCommunityFeasiblePrecisionRewardV46,
    "CostServiceCommunityFeasiblePrecisionRewardV47": CostServiceCommunityFeasiblePrecisionRewardV47,
    "CostServiceCommunityStorageValueRewardV49": CostServiceCommunityStorageValueRewardV49,
    "CostServiceCommunityDeadlineValueRewardV50": CostServiceCommunityDeadlineValueRewardV50,
    "CostServiceCommunityPrecisionValueRewardV51": CostServiceCommunityPrecisionValueRewardV51,
    "CostServiceCommunityPeakDeadlineRewardV52": CostServiceCommunityPeakDeadlineRewardV52,
}


def get_available_reward_function_names() -> list[str]:
    return sorted(REWARD_FUNCTION_MAP.keys())
