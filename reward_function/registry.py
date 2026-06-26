"""Reward-function registry used by run_experiment."""

from __future__ import annotations

from typing import Dict, Type

from citylearn.reward_function import RewardFunction

from reward_function.V2G_Reward import V2GPenaltyReward
from reward_function.ba_reward import BAReward
from reward_function.cc_reward import CCReward
from reward_function.cc_reward_level1 import CCRewardLevel1
from reward_function.cc_reward_level2 import CCRewardLevel2
from reward_function.cc_reward_phase1 import CCRewardPhase1
from reward_function.cost_hard_constraint_reward import (
    CostHardConstraintReward,
    CostServiceCostBalancedRewardV3,
    CostServiceCommunityBandRewardV4,
    CostServiceCommunityBatteryValueRewardV43,
    CostServiceCommunityFeasibleServiceRewardV45,
    CostServiceCommunityFeasiblePrecisionRewardV46,
    CostServiceCommunityFeasiblePrecisionRewardV47,
    CostServiceCommunityDeadlineValueRewardV50,
    CostServiceCommunityDenseEVResidualRewardV54,
    CostServiceCommunityPeakDeadlineRewardV52,
    CostServiceCommunityPrecisionValueRewardV51,
    CostServiceCommunityResidualConstraintRewardV53,
    CostServiceCommunityStorageValueRewardV49,
    CostServiceCommunityServiceBandRewardV42,
    CostServiceCommunitySmoothServiceRewardV44,
    CostServiceCommunityStorageBandRewardV41,
    CostServiceGuardRewardV2,
)
from reward_function.cost_minimization_reward import CostMinimizationReward

REWARD_FUNCTION_MAP: Dict[str, Type[RewardFunction]] = {
    "BAReward": BAReward,
    "RewardFunction": RewardFunction,
    "CCReward": CCReward,
    "CCRewardLevel1": CCRewardLevel1,
    "CCRewardLevel2": CCRewardLevel2,
    "CCRewardPhase1": CCRewardPhase1,
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
    "CostServiceCommunityResidualConstraintRewardV53": CostServiceCommunityResidualConstraintRewardV53,
    "CostServiceCommunityDenseEVResidualRewardV54": CostServiceCommunityDenseEVResidualRewardV54,
}


def get_available_reward_function_names() -> list[str]:
    return sorted(REWARD_FUNCTION_MAP.keys())
