from reward_function.registry import REWARD_FUNCTION_MAP, get_available_reward_function_names


def test_reward_registry_contains_expected_reward_functions():
    names = get_available_reward_function_names()

    assert "RewardFunction" in names
    assert "V2GPenaltyReward" in names
    assert "CostMinimizationReward" in names
    assert "CostHardConstraintReward" in names
    assert "CostServiceGuardRewardV2" in names
    assert "CostServiceCostBalancedRewardV3" in names
    assert "CostServiceCommunityBandRewardV4" in names
    assert "CostServiceCommunityStorageBandRewardV41" in names
    assert "CostServiceCommunityServiceBandRewardV42" in names
    assert "CostServiceCommunityBatteryValueRewardV43" in names
    assert "CostServiceCommunitySmoothServiceRewardV44" in names
    assert "CostServiceCommunityFeasibleServiceRewardV45" in names
    assert "CostServiceCommunityFeasiblePrecisionRewardV46" in names

    assert REWARD_FUNCTION_MAP["CostMinimizationReward"].__name__ == "CostMinimizationReward"
    assert REWARD_FUNCTION_MAP["CostHardConstraintReward"].__name__ == "CostHardConstraintReward"
    assert REWARD_FUNCTION_MAP["CostServiceGuardRewardV2"].__name__ == "CostServiceGuardRewardV2"
    assert REWARD_FUNCTION_MAP["CostServiceCostBalancedRewardV3"].__name__ == "CostServiceCostBalancedRewardV3"
    assert REWARD_FUNCTION_MAP["CostServiceCommunityBandRewardV4"].__name__ == "CostServiceCommunityBandRewardV4"
    assert REWARD_FUNCTION_MAP["CostServiceCommunityStorageBandRewardV41"].__name__ == "CostServiceCommunityStorageBandRewardV41"
    assert REWARD_FUNCTION_MAP["CostServiceCommunityServiceBandRewardV42"].__name__ == "CostServiceCommunityServiceBandRewardV42"
    assert REWARD_FUNCTION_MAP["CostServiceCommunityBatteryValueRewardV43"].__name__ == "CostServiceCommunityBatteryValueRewardV43"
    assert REWARD_FUNCTION_MAP["CostServiceCommunitySmoothServiceRewardV44"].__name__ == "CostServiceCommunitySmoothServiceRewardV44"
    assert REWARD_FUNCTION_MAP["CostServiceCommunityFeasibleServiceRewardV45"].__name__ == "CostServiceCommunityFeasibleServiceRewardV45"
    assert (
        REWARD_FUNCTION_MAP["CostServiceCommunityFeasiblePrecisionRewardV46"].__name__
        == "CostServiceCommunityFeasiblePrecisionRewardV46"
    )
