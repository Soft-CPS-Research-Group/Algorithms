from reward_function.registry import REWARD_FUNCTION_MAP, get_available_reward_function_names


def test_reward_registry_contains_expected_reward_functions():
    names = get_available_reward_function_names()

    assert "RewardFunction" in names
    assert "V2GPenaltyReward" in names
    assert "CostMinimizationReward" in names
    assert "CostHardConstraintReward" in names

    assert REWARD_FUNCTION_MAP["CostMinimizationReward"].__name__ == "CostMinimizationReward"
    assert REWARD_FUNCTION_MAP["CostHardConstraintReward"].__name__ == "CostHardConstraintReward"
