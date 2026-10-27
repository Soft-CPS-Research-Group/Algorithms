"""Known consumer limitations; importing this inventory needs no numerical stack.

Absence from this set is not certification: every export must pass execution.
"""

SPECIALISED_ADAPTERS = frozenset(
    {
        "BCAgent",
        "IQLAgent",
        "TIMARL",
        "CCLevel2Capacity",
        "CCLevel1",
        "CCLevel2",
        "CommunityCoordinator",
        "FixedServiceOracleReplayPolicy",
        "TotalHomeOracleReplayPolicy",
        "TotalOracleReplayPolicy",
    }
)
