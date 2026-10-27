"""Compatibility exports, loaded only when the requested agent is used."""

__all__ = [
    "NormalNoBatteryPolicy", "NormalPolicy", "RBCBasicPolicy", "RBCCommunityPolicy",
    "RBCSmartLocalPolicy", "RBCSmartPolicy", "RandomPolicy", "SignalAwareRBCSmartLocal",
    "MADDPG", "RuleBasedPolicy", "TotalOracleReplayPolicy",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from algorithms.registry import ALGORITHM_REGISTRY
    value = ALGORITHM_REGISTRY[name]
    globals()[name] = value
    return value
