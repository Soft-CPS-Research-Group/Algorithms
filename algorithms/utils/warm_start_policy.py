"""Shared helpers for constructing warm-start teacher policies."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.agents.base_agent import BaseAgent


def build_warm_start_policy(
    *,
    owner_name: str,
    policy_name: str,
    policy_hyperparameters: Mapping[str, Any] | None,
    config_template: Mapping[str, Any],
    observation_names: List[List[str]],
    action_names: List[List[str]],
    action_space: List[Any],
    observation_space: List[Any],
    metadata: Optional[Dict[str, Any]],
) -> BaseAgent:
    from algorithms.agents.baseline_policies import (  # Local import avoids registry cycles.
        NormalNoBatteryPolicy,
        NormalPolicy,
        RBCBasicPolicy,
        RBCCommunityPolicy,
        RBCSmartLocalPolicy,
        RBCSmartPolicy,
        SignalAwareRBCSmartLocal,
        RandomPolicy,
    )
    from algorithms.agents.oracle_replay_policy import FixedServiceOracleReplayPolicy
    from algorithms.agents.rbc_agent import RuleBasedPolicy
    from algorithms.agents.total_home_oracle_replay_policy import TotalHomeOracleReplayPolicy
    from algorithms.agents.total_oracle_replay_policy import TotalOracleReplayPolicy

    policy_registry = {
        "RuleBasedPolicy": RuleBasedPolicy,
        "RandomPolicy": RandomPolicy,
        "NormalNoBatteryPolicy": NormalNoBatteryPolicy,
        "NormalPolicy": NormalPolicy,
        "RBCBasicPolicy": RBCBasicPolicy,
        "RBCCommunityPolicy": RBCCommunityPolicy,
        "RBCSmartLocalPolicy": RBCSmartLocalPolicy,
        "RBCSmartPolicy": RBCSmartPolicy,
        "SignalAwareRBCSmartLocal": SignalAwareRBCSmartLocal,
        "FixedServiceOracleReplayPolicy": FixedServiceOracleReplayPolicy,
        "TotalHomeOracleReplayPolicy": TotalHomeOracleReplayPolicy,
        "TotalOracleReplayPolicy": TotalOracleReplayPolicy,
    }

    policy_cls = policy_registry.get(policy_name)
    if policy_cls is None:
        supported = ", ".join(sorted(policy_registry))
        raise ValueError(
            f"Unsupported {owner_name} warm-start policy '{policy_name}'. "
            f"Supported policies: {supported}."
        )

    if policy_hyperparameters is not None and not isinstance(policy_hyperparameters, Mapping):
        raise ValueError(
            f"{owner_name} warm-start policy '{policy_name}' hyperparameters must be a mapping when provided."
        )

    config = deepcopy(dict(config_template))
    config["algorithm"] = {
        "name": policy_name,
        "hyperparameters": dict(policy_hyperparameters or {}),
    }
    policy = policy_cls(config)
    policy.attach_environment(
        observation_names=observation_names,
        action_names=action_names,
        action_space=action_space,
        observation_space=observation_space,
        metadata=metadata,
    )
    return policy


__all__ = ["build_warm_start_policy"]
